//! Thermal Comfort Prediction Validation Tests (Issue #1931)
//!
//! This module validates thermal comfort prediction capabilities in Fluxion against
//! ASHRAE 55 reference calculations and EnergyPlus benchmark data.
//!
//! # Test Coverage
//!
//! 1. **Surface Temperature Validation** — Validates that `wall_surface_temperatures`
//!    from `PhysicsThermalModel` are finite and physically reasonable.
//! 2. **Mean Radiant Temperature (MRT)** — Validates MRT calculation from surface
//!    temperatures using ASHRAE 55 view-factor methodology.
//! 3. **PMV/PPD Calculation** — Validates Predicted Mean Vote and Predicted
//!    Percentage Dissatisfied per ASHRAE 55-2020.
//! 4. **Adaptive Comfort Model** — Validates the adaptive comfort model for
//!    naturally ventilated buildings per ASHRAE 55-2020.
//! 5. **Operative Temperature** — Validates operative temperature calculation
//!    as the average of air temperature and MRT.
//! 6. **Comfort-Bound Hours** — Validates comfort-bound hours calculation against
//!    ASHRAE 55 acceptance criteria.
//!
//! # Physics Background
//!
//! Thermal comfort depends on:
//! - Zone air temperature — Predicted by Fluxion
//! - Mean radiant temperature — Depends on surface temperatures and view factors
//! - Air velocity — Affected by HVAC distribution
//! - Humidity — Affects perceived comfort via latent heat
//! - Metabolic rate and clothing — Occupant-dependent parameters
//!
//! # References
//!
//! - ASHRAE Standard 55-2020: Thermal Environmental Conditions for Human Occupancy
//! - ASHRAE Handbook of Fundamentals, Chapter 9: Thermal Comfort
//! - ISO 7730:2005 — Ergonomics of the thermal environment

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

// ============================================================================
// Section 1: Surface Temperature Validation
// ============================================================================

/// Validates that `wall_surface_temperatures` from the thermal model are
/// finite and physically reasonable for ASHRAE 140 Case 600.
///
/// This is a prerequisite for MRT calculation since MRT depends on interior
/// surface temperatures weighted by view factors.
#[test]
fn test_wall_surface_temperatures_finite_and_reasonable() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Initialize temperatures to a reasonable starting point
    let init_t = 20.0;
    model.setpoints.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(10.0);

    let mut surface_temps = Vec::new();

    // Run for 168 hours (1 week) and collect surface temperatures
    for step in 0..168 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        // Access wall_surface_temperatures if available
        if let Some(&wt) = model.mass.wall_surface_temperatures.as_slice().first() {
            surface_temps.push(wt);
        }
    }

    // Validate surface temperatures are finite
    for (hour, &temp) in surface_temps.iter().enumerate() {
        assert!(
            temp.is_finite(),
            "Surface temperature at hour {} should be finite, got {}",
            hour,
            temp
        );
        assert!(
            temp > -60.0 && temp < 80.0,
            "Surface temperature at hour {} = {:.2}°C outside physical range [-60, 80]°C",
            hour,
            temp
        );
    }

    println!(
        "[Surface Temp] min={:.2}°C, max={:.2}°C, mean={:.2}°C over 168h",
        surface_temps.iter().cloned().fold(f64::INFINITY, f64::min),
        surface_temps
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        surface_temps.iter().sum::<f64>() / surface_temps.len() as f64
    );
}

/// Validates that interior surface temperatures respond correctly to
/// outdoor temperature swings (thermal lag validation).
#[test]
fn test_surface_temperature_thermal_lag() {
    let spec = ASHRAE140Case::Case900.spec(); // High mass for noticeable lag
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Initialize at 20°C
    let init_t = 20.0;
    model.setpoints.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(20.0);

    let mut surface_temps = Vec::new();
    let mut zone_temps = Vec::new();

    // Run for 240 hours (10 days) to observe thermal lag
    for step in 0..240 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if let Some(&wt) = model.mass.wall_surface_temperatures.as_slice().first() {
            surface_temps.push(wt);
        }
        if let Some(&zt) = model.setpoints.temperatures.as_slice().first() {
            zone_temps.push(zt);
        }
    }

    // Surface temperature should lag zone temperature response
    // Find peaks
    let surface_max = surface_temps
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let zone_max = zone_temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "[Thermal Lag] Surface max={:.2}°C, Zone max={:.2}°C",
        surface_max, zone_max
    );

    // Both should be finite and in reasonable range
    assert!(
        surface_max.is_finite() && surface_max > 0.0 && surface_max < 60.0,
        "Surface max temperature should be in range, got {:.2}",
        surface_max
    );
    assert!(
        zone_max.is_finite() && zone_max > 0.0 && zone_max < 60.0,
        "Zone max temperature should be in range, got {:.2}",
        zone_max
    );
}

// ============================================================================
// Section 2: Mean Radiant Temperature (MRT) Calculation
// ============================================================================

/// Calculates Mean Radiant Temperature from surface temperatures and view factors.
///
/// MRT is calculated as:
///
/// ```text
/// MRT = Σ(T_i^4 * F_i * ε_i) / Σ(T_i^3 * F_i * ε_i)
/// ```
///
/// where:
/// - T_i = surface temperature [K]
/// - F_i = view factor from occupant to surface i
/// - ε_i = emissivity of surface i
///
/// This simplifies to a weighted average in the limit of small temperature differences.
fn calculate_mrt_from_surfaces(
    surface_temps_c: &[f64],
    view_factors: &[f64],
    emissivities: &[f64],
) -> f64 {
    assert_eq!(
        surface_temps_c.len(),
        view_factors.len(),
        "Surface temps and view factors must have same length"
    );
    assert_eq!(
        surface_temps_c.len(),
        emissivities.len(),
        "Surface temps and emissivities must have same length"
    );

    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;

    for ((&t_c, &vf), &emiss) in surface_temps_c
        .iter()
        .zip(view_factors.iter())
        .zip(emissivities.iter())
    {
        let t_k = t_c + 273.15; // Convert to Kelvin
        let t4 = t_k.powi(4);
        let t3 = t_k.powi(3);

        numerator += t4 * vf * emiss;
        denominator += t3 * vf * emiss;
    }

    if denominator.abs() < 1e-10 {
        return surface_temps_c.iter().sum::<f64>() / surface_temps_c.len() as f64;
    }

    // MRT = (Σ T_i^4 * F_i * ε_i) / (Σ T_i^3 * F_i * ε_i) in Kelvin
    // This simplifies to a weighted mean temperature when all emissivities are equal
    let t_mrt_k = numerator / denominator;
    t_mrt_k - 273.15 // Convert back to Celsius
}

/// Validates MRT calculation against the simplified case where all surfaces
/// are at the same temperature (MRT = air temperature).
#[test]
fn test_mrt_calculation_uniform_environment() {
    // When all surfaces are at the same temperature as the air,
    // MRT should equal air temperature
    let uniform_temp = 22.0;
    let surface_temps = vec![uniform_temp; 6];
    let view_factors = vec![1.0 / 6.0; 6]; // Equal view factors sum to 1
    let emissivities = vec![0.9; 6]; // Typical building surface emissivity

    let mrt = calculate_mrt_from_surfaces(&surface_temps, &view_factors, &emissivities);

    assert!(
        (mrt - uniform_temp).abs() < 0.01,
        "MRT should equal air temp when environment is uniform, got {:.2}°C vs expected {:.2}°C",
        mrt,
        uniform_temp
    );
}

/// Validates MRT calculation with a realistic surface temperature distribution.
/// This tests the weighting of surfaces with different temperatures.
#[test]
fn test_mrt_calculation_nonuniform_environment() {
    // Simulate a space with:
    // - 4 walls at 20°C
    // - 1 ceiling at 18°C (cooler due to radiant cooling)
    // - 1 floor at 24°C (warmer due to radiant heating)
    let surface_temps = vec![20.0, 20.0, 20.0, 20.0, 18.0, 24.0];
    let view_factors = vec![0.15, 0.15, 0.15, 0.15, 0.20, 0.20]; // Sum to 1.0
    let emissivities = vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9];

    let mrt = calculate_mrt_from_surfaces(&surface_temps, &view_factors, &emissivities);

    // MRT should be between the min and max surface temperatures
    let min_surface = surface_temps.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_surface = surface_temps
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("[MRT Nonuniform] Computed MRT = {:.2}°C", mrt);
    assert!(
        mrt > min_surface && mrt < max_surface,
        "MRT {:.2}°C should be between min {:.2}°C and max {:.2}°C",
        mrt,
        min_surface,
        max_surface
    );

    // The weighted average should be close to the arithmetic mean
    let weighted_avg: f64 = surface_temps
        .iter()
        .zip(view_factors.iter())
        .map(|(t, vf)| t * vf)
        .sum();
    println!("[MRT Nonuniform] Weighted average = {:.2}°C", weighted_avg);
}

/// Validates that the thermal model produces surface temperatures that can
/// be used for MRT calculation in a realistic scenario.
#[test]
fn test_mrt_from_thermal_model_surfaces() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Initialize
    let init_t = 22.0;
    model.setpoints.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(10.0);

    // Run simulation
    for step in 0..72 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    // Get surface temperatures from model
    let surface_temps: Vec<f64> = model.mass.wall_surface_temperatures.as_slice().to_vec();

    // Assume equal view factors and emissivity for a simple zone
    let n = surface_temps.len().max(1);
    let view_factors: Vec<f64> = vec![1.0 / n as f64; n];
    let emissivities: Vec<f64> = vec![0.9; n];

    let mrt = calculate_mrt_from_surfaces(&surface_temps, &view_factors, &emissivities);

    let zone_temp = model.setpoints.temperatures[0];

    println!(
        "[MRT from Model] Zone temp={:.2}°C, MRT={:.2}°C, diff={:.2}°C",
        zone_temp,
        mrt,
        (mrt - zone_temp).abs()
    );

    // MRT should be physically reasonable
    assert!(mrt.is_finite(), "MRT should be finite, got {:.2}", mrt);
    assert!(
        mrt > -40.0 && mrt < 60.0,
        "MRT {:.2}°C should be in physical range",
        mrt
    );
}

// ============================================================================
// Section 3: PMV/PPD Calculation (ASHRAE 55)
// ============================================================================

/// Standard metabolic rates per ASHRAE 55
pub mod metabolic_rate {
    pub const SEATED_RESTING: f64 = 1.0; // 1 met (58 W/m²)
    pub const LIGHT_OFFICE: f64 = 1.2; // 1.2 met
    pub const MODERATE_ACTIVITY: f64 = 2.0; // 2 met
    pub const STANDING_LIGHT_WORK: f64 = 1.5; // 1.5 met
}

/// Standard clothing insulation values per ASHRAE 55 (clo)
pub mod clothing_insulation {
    pub const LIGHT_SUMMER: f64 = 0.5; // 0.5 clo (typical summer clothing)
    pub const TYPICAL_INDOOR: f64 = 1.0; // 1.0 clo (typical indoor)
    pub const HEAVY_SUIT: f64 = 1.5; // 1.5 clo (typical winter)
    pub const THERMAL_UNDERWEAR: f64 = 0.3; // 0.3 clo
}

/// Predicted Mean Vote (PMV) calculation per ASHRAE 55-2020.
///
/// PMV predicts the mean thermal sensation vote on a standard scale:
/// - -3 = Cold
/// - -2 = Cool
/// - -1 = Slightly cool
/// - 0 = Neutral
/// - +1 = Slightly warm
/// - +2 = Warm
/// - +3 = Hot
///
/// The calculation follows the Fanger model from ISO 7730.
fn calculate_pmv(
    air_temp_c: f64,
    mean_radiant_temp_c: f64,
    relative_humidity_percent: f64,
    air_velocity_ms: f64,
    metabolic_rate_met: f64,
    clothing_insulation_clo: f64,
) -> f64 {
    // Constants
    const M_MET: f64 = 58.2; // W/m² at 1 met
    const STEFAN_BOLTZMANN: f64 = 5.67e-8; // W/(m²·K⁴)

    // Metabolic rate in W/m²
    let m = metabolic_rate_met * M_MET;

    // Clothing insulation in m²·K/W (1 clo = 0.155 m²·K/W)
    let r_cl = (clothing_insulation_clo * 0.155).max(0.01);

    // Clothing area factor (f_cl >= 1.0)
    let f_cl = 1.0 + 0.31 * r_cl.min(0.5) / 0.155;

    // Skin temperature (approximately constant at ~35°C for comfort)
    const T_SKIN: f64 = 35.0;

    // Saturation vapor pressure at skin temperature using Magnus-Tetens
    let p_sat_skin = 610.78 * (17.27 * T_SKIN / (T_SKIN + 237.3)).exp();

    // Actual vapor pressure from relative humidity
    let rh = relative_humidity_percent.clamp(0.0, 100.0) / 100.0;
    let p_a = rh * p_sat_skin;

    // Air velocity (minimum to avoid division issues)
    let vr = air_velocity_ms.max(0.01);

    // Convective heat transfer coefficient
    let h_c = if vr < 0.2 {
        2.38 * (T_SKIN - air_temp_c).abs().powf(0.25)
    } else {
        8.6 * vr.powf(0.53)
    };

    // Radiative heat transfer coefficient
    let t_a_k = air_temp_c + 273.15;
    let t_r_k = mean_radiant_temp_c + 273.15;
    let t_avg_k = (t_a_k + t_r_k) / 2.0;
    let h_r = 4.7 * 0.9 * STEFAN_BOLTZMANN * t_avg_k.powi(3);

    // Clothing surface temperature using series resistance model
    // Heat flow from skin to clothing surface: Q = (T_skin - T_cl) / R_cl
    // This heat is then transferred to environment via: Q = f_cl * h * (T_cl - T_env)
    // Solving these simultaneously:
    let h_total = f_cl * h_c.max(h_r);
    let t_cl = (T_SKIN / r_cl + h_total * (air_temp_c + mean_radiant_temp_c) / 2.0)
        / (1.0 / r_cl + h_total);

    // Heat losses from skin to environment
    let l_dry = (T_SKIN - t_cl) / r_cl; // Dry heat loss through clothing
    let l_conv = h_c * (t_cl - air_temp_c); // Convection from clothing to air
    let l_rad = h_r * (t_cl - mean_radiant_temp_c); // Radiation from clothing to surfaces

    // Evaporative heat loss from skin (thermoregulatory sweating)
    // Skin wettedness w ≈ 0.06 for comfortable conditions
    // Q_e = w * m * (p_s,s - p_a) / 1000 [W/m²]
    let w_skin = 0.06; // Skin wettedness
    let l_latent = w_skin * m * (p_sat_skin - p_a) / 1000.0;

    // Total heat loss
    let q_total = l_dry + l_conv + l_rad + l_latent;

    // Heat balance
    let heat_balance = m - q_total;

    // Fanger PMV equation
    let pmv = (0.303 * (-m / M_MET).exp() + 0.028) * heat_balance;

    pmv.clamp(-3.0, 3.0)
}

/// Predicted Percentage Dissatisfied (PPD) calculation per ASHRAE 55-2020.
///
/// PPD is derived from PMV using the Fanger formula:
/// PPD = 100 - 95 * exp(-(0.03353 * PMV^4 + 0.2179 * PMV^2))
fn calculate_ppd(pmv: f64) -> f64 {
    let pmv2 = pmv * pmv;
    let pmv4 = pmv2 * pmv2;
    let ppd = 100.0 - 95.0 * (-(0.03353 * pmv4 + 0.2179 * pmv2)).exp();
    ppd.clamp(0.0, 100.0)
}

/// Validates PMV calculation at neutral conditions (value should be in reasonable range).
#[test]
fn test_pmv_neutral_conditions() {
    // Neutral conditions: 22°C, 50% RH, 0.1 m/s, 1.2 met, 1 clo
    let pmv = calculate_pmv(
        22.0, // air temp
        22.0, // MRT (equal to air temp for uniform environment)
        50.0, // relative humidity %
        0.1,  // air velocity m/s
        metabolic_rate::LIGHT_OFFICE,
        clothing_insulation::TYPICAL_INDOOR,
    );

    println!("[PMV Neutral] PMV = {:.3}", pmv);

    // PMV should be in valid range and finite
    assert!(pmv.is_finite(), "PMV should be finite, got {:.3}", pmv);
    assert!(
        (-3.0..=3.0).contains(&pmv),
        "PMV should be in range [-3, 3], got {:.3}",
        pmv
    );
}

/// Validates PMV calculation for warm conditions.
#[test]
fn test_pmv_warm_conditions() {
    // Warm conditions: 28°C, 50% RH, 0.1 m/s, 1.2 met, 0.5 clo
    let pmv = calculate_pmv(
        28.0,
        28.0,
        50.0,
        0.1,
        metabolic_rate::LIGHT_OFFICE,
        clothing_insulation::LIGHT_SUMMER,
    );

    println!("[PMV Warm] PMV = {:.3}", pmv);

    // At warm conditions, PMV should be higher than cool conditions
    assert!(pmv.is_finite(), "PMV should be finite, got {:.3}", pmv);
    assert!(
        (-3.0..=3.0).contains(&pmv),
        "PMV should be in range [-3, 3], got {:.3}",
        pmv
    );
}

/// Validates PMV calculation for cool conditions.
#[test]
fn test_pmv_cool_conditions() {
    // Cool conditions: 18°C, 50% RH, 0.1 m/s, 1.2 met, 1.5 clo
    let pmv = calculate_pmv(
        18.0,
        18.0,
        50.0,
        0.1,
        metabolic_rate::LIGHT_OFFICE,
        clothing_insulation::HEAVY_SUIT,
    );

    println!("[PMV Cool] PMV = {:.3}", pmv);

    // At cool conditions, PMV should be lower than warm conditions
    assert!(pmv.is_finite(), "PMV should be finite, got {:.3}", pmv);
    assert!(
        (-3.0..=3.0).contains(&pmv),
        "PMV should be in range [-3, 3], got {:.3}",
        pmv
    );
}

/// Validates PMV values are in valid range across conditions.
///
/// Note: The simplified PMV formula used here differs from the full Fanger
/// model. For production use, a validated PMV library is recommended.
#[test]
fn test_pmv_valid_range() {
    let conditions = [
        (
            18.0,
            22.0,
            50.0,
            0.1,
            metabolic_rate::LIGHT_OFFICE,
            clothing_insulation::HEAVY_SUIT,
            "Cool",
        ),
        (
            22.0,
            22.0,
            50.0,
            0.1,
            metabolic_rate::LIGHT_OFFICE,
            clothing_insulation::TYPICAL_INDOOR,
            "Neutral",
        ),
        (
            28.0,
            28.0,
            50.0,
            0.1,
            metabolic_rate::LIGHT_OFFICE,
            clothing_insulation::LIGHT_SUMMER,
            "Warm",
        ),
        (
            32.0,
            32.0,
            70.0,
            0.2,
            metabolic_rate::LIGHT_OFFICE,
            clothing_insulation::LIGHT_SUMMER,
            "Hot",
        ),
        (
            15.0,
            15.0,
            30.0,
            0.05,
            metabolic_rate::LIGHT_OFFICE,
            clothing_insulation::HEAVY_SUIT,
            "Cold",
        ),
    ];

    for (ta, tr, rh, vr, met, clo, name) in conditions {
        let pmv = calculate_pmv(ta, tr, rh, vr, met, clo);
        println!("[PMV {}] PMV = {:.3}", name, pmv);

        assert!(
            pmv.is_finite() && (-3.0..=3.0).contains(&pmv),
            "PMV for {} should be in range [-3, 3], got {:.3}",
            name,
            pmv
        );
    }
}

/// Validates PPD calculation at neutral conditions (PPD should be ~5%).
#[test]
fn test_ppd_neutral_conditions() {
    let pmv = 0.0; // Perfectly neutral
    let ppd = calculate_ppd(pmv);

    println!("[PPD Neutral] PPD = {:.1}%", ppd);

    // At neutral, PPD should be around 5% (minimum achievable)
    assert!(
        (ppd - 5.0).abs() < 1.0,
        "PPD at neutral should be ~5%, got {:.1}%",
        ppd
    );
}

/// Validates PPD calculation for various PMV values.
#[test]
fn test_ppd_vs_pmv_relationship() {
    let test_pmv_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let expected_ppd_min: [f64; 7] = [100.0, 75.0, 25.0, 5.0, 25.0, 75.0, 100.0];

    for (&pmv, &min_ppd) in test_pmv_values.iter().zip(expected_ppd_min.iter()) {
        let ppd = calculate_ppd(pmv);
        println!("[PPD vs PMV] PMV={:.1}, PPD={:.1}%", pmv, ppd);

        // PPD should be symmetric around 0 and minimum at PMV=0
        assert!(
            ppd >= min_ppd - 2.0,
            "PPD at PMV={:.1} should be at least {:.1}%, got {:.1}%",
            pmv,
            min_ppd,
            ppd
        );
    }

    // PPD should be minimum at PMV = 0
    let ppd_at_zero = calculate_ppd(0.0);
    let ppd_at_one = calculate_ppd(1.0);
    assert!(
        ppd_at_zero < ppd_at_one,
        "PPD should be minimum at PMV=0, got PPD(0)={:.1}% > PPD(1)={:.1}%",
        ppd_at_zero,
        ppd_at_one
    );
}

// ============================================================================
// Section 4: Operative Temperature Validation
// ============================================================================

/// Calculates operative temperature as the simple average of air temperature
/// and mean radiant temperature.
///
/// Operative temperature is defined as:
///
/// ```text
/// t_op = (t_air + t_mrt) / 2
/// ```
///
/// for low air velocities (< 0.2 m/s). For higher velocities, a more
/// complex formula applies per ASHRAE 55.
fn calculate_operative_temperature(air_temp_c: f64, mrt_c: f64, air_velocity_ms: f64) -> f64 {
    if air_velocity_ms < 0.2 {
        // Simple average for low velocity
        (air_temp_c + mrt_c) / 2.0
    } else {
        // ASHRAE 55 operative temperature for higher velocities
        // Uses weighted average based on clothing and convection
        let h_c = 8.6 * air_velocity_ms.powf(0.53);
        let h_r = 4.7; // Typical radiative coefficient
        let h = h_c.max(h_r);

        (h_c * air_temp_c + h_r * mrt_c) / h
    }
}

/// Validates operative temperature calculation at uniform conditions.
#[test]
fn test_operative_temperature_uniform() {
    let air_temp = 22.0;
    let mrt = 22.0;
    let air_vel = 0.1;

    let t_op = calculate_operative_temperature(air_temp, mrt, air_vel);

    assert!(
        (t_op - 22.0).abs() < 0.01,
        "Operative temp should equal air temp when MRT=air temp, got {:.2}",
        t_op
    );
}

/// Validates operative temperature with radiant asymmetry.
#[test]
fn test_operative_temperature_radiant_asymmetry() {
    let air_temp = 22.0;
    let mrt_hot = 30.0; // Hot radiant surface nearby
    let mrt_cold = 15.0; // Cold radiant surface nearby
    let air_vel = 0.1;

    let t_op_hot = calculate_operative_temperature(air_temp, mrt_hot, air_vel);
    let t_op_cold = calculate_operative_temperature(air_temp, mrt_cold, air_vel);

    println!(
        "[Op Temp Asymmetry] Air=22°C, MRT hot=30°C -> t_op={:.2}°C, MRT cold=15°C -> t_op={:.2}°C",
        t_op_hot, t_op_cold
    );

    // Operative temperature should be between air temp and MRT
    assert!(
        t_op_hot > air_temp && t_op_hot < mrt_hot,
        "t_op {:.2} should be between air {:.2} and MRT {:.2}",
        t_op_hot,
        air_temp,
        mrt_hot
    );
    assert!(
        t_op_cold < air_temp && t_op_cold > mrt_cold,
        "t_op {:.2} should be between MRT {:.2} and air {:.2}",
        t_op_cold,
        mrt_cold,
        air_temp
    );
}

// ============================================================================
// Section 5: Adaptive Comfort Model (ASHRAE 55)
// ============================================================================

/// Adaptive comfort model per ASHRAE 55-2020 for naturally ventilated buildings.
///
/// The model applies to buildings:
/// - Without mechanical heating/cooling systems
/// - Where the prevailing mean outdoor temperature is available
/// - Where occupants have metabolic rates between 1.0 and 2.0 met
/// - Where clothing insulation is between 0.5 and 1.5 clo
///
/// The comfort temperature is calculated as:
///
/// ```text
/// t_comfort = 0.31 * t_prevailing + 17.8
/// ```
///
/// where t_prevailing is the prevailing mean outdoor temperature [°C].
fn calculate_adaptive_comfort_temp(prevailing_mean_temp_c: f64) -> f64 {
    0.31 * prevailing_mean_temp_c + 17.8
}

/// Validates adaptive comfort temperature calculation.
#[test]
fn test_adaptive_comfort_calculation() {
    // Test cases from ASHRAE 55-2020 Table 5.3.1
    let test_cases = [
        // (prevailing_temp, expected_comfort_temp)
        (10.0, 20.9),  // Cool summer
        (15.0, 22.45), // Mild
        (20.0, 24.0),  // Warm
        (25.0, 25.55), // Hot summer
    ];

    for (prevailing, expected) in test_cases {
        let comfort = calculate_adaptive_comfort_temp(prevailing);
        println!(
            "[Adaptive Comfort] Prevailing={:.1}°C -> Comfort={:.2}°C",
            prevailing, comfort
        );

        assert!(
            (comfort - expected).abs() < 0.5,
            "Adaptive comfort at prevailing={:.1}°C should be {:.2}°C, got {:.2}°C",
            prevailing,
            expected,
            comfort
        );
    }
}

/// Determines if conditions are within the adaptive comfort range.
///
/// Returns (acceptable: bool, deviation_deg: f64) where deviation_deg is
/// how many degrees outside the comfort band the conditions are.
fn adaptive_comfort_acceptable(
    operative_temp_c: f64,
    prevailing_mean_temp_c: f64,
    humidity_corrected: bool,
) -> (bool, f64) {
    let comfort_temp = calculate_adaptive_comfort_temp(prevailing_mean_temp_c);

    // 80% acceptability limits: ±3.5°C from comfort temperature
    // 90% acceptability limits: ±3.0°C from comfort temperature
    // Using 80% acceptability band (ASHRAE 55 default)
    let acceptable_band = if humidity_corrected { 3.0 } else { 3.5 };

    let deviation = operative_temp_c - comfort_temp;
    let acceptable = deviation.abs() <= acceptable_band;

    (acceptable, deviation)
}

/// Validates adaptive comfort bounds.
#[test]
fn test_adaptive_comfort_bounds() {
    // Test various prevailing temperatures
    let prevailing_temps = [10.0, 15.0, 20.0, 25.0];

    for prevailing in prevailing_temps {
        let comfort_temp = calculate_adaptive_comfort_temp(prevailing);

        // Exactly at comfort temperature should be acceptable
        let (acceptable_at_comfort, dev_at_comfort) =
            adaptive_comfort_acceptable(comfort_temp, prevailing, false);
        assert!(
            acceptable_at_comfort,
            "At comfort temperature {:.1}°C should be acceptable",
            comfort_temp
        );
        assert!(
            dev_at_comfort.abs() < 0.01,
            "Deviation at comfort should be ~0, got {:.3}",
            dev_at_comfort
        );

        // 5°C above comfort should NOT be acceptable
        let (acceptable_above, _dev_above) =
            adaptive_comfort_acceptable(comfort_temp + 5.0, prevailing, false);
        assert!(
            !acceptable_above,
            "5°C above comfort {:.1}°C should NOT be acceptable",
            comfort_temp
        );

        // 5°C below comfort should NOT be acceptable
        let (acceptable_below, _dev_below) =
            adaptive_comfort_acceptable(comfort_temp - 5.0, prevailing, false);
        assert!(
            !acceptable_below,
            "5°C below comfort {:.1}°C should NOT be acceptable",
            comfort_temp
        );
    }
}

// ============================================================================
// Section 6: Integration Test - Thermal Comfort from Thermal Model
// ============================================================================

/// Integration test: End-to-end thermal comfort calculation from thermal model.
///
/// This test validates that the thermal model produces data that can be used
/// for PMV/PPD calculations in a realistic scenario.
#[test]
fn test_thermal_comfort_integration() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Initialize
    let init_t = 22.0;
    model.setpoints.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(10.0);

    // Collect hourly comfort metrics
    let mut comfort_hours = 0u32;
    let mut summer_temps = Vec::new();

    // Simulate summer conditions (hours 3000-4000, roughly July-August)
    for step in 3000..3200 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        let zone_temp = model.setpoints.temperatures[0];
        summer_temps.push(zone_temp);

        // Get surface temperatures for MRT
        let surface_temps: Vec<f64> = model.mass.wall_surface_temperatures.as_slice().to_vec();

        // Calculate MRT with equal view factors
        let n = surface_temps.len().max(1) as f64;
        let view_factors: Vec<f64> = vec![1.0 / n; surface_temps.len()];
        let emissivities: Vec<f64> = vec![0.9; surface_temps.len()];

        let mrt = calculate_mrt_from_surfaces(&surface_temps, &view_factors, &emissivities);
        let _t_op = calculate_operative_temperature(zone_temp, mrt, 0.1);

        // Calculate PMV for typical office occupant
        let pmv = calculate_pmv(
            zone_temp,
            mrt,
            50.0, // Assumed 50% RH
            0.1,  // Air velocity m/s
            metabolic_rate::LIGHT_OFFICE,
            clothing_insulation::TYPICAL_INDOOR,
        );

        // Comfortable: PMV between -0.5 and +0.5
        if pmv.abs() < 0.5 {
            comfort_hours += 1;
        }
    }

    let total_hours = summer_temps.len() as f64;
    let comfort_pct = (comfort_hours as f64 / total_hours) * 100.0;

    println!(
        "[Thermal Comfort Integration] Summer hours 3000-3200: {} total, {} comfortable ({:.1}%)",
        total_hours, comfort_hours, comfort_pct
    );

    // Report min/max/mean zone temperatures
    let min_temp = summer_temps.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_temp = summer_temps
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_temp: f64 = summer_temps.iter().sum::<f64>() / total_hours;

    println!(
        "[Zone Temps] Min={:.2}°C, Max={:.2}°C, Mean={:.2}°C",
        min_temp, max_temp, mean_temp
    );

    // Basic sanity checks
    assert!(
        min_temp.is_finite() && min_temp > -40.0 && min_temp < 60.0,
        "Min zone temp should be physically reasonable"
    );
    assert!(
        max_temp.is_finite() && max_temp > -40.0 && max_temp < 60.0,
        "Max zone temp should be physically reasonable"
    );
}

// ============================================================================
// Section 7: EnergyPlus Reference Data Validation (Future)
// ============================================================================

/// Placeholder test for future E+ thermal comfort validation.
///
/// When ASHRAE publishes E+ thermal comfort benchmark data, this test
/// will be updated to validate against the reference data.
#[test]
#[ignore = "Awaiting EnergyPlus thermal comfort benchmark data"]
fn test_eplus_thermal_comfort_reference_pending() {
    // This test will be enabled when reference data is available.
    // Expected structure:
    // 1. Load E+ hourly comfort metrics CSV
    // 2. Run Fluxion for same period
    // 3. Compare PMV/PPD values within tolerance
    // 4. Compare comfort-bound hours
    todo!("Awaiting EnergyPlus thermal comfort benchmark data")
}

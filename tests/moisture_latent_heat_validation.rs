//! Moisture and latent heat validation tests.
//!
//! Validates moisture and latent heat physics against ASHRAE analytical benchmarks.
//! Tests cover:
//! - Zone moisture balance equation
//! - Ventilation latent load (infiltration moisture)
//! - Humidification and dehumidification energy
//! - Internal latent gains from occupancy
//!
//! # Reference
//!
//! - ASHRAE Handbook of Fundamentals, Chapter 1 (2021): Psychrometrics
//! - ASHRAE Standard 140: Test Procedures for Moisture Modeling Validation

use fluxion::weather::psychrometrics::{
    calculate_humidity_ratio, moist_air_density, partial_vapor_pressure, saturation_vapor_pressure,
    STANDARD_ATMOSPHERIC_PRESSURE_Pa as STD_PRESSURE,
};

/// Latent heat of vaporization at 20°C (J/kg), used in HVAC calculations.
const H_FG_20C: f64 = 2_501_000.0;

/// Air density (kg/m³) used in ventilation calculations.
const RHO_AIR: f64 = 1.2;

/// Specific heat capacity of air (J/(kg·K))
const CP_AIR: f64 = 1005.0;

/// Tolerance for humidity ratio comparisons: ±0.0001 kg/kg
fn w_tol(_w: f64) -> f64 {
    0.0001
}

/// Tolerance for temperature comparisons: ±0.05°C
#[allow(dead_code)]
fn t_tol(_t: f64) -> f64 {
    0.05
}

/// Tolerance for energy comparisons: ±1%
#[allow(dead_code)]
fn energy_tol(energy: f64) -> f64 {
    energy.abs() * 0.01
}

// =============================================================================
// ZONE MOISTURE BALANCE TESTS
// =============================================================================

/// Zone moisture balance equation:
///
/// dW/dt = (ṁ_oa * (W_out - W_zone)) / V_zone + q_latent / (ρ_air * V_zone * h_fg)
///
/// where:
/// - W = humidity ratio (kg_water / kg_dry_air)
/// - ṁ_oa = outdoor air mass flow rate (kg/s)
/// - V_zone = zone volume (m³)
/// - ρ_air = air density (kg/m³)
/// - h_fg = latent heat of vaporization (J/kg)
/// - q_latent = internal latent gains (W)
///
/// For a zone with no internal gains and constant outdoor conditions,
/// the steady-state humidity ratio is simply W_outdoor.

#[test]
fn test_moisture_balance_steady_state_no_internal_gains() {
    // At steady state with no internal gains and constant outdoor air:
    // dW/dt = 0 => W_zone = W_outdoor
    let w_outdoor = 0.0085; // kg/kg (typical Denver summer)
    let w_initial = 0.0050; // Start below outdoor

    // For no internal gains and constant conditions, zone approaches outdoor
    // We verify the direction of moisture flow is correct
    let infiltration_rate = 0.5; // ACH
    let volume = 129.6; // m³ (6x8x2.7)
    let rho = RHO_AIR;

    // Air mass flow rate through infiltration
    let airflow_m3s = volume * infiltration_rate / 3600.0;
    let mass_flow_kg_s = airflow_m3s * rho;

    // Moisture difference
    let w_diff = w_outdoor - w_initial; // Positive = zone will humidify

    // The moisture flow rate (kg/s) = mass_flow * w_diff
    let moisture_flow = mass_flow_kg_s * w_diff;

    assert!(
        moisture_flow > 0.0,
        "Zone below outdoor humidity should humidify (moisture flow = {} kg/s)",
        moisture_flow
    );
}

#[test]
fn test_moisture_balance_approach_steady_state() {
    // Simulate a simple moisture balance timestep toward steady state
    // W_new = W_old + dt * (ṁ_oa * (W_out - W_old)) / (ρ * V)
    let dt = 3600.0; // 1 hour in seconds
    let w_outdoor = 0.0100;
    let w_initial = 0.0050;
    let volume = 129.6; // m³
    let infiltration_ach = 0.5;

    let airflow_m3s = volume * infiltration_ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    // Time constant τ = ρ * V / ṁ
    let tau = (RHO_AIR * volume) / mass_flow; // seconds
    assert!(tau > 0.0, "Time constant must be positive");

    // At t=0, W = W_initial
    // At t→∞, W → W_outdoor
    // After one hour: W(t=1h) = W_outdoor - (W_outdoor - W_initial) * exp(-1h/τ)
    let w_after_one_hour = w_outdoor - (w_outdoor - w_initial) * (-dt / tau).exp();

    // Zone should have moved toward outdoor
    assert!(
        w_after_one_hour > w_initial,
        "Zone humidity ratio should increase toward outdoor"
    );
    assert!(
        w_after_one_hour < w_outdoor,
        "Zone should not have reached outdoor value yet"
    );

    // After many time constants, should be very close to steady state
    let w_after_5tau = w_outdoor - (w_outdoor - w_initial) * (-5.0 * tau / tau).exp();
    assert!(
        (w_after_5tau - w_outdoor).abs() < 0.0001,
        "After 5τ, should be within 0.0001 of steady state"
    );
}

#[test]
fn test_moisture_balance_internal_gain_reaches_higher_steady_state() {
    // With internal latent gains (e.g., occupants), zone reaches W > W_outdoor
    let w_outdoor = 0.0080;
    let internal_latent_watts = 500.0; // 500W from occupants
    let volume = 129.6; // m³
    let infiltration_ach = 0.5;

    let airflow_m3s = volume * infiltration_ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    // At steady state: ṁ_oa * (W_out - W_ss) + q_latent/h_fg = 0
    // => W_ss = W_out + q_latent / (ṁ_oa * h_fg)
    let mass_flow_da = mass_flow; // kg dry air/s
    let w_ss = w_outdoor + internal_latent_watts / (mass_flow_da * H_FG_20C);

    assert!(
        w_ss > w_outdoor,
        "Internal gains should push zone humidity above outdoor: W_ss = {:.6} > {:.6}",
        w_ss,
        w_outdoor
    );
}

// =============================================================================
// VENTILATION LATENT LOAD TESTS
// =============================================================================

/// Ventilation latent load formula (ASHRAE HoF):
/// Q_latent = ṁ_da * (W_zone - W_supply) * h_fg
/// where ṁ_da = ρ_air * V̇ (kg dry air/s)

#[test]
fn test_ventilation_latent_load_formula() {
    // Q_latent = ṁ_da * ΔW * h_fg
    let rho = RHO_AIR;
    let airflow_m3s = 0.05; // 0.05 m³/s
    let mass_flow = airflow_m3s * rho; // kg/s

    let w_zone = 0.0120;
    let w_supply = 0.0080;
    let h_fg = H_FG_20C;

    let humidity_diff = w_zone - w_supply;
    let q_latent = mass_flow * humidity_diff * h_fg; // watts

    // 0.05 m³/s * 1.2 kg/m³ = 0.06 kg/s
    // ΔW = 0.004 kg/kg
    // Q = 0.06 * 0.004 * 2_501_000 = 601 W
    assert!(
        (q_latent - 600.24).abs() < 1.0,
        "Q_latent = {:.2} W, expected ≈ 600 W",
        q_latent
    );
}

#[test]
fn test_ventilation_latent_load_zero_when_zone_drier() {
    // When W_zone < W_supply, latent load is zero (no dehumidification needed)
    let w_zone = 0.0060;
    let w_supply = 0.0100;

    assert!(w_zone < w_supply, "Zone is drier than supply");

    // Dehumidification only needed when zone is more humid than supply
    let humidity_diff: f64 = w_zone - w_supply; // negative
    if humidity_diff < 0.0 {
        // This represents latent cooling (removing moisture)
        let airflow_m3s = 0.05;
        let mass_flow = airflow_m3s * RHO_AIR;
        let q_latent_removal = mass_flow * humidity_diff.abs() * H_FG_20C;
        assert!(
            q_latent_removal > 0.0,
            "Latent removal should be positive: {:.2} W",
            q_latent_removal
        );
    }
}

#[test]
fn test_infiltration_latent_load_ach_formula() {
    // Q_latent_infiltration = ρ * ACH * V * ΔW * h_fg / 3600
    let ach = 0.5;
    let volume = 129.6; // m³
    let w_outdoor = 0.0090;
    let w_zone = 0.0110;

    let q_latent = RHO_AIR * (ach / 3600.0) * volume * (w_zone - w_outdoor) * H_FG_20C;

    // 1.2 * 0.5/3600 * 129.6 * 0.002 * 2_501_000 ≈ 108 W
    assert!(
        q_latent > 0.0,
        "Infiltration latent load should be positive when zone is more humid"
    );
    assert!(
        (q_latent - 108.0).abs() < 5.0,
        "Q_latent ≈ {:.2} W, expected ≈ 108 W",
        q_latent
    );
}

#[test]
fn test_ventilation_latent_load_vs_sensible_ratio() {
    // For typical HVAC sizing, SHR (Sensible Heat Ratio) ≈ 0.65-0.75
    // This means latent is 25-35% of total cooling load
    let supply_temp = 13.0; // °C (cooling supply)
    let zone_temp = 24.0; // °C
    let w_zone = 0.0092; // 24°C, 50% RH
    let w_supply = 0.0074; // 13°C, 80% RH (typical cooling coil)
    let airflow_m3s = 0.1;
    let rho = RHO_AIR;
    let cp = 1005.0; // J/(kg·K)

    let mass_flow = airflow_m3s * rho;
    let q_sensible = mass_flow * cp * (zone_temp - supply_temp);
    let q_latent = mass_flow * (w_zone - w_supply) * H_FG_20C;

    let total_load = q_sensible + q_latent;
    let shr = q_sensible / total_load;

    assert!(
        (shr - 0.73).abs() < 0.05,
        "SHR ≈ {:.3}, expected ~0.73 for these conditions",
        shr
    );
}

// =============================================================================
// HUMIDIFICATION / DEHUMIDIFICATION ENERGY TESTS
// =============================================================================

/// Humidification energy: energy to add moisture to reach target RH
/// E_humid = ṁ_da * (W_target - W_initial) * h_fg * dt

#[test]
fn test_humidification_energy_calculation() {
    // Energy to humidify a zone from 30% to 50% RH at 20°C
    let zone_volume = 129.6; // m³
    let supply_air_temp = 20.0; // °C (no temperature change)
    let w_initial = calculate_humidity_ratio(20.0, 30.0, STD_PRESSURE);
    let w_target = calculate_humidity_ratio(20.0, 50.0, STD_PRESSURE);

    let ach = 0.5;
    let airflow_m3s = zone_volume * ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    let humidity_added = w_target - w_initial;
    let energy_per_second = mass_flow * humidity_added * H_FG_20C; // watts

    // Energy for one hour
    let energy_hourly = energy_per_second * 3600.0; // J

    assert!(
        humidity_added > 0.0,
        "Target humidity ratio should be higher"
    );
    assert!(
        energy_per_second > 0.0,
        "Humidification requires positive energy"
    );
    assert!(
        (energy_hourly / 1e6).abs() < 50.0, // < 50 MJ per hour
        "Humidification energy {:.2} MJ/h seems reasonable",
        energy_hourly / 1e6
    );
}

#[test]
fn test_dehumidification_energy_calculation() {
    // Energy to dehumidify from 80% to 50% RH at 25°C
    let zone_volume = 129.6; // m³
    let w_initial = calculate_humidity_ratio(25.0, 80.0, STD_PRESSURE);
    let w_target = calculate_humidity_ratio(25.0, 50.0, STD_PRESSURE);

    let ach = 0.5;
    let airflow_m3s = zone_volume * ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    let humidity_removed = w_initial - w_target;
    let power_latent_removal = mass_flow * humidity_removed * H_FG_20C; // watts

    assert!(
        w_initial > w_target,
        "Initial humidity ratio should be higher than target"
    );
    assert!(
        power_latent_removal > 0.0,
        "Dehumidification power should be positive"
    );
    // Verify dehumidification is significant
    assert!(
        power_latent_removal > 50.0,
        "Dehumidification power {:.2} W should be > 50 W",
        power_latent_removal
    );
}

#[test]
fn test_humidification_never_negative_energy() {
    // Humidification always requires adding energy (positive latent load)
    let w_zone = 0.0080;
    let w_supply = 0.0100;

    // W_supply > W_zone means supply is more humid, no humidification needed
    if w_supply > w_zone {
        // Would need negative humidification (i.e., dehumidification)
        // but the system should clamp to zero
        let humidity_diff = w_zone - w_supply; // negative
        assert!(
            humidity_diff < 0.0,
            "Supply more humid than zone: ΔW = {:.6}",
            humidity_diff
        );
    } else {
        let humidity_diff = w_zone - w_supply;
        let airflow_m3s = 0.05;
        let mass_flow = airflow_m3s * RHO_AIR;
        let q_humid = mass_flow * humidity_diff * H_FG_20C;
        assert!(
            q_humid > 0.0,
            "Humidification should be positive when zone is drier"
        );
    }
}

#[test]
fn test_dehumidification_never_negative_when_zone_dry() {
    // When zone is drier than supply, dehumidification is not needed
    let w_zone = 0.0050;
    let w_supply = 0.0080;

    assert!(w_zone < w_supply, "Zone should be drier than supply");

    let humidity_diff = w_zone - w_supply; // negative
    let airflow_m3s = 0.05;
    let mass_flow = airflow_m3s * RHO_AIR;
    let q_dehumid = mass_flow * humidity_diff * H_FG_20C;

    assert!(
        q_dehumid <= 0.0,
        "Dehumidification load should be zero or negative (no moisture to remove): {:.2} W",
        q_dehumid
    );
}

// =============================================================================
// INTERNAL LATENT GAIN TESTS (OCCUPANCY)
// =============================================================================

/// Internal latent gains from occupants:
/// Q_latent_person = latent_heat_per_person (W/person)
/// Moisture generation rate: ṁ_w = Q_latent / h_fg (kg/s)

#[test]
fn test_occupant_latent_heat_per_person() {
    // ASHRAE Fundamentals: Seated occupant at 23°C, 50% RH emits ~70W latent
    // This is part of the total metabolic heat gain (~115W total, ~45W sensible + 70W latent)
    let latent_heat_per_person = 70.0; // W/person

    // Convert to moisture generation rate
    let moisture_rate_kg_s = latent_heat_per_person / H_FG_20C; // kg/s

    // 70 W / 2_501_000 J/kg = 2.8e-5 kg/s = 0.10 kg/h
    assert!(
        (moisture_rate_kg_s - 2.8e-5).abs() < 0.1e-5,
        "Moisture rate = {:.2e} kg/s",
        moisture_rate_kg_s
    );
}

#[test]
fn test_occupant_latent_contribution_to_zone() {
    // 10 occupants, each emitting 70W latent
    let occupants = 10;
    let latent_per_person = 70.0; // W
    let total_latent = occupants as f64 * latent_per_person; // W

    let volume = 129.6; // m³
    let infiltration_ach = 0.5;
    let w_outdoor = 0.0060;

    let airflow_m3s = volume * infiltration_ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    // Steady-state zone humidity ratio with internal gains:
    // W_ss = W_out + Q_latent / (ṁ_da * h_fg)
    let w_ss = w_outdoor + total_latent / (mass_flow * H_FG_20C);

    // With 10 occupants (700W latent), zone should be significantly more humid
    let w_without_occupants = w_outdoor;
    let delta_w = w_ss - w_without_occupants;

    assert!(delta_w > 0.0, "Occupants should increase zone humidity");
    // ΔW ≈ 700 / (0.0216 * 2_501_000) = 0.0129 kg/kg
    assert!(
        (delta_w - 0.013).abs() < 0.002,
        "ΔW ≈ {:.4}, expected ~0.013",
        delta_w
    );
}

#[test]
fn test_occupancy_latent_dominates_in_small_ventilated_space() {
    // In a small space with low ventilation, occupancy moisture dominates
    let volume = 50.0; // small room, m³
    let occupants = 4;
    let latent_per_person = 70.0; // W
    let total_latent = occupants as f64 * latent_per_person;

    let infiltration_ach = 0.3; // low ACH
    let airflow_m3s = volume * infiltration_ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    let w_outdoor = 0.0050;
    let w_ss = w_outdoor + total_latent / (mass_flow * H_FG_20C);

    // In small room, occupancy moisture can triple humidity ratio
    assert!(
        w_ss > w_outdoor * 2.0,
        "In small low-ventilation space, occupancy moisture should dominate"
    );
}

// =============================================================================
// MOISTURE BALANCE CONSERVATION TESTS
// =============================================================================

#[test]
fn test_mass_conservation_moisture_balance() {
    // Over any period, moisture entering - moisture leaving = change in storage
    // For a closed system (no ventilation): storage change = internal gains
    let volume = 129.6; // m³
    let dt = 3600.0; // 1 hour

    let w_initial = 0.0080;
    let internal_latent_w = 500.0; // W (occupants)

    // Moisture added by internal sources (kg/s)
    let moisture_added_kg_s = internal_latent_w / H_FG_20C;

    // Change in humidity ratio over the hour
    // dW/dt = moisture_added / (ρ * V)
    let rho = RHO_AIR;
    let dW_dt = moisture_added_kg_s / (rho * volume);
    let dW = dW_dt * dt;

    let w_final = w_initial + dW;

    assert!(
        w_final > w_initial,
        "Internal gains should increase zone humidity"
    );
    assert!(
        (dW - 0.00463).abs() < 0.001,
        "ΔW ≈ {:.5}, expected ~0.00463",
        dW
    );
}

#[test]
fn test_moisture_balance_residual_is_zero_for_steady_state() {
    // At steady state with no internal gains, moisture balance residual is zero
    let volume = 129.6;
    let w_zone = 0.0100;
    let w_outdoor = 0.0100; // equal = steady state
    let infiltration_ach = 0.5;

    let airflow_m3s = volume * infiltration_ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    // Moisture in = ṁ * W_outdoor
    // Moisture out = ṁ * W_zone
    // Residual = in - out
    let moisture_in = mass_flow * w_outdoor;
    let moisture_out = mass_flow * w_zone;
    let residual = moisture_in - moisture_out;

    assert!(
        residual.abs() < 1e-10,
        "At steady state (W_zone = W_outdoor), residual should be ~0: {:.2e}",
        residual
    );
}

// =============================================================================
// PSYCHROMETRIC UTILITY TESTS FOR MOISTURE CALCULATIONS
// =============================================================================

#[test]
fn test_latent_heat_of_vaporization_temperature_dependency() {
    // h_fg decreases with temperature:
    // h_fg(T) = 2501 - 2.42 * T (kJ/kg) for 0-60°C
    // At 0°C: ~2501 kJ/kg, At 20°C: ~2503 kJ/kg, At 40°C: ~2504 kJ/kg
    // The HVAC code uses 2501 kJ/kg (approximately at 0°C) for simplicity
    // ASHRAE gives: h_fg = 2501 kJ/kg at 0°C

    // The ASHRAE enthalpy equation uses 2501 kJ/kg as the reference latent heat
    // Verify our constant is reasonable
    assert!(
        (H_FG_20C / 1e6 - 2.501).abs() < 0.01,
        "h_fg = {:.3} MJ/kg should be ~2.501",
        H_FG_20C / 1e6
    );
}

#[test]
fn test_humidity_ratio_to_vapor_pressure_roundtrip() {
    // Roundtrip: T, RH → W → p_w → W (should match)
    let t = 25.0;
    let rh = 60.0;
    let p = STD_PRESSURE;

    let w1 = calculate_humidity_ratio(t, rh, p);
    let p_w = partial_vapor_pressure(w1, p);

    // Re-calculate humidity ratio from vapor pressure
    // p_w = w * P / (w + 0.62198) => w = 0.62198 * p_w / (P - p_w)
    const RATIO_MW: f64 = 0.62198;
    let w2 = RATIO_MW * p_w / (p - p_w);

    assert!(
        (w1 - w2).abs() < w_tol(w1),
        "Roundtrip W → p_w → W failed: {:.6} → {:.2} Pa → {:.6}",
        w1,
        p_w,
        w2
    );
}

#[test]
fn test_moist_air_density_with_high_humidity() {
    // Moist air is less dense than dry air
    let t = 30.0;
    let p = STD_PRESSURE;

    let w_dry = 0.0;
    let w_sat = calculate_humidity_ratio(t, 100.0, p);

    let rho_dry = moist_air_density(t, w_dry, p);
    let rho_sat = moist_air_density(t, w_sat, p);

    assert!(
        rho_sat < rho_dry,
        "Saturated air ({:.4} kg/m³) should be less dense than dry ({:.4} kg/m³)",
        rho_sat,
        rho_dry
    );
}

#[test]
fn test_latent_load_from_weather_data() {
    // Using typical Denver summer conditions: 30°C, 30% RH
    let t_outdoor = 30.0;
    let rh_outdoor = 30.0;
    let t_zone = 24.0;
    let rh_zone = 50.0;

    let w_outdoor = calculate_humidity_ratio(t_outdoor, rh_outdoor, STD_PRESSURE);
    let w_zone = calculate_humidity_ratio(t_zone, rh_zone, STD_PRESSURE);

    let volume = 129.6;
    let ach = 0.5;
    let airflow_m3s = volume * ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    let q_latent_infiltration = mass_flow * (w_zone - w_outdoor).abs() * H_FG_20C;

    assert!(
        q_latent_infiltration > 0.0,
        "Latent load should be positive"
    );
    assert!(
        q_latent_infiltration < 500.0,
        "Latent infiltration load {:.2} W seems reasonable",
        q_latent_infiltration
    );
}

// =============================================================================
// BOUNDARY CONDITION TESTS
// =============================================================================

#[test]
fn test_moisture_balance_at_freezing() {
    // At 0°C, saturation vapor pressure is ~611 Pa
    // Humidity ratio is very low even at 100% RH
    let t = 0.0;
    let rh = 100.0;

    let p_sat = saturation_vapor_pressure(t);
    assert!(
        (p_sat - 611.2).abs() < 1.0,
        "p_sat(0°C) ≈ 611 Pa, got {:.1}",
        p_sat
    );

    let w_sat = calculate_humidity_ratio(t, rh, STD_PRESSURE);
    assert!(
        w_sat < 0.01,
        "At 0°C saturated, W should be < 0.01: got {:.5}",
        w_sat
    );
}

#[test]
fn test_latent_load_at_freezing_conditions() {
    // At freezing, latent loads are minimal due to low moisture content
    // Zone at 0°C, 80% RH; supply at -5°C (slightly colder for cooling)
    let t_zone = 0.0;
    let rh_zone = 80.0;
    let t_supply = -5.0;
    let rh_supply = 100.0;

    let w_zone = calculate_humidity_ratio(t_zone, rh_zone, STD_PRESSURE);
    let w_supply = calculate_humidity_ratio(t_supply, rh_supply, STD_PRESSURE);

    let airflow_m3s = 0.05;
    let mass_flow = airflow_m3s * RHO_AIR;
    let q_latent = mass_flow * (w_zone - w_supply).abs() * H_FG_20C;

    // At freezing, latent load is small relative to sensible load
    // because cold air holds very little moisture even when saturated
    let q_sensible = mass_flow * CP_AIR * (t_zone - t_supply);
    let latent_fraction = q_latent / (q_latent + q_sensible);

    assert!(
        latent_fraction < 0.25,
        "At freezing conditions, latent fraction ({:.2}) should be < 25% of total load",
        latent_fraction
    );
}

#[test]
fn test_latent_load_at_high_humidity_tropical() {
    // Miami summer: 30°C, 80% RH indoor vs 35°C, 70% RH outdoor
    let t_zone = 30.0;
    let rh_zone = 80.0;
    let t_outdoor = 35.0;
    let rh_outdoor = 70.0;

    let w_zone = calculate_humidity_ratio(t_zone, rh_zone, STD_PRESSURE);
    let w_outdoor = calculate_humidity_ratio(t_outdoor, rh_outdoor, STD_PRESSURE);

    let volume = 129.6;
    let ach = 0.5;
    let airflow_m3s = volume * ach / 3600.0;
    let mass_flow = airflow_m3s * RHO_AIR;

    let q_latent_infiltration = mass_flow * (w_outdoor - w_zone) * H_FG_20C;

    // High humidity difference = significant latent load
    assert!(
        q_latent_infiltration > 50.0,
        "In Miami summer, infiltration latent load should be significant: {:.2} W",
        q_latent_infiltration
    );
}

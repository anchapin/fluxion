//! Test scaffolds for ASHRAE 140 Case 900 reference values
//!
//! This module provides failing tests (TDD RED phase) that define expected behavior
//! for Case 900 (high-mass concrete building) validation against ASHRAE 140 reference values.
//!
//! Context: Phase 2 addresses thermal mass dynamics, specifically targeting Case 900
//! validation issues that remain after Phase 1 improvements.
//!
//! Research insight from Phase 1:
//! - Case 900FF shows under-damped behavior (max 37.52°C vs reference 41.8-46.4°C)
//! - Temperature swing reduction (~19.6% narrower than 600FF) not captured correctly
//! - Thermal mass dynamics need proper integration methods

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

/// ASHRAE 140 Case 900 specifications (high-mass concrete building)
///
/// Case 900 is the high-mass version of Case 600, with thick concrete walls
/// and floors that provide significant thermal mass. This case is critical
/// for validating thermal mass dynamics in the 5R1C thermal network.
///
/// Building Characteristics:
/// - Floor Area: 48 m² (8m × 6m)
/// - Wall Area: 75.6 m²
/// - Wall Construction: 200mm concrete + 50mm insulation
/// - Roof Construction: 200mm concrete slab
/// - Window-to-Wall Ratio: 0.15 (same as Case 600)
///
/// Reference Values (ASHRAE 140 Standard):
/// - Annual Heating: 1.17 - 2.04 MWh
/// - Annual Cooling: 2.13 - 3.67 MWh
/// - Peak Heating: 1.10 - 2.10 kW
/// - Peak Cooling: 2.10 - 3.50 kW
/// - Free-Floating Min: -6.40 to -1.60°C
/// - Free-Floating Max: 41.80 to 46.40°C

/// Reference ranges for Case 900 (ASHRAE 140)
#[derive(Debug, Clone)]
struct Case900Reference {
    /// Annual heating energy (MWh)
    annual_heating: (f64, f64),  // (min, max)

    /// Annual cooling energy (MWh)
    annual_cooling: (f64, f64),  // (min, max)

    /// Peak heating load (kW)
    peak_heating: (f64, f64),    // (min, max)

    /// Peak cooling load (kW)
    peak_cooling: (f64, f64),    // (min, max)

    /// Free-floating minimum temperature (°C)
    free_floating_min: (f64, f64), // (min, max)

    /// Free-floating maximum temperature (°C)
    free_floating_max: (f64, f64), // (min, max)

    /// Temperature swing reduction vs Case 600FF (%)
    swing_reduction: f64,
}

/// ASHRAE 140 reference values for Case 900
const CASE_900_REFERENCE: Case900Reference = Case900Reference {
    annual_heating: (1.17, 2.04),    // MWh
    annual_cooling: (2.13, 3.67),    // MWh
    peak_heating: (1.10, 2.10),      // kW
    peak_cooling: (2.10, 3.50),      // kW
    free_floating_min: (-6.40, -1.60), // °C
    free_floating_max: (41.80, 46.40), // °C
    swing_reduction: 19.6,           // % reduction vs 600FF
};

/// Tolerance for annual energy validation (±15% as per ASHRAE 140)
const ANNUAL_ENERGY_TOLERANCE: f64 = 0.15;

/// Tolerance for monthly energy validation (±10% as per ASHRAE 140)
const MONTHLY_ENERGY_TOLERANCE: f64 = 0.10;

/// Tolerance for peak loads (±10% as per ASHRAE 140)
const PEAK_LOAD_TOLERANCE: f64 = 0.10;

/// Tolerance for free-floating temperatures (±5% of reference range)
const TEMP_TOLERANCE: f64 = 0.05;

/// Convert energy from J to MWh (1 MWh = 3.6e9 J)
const J_TO_MWH: f64 = 1.0 / 3.6e9;

/// Convert power from W to kW (1 kW = 1000 W)
const W_TO_KW: f64 = 1.0 / 1000.0;

/// Simulate Case 900 for 1 year with HVAC
/// Returns: (annual_heating_J, annual_cooling_J, peak_heating_W, peak_cooling_W)
fn simulate_case_900() -> (f64, f64, f64, f64) {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    // Simulate 1 year (8760 hours)
    let steps = 8760;

    // Track energy and peak loads
    let mut total_heating = 0.0_f64;
    let mut total_cooling = 0.0_f64;
    let mut peak_heating = 0.0_f64;
    let mut peak_cooling = 0.0_f64;

    // Track solar gains for diagnostics
    let mut total_solar_gain = 0.0_f64;
    let mut peak_solar_gain = 0.0_f64;
    let mut summer_solar_gain = 0.0_f64;
    let mut summer_hours = 0_usize;

    // Track zone temperatures for diagnostics
    let mut min_zone_temp = f64::MAX;
    let mut max_zone_temp = f64::MIN;
    let mut summer_min_zone_temp = f64::MAX;
    let mut summer_max_zone_temp = f64::MIN;

    // Run simulation
    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        // Set weather data on model for solar gain calculation
        model.weather = Some(weather_data.clone());

        // Get zone temperature before HVAC to determine if heating or cooling is needed
        let zone_temp_before = model.temperatures.as_slice().first().copied().unwrap_or(20.0);

        // Run physics step (returns HVAC energy in kWh, positive for heating, negative for cooling)
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp);
        let energy_joules = energy_kwh * 3.6e6;  // Convert kWh to Joules

        // Track solar gains for diagnostics
        let solar_gain_watts = model.solar_gains.as_slice().first().copied().unwrap_or(0.0) * model.zone_area.as_slice().first().copied().unwrap_or(1.0);
        total_solar_gain += solar_gain_watts;  // This is in Watts, will convert to MWh later
        peak_solar_gain = peak_solar_gain.max(solar_gain_watts);

        // Track summer solar gains (June-August)
        let month = fluxion::sim::engine::ThermalModel::<VectorField>::timestep_to_date(step).1;
        if month >= 6 && month <= 8 {
            summer_solar_gain += solar_gain_watts;
            summer_hours += 1;
        }

        // Track zone temperatures for diagnostics
        if let Some(&zone_temp) = model.temperatures.as_slice().first() {
            min_zone_temp = min_zone_temp.min(zone_temp);
            max_zone_temp = max_zone_temp.max(zone_temp);
            if month >= 6 && month <= 8 {
                summer_min_zone_temp = summer_min_zone_temp.min(zone_temp);
                summer_max_zone_temp = summer_max_zone_temp.max(zone_temp);
            }
        }

        // Separate heating and cooling based on energy sign and zone temperature
        // Heating: energy > 0 or zone temp below heating setpoint
        // Cooling: energy < 0 or zone temp above cooling setpoint
        if energy_kwh > 0.0 || zone_temp_before < model.heating_setpoint {
            total_heating += energy_joules;
            let power_watts = energy_joules / 3600.0;  // Convert J/h to W
            peak_heating = peak_heating.max(power_watts);
        } else if energy_kwh < 0.0 || zone_temp_before > model.cooling_setpoint {
            total_cooling += -energy_joules;  // Cooling energy is negative
            let power_watts = -energy_joules / 3600.0;  // Convert J/h to W
            peak_cooling = peak_cooling.max(power_watts);
        }
    }

    let summer_avg_solar = if summer_hours > 0 { summer_solar_gain / summer_hours as f64 } else { 0.0 };

    println!("=== Solar Gain Diagnostics ===");
    println!("Total annual solar gain (raw): {:.2} W*h", total_solar_gain);
    println!("Total annual solar gain: {:.2} MWh", total_solar_gain / 1e6);  // W*h to MWh
    println!("Peak solar gain: {:.2} kW", peak_solar_gain / 1000.0);
    println!("Summer average solar gain: {:.2} kW", summer_avg_solar / 1000.0);
    println!("Summer hours tracked: {}", summer_hours);
    println!("=== Zone Temperature Diagnostics ===");
    println!("Min zone temp: {:.2}°C", min_zone_temp);
    println!("Max zone temp: {:.2}°C", max_zone_temp);
    println!("Summer min zone temp: {:.2}°C", summer_min_zone_temp);
    println!("Summer max zone temp: {:.2}°C", summer_max_zone_temp);

    (total_heating, total_cooling, peak_heating, peak_cooling)
}

/// Simulate Case 900FF (free-floating) for 1 year
/// Returns: (min_temp, max_temp, avg_temp)
fn simulate_case_900ff() -> (f64, f64, f64) {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    // Simulate 1 year (8760 hours)
    let steps = 8760;

    // Track temperatures
    let mut min_temp = f64::MAX;
    let mut max_temp = f64::MIN;
    let mut sum_temp = 0.0;

    // Run simulation
    for _step in 0..steps {
        let weather_data = weather.get_hourly_data(_step).unwrap();
        // Set weather data on model for solar gain calculation
        model.weather = Some(weather_data.clone());
        model.step_physics(_step, weather_data.dry_bulb_temp);

        // Get current zone temperature
        if let Some(&zone_temp) = model.temperatures.as_slice().first() {
            min_temp = min_temp.min(zone_temp);
            max_temp = max_temp.max(zone_temp);
            sum_temp += zone_temp;
        }
    }

    let avg_temp = sum_temp / steps as f64;
    (min_temp, max_temp, avg_temp)
}

/// Calculate temperature swing (max - min)
fn calculate_temperature_swing(min_temp: f64, max_temp: f64) -> f64 {
    max_temp - min_temp
}

#[test]
fn test_case_900_annual_heating_within_reference_range() {
    // Test 1: Case 900 annual heating energy within reference range [1.17, 2.04] MWh

    let (annual_heating_j, _, _, _) = simulate_case_900();
    let annual_heating_mwh = annual_heating_j * J_TO_MWH;

    let (ref_min, ref_max) = CASE_900_REFERENCE.annual_heating;
    let tolerance = (ref_max - ref_min) * ANNUAL_ENERGY_TOLERANCE;

    println!("Case 900 Annual Heating: {:.2} MWh", annual_heating_mwh);
    println!("Reference Range: [{:.2}, {:.2}] MWh", ref_min, ref_max);
    println!("Tolerance: ±{:.2} MWh", tolerance);

    // This test will fail until thermal mass dynamics are corrected
    assert!(
        annual_heating_mwh >= ref_min - tolerance && annual_heating_mwh <= ref_max + tolerance,
        "Annual heating {:.2} MWh outside reference range [{:.2}, {:.2}] MWh (±15% tolerance)",
        annual_heating_mwh,
        ref_min,
        ref_max
    );

    println!("✅ Test 1 PASSED: Annual heating within reference range");
}

#[test]
fn test_case_900_annual_cooling_within_reference_range() {
    // Test 2: Case 900 annual cooling energy within reference range [2.13, 3.67] MWh

    let (_, annual_cooling_j, _, _) = simulate_case_900();
    let annual_cooling_mwh = annual_cooling_j * J_TO_MWH;

    let (ref_min, ref_max) = CASE_900_REFERENCE.annual_cooling;
    let tolerance = (ref_max - ref_min) * ANNUAL_ENERGY_TOLERANCE;

    println!("Case 900 Annual Cooling: {:.2} MWh", annual_cooling_mwh);
    println!("Reference Range: [{:.2}, {:.2}] MWh", ref_min, ref_max);
    println!("Tolerance: ±{:.2} MWh", tolerance);

    // This test will fail until thermal mass dynamics are corrected
    assert!(
        annual_cooling_mwh >= ref_min - tolerance && annual_cooling_mwh <= ref_max + tolerance,
        "Annual cooling {:.2} MWh outside reference range [{:.2}, {:.2}] MWh (±15% tolerance)",
        annual_cooling_mwh,
        ref_min,
        ref_max
    );

    println!("✅ Test 2 PASSED: Annual cooling within reference range");
}

#[test]
fn test_case_900_peak_heating_within_reference_range() {
    // Test 3: Case 900 peak heating load within reference range [1.10, 2.10] kW

    let (_, _, peak_heating_w, _) = simulate_case_900();
    let peak_heating_kw = peak_heating_w * W_TO_KW;

    let (ref_min, ref_max) = CASE_900_REFERENCE.peak_heating;
    let tolerance = (ref_max - ref_min) * PEAK_LOAD_TOLERANCE;

    println!("Case 900 Peak Heating: {:.2} kW", peak_heating_kw);
    println!("Reference Range: [{:.2}, {:.2}] kW", ref_min, ref_max);
    println!("Tolerance: ±{:.2} kW", tolerance);

    // This test will fail until thermal mass dynamics are corrected
    assert!(
        peak_heating_kw >= ref_min - tolerance && peak_heating_kw <= ref_max + tolerance,
        "Peak heating {:.2} kW outside reference range [{:.2}, {:.2}] kW (±10% tolerance)",
        peak_heating_kw,
        ref_min,
        ref_max
    );

    println!("✅ Test 3 PASSED: Peak heating within reference range");
}

#[test]
fn test_case_900_peak_cooling_within_reference_range() {
    // Test 4: Case 900 peak cooling load within reference range [2.10, 3.50] kW

    let (_, _, _, peak_cooling_w) = simulate_case_900();
    let peak_cooling_kw = peak_cooling_w * W_TO_KW;

    let (ref_min, ref_max) = CASE_900_REFERENCE.peak_cooling;
    let tolerance = (ref_max - ref_min) * PEAK_LOAD_TOLERANCE;

    println!("Case 900 Peak Cooling: {:.2} kW", peak_cooling_kw);
    println!("Reference Range: [{:.2}, {:.2}] kW", ref_min, ref_max);
    println!("Tolerance: ±{:.2} kW", tolerance);

    // This test will fail until thermal mass dynamics are corrected
    assert!(
        peak_cooling_kw >= ref_min - tolerance && peak_cooling_kw <= ref_max + tolerance,
        "Peak cooling {:.2} kW outside reference range [{:.2}, {:.2}] kW (±10% tolerance)",
        peak_cooling_kw,
        ref_min,
        ref_max
    );

    println!("✅ Test 4 PASSED: Peak cooling within reference range");
}

#[test]
fn test_case_900ff_min_temperature_within_reference_range() {
    // Test 5: Case 900FF minimum temperature within reference range [-6.40, -1.60]°C

    let (min_temp, _, _) = simulate_case_900ff();

    let (ref_min, ref_max) = CASE_900_REFERENCE.free_floating_min;
    let tolerance = (ref_max - ref_min) * TEMP_TOLERANCE;

    println!("Case 900FF Min Temperature: {:.2}°C", min_temp);
    println!("Reference Range: [{:.2}, {:.2}]°C", ref_min, ref_max);
    println!("Tolerance: ±{:.2}°C", tolerance);

    // This test will fail until thermal mass dynamics are corrected
    assert!(
        min_temp >= ref_min - tolerance && min_temp <= ref_max + tolerance,
        "Min temperature {:.2}°C outside reference range [{:.2}, {:.2}]°C (±5% tolerance)",
        min_temp,
        ref_min,
        ref_max
    );

    println!("✅ Test 5 PASSED: Min temperature within reference range");
}

#[test]
fn test_case_900ff_max_temperature_within_reference_range() {
    // Test 6: Case 900FF maximum temperature within reference range [41.80, 46.40]°C

    let (_, max_temp, _) = simulate_case_900ff();

    let (ref_min, ref_max) = CASE_900_REFERENCE.free_floating_max;
    let tolerance = (ref_max - ref_min) * TEMP_TOLERANCE;

    println!("Case 900FF Max Temperature: {:.2}°C", max_temp);
    println!("Reference Range: [{:.2}, {:.2}]°C", ref_min, ref_max);
    println!("Tolerance: ±{:.2}°C", tolerance);

    // This test will fail until thermal mass dynamics are corrected
    assert!(
        max_temp >= ref_min - tolerance && max_temp <= ref_max + tolerance,
        "Max temperature {:.2}°C outside reference range [{:.2}, {:.2}]°C (±5% tolerance)",
        max_temp,
        ref_min,
        ref_max
    );

    println!("✅ Test 6 PASSED: Max temperature within reference range");
}

#[test]
fn test_case_900ff_temperature_swing_reduction() {
    // Test 7: Case 900FF temperature swing shows ~19.6% reduction vs 600FF

    // Simulate Case 900FF
    let (min_temp_900, max_temp_900, _) = simulate_case_900ff();
    let swing_900 = calculate_temperature_swing(min_temp_900, max_temp_900);

    // Simulate Case 600FF (low-mass baseline) using the same method
    let spec_600 = ASHRAE140Case::Case600FF.spec();
    let mut model_600 = ThermalModel::<VectorField>::from_spec(&spec_600);
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    let mut min_temp_600 = f64::MAX;
    let mut max_temp_600 = f64::MIN;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model_600.weather = Some(weather_data.clone());
        model_600.step_physics(step, weather_data.dry_bulb_temp);

        if let Some(&zone_temp) = model_600.temperatures.as_slice().first() {
            min_temp_600 = min_temp_600.min(zone_temp);
            max_temp_600 = max_temp_600.max(zone_temp);
        }
    }

    let swing_600 = calculate_temperature_swing(min_temp_600, max_temp_600);

    // Calculate swing reduction
    let swing_reduction = ((swing_600 - swing_900) / swing_600) * 100.0;
    let expected_reduction = CASE_900_REFERENCE.swing_reduction;
    let tolerance = 5.0;  // ±5% tolerance

    println!("Temperature Swing Comparison:");
    println!("  Case 600FF: {:.2}°C", swing_600);
    println!("  Case 900FF: {:.2}°C", swing_900);
    println!("  Reduction: {:.1}% (expected: ~{:.1}%)", swing_reduction, expected_reduction);

    // This test will fail until thermal mass dynamics are corrected
    assert!(
        (swing_reduction - expected_reduction).abs() <= tolerance,
        "Temperature swing reduction {:.1}% outside expected range [±{:.1}%] around {:.1}%",
        swing_reduction,
        tolerance,
        expected_reduction
    );

    println!("✅ Test 7 PASSED: Temperature swing reduction within expected range");
}

#[test]
fn test_case_900_thermal_mass_characteristics() {
    // Verify that Case 900 has the expected thermal mass characteristics

    let spec = ASHRAE140Case::Case900.spec();

    // Calculate thermal capacitance
    let wall_cap = spec.construction.wall.thermal_capacitance_per_area();
    let roof_cap = spec.construction.roof.thermal_capacitance_per_area();
    let floor_cap = spec.construction.floor.thermal_capacitance_per_area();

    let floor_area = spec.geometry[0].floor_area();
    let wall_area = spec.geometry[0].wall_area();

    let total_wall = wall_cap * wall_area;
    let total_roof = roof_cap * floor_area;
    let total_floor = floor_cap * floor_area;
    let total_cap = total_wall + total_roof + total_floor;

    println!("=== Case 900 Thermal Mass Characteristics ===");
    println!("Floor Area: {:.2} m²", floor_area);
    println!("Wall Area: {:.2} m²", wall_area);
    println!();
    println!("Thermal Capacitance per Area:");
    println!("  Wall: {:.2} kJ/m²K", wall_cap / 1000.0);
    println!("  Roof: {:.2} kJ/m²K", roof_cap / 1000.0);
    println!("  Floor: {:.2} kJ/m²K", floor_cap / 1000.0);
    println!();
    println!("Total Thermal Capacitance: {:.2} kJ/K", total_cap / 1000.0);

    // Verify high thermal mass (>500 kJ/K)
    assert!(
        total_cap > 500_000.0,  // 500 kJ/K
        "Case 900 should have high thermal capacitance (>500 kJ/K), got {:.2} kJ/K",
        total_cap / 1000.0
    );

    println!();
    println!("✅ Case 900 has expected high thermal mass characteristics");
}

fn main() {
    println!("=== ASHRAE 140 Case 900 Reference Values Test Suite ===\n");
    println!("Purpose: TDD RED phase - create failing tests for Case 900 validation");
    println!("Context: Phase 2 addresses high-mass building validation (Case 900, 900FF)");
    println!("Issue: Case 900 shows under-damped behavior, incorrect temperature swing");
    println!("Solution: Implement proper thermal mass integration and conductance validation\n");

    println!("Reference Values (ASHRAE 140):");
    println!("  Annual Heating: [{:.2}, {:.2}] MWh",
             CASE_900_REFERENCE.annual_heating.0, CASE_900_REFERENCE.annual_heating.1);
    println!("  Annual Cooling: [{:.2}, {:.2}] MWh",
             CASE_900_REFERENCE.annual_cooling.0, CASE_900_REFERENCE.annual_cooling.1);
    println!("  Peak Heating: [{:.2}, {:.2}] kW",
             CASE_900_REFERENCE.peak_heating.0, CASE_900_REFERENCE.peak_heating.1);
    println!("  Peak Cooling: [{:.2}, {:.2}] kW",
             CASE_900_REFERENCE.peak_cooling.0, CASE_900_REFERENCE.peak_cooling.1);
    println!("  Free-Floating Min: [{:.2}, {:.2}]°C",
             CASE_900_REFERENCE.free_floating_min.0, CASE_900_REFERENCE.free_floating_min.1);
    println!("  Free-Floating Max: [{:.2}, {:.2}]°C",
             CASE_900_REFERENCE.free_floating_max.0, CASE_900_REFERENCE.free_floating_max.1);
    println!("  Temperature Swing Reduction: ~{:.1}%", CASE_900_REFERENCE.swing_reduction);
    println!();

    println!("Running tests...\n");
}

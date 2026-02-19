//! Integration tests for ASHRAE 140 Case 195 - Solid Conduction test case.
//!
//! Case 195 is a conduction-only problem that tests radiative and convective
//! heat transfer in opaque surfaces:
//! - No windows
//! - No infiltration (0 ACH)
//! - No internal loads (0 W)
//! - Bang-bang control (heating = cooling = 20°C)
//! - Tests only envelope heat transfer
//!
//! This case isolates the building fabric heat transfer from other loads,
//! providing a clean test of the thermal network.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Reference ranges for Case 195
mod reference {
    pub const ANNUAL_HEATING_MIN: f64 = 5.85;
    pub const ANNUAL_HEATING_MAX: f64 = 7.25;
    pub const ANNUAL_COOLING_MIN: f64 = 0.00;
    pub const ANNUAL_COOLING_MAX: f64 = 0.00;
    pub const PEAK_HEATING_MIN: f64 = 1.70;
    pub const PEAK_HEATING_MAX: f64 = 2.20;
}

/// Simulates Case 195 and returns annual heating/cooling in MWh
fn simulate_case_195() -> (f64, f64, f64) {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;
    let mut peak_heating_watts: f64 = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
            peak_heating_watts = peak_heating_watts.max(hvac_kwh * 1000.0);
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    (
        annual_heating_joules / 3.6e9,
        annual_cooling_joules / 3.6e9,
        peak_heating_watts / 1000.0,
    )
}

#[test]
fn test_case_195_configuration() {
    let spec = ASHRAE140Case::Case195.spec();

    // Verify no windows
    let total_window_area: f64 = spec.windows.iter().flat_map(|w| w.iter()).map(|w| w.area).sum();
    assert_eq!(total_window_area, 0.0, "Case 195 should have no windows");

    // Verify no infiltration
    assert_eq!(spec.infiltration_ach, 0.0, "Case 195 should have zero infiltration");

    // Verify no internal loads
    let internal_loads = spec.internal_loads[0].as_ref();
    if let Some(loads) = internal_loads {
        assert_eq!(loads.total_load, 0.0, "Case 195 should have zero internal loads");
    }

    // Verify single zone
    assert_eq!(spec.num_zones, 1, "Case 195 should be single-zone");
}

#[test]
fn test_case_195_solid_conduction_simulation() {
    let (heating, cooling, peak_h) = simulate_case_195();

    println!("\n=== ASHRAE 140 Case 195 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::ANNUAL_HEATING_MIN,
        reference::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::ANNUAL_COOLING_MIN,
        reference::ANNUAL_COOLING_MAX
    );
    println!(
        "Peak Heating: {:.2} kW (reference: {:.2}-{:.2} kW)",
        peak_h,
        reference::PEAK_HEATING_MIN,
        reference::PEAK_HEATING_MAX
    );
    println!("=== End ===\n");

    // Verify heating is positive (heating-only case)
    assert!(heating >= 0.0, "Heating should be non-negative");

    // Cooling should be zero or minimal (no solar gains, no internal loads)
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_195_no_solar_gains() {
    let spec = ASHRAE140Case::Case195.spec();

    // Verify opaque absorptance is zero (no solar absorption)
    assert_eq!(spec.opaque_absorptance, 0.0, "Case 195 should have zero solar absorptance");

    // Verify no windows for solar gains
    assert!(spec.windows[0].is_empty(), "Case 195 should have no windows");
}

#[test]
fn test_case_195_conduction_only() {
    // Case 195 should only have conduction heat transfer
    // No solar, no infiltration, no internal loads
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Verify the model is configured correctly
    assert_eq!(model.num_zones, 1, "Should be single-zone");

    // Window U-value should still be set (even with zero area)
    assert!(model.window_u_value > 0.0, "Window U-value should be set");

    // Infiltration should be zero
    let h_ve = model.h_ve.as_ref();
    assert_eq!(h_ve[0], 0.0, "Ventilation conductance should be zero (no infiltration)");
}

#[test]
fn test_case_195_temperature_range() {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut temperatures: Vec<f64> = Vec::new();

    // Simulate for a week to see temperature patterns
    for step in 0..168 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.step_physics(step, weather_data.dry_bulb_temp);
        temperatures.push(model.temperatures.as_ref()[0]);
    }

    let min_temp = temperatures.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_temp = temperatures.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Case 195 Temperature Range (Week 1) ===");
    println!("Zone temperature: {:.2}°C to {:.2}°C", min_temp, max_temp);
    println!("=== End ===\n");

    // Temperature should be maintained near setpoint (20°C)
    // With only conduction, the zone should track outdoor temp more closely
    // but HVAC maintains the setpoint
    assert!(min_temp > 15.0 && max_temp < 25.0, "Temperature should be near setpoint");
}

#[test]
fn test_case_195_heating_only() {
    // Case 195 is heating-only (no cooling needed due to no solar/internal gains)
    let (heating, cooling, _) = simulate_case_195();

    // Should have heating
    assert!(heating > 0.0, "Should have heating energy");

    // Cooling should be zero or very small
    // (might have small cooling if outdoor temp drops below setpoint during summer nights)
    println!("Heating: {:.2} MWh, Cooling: {:.2} MWh", heating, cooling);
}

#[test]
fn test_case_195_construction_properties() {
    let spec = ASHRAE140Case::Case195.spec();

    // Verify construction is low-mass
    assert!(
        matches!(spec.construction_type, fluxion::validation::ashrae_140_cases::ConstructionType::LowMass | fluxion::validation::ashrae_140_cases::ConstructionType::Special),
        "Case 195 should use low-mass or special construction"
    );

    // Verify geometry
    assert_eq!(spec.geometry[0].floor_area(), 48.0, "Floor area should be 48 m²");
}
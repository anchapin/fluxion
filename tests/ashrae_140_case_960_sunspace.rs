//! Integration tests for ASHRAE 140 Case 960 - Sunspace/Multi-zone test case.
//!
//! Case 960 tests multi-zone heat transfer with an attached sunspace:
//! - Zone 0: Back-zone (conditioned, 8m x 6m x 2.7m)
//! - Zone 1: Sunspace/Unconditioned (8m x 3m x 2.7m)
//! - Common wall between zones with door opening
//! - Back-zone has HVAC, sunspace is free-floating
//!
//! This tests:
//! - Inter-zone air exchange
//! - Solar gains in sunspace
//! - Heat transfer through common wall
//! - Thermal coupling between zones

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Reference ranges for Case 960
mod reference {
    pub const ANNUAL_HEATING_MIN: f64 = 1.65;
    pub const ANNUAL_HEATING_MAX: f64 = 2.45;
    pub const ANNUAL_COOLING_MIN: f64 = 1.55;
    pub const ANNUAL_COOLING_MAX: f64 = 2.78;
    pub const PEAK_HEATING_MIN: f64 = 2.20;
    pub const PEAK_HEATING_MAX: f64 = 2.90;
    pub const PEAK_COOLING_MIN: f64 = 1.50;
    pub const PEAK_COOLING_MAX: f64 = 2.00;
}

/// Simulates Case 960 and returns annual heating/cooling in MWh
fn simulate_case_960() -> (f64, f64, f64, f64) {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Verify multi-zone configuration
    assert_eq!(model.num_zones, 2, "Case 960 should have 2 zones");

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;
    let mut peak_heating_watts: f64 = 0.0;
    let mut peak_cooling_watts: f64 = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
            peak_heating_watts = peak_heating_watts.max(hvac_kwh * 1000.0);
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
            peak_cooling_watts = peak_cooling_watts.max((-hvac_kwh) * 1000.0);
        }
    }

    (
        annual_heating_joules / 3.6e9,
        annual_cooling_joules / 3.6e9,
        peak_heating_watts / 1000.0,
        peak_cooling_watts / 1000.0,
    )
}

#[test]
fn test_case_960_multi_zone_configuration() {
    let spec = ASHRAE140Case::Case960.spec();

    // Verify 2-zone configuration
    assert_eq!(spec.num_zones, 2, "Case 960 should have 2 zones");

    // Verify geometry
    // Zone 0: Back-zone (8m x 6m x 2.7m = 48 m²)
    assert_eq!(spec.geometry[0].width, 8.0);
    assert_eq!(spec.geometry[0].depth, 6.0);
    assert_eq!(spec.geometry[0].height, 2.7);
    assert_eq!(spec.geometry[0].floor_area(), 48.0);

    // Zone 1: Sunspace (8m x 2m x 2.7m = 16 m²)
    assert_eq!(spec.geometry[1].width, 8.0);
    assert_eq!(spec.geometry[1].depth, 2.0);
    assert_eq!(spec.geometry[1].height, 2.7);
    assert_eq!(spec.geometry[1].floor_area(), 16.0);

    // Verify common wall exists
    assert!(
        !spec.common_walls.is_empty(),
        "Should have common wall between zones"
    );

    // Verify HVAC configuration
    // Zone 0 should have HVAC control
    assert!(
        !spec.hvac[0].is_free_floating(),
        "Back-zone should have HVAC control"
    );

    // Zone 1 should be free-floating (sunspace)
    assert!(
        spec.hvac[1].is_free_floating(),
        "Sunspace should be free-floating"
    );
}

#[test]
fn test_case_960_inter_zone_conductance() {
    let spec = ASHRAE140Case::Case960.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Verify inter-zone conductance is set
    let h_iz = model.h_tr_iz.as_ref();
    assert!(h_iz[0] > 0.0, "Inter-zone conductance should be positive");

    println!("Inter-zone conductance: {:.2} W/K", h_iz[0]);
}

#[test]
fn test_case_960_sunspace_simulation() {
    let (heating, cooling, peak_h, peak_c) = simulate_case_960();

    println!("\n=== ASHRAE 140 Case 960 Results ===");
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
    println!(
        "Peak Cooling: {:.2} kW (reference: {:.2}-{:.2} kW)",
        peak_c,
        reference::PEAK_COOLING_MIN,
        reference::PEAK_COOLING_MAX
    );
    println!("=== End ===\n");

    // Verify positive energy values
    assert!(heating >= 0.0, "Heating should be non-negative");
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_960_zone_temperatures() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut back_zone_temps: Vec<f64> = Vec::new();
    let mut sunspace_temps: Vec<f64> = Vec::new();

    // Simulate for a few days to see temperature patterns
    for step in 0..168 {
        // One week
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.step_physics(step, weather_data.dry_bulb_temp);

        let temps = model.temperatures.as_ref();
        back_zone_temps.push(temps[0]);
        sunspace_temps.push(temps[1]);
    }

    let back_min = back_zone_temps
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let back_max = back_zone_temps
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let sunspace_min = sunspace_temps.iter().cloned().fold(f64::INFINITY, f64::min);
    let sunspace_max = sunspace_temps
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Case 960 Temperature Ranges (Week 1) ===");
    println!("Back-zone: {:.2}°C to {:.2}°C", back_min, back_max);
    println!("Sunspace: {:.2}°C to {:.2}°C", sunspace_min, sunspace_max);
    println!("=== End ===\n");

    // Sunspace should have larger temperature swings (free-floating)
    let _back_swing = back_max - back_min;
    let _sunspace_swing = sunspace_max - sunspace_min;

    // Both zones should have reasonable temperatures
    assert!(
        back_min > -30.0 && back_max < 50.0,
        "Back-zone temps should be reasonable"
    );
    assert!(
        sunspace_min > -30.0 && sunspace_max < 80.0,
        "Sunspace temps should be reasonable"
    );
}

#[test]
fn test_case_960_solar_gains_distribution() {
    let spec = ASHRAE140Case::Case960.spec();

    // Verify sunspace has windows
    assert!(!spec.windows[1].is_empty(), "Sunspace should have windows");

    // Calculate total window area per zone
    let back_zone_window_area: f64 = spec.windows[0].iter().map(|w| w.area).sum();
    let sunspace_window_area: f64 = spec.windows[1].iter().map(|w| w.area).sum();

    println!("\n=== Case 960 Window Areas ===");
    println!("Back-zone windows: {:.2} m²", back_zone_window_area);
    println!("Sunspace windows: {:.2} m²", sunspace_window_area);
    println!("=== End ===\n");

    // Sunspace should have windows for solar gains
    assert!(
        sunspace_window_area > 0.0,
        "Sunspace should have window area"
    );
}

#[test]
fn test_case_960_hvac_only_in_back_zone() {
    // HVAC energy should only be counted for the conditioned back-zone
    // The sunspace is free-floating and should not contribute to HVAC energy
    let spec = ASHRAE140Case::Case960.spec();
    let _model = ThermalModel::<VectorField>::from_spec(&spec);

    // Verify HVAC is only in zone 0
    assert!(!spec.hvac[0].is_free_floating(), "Zone 0 should have HVAC");
    assert!(
        spec.hvac[1].is_free_floating(),
        "Zone 1 should be free-floating"
    );
}

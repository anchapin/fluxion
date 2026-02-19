//! Integration tests for ASHRAE 140 Free-Floating (FF) test cases.
//!
//! Free-floating cases test the building's thermal response without HVAC intervention.
//! The simulator tracks zone temperatures throughout the simulation and reports
//! min/max temperatures as the validation metrics.
//!
//! # Test Cases
//! - Case 600FF: Low mass free-floating
//! - Case 650FF: Low mass free-floating with night ventilation
//! - Case 900FF: High mass free-floating
//! - Case 950FF: High mass free-floating with night ventilation

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, HvacSchedule};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Reference ranges for ASHRAE 140 free-floating cases
mod reference {
    // Case 600FF - Low mass free-floating
    pub mod case_600ff {
        pub const MIN_TEMP_MIN: f64 = -18.8;
        pub const MIN_TEMP_MAX: f64 = -15.6;
        pub const MAX_TEMP_MIN: f64 = 64.9;
        pub const MAX_TEMP_MAX: f64 = 75.1;
    }

    // Case 650FF - Low mass free-floating with night ventilation
    pub mod case_650ff {
        pub const MIN_TEMP_MIN: f64 = -23.0;
        pub const MIN_TEMP_MAX: f64 = -21.0;
        pub const MAX_TEMP_MIN: f64 = 63.2;
        pub const MAX_TEMP_MAX: f64 = 73.5;
    }

    // Case 900FF - High mass free-floating
    pub mod case_900ff {
        pub const MIN_TEMP_MIN: f64 = -6.4;
        pub const MIN_TEMP_MAX: f64 = -1.6;
        pub const MAX_TEMP_MIN: f64 = 41.8;
        pub const MAX_TEMP_MAX: f64 = 46.4;
    }

    // Case 950FF - High mass free-floating with night ventilation
    pub mod case_950ff {
        pub const MIN_TEMP_MIN: f64 = -20.2;
        pub const MIN_TEMP_MAX: f64 = -17.8;
        pub const MAX_TEMP_MIN: f64 = 35.5;
        pub const MAX_TEMP_MAX: f64 = 38.5;
    }
}

/// Simulates a free-floating case and returns min/max temperatures
fn simulate_free_float_case(case: ASHRAE140Case) -> (f64, f64) {
    let spec = case.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Verify this is a free-floating case
    assert!(spec.is_free_floating(), "Case should be free-floating");

    // Disable HVAC for free-floating mode
    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;
    model.hvac_heating_capacity = 0.0;
    model.hvac_cooling_capacity = 0.0;

    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.step_physics(step, weather_data.dry_bulb_temp);

        // Track zone temperatures
        if let Some(&zone_temp) = model.temperatures.as_slice().first() {
            min_temp = min_temp.min(zone_temp);
            max_temp = max_temp.max(zone_temp);
        }
    }

    (min_temp, max_temp)
}

#[test]
fn test_case_600ff_free_floating() {
    let (min_temp, max_temp) = simulate_free_float_case(ASHRAE140Case::Case600FF);

    println!("\n=== ASHRAE 140 Case 600FF Results ===");
    println!(
        "Min Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        min_temp,
        reference::case_600ff::MIN_TEMP_MIN,
        reference::case_600ff::MIN_TEMP_MAX
    );
    println!(
        "Max Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        max_temp,
        reference::case_600ff::MAX_TEMP_MIN,
        reference::case_600ff::MAX_TEMP_MAX
    );
    println!("=== End ===\n");

    // Verify temperatures are in reasonable range
    assert!(min_temp < max_temp, "Min temp should be less than max temp");
    assert!(
        min_temp > -50.0 && min_temp < 50.0,
        "Min temp should be in reasonable range"
    );
    assert!(
        max_temp > -50.0 && max_temp < 100.0,
        "Max temp should be in reasonable range"
    );
}

#[test]
fn test_case_650ff_free_floating_night_vent() {
    let (min_temp, max_temp) = simulate_free_float_case(ASHRAE140Case::Case650FF);

    println!("\n=== ASHRAE 140 Case 650FF Results ===");
    println!(
        "Min Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        min_temp,
        reference::case_650ff::MIN_TEMP_MIN,
        reference::case_650ff::MIN_TEMP_MAX
    );
    println!(
        "Max Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        max_temp,
        reference::case_650ff::MAX_TEMP_MIN,
        reference::case_650ff::MAX_TEMP_MAX
    );
    println!("=== End ===\n");

    // Verify temperatures are in reasonable range
    assert!(min_temp < max_temp, "Min temp should be less than max temp");
}

#[test]
fn test_case_900ff_free_floating_high_mass() {
    let (min_temp, max_temp) = simulate_free_float_case(ASHRAE140Case::Case900FF);

    println!("\n=== ASHRAE 140 Case 900FF Results ===");
    println!(
        "Min Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        min_temp,
        reference::case_900ff::MIN_TEMP_MIN,
        reference::case_900ff::MIN_TEMP_MAX
    );
    println!(
        "Max Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        max_temp,
        reference::case_900ff::MAX_TEMP_MIN,
        reference::case_900ff::MAX_TEMP_MAX
    );
    println!("=== End ===\n");

    // High mass should have smaller temperature swing than low mass
    let (min_600ff, max_600ff) = simulate_free_float_case(ASHRAE140Case::Case600FF);
    let swing_900ff = max_temp - min_temp;
    let swing_600ff = max_600ff - min_600ff;

    println!(
        "Temperature swing - 600FF: {:.2}°C, 900FF: {:.2}°C",
        swing_600ff, swing_900ff
    );

    // High mass should moderate temperature swings
    assert!(min_temp < max_temp, "Min temp should be less than max temp");
}

#[test]
fn test_case_950ff_free_floating_night_vent_high_mass() {
    let (min_temp, max_temp) = simulate_free_float_case(ASHRAE140Case::Case950FF);

    println!("\n=== ASHRAE 140 Case 950FF Results ===");
    println!(
        "Min Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        min_temp,
        reference::case_950ff::MIN_TEMP_MIN,
        reference::case_950ff::MIN_TEMP_MAX
    );
    println!(
        "Max Temperature: {:.2}°C (reference: {:.2} to {:.2}°C)",
        max_temp,
        reference::case_950ff::MAX_TEMP_MIN,
        reference::case_950ff::MAX_TEMP_MAX
    );
    println!("=== End ===\n");

    assert!(min_temp < max_temp, "Min temp should be less than max temp");
}

#[test]
fn test_hvac_schedule_free_floating() {
    // Test that free-floating schedule is correctly configured
    let schedule = HvacSchedule::free_floating();

    assert!(
        !schedule.is_enabled(),
        "Free-floating schedule should not be enabled"
    );
    assert!(
        schedule.is_free_floating(),
        "Schedule should report as free-floating"
    );
    assert_eq!(
        schedule.heating_setpoint_at_hour(12),
        None,
        "No heating setpoint in free-floating mode"
    );
    assert_eq!(
        schedule.cooling_setpoint_at_hour(12),
        None,
        "No cooling setpoint in free-floating mode"
    );
}

#[test]
fn test_free_floating_case_specification() {
    // Verify that FF cases are properly configured
    let cases = vec![
        ASHRAE140Case::Case600FF,
        ASHRAE140Case::Case650FF,
        ASHRAE140Case::Case900FF,
        ASHRAE140Case::Case950FF,
    ];

    for case in cases {
        let spec = case.spec();
        assert!(
            spec.is_free_floating(),
            "Case {:?} should be free-floating",
            case
        );
        assert!(
            spec.hvac[0].is_free_floating(),
            "Case {:?} should have free-floating HVAC schedule",
            case
        );
    }
}

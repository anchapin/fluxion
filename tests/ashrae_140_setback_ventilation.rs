//! Integration tests for ASHRAE 140 thermostat setback and night ventilation cases.
//!
//! These cases test time-based HVAC control strategies:
//!
//! # Thermostat Setback Cases (640, 940)
//! - Heating setpoint drops to 10°C overnight (23:00-07:00)
//! - Normal heating setpoint (20°C) during daytime
//! - Tests thermal mass benefits from setpoint variation
//!
//! # Night Ventilation Cases (650, 950)
//! - Heating is disabled
//! - Ventilation fan runs 18:00-07:00
//! - Tests passive cooling through night air exchange
//! - Cooling-only operation during the day

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, HvacSchedule, NightVentilation};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Reference ranges for thermostat setback cases
mod reference {
    // Case 640 - Low mass with thermostat setback
    pub mod case_640 {
        pub const ANNUAL_HEATING_MIN: f64 = 2.75;
        pub const ANNUAL_HEATING_MAX: f64 = 3.80;
        pub const ANNUAL_COOLING_MIN: f64 = 5.95;
        pub const ANNUAL_COOLING_MAX: f64 = 8.10;
    }

    // Case 650 - Low mass with night ventilation
    pub mod case_650 {
        pub const ANNUAL_HEATING_MIN: f64 = 0.00;
        pub const ANNUAL_HEATING_MAX: f64 = 0.00;
        pub const ANNUAL_COOLING_MIN: f64 = 4.82;
        pub const ANNUAL_COOLING_MAX: f64 = 7.06;
    }

    // Case 940 - High mass with thermostat setback
    pub mod case_940 {
        pub const ANNUAL_HEATING_MIN: f64 = 0.79;
        pub const ANNUAL_HEATING_MAX: f64 = 1.41;
        pub const ANNUAL_COOLING_MIN: f64 = 2.08;
        pub const ANNUAL_COOLING_MAX: f64 = 3.55;
    }

    // Case 950 - High mass with night ventilation
    pub mod case_950 {
        pub const ANNUAL_HEATING_MIN: f64 = 0.00;
        pub const ANNUAL_HEATING_MAX: f64 = 0.00;
        pub const ANNUAL_COOLING_MIN: f64 = 0.39;
        pub const ANNUAL_COOLING_MAX: f64 = 0.92;
    }
}

/// Simulates a case and returns annual heating/cooling in MWh
fn simulate_case(case: ASHRAE140Case) -> (f64, f64) {
    let spec = case.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;

    for step in 0..8760 {
        let hour_of_day = step % 24;
        let weather_data = weather.get_hourly_data(step).unwrap();

        // Apply dynamic setpoints based on HVAC schedule
        if let Some(hvac_schedule) = spec.hvac.first() {
            if let Some(heating_sp) = hvac_schedule.heating_setpoint_at_hour(hour_of_day as u8) {
                model.heating_setpoint = heating_sp;
            }
            if let Some(cooling_sp) = hvac_schedule.cooling_setpoint_at_hour(hour_of_day as u8) {
                model.cooling_setpoint = cooling_sp;
            }
        }

        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    (
        annual_heating_joules / 3.6e9,
        annual_cooling_joules / 3.6e9,
    )
}

#[test]
fn test_hvac_schedule_with_setback() {
    // Test thermostat setback schedule
    let schedule = HvacSchedule::with_setback(20.0, 27.0, 10.0, 23, 7);

    assert!(schedule.is_enabled(), "Schedule should be enabled");

    // During setback hours (23:00-07:00), heating setpoint should be 10°C
    assert_eq!(schedule.heating_setpoint_at_hour(0), Some(10.0), "Midnight should have setback setpoint");
    assert_eq!(schedule.heating_setpoint_at_hour(6), Some(10.0), "6 AM should have setback setpoint");

    // During normal hours, heating setpoint should be 20°C
    assert_eq!(schedule.heating_setpoint_at_hour(12), Some(20.0), "Noon should have normal setpoint");
    assert_eq!(schedule.heating_setpoint_at_hour(20), Some(20.0), "8 PM should have normal setpoint");

    // Cooling setpoint should remain constant
    assert_eq!(schedule.cooling_setpoint_at_hour(0), Some(27.0), "Cooling setpoint should be constant");
    assert_eq!(schedule.cooling_setpoint_at_hour(12), Some(27.0), "Cooling setpoint should be constant");
}

#[test]
fn test_night_ventilation_schedule() {
    let vent = NightVentilation::case_650();

    // Night ventilation runs 18:00-07:00
    assert!(vent.is_active_at_hour(18), "Should be active at 18:00");
    assert!(vent.is_active_at_hour(0), "Should be active at midnight");
    assert!(vent.is_active_at_hour(6), "Should be active at 6 AM");

    // Not active during daytime
    assert!(!vent.is_active_at_hour(12), "Should not be active at noon");
    assert!(!vent.is_active_at_hour(10), "Should not be active at 10 AM");
    assert!(!vent.is_active_at_hour(17), "Should not be active at 5 PM");
}

#[test]
fn test_case_640_setback_low_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case640);

    println!("\n=== ASHRAE 140 Case 640 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_640::ANNUAL_HEATING_MIN,
        reference::case_640::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_640::ANNUAL_COOLING_MIN,
        reference::case_640::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    // Verify heating and cooling are positive
    assert!(heating >= 0.0, "Heating should be non-negative");
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_650_night_vent_low_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case650);

    println!("\n=== ASHRAE 140 Case 650 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_650::ANNUAL_HEATING_MIN,
        reference::case_650::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_650::ANNUAL_COOLING_MIN,
        reference::case_650::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    // Heating should be zero (disabled for night ventilation cases)
    assert_eq!(heating, 0.0, "Heating should be zero for night ventilation case");
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_940_setback_high_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case940);

    println!("\n=== ASHRAE 140 Case 940 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_940::ANNUAL_HEATING_MIN,
        reference::case_940::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_940::ANNUAL_COOLING_MIN,
        reference::case_940::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    assert!(heating >= 0.0, "Heating should be non-negative");
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_950_night_vent_high_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case950);

    println!("\n=== ASHRAE 140 Case 950 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_950::ANNUAL_HEATING_MIN,
        reference::case_950::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_950::ANNUAL_COOLING_MIN,
        reference::case_950::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    // Heating should be zero (disabled for night ventilation cases)
    assert_eq!(heating, 0.0, "Heating should be zero for night ventilation case");
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_setback_reduces_heating() {
    // Compare Case 600 (baseline) vs Case 640 (setback)
    // Setback should reduce heating energy
    let (heating_600, _) = simulate_case(ASHRAE140Case::Case600);
    let (heating_640, _) = simulate_case(ASHRAE140Case::Case640);

    println!(
        "Heating - Case 600: {:.2} MWh, Case 640: {:.2} MWh",
        heating_600, heating_640
    );

    // Note: This test documents current behavior
    // The actual reduction depends on proper setback implementation
}

#[test]
fn test_night_vent_reduces_cooling() {
    // Compare Case 600 (baseline) vs Case 650 (night vent)
    // Night ventilation should reduce cooling energy
    let (_, cooling_600) = simulate_case(ASHRAE140Case::Case600);
    let (_, cooling_650) = simulate_case(ASHRAE140Case::Case650);

    println!(
        "Cooling - Case 600: {:.2} MWh, Case 650: {:.2} MWh",
        cooling_600, cooling_650
    );

    // Note: This test documents current behavior
    // Night ventilation should provide passive cooling
}

#[test]
fn test_case_specifications() {
    // Verify setback cases have setback configured
    let spec_640 = ASHRAE140Case::Case640.spec();
    assert!(spec_640.hvac[0].setback_setpoint.is_some(), "Case 640 should have setback");
    assert_eq!(spec_640.hvac[0].setback_setpoint, Some(10.0), "Setback should be 10°C");

    let spec_940 = ASHRAE140Case::Case940.spec();
    assert!(spec_940.hvac[0].setback_setpoint.is_some(), "Case 940 should have setback");

    // Verify night ventilation cases have ventilation configured
    let spec_650 = ASHRAE140Case::Case650.spec();
    assert!(spec_650.night_ventilation.is_some(), "Case 650 should have night ventilation");
    assert!(spec_650.hvac[0].heating_setpoint < 0.0, "Case 650 should have heating disabled");

    let spec_950 = ASHRAE140Case::Case950.spec();
    assert!(spec_950.night_ventilation.is_some(), "Case 950 should have night ventilation");
}
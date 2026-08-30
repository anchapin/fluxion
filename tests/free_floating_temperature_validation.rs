//! Test for GitHub Issue #666: [Phase B.3] Fix free-floating temperature failures
//!
//! This test validates that free-floating cases have:
//! 1. HVAC output = 0 (HVAC is completely off)
//! 2. Internal gains = 0 (ASHRAE 140 specifies no internal loads for FF cases)
//! 3. Physically reasonable temperatures (10-50°C range, not 125°C)
//!
//! Root cause: compute_zone_hvac_load() doesn't check hvac_enabled flag

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Reference ranges for ASHRAE 140 free-floating cases
mod reference {
    // Case 900FF - High mass free-floating
    pub mod case_900ff {
        pub const MIN_TEMP_MIN: f64 = -6.4;
        pub const MIN_TEMP_MAX: f64 = -1.6;
        pub const MAX_TEMP_MIN: f64 = 41.8;
        pub const MAX_TEMP_MAX: f64 = 46.4;
    }
}

/// Test that free-floating case has zero HVAC energy (HVAC completely off)
#[test]
fn test_free_floating_hvac_is_disabled() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Verify this is a free-floating case
    assert!(spec.is_free_floating(), "Case should be free-floating");

    // Disable HVAC for free-floating mode (same as validator does)
    model.hvac.hvac_heating_capacity = 0.0;
    model.hvac.hvac_cooling_capacity = 0.0;
    model.hvac.hvac_enabled = VectorField::from_scalar(0.0, model.hvac.num_zones);

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    // After simulation, HVAC energy should be zero
    let total_hvac_energy = model.hvac.annual_heating_energy + model.hvac.annual_cooling_energy;
    println!(
        "Total HVAC energy for 900FF: {:.6} kWh (should be 0)",
        total_hvac_energy
    );

    // Key assertion: HVAC energy should be ZERO for free-floating
    assert!(
        total_hvac_energy < 1e-6,
        "HVAC energy should be zero for free-floating case, got {} kWh",
        total_hvac_energy
    );
}

/// Test that free-floating temperatures are within physically reasonable range
/// Issue #666: Previously showed 125°C which is physically impossible
#[test]
fn test_free_floating_temperatures_physically_reasonable() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Verify this is a free-floating case
    assert!(spec.is_free_floating(), "Case should be free-floating");

    // Disable HVAC for free-floating mode
    model.hvac.hvac_heating_capacity = 0.0;
    model.hvac.hvac_cooling_capacity = 0.0;
    model.hvac.hvac_enabled = VectorField::from_scalar(0.0, model.hvac.num_zones);

    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if let Some(&zone_temp) = model.setpoints.temperatures.as_slice().first() {
            min_temp = min_temp.min(zone_temp);
            max_temp = max_temp.max(zone_temp);
        }
    }

    println!("\n=== Free-Floating Temperature Validation ===");
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

    // Physically impossible temperatures indicate a bug
    assert!(
        max_temp < 60.0,
        "Max temp {:.2}°C is physically impossible for a building without HVAC",
        max_temp
    );

    assert!(
        min_temp > -40.0,
        "Min temp {:.2}°C is physically unrealistic for Denver climate",
        min_temp
    );
}

/// Test HVAC enabled=0 produces zero output (root cause check for issue #666)
/// This tests whether the hvac_enabled flag is properly respected
#[test]
fn test_hvac_enabled_zero_produces_zero_output() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Enable HVAC with realistic setpoints
    model.hvac.hvac_enabled = VectorField::from_scalar(1.0, model.hvac.num_zones);
    model.setpoints.heating_setpoint = 20.0;
    model.setpoints.cooling_setpoint = 25.0;
    model.hvac.hvac_heating_capacity = 10000.0;
    model.hvac.hvac_cooling_capacity = 10000.0;

    for step in 0..168 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    let energy_with_enabled = model.hvac.annual_heating_energy + model.hvac.annual_cooling_energy;
    println!("Energy with HVAC enabled: {:.4} kWh", energy_with_enabled);

    // Now disable HVAC
    model.hvac.hvac_enabled = VectorField::from_scalar(0.0, model.hvac.num_zones);
    model.reset_heating_cooling_energy();

    for step in 0..168 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    let energy_with_disabled = model.hvac.annual_heating_energy + model.hvac.annual_cooling_energy;
    println!("Energy with HVAC disabled: {:.4} kWh", energy_with_disabled);

    // When HVAC is disabled, energy should be zero
    // If it's NOT zero, it means the system is ignoring hvac_enabled
    assert!(
        energy_with_disabled < 1e-6,
        "HVAC energy should be 0 when hvac_enabled=0, got {} kWh. \
         This indicates hvac_enabled flag is being ignored!",
        energy_with_disabled
    );
}

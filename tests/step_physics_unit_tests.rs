//! Step Physics Unit Tests for ASHRAE 140 Validation
//!
//! These tests validate the core physics simulation step by step,
//! ensuring reasonable values are returned and temperatures remain stable.
//!
//! Test categories:
//! 1. Basic sanity checks (no NaN, no Inf, reasonable ranges)
//! 2. Single timestep behavior
//! 3. Multi-timestep stability
//! 4. Temperature bounds verification
//!
//! # NaN Checking
//!
//! Uses `noisy_float::N64` to catch silent NaN propagation.
//! N64 wraps f64 and panics if NaN is passed to `new()`.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use noisy_float::prelude::*;

// ============================================================================
// Basic Sanity Checks
// ============================================================================

/// Test that step_physics returns finite values for Case 600 (low-mass)
#[test]
fn test_step_physics_finite_case_600() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run a single timestep
    let hvac_kwh = model.step_physics(0, 10.0, 3600.0);

    // Wrap in N64 to catch NaN - will panic if hvac_kwh is NaN
    let _hvac_kwh_checked = n64(hvac_kwh);

    assert!(
        hvac_kwh.is_finite(),
        "Case 600 step_physics returned non-finite value: {}",
        hvac_kwh
    );
}

/// Test that step_physics returns finite values for Case 900 (high-mass)
#[test]
fn test_step_physics_finite_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run a single timestep
    let hvac_kwh = model.step_physics(0, 10.0, 3600.0);

    // Wrap in N64 to catch NaN - will panic if hvac_kwh is NaN
    let _hvac_kwh_checked = n64(hvac_kwh);

    assert!(
        hvac_kwh.is_finite(),
        "Case 900 step_physics returned non-finite value: {}",
        hvac_kwh
    );
}

/// Test that step_physics returns values in reasonable range for Case 600
#[test]
fn test_step_physics_reasonable_range_case_600() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Test at various outdoor temperatures
    for outdoor_temp in [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0] {
        let hvac_kwh = model.step_physics(0, outdoor_temp, 3600.0);

        // Wrap in N64 to catch NaN - will panic if hvac_kwh is NaN
        let hvac_kwh_checked = n64(hvac_kwh);

        // HVAC demand should be within ±100 kWh per hour for a small building
        // (equivalent to ±100 kW, which is very generous for a 20m² zone)
        // Using arithmetic on N64 to ensure NaN would propagate
        assert!(
            (hvac_kwh_checked - 0.0).abs() <= 100.0,
            "Case 600 step_physics returned extreme value {:.2} kWh at outdoor_temp {:.1}°C",
            hvac_kwh,
            outdoor_temp
        );
    }
}

/// Test that step_physics returns values in reasonable range for Case 900
#[test]
fn test_step_physics_reasonable_range_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Test at various outdoor temperatures
    for outdoor_temp in [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0] {
        let hvac_kwh = model.step_physics(0, outdoor_temp, 3600.0);

        // Wrap in N64 to catch NaN - will panic if hvac_kwh is NaN
        let hvac_kwh_checked = n64(hvac_kwh);

        // HVAC demand should be within ±100 kWh per hour
        // Using arithmetic on N64 to ensure NaN would propagate
        assert!(
            (hvac_kwh_checked - 0.0).abs() <= 100.0,
            "Case 900 step_physics returned extreme value {:.2} kWh at outdoor_temp {:.1}°C",
            hvac_kwh,
            outdoor_temp
        );
    }
}

// ============================================================================
// Temperature Stability Tests
// ============================================================================

/// Test that zone temperature remains stable for Case 600
#[test]
fn test_temperature_stability_case_600() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run 168 hours (1 week) with sinusoidal outdoor temperature
    for step in 0..168 {
        let hour_of_day = step % 24;
        let outdoor_temp = 10.0 + 15.0 * ((hour_of_day as f64) * std::f64::consts::PI / 12.0).sin();

        model.step_physics(step, outdoor_temp, 3600.0);

        // Check zone temperature is within reasonable bounds
        let zone_temp = model.temperatures.as_ref()[0];

        // Wrap in N64 to catch NaN - will panic if zone_temp is NaN
        let zone_temp_checked = n64(zone_temp);

        // Using arithmetic to ensure NaN would propagate
        assert!(
            zone_temp_checked > -40.0 && zone_temp_checked < 60.0,
            "Case 600 zone temperature out of range at step {}: {:.2}°C",
            step,
            zone_temp
        );
    }
}

/// Test that zone temperature remains stable for Case 900
#[test]
fn test_temperature_stability_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run 168 hours (1 week) with sinusoidal outdoor temperature
    for step in 0..168 {
        let hour_of_day = step % 24;
        let outdoor_temp = 10.0 + 15.0 * ((hour_of_day as f64) * std::f64::consts::PI / 12.0).sin();

        model.step_physics(step, outdoor_temp, 3600.0);

        // Check zone temperature is within reasonable bounds
        let zone_temp = model.temperatures.as_ref()[0];

        // Wrap in N64 to catch NaN - will panic if zone_temp is NaN
        let zone_temp_checked = n64(zone_temp);

        assert!(
            zone_temp_checked > -40.0 && zone_temp_checked < 60.0,
            "Case 900 zone temperature out of range at step {}: {:.2}°C",
            step,
            zone_temp
        );
    }
}

// ============================================================================
// Free-Floating Temperature Tests
// ============================================================================

/// Test free-floating temperature stability for Case 600FF
#[test]
fn test_free_floating_stability_case_600ff() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Verify this is a free-floating case (no HVAC)
    // Free-floating cases should have extreme setpoints
    assert!(
        model.heating_setpoint < -100.0,
        "Case 600FF should have very low heating setpoint"
    );
    assert!(
        model.cooling_setpoint > 100.0,
        "Case 600FF should have very high cooling setpoint"
    );

    // Run 168 hours with sinusoidal outdoor temperature
    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    for step in 0..168 {
        let hour_of_day = step % 24;
        let outdoor_temp = 10.0 + 15.0 * ((hour_of_day as f64) * std::f64::consts::PI / 12.0).sin();

        model.step_physics(step, outdoor_temp, 3600.0);

        let zone_temp = model.temperatures.as_ref()[0];

        // Wrap in N64 to catch NaN - will panic if zone_temp is NaN
        let zone_temp_checked = n64(zone_temp);

        min_temp = min_temp.min(zone_temp);
        max_temp = max_temp.max(zone_temp);

        // Check zone temperature is within reasonable bounds
        assert!(
            zone_temp_checked > -40.0 && zone_temp_checked < 80.0,
            "Case 600FF zone temperature out of range at step {}: {:.2}°C",
            step,
            zone_temp
        );
    }

    println!("Case 600FF: Min={:.2}°C, Max={:.2}°C", min_temp, max_temp);
}

/// Test free-floating temperature stability for Case 900FF
#[test]
fn test_free_floating_stability_case_900ff() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run 168 hours with sinusoidal outdoor temperature
    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    for step in 0..168 {
        let hour_of_day = step % 24;
        let outdoor_temp = 10.0 + 15.0 * ((hour_of_day as f64) * std::f64::consts::PI / 12.0).sin();

        model.step_physics(step, outdoor_temp, 3600.0);

        let zone_temp = model.temperatures.as_ref()[0];

        // Wrap in N64 to catch NaN - will panic if zone_temp is NaN
        let zone_temp_checked = n64(zone_temp);

        min_temp = min_temp.min(zone_temp);
        max_temp = max_temp.max(zone_temp);

        // Check zone temperature is within reasonable bounds
        assert!(
            zone_temp_checked > -40.0 && zone_temp_checked < 80.0,
            "Case 900FF zone temperature out of range at step {}: {:.2}°C",
            step,
            zone_temp
        );
    }

    println!("Case 900FF: Min={:.2}°C, Max={:.2}°C", min_temp, max_temp);
}

// ============================================================================
// Energy Accumulation Tests
// ============================================================================

/// Test that energy accumulation is consistent over multiple timesteps
#[test]
#[ignore] // TODO: Fix - energy accumulation calculation issue
fn test_energy_accumulation_consistency() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Reset energy tracking
    model.reset_heating_cooling_energy();

    // Run 24 hours with constant outdoor temperature
    let outdoor_temp = 5.0; // Cold, should require heating
    let mut total_hvac_kwh = 0.0;

    for step in 0..24 {
        let hvac_kwh = model.step_physics(step, outdoor_temp, 3600.0);
        total_hvac_kwh += hvac_kwh;
    }

    // Check that accumulated energy matches sum of individual steps
    let model_heating = model.annual_heating_energy;
    let model_cooling = model.annual_cooling_energy;
    let model_net = model_heating - model_cooling;

    // Allow small numerical difference
    let diff = (total_hvac_kwh - model_net).abs();
    assert!(
        diff < 0.01,
        "Energy accumulation mismatch: sum={:.4} kWh, model={:.4} kWh, diff={:.4} kWh",
        total_hvac_kwh,
        model_net,
        diff
    );
}

// ============================================================================
// HVAC Mode Detection Tests
// ============================================================================

/// Test that HVAC correctly detects heating mode
#[test]
fn test_hvac_heating_mode_detection() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Cold outdoor temperature should trigger heating
    let hvac_kwh = model.step_physics(0, -10.0, 3600.0);

    // Wrap in N64 to catch NaN - will panic if hvac_kwh is NaN
    let hvac_kwh_checked = n64(hvac_kwh);

    assert!(
        hvac_kwh_checked > 0.0,
        "Expected heating (positive hvac_kwh) at -10°C, got {:.4} kWh",
        hvac_kwh
    );
}

/// Test that HVAC correctly detects cooling mode
#[test]
fn test_hvac_cooling_mode_detection() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Warm up the zone for several hours before checking cooling activation.
    // With corrected h_ext=29.3 W/m²K (ASHRAE 140 Sec. 5.2), the model
    // initialises at a lower equilibrium temperature than with the old h=25.0,
    // so a single step at 35°C is insufficient to push the zone above the
    // cooling setpoint.  Running 6 hours of warm-up ensures steady-state
    // hot-weather operation before the assertion.
    for ts in 0..6 {
        model.step_physics(ts, 35.0, 3600.0);
    }
    let hvac_kwh = model.step_physics(6, 35.0, 3600.0);

    // Wrap in N64 to catch NaN - will panic if hvac_kwh is NaN
    let hvac_kwh_checked = n64(hvac_kwh);

    assert!(
        hvac_kwh_checked < 0.0,
        "Expected cooling (negative hvac_kwh) at 35°C after warm-up, got {:.4} kWh",
        hvac_kwh
    );
}

/// Test that HVAC is off in deadband
#[test]
fn test_hvac_deadband() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Set initial zone temperature to setpoint
    // Then test at moderate outdoor temperature
    let hvac_kwh = model.step_physics(0, 20.0, 3600.0);

    // Wrap in N64 to catch NaN - will panic if hvac_kwh is NaN
    let hvac_kwh_checked = n64(hvac_kwh);

    // At 20°C outdoor with 20°C heating setpoint, expect minimal HVAC
    assert!(
        hvac_kwh_checked.abs() < 1.0,
        "Expected minimal HVAC at 20°C outdoor, got {:.4} kWh",
        hvac_kwh
    );
}

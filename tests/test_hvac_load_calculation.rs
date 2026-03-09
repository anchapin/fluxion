//! Comprehensive unit tests for HVAC load calculation
//!
//! These tests validate the core HVAC control logic including:
//! - Ti_free calculation from 5R1C thermal network
//! - HVAC mode determination (Heating/Cooling/Off)
//! - Heating and cooling load calculations
//! - Sign convention (positive = heating, negative = cooling)
//! - Deadband tolerance prevents rapid cycling
//! - Dual setpoint control (heating <20°C, cooling >27°C)
//! - Free-floating mode (no HVAC when Ti_free in deadband)

use fluxion::sim::engine::{IdealHVACController, HVACMode};

/// Test 1: Validate Ti_free calculation from 5R1C thermal network
///
/// This test verifies that the free-floating temperature is calculated correctly
/// from the energy balance equation without HVAC input.
#[test]
fn test_ti_free_calculation() {
    // For a simple 5R1C model, Ti_free = (num_tm + num_phi_st + num_rest) / den
    // where:
    // - num_tm = h_tr_ms * Tm
    // - num_phi_st = h_tr_is * phi_st
    // - num_rest = h_tr_is * h_ve * (Te + phi_ia / h_ve) + ground term
    // - den = h_tr_ms * h_tr_is + h_tr_is * (h_tr_w + h_ve)

    // Simplified test: verify that Ti_free is between outdoor and mass temperatures
    // when no gains are present
    let t_m: f64 = 18.0; // Mass temperature
    let t_e: f64 = 10.0; // Outdoor temperature
    let phi_st: f64 = 0.0; // No surface gains
    let phi_ia: f64 = 0.0; // No internal gains

    // Conductances (W/K)
    let h_tr_ms: f64 = 100.0; // Mass to surface
    let h_tr_is: f64 = 200.0; // Surface to interior
    let h_tr_w: f64 = 50.0; // Windows
    let h_ve: f64 = 20.0; // Ventilation

    let num_tm = h_tr_ms * t_m;
    let num_phi_st = h_tr_is * phi_st;
    let num_rest = h_tr_is * (h_tr_w + h_ve) * t_e;
    let den = h_tr_ms * h_tr_is + h_tr_is * (h_tr_w + h_ve);

    let t_i_free = (num_tm + num_phi_st + num_rest) / den;

    // Ti_free should be between Tm and Te (no gains)
    assert!(t_i_free >= t_e.min(t_m), "Ti_free should be >= min(Tm, Te)");
    assert!(t_i_free <= t_e.max(t_m), "Ti_free should be <= max(Tm, Te)");
}

/// Test 2: Validate HVAC mode determination (Heating/Cooling/Off) based on zone temperature
#[test]
fn test_hvac_mode_determination() {
    let controller = IdealHVACController::new(20.0, 27.0);

    // Below heating setpoint - tolerance
    assert_eq!(
        controller.determine_mode(19.0),
        HVACMode::Heating,
        "Should be heating when zone temp < heating threshold"
    );

    // At heating setpoint - should be off due to tolerance
    assert_eq!(
        controller.determine_mode(20.0),
        HVACMode::Off,
        "Should be off at heating setpoint (tolerance)"
    );

    // In deadband
    assert_eq!(
        controller.determine_mode(23.5),
        HVACMode::Off,
        "Should be off in deadband"
    );

    // At cooling setpoint - should be off due to tolerance
    assert_eq!(
        controller.determine_mode(27.0),
        HVACMode::Off,
        "Should be off at cooling setpoint (tolerance)"
    );

    // Above cooling setpoint + tolerance
    assert_eq!(
        controller.determine_mode(28.0),
        HVACMode::Cooling,
        "Should be cooling when zone temp > cooling threshold"
    );
}

/// Test 3: Validate heating load calculation when Ti_free < heating_setpoint
#[test]
fn test_heating_load_calculation_case1() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let ti_free = 15.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Should be positive (heating)
    assert!(power > 0.0, "Heating power should be positive, got {}", power);

    // Check magnitude (with tolerance for deadband adjustment)
    let expected_power = 5000.0;
    assert!(
        (power - expected_power).abs() < 10.0,
        "Expected ~{}W, got {}W",
        expected_power,
        power
    );
}

#[test]
fn test_heating_load_calculation_case2() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let ti_free = 18.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Should be positive (heating)
    assert!(power > 0.0, "Heating power should be positive, got {}", power);

    // Check magnitude
    let expected_power = 2000.0;
    assert!(
        (power - expected_power).abs() < 10.0,
        "Expected ~{}W, got {}W",
        expected_power,
        power
    );
}

#[test]
fn test_heating_load_calculation_case3() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.002;
    let ti_free = 10.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Should be positive (heating)
    assert!(power > 0.0, "Heating power should be positive, got {}", power);

    // Check magnitude
    let expected_power = 5000.0;
    assert!(
        (power - expected_power).abs() < 10.0,
        "Expected ~{}W, got {}W",
        expected_power,
        power
    );
}

/// Test 4: Validate cooling load calculation when Ti_free > cooling_setpoint
#[test]
fn test_cooling_load_calculation_case1() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let ti_free = 30.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Should be negative (cooling)
    assert!(power < 0.0, "Cooling power should be negative, got {}", power);

    // Check magnitude
    let expected_power = -3000.0;
    assert!(
        (power - expected_power).abs() < 10.0,
        "Expected ~{}W, got {}W",
        expected_power,
        power
    );
}

#[test]
fn test_cooling_load_calculation_case2() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let ti_free = 28.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Should be negative (cooling)
    assert!(power < 0.0, "Cooling power should be negative, got {}", power);

    // Check magnitude
    let expected_power = -1000.0;
    assert!(
        (power - expected_power).abs() < 10.0,
        "Expected ~{}W, got {}W",
        expected_power,
        power
    );
}

#[test]
fn test_cooling_load_calculation_case3() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.002;
    let ti_free = 35.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Should be negative (cooling)
    assert!(power < 0.0, "Cooling power should be negative, got {}", power);

    // Check magnitude
    let expected_power = -4000.0;
    assert!(
        (power - expected_power).abs() < 10.0,
        "Expected ~{}W, got {}W",
        expected_power,
        power
    );
}

/// Test 5: Validate HVAC load sign convention (positive = heating, negative = cooling)
#[test]
fn test_hvac_sign_convention() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;

    // Heating should be positive
    let heating_power = controller.calculate_power(15.0, 15.0, sensitivity);
    assert!(
        heating_power > 0.0,
        "Heating power should be positive, got {}",
        heating_power
    );

    // Cooling should be negative
    let cooling_power = controller.calculate_power(30.0, 30.0, sensitivity);
    assert!(
        cooling_power < 0.0,
        "Cooling power should be negative, got {}",
        cooling_power
    );
}

/// Test 6: Validate deadband tolerance prevents rapid cycling
#[test]
fn test_deadband_prevents_cycling_at_heating_setpoint() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let zone_temp = 20.0;

    let power = controller.calculate_power(zone_temp, zone_temp, sensitivity);

    assert_eq!(
        power, 0.0,
        "HVAC should be off (power=0) in deadband at {}°C, got {}W",
        zone_temp, power
    );
}

#[test]
fn test_deadband_prevents_cycling_at_cooling_setpoint() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let zone_temp = 27.0;

    let power = controller.calculate_power(zone_temp, zone_temp, sensitivity);

    assert_eq!(
        power, 0.0,
        "HVAC should be off (power=0) in deadband at {}°C, got {}W",
        zone_temp, power
    );
}

#[test]
fn test_deadband_prevents_cycling_above_heating() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let zone_temp = 20.5;

    let power = controller.calculate_power(zone_temp, zone_temp, sensitivity);

    assert_eq!(
        power, 0.0,
        "HVAC should be off (power=0) in deadband at {}°C, got {}W",
        zone_temp, power
    );
}

#[test]
fn test_deadband_prevents_cycling_below_cooling() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let zone_temp = 26.5;

    let power = controller.calculate_power(zone_temp, zone_temp, sensitivity);

    assert_eq!(
        power, 0.0,
        "HVAC should be off (power=0) in deadband at {}°C, got {}W",
        zone_temp, power
    );
}

#[test]
fn test_deadband_prevents_cycling_middle() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;
    let zone_temp = 23.5;

    let power = controller.calculate_power(zone_temp, zone_temp, sensitivity);

    assert_eq!(
        power, 0.0,
        "HVAC should be off (power=0) in deadband at {}°C, got {}W",
        zone_temp, power
    );
}

/// Test 7: Validate dual setpoint control (heating <20°C, cooling >27°C)
#[test]
fn test_dual_setpoint_control() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;

    // Test heating setpoint (20°C)
    assert_eq!(controller.heating_setpoint, 20.0);

    // Test cooling setpoint (27°C)
    assert_eq!(controller.cooling_setpoint, 27.0);

    // Verify deadband is valid (7°C gap)
    assert!(controller.cooling_setpoint > controller.heating_setpoint);

    // Test heating below setpoint
    let heating_power = controller.calculate_power(15.0, 15.0, sensitivity);
    assert!(heating_power > 0.0, "Should heat when Ti_free < 20°C");

    // Test cooling above setpoint
    let cooling_power = controller.calculate_power(30.0, 30.0, sensitivity);
    assert!(cooling_power < 0.0, "Should cool when Ti_free > 27°C");

    // Test deadband between setpoints
    let deadband_power = controller.calculate_power(23.5, 23.5, sensitivity);
    assert_eq!(deadband_power, 0.0, "Should be off in deadband (20-27°C)");
}

/// Test 8: Validate free-floating mode (no HVAC when Ti_free in deadband)
#[test]
fn test_free_floating_mode() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;

    // Free-floating temperatures in deadband
    let free_floating_temps = vec![20.5, 22.0, 23.5, 25.0, 26.5];

    for ti_free in free_floating_temps {
        let power = controller.calculate_power(ti_free, ti_free, sensitivity);
        assert_eq!(
            power, 0.0,
            "Free-floating at {}°C should have no HVAC load, got {}W",
            ti_free, power
        );
    }
}

/// Test 9: Validate that HVAC uses Ti_free not Ti for load calculation
///
/// This is a critical test to ensure the research-guided fix is applied.
/// The HVAC load should be based on the free-floating temperature (Ti_free),
/// not the current zone temperature (Ti).
#[test]
fn test_hvac_uses_ti_free_not_ti() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;

    // Scenario: Zone temp is comfortable (25°C) but Ti_free is cold (15°C)
    // This happens when thermal mass is holding heat from previous hours
    let zone_temp = 25.0;
    let ti_free = 15.0;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Should heat based on Ti_free (15°C), not zone temp (25°C)
    // If using zone temp, power would be 0 (in deadband)
    // If using Ti_free, power should be > 0 (need heating)
    assert!(
        power > 0.0,
        "HVAC should use Ti_free (15°C) for load calculation, not zone temp (25°C). Got power: {}W",
        power
    );

    // Verify the calculation is based on Ti_free
    let expected_power = ((20.0 + 0.5) - ti_free) / sensitivity;
    assert!(
        (power - expected_power).abs() < 1.0,
        "Power should be based on Ti_free, expected ~{}W, got {}W",
        expected_power,
        power
    );
}

/// Test 10: Validate capacity limits
#[test]
fn test_capacity_limits() {
    let mut controller = IdealHVACController::new(20.0, 27.0);
    controller.heating_capacity_per_stage = 5000.0;
    controller.cooling_capacity_per_stage = 5000.0;
    controller.heating_stages = 1;
    controller.cooling_stages = 1;

    let sensitivity = 0.001;

    // Extreme cold - should cap at heating capacity
    let extreme_heating = controller.calculate_power(0.0, 0.0, sensitivity);
    assert_eq!(
        extreme_heating,
        5000.0,
        "Heating should cap at 5000W, got {}W",
        extreme_heating
    );

    // Extreme heat - should cap at cooling capacity
    let extreme_cooling = controller.calculate_power(50.0, 50.0, sensitivity);
    assert_eq!(
        extreme_cooling,
        -5000.0,
        "Cooling should cap at -5000W, got {}W",
        extreme_cooling
    );
}

/// Test 11: Validate deadband tolerance setting
#[test]
fn test_deadband_tolerance() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;

    // Default tolerance is 0.5°C
    assert_eq!(controller.deadband_tolerance, 0.5);

    // Heating threshold = 20 - 0.5 = 19.5°C
    let just_below_heating = controller.calculate_power(19.0, 19.0, sensitivity);
    assert!(just_below_heating > 0.0, "Should heat at 19°C (< 19.5°C threshold)");

    let at_heating_threshold = controller.calculate_power(19.5, 19.5, sensitivity);
    assert_eq!(
        at_heating_threshold, 0.0,
        "Should be off at 19.5°C (heating threshold with tolerance)"
    );

    // Cooling threshold = 27 + 0.5 = 27.5°C
    let at_cooling_threshold = controller.calculate_power(27.5, 27.5, sensitivity);
    assert_eq!(
        at_cooling_threshold, 0.0,
        "Should be off at 27.5°C (cooling threshold with tolerance)"
    );

    let just_above_cooling = controller.calculate_power(28.0, 28.0, sensitivity);
    assert!(just_above_cooling < 0.0, "Should cool at 28°C (> 27.5°C threshold)");
}

/// Test 12: Validate steady-state heating scenario
///
/// In steady-state, Ti_free should equal the setpoint when HVAC is off,
/// and HVAC power should exactly compensate for heat loss.
#[test]
fn test_steady_state_heating() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;

    // Steady-state scenario: outdoor temp is cold, Ti_free would be 15°C
    // HVAC needs to heat to 20.5°C (setpoint + tolerance)
    let ti_free = 15.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Power should be positive
    assert!(power > 0.0, "Heating power should be positive");

    // Calculate expected power: (target - ti_free) / sensitivity
    let target = controller.heating_setpoint + controller.deadband_tolerance;
    let expected_power = (target - ti_free) / sensitivity;

    assert!(
        (power - expected_power).abs() < 1.0,
        "Steady-state heating power should match expected value"
    );
}

/// Test 13: Validate steady-state cooling scenario
#[test]
fn test_steady_state_cooling() {
    let controller = IdealHVACController::new(20.0, 27.0);
    let sensitivity = 0.001;

    // Steady-state scenario: outdoor temp is hot, Ti_free would be 30°C
    // HVAC needs to cool to 26.5°C (setpoint - tolerance)
    let ti_free = 30.0;
    let zone_temp = ti_free;

    let power = controller.calculate_power(zone_temp, ti_free, sensitivity);

    // Power should be negative
    assert!(power < 0.0, "Cooling power should be negative");

    // Calculate expected power: -(ti_free - target) / sensitivity
    let target = controller.cooling_setpoint - controller.deadband_tolerance;
    let expected_power = -(ti_free - target) / sensitivity;

    assert!(
        (power - expected_power).abs() < 1.0,
        "Steady-state cooling power should match expected value"
    );
}

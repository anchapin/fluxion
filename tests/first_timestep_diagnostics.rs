//! First Timestep Diagnostics for ASHRAE 140 Validation
//!
//! These tests isolate the first timestep behavior to identify the root cause
//! of temperature instability in high-mass cases (Case 900, 900FF).
//!
//! Session 72 identified critical issues:
//! - Case 900: Temperature starts at 164.82°C (should be ~20°C)
//! - Case 900FF: Temperature starts at 148.93°C (should be realistic)
//!
//! This test suite systematically checks each component of the physics calculation.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

// ============================================================================
// Initial State Verification
// ============================================================================

/// Verify initial temperatures are correctly set before any simulation
#[test]
fn test_initial_temperatures_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // All temperatures should start at 20°C
    let zone_temp = model.temperatures.as_ref()[0];
    let mass_temp = model.mass_temperatures.as_ref()[0];

    assert!(
        (zone_temp - 20.0).abs() < 1.0,
        "Initial zone temp should be ~20°C, got {:.2}°C",
        zone_temp
    );
    assert!(
        (mass_temp - 20.0).abs() < 1.0,
        "Initial mass temp should be ~20°C, got {:.2}°C",
        mass_temp
    );
}

/// Verify initial temperatures for free-floating case
#[test]
fn test_initial_temperatures_case_900ff() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let zone_temp = model.temperatures.as_ref()[0];
    let mass_temp = model.mass_temperatures.as_ref()[0];

    assert!(
        (zone_temp - 20.0).abs() < 1.0,
        "Initial zone temp should be ~20°C, got {:.2}°C",
        zone_temp
    );
    assert!(
        (mass_temp - 20.0).abs() < 1.0,
        "Initial mass temp should be ~20°C, got {:.2}°C",
        mass_temp
    );
}

// ============================================================================
// Conductance Diagnostics
// ============================================================================

/// Verify thermal conductances are reasonable for Case 900
#[test]
fn test_conductances_reasonable_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let h_tr_em = model.h_tr_em.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_is = model.h_tr_is.as_ref()[0];
    let h_tr_w = model.h_tr_w.as_ref()[0];
    let h_ve = model.h_ve.as_ref()[0];

    // All conductances should be positive and finite
    assert!(
        h_tr_em.is_finite() && h_tr_em > 0.0,
        "h_tr_em: {:.2}",
        h_tr_em
    );
    assert!(
        h_tr_ms.is_finite() && h_tr_ms > 0.0,
        "h_tr_ms: {:.2}",
        h_tr_ms
    );
    assert!(
        h_tr_is.is_finite() && h_tr_is > 0.0,
        "h_tr_is: {:.2}",
        h_tr_is
    );
    assert!(h_tr_w.is_finite() && h_tr_w >= 0.0, "h_tr_w: {:.2}", h_tr_w);
    assert!(h_ve.is_finite() && h_ve >= 0.0, "h_ve: {:.2}", h_ve);

    // Check sensitivity (should be small positive number)
    let sensitivity = model.derived_sensitivity.as_ref()[0];
    assert!(
        sensitivity.is_finite() && sensitivity > 0.0 && sensitivity < 1.0,
        "Sensitivity should be 0-1 K/W, got {:.6} K/W",
        sensitivity
    );

    println!("Case 900 conductances:");
    println!("  h_tr_em: {:.2} W/K", h_tr_em);
    println!("  h_tr_ms: {:.2} W/K", h_tr_ms);
    println!("  h_tr_is: {:.2} W/K", h_tr_is);
    println!("  h_tr_w: {:.2} W/K", h_tr_w);
    println!("  h_ve: {:.2} W/K", h_ve);
    println!("  sensitivity: {:.6} K/W", sensitivity);
}

// ============================================================================
// CTF Solver Diagnostics
// ============================================================================

/// Check if CTF solver is enabled and has valid coefficients
#[test]
fn test_ctf_solver_state_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("Case 900 CTF state:");
    println!("  CTF enabled: {}", model.ctf_enabled);
    println!("  CTF coefficients: {:?}", model.ctf_coefficients.is_some());
    println!("  CTF solvers count: {}", model.ctf_solvers.len());
    println!("  FD enabled: {}", model.fd_enabled);

    // Check if model is using 6R2C
    println!("  Is 6R2C model: {}", model.is_6r2c_model());
}

// ============================================================================
// First Timestep Temperature Check
// ============================================================================

/// Check temperature after first timestep with no solar gains
#[test]
fn test_first_timestep_no_solar_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Print conductances for debugging
    println!("=== Case 900 Conductance Debug ===");
    println!("  h_tr_em: {:.4} W/K", model.h_tr_em.as_ref()[0]);
    println!("  h_tr_ms: {:.4} W/K", model.h_tr_ms.as_ref()[0]);
    println!("  h_tr_is: {:.4} W/K", model.h_tr_is.as_ref()[0]);
    println!("  h_tr_w: {:.4} W/K", model.h_tr_w.as_ref()[0]);
    println!("  h_ve: {:.4} W/K", model.h_ve.as_ref()[0]);
    println!("  h_tr_floor: {:.4} W/K", model.h_tr_floor.as_ref()[0]);
    println!(
        "  thermal_capacitance: {:.4} J/K",
        model.thermal_capacitance.as_ref()[0]
    );
    println!("  derived_den: {:.4}", model.derived_den.as_ref()[0]);
    println!(
        "  derived_sensitivity: {:.6} K/W",
        model.derived_sensitivity.as_ref()[0]
    );
    println!(
        "  derived_h_ext: {:.4} W/K",
        model.derived_h_ext.as_ref()[0]
    );
    println!(
        "  derived_term_rest_1: {:.4} W/K",
        model.derived_term_rest_1.as_ref()[0]
    );
    println!(
        "  derived_h_ms_is_prod: {:.4} W²/K²",
        model.derived_h_ms_is_prod.as_ref()[0]
    );
    println!(
        "  derived_ground_coeff: {:.4} W",
        model.derived_ground_coeff.as_ref()[0]
    );

    // Set solar gains to 0 to isolate the issue
    model.solar_gains = VectorField::from_scalar(0.0, model.num_zones);
    model.loads = VectorField::from_scalar(0.0, model.num_zones);

    // Run first timestep
    let hvac_kwh = model.step_physics(0, 10.0, 3600.0);

    // Check temperatures after first step
    let zone_temp = model.temperatures.as_ref()[0];
    let mass_temp = model.mass_temperatures.as_ref()[0];

    println!("Case 900 first timestep (no solar):");
    println!("  HVAC energy: {:.4} kWh", hvac_kwh);
    println!("  Zone temp: {:.2}°C", zone_temp);
    println!("  Mass temp: {:.2}°C", mass_temp);

    // Temperature should remain reasonable
    assert!(
        zone_temp > -30.0 && zone_temp < 70.0,
        "Zone temp after step 0 (no solar): {:.2}°C (unreasonable)",
        zone_temp
    );
    assert!(
        mass_temp > -30.0 && mass_temp < 70.0,
        "Mass temp after step 0 (no solar): {:.2}°C (unreasonable)",
        mass_temp
    );
}

/// Check temperature after first timestep with normal solar gains
#[test]
fn test_first_timestep_with_solar_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run first timestep with normal calculation
    let hvac_kwh = model.step_physics(0, 10.0, 3600.0);

    // Check temperatures after first step
    let zone_temp = model.temperatures.as_ref()[0];
    let mass_temp = model.mass_temperatures.as_ref()[0];

    println!("Case 900 first timestep (with solar):");
    println!("  HVAC energy: {:.4} kWh", hvac_kwh);
    println!("  Zone temp: {:.2}°C", zone_temp);
    println!("  Mass temp: {:.2}°C", mass_temp);

    // This is the failing test - temperature should be reasonable
    assert!(
        zone_temp > -30.0 && zone_temp < 70.0,
        "Zone temp after step 0 (with solar): {:.2}°C (unreasonable)",
        zone_temp
    );
}

// ============================================================================
// Free-Floating Case Diagnostics
// ============================================================================

/// Check free-floating case setpoints
#[test]
fn test_free_floating_setpoints_case_600ff() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("Case 600FF setpoints:");
    println!("  Heating setpoint: {:.1}°C", model.heating_setpoint);
    println!("  Cooling setpoint: {:.1}°C", model.cooling_setpoint);
    println!("  HVAC system mode: {:?}", model.hvac_system_mode);

    // Free-floating cases should have extreme setpoints
    // This test documents the current (possibly incorrect) behavior
}

/// Check free-floating case setpoints for 900FF
#[test]
fn test_free_floating_setpoints_case_900ff() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("Case 900FF setpoints:");
    println!("  Heating setpoint: {:.1}°C", model.heating_setpoint);
    println!("  Cooling setpoint: {:.1}°C", model.cooling_setpoint);
    println!("  HVAC system mode: {:?}", model.hvac_system_mode);
}

// ============================================================================
// HVAC Response Diagnostics
// ============================================================================

/// Check HVAC response to extreme outdoor temperatures
#[test]
fn test_hvac_response_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Test with very cold outdoor temperature
    let hvac_cold = model.step_physics(0, -20.0, 3600.0);
    let zone_temp_cold = model.temperatures.as_ref()[0];

    // Reset temperatures
    model.temperatures = VectorField::from_scalar(20.0, model.num_zones);
    model.mass_temperatures = VectorField::from_scalar(20.0, model.num_zones);

    // Test with very hot outdoor temperature
    let hvac_hot = model.step_physics(0, 40.0, 3600.0);
    let zone_temp_hot = model.temperatures.as_ref()[0];

    println!("Case 900 HVAC response:");
    println!(
        "  Cold (-20°C): HVAC={:.4} kWh, Zone={:.2}°C",
        hvac_cold, zone_temp_cold
    );
    println!(
        "  Hot (40°C): HVAC={:.4} kWh, Zone={:.2}°C",
        hvac_hot, zone_temp_hot
    );

    // HVAC energy should be finite and reasonable
    assert!(
        hvac_cold.is_finite(),
        "HVAC energy should be finite for cold case"
    );
    assert!(
        hvac_hot.is_finite(),
        "HVAC energy should be finite for hot case"
    );
}

// ============================================================================
// Thermal Capacitance Diagnostics
// ============================================================================

/// Check thermal capacitance values
#[test]
fn test_thermal_capacitance_case_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let thermal_cap = model.thermal_capacitance.as_ref()[0];

    println!("Case 900 thermal capacitance: {:.2} J/K", thermal_cap);

    assert!(
        thermal_cap.is_finite() && thermal_cap > 0.0,
        "Thermal capacitance should be positive, got {:.2} J/K",
        thermal_cap
    );

    // High-mass cases should have large thermal capacitance (> 1e6 J/K)
    assert!(
        thermal_cap > 1e6,
        "High-mass case should have thermal capacitance > 1e6 J/K, got {:.2}",
        thermal_cap
    );
}

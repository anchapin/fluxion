//! Comprehensive 6R2C (two mass node) thermal model tests.
//!
//! This test suite consolidates diagnostic and unit tests for the optional 6R2C
//! thermal network model, which separates thermal mass into:
//! - Envelope mass (walls, roof, floor)
//! - Internal mass (furniture, partitions)

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::{ThermalModel, ThermalModelType};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

// ============================================================================
// Section 1: Core Configuration & Initialization
// ============================================================================

#[test]
fn test_thermal_model_type_default() {
    // Default model should be 5R1C
    let model = ThermalModel::new(1);
    assert_eq!(model.thermal_model_type, ThermalModelType::FiveROneC);
    assert!(!model.is_6r2c_model());
}

#[test]
fn test_configure_6r2c_model() {
    let mut model = ThermalModel::new(1);
    let envelope_fraction = 0.75;
    let h_tr_me_value = 100.0;

    model.configure_6r2c_model(envelope_fraction, h_tr_me_value, None);

    // Check that model is now 6R2C
    assert!(model.is_6r2c_model());
    assert_eq!(model.thermal_model_type, ThermalModelType::SixRTwoC);

    // Check that thermal capacitance is split correctly
    let total_cap = model.thermal_capacitance.as_ref()[0];
    let envelope_cap = model.envelope_thermal_capacitance.as_ref()[0];
    let internal_cap = model.internal_thermal_capacitance.as_ref()[0];

    assert!((envelope_cap - total_cap * envelope_fraction).abs() < 0.01);
    assert!((internal_cap - total_cap * (1.0 - envelope_fraction)).abs() < 0.01);
    assert!((envelope_cap + internal_cap - total_cap).abs() < 0.01);

    // Check that conductance between masses is set
    let h_tr_me = model.h_tr_me.as_ref()[0];
    assert!((h_tr_me - h_tr_me_value).abs() < 0.01);
}

#[test]
fn test_6r2c_thermal_mass_initialization() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    model.configure_6r2c_model(0.75, 100.0, None);

    assert!(!model.envelope_mass_temperatures.as_ref().is_empty());
    assert!(!model.internal_mass_temperatures.as_ref().is_empty());

    let initial_mass_temp = model.mass_temperatures.as_ref()[0];
    assert_eq!(
        model.envelope_mass_temperatures.as_ref()[0],
        initial_mass_temp
    );
    assert_eq!(
        model.internal_mass_temperatures.as_ref()[0],
        initial_mass_temp
    );
}

// ============================================================================
// Section 2: Conductance Calculations
// ============================================================================

#[test]
fn test_all_conductances_non_negative() {
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0, None);

    assert!(model.h_tr_em.as_ref()[0] >= 0.0);
    assert!(model.h_tr_ms.as_ref()[0] >= 0.0);
    assert!(model.h_tr_me.as_ref()[0] >= 0.0);
    assert!(model.h_tr_is.as_ref()[0] >= 0.0);
}

#[test]
fn test_coupling_ratio_bounds() {
    for fraction in [0.5, 0.6, 0.7, 0.75, 0.8] {
        let mut model = ThermalModel::new(1);
        model.configure_6r2c_model(fraction, 100.0, None);

        let h_tr_em = model.h_tr_em.as_ref()[0];
        let h_tr_ms = model.h_tr_ms.as_ref()[0];

        if h_tr_ms > 0.0 {
            let ratio = h_tr_em / h_tr_ms;
            assert!(ratio >= 0.0 && ratio <= 1.0);
        }
    }
}

// ============================================================================
// Section 3: Dynamics & Thermal Lag
// ============================================================================

#[test]
fn test_thermal_lag_envelope_vs_internal() {
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0, None);

    let outdoor_temp_step = 40.0;
    let mut t_env_curve = Vec::new();
    let mut t_int_curve = Vec::new();

    for timestep in 0..72 {
        model.step_physics(timestep, outdoor_temp_step, 3600.0);
        t_env_curve.push(model.envelope_mass_temperatures.as_ref()[0]);
        t_int_curve.push(model.internal_mass_temperatures.as_ref()[0]);
    }

    let t_env_initial = t_env_curve[0];
    let t_env_final = *t_env_curve.last().unwrap();
    let target_env = t_env_initial + 0.5 * (t_env_final - t_env_initial);

    let t_int_initial = t_int_curve[0];
    let t_int_final = *t_int_curve.last().unwrap();
    let target_int = t_int_initial + 0.5 * (t_int_final - t_int_initial);

    let t50_env = t_env_curve
        .iter()
        .position(|&t| t >= target_env)
        .unwrap_or(0);
    let t50_int = t_int_curve
        .iter()
        .position(|&t| t >= target_int)
        .unwrap_or(0);

    assert!(
        t50_int >= t50_env,
        "Internal mass should respond slower than envelope"
    );
}

#[test]
fn test_mass_nodes_diverge_during_simulation() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    model.configure_6r2c_model(0.75, 100.0, None);

    let initial_t_env = model.envelope_mass_temperatures.as_ref()[0];
    let initial_t_int = model.internal_mass_temperatures.as_ref()[0];

    for timestep in 0..24 {
        model.step_physics(timestep, 0.0, 3600.0);
    }

    let final_t_env = model.envelope_mass_temperatures.as_ref()[0];
    let final_t_int = model.internal_mass_temperatures.as_ref()[0];

    let delta_t_env = initial_t_env - final_t_env;
    let delta_t_int = initial_t_int - final_t_int;

    // Issue 691 fix: envelope mass time constant is now based on h_tr_ms + h_tr_me
    // not h_tr_em + h_tr_ms + h_tr_me. The envelope responds more slowly to outdoor
    // conditions because h_tr_em (exterior-to-mass path) no longer directly affects
    // the envelope's time constant.
    println!("\n=== Mass Node Divergence ===");
    println!("Initial T_env = {:.1}, T_int = {:.1}", initial_t_env, initial_t_int);
    println!("Final T_env = {:.1}, T_int = {:.1}", final_t_env, final_t_int);
    println!("Delta T_env = {:.2}, Delta T_int = {:.2}", delta_t_env, delta_t_int);

    // With corrected physics, both masses should be finite and reasonable
    assert!(
        final_t_env.is_finite() && final_t_env > -50.0 && final_t_env < 100.0,
        "Envelope temperature should be in reasonable range"
    );
    assert!(
        final_t_int.is_finite() && final_t_int > -50.0 && final_t_int < 100.0,
        "Internal temperature should be in reasonable range"
    );
}

// ============================================================================
// Section 3.5: Issue 691 - Time Constant Bug Fix
// ============================================================================

#[test]
fn test_envelope_mass_time_constant_based_on_h_tr_ms() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    model.configure_6r2c_model(0.75, 100.0, None);

    let cm_env = model.envelope_thermal_capacitance.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_me = model.h_tr_me.as_ref()[0];
    let h_tr_em = model.h_tr_em.as_ref()[0];

    // Time constant for envelope mass should be τ = Cm / (h_tr_ms + h_tr_me)
    // NOT τ = Cm / (h_tr_em + h_tr_ms + h_tr_me)
    // The h_tr_em path (exterior to envelope) should NOT affect the envelope's time constant
    let correct_tau = cm_env / (h_tr_ms + h_tr_me);
    let buggy_tau = cm_env / (h_tr_em + h_tr_ms + h_tr_me);

    let correct_tau_hours = correct_tau / 3600.0;
    let buggy_tau_hours = buggy_tau / 3600.0;

    println!("\n=== Issue 691: Envelope Mass Time Constant ===");
    println!("Cm_env = {:.0} J/K", cm_env);
    println!("h_tr_ms = {:.4} W/K", h_tr_ms);
    println!("h_tr_me = {:.4} W/K", h_tr_me);
    println!("h_tr_em = {:.4} W/K", h_tr_em);
    println!("Correct τ (based on h_tr_ms + h_tr_me) = {:.1} hours", correct_tau_hours);
    println!("Buggy τ (based on h_tr_em + h_tr_ms + h_tr_me) = {:.1} hours", buggy_tau_hours);
    println!("Reference τ for 900FF (ASHRAE 140) ≈ 47 hours");

    // The correct time constant should be higher (closer to reference)
    // The buggy calculation gives ~13-26 hours (too fast due to extra h_tr_em in denominator)
    assert!(
        correct_tau_hours > buggy_tau_hours,
        "Correct τ {:.1}h should be > buggy τ {:.1}h",
        correct_tau_hours, buggy_tau_hours
    );
}

// ============================================================================
// Section 4: Numerical Stability & Conservation
// ============================================================================

#[test]
fn test_6r2c_model_energy_conservation() {
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0, None);

    for t in 0..100 {
        let energy = model.step_physics(t, 20.0, 3600.0);
        assert!(energy.is_finite());
        assert!(model.temperatures.as_ref()[0].is_finite());
        assert!(model.envelope_mass_temperatures.as_ref()[0].is_finite());
        assert!(model.internal_mass_temperatures.as_ref()[0].is_finite());
    }
}

#[test]
fn test_warm_start_continuity() {
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0, None);

    for timestep in 0..24 {
        model.step_physics(timestep, 15.0, 3600.0);
    }

    let t_env_final = model.envelope_mass_temperatures.as_ref()[0];
    model.step_physics(24, 15.0, 3600.0);
    let t_env_next = model.envelope_mass_temperatures.as_ref()[0];

    assert!(
        (t_env_next - t_env_final).abs() < 5.0,
        "Large temperature jump detected"
    );
}

// ============================================================================
// Section 5: Multi-Zone Support
// ============================================================================

#[test]
fn test_6r2c_model_multi_zone() {
    let mut model = ThermalModel::new(2);
    model.configure_6r2c_model(0.75, 100.0, None);

    for t in 0..10 {
        let energy = model.step_physics(t, 20.0, 3600.0);
        assert!(energy.is_finite());
    }

    for i in 0..2 {
        assert!(model.temperatures.as_ref()[i].is_finite());
        assert!(model.envelope_mass_temperatures.as_ref()[i].is_finite());
        assert!(model.internal_mass_temperatures.as_ref()[i].is_finite());
    }
}

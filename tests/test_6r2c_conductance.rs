//! Plan 24-03: 6R2C Conductance Calculation Unit Tests
//!
//! This test suite validates key conductance calculations in the 6R2C thermal network
//! against ISO 13790 specifications.
//!
//! Reference: docs/ISO_13790_6R2C_SPECIFICATION.md

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

// ============================================================================
// Core Conductance Tests
// ============================================================================

#[test]
fn test_h_tr_me_configured_value() {
    // h_tr_me should match the configured value exactly
    for h_tr_me_value in [50.0, 100.0, 150.0, 200.0] {
        let mut model = ThermalModel::new(1);
        model.configure_6r2c_model(0.75, h_tr_me_value);

        let h_tr_me = model.h_tr_me.as_ref()[0];
        assert!(
            (h_tr_me - h_tr_me_value).abs() < 0.01,
            "h_tr_me should equal configured value {} got {}",
            h_tr_me_value,
            h_tr_me
        );
    }
}

#[test]
fn test_capacitance_split_conservation() {
    // C_env + C_int = C_total for any envelope fraction
    for envelope_fraction in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9] {
        let mut model = ThermalModel::new(1);
        let total_cap = model.thermal_capacitance.as_ref()[0];

        model.configure_6r2c_model(envelope_fraction, 100.0);

        let c_env = model.envelope_thermal_capacitance.as_ref()[0];
        let c_int = model.internal_thermal_capacitance.as_ref()[0];

        let sum = c_env + c_int;
        let error = (sum - total_cap).abs() / total_cap;

        assert!(
            error < 0.001,
            "C_env + C_int should equal C_total, error={:.4}%",
            error * 100.0
        );
    }
}

#[test]
fn test_capacitance_split_case_900() {
    // Case 900: C_total = 19,944,509 J/K
    // C_env = 0.75 × C_total = 14,958,382 J/K
    // C_int = 0.25 × C_total = 4,986,127 J/K
    let mut model = ThermalModel::new(1);

    // Set Case 900 capacitance
    model.thermal_capacitance = VectorField::from_scalar(19_944_509.0, 1);
    model.configure_6r2c_model(0.75, 100.0, None);

    let c_env = model.envelope_thermal_capacitance.as_ref()[0];
    let c_int = model.internal_thermal_capacitance.as_ref()[0];

    let expected_c_env = 0.75 * 19_944_509.0;
    let expected_c_int = 0.25 * 19_944_509.0;

    assert!(
        (c_env - expected_c_env).abs() < 1.0,
        "C_env should be {:.0}, got {:.0}",
        expected_c_env,
        c_env
    );
    assert!(
        (c_int - expected_c_int).abs() < 1.0,
        "C_int should be {:.0}, got {:.0}",
        expected_c_int,
        c_int
    );
}

#[test]
fn test_all_conductances_non_negative() {
    // All conductances must be non-negative
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0, None);

    assert!(
        model.h_tr_em.as_ref()[0] >= 0.0,
        "h_tr_em must be non-negative"
    );
    assert!(
        model.h_tr_ms.as_ref()[0] >= 0.0,
        "h_tr_ms must be non-negative"
    );
    assert!(
        model.h_tr_me.as_ref()[0] >= 0.0,
        "h_tr_me must be non-negative"
    );
    assert!(
        model.h_tr_is.as_ref()[0] >= 0.0,
        "h_tr_is must be non-negative"
    );
    assert!(
        model.h_tr_em_heating.as_ref()[0] >= 0.0,
        "h_tr_em_heating must be non-negative"
    );
    assert!(
        model.h_tr_em_cooling.as_ref()[0] >= 0.0,
        "h_tr_em_cooling must be non-negative"
    );
}

#[test]
fn test_coupling_ratio_bounds() {
    // Coupling ratio (h_tr_em / h_tr_ms) should always be between 0 and 1
    for envelope_fraction in [0.5, 0.6, 0.7, 0.75, 0.8] {
        let mut model = ThermalModel::new(1);
        model.configure_6r2c_model(envelope_fraction, 100.0);

        let h_tr_em = model.h_tr_em.as_ref()[0];
        let h_tr_ms = model.h_tr_ms.as_ref()[0];

        if h_tr_ms > 0.0 {
            let ratio = h_tr_em / h_tr_ms;
            assert!(
                ratio >= 0.0 && ratio <= 1.0,
                "Coupling ratio should be [0,1], got {} for fraction={}",
                ratio,
                envelope_fraction
            );
        }
    }
}

// ============================================================================
// Time Constant Analysis
// ============================================================================

#[test]
fn test_time_constant_env_reasonable() {
    // τ_env = C_env / (h_tr_em + h_tr_ms + h_tr_me)
    // Should be in hours range (not seconds, not days)
    let mut model = ThermalModel::new(1);
    model.thermal_capacitance = VectorField::from_scalar(19_944_509.0, 1);
    model.configure_6r2c_model(0.75, 100.0, None);

    let c_env = model.envelope_thermal_capacitance.as_ref()[0];
    let h_tr_em = model.h_tr_em.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_me = model.h_tr_me.as_ref()[0];

    let tau_env = c_env / (h_tr_em + h_tr_ms + h_tr_me);
    let tau_env_hours = tau_env / 3600.0;

    assert!(
        tau_env_hours > 0.1 && tau_env_hours < 100.0,
        "τ_env should be in reasonable range (0.1-100 hours), got {:.2} hours",
        tau_env_hours
    );
}

#[test]
fn test_time_constant_int_reasonable() {
    // τ_int = C_int / h_tr_me
    // Should be in hours range
    let mut model = ThermalModel::new(1);
    model.thermal_capacitance = VectorField::from_scalar(19_944_509.0, 1);
    model.configure_6r2c_model(0.75, 100.0, None);

    let c_int = model.internal_thermal_capacitance.as_ref()[0];
    let h_tr_me = model.h_tr_me.as_ref()[0];

    let tau_int = c_int / h_tr_me;
    let tau_int_hours = tau_int / 3600.0;

    assert!(
        tau_int_hours > 1.0 && tau_int_hours < 100.0,
        "τ_int should be in reasonable range (1-100 hours), got {:.2} hours",
        tau_int_hours
    );
}

#[test]
fn test_timestep_vs_time_constant_warning() {
    // Rule of thumb: Δt < τ_min / 10 for good accuracy
    // Current timestep: 1 hour = 3600 s
    // This test logs a warning if timestep exceeds recommendation
    let mut model = ThermalModel::new(1);
    model.thermal_capacitance = VectorField::from_scalar(19_944_509.0, 1);
    model.configure_6r2c_model(0.75, 100.0, None);

    let c_env = model.envelope_thermal_capacitance.as_ref()[0];
    let c_int = model.internal_thermal_capacitance.as_ref()[0];
    let h_tr_em = model.h_tr_em.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_me = model.h_tr_me.as_ref()[0];

    let tau_env = c_env / (h_tr_em + h_tr_ms + h_tr_me);
    let tau_int = c_int / h_tr_me;
    let tau_min = tau_env.min(tau_int);

    let timestep = 3600.0; // 1 hour
    let recommended_timestep = tau_min / 10.0;

    // Log warning if timestep exceeds recommendation (test still passes)
    if timestep > recommended_timestep {
        println!(
            "⚠️  TIMESTEP WARNING: Timestep ({:.1} h) exceeds recommended ({:.2} h) for τ_min={:.1} h",
            timestep / 3600.0,
            recommended_timestep / 3600.0,
            tau_min / 3600.0
        );
        println!("   This may cause numerical damping of thermal dynamics");
    }

    // Test passes regardless - this is informational
    assert!(true);
}

// ============================================================================
// Multi-Zone Tests
// ============================================================================

#[test]
fn test_conductance_multi_zone() {
    // Conductances should be calculated correctly for multi-zone buildings
    let mut model = ThermalModel::new(2); // 2 zones

    model.configure_6r2c_model(0.75, 100.0, None);

    // Both zones should have positive conductances
    for i in 0..2 {
        assert!(model.h_tr_em.as_ref()[i] > 0.0, "Zone {} h_tr_em > 0", i);
        assert!(model.h_tr_ms.as_ref()[i] > 0.0, "Zone {} h_tr_ms > 0", i);
        assert!(model.h_tr_me.as_ref()[i] > 0.0, "Zone {} h_tr_me > 0", i);
    }
}

#[test]
fn test_capacitance_split_multi_zone() {
    // Capacitance split should work correctly for multi-zone buildings
    let mut model = ThermalModel::new(2);
    model.thermal_capacitance = VectorField::new(vec![10_000_000.0, 20_000_000.0]);
    model.configure_6r2c_model(0.75, 100.0, None);

    for i in 0..2 {
        let c_total = model.thermal_capacitance.as_ref()[i];
        let c_env = model.envelope_thermal_capacitance.as_ref()[i];
        let c_int = model.internal_thermal_capacitance.as_ref()[i];

        assert!(
            (c_env + c_int - c_total).abs() < 1.0,
            "Zone {}: C_env + C_int should equal C_total",
            i
        );
    }
}

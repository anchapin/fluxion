//! Runtime energy-balance residual monitoring tests (Issue #1896).
//!
//! These tests verify that the output-side safety guard correctly:
//! - Passes balanced predictions (residual ≈ 0)
//! - Triggers rerouting for corrupted predictions
//! - Handles threshold boundaries correctly

use fluxion::ai::surrogate::{
    ResidualViolation, SurrogateInputs, SurrogateManager, DEFAULT_RESIDUAL_TAU,
};

#[test]
fn test_residual_violation_display() {
    let violation = ResidualViolation {
        sample_index: 0,
        predicted: 100.5,
        expected: 50.0,
        residual: 2550.25,
    };
    let msg = violation.to_string();
    assert!(msg.contains("surrogate residual violation"));
    assert!(msg.contains("sample 0"));
    assert!(msg.contains("100.50 W"));
    assert!(msg.contains("50.00 W"));
    assert!(msg.contains("2550.25 W²"));
}

#[test]
fn test_default_residual_tau_is_one() {
    assert_eq!(DEFAULT_RESIDUAL_TAU, 1.0);
}

#[test]
fn test_check_inference_residual_passes_when_balanced() {
    let mut manager = SurrogateManager::new().unwrap();
    manager.model_loaded = true;

    // Inputs and predicted have matching lengths (one sample).
    // At hour 0, q_expected ≈ 10 W (from internal gains).
    // We predict exactly 10.0, so residual = 0.
    let inputs = vec![20.0, 20.0];
    let predicted = vec![10.0];

    let result = manager.check_inference_residual(&inputs, &predicted);
    assert!(
        result.is_ok(),
        "balanced prediction should not violate residual: {:?}",
        result
    );
}

#[test]
fn test_check_inference_residual_fails_when_corrupted() {
    let mut manager = SurrogateManager::new().unwrap();
    manager.model_loaded = true;

    // Inputs and predicted have matching lengths (one sample).
    // At any hour, predicting 10 kW when expected is ~10-50 W far exceeds tau=1.0 W².
    let inputs = vec![20.0, 20.0];
    let predicted = vec![10000.0];

    let result = manager.check_inference_residual(&inputs, &predicted);
    assert!(
        result.is_err(),
        "corrupted prediction should violate residual: {:?}",
        result
    );

    let violation = result.unwrap_err();
    assert_eq!(violation.sample_index, 0);
    assert!((violation.predicted - 10000.0).abs() < 1e-6);
    assert!(violation.residual > DEFAULT_RESIDUAL_TAU);
}

#[test]
fn test_check_inference_residual_threshold_boundary_at_tau() {
    let mut manager = SurrogateManager::new().unwrap();
    manager.model_loaded = true;
    manager.set_residual_tau(1.0);

    // At hour 0, q_expected ≈ 10 W. If predicted = 11.0, residual = 1.0.
    // This is exactly at tau, and the check uses strict > comparison,
    // so it should pass.
    let inputs = vec![20.0, 20.0];
    let predicted = vec![11.0];

    let result = manager.check_inference_residual(&inputs, &predicted);
    assert!(
        result.is_ok(),
        "residual exactly at tau should pass: {:?}",
        result
    );
}

#[test]
fn test_check_inference_residual_threshold_boundary_above_tau() {
    let mut manager = SurrogateManager::new().unwrap();
    manager.model_loaded = true;
    manager.set_residual_tau(1.0);

    // At hour 0, q_expected ≈ 10 W. If predicted = 11.1, residual = 1.21 > 1.0.
    // This exceeds tau and should fail.
    let inputs = vec![20.0, 20.0];
    let predicted = vec![11.1];

    let result = manager.check_inference_residual(&inputs, &predicted);
    assert!(
        result.is_err(),
        "residual above tau should fail: {:?}",
        result
    );
}

#[test]
fn test_check_inference_residual_length_mismatch_returns_ok() {
    let manager = SurrogateManager::new().unwrap();
    // Length mismatch: 3 inputs but only 2 predictions
    let inputs = vec![20.0, 20.0, 22.0];
    let predicted = vec![10.0, 10.0];

    let result = manager.check_inference_residual(&inputs, &predicted);
    assert!(
        result.is_ok(),
        "length mismatch should return Ok without violation"
    );
}

#[test]
fn test_check_inference_residual_no_model_loaded_returns_ok() {
    let manager = SurrogateManager::new().unwrap();
    assert!(!manager.model_loaded);

    // Lengths match, but since no model is loaded, check is skipped.
    let inputs = vec![20.0, 20.0];
    let predicted = vec![10000.0];

    let result = manager.check_inference_residual(&inputs, &predicted);
    assert!(
        result.is_ok(),
        "check should be skipped when no model loaded: {:?}",
        result
    );
}

#[test]
fn test_residual_reroute_count_increments() {
    let manager = SurrogateManager::new().unwrap();
    *manager.residual_reroute_count.lock() += 1;
    assert_eq!(manager.residual_reroute_count(), 1);

    *manager.residual_reroute_count.lock() += 1;
    assert_eq!(manager.residual_reroute_count(), 2);
}

#[test]
fn test_reset_residual_reroute_count() {
    let mut manager = SurrogateManager::new().unwrap();
    *manager.residual_reroute_count.lock() += 5;
    assert_eq!(manager.residual_reroute_count(), 5);

    manager.reset_residual_reroute_count();
    assert_eq!(manager.residual_reroute_count(), 0);
}

#[test]
fn test_set_residual_tau() {
    let mut manager = SurrogateManager::new().unwrap();
    assert_eq!(manager.residual_tau, DEFAULT_RESIDUAL_TAU);

    manager.set_residual_tau(5.0);
    assert_eq!(manager.residual_tau, 5.0);

    manager.set_residual_tau(0.01);
    assert_eq!(manager.residual_tau, 0.01);
}

#[test]
fn test_surrogate_inputs_from_temps_for_residual_check() {
    let temps = vec![10.0, 22.0];
    let inputs = SurrogateInputs::from_temps(&temps);

    assert_eq!(inputs.exterior_temp, 10.0);
    assert_eq!(inputs.zone_temp, 22.0);
    assert!(inputs.solar_rad >= 0.0);
    assert!(inputs.occupancy >= 0.0);
}

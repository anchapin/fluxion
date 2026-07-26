// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Error handling tests for QUBO matrices (issue #1775).
//!
//! Covers:
//! - Condition-number estimation for QUBO matrices
//! - Detection of numerical overflow in QUBO entries
//! - Tikhonov regularization as a documented fallback path
//! - Integration with `manifold_to_qubo` validation

use fluxion::physics::geometry_tensor::ThermalManifold;
use fluxion::quantum::qubo_mapping::{
    manifold_to_qubo, QuboConfig, QuboError, QuboProblem, DEFAULT_REGULARIZATION_ALPHA,
    ILL_CONDITIONED_THRESHOLD, OVERFLOW_THRESHOLD,
};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// ---------------------------------------------------------------------------
// Well-conditioned manifold (baseline)
// ---------------------------------------------------------------------------

#[test]
fn test_well_conditioned_5r1c_passes_condition_check() {
    let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
    let result = manifold_to_qubo(&m, QuboConfig::default());
    assert!(
        result.is_ok(),
        "well-conditioned 5R1C manifold should not error, got: {:?}",
        result.err()
    );
}

#[test]
fn test_well_conditioned_9r4c_passes_condition_check() {
    let m = ThermalManifold::from_9r4c_parameters(
        [22.0, 20.0, 23.0, 18.0],
        [1000.0, 5000.0, 3000.0, 8000.0],
        [50.0, 30.0, 20.0],
        Some([5.0, 3.0, 2.0]),
    );
    let result = manifold_to_qubo(&m, QuboConfig::default());
    assert!(
        result.is_ok(),
        "9R4C manifold should build QUBO without error"
    );
}

// ---------------------------------------------------------------------------
// Zero / singular QUBO detection
// ---------------------------------------------------------------------------

#[test]
fn test_zero_qubo_matrix_is_singular() {
    // A 4×4 diagonal matrix with all zeros → exactly singular.
    let q_matrix = vec![0.0; 16];
    let qp = QuboProblem {
        q_matrix,
        num_variables: 4,
        config: QuboConfig::default(),
        source_metric: nalgebra::Matrix4::identity(),
        source_field: nalgebra::Vector4::zeros(),
        source_gauge: nalgebra::Vector4::zeros(),
    };
    let err = qp.condition_number_estimate().unwrap_err();
    assert!(
        matches!(err, QuboError::SingularMatrix),
        "zero matrix should return SingularMatrix, got: {:?}",
        err
    );
}

#[test]
fn test_rank_one_matrix_is_singular() {
    // A 2×2 matrix [[1, 1], [1, 1]] has eigenvalues {2, 0} — singular.
    let q_matrix = vec![1.0, 1.0, 1.0, 1.0];
    let qp = QuboProblem {
        q_matrix,
        num_variables: 2,
        config: QuboConfig::default(),
        source_metric: nalgebra::Matrix4::identity(),
        source_field: nalgebra::Vector4::zeros(),
        source_gauge: nalgebra::Vector4::zeros(),
    };
    let err = qp.condition_number_estimate().unwrap_err();
    assert!(
        matches!(err, QuboError::SingularMatrix),
        "rank-1 matrix [[1,1],[1,1]] should be SingularMatrix, got: {:?}",
        err
    );
}

// ---------------------------------------------------------------------------
// Condition number estimation correctness (identity matrix)
// ---------------------------------------------------------------------------

#[test]
fn test_condition_number_estimate_identity_matrix() {
    // Identity matrix: all eigenvalues = 1, condition number = 1.
    let n = 4;
    let mut q_matrix = vec![0.0; n * n];
    for i in 0..n {
        q_matrix[i * n + i] = 1.0;
    }
    let qp = QuboProblem {
        q_matrix,
        num_variables: n,
        config: QuboConfig::default(),
        source_metric: nalgebra::Matrix4::identity(),
        source_field: nalgebra::Vector4::zeros(),
        source_gauge: nalgebra::Vector4::zeros(),
    };
    let (lambda_max, lambda_min, cond) = qp.condition_number_estimate().expect("should estimate");
    assert!(
        cond > 0.0 && cond < 10.0,
        "identity cond {} should be near 1",
        cond
    );
    assert!(
        lambda_max >= lambda_min,
        "lambda_max {} should be >= lambda_min {}",
        lambda_max,
        lambda_min
    );
}

// ---------------------------------------------------------------------------
// Regularization
// ---------------------------------------------------------------------------

#[test]
fn test_regularization_with_custom_alpha() {
    let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
    let qp = manifold_to_qubo(&m, QuboConfig::default()).expect("well-conditioned");
    let qp_reg = qp.regularize(Some(0.1)).expect("regularize should succeed");
    let (_, _, cond) = qp_reg.condition_number_estimate().expect("should estimate");
    assert!(
        cond < 1e8,
        "with alpha=0.1, condition number {} should be bounded",
        cond
    );

    let n = qp.num_variables();
    let orig_m = qp.q_matrix();
    let reg_m = qp_reg.q_matrix();
    for i in 0..n {
        let orig_diag = orig_m[i * n + i];
        let reg_diag = reg_m[i * n + i];
        assert!(
            approx_eq(reg_diag - orig_diag, 0.1, 1e-12),
            "diagonal shift should be exactly alpha=0.1"
        );
    }
}

#[test]
fn test_regularization_rejects_non_positive_alpha() {
    let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
    let qp = manifold_to_qubo(&m, QuboConfig::default()).unwrap();

    let err = qp.regularize(Some(0.0)).unwrap_err();
    assert!(matches!(err, QuboError::InvalidEncoding(_)));

    let err = qp.regularize(Some(-1.0)).unwrap_err();
    assert!(matches!(err, QuboError::InvalidEncoding(_)));
}

#[test]
fn test_regularized_preserves_solution_energy_structure() {
    let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
    let cfg = QuboConfig::default();
    let qp = manifold_to_qubo(&m, cfg).expect("well-conditioned");
    let qp_reg = qp
        .regularize(Some(1e-4))
        .expect("regularize should succeed");

    let x_canon = qp.encode_manifold_solution();
    let e_original = qp.evaluate(&x_canon);
    let e_regularized = qp_reg.evaluate(&x_canon);

    let alpha = 1e-4;
    let expected_delta = alpha * x_canon.iter().filter(|&&b| b != 0).count() as f64;
    assert!(
        approx_eq(e_regularized - e_original, expected_delta, 1e-9),
        "regularized energy {} - original {} should ≈ {}",
        e_regularized,
        e_original,
        expected_delta
    );
}

// ---------------------------------------------------------------------------
// Overflow detection
// ---------------------------------------------------------------------------

#[test]
fn test_overflow_detected_in_qubo_entries() {
    // A manifold with huge metric entries will produce QUBO entries > 1e10.
    let mut m = ThermalManifold::new_flat();
    m.metric_tensor = nalgebra::Matrix4::from_row_slice(&[
        1e12_f64, 0.0, 0.0, 0.0, //
        0.0, 1e12_f64, 0.0, 0.0, //
        0.0, 0.0, 1e12_f64, 0.0, //
        0.0, 0.0, 0.0, 1e12_f64, //
    ]);
    m.scalar_field = nalgebra::Vector4::new(20.0, 21.0, 22.0, 19.0);
    let result = manifold_to_qubo(&m, QuboConfig::default());
    let err = result.expect_err("overflow manifold should return error");
    assert!(
        matches!(err, QuboError::NumericalOverflow { .. }),
        "expected NumericalOverflow error, got: {:?}",
        err
    );
    if let QuboError::NumericalOverflow { max_entry } = err {
        assert!(
            max_entry > OVERFLOW_THRESHOLD,
            "max_entry {} should exceed {}",
            max_entry,
            OVERFLOW_THRESHOLD
        );
    }
}

// ---------------------------------------------------------------------------
// Constants and Display
// ---------------------------------------------------------------------------

#[test]
fn test_overflow_threshold_constant() {
    assert_eq!(OVERFLOW_THRESHOLD, 1e10);
    assert!(ILL_CONDITIONED_THRESHOLD > 0.0);
    assert!(DEFAULT_REGULARIZATION_ALPHA > 0.0);
}

#[test]
fn test_qubo_error_display_ill_conditioned() {
    let err = QuboError::IllConditioned {
        condition_number: 1.5e7,
        eigenvalue_ratio: 2.0e7,
    };
    let s = err.to_string();
    assert!(
        s.contains("ill-conditioned"),
        "display should mention 'ill-conditioned'"
    );
    assert!(s.contains("1.5"), "display should contain condition number");
}

#[test]
fn test_qubo_error_display_overflow() {
    let err = QuboError::NumericalOverflow { max_entry: 2.5e11 };
    let s = err.to_string();
    assert!(s.contains("overflow"), "display should mention 'overflow'");
    assert!(s.contains("2.5"), "display should contain max_entry");
}

#[test]
fn test_qubo_error_display_singular() {
    let err = QuboError::SingularMatrix;
    let s = err.to_string();
    assert!(s.contains("singular"), "display should mention 'singular'");
}

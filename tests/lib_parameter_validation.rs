//! Parameter validation tests for BatchOracle (src/lib.rs:830-908)
//!
//! Tests the validate_parameters() function indirectly via evaluate_population()
//! which validates parameters upfront and returns NaN for invalid ones.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

fn create_test_oracle() -> BatchOracle {
    let model = ThermalModel::<VectorField>::new(1);
    BatchOracle::from_model(model)
}

#[test]
fn test_validate_parameters_valid() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].is_finite(),
        "Expected finite EUI, got {}",
        results[0]
    );
}

#[test]
fn test_validate_parameters_u_value_nan() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![f64::NAN, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    // Invalid params should return NaN
    assert!(
        results[0].is_nan(),
        "Expected NaN for invalid U-value, got {}",
        results[0]
    );
}

#[test]
fn test_validate_parameters_u_value_inf() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![f64::INFINITY, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_u_value_negative_inf() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![f64::NEG_INFINITY, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_u_value_too_low() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![0.05, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(
        results[0].is_nan(),
        "Expected NaN for U-value 0.05, got {}",
        results[0]
    );
}

#[test]
fn test_validate_parameters_u_value_too_high() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![5.5, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_u_value_at_boundary_low() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![0.1, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_u_value_at_boundary_high() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![5.0, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_heating_setpoint_nan() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, f64::NAN, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_heating_setpoint_too_low() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 10.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_heating_setpoint_too_high() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 30.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_heating_setpoint_at_boundary_low() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 15.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_heating_setpoint_at_boundary_high() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 25.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_cooling_setpoint_nan() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, f64::NAN]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_cooling_setpoint_too_low() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 18.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_cooling_setpoint_too_high() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 35.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_cooling_setpoint_at_boundary_low() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 22.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_cooling_setpoint_at_boundary_high() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 32.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_heating_greater_than_cooling() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 25.0, 22.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_heating_equals_cooling() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 24.0, 24.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_nan());
}

#[test]
fn test_validate_parameters_partial_vector_u_value_only() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_partial_vector_u_value_and_heating() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_empty_vector() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_longer_vector() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 27.0, 100.0, 200.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results[0].is_finite());
}

#[test]
fn test_validate_parameters_multiple_configs_mixed_valid_invalid() {
    let oracle = create_test_oracle();
    let population = vec![
        vec![1.5, 20.0, 27.0],  // Valid
        vec![0.05, 20.0, 27.0], // Invalid U-value
        vec![2.0, 21.0, 28.0],  // Valid
        vec![1.5, 30.0, 27.0],  // Invalid heating setpoint
    ];
    let result = oracle.evaluate_population(population, false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 4);
    assert!(results[0].is_finite(), "Config 0 should be valid");
    assert!(results[1].is_nan(), "Config 1 should be invalid (U-value)");
    assert!(results[2].is_finite(), "Config 2 should be valid");
    assert!(
        results[3].is_nan(),
        "Config 3 should be invalid (heating setpoint)"
    );
}

#[test]
fn test_validate_parameters_empty_population() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_validate_parameters_multiple_valid_configs() {
    let oracle = create_test_oracle();
    let population = vec![
        vec![1.5, 20.0, 27.0],
        vec![2.0, 21.0, 28.0],
        vec![3.0, 22.0, 29.0],
    ];
    let result = oracle.evaluate_population(population, false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 3);
    for (i, eui) in results.iter().enumerate() {
        assert!(eui.is_finite(), "Config {} should have finite EUI", i);
    }
}

#[test]
fn test_validate_parameters_boundary_values() {
    let oracle = create_test_oracle();
    // Test all boundary combinations
    let test_cases = vec![
        (0.1, 15.0, 22.0), // All minimums
        (5.0, 25.0, 32.0), // All maximums
        (0.1, 25.0, 32.0), // Min U, max setpoints
        (5.0, 15.0, 22.0), // Max U, min setpoints
    ];
    for (u, h, c) in test_cases {
        let result = oracle.evaluate_population(vec![vec![u, h, c]], false);
        assert!(result.is_ok(), "Failed for params [{}, {}, {}]", u, h, c);
        let results = result.unwrap();
        assert!(
            results[0].is_finite(),
            "Expected finite EUI for [{}, {}, {}], got NaN",
            u,
            h,
            c
        );
    }
}

#[test]
fn test_validate_parameters_just_outside_boundaries() {
    let oracle = create_test_oracle();
    let test_cases = vec![
        (0.09, 20.0, 27.0), // U-value just below min
        (5.01, 20.0, 27.0), // U-value just above max
        (1.5, 14.9, 27.0),  // Heating just below min
        (1.5, 25.1, 27.0),  // Heating just above max
        (1.5, 20.0, 21.9),  // Cooling just below min
        (1.5, 20.0, 32.1),  // Cooling just above max
    ];
    for (u, h, c) in test_cases {
        let result = oracle.evaluate_population(vec![vec![u, h, c]], false);
        assert!(result.is_ok());
        let results = result.unwrap();
        assert!(
            results[0].is_nan(),
            "Expected NaN for [{}, {}, {}]",
            u,
            h,
            c
        );
    }
}

#[test]
fn test_validate_parameters_just_inside_boundaries() {
    let oracle = create_test_oracle();
    let test_cases = vec![
        (0.11, 20.0, 27.0), // U-value just above min
        (4.99, 20.0, 27.0), // U-value just below max
        (1.5, 15.1, 27.0),  // Heating just above min
        (1.5, 24.9, 27.0),  // Heating just below max
        (1.5, 20.0, 22.1),  // Cooling just above min
        (1.5, 20.0, 31.9),  // Cooling just below max
    ];
    for (u, h, c) in test_cases {
        let result = oracle.evaluate_population(vec![vec![u, h, c]], false);
        assert!(result.is_ok());
        let results = result.unwrap();
        assert!(
            results[0].is_finite(),
            "Expected finite EUI for [{}, {}, {}]",
            u,
            h,
            c
        );
    }
}

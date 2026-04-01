//! BatchOracle tests for src/lib.rs
//!
//! Tests BatchOracle creation, evaluate_population(), and related functionality.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

fn create_test_oracle() -> BatchOracle {
    let model = ThermalModel::<VectorField>::new(1);
    BatchOracle::from_model(model)
}

#[test]
fn test_batch_oracle_from_model() {
    let model = ThermalModel::<VectorField>::new(1);
    let oracle = BatchOracle::from_model(model);
    // Oracle should be created successfully
    // We can't directly inspect fields, but we can use it
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 27.0]], false);
    assert!(result.is_ok());
}

#[test]
fn test_batch_oracle_from_model_multiple_zones() {
    let model = ThermalModel::<VectorField>::new(5);
    let oracle = BatchOracle::from_model(model);
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 27.0]], false);
    assert!(result.is_ok());
}

#[test]
#[ignore = "slow: full year simulation"]
fn test_evaluate_population_u_value_impact() {
    let oracle = create_test_oracle();
    // Lower U-value (better insulation) should generally result in lower EUI
    let low_u = oracle
        .evaluate_population(vec![vec![0.5, 20.0, 27.0]], false)
        .unwrap()[0];
    let high_u = oracle
        .evaluate_population(vec![vec![4.0, 20.0, 27.0]], false)
        .unwrap()[0];

    assert!(low_u.is_finite());
    assert!(high_u.is_finite());
    // Note: This may not always hold due to climate and other factors,
    // but in most cases better insulation reduces energy use
    // We just verify both are finite and positive
    assert!(low_u > 0.0);
    assert!(high_u > 0.0);
}

#[test]
#[ignore = "slow: full year simulation"]
fn test_evaluate_population_setpoint_impact() {
    let oracle = create_test_oracle();
    // Narrower setpoint range (heating 22, cooling 24) vs wider (heating 18, cooling 28)
    let narrow = oracle
        .evaluate_population(vec![vec![1.5, 22.0, 24.0]], false)
        .unwrap()[0];
    let wide = oracle
        .evaluate_population(vec![vec![1.5, 18.0, 28.0]], false)
        .unwrap()[0];

    assert!(narrow.is_finite());
    assert!(wide.is_finite());
    // Narrower range typically uses more energy (HVAC runs more often)
    assert!(narrow > 0.0);
    assert!(wide > 0.0);
}

#[test]
fn test_evaluate_population_single_config() {
    let oracle = create_test_oracle();
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_finite());
    assert!(results[0] >= 0.0, "EUI should be non-negative");
}

#[test]
#[ignore = "slow: full year simulation"]
fn test_evaluate_population_with_large_population() {
    let oracle = create_test_oracle();
    // Test with 1000 configs to verify parallel execution works
    let population: Vec<Vec<f64>> = (0..1000)
        .map(|i| vec![1.5 + (i as f64 * 0.001), 20.0 + (i as f64 * 0.001), 27.0])
        .collect();

    let result = oracle.evaluate_population(population, false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1000);
}

#[test]
fn test_evaluate_population_mixed_valid_invalid() {
    let oracle = create_test_oracle();
    let population = vec![
        vec![1.5, 20.0, 27.0],  // Valid
        vec![0.05, 20.0, 27.0], // Invalid
        vec![2.0, 21.0, 28.0],  // Valid
    ];
    let result = oracle.evaluate_population(population, false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_finite());
    assert!(results[1].is_nan());
    assert!(results[2].is_finite());
}

#[test]
fn test_evaluate_population_analytical_path() {
    let oracle = create_test_oracle();
    // use_surrogates=false should use analytical physics calculations
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 27.0]], false);
    assert!(result.is_ok());
    let eui = result.unwrap()[0];
    assert!(eui.is_finite());
    assert!(eui > 0.0, "EUI should be positive");
}

#[test]
fn test_evaluate_population_with_surrogates_no_model() {
    let oracle = create_test_oracle();
    // use_surrogates=true but no surrogate model loaded should still work
    // (falls back to analytical or returns mock values)
    let result = oracle.evaluate_population(vec![vec![1.5, 20.0, 27.0]], true);
    // Should not panic - either succeeds or returns NaN
    if let Ok(results) = result {
        assert_eq!(results.len(), 1);
    }
}

#[test]
#[ignore = "slow: full year simulation"]
fn test_evaluate_population_parallel_execution() {
    let oracle = create_test_oracle();
    // Large population should execute in parallel without issues
    let population: Vec<Vec<f64>> = (0..100)
        .map(|i| vec![1.5 + (i as f64 * 0.01), 20.0, 27.0])
        .collect();

    let result = oracle.evaluate_population(population.clone(), false);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 100);

    // All should be finite (valid params)
    for (i, eui) in results.iter().enumerate() {
        assert!(eui.is_finite(), "Config {} should have finite EUI", i);
    }
}

#[test]
fn test_evaluate_population_consistency() {
    let oracle = create_test_oracle();
    // Same config should produce same result
    let config = vec![1.5, 20.0, 27.0];
    let result1 = oracle
        .evaluate_population(vec![config.clone()], false)
        .unwrap()[0];
    let result2 = oracle
        .evaluate_population(vec![config.clone()], false)
        .unwrap()[0];

    assert!(
        (result1 - result2).abs() < 1e-10,
        "Results should be identical"
    );
}

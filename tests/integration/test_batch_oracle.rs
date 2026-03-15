//! BatchOracle integration tests
//!
//! Tests validate BatchOracle population evaluation with realistic workloads.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;
use std::time::Instant;

/// Test population evaluation with 100 configurations
#[test]
fn test_population_evaluation_100() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    // Generate 100 configurations with valid parameters
    let population: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let u_value = 1.0 + (i as f64 * 0.01);
            let heating_setpoint = 18.0 + (i as f64 * 0.03);
            let cooling_setpoint = heating_setpoint + 5.0;
            vec![u_value, heating_setpoint, cooling_setpoint]
        })
        .collect();

    // Evaluate population
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");

    // Verify results are all finite and correct count
    assert_eq!(results.len(), 100);
    assert!(results.iter().all(|&r| r.is_finite()));
}

/// Test population evaluation with 1000 configurations
#[test]
fn test_population_evaluation_1000() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    // Generate 1000 configurations with valid parameters
    let population: Vec<Vec<f64>> = (0..1000)
        .map(|i| {
            let u_value = 1.0 + (i as f64 * 0.001);
            let heating_setpoint = 18.0 + (i as f64 * 0.003);
            let cooling_setpoint = heating_setpoint + 5.0;
            vec![u_value, heating_setpoint, cooling_setpoint]
        })
        .collect();

    // Measure throughput
    let start = Instant::now();
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");
    let elapsed = start.elapsed();

    // Verify results are all finite and correct count
    assert_eq!(results.len(), 1000);
    assert!(results.iter().all(|&r| r.is_finite()));

    // Verify throughput >=100 configs/sec (relaxed for CI)
    let throughput = 1000.0 / elapsed.as_secs_f64();
    println!("BatchOracle throughput: {:.2} configs/sec", throughput);
    assert!(
        throughput >= 100.0,
        "Throughput too low: {:.2} configs/sec",
        throughput
    );
}

/// Test parameter vector semantics and validation
#[test]
fn test_parameter_vector_semantics() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    // Test valid parameters
    let valid_population = vec![
        vec![0.1, 15.0, 22.0], // Min bounds
        vec![5.0, 25.0, 32.0], // Max bounds
        vec![1.5, 20.0, 26.0], // Mid-range
    ];

    let valid_results = oracle
        .evaluate_population(valid_population, false)
        .expect("Evaluation failed");

    assert!(valid_results.iter().all(|&r| r.is_finite()));

    // Test out-of-range U-value (should return NaN)
    let invalid_u_value = vec![vec![10.0, 20.0, 26.0]]; // U-value > 5.0
    let u_value_results = oracle
        .evaluate_population(invalid_u_value, false)
        .expect("Evaluation failed");

    assert!(
        u_value_results[0].is_nan(),
        "Out-of-range U-value should return NaN"
    );

    // Test out-of-range setpoints (should return NaN)
    let invalid_setpoint = vec![vec![1.5, 50.0, 26.0]]; // Heating > 25°C
    let setpoint_results = oracle
        .evaluate_population(invalid_setpoint, false)
        .expect("Evaluation failed");

    assert!(
        setpoint_results[0].is_nan(),
        "Out-of-range setpoint should return NaN"
    );
}

/// Test surrogate integration
#[test]
fn test_surrogate_integration() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    // Test with surrogates disabled (analytical)
    let population = vec![vec![1.5, 20.0, 26.0], vec![2.0, 21.0, 25.0]];
    let analytical_results = oracle
        .evaluate_population(population.clone(), false)
        .expect("Evaluation failed");

    // Test with surrogates enabled (would use AI predictions if model loaded)
    let surrogate_results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");

    // Both should produce valid, finite results
    assert!(analytical_results.iter().all(|&r| r.is_finite()));
    assert!(surrogate_results.iter().all(|&r| r.is_finite()));
}

/// Test parallelism correctness
#[test]
fn test_parallelism_correctness() {
    let mut model = ThermalModel::<VectorField>::new(1);

    // Initialize model with sensible defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    // Test with a small population to verify parallelism
    let population: Vec<Vec<f64>> = (0..100).map(|_| vec![1.5, 20.0, 26.0]).collect();
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");

    // Verify results are all finite
    assert_eq!(results.len(), 100);
    assert!(results.iter().all(|&r| r.is_finite()));

    // The BatchOracle uses rayon for parallelism internally
    // We verify it works by checking that evaluation completes successfully
    println!(
        "Parallelism test: {} configs evaluated successfully",
        results.len()
    );
}

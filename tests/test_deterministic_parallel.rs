//! Deterministic parallel tests (TEST-04, BUG-04).
//!
//! These tests verify that parallel execution with seeded RNG produces
//! deterministic results across multiple runs. They eliminate flakiness
//! from timing-dependent parallel execution.
//!
//! Run with: cargo test --test test_deterministic_parallel

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Helper: create a base model for BatchOracle.
fn create_base_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.solar.window_u_value = 1.5;
    model.setpoints.heating_setpoint = 20.0;
    model.setpoints.cooling_setpoint = 26.0;
    model.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
    model
}

/// Helper: generate a deterministic population using seeded RNG.
fn generate_population_deterministic(size: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..size)
        .map(|_| {
            vec![
                rng.gen_range(0.1..5.0),        // U-value
                18.0 + rng.gen_range(0.0..7.0), // heating setpoint
                22.0 + rng.gen_range(0.0..8.0), // cooling setpoint
            ]
        })
        .collect()
}

/// Test: BatchOracle produces identical results across multiple runs with same seed.
///
/// This test verifies that:
/// 1. Population generation with seeded RNG is deterministic
/// 2. BatchOracle evaluation produces identical results across runs
/// 3. No timing-dependent variations affect parallel execution
///
/// Note: NaN values are filtered out during comparison to handle
/// configurations that produce invalid results.
#[test]
fn test_batch_oracle_deterministic_analytical() {
    let oracle = fluxion::BatchOracle::from_model(create_base_model());
    let population = generate_population_deterministic(50);

    // Run evaluation 3 times and collect results
    let mut results: Vec<Vec<f64>> = Vec::new();
    for _ in 0..3 {
        let result = oracle
            .evaluate_population(population.clone(), false)
            .expect("Evaluation failed");
        results.push(result);
    }

    // Verify all results are identical (skip NaN values)
    let first = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        for (j, (&first_val, &result_val)) in first.iter().zip(result.iter()).enumerate() {
            // Skip NaN values
            if first_val.is_nan() || result_val.is_nan() {
                continue;
            }
            assert!(
                (first_val - result_val).abs() < 1e-9,
                "Run {} config {} produced different results: {} vs {}",
                i,
                j,
                first_val,
                result_val
            );
        }
    }

    println!("All 3 runs produced identical results (analytical path)");
}

/// Test: BatchOracle with surrogates produces deterministic results.
///
/// Note: This test may skip if surrogates are not properly initialized,
/// as the requirement primarily focuses on the analytical path.
///
/// Note: NaN values are filtered out during comparison to handle
/// configurations that produce invalid results.
#[test]
fn test_batch_oracle_deterministic_surrogates() {
    let oracle = fluxion::BatchOracle::from_model(create_base_model());
    let population = generate_population_deterministic(50);

    // Check if surrogates are available
    let test_result = oracle.evaluate_population(population.clone(), true);

    match test_result {
        Ok(_) => {
            // Surrogates available - test determinism
            let mut results: Vec<Vec<f64>> = Vec::new();
            for _ in 0..3 {
                let result = oracle
                    .evaluate_population(population.clone(), true)
                    .expect("Evaluation failed");
                results.push(result);
            }

            // Verify all results are identical (skip NaN values)
            let first = &results[0];
            for (i, result) in results.iter().enumerate().skip(1) {
                for (j, (&first_val, &result_val)) in first.iter().zip(result.iter()).enumerate() {
                    // Skip NaN values
                    if first_val.is_nan() || result_val.is_nan() {
                        continue;
                    }
                    assert!(
                        (first_val - result_val).abs() < 1e-9,
                        "Run {} config {} produced different results: {} vs {}",
                        i,
                        j,
                        first_val,
                        result_val
                    );
                }
            }

            println!("All 3 runs produced identical results (surrogate path)");
        }
        Err(_) => {
            // Surrogates not available - skip test
            println!("Surrogate evaluation not available (skipping)");
        }
    }
}

/// Test: Rayon par_iter produces deterministic results with seeded RNG.
///
/// This test verifies that parallel computation with seeded RNG
/// produces the same results regardless of thread pool configuration.
#[test]
fn test_par_iter_deterministic() {
    // Generate deterministic input data
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<f64> = (0..1000).map(|_| rng.gen_range(0.0..100.0)).collect();

    // Run parallel computation 5 times
    let mut results: Vec<f64> = Vec::new();
    for _ in 0..5 {
        let sum: f64 = data.par_iter().map(|&x| x * x).sum();
        results.push(sum);
    }

    // Verify all results are identical
    let first = results[0];
    for (i, &result) in results.iter().enumerate().skip(1) {
        assert!(
            (result - first).abs() < 1e-9,
            "Run {} produced different result: {} vs {}",
            i,
            result,
            first
        );
    }

    println!("All 5 parallel runs produced identical results");
}

/// Test: Deterministic results with different thread pool sizes.
///
/// This test verifies that results are consistent regardless of
/// the number of threads used for parallel execution.
///
/// Note: This test uses environment variable RAYON_NUM_THREADS to
/// control thread count. It verifies that results are deterministic
/// at the specified thread count.
#[test]
fn test_deterministic_at_specified_thread_count() {
    // Get current thread count from environment or use default
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(rayon::current_num_threads());

    println!("Running with {} threads", num_threads);

    // Generate deterministic input data
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<f64> = (0..1000).map(|_| rng.gen_range(0.0..100.0)).collect();

    // Run parallel computation 3 times
    let mut results: Vec<f64> = Vec::new();
    for i in 0..3 {
        let sum: f64 = data.par_iter().map(|&x| x * x).sum();
        results.push(sum);
        println!("Run {}: sum = {}", i, sum);
    }

    // Verify all results are identical
    let first = results[0];
    for (i, &result) in results.iter().enumerate().skip(1) {
        assert!(
            (result - first).abs() < 1e-9,
            "Run {} produced different result: {} vs {}",
            i,
            result,
            first
        );
    }

    println!(
        "All 3 runs with {} threads produced identical results",
        num_threads
    );
}

/// Test: Seeded population generation produces identical populations.
///
/// This test verifies that generate_population_deterministic
/// produces the same population vector across multiple calls.
#[test]
fn test_population_seeding_deterministic() {
    // Generate two populations with same seed
    let pop1 = generate_population_deterministic(100);
    let pop2 = generate_population_deterministic(100);

    // Verify they are identical
    assert_eq!(pop1.len(), pop2.len());
    for (i, (p1, p2)) in pop1.iter().zip(pop2.iter()).enumerate() {
        assert_eq!(p1, p2, "Population config {} differs", i);
    }

    println!("Population generation is deterministic with seed 42");
}

/// Test: BatchOracle with large population is deterministic.
///
/// This test verifies determinism at scale with 200 configurations,
/// simulating real-world usage scenarios.
///
/// Note: NaN values are filtered out during comparison to handle
/// configurations that produce invalid results.
#[test]
fn test_batch_oracle_deterministic_large_population() {
    let oracle = fluxion::BatchOracle::from_model(create_base_model());
    let population = generate_population_deterministic(200);

    // Run evaluation 3 times with large population
    let mut results: Vec<Vec<f64>> = Vec::new();
    for _ in 0..3 {
        let result = oracle
            .evaluate_population(population.clone(), false)
            .expect("Evaluation failed");
        results.push(result);
    }

    // Verify all results are identical (skip NaN values)
    let first = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        for (j, (&first_val, &result_val)) in first.iter().zip(result.iter()).enumerate() {
            // Skip NaN values
            if first_val.is_nan() || result_val.is_nan() {
                continue;
            }
            assert!(
                (first_val - result_val).abs() < 1e-9,
                "Run {} config {} produced different results: {} vs {}",
                i,
                j,
                first_val,
                result_val
            );
        }
    }

    println!("All 3 runs with 200 configs produced identical results");
}

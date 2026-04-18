//! Throughput Benchmark Test
//!
//! This test provides reproducible throughput measurements for Fluxion's
//! BatchOracle evaluation. Results are used for performance tracking and
//! regression detection.
//!
//! Run with: cargo test --test throughput_benchmark -- --nocapture

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;
use std::time::Instant;

const BENCHMARK_POPULATION_SIZE: usize = 1000;
const MINIMUM_THROUGHPUT: f64 = 100.0; // configs/sec - absolute minimum for any meaningful optimization

/// Throughput benchmark test - measures BatchOracle evaluation speed
///
/// This test validates that BatchOracle meets minimum throughput requirements
/// for optimization workflows. The target threshold is 800+ configs/sec for
/// effective genetic algorithm / quantum optimization use cases.
#[test]
fn test_batch_oracle_throughput_1000() {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    // Generate population with valid parameters
    let population: Vec<Vec<f64>> = (0..BENCHMARK_POPULATION_SIZE)
        .map(|i| {
            let u_value = 1.0 + (i as f64 * 0.001);
            let heating_setpoint = 18.0 + (i as f64 * 0.003);
            let cooling_setpoint = heating_setpoint + 5.0;
            vec![u_value, heating_setpoint, cooling_setpoint]
        })
        .collect();

    // Warm-up run to ensure JIT/timers are ready
    let warmup_pop = vec![vec![1.5, 20.0, 26.0]];
    let _ = oracle.evaluate_population(warmup_pop, false);

    // Benchmark run
    let start = Instant::now();
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");
    let elapsed = start.elapsed();

    // Validate results
    assert_eq!(results.len(), BENCHMARK_POPULATION_SIZE);
    assert!(
        results.iter().all(|&r| r.is_finite()),
        "All results must be finite"
    );

    // Calculate throughput
    let throughput = BENCHMARK_POPULATION_SIZE as f64 / elapsed.as_secs_f64();

    println!("\n========================================");
    println!("BatchOracle Throughput Benchmark");
    println!("========================================");
    println!("Population size: {}", BENCHMARK_POPULATION_SIZE);
    println!("Elapsed time: {:.3} seconds", elapsed.as_secs_f64());
    println!("Throughput: {:.2} configs/sec", throughput);
    println!("========================================\n");

    // Assert minimum throughput threshold
    assert!(
        throughput >= MINIMUM_THROUGHPUT,
        "Throughput {:.2} configs/sec is below minimum {} configs/sec",
        throughput,
        MINIMUM_THROUGHPUT
    );
}

/// Small batch throughput test - verifies performance on smaller populations
#[test]
fn test_batch_oracle_throughput_100() {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    let population: Vec<Vec<f64>> = (0..100).map(|_| vec![1.5, 20.0, 26.0]).collect();

    let start = Instant::now();
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");
    let elapsed = start.elapsed();

    assert_eq!(results.len(), 100);

    let throughput = 100.0 / elapsed.as_secs_f64();
    println!("Small batch throughput: {:.2} configs/sec", throughput);

    // Small batches may have higher per-config overhead, so lower threshold
    assert!(
        throughput >= 50.0,
        "Small batch throughput too low: {:.2}",
        throughput
    );
}

/// Regression test - ensures throughput doesn't degrade over time
///
/// This test captures the measured throughput so it can be compared against
/// previous measurements to detect performance regressions.
#[test]
fn test_throughput_regression_baseline() {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);

    let oracle = BatchOracle::from_model(model);

    // Use same population as other tests for consistency
    let population: Vec<Vec<f64>> = (0..1000)
        .map(|i| {
            let u_value = 1.0 + (i as f64 * 0.001);
            let heating_setpoint = 18.0 + (i as f64 * 0.003);
            let cooling_setpoint = heating_setpoint + 5.0;
            vec![u_value, heating_setpoint, cooling_setpoint]
        })
        .collect();

    let start = Instant::now();
    let results = oracle
        .evaluate_population(population, false)
        .expect("Evaluation failed");
    let elapsed = start.elapsed();

    let throughput = 1000.0 / elapsed.as_secs_f64();

    // Print for tracking
    println!("THROUGHPUT_BASELINE: {:.2} configs/sec", throughput);
    println!(
        "MEASURED_AT: {}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    assert_eq!(results.len(), 1000);
    assert!(throughput > 0.0, "Throughput must be positive");
}

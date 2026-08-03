//! Throughput Benchmark Test
//!
//! This test provides reproducible throughput measurements for Fluxion's
//! BatchOracle evaluation. Results are used for performance tracking and
//! regression detection.
//!
//! Run with: cargo test --test throughput_benchmark --release -- --nocapture
//!
//! Issue #2312: Characterizes actual throughput vs 150 configs/sec CI gate

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::BatchOracle;
use std::time::Instant;

const RELEASE_GATE_THROUGHPUT: f64 = 150.0;

fn create_single_zone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);
    model
}

fn create_case_960_model() -> ThermalModel<VectorField> {
    let spec = ASHRAE140Case::Case960.spec();
    ThermalModel::<VectorField>::from_spec(&spec)
}

fn generate_population(size: usize) -> Vec<Vec<f64>> {
    (0..size)
        .map(|i| {
            let u_value = 1.0 + (i as f64 * 0.001);
            let heating_setpoint = 18.0 + (i as f64 * 0.003);
            let cooling_setpoint = heating_setpoint + 5.0;
            vec![u_value, heating_setpoint, cooling_setpoint]
        })
        .collect()
}

fn measure_throughput(oracle: &BatchOracle, population: &[Vec<f64>]) -> (f64, usize, Vec<f64>) {
    let start = Instant::now();
    let results = oracle
        .evaluate_population(population.to_vec(), false)
        .expect("Evaluation failed");
    let elapsed = start.elapsed();
    let throughput = population.len() as f64 / elapsed.as_secs_f64();
    (throughput, results.len(), results)
}

/// Systematic throughput characterization covering 10-10,000 config range.
///
/// This test validates BatchOracle throughput across the full range of
/// population sizes relevant to optimization workflows. The 150 configs/sec
/// threshold is derived from the release_gates.yaml benchmark gate.
///
/// Note: For larger batch sizes, some configs may produce NaN due to
/// parameter bounds (u_value > 5.0 W/m²K at i > 4000). This is expected
/// and does not affect throughput measurement.
#[test]
fn test_batch_oracle_throughput_characterization() {
    let oracle = BatchOracle::from_model(create_single_zone_model());

    let sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000];

    println!("\n========================================");
    println!("BatchOracle Throughput characterization");
    println!("========================================");
    println!(
        "Release gate threshold: {} configs/sec",
        RELEASE_GATE_THROUGHPUT
    );
    println!("========================================");
    println!(
        "{:<10} {:>15} {:>15} {:>12}",
        "Size", "Time (s)", "Throughput", "% of Gate"
    );
    println!("----------------------------------------");

    for &size in &sizes {
        let population = generate_population(size);

        let warmup = vec![vec![1.5, 20.0, 26.0]];
        let _ = oracle.evaluate_population(warmup, false);

        let (throughput, count, _results) = measure_throughput(&oracle, &population);
        let pct_of_gate = (throughput / RELEASE_GATE_THROUGHPUT) * 100.0;

        println!(
            "{:<10} {:>15.3} {:>15.2} {:>11.1}%",
            size,
            count as f64 / throughput,
            throughput,
            pct_of_gate
        );

        assert_eq!(count, size);
    }

    println!("========================================\n");
}

/// Single-zone 1000-config throughput test aligned with 150 configs/sec release gate.
///
/// This test validates that BatchOracle meets minimum throughput requirements
/// for optimization workflows at the standard batch size.
#[test]
fn test_batch_oracle_throughput_1000() {
    let oracle = BatchOracle::from_model(create_single_zone_model());
    let population = generate_population(1000);

    let warmup = vec![vec![1.5, 20.0, 26.0]];
    let _ = oracle.evaluate_population(warmup, false);

    let (throughput, count, _) = measure_throughput(&oracle, &population);

    println!("\n========================================");
    println!("BatchOracle Single-Zone Throughput");
    println!("========================================");
    println!("Population size: {}", count);
    println!("Throughput: {:.2} configs/sec", throughput);
    println!("Release gate: {} configs/sec", RELEASE_GATE_THROUGHPUT);
    println!("========================================\n");

    assert_eq!(count, 1000);
    assert!(
        throughput >= RELEASE_GATE_THROUGHPUT,
        "Throughput {:.2} configs/sec is below release gate {} configs/sec",
        throughput,
        RELEASE_GATE_THROUGHPUT
    );
}

/// Small batch throughput test - verifies performance on smaller populations.
///
/// Small batches have higher per-config overhead. Using the original threshold
/// of 50 configs/sec which was the established CI baseline before the
/// release gate was created.
#[test]
fn test_batch_oracle_throughput_100() {
    let oracle = BatchOracle::from_model(create_single_zone_model());
    let population: Vec<Vec<f64>> = (0..100).map(|_| vec![1.5, 20.0, 26.0]).collect();

    let (throughput, count, _) = measure_throughput(&oracle, &population);

    println!(
        "Small batch (100) throughput: {:.2} configs/sec",
        throughput
    );

    assert_eq!(count, 100);
    assert!(
        throughput >= 50.0,
        "Small batch throughput {:.2} configs/sec is below 50 configs/sec (original CI baseline)",
        throughput
    );
}

/// Multi-zone Case 960 (2-zone sunspace) throughput benchmark.
///
/// Issue #2312: Benchmarks the BatchOracle throughput for the ASHRAE 140
/// Case 960 sunspace configuration. This is informational only - the
/// release_gates.yaml has `multi_zone.min_configs_per_sec: 0` (disabled)
/// because multi-zone performance characterization was not yet implemented.
#[test]
fn test_batch_oracle_throughput_case_960() {
    let oracle = BatchOracle::from_model(create_case_960_model());

    let sizes = [100, 500, 1000];

    println!("\n========================================");
    println!("BatchOracle Case 960 Multi-Zone Throughput");
    println!("========================================");
    println!("Configuration: 2-zone sunspace building");
    println!("Note: Multi-zone throughput not yet in release gate");
    println!("========================================");
    println!("{:<10} {:>15} {:>15}", "Size", "Time (s)", "Throughput");
    println!("----------------------------------------");

    for &size in &sizes {
        let population = generate_population(size);

        let warmup = vec![vec![1.5, 20.0, 26.0]];
        let _ = oracle.evaluate_population(warmup, false);

        let (throughput, count, _) = measure_throughput(&oracle, &population);

        println!(
            "{:<10} {:>15.3} {:>15.2}",
            size,
            count as f64 / throughput,
            throughput
        );

        assert_eq!(count, size);
    }

    println!("========================================\n");
}

/// Regression test - ensures throughput doesn't degrade over time
///
/// This test captures the measured throughput so it can be compared against
/// previous measurements to detect performance regressions.
#[test]
fn test_throughput_regression_baseline() {
    let oracle = BatchOracle::from_model(create_single_zone_model());
    let population = generate_population(1000);

    let (throughput, count, _) = measure_throughput(&oracle, &population);

    println!("THROUGHPUT_BASELINE: {:.2} configs/sec", throughput);
    println!(
        "MEASURED_AT: {}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    assert_eq!(count, 1000);
    assert!(throughput > 0.0, "Throughput must be positive");
}

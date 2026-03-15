//! Performance regression test for Fluxion
//!
//! This test measures performance against a stored baseline and fails if
//! performance degrades by more than 10%.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Baseline file location
const BASELINE_FILE: &str = "tests/perf_baseline.json";
/// Regression threshold (10%)
const REGRESSION_THRESHOLD: f64 = 0.10;

/// Load baseline metrics from JSON file
fn load_baseline() -> Option<BaselineMetrics> {
    let path = PathBuf::from(BASELINE_FILE);
    if !path.exists() {
        return None;
    }

    let content = fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    Some(BaselineMetrics {
        timestamp: json.get("timestamp")?.as_str()?.to_string(),
        throughput_analytical: json.get("throughput_analytical")?.as_f64()?,
        latency_ms: json.get("latency_ms")?.as_f64()?,
    })
}

/// Baseline metrics structure
struct BaselineMetrics {
    timestamp: String,
    throughput_analytical: f64,
    latency_ms: f64,
}

/// Performance metrics from a test run
struct PerformanceMetrics {
    elapsed_ms: f64,
    throughput: f64,
    latency_per_config_ms: f64,
}

/// Run a performance test and measure metrics
fn run_performance_test(population_size: usize) -> PerformanceMetrics {
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::BatchOracle;

    let base_model = ThermalModel::<VectorField>::new(1);
    let oracle = BatchOracle::from_model(base_model);

    // Generate test population with valid parameters
    let population: Vec<Vec<f64>> = (0..population_size)
        .map(|i| {
            let idx = i as f64;
            vec![
                1.5 + (idx % 10.0) * 0.1, // U-value: 1.5-2.5
                20.0 + (idx % 5.0),       // Heating: 20-24°C
                24.0 + (idx % 5.0),       // Cooling: 24-28°C
            ]
        })
        .collect();

    // Warm-up run
    let _ = oracle.evaluate_population(population.clone(), false);

    // Actual benchmark
    let start = Instant::now();
    let _ = oracle.evaluate_population(population.clone(), false);
    let elapsed = start.elapsed();

    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = population_size as f64 / elapsed.as_secs_f64();
    let latency_per_config_ms = elapsed_ms / population_size as f64;

    PerformanceMetrics {
        elapsed_ms,
        throughput,
        latency_per_config_ms,
    }
}

/// Check for performance regression
fn check_regression(current: &PerformanceMetrics, baseline: &BaselineMetrics) -> Option<f64> {
    let baseline_throughput = baseline.throughput_analytical;

    if baseline_throughput <= 0.0 {
        return None;
    }

    let percent_change = (current.throughput - baseline_throughput) / baseline_throughput;

    // Return Some(percent_change) if regression detected (>10% slowdown)
    if percent_change < -REGRESSION_THRESHOLD {
        Some(percent_change)
    } else {
        None
    }
}

/// Integration test: Performance regression detection
///
/// This test verifies that performance doesn't regress by more than 10%
/// from the stored baseline. Run with:
/// ```
/// cargo test performance_regression --release
/// ```
///
/// To update the baseline:
/// ```
/// python .githooks/perf-baseline.py --update-baseline
/// ```
#[test]
fn test_performance_regression() {
    let population_size = 100;

    // Run performance test
    let metrics = run_performance_test(population_size);

    println!("Performance metrics:");
    println!("  Population size: {}", population_size);
    println!("  Elapsed: {:.2}ms", metrics.elapsed_ms);
    println!("  Throughput: {:.0} configs/sec", metrics.throughput);
    println!(
        "  Latency per config: {:.3}ms",
        metrics.latency_per_config_ms
    );

    // Try to load baseline
    match load_baseline() {
        Some(baseline) => {
            println!("\nBaseline metrics:");
            println!(
                "  Throughput: {:.0} configs/sec",
                baseline.throughput_analytical
            );
            println!("  Latency: {:.3}ms", baseline.latency_ms);

            // Check for regression
            match check_regression(&metrics, &baseline) {
                Some(percent_change) => {
                    let change_percent = percent_change * 100.0;
                    panic!(
                        "PERFORMANCE REGRESSION DETECTED: {:.1}% slowdown\n\
                         Baseline: {:.0} configs/sec\n\
                         Current:  {:.0} configs/sec\n\
                         Threshold: -{}%",
                        change_percent,
                        baseline.throughput_analytical,
                        metrics.throughput,
                        (REGRESSION_THRESHOLD * 100.0) as i32
                    );
                }
                None => {
                    println!("\n✓ Performance OK: No regression detected");
                }
            }
        }
        None => {
            println!("\nℹ No baseline found. Run:");
            println!("  python .githooks/perf-baseline.py --update-baseline");
            println!("  to create a baseline for regression detection.");
        }
    }
}

/// Benchmark test: Quick smoke test for performance
///
/// This test verifies that performance meets minimum targets.
/// Run with:
/// ```
/// cargo test performance_smoke_test --release
/// ```
#[test]
fn test_performance_smoke_test() {
    let population_size = 100;
    let metrics = run_performance_test(population_size);

    // Target: >500 configs/sec in debug mode (1ms per config in release)
    // This is a smoke test to catch major regressions, not tight performance targets
    let min_throughput = 500.0;

    println!("Smoke test results:");
    println!(
        "  Throughput: {:.0} configs/sec (target: >{})",
        metrics.throughput, min_throughput
    );
    println!(
        "  Latency: {:.3}ms per config (target: <2ms)",
        metrics.latency_per_config_ms
    );

    assert!(
        metrics.throughput > min_throughput,
        "Performance below minimum target: {:.0} configs/sec (expected >{})",
        metrics.throughput,
        min_throughput
    );
}

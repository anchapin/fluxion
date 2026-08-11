//! Performance regression test for Fluxion
//!
//! This test measures performance against a stored baseline and fails if
//! performance degrades by more than the threshold defined in
//! `release_gates.yaml` (`benchmark.regression_threshold`, currently 5%).

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Baseline file location
const BASELINE_FILE: &str = "tests/perf_baseline.json";

/// Canonical location of the release-gate configuration (source of truth
/// for the regression threshold). Resolved relative to the workspace root,
/// which is the CWD when `cargo test` invokes this binary.
const RELEASE_GATES_FILE: &str = "release_gates.yaml";

/// Regression threshold fallback — **fraction** (0.05 = 5% slowdown allowed).
///
/// **Source of truth**: `release_gates.yaml` → `benchmark.regression_threshold`
/// (expressed as a percentage, currently `5.0`). The test reads that YAML at
/// runtime via [`regression_threshold`]; this constant is only a fallback for
/// environments where the YAML cannot be located (e.g. an exotic `--target-dir`
/// remap). The drift-guard test `test_regression_threshold_matches_yaml`
/// (below) asserts this constant and the YAML stay in sync, so the two cannot
/// silently diverge — if you change one, update the other (issue #2700).
const REGRESSION_THRESHOLD_FALLBACK: f64 = 0.05;

/// Read the regression threshold (as a fraction, 0.05 = 5%) from
/// `release_gates.yaml` → `benchmark.regression_threshold`.
///
/// Returns `None` if the file is absent or malformed; callers fall back to
/// [`REGRESSION_THRESHOLD_FALLBACK`]. Keeping the YAML as the source of truth
/// prevents the test/gate drift that issue #2700 was filed against.
fn regression_threshold_from_yaml() -> Option<f64> {
    let content = fs::read_to_string(RELEASE_GATES_FILE).ok()?;
    let yaml: serde_yaml::Value = serde_yaml::from_str(&content).ok()?;
    let pct = yaml
        .get("benchmark")?
        .get("regression_threshold")?
        .as_f64()?;
    // Reject nonsensical values (≤0 or >100%) rather than flipping the
    // comparison sign in `check_regression`.
    if !pct.is_finite() || pct <= 0.0 || pct > 100.0 {
        return None;
    }
    Some(pct / 100.0)
}

/// Resolve the active regression threshold: the YAML value if readable,
/// otherwise [`REGRESSION_THRESHOLD_FALLBACK`].
fn regression_threshold() -> f64 {
    regression_threshold_from_yaml().unwrap_or(REGRESSION_THRESHOLD_FALLBACK)
}

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
#[allow(dead_code)]
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
///
/// `threshold` is a fraction (e.g. 0.05 = 5%); a slowdown larger than this
/// fraction relative to baseline returns `Some(percent_change)`.
fn check_regression(
    current: &PerformanceMetrics,
    baseline: &BaselineMetrics,
    threshold: f64,
) -> Option<f64> {
    let baseline_throughput = baseline.throughput_analytical;

    if baseline_throughput <= 0.0 {
        return None;
    }

    let percent_change = (current.throughput - baseline_throughput) / baseline_throughput;

    // Return Some(percent_change) if regression detected (> threshold slowdown)
    if percent_change < -threshold {
        Some(percent_change)
    } else {
        None
    }
}

/// Integration test: Performance regression detection
///
/// This test verifies that performance doesn't regress by more than the
/// threshold documented in `release_gates.yaml`
/// (`benchmark.regression_threshold`, currently 5%). Run with:
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
    let threshold = regression_threshold();

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
    println!(
        "  Regression threshold: {:.1}% (from {})",
        threshold * 100.0,
        RELEASE_GATES_FILE
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
            match check_regression(&metrics, &baseline, threshold) {
                Some(percent_change) => {
                    let change_percent = percent_change * 100.0;
                    panic!(
                        "PERFORMANCE REGRESSION DETECTED: {:.1}% slowdown\n\
                         Baseline: {:.0} configs/sec\n\
                         Current:  {:.0} configs/sec\n\
                         Threshold: -{:.1}% (release_gates.yaml benchmark.regression_threshold)",
                        change_percent,
                        baseline.throughput_analytical,
                        metrics.throughput,
                        threshold * 100.0
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

/// Drift guard: the [`REGRESSION_THRESHOLD_FALLBACK`] constant MUST match
/// `release_gates.yaml` → `benchmark.regression_threshold`.
///
/// This is the "single source of truth" enforcement for issue #2700: even
/// though the regression test reads the YAML at runtime, the fallback
/// constant can still rot if someone changes the YAML without updating the
/// constant. This test fails CI in that case (the YAML is always present in
/// repo-rooted test runs), turning silent drift into a loud signal.
#[test]
fn test_regression_threshold_matches_yaml() {
    match regression_threshold_from_yaml() {
        Some(yaml_fraction) => {
            assert!(
                (yaml_fraction - REGRESSION_THRESHOLD_FALLBACK).abs() < 1e-9,
                "REGRESSION_THRESHOLD_FALLBACK ({}) does not match \
                 release_gates.yaml benchmark.regression_threshold ({}). \
                 The YAML is the source of truth — update the constant to match.",
                REGRESSION_THRESHOLD_FALLBACK,
                yaml_fraction,
            );
        }
        None => {
            // The YAML was not readable in this environment (unusual — cargo
            // tests run with the workspace root as CWD). CI exercises the
            // normal path where the file is present, so drift is still
            // caught there; we just cannot assert it locally here.
            eprintln!(
                "warning: could not read {RELEASE_GATES_FILE}; \
                 drift guard skipped in this environment"
            );
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

    // Debug mode is much slower than release mode
    // In release: >500 configs/sec (target)
    // In debug: ~40-50 configs/sec is typical
    // In coverage mode (tarpaulin): much slower due to instrumentation
    #[cfg(tarpaulin)]
    let min_throughput = 5.0; // Tarpaulin is ~10x slower
    #[cfg(not(tarpaulin))]
    let min_throughput = if cfg!(debug_assertions) {
        20.0 // Much lower threshold for debug builds
    } else {
        // 150 configs/sec: conservative floor that catches real regressions
        // while tolerating CI machine variability (shared runners measure
        // ~150-200; dev machines/release gates enforce the 200+ target).
        150.0
    };

    println!("Smoke test results:");
    println!(
        "  Throughput: {:.0} configs/sec (target: >{})",
        metrics.throughput, min_throughput
    );

    #[cfg(tarpaulin)]
    let target_latency = 200.0; // Tarpaulin is much slower
    #[cfg(not(tarpaulin))]
    let target_latency = if cfg!(debug_assertions) { 50.0 } else { 10.0 }; // Aligned with release_gates.yaml

    println!(
        "  Latency: {:.3}ms per config (target: <{}ms)",
        metrics.latency_per_config_ms, target_latency
    );

    assert!(
        metrics.throughput > min_throughput,
        "Performance below minimum target: {:.0} configs/sec (expected >{})",
        metrics.throughput,
        min_throughput
    );
}

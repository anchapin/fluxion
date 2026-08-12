//! Performance regression test for Fluxion
//!
//! This test measures performance against a stored baseline and fails if
//! performance degrades by more than the threshold defined in
//! `release_gates.yaml` (`benchmark.regression_threshold`, currently 5%).

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

/// Load baseline metrics from JSON file.
///
/// Reads the optional `_meta` block written by
/// `scripts/generate_perf_baseline.py`: `enforcement` (`"hard-gate"` panics on
/// regression; `"report-only"` warns — for baselines measured on a non-CI
/// runner), `measured_at` (ISO `YYYY-MM-DD`, for staleness), and `runner_class`
/// (provenance). Returns `None` only when the file is absent or malformed.
fn load_baseline() -> Option<BaselineMetrics> {
    let path = PathBuf::from(BASELINE_FILE);
    if !path.exists() {
        return None;
    }

    let content = fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    let meta = json.get("_meta");
    let hard_gate = meta
        .and_then(|m| m.get("enforcement"))
        .and_then(|v| v.as_str())
        .map(|s| s == "hard-gate")
        .unwrap_or(true);
    let measured_at = meta
        .and_then(|m| m.get("measured_at"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let runner_class = meta
        .and_then(|m| m.get("runner_class"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    Some(BaselineMetrics {
        timestamp: json.get("timestamp")?.as_str()?.to_string(),
        throughput_analytical: json.get("throughput_analytical")?.as_f64()?,
        latency_ms: json.get("latency_ms")?.as_f64()?,
        hard_gate,
        measured_at,
        runner_class,
    })
}

/// Days since the UNIX epoch (1970-01-01) for a proleptic Gregorian date.
///
/// Howard Hinnant's `days_from_civil` algorithm. Verified equal to Python's
/// `datetime.date.toordinal()` arithmetic across leap years and century
/// boundaries (RULES.md constraint #0 — the date math was checked in Python,
/// not by hand). Uses Euclidean (`div_euclid`) division so it matches Python's
/// floor `//` for every sign of input.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y2 = y - if m <= 2 { 1 } else { 0 };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2).div_euclid(5) + d - 1;
    let doe = yoe * 365 + yoe.div_euclid(4) - yoe.div_euclid(100) + doy;
    era * 146097 + doe - 719468
}

/// Parse the leading `YYYY-MM-DD` of an ISO timestamp into days-since-epoch.
/// Returns `None` on any parse failure (caller treats as "no staleness info").
fn parse_date_to_days(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.len() < 10 {
        return None;
    }
    let mut parts = s[..10].split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

/// Current date as days-since-epoch (UTC), derived from the system clock with
/// no external crate. Returns `None` only if the system clock is before epoch.
fn current_date_days() -> Option<i64> {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some((secs / 86400) as i64)
}

/// Baseline metrics structure
///
/// `hard_gate` controls enforcement: `true` → a detected regression PANICS
/// (the historical behaviour); `false` → it prints a loud WARNING instead.
/// Defaults to `true` when `_meta.enforcement` is absent, so the test stays
/// strict by default. A baseline measured on a dev machine sets it `false`
/// because cross-runner throughput is not comparable within the 5% threshold
/// (issue #2680).
#[allow(dead_code)]
struct BaselineMetrics {
    timestamp: String,
    throughput_analytical: f64,
    latency_ms: f64,
    hard_gate: bool,
    measured_at: Option<String>,
    runner_class: Option<String>,
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
/// python3 scripts/generate_perf_baseline.py tests/perf_baseline.json 7
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

    // Load baseline. Issue #2680: a missing baseline used to make this test a
    // SILENT no-op (it printed "No baseline found" and returned Ok), so the
    // regression guard could never fire. It now FAILS LOUD so a missing or
    // unparseable baseline can never pass silently again.
    let baseline = match load_baseline() {
        Some(b) => b,
        None => {
            panic!(
                "PERF BASELINE MISSING: {}\n\
                 The performance-regression test cannot detect regressions without \
                 it — before issue #2680 it silently passed here, giving false \
                 confidence that performance was guarded. Regenerate by running the \
                 SAME harness CI uses (median-of-N):\n\
                 \n    python3 scripts/generate_perf_baseline.py tests/perf_baseline.json 7\n\
                 \n\
                 (or, with the Python bindings installed via `maturin develop \
                 --release`):\n    python .githooks/perf-baseline.py --update-baseline\n\
                 \n\
                 Commit the generated file. See issue #2680.",
                BASELINE_FILE
            );
        }
    };

    println!("\nBaseline metrics:");
    println!(
        "  Throughput: {:.0} configs/sec",
        baseline.throughput_analytical
    );
    println!("  Latency: {:.3}ms", baseline.latency_ms);
    println!(
        "  Runner:    {}  (enforcement: {})",
        baseline.runner_class.as_deref().unwrap_or("unknown"),
        if baseline.hard_gate {
            "hard-gate"
        } else {
            "report-only"
        }
    );

    // Staleness check (issue #2680, mirrors the KNOWN_ISSUES staleness gate).
    // A baseline older than 90 days is stale. Hard-gate baselines panic;
    // report-only baselines warn — a stale dev baseline must not red dev/main
    // CI, but it must not be invisible either.
    if let (Some(measured), Some(now)) = (baseline.measured_at.as_deref(), current_date_days()) {
        if let Some(m_days) = parse_date_to_days(measured) {
            let age = now - m_days;
            const STALE_AFTER_DAYS: i64 = 90;
            if age > STALE_AFTER_DAYS {
                let msg = format!(
                    "PERF BASELINE STALE: {} measured {age} days ago (>{STALE_AFTER_DAYS}).\n\
                     Regenerate via:\n    \
                     python3 scripts/generate_perf_baseline.py tests/perf_baseline.json 7\n\
                     and commit the result.",
                    BASELINE_FILE
                );
                if baseline.hard_gate {
                    panic!("{}", msg);
                } else {
                    eprintln!("⚠ {msg}");
                }
            } else {
                println!("  Baseline age: {age} days (fresh)");
            }
        }
    }

    // Regression check. For `report-only` baselines (e.g. measured on a dev
    // machine) a regression is a LOUD WARNING, not a panic: cross-runner
    // throughput routinely varies far more than the 5% threshold (a dev machine
    // may measure 1.5-10x the GitHub-hosted ubuntu-latest runner), so a hard
    // panic against a non-CI baseline would red dev/main CI on every push. The
    // machine-independent hard gates are the absolute floor (#2693, 150 cfg/s +
    // 10 ms) and the criterion same-runner artifact diff (#1618). A future
    // baseline generated ON ubuntu-latest may set `_meta.enforcement=hard-gate`
    // to restore a panic-on-regression here.
    match check_regression(&metrics, &baseline, threshold) {
        Some(percent_change) => {
            let change_percent = percent_change * 100.0;
            let headline = format!(
                "PERFORMANCE REGRESSION DETECTED: {:.1}% slowdown\n\
                 Baseline: {:.0} configs/sec (runner: {})\n\
                 Current:  {:.0} configs/sec\n\
                 Threshold: -{:.1}% (release_gates.yaml benchmark.regression_threshold)",
                change_percent,
                baseline.throughput_analytical,
                baseline.runner_class.as_deref().unwrap_or("unknown"),
                metrics.throughput,
                threshold * 100.0
            );
            if baseline.hard_gate {
                panic!("{}", headline);
            } else {
                eprintln!(
                    "⚠ {}\n\
                     NOTE: report-only baseline — this is a WARNING, not a failure. \
                     Cross-runner throughput is not comparable within 5%; the hard \
                     gates are #2693 (absolute floor) and #1618 (same-runner diff).",
                    headline
                );
            }
        }
        None => {
            println!("\n✓ Performance OK: No regression detected");
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

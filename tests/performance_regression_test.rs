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

    // Optional fields — read additively so older baselines (which predate
    // the field) keep loading. `multi_zone_throughput` (issue #2772) is the
    // only additive metric today; if absent the multi-zone regression test
    // takes its "no baseline" branch (loud warning, never a silent no-op).
    let multi_zone_throughput = json.get("multi_zone_throughput").and_then(|v| v.as_f64());

    Some(BaselineMetrics {
        timestamp: json.get("timestamp")?.as_str()?.to_string(),
        throughput_analytical: json.get("throughput_analytical")?.as_f64()?,
        latency_ms: json.get("latency_ms")?.as_f64()?,
        hard_gate,
        measured_at,
        runner_class,
        multi_zone_throughput,
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
    /// Optional 10-zone throughput baseline (configs/sec) for
    /// `test_multi_zone_performance_regression` (issue #2772). `None` on
    /// baselines predating the field — the multi-zone test then falls back
    /// to the absolute floor (`benchmark.multi_zone.min_configs_per_sec`,
    /// currently 10) and skips the relative-regression leg.
    multi_zone_throughput: Option<f64>,
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

/// Zone count enforced by the multi-zone throughput gate
/// (`release_gates.yaml`: `benchmark.multi_zone.zones: 10`).
const MULTI_ZONE_GATE_ZONES: usize = 10;

/// Fallback for the absolute multi-zone floor (configs/sec) when
/// `release_gates.yaml` is unreadable. Matches
/// `benchmark.multi_zone.min_configs_per_sec`. Drift-guarded by
/// [`test_multi_zone_floor_matches_yaml`].
const MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK: f64 = 10.0;

/// Read `benchmark.multi_zone.min_configs_per_sec` (the ABSOLUTE floor for
/// the 10-zone gate) from `release_gates.yaml`. Returns the fallback when
/// the file is absent or malformed (e.g. an exotic `--target-dir` remap).
fn multi_zone_floor_from_yaml() -> f64 {
    let content = match fs::read_to_string(RELEASE_GATES_FILE) {
        Ok(s) => s,
        Err(_) => return MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK,
    };
    let yaml = match serde_yaml::from_str::<serde_yaml::Value>(&content) {
        Ok(v) => v,
        Err(_) => return MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK,
    };
    let pct = yaml
        .get("benchmark")
        .and_then(|b| b.get("multi_zone"))
        .and_then(|m| m.get("min_configs_per_sec"))
        .and_then(|v| v.as_f64());
    match pct {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK,
    }
}

/// Synthetic-population generator matching the fixture used by
/// `benches/multi_zone_throughput.rs` and `tests/performance_ci_test.rs`
/// (the gate test), so this regression measurement is directly comparable
/// to the release-gate assertion. Parameter ranges are documented for
/// `BatchOracle::evaluate_population`:
/// - `[0]` U-value: 0.1-5.0 W/m²K
/// - `[1]` Heating setpoint: 15-25 °C
/// - `[2]` Cooling setpoint: 22-32 °C
fn generate_multi_zone_population(size: usize) -> Vec<Vec<f64>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut population = Vec::with_capacity(size);
    for _ in 0..size {
        let u_value = rng.random_range(0.1..5.0);
        let heating_setpoint = rng.random_range(15.0..25.0);
        let cooling_setpoint = rng.random_range(22.0..32.0);
        population.push(vec![u_value, heating_setpoint, cooling_setpoint]);
    }
    population
}

/// Run the multi-zone performance test and measure throughput.
///
/// Mirrors [`run_performance_test`] but constructs a 10-zone
/// `ThermalModel` (the configuration the
/// `benchmark.multi_zone.min_configs_per_sec` release gate actually
/// targets) and feeds the shared synthetic-population fixture, so a
/// measurement here is directly comparable to
/// `tests/performance_ci_test.rs::test_multi_zone_throughput` and
/// `benches/multi_zone_throughput.rs`. Analytical mode
/// (`use_surrogates = false`) avoids the ONNX runtime dependency,
/// matching the existing gate test and bench.
fn run_multi_zone_performance_test(population_size: usize) -> PerformanceMetrics {
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::BatchOracle;

    let base_model = ThermalModel::<VectorField>::new(MULTI_ZONE_GATE_ZONES);
    let oracle = BatchOracle::from_model(base_model);

    let population = generate_multi_zone_population(population_size);

    // Warm-up run — populates any lazy per-zone state on the cold path so
    // the measured run reflects steady-state throughput.
    let _ = oracle.evaluate_population(population.clone(), false);

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

/// Multi-zone performance regression test (Issue #2772).
///
/// The Absolute Perf Gate (#2693) measures the *single-zone analytical*
/// path against the 150 cfg/s floor — a much easier target than the
/// multi-zone gate (`release_gates.yaml` →
/// `benchmark.multi_zone.min_configs_per_sec: 10` for a 10-zone model).
/// `benches/multi_zone_throughput.rs:17` documents that population_10k
/// runs at ~28 cfg/sec — only 2.8× margin over the 10 cfg/s floor — yet
/// before this test the slowest regime (pop_10k) was only gated by
/// manual `cargo bench`, and `performance_ci_test.rs` only exercised
/// population=100 on 10-zone. A change adding per-zone cost could pass
/// the #2693 absolute gate while blowing the multi-zone margin.
///
/// This test closes that gap. It mirrors [`test_performance_regression`]
/// — median-of-3 measurement profile, identical 5% regression threshold
/// from `release_gates.yaml` → `benchmark.regression_threshold`,
/// `_meta.enforcement`-aware panic-vs-warn semantics — but runs
/// [`run_multi_zone_performance_test`] at `population_size = 1000`
/// (10× the existing CI throughput test, large enough that per-zone
/// cost dominates the inner rayon loop).
///
/// The test enforces TWO invariants:
/// 1. **Absolute floor** — `multi_zone_throughput ≥
///    benchmark.multi_zone.min_configs_per_sec` (currently 10 cfg/s).
///    This is a HARD assertion (panics in every mode, including
///    `report-only`), because the floor is a release-gate contract, not
///    a noise-prone cross-runner comparison. Tarpaulin / debug builds
///    relax it via cfg gates (matching [`test_performance_smoke_test`]).
/// 2. **Relative regression** — if a `multi_zone_throughput` baseline is
///    present in `tests/perf_baseline.json`, a slowdown beyond the 5%
///    threshold is a hard panic for `enforcement: hard-gate` baselines
///    and a loud WARNING for `report-only` baselines (cross-runner
///    throughput is not comparable within 5% — see [`test_performance_regression`]).
///
/// Run with:
/// ```
/// cargo test --test performance_regression_test test_multi_zone_performance_regression --release
/// ```
#[test]
fn test_multi_zone_performance_regression() {
    let population_size = 1000;
    let threshold = regression_threshold();
    let absolute_floor = multi_zone_floor_from_yaml();

    let metrics = run_multi_zone_performance_test(population_size);

    // The metric line the dashboard's median-of-3 parser greps for. The
    // exact label `Multi-zone throughput:` is matched by
    // `.github/workflows/performance_dashboard.yml` → `multi-zone-perf-gate`
    // — keep them in sync.
    println!("\nMulti-zone ({MULTI_ZONE_GATE_ZONES} zones) performance metrics:");
    println!("  Population size: {population_size}");
    println!("  Elapsed: {:.2}ms", metrics.elapsed_ms);
    println!(
        "  Multi-zone throughput: {:.0} configs/sec",
        metrics.throughput
    );
    println!(
        "  Latency per config: {:.3}ms",
        metrics.latency_per_config_ms
    );
    println!(
        "  Absolute floor: {:.0} configs/sec (from {} → benchmark.multi_zone.min_configs_per_sec)",
        absolute_floor, RELEASE_GATES_FILE
    );
    println!(
        "  Regression threshold: {:.1}% (from {} → benchmark.regression_threshold)",
        threshold * 100.0,
        RELEASE_GATES_FILE
    );

    // ─────────────────────────────────────────────────────────────────
    // 1) ABSOLUTE FLOOR — release-gate contract, always enforced.
    // ─────────────────────────────────────────────────────────────────
    // Tarpaulin / debug builds measure far below release throughput, so
    // the floor is relaxed for those build modes (mirrors
    // `test_performance_smoke_test`). Release builds must clear the
    // gate's `min_configs_per_sec` regardless of any baseline.
    #[cfg(tarpaulin)]
    let effective_floor = absolute_floor * 0.1; // Tarpaulin is ~10x slower
    #[cfg(not(tarpaulin))]
    let effective_floor = if cfg!(debug_assertions) {
        // Debug builds are also much slower; relax to 10% of the
        // release floor so the test stays useful as a smoke check
        // without red-herring on every dev `cargo test` run.
        absolute_floor * 0.1
    } else {
        absolute_floor
    };

    assert!(
        metrics.throughput >= effective_floor,
        "MULTI-ZONE ABSOLUTE FLOOR BREACH: {:.0} configs/sec < {:.0} (floor from \
         release_gates.yaml → benchmark.multi_zone.min_configs_per_sec, {} zones).\n\
         This is a release-gate contract — the multi-zone throughput gate is the \
         binding constraint (per issue #2772). Investigate per-zone cost regressions \
         before merging.",
        metrics.throughput,
        effective_floor,
        MULTI_ZONE_GATE_ZONES,
    );

    println!(
        "✓ Absolute floor OK: {:.0} ≥ {:.0} configs/sec",
        metrics.throughput, effective_floor
    );

    // ─────────────────────────────────────────────────────────────────
    // 2) RELATIVE REGRESSION — baseline-driven, enforcement-mode aware.
    // ─────────────────────────────────────────────────────────────────
    let baseline = load_baseline();
    let baseline_mz = baseline
        .as_ref()
        .and_then(|b| b.multi_zone_throughput)
        .filter(|v| v.is_finite() && *v > 0.0);

    if let (Some(b), Some(baseline_throughput)) = (baseline.as_ref(), baseline_mz) {
        let percent_change = (metrics.throughput - baseline_throughput) / baseline_throughput;
        let runner = b.runner_class.as_deref().unwrap_or("unknown");
        println!(
            "  Multi-zone baseline: {:.0} configs/sec (runner: {}, enforcement: {})",
            baseline_throughput,
            runner,
            if b.hard_gate {
                "hard-gate"
            } else {
                "report-only"
            }
        );

        if percent_change < -threshold {
            let change_percent = percent_change * 100.0;
            let headline = format!(
                "MULTI-ZONE PERFORMANCE REGRESSION DETECTED: {:.1}% slowdown\n\
                 Baseline: {:.0} configs/sec (runner: {}, {} zones)\n\
                 Current:  {:.0} configs/sec\n\
                 Threshold: -{:.1}% (release_gates.yaml benchmark.regression_threshold)",
                change_percent,
                baseline_throughput,
                runner,
                MULTI_ZONE_GATE_ZONES,
                metrics.throughput,
                threshold * 100.0,
            );
            if b.hard_gate {
                panic!("{headline}");
            } else {
                eprintln!(
                    "⚠ {headline}\n\
                     NOTE: report-only baseline — this is a WARNING, not a failure. \
                     Cross-runner throughput is not comparable within 5%; the hard \
                     absolute floor above is the gate (#2772 + benchmark.multi_zone).",
                );
            }
        } else {
            println!(
                "✓ Multi-zone regression check OK: {:.0} ≥ {:.0} configs/sec (within -{:.1}% threshold)",
                metrics.throughput,
                baseline_throughput,
                threshold * 100.0
            );
        }
    } else if baseline.is_some() {
        // Baseline loaded but no usable multi_zone_throughput field.
        // The absolute-floor assertion above still enforces the
        // release-gate contract; we just skip the relative-regression
        // leg with a loud signal — never a silent no-op (issue #2680).
        println!(
            "⚠ No `multi_zone_throughput` field in {BASELINE_FILE}; skipping the \
             relative-regression leg. The absolute-floor assertion above still \
             enforces the release-gate contract. Capture a baseline by adding a \
             `multi_zone_throughput` field (cfg/sec on a 10-zone model, pop 1000)."
        );
    } else {
        // `load_baseline` returned None — whole file missing/unparseable.
        // `test_performance_regression` already panics on this case (issue
        // #2680); here we only warn because the absolute floor above
        // still gates the run.
        println!(
            "⚠ {BASELINE_FILE} unreadable; skipping the relative-regression leg. \
             The absolute-floor assertion above still enforces the release-gate \
             contract."
        );
    }
}

/// Drift guard for [`MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK`].
///
/// Mirrors [`test_regression_threshold_matches_yaml`] for the multi-zone
/// floor: if someone changes `release_gates.yaml →
/// benchmark.multi_zone.min_configs_per_sec` without updating the
/// fallback constant (or vice-versa), this test fails CI. The YAML is
/// the source of truth — update the constant to match.
#[test]
fn test_multi_zone_floor_matches_yaml() {
    let content = match fs::read_to_string(RELEASE_GATES_FILE) {
        Ok(s) => s,
        Err(_) => {
            eprintln!(
                "warning: could not read {RELEASE_GATES_FILE}; \
                 multi-zone floor drift guard skipped in this environment"
            );
            return;
        }
    };
    let yaml: serde_yaml::Value = match serde_yaml::from_str(&content) {
        Ok(v) => v,
        Err(_) => {
            eprintln!(
                "warning: could not parse {RELEASE_GATES_FILE}; \
                 multi-zone floor drift guard skipped in this environment"
            );
            return;
        }
    };
    let Some(yaml_floor) = yaml
        .get("benchmark")
        .and_then(|b| b.get("multi_zone"))
        .and_then(|m| m.get("min_configs_per_sec"))
        .and_then(|v| v.as_f64())
    else {
        eprintln!(
            "warning: benchmark.multi_zone.min_configs_per_sec not found in \
             {RELEASE_GATES_FILE}; multi-zone floor drift guard skipped"
        );
        return;
    };
    assert!(
        (yaml_floor - MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK).abs() < 1e-9,
        "MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK ({}) does not match \
         release_gates.yaml benchmark.multi_zone.min_configs_per_sec ({}). \
         The YAML is the source of truth — update the constant to match.",
        MULTI_ZONE_MIN_CONFIGS_PER_SEC_FALLBACK,
        yaml_floor,
    );
}

//! HybridThermalModel throughput gate (Issue #2922).
//!
//! The Absolute Perf Gate (#2693) and Multi-Zone Perf Gate (#2772) measure
//! `BatchOracle::evaluate_population` over the pure-physics `ThermalModel<VectorField>`
//! path. They do NOT exercise the production-default routing
//! (`ThermalModelMode::Hybrid` → `HybridThermalModel` with `HybridRouting::default()`:
//! loads → surrogate, conduction / ventilation / HVAC → physics). A regression
//! adding 50 µs/timestep to the surrogate-load branch fires only in Hybrid
//! mode and would slip through both gates.
//!
//! This test closes that gap. It mirrors
//! `tests/performance_regression_test.rs::test_performance_regression` (single
//! zone, pop_100) and `::test_multi_zone_performance_regression` (10 zones,
//! pop_1000), but uses `HybridThermalModel` with default routing. CI runs each
//! test 3 times and takes the median (same noise-suppression discipline as
//! #2693 / #2772; the absolute floors below come from
//! `release_gates.yaml` → `benchmark.hybrid.{min_configs_per_sec,multi_zone.min_configs_per_sec}`).
//!
//! Acceptance (Issue #2922):
//! - `test_hybrid_performance_regression`: pop_100, ≥ 80 cfg/s
//!   (single-zone HybridThermalModel). The debug-mode arm uses a
//!   best-of-3 statistic with a dedicated calibrated floor (Issue #3894,
//!   un-quarantining #3892) — see [`HYBRID_DEBUG_FLOOR`]. The
//!   `--release` CI gate (#2922) stays authoritative.
//! - `test_hybrid_multi_zone_performance_regression`: pop_1000, ≥ 8 cfg/s
//!   (10-zone HybridThermalModel).
//!
//! The "Hybrid throughput:" / "Hybrid multi-zone throughput:" lines printed
//! below are matched by `.github/workflows/performance_dashboard.yml` →
//! `hybrid-perf-gate` — keep them in sync.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::thermal_model::{HybridRouting, HybridThermalModel, ThermalModelTrait};

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

const RELEASE_GATES_FILE: &str = "release_gates.yaml";

/// Single-zone population size (mirrors `test_performance_regression`).
const SINGLE_ZONE_POP: usize = 100;
/// 10-zone population size (mirrors `test_multi_zone_performance_regression`).
const MULTI_ZONE_POP: usize = 1000;
/// Zone count for the multi-zone workload
/// (`release_gates.yaml` → `benchmark.multi_zone.zones: 10`).
const MULTI_ZONE_GATE_ZONES: usize = 10;

/// Fallback absolute floor for single-zone HybridThermalModel (configs/sec),
/// used when `release_gates.yaml` cannot be read. Mirrors
/// `release_gates.yaml` → `benchmark.hybrid.min_configs_per_sec`.
const HYBRID_FLOOR_FALLBACK: f64 = 80.0;
/// Fallback absolute floor for 10-zone HybridThermalModel (configs/sec).
/// Mirrors `release_gates.yaml` → `benchmark.hybrid_multi_zone.min_configs_per_sec`.
const HYBRID_MULTI_ZONE_FLOOR_FALLBACK: f64 = 8.0;

/// Measured-pass count for the single-zone best-of-N statistic (Issue
/// #3894). 1 discarded warmup pass + 3 measured passes; the gate asserts
/// on the MAX (best-of-3). Composes with the dashboard's outer
/// median-of-3 across process invocations
/// (`.github/workflows/performance_dashboard.yml` → `hybrid-perf-gate`).
const SINGLE_ZONE_MEASURED_RUNS: usize = 3;

/// Debug-mode floor for the single-zone HybridThermalModel gate
/// (configs/sec, asserted against the best-of-3 statistic; Issue #3894).
///
/// Calibrated 2026-09-19 on the 12-core reference machine (develop
/// `e9bd46d` lineage, pop_100, debug build, best-of-3):
///
/// | Condition | Best-of-3 throughput | Verdict vs 26 cfg/s |
/// |---|---|---|
/// | Warm isolated | ~50 cfg/s | pass (~1.9× headroom) |
/// | Ambient load (load-avg ~7-12, desktop + browser) | 40+ cfg/s | pass |
/// | All 12 cores saturated (CPU spinners) | ~26-30 cfg/s | pass (marginal by design) |
/// | Genuine ≥ 2× regression of warm (≤ 25 cfg/s sustained) | ≤ 25 cfg/s | **FAIL** (required) |
///
/// Rationale: the #3892 re-triage showed full core saturation only halves
/// warm steady state (~50 → ~26 cfg/s) while a one-time cold-start
/// outlier measured ~1.1 cfg/s (~35× below steady state). The warmup
/// pass absorbs the cold start; best-of-3 absorbs transient load on any
/// single pass. The floor therefore sits ABOVE half of warm steady state
/// (> 25, so a genuine 2× regression sustained across all 3 passes still
/// trips it) and AT the documented saturated steady-state level (~26).
/// Release-mode floors stay authoritative from `release_gates.yaml`
/// (`benchmark.hybrid.min_configs_per_sec`, Issue #2922) — this constant
/// only relaxes the local/tarpaulin debug statistic.
const HYBRID_DEBUG_FLOOR: f64 = 26.0;

/// Read `benchmark.hybrid.min_configs_per_sec` from `release_gates.yaml`.
/// Returns the fallback when the file is absent or malformed.
fn hybrid_floor_from_yaml() -> f64 {
    hybrid_floor_with_key("hybrid", "min_configs_per_sec", HYBRID_FLOOR_FALLBACK)
}

/// Read `benchmark.hybrid_multi_zone.min_configs_per_sec` from
/// `release_gates.yaml`. Returns the fallback when absent/malformed.
fn hybrid_multi_zone_floor_from_yaml() -> f64 {
    hybrid_floor_with_key(
        "hybrid_multi_zone",
        "min_configs_per_sec",
        HYBRID_MULTI_ZONE_FLOOR_FALLBACK,
    )
}

fn hybrid_floor_with_key(section: &str, key: &str, fallback: f64) -> f64 {
    let content = match fs::read_to_string(RELEASE_GATES_FILE) {
        Ok(s) => s,
        Err(_) => return fallback,
    };
    let yaml = match serde_yaml::from_str::<serde_yaml::Value>(&content) {
        Ok(v) => v,
        Err(_) => return fallback,
    };
    let v = yaml
        .get("benchmark")
        .and_then(|b| b.get(section))
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_f64());
    match v {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => fallback,
    }
}

/// Synthetic-population generator matching the fixture used by
/// `tests/performance_regression_test.rs::generate_multi_zone_population`
/// and `tests/performance_ci_test.rs::generate_synthetic_population`, with
/// the heating / cooling ranges constrained so `heating < cooling`
/// ALWAYS (otherwise `BatchOracle::validate_parameters` rejects the config
/// and `apply_parameters` may produce NaN-bearing temperatures downstream).
/// - `[0]` U-value: 0.1-5.0 W/m²K
/// - `[1]` Heating setpoint: 15-23 °C (strictly < 24)
/// - `[2]` Cooling setpoint: 24-32 °C (strictly ≥ 24)
fn generate_population(size: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut population = Vec::with_capacity(size);
    for _ in 0..size {
        let u_value = rng.random_range(0.1..5.0);
        let heating_setpoint = rng.random_range(15.0..23.0);
        let cooling_setpoint = rng.random_range(24.0..32.0);
        population.push(vec![u_value, heating_setpoint, cooling_setpoint]);
    }
    population
}

/// One timed full-population solve pass: clone the base, apply params,
/// solve 8 760 steps per config (Rayon-parallel across the population).
///
/// HybridThermalModel's manual `Clone` impl resets solver/schedule slots
/// to fresh defaults per clone, so each worker owns an independent solve.
fn timed_population_pass(
    base: &HybridThermalModel,
    surrogates: &SurrogateManager,
    population: &[Vec<f64>],
) -> HybridMetrics {
    let start = Instant::now();
    let _: Vec<f64> = population
        .par_iter()
        .map(|p| {
            let mut m = base.clone();
            m.apply_parameters(p);
            m.solve_timesteps(8760, surrogates, true)
        })
        .collect();
    let elapsed = start.elapsed();

    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = population.len() as f64 / elapsed.as_secs_f64();
    let latency_per_config_ms = elapsed_ms / population.len() as f64;

    HybridMetrics {
        elapsed_ms,
        throughput,
        latency_per_config_ms,
    }
}

/// Construct the measurement harness (base model, surrogate manager,
/// synthetic population) for `run_hybrid_performance_test` and
/// `run_hybrid_performance_best_of_n`.
fn build_harness(
    population_size: usize,
    num_zones: usize,
) -> (HybridThermalModel, SurrogateManager, Vec<Vec<f64>>) {
    let spec = ASHRAE140Case::Case600.spec();
    let base = if num_zones == 1 {
        HybridThermalModel::from_spec(&spec)
    } else {
        HybridThermalModel::new(num_zones, HybridRouting::default())
    };
    let surrogates = SurrogateManager::new().expect("SurrogateManager::new (mock mode)");
    let population = generate_population(population_size);
    (base, surrogates, population)
}

/// Run a HybridThermalModel population solve and measure throughput
/// (single-shot: 1 warmup pass + 1 measured pass). Used by the 10-zone
/// gate; the single-zone gate uses [`run_hybrid_performance_best_of_n`].
///
/// `num_zones == 1` uses `from_spec(ASHRAE140Case::Case600.spec())` (the
/// production-recommended default base); `num_zones > 1` uses
/// `HybridThermalModel::new(num_zones, HybridRouting::default())` (no
/// ASHRAE 140 spec applies for arbitrary zone counts). All other routing
/// parameters come from `HybridRouting::default()` — i.e. surrogate loads,
/// physics everything else. This is the configuration the absolute-perf-gate
/// (#2693) does NOT measure and that production users running
/// `ThermalModelMode::Hybrid` actually exercise.
///
/// The loop body mirrors what `BatchOracle::evaluate_population` does for
/// the pure-physics path: clone the base, apply params, solve 8 760 steps.
fn run_hybrid_performance_test(population_size: usize, num_zones: usize) -> HybridMetrics {
    let (base, surrogates, population) = build_harness(population_size, num_zones);

    // Warm-up pass — populates any lazy per-zone state on the cold path so
    // the measured run reflects steady-state throughput.
    let _ = timed_population_pass(&base, &surrogates, &population);

    // Measured pass.
    timed_population_pass(&base, &surrogates, &population)
}

/// Best-of-N measurement for the single-zone gate (Issue #3894).
///
/// 1 discarded warmup pass + `measured_runs` timed passes, returning the
/// per-pass metrics. The warmup absorbs the one-time cold-start cost
/// (cold page cache for the test binary / model / weather data) that
/// produced the unreproducible ~1.1 cfg/s outlier in the #3892 re-triage;
/// the caller asserting on the best (max) of the measured passes makes
/// transient background load that depresses individual passes unable to
/// flip the gate, while a genuine ≥ 2× regression depresses EVERY pass
/// and still fails.
fn run_hybrid_performance_best_of_n(
    population_size: usize,
    num_zones: usize,
    measured_runs: usize,
) -> Vec<HybridMetrics> {
    let (base, surrogates, population) = build_harness(population_size, num_zones);

    // Warm-up pass — absorbs cold-start; discarded (see doc comment).
    let _ = timed_population_pass(&base, &surrogates, &population);

    let mut metrics = Vec::with_capacity(measured_runs);
    for i in 0..measured_runs {
        let m = timed_population_pass(&base, &surrogates, &population);
        // NOTE: interim per-pass diagnostics deliberately do NOT use the
        // `Hybrid throughput:` label — that exact label is parsed by
        // `.github/workflows/performance_dashboard.yml` (which takes the
        // LAST match per run), so only the final summary line may carry it.
        println!(
            "  measured pass {}/{}: {:.1} configs/sec",
            i + 1,
            measured_runs,
            m.throughput
        );
        metrics.push(m);
    }
    metrics
}

#[allow(dead_code)]
struct HybridMetrics {
    elapsed_ms: f64,
    throughput: f64,
    latency_per_config_ms: f64,
}

/// Single-zone HybridThermalModel throughput regression test (Issue #2922).
///
/// Mirrors `test_performance_regression` but constructs a `HybridThermalModel`
/// via `from_spec(ASHRAE140Case::Case600.spec())` and runs an annual solve
/// (8 760 steps) per config. Enforces the absolute floor from
/// `release_gates.yaml` → `benchmark.hybrid.min_configs_per_sec` (≥ 80 cfg/s
/// on `HybridRouting::default()`, which fires the surrogate-load branch on
/// every step).
///
/// **Measurement statistic (Issue #3894, un-quarantining #3892): best-of-3
/// with a discarded warmup pass — load/cold-start robust.** The #3892
/// re-triage showed this test's DEBUG-mode throughput is load-dependent:
///
/// | Condition | Single-shot throughput | Best-of-3 throughput |
/// |---|---|---|
/// | Warm isolated, 5× back-to-back | 49–51 cfg/s | ~50 cfg/s |
/// | Ambient load (load-avg ~7-12) | ~40 cfg/s | 40+ cfg/s |
/// | All 12 cores saturated | ~26 cfg/s (2.0× slowdown) | ~26-30 cfg/s |
/// | 2026-09-19 cold isolated run (#3892) | ~1.1 cfg/s (~89 s) | absorbed by warmup + max |
///
/// Cold-start (~35× below steady state, one-time page-cache effect) is
/// absorbed by the discarded warmup pass and by taking the max of the 3
/// measured passes; steady-state load can at most halve throughput, which
/// the calibrated debug floor ([`HYBRID_DEBUG_FLOOR`], 26 cfg/s) still
/// admits. A genuine ≥ 2× regression (≤ 25 cfg/s sustained across ALL
/// passes — max included) still fails loudly. Debug-mode isolated results
/// are therefore a reliable signal again and may be classified in
/// "no new failures" diffs. The `--release` Hybrid Perf Gate (#2922)
/// remains the authoritative CI gate.
///
/// Run with:
/// ```
/// cargo test --test all_tests hybrid_perf_regression::test_hybrid_performance_regression --release
/// ```
#[test]
fn test_hybrid_performance_regression() {
    let absolute_floor = hybrid_floor_from_yaml();

    let runs = run_hybrid_performance_best_of_n(SINGLE_ZONE_POP, 1, SINGLE_ZONE_MEASURED_RUNS);
    let best = runs
        .into_iter()
        .max_by(|a, b| a.throughput.total_cmp(&b.throughput))
        .expect("at least one measured pass");

    // The metric line the dashboard's median-of-3 parser greps for. The
    // exact label `Hybrid throughput:` is matched by
    // `.github/workflows/performance_dashboard.yml` → `hybrid-perf-gate`
    // — keep them in sync. Asserted value = best (max) of the measured
    // passes; the dashboard's outer median-of-3 composes with it.
    println!("\nHybrid (1 zone) performance metrics (best of {SINGLE_ZONE_MEASURED_RUNS}):");
    println!("  Population size: {SINGLE_ZONE_POP}");
    println!("  Elapsed: {:.2}ms", best.elapsed_ms);
    println!("  Hybrid throughput: {:.0} configs/sec", best.throughput);
    println!("  Latency per config: {:.3}ms", best.latency_per_config_ms);
    println!(
        "  Absolute floor: {:.0} configs/sec (from {} → benchmark.hybrid.min_configs_per_sec)",
        absolute_floor, RELEASE_GATES_FILE
    );

    // Tarpaulin / debug builds measure far below release throughput, so
    // the floor is relaxed for those build modes (mirrors
    // `test_performance_smoke_test`). The debug arm uses the best-of-3
    // calibration floor from #3892/#3894 (see [`HYBRID_DEBUG_FLOOR`]).
    #[cfg(tarpaulin)]
    let effective_floor = absolute_floor * 0.1;
    #[cfg(not(tarpaulin))]
    let effective_floor = if cfg!(debug_assertions) {
        HYBRID_DEBUG_FLOOR
    } else {
        absolute_floor
    };
    println!("  Effective floor: {effective_floor:.0} configs/sec");

    assert!(
        best.throughput >= effective_floor,
        "HYBRID ABSOLUTE FLOOR BREACH: best-of-{SINGLE_ZONE_MEASURED_RUNS} {:.0} configs/sec \
         < {:.0} (floor from release_gates.yaml → benchmark.hybrid.min_configs_per_sec, \
         HybridThermalModel with HybridRouting::default()). This is a release-gate contract — \
         the hybrid throughput gate is the binding constraint (per issue #2922). The debug \
         floor is the best-of-{SINGLE_ZONE_MEASURED_RUNS} calibration floor from #3894 \
         ({HYBRID_DEBUG_FLOOR:.0} cfg/s, 12-core reference machine): a breach means either a \
         genuine ≥ 2× throughput regression (all measured passes depressed) or a machine \
         slower than the calibration reference. Investigate per-timestep dispatch overhead \
         in Hybrid mode before merging.",
        best.throughput,
        effective_floor,
    );

    println!(
        "✓ Absolute floor OK: best-of-{SINGLE_ZONE_MEASURED_RUNS} {:.0} ≥ {:.0} configs/sec",
        best.throughput, effective_floor
    );
}

/// 10-zone HybridThermalModel throughput regression test (Issue #2922).
///
/// Mirrors `test_multi_zone_performance_regression` but constructs a
/// 10-zone `HybridThermalModel::new(10, HybridRouting::default())` (no
/// ASHRAE 140 spec for arbitrary zone counts; the 1-zone variant uses
/// `from_spec(Case600)`). Enforces the absolute floor from
/// `release_gates.yaml` → `benchmark.hybrid_multi_zone.min_configs_per_sec`
/// (≥ 8 cfg/s). Issue #2772 set the same 10 cfg/s floor on the pure-
/// physics path; #2922 sets a slightly lower floor (8 cfg/s) for Hybrid
/// to reflect the per-timestep dispatch overhead introduced by the
/// surrogate-load branch (default routing fires it on every step).
///
/// Run with:
/// ```
/// cargo test --test hybrid_perf_regression --release test_hybrid_multi_zone_performance_regression
/// ```
#[test]
fn test_hybrid_multi_zone_performance_regression() {
    let absolute_floor = hybrid_multi_zone_floor_from_yaml();

    let metrics = run_hybrid_performance_test(MULTI_ZONE_POP, MULTI_ZONE_GATE_ZONES);

    // The metric line the dashboard's median-of-3 parser greps for. The
    // exact label `Hybrid multi-zone throughput:` is matched by
    // `.github/workflows/performance_dashboard.yml` → `hybrid-perf-gate`
    // — keep them in sync.
    println!("\nHybrid multi-zone ({MULTI_ZONE_GATE_ZONES} zones) performance metrics:");
    println!("  Population size: {MULTI_ZONE_POP}");
    println!("  Elapsed: {:.2}ms", metrics.elapsed_ms);
    println!(
        "  Hybrid multi-zone throughput: {:.0} configs/sec",
        metrics.throughput
    );
    println!(
        "  Latency per config: {:.3}ms",
        metrics.latency_per_config_ms
    );
    println!(
        "  Absolute floor: {:.0} configs/sec (from {} → benchmark.hybrid_multi_zone.min_configs_per_sec)",
        absolute_floor, RELEASE_GATES_FILE
    );

    #[cfg(tarpaulin)]
    let effective_floor = absolute_floor * 0.1;
    #[cfg(not(tarpaulin))]
    let effective_floor = if cfg!(debug_assertions) {
        absolute_floor * 0.1
    } else {
        absolute_floor
    };

    assert!(
        metrics.throughput >= effective_floor,
        "HYBRID MULTI-ZONE ABSOLUTE FLOOR BREACH: {:.0} configs/sec < {:.0} (floor from \
         release_gates.yaml → benchmark.hybrid_multi_zone.min_configs_per_sec, \
         HybridThermalModel 10 zones with HybridRouting::default()). This is a \
         release-gate contract — the hybrid multi-zone throughput gate is the \
         binding constraint (per issue #2922). Investigate per-zone dispatch \
         overhead in Hybrid mode before merging.",
        metrics.throughput,
        effective_floor,
    );

    println!(
        "✓ Absolute floor OK: {:.0} ≥ {:.0} configs/sec",
        metrics.throughput, effective_floor
    );
}

/// Drift guard for [`HYBRID_FLOOR_FALLBACK`].
///
/// Mirrors `test_regression_threshold_matches_yaml` for the single-zone
/// Hybrid floor: if someone changes `release_gates.yaml → benchmark.hybrid.
/// min_configs_per_sec` without updating the fallback constant (or
/// vice-versa), this test fails CI. The YAML is the source of truth.
#[test]
fn test_hybrid_floor_matches_yaml() {
    let yaml_floor = hybrid_floor_from_yaml();
    assert!(
        (yaml_floor - HYBRID_FLOOR_FALLBACK).abs() < 1e-9,
        "HYBRID_FLOOR_FALLBACK ({}) does not match release_gates.yaml \
         benchmark.hybrid.min_configs_per_sec ({}). The YAML is the source \
         of truth — update the constant to match.",
        HYBRID_FLOOR_FALLBACK,
        yaml_floor,
    );
}

/// Drift guard for [`HYBRID_MULTI_ZONE_FLOOR_FALLBACK`].
#[test]
fn test_hybrid_multi_zone_floor_matches_yaml() {
    let yaml_floor = hybrid_multi_zone_floor_from_yaml();
    assert!(
        (yaml_floor - HYBRID_MULTI_ZONE_FLOOR_FALLBACK).abs() < 1e-9,
        "HYBRID_MULTI_ZONE_FLOOR_FALLBACK ({}) does not match release_gates.yaml \
         benchmark.hybrid_multi_zone.min_configs_per_sec ({}). The YAML is the \
         source of truth — update the constant to match.",
        HYBRID_MULTI_ZONE_FLOOR_FALLBACK,
        yaml_floor,
    );
}

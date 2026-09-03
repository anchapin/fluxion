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
//!   (single-zone HybridThermalModel).
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

/// Run a HybridThermalModel population solve and measure throughput.
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
/// HybridThermalModel's manual `Clone` impl resets solver/schedule slots
/// to fresh defaults per clone, so each worker owns an independent solve.
fn run_hybrid_performance_test(population_size: usize, num_zones: usize) -> HybridMetrics {
    let spec = ASHRAE140Case::Case600.spec();
    let base = if num_zones == 1 {
        HybridThermalModel::from_spec(&spec)
    } else {
        HybridThermalModel::new(num_zones, HybridRouting::default())
    };
    let surrogates = SurrogateManager::new().expect("SurrogateManager::new (mock mode)");
    let population = generate_population(population_size);

    // Warm-up run — populates any lazy per-zone state on the cold path so
    // the measured run reflects steady-state throughput.
    let _: Vec<f64> = population
        .par_iter()
        .map(|p| {
            let mut m = base.clone();
            m.apply_parameters(p);
            m.solve_timesteps(8760, &surrogates, true)
        })
        .collect();

    // Measured run.
    let start = Instant::now();
    let _: Vec<f64> = population
        .par_iter()
        .map(|p| {
            let mut m = base.clone();
            m.apply_parameters(p);
            m.solve_timesteps(8760, &surrogates, true)
        })
        .collect();
    let elapsed = start.elapsed();

    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = population_size as f64 / elapsed.as_secs_f64();
    let latency_per_config_ms = elapsed_ms / population_size as f64;

    HybridMetrics {
        elapsed_ms,
        throughput,
        latency_per_config_ms,
    }
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
/// Run with:
/// ```
/// cargo test --test hybrid_perf_regression --release test_hybrid_performance_regression
/// ```
#[test]
fn test_hybrid_performance_regression() {
    let absolute_floor = hybrid_floor_from_yaml();

    let metrics = run_hybrid_performance_test(SINGLE_ZONE_POP, 1);

    // The metric line the dashboard's median-of-3 parser greps for. The
    // exact label `Hybrid throughput:` is matched by
    // `.github/workflows/performance_dashboard.yml` → `hybrid-perf-gate`
    // — keep them in sync.
    println!("\nHybrid (1 zone) performance metrics:");
    println!("  Population size: {SINGLE_ZONE_POP}");
    println!("  Elapsed: {:.2}ms", metrics.elapsed_ms);
    println!("  Hybrid throughput: {:.0} configs/sec", metrics.throughput);
    println!(
        "  Latency per config: {:.3}ms",
        metrics.latency_per_config_ms
    );
    println!(
        "  Absolute floor: {:.0} configs/sec (from {} → benchmark.hybrid.min_configs_per_sec)",
        absolute_floor, RELEASE_GATES_FILE
    );

    // Tarpaulin / debug builds measure far below release throughput, so
    // the floor is relaxed for those build modes (mirrors
    // `test_performance_smoke_test`).
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
        "HYBRID ABSOLUTE FLOOR BREACH: {:.0} configs/sec < {:.0} (floor from \
         release_gates.yaml → benchmark.hybrid.min_configs_per_sec, HybridThermalModel \
         with HybridRouting::default()). This is a release-gate contract — the hybrid \
         throughput gate is the binding constraint (per issue #2922). Investigate \
         per-timestep dispatch overhead in Hybrid mode before merging.",
        metrics.throughput,
        effective_floor,
    );

    println!(
        "✓ Absolute floor OK: {:.0} ≥ {:.0} configs/sec",
        metrics.throughput, effective_floor
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

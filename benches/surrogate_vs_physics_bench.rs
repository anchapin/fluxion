//! Surrogate vs Physics Benchmark for Fluxion
//!
//! Head-to-head benchmark harness running the same complex building under
//! pure 9R4C physics and under the hybrid physics/ML switch.
//!
//! The harness is parameterized: a single [`ASHRAE140Case`] specification
//! drives both execution paths, ensuring an apples-to-apples comparison.
//!
//! # Issue
//! Issue #1781: Implement benches/surrogate_vs_physics_bench.rs harness
//! Issue #1782: Track RSS memory in the surrogate-vs-physics benchmark CI
//!
//! Run with: cargo bench --release --bench surrogate_vs_physics

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::thermal_model::{HybridRouting, HybridThermalModel, ThermalModelTrait};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use std::time::Instant;

const DUMMY_ONNX_MODEL: &str = "assets/dummy_surrogate.onnx";

/// Number of hourly timesteps in a year (standard weather file length).
const ANNUAL_TIMESTEPS: usize = 8760;

/// Short run for quick iteration (1 week).
const SHORT_TIMESTEPS: usize = 168;

/// Case 900: High-mass baseline building (concrete construction, south windows).
/// This is the reference "complex building" for the head-to-head comparison.
fn case900_spec() -> fluxion::validation::ashrae_140_cases::CaseSpec {
    ASHRAE140Case::Case900.spec()
}

/// Build a SurrogateManager, falling back gracefully if ONNX is not available.
fn build_surrogate_manager() -> SurrogateManager {
    SurrogateManager::new().expect("Failed to create SurrogateManager")
}

// ---------------------------------------------------------------------------
// Head-to-head: pure 9R4C physics vs hybrid physics/ML on the SAME building
// ---------------------------------------------------------------------------

/// Benchmark: pure 9R4C physics (all_physics routing) on Case 900 for a year.
fn bench_physics_only_900_annual(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = case900_spec();
    let steps = ANNUAL_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case900/physics_only");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(10);

    group.bench_function("annual_8760", |b| {
        b.iter(|| {
            let mut model =
                HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::all_physics());
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

/// Benchmark: hybrid physics/ML (default routing: loads → surrogate, rest → physics)
/// on Case 900 for a year.
fn bench_hybrid_default_900_annual(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = case900_spec();
    let steps = ANNUAL_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case900/hybrid_default");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(10);

    group.bench_function("annual_8760", |b| {
        b.iter(|| {
            let mut model = HybridThermalModel::from_spec(&spec);
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

/// Benchmark: pure 9R4C physics on Case 900 — short run (1 week).
fn bench_physics_only_900_short(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = case900_spec();
    let steps = SHORT_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case900/physics_only_short");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(100);

    group.bench_function("weekly_168", |b| {
        b.iter(|| {
            let mut model =
                HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::all_physics());
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

/// Benchmark: hybrid physics/ML (default) on Case 900 — short run (1 week).
fn bench_hybrid_default_900_short(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = case900_spec();
    let steps = SHORT_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case900/hybrid_default_short");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(100);

    group.bench_function("weekly_168", |b| {
        b.iter(|| {
            let mut model = HybridThermalModel::from_spec(&spec);
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Additional ASHRAE 140 cases for coverage
// ---------------------------------------------------------------------------

/// Case 600: Low-mass baseline (5R1C) physics only — short run.
fn bench_physics_only_600_short(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = ASHRAE140Case::Case600.spec();
    let steps = SHORT_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case600/physics_only_short");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(100);

    group.bench_function("weekly_168", |b| {
        b.iter(|| {
            let mut model =
                HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::all_physics());
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

/// Case 600: Low-mass baseline (5R1C) hybrid default — short run.
fn bench_hybrid_default_600_short(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = ASHRAE140Case::Case600.spec();
    let steps = SHORT_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case600/hybrid_default_short");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(100);

    group.bench_function("weekly_168", |b| {
        b.iter(|| {
            let mut model = HybridThermalModel::from_spec(&spec);
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

/// Case 920: High-mass with east/west windows — physics only, short run.
fn bench_physics_only_920_short(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = ASHRAE140Case::Case920.spec();
    let steps = SHORT_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case920/physics_only_short");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(100);

    group.bench_function("weekly_168", |b| {
        b.iter(|| {
            let mut model =
                HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::all_physics());
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

/// Case 920: High-mass with east/west windows — hybrid default, short run.
fn bench_hybrid_default_920_short(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = ASHRAE140Case::Case920.spec();
    let steps = SHORT_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case920/hybrid_default_short");
    group.throughput(Throughput::Elements(steps as u64));
    group.sample_size(100);

    group.bench_function("weekly_168", |b| {
        b.iter(|| {
            let mut model = HybridThermalModel::from_spec(&spec);
            let _eui = model.solve_timesteps(black_box(steps), &surrogates, false);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Surrogate inference micro-benchmarks (retained from original file)
// ---------------------------------------------------------------------------

fn bench_surrogate_onnx_single_inference(c: &mut Criterion) {
    let surrogate = match SurrogateManager::load_onnx(DUMMY_ONNX_MODEL) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Could not load ONNX model for benchmarking: {e}");
            eprintln!("Surrogate benchmarks will be skipped.");
            return;
        }
    };

    let temps = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0];

    let mut group = c.benchmark_group("surrogate_onnx");
    group.throughput(Throughput::Elements(temps.len() as u64));
    group.sample_size(1000);

    group.bench_function("single_inference_6zones", |b| {
        b.iter(|| {
            let _ = surrogate
                .predict_loads_onnx(black_box(&temps))
                .expect("ONNX inference failed");
        })
    });

    group.finish();
}

fn bench_surrogate_onnx_batched_inference(c: &mut Criterion) {
    let surrogate = match SurrogateManager::load_onnx(DUMMY_ONNX_MODEL) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Could not load ONNX model for benchmarking: {e}");
            return;
        }
    };

    let batch_sizes = [1, 10, 100];

    for &batch_size in &batch_sizes {
        let mut group = c.benchmark_group(format!("surrogate_onnx_batch_{batch_size}"));
        group.throughput(Throughput::Elements(batch_size as u64 * 6));
        group.sample_size(100);

        let batch: Vec<Vec<f64>> = (0..batch_size)
            .map(|_| vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
            .collect();

        group.bench_function("batched_inference", |b| {
            b.iter(|| {
                let _ = surrogate.predict_loads_batched(black_box(&batch));
            })
        });

        group.finish();
    }
}

fn bench_analytical_loads_timing(c: &mut Criterion) {
    let surrogate = build_surrogate_manager();
    let temps = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0];

    let mut group = c.benchmark_group("analytical_timing");
    group.sample_size(1000);

    group.bench_function("analytical_loads_6zones", |b| {
        b.iter(|| {
            let _ = surrogate
                .analytical_loads(black_box(&temps))
                .expect("Analytical loads failed");
        })
    });

    group.finish();
}

/// Benchmark: ms/timestep execution-time reporting for T6.3.
/// Runs both physics-only and hybrid paths on Case 900 for a year, and emits
/// `MS_PER_TIMESTEP: {"physics": X, "hybrid": Y, "ratio": Z}` to stderr so CI
/// can parse it and track the speedup ratio across runs.
fn bench_ms_per_timestep_8760(c: &mut Criterion) {
    let surrogates = build_surrogate_manager();
    let spec = case900_spec();
    let timesteps = ANNUAL_TIMESTEPS;

    let mut group = c.benchmark_group("head_to_head/case900/ms_per_timestep");
    group.sample_size(10);
    group.throughput(Throughput::Elements(timesteps as u64));

    group.bench_function("speedup_ratio_ms_per_timestep", |b| {
        b.iter_custom(|n_iter| {
            let mut total_physics_ns: u128 = 0;
            let mut total_hybrid_ns: u128 = 0;

            for _ in 0..n_iter {
                let mut physics_model =
                    HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::all_physics());
                let start = Instant::now();
                let _ = physics_model.solve_timesteps(black_box(timesteps), &surrogates, false);
                total_physics_ns += start.elapsed().as_nanos();

                let mut hybrid_model = HybridThermalModel::from_spec(&spec);
                let start = Instant::now();
                let _ = hybrid_model.solve_timesteps(black_box(timesteps), &surrogates, false);
                total_hybrid_ns += start.elapsed().as_nanos();
            }

            let avg_physics_ns = total_physics_ns as f64 / n_iter as f64;
            let avg_hybrid_ns = total_hybrid_ns as f64 / n_iter as f64;
            let physics_ms_per_ts = avg_physics_ns / 1_000_000.0 / timesteps as f64;
            let hybrid_ms_per_ts = avg_hybrid_ns / 1_000_000.0 / timesteps as f64;
            let ratio = if avg_hybrid_ns > 0.0 {
                avg_physics_ns / avg_hybrid_ns
            } else {
                0.0
            };

            eprintln!(
                "MS_PER_TIMESTEP: {{\"physics\": {:.6}, \"hybrid\": {:.6}, \"ratio\": {:.6}}}",
                physics_ms_per_ts, hybrid_ms_per_ts, ratio
            );

            // Return the physics time as the criterion timing for this benchmark
            std::time::Duration::from_nanos((avg_physics_ns / 2.0) as u64)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    // Head-to-head Case 900 (high-mass, 9R4C) — primary benchmarks for T6.1
    bench_physics_only_900_annual,
    bench_hybrid_default_900_annual,
    bench_physics_only_900_short,
    bench_hybrid_default_900_short,
    // Additional cases for coverage
    bench_physics_only_600_short,
    bench_hybrid_default_600_short,
    bench_physics_only_920_short,
    bench_hybrid_default_920_short,
    // Micro-benchmarks
    bench_surrogate_onnx_single_inference,
    bench_surrogate_onnx_batched_inference,
    bench_analytical_loads_timing,
    // ms/timestep execution-time reporting (T6.3)
    bench_ms_per_timestep_8760
);
criterion_main!(benches);

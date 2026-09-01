//! Criterion bench for Issue #1439 — `BatchOracle` CPU surrogate
//! coordinator path (`BatchOrchestrator` + `par_chunks`).
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench rayon_chunks_bench
//! ```
//!
//! Each benchmark:
//!
//! 1. Generates a synthetic population of N configs (deterministic RNG seed).
//! 2. Calls `BatchOracle::evaluate_population(population, use_surrogates=true)`.
//! 3. Reports `Throughput::Elements(N)` so configs/sec is the headline number.
//!
//! Targets from Issue #1439:
//!
//! - `population_10000 CPU surrogate` ≥1 500 configs/sec (+50% over the
//!   previous ≈1 000 configs/sec reported on the `scope(N)` path).
//! - CI regression gate (`batch-oracle-bench` workflow) fails if
//!   `Throughput` row drops >5% vs the criterion baseline.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::orchestrator::{BatchOrchestrator, RayonChunksOrchestrator};
use fluxion::BatchOracle;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generate a synthetic population of building configurations for benchmarking.
///
/// Parameters are within valid bounds:
/// - U-value: 0.1-5.0 W/m²K
/// - Heating setpoint: 15-25°C
/// - Cooling setpoint: 22-32°C
fn generate_synthetic_population(size: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42); // Deterministic seed
    let mut population = Vec::with_capacity(size);
    for _ in 0..size {
        let u_value = rng.random_range(0.1..5.0);
        let heating_setpoint = rng.random_range(15.0..25.0);
        let cooling_setpoint = rng.random_range(22.0..32.0);
        population.push(vec![u_value, heating_setpoint, cooling_setpoint]);
    }
    population
}

/// Benchmark CPU surrogate path with `population_10000` (Issue #1439 AC line 1).
///
/// This is the canonical benchmark for the `BatchOracle::evaluate_population`
/// CPU surrogate branch. Pre-change it reported ≈1 000 configs/sec; the AC
/// line calls for ≥1 500 configs/sec (+50%) after the `par_chunks`
/// replacement.
fn bench_cpu_surrogate_population_10000(c: &mut Criterion) {
    let base_model = ThermalModel::<VectorField>::new(10);
    let oracle = BatchOracle::from_model(base_model);
    let population = generate_synthetic_population(10_000);

    let mut group = c.benchmark_group("batch_oracle_chunks");
    group.throughput(Throughput::Elements(10_000));
    group.sample_size(10);

    group.bench_function("population_10000_cpu_surrogate", |b| {
        b.iter(|| {
            let pop = population.clone();
            let _ = oracle.evaluate_population(pop, true);
        })
    });
    group.finish();
}

/// Sweep population sizes to characterize scaling on the new orchestrator.
fn bench_cpu_surrogate_scaling(c: &mut Criterion) {
    let base_model = ThermalModel::<VectorField>::new(10);
    let oracle = BatchOracle::from_model(base_model);

    let population_sizes = [100, 1_000, 10_000];
    for &size in &population_sizes {
        let population = generate_synthetic_population(size);

        let mut group = c.benchmark_group("batch_oracle_chunks_scaling");
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);

        let name = format!("population_{}", size);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let pop = population.clone();
                let _ = oracle.evaluate_population(pop, true);
            })
        });
        group.finish();
    }
}

/// Analytical path baseline (sanity check that the par_chunks refactor did
/// not regress the non-surrogate path). Same workload as the surrogate
/// bench above but `use_surrogates=false`.
fn bench_analytical_population_10000(c: &mut Criterion) {
    let base_model = ThermalModel::<VectorField>::new(10);
    let oracle = BatchOracle::from_model(base_model);
    let population = generate_synthetic_population(10_000);

    let mut group = c.benchmark_group("batch_oracle_analytical_chunks");
    group.throughput(Throughput::Elements(10_000));
    group.sample_size(10);

    group.bench_function("population_10000", |b| {
        b.iter(|| {
            let pop = population.clone();
            let _ = oracle.evaluate_population(pop, false);
        })
    });
    group.finish();
}

/// Build orchestrator-ready configs `(idx, ThermalModel)` with valid
/// parameters applied, matching `BatchOracle::evaluate_population`'s
/// `valid_configs` shape.
fn build_orchestrator_configs(size: usize) -> Vec<(usize, ThermalModel<VectorField>)> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..size)
        .map(|i| {
            let mut m = ThermalModel::<VectorField>::new(10);
            let params = vec![
                rng.random_range(0.1..5.0),
                rng.random_range(15.0..25.0),
                rng.random_range(22.0..32.0),
            ];
            m.apply_parameters(&params);
            (i, m)
        })
        .collect()
}

/// Issue #2520 — per-timestep ONNX batching overhead vs the unbatched
/// `par_chunks` path.
///
/// Both benches drive an identical mock surrogate (`SurrogateManager::new`,
/// `model_loaded == false`) so the *only* difference is the coordination
/// cost: the unbatched path runs each `par_chunks` worker to completion with
/// zero cross-thread hand-offs, while the batched path pays a
/// `crossbeam::channel` rendezvous per timestep (one batched inference call
/// instead of `N` per-config calls).
///
/// Acceptance criterion (#2520): `batched_mock_<N>` must be < 2×
/// `unbatched_mock_<N>`. With a real ONNX model loaded the batched path is
/// dramatically faster (1024× fewer inference calls), but a real model is
/// not staged in the bench environment, so we measure the rendezvous
/// overhead in isolation against the mock fallback.
fn bench_cpu_surrogate_batched_vs_unbatched(c: &mut Criterion) {
    let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
    let size = 1000;
    let orchestrator = RayonChunksOrchestrator::for_population(size);

    let mut group = c.benchmark_group("cpu_surrogate_batching");
    group.throughput(Throughput::Elements(size as u64));
    group.sample_size(10);

    group.bench_function("unbatched_mock_1000", |b| {
        b.iter_batched(
            || build_orchestrator_configs(size),
            |configs| {
                let _ = orchestrator.run_cpu_surrogate(configs, &surrogates);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("batched_mock_1000", |b| {
        b.iter_batched(
            || build_orchestrator_configs(size),
            |configs| {
                let _ = orchestrator.run_cpu_surrogate_batched(configs, &surrogates);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    batch_oracle_chunks,
    bench_cpu_surrogate_population_10000,
    bench_cpu_surrogate_scaling,
    bench_analytical_population_10000,
    bench_cpu_surrogate_batched_vs_unbatched
);
criterion_main!(batch_oracle_chunks);

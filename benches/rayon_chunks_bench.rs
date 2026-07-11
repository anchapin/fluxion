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
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
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
        let u_value = rng.gen_range(0.1..5.0);
        let heating_setpoint = rng.gen_range(15.0..25.0);
        let cooling_setpoint = rng.gen_range(22.0..32.0);
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

criterion_group!(
    batch_oracle_chunks,
    bench_cpu_surrogate_population_10000,
    bench_cpu_surrogate_scaling,
    bench_analytical_population_10000
);
criterion_main!(batch_oracle_chunks);

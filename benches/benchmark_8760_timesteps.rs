//! 8760-Timestep Benchmark for Fluxion
//!
//! This benchmark measures performance with realistic 8760-timestep (1 year) workloads.
//! Run with: cargo bench --release --bench benchmark_8760_timesteps

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Number of timesteps in a year (8760 hours)
const TIMESTEPS_PER_YEAR: usize = 8760;

/// Generate a synthetic population with valid building parameters
fn generate_population(size: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut population = Vec::with_capacity(size);

    for _ in 0..size {
        let u_value: f64 = rng.gen_range(0.1..5.0);
        let heating: f64 = rng.gen_range(15.0..25.0);
        let cooling_max: f64 = heating + 1.0;
        let cooling: f64 = rng.gen_range(22.0_f64..32.0_f64).max(cooling_max);
        population.push(vec![u_value, heating, cooling]);
    }

    population
}

/// Benchmark single-config 8760-timestep simulation
fn bench_single_config_8760(c: &mut Criterion) {
    let mut model = ThermalModel::<VectorField>::new(1);
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Warm-up
    model.solve_timesteps(100, &surrogates, false, None, None, None);

    let mut group = c.benchmark_group("single_config_8760");
    group.bench_function("analytical", |b| {
        b.iter(|| {
            let mut model = ThermalModel::<VectorField>::new(1);
            model.solve_timesteps(
                black_box(TIMESTEPS_PER_YEAR),
                black_box(&surrogates),
                black_box(false),
                black_box(None),
                black_box(None::<&[Box<dyn fluxion::sim::equipment::Equipment>]>),
                black_box(None),
            );
        })
    });
    group.finish();
}

/// Benchmark BatchOracle with 8760 timesteps for different population sizes
fn bench_batch_oracle_8760(c: &mut Criterion) {
    let population_sizes = [100, 1000, 10000];

    for &size in &population_sizes {
        let population = generate_population(size);

        // Analytical mode
        let mut group = c.benchmark_group(&format!("batch_oracle_8760_analytical_{}", size));
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);

        let base_model = ThermalModel::<VectorField>::new(1);
        let oracle = BatchOracle::from_model(base_model);

        let name = format!("population_{}", size);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let pop = population.clone();
                black_box(oracle.evaluate_population(pop, false));
            })
        });
        group.finish();
    }
}

/// Benchmark multi-zone 8760-timestep simulation
fn bench_multizone_8760(c: &mut Criterion) {
    let zone_counts = [1, 5, 10];
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    for &zones in &zone_counts {
        let mut group = c.benchmark_group(&format!("multizone_8760_{}zones", zones));
        group.sample_size(10);

        let name = format!("{}zones", zones);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let mut model = ThermalModel::<VectorField>::new(zones);
                model.solve_timesteps(
                    black_box(TIMESTEPS_PER_YEAR),
                    black_box(&surrogates),
                    black_box(false),
                    black_box(None),
                    black_box(None::<&[Box<dyn fluxion::sim::equipment::Equipment>]>),
                    black_box(None),
                );
            })
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_single_config_8760,
    bench_batch_oracle_8760,
    bench_multizone_8760
);
criterion_main!(benches);

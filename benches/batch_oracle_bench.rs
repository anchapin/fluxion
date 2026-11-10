use criterion::{criterion_group, criterion_main, Criterion};
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

/// Issue #4696 (spike): ceiling measurement for an "aggregate-only output
/// mode" for BatchOracle population runs.
///
/// The spike premise is that a population run wastes a material fraction of
/// its wall time materialising/writing *full* output (hourly series) that
/// sweep callers never read, and that an opt-in aggregate-only mode could
/// reclaim it.
///
/// These two measurement-only benchmarks quantify the ceiling on the
/// `BatchOracle` population path:
///
/// * `batch_oracle_api_boundary_output/{size}` — the cost of the output that
///   the BatchOracle API actually materialises today: one `f64` EUI per
///   config, written into one `Vec<f64>`. This is the *entire* output of
///   `evaluate_population`, so its cost is an upper bound on any saving
///   available from an aggregate-only mode on this path.
/// * `batch_oracle_full_hourly_output/{size}` — the cost a *full* hourly
///   output (8 760 timesteps x `size` configs) would cost if it were
///   materialised. This is the cost an aggregate-only mode would avoid — but
///   only on a path that actually materialises it. The BatchOracle
///   population path does not, so this number is a counterfactual, not a
///   saving.
fn bench_output_materialisation_ceiling(c: &mut Criterion) {
    for &size in &[100usize, 1000] {
        // (a) What the BatchOracle API materialises today: one f64 per config.
        let name = format!("batch_oracle_api_boundary_output/{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let mut out: Vec<f64> = Vec::with_capacity(size);
                for i in 0..size {
                    out.push(i as f64 * 0.5);
                }
                std::hint::black_box(out);
            })
        });

        // (b) Counterfactual: a full hourly series per config (8 760 steps).
        let name = format!("batch_oracle_full_hourly_output/{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let mut out: Vec<f64> = Vec::with_capacity(size * 8760);
                for i in 0..(size * 8760) {
                    out.push(i as f64 * 0.5);
                }
                std::hint::black_box(out);
            })
        });

        // (c) Per-config *state* materialisation (the base-model clone the
        // population path performs once per config). Not "output", but it is
        // the dominant per-config allocation on the population path and is
        // reported so the ceiling numbers are not misattributed.
        let base_model = ThermalModel::<VectorField>::new(10);
        let name = format!("batch_oracle_per_config_model_clone/{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let mut clones = Vec::with_capacity(size);
                for _ in 0..size {
                    clones.push(base_model.clone());
                }
                std::hint::black_box(clones);
            })
        });
    }
}

/// Generate a synthetic population of building configurations for benchmarking.
///
/// Parameters are within valid bounds:
/// - U-value: 0.1-5.0 W/m²K
/// - Heating setpoint: 15-25°C
/// - Cooling setpoint: 22-32°C
fn generate_synthetic_population(size: usize) -> Vec<Vec<f64>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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

fn bench_batch_oracle_throughput(c: &mut Criterion) {
    let base_model = ThermalModel::<VectorField>::new(10);
    let oracle = BatchOracle::from_model(base_model).unwrap();

    // Test population sizes relevant to performance requirements
    // Target: >1000 configs/sec. We test 100-1000 configs to measure scaling.
    let population_sizes = [100, 200, 500, 1000];

    for &size in &population_sizes {
        let population = generate_synthetic_population(size);

        let name = format!("batch_oracle_analytical/{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let pop = population.clone();
                let _results = oracle.evaluate_population(pop, false);
            })
        });

        let name = format!("batch_oracle_surrogates/{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let pop = population.clone();
                let _results = oracle.evaluate_population(pop, true);
            })
        });
    }
}

criterion_group!(
    batch_oracle_benches,
    bench_batch_oracle_throughput,
    bench_output_materialisation_ceiling,
);
criterion_main!(batch_oracle_benches);

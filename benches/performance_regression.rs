use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fluxion::ai::surrogate::SurrogateManager;
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

/// Benchmark thermal model solve performance for different zone counts.
///
/// Tests solve_timesteps for 1-year simulation (8760 steps) with varying
/// zone configurations. Uses mock surrogates to measure pure physics performance.
fn bench_thermal_model_solve(c: &mut Criterion) {
    let zone_counts = [1, 10, 50, 100];

    for &num_zones in &zone_counts {
        let mut model = ThermalModel::<VectorField>::new(num_zones);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        // Warm up
        model.solve_timesteps(100, &surrogates, false, None, None, None);

        let name = format!("thermal_model_solve_{}zones", num_zones);
        let mut group = c.benchmark_group("thermal_model");
        group.sample_size(10); // Reduce sample size for faster testing
        group.bench_function(&name, |b| {
            b.iter(|| {
                let mut model = ThermalModel::<VectorField>::new(num_zones);
                model.solve_timesteps(
                    black_box(8760),
                    black_box(&surrogates),
                    black_box(false),
                    None,
                    None,
                    None,
                );
            })
        });
        group.finish();
    }
}

/// Benchmark BatchOracle throughput for different population sizes.
///
/// Measures configs/second throughput for both analytical and surrogate modes.
/// Uses throughput measurement to quantify performance in terms of elements processed.
fn bench_batch_oracle_throughput(c: &mut Criterion) {
    let base_model = ThermalModel::<VectorField>::new(10);
    let oracle = BatchOracle::from_model(base_model);

    let population_sizes = [100, 1_000, 10_000];

    for &size in &population_sizes {
        let population = generate_synthetic_population(size);

        // Analytical mode
        let mut group = c.benchmark_group("batch_oracle_analytical");
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10); // Reduce sample size for faster testing

        let name = format!("population_{}", size);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let pop = population.clone();
                let _ = black_box(oracle.evaluate_population(pop, false));
            })
        });
        group.finish();

        // Surrogate mode
        let mut group = c.benchmark_group("batch_oracle_surrogates");
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10); // Reduce sample size for faster testing

        let name = format!("population_{}", size);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let pop = population.clone();
                let _ = black_box(oracle.evaluate_population(pop, true));
            })
        });
        group.finish();
    }
}

/// Benchmark VectorField operations for different vector sizes.
///
/// Tests key CTA operations (add, subtract, multiply, divide) to ensure
/// they are not regression points. Uses different vector sizes to test scaling.
fn bench_vectorfield_operations(c: &mut Criterion) {
    let sizes = [10, 100, 1_000];

    for &size in &sizes {
        let v1 = VectorField::new((0..size).map(|i| i as f64).collect());
        let v2 = VectorField::new((0..size).map(|i| (i as f64) * 0.5).collect());

        // Addition
        let name = format!("vectorfield_add_{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let _ = black_box(v1.clone()) + black_box(v2.clone());
            })
        });

        // Subtraction
        let name = format!("vectorfield_sub_{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let _ = black_box(v1.clone()) - black_box(v2.clone());
            })
        });

        // Multiplication
        let name = format!("vectorfield_mul_{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let _ = black_box(v1.clone()) * black_box(2.0);
            })
        });

        // Division
        let name = format!("vectorfield_div_{}", size);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let _ = black_box(v1.clone()) / black_box(2.0);
            })
        });
    }
}

criterion_group!(
    benches,
    bench_thermal_model_solve,
    bench_batch_oracle_throughput,
    bench_vectorfield_operations
);
criterion_main!(benches);

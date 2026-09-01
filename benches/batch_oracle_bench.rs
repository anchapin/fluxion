use criterion::{criterion_group, criterion_main, Criterion};
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

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
    let oracle = BatchOracle::from_model(base_model);

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

criterion_group!(batch_oracle_benches, bench_batch_oracle_throughput);
criterion_main!(batch_oracle_benches);

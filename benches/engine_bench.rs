use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

fn bench_solve_timesteps(c: &mut Criterion) {
    let mut model = ThermalModel::<VectorField>::new(10);
    // Use mock surrogates (CPU mode) to focus on physics engine performance
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Warm up
    model.solve_timesteps(100, &surrogates, false, None, None, None);

    c.bench_function("solve_timesteps_1year_10zones", |b| {
        b.iter(|| {
            // 8760 steps = 1 year
            // We clone to reset state? No, solve_timesteps continues from current state.
            // It's fine to continue simulation.
            model.solve_timesteps(8760, &surrogates, false, None, None, None);
        })
    });
}

/// Benchmark 5R1C single configuration performance
fn bench_5r1c_single_config(c: &mut Criterion) {
    let mut model = ThermalModel::<VectorField>::new(1);
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Warm up
    model.solve_timesteps(100, &surrogates, false, None, None, None);

    c.bench_function("5r1c_single_config_1year", |b| {
        b.iter(|| {
            // Clone to reset state for each iteration
            let mut model = model.clone();
            // Use 8760 timesteps (1 year) to match Phase 9 baseline
            model.solve_timesteps(8760, &surrogates, false, None, None, None);
        })
    });
}

/// Benchmark 6R2C single configuration performance
fn bench_6r2c_single_config(c: &mut Criterion) {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.configure_6r2c_model(0.75, 100.0, None);
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Warm up
    model.solve_timesteps(100, &surrogates, false, None, None, None);

    c.bench_function("6r2c_single_config_1year", |b| {
        b.iter(|| {
            // Clone to reset state for each iteration
            let mut model = model.clone();
            // Use 8760 timesteps (1 year) to match Phase 9 baseline
            model.solve_timesteps(8760, &surrogates, false, None, None, None);
        })
    });
}

/// Quick benchmark 5R1C single configuration (100 timesteps)
fn bench_5r1c_single_config_quick(c: &mut Criterion) {
    c.bench_function("5r1c_single_config_100steps", |b| {
        b.iter(|| {
            let mut model = ThermalModel::<VectorField>::new(1);
            let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");
            // Use 100 timesteps for quick benchmarking
            model.solve_timesteps(100, &surrogates, false, None, None, None);
        })
    });
}

/// Quick benchmark 6R2C single configuration (100 timesteps)
fn bench_6r2c_single_config_quick(c: &mut Criterion) {
    c.bench_function("6r2c_single_config_100steps", |b| {
        b.iter(|| {
            let mut model = ThermalModel::<VectorField>::new(1);
            model.configure_6r2c_model(0.75, 100.0, None);
            let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");
            // Use 100 timesteps for quick benchmarking
            model.solve_timesteps(100, &surrogates, false, None, None, None);
        })
    });
}

/// Benchmark 5R1C population throughput
fn bench_5r1c_throughput(c: &mut Criterion) {
    let population_size = 100; // Reduced from 1000 for faster benchmarking
    let population: Vec<Vec<f64>> = (0..population_size)
        .map(|_| vec![1.5, 21.0]) // Window U-value=1.5, HVAC setpoint=21.0
        .collect();

    c.bench_with_input(
        BenchmarkId::new("5r1c_throughput", population_size),
        &population,
        |b, population| {
            b.iter(|| {
                let mut total_energy = 0.0;
                for params in population.iter() {
                    let mut model = ThermalModel::<VectorField>::new(1);
                    model.apply_parameters(params);
                    let surrogates =
                        SurrogateManager::new().expect("Failed to create SurrogateManager");
                    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
                    total_energy += energy;
                }
                // Prevent compiler from optimizing away the computation
                black_box(total_energy);
            })
        },
    );
}

/// Benchmark 6R2C population throughput
fn bench_6r2c_throughput(c: &mut Criterion) {
    let population_size = 100; // Reduced from 1000 for faster benchmarking
    let population: Vec<Vec<f64>> = (0..population_size)
        .map(|_| vec![1.5, 21.0]) // Window U-value=1.5, HVAC setpoint=21.0
        .collect();

    c.bench_with_input(
        BenchmarkId::new("6r2c_throughput", population_size),
        &population,
        |b, population| {
            b.iter(|| {
                let mut total_energy = 0.0;
                for params in population.iter() {
                    let mut model = ThermalModel::<VectorField>::new(1);
                    model.configure_6r2c_model(0.75, 100.0, None);
                    model.apply_parameters(params);
                    let surrogates =
                        SurrogateManager::new().expect("Failed to create SurrogateManager");
                    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
                    total_energy += energy;
                }
                // Prevent compiler from optimizing away the computation
                black_box(total_energy);
            })
        },
    );
}

criterion_group!(
    benches,
    bench_solve_timesteps,
    bench_5r1c_single_config,
    bench_6r2c_single_config,
    bench_5r1c_single_config_quick,
    bench_6r2c_single_config_quick,
    bench_5r1c_throughput,
    bench_6r2c_throughput
);
criterion_main!(benches);

//! Benchmark comparing CTA VectorField performance vs raw Vec<f64>
//!
//! This benchmark compares the performance of the CTA VectorField implementation
//! against raw Vec<f64> operations to verify <10% overhead requirement.

use criterion::{criterion_group, criterion_main, Criterion};
use fluxion::physics::cta::{ContinuousTensor, VectorField};

/// Raw Vec<f64> implementation for comparison
fn raw_vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn raw_vec_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn raw_vec_map(a: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    a.iter().map(|&x| f(x)).collect()
}

fn raw_vec_reduce(a: &[f64], init: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    a.iter().fold(init, |acc, &x| f(acc, x))
}

fn raw_vec_zip_with(a: &[f64], b: &[f64], f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
}

/// Benchmark: VectorField add vs raw Vec add
fn bench_vector_add(c: &mut Criterion) {
    let size = 10_000;
    let v1 = VectorField::new((0..size).map(|i| i as f64).collect());
    let v2 = VectorField::new((0..size).map(|i| i as f64 * 0.5).collect());

    let a_raw: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let b_raw: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();

    let mut group = c.benchmark_group("vector_add");

    group.bench_function("VectorField", |b| {
        b.iter(|| {
            let _ = v1.clone() + v2.clone();
        })
    });

    group.bench_function("raw_vec", |b| {
        b.iter(|| {
            raw_vec_add(&a_raw, &b_raw);
        })
    });

    group.finish();
}

/// Benchmark: VectorField multiply vs raw Vec multiply
fn bench_vector_mul(c: &mut Criterion) {
    let size = 10_000;
    let v1 = VectorField::new((0..size).map(|i| i as f64).collect());
    let v2 = VectorField::new((0..size).map(|i| i as f64 * 0.5).collect());

    let a_raw: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let b_raw: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();

    let mut group = c.benchmark_group("vector_mul");

    group.bench_function("VectorField", |b| {
        b.iter(|| {
            let _ = v1.clone() * v2.clone();
        })
    });

    group.bench_function("raw_vec", |b| {
        b.iter(|| {
            raw_vec_mul(&a_raw, &b_raw);
        })
    });

    group.finish();
}

/// Benchmark: VectorField map vs raw Vec map
fn bench_vector_map(c: &mut Criterion) {
    let size = 10_000;
    let v = VectorField::new((0..size).map(|i| i as f64).collect());
    let raw: Vec<f64> = (0..size).map(|i| i as f64).collect();

    let mut group = c.benchmark_group("vector_map");

    group.bench_function("VectorField", |b| {
        b.iter(|| {
            let _ = v.map(|x| x * 1.001);
        })
    });

    group.bench_function("raw_vec", |b| {
        b.iter(|| {
            raw_vec_map(&raw, |x| x * 1.001);
        })
    });

    group.finish();
}

/// Benchmark: VectorField reduce vs raw Vec reduce
fn bench_vector_reduce(c: &mut Criterion) {
    let size = 10_000;
    let v = VectorField::new((0..size).map(|i| i as f64).collect());
    let raw: Vec<f64> = (0..size).map(|i| i as f64).collect();

    let mut group = c.benchmark_group("vector_reduce");

    group.bench_function("VectorField", |b| {
        b.iter(|| {
            let _ = v.reduce(0.0, |acc, x| acc + x);
        })
    });

    group.bench_function("raw_vec", |b| {
        b.iter(|| {
            raw_vec_reduce(&raw, 0.0, |acc, x| acc + x);
        })
    });

    group.finish();
}

/// Benchmark: VectorField zip_with vs raw Vec zip_with
fn bench_vector_zip_with(c: &mut Criterion) {
    let size = 10_000;
    let v1 = VectorField::new((0..size).map(|i| i as f64).collect());
    let v2 = VectorField::new((0..size).map(|i| i as f64 * 0.5).collect());

    let a_raw: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let b_raw: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();

    let mut group = c.benchmark_group("vector_zip_with");

    group.bench_function("VectorField", |b| {
        b.iter(|| {
            let _ = v1.zip_with(&v2, |x, y| x + y);
        })
    });

    group.bench_function("raw_vec", |b| {
        b.iter(|| {
            raw_vec_zip_with(&a_raw, &b_raw, |x, y| x + y);
        })
    });

    group.finish();
}

/// Benchmark: ThermalModel operations with VectorField
fn bench_thermal_model_ops(c: &mut Criterion) {
    use fluxion::sim::engine::ThermalModel;
    use fluxion::ai::surrogate::SurrogateManager;

    let mut model = ThermalModel::<VectorField>::new(10);
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Warm up
    model.solve_timesteps(100, &surrogates, false);

    let mut group = c.benchmark_group("thermal_model");

    group.bench_function("solve_timesteps_1year", |b| {
        b.iter(|| {
            model.solve_timesteps(8760, &surrogates, false);
        })
    });

    group.finish();
}

criterion_group!(
    cta_perf_benches,
    bench_vector_add,
    bench_vector_mul,
    bench_vector_map,
    bench_vector_reduce,
    bench_vector_zip_with,
    bench_thermal_model_ops
);
criterion_main!(cta_perf_benches);

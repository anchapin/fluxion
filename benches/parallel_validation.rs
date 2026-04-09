//! Parallel validation benchmarks for Fluxion
//!
//! Run with: cargo bench --bench parallel_validation

use criterion::{criterion_group, criterion_main, Criterion};
use fluxion::performance::benchmarking::*;
use fluxion::performance::parallel::validation::*;

/// Benchmark parallel validation with 100 cases
fn bench_parallel_validation_100(c: &mut Criterion) {
    let cases: Vec<HighMassCase> = (0..100)
        .map(|i| HighMassCase {
            case_id: format!("L{:03}", i),
            reference_loads: vec![100.0 + i as f64; 8760],
            simulation_results: vec![105.0 + i as f64; 8760],
            tolerance: 0.15,
        })
        .collect();

    c.bench_function("parallel validation 100 cases", |b| {
        b.iter(|| run_parallel_validation(cases.clone()))
    });
}

/// Benchmark parallel validation with 1000 cases
fn bench_parallel_validation_1000(c: &mut Criterion) {
    let cases: Vec<HighMassCase> = (0..1000)
        .map(|i| HighMassCase {
            case_id: format!("L{:03}", i),
            reference_loads: vec![100.0 + i as f64; 8760],
            simulation_results: vec![105.0 + i as f64; 8760],
            tolerance: 0.15,
        })
        .collect();

    c.bench_function("parallel validation 1000 cases", |b| {
        b.iter(|| run_parallel_validation(cases.clone()))
    });
}

/// Benchmark single timestep calculation
fn bench_single_timestep(c: &mut Criterion) {
    c.bench_function("single timestep", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                sum += (i as f64 * 0.1).sin();
            }
            sum
        })
    });
}

/// Benchmark validation with timing
fn bench_validation_with_timing(c: &mut Criterion) {
    let cases: Vec<HighMassCase> = (0..50)
        .map(|i| HighMassCase {
            case_id: format!("L{:03}", i),
            reference_loads: vec![100.0 + i as f64; 8760],
            simulation_results: vec![105.0 + i as f64; 8760],
            tolerance: 0.15,
        })
        .collect();

    c.bench_function("validation with timing (50 cases)", |b| {
        b.iter(|| validate_with_timing(cases.clone()))
    });
}

/// Benchmark chunked validation
fn bench_chunked_validation(c: &mut Criterion) {
    let cases: Vec<HighMassCase> = (0..100)
        .map(|i| HighMassCase {
            case_id: format!("L{:03}", i),
            reference_loads: vec![100.0 + i as f64; 8760],
            simulation_results: vec![105.0 + i as f64; 8760],
            tolerance: 0.15,
        })
        .collect();

    c.bench_function("chunked validation (10 chunks)", |b| {
        b.iter(|| run_parallel_validation_chunked(cases.clone(), 10))
    });
}

/// Benchmark throughput calculation
fn bench_throughput_calculation(c: &mut Criterion) {
    c.bench_function("throughput calc", |b| {
        b.iter(|| calculate_throughput(10000, 1.0))
    });
}

/// Benchmark performance metrics creation
fn bench_performance_metrics(c: &mut Criterion) {
    c.bench_function("benchmark metrics", |b| {
        b.iter(|| BenchmarkMetrics::from_run(1000, 0.05))
    });
}

/// Benchmark timestep measurement
fn bench_measure_timestep(c: &mut Criterion) {
    c.bench_function("measure timestep", |b| {
        b.iter(|| measure_timestep(|| 1 + 2, 1000))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(1));
    targets =
        bench_parallel_validation_100,
        bench_parallel_validation_1000,
        bench_single_timestep,
        bench_validation_with_timing,
        bench_chunked_validation,
        bench_throughput_calculation,
        bench_performance_metrics,
        bench_measure_timestep,
}

criterion_main!(benches);

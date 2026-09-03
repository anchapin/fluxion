//! Parallel Harness Benchmark Suite (Issue #2034)
//!
//! Measures throughput (buildings/sec), compares sequential vs parallel,
//! and reports speedup ratio using the criterion crate.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};

use fluxion_city::{BuildingGroup, UrbanRadiationSystem, UrbanStepDispatcher};

fn make_radiation() -> UrbanRadiationSystem {
    UrbanRadiationSystem::new(800.0, 120.0, 0.2, 0.85, 0.1, 2.0)
}

fn make_buildings(n: usize) -> Vec<BuildingGroup> {
    (0..n)
        .map(|i| {
            BuildingGroup::new(i as u32)
                .with_area(100.0 + (i as f64) * 10.0)
                .with_u_values(0.5, 0.3, 2.0)
        })
        .collect()
}

fn step_sequential(
    buildings: &mut [BuildingGroup],
    dt: &Duration,
    radiation: &UrbanRadiationSystem,
    outdoor_temp: f64,
) {
    for building in buildings.iter_mut() {
        building.step(dt, radiation, outdoor_temp);
    }
}

fn step_parallel(
    dispatcher: &mut UrbanStepDispatcher,
    dt: &Duration,
    radiation: &UrbanRadiationSystem,
    outdoor_temp: f64,
) {
    dispatcher.step_all(*dt, radiation, outdoor_temp);
}

fn run_sequential_benchmark(c: &mut Criterion, group_name: &str, building_counts: &[usize]) {
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let radiation = make_radiation();
    let dt = Duration::from_secs(3600);
    let outdoor_temp = 30.0;

    for &n in building_counts {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut buildings = make_buildings(n);
            b.iter(|| {
                step_sequential(
                    std::hint::black_box(&mut buildings),
                    std::hint::black_box(&dt),
                    std::hint::black_box(&radiation),
                    std::hint::black_box(outdoor_temp),
                );
            });
        });
    }

    group.finish();
}

fn run_parallel_benchmark(c: &mut Criterion, group_name: &str, building_counts: &[usize]) {
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let radiation = make_radiation();
    let dt = Duration::from_secs(3600);
    let outdoor_temp = 30.0;

    for &n in building_counts {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let buildings = make_buildings(n);
            let mut dispatcher = UrbanStepDispatcher::with_buildings(buildings);
            b.iter(|| {
                step_parallel(
                    std::hint::black_box(&mut dispatcher),
                    std::hint::black_box(&dt),
                    std::hint::black_box(&radiation),
                    std::hint::black_box(outdoor_temp),
                );
            });
        });
    }

    group.finish();
}

fn run_throughput_benchmark(c: &mut Criterion, building_counts: &[usize]) {
    let mut group = c.benchmark_group("throughput_buildings_per_sec");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let radiation = make_radiation();
    let dt = Duration::from_secs(3600);
    let outdoor_temp = 30.0;

    for &n in building_counts {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let buildings = make_buildings(n);
            let mut dispatcher = UrbanStepDispatcher::with_buildings(buildings);
            b.iter_custom(|iters| {
                let mut total_ms = 0u128;
                for _ in 0..iters {
                    let start = Instant::now();
                    step_parallel(
                        std::hint::black_box(&mut dispatcher),
                        std::hint::black_box(&dt),
                        std::hint::black_box(&radiation),
                        std::hint::black_box(outdoor_temp),
                    );
                    total_ms += start.elapsed().as_millis();
                }
                let avg_ms = total_ms as f64 / iters as f64;
                Duration::from_millis(avg_ms as u64)
            });
        });
    }

    group.finish();
}

fn run_speedup_benchmark(c: &mut Criterion, building_counts: &[usize]) {
    let mut group = c.benchmark_group("speedup_ratio");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let radiation = make_radiation();
    let dt = Duration::from_secs(3600);
    let outdoor_temp = 30.0;

    for &n in building_counts {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let buildings_par = make_buildings(n);
            let mut dispatcher = UrbanStepDispatcher::with_buildings(buildings_par);

            b.iter(|| {
                let mut buildings = make_buildings(n);
                let start_seq = Instant::now();
                step_sequential(
                    std::hint::black_box(&mut buildings),
                    std::hint::black_box(&dt),
                    std::hint::black_box(&radiation),
                    std::hint::black_box(outdoor_temp),
                );
                let time_seq = start_seq.elapsed().as_millis() as f64;

                let start_par = Instant::now();
                step_parallel(
                    std::hint::black_box(&mut dispatcher),
                    std::hint::black_box(&dt),
                    std::hint::black_box(&radiation),
                    std::hint::black_box(outdoor_temp),
                );
                let time_par = start_par.elapsed().as_millis() as f64;

                let speedup = time_seq / time_par;
                std::hint::black_box(speedup)
            });
        });
    }

    group.finish();
}

pub fn city_parallel_benchmark(c: &mut Criterion) {
    let building_counts = vec![10, 50, 100, 200, 500, 1000];

    run_sequential_benchmark(c, "sequential", &building_counts);
    run_parallel_benchmark(c, "parallel_rayon", &building_counts);
    run_throughput_benchmark(c, &building_counts);
    run_speedup_benchmark(c, &building_counts);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = city_parallel_benchmark
}
criterion_main!(benches);

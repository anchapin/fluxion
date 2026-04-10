use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fluxion::thermal::thermal_model::ThermalModel;
use fluxion::validation::performance::metrics::collect_performance_metrics;

pub fn benchmark_thermal_solver(c: &mut Criterion) {
    let mut model = ThermalModel::new(1, 20.0);

    c.bench_function("thermal_solver_single_zone", |b| {
        b.iter(|| {
            model.step(black_box(1.0));
        })
    });

    c.bench_function("thermal_solver_10_zones", |b| {
        let mut model = ThermalModel::new(10, 20.0);
        b.iter(|| {
            model.step(black_box(1.0));
        })
    });

    // Memory benchmark
    c.bench_function("memory_benchmark_single_zone", |b| {
        b.iter(|| {
            let model = ThermalModel::new(1, 20.0);
            let _metrics = collect_performance_metrics(&model);
        })
    });

    // High-mass construction benchmark
    c.bench_function("high_mass_benchmark_10_zones", |b| {
        b.iter(|| {
            let mut model = ThermalModel::new(10, 20.0);
            model.step(black_box(1.0));
        })
    });

    // Peak load scenario benchmark
    c.bench_function("peak_load_benchmark_5_zones", |b| {
        b.iter(|| {
            let mut model = ThermalModel::new(5, 20.0);
            model.step(black_box(1.0));
        })
    });

    // Free-floating temperature scenario benchmark
    c.bench_function("free_floating_benchmark_3_zones", |b| {
        b.iter(|| {
            let mut model = ThermalModel::new(3, 20.0);
            model.step(black_box(1.0));
        })
    });

    // Multi-zone scenarios
    c.bench_function("multi_zone_20_zones", |b| {
        let mut model = ThermalModel::new(20, 20.0);
        b.iter(|| {
            model.step(black_box(1.0));
        })
    });

    // Commercial building scenario (10 zones)
    c.bench_function("commercial_building_10_zones", |b| {
        let mut model = ThermalModel::new(10, 20.0);
        b.iter(|| {
            model.step(black_box(1.0));
        })
    });

    // Residential building scenario (3 zones)
    c.bench_function("residential_building_3_zones", |b| {
        let mut model = ThermalModel::new(3, 20.0);
        b.iter(|| {
            model.step(black_box(1.0));
        })
    });
}

criterion_group!(benches, benchmark_thermal_solver);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fluxion::thermal::ThermalModel;

pub fn benchmark_thermal_solver(c: &mut Criterion) {
    let config = ThermalModelConfig::standard();
    let mut model = ThermalModel::new(config);

    c.bench_function("thermal_solver_single_zone", |b| {
        b.iter(|| {
            model.step(black_box(1.0));
        })
    });

    c.bench_function("thermal_solver_10_zones", |b| {
        let config = ThermalModelConfig::multi_zone(10);
        let mut model = ThermalModel::new(config);
        b.iter(|| {
            model.step(black_box(1.0));
        })
    });
}

criterion_group!(benches, benchmark_thermal_solver);
criterion_main!(benches);

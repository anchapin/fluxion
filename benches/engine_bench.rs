use criterion::{criterion_group, criterion_main, Criterion};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

fn bench_solve_timesteps(c: &mut Criterion) {
    let mut model = ThermalModel::<VectorField>::new(10);
    // Use mock surrogates (CPU mode) to focus on physics engine performance
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Warm up
    model.solve_timesteps(100, &surrogates, false);

    c.bench_function("solve_timesteps_1year_10zones", |b| {
        b.iter(|| {
            // 8760 steps = 1 year
            // We clone to reset state? No, solve_timesteps continues from current state.
            // It's fine to continue simulation.
            model.solve_timesteps(8760, &surrogates, false);
        })
    });
}

criterion_group!(benches, bench_solve_timesteps);
criterion_main!(benches);

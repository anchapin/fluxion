use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fluxion::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};

fn high_mass_layers() -> Vec<CTFMaterial> {
    vec![
        CTFMaterial::new("Gypsum Board", 0.013, 0.16, 800.0, 1090.0),
        CTFMaterial::new("Concrete", 0.200, 1.95, 2300.0, 880.0),
        CTFMaterial::new("Insulation", 0.100, 0.04, 50.0, 840.0),
        CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
    ]
}

fn low_mass_layers() -> Vec<CTFMaterial> {
    vec![
        CTFMaterial::new("Steel Siding", 0.001, 45.0, 7800.0, 500.0),
        CTFMaterial::new("Insulation", 0.100, 0.04, 50.0, 840.0),
        CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
    ]
}

fn bench_ctf_coefficient_evaluation_high_mass(c: &mut Criterion) {
    let layers = high_mass_layers();
    let timestep = 3600.0;

    c.bench_function("ctf_coefficient_eval_high_mass_200mm_concrete", |b| {
        b.iter(|| {
            let calc = CTFCalculator::with_defaults(black_box(&layers), black_box(timestep));
            let _coeffs = calc.compute_coefficients();
        })
    });
}

fn bench_ctf_coefficient_evaluation_low_mass(c: &mut Criterion) {
    let layers = low_mass_layers();
    let timestep = 3600.0;

    c.bench_function("ctf_coefficient_eval_low_mass_steel_siding", |b| {
        b.iter(|| {
            let calc = CTFCalculator::with_defaults(black_box(&layers), black_box(timestep));
            let _coeffs = calc.compute_coefficients();
        })
    });
}

fn bench_ctf_state_update_high_mass(c: &mut Criterion) {
    let layers = high_mass_layers();
    let timestep = 3600.0;
    let calc = CTFCalculator::with_defaults(&layers, timestep);
    let coeffs = calc.compute_coefficients();

    let num_coeffs = coeffs.num_coeffs;
    let t_exterior_history: Vec<f64> = vec![25.0; num_coeffs];
    let t_interior_history: Vec<f64> = vec![20.0; num_coeffs.saturating_sub(1)];
    let flux_history: Vec<f64> = vec![10.0; num_coeffs.saturating_sub(1)];

    c.bench_function("ctf_state_update_high_mass_single_timestep", |b| {
        b.iter(|| {
            let _flux = coeffs.calculate_interior_flux(
                black_box(20.0),
                black_box(&t_exterior_history),
                black_box(&t_interior_history),
                black_box(&flux_history),
            );
        })
    });
}

fn bench_ctf_state_update_low_mass(c: &mut Criterion) {
    let layers = low_mass_layers();
    let timestep = 3600.0;
    let calc = CTFCalculator::with_defaults(&layers, timestep);
    let coeffs = calc.compute_coefficients();

    let num_coeffs = coeffs.num_coeffs;
    let t_exterior_history: Vec<f64> = vec![25.0; num_coeffs];
    let t_interior_history: Vec<f64> = vec![20.0; num_coeffs.saturating_sub(1)];
    let flux_history: Vec<f64> = vec![10.0; num_coeffs.saturating_sub(1)];

    c.bench_function("ctf_state_update_low_mass_single_timestep", |b| {
        b.iter(|| {
            let _flux = coeffs.calculate_interior_flux(
                black_box(20.0),
                black_box(&t_exterior_history),
                black_box(&t_interior_history),
                black_box(&flux_history),
            );
        })
    });
}

fn bench_ctf_flux_history_iteration(c: &mut Criterion) {
    let layers = high_mass_layers();
    let timestep = 3600.0;
    let calc = CTFCalculator::with_defaults(&layers, timestep);
    let coeffs = calc.compute_coefficients();

    let num_coeffs = coeffs.num_coeffs;
    let t_exterior_history: Vec<f64> = (0..num_coeffs).map(|i| 25.0 + i as f64 * 0.1).collect();
    let t_interior_history: Vec<f64> = (0..num_coeffs.saturating_sub(1))
        .map(|i| 20.0 + i as f64 * 0.05)
        .collect();
    let flux_history: Vec<f64> = (0..num_coeffs.saturating_sub(1))
        .map(|i| 10.0 + i as f64 * 0.2)
        .collect();

    c.bench_function("ctf_flux_history_iteration_50_terms", |b| {
        b.iter(|| {
            let _flux = coeffs.calculate_interior_flux(
                black_box(20.0),
                black_box(&t_exterior_history),
                black_box(&t_interior_history),
                black_box(&flux_history),
            );
        })
    });
}

fn bench_ctf_u_value_computation(c: &mut Criterion) {
    let layers = high_mass_layers();
    let timestep = 3600.0;
    let calc = CTFCalculator::with_defaults(&layers, timestep);
    let coeffs = calc.compute_coefficients();

    c.bench_function("ctf_u_value_from_coefficients", |b| {
        b.iter(|| {
            let _u = coeffs.u_value();
        })
    });
}

fn bench_ctf_multiple_timestep_sequence(c: &mut Criterion) {
    let layers = high_mass_layers();
    let timestep = 3600.0;
    let calc = CTFCalculator::with_defaults(&layers, timestep);
    let coeffs = calc.compute_coefficients();

    let num_coeffs = coeffs.num_coeffs;
    let mut t_ext_buf = vec![25.0; num_coeffs];
    let mut t_int_buf = vec![20.0; num_coeffs.saturating_sub(1)];
    let mut flux_buf = vec![10.0; num_coeffs.saturating_sub(1)];

    let num_steps = 24;

    c.bench_function("ctf_24hour_timestep_sequence", |b| {
        b.iter(|| {
            for step in 0..num_steps {
                let t_ext = 25.0 + (step as f64 * 2.0).sin() * 10.0;
                let t_int = 20.0 + (step as f64 * 2.0).cos() * 3.0;

                t_ext_buf.pop();
                t_ext_buf.insert(0, t_ext);

                t_int_buf.pop();
                t_int_buf.insert(0, t_int);

                let flux = coeffs.calculate_interior_flux(
                    black_box(t_int),
                    black_box(&t_ext_buf),
                    black_box(&t_int_buf),
                    black_box(&flux_buf),
                );

                flux_buf.pop();
                flux_buf.insert(0, flux);
            }
        })
    });
}

fn bench_ctf_various_timesteps(c: &mut Criterion) {
    let layers = high_mass_layers();
    let timesteps = [300.0, 600.0, 1800.0, 3600.0, 7200.0];

    let mut group = c.benchmark_group("ctf_coefficient_eval_timestep_scaling");
    for &dt in &timesteps {
        group.bench_with_input(
            BenchmarkId::from_parameter(dt as u32),
            &dt,
            |b, &timestep| {
                b.iter(|| {
                    let calc =
                        CTFCalculator::with_defaults(black_box(&layers), black_box(timestep));
                    let _coeffs = calc.compute_coefficients();
                })
            },
        );
    }
    group.finish();
}

fn bench_ctf_low_mass_various_timesteps(c: &mut Criterion) {
    let layers = low_mass_layers();
    let timesteps = [300.0, 600.0, 1800.0, 3600.0, 7200.0];

    let mut group = c.benchmark_group("ctf_coefficient_eval_lowmass_timestep_scaling");
    for &dt in &timesteps {
        group.bench_with_input(
            BenchmarkId::from_parameter(dt as u32),
            &dt,
            |b, &timestep| {
                b.iter(|| {
                    let calc =
                        CTFCalculator::with_defaults(black_box(&layers), black_box(timestep));
                    let _coeffs = calc.compute_coefficients();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    ctf_solver_benches,
    bench_ctf_coefficient_evaluation_high_mass,
    bench_ctf_coefficient_evaluation_low_mass,
    bench_ctf_state_update_high_mass,
    bench_ctf_state_update_low_mass,
    bench_ctf_flux_history_iteration,
    bench_ctf_u_value_computation,
    bench_ctf_multiple_timestep_sequence,
    bench_ctf_various_timesteps,
    bench_ctf_low_mass_various_timesteps,
);
criterion_main!(ctf_solver_benches);

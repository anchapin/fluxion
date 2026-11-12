//! ASHRAE 140 §8 (ground-coupled slab-on-grade) — transient analytical
//! verification for the 2-D ground-conduction domain model.
//!
//! Issue #4774 (transient half; the periodic half is a later split). The
//! solver under test is [`fluxion::physics::ground_conduction`], a
//! standalone 2-D (x–z) ground-conduction domain model. It is *not* wired
//! into the Section 7 floor boundary or the main simulation loop: §8
//! verifies the ground domain standalone first.
//!
//! # Case — step surface temperature on the 1-D limit
//!
//! The configuration is the one-dimensional limit of the domain (slab
//! footprint spans the whole width, lateral boundaries adiabatic) with a
//! **homogeneous** material: slab and soil share conductivity
//! `k = 1.9 W/(m·K)` and volumetric heat capacity
//! `rho c = 2.0e6 J/(m³·K)`, so the slab/soil interface is invisible to
//! the temperature field. The domain starts uniform at `T0 = 10 °C`;
//! at `t = 0` the surface is stepped to `Ts = 30 °C` (Dirichlet) and
//! held there. The deep boundary (4 m) also sits at 10 °C, but at the
//! comparison times the thermal penetration depth
//! `sqrt(alpha t) <= 0.29 m` is more than an order of magnitude smaller
//! than 4 m, so the finite depth does not contaminate the comparison
//! and the semi-infinite closed form applies:
//!
//! ```text
//! T(z, t) = Ts + (T0 - Ts) * erf( z / (2 sqrt(alpha t)) ),
//! alpha = k / (rho c) = 9.5e-7 m²/s
//! ```
//!
//! (Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., §2.4 —
//! the step-surface-temperature solution of the semi-infinite solid.)
//! `erf` is evaluated with the Abramowitz & Stegun 7.1.26 rational
//! approximation (absolute error <= 1.5e-7), far below the comparison
//! tolerance.
//!
//! # Tolerance
//!
//! The comparison tolerance is **0.75 K**. The error budget is dominated
//! by the cell-average vs point-value difference over the 0.05 m surface
//! cells and the first-order backward-Euler time error at `dt = 3600 s`;
//! both shrink under refinement, and the observed maximum deviation at
//! the asserted depths/times is printed by the test for the record. The
//! signal at the shallowest asserted depth is a ~16 K rise, so 0.75 K
//! is a discriminating bound, not a rubber stamp: dropping the heat
//! capacity term (a steady solve in disguise) would miss by >10 K.
//!
//! A second test asserts the transient solution settles onto the
//! discrete steady-state solution of the same configuration: with a
//! large time step the backward-Euler state's fixed point *is* the
//! steady solve, so after enough large steps the two must agree to
//! solver tolerance.

use fluxion::physics::ground_conduction::{
    GridSegment, GroundDomainConfig, GroundSolver, SurfaceCondition, TopCondition, TopSegment,
};

/// Surface temperature after the step [°C].
const T_SURFACE: f64 = 30.0;
/// Initial (and deep-ground) temperature [°C].
const T_INITIAL: f64 = 10.0;
/// Homogeneous conductivity [W/(m·K)].
const K_W_MK: f64 = 1.9;
/// Homogeneous volumetric heat capacity [J/(m³·K)].
const RHO_C: f64 = 2_000_000.0;

/// Error function, Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Closed-form semi-infinite step response at depth `z_m`, time `t_s`.
fn analytical_step_temperature(z_m: f64, t_s: f64) -> f64 {
    let alpha = K_W_MK / RHO_C;
    let eta = z_m / (2.0 * (alpha * t_s).sqrt());
    T_SURFACE + (T_INITIAL - T_SURFACE) * erf(eta)
}

/// Homogeneous 1-D configuration: 4 m wide, 4 m deep, fine uniform
/// cells over the top metre (the penetration region), geometric below.
fn homogeneous_config() -> GroundDomainConfig {
    GroundDomainConfig {
        domain_length_m: 4.0,
        domain_depth_m: 4.0,
        slab_x0_m: 0.0,
        slab_x1_m: 4.0,
        slab_thickness_m: 0.2,
        slab_conductivity_w_mk: K_W_MK,
        ground_conductivity_w_mk: K_W_MK,
        slab_volumetric_heat_capacity_j_m3k: RHO_C,
        ground_volumetric_heat_capacity_j_m3k: RHO_C,
        lateral_segments: vec![GridSegment::Uniform {
            start_m: 0.0,
            end_m: 4.0,
            cells: 4,
        }],
        vertical_segments: vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: 1.0,
                cells: 20,
            },
            GridSegment::Geometric {
                start_m: 1.0,
                end_m: 4.0,
                first_cell_m: 0.05,
                growth: 1.15,
                max_cell_m: 0.4,
            },
        ],
        top_segments: vec![TopSegment {
            x_start_m: 0.0,
            x_end_m: 4.0,
            condition: TopCondition::Dirichlet(T_SURFACE),
        }],
        deep_ground: SurfaceCondition::Dirichlet(T_INITIAL),
        lateral_boundary: SurfaceCondition::Adiabatic,
        max_iterations: 50_000,
        tolerance: 1.0e-11,
    }
}

#[test]
fn transient_step_matches_erf_closed_form() {
    let solver = GroundSolver::new(homogeneous_config()).expect("valid config");
    let mut state = solver
        .initial_transient_state(T_INITIAL)
        .expect("initial state");

    // One day of hourly steps.
    let dt_s = 3600.0;
    let steps = 24;
    for _ in 0..steps {
        solver.step_transient(&mut state, dt_s).expect("step");
    }
    let t_s = dt_s * steps as f64;
    assert!((state.time_s - t_s).abs() < 1.0e-9);

    // Compare every cell centre in the penetration region (z <= 0.6 m)
    // at the centre column against the closed form.
    let i = state.grid.x.len() / 2;
    let mut max_deviation = 0.0_f64;
    let mut compared = 0usize;
    for (j, &z) in state.grid.z.centers.iter().enumerate() {
        if z > 0.6 {
            break;
        }
        let expected = analytical_step_temperature(z, t_s);
        let actual = state.cell_temperature(i, j);
        let deviation = (actual - expected).abs();
        max_deviation = max_deviation.max(deviation);
        compared += 1;
        assert!(
            deviation <= 0.75,
            "z={z:.3} m: solver {actual:.4} °C vs closed form {expected:.4} °C \
             (deviation {deviation:.4} K > 0.75 K)"
        );
    }
    assert!(compared >= 10, "expected at least 10 compared depths");
    println!(
        "transient step: {compared} depths compared at t={t_s:.0}s, \
         max deviation {max_deviation:.4} K"
    );
}

#[test]
fn transient_state_settles_onto_steady_solution() {
    let config = homogeneous_config();
    let solver = GroundSolver::new(config).expect("valid config");
    let steady = solver.solve().expect("steady solve");
    let mut state = solver
        .initial_transient_state(T_INITIAL)
        .expect("initial state");

    // Large steps: the backward-Euler fixed point is the steady discrete
    // solution, so 60 steps of 10 days must land on it.
    for _ in 0..60 {
        solver.step_transient(&mut state, 864_000.0).expect("step");
    }

    let mut max_diff = 0.0_f64;
    for j in 0..state.grid.z.len() {
        for i in 0..state.grid.x.len() {
            let diff = (state.cell_temperature(i, j) - steady.cell_temperature(i, j)).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(
        max_diff <= 0.05,
        "transient state has not settled onto the steady solution: max diff {max_diff:.6} K"
    );
}

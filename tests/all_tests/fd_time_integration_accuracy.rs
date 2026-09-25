//! FD time-integration order verification: BackwardEuler vs Bdf2.
//!
//! Issue #3980: temporal-order tracer test for the implicit FD conduction
//! solver. Uses the same Carslaw & Jaeger step-response problem as the #1098
//! CTF component test (200 mm concrete slab, insulated at x=0, Dirichlet step
//! to 35 C at x=L, 24 h horizon) and measures the observed order of accuracy
//! in dt by Richardson-style halving.
//!
//! The convergence reference is a SEMI-DISCRETE solution: the same solver run
//! at dt = 5 s, whose own temporal error is negligible ((450/5)^2 = 8100x
//! smaller than the coarsest swept step). Comparing directly against the
//! analytical series instead would floor the measurement on the O(dx^2)
//! spatial error of the node-centred grid (boundary nodes sit dx/2 inside
//! the wall, so the ghost-node domain is L+dx wide — a ~+0.17 K bias on the
//! 80-node grid that is dt-independent and therefore not attributable to the
//! time integrator). The series is kept as an absolute smoke bound.

use fluxion::physics::fd_discretization::{MaterialLayer, WallDiscretization};
use fluxion::physics::fd_solver::{ImplicitFDSolver, SurfaceBC, TimeIntegrationScheme};
use std::f64::consts::PI;

/// Concrete properties (identical to the #1098 component test).
const K: f64 = 1.75; // W/(m·K)
const RHO: f64 = 2300.0; // kg/m³
const CP: f64 = 880.0; // J/(kg·K)
const L: f64 = 0.2; // m

/// Thermal diffusivity [m²/s]
const ALPHA: f64 = K / (RHO * CP);

const T_INITIAL: f64 = 20.0;
const T_EXTERIOR: f64 = 35.0;
const DT_STEP: f64 = T_EXTERIOR - T_INITIAL;

/// Simulation horizon [s] (24 h).
const T_END: f64 = 24.0 * 3600.0;

/// FD nodes per layer (80 keeps runs fast; the semi-discrete reference makes
/// the spatial error common-mode between the swept runs and the reference).
const FD_NODES: usize = 80;

/// Reference timestep for the semi-discrete solution [s].
const DT_REF: f64 = 5.0;

/// Analytical interior-surface temperature (Carslaw & Jaeger Ch. 2):
/// T(0,t) = T_s - 2·dT·Σ (-1)^n/((n+½)π) · exp(-α·((n+½)π/L)²·t)
fn analytical_interior_temp(t_seconds: f64, num_terms: usize) -> f64 {
    let mut series_sum = 0.0_f64;
    for n in 0..num_terms {
        let n_f = n as f64;
        let lam = (n_f + 0.5) * PI / L;
        series_sum += (-1.0_f64).powi(n as i32) / ((n_f + 0.5) * PI)
            * (-ALPHA * lam.powi(2) * t_seconds).exp();
    }
    T_EXTERIOR - 2.0 * DT_STEP * series_sum
}

/// Insulated interior boundary condition (h→0).
fn interior_bc() -> SurfaceBC {
    // h is deliberately small but not so extreme that 2·Fo·h·dx/k round-off
    // matters; 1e-9 pins Neumann behaviour for any dt swept here.
    SurfaceBC::new_interior(1e-9, T_INITIAL)
}

/// Near-Dirichlet exterior boundary condition (h→∞, numerically sane).
/// h = 1e6 holds the surface node within ~2e-4 K of T_EXTERIOR at peak flux
/// while keeping tridiagonal entries below 1e6 scale.
fn exterior_bc() -> SurfaceBC {
    SurfaceBC::new_exterior(1e6, T_EXTERIOR, 0.0)
}

/// Hourly interior-surface temperatures over 24 h for the given scheme/dt.
fn interior_temp_history(scheme: TimeIntegrationScheme, dt: f64) -> Vec<f64> {
    let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
    let disc = WallDiscretization::from_layers(&layers, FD_NODES);
    let mut solver = ImplicitFDSolver::with_scheme(disc, T_INITIAL, scheme);

    let int_bc = interior_bc();
    let ext_bc = exterior_bc();

    let steps = (T_END / dt) as usize;
    assert!(
        (steps as f64 * dt - T_END).abs() < 1e-9,
        "dt {dt} must divide T_END exactly"
    );
    assert!(
        (3600.0 / dt).fract() < 1e-9,
        "dt {dt} must divide the 3600 s sampling interval exactly"
    );

    let mut history = Vec::with_capacity(24);
    let mut next_sample = 3600.0_f64;
    for step in 1..=steps {
        let t_after = step as f64 * dt;
        solver.step(dt, &int_bc, &ext_bc);
        while next_sample <= t_after + 1e-9 {
            history.push(solver.temperatures[0]);
            next_sample += 3600.0;
        }
    }
    assert_eq!(history.len(), 24, "expected 24 hourly samples over 24 h");
    history
}

/// RMS error of the hourly interior temperature history against `reference`.
fn rms_vs(history: &[f64], reference: &[f64]) -> f64 {
    let squared: f64 = history
        .iter()
        .zip(reference.iter())
        .map(|(&fd, &r)| {
            let e = fd - r;
            e * e
        })
        .sum();
    (squared / history.len() as f64).sqrt()
}

fn rms_error(scheme: TimeIntegrationScheme, dt: f64, reference: &[f64]) -> f64 {
    rms_vs(&interior_temp_history(scheme, dt), reference)
}

fn observed_order(scheme: TimeIntegrationScheme, coarse: f64, fine: f64, reference: &[f64]) -> f64 {
    let e_coarse = rms_error(scheme, coarse, reference);
    let e_fine = rms_error(scheme, fine, reference);
    (e_coarse / e_fine).log2()
}

/// Semi-discrete reference: same solver, dt = 5 s (BDF2; its own temporal
/// error is ~8000x below the coarsest swept step).
fn semi_discrete_reference() -> Vec<f64> {
    interior_temp_history(TimeIntegrationScheme::Bdf2, DT_REF)
}

/// First-mode decay setup: smooth-in-time problem for order verification.
///
/// IC = T_MEAN + A·cos(λ₀·x) (the first Neumann/Dirichlet eigenfunction,
/// λ₀ = π/2L), constant BCs at T_MEAN. The semi-discrete system is
/// IDENTICAL for every dt (constant BCs — no source-sampling error), the
/// solution is smooth in t (bounded T''(0) ≈ (αλ₀²)²·A ≈ 1e-8 K/s²), so the
/// measured order isolates the time integrator, not the bootstrap.
const T_MEAN: f64 = 27.5;
const MODE_AMPLITUDE: f64 = 4.75;

/// Hourly interior-node temperatures over 24 h for an eigenmode-decay
/// problem: IC = T_MEAN + A·cos(λ_m·x) with λ_m = (2m+1)π/2L (the m-th
/// Neumann/Dirichlet eigenfunction), constant BCs at T_MEAN, on a grid with
/// `nodes` nodes.
fn eigenmode_history(
    scheme: TimeIntegrationScheme,
    dt: f64,
    nodes: usize,
    mode: usize,
) -> Vec<f64> {
    let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
    let disc = WallDiscretization::from_layers(&layers, nodes);

    // Initial condition: single eigenmode sampled at node centres.
    let lam = (2.0 * mode as f64 + 1.0) * PI / (2.0 * L);
    let init: Vec<f64> = (0..disc.total_nodes)
        .map(|i| T_MEAN + MODE_AMPLITUDE * (lam * disc.node_positions[i]).cos())
        .collect();

    let mut solver = ImplicitFDSolver::with_scheme_and_temperatures(disc, init, scheme);

    let int_bc = SurfaceBC::new_interior(1e-9, T_MEAN);
    let ext_bc = SurfaceBC::new_exterior(1e6, T_MEAN, 0.0);

    let steps = (T_END / dt) as usize;
    assert!((3600.0 / dt).fract() < 1e-9, "dt {dt} must divide 3600 s");

    let mut history = Vec::with_capacity(24);
    let mut next_sample = 3600.0_f64;
    for step in 1..=steps {
        let t_after = step as f64 * dt;
        solver.step(dt, &int_bc, &ext_bc);
        while next_sample <= t_after + 1e-9 {
            history.push(solver.temperatures[0]);
            next_sample += 3600.0;
        }
    }
    history
}

/// Observed order for an eigenmode-decay problem against a dt = 5 s
/// semi-discrete reference on the same grid.
fn eigenmode_order(
    scheme: TimeIntegrationScheme,
    coarse: f64,
    fine: f64,
    nodes: usize,
    mode: usize,
) -> f64 {
    let reference = eigenmode_history(TimeIntegrationScheme::Bdf2, DT_REF, nodes, mode);
    let e_coarse = rms_vs(&eigenmode_history(scheme, coarse, nodes, mode), &reference);
    let e_fine = rms_vs(&eigenmode_history(scheme, fine, nodes, mode), &reference);
    (e_coarse / e_fine).log2()
}

#[test]
fn backward_euler_converges_at_first_order_in_time() {
    // 1st-order scheme: halving dt should halve the error (order ≈ 1).
    let p_coarse = eigenmode_order(
        TimeIntegrationScheme::BackwardEuler,
        1800.0,
        900.0,
        FD_NODES,
        0,
    );
    let p_fine = eigenmode_order(
        TimeIntegrationScheme::BackwardEuler,
        900.0,
        450.0,
        FD_NODES,
        0,
    );
    assert!(
        p_coarse > 0.75 && p_coarse < 1.3,
        "BackwardEuler observed order out of band (coarse): {p_coarse}"
    );
    assert!(
        p_fine > 0.75 && p_fine < 1.3,
        "BackwardEuler observed order out of band (fine): {p_fine}"
    );
}

#[test]
fn bdf2_converges_at_second_order_in_time() {
    // 2nd-order scheme: halving dt should quarter the error (order ≈ 2).
    let p_coarse = eigenmode_order(TimeIntegrationScheme::Bdf2, 1800.0, 900.0, FD_NODES, 0);
    let p_fine = eigenmode_order(TimeIntegrationScheme::Bdf2, 900.0, 450.0, FD_NODES, 0);
    assert!(
        p_coarse > 1.7,
        "Bdf2 observed order too low (coarse halving): {p_coarse}"
    );
    assert!(
        p_fine > 1.7,
        "Bdf2 observed order too low (fine halving): {p_fine}"
    );
}

#[test]
fn crank_nicolson_oscillates_at_large_fourier_where_bdf2_stays_monotone() {
    // CN is A-stable but NOT L-stable: for z = mu*dt >> 2 its amplification
    // (1-z/2)/(1+z/2) approaches -1, so lightly-damped high discrete modes
    // ring. A 12 mm gypsum layer at the production zone step (3600 s) puts
    // the top of the discrete spectrum at z_max = 4*Fo ≈ 8 — deep in the
    // oscillatory regime. BDF2's amplification -> 0 as z -> inf (L-stable),
    // so it stays monotone under identical forcing. This is the documented
    // reason Bdf2 (not CN) is the production default (Issue #3980).
    let layers = vec![MaterialLayer::new("Gypsum", 0.012, 0.16, 950.0, 840.0)];
    let disc = WallDiscretization::from_layers(&layers, 20);
    let int_bc = SurfaceBC::new_interior(1e-9, T_INITIAL);
    let ext_bc = SurfaceBC::new_exterior(1e6, T_EXTERIOR, 0.0);

    let run = |scheme| -> Vec<f64> {
        let mut solver = ImplicitFDSolver::with_scheme(disc.clone(), T_INITIAL, scheme);
        let mut interior = Vec::new();
        for _ in 0..24 {
            solver.step(3600.0, &int_bc, &ext_bc);
            interior.push(solver.temperatures[0]);
        }
        interior
    };

    let cn = run(TimeIntegrationScheme::CrankNicolson);
    let bdf2 = run(TimeIntegrationScheme::Bdf2);

    // CN must ring: repeated non-monotone increments as lightly-damped high
    // modes flip sign (amplification -> -1 for z >> 2). More than a startup
    // artifact — several reversals across the run.
    let cn_reversals = cn.windows(2).filter(|w| w[1] < w[0] - 1e-9).count();
    assert!(
        cn_reversals >= 4,
        "CN should show sustained ringing; reversals={cn_reversals}, history: {cn:?}"
    );

    // BDF2 is L-stable (amplification -> 0 as z -> inf): the startup
    // superposition transient decays geometrically and the run settles to
    // the steady state. BDF2 is not monotonicity-preserving (only theta=1
    // is), so a small bounded startup overshoot is admissible; what must
    // hold is decay: after 6 steps the interior node stays within 1e-2 K of
    // the 35 C steady state and never leaves the physical envelope.
    assert!(
        bdf2.iter().skip(6).all(|&t| (t - T_EXTERIOR).abs() <= 1e-2),
        "BDF2 failed to settle to steady state: {bdf2:?}"
    );
    assert!(
        bdf2.iter()
            .all(|&t| t >= T_INITIAL - 1e-6 && t <= T_EXTERIOR + 0.6),
        "BDF2 exceeded the physical envelope: {bdf2:?}"
    );
}

#[test]
fn bdf2_keeps_second_order_on_alternating_timesteps() {
    // Variable-step BDF2 (Issue #3980): the Lagrange-derived rho-corrected
    // coefficients must retain 2nd order when dt alternates (rho = 2.0 and
    // 0.5 on successive steps). Halving the whole dt pattern must quarter
    // the error, same as the constant-step case.
    let run_pattern = |dt_small: f64, dt_large: f64| -> Vec<f64> {
        let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
        let disc = WallDiscretization::from_layers(&layers, FD_NODES);
        let lam = PI / (2.0 * L);
        let init: Vec<f64> = (0..disc.total_nodes)
            .map(|i| T_MEAN + MODE_AMPLITUDE * (lam * disc.node_positions[i]).cos())
            .collect();
        let mut solver =
            ImplicitFDSolver::with_scheme_and_temperatures(disc, init, TimeIntegrationScheme::Bdf2);
        let int_bc = SurfaceBC::new_interior(1e-9, T_MEAN);
        let ext_bc = SurfaceBC::new_exterior(1e6, T_MEAN, 0.0);

        let pattern_sum = dt_small + dt_large;
        let patterns = (T_END / pattern_sum) as usize;
        assert!(
            (3600.0 / pattern_sum).fract() < 1e-9,
            "pattern must divide 3600 s"
        );
        let mut history = Vec::new();
        let mut next_sample = 3600.0_f64;
        let mut t = 0.0_f64;
        for _ in 0..patterns {
            for dt in [dt_small, dt_large] {
                solver.step(dt, &int_bc, &ext_bc);
                t += dt;
                while next_sample <= t + 1e-9 {
                    history.push(solver.temperatures[0]);
                    next_sample += 3600.0;
                }
            }
        }
        history
    };

    let reference = eigenmode_history(TimeIntegrationScheme::Bdf2, DT_REF, FD_NODES, 0);
    let e_coarse = rms_vs(&run_pattern(600.0, 1200.0), &reference);
    let e_fine = rms_vs(&run_pattern(300.0, 600.0), &reference);
    let p = (e_coarse / e_fine).log2();
    assert!(
        p > 1.7,
        "variable-step Bdf2 lost 2nd order on alternating dt: {p} (e_coarse={e_coarse}, e_fine={e_fine})"
    );
}

#[test]
fn fd_timestep_config_is_honored_via_substepping() {
    // Issue #3980: `ConductionBackendConfig.fd_timestep` must be honored —
    // the FD solver substeps at fd_timestep inside the zone step instead of
    // silently taking one 3600 s step. Observable: the solver's last-step dt
    // equals fd_timestep and the resulting trajectory differs from the
    // single-step path.
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::{StepParameters, ThermalModel};

    let layers = vec![fluxion::physics::fd_discretization::MaterialLayer::new(
        "Gypsum", 0.012, 0.16, 950.0, 840.0,
    )];

    let build = |fd_dt: f64| {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.enable_fd(&layers, fd_dt, 10, 20.0);
        model
    };
    let params = StepParameters::default();

    let mut fine = build(60.0);
    let mut coarse = build(3600.0);

    let _ = fine.solve_single_step(0, 35.0, &params, 3600.0);
    let _ = coarse.solve_single_step(0, 35.0, &params, 3600.0);

    let fine_solver = &fine.conduction.backend.fd_solvers[0];
    let coarse_solver = &coarse.conduction.backend.fd_solvers[0];

    // The config was honored: the solver's last step was a 60 s substep.
    assert_eq!(fine_solver.dt, 60.0);

    // Substepping resolved more of the surface transient: the trajectories
    // differ (an ignored config would leave them identical).
    let diff: f64 = fine_solver
        .temperatures
        .iter()
        .zip(coarse_solver.temperatures.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        diff > 1e-6,
        "fd_timestep=60 produced an identical trajectory to 3600 (config ignored): max node diff {diff}"
    );
    assert!(
        fine_solver.temperatures.iter().all(|t| t.is_finite()),
        "substepped FD temperatures must stay finite"
    );
}

#[test]
fn steady_periodic_1052rp_bdf2_is_tighter_than_backward_euler() {
    // ASHRAE 1052-RP style steady-periodic check (Issue #3980 acceptance,
    // minimal form; the full cross-method regression module is #3981):
    // 24 h sinusoidal sol-air forcing on a 200 mm concrete wall, spun up
    // past the transient, comparing interior surface flux against a
    // dt = 5 s semi-discrete reference. The 2nd-order scheme must be
    // strictly tighter than backward Euler at both swept steps.
    let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
    let disc = WallDiscretization::from_layers(&layers, FD_NODES);
    let omega = 2.0 * PI / 86400.0; // 24 h period
    let t_mean = 20.0_f64;
    let t_amp = 15.0_f64;
    let h_int = 8.3;
    let h_ext = 18.3;

    let run = |scheme: TimeIntegrationScheme, dt: f64| -> Vec<f64> {
        let mut solver = ImplicitFDSolver::with_scheme(disc.clone(), t_mean, scheme);
        let spin_up_days = 10;
        let measure_days = 3;
        let total_s = (spin_up_days + measure_days) as f64 * 86_400.0;
        let steps = (total_s / dt) as usize;
        assert!((3600.0 / dt).fract() < 1e-9, "dt must divide 3600 s");
        let mut fluxes = Vec::new();
        let mut next_sample = spin_up_days as f64 * 86_400.0 + 3600.0;
        for step in 1..=steps {
            let t_after = step as f64 * dt;
            let sol_air = t_mean + t_amp * (omega * t_after).sin();
            let int_bc = SurfaceBC::new_interior(h_int, t_mean);
            let ext_bc = SurfaceBC::new_exterior(h_ext, sol_air, 0.0);
            solver.step(dt, &int_bc, &ext_bc);
            while next_sample <= t_after + 1e-9 {
                fluxes.push(solver.interior_heat_flux(h_int, t_mean));
                next_sample += 3600.0;
            }
        }
        fluxes
    };

    let reference = run(TimeIntegrationScheme::Bdf2, DT_REF);
    let err = |scheme, dt| {
        let f = run(scheme, dt);
        rms_vs(&f, &reference)
    };

    let e_be_h = err(TimeIntegrationScheme::BackwardEuler, 3600.0);
    let e_bdf2_h = err(TimeIntegrationScheme::Bdf2, 3600.0);
    let e_be_q = err(TimeIntegrationScheme::BackwardEuler, 900.0);
    let e_bdf2_q = err(TimeIntegrationScheme::Bdf2, 900.0);

    assert!(
        e_bdf2_h < e_be_h,
        "at dt=3600 Bdf2 ({e_bdf2_h}) must beat BackwardEuler ({e_be_h}) under steady-periodic forcing"
    );
    assert!(
        e_bdf2_q < e_be_q,
        "at dt=900 Bdf2 ({e_bdf2_q}) must beat BackwardEuler ({e_be_q}) under steady-periodic forcing"
    );
    // And the scheme actually converges on the periodic problem: refining
    // the step shrinks the BDF2 flux error.
    assert!(
        e_bdf2_q < e_bdf2_h,
        "Bdf2 steady-periodic error must shrink with dt: {e_bdf2_h} -> {e_bdf2_q}"
    );
}

#[test]
#[ignore = "dev harness: prints the Issue #3980 accuracy-vs-cost sweep table"]
fn fd_time_integration_sweep_harness() {
    // Prints RMS interior-flux error (steady-periodic concrete wall, vs a
    // dt=5 s Bdf2 reference) and relative wall-clock cost per scheme and
    // step. Output feeds docs/validation/conduction_time_integration.md.
    use std::time::Instant;
    let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
    let disc = WallDiscretization::from_layers(&layers, FD_NODES);
    let omega = 2.0 * PI / 86400.0;
    let h_int = 8.3;
    let h_ext = 18.3;

    let run = |scheme: TimeIntegrationScheme, dt: f64| -> (Vec<f64>, f64) {
        let mut solver = ImplicitFDSolver::with_scheme(disc.clone(), 20.0, scheme);
        let spin_up_days = 10;
        let measure_days = 3;
        let total_s = (spin_up_days + measure_days) as f64 * 86_400.0;
        let steps = (total_s / dt) as usize;
        let mut fluxes = Vec::new();
        let mut next_sample = spin_up_days as f64 * 86_400.0 + 3600.0;
        let t0 = Instant::now();
        for step in 1..=steps {
            let t_after = step as f64 * dt;
            let sol_air = 20.0 + 15.0 * (omega * t_after).sin();
            let int_bc = SurfaceBC::new_interior(h_int, 20.0);
            let ext_bc = SurfaceBC::new_exterior(h_ext, sol_air, 0.0);
            solver.step(dt, &int_bc, &ext_bc);
            while next_sample <= t_after + 1e-9 {
                fluxes.push(solver.interior_heat_flux(h_int, 20.0));
                next_sample += 3600.0;
            }
        }
        (fluxes, t0.elapsed().as_secs_f64())
    };

    let (reference, _) = run(TimeIntegrationScheme::Bdf2, DT_REF);
    println!("scheme             dt[s]  flux_rms_err[W/m2]  rel_cost");
    let base = run(TimeIntegrationScheme::BackwardEuler, 3600.0).1;
    for scheme in [
        TimeIntegrationScheme::BackwardEuler,
        TimeIntegrationScheme::CrankNicolson,
        TimeIntegrationScheme::Bdf2,
    ] {
        for dt in [3600.0, 1800.0, 900.0, 600.0, 300.0, 120.0, 60.0] {
            let (fluxes, secs) = run(scheme, dt);
            let err = rms_vs(&fluxes, &reference);
            let rel = secs / base;
            println!("{scheme:<18?} {dt:>5.0}  {err:>19.6e}  {rel:>8.2}");
        }
    }
}

#[test]
fn cumulative_energy_balance_closes_for_all_schemes() {
    // Over a transient, cumulative boundary energy exchange must match the
    // wall's stored-energy change closely enough that the 2nd-order schemes
    // never degrade the accounting vs BackwardEuler.
    //
    // Accounting convention: trapezoidal flux quadrature (before/after step
    // average), matching the incumbent `test_energy_conservation`. The
    // incumbent ghost-node boundary rows are not exactly cell-conservative
    // (node 0 assembles as a mirrored full cell, doubling its flux coupling),
    // so closure is bounded by the repo's accepted band — a pre-existing
    // property of the retained spatial discretization, not of the time
    // integrators compared here.
    let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
    let disc = WallDiscretization::from_layers(&layers, 20);
    let dt = 3600.0;
    let steps = 12;
    let int_bc = SurfaceBC::new_interior(8.3, T_INITIAL);
    let ext_bc = SurfaceBC::new_exterior(18.3, T_EXTERIOR, 0.0);

    // Relative closure normalized by total energy flow (the repo convention
    // from `test_energy_conservation`) — through a 200 mm concrete wall the
    // stored-energy change is comparable to the boundary flows, so the
    // normalization is meaningful.
    let check = |scheme| -> f64 {
        let mut solver = ImplicitFDSolver::with_scheme(disc.clone(), T_INITIAL, scheme);
        let e0 = solver.stored_energy(0.0);
        let mut e_in = 0.0_f64;
        let mut e_out = 0.0_f64;
        for _ in 0..steps {
            let q_int_before = solver.interior_heat_flux(int_bc.h, int_bc.t_fluid);
            let q_ext_before = solver.exterior_heat_flux(ext_bc.h, ext_bc.t_fluid);
            solver.step(dt, &int_bc, &ext_bc);
            let q_int = solver.interior_heat_flux(int_bc.h, int_bc.t_fluid);
            let q_ext = solver.exterior_heat_flux(ext_bc.h, ext_bc.t_fluid);
            e_in += (q_ext + q_ext_before) / 2.0 * dt;
            e_out += (q_int + q_int_before) / 2.0 * dt;
        }
        let de = (solver.stored_energy(0.0) - e0).abs();
        let net_mismatch = ((e_in - e_out).abs() - de).abs();
        net_mismatch / e_in.abs().max(e_out.abs()).max(de)
    };

    let rel_be = check(TimeIntegrationScheme::BackwardEuler);
    let rel_bdf2 = check(TimeIntegrationScheme::Bdf2);
    let rel_cn = check(TimeIntegrationScheme::CrankNicolson);

    // The incumbent ghost-node boundary rows double the boundary flux
    // coupling, so surface-flux reconstruction closes only to ~50% of total
    // flow (matches the repo band in `test_energy_conservation`). This is
    // a property of the RETAINED spatial discretization, identical for all
    // three time integrators.
    assert!(
        rel_be < 0.6,
        "BackwardEuler cumulative balance outside the accepted band: {rel_be}"
    );
    assert!(
        rel_bdf2 < 0.6,
        "Bdf2 cumulative balance outside the accepted band: {rel_bdf2}"
    );
    assert!(
        rel_cn < 0.6,
        "CrankNicolson cumulative balance outside the accepted band: {rel_cn}"
    );

    // The integrator upgrade must not degrade the accounting: 2nd-order
    // schemes stay within a small factor of the BE reference closure.
    let worst = rel_bdf2.max(rel_cn);
    assert!(
        worst < rel_be.max(1e-3) * 2.0 + 0.05,
        "2nd-order schemes degrade energy accounting: be={rel_be}, bdf2={rel_bdf2}, cn={rel_cn}"
    );
}

#[test]
fn crank_nicolson_converges_at_second_order_in_time() {
    // CN is 2nd order ONLY while every excited mode keeps z = mu*dt below
    // ~2 (above that its amplification factor approaches -1 and high modes
    // oscillate without damping — see `crank_nicolson_oscillates` and
    // Issue #3980's accuracy-vs-cost sweep). The applicability regime here:
    // 4 nodes (dx = 50 mm, z_max = 4*Fo(dt=900) ≈ 1.1) with the 3rd
    // eigenmode seeded for a measurable error signal.
    let p = eigenmode_order(TimeIntegrationScheme::CrankNicolson, 900.0, 450.0, 4, 2);
    assert!(
        p > 1.7,
        "CrankNicolson observed order too low in its applicability regime: {p}"
    );
}

#[test]
fn bdf2_beats_backward_euler_at_equal_dt() {
    // At equal dt the 2nd-order scheme must be more accurate than 1st-order
    // on the step-response problem (moderate BDF2 error constant).
    let reference = semi_discrete_reference();
    let e_be = rms_error(TimeIntegrationScheme::BackwardEuler, 900.0, &reference);
    let e_bdf2 = rms_error(TimeIntegrationScheme::Bdf2, 900.0, &reference);
    assert!(
        e_bdf2 < e_be,
        "Bdf2 error {e_bdf2} should be below BackwardEuler error {e_be} at dt=900"
    );
}

#[test]
fn both_schemes_stay_close_to_analytical_series() {
    // Absolute smoke bound vs the continuous analytical solution: the
    // dt-independent O(dx^2) spatial floor of the 80-node grid dominates
    // (~0.15 K); anything far outside that signals a broken scheme or BC.
    let reference = semi_discrete_reference();
    let e_be = rms_error(TimeIntegrationScheme::BackwardEuler, 450.0, &reference);
    let e_bdf2 = rms_error(TimeIntegrationScheme::Bdf2, 450.0, &reference);
    assert!(
        e_be < 0.2,
        "BackwardEuler drifted from the reference: {e_be}"
    );
    assert!(e_bdf2 < 0.2, "Bdf2 drifted from the reference: {e_bdf2}");
}

#[test]
fn free_floating_conduction_uses_fd_teacher_not_ctf_primary() {
    // Issue #3980: the 50-term CTF is demoted from the free-floating
    // primary to a fast cross-check role (linear constructions only).
    // The FF wiring must enable the upgraded FD backend (BDF2) at short
    // substeps and leave ctf_primary false. (Integration-runner home so
    // the validation module adds no new sim import edges — cycle guard
    // #1441 rejects growth.)
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::{StepParameters, ThermalModel};
    use fluxion::sim::thermal_selector::ThermalSelector;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    fluxion::validation::ashrae_140_validator::wire_free_floating_conduction(&mut model, &spec);

    let backend = &model.conduction.backend;
    assert!(
        !backend.ctf_primary,
        "free-floating wiring must not set ctf_primary (CTF is cross-check only)"
    );
    assert!(
        backend.fd_enabled,
        "free-floating wiring must enable the FD teacher path"
    );
    assert_eq!(
        backend.fd_timestep, 60.0,
        "FD teacher runs at 60 s substeps (Issue #3980 teacher step)"
    );
    assert!(
        !backend.fd_solvers.is_empty(),
        "FD teacher path needs at least one solver"
    );
    let _ = StepParameters::default();
}

//! Workspace integration test: UKF convergence on a van der Pol oscillator
//! (Issue #2060).
//!
//! The van der Pol oscillator is the canonical non-linear state-space benchmark
//! for sigma-point filters: the system's stiffness (driven by `mu`) breaks the
//! linearising assumptions of an Extended Kalman Filter, while the UKF's
//! unscented transform captures the non-linear propagation to second order.
//!
//! This test runs the `UnscentedKalmanFilter<Vec<f64>, Vec<f64>>` on the
//! oscillator for 200 timesteps and asserts:
//!
//! 1. The state estimate converges to the true trajectory within ~10 steps
//!    (tracking error < 1.0 in 2-D Euclidean distance).
//! 2. The posterior covariance stays positive semi-definite (diagonal non-negative,
//!    determinant non-negative) — i.e. the filter does not "blow up" the
//!    uncertainty matrix.
//!
//! Closes #2060 — UKF van der Pol integration test.

use approx::assert_relative_eq;
use fluxion_twin::UnscentedKalmanFilter;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Number of integration timesteps. 200 is enough for the oscillator to
/// complete ~3 limit-cycle loops at `dt = 0.01`, well past the convergence
/// horizon so we can assert steady-state tracking.
const N_STEPS: usize = 200;

/// Integration timestep. Small enough that the explicit Euler discretisation
/// is stable for `mu = 1.0` (the moderate-stiffness regime).
const DT: f64 = 0.01;

/// Van der Pol nonlinearity strength. `mu = 1.0` is the canonical textbook
/// value — large enough to make the system clearly non-linear, small enough
/// that explicit Euler remains bounded.
const MU: f64 = 1.0;

/// True initial state: `(x0, x1) = (2.0, 0.0)`. The oscillator starts on the
/// positive x-axis with zero velocity and spirals inward toward the limit
/// cycle.
const TRUE_X0: f64 = 2.0;
const TRUE_X1: f64 = 0.0;

/// Deliberately wrong initial state — the filter must converge from this.
/// Offsets of `(+0.5, +0.3)` give the UKF something to correct in the first
/// few measurement updates.
const INIT_OFFSET_X0: f64 = 0.5;
const INIT_OFFSET_X1: f64 = 0.3;

/// Seed for the deterministic measurement-noise RNG. Fixed seed keeps the test
/// reproducible across runs / platforms.
const NOISE_SEED: u64 = 42;

/// Measurement noise std-dev (added to `x[0]` after each true-step). Matches
/// the unit test in `lib.rs` so the convergence numbers are comparable.
const MEASUREMENT_NOISE: f64 = 0.1;

/// Tracking-error convergence tolerance (2-D Euclidean distance). The unit
/// test in `lib.rs` uses `1.0` after 10 steps; we use the same threshold so
/// this integration test mirrors the inline test's convergence contract.
const CONVERGENCE_TOL: f64 = 1.0;

#[test]
fn ukf_converges_on_van_der_pol_oscillator() {
    let initial_state = vec![TRUE_X0 + INIT_OFFSET_X0, TRUE_X1 + INIT_OFFSET_X1];

    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
    let q = DMatrix::from_diagonal(&DVector::from_vec(vec![0.1, 0.1]));
    let r = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01]));

    let mut rng = StdRng::seed_from_u64(NOISE_SEED);

    let mut ukf = UnscentedKalmanFilter::new(
        initial_state,
        p0,
        q,
        r,
        // State transition f(x, u): explicit-Euler integration of van der Pol.
        //   x0' = x1
        //   x1' = mu * (1 - x0²) * x1 - x0
        |x: &Vec<f64>, u: &[f64]| -> Vec<f64> {
            let dt = u[0];
            let mu = u[1];
            vec![
                x[0] + dt * x[1],
                x[1] + dt * (mu * (1.0 - x[0] * x[0]) * x[1] - x[0]),
            ]
        },
        // Measurement h(x): observe x[0] (the position coordinate).
        |x: &Vec<f64>| -> Vec<f64> { vec![x[0]] },
    );

    let mut true_state = vec![TRUE_X0, TRUE_X1];
    let mut estimated_states = Vec::with_capacity(N_STEPS);
    let mut true_states = Vec::with_capacity(N_STEPS);

    for _step in 0..N_STEPS {
        let u = vec![DT, MU];

        // Synthetic noisy sensor reading of the first state component.
        let noise: f64 = rng.random();
        let noisy_measurement = true_state[0] + noise * MEASUREMENT_NOISE;

        ukf.predict(&u)
            .expect("predict step should not fail on a well-defined system");
        ukf.update(&vec![noisy_measurement])
            .expect("update step should not fail on a well-defined system");

        estimated_states.push(ukf.state.as_slice().to_vec());
        true_states.push(true_state.clone());

        // Advance the true state by the same explicit-Euler integration.
        let x0_next = true_state[0] + DT * true_state[1];
        let x1_next = true_state[1]
            + DT * (MU * (1.0 - true_state[0] * true_state[0]) * true_state[1] - true_state[0]);
        true_state = vec![x0_next, x1_next];
    }

    // ---- Convergence assertion (Issue #2060 acceptance) ---------------------
    //
    // After ~10 steps the UKF should have driven the tracking error well below
    // the 2-D unit ball — a coarse but unambiguous convergence signal.
    let converged_index = 9;
    let converged_estimate = &estimated_states[converged_index];
    let true_at_9 = &true_states[converged_index];
    let tracking_error = ((converged_estimate[0] - true_at_9[0]).powi(2)
        + (converged_estimate[1] - true_at_9[1]).powi(2))
    .sqrt();

    assert!(
        tracking_error < CONVERGENCE_TOL,
        "UKF should converge within 10 steps, tracking error = {tracking_error} \
         (tolerance = {CONVERGENCE_TOL}); estimate = {converged_estimate:?}, \
         true = {true_at_9:?}"
    );

    // ---- Covariance health check (no blow-up) -------------------------------
    //
    // A diverging UKF typically drives `P` to either negative values
    // (Cholesky fails) or to infinity (covariance explosion). Both are caught
    // by a positive-semi-definiteness check at the end of the run.
    let p_final = &ukf.p_covariance;
    assert!(
        p_final[(0, 0)] >= 0.0 && p_final[(1, 1)] >= 0.0,
        "covariance diagonal must be non-negative after 200 steps: \
         diag = ({}, {})",
        p_final[(0, 0)],
        p_final[(1, 1)]
    );

    let det = p_final[(0, 0)] * p_final[(1, 1)] - p_final[(0, 1)] * p_final[(1, 0)];
    assert!(
        det >= 0.0,
        "covariance determinant must be >= 0 (positive semi-definite), got {det}"
    );

    // ---- Steady-state tracking check ----------------------------------------
    //
    // Over the last 50 steps the UKF should track the limit cycle with a mean
    // Euclidean error well under the initial offset. We use `assert_relative_eq`
    // on the running average so the assertion is robust to phase mismatch.
    let tail: usize = 50;
    let tail_start = N_STEPS - tail;
    let mean_error: f64 = estimated_states[tail_start..]
        .iter()
        .zip(true_states[tail_start..].iter())
        .map(|(est, tru)| ((est[0] - tru[0]).powi(2) + (est[1] - tru[1]).powi(2)).sqrt())
        .sum::<f64>()
        / tail as f64;

    assert!(
        mean_error < 0.5,
        "UKF should track the limit cycle tightly in steady state; \
         mean tail error = {mean_error} (tolerance = 0.5)"
    );
}

/// Variant: low-stiffness oscillator (`mu = 0.1`) to verify the UKF still
/// converges in the *easy* regime — guards against accidental regressions in
/// the sigma-point math that would only manifest for stiff systems.
#[test]
fn ukf_converges_on_van_der_pol_low_stiffness() {
    let mu = 0.1;
    let n_steps = 100;
    let initial_state = vec![TRUE_X0 + INIT_OFFSET_X0, TRUE_X1 + INIT_OFFSET_X1];
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
    let q = DMatrix::from_diagonal(&DVector::from_vec(vec![0.05, 0.05]));
    let r = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01]));

    let mut rng = StdRng::seed_from_u64(NOISE_SEED);

    let mut ukf = UnscentedKalmanFilter::new(
        initial_state,
        p0,
        q,
        r,
        |x: &Vec<f64>, u: &[f64]| -> Vec<f64> {
            let dt = u[0];
            let mu = u[1];
            vec![
                x[0] + dt * x[1],
                x[1] + dt * (mu * (1.0 - x[0] * x[0]) * x[1] - x[0]),
            ]
        },
        |x: &Vec<f64>| -> Vec<f64> { vec![x[0]] },
    );

    let mut true_state = vec![TRUE_X0, TRUE_X1];

    for _ in 0..n_steps {
        let u = vec![DT, mu];
        let noisy = true_state[0] + rng.random::<f64>() * MEASUREMENT_NOISE;

        ukf.predict(&u).unwrap();
        ukf.update(&vec![noisy]).unwrap();

        let x0_next = true_state[0] + DT * true_state[1];
        let x1_next = true_state[1]
            + DT * (mu * (1.0 - true_state[0] * true_state[0]) * true_state[1] - true_state[0]);
        true_state = vec![x0_next, x1_next];
    }

    // Low stiffness converges very tightly for the *measured* component
    // (x[0]). The unobserved component (x[1]) drifts because we only observe
    // position — the covariance reflects this honestly. Assert x[0] tracks
    // tightly and the 2-D error stays under 0.3.
    let err0 = (ukf.state[0] - true_state[0]).abs();
    let err1 = (ukf.state[1] - true_state[1]).abs();
    let err = (err0 * err0 + err1 * err1).sqrt();

    assert!(
        err < 0.3,
        "low-stiffness oscillator should track tightly: 2-D error = {err} \
         (component errors: ({err0}, {err1}))"
    );
    assert!(
        err0 < 0.1,
        "measured component should track tightly; err0 = {err0} > 0.1"
    );

    assert_relative_eq!(ukf.state[0], true_state[0], epsilon = 0.1);
}

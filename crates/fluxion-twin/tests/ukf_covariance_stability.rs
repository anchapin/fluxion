//! Workspace integration test: UKF 1000-step covariance stability (Issue #2061).
//!
//! Long-running state estimators in production (digital twins, model-predictive
//! control, sensor fusion) run for thousands of timesteps without restart.
//! A UKF that loses positive semi-definiteness in `P` (covariance blow-up or
//! collapse) becomes numerically invalid: the Cholesky factorisation that
//! underpins the sigma-point transform fails (`NonPositiveDefiniteMatrix`),
//! and downstream consumers see `NaN` state estimates.
//!
//! This test runs the filter for **1000 predict/update cycles** and asserts:
//!
//! 1. Every intermediate covariance is positive semi-definite (diagonal
//!    non-negative, determinant non-negative).
//! 2. The trace stays bounded — i.e. the filter neither explodes nor
//!    collapses uncertainty.
//! 3. The numerical Cholesky never fails — i.e. `predict` / `update` succeed
//!    for every step.
//!
//! Closes #2061 — UKF 1000-step covariance stability integration test.

use fluxion_twin::UnscentedKalmanFilter;
use nalgebra::{DMatrix, DVector};

/// Number of predict / update cycles. 1000 is the Issue #2061 acceptance
/// threshold.
const N_STEPS: usize = 1000;

/// Upper bound on the trace of `P` after the run. A well-designed filter on
/// this system should converge to a steady-state uncertainty well under
/// 1e6 — anything above this indicates covariance explosion.
const TRACE_UPPER_BOUND: f64 = 1.0e6;

/// Lower bound on the trace of `P` after the run. A filter that collapses
/// its own uncertainty (becoming overconfident) drives `P → 0`. A trace
/// below this threshold indicates covariance collapse — the filter would
/// ignore new measurements and stop responding to reality.
const TRACE_LOWER_BOUND: f64 = 1.0e-8;

/// 2-D state — same shape as the inline `test_covariance_positive_semi_definite`
/// in `lib.rs` so we test the same code paths.
fn make_ukf() -> UnscentedKalmanFilter<Vec<f64>, Vec<f64>> {
    let initial_state = vec![1.0, 0.0];

    // Initial covariance: deliberately not diagonal so we exercise the
    // off-diagonal terms in every Cholesky factorisation.
    let mut p0 = DMatrix::zeros(2, 2);
    p0[(0, 0)] = 1.0;
    p0[(0, 1)] = 0.5;
    p0[(1, 0)] = 0.5;
    p0[(1, 1)] = 1.0;

    let q = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01]));
    let r = DMatrix::from_diagonal(&DVector::from_vec(vec![0.1]));

    UnscentedKalmanFilter::new(
        initial_state,
        p0,
        q,
        r,
        // Stable linear decay: x' = (0.9 * x0, 0.8 * x1). No eigenvalues on
        // the unit circle, so the filter's posterior must converge to a
        // finite steady-state covariance.
        |x: &Vec<f64>, _: &[f64]| -> Vec<f64> { vec![x[0] * 0.9, x[1] * 0.8] },
        |x: &Vec<f64>| -> Vec<f64> { vec![x[0]] },
    )
}

#[test]
fn ukf_covariance_stable_over_1000_steps() {
    let mut ukf = make_ukf();
    let mut max_trace: f64 = 0.0;
    let mut min_trace: f64 = f64::INFINITY;

    for step in 0..N_STEPS {
        ukf.predict(&vec![0.0])
            .unwrap_or_else(|e| panic!("predict failed at step {step}: {e}"));
        ukf.update(&vec![1.0])
            .unwrap_or_else(|e| panic!("update failed at step {step}: {e}"));

        let p = &ukf.p_covariance;

        // ---- Positive semi-definiteness every step (Issue #2061) -----------
        assert!(
            p[(0, 0)] >= 0.0 && p[(1, 1)] >= 0.0,
            "covariance diagonal became negative at step {step}: \
             p = {p:?}"
        );

        let det = p[(0, 0)] * p[(1, 1)] - p[(0, 1)] * p[(1, 0)];
        assert!(
            det >= 0.0,
            "covariance lost positive semi-definiteness at step {step}: \
             det = {det} (p = {p:?})"
        );

        // ---- Track trace bounds ---------------------------------------------
        let trace = p[(0, 0)] + p[(1, 1)];
        max_trace = max_trace.max(trace);
        min_trace = min_trace.min(trace);

        assert!(
            trace.is_finite(),
            "covariance trace went non-finite at step {step}: trace = {trace}"
        );
    }

    // ---- Final-state assertions ---------------------------------------------
    let p_final = &ukf.p_covariance;
    let final_trace = p_final[(0, 0)] + p_final[(1, 1)];

    assert!(
        final_trace < TRACE_UPPER_BOUND,
        "covariance exploded after {N_STEPS} steps: trace = {final_trace} \
         (bound = {TRACE_UPPER_BOUND}); max observed = {max_trace}"
    );

    assert!(
        final_trace > TRACE_LOWER_BOUND,
        "covariance collapsed after {N_STEPS} steps: trace = {final_trace} \
         (bound = {TRACE_LOWER_BOUND}); min observed = {min_trace}"
    );

    // Sanity: the max and min we observed over the run should both fall inside
    // the same bounds (otherwise an intermediate excursion was masked by a
    // late-step correction — unlikely but worth flagging).
    assert!(
        max_trace < TRACE_UPPER_BOUND,
        "max trace over {N_STEPS} steps exceeded bound: {max_trace}"
    );
    assert!(
        min_trace > TRACE_LOWER_BOUND,
        "min trace over {N_STEPS} steps dropped below bound: {min_trace}"
    );
}

/// Sanity variant: zero process noise drives the filter toward full
/// uncertainty in the unobserved dimension. Verifies the bound assertions
/// above do not silently fail when `Q` is small but finite.
#[test]
fn ukf_covariance_stable_with_tiny_process_noise() {
    let mut ukf = make_ukf();
    // Drop the process noise to 1e-6 — should still be stable.
    ukf.process_noise = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0e-6, 1.0e-6]));

    for _ in 0..N_STEPS {
        ukf.predict(&vec![0.0]).unwrap();
        ukf.update(&vec![1.0]).unwrap();
    }

    let p_final = &ukf.p_covariance;
    let trace = p_final[(0, 0)] + p_final[(1, 1)];
    assert!(
        trace.is_finite() && trace > 0.0,
        "tiny-process-noise filter produced non-positive or non-finite trace: {trace}"
    );
    assert!(trace < TRACE_UPPER_BOUND);
}

/// Sanity variant: the *measured* component's variance must converge
/// (information from `h(x) = x[0]` reduces uncertainty in `x[0]`). The
/// unobserved component (`x[1]`) accumulates process noise and its variance
/// can grow — this is correct UKF behaviour for a partially-observable
/// system. We assert `p[0,0]` stabilises well below the initial value.
#[test]
fn ukf_measured_state_covariance_converges() {
    let mut ukf = make_ukf();

    // Warm-up: 200 steps so the filter reaches steady state.
    for _ in 0..200 {
        ukf.predict(&vec![0.0]).unwrap();
        ukf.update(&vec![1.0]).unwrap();
    }

    let p00_warmup = ukf.p_covariance[(0, 0)];

    // Continue for 800 more steps and verify the measured-state variance
    // does not grow and is well below the initial value of 1.0.
    for _ in 0..800 {
        ukf.predict(&vec![0.0]).unwrap();
        ukf.update(&vec![1.0]).unwrap();
    }

    let p00_final = ukf.p_covariance[(0, 0)];

    assert!(
        p00_final < p00_warmup * 1.10,
        "measured-state variance grew in steady state: warm-up = {p00_warmup}, \
         final = {p00_final}"
    );
    // Steady-state should be much smaller than initial p[0,0] = 1.0.
    assert!(
        p00_final < 0.5,
        "filter did not converge for measured state: p[0,0] = {p00_final}"
    );
}

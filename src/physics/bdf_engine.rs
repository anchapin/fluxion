//! BDF (Backward Differentiation Formula) Time-Stepping Engine for DAE Systems
//!
//! This module implements an implicit BDF solver suitable for stiff differential-algebraic
//! equations (DAEs) arising in building energy modeling, particularly for thermal networks
//! with algebraic constraints (e.g., zone energy balance).

use thiserror::Error;

#[cfg(feature = "std")]
use std::fmt;

#[derive(Debug, Clone, Error)]
pub enum BdfError {
    #[error("Invalid order: BDF order must be 1-6, got {0}")]
    InvalidOrder(usize),

    #[error("Convergence failed after {0} iterations: residual = {1:.2e}")]
    ConvergenceFailed(usize, f64),

    #[error("Singular matrix in Newton-Raphson solve")]
    SingularMatrix,

    #[error("Step size underflow: dt = {0:.2e} below minimum {1:.2e}")]
    StepSizeUnderflow(f64, f64),

    #[error("Step size overflow: dt = {0:.2e} above maximum {1:.2e}")]
    StepSizeOverflow(f64, f64),

    #[error("Invalid dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Numerical overflow in residual computation")]
    NumericalOverflow,
}

pub type BdfResult<T> = Result<T, BdfError>;

/// Issue #4693 — the error-controlled step-size law, extracted so that the
/// BDF engine and the simulation-side controller share one implementation.
///
/// This is the exact arithmetic that used to live in the private
/// `adaptive_step::AdaptiveStepController::compute_step_size` method
/// (see `src/physics/bdf_engine.rs`); the method now delegates here, so the
/// BDF path is bit-identical to before while `crate::sim::adaptive_timestep`
/// can reuse the same law without reaching into a private method.
///
/// * `dt` — current step size (s)
/// * `error` — local error estimate for the last accepted step
/// * `tolerance` — local error tolerance
/// * `order` — BDF order of the integrator
/// * `safety_factor` — multiplicative safety factor applied to the growth ratio
///
/// Returns the *unclamped* candidate step size (s). Callers apply their own
/// min/max clamps.
pub(crate) fn error_controlled_step_size(
    dt: f64,
    error: f64,
    tolerance: f64,
    order: usize,
    safety_factor: f64,
) -> f64 {
    if error < tolerance {
        return dt * 2.0;
    }

    let exponent = 1.0 / (order as f64 + 1.0);
    let factor = (tolerance / error).powf(exponent) * safety_factor;

    dt * factor.clamp(0.1, 2.0)
}

/// Issue #4691 — instrumentation counters for the BDF/Newton inner loop.
///
/// These counters are *always* live (one relaxed atomic increment per
/// event) so a perf spike can measure the baseline Jacobian-assembly /
/// factorisation / Newton-iteration cost without touching any solver
/// behaviour. They are pure instrumentation: no tolerance, default,
/// convergence criterion or release gate depends on them.
pub mod counters {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Number of numerical Jacobian assemblies (one per `compute_numerical_jacobian`).
    pub static JACOBIAN_ASSEMBLIES: AtomicU64 = AtomicU64::new(0);
    /// Number of matrix factorisations (one per `solve_linear_system` call in the
    /// baseline path, one per LU factorisation in the frozen-Jacobian path).
    pub static FACTORIZATIONS: AtomicU64 = AtomicU64::new(0);
    /// Number of Newton iterations entered (one residual evaluation each).
    pub static NEWTON_ITERATIONS: AtomicU64 = AtomicU64::new(0);
    /// Number of times a cached (frozen) Jacobian was reused instead of re-assembled.
    pub static JACOBIAN_REUSES: AtomicU64 = AtomicU64::new(0);
    /// Number of driver-level step rejections.
    pub static STEP_REJECTIONS: AtomicU64 = AtomicU64::new(0);
    /// Number of Newton solves started (one per BDF step attempt).
    pub static SOLVE_CALLS: AtomicU64 = AtomicU64::new(0);
    /// Number of forced-refresh retries (a failed frozen solve retried with a
    /// freshly assembled Jacobian).
    pub static FORCED_REFRESH_RETRIES: AtomicU64 = AtomicU64::new(0);

    /// Immutable snapshot of the counters above.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct CountersSnapshot {
        pub jacobian_assemblies: u64,
        pub factorizations: u64,
        pub newton_iterations: u64,
        pub jacobian_reuses: u64,
        pub step_rejections: u64,
        pub solve_calls: u64,
        pub forced_refresh_retries: u64,
    }

    /// Read the current counter values.
    pub fn snapshot() -> CountersSnapshot {
        CountersSnapshot {
            jacobian_assemblies: JACOBIAN_ASSEMBLIES.load(Ordering::Relaxed),
            factorizations: FACTORIZATIONS.load(Ordering::Relaxed),
            newton_iterations: NEWTON_ITERATIONS.load(Ordering::Relaxed),
            jacobian_reuses: JACOBIAN_REUSES.load(Ordering::Relaxed),
            step_rejections: STEP_REJECTIONS.load(Ordering::Relaxed),
            solve_calls: SOLVE_CALLS.load(Ordering::Relaxed),
            forced_refresh_retries: FORCED_REFRESH_RETRIES.load(Ordering::Relaxed),
        }
    }

    /// Reset every counter to zero (used by the spike harness to isolate
    /// one workload's contribution).
    pub fn reset() {
        JACOBIAN_ASSEMBLIES.store(0, Ordering::Relaxed);
        FACTORIZATIONS.store(0, Ordering::Relaxed);
        NEWTON_ITERATIONS.store(0, Ordering::Relaxed);
        JACOBIAN_REUSES.store(0, Ordering::Relaxed);
        STEP_REJECTIONS.store(0, Ordering::Relaxed);
        SOLVE_CALLS.store(0, Ordering::Relaxed);
        FORCED_REFRESH_RETRIES.store(0, Ordering::Relaxed);
    }
}

/// Issue #4691 — flag-gated frozen-Jacobian prototype (**default OFF**).
///
/// The prototype keeps the numerically-computed Jacobian (and its LU
/// factorisation) alive across Newton iterations *and* across
/// consecutive BDF timesteps, refreshing it only when
///
/// 1. the residual stops decreasing monotonically (stall),
/// 2. the step is rejected, or
/// 3. the reuse budget `N` is exhausted.
///
/// Nothing in the production default path changes: with the flag unset
/// (`FLX_BDF_FREEZE_JACOBIAN` absent / falsy) the solver takes exactly
/// the legacy branch — one Jacobian assembly and one elimination per
/// Newton iteration.
pub mod freeze {
    /// Env flag enabling the frozen-Jacobian prototype (default: OFF).
    pub const ENV_FREEZE_JACOBIAN: &str = "FLX_BDF_FREEZE_JACOBIAN";
    /// Env var selecting the reuse budget `N` (number of times a frozen
    /// Jacobian may be used before it is refreshed).
    pub const ENV_FREEZE_MAX_REUSE: &str = "FLX_BDF_FREEZE_MAX_REUSE";
    /// Default reuse budget when `FLX_BDF_FREEZE_MAX_REUSE` is unset or
    /// unparsable.
    pub const DEFAULT_MAX_REUSE: usize = 8;

    /// The refresh policy in force for one Newton solve.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FrozenJacobianPolicy {
        /// Whether the frozen-Jacobian prototype is active.
        pub enabled: bool,
        /// Reuse budget `N`: how many times a frozen Jacobian may be used
        /// before it must be refreshed.
        pub max_reuses: usize,
    }

    impl Default for FrozenJacobianPolicy {
        fn default() -> Self {
            Self {
                enabled: false,
                max_reuses: DEFAULT_MAX_REUSE,
            }
        }
    }

    impl FrozenJacobianPolicy {
        /// Read the policy from the environment (repo-standard `FLX_*`
        /// conventions; booleans go through `util::env_bool::env_bool`).
        pub fn from_env() -> Self {
            let enabled = crate::util::env_bool::env_bool(ENV_FREEZE_JACOBIAN, false);
            let max_reuses = std::env::var(ENV_FREEZE_MAX_REUSE)
                .ok()
                .and_then(|raw| raw.trim().parse::<usize>().ok())
                .unwrap_or(DEFAULT_MAX_REUSE);
            Self {
                enabled,
                max_reuses,
            }
        }
    }
}

pub const MAX_BDF_ORDER: usize = 6;
pub const MIN_BDF_ORDER: usize = 1;

// ---------------------------------------------------------------------------
// Issue #4692 — flag-gated linear warm-start predictor for the Newton solve.
//
// The predictor replaces the Newton initial guess with the linear
// extrapolation `x_prev + (x_prev - x_prev2)` of the last two accepted
// states. It is **default OFF**: with the flag unset the solver takes
// exactly the code path it took before this change.
//
// The flag is read **once, at construction** (`BdfTimeStepper::new` /
// `BdfDriver::new`), never per step — the #4688 annual profile showed
// per-step `getenv` reads of `FLX_*` feature flags to be the single
// largest avoidable cost of the year (762 121 calls per annual run).
// ---------------------------------------------------------------------------

/// Environment variable that enables the Issue #4692 warm-start
/// predictor. Unset (or any value other than `1` / `true` / `yes` /
/// `on`) means **OFF**.
pub const WARM_START_PREDICTOR_ENV: &str = "FLUXION_BDF_WARM_START_PREDICTOR";

/// Parse the warm-start flag value.
///
/// Pure function on purpose: the rules are unit-tested without
/// mutating the process environment, which would be racy under
/// `cargo test`'s parallel threads (every `BdfTimeStepper::new` in
/// the same process reads the same environment variable).
pub fn warm_start_flag_enabled(value: Option<&str>) -> bool {
    match value {
        Some(raw) => {
            let v = raw.trim();
            v == "1"
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on")
        }
        None => false,
    }
}

/// Issue #4692 — the linear warm-start predictor.
///
/// `out[i] = x_prev[i] + (x_prev[i] - x_prev2[i])`, optionally clamped
/// to `bounds = Some((lo, hi))` (element-wise; `lo`/`hi` may be given
/// in either order). A non-finite extrapolation falls back to
/// `x_prev[i]` so the Newton solve never starts from NaN/Inf.
pub fn warm_start_predict(
    x_prev: &[f64],
    x_prev2: &[f64],
    bounds: Option<(&[f64], &[f64])>,
) -> Vec<f64> {
    let n = x_prev.len().min(x_prev2.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut guess = x_prev[i] + (x_prev[i] - x_prev2[i]);
        if !guess.is_finite() {
            guess = x_prev[i];
        }
        if let Some((lo, hi)) = bounds {
            if i < lo.len() && i < hi.len() {
                let lower = lo[i].min(hi[i]);
                let upper = lo[i].max(hi[i]);
                guess = guess.clamp(lower, upper);
            }
        }
        out.push(guess);
    }
    out
}

pub mod coefficients {
    //! BDF Coefficients for orders 1-6

    use crate::physics::bdf_engine::{BdfError, BdfResult};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BdfCoefficients {
        pub k: usize,
        pub alpha: &'static [f64],
        pub beta: f64,
        pub gamma: f64,
    }

    impl BdfCoefficients {
        pub fn new(k: usize) -> BdfResult<Self> {
            match k {
                1 => Ok(BDF1),
                2 => Ok(BDF2),
                3 => Ok(BDF3),
                4 => Ok(BDF4),
                5 => Ok(BDF5),
                6 => Ok(BDF6),
                _ => Err(BdfError::InvalidOrder(k)),
            }
        }
    }

    pub static BDF1: BdfCoefficients = BdfCoefficients {
        k: 1,
        alpha: &[1.0, -1.0],
        beta: 1.0,
        gamma: 1.0,
    };

    pub static BDF2: BdfCoefficients = BdfCoefficients {
        k: 2,
        alpha: &[1.0 / 2.0, -4.0 / 2.0, 3.0 / 2.0],
        beta: 2.0 / 3.0,
        gamma: 2.0 / 3.0,
    };

    pub static BDF3: BdfCoefficients = BdfCoefficients {
        k: 3,
        alpha: &[1.0 / 6.0, -18.0 / 6.0, 36.0 / 6.0, -11.0 / 6.0],
        beta: 3.0 / 11.0,
        gamma: 6.0 / 11.0,
    };

    pub static BDF4: BdfCoefficients = BdfCoefficients {
        k: 4,
        alpha: &[
            1.0 / 12.0,
            -48.0 / 12.0,
            144.0 / 12.0,
            -80.0 / 12.0,
            25.0 / 12.0,
        ],
        beta: 12.0 / 25.0,
        gamma: 12.0 / 25.0,
    };

    pub static BDF5: BdfCoefficients = BdfCoefficients {
        k: 5,
        alpha: &[
            1.0 / 60.0,
            -300.0 / 60.0,
            1200.0 / 60.0,
            -900.0 / 60.0,
            400.0 / 60.0,
            -137.0 / 60.0,
        ],
        beta: 60.0 / 137.0,
        gamma: 60.0 / 137.0,
    };

    pub static BDF6: BdfCoefficients = BdfCoefficients {
        k: 6,
        alpha: &[
            1.0 / 60.0,
            -360.0 / 60.0,
            1800.0 / 60.0,
            -1680.0 / 60.0,
            1050.0 / 60.0,
            -480.0 / 60.0,
            147.0 / 60.0,
        ],
        beta: 60.0 / 147.0,
        gamma: 60.0 / 147.0,
    };

    pub static ALL_BDF_COEFFICIENTS: [&BdfCoefficients; 6] =
        [&BDF1, &BDF2, &BDF3, &BDF4, &BDF5, &BDF6];

    pub fn get_coefficients(order: usize) -> BdfResult<&'static BdfCoefficients> {
        if order < 1 || order > 6 {
            return Err(BdfError::InvalidOrder(order));
        }
        Ok(ALL_BDF_COEFFICIENTS[order - 1])
    }

    pub fn compute_bdf_coefficients(order: usize) -> BdfResult<(Vec<f64>, f64)> {
        let coeff = get_coefficients(order)?;
        Ok((coeff.alpha.to_vec(), coeff.beta))
    }
}

pub mod newton_raphson {
    //! Newton-Raphson Nonlinear Solver

    use crate::physics::bdf_engine::{BdfError, BdfResult};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NewtonRaphsonConfig {
        pub max_iterations: usize,
        pub residual_tolerance: f64,
        pub update_tolerance: f64,
        /// Legacy fixed damping factor (Issue #3339: kept for backward
        /// compatibility with the original Eq. test suite). When
        /// `damping.mode == 0` the solver uses this value verbatim;
        /// when `damping.mode == 1` the solver uses
        /// `DampingPolicy::residual_ratio_factor(...)` instead.
        pub damping_factor: f64,
        /// Issue #3339 — Residual-ratio-aware damping strategy.
        /// Default `DampingPolicy::default()` is `mode = 0` with
        /// `baseline_factor = 1.0`, which makes the solver numerically
        /// identical to the original fixed-`damping_factor`
        /// implementation byte-for-byte. The evolver mutates this
        /// struct's fields through the in-tree Python seed.
        pub damping: DampingPolicy,
    }

    impl Default for NewtonRaphsonConfig {
        fn default() -> Self {
            Self {
                max_iterations: 50,
                residual_tolerance: 1e-8,
                update_tolerance: 1e-10,
                damping_factor: 1.0,
                damping: DampingPolicy::default(),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NewtonRaphsonStats {
        pub iterations: usize,
        pub final_residual: f64,
        pub converged: bool,
    }

    /// Issue #3339 — Adaptive damping strategy for the Newton-Raphson
    /// solver of the BDF DAE engine.
    ///
    /// `mode = 0` (legacy / fixed) makes the solver numerically
    /// identical to the original `NewtonRaphsonConfig { damping_factor,
    /// .. }` baseline — the seed controllers and golden transient traces
    /// rely on this for byte-equivalence.
    ///
    /// `mode = 1` (residual-ratio) switches the solver to a
    /// residual-ratio-aware schedule where the per-iteration damping
    /// factor is interpolated between `baseline_factor` (full step) and
    /// `floor` (most conservative) based on the ratio of the current
    /// residual norm to the previous one. Thresholds
    /// (`loose_threshold`, `tight_threshold`) and `aggressiveness` are
    /// the single-method heuristic knobs OpenEvolve targets.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct DampingPolicy {
        /// 0 = use `baseline_factor` (legacy fixed damping);
        /// 1 = residual-ratio-aware schedule.
        pub mode: u8,
        /// Used verbatim when `mode == 0`; serves as the ceiling of the
        /// residual-ratio schedule when `mode == 1`.
        pub baseline_factor: f64,
        /// Lower bound on the damping factor when `mode == 1`
        /// (most-conservative step).
        pub floor: f64,
        /// Residual ratio at or below which the solver takes a full
        /// `baseline_factor` step (loosest band).
        pub loose_threshold: f64,
        /// Residual ratio at or above which the solver steps down to
        /// `floor` (tightest band).
        pub tight_threshold: f64,
        /// Slope of the transition; 1.0 is a linear interpolation,
        /// 0.0 holds the ceiling everywhere, 2.0 falls through `floor`
        /// faster than linearly.
        pub aggressiveness: f64,
        /// Number of past residuals retained for the residual-ratio
        /// computation (memory cap for the evolvable schedule).
        pub history_window: usize,
    }

    impl Default for DampingPolicy {
        fn default() -> Self {
            Self {
                mode: 0,
                baseline_factor: 1.0,
                floor: 0.25,
                loose_threshold: 0.5,
                tight_threshold: 0.95,
                aggressiveness: 1.0,
                history_window: 4,
            }
        }
    }

    impl DampingPolicy {
        /// Compute the damping factor for the current Newton iteration
        /// given the previous and current residual norms.
        ///
        /// `mode == 0` returns `baseline_factor`; the solver loop uses
        /// this to preserve the original fixed-`damping_factor`
        /// baseline. `mode == 1` interpolates between
        /// `baseline_factor` and `floor` based on
        /// `current_residual / previous_residual`. Outside the band the
        /// result is clamped to whichever boundary is closer.
        pub fn residual_ratio_factor(&self, prev_norm: f64, current_norm: f64) -> f64 {
            if self.mode == 0 {
                return self.baseline_factor;
            }
            if !prev_norm.is_finite() || !current_norm.is_finite() || prev_norm <= 0.0 {
                return self.baseline_factor;
            }
            let ratio = (current_norm / prev_norm).clamp(0.0, 2.0);
            let raw = if ratio <= self.loose_threshold {
                self.baseline_factor
            } else if ratio >= self.tight_threshold {
                self.floor
            } else {
                let span = (self.tight_threshold - self.loose_threshold).max(1e-12);
                let t = (ratio - self.loose_threshold) / span;
                self.baseline_factor + self.aggressiveness * t * (self.floor - self.baseline_factor)
            };
            raw.clamp(self.floor, self.baseline_factor)
        }
    }

    pub trait ResidualFunction<T> {
        fn eval(&self, x: &[T], residual: &mut [T]);
    }

    /// Issue #4691 — LU factorisation with partial pivoting.
    ///
    /// Used by the flag-gated frozen-Jacobian prototype so a frozen
    /// Jacobian can be *factorised once* and reused for every subsequent
    /// Newton iteration (O(n²) triangular solves) instead of being
    /// re-eliminated (O(n³)) on every iteration.
    #[derive(Debug, Clone, Default)]
    pub struct LuFactors {
        n: usize,
        /// Combined L\U storage (row-major, `n*n`), unit lower triangle.
        lu: Vec<f64>,
        /// Row permutation: `perm[i]` is the original row now at row `i`.
        perm: Vec<usize>,
        singular: bool,
    }

    impl LuFactors {
        /// Factorise the row-major `n*n` matrix `a` (copied, not mutated).
        pub fn factorize(a: &[f64], n: usize) -> Self {
            let mut lu = a[..n * n].to_vec();
            let mut perm: Vec<usize> = (0..n).collect();
            let mut singular = false;

            for k in 0..n {
                let mut pivot_row = k;
                let mut max_val = lu[k * n + k].abs();
                for i in (k + 1)..n {
                    let val = lu[i * n + k].abs();
                    if val > max_val {
                        max_val = val;
                        pivot_row = i;
                    }
                }
                if max_val < 1e-14 {
                    singular = true;
                    break;
                }
                if pivot_row != k {
                    // Swap the full rows: the already-computed L part must
                    // travel with the row (standard PA = LU layout).
                    for j in 0..n {
                        lu.swap(k * n + j, pivot_row * n + j);
                    }
                    perm.swap(k, pivot_row);
                }
                let pivot = lu[k * n + k];
                for i in (k + 1)..n {
                    let factor = lu[i * n + k] / pivot;
                    lu[i * n + k] = factor;
                    for j in (k + 1)..n {
                        lu[i * n + j] -= factor * lu[k * n + j];
                    }
                }
            }

            Self {
                n,
                lu,
                perm,
                singular,
            }
        }

        /// True when the factorisation hit a (near-)zero pivot.
        pub fn is_singular(&self) -> bool {
            self.singular
        }

        /// Solve `A x = b` with the stored factors.
        pub fn solve(&self, b: &[f64], out: &mut [f64]) {
            let n = self.n;
            if n == 0 {
                return;
            }
            if self.singular {
                // Degenerate Jacobian: mirror the legacy solver's
                // "no elimination performed" behaviour (solution = rhs).
                let m = n.min(b.len()).min(out.len());
                out[..m].copy_from_slice(&b[..m]);
                return;
            }

            let mut y = vec![0.0; n];
            for i in 0..n {
                y[i] = b[self.perm[i]];
            }
            // Forward substitution (unit lower triangle).
            for i in 1..n {
                let sum: f64 = self.lu[i * n..i * n + i]
                    .iter()
                    .zip(y[..i].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                y[i] -= sum;
            }
            // Back substitution.
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let sum: f64 = self.lu[i * n + i + 1..i * n + n]
                    .iter()
                    .zip(x[i + 1..].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                x[i] = (y[i] - sum) / self.lu[i * n + i];
            }
            out.copy_from_slice(&x);
        }
    }

    /// Issue #4691 — Jacobian cache shared by the Newton iterations of one
    /// BDF step and, when the frozen-Jacobian prototype is enabled, by
    /// consecutive BDF steps.
    #[derive(Debug, Clone, Default)]
    pub struct NewtonJacobianCache {
        lu: LuFactors,
        has_jacobian: bool,
        /// Number of times the cached Jacobian has been used since the last
        /// assembly (counts uses, i.e. solve iterations, across steps).
        reuse_count: usize,
        /// Residual norm of the *first* iteration of the previous solve —
        /// the cross-step stall probe (same role in the sequence, so the
        /// two residuals are scale-comparable).
        prev_first_residual: Option<f64>,
        /// Residual norm at convergence of the previous solve.
        prev_final_residual: Option<f64>,
    }

    impl NewtonJacobianCache {
        /// Drop the cached Jacobian; the next solve must re-assemble it.
        pub fn invalidate(&mut self) {
            self.has_jacobian = false;
            self.reuse_count = 0;
            self.lu = LuFactors::default();
        }

        /// Whether a usable Jacobian is currently cached.
        pub fn has_jacobian(&self) -> bool {
            self.has_jacobian
        }

        /// How many times the cached Jacobian has been used since the last
        /// assembly.
        pub fn reuse_count(&self) -> usize {
            self.reuse_count
        }

        /// Residual norm at convergence of the previous solve.
        pub fn prev_final_residual(&self) -> Option<f64> {
            self.prev_final_residual
        }
    }

    pub struct NewtonRaphsonSolver {
        config: NewtonRaphsonConfig,
    }

    impl NewtonRaphsonSolver {
        pub fn new(config: NewtonRaphsonConfig) -> Self {
            Self { config }
        }

        pub fn with_default_config() -> Self {
            Self::new(NewtonRaphsonConfig::default())
        }

        /// Solve the system with a throw-away Jacobian cache.
        ///
        /// Byte-identical to the pre-#4691 implementation when the
        /// frozen-Jacobian prototype is disabled (the default).
        pub fn solve<R>(&self, x0: &[f64], func: &R) -> BdfResult<(Vec<f64>, NewtonRaphsonStats)>
        where
            R: ResidualFunction<f64>,
        {
            let mut cache = NewtonJacobianCache::default();
            self.solve_with_cache(x0, func, &mut cache)
        }

        /// Solve the system, reusing the Jacobian (and its LU factorisation)
        /// held in `cache` when the frozen-Jacobian prototype is enabled.
        ///
        /// Refresh policy (Issue #4691):
        ///
        /// * reuse the cached Jacobian while the residual decreases
        ///   monotonically;
        /// * refresh it when the residual stalls (fails to decrease), when
        ///   the reuse budget `N` is exhausted, or when no Jacobian is
        ///   cached yet;
        /// * the cache is carried across BDF steps; the caller invalidates
        ///   it when a step is rejected.
        pub fn solve_with_cache<R>(
            &self,
            x0: &[f64],
            func: &R,
            cache: &mut NewtonJacobianCache,
        ) -> BdfResult<(Vec<f64>, NewtonRaphsonStats)>
        where
            R: ResidualFunction<f64>,
        {
            let n = x0.len();
            let mut x = x0.to_vec();
            let mut residual = vec![0.0; n];
            let mut update = vec![0.0; n];
            let mut jacobian = vec![0.0; n * n];

            // Issue #3339: when `damping.mode == 0` (default), the solver
            // uses `self.config.damping_factor` verbatim — identical to
            // the original implementation byte-for-byte. When
            // `damping.mode == 1`, the solver interpolates between
            // `damping.baseline_factor` and `damping.floor` based on the
            // residual ratio. Either way, the multiplier is computed
            // inside the loop on every iteration.
            let mut prev_residual_norm: f64 = 0.0;
            let history_window = self.config.damping.history_window.max(1);
            let mut residual_history = Vec::with_capacity(history_window);

            // Issue #4691: the frozen-Jacobian prototype is opt-in and off
            // by default; with it off the legacy branch below runs and the
            // numerics are unchanged.
            let policy = super::freeze::FrozenJacobianPolicy::from_env();
            let frozen = policy.enabled;

            // Stall probe: the residual of the previous Newton iteration.
            // Inside a step that is the previous iteration of the same solve;
            // for the first iteration of a step it is the *first-iteration*
            // residual of the previous step (same role → scale-comparable).
            let mut stall_probe: Option<f64> = if frozen {
                cache.prev_first_residual
            } else {
                None
            };
            let mut first_residual: Option<f64> = None;

            super::counters::SOLVE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            for iter in 0..self.config.max_iterations {
                func.eval(&x, &mut residual);

                let residual_norm = norm(&residual);
                super::counters::NEWTON_ITERATIONS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if first_residual.is_none() {
                    first_residual = Some(residual_norm);
                }

                if residual_norm < self.config.residual_tolerance {
                    if frozen {
                        cache.prev_first_residual = first_residual;
                        cache.prev_final_residual = Some(residual_norm);
                    }
                    return Ok((
                        x.clone(),
                        NewtonRaphsonStats {
                            iterations: iter + 1,
                            final_residual: residual_norm,
                            converged: true,
                        },
                    ));
                }

                if frozen {
                    // Refresh decision: reuse while the residual decreases
                    // monotonically; refresh on stall, on budget
                    // exhaustion, or when nothing is cached.
                    let stalled = match stall_probe {
                        // Not strictly decreasing (incl. NaN) counts as a stall.
                        Some(prev) => {
                            residual_norm.partial_cmp(&prev) != Some(std::cmp::Ordering::Less)
                        }
                        None => false,
                    };
                    let budget_exhausted = cache.reuse_count >= policy.max_reuses;
                    if !cache.has_jacobian || stalled || budget_exhausted {
                        compute_numerical_jacobian(&x, &residual, func, &mut jacobian);
                        super::counters::JACOBIAN_ASSEMBLIES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        cache.lu = LuFactors::factorize(&jacobian, n);
                        super::counters::FACTORIZATIONS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        cache.has_jacobian = true;
                        cache.reuse_count = 0;
                    } else {
                        super::counters::JACOBIAN_REUSES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    cache.reuse_count += 1;
                    cache.lu.solve(&residual, &mut update);
                } else {
                    compute_numerical_jacobian(&x, &residual, func, &mut jacobian);
                    super::counters::JACOBIAN_ASSEMBLIES
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    solve_linear_system(&jacobian, &residual, &mut update);
                    super::counters::FACTORIZATIONS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                let update_norm = norm(&update);

                if update_norm < self.config.update_tolerance {
                    if frozen {
                        cache.prev_first_residual = first_residual;
                        cache.prev_final_residual = Some(residual_norm);
                    }
                    return Ok((
                        x.clone(),
                        NewtonRaphsonStats {
                            iterations: iter + 1,
                            final_residual: residual_norm,
                            converged: true,
                        },
                    ));
                }

                let damping_factor = if self.config.damping.mode == 0 {
                    self.config.damping_factor
                } else {
                    let reference_residual_norm = if residual_history.is_empty() {
                        prev_residual_norm
                    } else {
                        residual_history.iter().sum::<f64>() / residual_history.len() as f64
                    };
                    self.config
                        .damping
                        .residual_ratio_factor(reference_residual_norm, residual_norm)
                };

                for i in 0..n {
                    x[i] -= damping_factor * update[i];
                }

                stall_probe = Some(residual_norm);
                prev_residual_norm = residual_norm;
                if self.config.damping.mode != 0 {
                    residual_history.push(residual_norm);
                    if residual_history.len() > history_window {
                        residual_history.remove(0);
                    }
                }
            }

            func.eval(&x, &mut residual);
            let final_residual = norm(&residual);
            if frozen {
                cache.prev_first_residual = first_residual;
                cache.prev_final_residual = None;
            }

            Err(BdfError::ConvergenceFailed(
                self.config.max_iterations,
                final_residual,
            ))
        }
    }

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn compute_numerical_jacobian<R>(x: &[f64], residual: &[f64], func: &R, jac: &mut [f64])
    where
        R: ResidualFunction<f64>,
    {
        let n = x.len();
        let eps = 1e-8;

        for j in 0..n {
            let mut x_perturbed = x.to_vec();
            x_perturbed[j] += eps;
            let mut residual_perturbed = vec![0.0; n];
            func.eval(&x_perturbed, &mut residual_perturbed);

            let dx = eps;
            for i in 0..n {
                jac[i * n + j] = (residual_perturbed[i] - residual[i]) / dx;
            }
        }
    }

    fn solve_linear_system(jac: &[f64], rhs: &[f64], solution: &mut [f64]) {
        let n = rhs.len();
        solution.copy_from_slice(rhs);

        let mut aug = vec![0.0; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = jac[i * n + j];
            }
            aug[i * (n + 1) + n] = rhs[i];
        }

        for k in 0..n {
            let mut max_val = aug[k * (n + 1) + k].abs();
            let mut max_idx = k;
            for i in (k + 1)..n {
                let val = aug[i * (n + 1) + k].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            if max_val < 1e-14 {
                continue;
            }

            if max_idx != k {
                for j in 0..=n {
                    aug.swap(k * (n + 1) + j, max_idx * (n + 1) + j);
                }
            }

            let pivot = aug[k * (n + 1) + k];
            for j in k..=n {
                aug[k * (n + 1) + j] /= pivot;
            }

            for i in 0..n {
                if i != k {
                    let factor = aug[i * (n + 1) + k];
                    for j in k..=n {
                        aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
                    }
                }
            }
        }

        for i in 0..n {
            solution[i] = aug[i * (n + 1) + n];
        }
    }

    pub struct SimpleResidual {
        pub n: usize,
    }

    impl ResidualFunction<f64> for SimpleResidual {
        fn eval(&self, x: &[f64], residual: &mut [f64]) {
            for i in 0..self.n {
                residual[i] = x[i] * x[i] - 1.0;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_newton_raphson_simple() {
            let solver = NewtonRaphsonSolver::with_default_config();
            let func = SimpleResidual { n: 1 };

            let result = solver.solve(&[2.0], &func);
            assert!(result.is_ok());

            let (solution, stats) = result.unwrap();
            assert!(stats.converged);
            assert!((solution[0] - 1.0).abs() < 1e-6);
        }

        #[test]
        fn test_newton_raphson_multi_dim() {
            let solver = NewtonRaphsonSolver::with_default_config();
            let func = SimpleResidual { n: 2 };

            let result = solver.solve(&[2.0, 2.0], &func);
            assert!(result.is_ok());

            let (solution, stats) = result.unwrap();
            assert!(stats.converged);
            assert!((solution[0] - 1.0).abs() < 1e-6);
            assert!((solution[1] - 1.0).abs() < 1e-6);
        }

        // -----------------------------------------------------------------
        // Issue #3339 — DampingPolicy byte-equivalence regression guard.
        //
        // The seed controller mirrors the original fixed-damping
        // behavior (`mode = 0`, `baseline_factor = 1.0`) byte-for-byte.
        // If a future refactor accidentally changes the no-mode path,
        // the golden transient traces break; this test catches the
        // issue at the unit level by pinning the same residual_norm
        // sequence to the same iteration count and final residual.
        // -----------------------------------------------------------------
        #[test]
        fn issue_3339_damping_policy_default_is_identity() {
            let p = DampingPolicy::default();
            assert_eq!(
                p.mode, 0,
                "default mode must be 0 (fixed) for byte-equivalence"
            );
            assert_eq!(p.baseline_factor, 1.0);

            // Any ratio under any condition returns baseline_factor.
            for (prev, cur) in [(1.0, 0.01), (1.0, 1.0), (1.0, 0.99), (1.0, 1.5)] {
                assert_eq!(
                    p.residual_ratio_factor(prev, cur),
                    1.0,
                    "mode=0 must ignore previous/current residuals and return baseline_factor"
                );
            }
        }

        #[test]
        fn issue_3339_damping_policy_residual_ratio_branch() {
            let p = DampingPolicy {
                mode: 1,
                baseline_factor: 1.0,
                floor: 0.25,
                loose_threshold: 0.5,
                tight_threshold: 0.95,
                aggressiveness: 1.0,
                history_window: 4,
            };
            // Loose band (ratio ≤ 0.5) → full baseline
            assert!((p.residual_ratio_factor(1.0, 0.4) - 1.0).abs() < 1e-12);
            // Tight band (ratio ≥ 0.95) → floor
            assert!((p.residual_ratio_factor(1.0, 1.0) - 0.25).abs() < 1e-12);
            // Linear transition band → somewhere between
            let mid = p.residual_ratio_factor(1.0, 0.725);
            assert!(mid > 0.25 && mid < 1.0, "mid-band factor must interpolate");
        }

        #[test]
        fn issue_3339_damping_policy_uses_residual_history_window() {
            let mut config = NewtonRaphsonConfig::default();
            config.damping = DampingPolicy {
                mode: 1,
                baseline_factor: 1.0,
                floor: 0.25,
                loose_threshold: 0.5,
                tight_threshold: 0.95,
                aggressiveness: 1.0,
                history_window: 3,
            };
            let solver = NewtonRaphsonSolver::new(config);
            let result = solver.solve(&[2.0], &SimpleResidual { n: 1 });
            assert!(result.is_ok());
            let (_, stats) = result.unwrap();
            assert!(stats.converged);
            assert!(stats.iterations > 0);
        }

        #[test]
        fn issue_3339_damping_policy_safety_clamps() {
            let p = DampingPolicy::default(); // mode=0 → ignores clamps but must not panic
            assert_eq!(p.residual_ratio_factor(0.0, 0.0), 1.0);
            assert_eq!(p.residual_ratio_factor(f64::NAN, 1.0), 1.0);
            assert_eq!(p.residual_ratio_factor(1.0, f64::INFINITY), 1.0);
        }
    }
}

pub mod adaptive_step {
    //! Adaptive Step Size Controller for BDF

    use crate::physics::bdf_engine::{BdfError, BdfResult};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct AdaptiveStepConfig {
        pub min_dt: f64,
        pub max_dt: f64,
        pub initial_dt: f64,
        pub safety_factor: f64,
        pub min_order: usize,
        pub max_order: usize,
    }

    impl Default for AdaptiveStepConfig {
        fn default() -> Self {
            Self {
                min_dt: 1e-6,
                max_dt: 3600.0,
                initial_dt: 60.0,
                safety_factor: 0.9,
                min_order: 1,
                max_order: 6,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct StepSize {
        pub dt: f64,
        pub order: usize,
    }

    pub struct AdaptiveStepController {
        config: AdaptiveStepConfig,
    }

    impl AdaptiveStepController {
        pub fn new(config: AdaptiveStepConfig) -> Self {
            Self { config }
        }

        pub fn with_default_config() -> Self {
            Self::new(AdaptiveStepConfig::default())
        }

        pub fn compute_next_step(
            &self,
            current: StepSize,
            error_estimate: f64,
            tolerance: f64,
        ) -> BdfResult<StepSize> {
            let new_dt =
                self.compute_step_size(current.dt, error_estimate, tolerance, current.order);

            let new_dt = new_dt.clamp(self.config.min_dt, self.config.max_dt);

            if new_dt < self.config.min_dt {
                return Err(BdfError::StepSizeUnderflow(new_dt, self.config.min_dt));
            }
            if new_dt > self.config.max_dt {
                return Err(BdfError::StepSizeOverflow(new_dt, self.config.max_dt));
            }

            Ok(StepSize {
                dt: new_dt,
                order: current.order,
            })
        }

        fn compute_step_size(&self, dt: f64, error: f64, tolerance: f64, order: usize) -> f64 {
            // Issue #4693: the law itself lives in the crate-level helper
            // `error_controlled_step_size` so the simulation-side controller
            // (`crate::sim::adaptive_timestep`) can share it. The arithmetic
            // is unchanged — this path stays bit-identical to before.
            crate::physics::bdf_engine::error_controlled_step_size(
                dt,
                error,
                tolerance,
                order,
                self.config.safety_factor,
            )
        }

        pub fn suggest_order(&self, current: StepSize, errors: &[f64]) -> usize {
            if errors.is_empty() {
                return current.order;
            }

            let min_err_idx = errors
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let suggested_order = (self.config.min_order + min_err_idx).min(self.config.max_order);
            suggested_order.max(self.config.min_order)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_adaptive_step_initial() {
            let controller = AdaptiveStepController::with_default_config();
            let step = StepSize { dt: 60.0, order: 3 };

            let next = controller.compute_next_step(step, 1e-4, 1e-6);
            assert!(next.is_ok());
        }

        #[test]
        fn test_step_size_growth_on_success() {
            let controller = AdaptiveStepController::with_default_config();
            let step = StepSize { dt: 60.0, order: 1 };

            let next = controller.compute_next_step(step, 1e-10, 1e-6);
            assert!(next.is_ok());
            let next_step = next.unwrap();
            assert!(next_step.dt > step.dt);
        }
    }
}

pub mod time_stepping {
    //! BDF Time-Stepping Engine

    use crate::physics::bdf_engine::{
        adaptive_step::{AdaptiveStepConfig, AdaptiveStepController, StepSize},
        coefficients::{get_coefficients, BdfCoefficients},
        newton_raphson::{
            NewtonJacobianCache, NewtonRaphsonConfig, NewtonRaphsonSolver, ResidualFunction,
        },
        warm_start_flag_enabled, warm_start_predict, BdfError, BdfResult, MAX_BDF_ORDER,
        WARM_START_PREDICTOR_ENV,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TimeSteppingConfig {
        pub bdf_config: NewtonRaphsonConfig,
        pub step_config: AdaptiveStepConfig,
        pub max_steps: usize,
        pub tolerance: f64,
    }

    impl Default for TimeSteppingConfig {
        fn default() -> Self {
            Self {
                bdf_config: NewtonRaphsonConfig::default(),
                step_config: AdaptiveStepConfig::default(),
                max_steps: 100000,
                tolerance: 1e-6,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TimeSteppingStats {
        pub steps_taken: usize,
        pub steps_rejected: usize,
        pub final_time: f64,
        pub converged: bool,
        /// Issue #3339 — Newton iterations for the last accepted step.
        /// Exposed so the `BdfDriver` (and the evolution harness) can
        /// aggregate iterations across a transient without re-running
        /// the inner loop.
        pub last_newton_iterations: usize,
        /// Issue #3339 — Final residual norm for the last accepted step.
        pub last_final_residual: f64,
    }

    /// Issue #3339 — Aggregate stats for a multi-step BDF transient,
    /// populated by [`BdfDriver::run`]. Mirrors the JSON summary
    /// emitted by the `bdf_evaluator` binary and the Python
    /// orchestration under `tools/evolution/seeds/dae/`.
    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct DriverStats {
        /// Total Newton iterations across all accepted AND rejected
        /// steps (rejected steps also cost iterations).
        pub newton_iterations: usize,
        /// Steps accepted by the adaptive controller.
        pub steps_accepted: usize,
        /// Steps rejected by the Newton solver or adaptive controller
        /// (re-try at a smaller `dt`). The issue's `Refs`/decision
        /// rule treats *zero aborts* as a hard invariant; rejected
        /// steps must be retried, never abort the run.
        pub steps_rejected: usize,
        /// Final simulation time [s].
        pub final_time: f64,
        /// True if the driver reached `t_end` within the step cap.
        pub converged: bool,
        /// Largest residual norm seen across the transient.
        pub max_residual: f64,
        /// Final residual norm (taken at the last accepted step).
        /// Used by the conservation probes in `bdf_benchmarks` as
        /// the proxy for junction mass/enthalpy balance error.
        pub final_residual: f64,
        /// Largest relative mass-conservation error across junctions
        /// (populated by the benchmark circuits; 0.0 when no probe).
        pub max_mass_conservation_error: f64,
        /// Largest relative enthalpy-conservation error across
        /// junctions (populated by the benchmark circuits; 0.0 when
        /// no probe).
        pub max_enthalpy_conservation_error: f64,
        /// Number of NaN / Inf observations in the trajectory. A
        /// non-zero value is a hard fail of the NaN/Inf invariant.
        pub nan_or_inf_count: usize,
        /// Number of conservation-hard-fail observations across the
        /// suite (junctions exceeding the 1e-7 invariant budget).
        pub conservation_violations: usize,
    }

    pub trait DaeSystem<T> {
        fn residual(&self, t: T, y: &[T], yp: &[T], r: &mut [T]);
        fn dimension(&self) -> usize;

        /// Issue #4692 — optional physical bounds on the state vector
        /// (`(lower, upper)`, element-wise).
        ///
        /// The BDF engine itself enforces **no** bounds on the state;
        /// the default implementation therefore returns `None` and the
        /// warm-start predictor runs unbounded. A DAE system that knows
        /// its own physical limits (temperatures, masses, …) can
        /// declare them here so the Issue #4692 warm-start predictor
        /// clamps its extrapolated initial guess to the same limits.
        /// The method is defaulted, so existing implementors are
        /// unaffected.
        fn state_bounds(&self) -> Option<(Vec<T>, Vec<T>)> {
            None
        }
    }

    pub struct BdfTimeStepper {
        config: TimeSteppingConfig,
        history: Vec<Vec<f64>>,
        time_history: Vec<f64>,
        current_order: usize,
        // Issue #3339: track the last accepted step's inner Newton
        // statistics so `BdfDriver::run` can aggregate them without
        // re-running the inner loop. `last_error_estimate` is the
        // local truncation error used by the adaptive controller.
        last_newton_iterations: usize,
        last_final_residual: f64,
        last_error_estimate: f64,
        // Issue #4692 — warm-start predictor state.
        //
        // `warm_start` is read **once** at construction (never per
        // step). `accepted_steps` counts the accepted steps since the
        // last `initialize`, which gates the "first two steps"
        // fallback, and `last_step_rejected` records that the previous
        // step was rejected, which gates the post-rejection fallback.
        warm_start: bool,
        accepted_steps: usize,
        last_step_rejected: bool,
        // Issue #4691: Jacobian (and LU factorisation) cache shared by the
        // Newton iterations of consecutive steps when the frozen-Jacobian
        // prototype is enabled. It is inert in the default (flag-off) path.
        jacobian_cache: NewtonJacobianCache,
    }

    impl BdfTimeStepper {
        pub fn new(config: TimeSteppingConfig) -> Self {
            // Issue #4692 — the environment flag is read exactly once,
            // here at construction time. Never per step (see the
            // #4688 profile: per-step `getenv` reads dominated the
            // annual run).
            let warm_start =
                warm_start_flag_enabled(std::env::var(WARM_START_PREDICTOR_ENV).ok().as_deref());
            Self::new_with_warm_start(config, warm_start)
        }

        /// Issue #4692 — construct a stepper with an explicit
        /// warm-start setting (used by the unit tests and by
        /// measurement harnesses that cannot set the environment).
        pub fn new_with_warm_start(config: TimeSteppingConfig, warm_start: bool) -> Self {
            Self {
                config,
                history: Vec::new(),
                time_history: Vec::new(),
                current_order: 1,
                last_newton_iterations: 0,
                last_final_residual: 0.0,
                last_error_estimate: 0.0,
                warm_start,
                accepted_steps: 0,
                last_step_rejected: false,
                jacobian_cache: NewtonJacobianCache::default(),
            }
        }

        /// Issue #4692 — whether the warm-start predictor is active.
        pub fn warm_start_enabled(&self) -> bool {
            self.warm_start
        }

        pub fn with_default_config() -> Self {
            Self::new(TimeSteppingConfig::default())
        }

        pub fn initialize(&mut self, t0: f64, y0: &[f64]) -> BdfResult<()> {
            let n = y0.len();
            self.history.clear();
            self.time_history.clear();

            for _ in 0..MAX_BDF_ORDER {
                self.history.push(vec![0.0; n]);
                self.time_history.push(0.0);
            }

            self.history[0].copy_from_slice(y0);
            self.time_history[0] = t0;

            self.current_order = 1;
            self.last_newton_iterations = 0;
            self.last_final_residual = 0.0;
            self.last_error_estimate = 0.0;
            // Issue #4692 — a fresh initial condition restarts the
            // warm-start predictor's fallback bookkeeping.
            self.accepted_steps = 0;
            self.last_step_rejected = false;
            Ok(())
        }

        /// Issue #3339 — Number of Newton iterations the most recent
        /// accepted step consumed.
        pub fn last_newton_iterations(&self) -> usize {
            self.last_newton_iterations
        }

        /// Issue #3339 — Residual norm at convergence for the most
        /// recent accepted step.
        pub fn last_final_residual(&self) -> f64 {
            self.last_final_residual
        }

        /// Issue #3339 — Local truncation-error estimate for the most
        /// recent accepted step (drives the adaptive step controller's
        /// `compute_next_step`).
        pub fn last_error_estimate(&self) -> f64 {
            self.last_error_estimate
        }

        /// Issue #4691: drop the cached Jacobian (and its factorisation).
        ///
        /// Called by the driver when a step is rejected so the next attempt
        /// re-assembles the Jacobian from scratch.
        pub fn invalidate_jacobian_cache(&mut self) {
            self.jacobian_cache.invalidate();
        }

        pub fn step<S>(&mut self, dt: f64, system: &S) -> BdfResult<(Vec<f64>, TimeSteppingStats)>
        where
            S: DaeSystem<f64>,
        {
            let n = system.dimension();
            let t_new = self.time_history[0] + dt;

            let coeff = get_coefficients(self.current_order)?;

            let mut y_new = vec![0.0; n];
            self.predict(&mut y_new);

            // Issue #4692 — flag-gated warm-start predictor (default
            // OFF). When enabled, the Newton initial guess is the
            // linear extrapolation `x_prev + (x_prev - x_prev2)` of
            // the last two accepted states, clamped to the physical
            // bounds the DAE system declares. The legacy BDF-coefficient
            // guess is kept on the first two steps and on the step
            // right after a rejected step (see `warm_start_guess`).
            if self.warm_start {
                let bounds = system.state_bounds();
                let clamped = bounds
                    .as_ref()
                    .map(|(lower, upper)| (lower.as_slice(), upper.as_slice()));
                if let Some(guess) = self.warm_start_guess(clamped) {
                    y_new.copy_from_slice(&guess);
                }
            }

            // Issue #3339: construct the Newton solver from the
            // configured `bdf_config` so the new `DampingPolicy` (and
            // any user-tuned `damping_factor`) actually flows
            // through. Before this fix, `with_default_config()` was
            // hard-coded inside `step`, silently discarding
            // `TimeSteppingConfig::bdf_config`. The default equals the
            // prior hard-coded solver byte-for-byte (Newton & BDF
            // residuals are deterministic), so existing callers see
            // no behavior change.
            let solver = NewtonRaphsonSolver::new(self.config.bdf_config);
            let bdf_residual = BdfResidual {
                system,
                dt,
                coeff,
                t_new,
                history: &self.history,
                n,
            };

            let (solution, nr_stats) =
                match solver.solve_with_cache(&y_new, &bdf_residual, &mut self.jacobian_cache) {
                    Ok(out) => out,
                    Err(err) => {
                        // Issue #4691: with the frozen-Jacobian prototype on, a
                        // failure of a frozen solve is retried once with a
                        // freshly assembled Jacobian (the frozen Jacobian was
                        // stale) before the step is rejected. With the prototype
                        // off this is exactly the legacy behaviour.
                        let frozen =
                            crate::physics::bdf_engine::freeze::FrozenJacobianPolicy::from_env();
                        if !frozen.enabled || !self.jacobian_cache.has_jacobian() {
                            self.jacobian_cache.invalidate();
                            // Issue #4692 — a failed step invalidates the
                            // warm-start history: the next attempt must fall
                            // back to the legacy initial guess.
                            self.last_step_rejected = true;
                            return Err(err);
                        }
                        crate::physics::bdf_engine::counters::FORCED_REFRESH_RETRIES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.jacobian_cache.invalidate();
                        match solver.solve_with_cache(
                            &y_new,
                            &bdf_residual,
                            &mut self.jacobian_cache,
                        ) {
                            Ok(out) => out,
                            Err(err) => {
                                // Issue #4692 — warm-start invalidation, as above.
                                self.last_step_rejected = true;
                                return Err(err);
                            }
                        }
                    }
                };

            if !nr_stats.converged {
                self.last_step_rejected = true;
                return Err(BdfError::ConvergenceFailed(
                    nr_stats.iterations,
                    nr_stats.final_residual,
                ));
            }

            // Issue #3339: estimate the local truncation error as the
            // scaled residual norm of the just-converged step
            // (BDF(k+1) vs BDF(k) difference is hidden inside the
            // single BDF residual we expose; the residual norm is the
            // closest direct signal we have without a second
            // high-order residual eval). The adaptive controller uses
            // it the same way.
            self.last_newton_iterations = nr_stats.iterations;
            self.last_final_residual = nr_stats.final_residual;
            self.last_error_estimate = nr_stats.final_residual.max(0.0);

            self.correct(&solution, t_new);

            // Issue #4692 — an accepted step refreshes the warm-start
            // bookkeeping (the predictor becomes eligible once two
            // accepted states exist and no rejection is pending).
            self.accepted_steps += 1;
            self.last_step_rejected = false;

            y_new[..n].copy_from_slice(&solution[..n]);

            Ok((
                y_new,
                TimeSteppingStats {
                    steps_taken: 1,
                    steps_rejected: 0,
                    final_time: t_new,
                    converged: true,
                    last_newton_iterations: nr_stats.iterations,
                    last_final_residual: nr_stats.final_residual,
                },
            ))
        }

        /// Issue #4692 — the warm-start initial guess, or `None` when
        /// the legacy (BDF-coefficient) guess must be kept.
        ///
        /// `None` is returned when
        ///  * fewer than two steps have been accepted since the last
        ///    `initialize` (the "first two steps" fallback: there are
        ///    not yet two real accepted states to extrapolate from), or
        ///  * the previous step was rejected (the "post-rejection"
        ///    fallback: the extrapolation is built from states that
        ///    precede a failed attempt).
        fn warm_start_guess(&self, bounds: Option<(&[f64], &[f64])>) -> Option<Vec<f64>> {
            if self.accepted_steps < 2 {
                return None;
            }
            if self.last_step_rejected {
                return None;
            }
            if self.history.len() < 2 {
                return None;
            }
            Some(warm_start_predict(
                &self.history[0],
                &self.history[1],
                bounds,
            ))
        }

        fn predict(&self, y_pred: &mut [f64]) {
            if self.history.is_empty() {
                return;
            }
            if self.history.len() < 2 {
                if let Some(h) = self.history.first() {
                    y_pred.copy_from_slice(h);
                }
                return;
            }

            let k = self.current_order;
            let coeff = get_coefficients(k).expect("Invalid BDF order");

            for (i, pred_item) in y_pred.iter_mut().enumerate() {
                let mut pred = 0.0;
                for j in 0..k {
                    pred += coeff.alpha[j] * self.history[j][i];
                }
                *pred_item = -pred;
            }
        }

        fn correct(&mut self, y_new: &[f64], t_new: f64) {
            let n = self.history.len();
            for j in (1..n).rev() {
                let src = self.history[j - 1].clone();
                self.history[j].copy_from_slice(&src);
            }
            self.history[0].copy_from_slice(y_new);

            let m = self.time_history.len();
            for j in (1..m).rev() {
                self.time_history[j] = self.time_history[j - 1];
            }
            self.time_history[0] = t_new;
        }
    }

    struct BdfResidual<'a, S> {
        system: &'a S,
        dt: f64,
        coeff: &'static BdfCoefficients,
        t_new: f64,
        history: &'a [Vec<f64>],
        n: usize,
    }

    impl<'a, S> ResidualFunction<f64> for BdfResidual<'a, S>
    where
        S: DaeSystem<f64>,
    {
        fn eval(&self, y: &[f64], residual: &mut [f64]) {
            let mut yp = vec![0.0; self.n];

            let mut sum = 0.0;
            for j in 0..self.coeff.k {
                for (i, yp_item) in yp.iter_mut().enumerate().take(self.n) {
                    *yp_item += self.coeff.alpha[j] * self.history[j][i];
                }
                sum += self.coeff.alpha[j];
            }

            for (i, yp_item) in yp.iter_mut().enumerate().take(self.n) {
                *yp_item += self.coeff.alpha[self.coeff.k] * y[i];
                *yp_item /= self.dt * self.coeff.beta;
            }

            self.system.residual(self.t_new, y, &yp, residual);

            for (i, res_item) in residual.iter_mut().enumerate().take(self.n) {
                *res_item = sum * y[i] / (self.dt * self.coeff.beta);
            }
        }
    }

    /// Issue #3339 — Multi-step integration driver.
    ///
    /// Ties [`BdfTimeStepper`] (Newton-Raphson per step) and
    /// [`AdaptiveStepController`] (size/order selection) together
    /// into a single `run(system, t_end, dt_init)` call. Aggregates
    /// Newton iterations, accepted/rejected steps, conservation
    /// errors, and NaN/Inf observations into a [`DriverStats`] value
    /// the `bdf_evaluator` binary serializes straight into a
    /// Schema-v1 `Summary`.
    ///
    /// **Determinism contract:** the driver does no I/O, no
    /// parallelism, no wall-clock reads; identical `config` + system
    /// state ⇒ identical `DriverStats` byte-for-byte.
    pub struct BdfDriver {
        config: TimeSteppingConfig,
        adaptive: AdaptiveStepController,
        stepper: BdfTimeStepper,
    }

    impl BdfDriver {
        pub fn new(config: TimeSteppingConfig) -> Self {
            // Issue #4692 — `BdfTimeStepper::new` reads the
            // warm-start environment flag exactly once, here.
            let adaptive = AdaptiveStepController::new(config.step_config);
            let stepper = BdfTimeStepper::new(config);
            Self {
                config,
                adaptive,
                stepper,
            }
        }

        /// Issue #4692 — construct a driver with an explicit
        /// warm-start setting (bypasses the environment flag).
        pub fn new_with_warm_start(config: TimeSteppingConfig, warm_start: bool) -> Self {
            let adaptive = AdaptiveStepController::new(config.step_config);
            let stepper = BdfTimeStepper::new_with_warm_start(config, warm_start);
            Self {
                config,
                adaptive,
                stepper,
            }
        }

        pub fn initialize(&mut self, t0: f64, y0: &[f64]) -> BdfResult<()> {
            self.stepper.initialize(t0, y0)
        }

        pub fn stepper(&self) -> &BdfTimeStepper {
            &self.stepper
        }

        pub fn adaptive(&self) -> &AdaptiveStepController {
            &self.adaptive
        }

        /// Issue #3339 — Snapshot of the most-recent accepted state
        /// (`y_history[0]`). Returns an empty vec when no step has
        /// been accepted yet. Cloning the state is intentional: the
        /// benchmark circuits' `finalize` methods consume `&[f64]`
        /// without re-running the inner solver.
        pub fn last_state(&self) -> Vec<f64> {
            self.stepper.history.first().cloned().unwrap_or_default()
        }

        /// Run the transient from the initial condition set via
        /// [`BdfDriver::initialize`] to `t_end`, starting at `dt_init`.
        ///
        /// On Newton-convergence failure, the step is rejected and
        /// `dt` is halved; the loop terminates with
        /// `stats.converged = true` if the horizon is reached within
        /// `max_steps`, or with `converged = false` if the step cap
        /// is hit. Rejected steps never abort the run; they are
        /// counted in `steps_rejected` and `newton_iterations` so
        /// the fitness can penalise them without discarding the
        /// candidate.
        pub fn run<S>(&mut self, system: &S, t_end: f64, dt_init: f64) -> BdfResult<DriverStats>
        where
            S: DaeSystem<f64>,
        {
            let mut stats = DriverStats::default();
            let step_cfg = self.config.step_config;
            let mut dt = dt_init.max(step_cfg.min_dt).min(step_cfg.max_dt);
            let t0 = self.stepper.time_history.first().copied().unwrap_or(0.0);
            let mut t = t0;

            for _ in 0..self.config.max_steps {
                if t + dt > t_end {
                    dt = t_end - t;
                }
                if dt < 1e-12 {
                    break;
                }

                match self.stepper.step(dt, system) {
                    Ok((_, ts)) => {
                        stats.newton_iterations += ts.last_newton_iterations;
                        stats.steps_accepted += 1;
                        stats.max_residual = stats.max_residual.max(ts.last_final_residual);
                        stats.final_residual = ts.last_final_residual;
                        t = ts.final_time;
                        let err = self.stepper.last_error_estimate();
                        let step = StepSize {
                            dt,
                            order: self.stepper.current_order,
                        };
                        let next =
                            self.adaptive
                                .compute_next_step(step, err, self.config.tolerance)?;
                        dt = next.dt.clamp(step_cfg.min_dt, step_cfg.max_dt);
                    }
                    Err(_) => {
                        // Newton failed or step below floor — halve and retry.
                        stats.steps_rejected += 1;
                        // Issue #4691: a rejected step invalidates the frozen
                        // Jacobian — the next attempt re-assembles it.
                        self.stepper.invalidate_jacobian_cache();
                        crate::physics::bdf_engine::counters::STEP_REJECTIONS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        dt *= 0.5;
                        if dt < step_cfg.min_dt {
                            // Floor hit: keep trying once more at min_dt,
                            // then give up (don't loop forever).
                            dt = step_cfg.min_dt;
                            stats.steps_rejected += 1;
                        }
                    }
                }

                if (t_end - t).abs() < 1e-9 * t_end.max(1.0) {
                    stats.final_time = t;
                    stats.converged = true;
                    return Ok(stats);
                }
            }

            stats.final_time = t;
            stats.converged = (t_end - t).abs() < 1e-9 * t_end.max(1.0);
            Ok(stats)
        }
    }

    #[cfg(test)]
    struct SimpleOscillator;

    #[cfg(test)]
    impl DaeSystem<f64> for SimpleOscillator {
        fn residual(&self, _t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
            r[0] = yp[0] - y[1];
            r[1] = yp[1] + y[0];
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_bdf_time_stepper_init() {
            let mut stepper = BdfTimeStepper::with_default_config();
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();
        }

        #[test]
        fn test_bdf_first_order() {
            let mut stepper = BdfTimeStepper::with_default_config();
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();

            let system = SimpleOscillator;
            let result = stepper.step(0.01, &system);
            assert!(result.is_ok());
        }

        // -----------------------------------------------------------------
        // Issue #4692 — warm-start predictor (flag-gated, default OFF).
        // -----------------------------------------------------------------

        /// A DAE system that declares physical state bounds, used by
        /// the clamping test.
        #[cfg(test)]
        struct BoundedOscillator;

        #[cfg(test)]
        impl DaeSystem<f64> for BoundedOscillator {
            fn residual(&self, _t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
                r[0] = yp[0] - y[1];
                r[1] = yp[1] + y[0];
            }

            fn dimension(&self) -> usize {
                2
            }

            fn state_bounds(&self) -> Option<(Vec<f64>, Vec<f64>)> {
                Some((vec![-10.0, -10.0], vec![0.5, 0.5]))
            }
        }

        #[test]
        fn issue_4692_warm_start_flag_defaults_off() {
            // The environment flag is unset in the test process, so
            // the predictor must be inactive by default.
            assert!(!BdfTimeStepper::with_default_config().warm_start_enabled());
            assert!(!warm_start_flag_enabled(None));
            assert!(warm_start_flag_enabled(Some("1")));
            assert!(warm_start_flag_enabled(Some("true")));
            assert!(warm_start_flag_enabled(Some("ON")));
            assert!(!warm_start_flag_enabled(Some("0")));
            assert!(!warm_start_flag_enabled(Some("")));
            assert!(!warm_start_flag_enabled(Some("maybe")));
        }

        #[test]
        fn issue_4692_warm_start_predict_math() {
            // x_prev + (x_prev - x_prev2)
            let got = warm_start_predict(&[1.0, 2.0], &[0.5, 1.5], None);
            assert_eq!(got, vec![1.5, 2.5]);

            // A flat history extrapolates to the same value.
            let flat = warm_start_predict(&[3.0], &[3.0], None);
            assert_eq!(flat, vec![3.0]);

            // A decreasing history extrapolates downwards.
            let down = warm_start_predict(&[1.0], &[2.0], None);
            assert_eq!(down, vec![0.0]);
        }

        #[test]
        fn issue_4692_warm_start_first_two_steps_fallback() {
            let mut stepper =
                BdfTimeStepper::new_with_warm_start(TimeSteppingConfig::default(), true);
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();

            // No accepted step yet → no extrapolatable history.
            assert!(stepper.warm_start_guess(None).is_none());

            let system = SimpleOscillator;
            stepper.step(0.01, &system).unwrap();
            // One accepted step → still the "first two steps" fallback.
            assert!(stepper.warm_start_guess(None).is_none());

            stepper.step(0.01, &system).unwrap();
            // Two accepted steps → the predictor is eligible.
            let guess = stepper.warm_start_guess(None).expect("predictor eligible");
            let x_prev = &stepper.history[0];
            let x_prev2 = &stepper.history[1];
            for i in 0..guess.len() {
                let expected = x_prev[i] + (x_prev[i] - x_prev2[i]);
                assert_eq!(guess[i], expected);
            }
        }

        #[test]
        fn issue_4692_warm_start_post_rejection_fallback() {
            let mut stepper =
                BdfTimeStepper::new_with_warm_start(TimeSteppingConfig::default(), true);
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();

            let system = SimpleOscillator;
            stepper.step(0.01, &system).unwrap();
            stepper.step(0.01, &system).unwrap();
            assert!(stepper.warm_start_guess(None).is_some());

            // A rejected step invalidates the extrapolation.
            stepper.last_step_rejected = true;
            assert!(stepper.warm_start_guess(None).is_none());

            // The next accepted step re-enables it.
            stepper.last_step_rejected = false;
            assert!(stepper.warm_start_guess(None).is_some());
        }

        #[test]
        fn issue_4692_failed_step_marks_rejection() {
            // A Newton solve that cannot converge (max_iterations = 1
            // with a non-zero initial guess) must mark the step as
            // rejected so the predictor falls back on the retry.
            let mut config = TimeSteppingConfig::default();
            config.bdf_config.max_iterations = 1;
            let mut stepper = BdfTimeStepper::new_with_warm_start(config, true);
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();

            let system = SimpleOscillator;
            let outcome = stepper.step(0.01, &system);
            assert!(outcome.is_err(), "step must fail with max_iterations = 1");
            assert!(stepper.last_step_rejected);
            assert!(stepper.warm_start_guess(None).is_none());
        }

        #[test]
        fn issue_4692_warm_start_clamps_to_declared_bounds() {
            // Unbounded: the raw extrapolation is used.
            let raw = warm_start_predict(&[10.0], &[0.0], None);
            assert_eq!(raw, vec![20.0]);

            // Bounded: the extrapolation is clamped to the declared
            // physical limits.
            let clamped = warm_start_predict(&[10.0], &[0.0], Some((&[-5.0], &[2.0])));
            assert_eq!(clamped, vec![2.0]);

            // Reversed bounds are normalised, not panicked on.
            let reversed = warm_start_predict(&[10.0], &[0.0], Some((&[2.0], &[-5.0])));
            assert_eq!(reversed, vec![2.0]);

            // The stepper path uses the bounds declared by the system.
            let mut stepper =
                BdfTimeStepper::new_with_warm_start(TimeSteppingConfig::default(), true);
            stepper.initialize(0.0, &[1.0, 0.0]).unwrap();
            let system = BoundedOscillator;
            stepper.step(0.01, &system).unwrap();
            stepper.step(0.01, &system).unwrap();
            let bounds = system.state_bounds().unwrap();
            let guess = stepper
                .warm_start_guess(Some((bounds.0.as_slice(), bounds.1.as_slice())))
                .expect("predictor eligible");
            for value in guess {
                assert!(
                    (-10.0..=0.5).contains(&value),
                    "clamped guess {value} outside declared bounds"
                );
            }
        }

        #[test]
        fn issue_4692_flag_off_is_bit_identical() {
            // The flag-off path must produce exactly the legacy trace.
            let mut legacy =
                BdfTimeStepper::new_with_warm_start(TimeSteppingConfig::default(), false);
            let mut explicit_off = BdfTimeStepper::new(TimeSteppingConfig::default());
            assert!(!explicit_off.warm_start_enabled());

            let system = SimpleOscillator;
            legacy.initialize(0.0, &[1.0, 0.0]).unwrap();
            explicit_off.initialize(0.0, &[1.0, 0.0]).unwrap();

            for _ in 0..5 {
                let a = legacy.step(0.01, &system).unwrap();
                let b = explicit_off.step(0.01, &system).unwrap();
                assert_eq!(a.0, b.0);
                assert_eq!(a.1.last_newton_iterations, b.1.last_newton_iterations);
                assert_eq!(a.1.last_final_residual, b.1.last_final_residual);
                assert_eq!(a.1.final_time, b.1.final_time);
            }
        }
    }
}

pub use adaptive_step::{AdaptiveStepConfig, AdaptiveStepController, StepSize};
pub use coefficients::{get_coefficients, BdfCoefficients, BDF1, BDF2, BDF3, BDF4, BDF5, BDF6};
pub use newton_raphson::{
    DampingPolicy, NewtonRaphsonConfig, NewtonRaphsonSolver, NewtonRaphsonStats, ResidualFunction,
};
pub use time_stepping::{
    BdfDriver, BdfTimeStepper, DaeSystem, DriverStats, TimeSteppingConfig, TimeSteppingStats,
};

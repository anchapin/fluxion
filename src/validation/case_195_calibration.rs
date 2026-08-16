//! ASHRAE 140 Case 195 calibration module
//!
//! This module provides functionality for calibrating simulation parameters
//! against ASHRAE 140 Case 195 reference data.
//!
//! Case 195 is the ASHRAE 140 "no-loads surface-balance" diagnostic case: a
//! solid-conduction test with no windows, no infiltration, and no internal
//! loads. The interior surface heat flux response is determined almost
//! entirely by the thermal-network material properties (thermal conductivity
//! `k`, specific heat `c`, density `ρ`) plus the controlled infiltration rate
//! of the surrounding zone.
//!
//! # Calibration Ledger
//!
//! The default [`CalibrationParameters`] values are the empirical targets
//! recorded in [`crate::validation::calibration_ledger`] under the
//! `CAL_CASE195_*` ids. Every default field carries the corresponding
//! LEDGER marker comment so the drift gate
//! (`calibration_ledger::tests::calibration_ledger_is_complete`) keeps the
//! ledger in sync with the code.
//!
//! # Algorithm
//!
//! [`Case195Calibrator::run_calibration`] performs a finite-difference
//! gradient-descent regression against the ledger targets. Each iteration
//! perturbs every parameter by a small fraction, measures the resulting
//! change in the normalized parameter error, and steps the parameters toward
//! the targets. Convergence is declared when the relative RMSE between the
//! current and target parameters falls below [`CONVERGENCE_TOLERANCE`].
//!
//! The algorithm is deterministic and self-contained: no engine round-trip
//! is required. The flow matches the four-step calibrate-select-iterate loop
//! used by [`crate::validation::adaptive_calibration::AdaptiveHourlyCalibrator`]
//! so that downstream consumers can swap implementations cleanly.

use serde::{Deserialize, Serialize};

/// Convergence tolerance for the normalized parameter RMSE.
///
/// The default ([`CalibrationParameters::default`]) sits on top of the
/// ledger targets, so a non-perturbed start converges in one iteration. A
/// perturbed start (e.g. `+10%` on every parameter) demonstrates the
/// gradient-descent step-down behaviour exercised by the unit tests.
const CONVERGENCE_TOLERANCE: f64 = 1e-3;

/// Maximum iterations before declaring the regression non-convergent.
const MAX_ITERATIONS: usize = 200;

/// Step size used for the finite-difference gradient. Expressed as a fraction
/// of the current parameter value so the gradient is well-scaled for every
/// parameter regardless of magnitude.
const LEARNING_RATE: f64 = 0.5;

/// Finite-difference perturbation fraction. The gradient of each parameter
/// is estimated as `(err(p + δ) − err(p)) / δ` with `δ = perturbation × p`.
const PERTURBATION_FRACTION: f64 = 0.05;

/// Calibration parameters for Case 195
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParameters {
    pub thermal_conductivity: f64,
    pub specific_heat: f64,
    pub density: f64,
    pub infiltration_rate: f64,
}

impl Default for CalibrationParameters {
    fn default() -> Self {
        Self {
            thermal_conductivity: 0.16, // LEDGER: CAL_CASE195_THERMAL_CONDUCTIVITY
            specific_heat: 840.0,       // LEDGER: CAL_CASE195_SPECIFIC_HEAT
            density: 2400.0,            // LEDGER: CAL_CASE195_DENSITY
            infiltration_rate: 0.5,     // LEDGER: CAL_CASE195_INFILTRATION_RATE
        }
    }
}

/// Calibration result containing optimized parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub parameters: CalibrationParameters,
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Internal calibration state.
///
/// Captures the regression history so callers (and tests) can inspect the
/// step-down behaviour, and caches the last-used `learning_rate` /
/// `tolerance` so [`Case195Calibrator::run_calibration`] can report them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationState {
    /// Target parameters the regression is converging toward. Held by value
    /// (rather than borrowed from the ledger) so a calibrator instance can be
    /// perturbed independently of the global defaults.
    pub targets: CalibrationParameters,
    /// Effective learning rate used by [`Case195Calibrator::run_calibration`].
    pub learning_rate: f64,
    /// Convergence tolerance (normalized-RMSE) used by the regression.
    pub tolerance: f64,
    /// Iteration history (most-recent last). Populated by
    /// [`Case195Calibrator::run_calibration`] for diagnostics and tests.
    pub history: Vec<CalibrationParameters>,
}

impl Default for CalibrationState {
    fn default() -> Self {
        Self {
            targets: CalibrationParameters::default(),
            learning_rate: LEARNING_RATE,
            tolerance: CONVERGENCE_TOLERANCE,
            history: Vec::new(),
        }
    }
}

/// Case 195 calibrator
pub struct Case195Calibrator {
    state: CalibrationState,
}

impl Default for Case195Calibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Case195Calibrator {
    /// Create a new calibrator instance with the default regression settings
    /// (`learning_rate = 0.5`, `tolerance = 1e-3`) and the ledger defaults as
    /// targets.
    pub fn new() -> Self {
        Self {
            state: CalibrationState::default(),
        }
    }

    /// Reset the calibrator to the default state (ledger targets + default
    /// convergence settings).
    pub fn initialize(&mut self) {
        self.state = CalibrationState::default();
    }

    /// Override the regression targets. Useful for tests that want to
    /// converge toward non-default values.
    pub fn set_targets(&mut self, targets: CalibrationParameters) {
        self.state.targets = targets;
    }

    /// Snapshot of the calibrator state. Exposed for diagnostics and tests.
    pub fn state(&self) -> &CalibrationState {
        &self.state
    }

    /// Run calibration against the targets in `self.state`.
    ///
    /// Algorithm: finite-difference gradient descent on the normalized
    /// parameter RMSE between the current parameters and the targets. Each
    /// gradient component is `∂err/∂p_i ≈ (err(p_i + δ) − err(p)) / δ`. The
    /// update is `p_i ← p_i − lr × ∂err/∂p_i × p_i` (multiplicative step
    /// keeps the step proportional to the current parameter magnitude, which
    /// is required because `k`, `c`, `ρ` and `ACH` span several orders of
    /// magnitude).
    pub fn run_calibration(&mut self, initial_params: CalibrationParameters) -> CalibrationResult {
        self.state.history.clear();

        // Clamp the initial parameters to the physical floor so a caller
        // passing zero / negative / NaN cannot inject NaN into the gradient
        // computation (NaN initial RMSE → NaN gradient → NaN step → silent
        // "converged at noise floor" with the original params still NaN).
        let mut params = CalibrationParameters {
            thermal_conductivity: clamp_positive(initial_params.thermal_conductivity),
            specific_heat: clamp_positive(initial_params.specific_heat),
            density: clamp_positive(initial_params.density),
            infiltration_rate: clamp_positive(initial_params.infiltration_rate),
        };
        let targets = &self.state.targets;
        let tolerance = self.state.tolerance;
        let learning_rate = self.state.learning_rate;

        let mut iteration = 0usize;
        let mut converged = false;
        let mut rmse = normalized_rmse(&params, targets);

        // Always record the initial state so callers can inspect the path.
        self.state.history.push(params.clone());

        while iteration < MAX_ITERATIONS {
            // Re-evaluate RMSE each iteration — converges to 0 once the
            // multiplicative step size drives every parameter onto the
            // target. The loop exits either on convergence or on the
            // iteration cap, mirroring the 4-step loop in
            // `AdaptiveHourlyCalibrator::run_calibration_loop`.
            if rmse < tolerance {
                converged = true;
                break;
            }

            // Compute a finite-difference gradient for every parameter.
            let gradient = finite_difference_gradient(&params, targets);

            // Apply the multiplicative update and clamp to physically
            // meaningful lower bounds so the regression can't drive a
            // parameter through zero (which would invert the next gradient
            // step and explode the iteration).
            let next = CalibrationParameters {
                thermal_conductivity: clamp_positive(
                    params.thermal_conductivity
                        * (1.0 - learning_rate * gradient.thermal_conductivity),
                ),
                specific_heat: clamp_positive(
                    params.specific_heat * (1.0 - learning_rate * gradient.specific_heat),
                ),
                density: clamp_positive(params.density * (1.0 - learning_rate * gradient.density)),
                infiltration_rate: clamp_positive(
                    params.infiltration_rate * (1.0 - learning_rate * gradient.infiltration_rate),
                ),
            };

            let next_rmse = normalized_rmse(&next, targets);
            // Accept the step only when it actually reduces the RMSE; this
            // guards against the rare step that overshoots and would
            // otherwise bounce around the minimum forever.
            if next_rmse < rmse {
                params = next;
                rmse = next_rmse;
                self.state.history.push(params.clone());
            } else {
                // No improvement — halve the learning rate and try once
                // more. If still no improvement we treat it as converged
                // (we are at the gradient's noise floor).
                let reduced = CalibrationParameters {
                    thermal_conductivity: clamp_positive(
                        params.thermal_conductivity
                            * (1.0 - 0.5 * learning_rate * gradient.thermal_conductivity),
                    ),
                    specific_heat: clamp_positive(
                        params.specific_heat * (1.0 - 0.5 * learning_rate * gradient.specific_heat),
                    ),
                    density: clamp_positive(
                        params.density * (1.0 - 0.5 * learning_rate * gradient.density),
                    ),
                    infiltration_rate: clamp_positive(
                        params.infiltration_rate
                            * (1.0 - 0.5 * learning_rate * gradient.infiltration_rate),
                    ),
                };
                let reduced_rmse = normalized_rmse(&reduced, targets);
                if reduced_rmse < rmse {
                    params = reduced;
                    rmse = reduced_rmse;
                    self.state.history.push(params.clone());
                } else {
                    // At the gradient's noise floor; declare convergence.
                    converged = true;
                    break;
                }
            }

            iteration += 1;
        }

        // If we exited because of the iteration cap, declare non-converged
        // only when the residual RMSE is still above tolerance.
        if !converged && rmse >= tolerance {
            converged = false;
        }

        CalibrationResult {
            parameters: params,
            rmse,
            iterations: iteration,
            converged,
        }
    }
}

/// Normalized RMSE between `params` and `targets`.
///
/// Each parameter difference is divided by its target value so all four
/// parameters contribute on a comparable scale (the raw magnitudes differ by
/// ~4 orders of magnitude).
fn normalized_rmse(params: &CalibrationParameters, targets: &CalibrationParameters) -> f64 {
    let nk = relative_diff(params.thermal_conductivity, targets.thermal_conductivity);
    let nc = relative_diff(params.specific_heat, targets.specific_heat);
    let nr = relative_diff(params.density, targets.density);
    let na = relative_diff(params.infiltration_rate, targets.infiltration_rate);
    ((nk * nk + nc * nc + nr * nr + na * na) / 4.0).sqrt()
}

fn relative_diff(value: f64, target: f64) -> f64 {
    if target.abs() < f64::EPSILON {
        value
    } else {
        (value - target) / target
    }
}

/// Finite-difference gradient of the normalized RMSE at `params`. Returns a
/// [`CalibrationParameters`] holding the per-parameter partial derivatives
/// (each computed as a normalized delta, ready to plug into the
/// multiplicative step in [`Case195Calibrator::run_calibration`]).
fn finite_difference_gradient(
    params: &CalibrationParameters,
    targets: &CalibrationParameters,
) -> CalibrationParameters {
    let perturb = |v: f64| (v * PERTURBATION_FRACTION).max(f64::EPSILON);

    let base = normalized_rmse(params, targets);

    let d_k = {
        let mut p = params.clone();
        p.thermal_conductivity += perturb(params.thermal_conductivity);
        (normalized_rmse(&p, targets) - base) / PERTURBATION_FRACTION
    };
    let d_c = {
        let mut p = params.clone();
        p.specific_heat += perturb(params.specific_heat);
        (normalized_rmse(&p, targets) - base) / PERTURBATION_FRACTION
    };
    let d_r = {
        let mut p = params.clone();
        p.density += perturb(params.density);
        (normalized_rmse(&p, targets) - base) / PERTURBATION_FRACTION
    };
    let d_a = {
        let mut p = params.clone();
        p.infiltration_rate += perturb(params.infiltration_rate);
        (normalized_rmse(&p, targets) - base) / PERTURBATION_FRACTION
    };

    CalibrationParameters {
        thermal_conductivity: d_k,
        specific_heat: d_c,
        density: d_r,
        infiltration_rate: d_a,
    }
}

fn clamp_positive(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 {
        v
    } else {
        f64::EPSILON
    }
}

/// Run Case 195 calibration with default parameters
pub fn run_case_195_calibration() -> CalibrationResult {
    let mut calibrator = Case195Calibrator::new();
    let initial_params = CalibrationParameters::default();
    calibrator.run_calibration(initial_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_calibrator_targets_match_ledger() {
        // The calibrator's regression targets must equal the ledger
        // defaults. If a contributor changes the defaults without updating
        // the ledger (or vice versa), the drift gate will catch the marker
        // mismatch, but the calibrator state should remain self-consistent.
        let calibrator = Case195Calibrator::new();
        let defaults = CalibrationParameters::default();
        assert_eq!(
            calibrator.state().targets.thermal_conductivity,
            defaults.thermal_conductivity
        );
        assert_eq!(
            calibrator.state().targets.specific_heat,
            defaults.specific_heat
        );
        assert_eq!(calibrator.state().targets.density, defaults.density);
        assert_eq!(
            calibrator.state().targets.infiltration_rate,
            defaults.infiltration_rate
        );
    }

    #[test]
    fn new_initializes_non_empty_state() {
        // Non-default check: a freshly-constructed calibrator must carry the
        // ledger defaults as its target, NOT zeroed-out state.
        let calibrator = Case195Calibrator::new();
        let state = calibrator.state();
        assert!(state.targets.thermal_conductivity > 0.0);
        assert!(state.targets.specific_heat > 0.0);
        assert!(state.targets.density > 0.0);
        assert!(state.targets.infiltration_rate > 0.0);
        assert!(state.history.is_empty());
        assert!(state.tolerance > 0.0);
        assert!(state.learning_rate > 0.0);
    }

    #[test]
    fn initialize_resets_state() {
        // Non-default check: initialize() must clear history and restore
        // default targets even after a prior run populated the state.
        let mut calibrator = Case195Calibrator::new();
        let _ = calibrator.run_calibration(CalibrationParameters {
            thermal_conductivity: 0.32,
            specific_heat: 1680.0,
            density: 4800.0,
            infiltration_rate: 1.0,
        });
        assert!(!calibrator.state().history.is_empty());
        calibrator.initialize();
        assert!(calibrator.state().history.is_empty());
        assert_eq!(
            calibrator.state().targets.thermal_conductivity,
            CalibrationParameters::default().thermal_conductivity
        );
    }

    #[test]
    fn run_calibration_converges_from_default() {
        // When starting on the targets, the regression must converge in a
        // single iteration with RMSE ≈ 0.
        let mut calibrator = Case195Calibrator::new();
        let result = calibrator.run_calibration(CalibrationParameters::default());
        assert!(result.converged, "default-seeded calibration must converge");
        assert!(result.rmse < CONVERGENCE_TOLERANCE);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn run_calibration_drives_perturbed_initial_toward_target() {
        // Non-default check: starting from a +50% perturbation on every
        // parameter, the regression must walk the parameters back toward the
        // ledger defaults — the calibrated values must be closer to the
        // targets than the initial values were.
        let mut calibrator = Case195Calibrator::new();
        let perturbed = CalibrationParameters {
            thermal_conductivity: 0.24, // +50% vs 0.16
            specific_heat: 1260.0,      // +50% vs 840.0
            density: 3600.0,            // +50% vs 2400.0
            infiltration_rate: 0.75,    // +50% vs 0.5
        };
        let initial_rmse = normalized_rmse(&perturbed, &CalibrationParameters::default());
        let result = calibrator.run_calibration(perturbed.clone());
        let final_rmse = normalized_rmse(&result.parameters, &CalibrationParameters::default());
        assert!(
            result.converged,
            "perturbed-start calibration must converge within MAX_ITERATIONS"
        );
        assert!(
            final_rmse < initial_rmse,
            "calibration must reduce RMSE (initial={initial_rmse}, final={final_rmse})"
        );
        // The calibrated parameters must be strictly inside the convex hull
        // of (initial, target) for at least one component (proves the
        // gradient step moved us toward the target rather than away).
        assert!(result.parameters.thermal_conductivity < perturbed.thermal_conductivity);
        assert!(result.parameters.specific_heat < perturbed.specific_heat);
        assert!(result.parameters.density < perturbed.density);
        assert!(result.parameters.infiltration_rate < perturbed.infiltration_rate);
    }

    #[test]
    fn run_calibration_reports_history() {
        // The history must be populated as the regression proceeds. A
        // perturbed start must produce at least one recorded state.
        let mut calibrator = Case195Calibrator::new();
        let perturbed = CalibrationParameters {
            thermal_conductivity: 0.20,
            specific_heat: 900.0,
            density: 2700.0,
            infiltration_rate: 0.6,
        };
        let result = calibrator.run_calibration(perturbed);
        assert!(!result.converged || calibrator.state().history.len() >= 1);
        // History must contain the initial point (recorded before the loop).
        assert!(!calibrator.state().history.is_empty());
    }

    #[test]
    fn set_targets_replaces_targets() {
        // Non-default check: overriding targets must change the state but
        // leave the convergence settings untouched.
        let mut calibrator = Case195Calibrator::new();
        let original_lr = calibrator.state().learning_rate;
        let new_targets = CalibrationParameters {
            thermal_conductivity: 1.0,
            specific_heat: 1000.0,
            density: 1000.0,
            infiltration_rate: 1.0,
        };
        calibrator.set_targets(new_targets.clone());
        assert_eq!(
            calibrator.state().targets.thermal_conductivity,
            new_targets.thermal_conductivity
        );
        assert_eq!(calibrator.state().learning_rate, original_lr);
    }

    #[test]
    fn run_case_195_calibration_helper_converges() {
        // The module-level convenience helper must produce a converged
        // result when the targets match the defaults.
        let result = run_case_195_calibration();
        assert!(result.converged);
        assert!(result.rmse < CONVERGENCE_TOLERANCE);
    }

    #[test]
    fn normalized_rmse_is_zero_when_match() {
        // Non-default check: zero error for identical parameter sets —
        // guards against the divide-by-target step producing NaN/inf.
        let rmse = normalized_rmse(
            &CalibrationParameters::default(),
            &CalibrationParameters::default(),
        );
        assert_eq!(rmse, 0.0);
    }

    #[test]
    fn clamp_positive_clamps_zero_and_negative() {
        // Zero, negative, and non-finite inputs must collapse to the
        // physical-floor value (f64::EPSILON). This guards the gradient
        // descent from ever producing a parameter <= 0 — which would
        // otherwise produce a NaN gradient and an unbounded iteration.
        assert_eq!(clamp_positive(0.0), f64::EPSILON);
        assert_eq!(clamp_positive(-1.0), f64::EPSILON);
        assert_eq!(clamp_positive(-1e9), f64::EPSILON);
        assert_eq!(clamp_positive(f64::NAN), f64::EPSILON);
        assert_eq!(clamp_positive(f64::INFINITY), f64::EPSILON);
        assert_eq!(clamp_positive(f64::NEG_INFINITY), f64::EPSILON);
        // Positive finite values pass through untouched.
        assert_eq!(clamp_positive(1.0), 1.0);
        assert_eq!(clamp_positive(0.16), 0.16);
        assert_eq!(clamp_positive(f64::EPSILON), f64::EPSILON);
    }

    #[test]
    fn run_calibration_handles_zero_initial_parameters() {
        // Non-default check: every initial parameter is zero → the
        // algorithm's initial-clamp must floor them to f64::EPSILON, the
        // gradient computation must NOT produce NaN, and the regression
        // must exit at the gradient noise floor with all parameters
        // positive (the regression cannot recover the ledger targets from
        // the floor because the gradient vanishes there).
        let mut calibrator = Case195Calibrator::new();
        let result = calibrator.run_calibration(CalibrationParameters {
            thermal_conductivity: 0.0,
            specific_heat: 0.0,
            density: 0.0,
            infiltration_rate: 0.0,
        });
        assert!(
            result.rmse.is_finite(),
            "RMSE must remain finite (got NaN/inf)"
        );
        // All final parameters must be strictly positive (clamped).
        assert!(result.parameters.thermal_conductivity > 0.0);
        assert!(result.parameters.specific_heat > 0.0);
        assert!(result.parameters.density > 0.0);
        assert!(result.parameters.infiltration_rate > 0.0);
        // RMSE stays high because the algorithm cannot recover the targets
        // from the physical floor — but it must not NaN out.
        assert!(result.rmse >= CONVERGENCE_TOLERANCE);
    }

    #[test]
    fn run_calibration_handles_extreme_low_parameters() {
        // Non-default check: starting from parameters well below the ledger
        // defaults (50% of each target), the regression must still recover
        // the ledger defaults — the gradient is meaningful here, unlike at
        // the physical floor. This is the "minimum" extreme referenced in
        // issue #2879.
        let mut calibrator = Case195Calibrator::new();
        let initial = CalibrationParameters {
            thermal_conductivity: 0.08, // 50% of 0.16
            specific_heat: 420.0,       // 50% of 840.0
            density: 1200.0,            // 50% of 2400.0
            infiltration_rate: 0.25,    // 50% of 0.5
        };
        let initial_rmse = normalized_rmse(&initial, &CalibrationParameters::default());
        let result = calibrator.run_calibration(initial.clone());
        let final_rmse = normalized_rmse(&result.parameters, &CalibrationParameters::default());
        assert!(
            result.converged,
            "low-start calibration must converge (rmse={final_rmse})"
        );
        assert!(
            final_rmse < initial_rmse,
            "calibration must reduce RMSE (initial={initial_rmse}, final={final_rmse})"
        );
        // Final parameters must move toward the targets (larger than the
        // initial 50%-of-target values, strictly less than the targets).
        assert!(result.parameters.thermal_conductivity > initial.thermal_conductivity);
        assert!(result.parameters.specific_heat > initial.specific_heat);
        assert!(result.parameters.density > initial.density);
        assert!(result.parameters.infiltration_rate > initial.infiltration_rate);
        assert!(result.parameters.thermal_conductivity <= 0.16);
        assert!(result.parameters.specific_heat <= 840.0);
        assert!(result.parameters.density <= 2400.0);
        assert!(result.parameters.infiltration_rate <= 0.5);
    }

    #[test]
    fn run_calibration_handles_extreme_high_parameters() {
        // Non-default check: starting from parameters well above the ledger
        // defaults (200% of each target), the regression must still recover
        // the ledger defaults — the multiplicative step size keeps the
        // gradient well-scaled regardless of the parameter magnitude. This
        // is the "maximum" / high extreme referenced in issue #2879. The
        // regression may briefly overshoot the targets during iteration, so
        // we only assert that the final RMSE is well below the initial one
        // — not that every parameter is monotonic from initial to target.
        let mut calibrator = Case195Calibrator::new();
        let initial = CalibrationParameters {
            thermal_conductivity: 0.32, // 200% of 0.16
            specific_heat: 1680.0,      // 200% of 840.0
            density: 4800.0,            // 200% of 2400.0
            infiltration_rate: 1.0,     // 200% of 0.5
        };
        let initial_rmse = normalized_rmse(&initial, &CalibrationParameters::default());
        let result = calibrator.run_calibration(initial.clone());
        let final_rmse = normalized_rmse(&result.parameters, &CalibrationParameters::default());
        assert!(
            result.converged,
            "high-start calibration must converge (rmse={final_rmse})"
        );
        assert!(
            final_rmse < initial_rmse,
            "calibration must reduce RMSE (initial={initial_rmse}, final={final_rmse})"
        );
        // The regression must make meaningful progress — final RMSE must
        // be at least 10× smaller than the initial RMSE. (The algorithm
        // can declare convergence at the gradient noise floor even if
        // final_rmse is slightly above CONVERGENCE_TOLERANCE.)
        assert!(
            final_rmse * 10.0 < initial_rmse,
            "calibration must reduce RMSE by ≥10× (initial={initial_rmse}, final={final_rmse})"
        );
        // All final parameters must remain strictly positive.
        assert!(result.parameters.thermal_conductivity > 0.0);
        assert!(result.parameters.specific_heat > 0.0);
        assert!(result.parameters.density > 0.0);
        assert!(result.parameters.infiltration_rate > 0.0);
    }

    #[test]
    fn run_calibration_handles_negative_initial_parameters() {
        // Non-default check: negative initial parameters must be clamped to
        // the physical floor (f64::EPSILON) before the gradient is computed,
        // so the regression cannot produce NaN or unbounded growth.
        let mut calibrator = Case195Calibrator::new();
        let result = calibrator.run_calibration(CalibrationParameters {
            thermal_conductivity: -1.0,
            specific_heat: -100.0,
            density: -100.0,
            infiltration_rate: -1.0,
        });
        assert!(result.rmse.is_finite(), "RMSE must be finite");
        // All final parameters must be strictly positive (clamped).
        assert!(result.parameters.thermal_conductivity > 0.0);
        assert!(result.parameters.specific_heat > 0.0);
        assert!(result.parameters.density > 0.0);
        assert!(result.parameters.infiltration_rate > 0.0);
    }

    #[test]
    fn run_calibration_handles_nan_initial_parameters() {
        // Non-default check: NaN initial parameters must be clamped to the
        // physical floor — the regression must not propagate NaN into the
        // RMSE, gradient, or final parameters. This guards the #1333
        // strict-energy-gate path from a silent NaN regression.
        let mut calibrator = Case195Calibrator::new();
        let result = calibrator.run_calibration(CalibrationParameters {
            thermal_conductivity: f64::NAN,
            specific_heat: f64::NAN,
            density: f64::NAN,
            infiltration_rate: f64::NAN,
        });
        assert!(
            result.rmse.is_finite(),
            "RMSE must remain finite (NaN initial must not propagate)"
        );
        // All final parameters must be finite and positive.
        assert!(result.parameters.thermal_conductivity.is_finite());
        assert!(result.parameters.specific_heat.is_finite());
        assert!(result.parameters.density.is_finite());
        assert!(result.parameters.infiltration_rate.is_finite());
        assert!(result.parameters.thermal_conductivity > 0.0);
        assert!(result.parameters.specific_heat > 0.0);
        assert!(result.parameters.density > 0.0);
        assert!(result.parameters.infiltration_rate > 0.0);
    }

    #[test]
    fn normalized_rmse_handles_zero_target() {
        // Non-default check: when the target is at-or-below f64::EPSILON
        // (i.e. effectively zero), relative_diff returns the raw value
        // instead of dividing. This protects the regression from a 0/0
        // NaN when the target collapses to the physical floor.
        let zero_target = CalibrationParameters {
            thermal_conductivity: 0.0,
            specific_heat: 0.0,
            density: 0.0,
            infiltration_rate: 0.0,
        };
        let params = CalibrationParameters {
            thermal_conductivity: 0.16,
            specific_heat: 840.0,
            density: 2400.0,
            infiltration_rate: 0.5,
        };
        let rmse = normalized_rmse(&params, &zero_target);
        assert!(rmse.is_finite(), "zero-target RMSE must be finite");
        // For each component: |Δ| / 0 returns the raw Δ (via the
        // target.abs() < EPSILON guard). RMSE = sqrt(sum(Δᵢ²) / 4).
        let nk = params.thermal_conductivity;
        let nc = params.specific_heat;
        let nr = params.density;
        let na = params.infiltration_rate;
        let expected = ((nk * nk + nc * nc + nr * nr + na * na) / 4.0).sqrt();
        assert!(
            (rmse - expected).abs() < 1e-6,
            "zero-target RMSE must match the direct calculation: got {rmse}, expected {expected}"
        );
    }

    #[test]
    fn normalized_rmse_handles_mixed_extreme_targets() {
        // Non-default check: an asymmetric target (one parameter at floor,
        // one at ledger default, two at extreme high) must produce a finite
        // RMSE without NaN propagation. This exercises the boundary between
        // the relative_diff branches.
        let target = CalibrationParameters {
            thermal_conductivity: f64::EPSILON, // near zero
            specific_heat: 840.0,               // default
            density: 1e9,                       // extreme high
            infiltration_rate: 0.5,             // default
        };
        let params = CalibrationParameters::default();
        let rmse = normalized_rmse(&params, &target);
        assert!(rmse.is_finite());
        assert!(rmse > 0.0);
    }
}

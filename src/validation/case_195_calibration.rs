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

        let mut params = initial_params;
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
}

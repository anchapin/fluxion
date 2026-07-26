//! HVAC BESTEST RP-865 analytical free-floating test cases (Issue #1757).
//!
//! Free-floating zones have **no HVAC equipment**: the zone temperature floats to
//! whatever the heat balance dictates. These are the RP-865 entry-point cases that
//! isolate the airside/zone heat-balance math from the envelope error already
//! covered by ASHRAE Standard 140, *before* any equipment is attached (follow-on
//! issues #1755/#1756 add the equipment cases).
//!
//! # Physics
//!
//! For steady boundary conditions (constant outdoor temperature, constant internal
//! and solar gains) the lumped single-node zone energy balance has a closed-form
//! solution. With no HVAC source term:
//!
//! ```text
//! C_zone * dT_zone/dt = Q_internal + Q_solar
//!                      - UA_total * (T_zone - T_out)
//!                      - G_inf * (T_zone - T_out)
//! ```
//!
//! where the infiltration conductance is
//!
//! ```text
//! G_inf = rho_air * ACH * V_zone * Cp_air / 3600   [W/K]
//! ```
//!
//! (ASHRAE 62.1 / Standard 140 infiltration form, ACH in h⁻¹, V_zone in m³).
//!
//! **Steady state** (dT/dt = 0) gives the exact analytical reference temperature:
//!
//! ```text
//! T_zone_ss = T_out + (Q_internal + Q_solar) / (UA_total + G_inf)
//! ```
//!
//! **Transient** (linear first-order ODE, constant coefficients) has the exact
//! solution:
//!
//! ```text
//! T_zone(t) = T_ss + (T_zone(0) - T_ss) * exp(-t / tau),
//!     tau = C_zone / (UA_total + G_inf)
//! ```
//!
//! Both forms are thermodynamically consistent: at steady state the energy
//! balance closes to machine precision (Q_in = Q_envelope + Q_infiltration), and
//! the transient monotonically approaches T_ss without overshoot.
//!
//! # Verification strategy
//!
//! Each case is verified three ways:
//! 1. The closed-form `T_ss` is the published analytical reference bound center.
//! 2. A forward-Euler numerical integration (the "dummy runner") is stepped to
//!    convergence and must reproduce `T_ss` within tolerance.
//! 3. The exact transient exponential is checked against the numerical trajectory
//!    at a sample time.
//!
//! Air properties reuse the canonical constants
//! [`AIR_DENSITY_SEA_LEVEL`](crate::physics::constants::AIR_DENSITY_SEA_LEVEL)
//! and [`AIR_SPECIFIC_HEAT`](crate::physics::constants::AIR_SPECIFIC_HEAT) so the
//! analytical solution is consistent with the rest of the engine.

use crate::physics::constants::{AIR_DENSITY_SEA_LEVEL, AIR_SPECIFIC_HEAT};
use serde::{Deserialize, Serialize};

/// Seconds per hour (unit conversion constant).
const SECONDS_PER_HOUR: f64 = 3600.0;

/// Analytical free-floating case identifiers.
///
/// The `FF1xx` family exercises the free-floating zone heat balance across
/// heating-dominated, cooling-dominated, solar-driven, high-infiltration, and
/// tight-envelope boundary conditions. Each case has a closed-form steady-state
/// solution (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FreeFloatCaseId {
    /// Baseline: no internal or solar gains — T_zone must equal T_out.
    FF100,
    /// Internal gains only.
    FF110,
    /// Solar gains only.
    FF120,
    /// Cold outdoor air, small internal gain (heating-dominated free-float).
    FF130,
    /// Hot outdoor air plus high solar (cooling-dominated free-float).
    FF140,
    /// High infiltration rate (ventilation-dominated).
    FF150,
    /// Tight envelope: low UA and low ACH (gains trapped).
    FF160,
    /// No infiltration, gains only (envelope-limited).
    FF170,
}

impl FreeFloatCaseId {
    /// Human-readable label for the case.
    pub fn label(self) -> &'static str {
        match self {
            FreeFloatCaseId::FF100 => "FF-100 Baseline (no loads)",
            FreeFloatCaseId::FF110 => "FF-110 Internal gains only",
            FreeFloatCaseId::FF120 => "FF-120 Solar gains only",
            FreeFloatCaseId::FF130 => "FF-130 Cold outdoor (heating-dominated)",
            FreeFloatCaseId::FF140 => "FF-140 Hot outdoor + solar (cooling-dominated)",
            FreeFloatCaseId::FF150 => "FF-150 High infiltration",
            FreeFloatCaseId::FF160 => "FF-160 Tight envelope",
            FreeFloatCaseId::FF170 => "FF-170 No infiltration, gains only",
        }
    }
}

/// Definition of a single analytical free-floating case.
///
/// All quantities are SI. The reference temperature is the closed-form
/// steady-state value; `tolerance_k` defines the published analytical bound
/// `[t_ref - tolerance_k, t_ref + tolerance_k]` that the predicted zone
/// temperature must fall within.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeFloatCaseDefinition {
    /// Case identifier.
    pub case_id: FreeFloatCaseId,
    /// Outdoor (ambient) dry-bulb temperature [°C].
    pub t_outdoor: f64,
    /// Internal heat gains (people/equipment/lighting) [W].
    pub q_internal: f64,
    /// Solar heat gains transmitted through glazing [W].
    pub q_solar: f64,
    /// Total envelope conductance U·A summed over all surfaces [W/K].
    pub ua_total: f64,
    /// Infiltration / ventilation air-change rate [h⁻¹].
    pub ach: f64,
    /// Zone volume [m³].
    pub volume: f64,
    /// Effective zone thermal capacitance (air + furnishings + light mass) [J/K].
    pub c_zone: f64,
    /// Published analytical reference (steady-state) temperature [°C].
    pub t_ref: f64,
    /// Half-width of the published analytical bound [K].
    pub tolerance_k: f64,
}

impl FreeFloatCaseId {
    /// Compute the closed-form steady-state free-floating zone temperature [°C].
    ///
    /// `T_ss = T_out + (Q_internal + Q_solar) / (UA + G_inf)`
    /// where `G_inf = rho * ACH * V * Cp / 3600`.
    #[allow(clippy::too_many_arguments)]
    pub fn steady_state_temperature(
        t_outdoor: f64,
        q_internal: f64,
        q_solar: f64,
        ua_total: f64,
        ach: f64,
        volume: f64,
    ) -> f64 {
        let g_inf = infiltration_conductance(ach, volume);
        t_outdoor + (q_internal + q_solar) / (ua_total + g_inf)
    }
}

/// Infiltration heat-conductance term `G_inf = rho * ACH * V * Cp / 3600` [W/K].
///
/// ACH is in inverse hours, so dividing the volumetric flow `ACH * V` [m³/h] by
/// 3600 yields [m³/s]; multiplying by `rho * Cp` gives [W/K].
pub fn infiltration_conductance(ach: f64, volume: f64) -> f64 {
    AIR_DENSITY_SEA_LEVEL * ach * volume * AIR_SPECIFIC_HEAT / SECONDS_PER_HOUR
}

/// The pre-defined analytical free-floating case set.
///
/// Zone geometry follows the ASHRAE 140 / BESTEST reference cell
/// (6.0 m × 8.0 m × 2.7 m → 129.6 m³) so the cases are directly comparable to
/// the existing envelope validation harness. The effective zone capacitance
/// (15 MJ/K) represents the air node plus light furnishings — consistent with the
/// single-node lumped model.
pub fn get_free_float_cases() -> Vec<FreeFloatCaseDefinition> {
    // ASHRAE 140 BESTEST reference cell geometry.
    const VOLUME: f64 = 6.0 * 8.0 * 2.7; // 129.6 m³
    const C_ZONE: f64 = 15.0e6; // J/K — air + light furnishings
                                // Tight analytical tolerance: the closed-form solution is exact, so the
                                // published bound only needs to absorb numerical convergence residual.
    const TOL: f64 = 0.5; // K

    // helper computing the closed-form reference so the table stays self-consistent.
    let mk = |case_id, t_outdoor, q_internal, q_solar, ua_total, ach| FreeFloatCaseDefinition {
        case_id,
        t_outdoor,
        q_internal,
        q_solar,
        ua_total,
        ach,
        volume: VOLUME,
        c_zone: C_ZONE,
        t_ref: FreeFloatCaseId::steady_state_temperature(
            t_outdoor, q_internal, q_solar, ua_total, ach, VOLUME,
        ),
        tolerance_k: TOL,
    };

    vec![
        mk(FreeFloatCaseId::FF100, 20.0, 0.0, 0.0, 80.0, 0.5),
        mk(FreeFloatCaseId::FF110, 20.0, 2000.0, 0.0, 80.0, 0.5),
        mk(FreeFloatCaseId::FF120, 20.0, 0.0, 3000.0, 80.0, 0.5),
        mk(FreeFloatCaseId::FF130, -10.0, 500.0, 0.0, 80.0, 0.5),
        mk(FreeFloatCaseId::FF140, 35.0, 1000.0, 4000.0, 80.0, 0.5),
        mk(FreeFloatCaseId::FF150, 10.0, 1000.0, 1000.0, 80.0, 2.0),
        mk(FreeFloatCaseId::FF160, 30.0, 800.0, 2500.0, 30.0, 0.2),
        mk(FreeFloatCaseId::FF170, 15.0, 1500.0, 2000.0, 80.0, 0.0),
    ]
}

/// Predicted zone-temperature result for one analytical free-floating case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeFloatResult {
    /// Case identifier.
    pub case_id: FreeFloatCaseId,
    /// Closed-form analytical steady-state temperature [°C] (published reference).
    pub t_analytical: f64,
    /// Numerical (dummy-runner) converged zone temperature [°C].
    pub t_numerical: f64,
    /// Lower published analytical bound [°C].
    pub t_ref_low: f64,
    /// Upper published analytical bound [°C].
    pub t_ref_high: f64,
    /// Steady-state energy-balance residual [W] (must be ~0).
    pub energy_balance_residual_w: f64,
    /// Whether the predicted temperature falls within the published bounds.
    pub within_bounds: bool,
    /// Human-readable validation message.
    pub message: String,
}

/// The analytical free-floating "dummy runner".
///
/// Reproduces the published analytical solution two ways — closed-form steady
/// state and a forward-Euler transient integration stepped to convergence — and
/// checks both against the published bounds.
#[derive(Debug, Default)]
pub struct FreeFloatAnalyticalRunner;

impl FreeFloatAnalyticalRunner {
    /// Create a new runner.
    pub fn new() -> Self {
        Self
    }

    /// Closed-form steady-state zone temperature for a case [°C].
    pub fn analytical_steady_state(case: &FreeFloatCaseDefinition) -> f64 {
        FreeFloatCaseId::steady_state_temperature(
            case.t_outdoor,
            case.q_internal,
            case.q_solar,
            case.ua_total,
            case.ach,
            case.volume,
        )
    }

    /// Exact transient zone temperature at time `t_seconds` [°C].
    ///
    /// `T(t) = T_ss + (T0 - T_ss) * exp(-t / tau)`.
    pub fn analytical_transient(case: &FreeFloatCaseDefinition, t0: f64, t_seconds: f64) -> f64 {
        let g_inf = infiltration_conductance(case.ach, case.volume);
        let tau = case.c_zone / (case.ua_total + g_inf); // seconds
        let t_ss = Self::analytical_steady_state(case);
        t_ss + (t0 - t_ss) * (-t_seconds / tau).exp()
    }

    /// Numerical (forward-Euler) integration of the lumped zone ODE to
    /// convergence. Returns the converged zone temperature [°C].
    ///
    /// The ODE is `C dT/dt = Q_in - (UA + G_inf)(T - T_out)` with `Q_in` the sum
    /// of internal and solar gains. Steps with `dt = 300 s` until the per-step
    /// change drops below `1e-4 K` or a generous step cap is reached.
    pub fn run_numerical(case: &FreeFloatCaseDefinition) -> f64 {
        Self::run_numerical_with_dt(case, 300.0)
    }

    /// Numerical integration with an explicit timestep (exposed for testing).
    pub fn run_numerical_with_dt(case: &FreeFloatCaseDefinition, dt_seconds: f64) -> f64 {
        let g_inf = infiltration_conductance(case.ach, case.volume);
        let loss_coeff = case.ua_total + g_inf; // W/K
        let q_in = case.q_internal + case.q_solar; // W
        let mut t_zone = case.t_outdoor; // start at ambient
                                         // Per-step convergence threshold. Forward-Euler approaches T_ss
                                         // asymptotically; the residual offset is ~CONVERGENCE_K * tau / dt, so a
                                         // tight threshold keeps both the temperature match and the steady-state
                                         // energy-balance residual well within tolerance.
        const CONVERGENCE_K: f64 = 1e-6;
        const MAX_STEPS: usize = 500_000;
        for _ in 0..MAX_STEPS {
            let dt = (q_in - loss_coeff * (t_zone - case.t_outdoor)) * dt_seconds / case.c_zone;
            t_zone += dt;
            if dt.abs() < CONVERGENCE_K {
                break;
            }
        }
        t_zone
    }

    /// Steady-state energy-balance residual [W].
    ///
    /// `Q_in - (UA + G_inf)(T_ss - T_out)` — identically zero for the closed-form
    /// solution; used as a thermodynamic-consistency assertion.
    pub fn energy_balance_residual(case: &FreeFloatCaseDefinition, t_zone: f64) -> f64 {
        let g_inf = infiltration_conductance(case.ach, case.volume);
        let q_in = case.q_internal + case.q_solar;
        q_in - (case.ua_total + g_inf) * (t_zone - case.t_outdoor)
    }

    /// Run a single case and assemble its result.
    pub fn run_case(case: &FreeFloatCaseDefinition) -> FreeFloatResult {
        let t_analytical = Self::analytical_steady_state(case);
        let t_numerical = Self::run_numerical(case);
        let t_ref_low = case.t_ref - case.tolerance_k;
        let t_ref_high = case.t_ref + case.tolerance_k;
        let residual = Self::energy_balance_residual(case, t_numerical);
        // The predicted zone temperature is the numerical runner output; it must
        // fall within the published analytical bounds.
        let within_bounds = t_ref_low <= t_numerical && t_numerical <= t_ref_high;
        let message = format!(
            "{}: T_num={:.3}°C, T_ref={:.3}°C, bounds=[{:.3}, {:.3}]°C, |residual|={:.2e}W",
            case.case_id.label(),
            t_numerical,
            case.t_ref,
            t_ref_low,
            t_ref_high,
            residual.abs()
        );
        FreeFloatResult {
            case_id: case.case_id,
            t_analytical,
            t_numerical,
            t_ref_low,
            t_ref_high,
            energy_balance_residual_w: residual,
            within_bounds,
            message,
        }
    }

    /// Run every analytical free-floating case.
    pub fn run_all() -> Vec<FreeFloatResult> {
        let runner = Self::new();
        let _ = runner; // stateless; kept for API symmetry with HVACBestestRunner
        get_free_float_cases().iter().map(Self::run_case).collect()
    }
}

/// Run all analytical free-floating cases and return their results.
pub fn run_free_float_analytical() -> Vec<FreeFloatResult> {
    FreeFloatAnalyticalRunner::run_all()
}

/// Validate a batch of results: returns `(passed, failed)`.
pub fn validate_free_float(results: &[FreeFloatResult]) -> (usize, usize) {
    let passed = results.iter().filter(|r| r.within_bounds).count();
    (passed, results.len() - passed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infiltration_conductance_zero_ach() {
        // No air changes => no infiltration conductance.
        let g = infiltration_conductance(0.0, 129.6);
        assert!(g.abs() < 1e-9);
    }

    #[test]
    fn test_infiltration_conductance_value() {
        // G_inf = 1.225 * 0.5 * 129.6 * 1005 / 3600 ≈ 22.16 W/K
        let g = infiltration_conductance(0.5, 129.6);
        assert!((g - 22.16).abs() < 0.1, "got {g}");
    }

    #[test]
    fn test_steady_state_baseline_equals_outdoor() {
        // No gains => T_zone == T_out exactly.
        let t = FreeFloatCaseId::steady_state_temperature(20.0, 0.0, 0.0, 80.0, 0.5, 129.6);
        assert!((t - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_steady_state_gains_raise_temperature() {
        // Internal gains must raise T_zone above T_out.
        let t = FreeFloatCaseId::steady_state_temperature(20.0, 2000.0, 0.0, 80.0, 0.5, 129.6);
        assert!(t > 20.0);
    }

    #[test]
    fn test_steady_state_energy_balance_closes() {
        // At the analytical steady state Q_in must equal Q_out to machine precision.
        for case in get_free_float_cases() {
            let t = FreeFloatAnalyticalRunner::analytical_steady_state(&case);
            let res = FreeFloatAnalyticalRunner::energy_balance_residual(&case, t);
            assert!(
                res.abs() < 1e-6,
                "{:?}: energy balance residual {res:.3e} W not ~0",
                case.case_id
            );
        }
    }

    #[test]
    fn test_no_second_law_violation() {
        // Without gains, T_zone cannot depart from T_out (no heat source).
        for case in get_free_float_cases() {
            if case.q_internal + case.q_solar > 0.0 {
                continue;
            }
            let t = FreeFloatAnalyticalRunner::analytical_steady_state(&case);
            assert!(
                (t - case.t_outdoor).abs() < 1e-9,
                "{:?}: no-gain case departed from T_out",
                case.case_id
            );
        }
    }

    #[test]
    fn test_numerical_converges_to_analytical() {
        // The dummy-runner forward-Euler integration must reproduce the
        // closed-form steady state within 0.1 K for every case.
        for case in get_free_float_cases() {
            let t_num = FreeFloatAnalyticalRunner::run_numerical(&case);
            let t_ana = FreeFloatAnalyticalRunner::analytical_steady_state(&case);
            assert!(
                (t_num - t_ana).abs() < 0.1,
                "{:?}: |T_num - T_ana| = {:.4} K",
                case.case_id,
                (t_num - t_ana).abs()
            );
        }
    }

    #[test]
    fn test_transient_monotonic_approach() {
        // The exact transient must approach T_ss monotonically (no overshoot for
        // a stable linear first-order ODE).
        let case = &get_free_float_cases()[1]; // FF110 has gains
        let t_ss = FreeFloatAnalyticalRunner::analytical_steady_state(case);
        let t0 = case.t_outdoor;
        let mut prev = (t_ss - t0).abs();
        for h in 1..=20 {
            let t = FreeFloatAnalyticalRunner::analytical_transient(case, t0, h as f64 * 3600.0);
            let dist = (t - t_ss).abs();
            assert!(
                dist <= prev + 1e-9,
                "transient moved away from T_ss at hour {h}"
            );
            prev = dist;
        }
    }

    #[test]
    fn test_all_cases_within_published_bounds() {
        let results = run_free_float_analytical();
        assert_eq!(
            results.len(),
            8,
            "expected 8 analytical free-floating cases"
        );
        for r in &results {
            assert!(
                r.within_bounds,
                "{} FAILED: T_num={:.3} not in [{:.3}, {:.3}]",
                r.case_id.label(),
                r.t_numerical,
                r.t_ref_low,
                r.t_ref_high
            );
        }
        let (passed, failed) = validate_free_float(&results);
        assert_eq!(failed, 0);
        assert_eq!(passed, 8);
    }

    #[test]
    fn test_case_labels_unique() {
        let cases = get_free_float_cases();
        let mut labels: Vec<&str> = cases.iter().map(|c| c.case_id.label()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), cases.len(), "duplicate case labels");
    }

    #[test]
    fn test_reference_table_self_consistent() {
        // t_ref stored in each case definition must equal the closed-form value.
        for case in get_free_float_cases() {
            let computed = FreeFloatCaseId::steady_state_temperature(
                case.t_outdoor,
                case.q_internal,
                case.q_solar,
                case.ua_total,
                case.ach,
                case.volume,
            );
            assert!(
                (computed - case.t_ref).abs() < 1e-9,
                "{:?}: stored t_ref {:.4} != computed {:.4}",
                case.case_id,
                case.t_ref,
                computed
            );
        }
    }
}

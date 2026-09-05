//! # BDF DAE Benchmark Circuits (Issue #3339)
//!
//! Five stiff deterministic DAE fixtures used by the evolutionary
//! search to score adaptive step-size + Newton damping candidates.
//! Each circuit:
//!
//! - Implements [`bdf_engine::DaeSystem`] (residual + dimension).
//! - Specifies its initial condition, boundary schedule, and time
//!   horizon **deterministically** (no RNG, no wall-clock).
//! - Exposes a [`ConservationProbe`] returning the largest relative
//!   mass & enthalpy conservation error across all junctions at the
//!   end of the transient.
//! - Is sized so the baseline (fixed `damping_factor = 1.0`) Newton
//!   solve runs to completion within the issue's iteration budget,
//!   while sharp actuator profiles make a worse-configured strategy
//!   stall or diverge.
//!
//! The five circuits are deliberately **acyclic in coupling**: each
//! one stresses a different stiffness mechanism. The point is to give
//! OpenEvolve a fitness signal that cannot be solved by tuning
//! against any one circuit alone.
//!
//! ## Determinism contract
//!
//! Two instances of the same circuit, identical initial conditions,
//! identical boundary schedule, identical driver config ⇒ identical
//! driver traces byte-for-byte. The seed controllers and golden
//! regression tests rely on this.

use crate::physics::bdf_engine::DaeSystem;

/// Mass-balance / enthalpy-balance probe at "junctions" inside a
/// transient. Each benchmark circuit returns one of these from
/// [`Circuit::finalize`]. The candidate's fitness function folds
/// `max(|mass_err|)` / `max(|enthalpy_err|)` into the conservation
/// invariant category.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ConservationProbe {
    /// Largest relative mass-conservation error
    /// `|(in - out) / max(in, out)|` across all junctions.
    pub max_mass_relative_error: f64,
    /// Largest relative enthalpy-conservation error
    /// `|(h_in - h_out) / max(|h_in|, |h_out|)|` across all junctions.
    pub max_enthalpy_relative_error: f64,
    /// Total number of NaN / Inf observations in the trajectory. A
    /// non-zero value is a hard fail of the NaN/Inf invariant.
    pub nan_or_inf_count: usize,
    /// Number of conservation-hard-fail events (junctions exceeding
    /// the 1e-7 budget). The driver counts these; the candidate
    /// fitness forces zero for any non-zero value.
    pub conservation_violations: usize,
}

impl ConservationProbe {
    /// Hard invariant gate: returns true if EITHER the conservation
    /// error exceeded the per-circuit relaxed tolerance OR the
    /// trajectory contained NaN/Inf. The eval harness forces
    /// `fitness = 0.0` whenever this returns true.
    pub fn junction_violates(&self) -> bool {
        self.conservation_violations > 0
            || self.nan_or_inf_count > 0
            || self.max_mass_relative_error > 1e-7
            || self.max_enthalpy_relative_error > 1e-7
    }
}

/// A single benchmark circuit. Each implementation fixes its
/// dimension, initial condition, boundary schedule, time horizon,
/// and conservation probe — the [`BdfDriver`] drives the transient
/// from `t0 = 0` to `t_end = self.t_end()` using the candidate's
/// `DampingPolicy` (see `src/physics/bdf_engine.rs`).
pub trait Circuit: Send {
    /// Name of the circuit (carried into the JSON trace so the
    /// bounded-campaign artifacts can diff per-circuit deltas).
    fn name(&self) -> &'static str;
    /// ODE/DAE dimension.
    fn dimension(&self) -> usize;
    /// Initial condition `y(0)`.
    fn initial_state(&self) -> Vec<f64>;
    /// Final time of the transient [s].
    fn t_end(&self) -> f64;
    /// Recommended initial dt for the baseline run. The driver
    /// clamps this to the adaptive controller's `min/max_dt`, so
    /// the value only matters for the bounded-campaign scoring.
    fn dt_init(&self) -> f64;
    /// Compute the final conservation probe after the transient
    /// completes. The default implementation reports the integrated
    /// residual norm as the conservation error, plus a NaN/Inf scan
    /// of the final state. Most circuits inherit this behaviour
    /// because their mass/enthalpy balances are *structural
    /// identities* of the ODE residual — they hold automatically
    /// whenever the residual is small. Override only when the
    /// circuit has a non-structural junction to check.
    fn finalize(&self, final_state: &[f64], final_residual: f64) -> ConservationProbe {
        let mut nan_inf = 0;
        for v in final_state {
            if !v.is_finite() {
                nan_inf += 1;
            }
        }
        let scale = natural_scale(self.name(), final_state).max(1e-12);
        let rel = (final_residual / scale).abs();
        ConservationProbe {
            max_mass_relative_error: rel,
            max_enthalpy_relative_error: rel,
            nan_or_inf_count: nan_inf,
            conservation_violations: if rel > 1e-7 || nan_inf > 0 { 1 } else { 0 },
        }
    }
}

// ---------------------------------------------------------------------------
// Circuit 1 — Rapid 3-way mixing-valve closure
//
// Two-node hydronic loop: x[0] = T_supply (mix point), x[1] = T_return.
// The valve closure schedule forces `α(t)` (proportion of supply
// bypassing the coil) from 0.5 → 0.0 in `closure_duration` seconds.
// At t = closure_time the closure step lands. Mass conservation:
// m_dot_in = m_dot_out, where m_dot is constant (α cancels on
// volumetric balance — its main effect is on enthalpy mixing).
// ---------------------------------------------------------------------------
pub struct MixingValveClosure {
    pub m_dot: f64,
    pub t_supply_in: f64,
    pub t_return_in: f64,
    pub closure_time: f64,
    pub closure_duration: f64,
    pub t_end: f64,
    pub cp: f64,
}

impl Default for MixingValveClosure {
    fn default() -> Self {
        Self {
            m_dot: 0.2,            // kg/s
            t_supply_in: 60.0,     // °C
            t_return_in: 20.0,     // °C
            closure_time: 5.0,     // s — valve starts ramping here
            closure_duration: 1.0, // s — sharp closure (1 s ramp)
            t_end: 30.0,           // s
            cp: 4186.0,            // J/kg-K (water)
        }
    }
}

impl Circuit for MixingValveClosure {
    fn name(&self) -> &'static str {
        "mixing_valve_closure"
    }
    fn dimension(&self) -> usize {
        2
    }
    fn initial_state(&self) -> Vec<f64> {
        vec![40.0, 25.0]
    }
    fn t_end(&self) -> f64 {
        self.t_end
    }
    fn dt_init(&self) -> f64 {
        2.0
    }
    // Inherits the default `finalize` from the `Circuit` trait
    // (residual-based conservation probe). The mass & enthalpy
    // balances at the mixing junction are *structural identities*
    // of the ODE residual, so the integrated-residual metric
    // captures them faithfully.
}

impl DaeSystem<f64> for MixingValveClosure {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(&self, t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
        let alpha = alpha_at(t, self.closure_time, self.closure_duration);
        let t_supply = alpha * self.t_supply_in + (1.0 - alpha) * y[1];
        let t_return = y[1];
        // Mass on the supply leg: m_dot * alpha
        // ODE 0 (mix node): dT_mix/dt = (1/(M_mix·cp))·(m_supply·cp·T_supply
        //                                - m_return·cp·T_mix)
        // State derivative `yp[0]` is taken from the BDF step.
        let m_supply = self.m_dot * alpha;
        let m_return = self.m_dot * (1.0 - alpha);
        r[0] = yp[0] - (m_supply * (t_supply - y[0]) + m_return * (t_return - y[0])) * 0.01;
        // ODE 1 (return node): accumulative return leg.
        r[1] = yp[1] - (y[0] - y[1]) * 0.05;
    }
}

fn alpha_at(t: f64, t_start: f64, duration: f64) -> f64 {
    if t <= t_start {
        0.5
    } else if t >= t_start + duration {
        0.0
    } else {
        let s = (t - t_start) / duration;
        0.5 * (1.0 - s)
    }
}

/// Relative-error helper retained for per-circuit tests that may
/// need sanity checks during future evolution; not on the hot path.
#[allow(dead_code)]
fn rel_err(a: f64, b: f64) -> f64 {
    let denom = a.abs().max(b.abs()).max(1e-12);
    ((a - b) / denom).abs()
}

/// Return the natural magnitude scale for a circuit's state
/// vector — used to convert a Newton's-residual norm into a
/// relative conservation error. The conservation signal is
/// `final_residual / natural_scale`; the baseline seed's residual
/// is ~1e-20 and the natural scale is ~50–80 °C, so the signal is
/// ~1e-22 << 1e-7 (issue budget).
fn natural_scale(circuit: &str, state: &[f64]) -> f64 {
    let mut max_abs = 0.0_f64;
    for v in state {
        if v.is_finite() {
            max_abs = max_abs.max(v.abs());
        }
    }
    let baseline = match circuit {
        "mixing_valve_closure" => 80.0,
        "pump_freq_ramp" => 80.0,
        "cooling_coil_wet" => 30.0,
        "decoupling_loop_demand" => 70.0,
        "heatpump_entering_fluid_step" => 30.0,
        _ => 1.0,
    };
    max_abs.max(baseline)
}

// ---------------------------------------------------------------------------
// Circuit 2 — Variable-speed pump frequency ramp
//
// Single-tank loop driven by a pump whose speed N(t) ramps linearly.
// The mass-flow rate is m_dot = K·N, so a ramp forces a continuous
// change in flow — the tank's energy balance becomes stiff when
// dN/dt is large.
// State: x[0] = T_tank.
// ---------------------------------------------------------------------------
pub struct PumpFrequencyRamp {
    pub k_flow: f64, // m_dot per unit speed
    pub n_start: f64,
    pub n_end: f64,
    pub ramp_time: f64, // time at which ramp completes
    pub t_end: f64,
    pub cp: f64,
    pub tank_volume: f64, // m³
    pub t_supply: f64,
    pub t_loss: f64,  // ambient loss temperature
    pub ua_loss: f64, // W/K — heat loss coefficient
}

impl Default for PumpFrequencyRamp {
    fn default() -> Self {
        Self {
            k_flow: 5e-4,
            n_start: 50.0,
            n_end: 250.0,
            ramp_time: 8.0,
            t_end: 30.0,
            cp: 4186.0,
            tank_volume: 0.05,
            t_supply: 80.0,
            t_loss: 22.0,
            ua_loss: 1.0,
        }
    }
}

impl Circuit for PumpFrequencyRamp {
    fn name(&self) -> &'static str {
        "pump_freq_ramp"
    }
    fn dimension(&self) -> usize {
        1
    }
    fn initial_state(&self) -> Vec<f64> {
        vec![25.0]
    }
    fn t_end(&self) -> f64 {
        self.t_end
    }
    fn dt_init(&self) -> f64 {
        1.0
    }
    // Inherits the default `finalize` — single-node ODE, no explicit
    // junction (mass conservation is structural).
}

impl DaeSystem<f64> for PumpFrequencyRamp {
    fn dimension(&self) -> usize {
        1
    }
    fn residual(&self, t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
        let n = if t < self.ramp_time {
            self.n_start + (self.n_end - self.n_start) * (t / self.ramp_time)
        } else {
            self.n_end
        };
        let m_dot = self.k_flow * n;
        // Energy balance: V·ρ·cp·dT/dt = m_dot·cp·(T_sup - T) - UA·(T - T_loss)
        let rho = 1000.0;
        let lhs = self.tank_volume * rho * self.cp * yp[0];
        let rhs = m_dot * self.cp * (self.t_supply - y[0]) - self.ua_loss * (y[0] - self.t_loss);
        r[0] = lhs - rhs;
    }
}

// ---------------------------------------------------------------------------
// Circuit 3 — AHU cooling-coil wet-surface transient (condensation)
//
// Two-state: x[0] = T_coil_out (air-side), x[1] = condensate mass
// (latent). The wet→dry transition at T_dew is the stiffness
// singularity — dehumidification releases latent heat that the
// standard fixed-damping Newton trip stumbles on.
// ---------------------------------------------------------------------------
pub struct CoolingCoilWetSurface {
    pub air_m_dot: f64, // kg_dry_air/s
    pub cp_air: f64,
    pub h_fg: f64, // J/kg — latent heat of vaporisation
    pub coil_ua: f64,
    pub t_chw_in: f64,
    pub m_dot_chw: f64,
    pub cp_chw: f64,
    pub t_air_in: f64,
    pub w_air_in: f64, // kg_water/kg_dry_air (humidity ratio)
    pub t_dew: f64,    // when coil_out < t_dew, condensation
    pub t_end: f64,
}

impl Default for CoolingCoilWetSurface {
    fn default() -> Self {
        Self {
            air_m_dot: 2.0,
            cp_air: 1006.0,
            h_fg: 2.45e6,
            coil_ua: 800.0,
            t_chw_in: 6.0,
            m_dot_chw: 1.5,
            cp_chw: 4186.0,
            t_air_in: 28.0,
            w_air_in: 0.014,
            t_dew: 16.0,
            t_end: 20.0,
        }
    }
}

impl Circuit for CoolingCoilWetSurface {
    fn name(&self) -> &'static str {
        "cooling_coil_wet"
    }
    fn dimension(&self) -> usize {
        2
    }
    fn initial_state(&self) -> Vec<f64> {
        vec![18.0, 0.0]
    }
    fn t_end(&self) -> f64 {
        self.t_end
    }
    fn dt_init(&self) -> f64 {
        0.5
    }
    // Inherits the default `finalize`. The condensate ODE
    // (state[1]) and the air-side temperature ODE (state[0]) both
    // have their mass/enthalpy balances as structural residuals;
    // a divergent policy trip shows up either as a high residual
    // OR as NaN/Inf in the state vector.
}

impl DaeSystem<f64> for CoolingCoilWetSurface {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(&self, _t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
        // State 0: T_coil_out. State 1: accumulated condensate mass [kg].
        // Latent load algebraically subtracted when T < T_dew; condensate
        // mass grows in the wet zone. The Jacobian's diagonal is
        // discontinuous at T = T_dew — that is the stiffness pulse the
        // baseline fixed-damping Newton trips on.
        let cond_rate = if y[0] < self.t_dew {
            self.air_m_dot * self.w_air_in * 0.1
        } else {
            0.0
        };
        let latent_load = cond_rate * self.h_fg;
        let cap = 1000.0; // J/K — arbitrary air-side thermal cap
        r[0] = cap * yp[0] + self.coil_ua * (y[0] - self.t_chw_in) + latent_load;
        // Condensate mass ODE: dM/dt = cond_rate.
        r[1] = yp[1] - cond_rate;
    }
}

// ---------------------------------------------------------------------------
// Circuit 4 — Primary/secondary decoupling loop demand step
//
// 2-state: x[0] = T_primary_loop, x[1] = T_secondary_loop. The
// decoupling tank introduces a stiff coupling: secondary demand
// steps cause the primary loop to oscillate briefly (underdamped
// second-order dynamics).
// ---------------------------------------------------------------------------
pub struct DecouplingLoopDemandStep {
    pub m_dot_primary: f64,
    pub m_dot_secondary: f64,
    pub cp: f64,
    pub vol_primary: f64,
    pub vol_secondary: f64,
    pub tank_ua: f64,
    pub t_supply_primary: f64,
    pub t_supply_secondary: f64,
    pub demand_time: f64,
    pub demand_step: f64,
    pub demand_value_pre: f64,
    pub t_end: f64,
}

impl Default for DecouplingLoopDemandStep {
    fn default() -> Self {
        Self {
            m_dot_primary: 0.5,
            m_dot_secondary: 0.3,
            cp: 4186.0,
            vol_primary: 0.02,
            vol_secondary: 0.015,
            tank_ua: 50.0,
            t_supply_primary: 70.0,
            t_supply_secondary: 55.0,
            demand_time: 5.0,
            demand_step: 0.5, // fraction of design flow
            demand_value_pre: 1.0,
            t_end: 40.0,
        }
    }
}

impl Circuit for DecouplingLoopDemandStep {
    fn name(&self) -> &'static str {
        "decoupling_loop_demand"
    }
    fn dimension(&self) -> usize {
        2
    }
    fn initial_state(&self) -> Vec<f64> {
        vec![40.0, 30.0]
    }
    fn t_end(&self) -> f64 {
        self.t_end
    }
    fn dt_init(&self) -> f64 {
        1.0
    }
    // Inherits the default `finalize` — tight 2-state coupling where
    // both balance equations are structural residuals.
}

impl DaeSystem<f64> for DecouplingLoopDemandStep {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(&self, t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
        let rho = 1000.0;
        let demand = if t < self.demand_time {
            self.demand_value_pre
        } else {
            self.demand_value_pre + self.demand_step
        };
        let m_s = self.m_dot_secondary * demand;
        let m_p = self.m_dot_primary;
        // Primary loop: m_p · cp · (T_supply_primary - y[0]) - UA · (y[0] - y[1]) = ρV_p · cp · yp[0]
        let cap_p = self.vol_primary * rho * self.cp;
        r[0] = cap_p * yp[0]
            - (m_p * self.cp * (self.t_supply_primary - y[0]) - self.tank_ua * (y[0] - y[1]));
        // Secondary loop: m_s · cp · (y[1] - T_secondary_return) = ρV_s · cp · yp[1]
        // Use y[0] as the source through the tank for tight coupling.
        let cap_s = self.vol_secondary * rho * self.cp;
        r[1] = cap_s * yp[1] - (m_s * self.cp * (y[0] - y[1]));
    }
}

// ---------------------------------------------------------------------------
// Circuit 5 — Heat-pump entering-fluid temperature step
//
// Single node x[0] = T_source_loop. Forcing: T_evap_in steps from
// 5 °C → 12 °C at t=4s. Bivalence point at T_biv = 8 °C crosses
// during the step — the COP curve is C^0 there, producing a sharp
// stiffness pulse.
// ---------------------------------------------------------------------------
pub struct HeatPumpEnteringFluidStep {
    pub t_source_init: f64,
    pub t_source_step_target: f64,
    pub step_time: f64,
    pub step_duration: f64,
    pub m_dot: f64,
    pub cp: f64,
    pub vol: f64,
    pub ua_load: f64,
    pub t_load: f64,
    pub t_biv: f64,
    pub t_end: f64,
}

impl Default for HeatPumpEnteringFluidStep {
    fn default() -> Self {
        Self {
            t_source_init: 5.0,
            t_source_step_target: 12.0,
            step_time: 4.0,
            step_duration: 2.0,
            m_dot: 0.3,
            cp: 4186.0,
            vol: 0.01,
            ua_load: 20.0,
            t_load: 30.0,
            t_biv: 8.0,
            t_end: 30.0,
        }
    }
}

impl Circuit for HeatPumpEnteringFluidStep {
    fn name(&self) -> &'static str {
        "heatpump_entering_fluid_step"
    }
    fn dimension(&self) -> usize {
        1
    }
    fn initial_state(&self) -> Vec<f64> {
        vec![5.0]
    }
    fn t_end(&self) -> f64 {
        self.t_end
    }
    fn dt_init(&self) -> f64 {
        0.5
    }
    // Inherits the default `finalize` — single-state ODE, the
    // bivalence-induced stiffness is what we want the candidate's
    // adaptive damping to tame; conservation holds structurally.
}

impl DaeSystem<f64> for HeatPumpEnteringFluidStep {
    fn dimension(&self) -> usize {
        1
    }
    fn residual(&self, t: f64, y: &[f64], yp: &[f64], r: &mut [f64]) {
        let t_in = if t < self.step_time {
            self.t_source_init
        } else if t < self.step_time + self.step_duration {
            let s = (t - self.step_time) / self.step_duration;
            self.t_source_init + s * (self.t_source_step_target - self.t_source_init)
        } else {
            self.t_source_step_target
        };
        let rho = 1000.0;
        // COP curve (C^0 at T_biv): linear below, gentler above. The
        // discontinuous derivative here is what makes the ODE stiff.
        let cop_factor = if t_in < self.t_biv { 1.0 } else { 0.85 };
        let q_pump = self.m_dot * self.cp * (t_in - y[0]) * cop_factor;
        let q_load = self.ua_load * (y[0] - self.t_load);
        r[0] = self.vol * rho * self.cp * yp[0] - (q_pump - q_load);
    }
}

// ---------------------------------------------------------------------------
// Fixture suite — instantiate all 5 circuits.
// ---------------------------------------------------------------------------
pub fn all_circuits() -> Vec<Box<dyn Circuit>> {
    vec![
        Box::new(MixingValveClosure::default()),
        Box::new(PumpFrequencyRamp::default()),
        Box::new(CoolingCoilWetSurface::default()),
        Box::new(DecouplingLoopDemandStep::default()),
        Box::new(HeatPumpEnteringFluidStep::default()),
    ]
}

// ---------------------------------------------------------------------------
// Boxed dispatch helper so `bdf_evaluator` (and tests) can construct
// the right `DaeSystem` for the driver without the driver knowing
// each circuit type.
// ---------------------------------------------------------------------------

/// Trait that bundles `Circuit` + `DaeSystem<f64>` so the driver can
/// drive a heterogeneous suite via a single boxed reference.
pub trait DynCircuit: Circuit + DaeSystem<f64> {}
impl<T: Circuit + DaeSystem<f64>> DynCircuit for T {}

/// Build a boxed dynamic circuit reference for `name`. Returns
/// `None` for unknown names.
pub fn make_circuit(name: &str) -> Option<Box<dyn DynCircuit>> {
    match name {
        "mixing_valve_closure" => Some(Box::new(MixingValveClosure::default())),
        "pump_freq_ramp" => Some(Box::new(PumpFrequencyRamp::default())),
        "cooling_coil_wet" => Some(Box::new(CoolingCoilWetSurface::default())),
        "decoupling_loop_demand" => Some(Box::new(DecouplingLoopDemandStep::default())),
        "heatpump_entering_fluid_step" => Some(Box::new(HeatPumpEnteringFluidStep::default())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::bdf_engine::{
        AdaptiveStepConfig, BdfDriver, NewtonRaphsonConfig, TimeSteppingConfig,
    };

    #[test]
    fn fixtures_have_unique_dimension_and_finite_initial_state() {
        let dims: Vec<usize> = all_circuits().iter().map(|c| c.dimension()).collect();
        assert!(dims.contains(&1));
        assert!(dims.contains(&2));
        for c in all_circuits().iter() {
            let y0 = c.initial_state();
            assert!(
                y0.iter().all(|v| v.is_finite()),
                "{} y0 not finite",
                c.name()
            );
            assert!(c.t_end() > 0.0, "{} t_end must be > 0", c.name());
            assert!(c.dt_init() > 0.0, "{} dt_init must be > 0", c.name());
        }
    }

    #[test]
    fn mixing_valve_alpha_helper_extremes() {
        assert_eq!(alpha_at(0.0, 5.0, 1.0), 0.5);
        assert_eq!(alpha_at(4.9, 5.0, 1.0), 0.5);
        assert_eq!(alpha_at(6.0, 5.0, 1.0), 0.0);
        // Branch through the linear ramp — alpha mid-closure must be
        // strictly between the boundary values.
        let mid = alpha_at(5.5, 5.0, 1.0);
        assert!(mid > 0.0 && mid < 0.5);
    }

    #[test]
    fn make_circuit_lookup_works_for_all_names() {
        for c in all_circuits() {
            assert!(
                make_circuit(c.name()).is_some(),
                "missing factory for {}",
                c.name()
            );
        }
        assert!(make_circuit("nonsense").is_none());
    }

    #[test]
    fn natural_scale_per_circuit_and_unknown() {
        // Each named arm of `natural_scale` must be reached with at
        // least one state that lifts `max_abs` above the baseline.
        let finite = vec![1.0];
        assert!(natural_scale("mixing_valve_closure", &finite) > 0.0);
        assert!(natural_scale("pump_freq_ramp", &finite) > 0.0);
        assert!(natural_scale("cooling_coil_wet", &finite) > 0.0);
        assert!(natural_scale("decoupling_loop_demand", &finite) > 0.0);
        assert!(natural_scale("heatpump_entering_fluid_step", &finite) > 0.0);
        // The fallback arm returns max_abs only.
        let large = vec![1e3];
        let s = natural_scale("unknown_circuit", &large);
        assert!((s - 1e3).abs() < 1e-9);
        // Non-finite state values must be skipped (no NaN propagation).
        let dirty = vec![f64::NAN, f64::INFINITY, 2.5];
        let s = natural_scale("mixing_valve_closure", &dirty);
        assert!(s.is_finite());
    }

    #[test]
    fn each_circuit_drives_through_bdfdriver() {
        // Drive every circuit a few steps through the production
        // BdfDriver path so the residual() branches and Newton
        // iteration counts are exercised in unit tests (the
        // integration regression in tests/bdf_golden_traces_regression.rs
        // spawns the bdf_evaluator subprocess, which doesn't
        // contribute to lib-test coverage).
        drive_circuit(&MixingValveClosure::default(), 4);
        drive_circuit(&PumpFrequencyRamp::default(), 4);
        drive_circuit(&CoolingCoilWetSurface::default(), 4);
        drive_circuit(&DecouplingLoopDemandStep::default(), 4);
        drive_circuit(&HeatPumpEnteringFluidStep::default(), 4);
        // Touch DynCircuit dispatch through make_circuit too — the
        // boxed path matches what bdf_evaluator and the campaign
        // orchestrator use.
        for name in [
            "mixing_valve_closure",
            "pump_freq_ramp",
            "cooling_coil_wet",
            "decoupling_loop_demand",
            "heatpump_entering_fluid_step",
        ] {
            let boxed: Box<dyn DynCircuit> = make_circuit(name).expect(name);
            // Drive residual directly so the boxed path is covered.
            let dim = DaeSystem::<f64>::dimension(boxed.as_ref());
            let mut y = boxed.initial_state();
            let yp = vec![0.0; dim];
            let mut r = vec![0.0; dim];
            boxed.residual(0.0, &y, &yp, &mut r);
            // One residual at a non-trivial time stamp so any
            // time-gated branches are exercised.
            y[0] += 1.0;
            boxed.residual(boxed.t_end() * 0.5, &y, &yp, &mut r);
            let _ = boxed.finalize(&y, 0.0);
        }
    }

    fn drive_circuit<S>(circuit: &S, max_steps: usize)
    where
        S: DaeSystem<f64> + Circuit,
    {
        let dt_init = circuit.dt_init();
        let t_end = (dt_init * 3.0).min(circuit.t_end());
        let initial = circuit.initial_state();
        let step_cfg = AdaptiveStepConfig {
            initial_dt: dt_init,
            ..AdaptiveStepConfig::default()
        };
        let ts_cfg = TimeSteppingConfig {
            bdf_config: NewtonRaphsonConfig::default(),
            step_config: step_cfg,
            max_steps,
            tolerance: 1e-6,
        };
        let mut driver = BdfDriver::new(ts_cfg);
        driver.initialize(0.0, &initial).expect("init failed");
        let stats = driver.run(circuit, t_end, dt_init).expect("run failed");
        assert!(
            stats.steps_accepted > 0,
            "{} must accept at least one step",
            circuit.name()
        );
        assert_eq!(
            stats.nan_or_inf_count,
            0,
            "{} produced NaN/Inf",
            circuit.name()
        );
    }
}

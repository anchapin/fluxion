// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Sub-hourly nodal temperature traces for the 9R4C multi-node solver.
//!
//! Issue #1800 (T9.6): Exposes the 9R4C node state evolution at every
//! timestep to Node.js consumers via NAPI, with a canonical solver trace
//! that the Python binding (T9.5) will share.
//!
//! The canonical entry point is [`run_sub_hourly_nodal_trace`]. It steps a
//! [`MultiNodeSolver`] forward by `timesteps` sub-steps of `dt_seconds`
//! duration and records the four mass-node temperatures plus the computed
//! zone air temperature after each step. The function is deterministic
//! (no clock, no thread spawns), so the NAPI and Python bindings can both
//! call it and the resulting traces are bit-identical for identical inputs.
//!
//! # Layer separation
//!
//! This module is a thin dependency-light wrapper over
//! [`crate::physics::multi_node_solver::MultiNodeSolver`]; it adds no new
//! state to the solver itself. The trace is allocated up front (one
//! `Vec<f64>` per series) so the inner loop performs no heap allocations
//! and can be hot-pathed into ML feature pipelines.
//!
//! The node temperature series produced here are exactly the 9R4C node
//! states: `wall`, `roof`, `floor`, `internal`. The `zone` series is the
//! zone air temperature computed from the envelope mass states via
//! [`MultiNodeSolver::compute_zone_air_temperature`] with the configured
//! [`MassAirCouplingMode`].
//!
//! # Coupling mode parity
//!
//! Both `MassAirCouplingMode::AdditiveSum` (default, backward-compatible)
//! and `MassAirCouplingMode::ParallelResistance` (#1281) are supported.
//! The same [`MassAirCouplingMode`] is used for the step and the zone air
//! temperature computation, so the trace is internally consistent.

use crate::physics::multi_node_solver::{MultiNodeSolver, SurfaceExteriorTemperatures};
use fluxion_core::multi_node::{MassAirCouplingMode, ThermalMassNode};

/// Configuration for a sub-hourly nodal temperature trace run.
///
/// This is the language-neutral input struct; both the NAPI and (future)
/// Python bindings construct one of these from their respective parameter
/// types and forward it to [`run_sub_hourly_nodal_trace`].
///
/// All fields are required; there are no hidden defaults. Callers that
/// want "9R4C defaults" should construct via
/// [`NineR4CTraceConfig::defaults`].
#[derive(Clone, Debug)]
pub struct NineR4CTraceConfig {
    /// Timestep duration in seconds. Must be positive and finite.
    ///
    /// The 9R4C backward Euler solver tolerates arbitrarily small
    /// timesteps (it auto-substeps internally for stiff nodes), so
    /// callers may pass values like `60.0` (1 minute) or `300.0`
    /// (5 minutes) for sub-hourly resolution.
    pub dt_seconds: f64,

    /// Number of sub-steps to run. The output series will have exactly
    /// this length. Must be non-zero.
    pub timesteps: usize,

    /// Mass-to-air coupling mode. See [`MassAirCouplingMode`] for the
    /// two supported variants.
    pub coupling_mode: MassAirCouplingMode,

    /// Initial zone air temperature (°C). Also seeds the four mass nodes
    /// if [`NineR4CTraceConfig::initial_node_temperature`] is `None`.
    pub initial_zone_temperature: f64,

    /// Optional per-node initial temperatures. When `None`, all four
    /// mass nodes are seeded at [`NineR4CTraceConfig::initial_zone_temperature`].
    /// When `Some`, must provide exactly four values in the order
    /// `[wall, roof, floor, internal]`.
    pub initial_node_temperature: Option<[f64; 4]>,

    /// Per-surface exterior boundary temperatures (°C). Use
    /// [`SurfaceExteriorTemperatures::uniform`] for the legacy
    /// single-temperature case.
    pub surface_exterior_temperatures: SurfaceExteriorTemperatures,

    /// Interior surface-to-air conductance [W/K]. Typical ISO 13790
    /// value: `8.0` (1 / R_si with R_si = 0.13 m²K/W).
    pub h_tr_is: f64,

    /// Wall thermal mass node. Encapsulates initial temperature,
    /// capacitance, and per-surface conductances.
    pub wall: ThermalMassNode,
    /// Roof thermal mass node.
    pub roof: ThermalMassNode,
    /// Floor thermal mass node.
    pub floor: ThermalMassNode,
    /// Internal (furniture / partitions) thermal mass node.
    pub internal: ThermalMassNode,

    /// Per-timestep radiative/convective gains injected into the
    /// backward Euler update of each mass node. Length must match
    /// `timesteps`; each element is `(gains_wall, gains_roof,
    /// gains_floor, gains_internal)` in watts.
    ///
    /// If empty, the solver runs with zero gains (pure conduction
    /// response to the exterior boundary temperatures).
    pub gains: Vec<(f64, f64, f64, f64)>,
}

impl NineR4CTraceConfig {
    /// Construct a config with the documented 9R4C defaults.
    ///
    /// Capacitances and conductances match the values used by the
    /// NAPI `NineR4CConfig::new()` constructor (issue #1796 / T9.2)
    /// so the trace produced by this function and the trace produced
    /// through the existing NAPI class agree to the bit for the
    /// default case.
    pub fn defaults() -> Self {
        let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
        let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
        let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 1e6, 0.0, 0.0).with_h_tr_me(100.0);

        Self {
            dt_seconds: 3600.0,
            timesteps: 24,
            coupling_mode: MassAirCouplingMode::default(),
            initial_zone_temperature: 20.0,
            initial_node_temperature: None,
            surface_exterior_temperatures: SurfaceExteriorTemperatures::uniform(10.0),
            h_tr_is: 10.0,
            wall,
            roof,
            floor,
            internal,
            gains: Vec::new(),
        }
    }

    /// Build the configured [`MultiNodeSolver`].
    pub fn build_solver(&self) -> MultiNodeSolver {
        let mut solver = MultiNodeSolver::new_with_mode(
            self.h_tr_is,
            self.wall,
            self.roof,
            self.floor,
            self.internal,
            self.coupling_mode,
        );
        solver.coupling_mode = self.coupling_mode;
        solver.timestep_seconds = self.dt_seconds;
        solver.zone_temperature = self.initial_zone_temperature;
        solver.surface_temperature = self.initial_zone_temperature;
        solver.exterior_temperature = (self.surface_exterior_temperatures.t_ext_wall
            + self.surface_exterior_temperatures.t_ext_roof
            + self.surface_exterior_temperatures.t_ext_floor)
            / 3.0;
        solver.exterior_temperatures = self.surface_exterior_temperatures.clone();

        let node_initials = self
            .initial_node_temperature
            .unwrap_or([self.initial_zone_temperature; 4]);
        solver.mass.wall.temperature = node_initials[0];
        solver.mass.roof.temperature = node_initials[1];
        solver.mass.floor.temperature = node_initials[2];
        solver.mass.internal.temperature = node_initials[3];
        solver
    }
}

/// Sub-hourly nodal temperature trace output.
///
/// Five parallel series, each of length [`NineR4CNodalTrace::timesteps`],
/// indexed by sub-step (0..timesteps).
///
/// The series are stored in **separate** vectors (not a single
/// `[f64; 5]`) so the NAPI and Python bindings can hand each one off as
/// a typed array / NumPy view without an additional copy step.
#[derive(Clone, Debug, PartialEq)]
pub struct NineR4CNodalTrace {
    /// Number of sub-steps recorded (length of each per-node series).
    pub timesteps: usize,
    /// Sub-step duration in seconds.
    pub dt_seconds: f64,
    /// Mass-to-air coupling mode that produced this trace.
    pub coupling_mode: MassAirCouplingMode,

    /// Wall mass-node temperature after each sub-step [°C].
    pub wall: Vec<f64>,
    /// Roof mass-node temperature after each sub-step [°C].
    pub roof: Vec<f64>,
    /// Floor mass-node temperature after each sub-step [°C].
    pub floor: Vec<f64>,
    /// Internal mass-node temperature after each sub-step [°C].
    pub internal: Vec<f64>,
    /// Zone air temperature after each sub-step [°C], computed via
    /// [`MultiNodeSolver::compute_zone_air_temperature`] using the
    /// same coupling mode as the step.
    pub zone: Vec<f64>,
}

/// Canonical solver trace function. Issue #1800 acceptance criterion
/// "validated against the same solver trace" is satisfied by having both
/// the NAPI binding and the Python binding call this function.
///
/// The function is `pub` and dependency-light (only depends on
/// `MultiNodeSolver` and `fluxion_core::multi_node`) so it can be reused
/// from any bindings layer without a second copy of the stepping logic.
///
/// # Validation
///
/// This function refuses to run with non-positive `dt_seconds`, zero
/// `timesteps`, or a mismatched gains vector length; in those cases it
/// returns [`NineR4CTraceError`] without mutating any state.
///
/// # Determinism
///
/// The function is fully deterministic: it does not read the wall clock,
/// does not allocate after the initial Vec reservation, and does not
/// spawn threads. Two calls with identical configs produce bit-identical
/// [`NineR4CNodalTrace`]s (modulo IEEE-754 reproducibility of the
/// underlying solver, which is itself deterministic for fixed inputs).
pub fn run_sub_hourly_nodal_trace(
    config: &NineR4CTraceConfig,
) -> Result<NineR4CNodalTrace, NineR4CTraceError> {
    if !config.dt_seconds.is_finite() || config.dt_seconds <= 0.0 {
        return Err(NineR4CTraceError::InvalidDt(config.dt_seconds));
    }
    if config.timesteps == 0 {
        return Err(NineR4CTraceError::ZeroTimesteps);
    }
    if !config.gains.is_empty() && config.gains.len() != config.timesteps {
        return Err(NineR4CTraceError::GainsLengthMismatch {
            gains: config.gains.len(),
            timesteps: config.timesteps,
        });
    }

    let mut solver = config.build_solver();
    let n = config.timesteps;
    let mut wall = Vec::with_capacity(n);
    let mut roof = Vec::with_capacity(n);
    let mut floor = Vec::with_capacity(n);
    let mut internal = Vec::with_capacity(n);
    let mut zone = Vec::with_capacity(n);

    // We call step_with_gains even when gains are empty — it dispatches
    // to the same backward Euler path with zero gain terms. This keeps
    // the trace identical to a hand-stepped trace at the bit level.
    for t in 0..n {
        let (gw, gr, gf, gi) = config.gains.get(t).copied().unwrap_or((0.0, 0.0, 0.0, 0.0));
        solver.step_with_gains(
            config.dt_seconds,
            gw,
            gr,
            gf,
            gi,
            0.0,
            solver.exterior_temperature,
        );

        let t_zone = solver.compute_zone_air_temperature(
            solver.exterior_temperature,
            /* h_ve = */ 0.0,
            /* h_ve_night = */ 0.0,
            /* phi_ia = */ 0.0,
        );
        solver.zone_temperature = t_zone;

        wall.push(solver.wall_temperature());
        roof.push(solver.roof_temperature());
        floor.push(solver.floor_temperature());
        internal.push(solver.internal_temperature());
        zone.push(t_zone);
    }

    Ok(NineR4CNodalTrace {
        timesteps: n,
        dt_seconds: config.dt_seconds,
        coupling_mode: config.coupling_mode,
        wall,
        roof,
        floor,
        internal,
        zone,
    })
}

/// Errors produced by [`run_sub_hourly_nodal_trace`].
#[derive(Debug, Clone, PartialEq)]
pub enum NineR4CTraceError {
    /// `dt_seconds` was non-finite or non-positive.
    InvalidDt(f64),
    /// `timesteps` was zero.
    ZeroTimesteps,
    /// `gains.len() != timesteps`.
    GainsLengthMismatch {
        /// Number of gain entries supplied.
        gains: usize,
        /// Number of sub-steps requested.
        timesteps: usize,
    },
}

impl std::fmt::Display for NineR4CTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NineR4CTraceError::InvalidDt(dt) => write!(
                f,
                "Invalid dt_seconds ({dt}): must be a positive finite number"
            ),
            NineR4CTraceError::ZeroTimesteps => {
                write!(f, "timesteps must be greater than zero (got 0)")
            }
            NineR4CTraceError::GainsLengthMismatch { gains, timesteps } => write!(
                f,
                "gains vector length ({gains}) must equal timesteps ({timesteps})"
            ),
        }
    }
}

impl std::error::Error for NineR4CTraceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_default_config_runs_without_error() {
        let trace = run_sub_hourly_nodal_trace(&NineR4CTraceConfig::defaults())
            .expect("default config must succeed");
        assert_eq!(trace.timesteps, 24);
        assert_eq!(trace.wall.len(), 24);
        assert_eq!(trace.roof.len(), 24);
        assert_eq!(trace.floor.len(), 24);
        assert_eq!(trace.internal.len(), 24);
        assert_eq!(trace.zone.len(), 24);
    }

    #[test]
    fn trace_rejects_zero_timesteps() {
        let mut config = NineR4CTraceConfig::defaults();
        config.timesteps = 0;
        let err = run_sub_hourly_nodal_trace(&config).unwrap_err();
        assert_eq!(err, NineR4CTraceError::ZeroTimesteps);
    }

    #[test]
    fn trace_rejects_non_positive_dt() {
        let mut config = NineR4CTraceConfig::defaults();
        config.dt_seconds = -1.0;
        let err = run_sub_hourly_nodal_trace(&config).unwrap_err();
        assert!(matches!(err, NineR4CTraceError::InvalidDt(_)));

        config.dt_seconds = 0.0;
        let err = run_sub_hourly_nodal_trace(&config).unwrap_err();
        assert!(matches!(err, NineR4CTraceError::InvalidDt(_)));

        config.dt_seconds = f64::NAN;
        let err = run_sub_hourly_nodal_trace(&config).unwrap_err();
        assert!(matches!(err, NineR4CTraceError::InvalidDt(_)));
    }

    #[test]
    fn trace_rejects_gains_length_mismatch() {
        let mut config = NineR4CTraceConfig::defaults();
        config.timesteps = 10;
        config.gains = vec![(0.0, 0.0, 0.0, 0.0); 5];
        let err = run_sub_hourly_nodal_trace(&config).unwrap_err();
        assert!(matches!(
            err,
            NineR4CTraceError::GainsLengthMismatch {
                gains: 5,
                timesteps: 10
            }
        ));
    }

    #[test]
    fn trace_is_deterministic_across_runs() {
        let trace_a = run_sub_hourly_nodal_trace(&NineR4CTraceConfig::defaults())
            .expect("first run must succeed");
        let trace_b = run_sub_hourly_nodal_trace(&NineR4CTraceConfig::defaults())
            .expect("second run must succeed");
        assert_eq!(trace_a, trace_b);
    }

    #[test]
    fn trace_mass_node_temperatures_stay_finite() {
        let trace = run_sub_hourly_nodal_trace(&NineR4CTraceConfig::defaults())
            .expect("default config must succeed");
        for series in [
            &trace.wall,
            &trace.roof,
            &trace.floor,
            &trace.internal,
            &trace.zone,
        ] {
            for &v in series {
                assert!(v.is_finite(), "non-finite value in series: {v}");
            }
        }
    }

    #[test]
    fn trace_zone_temperature_stays_between_extremes() {
        // With gains = 0 and an exterior below the initial zone
        // temperature, the envelope should cool and the zone air
        // temperature should track towards the exterior but stay above
        // it (the floor node is coupled to the exterior too).
        let mut config = NineR4CTraceConfig::defaults();
        config.timesteps = 200;
        config.dt_seconds = 60.0;
        config.surface_exterior_temperatures = SurfaceExteriorTemperatures::uniform(0.0);
        config.initial_zone_temperature = 20.0;

        let trace = run_sub_hourly_nodal_trace(&config).expect("config must succeed");
        for &t_zone in &trace.zone {
            assert!(t_zone.is_finite());
            // The zone can briefly drift slightly above or below the
            // initial 20°C envelope temperature depending on which mass
            // nodes dominate. We assert only that it stays inside the
            // physical envelope [exterior, initial].
            assert!(
                t_zone <= 20.5,
                "zone {t_zone} exceeded initial 20°C after cooling"
            );
            assert!(
                t_zone >= -0.5,
                "zone {t_zone} dropped below exterior 0°C envelope"
            );
        }
    }

    #[test]
    fn trace_initial_node_temperatures_override_seed() {
        let mut config = NineR4CTraceConfig::defaults();
        config.timesteps = 1;
        config.dt_seconds = 60.0;
        config.initial_node_temperature = Some([25.0, 22.0, 18.0, 21.0]);
        let trace = run_sub_hourly_nodal_trace(&config).expect("config must succeed");

        // After a single 60s step the node temperatures have barely
        // moved from their initial seeds, so we can assert that the
        // initial seeding worked.
        assert!(
            trace.wall[0] > 24.0 && trace.wall[0] < 26.0,
            "wall: {}",
            trace.wall[0]
        );
        assert!(
            trace.roof[0] > 21.0 && trace.roof[0] < 23.0,
            "roof: {}",
            trace.roof[0]
        );
        assert!(
            trace.floor[0] > 17.0 && trace.floor[0] < 19.0,
            "floor: {}",
            trace.floor[0]
        );
        assert!(
            trace.internal[0] > 20.0 && trace.internal[0] < 22.0,
            "internal: {}",
            trace.internal[0]
        );
    }

    #[test]
    fn trace_parallel_resistance_mode_produces_different_zone_series() {
        // The two coupling modes disagree on how the envelope mass
        // nodes couple to the air node; the trace must reflect that
        // by yielding different zone-temperature series for identical
        // initial conditions.
        let mut additive = NineR4CTraceConfig::defaults();
        additive.coupling_mode = MassAirCouplingMode::AdditiveSum;
        additive.timesteps = 50;
        additive.dt_seconds = 60.0;
        let trace_additive = run_sub_hourly_nodal_trace(&additive).unwrap();

        let mut parallel = additive.clone();
        parallel.coupling_mode = MassAirCouplingMode::ParallelResistance;
        let trace_parallel = run_sub_hourly_nodal_trace(&parallel).unwrap();

        // Mass node series should be very close but not bit-identical
        // (the two modes share the same backward Euler formula on the
        // envelope nodes, but the per-surface T_s_k path inside the
        // parallel-resistance mode shifts the surface-temperature
        // boundary used by the wall/roof/floor update). Use a tight
        // but realistic tolerance.
        for i in 0..additive.timesteps {
            let dw = (trace_additive.wall[i] - trace_parallel.wall[i]).abs();
            assert!(dw < 1e-3, "wall series diverged at t={i}: {dw}");
        }

        // The zone air series MUST differ (this is the whole point of
        // the parallel-resistance formulation).
        let zone_diff = trace_additive
            .zone
            .iter()
            .zip(trace_parallel.zone.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            zone_diff > 1e-6,
            "zone series should differ between coupling modes (max diff {zone_diff})"
        );
    }

    #[test]
    fn trace_matches_canonical_solver_loop() {
        // The canonical solver trace must agree with a hand-rolled loop
        // that calls the public `MultiNodeSolver::step_with_gains` and
        // `compute_zone_air_temperature` APIs in the same order. If
        // this test ever fails, the NAPI / Python traces have drifted
        // out of sync with the underlying solver.
        let config = NineR4CTraceConfig::defaults();
        let trace = run_sub_hourly_nodal_trace(&config).expect("canonical trace must succeed");

        let mut solver = config.build_solver();
        let n = config.timesteps;
        let dt = config.dt_seconds;

        let mut expected_wall = Vec::with_capacity(n);
        let mut expected_roof = Vec::with_capacity(n);
        let mut expected_floor = Vec::with_capacity(n);
        let mut expected_internal = Vec::with_capacity(n);
        let mut expected_zone = Vec::with_capacity(n);

        for _ in 0..n {
            solver.step_with_gains(dt, 0.0, 0.0, 0.0, 0.0, 0.0, solver.exterior_temperature);
            let tz =
                solver.compute_zone_air_temperature(solver.exterior_temperature, 0.0, 0.0, 0.0);
            solver.zone_temperature = tz;
            expected_wall.push(solver.wall_temperature());
            expected_roof.push(solver.roof_temperature());
            expected_floor.push(solver.floor_temperature());
            expected_internal.push(solver.internal_temperature());
            expected_zone.push(tz);
        }

        assert_eq!(trace.wall, expected_wall);
        assert_eq!(trace.roof, expected_roof);
        assert_eq!(trace.floor, expected_floor);
        assert_eq!(trace.internal, expected_internal);
        assert_eq!(trace.zone, expected_zone);
    }

    #[test]
    fn trace_supports_5_minute_substeps() {
        // The headline use case is sub-hourly resolution. 5-minute
        // substeps over 24 hours is 288 steps; verify the trace
        // produces that many entries and stays finite.
        let mut config = NineR4CTraceConfig::defaults();
        config.dt_seconds = 300.0;
        config.timesteps = 288;
        let trace = run_sub_hourly_nodal_trace(&config).expect("5-min trace must succeed");
        assert_eq!(trace.wall.len(), 288);
        assert!(trace.wall.iter().all(|v| v.is_finite()));
        assert!(trace.zone.iter().all(|v| v.is_finite()));
    }
}

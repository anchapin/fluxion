// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI bindings for the 9R4C sub-hourly nodal temperature trace (issue #1800).
//!
//! Node parity with T9.5 ([`crate::python`] PyO3 binding — once it lands).
//! Both bindings are expected to delegate to
//! [`crate::physics::nine_r4c_nodal_trace::run_sub_hourly_nodal_trace`],
//! which is the canonical solver trace function. The unit test
//! `nine_r4c_nodal_trace_round_trip_matches_canonical_solver` below
//! asserts that the NAPI-returned trace is bit-identical to a direct
//! call into the canonical function for the same configuration.

use crate::physics::multi_node_solver::SurfaceExteriorTemperatures;
use crate::physics::nine_r4c_nodal_trace::{
    run_sub_hourly_nodal_trace as run_canonical_trace, NineR4CTraceConfig, NineR4CTraceError,
};
use fluxion_core::multi_node::{MassAirCouplingMode, ThermalMassNode};
use napi::bindgen_prelude::Float64Array;

/// JavaScript-accessible 9R4C nodal temperature tracer.
///
/// Mirrors the defaults of the T9.2 [`crate::napi::nine_r4c_config::NineR4CConfig`]
/// so a Node.js caller that previously configured a `NineR4CConfig` can
/// obtain the same temperature evolution by constructing a
/// `NineR4CNodalTracer` and calling [`NineR4CNodalTracer::run_sub_hourly_trace`]
/// with identical inputs.
///
/// # TypeScript Example
/// ```typescript
/// import { NineR4CNodalTracer } from '@fluxion/native';
///
/// const tracer = new NineR4CNodalTracer();
/// const trace = tracer.runSubHourlyTrace({
///   dtSeconds: 300.0,         // 5-minute sub-steps
///   timesteps: 288,           // 24 hours
///   couplingMode: 'additive_sum',
///   initialZoneTemperature: 20.0,
///   surfaceExteriorTemperatures: { tExtWall: 0, tExtRoof: 0, tExtFloor: 0 },
///   hTrIs: 10.0,
///   gains: [],
/// });
///
/// console.log(`Wall[0]   = ${trace.wall[0].toFixed(2)} °C`);
/// console.log(`Zone[287] = ${trace.zone[287].toFixed(2)} °C`);
/// ```
#[napi_derive::napi]
pub struct NineR4CNodalTracer {
    config: NineR4CTraceConfig,
}

#[napi_derive::napi]
impl NineR4CNodalTracer {
    /// Create a new `NineR4CNodalTracer` with documented 9R4C defaults.
    ///
    /// The defaults match [`NineR4CConfig::defaults`] (in
    /// [`crate::physics::nine_r4c_nodal_trace`]) and the T9.2
    /// `NineR4CConfig::new()` constructor so the resulting trace agrees
    /// bit-for-bit with the equivalent default `NineR4CConfig` driven
    /// via the per-step methods.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            config: NineR4CTraceConfig::defaults(),
        }
    }

    /// Replace the per-node thermal mass parameters. Mirrors the
    /// constructor arguments of `NineR4CConfig::from_surface_parameters`
    /// from T9.2; both bindings share the same ISO 13790 9R4C layout.
    ///
    /// # Arguments
    /// * `wall_cm`, `roof_cm`, `floor_cm`, `internal_cm` —
    ///   Thermal capacitances per node [J/K].
    /// * `wall_h_tr_ms`, `wall_h_tr_em`, `wall_h_tr_me`,
    ///   `roof_h_tr_ms`, `roof_h_tr_em`, `roof_h_tr_me`,
    ///   `floor_h_tr_ms`, `floor_h_tr_em`, `floor_h_tr_me` —
    ///   Per-surface conductances [W/K].
    /// * `internal_h_tr_me` — Furniture-to-envelope conductance [W/K].
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn configure_nodes(
        &mut self,
        wall_cm: f64,
        wall_h_tr_ms: f64,
        wall_h_tr_em: f64,
        wall_h_tr_me: f64,
        roof_cm: f64,
        roof_h_tr_ms: f64,
        roof_h_tr_em: f64,
        roof_h_tr_me: f64,
        floor_cm: f64,
        floor_h_tr_ms: f64,
        floor_h_tr_em: f64,
        floor_h_tr_me: f64,
        internal_cm: f64,
        internal_h_tr_ms: f64,
        internal_h_tr_em: f64,
        internal_h_tr_me: f64,
    ) {
        self.config.wall = ThermalMassNode::new(
            self.config.initial_zone_temperature,
            wall_cm,
            wall_h_tr_ms,
            wall_h_tr_em,
        )
        .with_h_tr_me(wall_h_tr_me);
        self.config.roof = ThermalMassNode::new(
            self.config.initial_zone_temperature,
            roof_cm,
            roof_h_tr_ms,
            roof_h_tr_em,
        )
        .with_h_tr_me(roof_h_tr_me);
        self.config.floor = ThermalMassNode::new(
            self.config.initial_zone_temperature,
            floor_cm,
            floor_h_tr_ms,
            floor_h_tr_em,
        )
        .with_h_tr_me(floor_h_tr_me);
        self.config.internal = ThermalMassNode::new(
            self.config.initial_zone_temperature,
            internal_cm,
            internal_h_tr_ms,
            internal_h_tr_em,
        )
        .with_h_tr_me(internal_h_tr_me);
    }

    /// Run the sub-hourly nodal temperature trace and return the
    /// per-node temperature series.
    ///
    /// # Arguments
    /// * `params` — Trace parameters:
    ///   - `dtSeconds`: sub-step duration in seconds (must be > 0).
    ///   - `timesteps`: number of sub-steps to record (must be > 0).
    ///   - `couplingMode`: `"additive_sum"` (default) or
    ///     `"parallel_resistance"` (#1281).
    ///   - `initialZoneTemperature`: zone air temperature used to seed
    ///     the four mass nodes [°C].
    ///   - `initialNodeTemperature`: optional `[wall, roof, floor, internal]`
    ///     seed vector; when omitted, all four nodes are seeded at
    ///     `initialZoneTemperature`.
    ///   - `surfaceExteriorTemperatures`: `{ tExtWall, tExtRoof, tExtFloor }`
    ///     [°C]. Defaults to `{0, 0, 0}`.
    ///   - `hTrIs`: zone-air-to-surface conductance [W/K].
    ///   - `gains`: optional array of per-timestep
    ///     `[gainsWall, gainsRoof, gainsFloor, gainsInternal]` gain
    ///     vectors in watts. Length must equal `timesteps` when provided.
    ///
    /// # Returns
    /// A [`NineR4CNodalTrace`] object exposing five typed arrays
    /// (`Float64Array`) of length `timesteps`:
    ///   - `wall`, `roof`, `floor`, `internal`: mass-node temperatures
    ///     [°C].
    ///   - `zone`: zone air temperature computed via the
    ///     same coupling mode used to step the solver [°C].
    ///
    /// # Throws
    /// A `napi::Error` if `dtSeconds` is non-finite or non-positive,
    /// `timesteps` is zero, or `gains.length !== timesteps`.
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn run_sub_hourly_trace(
        &mut self,
        params: NineR4CTraceParams,
    ) -> napi::bindgen_prelude::Result<NineR4CNodalTrace> {
        // Translate JS-friendly params into the language-neutral config.
        let coupling_mode = match params.coupling_mode.as_deref() {
            Some("parallel_resistance") => MassAirCouplingMode::ParallelResistance,
            _ => MassAirCouplingMode::AdditiveSum,
        };

        let (t_ext_wall, t_ext_roof, t_ext_floor) = params
            .surface_exterior_temperatures
            .map(|t| (t.t_ext_wall, t.t_ext_roof, t.t_ext_floor))
            .unwrap_or((0.0, 0.0, 0.0));

        let initial_node_temperature = params.initial_node_temperature.map(|arr| {
            // The JS-side contract is a 4-element array; pad with the
            // zone seed if the caller passed fewer entries (defensive).
            let zone_seed = params.initial_zone_temperature.unwrap_or(20.0);
            let mut it = arr.into_iter();
            [
                it.next().unwrap_or(zone_seed),
                it.next().unwrap_or(zone_seed),
                it.next().unwrap_or(zone_seed),
                it.next().unwrap_or(zone_seed),
            ]
        });

        let gains = params
            .gains
            .map(|g| {
                g.into_iter()
                    .map(|quad| (quad[0], quad[1], quad[2], quad[3]))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let config = NineR4CTraceConfig {
            dt_seconds: params.dt_seconds,
            timesteps: params.timesteps as usize,
            coupling_mode,
            initial_zone_temperature: params.initial_zone_temperature.unwrap_or(20.0),
            initial_node_temperature,
            surface_exterior_temperatures: SurfaceExteriorTemperatures {
                t_ext_wall,
                t_ext_roof,
                t_ext_floor,
            },
            h_tr_is: params.h_tr_is.unwrap_or(10.0),
            wall: self.config.wall,
            roof: self.config.roof,
            floor: self.config.floor,
            internal: self.config.internal,
            gains,
        };

        let trace = run_canonical_trace(&config).map_err(trace_error_to_napi)?;

        // Stash the configured mass nodes back so that repeated calls
        // (e.g. sweeping `dtSeconds`) reuse the same nodes without
        // re-requiring `configure_nodes`.
        self.config = config;

        Ok(NineR4CNodalTrace {
            timesteps: trace.timesteps as u32,
            dt_seconds: trace.dt_seconds,
            coupling_mode: coupling_mode_str(trace.coupling_mode).to_string(),
            wall: Float64Array::from(trace.wall),
            roof: Float64Array::from(trace.roof),
            floor: Float64Array::from(trace.floor),
            internal: Float64Array::from(trace.internal),
            zone: Float64Array::from(trace.zone),
        })
    }
}

impl Default for NineR4CNodalTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Plain-data view of the trace parameters. Used as the NAPI argument
/// for [`NineR4CNodalTracer::run_sub_hourly_trace`].
///
/// Optional fields default to the canonical 9R4C defaults when `None`
/// or omitted by the JavaScript caller.
#[napi_derive::napi(object)]
pub struct NineR4CTraceParams {
    /// Sub-step duration in seconds. Must be positive and finite.
    pub dt_seconds: f64,
    /// Number of sub-steps to record. Must be non-zero.
    pub timesteps: u32,
    /// Coupling mode string: `"additive_sum"` (default) or
    /// `"parallel_resistance"`.
    #[napi(js_name = "couplingMode")]
    pub coupling_mode: Option<String>,
    /// Initial zone air temperature [°C]. Defaults to `20.0`.
    #[napi(js_name = "initialZoneTemperature")]
    pub initial_zone_temperature: Option<f64>,
    /// Optional `[wall, roof, floor, internal]` seed vector [°C].
    /// Defaults to seeding all four nodes at `initialZoneTemperature`.
    #[napi(js_name = "initialNodeTemperature")]
    pub initial_node_temperature: Option<Vec<f64>>,
    /// Per-surface exterior boundary temperatures [°C]. Defaults to
    /// `{ tExtWall: 0, tExtRoof: 0, tExtFloor: 0 }`.
    #[napi(js_name = "surfaceExteriorTemperatures")]
    pub surface_exterior_temperatures: Option<ExteriorTemperatureSet>,
    /// Zone-air-to-surface conductance [W/K]. Defaults to `10.0`.
    #[napi(js_name = "hTrIs")]
    pub h_tr_is: Option<f64>,
    /// Optional array of length `timesteps`, each entry being
    /// `[gainsWall, gainsRoof, gainsFloor, gainsInternal]` in watts.
    /// Defaults to zero gains on every step.
    pub gains: Option<Vec<Vec<f64>>>,
}

/// Plain-data view of the three per-surface exterior boundary
/// temperatures.
#[napi_derive::napi(object)]
pub struct ExteriorTemperatureSet {
    #[napi(js_name = "tExtWall")]
    pub t_ext_wall: f64,
    #[napi(js_name = "tExtRoof")]
    pub t_ext_roof: f64,
    #[napi(js_name = "tExtFloor")]
    pub t_ext_floor: f64,
}

/// Sub-hourly nodal temperature trace returned to JavaScript.
///
/// All five series are returned as `Float64Array` (typed arrays) so
/// JavaScript can iterate or copy them without an additional JSON
/// round-trip.
#[napi_derive::napi]
pub struct NineR4CNodalTrace {
    /// Number of sub-steps recorded (length of each series).
    pub timesteps: u32,
    /// Sub-step duration in seconds.
    #[napi(js_name = "dtSeconds")]
    pub dt_seconds: f64,
    /// Coupling mode used for this trace.
    #[napi(js_name = "couplingMode")]
    pub coupling_mode: String,
    /// Wall mass-node temperatures after each sub-step [°C].
    pub wall: Float64Array,
    /// Roof mass-node temperatures after each sub-step [°C].
    pub roof: Float64Array,
    /// Floor mass-node temperatures after each sub-step [°C].
    pub floor: Float64Array,
    /// Internal mass-node temperatures after each sub-step [°C].
    pub internal: Float64Array,
    /// Zone air temperature after each sub-step [°C].
    pub zone: Float64Array,
}

impl std::fmt::Debug for NineR4CNodalTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NineR4CNodalTrace")
            .field("timesteps", &self.timesteps)
            .field("dt_seconds", &self.dt_seconds)
            .field("coupling_mode", &self.coupling_mode)
            .field("wall", &self.wall.to_vec())
            .field("roof", &self.roof.to_vec())
            .field("floor", &self.floor.to_vec())
            .field("internal", &self.internal.to_vec())
            .field("zone", &self.zone.to_vec())
            .finish()
    }
}

fn coupling_mode_str(mode: MassAirCouplingMode) -> &'static str {
    match mode {
        MassAirCouplingMode::AdditiveSum => "additive_sum",
        MassAirCouplingMode::ParallelResistance => "parallel_resistance",
    }
}

fn trace_error_to_napi(err: NineR4CTraceError) -> napi::bindgen_prelude::Error {
    napi::bindgen_prelude::Error::from_reason(err.to_string())
}

#[cfg(all(test, feature = "napi-bindings"))]
mod tests {
    use super::*;
    use crate::physics::nine_r4c_nodal_trace::NineR4CNodalTrace as PhysicsNodalTrace;

    /// NAPI-exposed trace must be bit-identical to a direct call into
    /// the canonical solver trace for identical inputs. This is the
    /// validation hook for the issue #1800 acceptance criterion
    /// "validated against the same solver trace used for Python".
    #[test]
    fn nine_r4c_nodal_trace_round_trip_matches_canonical_solver() {
        let mut tracer = NineR4CNodalTracer::new();

        let params = NineR4CTraceParams {
            dt_seconds: 300.0,
            timesteps: 48,
            coupling_mode: Some("additive_sum".to_string()),
            initial_zone_temperature: Some(20.0),
            initial_node_temperature: Some(vec![20.0, 20.0, 20.0, 20.0]),
            surface_exterior_temperatures: Some(ExteriorTemperatureSet {
                t_ext_wall: 5.0,
                t_ext_roof: 5.0,
                t_ext_floor: 5.0,
            }),
            h_tr_is: Some(10.0),
            gains: Some(vec![vec![0.0, 0.0, 0.0, 0.0]; 48]),
        };

        let napi_trace = tracer
            .run_sub_hourly_trace(params)
            .expect("NAPI trace must succeed for valid params");

        // Independently invoke the canonical trace function for the
        // same configuration. The two outputs MUST agree to the bit.
        let mut canonical_config = NineR4CTraceConfig::defaults();
        canonical_config.dt_seconds = 300.0;
        canonical_config.timesteps = 48;
        canonical_config.surface_exterior_temperatures = SurfaceExteriorTemperatures::uniform(5.0);
        canonical_config.initial_zone_temperature = 20.0;
        let canonical_trace: PhysicsNodalTrace =
            run_canonical_trace(&canonical_config).expect("canonical trace must succeed");

        assert_eq!(napi_trace.timesteps, canonical_trace.timesteps as u32);
        assert_eq!(napi_trace.dt_seconds, canonical_trace.dt_seconds);
        assert_eq!(napi_trace.coupling_mode, "additive_sum");

        let napi_wall: Vec<f64> = napi_trace.wall.to_vec();
        let napi_roof: Vec<f64> = napi_trace.roof.to_vec();
        let napi_floor: Vec<f64> = napi_trace.floor.to_vec();
        let napi_internal: Vec<f64> = napi_trace.internal.to_vec();
        let napi_zone: Vec<f64> = napi_trace.zone.to_vec();

        assert_eq!(napi_wall, canonical_trace.wall, "wall series mismatch");
        assert_eq!(napi_roof, canonical_trace.roof, "roof series mismatch");
        assert_eq!(napi_floor, canonical_trace.floor, "floor series mismatch");
        assert_eq!(
            napi_internal, canonical_trace.internal,
            "internal series mismatch"
        );
        assert_eq!(napi_zone, canonical_trace.zone, "zone series mismatch");
    }

    /// NAPI surface must reject non-positive `dt_seconds`. Mirrors the
    /// canonical trace's `InvalidDt` error path.
    #[test]
    fn napi_trace_rejects_zero_dt() {
        let mut tracer = NineR4CNodalTracer::new();
        let params = NineR4CTraceParams {
            dt_seconds: 0.0,
            timesteps: 4,
            coupling_mode: None,
            initial_zone_temperature: None,
            initial_node_temperature: None,
            surface_exterior_temperatures: None,
            h_tr_is: None,
            gains: None,
        };
        let err = tracer
            .run_sub_hourly_trace(params)
            .expect_err("zero dt must be rejected");
        assert!(err.to_string().contains("dt_seconds"));
    }

    /// NAPI surface must reject mismatched gains length.
    #[test]
    fn napi_trace_rejects_gains_length_mismatch() {
        let mut tracer = NineR4CNodalTracer::new();
        let params = NineR4CTraceParams {
            dt_seconds: 60.0,
            timesteps: 10,
            coupling_mode: None,
            initial_zone_temperature: None,
            initial_node_temperature: None,
            surface_exterior_temperatures: None,
            h_tr_is: None,
            gains: Some(vec![vec![0.0, 0.0, 0.0, 0.0]; 5]),
        };
        let err = tracer
            .run_sub_hourly_trace(params)
            .expect_err("gains length mismatch must be rejected");
        assert!(err.to_string().contains("gains vector length"));
    }

    /// Parallel-resistance coupling mode is wired through to the trace.
    #[test]
    fn napi_trace_parallel_resistance_mode_accepted() {
        let mut tracer = NineR4CNodalTracer::new();
        let params = NineR4CTraceParams {
            dt_seconds: 60.0,
            timesteps: 4,
            coupling_mode: Some("parallel_resistance".to_string()),
            initial_zone_temperature: Some(20.0),
            initial_node_temperature: None,
            surface_exterior_temperatures: Some(ExteriorTemperatureSet {
                t_ext_wall: 0.0,
                t_ext_roof: 0.0,
                t_ext_floor: 0.0,
            }),
            h_tr_is: Some(10.0),
            gains: None,
        };
        let trace = tracer
            .run_sub_hourly_trace(params)
            .expect("parallel-resistance trace must succeed");
        assert_eq!(trace.coupling_mode, "parallel_resistance");
        assert_eq!(trace.wall.to_vec().len(), 4);
    }
}

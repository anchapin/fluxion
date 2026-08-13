// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI bindings for NineR4CConfig - 9R4C thermal solver configuration.
//!
//! Exposes all internal configuration parameters of the 9R4C multi-node thermal
//! solver to JavaScript/TypeScript consumers, enabling Node.js parity with the
//! PyO3 exposure in T9.1.

use crate::physics::multi_node_solver::{MultiNodeSolver, SurfaceExteriorTemperatures};
use fluxion_core::multi_node::{MassAirCouplingMode, ThermalMassNode};

/// Plain-data view of a single thermal mass node (wall/roof/floor/internal).
///
/// Exposed to JavaScript as an object with named properties so the npm test
/// suite can read `config.wall.temperature`, `config.wall.capacitance`, etc.
#[napi_derive::napi(object)]
pub struct MassNode {
    /// Node temperature [°C].
    pub temperature: f64,
    /// Node thermal capacitance [J/K].
    pub capacitance: f64,
    /// Surface-to-mass conductance [W/K].
    #[napi(js_name = "hTrMs")]
    pub h_tr_ms: f64,
    /// Exterior-to-mass conductance [W/K].
    #[napi(js_name = "hTrEm")]
    pub h_tr_em: f64,
    /// Mass-to-envelope conductance [W/K].
    #[napi(js_name = "hTrMe")]
    pub h_tr_me: f64,
}

/// Optional constructor parameters for [`NineR4CConfig`].
///
/// All fields default to the canonical 9R4C defaults when `None` or omitted
/// by the JavaScript caller. Mirrors the structure accepted by the npm test
/// suite (issue #1796 / #2832).
#[napi_derive::napi(object)]
pub struct NineR4CConfigInit {
    /// Interior surface-to-indoor air conductance [W/K]. Defaults to 10.0.
    #[napi(js_name = "hTrIs")]
    pub h_tr_is: Option<f64>,
    /// Wall mass node. Defaults to `{ temperature: 20.0, capacitance: 5e6,
    /// hTrMs: 50.0, hTrEm: 20.0, hTrMe: 0.0 }`.
    pub wall: Option<MassNode>,
    /// Roof mass node. Defaults to `{ temperature: 20.0, capacitance: 3e6,
    /// hTrMs: 30.0, hTrEm: 15.0, hTrMe: 0.0 }`.
    pub roof: Option<MassNode>,
    /// Floor mass node. Defaults to `{ temperature: 20.0, capacitance: 2e6,
    /// hTrMs: 20.0, hTrEm: 10.0, hTrMe: 0.0 }`.
    pub floor: Option<MassNode>,
    /// Internal mass node. Defaults to `{ temperature: 20.0, capacitance: 1e6,
    /// hTrMs: 0.0, hTrEm: 0.0, hTrMe: 100.0 }`.
    pub internal: Option<MassNode>,
    /// Initial zone air temperature [°C]. Defaults to 20.0.
    #[napi(js_name = "zoneTemperature")]
    pub zone_temperature: Option<f64>,
    /// Initial surface temperature [°C]. Defaults to 20.0.
    #[napi(js_name = "surfaceTemperature")]
    pub surface_temperature: Option<f64>,
    /// Initial exterior air temperature [°C]. Defaults to 10.0.
    #[napi(js_name = "exteriorTemperature")]
    pub exterior_temperature: Option<f64>,
    /// Air-mass coupling mode: `"additive_sum"` (default) or
    /// `"parallel_resistance"`.
    #[napi(js_name = "couplingMode")]
    pub coupling_mode: Option<String>,
}

impl Default for NineR4CConfigInit {
    fn default() -> Self {
        Self {
            h_tr_is: None,
            wall: None,
            roof: None,
            floor: None,
            internal: None,
            zone_temperature: None,
            surface_temperature: None,
            exterior_temperature: None,
            coupling_mode: None,
        }
    }
}

impl NineR4CConfigInit {
    /// Resolve every `Option` against the canonical 9R4C defaults.
    fn resolve(self) -> ResolvedNineR4CConfig {
        fn resolve_node(override_node: Option<MassNode>, default: MassNode) -> MassNode {
            override_node.unwrap_or(default)
        }

        let wall = resolve_node(self.wall, MassNode {
            temperature: 20.0,
            capacitance: 5e6,
            h_tr_ms: 50.0,
            h_tr_em: 20.0,
            h_tr_me: 0.0,
        });
        let roof = resolve_node(self.roof, MassNode {
            temperature: 20.0,
            capacitance: 3e6,
            h_tr_ms: 30.0,
            h_tr_em: 15.0,
            h_tr_me: 0.0,
        });
        let floor = resolve_node(self.floor, MassNode {
            temperature: 20.0,
            capacitance: 2e6,
            h_tr_ms: 20.0,
            h_tr_em: 10.0,
            h_tr_me: 0.0,
        });
        let internal = resolve_node(self.internal, MassNode {
            temperature: 20.0,
            capacitance: 1e6,
            h_tr_ms: 0.0,
            h_tr_em: 0.0,
            h_tr_me: 100.0,
        });

        let coupling = match self.coupling_mode.as_deref() {
            Some("parallel_resistance") => MassAirCouplingMode::ParallelResistance,
            _ => MassAirCouplingMode::AdditiveSum,
        };

        ResolvedNineR4CConfig {
            h_tr_is: self.h_tr_is.unwrap_or(10.0),
            wall,
            roof,
            floor,
            internal,
            zone_temperature: self.zone_temperature.unwrap_or(20.0),
            surface_temperature: self.surface_temperature.unwrap_or(20.0),
            exterior_temperature: self.exterior_temperature.unwrap_or(10.0),
            coupling,
        }
    }
}

/// Resolved (no-Option) values for building a [`NineR4CConfig`].
struct ResolvedNineR4CConfig {
    h_tr_is: f64,
    wall: MassNode,
    roof: MassNode,
    floor: MassNode,
    internal: MassNode,
    zone_temperature: f64,
    surface_temperature: f64,
    exterior_temperature: f64,
    coupling: MassAirCouplingMode,
}

/// JavaScript-accessible 9R4C thermal solver configuration.
///
/// This class exposes all internal configuration parameters of the 9R4C multi-node
/// thermal solver, enabling Node.js consumers to configure and query the solver
/// with the same level of control as the Rust core.
#[napi_derive::napi]
pub struct NineR4CConfig {
    inner: MultiNodeSolver,
}

#[napi_derive::napi]
impl NineR4CConfig {
    /// Create a NineR4CConfig with optional parameters.
    ///
    /// When `params` is `None` (i.e. `new NineR4CConfig()`) the canonical 9R4C
    /// defaults are used. The full default set is documented on
    /// [`NineR4CConfigInit`].
    #[napi(constructor)]
    pub fn new(params: Option<NineR4CConfigInit>) -> Self {
        let resolved = params.unwrap_or_default().resolve();
        let wall = ThermalMassNode::new(
            resolved.wall.temperature,
            resolved.wall.capacitance,
            resolved.wall.h_tr_ms,
            resolved.wall.h_tr_em,
        );
        let wall = if resolved.wall.h_tr_me > 0.0 {
            wall.with_h_tr_me(resolved.wall.h_tr_me)
        } else {
            wall
        };

        let roof = ThermalMassNode::new(
            resolved.roof.temperature,
            resolved.roof.capacitance,
            resolved.roof.h_tr_ms,
            resolved.roof.h_tr_em,
        );
        let roof = if resolved.roof.h_tr_me > 0.0 {
            roof.with_h_tr_me(resolved.roof.h_tr_me)
        } else {
            roof
        };

        let floor = ThermalMassNode::new(
            resolved.floor.temperature,
            resolved.floor.capacitance,
            resolved.floor.h_tr_ms,
            resolved.floor.h_tr_em,
        );
        let floor = if resolved.floor.h_tr_me > 0.0 {
            floor.with_h_tr_me(resolved.floor.h_tr_me)
        } else {
            floor
        };

        let internal = ThermalMassNode::new(
            resolved.internal.temperature,
            resolved.internal.capacitance,
            resolved.internal.h_tr_ms,
            resolved.internal.h_tr_em,
        );
        let internal = if resolved.internal.h_tr_me > 0.0 {
            internal.with_h_tr_me(resolved.internal.h_tr_me)
        } else {
            internal
        };

        let mut solver =
            MultiNodeSolver::new_with_mode(resolved.h_tr_is, wall, roof, floor, internal, resolved.coupling);
        solver.zone_temperature = resolved.zone_temperature;
        solver.surface_temperature = resolved.surface_temperature;
        solver.exterior_temperature = resolved.exterior_temperature;
        solver.exterior_temperatures =
            SurfaceExteriorTemperatures::uniform(resolved.exterior_temperature);

        NineR4CConfig { inner: solver }
    }

    /// Create a NineR4CConfig from individual surface and coupling parameters.
    ///
    /// # Arguments
    /// * `h_tr_is` - Interior surface-to-indoor air conductance [W/K]
    /// * `wall_cm` - Wall node thermal capacitance [J/K]
    /// * `wall_h_tr_ms` - Wall surface-to-mass conductance [W/K]
    /// * `wall_h_tr_em` - Wall exterior-to-mass conductance [W/K]
    /// * `wall_h_tr_me` - Wall mass-to-envelope conductance [W/K]
    /// * `roof_cm` - Roof node thermal capacitance [J/K]
    /// * `roof_h_tr_ms` - Roof surface-to-mass conductance [W/K]
    /// * `roof_h_tr_em` - Roof exterior-to-mass conductance [W/K]
    /// * `roof_h_tr_me` - Roof mass-to-envelope conductance [W/K]
    /// * `floor_cm` - Floor node thermal capacitance [J/K]
    /// * `floor_h_tr_ms` - Floor surface-to-mass conductance [W/K]
    /// * `floor_h_tr_em` - Floor exterior-to-mass conductance [W/K]
    /// * `floor_h_tr_me` - Floor mass-to-envelope conductance [W/K]
    /// * `internal_cm` - Internal node thermal capacitance [J/K]
    /// * `internal_h_tr_me` - Internal mass-to-envelope conductance [W/K]
    /// * `zone_temperature` - Initial zone air temperature [°C]
    /// * `surface_temperature` - Initial surface temperature [°C]
    /// * `exterior_temperature` - Initial exterior air temperature [°C]
    /// * `coupling_mode` - Air-mass coupling mode: "additive_sum" or "parallel_resistance"
    #[napi(factory)]
    pub fn from_surface_parameters(
        h_tr_is: f64,
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
        internal_h_tr_me: f64,
        zone_temperature: f64,
        surface_temperature: f64,
        exterior_temperature: f64,
        coupling_mode: String,
    ) -> Self {
        let mut wall = ThermalMassNode::new(zone_temperature, wall_cm, wall_h_tr_ms, wall_h_tr_em);
        if wall_h_tr_me > 0.0 {
            wall = wall.with_h_tr_me(wall_h_tr_me);
        }

        let mut roof = ThermalMassNode::new(zone_temperature, roof_cm, roof_h_tr_ms, roof_h_tr_em);
        if roof_h_tr_me > 0.0 {
            roof = roof.with_h_tr_me(roof_h_tr_me);
        }

        let mut floor =
            ThermalMassNode::new(zone_temperature, floor_cm, floor_h_tr_ms, floor_h_tr_em);
        if floor_h_tr_me > 0.0 {
            floor = floor.with_h_tr_me(floor_h_tr_me);
        }

        let mut internal = ThermalMassNode::new(zone_temperature, internal_cm, 0.0, 0.0);
        if internal_h_tr_me > 0.0 {
            internal = internal.with_h_tr_me(internal_h_tr_me);
        }

        let coupling = match coupling_mode.as_str() {
            "parallel_resistance" => MassAirCouplingMode::ParallelResistance,
            _ => MassAirCouplingMode::AdditiveSum,
        };

        let mut solver =
            MultiNodeSolver::new_with_mode(h_tr_is, wall, roof, floor, internal, coupling);
        solver.zone_temperature = zone_temperature;
        solver.surface_temperature = surface_temperature;
        solver.exterior_temperature = exterior_temperature;
        solver.exterior_temperatures = SurfaceExteriorTemperatures::uniform(exterior_temperature);

        NineR4CConfig { inner: solver }
    }

    // ── Zone-level parameters ─────────────────────────────────────────

    #[napi(getter)]
    pub fn get_h_tr_is(&self) -> f64 {
        self.inner.h_tr_is
    }

    #[napi(setter)]
    pub fn set_h_tr_is(&mut self, value: f64) {
        self.inner.h_tr_is = value;
    }

    #[napi(getter)]
    pub fn get_zone_temperature(&self) -> f64 {
        self.inner.zone_temperature
    }

    #[napi(setter)]
    pub fn set_zone_temperature(&mut self, value: f64) {
        self.inner.zone_temperature = value;
    }

    #[napi(getter)]
    pub fn get_surface_temperature(&self) -> f64 {
        self.inner.surface_temperature
    }

    #[napi(setter)]
    pub fn set_surface_temperature(&mut self, value: f64) {
        self.inner.surface_temperature = value;
    }

    #[napi(getter)]
    pub fn get_exterior_temperature(&self) -> f64 {
        self.inner.exterior_temperature
    }

    #[napi(setter)]
    pub fn set_exterior_temperature(&mut self, value: f64) {
        self.inner.set_exterior_temperature(value);
    }

    // ── Per-surface exterior temperatures ──────────────────────────────

    #[napi(getter)]
    pub fn get_t_ext_wall(&self) -> f64 {
        self.inner.exterior_temperatures.t_ext_wall
    }

    #[napi(getter)]
    pub fn get_t_ext_roof(&self) -> f64 {
        self.inner.exterior_temperatures.t_ext_roof
    }

    #[napi(getter)]
    pub fn get_t_ext_floor(&self) -> f64 {
        self.inner.exterior_temperatures.t_ext_floor
    }

    #[napi]
    pub fn set_surface_exterior_temperatures(
        &mut self,
        t_ext_wall: f64,
        t_ext_roof: f64,
        t_ext_floor: f64,
    ) {
        self.inner
            .set_surface_exterior_temperatures(SurfaceExteriorTemperatures {
                t_ext_wall,
                t_ext_roof,
                t_ext_floor,
            });
    }

    // ── Timestep and solver state ────────────────────────────────────

    #[napi(getter)]
    pub fn get_timestep_seconds(&self) -> f64 {
        self.inner.timestep_seconds
    }

    #[napi(setter)]
    pub fn set_timestep_seconds(&mut self, value: f64) {
        self.inner.timestep_seconds = value;
    }

    #[napi(getter)]
    pub fn get_initialized(&self) -> bool {
        self.inner.initialized
    }

    #[napi(getter)]
    pub fn get_r_total(&self) -> f64 {
        self.inner.r_total
    }

    #[napi(getter)]
    pub fn get_r_se(&self) -> f64 {
        self.inner.r_se
    }

    #[napi(getter)]
    pub fn get_coupling_mode(&self) -> String {
        match self.inner.coupling_mode {
            MassAirCouplingMode::AdditiveSum => "additive_sum".to_string(),
            MassAirCouplingMode::ParallelResistance => "parallel_resistance".to_string(),
        }
    }

    #[napi]
    pub fn set_coupling_mode(&mut self, mode: String) {
        self.inner.coupling_mode = match mode.as_str() {
            "parallel_resistance" => MassAirCouplingMode::ParallelResistance,
            _ => MassAirCouplingMode::AdditiveSum,
        };
    }

    // ── Temperature accessors ────────────────────────────────────────

    #[napi(getter)]
    pub fn get_wall_temperature(&self) -> f64 {
        self.inner.wall_temperature()
    }

    #[napi(getter)]
    pub fn get_roof_temperature(&self) -> f64 {
        self.inner.roof_temperature()
    }

    #[napi(getter)]
    pub fn get_floor_temperature(&self) -> f64 {
        self.inner.floor_temperature()
    }

    #[napi(getter)]
    pub fn get_internal_temperature(&self) -> f64 {
        self.inner.internal_temperature()
    }

    #[napi(getter)]
    pub fn get_envelope_temperature(&self) -> f64 {
        self.inner.envelope_temperature()
    }

    // ── Conductance setters ───────────────────────────────────────────

    #[napi]
    pub fn set_wall_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.inner.set_wall_conductances(h_tr_em, h_tr_ms);
    }

    #[napi]
    pub fn set_roof_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.inner.set_roof_conductances(h_tr_em, h_tr_ms);
    }

    #[napi]
    pub fn set_floor_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.inner.set_floor_conductances(h_tr_em, h_tr_ms);
    }

    #[napi]
    pub fn set_internal_conductance(&mut self, h_tr_me: f64) {
        self.inner.set_internal_conductance(h_tr_me);
    }

    // ── Capacitance setters ──────────────────────────────────────────

    #[napi]
    pub fn set_wall_capacitance(&mut self, cm: f64) {
        self.inner.set_wall_capacitance(cm);
    }

    #[napi]
    pub fn set_roof_capacitance(&mut self, cm: f64) {
        self.inner.set_roof_capacitance(cm);
    }

    #[napi]
    pub fn set_floor_capacitance(&mut self, cm: f64) {
        self.inner.set_floor_capacitance(cm);
    }

    #[napi]
    pub fn set_internal_capacitance(&mut self, cm: f64) {
        self.inner.set_internal_capacitance(cm);
    }

    // ── Node parameter views ──────────────────────────────────────────
    // Returns a MassNode object so JavaScript can read `config.wall.temperature`,
    // `config.wall.hTrMs`, etc. (matches the npm test expectations, issue #1796).

    #[napi(getter)]
    pub fn get_wall(&self) -> MassNode {
        MassNode {
            temperature: self.inner.mass.wall.temperature,
            capacitance: self.inner.mass.wall.capacitance,
            h_tr_ms: self.inner.mass.wall.h_tr_ms,
            h_tr_em: self.inner.mass.wall.h_tr_em,
            h_tr_me: self.inner.mass.wall.h_tr_me,
        }
    }

    #[napi(getter)]
    pub fn get_roof(&self) -> MassNode {
        MassNode {
            temperature: self.inner.mass.roof.temperature,
            capacitance: self.inner.mass.roof.capacitance,
            h_tr_ms: self.inner.mass.roof.h_tr_ms,
            h_tr_em: self.inner.mass.roof.h_tr_em,
            h_tr_me: self.inner.mass.roof.h_tr_me,
        }
    }

    #[napi(getter)]
    pub fn get_floor(&self) -> MassNode {
        MassNode {
            temperature: self.inner.mass.floor.temperature,
            capacitance: self.inner.mass.floor.capacitance,
            h_tr_ms: self.inner.mass.floor.h_tr_ms,
            h_tr_em: self.inner.mass.floor.h_tr_em,
            h_tr_me: self.inner.mass.floor.h_tr_me,
        }
    }

    #[napi(getter)]
    pub fn get_internal(&self) -> MassNode {
        MassNode {
            temperature: self.inner.mass.internal.temperature,
            capacitance: self.inner.mass.internal.capacitance,
            h_tr_ms: self.inner.mass.internal.h_tr_ms,
            h_tr_em: self.inner.mass.internal.h_tr_em,
            h_tr_me: self.inner.mass.internal.h_tr_me,
        }
    }

    // ── Simulation methods ────────────────────────────────────────────

    #[napi]
    pub fn initialize_temperatures(&mut self, t_initial: f64) {
        self.inner.initialize_temperatures(t_initial);
    }

    #[napi]
    pub fn step(&mut self, dt: f64) {
        self.inner.step(dt);
    }

    #[napi]
    pub fn step_with_gains(
        &mut self,
        dt: f64,
        gains_wall: f64,
        gains_roof: f64,
        gains_floor: f64,
        gains_internal: f64,
        #[napi(default, ts_arg_type = "number")] h_ve_night: f64,
        #[napi(default, ts_arg_type = "number")] outdoor_temp: f64,
    ) {
        self.inner.step_with_gains(
            dt,
            gains_wall,
            gains_roof,
            gains_floor,
            gains_internal,
            h_ve_night,
            outdoor_temp,
        );
    }

    #[napi]
    pub fn compute_zone_air_temperature(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        h_ve_night: f64,
        phi_ia: f64,
    ) -> f64 {
        self.inner
            .compute_zone_air_temperature(t_outdoor, h_ve, h_ve_night, phi_ia)
    }

    #[napi]
    pub fn compute_hvac_demand(
        &self,
        t_air_free: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> f64 {
        self.inner
            .compute_hvac_demand(t_air_free, heating_setpoint, cooling_setpoint)
    }

    #[napi(getter)]
    pub fn get_effective_time_constant(&self) -> f64 {
        self.inner.effective_time_constant()
    }
}

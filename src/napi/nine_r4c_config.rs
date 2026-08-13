// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI bindings for NineR4CConfig - 9R4C thermal solver configuration.
//!
//! Exposes all internal configuration parameters of the 9R4C multi-node thermal
//! solver to JavaScript/TypeScript consumers, enabling Node.js parity with the
//! PyO3 exposure in T9.1.

use crate::physics::multi_node_solver::{MultiNodeSolver, SurfaceExteriorTemperatures};
use fluxion_core::multi_node::{MassAirCouplingMode, ThermalMassNode};

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
    /// Create a NineR4CConfig with default parameters.
    #[napi(constructor)]
    pub fn new() -> Self {
        let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
        let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
        let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 1e6, 0.0, 0.0).with_h_tr_me(100.0);

        let solver = MultiNodeSolver::new(10.0, wall, roof, floor, internal);
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
    // Returns [temperature, capacitance, h_tr_ms, h_tr_em, h_tr_me]

    #[napi(getter)]
    pub fn get_wall(&self) -> Vec<f64> {
        vec![
            self.inner.mass.wall.temperature,
            self.inner.mass.wall.capacitance,
            self.inner.mass.wall.h_tr_ms,
            self.inner.mass.wall.h_tr_em,
            self.inner.mass.wall.h_tr_me,
        ]
    }

    #[napi(getter)]
    pub fn get_roof(&self) -> Vec<f64> {
        vec![
            self.inner.mass.roof.temperature,
            self.inner.mass.roof.capacitance,
            self.inner.mass.roof.h_tr_ms,
            self.inner.mass.roof.h_tr_em,
            self.inner.mass.roof.h_tr_me,
        ]
    }

    #[napi(getter)]
    pub fn get_floor(&self) -> Vec<f64> {
        vec![
            self.inner.mass.floor.temperature,
            self.inner.mass.floor.capacitance,
            self.inner.mass.floor.h_tr_ms,
            self.inner.mass.floor.h_tr_em,
            self.inner.mass.floor.h_tr_me,
        ]
    }

    #[napi(getter)]
    pub fn get_internal(&self) -> Vec<f64> {
        vec![
            self.inner.mass.internal.temperature,
            self.inner.mass.internal.capacitance,
            self.inner.mass.internal.h_tr_ms,
            self.inner.mass.internal.h_tr_em,
            self.inner.mass.internal.h_tr_me,
        ]
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

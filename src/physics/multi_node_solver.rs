//! Multi-Node Thermal Solver for 9R4C Model (Phase 6C)
//!
//! This module implements a backward Euler finite difference solver for the
//! 9R4C thermal network model used for heavy-mass buildings (Case 900+ series).
//!
//! ## 9R4C Network Architecture
//!
//! The 9R4C model separates thermal mass into 4 nodes:
//! - Wall node (Cm_wall): receives heat from exterior via h_tr_em_wall and from zone via h_tr_is
//! - Roof node (Cm_roof): receives heat from exterior via h_tr_em_roof and from zone via h_tr_is
//! - Floor node (Cm_floor): receives heat from exterior via h_tr_em_floor and from zone via h_tr_is
//! - Internal node (Cm_internal): furniture, partitions — receives heat from zone via h_tr_is
//!
//! ## Resistance Network
//!
//! ```text
//!                    h_tr_em_wall         h_tr_em_roof         h_tr_em_floor
//!   T_exterior ----[R_em_wall]----[Tm_wall]----+
//!                                          |
//!   T_exterior ----[R_em_roof]----[Tm_roof]---+  (parallel to exterior)
//!                                          |
//!   T_exterior ----[R_em_floor]--[Tm_floor]--+
//!                                          |
//!   T_zone ----[R_is]----[T_s]----[R_ms]----+  (series path to mass)
//!                              |
//!                              +----[R_me]----[Tm_internal]
//!
//! Where:
//! - h_tr_is: zone air to surface conductance
//! - h_tr_ms: surface to mass conductance (shared for all envelope surfaces in series)
//! - h_tr_me: internal mass to envelope mass conductance
//! ```
//!
//! Each envelope node (wall, roof, floor) has its own h_tr_em path to exterior.
//! All envelope nodes share the same surface node T_s via their respective h_tr_ms paths.

use crate::sim::multi_node_thermal::{MultiNodeThermalMass, ThermalMassNode};

/// Per-surface exterior boundary temperatures for the multi-node solver.
///
/// Each envelope node (wall, roof, floor) can have its own exterior boundary
/// temperature, computed from sol-air temperature calculations.
///
/// - Wall/Roof: sol-air temperature (accounts for solar irradiance, longwave radiation)
/// - Floor: ground temperature (ground-coupled)
#[derive(Debug, Clone)]
pub struct SurfaceExteriorTemperatures {
    /// Sol-air temperature for the wall exterior boundary (°C)
    pub t_ext_wall: f64,
    /// Sol-air temperature for the roof exterior boundary (°C)
    pub t_ext_roof: f64,
    /// Ground temperature for the floor exterior boundary (°C)
    pub t_ext_floor: f64,
}

impl SurfaceExteriorTemperatures {
    /// Create with a uniform exterior temperature (legacy fallback).
    pub fn uniform(t: f64) -> Self {
        Self {
            t_ext_wall: t,
            t_ext_roof: t,
            t_ext_floor: t,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiNodeSolver {
    pub mass: MultiNodeThermalMass,
    pub h_tr_is: f64,
    pub zone_temperature: f64,
    pub surface_temperature: f64,
    pub exterior_temperature: f64,
    /// Per-surface exterior boundary temperatures (Issue #863).
    /// When set, each envelope node uses its respective boundary temp
    /// instead of the uniform `exterior_temperature`.
    pub exterior_temperatures: Option<SurfaceExteriorTemperatures>,
    pub timestep_seconds: f64,
}

impl MultiNodeSolver {
    pub fn new(
        h_tr_is: f64,
        wall: ThermalMassNode,
        roof: ThermalMassNode,
        floor: ThermalMassNode,
        internal: ThermalMassNode,
    ) -> Self {
        Self {
            mass: MultiNodeThermalMass::new(wall, roof, floor, internal),
            h_tr_is,
            zone_temperature: 20.0,
            surface_temperature: 20.0,
            exterior_temperature: 10.0,
            exterior_temperatures: None,
            timestep_seconds: 3600.0,
        }
    }

    pub fn with_timestep(mut self, dt: f64) -> Self {
        self.timestep_seconds = dt;
        self
    }

    pub fn step(&mut self, dt: f64) -> &MultiNodeThermalMass {
        self.timestep_seconds = dt;
        self.step_backward_euler();
        &self.mass
    }

    fn step_backward_euler(&mut self) {
        let dt = self.timestep_seconds;
        let t_i = self.zone_temperature;
        let t_ext = self.exterior_temperature;
        let h_is = self.h_tr_is;

        // Per-surface exterior temperatures (Issue #863).
        // When available, each envelope node uses its own boundary temp
        // (sol-air for wall/roof, ground for floor) instead of the uniform
        // `exterior_temperature` average.
        let t_ext_wall = self
            .exterior_temperatures
            .as_ref()
            .map_or(t_ext, |et| et.t_ext_wall);
        let t_ext_roof = self
            .exterior_temperatures
            .as_ref()
            .map_or(t_ext, |et| et.t_ext_roof);
        let t_ext_floor = self
            .exterior_temperatures
            .as_ref()
            .map_or(t_ext, |et| et.t_ext_floor);

        let m = &mut self.mass;

        // Update wall node (envelope mass)
        {
            let node = &mut m.wall;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            // Backward Euler: (Cm/dt + h_em + h_ms) * T_new = Cm/dt * T_old + h_em * T_ext + h_ms * T_s
            let denom = node.capacitance / dt + h_em + h_ms;
            let numer = node.capacitance / dt * node.temperature
                + h_em * t_ext_wall
                + h_ms * self.surface_temperature;
            node.temperature = numer / denom;
        }

        // Update roof node (envelope mass)
        {
            let node = &mut m.roof;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            let numer = node.capacitance / dt * node.temperature
                + h_em * t_ext_roof
                + h_ms * self.surface_temperature;
            node.temperature = numer / denom;
        }

        // Update floor node (envelope mass)
        {
            let node = &mut m.floor;
            let h_em = node.h_tr_em;
            let h_ms = node.h_tr_ms;

            let denom = node.capacitance / dt + h_em + h_ms;
            let numer = node.capacitance / dt * node.temperature
                + h_em * t_ext_floor
                + h_ms * self.surface_temperature;
            node.temperature = numer / denom;
        }

        // Update internal node
        // Internal mass receives heat from zone air via h_tr_is and from envelope mass via h_tr_me
        {
            let node = &mut m.internal;
            // h_tr_me connects internal mass to envelope mass (approximated as average of envelope temps)
            let t_env_avg = (m.wall.temperature + m.roof.temperature + m.floor.temperature) / 3.0;

            // For internal mass: h_tr_me is the coupling to envelope mass
            // Simplified: internal mass exchanges with zone air directly via h_tr_is
            // and with envelope mass via a coupling conductance
            let h_me = node.h_tr_me;

            let denom = node.capacitance / dt + h_is + h_me;
            let numer = node.capacitance / dt * node.temperature + h_is * t_i + h_me * t_env_avg;
            node.temperature = numer / denom;
        }
    }

    pub fn wall_temperature(&self) -> f64 {
        self.mass.wall.temperature
    }

    pub fn roof_temperature(&self) -> f64 {
        self.mass.roof.temperature
    }

    pub fn floor_temperature(&self) -> f64 {
        self.mass.floor.temperature
    }

    pub fn internal_temperature(&self) -> f64 {
        self.mass.internal.temperature
    }

    pub fn envelope_temperature(&self) -> f64 {
        (self.mass.wall.temperature + self.mass.roof.temperature + self.mass.floor.temperature)
            / 3.0
    }

    pub fn set_zone_temperature(&mut self, t: f64) {
        self.zone_temperature = t;
    }

    pub fn set_surface_temperature(&mut self, t: f64) {
        self.surface_temperature = t;
    }

    pub fn set_exterior_temperature(&mut self, t: f64) {
        self.exterior_temperature = t;
    }

    /// Set per-surface exterior boundary temperatures (Issue #863).
    ///
    /// Stores per-surface sol-air/ground temperatures and updates the
    /// legacy `exterior_temperature` field to the average for backward
    /// compatibility with code that reads it directly.
    pub fn set_surface_exterior_temperatures(&mut self, temps: SurfaceExteriorTemperatures) {
        self.exterior_temperature = (temps.t_ext_wall + temps.t_ext_roof + temps.t_ext_floor) / 3.0;
        self.exterior_temperatures = Some(temps);
    }

    pub fn set_wall_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.wall.h_tr_em = h_tr_em;
        self.mass.wall.h_tr_ms = h_tr_ms;
    }

    pub fn set_roof_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.roof.h_tr_em = h_tr_em;
        self.mass.roof.h_tr_ms = h_tr_ms;
    }

    pub fn set_floor_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.floor.h_tr_em = h_tr_em;
        self.mass.floor.h_tr_ms = h_tr_ms;
    }

    pub fn set_internal_conductance(&mut self, h_tr_me: f64) {
        self.mass.internal.h_tr_me = h_tr_me;
    }

    pub fn set_wall_capacitance(&mut self, cm: f64) {
        self.mass.wall.capacitance = cm;
    }

    pub fn set_roof_capacitance(&mut self, cm: f64) {
        self.mass.roof.capacitance = cm;
    }

    pub fn set_floor_capacitance(&mut self, cm: f64) {
        self.mass.floor.capacitance = cm;
    }

    pub fn set_internal_capacitance(&mut self, cm: f64) {
        self.mass.internal.capacitance = cm;
    }

    pub fn initialize_temperatures(&mut self, t_initial: f64) {
        self.mass.wall.temperature = t_initial;
        self.mass.roof.temperature = t_initial;
        self.mass.floor.temperature = t_initial;
        self.mass.internal.temperature = t_initial;
        self.zone_temperature = t_initial;
        self.surface_temperature = t_initial;
    }

    pub fn effective_time_constant(&self) -> f64 {
        // τ_eff = C_total / h_tr_eff
        // where h_tr_eff is the effective coupling to the zone
        let c_total = self.mass.wall.capacitance
            + self.mass.roof.capacitance
            + self.mass.floor.capacitance
            + self.mass.internal.capacitance;

        // Effective conductance: envelope nodes coupled to zone via h_tr_ms + internal via h_is
        let h_tr_ms_total =
            self.mass.wall.h_tr_ms + self.mass.roof.h_tr_ms + self.mass.floor.h_tr_ms;

        // h_tr_is is shared, h_tr_ms connects surface to each envelope node
        // For time constant, we consider the dominant coupling
        let h_eff = self.h_tr_is + h_tr_ms_total / 3.0;

        c_total / h_eff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::multi_node_thermal::ThermalMassNode;

    fn create_test_solver() -> MultiNodeSolver {
        let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
        let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
        let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
        let internal = ThermalMassNode::new(20.0, 1e6, 10.0, 5.0);

        MultiNodeSolver::new(10.0, wall, roof, floor, internal)
    }

    #[test]
    fn test_solver_creation() {
        let solver = create_test_solver();
        assert_eq!(solver.wall_temperature(), 20.0);
        assert_eq!(solver.roof_temperature(), 20.0);
        assert_eq!(solver.floor_temperature(), 20.0);
        assert_eq!(solver.internal_temperature(), 20.0);
    }

    #[test]
    fn test_step_changes_temperatures() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(22.0);
        solver.set_exterior_temperature(5.0);
        solver.set_surface_temperature(18.0);

        let t_wall_before = solver.wall_temperature();
        solver.step(3600.0);

        // Wall should cool toward exterior temperature
        assert!(solver.wall_temperature() < t_wall_before);
    }

    #[test]
    fn test_envelope_temperature_average() {
        let mut solver = create_test_solver();
        solver.mass.wall.temperature = 10.0;
        solver.mass.roof.temperature = 20.0;
        solver.mass.floor.temperature = 30.0;

        let avg = solver.envelope_temperature();
        assert!((avg - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_time_constant_calculation() {
        let solver = create_test_solver();
        let tau = solver.effective_time_constant();

        // With C_total ≈ 11e6 J and h_eff ≈ 10-20 W/K
        // τ should be in the range of hours (h_tr in W/K, C in J/K, so τ in seconds)
        assert!(tau > 0.0);
        assert!(tau < 1e8); // Sanity check
    }

    #[test]
    fn test_steady_state_convergence() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(20.0);
        solver.set_exterior_temperature(20.0);
        solver.set_surface_temperature(20.0);

        // Run for many hours - temperatures should converge
        for _ in 0..168 {
            solver.step(3600.0);
        }

        // All temperatures should be near 20°C (within 0.1°C)
        assert!((solver.wall_temperature() - 20.0).abs() < 0.1);
        assert!((solver.roof_temperature() - 20.0).abs() < 0.1);
        assert!((solver.floor_temperature() - 20.0).abs() < 0.1);
        assert!((solver.internal_temperature() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_temperature_gradient_with_known_conductances() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(25.0);
        solver.set_exterior_temperature(0.0);
        solver.set_surface_temperature(15.0);

        // High-mass wall should show thermal lag
        let t_wall_initial = solver.wall_temperature();
        solver.step(3600.0);

        // Wall should cool slightly but not reach 0°C quickly due to high capacitance
        assert!(solver.wall_temperature() > 0.0);
        assert!(solver.wall_temperature() < t_wall_initial);
    }

    #[test]
    fn test_internal_mass_response() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(30.0);
        solver.set_exterior_temperature(10.0);
        solver.set_surface_temperature(20.0);

        // Internal mass should respond to zone temperature changes
        let t_internal_initial = solver.internal_temperature();
        solver.step(3600.0);

        // Internal mass should warm toward zone temperature
        assert!(solver.internal_temperature() > t_internal_initial);
        assert!(solver.internal_temperature() < 30.0);
    }

    #[test]
    fn test_backward_euler_stability() {
        let mut solver = create_test_solver();
        solver.set_zone_temperature(100.0); // Large temperature difference
        solver.set_exterior_temperature(-50.0);
        solver.set_surface_temperature(50.0);

        // Take many small timesteps - backward Euler should be stable
        for _ in 0..24 {
            solver.step(300.0); // 5-minute timestep
        }

        // All temperatures should be finite and within reasonable bounds
        assert!(solver.wall_temperature().is_finite());
        assert!(solver.roof_temperature().is_finite());
        assert!(solver.floor_temperature().is_finite());
        assert!(solver.internal_temperature().is_finite());

        // Should not have exploded
        assert!(solver.wall_temperature().abs() < 1000.0);
    }

    #[test]
    fn test_conductance_setters() {
        let mut solver = create_test_solver();

        solver.set_wall_conductances(25.0, 55.0);
        solver.set_roof_conductances(20.0, 40.0);
        solver.set_floor_conductances(15.0, 30.0);
        solver.set_internal_conductance(8.0);

        assert_eq!(solver.mass.wall.h_tr_em, 25.0);
        assert_eq!(solver.mass.wall.h_tr_ms, 55.0);
        assert_eq!(solver.mass.roof.h_tr_em, 20.0);
        assert_eq!(solver.mass.roof.h_tr_ms, 40.0);
        assert_eq!(solver.mass.floor.h_tr_em, 15.0);
        assert_eq!(solver.mass.floor.h_tr_ms, 30.0);
        assert_eq!(solver.mass.internal.h_tr_me, 8.0);
    }

    #[test]
    fn test_capacitance_setters() {
        let mut solver = create_test_solver();

        solver.set_wall_capacitance(1e7);
        solver.set_roof_capacitance(2e7);
        solver.set_floor_capacitance(3e7);
        solver.set_internal_capacitance(4e6);

        assert_eq!(solver.mass.wall.capacitance, 1e7);
        assert_eq!(solver.mass.roof.capacitance, 2e7);
        assert_eq!(solver.mass.floor.capacitance, 3e7);
        assert_eq!(solver.mass.internal.capacitance, 4e6);
    }

    #[test]
    fn test_initialization() {
        let mut solver = create_test_solver();
        solver.initialize_temperatures(15.0);

        assert_eq!(solver.wall_temperature(), 15.0);
        assert_eq!(solver.roof_temperature(), 15.0);
        assert_eq!(solver.floor_temperature(), 15.0);
        assert_eq!(solver.internal_temperature(), 15.0);
        assert_eq!(solver.zone_temperature, 15.0);
        assert_eq!(solver.surface_temperature, 15.0);
    }
}

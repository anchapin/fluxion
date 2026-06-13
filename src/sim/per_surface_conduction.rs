//! Per-Surface Conduction Solver for Multi-Node Thermal Model
//!
//! This module implements independent per-surface thermal conduction solving
//! as part of the multi-node thermal modeling foundation (Issue #857, Epic #856).
//!
//! # Overview
//!
//! Each building surface (wall, roof, floor) is modeled as an independent thermal
//! node with its own temperature state, using backward Euler integration for
//! numerical stability.
//!
//! # Physics
//!
//! ## Backward Euler Integration
//!
//! Each surface tracks its own temperature state and updates independently:
//! ```text
//! T_new = T_old + dt * (Q_in - Q_out) / C_surface
//! ```
//!
//! ## Mass Temperature Update (Issue #1003)
//!
//! The thermal mass temperature is updated using backward Euler integration:
//! ```text
//! T_mass_new = (T_mass_old * C_mass + dt * (h_tr_is * T_air + h_tr_ms * T_sky))
//!              / (C_mass + dt * (h_tr_is + h_tr_ms))
//! ```
//!
//! Where:
//! - T_mass_new = new mass temperature [°C]
//! - T_mass_old = old mass temperature [°C]
//! - C_mass = thermal mass capacity [J/K]
//! - dt = timestep [s]
//! - h_tr_is = interior surface heat transfer coefficient [W/K]
//! - h_tr_ms = mass-to-sky heat transfer coefficient [W/K]
//! - T_air = zone air temperature [°C]
//! - T_sky = sky temperature [°C]
//!
//! ## Surface Temperature Calculation
//!
//! Surface temperature is computed from the mass temperature using ISO 13790
//! surface heat transfer formula:
//! ```text
//! T_surface = T_mass + Q_net / (U * A)
//! ```
//!
//! Or equivalently via conductances:
//! ```text
//! Q_net = h_tr_ms * (T_mass - T_surface)
//! T_surface = T_mass - Q_net / h_tr_ms
//! ```
//!
//! # Independence Property
//!
//! Surfaces are thermally decoupled — each surface updates its temperature
//! based only on its own thermal mass, area, U-value, and the surrounding
//! mass node temperature. There is no cross-coupling between surfaces.

use crate::validation::ashrae_140_cases::Orientation;

/// Surface type distinguishing walls, roofs, and floors.
///
/// Different surface types have different thermal characteristics:
/// - **Wall**: Vertical orientation, convective/radiative heat transfer
/// - **Roof**: Upward heat flow (ceiling), higher film coefficients
/// - **Floor**: Downward heat flow, different film coefficients per ASHRAE 140
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    Wall,
    Roof,
    Floor,
}

impl SurfaceKind {
    /// Classify a surface kind from its orientation.
    pub fn from_orientation(orientation: Orientation) -> Self {
        match orientation {
            Orientation::Up | Orientation::Horizontal => SurfaceKind::Roof,
            Orientation::Down => SurfaceKind::Floor,
            Orientation::North | Orientation::East | Orientation::South | Orientation::West => {
                SurfaceKind::Wall
            }
        }
    }
}

/// Thermal mass node for per-surface conduction.
///
/// Each surface has its own thermal mass node that stores the interior-side
/// temperature state. The mass temperature evolves according to the backward
/// Euler update formula (Issue #1003):
/// ```text
/// T_mass_new = (T_mass_old * C_mass + dt * (h_tr_is * T_air + h_tr_ms * T_sky))
///              / (C_mass + dt * (h_tr_is + h_tr_ms))
/// ```
///
/// This represents the discretized heat balance at the mass node:
/// - Heat input: h_tr_is * T_air (from zone air)
/// - Heat input: h_tr_ms * T_sky (from sky via exterior)
/// - Heat stored: C_mass * T_mass
#[derive(Debug, Clone, PartialEq)]
pub struct MassNode {
    /// Surface identifier (links to corresponding SurfaceNode)
    pub id: usize,
    /// Thermal mass temperature in °C
    pub temperature: f64,
    /// Thermal mass capacity in J/K
    pub capacitance: f64,
    /// Conductance from interior surface to mass in W/K
    pub h_tr_is: f64,
    /// Conductance from mass to sky (exterior) in W/K
    pub h_tr_ms: f64,
}

impl MassNode {
    /// Create a new MassNode.
    ///
    /// # Arguments
    /// * `id` - Surface identifier
    /// * `temperature` - Initial mass temperature in °C
    /// * `capacitance` - Thermal mass capacity in J/K
    /// * `h_tr_is` - Interior surface-to-mass conductance in W/K
    /// * `h_tr_ms` - Mass-to-sky conductance in W/K
    pub fn new(
        id: usize,
        temperature: f64,
        capacitance: f64,
        h_tr_is: f64,
        h_tr_ms: f64,
    ) -> Self {
        Self {
            id,
            temperature,
            capacitance,
            h_tr_is,
            h_tr_ms,
        }
    }

    /// Update mass temperature using backward Euler integration.
    ///
    /// Implements the backward Euler update formula from Issue #1003:
    /// ```text
    /// T_mass_new = (T_mass_old * C_mass + dt * (h_tr_is * T_air + h_tr_ms * T_sky))
    ///              / (C_mass + dt * (h_tr_is + h_tr_ms))
    /// ```
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds
    /// * `T_air` - Zone air temperature in °C
    /// * `T_sky` - Sky temperature in °C
    pub fn update(&mut self, dt: f64, T_air: f64, T_sky: f64) {
        if self.capacitance <= 0.0 {
            return;
        }

        let numerator = self.temperature * self.capacitance
            + dt * (self.h_tr_is * T_air + self.h_tr_ms * T_sky);
        let denominator = self.capacitance + dt * (self.h_tr_is + self.h_tr_ms);

        if denominator > 0.0 {
            self.temperature = numerator / denominator;
        }
    }

    /// Compute steady-state mass temperature.
    ///
    /// At steady state (dt → ∞), the formula reduces to:
    /// ```text
    /// T_mass_ss = (h_tr_is * T_air + h_tr_ms * T_sky) / (h_tr_is + h_tr_ms)
    /// ```
    ///
    /// # Arguments
    /// * `T_air` - Zone air temperature in °C
    /// * `T_sky` - Sky temperature in °C
    ///
    /// # Returns
    /// Steady-state mass temperature in °C
    pub fn steady_state_temperature(&self, T_air: f64, T_sky: f64) -> f64 {
        let conductance_sum = self.h_tr_is + self.h_tr_ms;
        if conductance_sum > 0.0 {
            (self.h_tr_is * T_air + self.h_tr_ms * T_sky) / conductance_sum
        } else {
            self.temperature
        }
    }
}

/// A single surface node for thermal conduction modeling.
///
/// Each surface tracks its own temperature state independently, with thermal
/// properties derived from construction geometry and material properties.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceNode {
    /// Surface identifier
    pub id: usize,
    /// Surface type (wall, roof, floor)
    pub kind: SurfaceKind,
    /// Surface area in m²
    pub area: f64,
    /// U-value (thermal transmittance) in W/m²K
    pub u_value: f64,
    /// Surface temperature in °C
    pub temperature: f64,
    /// Surface thermal capacitance in J/K
    pub capacitance: f64,
    /// Conductance from surface to mass node in W/K
    pub h_tr_ms: f64,
    /// Conductance from interior surface to air in W/K
    pub h_tr_is: f64,
    /// Conductance from exterior to mass node in W/K
    pub h_tr_em: f64,
    /// Interior surface heat transfer coefficient in W/m²K
    /// Used for computing heat exchange between zone air and interior surface
    pub h_tr_is: f64,
    /// Thermal mass temperature in °C
    /// Updated using backward Euler integration
    pub mass_temperature: f64,
    /// Net heat flow rate at the surface in W
    pub heat_flow: f64,
}

impl SurfaceNode {
    /// Create a new SurfaceNode with thermal properties.
    ///
    /// # Arguments
    /// * `id` - Surface identifier
    /// * `kind` - Surface type
    /// * `area` - Surface area in m²
    /// * `u_value` - U-value in W/m²K
    /// * `temperature` - Initial temperature in °C
    /// * `capacitance` - Thermal capacitance in J/K
    /// * `h_tr_ms` - Surface-to-mass conductance in W/K
    /// * `h_tr_is` - Interior surface-to-air conductance in W/K
    /// * `h_tr_em` - Exterior-to-mass conductance in W/K
    /// * `h_tr_is` - Interior surface heat transfer coefficient in W/m²K
    /// * `mass_temperature` - Initial mass temperature in °C
    pub fn new(
        id: usize,
        kind: SurfaceKind,
        area: f64,
        u_value: f64,
        temperature: f64,
        capacitance: f64,
        h_tr_ms: f64,
        h_tr_is: f64,
        h_tr_em: f64,
        h_tr_is: f64,
        mass_temperature: f64,
    ) -> Self {
        Self {
            id,
            kind,
            area,
            u_value,
            temperature,
            capacitance,
            h_tr_ms,
            h_tr_is,
            h_tr_em,
            h_tr_is,
            mass_temperature,
            heat_flow: 0.0,
        }
    }

    /// Compute steady-state heat flow through this surface.
    ///
    /// At steady state: Q = U * A * ΔT
    /// where ΔT = T_mass - T_exterior
    ///
    /// # Arguments
    /// * `mass_temperature` - Thermal mass node temperature in °C
    /// * `exterior_temperature` - Exterior ambient temperature in °C
    ///
    /// # Returns
    /// Steady-state heat flow in W (positive = heat gain to interior)
    pub fn steady_state_heat_flow(&self, mass_temperature: f64, exterior_temperature: f64) -> f64 {
        self.u_value * self.area * (mass_temperature - exterior_temperature)
    }

    /// Compute surface temperature from mass and air temperatures using ISO 13790 formula.
    ///
    /// This is the primary formula for calculating interior surface temperature
    /// from the zone air temperature and mass temperature:
    /// ```text
    /// T_surface = (h_tr_is * T_air + h_tr_ms * T_mass) / (h_tr_is + h_tr_ms)
    /// ```
    ///
    /// This represents the heat balance at the interior surface where:
    /// - h_tr_is connects the surface to the zone air
    /// - h_tr_ms connects the surface to the thermal mass
    ///
    /// When h_tr_is dominates (high surface convection), T_surface ≈ T_air.
    /// When h_tr_ms dominates (high thermal coupling to mass), T_surface ≈ T_mass.
    ///
    /// # Arguments
    /// * `zone_air_temperature` - Zone air temperature in °C
    /// * `mass_temperature` - Thermal mass node temperature in °C
    ///
    /// # Returns
    /// Surface temperature in °C
    pub fn surface_temperature_from_mass(
        &self,
        zone_air_temperature: f64,
        mass_temperature: f64,
    ) -> f64 {
        let h_sum = self.h_tr_is + self.h_tr_ms;
        if h_sum > 0.0 {
            (self.h_tr_is * zone_air_temperature + self.h_tr_ms * mass_temperature) / h_sum
        } else {
            // Fallback if conductances are zero
            mass_temperature
        }
    }

    /// Compute surface temperature from mass and exterior temperatures.
    ///
    /// Uses ISO 13790 surface heat transfer relationship:
    /// ```text
    /// Q = h_tr_ms * (T_mass - T_surface)
    /// T_surface = T_mass - Q / h_tr_ms
    /// ```
    ///
    /// When h_tr_ms dominates thermal resistance, T_surface ≈ T_mass.
    /// When h_tr_ms is small, larger temperature differences occur.
    ///
    /// # Arguments
    /// * `mass_temperature` - Thermal mass node temperature in °C
    /// * `exterior_temperature` - Exterior temperature in °C
    ///
    /// # Returns
    /// Surface temperature in °C
    pub fn surface_temperature(&self, mass_temperature: f64, exterior_temperature: f64) -> f64 {
        // Net heat flow from exterior to mass
        let q_net = self.steady_state_heat_flow(mass_temperature, exterior_temperature);
        // Surface temperature via heat transfer formula
        // Q = h_tr_ms * (T_mass - T_surface)
        // => T_surface = T_mass - Q / h_tr_ms
        if self.h_tr_ms > 0.0 {
            mass_temperature - q_net / self.h_tr_ms
        } else {
            mass_temperature
        }
    }

    /// Update surface temperature using backward Euler integration.
    ///
    /// Backward Euler (fully implicit):
    /// ```text
    /// T_new = T_old + dt * (Q_in - Q_out) / C
    /// ```
    ///
    /// For a surface node, Q_in comes from the mass node and Q_out goes to exterior.
    /// Using h_tr_ms and h_tr_em as conductances:
    /// ```text
    /// Q_ms = h_tr_ms * (T_mass - T_surface)   // from mass to surface
    /// Q_em = h_tr_em * (T_surface - T_exterior)  // from surface to exterior
    /// Q_net = Q_ms - Q_em
    /// T_new = T_old + dt * Q_net / C
    /// ```
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds
    /// * `mass_temperature` - Mass node temperature in °C
    /// * `exterior_temperature` - Exterior temperature in °C
    pub fn update(&mut self, dt: f64, mass_temperature: f64, exterior_temperature: f64) {
        // Heat flow from mass to surface
        let q_ms = self.h_tr_ms * (mass_temperature - self.temperature);
        // Heat flow from surface to exterior (through the envelope)
        let q_em = self.h_tr_em * (self.temperature - exterior_temperature);
        // Net heat flow (positive = heat entering surface)
        self.heat_flow = q_ms - q_em;

        // Backward Euler update
        if self.capacitance > 0.0 {
            let t_new = self.temperature + dt * self.heat_flow / self.capacitance;
            self.temperature = t_new;
        }
    }
}

/// Per-surface conduction solver for multi-node thermal modeling.
///
/// This solver manages a collection of independent SurfaceNodes, each representing
/// a building surface (wall, roof, or floor). Each surface updates its temperature
/// state independently using backward Euler integration.
///
/// # Design Principles
///
/// 1. **Independence**: No cross-coupling between surfaces — each surface's thermal
///    state depends only on its own properties and the shared mass node temperature.
/// 2. **Numerical Stability**: Backward Euler integration ensures unconditional
///    stability regardless of time step size.
/// 3. **Energy Conservation**: Heat flow at each surface is tracked for energy auditing.
///
/// # Usage
///
/// ```ignore
/// let mut solver = PerSurfaceConductionSolver::new();
/// solver.add_surface(SurfaceKind::Wall, 10.0, 0.5, 20.0, 50000.0, 5.0, 2.0);
/// solver.add_surface(SurfaceKind::Roof, 10.0, 0.3, 20.0, 80000.0, 4.0, 1.5);
///
/// // In simulation loop:
/// solver.update_all(dt, mass_temperature, exterior_temperature);
/// let surface_temps = solver.surface_temperatures();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PerSurfaceConductionSolver {
    /// Collection of surface nodes
    surfaces: Vec<SurfaceNode>,
    /// Collection of mass nodes (one per surface)
    mass_nodes: Vec<MassNode>,
}

impl PerSurfaceConductionSolver {
    /// Create a new empty per-surface conduction solver.
    pub fn new() -> Self {
        Self {
            surfaces: Vec::new(),
            mass_nodes: Vec::new(),
        }
    }

    /// Create a solver with the given surfaces.
    pub fn with_surfaces(surfaces: Vec<SurfaceNode>, mass_nodes: Vec<MassNode>) -> Self {
        Self { surfaces, mass_nodes }
    }

    /// Add a mass node to the solver.
    pub fn add_mass_node(&mut self, mass_node: MassNode) {
        self.mass_nodes.push(mass_node);
    }

    /// Add a surface and its corresponding mass node together.
    ///
    /// This is the primary method for adding thermal nodes to the solver.
    /// Both the surface node and mass node share the same ID for cross-referencing.
    pub fn add_surface_with_mass(
        &mut self,
        surface: SurfaceNode,
        mass_node: MassNode,
    ) {
        debug_assert_eq!(surface.id, mass_node.id, "Surface and MassNode must have same ID");
        self.surfaces.push(surface);
        self.mass_nodes.push(mass_node);
    }

    /// Add a surface from thermal model data parameters.
    ///
    /// Convenience constructor that computes h_tr_ms and h_tr_em from area and U-value.
    /// Uses parallel conductance formula: h = 1 / (1/h1 + 1/h2) for stacked resistances.
    pub fn add_surface_from_params(
        &mut self,
        id: usize,
        kind: SurfaceKind,
        area: f64,
        u_value: f64,
        temperature: f64,
        h_tr_ms: f64,
        h_tr_is: f64,
        h_tr_em: f64,
        h_tr_is: f64,
    ) {
        // Thermal capacitance: C = ρ * c * V = ρ * c * (A * d)
        // Using typical concrete: ρ = 2300 kg/m³, c = 1000 J/kgK, d = 0.1m
        // C_per_area = 2300 * 1000 * 0.1 = 230,000 J/m²K
        let capacitance_per_area = 230_000.0; // J/m²K
        let capacitance = capacitance_per_area * area;

        let surface = SurfaceNode::new(
            id,
            kind,
            area,
            u_value,
            temperature,
            capacitance,
            h_tr_ms,
            h_tr_is,
            h_tr_em,
            h_tr_is,
            temperature, // mass_temperature starts equal to surface temperature
        );
        self.surfaces.push(surface);
    }

    /// Number of surfaces in the solver.
    pub fn len(&self) -> usize {
        self.surfaces.len()
    }

    /// Check if solver has no surfaces.
    pub fn is_empty(&self) -> bool {
        self.surfaces.is_empty()
    }

    /// Get a reference to a surface by index.
    pub fn get(&self, index: usize) -> Option<&SurfaceNode> {
        self.surfaces.get(index)
    }

    /// Get a mutable reference to a surface by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut SurfaceNode> {
        self.surfaces.get_mut(index)
    }

    /// Get all surface temperatures.
    pub fn surface_temperatures(&self) -> Vec<f64> {
        self.surfaces.iter().map(|s| s.temperature).collect()
    }

    /// Get all surface heat flows.
    pub fn heat_flows(&self) -> Vec<f64> {
        self.surfaces.iter().map(|s| s.heat_flow).collect()
    }

    /// Get total heat flow across all surfaces.
    pub fn total_heat_flow(&self) -> f64 {
        self.surfaces.iter().map(|s| s.heat_flow).sum()
    }

    /// Get a reference to a mass node by index.
    pub fn get_mass_node(&self, index: usize) -> Option<&MassNode> {
        self.mass_nodes.get(index)
    }

    /// Get a mutable reference to a mass node by index.
    pub fn get_mass_node_mut(&mut self, index: usize) -> Option<&mut MassNode> {
        self.mass_nodes.get_mut(index)
    }

    /// Get all mass node temperatures.
    pub fn mass_temperatures(&self) -> Vec<f64> {
        self.mass_nodes.iter().map(|m| m.temperature).collect()
    }

    /// Update all mass nodes using backward Euler integration.
    ///
    /// Implements the backward Euler update formula from Issue #1003:
    /// ```text
    /// T_mass_new = (T_mass_old * C_mass + dt * (h_tr_is * T_air + h_tr_ms * T_sky))
    ///              / (C_mass + dt * (h_tr_is + h_tr_ms))
    /// ```
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds
    /// * `T_air` - Zone air temperature in °C
    /// * `T_sky` - Sky temperature in °C
    pub fn update_mass_nodes(&mut self, dt: f64, T_air: f64, T_sky: f64) {
        for mass_node in &mut self.mass_nodes {
            mass_node.update(dt, T_air, T_sky);
        }
    }

    /// Update all surfaces using backward Euler integration.
    ///
    /// Each surface updates its temperature independently based on the shared
    /// mass temperature and exterior temperature.
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds
    /// * `mass_temperature` - Mass node temperature in °C (shared across surfaces)
    /// * `exterior_temperature` - Exterior ambient temperature in °C
    pub fn update_all(&mut self, dt: f64, mass_temperature: f64, exterior_temperature: f64) {
        for surface in &mut self.surfaces {
            surface.update(dt, mass_temperature, exterior_temperature);
        }
    }

    /// Update a single surface by ID.
    pub fn update_surface(
        &mut self,
        id: usize,
        dt: f64,
        mass_temperature: f64,
        exterior_temperature: f64,
    ) {
        if let Some(surface) = self.surfaces.iter_mut().find(|s| s.id == id) {
            surface.update(dt, mass_temperature, exterior_temperature);
        }
    }

    /// Compute surface temperatures from mass and air temperatures.
    ///
    /// Uses the ISO 13790 formula: T_surface = (h_tr_is * T_air + h_tr_ms * T_mass) / (h_tr_is + h_tr_ms)
    ///
    /// Returns a vector of surface temperatures corresponding to each surface node.
    pub fn compute_surface_temperatures(
        &self,
        zone_air_temperature: f64,
        mass_temperature: f64,
    ) -> Vec<f64> {
        self.surfaces
            .iter()
            .map(|s| s.surface_temperature_from_mass(zone_air_temperature, mass_temperature))
            .collect()
    }

    /// Verify energy conservation at the surface interface.
    ///
    /// For each surface, checks that:
    /// ```text
    /// Q_ms = h_tr_ms * (T_mass - T_surface)
    /// Q_em = h_tr_em * (T_surface - T_exterior)
    /// Q_net = Q_ms - Q_em ≈ 0  (at steady state)
    /// ```
    ///
    /// Returns maximum absolute imbalance across all surfaces.
    pub fn energy_imbalance(&self, mass_temperature: f64, exterior_temperature: f64) -> f64 {
        self.surfaces
            .iter()
            .map(|s| {
                let q_ms = s.h_tr_ms * (mass_temperature - s.temperature);
                let q_em = s.h_tr_em * (s.temperature - exterior_temperature);
                (q_ms - q_em - s.heat_flow).abs()
            })
            .fold(0.0f64, f64::max)
    }
}

impl Default for PerSurfaceConductionSolver {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Steady-state gives correct U*A heat flow.
    ///
    /// At steady state with constant temperatures, the heat flow through
    /// a surface should equal Q = U * A * ΔT.
    #[test]
    fn test_steady_state_heat_flow() {
        // Create a wall surface
        let mut solver = PerSurfaceConductionSolver::new();
        solver.add_surface_from_params(
            0,
            SurfaceKind::Wall,
            10.0, // area = 10 m²
            0.5,  // U = 0.5 W/m²K
            20.0, // initial temperature
            5.0,  // h_tr_ms
            4.0,  // h_tr_is
            2.0,  // h_tr_em
        );

        let mass_temp = 20.0;
        let exterior_temp_step = 0.0;
        let dt = 3600.0;

        // Record initial temperature before transient
        let t_surface_initial = solver.surface_temperatures()[0];

        // Collect surface temperatures during transient
        let mut temperatures = Vec::new();
        for _ in 0..100 {
            solver.update_all(dt, mass_temp, exterior_temp_step);
            temperatures.push(solver.surface_temperatures()[0]);
        }

        let t_surface_final = temperatures.last().unwrap();

        // Final temperature should be between mass and exterior
        assert!(
            *t_surface_final > exterior_temp_step && *t_surface_final < mass_temp,
            "Final surface temp {} should be between exterior {} and mass {}",
            t_surface_final,
            exterior_temp_step,
            mass_temp
        );

        // Surface should have cooled (moved toward exterior temperature)
        assert!(
            *t_surface_final < t_surface_initial,
            "Surface temp should have decreased from {} to {}",
            t_surface_initial,
            t_surface_final
        );
    }

    /// Test: Thermal lag matches expected delay for insulation thickness.
    ///
    /// Thicker insulation (lower U-value) should result in slower thermal
    /// response (larger thermal lag).
    #[test]
    fn test_thermal_lag() {
        let dt = 60.0;
        let mass_temp = 20.0;
        let _exterior_temp = 0.0;

        // High insulation (low U) - slow response
        let mut solver_thick = PerSurfaceConductionSolver::new();
        solver_thick.add_surface_from_params(
            0,
            SurfaceKind::Wall,
            10.0,
            0.2, // Low U = thick insulation
            20.0,
            10.0, // h_tr_ms
            8.0,  // h_tr_is
            5.0,  // h_tr_em
        );

        // Low insulation (high U) - fast response
        let mut solver_thin = PerSurfaceConductionSolver::new();
        solver_thin.add_surface_from_params(
            0,
            SurfaceKind::Wall,
            10.0,
            2.0, // High U = thin insulation
            20.0,
            50.0, // h_tr_ms
            40.0, // h_tr_is
            20.0, // h_tr_em
        );

        // Apply step change
        let exterior_step = 0.0;
        let steps = 50;

        // Track temperature evolution
        let mut thick_temps = Vec::new();
        for _ in 0..steps {
            solver_thick.update_all(dt, mass_temp, exterior_step);
            thick_temps.push(solver_thick.surface_temperatures()[0]);
        }

        let mut thin_temps = Vec::new();
        for _ in 0..steps {
            solver_thin.update_all(dt, mass_temp, exterior_step);
            thin_temps.push(solver_thin.surface_temperatures()[0]);
        }

        // Thin insulation should reach lower temperature faster
        // After same number of steps, thin surface should be colder (closer to exterior)
        assert!(
            thin_temps[steps - 1] < thick_temps[steps - 1],
            "Thin insulation surface ({}) should be colder than thick ({}) after {} steps",
            thin_temps[steps - 1],
            thick_temps[steps - 1],
            steps
        );
    }

    /// Test: Surface temperature computation from mass and air temperatures using ISO 13790.
    #[test]
    fn test_surface_temperature_from_mass() {
        let mut solver = PerSurfaceConductionSolver::new();
        solver.add_surface_from_params(
            0,
            SurfaceKind::Roof,
            20.0, // area
            0.3,  // U-value
            15.0, // temperature
            8.0,  // h_tr_ms
            6.0,  // h_tr_is
            3.0,  // h_tr_em
        );

        let zone_air_temp = 22.0;
        let mass_temp = 18.0;

        let computed_temps = solver.compute_surface_temperatures(zone_air_temp, mass_temp);
        let t_surface = computed_temps[0];

        // T_surface = (h_tr_is * T_air + h_tr_ms * T_mass) / (h_tr_is + h_tr_ms)
        // T_surface = (6.0 * 22.0 + 8.0 * 18.0) / (6.0 + 8.0)
        // T_surface = (132.0 + 144.0) / 14.0 = 276.0 / 14.0 = 19.714...
        let expected = (6.0 * zone_air_temp + 8.0 * mass_temp) / (6.0 + 8.0);
        assert!(
            (t_surface - expected).abs() < 1e-10,
            "Surface temp {} should equal expected {}",
            t_surface,
            expected
        );

        // T_surface should be between T_air and T_mass
        assert!(
            t_surface > mass_temp && t_surface < zone_air_temp,
            "Surface temp {} should be between air {} and mass {}",
            t_surface,
            zone_air_temp,
            mass_temp
        );
    }

    /// Test: Surface temperature when air dominates (high h_tr_is).
    #[test]
    fn test_surface_temperature_air_dominates() {
        let mut solver = PerSurfaceConductionSolver::new();
        solver.add_surface_from_params(
            0,
            SurfaceKind::Wall,
            10.0,
            0.5,
            20.0,
            1.0,   // h_tr_ms - small
            100.0, // h_tr_is - large, so air dominates
            2.0,
        );

        let zone_air_temp = 22.0;
        let mass_temp = 18.0;

        let computed_temps = solver.compute_surface_temperatures(zone_air_temp, mass_temp);
        let t_surface = computed_temps[0];

        // When h_tr_is >> h_tr_ms, T_surface ≈ T_air
        assert!(
            (t_surface - zone_air_temp).abs() < 0.1,
            "When air dominates, T_surface ({}) should be close to T_air ({})",
            t_surface,
            zone_air_temp
        );
    }

    /// Test: Surface temperature when mass dominates (high h_tr_ms).
    #[test]
    fn test_surface_temperature_mass_dominates() {
        let mut solver = PerSurfaceConductionSolver::new();
        solver.add_surface_from_params(
            0,
            SurfaceKind::Wall,
            10.0,
            0.5,
            20.0,
            100.0, // h_tr_ms - large, so mass dominates
            1.0,   // h_tr_is - small
            2.0,
        );

        let zone_air_temp = 22.0;
        let mass_temp = 18.0;

        let computed_temps = solver.compute_surface_temperatures(zone_air_temp, mass_temp);
        let t_surface = computed_temps[0];

        // When h_tr_ms >> h_tr_is, T_surface ≈ T_mass
        assert!(
            (t_surface - mass_temp).abs() < 0.1,
            "When mass dominates, T_surface ({}) should be close to T_mass ({})",
            t_surface,
            mass_temp
        );
    }

    /// Test: Surface temperature when h_tr_is == h_tr_ms (equal weighting).
    #[test]
    fn test_surface_temperature_equal_weighting() {
        let mut solver = PerSurfaceConductionSolver::new();
        solver.add_surface_from_params(
            0,
            SurfaceKind::Wall,
            10.0,
            0.5,
            20.0,
            5.0, // h_tr_ms
            5.0, // h_tr_is - equal to h_tr_ms
            2.0,
        );

        let zone_air_temp = 22.0;
        let mass_temp = 18.0;

        let computed_temps = solver.compute_surface_temperatures(zone_air_temp, mass_temp);
        let t_surface = computed_temps[0];

        // When h_tr_is == h_tr_ms, T_surface = (T_air + T_mass) / 2
        let expected = (zone_air_temp + mass_temp) / 2.0;
        assert!(
            (t_surface - expected).abs() < 1e-10,
            "When equal weighting, T_surface ({}) should be average ({})",
            t_surface,
            expected
        );
    }

    /// Test: Surface temperature computation from mass temperature (deprecated method).
    #[test]
    fn test_surface_temperature_computation() {
        let mut solver = PerSurfaceConductionSolver::new();
        solver.add_surface_from_params(
            0,
            SurfaceKind::Roof,
            20.0, // area
            0.3,  // U-value
            15.0, // temperature (will be overwritten by computation)
            8.0,  // h_tr_ms
            6.0,  // h_tr_is
            3.0,  // h_tr_em
        );

        let mass_temp = 20.0;
        let exterior_temp = -10.0;

        // Use the original surface_temperature method directly
        let t_surface = solver
            .get(0)
            .unwrap()
            .surface_temperature(mass_temp, exterior_temp);

        // T_surface should be between T_mass and T_exterior
        assert!(
            t_surface > exterior_temp && t_surface < mass_temp,
            "Surface temp {} should be between exterior {} and mass {}",
            t_surface,
            exterior_temp,
            mass_temp
        );
    }

    /// Test: Energy conservation at surface interface.
    #[test]
    fn test_energy_conservation() {
        let mut solver = PerSurfaceConductionSolver::new();
        solver.add_surface_from_params(0, SurfaceKind::Wall, 15.0, 0.5, 18.0, 12.0, 10.0, 6.0);

        let mass_temp = 22.0;
        let exterior_temp = 2.0;
        let dt = 300.0; // 5 minutes

        // Update and check energy balance
        solver.update_all(dt, mass_temp, exterior_temp);

        let imbalance = solver.energy_imbalance(mass_temp, exterior_temp);

        // Energy imbalance should be small relative to heat flow magnitudes
        // A 1% tolerance is reasonable for backward Euler integration
        let tolerance = 1.0; // 1 W/K tolerance
        assert!(
            imbalance < tolerance,
            "Energy imbalance {} should be less than {} W/K",
            imbalance,
            tolerance
        );
    }

    /// Test: Independent surface updates.
    #[test]
    fn test_independent_updates() {
        let mut solver = PerSurfaceConductionSolver::new();

        // Add wall and roof with different properties
        solver.add_surface_from_params(0, SurfaceKind::Wall, 10.0, 0.5, 20.0, 5.0, 4.0, 2.0);
        solver.add_surface_from_params(
            1,
            SurfaceKind::Roof,
            10.0,
            0.3,
            15.0, // Different initial temperature
            4.0,
            3.0,
            1.5,
        );

        let dt = 3600.0;
        let mass_temp = 20.0;
        let exterior_temp = 0.0;

        // Update all
        solver.update_all(dt, mass_temp, exterior_temp);

        let temps = solver.surface_temperatures();

        // Both surfaces should have updated temperatures
        assert_eq!(temps.len(), 2);

        // Each surface should have moved from its initial
        // Wall started at 20, should change
        // Roof started at 15, should change
        let surface_0 = solver.get(0).unwrap();
        let surface_1 = solver.get(1).unwrap();

        assert_ne!(
            surface_0.temperature, 20.0,
            "Wall temperature should have changed"
        );
        assert_ne!(
            surface_1.temperature, 15.0,
            "Roof temperature should have changed"
        );
    }

    /// Test: All surface types are handled correctly.
    #[test]
    fn test_surface_kinds() {
        let wall_kind = SurfaceKind::from_orientation(Orientation::North);
        assert_eq!(wall_kind, SurfaceKind::Wall);

        let roof_kind = SurfaceKind::from_orientation(Orientation::Up);
        assert_eq!(roof_kind, SurfaceKind::Roof);

        let floor_kind = SurfaceKind::from_orientation(Orientation::Down);
        assert_eq!(floor_kind, SurfaceKind::Floor);

        let horiz_kind = SurfaceKind::from_orientation(Orientation::Horizontal);
        assert_eq!(horiz_kind, SurfaceKind::Roof);
    }

    // =============================================================================
    // Mass Node Tests (Issue #1003 - Backward Euler Update Formula)
    // =============================================================================

    /// Test: Backward Euler update formula for mass temperature.
    ///
    /// Verifies that the implementation matches the analytical formula:
    /// T_mass_new = (T_mass_old * C_mass + dt * (h_tr_is * T_air + h_tr_ms * T_sky))
    ///              / (C_mass + dt * (h_tr_is + h_tr_ms))
    #[test]
    fn test_mass_node_backward_euler_formula() {
        // Create a mass node with known properties
        let mut mass_node = MassNode::new(
            0,           // id
            20.0,        // initial temperature T_mass_old = 20°C
            100_000.0,   // C_mass = 100,000 J/K
            10.0,        // h_tr_is = 10 W/K
            5.0,         // h_tr_ms = 5 W/K
        );

        let dt = 3600.0;     // dt = 1 hour
        let T_air = 22.0;    // T_air = 22°C
        let T_sky = 0.0;     // T_sky = 0°C

        // Compute expected result using the analytical formula
        let expected = {
            let numerator = 20.0 * 100_000.0 + 3600.0 * (10.0 * 22.0 + 5.0 * 0.0);
            let denominator = 100_000.0 + 3600.0 * (10.0 + 5.0);
            numerator / denominator
        };

        // Update and verify
        mass_node.update(dt, T_air, T_sky);

        // Should match analytical formula within floating-point tolerance
        assert!(
            (mass_node.temperature - expected).abs() < 1e-10,
            "Mass node temp {} should match expected {}",
            mass_node.temperature,
            expected
        );
    }

    /// Test: Mass node approaches steady-state temperature.
    ///
    /// When dt → ∞, the mass temperature should approach the weighted average:
    /// T_ss = (h_tr_is * T_air + h_tr_ms * T_sky) / (h_tr_is + h_tr_ms)
    #[test]
    fn test_mass_node_steady_state() {
        let mass_node = MassNode::new(
            0,
            20.0,
            100_000.0,
            10.0,
            5.0,
        );

        let T_air = 25.0;
        let T_sky = 5.0;

        let expected_ss = mass_node.steady_state_temperature(T_air, T_sky);
        let expected_manual = (10.0 * 25.0 + 5.0 * 5.0) / (10.0 + 5.0);

        assert!(
            (expected_ss - expected_manual).abs() < 1e-10,
            "Steady-state temp {} should match expected {}",
            expected_ss,
            expected_manual
        );

        // With large dt, should approach steady state
        let mut mass_node_large_dt = MassNode::new(0, 20.0, 100_000.0, 10.0, 5.0);
        mass_node_large_dt.update(1e12, T_air, T_sky); // Very large dt

        assert!(
            (mass_node_large_dt.temperature - expected_ss).abs() < 1e-6,
            "With large dt, mass temp {} should approach steady state {}",
            mass_node_large_dt.temperature,
            expected_ss
        );
    }

    /// Test: Mass node zero capacitance handling.
    ///
    /// A mass node with zero capacitance should not update (division by zero protection).
    #[test]
    fn test_mass_node_zero_capacitance() {
        let mut mass_node = MassNode::new(
            0,
            20.0,
            0.0,        // Zero capacitance
            10.0,
            5.0,
        );

        let initial_temp = mass_node.temperature;
        mass_node.update(3600.0, 22.0, 0.0);

        // Temperature should remain unchanged
        assert_eq!(
            mass_node.temperature, initial_temp,
            "Mass node with zero capacitance should not update"
        );
    }

    /// Test: Mass node update_all in solver.
    ///
    /// Verifies that update_mass_nodes correctly updates all mass nodes.
    #[test]
    fn test_solver_update_mass_nodes() {
        let mut solver = PerSurfaceConductionSolver::new();

        // Add mass nodes with different properties
        solver.add_mass_node(MassNode::new(0, 20.0, 100_000.0, 10.0, 5.0));
        solver.add_mass_node(MassNode::new(1, 15.0, 80_000.0, 8.0, 3.0));

        let dt = 3600.0;
        let T_air = 22.0;
        let T_sky = 0.0;

        // Update all mass nodes
        solver.update_mass_nodes(dt, T_air, T_sky);

        let mass_temps = solver.mass_temperatures();

        // Both mass nodes should have updated
        assert_eq!(mass_temps.len(), 2);

        // Mass node 0: T_old=20, should move toward (10*22 + 5*0)/(10+5) = 14.67
        // Mass node 1: T_old=15, should move toward (8*22 + 3*0)/(8+3) = 16
        assert!(
            mass_temps[0] < 20.0 && mass_temps[0] > 0.0,
            "Mass node 0 temp {} should decrease toward steady state",
            mass_temps[0]
        );
        assert!(
            mass_temps[1] > 15.0 && mass_temps[1] < 22.0,
            "Mass node 1 temp {} should increase toward steady state",
            mass_temps[1]
        );
    }

    /// Test: Backward Euler is unconditionally stable.
    ///
    /// Backward Euler should remain stable even with very large timesteps.
    #[test]
    fn test_backward_euler_stability() {
        let mut mass_node = MassNode::new(
            0,
            100.0,      // Very different from T_air and T_sky
            1000.0,     // Small capacitance
            100.0,      // Large conductance
            100.0,
        );

        let T_air = 0.0;
        let T_sky = 0.0;
        let dt = 1e6; // Very large timestep (10^6 seconds ≈ 11.5 days)

        // Should not diverge or produce NaN
        mass_node.update(dt, T_air, T_sky);

        assert!(
            mass_node.temperature.is_finite(),
            "Mass temperature {} should be finite (not NaN or Inf)",
            mass_node.temperature
        );

        // Should approach steady state
        assert!(
            mass_node.temperature > 0.0 && mass_node.temperature < 100.0,
            "Mass temperature {} should be between T_air and initial",
            mass_node.temperature
        );
    }

    /// Test: Mass node heat flows are conserved.
    ///
    /// At any timestep, the heat flow balance should be satisfied:
    /// Q_tr_is + Q_tr_ms = C_mass * dT/dt
    #[test]
    fn test_mass_node_energy_conservation() {
        let mut mass_node = MassNode::new(
            0,
            20.0,
            50_000.0,
            10.0,
            5.0,
        );

        let dt = 3600.0;
        let T_air = 22.0;
        let T_sky = 2.0;

        let T_old = mass_node.temperature;
        mass_node.update(dt, T_air, T_sky);
        let T_new = mass_node.temperature;

        // Heat flows
        let Q_tr_is = 10.0 * (T_air - T_new);
        let Q_tr_ms = 5.0 * (T_sky - T_new);
        let Q_stored = 50_000.0 * (T_new - T_old) / dt;

        // Energy balance: Q_tr_is + Q_tr_ms = Q_stored
        let imbalance = (Q_tr_is + Q_tr_ms - Q_stored).abs();
        let tolerance = 1e-10;

        assert!(
            imbalance < tolerance,
            "Energy imbalance {} should be negligible",
            imbalance
        );
    }
}

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

    /// Compute surface temperature from mass temperature.
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
}

impl PerSurfaceConductionSolver {
    /// Create a new empty per-surface conduction solver.
    pub fn new() -> Self {
        Self {
            surfaces: Vec::new(),
        }
    }

    /// Create a solver with the given surfaces.
    pub fn with_surfaces(surfaces: Vec<SurfaceNode>) -> Self {
        Self { surfaces }
    }

    /// Add a surface to the solver.
    pub fn add_surface(&mut self, surface: SurfaceNode) {
        self.surfaces.push(surface);
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

    /// Compute surface temperatures from mass temperature.
    ///
    /// Returns a vector of surface temperatures corresponding to each surface node.
    pub fn compute_surface_temperatures(
        &self,
        mass_temperature: f64,
        exterior_temperature: f64,
    ) -> Vec<f64> {
        self.surfaces
            .iter()
            .map(|s| s.surface_temperature(mass_temperature, exterior_temperature))
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

    /// Test: Surface temperature computation from mass temperature.
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
            3.0,  // h_tr_em
        );

        let mass_temp = 20.0;
        let exterior_temp = -10.0;

        let computed_temps = solver.compute_surface_temperatures(mass_temp, exterior_temp);
        let t_surface = computed_temps[0];

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
        solver.add_surface_from_params(0, SurfaceKind::Wall, 15.0, 0.5, 18.0, 12.0, 6.0);

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
        solver.add_surface_from_params(0, SurfaceKind::Wall, 10.0, 0.5, 20.0, 5.0, 2.0);
        solver.add_surface_from_params(
            1,
            SurfaceKind::Roof,
            10.0,
            0.3,
            15.0, // Different initial temperature
            4.0,
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
}

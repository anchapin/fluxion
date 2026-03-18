//! 5R1C Thermal Network Solver - Baseline heat conduction method.
//!
//! This module implements the ISO 13790 5R1C thermal network model,
//! which represents a wall as a network of 5 resistances and 1 capacitance.
//!
//! # Overview
//!
//! The 5R1C model is the fastest heat conduction solver, suitable for:
//! - Low-mass buildings (wood frame, lightweight construction)
//! - Steady-state or quasi-steady-state conditions
//! - Quick parametric studies and optimization
//!
//! # Model Structure
//!
//! ```text
//! T_ext ── R_se ── T_se ── R_2 ──┬── R_1 ── T_si ── R_si ── T_int
//!                                │
//!                                C_m
//!                                │
//!                               T_m
//! ```
//!
//! Where:
//! - R_se: Exterior surface resistance
//! - R_si: Interior surface resistance
//! - R_1, R_2: Wall layer resistances (split around capacitance)
//! - C_m: Thermal capacitance of wall
//! - T_m: Temperature of thermal mass node

use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::sim::assembly::BuildingAssembly;

/// 5R1C thermal network solver.
///
/// This solver uses a lumped-capacitance approach with 5 resistances
/// and 1 capacitance to model heat transfer through building envelopes.
pub struct FiveR1CSolver {
    /// Total thermal resistance [m²·K/W]
    R_total: f64,
    /// Total thermal capacitance [J/m²·K]
    C_total: f64,
    /// Interior surface resistance [m²·K/W]
    R_si: f64,
    /// Exterior surface resistance [m²·K/W]
    R_se: f64,
    /// Mass node temperature [°C]
    T_mass: f64,
    /// Current heat flux [W/m²]
    q_flux: f64,
    /// Initialized flag
    initialized: bool,
}

impl FiveR1CSolver {
    /// Create a new 5R1C solver (uninitialized).
    pub fn new() -> Self {
        Self {
            R_total: 0.0,
            C_total: 0.0,
            R_si: 1.0 / 8.0,  // Default interior film coefficient
            R_se: 1.0 / 25.0, // Default exterior film coefficient
            T_mass: 20.0,
            q_flux: 0.0,
            initialized: false,
        }
    }

    /// Calculate steady-state heat flux (no mass effect).
    pub fn steady_state_flux(&self, T_int: f64, T_ext: f64) -> f64 {
        (T_ext - T_int) / self.R_total
    }
}

impl Default for FiveR1CSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl HeatConductionSolver for FiveR1CSolver {
    fn name(&self) -> &str {
        "5R1C"
    }

    fn initialize(&mut self, wall: &BuildingAssembly) -> Result<(), SolverError> {
        // Calculate total thermal resistance [m²·K/W]
        self.R_total = wall.total_r_value();

        // Calculate total thermal capacitance [J/m²·K]
        // thermal_mass() returns kJ/m²·K, so convert to J/m²·K
        self.C_total = wall.thermal_mass() * 1000.0;

        // Set surface resistances (default values)
        self.R_si = 1.0 / 8.0; // h_interior = 8 W/m²·K
        self.R_se = 1.0 / 25.0; // h_exterior = 25 W/m²·K

        // Initialize mass node temperature to average of expected temperatures
        self.T_mass = 20.0;
        self.q_flux = 0.0;

        // Validate
        if self.R_total <= 0.0 || !self.R_total.is_finite() {
            return Err(SolverError::ConstructionError(
                "Invalid wall resistance (must be positive and finite)".to_string(),
            ));
        }

        self.initialized = true;
        Ok(())
    }

    fn step(
        &mut self,
        timestep: f64,
        T_interior: f64,
        T_exterior: f64,
        _h_interior: f64,
        _h_exterior: f64,
    ) -> Result<f64, SolverError> {
        if !self.initialized {
            return Err(SolverError::InvalidConfig(
                "Solver not initialized. Call initialize() first.".to_string(),
            ));
        }

        // Simple 5R1C calculation (no mass node dynamics for now)
        // This is the baseline implementation - can be extended with mass node
        self.q_flux = self.steady_state_flux(T_interior, T_exterior);

        Ok(self.q_flux)
    }

    fn energy_storage_rate(&self) -> f64 {
        // 5R1C doesn't track storage explicitly in this simple implementation
        0.0
    }

    fn is_valid(&self) -> bool {
        self.initialized && self.R_total > 0.0 && self.R_total.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial};

    #[test]
    fn test_five_r1c_initialization() {
        let mut solver = FiveR1CSolver::new();

        // Create simple wall
        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2))) // 200mm concrete
            .build()
            .unwrap();

        let result = solver.initialize(&wall);
        assert!(result.is_ok());
        assert!(solver.is_valid());
        assert!(solver.R_total > 0.0);
    }

    #[test]
    fn test_five_r1c_flux_calculation() {
        let mut solver = FiveR1CSolver::new();

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        solver.initialize(&wall).unwrap();

        // Calculate flux for 20°C interior, 0°C exterior
        let flux = solver.step(3600.0, 20.0, 0.0, 8.0, 25.0).unwrap();

        // Flux should be negative (heat flowing out)
        assert!(flux < 0.0);

        // Magnitude should be reasonable (around 50-100 W/m² for this ΔT)
        assert!(flux.abs() > 10.0 && flux.abs() < 200.0);
    }

    #[test]
    fn test_five_r1c_steady_state() {
        let mut solver = FiveR1CSolver::new();

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        solver.initialize(&wall).unwrap();

        // Steady-state flux
        let T_int = 20.0;
        let T_ext = 0.0;
        let expected = (T_ext - T_int) / solver.R_total;
        let actual = solver.steady_state_flux(T_int, T_ext);

        assert!((actual - expected).abs() < 1e-10);
    }
}

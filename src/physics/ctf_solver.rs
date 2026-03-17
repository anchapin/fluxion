//! Conduction Transfer Function (CTF) runtime solver.
//!
//! This module provides the runtime engine for CTF-based heat conduction
//! calculations. It maintains temperature and flux history buffers and
//! computes surface heat flux using precomputed CTF coefficients.
//!
//! # Overview
//!
//! The `CTFSolver` uses CTF coefficients to calculate interior and exterior
//! surface heat flux at each timestep:
//!
//! ```text
//! q''_int,t = -Z₀·T_int,t + Σ(X_j·T_ext,t-j) - Σ(Y_j·T_int,t-j) - Σ(Φ_j·q''_t-j)
//! q''_ext,t = -X₀·T_ext,t + Σ(Y_j·T_ext,t-j) - Σ(Z_j·T_int,t-j) - Σ(Φ_j·q''_t-j)
//! ```
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::ctf_solver::{CTFSolver, CTFSolverConfig};
//! use fluxion::physics::ctf_coefficients::{CTFCalculator, CTFMaterial, CTFCoefficients};
//!
//! // Define wall construction
//! let layers = vec![
//!     CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
//!     CTFMaterial::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
//!     CTFMaterial::new("Insulation", 0.050, 0.04, 50.0, 840.0),
//!     CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
//! ];
//!
//! // Compute coefficients
//! let coeffs = CTFCalculator::with_defaults(&layers, 3600.0).compute_coefficients();
//!
//! // Create solver
//! let config = CTFSolverConfig::new(3600.0, 50);
//! let mut solver = CTFSolver::new(coeffs, config);
//!
//! // At each timestep
//! let t_interior = 20.0;
//! let t_exterior = 5.0;
//! let q_flux = solver.step(t_interior, t_exterior);
//! ```

use crate::physics::ctf_coefficients::CTFCoefficients;
use std::fmt;

/// CTF solver configuration.
#[derive(Debug, Clone)]
pub struct CTFSolverConfig {
    /// Timestep duration [s].
    pub timestep: f64,
    /// Number of history elements to retain.
    pub history_size: usize,
    /// Interior surface area [m²].
    pub surface_area: f64,
    /// Interior convective coefficient [W/m²·K].
    pub h_interior: f64,
    /// Exterior convective coefficient [W/m²·K].
    pub h_exterior: f64,
    /// Solar absorptance (0-1).
    pub alpha_solar: f64,
}

impl CTFSolverConfig {
    /// Create new configuration.
    pub fn new(timestep: f64, history_size: usize) -> Self {
        Self {
            timestep,
            history_size,
            surface_area: 1.0,
            h_interior: 8.0,
            h_exterior: 25.0,
            alpha_solar: 0.7,
        }
    }

    /// Create configuration for ASHRAE 140 Case 900.
    pub fn case_900(timestep: f64) -> Self {
        Self {
            timestep,
            history_size: 50,
            surface_area: 97.2, // 4 walls × 8m × 2.7m - windows
            h_interior: 8.0,
            h_exterior: 25.0,
            alpha_solar: 0.7,
        }
    }
}

/// CTF solver state and history buffers.
pub struct CTFSolver {
    /// CTF coefficients (X, Y, Z, Φ).
    pub coefficients: CTFCoefficients,
    /// Solver configuration.
    pub config: CTFSolverConfig,
    /// Interior temperature history [T_t, T_t-1, T_t-2, ...].
    t_interior_history: Vec<f64>,
    /// Exterior temperature history [T_t, T_t-1, T_t-2, ...].
    t_exterior_history: Vec<f64>,
    /// Interior heat flux history [q_t, q_t-1, q_t-2, ...].
    q_interior_history: Vec<f64>,
    /// Exterior heat flux history [q_t, q_t-1, q_t-2, ...].
    q_exterior_history: Vec<f64>,
    /// Current interior surface temperature [°C].
    t_interior_surface: f64,
    /// Current exterior surface temperature [°C].
    t_exterior_surface: f64,
}

impl CTFSolver {
    /// Create new CTF solver.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Precomputed CTF coefficients
    /// * `config` - Solver configuration
    ///
    /// # Returns
    ///
    /// Solver initialized with uniform temperature (20°C).
    pub fn new(coefficients: CTFCoefficients, config: CTFSolverConfig) -> Self {
        let history_size = config.history_size.max(coefficients.num_coeffs);

        Self {
            coefficients,
            config,
            t_interior_history: vec![20.0; history_size],
            t_exterior_history: vec![20.0; history_size],
            q_interior_history: vec![0.0; history_size],
            q_exterior_history: vec![0.0; history_size],
            t_interior_surface: 20.0,
            t_exterior_surface: 20.0,
        }
    }

    /// Create solver for ASHRAE 140 Case 900.
    pub fn case_900(coefficients: CTFCoefficients, timestep: f64) -> Self {
        Self::new(coefficients, CTFSolverConfig::case_900(timestep))
    }

    /// Advance solver by one timestep.
    ///
    /// # Arguments
    ///
    /// * `t_interior` - Current interior surface temperature [°C]
    /// * `t_exterior` - Current exterior surface temperature [°C]
    ///
    /// # Returns
    ///
    /// Interior surface heat flux [W/m²] (positive = into zone).
    pub fn step(&mut self, t_interior: f64, t_exterior: f64) -> f64 {
        // Update surface temperatures
        self.t_interior_surface = t_interior;
        self.t_exterior_surface = t_exterior;

        // Shift history buffers
        self.shift_history();

        // Store new temperatures at front of history
        self.t_interior_history[0] = t_interior;
        self.t_exterior_history[0] = t_exterior;

        // Calculate interior heat flux using CTF coefficients
        let q_interior = self.coefficients.calculate_interior_flux(
            t_interior,
            &self.t_exterior_history,
            &self.t_interior_history[1..], // Exclude current from history
            &self.q_interior_history[1..], // Exclude current from history
        );

        // Store flux in history
        self.q_interior_history[0] = q_interior;

        q_interior
    }

    /// Calculate exterior heat flux (positive = into wall from outside).
    pub fn exterior_flux(&self) -> f64 {
        // Simplified: use energy balance
        // q_ext = q_int - dE_stored/dt (approximately)
        self.q_interior_history[0]
    }

    /// Shift history buffers by one position.
    fn shift_history(&mut self) {
        // Shift temperatures
        for i in (1..self.t_interior_history.len()).rev() {
            self.t_interior_history[i] = self.t_interior_history[i - 1];
            self.t_exterior_history[i] = self.t_exterior_history[i - 1];
        }

        // Shift fluxes
        for i in (1..self.q_interior_history.len()).rev() {
            self.q_interior_history[i] = self.q_interior_history[i - 1];
            self.q_exterior_history[i] = self.q_exterior_history[i - 1];
        }
    }

    /// Get interior surface temperature.
    #[inline]
    pub fn interior_surface_temp(&self) -> f64 {
        self.t_interior_surface
    }

    /// Get exterior surface temperature.
    #[inline]
    pub fn exterior_surface_temp(&self) -> f64 {
        self.t_exterior_surface
    }

    /// Get interior heat flux.
    #[inline]
    pub fn interior_flux(&self) -> f64 {
        self.q_interior_history[0]
    }

    /// Get total heat transferred over simulation [J/m²].
    pub fn total_energy_transferred(&self) -> f64 {
        self.q_interior_history.iter().sum::<f64>() * self.config.timestep
    }

    /// Reset solver state to initial conditions.
    pub fn reset(&mut self, initial_temp: f64) {
        self.t_interior_history.fill(initial_temp);
        self.t_exterior_history.fill(initial_temp);
        self.q_interior_history.fill(0.0);
        self.q_exterior_history.fill(0.0);
        self.t_interior_surface = initial_temp;
        self.t_exterior_surface = initial_temp;
    }

    /// Get current timestep.
    #[inline]
    pub fn timestep(&self) -> f64 {
        self.config.timestep
    }
}

impl fmt::Display for CTFSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CTF Solver:")?;
        writeln!(f, "  Timestep: {:.0} s", self.config.timestep)?;
        writeln!(f, "  History size: {}", self.config.history_size)?;
        writeln!(f, "  Surface area: {:.1} m²", self.config.surface_area)?;
        writeln!(f, "  T_interior: {:.2}°C", self.t_interior_surface)?;
        writeln!(f, "  T_exterior: {:.2}°C", self.t_exterior_surface)?;
        writeln!(f, "  q_interior: {:.2} W/m²", self.q_interior_history[0])?;
        Ok(())
    }
}

/// CTF-based wall thermal model for system integration.
pub struct CTFWallModel {
    /// CTF solver for wall conduction.
    pub solver: CTFSolver,
    /// Interior zone temperature [°C].
    pub t_zone: f64,
    /// Exterior conditions (sol-air temperature) [°C].
    pub t_sol_air: f64,
}

impl CTFWallModel {
    /// Create new wall model.
    pub fn new(solver: CTFSolver) -> Self {
        Self {
            solver,
            t_zone: 20.0,
            t_sol_air: 20.0,
        }
    }

    /// Update boundary conditions.
    pub fn update_conditions(&mut self, t_zone: f64, t_sol_air: f64) {
        self.t_zone = t_zone;
        self.t_sol_air = t_sol_air;
    }

    /// Advance wall model by one timestep.
    ///
    /// Assumes surface temperatures equal adjacent fluid temperatures
    /// (simplified boundary condition).
    pub fn step(&mut self) -> f64 {
        self.solver.step(self.t_zone, self.t_sol_air)
    }

    /// Get heat flux into zone [W/m²].
    pub fn heat_flux(&self) -> f64 {
        self.solver.interior_flux()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};

    fn case_900_coefficients() -> CTFCoefficients {
        let layers = vec![
            CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            CTFMaterial::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
            CTFMaterial::new("Insulation", 0.050, 0.04, 50.0, 840.0),
            CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
        ];
        CTFCalculator::with_defaults(&layers, 3600.0).compute_coefficients()
    }

    #[test]
    fn test_solver_creation() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let solver = CTFSolver::new(coeffs, config);

        assert_eq!(solver.t_interior_history.len(), 50);
        assert_eq!(solver.t_exterior_history.len(), 50);
        assert_eq!(solver.q_interior_history.len(), 50);
    }

    #[test]
    fn test_single_step() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        let q = solver.step(20.0, 30.0);

        // Flux should be finite and reasonable
        assert!(q.is_finite(), "Flux should be finite");
        assert!(q.abs() < 1000.0, "Flux {:.2} unreasonably large", q);
    }

    #[test]
    fn test_temperature_step() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        // Apply temperature difference
        let _q1 = solver.step(20.0, 30.0);
        let _q2 = solver.step(20.0, 30.0);
        let q3 = solver.step(20.0, 30.0);

        // Flux should stabilize after initial transient
        assert!(q3.is_finite());
    }

    #[test]
    fn test_history_shift() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 10);
        let mut solver = CTFSolver::new(coeffs, config);

        // Initial state
        assert_eq!(solver.t_interior_history[0], 20.0);

        // Step with different temperature
        solver.step(25.0, 30.0);

        // New temperature should be at front
        assert_eq!(solver.t_interior_history[0], 25.0);
        // Old temperature should have shifted
        assert_eq!(solver.t_interior_history[1], 20.0);
    }

    #[test]
    fn test_reset() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        // Change state
        solver.step(30.0, 40.0);

        // Reset
        solver.reset(15.0);

        assert_eq!(solver.t_interior_history[0], 15.0);
        assert_eq!(solver.t_exterior_history[0], 15.0);
        assert_eq!(solver.q_interior_history[0], 0.0);
    }

    #[test]
    fn test_wall_model() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::case_900(3600.0);
        let solver = CTFSolver::new(coeffs, config);
        let mut model = CTFWallModel::new(solver);

        model.update_conditions(20.0, 30.0);
        let q = model.step();

        assert!(q.is_finite());
    }

    #[test]
    fn test_diurnal_simulation() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        // 24-hour simulation with diurnal exterior temperature
        let mut total_flux = 0.0;
        for hour in 0..24 {
            let t_ext = 10.0 + 10.0 * ((hour as f64 - 6.0) * std::f64::consts::PI / 12.0).sin();
            let q = solver.step(20.0, t_ext);
            total_flux += q;
        }

        // Net flux should be reasonable (not exploding)
        assert!(
            total_flux.abs() < 10000.0,
            "Total flux {:.2} unreasonably large",
            total_flux
        );
    }

    #[test]
    fn test_config_case_900() {
        let config = CTFSolverConfig::case_900(3600.0);

        assert_eq!(config.timestep, 3600.0);
        assert_eq!(config.history_size, 50);
        assert!((config.surface_area - 97.2).abs() < 0.1);
        assert_eq!(config.h_interior, 8.0);
        assert_eq!(config.h_exterior, 25.0);
    }
}

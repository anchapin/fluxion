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

use crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF;
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
            h_interior: 8.29,                // ASHRAE 140 Section 5.2
            h_exterior: EXTERIOR_FILM_COEFF, // ASHRAE 140 Section 5.2 (Issue #1419, v2023)
            alpha_solar: 0.7,
        }
    }

    /// Create configuration for ASHRAE 140 Case 900.
    pub fn case_900(timestep: f64) -> Self {
        Self {
            timestep,
            history_size: 50,
            surface_area: 63.6, // m² (corrected: 2(8+6)×2.7 - 12m² window = 63.6, was 97.2)
            h_interior: 8.29,   // ASHRAE 140 Section 5.2
            h_exterior: EXTERIOR_FILM_COEFF, // ASHRAE 140 Section 5.2 (Issue #1419, v2023)
            alpha_solar: 0.7,
        }
    }
}

/// CTF solver state and history buffers.
#[derive(Debug, Clone)]
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

    /// Create new CTF solver with warmup period.
    ///
    /// This initializes the solver by running a warmup period with realistic
    /// diurnal temperature cycles to fill history buffers with physically
    /// meaningful values, avoiding artificial transients at simulation start.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Precomputed CTF coefficients
    /// * `config` - Solver configuration
    /// * `t_interior_initial` - Initial interior temperature [°C]
    /// * `t_exterior_initial` - Initial exterior temperature [°C]
    /// * `warmup_days` - Number of warmup days (default: 7)
    ///
    /// # Returns
    ///
    /// Solver with history buffers initialized from warmup simulation.
    pub fn with_warmup(
        coefficients: CTFCoefficients,
        config: CTFSolverConfig,
        t_interior_initial: f64,
        t_exterior_initial: f64,
        warmup_days: usize,
    ) -> Self {
        let history_size = config.history_size.max(coefficients.num_coeffs);

        // Start with initial conditions
        let mut solver = Self {
            coefficients,
            config,
            t_interior_history: vec![t_interior_initial; history_size],
            t_exterior_history: vec![t_exterior_initial; history_size],
            q_interior_history: vec![0.0; history_size],
            q_exterior_history: vec![0.0; history_size],
            t_interior_surface: t_interior_initial,
            t_exterior_surface: t_exterior_initial,
        };

        // Run warmup period with diurnal cycles
        // Use simple sinusoidal diurnal variation: T_ext = T_avg + A*sin(2π*t/24)
        let t_avg = t_exterior_initial;
        let amplitude = 8.0; // Typical diurnal amplitude [°C]
        let hours_per_day = 24;
        let total_warmup_hours = warmup_days * hours_per_day;

        for hour in 0..total_warmup_hours {
            // Diurnal exterior temperature
            let t_ext =
                t_avg + amplitude * ((hour as f64 - 6.0) * std::f64::consts::PI / 12.0).sin();

            // Constant interior temperature (simplified - assumes HVAC maintains setpoint)
            let t_int = t_interior_initial;

            // Step solver
            solver.step(t_int, t_ext);
        }

        solver
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

    /// Get exterior temperature history buffer.
    /// Used by coupled solver for CTF history terms calculation.
    #[inline]
    pub fn exterior_temperature_history(&self) -> &[f64] {
        &self.t_exterior_history
    }

    /// Get interior temperature history buffer.
    /// Used by coupled solver for CTF history terms calculation.
    #[inline]
    pub fn interior_temperature_history(&self) -> &[f64] {
        &self.t_interior_history
    }

    /// Get interior heat flux history buffer.
    /// Used by coupled solver for CTF history terms calculation.
    #[inline]
    pub fn interior_flux_history(&self) -> &[f64] {
        &self.q_interior_history
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

        // Phase D update: the solver history length is
        //   max(config.history_size, coefficients.num_coeffs)
        // (the larger of the two — we need at least num_coeffs entries
        // for the CTF evaluation, and at least config.history_size for
        // user expectations). The second argument to CTFSolverConfig::new
        // is a LOWER BOUND on history size.
        let expected_len = 50_usize.max(solver.coefficients.num_coeffs);
        assert_eq!(solver.t_interior_history.len(), expected_len);
        assert_eq!(solver.t_exterior_history.len(), expected_len);
        assert_eq!(solver.q_interior_history.len(), expected_len);
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
        assert!((config.surface_area - 63.6).abs() < 0.1);
        assert_eq!(config.h_interior, 8.29);
        assert_eq!(config.h_exterior, EXTERIOR_FILM_COEFF);
    }

    #[test]
    fn test_solver_with_warmup() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let solver = CTFSolver::with_warmup(coeffs, config, 20.0, 15.0, 7);

        assert!(solver.interior_flux().is_finite());
    }

    #[test]
    fn test_exterior_flux() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        solver.step(20.0, 30.0);
        let q_int = solver.interior_flux();
        let q_ext = solver.exterior_flux();

        // In simplified implementation, they should be equal
        assert_eq!(q_int, q_ext);
    }

    #[test]
    fn test_interior_surface_temp() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        solver.step(22.0, 28.0);
        assert_eq!(solver.interior_surface_temp(), 22.0);

        solver.step(21.0, 27.0);
        assert_eq!(solver.interior_surface_temp(), 21.0);
    }

    #[test]
    fn test_exterior_surface_temp() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        solver.step(22.0, 28.0);
        assert_eq!(solver.exterior_surface_temp(), 28.0);

        solver.step(21.0, 27.0);
        assert_eq!(solver.exterior_surface_temp(), 27.0);
    }

    #[test]
    fn test_total_energy_transferred() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        // Step with constant temperature difference
        let q1 = solver.step(20.0, 30.0);
        let q2 = solver.step(20.0, 30.0);
        let q3 = solver.step(20.0, 30.0);

        let total_energy = solver.total_energy_transferred();
        let expected = (q1 + q2 + q3) * 3600.0;

        assert!((total_energy - expected).abs() < 1e-6);
    }

    #[test]
    fn test_timestep() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let solver = CTFSolver::new(coeffs.clone(), config);

        assert_eq!(solver.timestep(), 3600.0);

        let config2 = CTFSolverConfig::new(1800.0, 50);
        let solver2 = CTFSolver::new(coeffs, config2);
        assert_eq!(solver2.timestep(), 1800.0);
    }

    #[test]
    fn test_history_accessors() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 10);
        let mut solver = CTFSolver::new(coeffs, config);

        solver.step(25.0, 30.0);

        // Check history accessors return non-empty slices
        let ext_hist = solver.exterior_temperature_history();
        let int_hist = solver.interior_temperature_history();
        let flux_hist = solver.interior_flux_history();

        assert!(!ext_hist.is_empty());
        assert!(!int_hist.is_empty());
        assert!(!flux_hist.is_empty());
        assert_eq!(ext_hist[0], 30.0);
        assert_eq!(int_hist[0], 25.0);
    }

    #[test]
    fn test_wall_model_update_conditions() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let solver = CTFSolver::new(coeffs, config);
        let mut model = CTFWallModel::new(solver);

        model.update_conditions(22.0, 28.0);
        assert_eq!(model.t_zone, 22.0);
        assert_eq!(model.t_sol_air, 28.0);

        model.update_conditions(21.0, 27.0);
        assert_eq!(model.t_zone, 21.0);
        assert_eq!(model.t_sol_air, 27.0);
    }

    #[test]
    fn test_wall_model_heat_flux() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let solver = CTFSolver::new(coeffs, config);
        let mut model = CTFWallModel::new(solver);

        model.update_conditions(20.0, 30.0);
        model.step();

        assert!(model.heat_flux().is_finite());
    }

    #[test]
    fn test_solver_display() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let solver = CTFSolver::new(coeffs, config);

        let display_str = format!("{}", solver);
        assert!(display_str.contains("CTF Solver"));
        assert!(display_str.contains("Timestep"));
        assert!(display_str.contains("History size"));
    }

    #[test]
    fn test_solver_clone() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        solver.step(22.0, 28.0);
        let cloned = solver.clone();

        assert_eq!(
            cloned.interior_surface_temp(),
            solver.interior_surface_temp()
        );
        assert_eq!(
            cloned.exterior_surface_temp(),
            solver.exterior_surface_temp()
        );
    }

    #[test]
    fn test_config_clone() {
        let config = CTFSolverConfig::new(3600.0, 50);
        let cloned = config.clone();

        assert_eq!(cloned.timestep, 3600.0);
        assert_eq!(cloned.history_size, 50);
        assert_eq!(cloned.surface_area, 1.0);
        assert_eq!(cloned.h_interior, 8.29);
        assert_eq!(cloned.h_exterior, EXTERIOR_FILM_COEFF);
    }

    #[test]
    fn test_config_debug() {
        let config = CTFSolverConfig::new(3600.0, 50);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CTFSolverConfig"));
    }

    #[test]
    fn test_solver_debug() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let solver = CTFSolver::new(coeffs, config);
        let debug_str = format!("{:?}", solver);
        assert!(debug_str.contains("CTFSolver"));
    }

    #[test]
    fn test_case_900_constructor() {
        let coeffs = case_900_coefficients();
        let solver = CTFSolver::case_900(coeffs.clone(), 3600.0);

        assert_eq!(solver.config.timestep, 3600.0);
        assert_eq!(solver.config.history_size, 50);
        assert!((solver.config.surface_area - 63.6).abs() < 0.1);
    }

    #[test]
    fn test_warmup_initializes_history() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 20);
        let solver = CTFSolver::with_warmup(coeffs, config, 20.0, 15.0, 3);

        // History should be populated from warmup
        assert!(!solver.q_interior_history.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_multiple_steps_consistent() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        let mut solver = CTFSolver::new(coeffs, config);

        let mut prev_flux = 0.0;
        for _ in 0..10 {
            let q = solver.step(20.0, 30.0);
            assert!(q.is_finite());
            // Flux should stabilize (not jump wildly)
            if prev_flux != 0.0 {
                assert!((q - prev_flux).abs() < 100.0);
            }
            prev_flux = q;
        }
    }

    #[test]
    fn test_temperature_history_accurate() {
        let coeffs = case_900_coefficients();
        let config = CTFSolverConfig::new(3600.0, 10);
        let mut solver = CTFSolver::new(coeffs, config);

        solver.step(20.0, 30.0);
        solver.step(21.0, 31.0);
        solver.step(22.0, 32.0);

        let int_hist = solver.interior_temperature_history();
        assert_eq!(int_hist[0], 22.0);
        assert_eq!(int_hist[1], 21.0);
        assert_eq!(int_hist[2], 20.0);

        let ext_hist = solver.exterior_temperature_history();
        assert_eq!(ext_hist[0], 32.0);
        assert_eq!(ext_hist[1], 31.0);
        assert_eq!(ext_hist[2], 30.0);
    }
}

//! Heat Conduction Solver Trait - Common interface for all thermal solvers.
//!
//! This module defines the trait interface for heat conduction solvers,
//! enabling unified treatment of 5R1C, CTF, and finite difference methods.
//!
//! # Overview
//!
//! The `HeatConductionSolver` trait provides a common interface for:
//! - 5R1C thermal network (fast, low-mass buildings)
//! - CTF (Conduction Transfer Functions, accurate for high-mass)
//! - FD (Finite Difference, robust fallback for complex constructions)
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::solver_trait::{HeatConductionSolver, SolverError};
//! use fluxion::physics::five_r1c_solver::FiveR1CSolver;
//!
//! let mut solver = FiveR1CSolver::new();
//! solver.initialize(&wall_assembly)?;
//!
//! let flux = solver.step(3600.0, 20.0, 5.0, 8.0, 25.0)?;
//! ```

use crate::sim::assembly::BuildingAssembly;
use std::error::Error;
use std::fmt;

/// Error type for solver operations
#[derive(Debug, Clone)]
pub enum SolverError {
    /// Invalid configuration parameters
    InvalidConfig(String),
    /// Coefficient calculation failed
    CoefficientError(String),
    /// Numerical instability detected
    Instability(String),
    /// Convergence failure
    ConvergenceError(String),
    /// Invalid wall construction
    ConstructionError(String),
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            SolverError::CoefficientError(msg) => write!(f, "Coefficient error: {}", msg),
            SolverError::Instability(msg) => write!(f, "Numerical instability: {}", msg),
            SolverError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
            SolverError::ConstructionError(msg) => write!(f, "Construction error: {}", msg),
        }
    }
}

impl Error for SolverError {}

/// Common trait for all heat conduction solvers.
///
/// This trait defines the interface for calculating heat transfer through
/// building envelope constructions (walls, roofs, floors).
///
/// # Lifecycle
///
/// 1. Create solver instance
/// 2. Call `initialize()` with wall construction
/// 3. Call `step()` at each timestep
/// 4. Query results via `energy_storage_rate()`
///
/// # Example
///
/// ```rust
/// # use fluxion::physics::solver_trait::{HeatConductionSolver, SolverError};
/// # struct MySolver;
/// # impl MySolver { fn new() -> Self { MySolver } }
/// # impl HeatConductionSolver for MySolver {
/// #     fn name(&self) -> &str { "test" }
/// #     fn initialize(&mut self, wall: &BuildingAssembly) -> Result<(), SolverError> { Ok(()) }
/// #     fn step(&mut self, dt: f64, T_int: f64, T_ext: f64, h_int: f64, h_ext: f64) -> Result<f64, SolverError> { Ok(0.0) }
/// #     fn energy_storage_rate(&self) -> f64 { 0.0 }
/// #     fn is_valid(&self) -> bool { true }
/// # }
/// let mut solver = MySolver::new();
/// solver.initialize(&wall)?;
///
/// let flux = solver.step(3600.0, T_zone, T_outdoor, h_int, h_ext)?;
/// ```
pub trait HeatConductionSolver: Send + Sync {
    /// Get solver name/type identifier
    fn name(&self) -> &str;

    /// Initialize solver with wall construction
    ///
    /// # Arguments
    /// * `wall` - Wall assembly with material layers and properties
    ///
    /// # Returns
    /// Ok if initialization successful, Err if construction is invalid
    fn initialize(&mut self, wall: &BuildingAssembly) -> Result<(), SolverError>;

    /// Advance solver by one timestep
    ///
    /// # Arguments
    /// * `timestep` - Timestep duration [s]
    /// * `T_interior` - Interior air temperature [°C]
    /// * `T_exterior` - Exterior air temperature [°C]
    /// * `h_interior` - Interior convective heat transfer coefficient [W/m²·K]
    /// * `h_exterior` - Exterior convective heat transfer coefficient [W/m²·K]
    ///
    /// # Returns
    /// Heat flux through wall [W/m²] (positive = heat flowing into zone)
    fn step(
        &mut self,
        timestep: f64,
        T_interior: f64,
        T_exterior: f64,
        h_interior: f64,
        h_exterior: f64,
    ) -> Result<f64, SolverError>;

    /// Get current energy storage rate in wall [W/m²]
    ///
    /// Positive value means wall is storing energy (heating up),
    /// negative means wall is releasing energy (cooling down).
    fn energy_storage_rate(&self) -> f64;

    /// Check if solver is valid (coefficients converged, etc.)
    fn is_valid(&self) -> bool;
}

//! Heat Conduction Solver Trait - Common interface for all thermal solvers.
//!
//! This module defines the trait interface for heat conduction solvers,
//! enabling unified treatment of 5R1C, CTF, and finite difference methods.
//!
//! # Unified Solver Architecture (Issue #624)
//!
//! This trait is part of a consolidated solver architecture that prevents
//! solver proliferation. Rather than creating separate solver types for each
//! use case, we use a single trait with runtime dispatch via SolverManager.
//!
//! ## Design Principles
//!
//! 1. **Single Interface**: All heat conduction solvers implement this trait
//! 2. **Explicit Selection**: SolverManager selects the appropriate method
//! 3. **No Duplication**: CTF, FD, and 5R1C share the same lifecycle pattern
//! 4. **Validation Required**: New solvers only if existing ones fail validation
//!
//! ## Solver Lifecycle
//!
//! ```text
//! +-------------+     +---------------------+     +-------------+
//! | HeatConduction|-->|   SolverManager     |-->| 5R1C/CTF/FD |
//! |    Trait     |   | (automatic select)  |     |  Wrappers  |
//! +-------------+     +---------------------+     +-------------+
//! ```
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_error_display_invalid_config() {
        let err = SolverError::InvalidConfig("test message".to_string());
        assert_eq!(format!("{}", err), "Invalid configuration: test message");
    }

    #[test]
    fn test_solver_error_display_coefficient() {
        let err = SolverError::CoefficientError("calculation failed".to_string());
        assert_eq!(format!("{}", err), "Coefficient error: calculation failed");
    }

    #[test]
    fn test_solver_error_display_instability() {
        let err = SolverError::Instability("diverged".to_string());
        assert_eq!(format!("{}", err), "Numerical instability: diverged");
    }

    #[test]
    fn test_solver_error_display_convergence() {
        let err = SolverError::ConvergenceError("max iterations exceeded".to_string());
        assert_eq!(
            format!("{}", err),
            "Convergence error: max iterations exceeded"
        );
    }

    #[test]
    fn test_solver_error_display_construction() {
        let err = SolverError::ConstructionError("invalid layer".to_string());
        assert_eq!(format!("{}", err), "Construction error: invalid layer");
    }

    #[test]
    fn test_solver_error_is_clone() {
        let err = SolverError::InvalidConfig("test".to_string());
        let cloned = err.clone();
        assert_eq!(format!("{}", err), format!("{}", cloned));
    }

    #[test]
    fn test_solver_error_is_debug() {
        let err = SolverError::InvalidConfig("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidConfig"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_solver_error_implements_error_trait() {
        let err: Box<dyn Error> = Box::new(SolverError::InvalidConfig("test".to_string()));
        assert!(err.to_string().contains("Invalid configuration"));
    }

    #[test]
    fn test_heat_conduction_solver_trait_can_be_implemented() {
        struct TestSolver {
            valid: bool,
            storage_rate: f64,
        }

        impl HeatConductionSolver for TestSolver {
            fn name(&self) -> &str {
                "TestSolver"
            }

            fn initialize(&mut self, _wall: &BuildingAssembly) -> Result<(), SolverError> {
                Ok(())
            }

            fn step(
                &mut self,
                _timestep: f64,
                _T_interior: f64,
                _T_exterior: f64,
                _h_interior: f64,
                _h_exterior: f64,
            ) -> Result<f64, SolverError> {
                Ok(42.0)
            }

            fn energy_storage_rate(&self) -> f64 {
                self.storage_rate
            }

            fn is_valid(&self) -> bool {
                self.valid
            }
        }

        let mut solver = TestSolver {
            valid: true,
            storage_rate: 10.0,
        };

        assert_eq!(solver.name(), "TestSolver");
        assert!(solver.is_valid());
        assert_eq!(solver.energy_storage_rate(), 10.0);

        let result = solver.step(3600.0, 20.0, 5.0, 8.0, 25.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42.0);
    }

    #[test]
    fn test_heat_conduction_solver_can_return_error() {
        struct FailingSolver;

        impl HeatConductionSolver for FailingSolver {
            fn name(&self) -> &str {
                "FailingSolver"
            }

            fn initialize(&mut self, _wall: &BuildingAssembly) -> Result<(), SolverError> {
                Err(SolverError::ConstructionError("bad wall".to_string()))
            }

            fn step(
                &mut self,
                _timestep: f64,
                _T_interior: f64,
                _T_exterior: f64,
                _h_interior: f64,
                _h_exterior: f64,
            ) -> Result<f64, SolverError> {
                Err(SolverError::Instability("NaN detected".to_string()))
            }

            fn energy_storage_rate(&self) -> f64 {
                0.0
            }

            fn is_valid(&self) -> bool {
                false
            }
        }

        let mut solver = FailingSolver;
        assert!(!solver.is_valid());

        // Note: initialize() returns error before needing BuildingAssembly
        let step_result = solver.step(3600.0, 20.0, 5.0, 8.0, 25.0);
        assert!(step_result.is_err());
        assert!(step_result.unwrap_err().to_string().contains("instability"));
    }
}

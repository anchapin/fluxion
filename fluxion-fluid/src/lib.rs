//! fluxion-fluid: Compile-time strongly typed fluid port traits for DAE systems
//!
//! This crate provides:
//! - [`mediums`] - Physical medium types (Air, Water, Refrigerant, Steam)
//! - [`ports`] - FluidPort trait and concrete port implementations
//! - [`pantelides`] - Pantelides symbolic index reduction for DAE systems
//! - [`energy`] - Energy conservation verification for fluid networks
//! - [`autodiff`] - Analytical Jacobian traits for HVAC MPC control
//!
//! # Port Type Safety
//!
//! The crate uses Rust's type system to enforce compile-time compatibility between
//! fluid ports. Connecting mismatched ports (e.g., hydronic water to air duct) is
//! a **compile error**, not a runtime crash.
//!
//! # DAE Index Reduction
//!
//! The Pantelides algorithm reduces high-index DAE systems to index-1 form,
//! making them suitable for BDF timestepping.
//!
//! # Energy Conservation
//!
//! The [`energy`] module provides [`EnergyConservationVerifier`] which verifies that
//! fluid networks conserve energy according to the first law of thermodynamics.
//!
//! # Automatic Differentiation
//!
//! The [`autodiff`] module provides [`DifferentiableComponent`] for exposing exact
//! analytical Jacobian matrices from HVAC equipment, enabling reverse-mode automatic
//! differentiation for Model Predictive Control (MPC) and setpoint optimization.

pub mod autodiff;
pub mod ecs;
pub mod energy;
pub mod mediums;
pub mod pantelides;
pub mod ports;

pub use autodiff::{
    finite_diff_epsilon, finite_diff_jacobian, optimize_with_gradient_descent, relative_diff,
    relative_error, verify_jacobian_entries, Boiler, Chiller, CoolingCoil, DifferentiableComponent,
    GradientDescentOptimizer, Pump, VavBox,
};
pub use energy::{
    ConservationNode, EnergyConservationError, EnergyConservationResult,
    EnergyConservationVerifier, EnthalpyFlow, FluidNetworkGraph, SimulationResults,
};
pub use mediums::{Air, CompatibleWith, Medium, Water};
pub use pantelides::{
    pantelides_reduce, EqIndex, Equation, IncidenceMatrix, PantelidesError, PantelidesOutput,
    PantelidesResult, VarIndex,
};
pub use ports::{
    AirPort, BoundaryConditions, EquationSystem, HydronicPort, PortError, PortResult,
    RefrigerantPort, SteamPort,
};

//! fluxion-fluid: Compile-time strongly typed fluid port traits for DAE systems
//!
//! This crate provides:
//! - [`mediums`] - Physical medium types (Air, Water, Refrigerant, Steam)
//! - [`ports`] - FluidPort trait and concrete port implementations
//! - [`pantelides`] - Pantelides symbolic index reduction for DAE systems
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

pub mod mediums;
pub mod pantelides;
pub mod ports;

pub use mediums::{Air, CompatibleWith, Medium, Water};
pub use pantelides::{
    pantelides_reduce, EqIndex, Equation, IncidenceMatrix, PantelidesError, PantelidesOutput,
    PantelidesResult, VarIndex,
};
pub use ports::{
    AirPort, BoundaryConditions, EquationSystem, HydronicPort, PortError, PortResult,
    RefrigerantPort, SteamPort,
};

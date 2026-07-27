//! # fluxion-fluid
//!
//! Compile-time strongly typed fluid port traits for Fluxion HVAC modeling.
//!
//! This crate provides the foundation for acausal HVAC port connections with
//! compile-time type safety. Ports are typed by their medium (Water, Air, Refrigerant)
//! at build time, eliminating runtime type erasure.
//!
//! ## Design Principles
//!
//! - **No runtime type erasure**: Use generics and enums, not `dyn Trait`
//! - **Compile-time type safety**: Port types are verified at compile time
//! - **Trait-based abstraction**: `FluidMedium` for thermophysical properties,
//!   `FluidPort` for inlet/outlet port connections
//!
//! ## Core Traits
//!
//! - [`FluidMedium`] — Thermophysical properties of working fluids
//! - [`FluidPort`] — Typed inlet/outlet ports with medium, temperature, pressure,
//!   mass flow rate

#![deny(clippy::all)]
#![warn(clippy::nursery, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn
)]

pub mod medium;
pub mod port;
pub mod properties;

pub use medium::Medium;
pub use port::{FluidPort, PortDirection, PortSide};
pub use properties::FluidProperties;

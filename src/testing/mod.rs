//! Testing framework for Fluxion
//!
//! Provides integration testing infrastructure, reusable fixtures,
//! and validation utilities for comprehensive system testing.
//!
//! # Modules
//!
//! - `integration`: Integration testing infrastructure with wiring tracing
//! - `reference_data`: Typed loaders for EnergyPlus reference CSVs
//! - `tdd`: TDD helper functions (blind simulation, test climate, assertions)
//! - `tdd_framework`: Test-Driven Development framework for physics accuracy
//!   validation against EnergyPlus reference CSVs

pub mod integration;
pub mod reference_data;
pub mod tdd;
pub mod tdd_framework;

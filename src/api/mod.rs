// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Python API support modules.
//!
//! This module contains components specifically for the Python API,
//! including parameter types and custom exception definitions.

pub mod error;
pub mod parameters;

// Re-export commonly used types
pub use error::FluxionError;
pub use parameters::BuildingParameters;

#[cfg(feature = "python-bindings")]
pub use error::{FluxionErrorPy, SimulationError, SurrogateError, ValidationError};

//! Custom exception types for Fluxion Python API.
//!
//! This module defines domain-specific exception types that provide clear,
//! actionable error messages to Python users. All exceptions inherit from
//! a base FluxionError to enable structured error handling.

#[cfg(feature = "python-bindings")]
use pyo3::create_exception;
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyException;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Rust-side error enumeration for Fluxion.
///
/// This enum provides type-safe error handling within Rust code and
/// automatically converts to appropriate Python exception via
/// `From<FluxionError> for PyErr` implementation.
#[derive(Debug, thiserror::Error)]
pub enum FluxionError {
    /// Parameter validation error (maps to ValidationError in Python)
    #[error("Parameter validation error: {0}")]
    Validation(String),

    /// Surrogate model error (maps to SurrogateError in Python)
    #[error("Surrogate model error: {0}")]
    Surrogate(String),

    /// Simulation error (maps to SimulationError in Python)
    #[error("Simulation error: {0}")]
    Simulation(String),
}

#[cfg(feature = "python-bindings")]
/// Base exception for all Fluxion-specific errors.
///
/// Python users can catch this base type to handle all Fluxion errors uniformly,
/// or catch specific subclasses for fine-grained error handling.
create_exception!(fluxion, PyFluxionError, PyException);

#[cfg(feature = "python-bindings")]
/// Exception raised for parameter validation errors.
///
/// This includes:
/// - Parameter values outside valid ranges (e.g., U-value < 0.1 or > 5.0)
/// - NaN or Infinity values in parameters
/// - Invalid parameter vector lengths
/// - Heating/cooling setpoint conflicts
create_exception!(fluxion, ValidationError, PyFluxionError);

#[cfg(feature = "python-bindings")]
/// Exception raised for surrogate model errors.
///
/// This includes:
/// - ONNX Runtime initialization failures
/// - Model loading errors (file not found, invalid format)
/// - Inference failures (e.g., GPU not available)
/// - Session pool exhaustion
create_exception!(fluxion, SurrogateError, PyFluxionError);

#[cfg(feature = "python-bindings")]
/// Exception raised for simulation errors.
///
/// This includes:
/// - Physics calculation failures (e.g., singular matrices)
/// - NaN/Infinity propagation during simulation
/// - Integration errors
/// - State corruption or invalid thermal network states
create_exception!(fluxion, SimulationError, PyFluxionError);

#[cfg(feature = "python-bindings")]
impl From<FluxionError> for PyErr {
    fn from(err: FluxionError) -> PyErr {
        match err {
            FluxionError::Validation(msg) => ValidationError::new_err(msg),
            FluxionError::Surrogate(msg) => SurrogateError::new_err(msg),
            FluxionError::Simulation(msg) => SimulationError::new_err(msg),
        }
    }
}

#[cfg(feature = "python-bindings")]
pub type FluxionErrorPy = PyFluxionError;

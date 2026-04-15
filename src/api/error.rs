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
create_exception!(fluxion, PyFluxionError, PyException);

#[cfg(feature = "python-bindings")]
create_exception!(fluxion, ValidationError, PyFluxionError);

#[cfg(feature = "python-bindings")]
create_exception!(fluxion, SurrogateError, PyFluxionError);

#[cfg(feature = "python-bindings")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluxion_error_validation_display() {
        let err = FluxionError::Validation("U-value out of range".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Parameter validation error"));
        assert!(msg.contains("U-value out of range"));
    }

    #[test]
    fn test_fluxion_error_surrogate_display() {
        let err = FluxionError::Surrogate("Model not found".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Surrogate model error"));
        assert!(msg.contains("Model not found"));
    }

    #[test]
    fn test_fluxion_error_simulation_display() {
        let err = FluxionError::Simulation("Singular matrix".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Simulation error"));
        assert!(msg.contains("Singular matrix"));
    }

    #[test]
    fn test_fluxion_error_is_debug() {
        let err = FluxionError::Validation("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Validation"));
    }

    #[test]
    fn test_fluxion_error_implements_error_trait() {
        let err: &dyn std::error::Error = &FluxionError::Validation("test".to_string());
        assert!(err.to_string().contains("Parameter validation error"));
    }

    #[test]
    fn test_fluxion_error_matches() {
        let err = FluxionError::Validation("param missing".to_string());
        assert!(matches!(err, FluxionError::Validation(_)));

        let err = FluxionError::Surrogate("onnx failed".to_string());
        assert!(matches!(err, FluxionError::Surrogate(_)));

        let err = FluxionError::Simulation("nan detected".to_string());
        assert!(matches!(err, FluxionError::Simulation(_)));
    }
}

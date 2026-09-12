//! Custom exception types for Fluxion Python API.
//!
//! This module defines domain-specific exception types that provide clear,
//! actionable error messages to Python users. All exceptions inherit from
//! a base FluxionError to enable structured error handling.
//!
//! The Rust-side error type ([`FluxionError`]) and [`SimulationDiagnostics`]
//! are re-exported from the `fluxion-core` leaf crate
//! ([`fluxion_core::error`]) — the ONE unified engine error type. The PyO3
//! exception classes and the `From<FluxionError> for PyErr` mapping below
//! are defined here and keep their exact historic behavior, so the Python
//! `FluxionError` / `ValidationError` / `SurrogateError` /
//! `SimulationError` hierarchy (including the `diagnostics` attribute on
//! `SimulationError`, issue #2547) is unchanged.
//!
//! It also defines [`SimulationDiagnostics`] — a machine-readable record of
//! why a simulation diverged (NaN, infinite temperature, energy-balance
//! violation, non-convergent timestep). Issue #2547 surfaces this on the
//! `ApiError::SimulationFailed` REST envelope and the Python
//! `SimulationError` exception so clients can attribute failure to a
//! specific timestep / zone instead of receiving a bare string.

// Re-exported from the unified leaf-crate error module so every binding
// layer and engine caller shares one `FluxionError` type. The Python
// exception mapping below is defined against this re-exported type.
pub use fluxion_core::error::{FluxionError, SimulationDiagnostics};

#[cfg(feature = "python-bindings")]
use pyo3::create_exception;
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyException;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
create_exception!(fluxion, PyFluxionError, PyException);

#[cfg(feature = "python-bindings")]
create_exception!(fluxion, ValidationError, PyFluxionError);

#[cfg(feature = "python-bindings")]
create_exception!(fluxion, SurrogateError, PyFluxionError);

#[cfg(feature = "python-bindings")]
create_exception!(fluxion, SimulationError, PyFluxionError);

#[cfg(feature = "python-bindings")]
// E0117 note: FluxionError now lives in fluxion-core (a foreign crate), so
// `impl From<FluxionError> for PyErr` is an illegal orphan impl under the
// python-bindings feature. Convert via the local helper instead:
// `something.map_err(fluxion_err_to_pyerr)?`.
#[cfg(feature = "python-bindings")]
pub(crate) fn fluxion_err_to_pyerr(err: FluxionError) -> PyErr {
    py_err_from_fluxion(err)
}

#[cfg(feature = "python-bindings")]
fn py_err_from_fluxion(err: FluxionError) -> PyErr {
    match err {
        FluxionError::Validation(msg) => ValidationError::new_err(msg),
        FluxionError::Surrogate(msg) => SurrogateError::new_err(msg),
        // Issue #2547 — attach the diagnostics dict as a `diagnostics`
        // attribute on the Python `SimulationError` so Python clients
        // can read failing_timestep / failing_zone / max_residual_pct /
        // last_known_good_timestep without parsing the error message.
        FluxionError::Simulation(msg, diagnostics) => Python::attach(|py| {
            let py_err = SimulationError::new_err(msg);
            if let Some(diag) = diagnostics {
                let dict = pyo3::types::PyDict::new(py);
                let _ = dict.set_item("failing_timestep", diag.failing_timestep);
                let _ = dict.set_item("failing_zone", diag.failing_zone.as_deref());
                let _ = dict.set_item("max_residual_pct", diag.max_residual_pct);
                let _ = dict.set_item("last_known_good_timestep", diag.last_known_good_timestep);
                let bound = py_err.value(py);
                let _ = bound.setattr("diagnostics", dict);
            }
            py_err
        }),
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
        let err = FluxionError::Simulation("Singular matrix".to_string(), None);
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

        let err = FluxionError::Simulation("nan detected".to_string(), None);
        assert!(matches!(err, FluxionError::Simulation(..)));
    }

    #[test]
    fn test_simulation_diagnostics_from_clean_trace_is_none() {
        let trace = vec![vec![20.0, 20.5, 21.0]];
        assert!(SimulationDiagnostics::from_temperature_trace(&trace).is_none());
    }

    #[test]
    fn test_simulation_diagnostics_from_nan_trace() {
        let trace = vec![
            vec![20.0, 21.0, f64::NAN, 22.0],
            vec![20.0, 21.0, 22.0, 23.0],
        ];
        let diag = SimulationDiagnostics::from_temperature_trace(&trace)
            .expect("NaN should produce diagnostics");
        assert_eq!(diag.failing_timestep, 2);
        assert_eq!(diag.failing_zone.as_deref(), Some("zone_0"));
        assert_eq!(diag.last_known_good_timestep, 1);
        assert_eq!(diag.max_residual_pct, 0.0);
    }

    #[test]
    fn test_simulation_diagnostics_from_inf_trace() {
        let trace = vec![vec![20.0, f64::INFINITY]];
        let diag = SimulationDiagnostics::from_temperature_trace(&trace).unwrap();
        assert_eq!(diag.failing_timestep, 1);
        assert_eq!(diag.last_known_good_timestep, 0);
    }

    #[test]
    fn test_simulation_diagnostics_serde_round_trip() {
        let diag = SimulationDiagnostics {
            failing_timestep: 42,
            failing_zone: Some("zone_3".to_string()),
            max_residual_pct: 137.5,
            last_known_good_timestep: 41,
        };
        let json = serde_json::to_string(&diag).unwrap();
        let back: SimulationDiagnostics = serde_json::from_str(&json).unwrap();
        assert_eq!(diag, back);
        // Verify JSON field names match the spec in issue #2547.
        assert!(json.contains("\"failing_timestep\":42"));
        assert!(json.contains("\"failing_zone\":\"zone_3\""));
        assert!(json.contains("\"max_residual_pct\":137.5"));
        assert!(json.contains("\"last_known_good_timestep\":41"));
    }
}

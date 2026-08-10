//! Custom exception types for Fluxion Python API.
//!
//! This module defines domain-specific exception types that provide clear,
//! actionable error messages to Python users. All exceptions inherit from
//! a base FluxionError to enable structured error handling.
//!
//! It also defines [`SimulationDiagnostics`] — a machine-readable record of
//! why a simulation diverged (NaN, infinite temperature, energy-balance
//! violation, non-convergent timestep). Issue #2547 surfaces this on the
//! `ApiError::SimulationFailed` REST envelope and the Python
//! `SimulationError` exception so clients can attribute failure to a
//! specific timestep / zone instead of receiving a bare string.

#[cfg(feature = "python-bindings")]
use pyo3::create_exception;
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyException;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Machine-readable divergence diagnostics for a failed simulation.
///
/// Populated from the data the simulation already computes while running
/// (per-timestep zone temperatures, energy-balance residual). When the
/// REST handler or Python binding detects divergence (NaN / infinity /
/// energy-balance violation / non-convergence), it builds a
/// `SimulationDiagnostics` from that data and threads it into
/// `ApiError::SimulationFailed` (REST) and `FluxionError::Simulation`
/// (Python) so clients get failing-timestep, failing-zone, residual and
/// last-known-good-timestep attribution instead of a plain string.
///
/// All fields are `Serialize` so the struct embeds cleanly into the JSON
/// error envelope; `failing_zone` is optional because single-zone models
/// have no inter-zone attribution to report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimulationDiagnostics {
    /// First timestep index (0-based, hourly) at which divergence was
    /// detected — a NaN / infinity in the zone temperature trace, or the
    /// timestep at which the energy-balance residual exceeded tolerance.
    pub failing_timestep: u64,
    /// Zone identifier (1-based label, e.g. `"zone_0"`) attributed with
    /// the divergence, when the diagnostician could isolate one. `None`
    /// for whole-system residuals (e.g. global energy-balance violation)
    /// or for single-zone models.
    pub failing_zone: Option<String>,
    /// Worst energy-balance residual observed across the run, expressed
    /// as a percentage of the tolerance window (`residual / tolerance * 100`).
    /// `0.0` when divergence was detected by NaN/inf scan rather than by
    /// the residual check.
    pub max_residual_pct: f64,
    /// Last timestep index (0-based, hourly) for which all zone
    /// temperatures were finite AND the energy-balance residual was
    /// within tolerance. `0` if divergence was present from the first
    /// timestep. Clients can use this as a safe restart point.
    pub last_known_good_timestep: u64,
}

impl SimulationDiagnostics {
    /// Construct a diagnostics record from the per-zone hourly temperature
    /// trace that `ThermalModel::get_hourly_temperatures` already collects.
    ///
    /// Scans for the first (zone, timestep) cell containing NaN or
    /// infinity, sets `failing_timestep` / `failing_zone` to that cell,
    /// and `last_known_good_timestep` to the preceding timestep (clamped
    /// to 0). `max_residual_pct` is `0.0` because the residual check is
    /// not the source of this divergence.
    ///
    /// Returns `None` when no divergence is present in the trace (no NaN
    /// and no infinity in any zone at any timestep).
    pub fn from_temperature_trace(hourly: &[Vec<f64>]) -> Option<Self> {
        let mut failing_timestep: Option<u64> = None;
        let mut failing_zone: Option<String> = None;

        for (zone_idx, zone_trace) in hourly.iter().enumerate() {
            for (t, &temp) in zone_trace.iter().enumerate() {
                if !temp.is_finite() {
                    let t = t as u64;
                    // Earliest divergence wins across zones — keep the
                    // first (zone, timestep) we see so attribution is
                    // deterministic across multi-zone models.
                    if failing_timestep.is_none_or(|ft| t < ft) {
                        failing_timestep = Some(t);
                        failing_zone = Some(format!("zone_{}", zone_idx));
                    }
                }
            }
        }

        let failing_timestep = failing_timestep?;
        let last_known_good = failing_timestep.saturating_sub(1);

        Some(SimulationDiagnostics {
            failing_timestep,
            failing_zone,
            max_residual_pct: 0.0,
            last_known_good_timestep: last_known_good,
        })
    }
}

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

    /// Simulation error (maps to SimulationError in Python). Carries an
    /// optional [`SimulationDiagnostics`] (Issue #2547) so the Python
    /// exception can surface failing-timestep / failing-zone attribution
    /// instead of a bare message string.
    #[error("Simulation error: {0}")]
    Simulation(String, Option<SimulationDiagnostics>),
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
            // Issue #2547 — attach the diagnostics dict as a `diagnostics`
            // attribute on the Python `SimulationError` so Python clients
            // can read failing_timestep / failing_zone / max_residual_pct /
            // last_known_good_timestep without parsing the error message.
            FluxionError::Simulation(msg, diagnostics) => Python::with_gil(|py| {
                let py_err = SimulationError::new_err(msg);
                if let Some(diag) = diagnostics {
                    let dict = pyo3::types::PyDict::new_bound(py);
                    let _ = dict.set_item("failing_timestep", diag.failing_timestep);
                    let _ = dict.set_item("failing_zone", diag.failing_zone.as_deref());
                    let _ = dict.set_item("max_residual_pct", diag.max_residual_pct);
                    let _ =
                        dict.set_item("last_known_good_timestep", diag.last_known_good_timestep);
                    let bound = py_err.value_bound(py);
                    let _ = bound.setattr("diagnostics", dict);
                }
                py_err
            }),
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

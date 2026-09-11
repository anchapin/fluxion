//! Unified error type for the Fluxion engine.
//!
//! Historically the workspace carried three parallel error systems:
//!
//! - `fluxion::api::error::FluxionError` — a `thiserror` enum with a PyO3
//!   exception mapping for the Python bindings;
//! - `fluxion::napi::error::FluxionError` — a *different* `#[napi]` struct
//!   that happened to share the same name (a trap for anyone wiring new
//!   bindings);
//! - ~165 `Result<_, String>` sites across `src/` and `fluxion-core/src`.
//!
//! This module introduces the ONE unified type — [`FluxionError`] — that all
//! bindings and engine code converge on. It lives in the `fluxion-core` leaf
//! crate so it can be shared without pulling in heavy dependencies: the only
//! crates used here are `thiserror` (the enum) and `serde` (the
//! [`SimulationDiagnostics`] payload), both inside the leaf's dependency
//! budget (see `scripts/check_fluxion_core_dep_budget.py`).
//!
//! [`FluxionError`] is deliberately dependency-light and `Clone`, so binding
//! layers (Python via `fluxion::api::error`, Node via `fluxion::napi::error`)
//! can map it onto their own exception classes while the engine speaks a
//! single type.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Machine-readable divergence diagnostics for a failed simulation.
///
/// Populated from the data the simulation already computes while running
/// (per-timestep zone temperatures, energy-balance residual). When the
/// REST handler or Python binding detects divergence (NaN / infinity /
/// energy-balance violation / non-convergence), it builds a
/// `SimulationDiagnostics` from that data and threads it into
/// `FluxionError::Simulation` so clients get failing-timestep,
/// failing-zone, residual and last-known-good-timestep attribution instead
/// of a plain string.
///
/// All fields are `Serialize` so the struct embeds cleanly into the JSON
/// error envelope; `failing_zone` is optional because single-zone models
/// have no inter-zone attribution to report.
///
/// (Hoisted verbatim from `fluxion::api::error`, issue #2547, so the Python
/// `SimulationError.diagnostics` attribute and the REST
/// `ApiError::SimulationFailed` envelope keep byte-identical payloads.)
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

/// The single unified error type for the Fluxion engine.
///
/// Replaces the fragmented pair of same-named types
/// (`fluxion::api::error::FluxionError`, the `thiserror` enum, and
/// `fluxion::napi::error::FluxionError`, the `#[napi]` struct — the latter
/// is now `NapiFluxionError` with `js_name = "FluxionError"` so the
/// JavaScript API is unchanged).
///
/// Display strings are kept byte-identical to the historic
/// `fluxion::api::error::FluxionError` messages so log lines, REST error
/// envelopes, and the Python exception messages do not change.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FluxionError {
    /// Parameter validation error (maps to `ValidationError` in Python).
    #[error("Parameter validation error: {0}")]
    Validation(String),

    /// Surrogate model error (maps to `SurrogateError` in Python).
    #[error("Surrogate model error: {0}")]
    Surrogate(String),

    /// Simulation error (maps to `SimulationError` in Python). Carries an
    /// optional [`SimulationDiagnostics`] so clients get
    /// failing-timestep / failing-zone attribution instead of a bare
    /// message string.
    #[error("Simulation error: {0}")]
    Simulation(String, Option<SimulationDiagnostics>),
}

/// Convenience alias for fallible Fluxion engine operations.
///
/// New code that would previously have returned `Result<T, String>` should
/// return `FluxionResult<T>` instead, picking the variant that best
/// describes the failure.
pub type FluxionResult<T> = Result<T, FluxionError>;

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
    fn test_fluxion_error_implements_error_trait() {
        let err: &dyn std::error::Error = &FluxionError::Validation("test".to_string());
        assert!(err.to_string().contains("Parameter validation error"));
    }

    #[test]
    fn test_fluxion_error_is_clone_and_partial_eq() {
        let err = FluxionError::Validation("test".to_string());
        assert_eq!(err.clone(), err);
        assert_ne!(err, FluxionError::Surrogate("test".to_string()));
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

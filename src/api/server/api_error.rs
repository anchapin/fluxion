// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `ApiError` and its HTTP envelope (Issue #2530 / #2547).
//!
//! Decomposed from the legacy `server.rs` so the error type and its
//! `IntoResponse` impl are isolated from the rest of the REST surface
//! (Issue #3457 / #3543 — module-size ratchet).

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use thiserror::Error;

use crate::api::error::SimulationDiagnostics;
use crate::api::server::constants::MAX_BATCH_SIMULATIONS;

/// Errors that handlers convert to HTTP responses. Kept inside the module so
/// the public `AppState` / `router` API stays small.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("invalid schema: {0}")]
    InvalidSchema(String),
    #[error("invalid request body: {0}")]
    InvalidRequest(String),
    #[error("schema id not found: {0}")]
    SchemaNotFound(String),
    #[error("simulation id not found: {0}")]
    SimulationNotFound(String),
    #[error("campaign id not found: {0}")]
    CampaignNotFound(String),
    #[error("format '{0}' is not supported by this endpoint")]
    UnsupportedFormat(String),
    #[error("idf import is not yet implemented")]
    IdfNotImplemented,
    #[error("import failed: {0}")]
    ImportFailed(String),
    #[error("simulation failed: {0}")]
    SimulationFailed(String, Option<SimulationDiagnostics>),
    #[error("batch request is empty")]
    EmptyBatch,
    #[error("batch request has {0} simulations, exceeds limit of {MAX_BATCH_SIMULATIONS}")]
    BatchTooLarge(usize),
    #[error("request would schedule {requested} timesteps, exceeds per-request limit of {limit}")]
    StepBudgetExceeded { requested: usize, limit: usize },
    #[error("serialization failed: {0}")]
    SerializationFailed(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self {
            ApiError::InvalidSchema(_) => (StatusCode::BAD_REQUEST, "invalid_schema"),
            ApiError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
            ApiError::SchemaNotFound(_) => (StatusCode::NOT_FOUND, "schema_not_found"),
            ApiError::SimulationNotFound(_) => (StatusCode::NOT_FOUND, "simulation_not_found"),
            ApiError::CampaignNotFound(_) => (StatusCode::NOT_FOUND, "campaign_not_found"),
            ApiError::UnsupportedFormat(_) => (StatusCode::BAD_REQUEST, "unsupported_format"),
            ApiError::IdfNotImplemented => (StatusCode::NOT_IMPLEMENTED, "not_implemented"),
            ApiError::ImportFailed(_) => (StatusCode::UNPROCESSABLE_ENTITY, "import_failed"),
            ApiError::SimulationFailed(_, _) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "simulation_failed")
            }
            ApiError::EmptyBatch => (StatusCode::BAD_REQUEST, "empty_batch"),
            ApiError::BatchTooLarge(_) => (StatusCode::BAD_REQUEST, "batch_too_large"),
            ApiError::StepBudgetExceeded { .. } => {
                (StatusCode::BAD_REQUEST, "step_budget_exceeded")
            }
            ApiError::SerializationFailed(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "serialization_failed")
            }
        };
        // Issue #2547 — when a simulation diverged, embed the machine-readable
        // diagnostics (failing_timestep / failing_zone / max_residual_pct /
        // last_known_good_timestep) into the error envelope so REST clients
        // don't have to parse the human-readable message string. The field is
        // omitted entirely when no diagnostics are present so existing callers
        // see no schema change for non-divergence failures.
        let diagnostics_value = match &self {
            ApiError::SimulationFailed(_, Some(d)) => Some(serde_json::to_value(d).unwrap_or_else(
                |_| serde_json::json!({"error": "diagnostics serialization failed"}),
            )),
            _ => None,
        };
        let mut error_obj = serde_json::json!({
            "kind": kind,
            "message": self.to_string(),
        });
        if let Some(d) = diagnostics_value {
            error_obj["diagnostics"] = d;
        }
        let body = Json(serde_json::json!({
            "error": error_obj
        }));
        (status, body).into_response()
    }
}

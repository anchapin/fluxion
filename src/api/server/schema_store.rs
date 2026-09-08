// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OpenAPI and schema-store endpoints:
//! - `GET /v1/openapi.json` / `GET /v1/openapi.yaml` — embed the OpenAPI 3.1
//!   spec at compile time
//! - `GET /v1/schema/{id}` — fetch a previously stored schema
//!
//! Decomposed from the legacy `server.rs` so the schema-related endpoints
//! live in one focused submodule (Issue #3457 / #3543 — module-size
//! ratchet).

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::api::schema::SimulationSchemaV1;
use crate::api::server::api_error::ApiError;
use crate::api::server::state::AppState;

/// Embed the OpenAPI 3.1 spec at compile time so the binary is self-contained
/// and the spec can never drift from the running code without a rebuild.
const OPENAPI_SPEC: &str = include_str!("../openapi.yaml");

pub async fn openapi_json() -> Response {
    match serde_json::to_string(&serde_json::json!({
        "openapi": "3.1.0",
        "_fluxion_internal_note": "Hand-authored YAML at src/api/openapi.yaml; this JSON envelope mirrors the YAML for clients that prefer JSON.",
        "spec": OPENAPI_SPEC,
    })) {
        Ok(s) => (StatusCode::OK, [("content-type", "application/json")], s).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to serialize OpenAPI envelope: {e}"),
        )
            .into_response(),
    }
}

/// YAML endpoint — exact mirror of the on-disk `openapi.yaml`. Some clients
/// (e.g. swagger-cli) prefer raw YAML over the envelope.
pub async fn openapi_yaml() -> Response {
    (
        StatusCode::OK,
        [("content-type", "application/yaml")],
        OPENAPI_SPEC.to_string(),
    )
        .into_response()
}

/// Fetch a previously stored schema.
pub async fn get_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SimulationSchemaV1>, ApiError> {
    state
        .get(&id)
        .await
        .map(Json)
        .ok_or(ApiError::SchemaNotFound(id))
}

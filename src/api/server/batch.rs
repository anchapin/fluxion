// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `/v1/batch` and `/v1/simulation/{id}/status` endpoints. Decomposed from
//! the legacy `server.rs` so the batch + status polling paths live in one
//! focused submodule (Issue #3457 / #3543 — module-size ratchet).

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::api::metrics;
use crate::api::server::api_error::ApiError;
use crate::api::server::constants::{MAX_BATCH_SIMULATIONS, MAX_CAMPAIGN_STEPS};
use crate::api::server::simulate::{
    parse_selector_from_options, run_simulation, SimulateRequest, SimulateResponse,
};
use crate::api::server::state::AppState;

/// Request body for `POST /v1/batch`.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchRequest {
    pub simulations: Vec<SimulateRequest>,
}

/// Response body for `POST /v1/batch`.
#[derive(Debug, Clone, Serialize)]
pub struct BatchResponse {
    pub results: Vec<Result<SimulateResponse, String>>,
}

/// Get simulation status for async polling via `GET /v1/simulation/{id}/status`.
pub async fn get_simulation_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<crate::api::server::state::SimulationStatus>, ApiError> {
    state
        .get_simulation_status(&id)
        .await
        .map(Json)
        .ok_or(ApiError::SimulationNotFound(id))
}

/// Batch simulation handler for `POST /v1/batch`. Runs multiple simulations
/// concurrently using rayon and returns all results.
#[tracing::instrument(skip_all, fields(request_id, batch_size))]
pub async fn batch_simulate(
    State(_state): State<AppState>,
    headers: axum::http::HeaderMap,
    crate::api::server::simulate::ValidatedJson(req): crate::api::server::simulate::ValidatedJson<
        BatchRequest,
    >,
) -> Result<Json<BatchResponse>, ApiError> {
    use crate::api::server::X_REQUEST_ID;
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let request_id = headers
        .get(X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    tracing::Span::current().record("request_id", request_id.as_str());
    tracing::Span::current().record("batch_size", req.simulations.len());

    if req.simulations.is_empty() {
        return Err(ApiError::EmptyBatch);
    }

    metrics::record_batch_size(req.simulations.len());

    if req.simulations.len() > MAX_BATCH_SIMULATIONS {
        return Err(ApiError::BatchTooLarge(req.simulations.len()));
    }
    let total_steps: usize = req
        .simulations
        .iter()
        .map(|r| (r.options.years as usize) * 8760)
        .sum();
    if total_steps > MAX_CAMPAIGN_STEPS {
        return Err(ApiError::StepBudgetExceeded {
            requested: total_steps,
            limit: MAX_CAMPAIGN_STEPS,
        });
    }

    let schemas: Vec<_> = req
        .simulations
        .iter()
        .map(|r| r.schema.clone().into_v1())
        .collect();
    let opts: Vec<_> = req.simulations.iter().map(|r| r.options.clone()).collect();

    let request_id_for_batch = request_id.clone();
    let results = tokio::task::spawn_blocking(move || {
        schemas
            .into_par_iter()
            .zip(opts.into_par_iter())
            .map(|(schema, options)| {
                let selector = match parse_selector_from_options(&options) {
                    Ok(s) => s,
                    Err(e) => {
                        return Err(e.to_string());
                    }
                };
                run_simulation(
                    &schema,
                    options.years,
                    options.use_surrogates,
                    selector,
                    &request_id_for_batch,
                )
                .map(|output| SimulateResponse {
                    schema_id: None,
                    output,
                })
                .map_err(|e| e.to_string())
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|join_err| {
        ApiError::SimulationFailed(format!("batch blocking task failed: {join_err}"), None)
    })?;

    Ok(Json(BatchResponse { results }))
}

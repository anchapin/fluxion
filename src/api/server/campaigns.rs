// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `/v1/campaigns` fire-and-forget endpoint (Issue #1786). Decomposed from
//! the legacy `server.rs` so the campaign submit / status polling path
//! lives in one focused submodule (Issue #3457 / #3543 — module-size
//! ratchet).

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};

use crate::api::server::api_error::ApiError;
use crate::api::server::constants::{MAX_BATCH_SIMULATIONS, MAX_CAMPAIGN_STEPS};
use crate::api::server::simulate::{parse_selector_from_options, run_simulation, ValidatedJson};
use crate::api::server::state::{
    AppState, CampaignSpec, CampaignState, CampaignStatus, CampaignSubmitResponse,
};

/// Submit a campaign for fire-and-forget execution (Issue #1786).
///
/// The coordinator accepts a campaign spec and returns a campaign ID immediately
/// without waiting for simulations to complete. Workers push status to the
/// state store enabling async polling via `GET /v1/campaigns/{id}/status`.
pub async fn submit_campaign(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    ValidatedJson(spec): ValidatedJson<CampaignSpec>,
) -> Result<Json<CampaignSubmitResponse>, ApiError> {
    use crate::api::server::X_REQUEST_ID;

    let request_id = headers
        .get(X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    if spec.simulations.is_empty() {
        return Err(ApiError::EmptyBatch);
    }

    if spec.simulations.len() > MAX_BATCH_SIMULATIONS {
        return Err(ApiError::BatchTooLarge(spec.simulations.len()));
    }
    let total_steps: usize = spec
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

    let campaign_id = state.register_campaign(spec.clone()).await;
    let campaign_id_for_task = campaign_id.clone();
    let request_id_for_task = request_id.clone();

    let campaigns = Arc::clone(&state.campaigns);

    tokio::spawn(async move {
        let total = spec.simulations.len();

        {
            let mut guard = campaigns.write();
            if let Some(current) = guard.get_mut(&campaign_id_for_task) {
                *current = CampaignState::Running {
                    spec: spec.clone(),
                    progress: 0.0,
                    completed: 0,
                };
            }
        }

        let mut results: Vec<Result<crate::api::schema::SimulationOutput, String>> =
            Vec::with_capacity(total);

        for (i, sim_req) in spec.simulations.iter().enumerate() {
            let schema = sim_req.schema.clone().into_v1();
            let years = sim_req.options.years;
            let use_surrogates = sim_req.options.use_surrogates;
            let selector = match parse_selector_from_options(&sim_req.options) {
                Ok(s) => s,
                Err(e) => {
                    results.push(Err(e.to_string()));
                    continue;
                }
            };

            let result = run_simulation(
                &schema,
                years,
                use_surrogates,
                selector,
                &request_id_for_task,
            )
            .map_err(|e| e.to_string());

            results.push(result);

            let progress = (i + 1) as f32 / total as f32;
            let completed = i + 1;

            {
                let mut guard = campaigns.write();
                if let Some(current) = guard.get_mut(&campaign_id_for_task) {
                    *current = CampaignState::Running {
                        spec: spec.clone(),
                        progress,
                        completed,
                    };
                }
            }
        }

        {
            let mut guard = campaigns.write();
            if let Some(current) = guard.get_mut(&campaign_id_for_task) {
                *current = CampaignState::Completed {
                    spec: spec.clone(),
                    results,
                };
            }
        }
    });

    Ok(Json(CampaignSubmitResponse { campaign_id }))
}

/// Get campaign status for async polling via `GET /v1/campaigns/{id}/status`.
pub async fn get_campaign_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<CampaignStatus>, ApiError> {
    state
        .get_campaign_status(&id)
        .await
        .map(Json)
        .ok_or(ApiError::CampaignNotFound(id))
}

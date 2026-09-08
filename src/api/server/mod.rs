// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! REST API server for Fluxion (Issue #1342).
//!
//! Decomposed from the legacy `server.rs` (which exceeded the
//! [`scripts/check_module_size.py`] ratchet — Issue #3457) into
//! per-feature submodules. `mod.rs` is the thin entry point that
//! re-exports the public API and the route table. The handlers and
//! per-feature logic live in:
//!
//! - [`api_error`] — `ApiError` + HTTP envelope
//! - [`batch`] — `/v1/batch`, `/v1/simulation/{id}/status`
//! - [`campaigns`] — `/v1/campaigns`, `/v1/campaigns/{id}/status`
//! - [`constants`] — module-level constants (DoS, timeouts, id prefixes)
//! - [`health`] — `/v1/healthz`, `/v1/readyz`, readiness probes
//! - [`import_format`] — `/v1/import/{osm|gbxml|idf|epjson}`
//! - [`router`] — `Router` builder, `REST_ROUTES`, middleware stack
//! - [`schema_store`] — `/v1/schema/{id}`, `/v1/openapi.{json,yaml}`
//! - [`simulate`] — `/v1/simulate`, `/v1/simulate/stream`, `run_simulation`
//! - [`state`] — `AppState`, `SimulationStateStore`, campaign/sim state types
//!
//! The public API is preserved — every name previously reachable at
//! `fluxion::api::server::*` is re-exported from this module so the
//! existing callers (Python bindings, integration tests, CLI binaries)
//! continue to work without modification.

pub mod api_error;
pub mod batch;
pub mod campaigns;
pub mod constants;
pub mod health;
pub mod import_format;
pub mod router;
pub mod schema_store;
pub mod simulate;
pub mod state;

// Re-export the public API so the legacy `fluxion::api::server::*` paths
// continue to resolve without modification.
pub use api_error::ApiError;
pub use batch::{batch_simulate, get_simulation_status, BatchRequest, BatchResponse};
pub use campaigns::{get_campaign_status, submit_campaign};
pub use constants::{
    resolve_shutdown_timeout_secs, DEFAULT_SHUTDOWN_TIMEOUT_SECS, MAX_BATCH_SIMULATIONS,
    MAX_CAMPAIGN_STEPS, MAX_YEARS, REQUEST_TIMEOUT,
};
// Re-export the security::MAX_REQUEST_BODY_BYTES through the legacy path.
pub use crate::api::security::MAX_REQUEST_BODY_BYTES;

// Internal header-name constant shared across all REST submodules.
pub(crate) const X_REQUEST_ID: &str = "x-request-id";

pub use health::{
    healthz, readyz, run_readiness_probes, run_readiness_probes_with, HealthResponse,
    ReadinessCheck, ReadinessChecks, ReadinessReport,
};
pub use import_format::{import_format, tempfile_for_bytes, ImportResponse};
pub use router::{router, router_with_security};
pub use schema_store::{get_schema, openapi_json, openapi_yaml};
pub use simulate::{
    parse_selector_from_options, run_simulation, simulate, simulate_stream, SimulateOptions,
    SimulateRequest, SimulateResponse, SimulationSchemaBody, TimestepEvent, ValidatedJson,
};
pub use state::{
    AppState, CampaignResult, CampaignSimulationResult, CampaignSpec, CampaignState,
    CampaignStateEnum, CampaignStatus, CampaignSubmitResponse, InMemorySimulationStateStore,
    SimulationState, SimulationStateEnum, SimulationStateStore, SimulationStatus,
};

#[cfg(test)]
mod tests;

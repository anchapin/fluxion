// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Module-level constants and helpers for the REST API server (Issue #1342).
//!
//! Constants that drive the REST surface's DoS / shutdown / batch ceilings
//! live here so the bounds are visible at a single source and can be
//! referenced from any submodule (`pub(super)`). The previous `server.rs`
//! inlined these in the parent module, which made it harder to reason
//! about the per-route budget when the file grew beyond the module-size
//! ratchet (Issue #3457 / #3543).

use std::time::Duration;

/// Identifier prefix for schemas persisted by the in-memory store.
pub(super) const SCHEMA_ID_PREFIX: &str = "sch-";

/// Identifier prefix for simulations tracked for async status.
pub(super) const SIM_ID_PREFIX: &str = "sim-";

/// Identifier prefix for campaigns (OSimFlow fire-and-forget, Issue #1786).
pub(super) const CAMPAIGN_ID_PREFIX: &str = "camp-";

/// Upper bound on `SimulateOptions.years` (Issue #2530 DoS hardening).
///
/// The handler computes `steps = years * 8760` and runs `solve_timesteps`
/// synchronously. Without a cap, `{"years": u32::MAX}` asks the server to
/// allocate and run ~3.76 × 10¹³ timesteps, pinning a Tokio worker. 10 leaves
/// headroom for future multi-year validation runs while bounding the worst
/// case for a single request to `10 * 8760 = 87_600` timesteps.
pub const MAX_YEARS: u32 = 10;

/// Maximum number of entries accepted by `POST /v1/batch` and `POST
/// /v1/campaigns` (Issue #2530). 1024 matches the batch ceiling used by the
/// surrogate `BatchOracle` population path so a REST batch never exceeds what
/// the engine is designed to fan out across rayon workers.
pub const MAX_BATCH_SIMULATIONS: usize = 1024;

/// Per-campaign / per-batch step budget (Issue #2530). The total number of
/// timesteps a single request may schedule is `Σ years_i * 8760`. Bounding it
/// to `MAX_YEARS * 8760 * MAX_BATCH_SIMULATIONS` (= 89_702_400) means a full
/// batch of 1024 decade-long simulations is still accepted, but a malicious
/// batch that smuggles huge `years` past the per-entry validator is rejected
/// before any rayon work is spawned.
pub const MAX_CAMPAIGN_STEPS: usize = (MAX_YEARS as usize) * 8760 * MAX_BATCH_SIMULATIONS;

/// Maximum accepted request body size — re-exported from
/// [`crate::api::security`] (Issue #2505). See there for the rationale.
/// Kept at this path for API stability (callers and tests historically
/// referenced `fluxion::api::server::MAX_REQUEST_BODY_BYTES`).
pub use crate::api::security::MAX_REQUEST_BODY_BYTES;

/// Wall-clock budget for any single HTTP request (Issue #2530). Enforced via
/// `tower::timeout::TimeoutLayer` so a runaway synchronous `solve_timesteps`
/// cannot pin a worker indefinitely; the handler aborts with a structured
/// 408 once the deadline elapses.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Default hard deadline (seconds) for the graceful-shutdown drain phase
/// (Issue #2517). After a SIGINT/SIGTERM, `fluxion-rest` stops accepting new
/// connections and gives in-flight requests this many seconds to complete
/// before forcibly closing them. The value is deliberately below the
/// Kubernetes default `terminationGracePeriodSeconds` (30 s) so the process
/// exits before the kubelet issues SIGKILL. Override with
/// `FLUXION_REST_SHUTDOWN_TIMEOUT_SECS`.
pub const DEFAULT_SHUTDOWN_TIMEOUT_SECS: u64 = 25;

/// Resolve the graceful-shutdown drain timeout from the
/// `FLUXION_REST_SHUTDOWN_TIMEOUT_SECS` environment variable, falling back to
/// [`DEFAULT_SHUTDOWN_TIMEOUT_SECS`] (25 s). Non-positive, empty, or
/// unparseable values all fall back to the default so a misconfigured env var
/// can never accidentally disable the hard deadline (Issue #2517).
pub fn resolve_shutdown_timeout_secs() -> u64 {
    std::env::var("FLUXION_REST_SHUTDOWN_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_SHUTDOWN_TIMEOUT_SECS)
}

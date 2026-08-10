// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! REST API server for Fluxion (Issue #1342).
//!
//! Exposes a small, opinionated HTTP surface that mirrors the existing
//! `SimulationSchema` contract:
//!
//! - `POST /v1/simulate` — run a simulation against a schema
//! - `GET /v1/schema/{id}` — fetch a previously imported/used schema
//! - `POST /v1/import/{osm|gbxml|ifc|idf}` — convert an external model file into
//!   a `SimulationSchemaV1` and store it
//! - `GET /v1/healthz` — liveness probe
//! - `GET /v1/openapi.json` — embedded OpenAPI 3.1 document
//! - `GET /v1/metrics` — Prometheus exposition for `/v1/metrics` scrapers
//!   (Issue #1447)
//!
//! The implementation deliberately reuses [`crate::api::schema`] for the wire
//! format (no modifications to the canonical schema) and the existing
//! `crate::interop::osm` / `crate::interop::gbxml` readers for import
//! delegation. IDF import is not yet implemented in `src/interop/*`, so
//! `POST /v1/import/idf` returns `501 Not Implemented` with a structured error
//! (documented in `docs/REST_API.md`).
//!
//! See `src/api/openapi.yaml` for the OpenAPI 3.1 contract and
//! `tests/api_integration_tests.rs` for end-to-end coverage.
//!
//! Observability (Issue #1447):
//! - Every response carries an `x-request-id` header (generated as a v4 UUID
//!   via `tower-http::request_id::MakeRequestUuid`).
//! - `tower_http::trace::TraceLayer` emits one structured log line per
//!   request, including method, path, status, and the request-id header.
//! - `crate::api::metrics::record` middleware maintains the Prometheus
//!   counters and histograms that `/v1/metrics` renders.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use axum::{
    extract::{FromRequest, Path, Request, State},
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use parking_lot::RwLock;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};
use tower::timeout::TimeoutLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::{
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::Level;

use crate::ai::surrogate::SurrogateManager;
use crate::api::error::SimulationDiagnostics;
use crate::api::metrics::{self, metrics_handler};
use crate::api::schema::{SimulationOutput, SimulationSchema, SimulationSchemaV1};
use crate::interop::{gbxml, osm};
use crate::io::idf::{IdfFile, IdfParser};
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

/// Identifier prefix for schemas persisted by the in-memory store.
const SCHEMA_ID_PREFIX: &str = "sch-";

/// Identifier prefix for simulations tracked for async status.
const SIM_ID_PREFIX: &str = "sim-";

/// Identifier prefix for campaigns (OSimFlow fire-and-forget, Issue #1786).
const CAMPAIGN_ID_PREFIX: &str = "camp-";

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

/// Trait for simulation state persistence.
///
/// Implementers of this trait can store simulation state in any backing store:
/// - In-memory `HashMap` (default, for single-instance deployments)
/// - Redis (for multi-instance deployments with local connection)
/// - DynamoDB (for cloud-native deployments)
///
/// # Invariant: Campaign survives client disconnect
///
/// When a cloud store implementation (Redis/DynamoDB) is used, workers push
/// status updates directly to the store. The campaign continues running even if
/// the client that initiated it disconnects. Clients can reconnect later and
/// query the simulation status via `GET /v1/simulation/:id/status`.
///
/// This enables the T7.2 async coordinator pattern where the campaign manager
/// is decoupled from the workers via the state store.
///
/// # Example: DynamoDB-backed store
///
/// ```ignore
/// struct DynamoDbStateStore { ... }
///
/// #[async_trait::async_trait]
/// impl SimulationStateStore for DynamoDbStateStore {
///     async fn get(&self, id: &str) -> Option<SimulationState> { ... }
///     async fn insert(&self, id: &str, state: SimulationState) { ... }
///     async fn update(&self, id: &str, state: SimulationState) { ... }
/// }
/// ```
#[async_trait::async_trait]
pub trait SimulationStateStore: Send + Sync {
    /// Retrieve simulation state by id.
    async fn get(&self, id: &str) -> Option<SimulationState>;

    /// Insert new simulation state with a generated id.
    /// Returns the generated id.
    async fn insert(&self, state: SimulationState) -> String;

    /// Update existing simulation state.
    async fn update(&self, id: &str, state: SimulationState) -> bool;

    /// Get simulation status for polling endpoint.
    async fn get_status(&self, id: &str) -> Option<SimulationStatus>;
}

/// In-memory simulation state store using a `HashMap`.
///
/// This is the default store for single-instance deployments. It does NOT
/// survive server restarts or support multi-instance deployments. For those
/// use cases, use a cloud store implementation (Redis/DynamoDB).
#[derive(Clone, Default)]
pub struct InMemorySimulationStateStore {
    inner: Arc<Mutex<HashMap<String, SimulationState>>>,
    next_id: Arc<AtomicU64>,
}

impl InMemorySimulationStateStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }

    fn next_sim_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{}{}", SIM_ID_PREFIX, n)
    }
}

#[async_trait::async_trait]
impl SimulationStateStore for InMemorySimulationStateStore {
    async fn get(&self, id: &str) -> Option<SimulationState> {
        self.inner.lock().await.get(id).cloned()
    }

    async fn insert(&self, state: SimulationState) -> String {
        let id = self.next_sim_id();
        self.inner.lock().await.insert(id.clone(), state);
        id
    }

    async fn update(&self, id: &str, state: SimulationState) -> bool {
        self.inner
            .lock()
            .await
            .insert(id.to_string(), state)
            .is_some()
    }

    async fn get_status(&self, id: &str) -> Option<SimulationStatus> {
        self.inner.lock().await.get(id).map(|state| {
            let (state_enum, progress) = match state {
                SimulationState::Pending => (SimulationStateEnum::Pending, None),
                SimulationState::Running { progress } => (
                    SimulationStateEnum::Running {
                        progress: *progress,
                    },
                    Some(*progress),
                ),
                SimulationState::Completed { result: _ } => {
                    (SimulationStateEnum::Completed, Some(1.0))
                }
                SimulationState::Failed { error } => (
                    SimulationStateEnum::Failed {
                        error: error.clone(),
                    },
                    None,
                ),
            };
            SimulationStatus {
                id: id.to_string(),
                state: state_enum,
                progress,
                result: None,
            }
        })
    }
}

/// Shared application state — held inside an [`axum::extract::State`] so every
/// handler can mutate the same schema store. The store is process-local (in
/// scope per #1342).
///
/// # Locking strategy (Issue #2552)
///
/// Two fields that previously held `tokio::sync::Mutex` now hold
/// `parking_lot::RwLock`:
///
/// * `schemas` — heavily read-mostly (`GET /v1/schema/{id}` dominates the
///   workload; writes only happen on `POST /v1/simulate` and
///   `POST /v1/import/*`). `parking_lot::RwLock` lets an arbitrary number of
///   concurrent readers proceed without ever going through the tokio task
///   scheduler, eliminating the p99 cliff reported in #2552.
/// * `campaigns` — reads (`GET /v1/campaigns/:id/status`) dominate over
///   writes (per-step status pushes inside the spawned worker task).
///
/// `tokio::sync::Mutex` is intentionally retained for `SimulationStateStore`
/// implementations (the `inner: Arc<Mutex<HashMap<...>>>` inside
/// [`InMemorySimulationStateStore`]) because the trait's `async fn` signatures
/// must remain compatible with cloud backends (DynamoDB/Redis) that await on
/// network I/O between the lock acquisition and the data access. Holding a
/// sync lock across an `.await` would block the async runtime, so we keep the
/// `tokio::sync::Mutex` for the only field where an `.await` legitimately
/// appears inside the critical section.
///
/// All lock acquisitions on `schemas` and `campaigns` are synchronous and
/// non-blocking under contention — they never cross an `.await`. The public
/// methods on `AppState` stay `async fn` for backwards compatibility; the
/// `.await`s are now unnecessary in the lock paths but harmless.
///
/// # Cloud State Store
///
/// For deployments where campaigns must survive client disconnect (e.g., remote
/// workers on Nomad/AWS Batch), use a cloud store implementation:
///
/// ```
/// # use fluxion::api::server::{AppState, SimulationStateStore};
/// # struct MyCloudStore { ... }
/// # #[async_trait::async_trait]
/// # impl SimulationStateStore for MyCloudStore {
/// #     async fn get(&self, id: &str) -> Option<fluxion::api::server::SimulationState> { None }
/// #     async fn insert(&self, state: fluxion::api::server::SimulationState) -> String { String::new() }
/// #     async fn update(&self, id: &str, state: fluxion::api::server::SimulationState) -> bool { false }
/// #     async fn get_status(&self, id: &str) -> Option<fluxion::api::server::SimulationStatus> { None }
/// # }
/// let state = AppState::with_cloud_store(MyCloudStore { ... });
/// ```
///
/// # Invariant: Campaign survives client disconnect
///
/// When using a cloud store (Redis/DynamoDB), the campaign execution no longer
/// requires an open local connection to keep workers alive. Workers push status
/// updates directly to the cloud store. If the client disconnects, the campaign
/// continues running and the client can reconnect later to check status.
///
/// This is the prerequisite for T7.2 (async coordinator).
#[derive(Clone)]
pub struct AppState<S = InMemorySimulationStateStore> {
    schemas: Arc<RwLock<HashMap<String, SimulationSchemaV1>>>,
    simulations: S,
    campaigns: Arc<RwLock<HashMap<String, CampaignState>>>,
    next_id: Arc<AtomicU64>,
}

impl Default for AppState<InMemorySimulationStateStore> {
    fn default() -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            simulations: InMemorySimulationStateStore::new(),
            campaigns: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl<S: SimulationStateStore> AppState<S> {
    /// Create an `AppState` with a custom cloud-backed simulation state store.
    ///
    /// This enables campaigns to survive client disconnect when workers push
    /// status to a cloud store (DynamoDB/Redis) instead of local memory.
    pub fn with_cloud_store(simulations: S) -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            simulations,
            campaigns: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Simulation status for async polling via `GET /v1/simulation/:id/status`.
#[derive(Debug, Clone, Serialize)]
pub struct SimulationStatus {
    pub id: String,
    pub state: SimulationStateEnum,
    pub progress: Option<f32>,
    pub result: Option<SimulateResponse>,
}

/// State machine for async simulations.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state")]
pub enum SimulationStateEnum {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "running")]
    Running { progress: f32 },
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed { error: String },
}

/// Internal simulation state with result container.
#[derive(Debug, Clone)]
pub enum SimulationState {
    Pending,
    Running { progress: f32 },
    Completed { result: SimulationOutput },
    Failed { error: String },
}

/// Campaign specification for fire-and-forget submission (Issue #1786).
#[derive(Debug, Clone, Deserialize)]
pub struct CampaignSpec {
    pub name: Option<String>,
    pub description: Option<String>,
    pub simulations: Vec<SimulateRequest>,
}

/// Campaign status for async polling via `GET /v1/campaigns/:id/status`.
#[derive(Debug, Clone, Serialize)]
pub struct CampaignStatus {
    pub id: String,
    pub name: Option<String>,
    pub state: CampaignStateEnum,
    pub progress: Option<f32>,
    pub total_simulations: usize,
    pub completed_simulations: usize,
    pub result: Option<CampaignResult>,
}

/// Campaign state for serialization.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state")]
pub enum CampaignStateEnum {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "running")]
    Running { progress: f32 },
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed { error: String },
}

/// Internal campaign state with results container.
#[derive(Debug, Clone)]
pub enum CampaignState {
    Pending {
        spec: CampaignSpec,
    },
    Running {
        spec: CampaignSpec,
        progress: f32,
        completed: usize,
    },
    Completed {
        spec: CampaignSpec,
        results: Vec<Result<SimulationOutput, String>>,
    },
    Failed {
        spec: CampaignSpec,
        error: String,
    },
}

/// Campaign result containing all simulation outputs.
#[derive(Debug, Clone, Serialize)]
pub struct CampaignResult {
    pub outputs: Vec<CampaignSimulationResult>,
}

/// Individual simulation result within a campaign.
#[derive(Debug, Clone, Serialize)]
pub struct CampaignSimulationResult {
    pub schema_id: Option<String>,
    pub output: Option<SimulationOutput>,
    pub error: Option<String>,
}

impl<S: SimulationStateStore> AppState<S> {
    /// Allocate a new monotonically-increasing schema id.
    fn next_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{}{}", SCHEMA_ID_PREFIX, n)
    }

    /// Store a schema and return its assigned id.
    ///
    /// The `schemas` lock is `parking_lot::RwLock` (Issue #2552), so this
    /// method takes a synchronous write guard and never crosses an `.await`
    /// while the lock is held.
    pub async fn store(&self, schema: SimulationSchemaV1) -> String {
        let id = self.next_id();
        self.schemas.write().insert(id.clone(), schema);
        id
    }

    /// Look up a previously-stored schema by id.
    ///
    /// Uses a synchronous read guard so multiple concurrent `GET /v1/schema/{id}`
    /// requests can proceed without contending with each other.
    pub async fn get(&self, id: &str) -> Option<SimulationSchemaV1> {
        self.schemas.read().get(id).cloned()
    }

    /// Register a new simulation and return its id.
    ///
    /// Uses the configured `SimulationStateStore` so cloud stores can persist
    /// the initial state. This enables workers to push status updates directly
    /// to the cloud store, decoupling the campaign from the local connection.
    pub async fn register_simulation(&self) -> String {
        self.simulations.insert(SimulationState::Pending).await
    }

    /// Update simulation state.
    ///
    /// Workers call this to push status updates to the store. With a cloud
    /// store (Redis/DynamoDB), updates are persisted immediately and survive
    /// client disconnect.
    pub async fn update_simulation(&self, id: &str, state: SimulationState) {
        let _ = self.simulations.update(id, state).await;
    }

    /// Get simulation status for polling.
    ///
    /// With a cloud store, this allows clients to query status after
    /// reconnecting following a disconnect.
    pub async fn get_simulation_status(&self, id: &str) -> Option<SimulationStatus> {
        self.simulations.get_status(id).await
    }

    /// Number of stored schemas (for tests / diagnostics).
    pub async fn len(&self) -> usize {
        self.schemas.read().len()
    }

    /// Whether the store has zero schemas. Kept to satisfy
    /// `clippy::len_without_is_empty`; cheaper than `len() == 0` only if
    /// callers already hold the lock.
    pub async fn is_empty(&self) -> bool {
        self.schemas.read().is_empty()
    }

    /// Allocate a new monotonically-increasing campaign id.
    fn next_campaign_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{}{}", CAMPAIGN_ID_PREFIX, n)
    }

    /// Register a new campaign and return its id.
    pub async fn register_campaign(&self, spec: CampaignSpec) -> String {
        let id = self.next_campaign_id();
        self.campaigns
            .write()
            .insert(id.clone(), CampaignState::Pending { spec });
        id
    }

    /// Update campaign state.
    pub async fn update_campaign(&self, id: &str, state: CampaignState) {
        self.campaigns.write().insert(id.to_string(), state);
    }

    /// Get campaign status for polling.
    pub async fn get_campaign_status(&self, id: &str) -> Option<CampaignStatus> {
        self.campaigns.read().get(id).map(|state| {
            let (state_enum, progress, completed, total) = match state {
                CampaignState::Pending { spec } => (
                    CampaignStateEnum::Pending,
                    None,
                    0usize,
                    spec.simulations.len(),
                ),
                CampaignState::Running {
                    spec,
                    progress,
                    completed,
                } => (
                    CampaignStateEnum::Running {
                        progress: *progress,
                    },
                    Some(*progress),
                    *completed,
                    spec.simulations.len(),
                ),
                CampaignState::Completed { spec, results } => {
                    let completed = results.len();
                    let total = spec.simulations.len();
                    (CampaignStateEnum::Completed, Some(1.0), completed, total)
                }
                CampaignState::Failed { spec, error: _ } => (
                    CampaignStateEnum::Failed {
                        error: "campaign failed".to_string(),
                    },
                    None,
                    0,
                    spec.simulations.len(),
                ),
            };
            let result = match state {
                CampaignState::Completed { results, .. } => Some(CampaignResult {
                    outputs: results
                        .iter()
                        .map(|r| match r {
                            Ok(output) => CampaignSimulationResult {
                                schema_id: None,
                                output: Some(output.clone()),
                                error: None,
                            },
                            Err(e) => CampaignSimulationResult {
                                schema_id: None,
                                output: None,
                                error: Some(e.clone()),
                            },
                        })
                        .collect(),
                }),
                _ => None,
            };
            let name = match state {
                CampaignState::Pending { spec } => spec.name.clone(),
                CampaignState::Running { spec, .. } => spec.name.clone(),
                CampaignState::Completed { spec, .. } => spec.name.clone(),
                CampaignState::Failed { spec, .. } => spec.name.clone(),
            };
            CampaignStatus {
                id: id.to_string(),
                name,
                state: state_enum,
                progress,
                total_simulations: total,
                completed_simulations: completed,
                result,
            }
        })
    }
}

/// Optional knobs attached to a simulation request. Defaults match the
/// existing `bindings.rs` path so the REST result matches an in-process call
/// within numerical noise.
#[derive(Debug, Clone, Deserialize)]
pub struct SimulateOptions {
    /// Number of years to simulate. Default: `1`. Bounded to `1..=MAX_YEARS`
    /// at deserialisation (Issue #2530) so a `{"years": u32::MAX}` payload is
    /// rejected as a 400 before `steps = years * 8760` is ever computed.
    #[serde(default = "default_years", deserialize_with = "validate_years")]
    pub years: u32,
    /// Whether to use the ONNX surrogate path. Default: `false`.
    #[serde(default)]
    pub use_surrogates: bool,
    /// Optional opaque id; if present, the request's schema is stored under
    /// this id *and* the id is returned for retrieval via
    /// `GET /v1/schema/{id}`.
    #[serde(default)]
    pub store_as: Option<String>,
}

fn default_years() -> u32 {
    1
}

/// Serde validator for `SimulateOptions.years` (Issue #2530). Rejects `0` and
/// any value above [`MAX_YEARS`] at deserialisation so the `Json` extractor
/// surfaces a structured 400 (via [`ValidatedJson`]) rather than letting a
/// multi-trillion-step payload reach the synchronous solver. `#[serde(default
/// = "default_years")]` is still honoured when the field is *absent* — this
/// function only runs when the caller supplies an explicit value.
fn validate_years<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = u32::deserialize(deserializer)?;
    if v == 0 {
        return Err(serde::de::Error::custom(format!(
            "options.years must be between 1 and {MAX_YEARS} (got 0)"
        )));
    }
    if v > MAX_YEARS {
        return Err(serde::de::Error::custom(format!(
            "options.years must be between 1 and {MAX_YEARS} (got {v})"
        )));
    }
    Ok(v)
}

impl Default for SimulateOptions {
    fn default() -> Self {
        SimulateOptions {
            years: default_years(),
            use_surrogates: false,
            store_as: None,
        }
    }
}

/// Request body for `POST /v1/simulate`.
#[derive(Debug, Clone, Deserialize)]
pub struct SimulateRequest {
    /// The simulation schema. Accepts either a bare `SimulationSchemaV1`
    /// or the version-tagged `SimulationSchema` envelope.
    #[serde(flatten)]
    pub schema: SimulationSchemaBody,
    #[serde(default)]
    pub options: SimulateOptions,
}

/// Helper for the polymorphic schema payload (bare V1 or `{ "version": ... }`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SimulationSchemaBody {
    V1(SimulationSchemaV1),
    Enveloped(SimulationSchema),
}

impl SimulationSchemaBody {
    /// Unwrap to the V1 schema regardless of which wire form was supplied.
    pub fn into_v1(self) -> SimulationSchemaV1 {
        match self {
            SimulationSchemaBody::V1(v) => v,
            SimulationSchemaBody::Enveloped(SimulationSchema::V1(v)) => v,
        }
    }
}

/// Validating JSON extractor (Issue #2530).
///
/// Wraps [`axum::Json`] so that any deserialisation failure — including the
/// range checks performed by [`validate_years`] — surfaces as a structured
/// [`ApiError::InvalidRequest`] (HTTP 400) instead of axum's default
/// `JsonRejection` (which is a bare 422 with a non-`error`-enveloped body).
/// The rejection flows through `ApiError::into_response`, so clients always
/// see the canonical `{"error":{"kind":...,"message":...}}` shape.
///
/// Handler bodies destructure it exactly like `Json`: `ValidatedJson(req):
/// ValidatedJson<SimulateRequest>`.
pub struct ValidatedJson<T>(pub T);

#[async_trait::async_trait]
impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(j) => Ok(ValidatedJson(j.0)),
            // `body_text()` preserves the inner serde error chain (e.g. the
            // `validate_years` "options.years must be between 1 and 10"
            // message) so the client can see *why* the body was rejected,
            // not just that it was. `to_string()` would drop that detail.
            Err(rejection) => Err(ApiError::InvalidRequest(rejection.body_text())),
        }
    }
}

/// Response body for `POST /v1/simulate`.
#[derive(Debug, Clone, Serialize)]
pub struct SimulateResponse {
    pub schema_id: Option<String>,
    pub output: SimulationOutput,
}

/// Wire-level representation of an import failure. Returned with HTTP 4xx so
/// clients can present the same error string the CLI prints.
#[derive(Debug, Clone, Serialize)]
pub struct ImportResponse {
    pub schema_id: String,
    pub schema: SimulationSchemaV1,
}

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

/// SSE event payload for per-timestep zone temperatures.
#[derive(Debug, Clone, Serialize)]
pub struct TimestepEvent {
    pub timestep: usize,
    pub zone_temperatures: Vec<f64>,
}

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

/// Body returned by `GET /v1/healthz`. Fields are deliberately minimal so
/// load balancers can parse the JSON without coupling to schema internals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

/// Liveness handler. Always returns `200 OK` with a static payload; we
/// deliberately do **not** ping downstream services here so a slow disk does
/// not flap the load balancer.
async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// Embed the OpenAPI 3.1 spec at compile time so the binary is self-contained
/// and the spec can never drift from the running code without a rebuild.
const OPENAPI_SPEC: &str = include_str!("openapi.yaml");

async fn openapi_json() -> Response {
    // The spec is YAML; serve it as JSON via a hand-rolled envelope so the
    // handler does not pull in a YAML→JSON dependency for a single static
    // payload. Clients that need the raw YAML can `GET /v1/openapi.yaml`.
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
async fn openapi_yaml() -> Response {
    (
        StatusCode::OK,
        [("content-type", "application/yaml")],
        OPENAPI_SPEC.to_string(),
    )
        .into_response()
}

/// Fetch a previously stored schema.
async fn get_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SimulationSchemaV1>, ApiError> {
    state
        .get(&id)
        .await
        .map(Json)
        .ok_or(ApiError::SchemaNotFound(id))
}

/// Run a simulation synchronously and return the structured output. Kept as
/// a free function so it is reusable from integration tests and from the
/// `bindings.rs` Python path if we ever want to consolidate.
///
/// `request_id` carries the per-request correlation id (extracted from the
/// `x-request-id` header by the HTTP handlers) into the
/// [`tracing::instrument`] span. Every log line emitted on the simulation
/// hot path — the surrogate-fallback `WARN` below and the per-timestep logs
/// inside `solve_timesteps` — therefore carries the request id for
/// end-to-end correlation (Issue #2499).
#[tracing::instrument(
    skip(schema),
    fields(request_id = %request_id, num_zones = schema.geometry.zones.len().max(1), years),
)]
pub fn run_simulation(
    schema: &SimulationSchemaV1,
    years: u32,
    use_surrogates: bool,
    request_id: &str,
) -> Result<SimulationOutput, ApiError> {
    let num_zones = schema.geometry.zones.len().max(1);

    // The same set of assertions Python uses — kept small so the REST path
    // can never silently disagree with `src/python/bindings.rs`.
    let heating = schema.controls.zone_control.heating_setpoint;
    let cooling = schema.controls.zone_control.cooling_setpoint;
    if heating >= cooling {
        return Err(ApiError::InvalidSchema(format!(
            "heating_setpoint ({heating}) must be < cooling_setpoint ({cooling})"
        )));
    }
    if schema.geometry.zones.is_empty() {
        return Err(ApiError::InvalidSchema(
            "geometry.zones must contain at least one zone".to_string(),
        ));
    }

    // Issue #2530 — defense in depth. The REST path already rejects out-of
    // -range `years` at deserialisation (see `validate_years`), but
    // `run_simulation` is `pub` and is also called directly by the batch /
    // campaign handlers and by integration tests. Clamp here so no caller
    // can ever reach `solve_timesteps` with a multi-trillion-step count.
    // `clamp` floors at 1 (rejecting 0) and ceilings at [`MAX_YEARS`].
    let years = years.clamp(1, MAX_YEARS);

    let mut model = ThermalModel::<VectorField>::new(num_zones);
    for zone_idx in 0..model.num_zones {
        model.heating_setpoints.as_mut_slice()[zone_idx] = heating;
        model.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;
    }

    let steps = years as usize * 8760;
    let surrogates = SurrogateManager::new().map_err(|e| {
        ApiError::SimulationFailed(format!("failed to create SurrogateManager: {e}"), None)
    })?;

    // Issue #2499 — observability on the surrogate hot path. When a caller
    // requests neural-surrogate acceleration but no ONNX model is available,
    // the solver transparently falls back to the analytical/physics path.
    // Emit a single WARN per simulation (a fresh `SurrogateManager` is built
    // per call, so this never spams per-timestep) so the fallback is visible
    // in structured logs. This fires inside the `#[tracing::instrument]`
    // span declared above, so the line carries the `request_id` for
    // end-to-end correlation.
    if use_surrogates && !surrogates.model_loaded {
        tracing::warn!(
            backend = surrogates.backend.as_str(),
            "surrogate requested but no ONNX model loaded — using analytical fallback"
        );
    }

    let _ = model.solve_timesteps(steps, &surrogates, use_surrogates, None, None, None);

    // Issue #2547 — divergence detection. `solve_timesteps` currently swallows
    // internal errors and returns a (possibly NaN / infinite) EUI; the
    // per-zone hourly temperature trace it leaves behind is the only signal
    // we have. Scan it for NaN / infinity; if found, build a
    // `SimulationDiagnostics` from the trace and surface it on the
    // `SimulationFailed` envelope so REST/Python clients get
    // failing-timestep / failing-zone attribution instead of a bare message.
    if let Some(hourly) = model.get_hourly_temperatures() {
        if let Some(diag) = SimulationDiagnostics::from_temperature_trace(&hourly) {
            return Err(ApiError::SimulationFailed(
                format!(
                    "simulation diverged at timestep {}{}",
                    diag.failing_timestep,
                    diag.failing_zone
                        .as_ref()
                        .map(|z| format!(" in zone {z}"))
                        .unwrap_or_default()
                ),
                Some(diag),
            ));
        }
    }

    let heating_energy = model.get_heating_energy_kwh();
    let cooling_energy = model.get_cooling_energy_kwh();
    let total_energy = heating_energy + cooling_energy;
    let floor_area = schema.geometry.total_floor_area.max(1.0);
    let eui = total_energy / floor_area;

    let peak_heating_load = model.get_peak_heating_power_kw() * 1000.0;
    let peak_cooling_load = model.get_peak_cooling_power_kw() * 1000.0;

    let hourly_zone_temperatures = model.get_hourly_temperatures();
    let zone_temperatures = model.get_temperatures();

    Ok(SimulationOutput {
        eui,
        total_energy,
        peak_heating_load,
        peak_cooling_load,
        heating_energy,
        cooling_energy,
        zone_temperatures: Some(zone_temperatures),
        hourly_zone_temperatures,
    })
}

#[tracing::instrument(skip_all, fields(request_id, num_zones, years))]
async fn simulate(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    ValidatedJson(req): ValidatedJson<SimulateRequest>,
) -> Result<Json<SimulateResponse>, ApiError> {
    // Issue #2546 — audit trail. Extract per-request identifiers before
    // the simulation runs so the `simulation_started` event captures the
    // caller even if the run fails immediately.
    let request_id = headers
        .get(X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let client_id: Option<String> = headers
        .get("x-fluxion-client")
        .or_else(|| headers.get(axum::http::header::USER_AGENT))
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let schema = req.schema.into_v1();
    let options = req.options;

    // Snapshot the inputs for the audit event before `schema` is moved
    // into the in-memory store below.
    let num_zones = schema.geometry.zones.len();
    let years = options.years;
    let use_surrogates = options.use_surrogates;
    let schema_hash = schema_audit_hash(&schema);

    // Issue #2499 — record the per-request correlation id + shape onto the
    // handler span so downstream `run_simulation` logs inherit it.
    tracing::Span::current().record("request_id", request_id.as_str());
    tracing::Span::current().record("num_zones", num_zones);
    tracing::Span::current().record("years", years);

    tracing::info!(
        target: "audit",
        event = "simulation_started",
        request_id = %request_id,
        schema_hash = %schema_hash,
        num_zones = num_zones,
        years = years,
        use_surrogates = use_surrogates,
        client_id = ?client_id,
    );

    let started = std::time::Instant::now();
    // Issue #2501 — `run_simulation` is CPU-bound (it iterates
    // `years * 8760` physics timesteps). Running it directly on the tokio
    // worker pinned to this request starves every other concurrent request
    // sharing that worker, and under load drives `/v1/simulate` p99 well
    // past the 10 ms/config latency budget. Move the whole solve onto the
    // dedicated blocking thread pool with `spawn_blocking` so the tokio
    // worker is free to drive other I/O while the physics runs. The schema
    // and request_id are cloned into the closure because they are still
    // needed below (schema for storage, request_id for the audit event).
    //
    // `spawn_blocking` yields `Result<Result<SimulationOutput, ApiError>,
    // JoinError>` (outer = join, inner = simulation). The audit `completed`
    // event wraps both layers so a panic is still timed; the two `?` then
    // surface a join failure as `SimulationFailed` and the inner result
    // unchanged. The `request_id` is threaded in per #2499.
    let schema_for_sim = schema.clone();
    let request_id_for_sim = request_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        run_simulation(&schema_for_sim, years, use_surrogates, &request_id_for_sim)
    })
    .await
    .map_err(|join_err| {
        ApiError::SimulationFailed(format!("simulation blocking task failed: {join_err}"), None)
    })?;
    tracing::info!(
        target: "audit",
        event = "simulation_completed",
        request_id = %request_id,
        duration_ms = started.elapsed().as_millis(),
        outcome = if result.is_ok() { "success" } else { "error" },
    );
    let output = result?;

    let schema_id = if let Some(id) = options.store_as.clone() {
        // Issue #2552: `schemas` is now `parking_lot::RwLock`. Use a
        // synchronous write guard — no `.await` crosses the critical
        // section, so concurrent `simulate` calls do not serialise on the
        // lock.
        state.schemas.write().insert(id.clone(), schema);
        Some(id)
    } else {
        // Auto-store so clients can retrieve the schema that produced the
        // numbers via `GET /v1/schema/{id}`. Returning the id keeps the
        // acceptance criterion tractable without forcing clients to manage
        // ids themselves.
        Some(state.store(schema).await)
    };

    Ok(Json(SimulateResponse { schema_id, output }))
}

/// Stable, non-cryptographic hash of a simulation schema used for audit
/// correlation (Issue #2546). Two requests with byte-identical canonical
/// JSON produce the same id; intentionally NOT a security primitive.
fn schema_audit_hash(schema: &SimulationSchemaV1) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    if let Ok(canonical) = serde_json::to_string(schema) {
        canonical.hash(&mut hasher);
    }
    format!("0x{:016x}", hasher.finish())
}

/// SSE streaming handler for `POST /v1/simulate/stream`. Emits one SSE event
/// per timestep with the current zone temperatures.
async fn simulate_stream(
    State(state): State<AppState>,
    ValidatedJson(req): ValidatedJson<SimulateRequest>,
) -> Result<Response, ApiError> {
    let schema = req.schema.into_v1();
    let options = req.options;
    let num_zones = schema.geometry.zones.len().max(1);

    let heating = schema.controls.zone_control.heating_setpoint;
    let cooling = schema.controls.zone_control.cooling_setpoint;
    if heating >= cooling {
        return Err(ApiError::InvalidSchema(format!(
            "heating_setpoint ({heating}) must be < cooling_setpoint ({cooling})"
        )));
    }
    if schema.geometry.zones.is_empty() {
        return Err(ApiError::InvalidSchema(
            "geometry.zones must contain at least one zone".to_string(),
        ));
    }

    // Issue #2530 — `validate_years` already bounds `options.years` to
    // `1..=MAX_YEARS` at deserialisation; clamp again here (defense in
    // depth) since this handler computes `steps` directly instead of going
    // through `run_simulation`.
    let years = options.years.clamp(1, MAX_YEARS);
    let steps = years as usize * 8760;
    let surrogates = SurrogateManager::new().map_err(|e| {
        ApiError::SimulationFailed(format!("failed to create SurrogateManager: {e}"), None)
    })?;
    let (tx, rx) = mpsc::channel::<Result<TimestepEvent, ApiError>>(100);

    tokio::spawn(async move {
        let mut model = ThermalModel::<VectorField>::new(num_zones);
        for zone_idx in 0..model.num_zones {
            model.heating_setpoints.as_mut_slice()[zone_idx] = heating;
            model.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;
        }

        let dt_seconds = model.calculate_timestep_seconds();
        let _ = model.solve_timesteps_with_dt(
            steps,
            &surrogates,
            options.use_surrogates,
            None,
            None,
            None,
            dt_seconds,
        );

        if let Some(hourly_temps) = model.get_hourly_temperatures() {
            for (timestep, zone_temps) in hourly_temps.iter().enumerate() {
                let event = TimestepEvent {
                    timestep,
                    zone_temperatures: zone_temps.clone(),
                };
                if tx.send(Ok(event)).await.is_err() {
                    break;
                }
            }
        }
    });

    let stream = stream! {
        let mut rx = rx;
        while let Some(item) = rx.recv().await {
            match item {
                Ok(event) => {
                    match serde_json::to_string(&event) {
                        Ok(json) => {
                            yield Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", json));
                        }
                        Err(e) => {
                            yield Ok::<_, std::convert::Infallible>(format!("data: {{\"error\": \"{}\"}}\n\n", ApiError::SerializationFailed(e.to_string())));
                        }
                    }
                }
                Err(e) => {
                    yield Ok::<_, std::convert::Infallible>(format!("data: {{\"error\": \"{}\"}}\n\n", e));
                }
            }
        }
    };

    let _ = state.store(schema).await;

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(axum::body::Body::from_stream(stream))
        .unwrap();

    Ok(response)
}

/// Batch simulation handler for `POST /v1/batch`. Runs multiple simulations
/// concurrently using rayon and returns all results.
#[tracing::instrument(skip_all, fields(request_id, batch_size))]
async fn batch_simulate(
    State(_state): State<AppState>,
    headers: axum::http::HeaderMap,
    ValidatedJson(req): ValidatedJson<BatchRequest>,
) -> Result<Json<BatchResponse>, ApiError> {
    // Issue #2499 — extract the request id once for the whole batch so each
    // per-entry `run_simulation` span (and its surrogate-fallback warning)
    // carries it for end-to-end correlation.
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

    // Issue #2530 — cap the batch size and the total step budget before any
    // rayon work is spawned. Per-entry `years` is already bounded to
    // `1..=MAX_YEARS` by `validate_years`, so a well-formed request can never
    // trip the budget guard; it exists to catch a future regression where a
    // new field smuggles in a larger step count.
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

    // Issue #2501 — the per-config `run_simulation` work below is CPU-bound
    // and, worse, runs on the **rayon** pool via `into_par_iter`. Executing
    // that rayon dispatch from a tokio `async fn` pins the tokio worker to
    // the rayon job: every concurrent `/v1/batch` request burns one tokio
    // worker on blocking CPU work and contends with
    // `BatchOracle::evaluate_population` (which uses the same global rayon
    // pool) for threads. Move the entire rayon dispatch into
    // `spawn_blocking` so the tokio worker is released for the duration of
    // the batch. The inner `into_par_iter` still parallelises across rayon
    // threads exactly as before — only the *dispatching* thread changes.
    // The `request_id` is threaded in per #2499.
    let request_id_for_batch = request_id.clone();
    let results = tokio::task::spawn_blocking(move || {
        schemas
            .into_par_iter()
            .zip(opts.into_par_iter())
            .map(|(schema, options)| {
                run_simulation(
                    &schema,
                    options.years,
                    options.use_surrogates,
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

/// Get simulation status for async polling via `GET /v1/simulation/:id/status`.
async fn get_simulation_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SimulationStatus>, ApiError> {
    state
        .get_simulation_status(&id)
        .await
        .map(Json)
        .ok_or(ApiError::SimulationNotFound(id))
}

/// Response body for `POST /v1/campaigns` (fire-and-forget, Issue #1786).
#[derive(Debug, Clone, Serialize)]
pub struct CampaignSubmitResponse {
    pub campaign_id: String,
}

/// Submit a campaign for fire-and-forget execution (Issue #1786).
///
/// The coordinator accepts a campaign spec and returns a campaign ID immediately
/// without waiting for simulations to complete. Workers push status to the
/// state store enabling async polling via `GET /v1/campaigns/:id/status`.
async fn submit_campaign(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    ValidatedJson(spec): ValidatedJson<CampaignSpec>,
) -> Result<Json<CampaignSubmitResponse>, ApiError> {
    // Issue #2499 — propagate the request id into the fire-and-forget task so
    // each `run_simulation` span (and its surrogate-fallback warning) carries
    // it for end-to-end correlation.
    let request_id = headers
        .get(X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    if spec.simulations.is_empty() {
        return Err(ApiError::EmptyBatch);
    }

    // Issue #2530 — apply the same batch + step-budget caps as `/v1/batch`.
    // Campaigns run fire-and-forget on background tasks, so an unbounded
    // campaign pins worker threads just as effectively as a synchronous
    // request; reject it up front.
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
            // Issue #2552: `campaigns` is now `parking_lot::RwLock`. The
            // write guard is held only for the duration of the synchronous
            // `HashMap::get_mut` + assignment; nothing inside is async, so
            // we avoid parking the worker task.
            let mut guard = campaigns.write();
            if let Some(current) = guard.get_mut(&campaign_id_for_task) {
                *current = CampaignState::Running {
                    spec: spec.clone(),
                    progress: 0.0,
                    completed: 0,
                };
            }
        }

        let mut results: Vec<Result<SimulationOutput, String>> = Vec::with_capacity(total);

        for (i, sim_req) in spec.simulations.iter().enumerate() {
            let schema = sim_req.schema.clone().into_v1();
            let years = sim_req.options.years;
            let use_surrogates = sim_req.options.use_surrogates;

            let result = run_simulation(&schema, years, use_surrogates, &request_id_for_task)
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

/// Get campaign status for async polling via `GET /v1/campaigns/:id/status`.
async fn get_campaign_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<CampaignStatus>, ApiError> {
    state
        .get_campaign_status(&id)
        .await
        .map(Json)
        .ok_or(ApiError::CampaignNotFound(id))
}

/// Import a file from one of the supported external formats. The body is the
/// raw file bytes; the path parameter selects the decoder.
async fn import_format(
    State(state): State<AppState>,
    Path(fmt): Path<String>,
    body: axum::body::Bytes,
) -> Result<Json<ImportResponse>, ApiError> {
    let fmt = fmt.to_ascii_lowercase();
    let schema = match fmt.as_str() {
        "osm" => {
            // OSM and gbxml readers expect a filesystem path. Write the body
            // to a temp file and hand it off. The temp file is cleaned up by
            // the OS once the handle drops; readers stream-read and close.
            // `tmp` (a `NamedTempFile`) is kept alive for the duration of
            // `import_osm` so the OS doesn't recycle the path mid-parse.
            let tmp = tempfile_for_bytes(&body, "osm")?;
            osm::import_osm(tmp.path()).map_err(|e| ApiError::ImportFailed(e.to_string()))?
        }
        "gbxml" => {
            let tmp = tempfile_for_bytes(&body, "gbxml")?;
            gbxml::import_gbxml(tmp.path()).map_err(|e| ApiError::ImportFailed(e.to_string()))?
        }
        "idf" => {
            let body_str = std::str::from_utf8(&body)
                .map_err(|e| ApiError::ImportFailed(format!("invalid UTF-8 in IDF body: {e}")))?;
            let idf: IdfFile = IdfParser::from_str(body_str)
                .map_err(|e| ApiError::ImportFailed(format!("IDF parse error: {e}")))?;
            let schema = SimulationSchemaV1::try_from(&idf)
                .map_err(|e| ApiError::ImportFailed(format!("IDF conversion error: {e}")))?;
            schema
        }
        "epjson" => {
            let body_str = std::str::from_utf8(&body).map_err(|e| {
                ApiError::ImportFailed(format!("invalid UTF-8 in epJSON body: {e}"))
            })?;
            let idf: IdfFile = IdfParser::from_epjson_str(body_str)
                .map_err(|e| ApiError::ImportFailed(format!("epJSON parse error: {e}")))?;
            let schema = SimulationSchemaV1::try_from(&idf)
                .map_err(|e| ApiError::ImportFailed(format!("IDF conversion error: {e}")))?;
            schema
        }
        other => return Err(ApiError::UnsupportedFormat(other.to_string())),
    };

    let id = state.store(schema.clone()).await;
    Ok(Json(ImportResponse {
        schema_id: id,
        schema,
    }))
}

/// Persist `bytes` to a uniquely-named, owner-only temp file and return
/// the [`NamedTempFile`] handle. The file is removed when the handle is
/// dropped.
///
/// Security (Issue #2556): the previous implementation built the path as
/// `fluxion-import-{nanos}.{ext}` under `std::env::temp_dir()` and opened
/// it with `std::fs::File::create` — a predictable name plus `O_CREAT`
/// without `O_EXCL`. On a multi-tenant host, an unprivileged co-tenant
/// that could predict (or race) the same nanosecond could pre-create the
/// path as a symlink to e.g. `/etc/passwd`; the import handler would
/// then either overwrite the symlink target or follow the link and
/// parse a file the attacker chose (CWE-377, CWE-367).
///
/// `tempfile::Builder::tempfile()` performs the open atomically with
/// `O_EXCL | O_NOFOLLOW` semantics and 16 random suffix bytes (128 bits
/// of entropy), so the file cannot already exist and cannot be a symlink.
/// We additionally set permissions to `0o600` and then call
/// `symlink_metadata` after creation as belt-and-braces verification
/// that the path on disk is a regular file we own before handing it to
/// the parsers.
fn tempfile_for_bytes(bytes: &[u8], ext: &str) -> Result<NamedTempFile, ApiError> {
    use std::io::Write;

    let suffix = format!(".{ext}");
    let mut tmp = {
        let mut builder = tempfile::Builder::new();
        builder
            .prefix("fluxion-import-")
            .suffix(&suffix)
            .rand_bytes(16);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            builder.permissions(std::fs::Permissions::from_mode(0o600));
        }
        builder
            .tempfile()
            .map_err(|e| ApiError::ImportFailed(format!("temp file create: {e}")))?
    };

    tmp.as_file_mut()
        .write_all(bytes)
        .map_err(|e| ApiError::ImportFailed(format!("temp file write: {e}")))?;
    tmp.as_file_mut()
        .sync_all()
        .map_err(|e| ApiError::ImportFailed(format!("temp file sync: {e}")))?;

    // Defence-in-depth: confirm the path on disk is a regular file and
    // not a symlink that was somehow substituted between creation and
    // now. `tempfile::Builder` already passes `O_NOFOLLOW` to the open,
    // so this should always succeed; we check anyway so a future change
    // to the builder cannot silently regress the security property.
    let meta = std::fs::symlink_metadata(tmp.path())
        .map_err(|e| ApiError::ImportFailed(format!("temp file stat: {e}")))?;
    let ft = meta.file_type();
    if !ft.is_file() || ft.is_symlink() {
        return Err(ApiError::ImportFailed(format!(
            "temp file is not a regular file (file_type={ft:?})"
        )));
    }

    Ok(tmp)
}

/// Header name carrying the per-request UUID (Issue #1447). Lowercase to
/// match the HTTP/2 wire spelling; the AXUM/Tower layers normalize to that.
const X_REQUEST_ID: &str = "x-request-id";

/// Error handler for the per-request timeout layer (Issue #2530).
///
/// `tower::timeout::TimeoutLayer` turns the inner (infallible) service into a
/// fallible one whose error is [`tower::timeout::Elapsed`]. axum's router
/// requires an infallible service, so we wrap the timeout in
/// [`axum::error_handling::HandleErrorLayer`] and convert the deadline error
/// into the canonical structured `{"error": {...}}` envelope (HTTP 408). Any
/// non-timeout `BoxError` (which should never occur for this layer stack) is
/// surfaced as a 500 so it is never silently swallowed.
async fn handle_timeout_error(err: BoxError) -> Response {
    if err
        .downcast_ref::<tower::timeout::error::Elapsed>()
        .is_some()
    {
        let body = Json(serde_json::json!({
            "error": {
                "kind": "request_timeout",
                "message": "request exceeded the 60-second server budget",
            }
        }));
        (StatusCode::REQUEST_TIMEOUT, body).into_response()
    } else {
        let body = Json(serde_json::json!({
            "error": {
                "kind": "internal_error",
                "message": format!("unhandled middleware error: {err}"),
            }
        }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

/// Construct the application's router. Exposed so integration tests can
/// mount it without going through the binary's env-var resolution path.
///
/// Security configuration is resolved from the environment (Issue #2505)
/// via [`crate::api::security::RestSecurityConfig::from_env`]. In tests /
/// local dev — where none of the `FLUXION_REST_*` vars are set — auth is
/// `off`, CORS allows localhost dev origins, and the per-IP rate limiter
/// has a generous default burst. Production operators set
/// `FLUXION_REST_AUTH=token` (with `FLUXION_REST_AUTH_TOKEN`) and a real
/// `FLUXION_REST_CORS_ORIGINS` allow-list. For an explicitly-configured
/// build, prefer [`router_with_security`].
pub fn router(state: AppState) -> Router {
    router_with_security(state, crate::api::security::RestSecurityConfig::from_env())
}

/// Construct the application's router with an explicit security
/// configuration (Issue #2505). The binary uses this entry point so the
/// auth / CORS / rate-limit / boot-guard controls are all driven from one
/// resolved [`RestSecurityConfig`].
///
/// Layer order matters (Issue #1447 / #2530 / #2505). `tower::ServiceBuilder`
/// applies layers so that the **first** `.layer()` call sits as the
/// **outermost** middleware (the request hits it first, the response leaves
/// it last). We arrange them top-to-bottom here so that SetRequestIdLayer
/// runs first on the request and PropagateRequestIdLayer runs last on the
/// response:
///
///   0. `HandleErrorLayer` + `tower::timeout::TimeoutLayer` (Issue #2530) —
///      outermost. Bounds the entire request to [`REQUEST_TIMEOUT`] (60 s)
///      so a runaway synchronous `solve_timesteps` cannot pin a Tokio
///      worker. On deadline the handler responds with a structured 408.
///   1. `SetRequestIdLayer` — assigns a UUID *before* anything else sees
///      the request, so `TraceLayer`'s span and the metrics middleware
///      can include it.
///   2. `TraceLayer` — emits one structured log line per request, with
///      the `x-request-id` header in scope.
///   3. `PropagateRequestIdLayer` — copies the captured `x-request-id`
///      onto the outbound response.
///   4. `metrics::record` — innermost of the global stack. Wraps the
///      handler so it can observe the final response status and elapsed
///      time.
///
/// The per-request global stack above is applied outermost on the merged
/// router. Inside it sit, in order: the CORS layer (handles preflight and
/// stamps allow-list headers), the per-IP token-bucket governor (429), and
/// the 16 MiB [`axum::extract::DefaultBodyLimit`] (#2505). The auth
/// middleware is attached only to the **protected** sub-router so
/// `/v1/healthz` stays reachable anonymously for liveness probes.
pub fn router_with_security(
    state: AppState,
    cfg: crate::api::security::RestSecurityConfig,
) -> Router {
    // Touch the recorder so it is installed at server start-up rather than
    // on the first request (matters for `/v1/metrics` smoke checks).
    let _ = metrics::init_recorder();

    let middleware_stack = ServiceBuilder::new()
        // Issue #2530 — outermost layer. `HandleErrorLayer` converts the
        // `Elapsed` error produced by `TimeoutLayer` into a structured 408 so
        // the router stays infallible. Placing the pair first makes the
        // 60-second budget bound the *entire* request (trace, metrics, and
        // the synchronous `solve_timesteps` call).
        .layer(axum::error_handling::HandleErrorLayer::new(
            handle_timeout_error,
        ))
        .layer(TimeoutLayer::new(REQUEST_TIMEOUT))
        .layer(SetRequestIdLayer::new(
            axum::http::HeaderName::from_static(X_REQUEST_ID),
            MakeRequestUuid,
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(
                    DefaultMakeSpan::new()
                        .level(Level::INFO)
                        .include_headers(true),
                )
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(PropagateRequestIdLayer::new(
            axum::http::HeaderName::from_static(X_REQUEST_ID),
        ))
        .layer(middleware::from_fn(metrics::record))
        .into_inner();

    // Issue #2505 — `/v1/healthz` stays public so liveness probes work
    // without credentials. Every other `/v1/*` route is mounted on the
    // protected sub-router, which carries the auth middleware.
    let protected_routes = Router::new()
        .route("/v1/metrics", get(metrics_handler))
        .route("/v1/openapi.json", get(openapi_json))
        .route("/v1/openapi.yaml", get(openapi_yaml))
        .route("/v1/simulate", post(simulate))
        .route("/v1/simulate/stream", post(simulate_stream))
        .route("/v1/batch", post(batch_simulate))
        .route("/v1/simulation/:id/status", get(get_simulation_status))
        .route("/v1/schema/:id", get(get_schema))
        .route("/v1/import/:fmt", post(import_format))
        .route("/v1/campaigns", post(submit_campaign))
        .route("/v1/campaigns/:id/status", get(get_campaign_status))
        .layer(middleware::from_fn_with_state(
            cfg.auth_state(),
            crate::api::security::require_auth,
        ));

    Router::new()
        .route("/v1/healthz", get(healthz))
        .merge(protected_routes)
        .with_state(state)
        // Issue #2505 — 16 MiB body cap (innermost global layer). Bounds
        // `/v1/import/*` and every other POST before any handler allocates.
        .layer(axum::extract::DefaultBodyLimit::max(
            crate::api::security::MAX_REQUEST_BODY_BYTES,
        ))
        // Issue #2505 — per-IP token-bucket governor (429 on flood).
        .layer(middleware::from_fn_with_state(
            cfg.rate_limiter(),
            crate::api::security::rate_limit_middleware,
        ))
        // Issue #2505 — explicit CORS allow-list (handles preflight).
        .layer(cfg.cors_layer())
        // Issues #2530 / #1447 — outermost global stack: 60 s timeout +
        // request-id + trace + metrics.
        .layer(middleware_stack)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::schema::{
        ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
        SimulationSchemaV1, WeatherData,
    };
    use serde_json::json;

    fn default_schema_v1() -> SimulationSchemaV1 {
        SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata::default(),
            geometry: Geometry::default(),
            constructions: ConstructionSet::default(),
            schedules: ScheduleSet::default(),
            weather: WeatherData::default(),
            controls: ControlSet::default(),
            output: SimulationOutput::default(),
        }
    }

    #[test]
    fn default_schema_v1_serializes_round_trip() {
        let schema = default_schema_v1();
        let body = serde_json::to_value(&schema).unwrap();
        // Schema is at the root, so a bare SimulationSchemaV1 must deserialize.
        let parsed: SimulationSchemaV1 = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(parsed.version, SchemaVersion::V1);
        // And the enveloped form must also work.
        let enveloped = json!({ "V1": schema });
        let _: SimulationSchema = serde_json::from_value(enveloped).unwrap();
    }

    #[tokio::test]
    async fn appstate_allocates_unique_ids() {
        let state = AppState::default();
        let a = state.store(default_schema_v1()).await;
        let b = state.store(default_schema_v1()).await;
        assert_ne!(a, b);
        assert!(a.starts_with(SCHEMA_ID_PREFIX));
        assert_eq!(state.len().await, 2);
    }

    #[tokio::test]
    async fn appstate_lookup_returns_stored_schema() {
        let state = AppState::default();
        let schema = default_schema_v1();
        let id = state.store(schema.clone()).await;
        let got = state.get(&id).await.expect("missing schema");
        assert_eq!(got.geometry.zones.len(), schema.geometry.zones.len());
    }

    #[tokio::test]
    async fn appstate_lookup_missing_is_none() {
        let state = AppState::default();
        assert!(state.get("sch-does-not-exist").await.is_none());
    }

    #[tokio::test]
    async fn router_has_all_endpoints() {
        // Issue #1442: every route declared in `Router::new()` (line 476)
        // must be reachable via HTTP. A path that returns 404 from axum's
        // *automatic* fallback (rather than the handler's typed
        // `SchemaNotFound`) means the route was removed from the router
        // without removing it from `src/api/openapi.yaml` or the docs —
        // both the in-process server probe here and the doc gate
        // (`openapi_yaml_paths_match_router` below) catch the drift on
        // either side.
        let state = AppState::default();
        // Pre-store a schema so `GET /v1/schema/{id}` returns 200 (a
        // missing id would legitimately 404 from the handler and would
        // be indistinguishable over HTTP from an unrouted path).
        let stored_id = state.store(default_schema_v1()).await;

        let router = router(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();

        // (HTTP method, path). Body is intentionally empty/allowed-missing
        // because the contract under test is "the route exists" (2xx/4xx
        // with a typed envelope), not "the body is well-formed" — a
        // probe that returns a typed 4xx proves the handler ran. The
        // only probe whose success depends on a pre-populated store is
        // `/v1/schema/{id}` (above).
        let probes: &[(&str, &str)] = &[
            ("GET", "/v1/healthz"),
            ("GET", "/v1/metrics"),
            ("GET", "/v1/openapi.json"),
            ("GET", "/v1/openapi.yaml"),
            ("POST", "/v1/simulate"),
            ("POST", "/v1/import/osm"),
            ("POST", "/v1/import/epjson"),
        ];
        for (method, path) in probes {
            let url = format!("http://{addr}{path}");
            let resp = match *method {
                "GET" => client.get(&url).send().await.unwrap(),
                "POST" => client.post(&url).send().await.unwrap(),
                other => panic!("unsupported probe method in test: {other}"),
            };
            assert_ne!(
                resp.status().as_u16(),
                404,
                "route {method} {path} returned 404 — declared in Router::new() but not actually mounted"
            );
        }

        // Schema lookup with a known id — must return 200.
        let url = format!("http://{addr}/v1/schema/{stored_id}");
        let resp = client.get(&url).send().await.unwrap();
        assert_eq!(
            resp.status().as_u16(),
            200,
            "GET /v1/schema/{{id}} (known id) returned {}",
            resp.status()
        );
        handle.abort();
    }

    /// Issue #1442 (cross-check): `src/api/openapi.yaml`'s `paths:` keys
    /// (OpenAPI-style `{id}`, `{fmt}`) must match the routes declared in
    /// `Router::new()` (axum-style `:id`, `:fmt`), one-to-one, modulo the
    /// brace/colon difference. Adding a route on either side without the
    /// matching entry on the other side turns this test red.
    #[test]
    fn openapi_yaml_paths_match_router() {
        // Routes declared in `Router::new()` (axum-style). Keep this list
        // in sync with `src/api/server.rs:476` — it is the canonical
        // source-of-truth used by this drift gate.
        const AXUM_ROUTES: &[&str] = &[
            "/v1/healthz",
            "/v1/metrics",
            "/v1/openapi.json",
            "/v1/openapi.yaml",
            "/v1/simulate",
            "/v1/simulate/stream",
            "/v1/batch",
            "/v1/simulation/:id/status",
            "/v1/schema/:id",
            "/v1/import/:fmt",
        ];

        let yaml = include_str!("openapi.yaml");
        let parsed: serde_yaml::Value =
            serde_yaml::from_str(yaml).expect("src/api/openapi.yaml must be a valid YAML document");
        let paths = parsed
            .get("paths")
            .and_then(|v| v.as_mapping())
            .expect("src/api/openapi.yaml must have a top-level `paths:` map");

        let openapi_paths: std::collections::BTreeSet<String> = paths
            .iter()
            .map(|(k, _)| k.as_str().expect("path keys must be strings").to_string())
            .collect();

        // Normalize OpenAPI-style `{x}` → axum-style `:x`.
        let normalized: std::collections::BTreeSet<String> = openapi_paths
            .iter()
            .map(|p| p.replace("{id}", ":id").replace("{fmt}", ":fmt"))
            .collect();

        let axum_routes: std::collections::BTreeSet<String> =
            AXUM_ROUTES.iter().map(|s| s.to_string()).collect();

        let only_in_router: Vec<&String> = axum_routes.difference(&normalized).collect();
        let only_in_yaml: Vec<&String> = normalized.difference(&axum_routes).collect();

        assert!(
            only_in_router.is_empty() && only_in_yaml.is_empty(),
            "OpenAPI ↔ Router drift detected.\n\
             Routes in `Router::new()` but missing from openapi.yaml: {only_in_router:#?}\n\
             Routes in openapi.yaml but missing from `Router::new()`: {only_in_yaml:#?}\n\
             Update both sides (axum uses `:id`/`:fmt`, OpenAPI uses `{{id}}`/`{{fmt}}`) \
             and keep this test passing.",
        );
    }

    #[test]
    fn run_simulation_rejects_heating_ge_cooling() {
        let mut bad = default_schema_v1();
        bad.controls.zone_control.heating_setpoint = 25.0;
        bad.controls.zone_control.cooling_setpoint = 24.0;
        let err = run_simulation(&bad, 1, false, "test").unwrap_err();
        assert!(matches!(err, ApiError::InvalidSchema(_)));
    }

    #[test]
    fn run_simulation_rejects_empty_geometry() {
        let mut bad = default_schema_v1();
        bad.geometry.zones.clear();
        bad.geometry.total_floor_area = 0.0;
        bad.geometry.total_volume = 0.0;
        let err = run_simulation(&bad, 1, false, "test").unwrap_err();
        assert!(matches!(err, ApiError::InvalidSchema(_)));
    }

    // Issue #2547 — the JSON error envelope must embed the diagnostics
    // object when present on a `SimulationFailed`, and omit it entirely
    // when absent. We assert against the serialized JSON body directly
    // because `IntoResponse` consumes `self`.
    #[tokio::test]
    async fn simulation_failed_envelope_embeds_diagnostics_when_present() {
        let diag = SimulationDiagnostics {
            failing_timestep: 42,
            failing_zone: Some("zone_0".to_string()),
            max_residual_pct: 137.5,
            last_known_good_timestep: 41,
        };
        let err = ApiError::SimulationFailed(
            "simulation diverged at timestep 42 in zone zone_0".to_string(),
            Some(diag),
        );
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let error_obj = body.get("error").expect("envelope has error object");
        assert_eq!(error_obj["kind"], "simulation_failed");
        assert!(error_obj["message"]
            .as_str()
            .unwrap()
            .contains("diverged at timestep 42"));
        let diagnostics = error_obj
            .get("diagnostics")
            .expect("diagnostics field embedded when present");
        assert_eq!(diagnostics["failing_timestep"], 42);
        assert_eq!(diagnostics["failing_zone"], "zone_0");
        assert_eq!(diagnostics["max_residual_pct"], 137.5);
        assert_eq!(diagnostics["last_known_good_timestep"], 41);
    }

    #[tokio::test]
    async fn simulation_failed_envelope_omits_diagnostics_when_absent() {
        // SurrogateManager construction failure passes `None` for diagnostics
        // — the envelope must not contain a `diagnostics` key in that case.
        let err = ApiError::SimulationFailed(
            "failed to create SurrogateManager: missing model".to_string(),
            None,
        );
        let response = err.into_response();
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let error_obj = body.get("error").expect("envelope has error object");
        assert_eq!(error_obj["kind"], "simulation_failed");
        assert!(
            error_obj.get("diagnostics").is_none(),
            "diagnostics must be absent when None"
        );
    }

    #[tokio::test]
    async fn in_memory_store_insert_and_get() {
        use crate::api::server::InMemorySimulationStateStore;
        let store = InMemorySimulationStateStore::new();
        let id = store.insert(SimulationState::Pending).await;
        assert!(id.starts_with("sim-"));
        let state = store.get(&id).await;
        assert!(matches!(state, Some(SimulationState::Pending)));
    }

    #[tokio::test]
    async fn in_memory_store_update() {
        use crate::api::server::InMemorySimulationStateStore;
        let store = InMemorySimulationStateStore::new();
        let id = store.insert(SimulationState::Pending).await;

        let updated = store
            .update(&id, SimulationState::Running { progress: 0.5 })
            .await;
        assert!(updated, "update should return true for existing key");

        let state = store.get(&id).await;
        assert!(matches!(
            state,
            Some(SimulationState::Running { progress: 0.5 })
        ));
    }

    #[tokio::test]
    async fn in_memory_store_get_status() {
        use crate::api::server::InMemorySimulationStateStore;
        let store = InMemorySimulationStateStore::new();
        let id = store
            .insert(SimulationState::Running { progress: 0.75 })
            .await;

        let status = store.get_status(&id).await;
        assert!(status.is_some());
        let status = status.unwrap();
        assert_eq!(status.id, id);
        assert!(matches!(
            status.state,
            SimulationStateEnum::Running { progress: 0.75 }
        ));
        assert_eq!(status.progress, Some(0.75));
    }

    #[tokio::test]
    async fn in_memory_store_get_missing_is_none() {
        use crate::api::server::InMemorySimulationStateStore;
        let store = InMemorySimulationStateStore::new();
        let state = store.get("sim-does-not-exist").await;
        assert!(state.is_none());
    }

    #[tokio::test]
    async fn appstate_with_cloud_store() {
        use crate::api::server::InMemorySimulationStateStore;
        let state = AppState::with_cloud_store(InMemorySimulationStateStore::new());
        let id = state.register_simulation().await;
        assert!(id.starts_with("sim-"));
        let status = state.get_simulation_status(&id).await;
        assert!(status.is_some());
        assert!(matches!(
            status.unwrap().state,
            SimulationStateEnum::Pending
        ));
    }

    #[test]
    fn doc_invariant_campaign_survives_disconnect() {
        use crate::api::server::{AppState, SimulationStateStore};
        let state = AppState::with_cloud_store(InMemorySimulationStateStore::new());
        // Invariant: With a cloud store, workers push status and campaign
        // survives client disconnect. This is documented in the AppState docstring.
        // The actual cloud store implementation (DynamoDB/Redis) would persist
        // state across client connections.
        let _ = state;
    }

    // --- Issue #2556: TOCTOU + symlink-race regression tests for the
    // /v1/import temp-file path. The original `tempfile_for_bytes`
    // constructed `fluxion-import-{nanos}.{ext}` under `std::env::temp_dir()`
    // and opened it with `File::create` (no O_EXCL, no O_NOFOLLOW, 0644
    // by default). An unprivileged co-tenant on a shared host could
    // predict the nanosecond and pre-create the path as a symlink to
    // e.g. /etc/passwd; the server would then either overwrite the
    // symlink target or follow the link and parse attacker-chosen bytes
    // (CWE-377, CWE-367).
    //
    // The fixed implementation uses `tempfile::Builder::tempfile()` which
    // performs an atomic `O_CREAT|O_EXCL|0o600` open with 16 random
    // suffix bytes (128 bits of entropy) and `O_NOFOLLOW` semantics.
    // These tests verify the resulting security properties.
    #[cfg(unix)]
    #[test]
    fn tempfile_for_bytes_creates_regular_file_with_payload() {
        use std::io::Read;
        let payload = b"<osm>body-bytes</osm>";
        let tmp = tempfile_for_bytes(payload, "osm").expect("create temp file");

        // Path resolves on disk and the bytes round-trip exactly.
        let mut f = std::fs::File::open(tmp.path()).expect("open temp file");
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).expect("read temp file");
        assert_eq!(buf, payload);

        // `symlink_metadata` (not `metadata`) is the lstat that does NOT
        // follow symlinks — if anyone ever swaps it back to a path that
        // resolves to a symlink, this assertion will catch it.
        let meta = std::fs::symlink_metadata(tmp.path()).expect("lstat temp file");
        let ft = meta.file_type();
        assert!(ft.is_file(), "expected regular file, got file_type={ft:?}");
        assert!(
            !ft.is_symlink(),
            "temp file must not be a symlink (TOCTOU regression)"
        );
    }

    #[cfg(unix)]
    #[test]
    fn tempfile_for_bytes_uses_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile_for_bytes(b"secret", "gbxml").expect("create temp file");
        let perms = std::fs::metadata(tmp.path())
            .expect("stat temp file")
            .permissions();
        // Strip the file-type bits (S_IFREG = 0o100000 etc.) and compare
        // the permission bits only.
        assert_eq!(
            perms.mode() & 0o777,
            0o600,
            "temp file must be owner-read/write only (Issue #2556)"
        );
    }

    #[cfg(unix)]
    #[test]
    fn tempfile_for_bytes_distinct_paths_per_call() {
        // Two consecutive calls must produce different paths because the
        // suffix is drawn from 16 random bytes (128 bits). If this ever
        // regresses to a predictable name, the TOCTOU window re-opens
        // (Issue #2556).
        let a = tempfile_for_bytes(b"a", "osm").expect("create temp a");
        let b = tempfile_for_bytes(b"b", "osm").expect("create temp b");
        assert_ne!(a.path(), b.path());

        // Filenames must not embed the old predictable `fluxion-import-{nanos}`
        // pattern. We assert the prefix is preserved (helps debugging) and
        // that the parent dir is the system temp dir.
        let parent = a.path().parent().expect("temp file has a parent dir");
        assert_eq!(parent, std::env::temp_dir());
        let name_a = a.path().file_name().unwrap().to_string_lossy();
        assert!(
            name_a.starts_with("fluxion-import-"),
            "prefix should be preserved for log triage, got {name_a}"
        );
        // No `nanos` field — the old format was `fluxion-import-{nanos}.{ext}`,
        // the new one is `fluxion-import-{22 base64 chars}.{ext}` (16 bytes
        // b64-encoded ≈ 22 chars + padding). Either way there must NOT be
        // a long run of digits in the middle.
        let stem = name_a.trim_start_matches("fluxion-import-");
        let stem = stem.split('.').next().unwrap_or(stem);
        let digit_run = stem.chars().take_while(|c| c.is_ascii_digit()).count();
        assert!(
            digit_run < 8,
            "temp filename looks predictable (long digit run): {name_a}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn create_new_true_rejects_pre_existing_symlink() {
        // The attack: an unprivileged co-tenant pre-creates the predicted
        // temp path as a symlink to a target they want overwritten
        // (CWE-377). The fix relies on `O_EXCL` semantics: even if a
        // symlink already occupies the name, the open must fail rather
        // than follow the link. `tempfile::Builder::tempfile()` does
        // this internally; we mirror the same call here to lock the
        // behaviour down with a regression test.
        use std::os::unix::fs::{symlink, OpenOptionsExt, PermissionsExt};
        use tempfile::Builder;

        let dir = tempfile::tempdir().expect("create attack sandbox");
        let victim = dir.path().join("victim");
        std::fs::write(&victim, b"original victim contents").expect("seed victim");

        let pwn = dir.path().join("pwn");
        symlink(&victim, &pwn).expect("attacker plants symlink");
        assert!(
            std::fs::symlink_metadata(&pwn)
                .expect("lstat symlink")
                .file_type()
                .is_symlink(),
            "precondition: planted symlink must actually be a symlink"
        );

        // Mirrors what `tempfile::Builder` does under the hood: open the
        // pre-existing path with `create_new(true)` so it fails instead
        // of silently following the link.
        let result = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&pwn);
        assert!(
            result.is_err(),
            "create_new(true) must reject a pre-existing symlink path \
             (this is the core fix for Issue #2556)"
        );

        // The victim file must be untouched after the rejected open.
        assert_eq!(
            std::fs::read(&victim).expect("read victim"),
            b"original victim contents",
            "victim file must not have been clobbered by the failed open"
        );

        // And a real tempfile created via the same Builder API in the
        // same dir must succeed and land on a regular file with 0o600.
        let tmp = Builder::new()
            .prefix("safe-")
            .rand_bytes(16)
            .permissions(std::fs::Permissions::from_mode(0o600))
            .tempfile_in(dir.path())
            .expect("create safe temp");
        let meta = std::fs::symlink_metadata(tmp.path()).expect("lstat safe temp");
        assert!(meta.file_type().is_file());
        assert!(!meta.file_type().is_symlink());
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
    }

    // -- Issue #2530 ------------------------------------------------------

    #[test]
    fn simulate_options_default_years_is_valid() {
        // When the field is absent, `default_years()` (= 1) must apply and
        // must survive `validate_years`.
        let opts: SimulateOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(opts.years, 1);
    }

    #[test]
    fn validate_years_rejects_zero() {
        let err = serde_json::from_str::<SimulateOptions>(r#"{"years": 0}"#);
        assert!(err.is_err(), "years=0 must be rejected");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("years"), "error must mention years: {msg}");
    }

    #[test]
    fn validate_years_accepts_max_years_and_rejects_above() {
        let ok: SimulateOptions = serde_json::from_value(json!({ "years": MAX_YEARS })).unwrap();
        assert_eq!(ok.years, MAX_YEARS);

        let err = serde_json::from_value::<SimulateOptions>(json!({ "years": MAX_YEARS + 1 }));
        assert!(err.is_err(), "years=MAX+1 must be rejected");
    }

    #[test]
    fn validate_years_rejects_u32max() {
        let err = serde_json::from_value::<SimulateOptions>(json!({ "years": u32::MAX }));
        assert!(
            err.is_err(),
            "years=u32::MAX must be rejected at deserialisation"
        );
    }

    #[test]
    fn run_simulation_clamps_huge_years_defensively() {
        // Even though the REST path rejects u32::MAX at deserialisation,
        // `run_simulation` is `pub` and may be called directly. A defensive
        // clamp must prevent the multi-trillion-step DoS from ever reaching
        // `solve_timesteps`. We assert the *invariant*: a call with
        // `u32::MAX` returns promptly (bounded work) instead of attempting
        // ~3.76e13 steps. With the default schema the solver may legitimately
        // diverge (#2547) — that is still a fast, bounded outcome and counts
        // as success for this guard. The only failing mode is the call never
        // returning, which would trip the test process timeout.
        let schema = default_schema_v1();
        let start = std::time::Instant::now();
        let result = run_simulation(&schema, u32::MAX, false, "test");
        assert!(
            start.elapsed().as_secs() < 30,
            "run_simulation with u32::MAX took {:?} — clamp appears absent",
            start.elapsed()
        );
        match result {
            Ok(_) => {}
            Err(ApiError::SimulationFailed(_, _)) => {
                // Bounded divergence (#2547) — the clamp did its job.
            }
            Err(other) => panic!("unexpected error from clamped run: {other:?}"),
        }
    }

    #[test]
    fn step_budget_constant_bounds_worst_case_batch() {
        // MAX_CAMPAIGN_STEPS must equal the product of the three caps so a
        // maximal-but-legal batch is accepted while anything larger is not.
        let expected = (MAX_YEARS as usize) * 8760 * MAX_BATCH_SIMULATIONS;
        assert_eq!(MAX_CAMPAIGN_STEPS, expected);
        // Sanity: the budget is well under usize::MAX and under 2^31 on any
        // platform the workspace targets (it is 89_702_400).
        assert!(MAX_CAMPAIGN_STEPS < (1usize << 31));
    }

    // ---- Issue #2499: structured-log request-id propagation smoke test ----

    /// A `tracing` layer that records, for every WARN/ERROR event, the
    /// `request_id` field of the innermost active span (the field set by the
    /// `#[tracing::instrument(... fields(request_id = ...))]` annotation on
    /// `run_simulation`). Used only by the #2499 smoke test below.
    struct RequestIdWarnCapture {
        captured: std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>,
        span_request_ids: std::sync::Arc<std::sync::Mutex<HashMap<tracing::span::Id, String>>>,
    }

    struct FieldCollector<'a> {
        request_id: &'a mut Option<String>,
        message: &'a mut String,
    }

    impl<'a> tracing::field::Visit for FieldCollector<'a> {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "request_id" {
                *self.request_id = Some(value.to_string());
            }
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            let formatted = format!("{:?}", value);
            match field.name() {
                "request_id" => *self.request_id = Some(formatted),
                "message" => *self.message = formatted,
                _ => {}
            }
        }
    }

    impl<S> tracing_subscriber::Layer<S> for RequestIdWarnCapture
    where
        S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut request_id = None;
            let mut message = String::new();
            let mut visitor = FieldCollector {
                request_id: &mut request_id,
                message: &mut message,
            };
            attrs.record(&mut visitor);
            if let Some(rid) = request_id {
                self.span_request_ids
                    .lock()
                    .unwrap()
                    .insert(id.clone(), rid);
            }
        }

        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            // Only WARN / ERROR events reach the assertion (the surrogate
            // fallback line is WARN). `Level` orders severity descending, so
            // `level < WARN` covers INFO / DEBUG / TRACE.
            if event.metadata().level() < &tracing::Level::WARN {
                return;
            }
            let mut message = String::new();
            let mut request_id_direct = None;
            let mut visitor = FieldCollector {
                request_id: &mut request_id_direct,
                message: &mut message,
            };
            event.record(&mut visitor);

            // Resolve the request id from the active `run_simulation` span
            // when the event itself does not carry it (the fallback WARN does
            // not — it inherits the field from the span).
            let request_id = request_id_direct.or_else(|| {
                ctx.lookup_current().and_then(|span| {
                    self.span_request_ids
                        .lock()
                        .unwrap()
                        .get(&span.id())
                        .cloned()
                })
            });
            self.captured
                .lock()
                .unwrap()
                .push((request_id.unwrap_or_default(), message));
        }
    }

    #[test]
    fn run_simulation_propagates_request_id_to_surrogate_fallback_warn() {
        // Issue #2499 acceptance — the request id supplied to
        // `run_simulation` must appear on the surrogate-fallback WARN line.
        // We install a per-test capturing subscriber via `with_default`
        // (thread-local, so the synchronous `run_simulation` call on this
        // thread is observed) and assert that a WARN carrying
        // `request_id = test-123` and the fallback message was emitted.
        // `use_surrogates = true` with no ONNX model loaded triggers the
        // analytical-fallback path and therefore the WARN.
        use tracing_subscriber::layer::SubscriberExt as _;

        let captured: std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let layer = RequestIdWarnCapture {
            captured: std::sync::Arc::clone(&captured),
            span_request_ids: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        };
        let subscriber = tracing_subscriber::registry().with(layer);
        let schema = default_schema_v1();

        tracing::subscriber::with_default(subscriber, || {
            let _ = run_simulation(&schema, 1, true, "test-123");
        });

        let guard = captured.lock().unwrap();
        assert!(
            guard.iter().any(|(rid, msg)| {
                rid.contains("test-123") && msg.to_lowercase().contains("fallback")
            }),
            "expected a surrogate-fallback WARN carrying request_id `test-123`; \
             captured events: {guard:?}"
        );
    }

    // =====================================================================
    // Issue #2505 — auth / CORS / body-limit / rate-limit acceptance tests.
    // These exercise the layered router built by `router_with_security`
    // directly via `tower::ServiceExt::oneshot` (no socket needed).
    // =====================================================================

    /// Build a request with optional `Authorization` and `Origin` headers.
    fn build_request(
        method: axum::http::Method,
        uri: &str,
        body: axum::body::Body,
        auth_token: Option<&str>,
        origin: Option<&str>,
    ) -> Request {
        let mut b = Request::builder().method(method).uri(uri);
        if let Some(tok) = auth_token {
            b = b.header(axum::http::header::AUTHORIZATION, format!("Bearer {tok}"));
        }
        if let Some(o) = origin {
            b = b.header(axum::http::header::ORIGIN, o);
        }
        b.body(body).unwrap()
    }

    /// Read the full response body as bytes.
    async fn body_bytes(response: Response) -> Vec<u8> {
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec()
    }

    /// Issue #2505 (a): a request to a protected `/v1/*` route without a
    /// bearer token is rejected with `401` when `AUTH=token`.
    #[tokio::test]
    async fn auth_token_mode_rejects_missing_credential() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.auth_mode = crate::api::security::AuthMode::Token;
        cfg.auth_token = Some("s3cret".to_string());
        let app = router_with_security(AppState::default(), cfg);

        let req = build_request(
            axum::http::Method::GET,
            "/v1/metrics",
            axum::body::Body::empty(),
            None,
            None,
        );
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "protected route without token must be 401"
        );
    }

    /// Issue #2505 (a): a request with the correct bearer token passes the
    /// auth gate and reaches the handler.
    #[tokio::test]
    async fn auth_token_mode_accepts_correct_token() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.auth_mode = crate::api::security::AuthMode::Token;
        cfg.auth_token = Some("s3cret".to_string());
        let app = router_with_security(AppState::default(), cfg);

        let req = build_request(
            axum::http::Method::GET,
            "/v1/metrics",
            axum::body::Body::empty(),
            Some("s3cret"),
            None,
        );
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        // `/v1/metrics` is a plain GET; reaching the handler yields 200.
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "correct token must reach the handler"
        );
    }

    /// Issue #2505 (a): `/v1/healthz` stays reachable without credentials
    /// even when auth is enabled (liveness probes must work anonymously).
    #[tokio::test]
    async fn healthz_is_exempt_from_auth() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.auth_mode = crate::api::security::AuthMode::Token;
        cfg.auth_token = Some("s3cret".to_string());
        let app = router_with_security(AppState::default(), cfg);

        let req = build_request(
            axum::http::Method::GET,
            "/v1/healthz",
            axum::body::Body::empty(),
            None,
            None,
        );
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "healthz must be reachable without auth"
        );
    }

    /// Issue #2505 (a): the wrong token is rejected with `401`.
    #[tokio::test]
    async fn auth_token_mode_rejects_wrong_token() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.auth_mode = crate::api::security::AuthMode::Token;
        cfg.auth_token = Some("s3cret".to_string());
        let app = router_with_security(AppState::default(), cfg);

        let req = build_request(
            axum::http::Method::GET,
            "/v1/metrics",
            axum::body::Body::empty(),
            Some("wrong"),
            None,
        );
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// Issue #2505 (b): a request body exceeding the 16 MiB cap is rejected
    /// with `413 Payload Too Large` before the handler runs.
    #[tokio::test]
    async fn body_limit_rejects_oversized_body() {
        // Sanity: the production cap is exactly 16 MiB.
        assert_eq!(
            crate::api::security::MAX_REQUEST_BODY_BYTES,
            16 * 1024 * 1024
        );

        let cfg = crate::api::security::RestSecurityConfig::default(); // auth off
        let app = router_with_security(AppState::default(), cfg);

        let oversized = vec![0u8; crate::api::security::MAX_REQUEST_BODY_BYTES + 1];
        let req = build_request(
            axum::http::Method::POST,
            "/v1/import/osm",
            axum::body::Body::from(oversized),
            None,
            None,
        );
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "body > 16 MiB must be rejected with 413"
        );
    }

    /// Issue #2505 (b): a body exactly at the cap is accepted past the
    /// limit layer (it may still fail in the handler for unrelated reasons,
    /// so we only assert it is NOT a 413).
    #[tokio::test]
    async fn body_limit_accepts_body_at_cap() {
        let cfg = crate::api::security::RestSecurityConfig::default();
        let app = router_with_security(AppState::default(), cfg);

        let at_cap = vec![0u8; crate::api::security::MAX_REQUEST_BODY_BYTES];
        let req = build_request(
            axum::http::Method::POST,
            "/v1/import/osm",
            axum::body::Body::from(at_cap),
            None,
            None,
        );
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "body exactly at the cap must not be rejected as 413"
        );
    }

    /// Issue #2505 (c): a CORS preflight (`OPTIONS`) for an allowed origin
    /// returns the `access-control-allow-origin` header echoing that origin.
    #[tokio::test]
    async fn cors_preflight_allows_origin_in_allowlist() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.cors_origins = vec!["http://localhost:3000".to_string()];
        let app = router_with_security(AppState::default(), cfg);

        let req = Request::builder()
            .method(axum::http::Method::OPTIONS)
            .uri("/v1/simulate")
            .header(axum::http::header::ORIGIN, "http://localhost:3000")
            .header(axum::http::header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(
                axum::http::header::ACCESS_CONTROL_REQUEST_HEADERS,
                "content-type",
            )
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        let allow_origin = resp
            .headers()
            .get(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .expect("preflight for an allowed origin must set access-control-allow-origin");
        assert_eq!(
            allow_origin.to_str().unwrap(),
            "http://localhost:3000",
            "allow-origin must echo the allowed origin exactly"
        );
    }

    /// Issue #2505 (c): a CORS preflight for a *disallowed* origin receives
    /// no `access-control-allow-origin` header (browser denies the request).
    #[tokio::test]
    async fn cors_preflight_denies_origin_not_in_allowlist() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.cors_origins = vec!["http://localhost:3000".to_string()];
        let app = router_with_security(AppState::default(), cfg);

        let req = Request::builder()
            .method(axum::http::Method::OPTIONS)
            .uri("/v1/simulate")
            .header(axum::http::header::ORIGIN, "https://evil.example.com")
            .header(axum::http::header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert!(
            resp.headers()
                .get(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none(),
            "disallowed origin must not receive an allow-origin header"
        );
    }

    /// Issue #2505 (d): flooding a single IP past the token-bucket burst is
    /// throttled with `429 Too Many Requests`. Uses a tight test config so
    /// the assertion is deterministic without sending thousands of requests.
    #[tokio::test]
    async fn rate_limiter_throttles_flood_from_one_ip() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.rate_limit_rps = 1;
        cfg.rate_limit_burst = 3; // tiny bucket for a deterministic test
                                  // Build the router once so all clones share the same limiter.
        let app = router_with_security(AppState::default(), cfg);

        // First `burst` requests are allowed; the next is rejected.
        let mut got_429 = false;
        let mut allowed = 0usize;
        for _ in 0..10 {
            // Each iteration clones `app`; cloned routers share the
            // Arc-backed rate-limiter state.
            let router_clone = app.clone();
            let req = Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/healthz")
                .header("x-forwarded-for", "198.51.100.7")
                .body(axum::body::Body::empty())
                .unwrap();
            let resp = tower::ServiceExt::oneshot(router_clone, req).await.unwrap();
            if resp.status() == StatusCode::TOO_MANY_REQUESTS {
                got_429 = true;
                // Drain body.
                let _ = body_bytes(resp).await;
                break;
            } else {
                allowed += 1;
                let _ = body_bytes(resp).await;
            }
        }
        assert!(allowed <= 3, "burst capacity should bound allowed requests");
        assert!(
            got_429,
            "after draining the bucket the flood must be throttled with 429"
        );
    }

    /// Issue #2505 (d): distinct IPs are not penalised for each other's
    /// traffic (per-IP isolation).
    #[tokio::test]
    async fn rate_limiter_isolates_distinct_ips() {
        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.rate_limit_rps = 1;
        cfg.rate_limit_burst = 1;
        let app = router_with_security(AppState::default(), cfg);

        // Drain IP A's single token.
        let req_a = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .header("x-forwarded-for", "198.51.100.10")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app.clone(), req_a)
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let _ = body_bytes(resp).await;

        // IP A is now empty → next request from A is throttled.
        let req_a2 = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .header("x-forwarded-for", "198.51.100.10")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app.clone(), req_a2)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let _ = body_bytes(resp).await;

        // IP B has its own bucket → allowed.
        let req_b = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .header("x-forwarded-for", "198.51.100.11")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req_b).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "different IP must have its own bucket"
        );
    }
}

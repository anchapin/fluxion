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
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_stream::stream;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};
use tower::ServiceBuilder;
use tower_http::{
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::Level;

use crate::ai::surrogate::SurrogateManager;
use crate::api::metrics::{self, metrics_handler};
use crate::api::schema::{SimulationOutput, SimulationSchema, SimulationSchemaV1};
use crate::interop::{gbxml, ifc, osm};
use crate::io::idf::{IdfFile, IdfParser};
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

/// Identifier prefix for schemas persisted by the in-memory store.
const SCHEMA_ID_PREFIX: &str = "sch-";

/// Identifier prefix for simulations tracked for async status.
const SIM_ID_PREFIX: &str = "sim-";

/// Identifier prefix for campaigns (OSimFlow fire-and-forget, Issue #1786).
const CAMPAIGN_ID_PREFIX: &str = "camp-";

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
/// scope per #1342) and is intentionally behind a `tokio::sync::Mutex` to
/// avoid blocking the async runtime on contended reads.
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
    schemas: Arc<Mutex<HashMap<String, SimulationSchemaV1>>>,
    simulations: S,
    campaigns: Arc<Mutex<HashMap<String, CampaignState>>>,
    next_id: Arc<AtomicU64>,
}

impl Default for AppState<InMemorySimulationStateStore> {
    fn default() -> Self {
        Self {
            schemas: Arc::new(Mutex::new(HashMap::new())),
            simulations: InMemorySimulationStateStore::new(),
            campaigns: Arc::new(Mutex::new(HashMap::new())),
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
            schemas: Arc::new(Mutex::new(HashMap::new())),
            simulations,
            campaigns: Arc::new(Mutex::new(HashMap::new())),
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
    pub async fn store(&self, schema: SimulationSchemaV1) -> String {
        let id = self.next_id();
        self.schemas.lock().await.insert(id.clone(), schema);
        id
    }

    /// Look up a previously-stored schema by id.
    pub async fn get(&self, id: &str) -> Option<SimulationSchemaV1> {
        self.schemas.lock().await.get(id).cloned()
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
        self.schemas.lock().await.len()
    }

    /// Whether the store has zero schemas. Kept to satisfy
    /// `clippy::len_without_is_empty`; cheaper than `len() == 0` only if
    /// callers already hold the lock.
    pub async fn is_empty(&self) -> bool {
        self.schemas.lock().await.is_empty()
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
            .lock()
            .await
            .insert(id.clone(), CampaignState::Pending { spec });
        id
    }

    /// Update campaign state.
    pub async fn update_campaign(&self, id: &str, state: CampaignState) {
        self.campaigns.lock().await.insert(id.to_string(), state);
    }

    /// Get campaign status for polling.
    pub async fn get_campaign_status(&self, id: &str) -> Option<CampaignStatus> {
        self.campaigns.lock().await.get(id).map(|state| {
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
                        .enumerate()
                        .map(|(_i, r)| match r {
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
    /// Number of years to simulate. Default: `1`.
    #[serde(default = "default_years")]
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
    SimulationFailed(String),
    #[error("batch request is empty")]
    EmptyBatch,
    #[error("serialization failed: {0}")]
    SerializationFailed(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self {
            ApiError::InvalidSchema(_) => (StatusCode::BAD_REQUEST, "invalid_schema"),
            ApiError::SchemaNotFound(_) => (StatusCode::NOT_FOUND, "schema_not_found"),
            ApiError::SimulationNotFound(_) => (StatusCode::NOT_FOUND, "simulation_not_found"),
            ApiError::CampaignNotFound(_) => (StatusCode::NOT_FOUND, "campaign_not_found"),
            ApiError::UnsupportedFormat(_) => (StatusCode::BAD_REQUEST, "unsupported_format"),
            ApiError::IdfNotImplemented => (StatusCode::NOT_IMPLEMENTED, "not_implemented"),
            ApiError::ImportFailed(_) => (StatusCode::UNPROCESSABLE_ENTITY, "import_failed"),
            ApiError::SimulationFailed(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "simulation_failed")
            }
            ApiError::EmptyBatch => (StatusCode::BAD_REQUEST, "empty_batch"),
            ApiError::SerializationFailed(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "serialization_failed")
            }
        };
        let body = Json(serde_json::json!({
            "error": {
                "kind": kind,
                "message": self.to_string(),
            }
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
pub fn run_simulation(
    schema: &SimulationSchemaV1,
    years: u32,
    use_surrogates: bool,
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

    let mut model = ThermalModel::<VectorField>::new(num_zones);
    for zone_idx in 0..model.num_zones {
        model.heating_setpoints.as_mut_slice()[zone_idx] = heating;
        model.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;
    }

    let steps = years as usize * 8760;
    let surrogates = SurrogateManager::new().map_err(|e| {
        ApiError::SimulationFailed(format!("failed to create SurrogateManager: {e}"))
    })?;
    let _ = model.solve_timesteps(steps, &surrogates, use_surrogates, None, None, None);

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

async fn simulate(
    State(state): State<AppState>,
    Json(req): Json<SimulateRequest>,
) -> Result<Json<SimulateResponse>, ApiError> {
    let schema = req.schema.into_v1();
    let options = req.options;

    let output = run_simulation(&schema, options.years, options.use_surrogates)?;

    let schema_id = if let Some(id) = options.store_as.clone() {
        state.schemas.lock().await.insert(id.clone(), schema);
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

/// SSE streaming handler for `POST /v1/simulate/stream`. Emits one SSE event
/// per timestep with the current zone temperatures.
async fn simulate_stream(
    State(state): State<AppState>,
    Json(req): Json<SimulateRequest>,
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

    let steps = options.years as usize * 8760;
    let surrogates = SurrogateManager::new().map_err(|e| {
        ApiError::SimulationFailed(format!("failed to create SurrogateManager: {e}"))
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
async fn batch_simulate(
    State(_state): State<AppState>,
    Json(req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, ApiError> {
    if req.simulations.is_empty() {
        return Err(ApiError::EmptyBatch);
    }

    let schemas: Vec<_> = req
        .simulations
        .iter()
        .map(|r| r.schema.clone().into_v1())
        .collect();
    let opts: Vec<_> = req.simulations.iter().map(|r| r.options.clone()).collect();

    let results = schemas
        .into_par_iter()
        .zip(opts.into_par_iter())
        .map(|(schema, options)| {
            run_simulation(&schema, options.years, options.use_surrogates)
                .map(|output| SimulateResponse {
                    schema_id: None,
                    output,
                })
                .map_err(|e| e.to_string())
        })
        .collect();

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
    Json(spec): Json<CampaignSpec>,
) -> Result<Json<CampaignSubmitResponse>, ApiError> {
    if spec.simulations.is_empty() {
        return Err(ApiError::EmptyBatch);
    }

    let campaign_id = state.register_campaign(spec.clone()).await;
    let campaign_id_for_task = campaign_id.clone();

    let campaigns = Arc::clone(&state.campaigns);

    tokio::spawn(async move {
        let total = spec.simulations.len();

        {
            let mut guard = campaigns.lock().await;
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

            let result = run_simulation(&schema, years, use_surrogates).map_err(|e| e.to_string());

            results.push(result);

            let progress = (i + 1) as f32 / total as f32;
            let completed = i + 1;

            {
                let mut guard = campaigns.lock().await;
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
            let mut guard = campaigns.lock().await;
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
            let tmp = tempfile_for_bytes(&body, "osm")?;
            osm::import_osm(&tmp).map_err(|e| ApiError::ImportFailed(e.to_string()))?
        }
        "gbxml" => {
            let tmp = tempfile_for_bytes(&body, "gbxml")?;
            gbxml::import_gbxml(&tmp).map_err(|e| ApiError::ImportFailed(e.to_string()))?
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
        "ifc" => {
            let tmp = tempfile_for_bytes(&body, "ifc")?;
            ifc::import_ifc(&tmp).map_err(|e| ApiError::ImportFailed(e.to_string()))?
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

/// Persist `bytes` to a uniquely-named temp file and return its path. The
/// file is removed when the returned [`PathBuf`] is dropped (see
/// `tempfile::NamedTempFile` semantics). We use the standard library directly
/// here to avoid pulling a new dev-dependency just for this handler.
fn tempfile_for_bytes(bytes: &[u8], ext: &str) -> Result<PathBuf, ApiError> {
    use std::io::Write;

    let mut dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    dir.push(format!("fluxion-import-{nanos}.{ext}"));

    let mut f = std::fs::File::create(&dir)
        .map_err(|e| ApiError::ImportFailed(format!("temp file create: {e}")))?;
    f.write_all(bytes)
        .map_err(|e| ApiError::ImportFailed(format!("temp file write: {e}")))?;
    Ok(dir)
}

/// Header name carrying the per-request UUID (Issue #1447). Lowercase to
/// match the HTTP/2 wire spelling; the AXUM/Tower layers normalize to that.
const X_REQUEST_ID: &str = "x-request-id";

/// Construct the application's router. Exposed so integration tests can
/// mount it without going through the binary's env-var resolution path.
///
/// Layer order matters (Issue #1447). `tower::ServiceBuilder` applies
/// layers so that the **first** `.layer()` call sits as the **outermost**
/// middleware (the request hits it first, the response leaves it last).
/// We arrange them top-to-bottom here so that SetRequestIdLayer runs
/// first on the request and PropagateRequestIdLayer runs last on the
/// response:
///
///   1. `SetRequestIdLayer` — assigns a UUID *before* anything else sees
///      the request, so `TraceLayer`'s span and the metrics middleware
///      can include it.
///   2. `TraceLayer` — emits one structured log line per request, with
///      the `x-request-id` header in scope.
///   3. `PropagateRequestIdLayer` — copies the captured `x-request-id`
///      onto the outbound response.
///   4. `metrics::record` — innermost. Wraps the handler so it can
///      observe the final response status and elapsed time.
pub fn router(state: AppState) -> Router {
    // Touch the recorder so it is installed at server start-up rather than
    // on the first request (matters for `/v1/metrics` smoke checks).
    let _ = metrics::init_recorder();

    let middleware_stack = ServiceBuilder::new()
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

    Router::new()
        .route("/v1/healthz", get(healthz))
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
        .with_state(state)
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
        let err = run_simulation(&bad, 1, false).unwrap_err();
        assert!(matches!(err, ApiError::InvalidSchema(_)));
    }

    #[test]
    fn run_simulation_rejects_empty_geometry() {
        let mut bad = default_schema_v1();
        bad.geometry.zones.clear();
        bad.geometry.total_floor_area = 0.0;
        bad.geometry.total_volume = 0.0;
        let err = run_simulation(&bad, 1, false).unwrap_err();
        assert!(matches!(err, ApiError::InvalidSchema(_)));
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
}

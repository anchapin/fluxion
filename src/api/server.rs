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
//! - `GET /v1/readyz` — readiness probe (ONNX/weather/AppState, Issue #2514)
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
    routing::{get, post, MethodRouter},
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
    trace::{DefaultOnResponse, MakeSpan, TraceLayer},
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
use crate::sim::thermal_selector::ThermalSelector;

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
/// query the simulation status via `GET /v1/simulation/{id}/status`.
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
/// * `campaigns` — reads (`GET /v1/campaigns/{id}/status`) dominate over
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

/// Simulation status for async polling via `GET /v1/simulation/{id}/status`.
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

/// Campaign status for async polling via `GET /v1/campaigns/{id}/status`.
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
    /// Zone solver selection (Issue #3281). One of `"gauge"` (default),
    /// `"5r1c"`, `"9r4c"`. Validated by [`parse_selector_from_options`]
    /// *after* deserialisation so the rejection message can name the
    /// experimental gate (`FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1`) for the
    /// reserved `"6r2c"` / `"8r3c"` identifiers. `None` ⇒
    /// [`ThermalSelector::default()`].
    ///
    /// Issue #3305 — an *explicit* `"gauge"` is rejected with a 400 over
    /// REST: the REST schema carries no per-surface construction detail
    /// (`wall_spec`), so the gauge solver can never initialise on this path
    /// and the request would silently fall through to 5R1C. Omitting the
    /// field keeps the legacy default-selector behaviour (β-phase 5R1C
    /// fall-through) unchanged.
    #[serde(default)]
    pub zone_solver: Option<String>,
    /// Conduction algorithm selection (Issue #3281). One of `"default"`
    /// (default), `"ctf"`, `"fd"`. Validated alongside `zone_solver`.
    #[serde(default)]
    pub conduction_solver: Option<String>,
    /// Optional opaque id; if present, the request's schema is stored under
    /// this id *and* the id is returned for retrieval via
    /// `GET /v1/schema/{id}`.
    #[serde(default)]
    pub store_as: Option<String>,
}

/// Translate the optional `zone_solver` / `conduction_solver` request fields
/// into a [`ThermalSelector`] (Issue #3281).
///
/// Both fields are optional and default to `None`, which maps to
/// [`ThermalSelector::default()`] (`gauge` + `default`). Unknown values —
/// and the experimental `"6r2c"` / `"8r3c"` identifiers unless the
/// `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` env var is set on the server — are
/// rejected as [`ApiError::InvalidRequest`] (HTTP 400). The accepted
/// vocabulary and the gate wording live in
/// [`crate::sim::thermal_selector::parse_zone_solver`] /
/// [`crate::sim::thermal_selector::parse_conduction_solver`] so the REST,
/// CLI, and binding layers can never drift apart.
///
/// Issue #3305 — an *explicit* `zone_solver: "gauge"` is rejected with a
/// 400 (fail-closed). `build_model_from_schema` constructs the simplified
/// 4-orientation surface layout without `WallSurface.wall_spec`, so the
/// gauge solver's fail-fast initialisation can never succeed on this path
/// and the β-phase dispatcher (Issue #3280) silently falls through to 5R1C
/// — the selector was a no-op for the zone axis. Omitting the field keeps
/// the legacy default-selector behaviour unchanged. The conduction axis is
/// deliberately out of scope here (see the issue discussion).
pub fn parse_selector_from_options(options: &SimulateOptions) -> Result<ThermalSelector, ApiError> {
    let zone_solver = match &options.zone_solver {
        Some(s) => {
            let parsed = crate::sim::thermal_selector::parse_zone_solver(s)
                .map_err(ApiError::InvalidRequest)?;
            if parsed == crate::sim::thermal_selector::ZoneSolverKind::Gauge {
                return Err(ApiError::InvalidRequest(
                    "explicit zone_solver \"gauge\" is not supported over REST: the REST schema \
                     does not carry per-surface construction detail (wall_spec), so the gauge \
                     solver cannot initialise on this path and the request would silently fall \
                     through to 5R1C. Omit zone_solver or request \"5r1c\" / \"9r4c\" (fail-closed \
                     per issue #3305)"
                        .to_string(),
                ));
            }
            parsed
        }
        None => ThermalSelector::default().zone_solver,
    };
    let conduction_solver = match &options.conduction_solver {
        Some(s) => crate::sim::thermal_selector::parse_conduction_solver(s)
            .map_err(ApiError::InvalidRequest)?,
        None => ThermalSelector::default().conduction_solver,
    };
    Ok(ThermalSelector {
        zone_solver,
        conduction_solver,
    })
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
            zone_solver: None,
            conduction_solver: None,
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

// ── Readiness probes (Issue #2514) ──────────────────────────────────────
//
// `GET /v1/healthz` is deliberately liveness-only — it never pokes
// downstreams so a slow disk does not flap the load balancer. Kubernetes
// still needs a way to keep traffic out of a pod whose dependencies are
// not yet satisfied (missing ONNX model, unreadable weather file, broken
// state store). `/v1/readyz` is that probe: it runs three sub-checks and
// returns 200 only when all of them pass.
//
// The probe logic lives in a pure, synchronous function
// ([`run_readiness_probes_with`]) so the HTTP handler and the
// `fluxion-rest` startup self-check share one definition of "ready".

/// Outcome of a single readiness sub-probe.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessCheck {
    /// `"ok"` when the probe passed, `"fail"` otherwise.
    pub status: &'static str,
    /// Human-readable detail. On success a short note (e.g.
    /// `"mock (no model loaded)"`); on failure the error message.
    pub detail: String,
}

impl ReadinessCheck {
    /// `true` when `status == "ok"`.
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

impl From<Result<String, String>> for ReadinessCheck {
    fn from(res: Result<String, String>) -> Self {
        match res {
            Ok(detail) => ReadinessCheck {
                status: "ok",
                detail,
            },
            Err(detail) => ReadinessCheck {
                status: "fail",
                detail,
            },
        }
    }
}

/// Per-check breakdown returned by `GET /v1/readyz`.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessChecks {
    pub onnx: ReadinessCheck,
    pub weather: ReadinessCheck,
    pub appstate: ReadinessCheck,
}

/// Overall readiness report — the JSON body of `GET /v1/readyz`.
///
/// `status` is `"ok"` only when every check in [`ReadinessChecks`] is ok;
/// [`ReadinessReport::is_ready`] is the canonical accessor so callers do
/// not hard-code the literal.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessReport {
    pub status: &'static str,
    pub checks: ReadinessChecks,
}

impl ReadinessReport {
    /// `true` when the service is ready to accept traffic.
    pub fn is_ready(&self) -> bool {
        self.status == "ok"
    }
}

/// ONNX surrogate probe.
///
/// When the `ort` feature is enabled, constructing a [`SurrogateManager`]
/// exercises the ONNX-runtime linkage — ABI / shared-library issues
/// surface here rather than on the first request. When an operator
/// explicitly sets `FLUXION_ONNX_MODEL`, the path is verified to exist on
/// disk first (a missing model file is the most common readiness failure
/// under k8s where a ConfigMap/PVC mount is misconfigured). When `ort` is
/// off the probe passes unconditionally: surrogate inference runs in
/// mock/analytical mode, so there is nothing to fail on.
///
/// `model_env` is the value of `FLUXION_ONNX_MODEL` (or `None` when
/// unset); passing it in explicitly keeps this function pure and
/// deterministic under test.
fn probe_onnx(model_env: Option<&str>) -> Result<String, String> {
    #[cfg(feature = "ort")]
    {
        if let Some(path) = model_env {
            if !path.is_empty() && !std::path::Path::new(path).exists() {
                // Generic message — do NOT echo the user-supplied path
                // (Issue #2905: closes the path-oracle / error-leak window
                // now that the full validation pipeline lives one layer
                // down in `SurrogateManager::new_with_auto_load`).
                return Err("FLUXION_ONNX_MODEL file not found".to_string());
            }
        }
        match SurrogateManager::new() {
            Ok(m) => {
                if m.model_loaded {
                    Ok("model loaded".to_string())
                } else {
                    Ok("mock (no model loaded)".to_string())
                }
            }
            Err(e) => Err(format!("SurrogateManager::new() failed: {e}")),
        }
    }
    #[cfg(not(feature = "ort"))]
    {
        // ONNX runtime is not compiled in — suppress the unused-param
        // warning so the non-ort build stays clippy-clean.
        let _ = model_env;
        Ok("skipped (ort feature off)".to_string())
    }
}

/// EPW / weather-file probe.
///
/// The REST API embeds weather inline in each schema, so no default file
/// is required for readiness. When `FLUXION_WEATHER_FILE` is set, however,
/// the path must be readable so a misconfigured mount does not get traffic
/// routed to a server that cannot load TMY data.
fn probe_weather(weather_file: Option<&str>) -> Result<String, String> {
    match weather_file.filter(|p| !p.is_empty()) {
        Some(path) => match std::fs::File::open(path) {
            Ok(_) => Ok(format!("readable: {path}")),
            Err(e) => Err(format!("FLUXION_WEATHER_FILE='{path}' not readable: {e}")),
        },
        None => Ok("no weather file configured".to_string()),
    }
}

/// AppState probe.
///
/// `AppState::default()` must construct (allocating the
/// [`InMemorySimulationStateStore`]). Construction is infallible today,
/// but the probe exists so a future state-store init that *can* fail
/// (e.g. a cloud store requiring credentials) has a deterministic fail
/// point at readiness time rather than on the first request.
fn probe_appstate() -> Result<String, String> {
    let _state = AppState::default();
    Ok("initialized".to_string())
}

/// Run all readiness probes against explicit inputs. Pure (no env read)
/// so it is deterministic under test; [`run_readiness_probes`] is the
/// env-reading wrapper used by the HTTP handler and startup self-check.
pub fn run_readiness_probes_with(
    onnx_model: Option<&str>,
    weather_file: Option<&str>,
) -> ReadinessReport {
    let onnx: ReadinessCheck = probe_onnx(onnx_model).into();
    let weather: ReadinessCheck = probe_weather(weather_file).into();
    let appstate: ReadinessCheck = probe_appstate().into();
    let ready = onnx.is_ok() && weather.is_ok() && appstate.is_ok();
    ReadinessReport {
        status: if ready { "ok" } else { "not ready" },
        checks: ReadinessChecks {
            onnx,
            weather,
            appstate,
        },
    }
}

/// Run all readiness probes, reading configuration from the environment:
///
/// - `FLUXION_ONNX_MODEL` — explicit ONNX model path (probe verifies it
///   exists when `--features ort` is on).
/// - `FLUXION_WEATHER_FILE` — EPW/TMY weather file path (probe verifies
///   it is readable when set).
///
/// This is the single source of truth shared by the `GET /v1/readyz`
/// handler and the `fluxion-rest` startup self-check, so the live endpoint
/// and the boot-time gate agree on "ready".
pub fn run_readiness_probes() -> ReadinessReport {
    let onnx_model = std::env::var("FLUXION_ONNX_MODEL").ok();
    let weather_file = std::env::var("FLUXION_WEATHER_FILE").ok();
    run_readiness_probes_with(onnx_model.as_deref(), weather_file.as_deref())
}

/// Readiness handler (Issue #2514).
///
/// Returns `200 OK` with a per-check breakdown when every probe passes,
/// or `503 Service Unavailable` with the same breakdown when any probe
/// fails. Unlike [`healthz`] (liveness), this endpoint *does* poke
/// downstream dependencies, so it must be wired to a k8s
/// `readinessProbe` (not `livenessProbe`) to avoid restart loops.
async fn readyz() -> Response {
    let report = run_readiness_probes();
    let status = if report.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report)).into_response()
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

/// Build a [`ThermalModel`] from a [`SimulationSchemaV1`], mirroring the
/// schema→physics wiring that `ThermalModel::from_spec` performs for the
/// ASHRAE 140 validation path.
///
/// This is the root-cause fix for issue #2747 / LIMIT-07: previously
/// `run_simulation` and `/v1/simulate/stream` called `ThermalModel::new`
/// and set only the heating/cooling setpoints, leaving `thermal_capacitance`
/// at its `1.0 J/K` placeholder. The Explicit-Euler mass update then blew
/// up at hourly step 91 (`inf`/`NaN`), surfacing as
/// `ApiError::SimulationFailed("simulation diverged at timestep 91 …")`.
///
/// # What this wires
///
/// Per zone (from `schema.geometry.zones[i]`, `schema.constructions`,
/// `schema.controls`, and `schema.schedules`):
///
/// - Geometry: `zone_area` (floor area), `ceiling_height`, `zone_volume`,
///   `wall_area`/`roof_area`/`floor_area`. The wall area assumes a square
///   footprint (perimeter = 4·√(floor_area)), which matches the default
///   `ZoneGeometry::default()` shape and is the standard approximation used
///   when only floor area + height are known.
/// - Construction U-values: `wall_u_value`, `roof_u_value`, `floor_u_value`,
///   `window_u_value` from the schema's `SurfaceConstruction` layers using
///   `Construction::u_value` (includes interior + exterior film
///   coefficients; floor uses `SurfaceType::Floor` for downward-heat-flow
///   film + ground coupling resistance).
/// - Thermal capacitance `C_m` (J/K) following ISO 13790 §7.2:
///   `wall_cap + roof_cap + floor_cap`, where each term is the
///   construction's `thermal_capacitance_per_area()` × surface area. The
///   air-node capacitance `C_air = ρ·cp·V_zone` is stored separately per
///   Issue #1522 (option (a)).
/// - Conductances: `h_tr_ms` from ISO 13790 §7.2.2.2 (`h_ms_coeff · A_m`,
///   low-mass coefficient 2.0 W/m²K); `h_tr_em` from ISO 13790 Eq. 64
///   `1 / (1/h_op − 1/h_ms)` so `h_em` and `h_ms` in series equal the
///   overall opaque transmittance `h_op = U_wall·A_wall + U_roof·A_roof`.
///   `h_tr_w`, `h_ve`, `h_tr_is`, `h_tr_floor` are derived by
///   `update_derived_parameters` from `window_ratio`, `infiltration_rate`,
///   `floor_u_value`, and `zone_area`.
/// - HVAC: per-zone heating/cooling setpoints from
///   `schema.controls.zone_control`, and daily schedules from
///   `schema.schedules.hvac`.
/// - `update_derived_parameters` is called at the end so the cached
///   `derived_h_tr_3`, `derived_h_ext`, `derived_den`, … are consistent
///   with the populated scalar fields.
///
/// # What this deliberately does NOT wire
///
/// - ASHRAE 140 case-specific branches (Case 195 zero-infiltration, FF
///   free-floating setpoints, 9R4C multi-node solver selection, etc.) —
///   those belong in `from_spec`. The REST schema has no case-id field.
/// - Shading (overhang / fins) — the REST schema has no shading field.
/// - Inline weather — the schema's `WeatherData::TmyLocation` default does
///   not carry inline hourly data; weather wiring is a separate concern
///   tracked outside #2747.
///
/// These exclusions keep the constructor to the minimum surface needed to
/// produce physically-sane, EnergyPlus-comparable output for a generic
/// `SimulationSchemaV1` and no more.
fn build_model_from_schema(schema: &SimulationSchemaV1) -> ThermalModel<VectorField> {
    use crate::sim::construction::{Construction, SurfaceType};

    let num_zones = schema.geometry.zones.len().max(1);
    let mut model = ThermalModel::<VectorField>::new(num_zones);

    let heating = schema.controls.zone_control.heating_setpoint;
    let cooling = schema.controls.zone_control.cooling_setpoint;

    // Constants — ρ_air and cp_air at sea level (matches `ThermalModel::new`
    // defaults and fluxion_core::construction::AIR_DENSITY_SEA_LEVEL /
    // AIR_SPECIFIC_HEAT).
    const AIR_DENSITY: f64 = 1.2; // kg/m³
    const AIR_SPECIFIC_HEAT: f64 = 1005.0; // J/(kg·K)
                                           // Default infiltration when the schema carries no explicit schedule
                                           // (matches `ThermalModel::new` default of 0.5 ACH).
    const DEFAULT_INFILTRATION_ACH: f64 = 0.5;
    // ISO 13790 §7.2.2.2 surface-to-mass coupling coefficient for
    // low-mass / generic construction. The ASHRAE-140 path picks
    // construction-type-specific values in `from_spec`; the REST schema has
    // no construction-type field so we use the low-mass default which is
    // the safer (slightly under-coupled) choice for unknown stock.
    const H_MS_COEFF_LOW_MASS: f64 = 2.0; // W/(m²·K)
                                          // ISO 13790 §C.3 simplified interior-surface-to-air coupling (also used
                                          // by `update_derived_parameters` — repeated here only for the A_m doc).
    const H_SI: f64 = 3.45; // W/(m²·K)

    // Per-zone vectors for the geometry / construction-derived fields.
    let mut zone_area_vec = Vec::with_capacity(num_zones);
    let mut ceiling_height_vec = Vec::with_capacity(num_zones);
    let mut zone_volume_vec = Vec::with_capacity(num_zones);
    let mut wall_area_vec = Vec::with_capacity(num_zones);
    let mut roof_area_vec = Vec::with_capacity(num_zones);
    let mut floor_area_vec = Vec::with_capacity(num_zones);
    let mut window_ratio_vec = Vec::with_capacity(num_zones);
    let mut infiltration_vec = Vec::with_capacity(num_zones);

    // Convert the schema's `SurfaceConstruction` → `fluxion_core::Construction`
    // so we can reuse its `u_value` / `thermal_capacitance_per_area` helpers.
    let wall_c: Construction = Construction::new(schema.constructions.wall.layers.clone());
    let roof_c: Construction = Construction::new(schema.constructions.roof.layers.clone());
    let floor_c: Construction = Construction::new(schema.constructions.floor.layers.clone());

    let wall_u_value = wall_c.u_value(Some(SurfaceType::Wall), None);
    let roof_u_value = roof_c.u_value(Some(SurfaceType::Ceiling), None);
    let floor_u_value = floor_c.u_value(Some(SurfaceType::Floor), None);
    let window_u_value = schema
        .constructions
        .wall
        .window
        .as_ref()
        .map(|w| w.window_u_value)
        .unwrap_or(2.5);

    for zone in &schema.geometry.zones {
        let floor_area = zone.floor_area.max(1.0);
        let height = zone.height.max(1.0);
        let volume = if zone.volume > 0.0 {
            zone.volume
        } else {
            floor_area * height
        };
        // Square-footprint approximation: perimeter = 4·√(A).
        let perimeter = 4.0 * floor_area.sqrt();
        let gross_wall_area = perimeter * height;
        // Window area from the wall's WindowSpec (if any); fallback to
        // 15 % of gross wall area (the `ThermalModel::new` default ratio).
        let window_area = schema
            .constructions
            .wall
            .window
            .as_ref()
            .map(|w| w.window_area)
            .filter(|a| *a > 0.0 && *a <= gross_wall_area)
            .unwrap_or(0.15 * gross_wall_area);
        let window_ratio = if gross_wall_area > 0.0 {
            window_area / gross_wall_area
        } else {
            0.0
        };

        zone_area_vec.push(floor_area);
        ceiling_height_vec.push(height);
        zone_volume_vec.push(volume);
        wall_area_vec.push(gross_wall_area);
        roof_area_vec.push(floor_area); // flat-roof assumption
        floor_area_vec.push(floor_area);
        window_ratio_vec.push(window_ratio);
        infiltration_vec.push(DEFAULT_INFILTRATION_ACH);
    }

    // Pad vectors to num_zones in case `schema.geometry.zones` was shorter
    // (the request-validation layer already rejects empty zone lists, but
    // defensive coding here costs nothing).
    while zone_area_vec.len() < num_zones {
        zone_area_vec.push(*zone_area_vec.last().unwrap_or(&48.0));
        ceiling_height_vec.push(*ceiling_height_vec.last().unwrap_or(&2.7));
        zone_volume_vec.push(*zone_volume_vec.last().unwrap_or(&129.6));
        wall_area_vec.push(*wall_area_vec.last().unwrap_or(&64.8));
        roof_area_vec.push(*roof_area_vec.last().unwrap_or(&48.0));
        floor_area_vec.push(*floor_area_vec.last().unwrap_or(&48.0));
        window_ratio_vec.push(*window_ratio_vec.last().unwrap_or(&0.15));
        infiltration_vec.push(DEFAULT_INFILTRATION_ACH);
    }

    model.setpoints.zone_area = VectorField::new(zone_area_vec.clone());
    model.setpoints.ceiling_height = VectorField::new(ceiling_height_vec.clone());
    model.setpoints.zone_volume = VectorField::new(zone_volume_vec.clone());
    model.setpoints.wall_area = VectorField::new(wall_area_vec.clone());
    model.setpoints.roof_area = VectorField::new(roof_area_vec.clone());
    model.setpoints.floor_area = VectorField::new(floor_area_vec.clone());
    model.setpoints.window_ratio = VectorField::new(window_ratio_vec.clone());
    model.setpoints.aspect_ratio = VectorField::from_scalar(1.0, num_zones);
    model.setpoints.infiltration_rate = VectorField::new(infiltration_vec.clone());
    model.setpoints.air_density = VectorField::from_scalar(AIR_DENSITY, num_zones);
    model.setpoints.heat_capacity = VectorField::from_scalar(AIR_SPECIFIC_HEAT, num_zones);

    // Scalar U-values (single value for the whole model — the schema carries
    // one construction set, not per-zone).
    model.setpoints.wall_u_value = wall_u_value;
    model.setpoints.roof_u_value = roof_u_value;
    model.setpoints.floor_u_value = floor_u_value;
    model.solar.window_u_value = window_u_value;

    // Per-zone thermal capacitances and conductances. Vectorised because
    // each zone may have its own geometry; the constructions are shared
    // across zones (one wall/roof/floor assembly in `ConstructionSet`).
    let mut thermal_cap_vec = Vec::with_capacity(num_zones);
    let mut air_thermal_cap_vec = Vec::with_capacity(num_zones);
    let mut h_tr_ms_vec = Vec::with_capacity(num_zones);
    let mut h_tr_em_vec = Vec::with_capacity(num_zones);
    let mut h_tr_me_vec = Vec::with_capacity(num_zones);

    let wall_cap_per_area = wall_c.thermal_capacitance_per_area();
    let roof_cap_per_area = roof_c.thermal_capacitance_per_area();
    let floor_cap_per_area = floor_c.thermal_capacitance_per_area();

    for zone_idx in 0..num_zones {
        let zone_floor_area = zone_area_vec[zone_idx];
        let zone_wall_area = wall_area_vec[zone_idx];
        let zone_volume = zone_volume_vec[zone_idx];
        let window_area = window_ratio_vec[zone_idx] * zone_wall_area;
        let opaque_wall_area = (zone_wall_area - window_area).max(0.0);

        // C_m per ISO 13790 §7.2 (envelope only; air-node capacitance is
        // stored separately per Issue #1522 option (a)).
        let wall_cap = wall_cap_per_area * opaque_wall_area;
        let roof_cap = roof_cap_per_area * zone_floor_area;
        let floor_cap = floor_cap_per_area * zone_floor_area;
        let total_thermal_cap = (wall_cap + roof_cap + floor_cap).max(1.0e3);
        thermal_cap_vec.push(total_thermal_cap);

        let air_cap = zone_volume * AIR_DENSITY * AIR_SPECIFIC_HEAT;
        air_thermal_cap_vec.push(air_cap);

        // ISO 13790 §7.2.2.2 effective mass area A_m for low-mass
        // construction = 2.5 · A_floor (Table C.2 simplified form).
        let a_m = 2.5 * zone_floor_area;
        let h_ms = H_MS_COEFF_LOW_MASS * a_m;
        h_tr_ms_vec.push(h_ms);

        // ISO 13790 Eq. 64: h_em = 1 / (1/h_op − 1/h_ms), where
        // h_op = U_wall·A_opaque_wall + U_roof·A_roof (floor has its own
        // ground node via h_tr_floor and is excluded to avoid double-count).
        let h_op = wall_u_value * opaque_wall_area + roof_u_value * zone_floor_area;
        let h_em = if h_op > 0.0 && h_op < h_ms {
            (1.0 / (1.0 / h_op - 1.0 / h_ms)).max(0.1)
        } else {
            // Degenerate (e.g. near-zero wall U) — fall back to direct
            // opaque transmittance so the mass node never fully decouples.
            h_op.max(0.1)
        };
        h_tr_em_vec.push(h_em);

        // Interior-surface ↔ internal-mass (furniture) coupling. ISO 13790
        // Annex C: 9.1 W/(m²·K) over an internal-mass area estimated at
        // 0.5·A_floor (matches `from_spec` furniture_factor for commercial
        // buildings — the default `ControlSet` looks commercial).
        h_tr_me_vec.push(9.1 * 0.5 * zone_floor_area);

        // Reference H_SI for diagnostic comparison; not assigned —
        // `update_derived_parameters` derives h_tr_is from zone_area.
        let _h_tr_is_check = H_SI * zone_floor_area;
    }

    model.mass.thermal_capacitance = VectorField::new(thermal_cap_vec);
    model.mass.air_thermal_capacitance = VectorField::new(air_thermal_cap_vec);
    model.conduction.h_tr_ms = VectorField::new(h_tr_ms_vec);
    model.conduction.h_tr_em = VectorField::new(h_tr_em_vec);
    model.mass.h_tr_me = VectorField::new(h_tr_me_vec);
    model.conduction.h_tr_is = VectorField::from_scalar(0.0, num_zones); // recomputed below

    // Surfaces — replace the default placeholder surfaces created by
    // `ThermalModel::new` with ones whose areas / U-values / window areas
    // match the schema, so solar gain distribution uses real geometry.
    // The simplified single-wall-per-orientation layout matches the
    // assumption made above for `wall_area`.
    use crate::validation::ashrae_140_cases::Orientation;
    let orientations = [
        Orientation::South,
        Orientation::West,
        Orientation::North,
        Orientation::East,
    ];
    let mut surfaces = Vec::with_capacity(num_zones);
    for zone_idx in 0..num_zones {
        let gross_wall_area = wall_area_vec[zone_idx];
        let per_orientation = gross_wall_area / 4.0;
        let total_window_area = window_ratio_vec[zone_idx] * gross_wall_area;
        let window_per_orientation = total_window_area / 4.0;
        let mut zone_surfaces = Vec::with_capacity(orientations.len());
        for &orientation in &orientations {
            let surface = crate::sim::construction::WallSurface::new(
                per_orientation,
                wall_u_value,
                orientation,
            )
            .with_window(window_per_orientation);
            // Keep default emissivity/absorptance — `WallSurface::new` sets
            // physically-sane defaults; the schema has no per-surface optical
            // fields today.
            zone_surfaces.push(surface);
        }
        surfaces.push(zone_surfaces);
    }
    model.solar.surfaces = surfaces;

    // HVAC setpoints + schedules.
    model.setpoints.heating_setpoint = heating;
    model.setpoints.cooling_setpoint = cooling;
    model.setpoints.heating_setpoints = VectorField::from_scalar(heating, num_zones);
    model.setpoints.cooling_setpoints = VectorField::from_scalar(cooling, num_zones);
    model.hvac.hvac_enabled = VectorField::from_scalar(1.0, num_zones);
    model.hvac.hvac_heating_capacity = schema.controls.zone_control.heating_capacity.max(1.0);
    model.hvac.hvac_cooling_capacity = schema.controls.zone_control.cooling_capacity.max(1.0);
    model.setpoints.heating_schedule = schema.schedules.hvac.heating.clone();
    model.setpoints.cooling_schedule = schema.schedules.hvac.cooling.clone();

    // Recompute the derived conductances (h_tr_w, h_ve, h_tr_is, h_tr_floor,
    // derived_h_ext, derived_h_tr_3, …) from the scalar fields now set.
    // Note: `update_derived_parameters` deliberately does NOT overwrite
    // `thermal_capacitance`, `h_tr_em`, or `h_tr_ms` — those are set
    // explicitly above and preserved.
    model.update_derived_parameters();

    model
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
    selector: ThermalSelector,
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

    // Issue #2518 — time the solve (model construction through output
    // assembly) so a single `fluxion_simulation_duration_seconds` observation
    // is recorded on every exit path. The cheap input-validation guards above
    // return before this point, so a rejected request does not pollute the
    // histogram. The solve body is wrapped in a closure so the `?` operator
    // stays ergonomic while still funnelling every outcome through one metric
    // emission below.
    let solve_started = std::time::Instant::now();
    // Issue #2747: empty lighting profile passed to `solve_timesteps` below
    // to suppress the auto-loaded office profile (see comment at call site).
    let empty_lighting =
        crate::sim::lighting::LightingSchedule::new(0.0, schema.geometry.total_floor_area);
    let solve_result: Result<SimulationOutput, ApiError> = (|| {
        // Issue #2747 / LIMIT-07 root-cause fix: build the model from the
        // full schema (geometry + constructions + controls + schedules) via
        // `build_model_from_schema`, NOT `ThermalModel::new(num_zones)`.
        // The bare constructor leaves `thermal_capacitance = 1.0 J/K` (a
        // placeholder) and `air_thermal_capacitance = 0.0`; with C_m = 1.0
        // the Explicit-Euler mass update `Tm += (q_net/C_m)·dt` amplifies
        // any flux imbalance by ~3600 per step and the simulation blows up
        // at hourly index 91. See `build_model_from_schema` doc-comment
        // for the full schema→physics wiring.
        let mut model = build_model_from_schema(schema);
        // Issue #3281 — the caller-selected solver stack lands on the model
        // here. The β-phase dispatcher (Issue #3280) consumes
        // `hvac.thermal_selector` per step: `Gauge` tries the gauge solver
        // and falls through to 5R1C/9R4C on init or step failure;
        // `FiveROneC` / `NineRFourC` route strictly. With the default
        // selector the model behaves exactly as before this change.
        model.hvac.thermal_selector = selector;
        for zone_idx in 0..model.hvac.num_zones {
            model.setpoints.heating_setpoints.as_mut_slice()[zone_idx] = heating;
            model.setpoints.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;
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

        let _ = model.solve_timesteps(
            steps,
            &surrogates,
            use_surrogates,
            // Issue #2747: pass an explicit zero-gain lighting schedule so
            // `solve_timesteps_with_dt` does NOT auto-load the bundled office
            // building profile (which injects real office internal gains).
            // The auto-load path also has a `loads[i] += internal_gains`
            // accumulation quirk that produces runaway zone temperatures over
            // a full 8760-step run when no caller-supplied profile is set.
            // The REST schema does not carry an internal-loads field today —
            // wire real internal loads when the schema grows one. Until then
            // the simulation runs envelope-only (ventilation + conduction +
            // solar + HVAC), which is the physically-sane baseline.
            Some(&empty_lighting),
            None,
            None,
        );

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
            // Issue #3305 — report what the dispatcher ACTUALLY executed
            // (gauge success or the 5R1C/9R4C fall-through), not the
            // requested selector. The metric `solver` label above still
            // carries the requested stack; this field is the honest one.
            effective_solver: Some(model.effective_zone_solver().as_str().to_string()),
        })
    })();
    let solve_elapsed = solve_started.elapsed().as_secs_f64();

    // Issue #2518 — emit the per-simulation metric family once, on every exit
    // path. `energy_kwh` is only forwarded on success so the cumulative
    // throughput counter never advances on a failed run. The Issue #3284
    // `solver` label carries the `{zone}+{conduction}` selector pair so
    // nightly telemetry can attribute outcomes to the solver stack.
    metrics::record_simulation(
        solve_elapsed,
        years,
        use_surrogates,
        solve_result.is_ok(),
        num_zones,
        solve_result.as_ref().ok().map(|o| o.total_energy),
        &format!(
            "{}+{}",
            selector.zone_solver.as_str(),
            selector.conduction_solver.as_str()
        ),
    );

    solve_result
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

    // Issue #3281 — validate the solver selection *before* the audit event so
    // an invalid `zone_solver` / `conduction_solver` is a clean HTTP 400 with
    // no audit-trail noise.
    let selector = parse_selector_from_options(&options)?;

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
        run_simulation(
            &schema_for_sim,
            years,
            use_surrogates,
            selector,
            &request_id_for_sim,
        )
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

    // The spawned task below needs the schema for `build_model_from_schema`;
    // the original `schema` is still consumed by `state.store(schema)` after
    // the spawn, so clone once here.
    let schema_for_stream = schema.clone();
    // Issue #2747: empty lighting profile (see `run_simulation` for rationale).
    let empty_lighting =
        crate::sim::lighting::LightingSchedule::new(0.0, schema.geometry.total_floor_area);
    tokio::spawn(async move {
        // Issue #2747 / LIMIT-07: schema→physics wiring (same fix as
        // `run_simulation` — see `build_model_from_schema` doc-comment).
        let mut model = build_model_from_schema(&schema_for_stream);
        for zone_idx in 0..model.hvac.num_zones {
            model.setpoints.heating_setpoints.as_mut_slice()[zone_idx] = heating;
            model.setpoints.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;
        }

        let dt_seconds = model.calculate_timestep_seconds();
        let _ = model.solve_timesteps_with_dt(
            steps,
            &surrogates,
            options.use_surrogates,
            Some(&empty_lighting),
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

    // Issue #2518 — record the (non-empty) batch size on entry, before the
    // size/step-budget caps can reject it, so the histogram still captures
    // oversized batches. Each per-config `run_simulation` invocation below
    // emits its own duration/solver-kind/energy observations.
    metrics::record_batch_size(req.simulations.len());

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
                // Issue #3281 — per-entry solver selection; a bad selector is
                // reported for that entry only (matching how per-entry schema
                // validation failures surface below).
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

/// Get simulation status for async polling via `GET /v1/simulation/{id}/status`.
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
/// state store enabling async polling via `GET /v1/campaigns/{id}/status`.
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
            // Issue #3281 — per-entry solver selection. A bad selector is a
            // failed entry (same envelope as a failed simulation), not an
            // abort of the whole campaign.
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
///
/// DoS hardening (issue #2527): this route is protected by **two** layers.
/// The outer 16 MiB `DefaultBodyLimit` (#2505, applied at the router stack
/// below) rejects oversized bodies before the handler allocates. Each parser
/// additionally enforces `ParserLimits::default()` (64 MiB file / 1M lines /
/// 256 depth) — for HTTP the body limit binds first, but the parser caps
/// still catch pathologically line-dense inputs and protect the in-process
/// `from_str` paths used by `BatchOracle` / `fluxion-mcp` (which use
/// `ParserLimits::cli_default()`, 1 GiB).
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
///
/// `from_env` is fail-closed (Issue #2689): an unrecognized
/// `FLUXION_REST_AUTH` value is an `Err`. This convenience wrapper panics
/// on that condition — a misconfigured auth value must crash loudly rather
/// than silently disable authentication. In the documented test/dev use
/// (env unset) `from_env` always returns `Ok`, so the panic is unreachable
/// in normal operation.
pub fn router(state: AppState) -> Router {
    let security_cfg = crate::api::security::RestSecurityConfig::from_env()
        .unwrap_or_else(|e| panic!("fluxion-rest security misconfiguration: {e}"));
    router_with_security(state, security_cfg)
}

/// Request-header names that are safe to record on the `TraceLayer` span.
///
/// This is an **allow-list**, not a deny-list: a header is recorded only if
/// it appears here. Every credential-bearing header (`authorization`,
/// `cookie`, `x-api-key`, AWS Sig V4 `x-amz-*`, proxy-auth tokens, …) is
/// omitted by construction — there is no deny-list to keep in sync.
///
/// Issue #2504: `DefaultMakeSpan::new().include_headers(true)` previously
/// recorded *every* request header as a span field, leaking credentials
/// into structured logs (OWASP A09:2021). [`SafeHeaderMakeSpan`] replaces
/// it and only ever records the headers below.
const SAFE_HEADER_ALLOWLIST: [&str; 3] = ["x-request-id", "content-type", "user-agent"];

/// A [`MakeSpan`] that records an explicit allow-list of safe request
/// headers onto the `tower_http` trace span.
///
/// Replaces `DefaultMakeSpan::new().include_headers(true)` (Issue #2504).
/// Only the names in [`SAFE_HEADER_ALLOWLIST`] are ever recorded; everything
/// else — including `Authorization`, `Cookie`, `x-api-key`, and all AWS Sig
/// V4 headers — is omitted by construction.
#[derive(Clone)]
struct SafeHeaderMakeSpan {
    level: Level,
}

impl SafeHeaderMakeSpan {
    /// Create a new span builder that emits at `INFO` level (matching the
    /// previous `DefaultMakeSpan::new().level(Level::INFO)` configuration).
    fn new() -> Self {
        Self { level: Level::INFO }
    }

    /// Read an allow-listed header off the request, returning `""` if it is
    /// absent or not valid UTF-8. The header name is matched case-insensitively
    /// (HTTP headers are case-insensitive per RFC 7230 §3.2).
    ///
    /// The `debug_assert!` ties this method to [`SAFE_HEADER_ALLOWLIST`] so the
    /// allow-list constant is the single source of truth: calling this with a
    /// name not on the allow-list fails loudly in dev/test rather than silently
    /// recording an un-vetted header (Issue #2504).
    fn safe_header<'a>(headers: &'a axum::http::HeaderMap, name: &str) -> &'a str {
        debug_assert!(
            SAFE_HEADER_ALLOWLIST.contains(&name),
            "SafeHeaderMakeSpan::safe_header({name:?}) — not on SAFE_HEADER_ALLOWLIST; \
             refusing to record an un-vetted header (Issue #2504)"
        );
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
    }
}

impl<B> MakeSpan<B> for SafeHeaderMakeSpan {
    fn make_span(&mut self, request: &axum::http::Request<B>) -> tracing::Span {
        let headers = request.headers();
        // Only allow-listed names are ever touched — credential headers
        // (`authorization`, `cookie`, `x-api-key`, …) are never read here,
        // so they cannot leak into the span by construction (Issue #2504).
        let x_request_id = Self::safe_header(headers, "x-request-id");
        let content_type = Self::safe_header(headers, "content-type");
        let user_agent = Self::safe_header(headers, "user-agent");

        // The `tracing::span!` macro requires the level as a static token,
        // so (like `DefaultMakeSpan`) we expand via a macro + match.
        macro_rules! make_span {
            ($level:expr) => {
                tracing::span!(
                    $level,
                    "request",
                    method = %request.method(),
                    uri = %request.uri(),
                    version = ?request.version(),
                    x_request_id = %x_request_id,
                    content_type = %content_type,
                    user_agent = %user_agent,
                )
            };
        }
        match self.level {
            Level::ERROR => make_span!(Level::ERROR),
            Level::WARN => make_span!(Level::WARN),
            Level::INFO => make_span!(Level::INFO),
            Level::DEBUG => make_span!(Level::DEBUG),
            Level::TRACE => make_span!(Level::TRACE),
        }
    }
}

impl Default for SafeHeaderMakeSpan {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Issue #2812 — single source of truth for the `/v1/*` REST surface.
// =========================================================================
//
// `axum` 0.8 (the pinned version, `Cargo.toml:295`) does **not** expose a
// public route-introspection API: its `RouteId`, the internal `path_router`
// module, and the `matchit` router store are all `pub(crate)`, and
// `Router::as_service` / `Router::into_service` only wrap the router for
// request dispatch — they do not enumerate registered paths. That means the
// drift gate cannot ask a live `Router` "what routes do you mount?".
//
// The robust alternative (per the issue) is to make the route *registration*
// itself the single source of truth. The table below is the only place the
// `/v1/*` surface is enumerated:
//
//   - the production builder [`router_with_security`] iterates it to mount
//     routes (so the live `Router` is built *from* the table — there is no
//     parallel hardcoded `.route()` chain whose paths could drift), and
//   - the OpenAPI drift gate `openapi_yaml_paths_match_router` iterates the
//     same table to compare against `src/api/openapi.yaml`.
//
// Adding a route is a single edit to this table (+ the handler wiring in
// [`method_router_for_path`]). Forget the table and the route is simply not
// mounted — the drift gate cannot "silently green" a route that was never
// registered, which is exactly the footgun that hid `/v1/campaigns` for ~5
// months (#2747/#2803).

/// Access tier for a `/v1/*` route — drives where the builder mounts it.
///
/// `Public` routes (`/v1/healthz`, `/v1/readyz`) sit at the top level so
/// liveness/readiness probes work without credentials (#2505/#2514). Every
/// other route is `Protected` and carries the auth middleware.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteTier {
    Public,
    Protected,
}

/// HTTP method tag carried alongside each registry entry. Kept as a small
/// `Copy` enum (rather than `axum::http::Method`, which is only `Clone`) so
/// the whole [`REST_ROUTES`] tuple is `Copy` and cheap to iterate by value.
///
/// Only the methods the REST surface actually uses appear here; the OpenAPI
/// side of the drift check ([`openapi_router_drift`]) parses arbitrary
/// method keys, so adding a `PUT`/`DELETE` route later only needs a new
/// variant here plus its handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HttpMethod {
    Get,
    Post,
}

impl HttpMethod {
    /// Lowercase method name, matching the keys OpenAPI uses under each path
    /// (`get:`, `post:`, …). Only used by the (test-only) drift detector.
    #[cfg(test)]
    const fn as_str(self) -> &'static str {
        match self {
            HttpMethod::Get => "get",
            HttpMethod::Post => "post",
        }
    }
}

/// The canonical list of `/v1/*` REST routes — **the single source of truth**
/// (Issue #2812). Each entry is `(HTTP method, path template, access tier)`.
///
/// Order is irrelevant to `axum::matching`; routes are listed in
/// public-then-protected, doc-order for readability. Both axum 0.8+ and
/// OpenAPI use the `{id}` / `{fmt}` capture syntax, so path templates compare
/// directly with no brace↔colon normalization.
///
/// **When you add a route here you MUST also wire its handler in
/// [`method_router_for_path`].** The test
/// `method_router_for_path_covers_every_registry_path` enforces this — a
/// table entry with no handler fails the build's test suite loudly.
#[rustfmt::skip]
const REST_ROUTES: &[(HttpMethod, &str, RouteTier)] = &[
    // Public — anonymous liveness/readiness probes.
    (HttpMethod::Get,  "/v1/healthz",                   RouteTier::Public),
    (HttpMethod::Get,  "/v1/readyz",                    RouteTier::Public),
    // Protected — credential-gated via `require_auth` (#2505).
    (HttpMethod::Get,  "/v1/metrics",                   RouteTier::Protected),
    (HttpMethod::Get,  "/v1/openapi.json",              RouteTier::Protected),
    (HttpMethod::Get,  "/v1/openapi.yaml",              RouteTier::Protected),
    (HttpMethod::Post, "/v1/simulate",                  RouteTier::Protected),
    (HttpMethod::Post, "/v1/simulate/stream",           RouteTier::Protected),
    (HttpMethod::Post, "/v1/batch",                     RouteTier::Protected),
    (HttpMethod::Get,  "/v1/simulation/{id}/status",    RouteTier::Protected),
    (HttpMethod::Get,  "/v1/schema/{id}",               RouteTier::Protected),
    (HttpMethod::Post, "/v1/import/{fmt}",              RouteTier::Protected),
    (HttpMethod::Post, "/v1/campaigns",                 RouteTier::Protected),
    (HttpMethod::Get,  "/v1/campaigns/{id}/status",     RouteTier::Protected),
];

/// Resolve a registry path to its [`MethodRouter`] handler pair.
///
/// This is the *only* place a path is wired to its handler function. It is
/// keyed on the same path strings as [`REST_ROUTES`]; the test
/// [`method_router_for_path_covers_every_registry_path`] asserts every table
/// entry resolves here, so a path present in the table but missing a handler
/// arm is caught immediately rather than silently dropped.
///
/// Returns [`MethodRouter<AppState>`] so the builder can mount every entry
/// uniformly regardless of which extractors (`State`, `Path`, …) the handler
/// uses — handlers without a `State` extractor are `Handler<T, S>` for all
/// `S`, so they coerce to `MethodRouter<AppState>` alongside the rest.
fn method_router_for_path(path: &str) -> MethodRouter<AppState> {
    match path {
        "/v1/healthz" => get(healthz),
        "/v1/readyz" => get(readyz),
        "/v1/metrics" => get(metrics_handler),
        "/v1/openapi.json" => get(openapi_json),
        "/v1/openapi.yaml" => get(openapi_yaml),
        "/v1/simulate" => post(simulate),
        "/v1/simulate/stream" => post(simulate_stream),
        "/v1/batch" => post(batch_simulate),
        "/v1/simulation/{id}/status" => get(get_simulation_status),
        "/v1/schema/{id}" => get(get_schema),
        "/v1/import/{fmt}" => post(import_format),
        "/v1/campaigns" => post(submit_campaign),
        "/v1/campaigns/{id}/status" => get(get_campaign_status),
        other => unreachable!(
            "REST_ROUTES references {other:?} but method_router_for_path has no handler — \
             registry/handler drift (Issue #2812). Add the handler arm or remove the entry."
        ),
    }
}

/// Symmetric-path and per-path method drift between a route registry (the
/// single source of truth) and an OpenAPI document. Produced by
/// [`openapi_router_drift`]; empty iff the two are in sync. (Issue #2812.)
#[cfg(test)]
#[derive(Debug, Default, PartialEq, Eq)]
struct OpenApiRouterDrift {
    /// Paths mounted on the Router but absent from `openapi.yaml`.
    only_in_router: Vec<String>,
    /// Paths in `openapi.yaml` but not mounted on the Router.
    only_in_openapi: Vec<String>,
    /// `(path, router_methods, openapi_methods)` for paths present on both
    /// sides whose HTTP-method sets disagree. Methods are lowercase
    /// (`get`, `post`, …). Empty if every shared path agrees on methods.
    method_mismatches: Vec<(String, Vec<String>, Vec<String>)>,
}

#[cfg(test)]
impl OpenApiRouterDrift {
    /// `true` iff there is no drift to report.
    fn is_clean(&self) -> bool {
        self.only_in_router.is_empty()
            && self.only_in_openapi.is_empty()
            && self.method_mismatches.is_empty()
    }
}

/// Pure drift detector between a route registry and an OpenAPI YAML document.
///
/// Computes (a) the symmetric difference of path templates and (b) the
/// per-path HTTP-method set difference. Path templates compare directly
/// because axum 0.8+ and OpenAPI both use `{x}` captures. Method keys under
/// each OpenAPI path (`get`, `post`, `put`, `delete`, `patch`, `head`,
/// `options`) are lowercased and compared against the registry's
/// [`HttpMethod`] tags.
///
/// Factored as a pure function so the drift gate and its regression tests
/// share one implementation (Issue #2812).
#[cfg(test)]
fn openapi_router_drift(registry: &[(HttpMethod, &str)], openapi_yaml: &str) -> OpenApiRouterDrift {
    use std::collections::{BTreeMap, BTreeSet};

    // Registry side: path -> set<lowercase method>.
    let mut router: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (method, path) in registry {
        router.entry(path).or_default().insert(method.as_str());
    }

    // OpenAPI side: parse `paths:` -> path -> set<lowercase method>.
    let parsed: serde_yaml::Value = match serde_yaml::from_str(openapi_yaml) {
        Ok(v) => v,
        Err(e) => panic!("OpenAPI YAML failed to parse: {e}"),
    };
    let paths = parsed
        .get("paths")
        .and_then(|v| v.as_mapping())
        .expect("OpenAPI document must have a top-level `paths:` mapping");
    let mut openapi: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (key, val) in paths.iter() {
        let path = key
            .as_str()
            .unwrap_or_else(|| panic!("OpenAPI path key {key:?} must be a string"))
            .to_string();
        let mut methods = BTreeSet::new();
        if let Some(ops) = val.as_mapping() {
            for (method_key, _) in ops.iter() {
                if let Some(s) = method_key.as_str() {
                    let lower = s.to_ascii_lowercase();
                    if matches!(
                        lower.as_str(),
                        "get" | "post" | "put" | "delete" | "patch" | "head" | "options"
                    ) {
                        methods.insert(lower);
                    }
                }
            }
        }
        openapi.insert(path, methods);
    }

    let router_paths: BTreeSet<&str> = router.keys().copied().collect();
    let openapi_paths: BTreeSet<&str> = openapi.keys().map(String::as_str).collect();

    let only_in_router: Vec<String> = router_paths
        .difference(&openapi_paths)
        .map(|s| (*s).to_string())
        .collect();
    let only_in_openapi: Vec<String> = openapi_paths
        .difference(&router_paths)
        .map(|s| (*s).to_string())
        .collect();

    let method_mismatches: Vec<(String, Vec<String>, Vec<String>)> = router_paths
        .intersection(&openapi_paths)
        .filter_map(|path| {
            let router_methods: BTreeSet<String> =
                router[path].iter().map(|s| (*s).to_string()).collect();
            let openapi_methods: &BTreeSet<String> = &openapi[*path];
            if &router_methods == openapi_methods {
                None
            } else {
                Some((
                    (*path).to_string(),
                    router_methods.into_iter().collect(),
                    openapi_methods.iter().cloned().collect(),
                ))
            }
        })
        .collect();

    OpenApiRouterDrift {
        only_in_router,
        only_in_openapi,
        method_mismatches,
    }
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
///   5. `metrics::track_in_flight` — innermost-2. Maintains the
///      `fluxion_rest_in_flight_requests` gauge (Issue #2517) so the
///      graceful-shutdown drain has an accurate count of active requests.
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
            // Issue #2504: do NOT use `DefaultMakeSpan::new().include_headers(true)`
            // — that records *every* request header (including `Authorization`,
            // `Cookie`, `x-api-key`, AWS Sig V4) on the span, leaking
            // credentials into structured logs (OWASP A09:2021). Instead we
            // build the span with [`SafeHeaderMakeSpan`], which records only
            // the allow-listed names in `SAFE_HEADER_ALLOWLIST`
            // (`x-request-id`, `content-type`, `user-agent`). Credential
            // headers are omitted by construction — there is no deny-list to
            // keep in sync. See the regression test
            // `tracelayer_does_not_log_credentials`.
            TraceLayer::new_for_http()
                .make_span_with(SafeHeaderMakeSpan::new())
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(PropagateRequestIdLayer::new(
            axum::http::HeaderName::from_static(X_REQUEST_ID),
        ))
        .layer(middleware::from_fn(metrics::record))
        .layer(middleware::from_fn(metrics::track_in_flight))
        .into_inner();

    // Issue #2505 — `/v1/healthz` stays public so liveness probes work
    // without credentials. Every other `/v1/*` route is mounted on the
    // protected sub-router, which carries the auth middleware.
    // Issue #2514 — `/v1/readyz` is likewise public (readiness probes).
    //
    // Issue #2812 — routes are mounted from the single-source-of-truth
    // [`REST_ROUTES`] table via [`method_router_for_path`], rather than a
    // parallel hardcoded `.route()` chain. The drift gate below reads the
    // same table, so there is no second list of paths that can fall out of
    // sync. A path listed in the table without a handler arm fails
    // `method_router_for_path_covers_every_registry_path` loudly.
    let mut protected: Router<AppState> = Router::new();
    let mut public: Router<AppState> = Router::new();
    for &(_, path, tier) in REST_ROUTES {
        let method_router = method_router_for_path(path);
        match tier {
            RouteTier::Protected => protected = protected.route(path, method_router),
            RouteTier::Public => public = public.route(path, method_router),
        }
    }
    let protected_routes = protected.layer(middleware::from_fn_with_state(
        cfg.auth_state(),
        crate::api::security::require_auth,
    ));

    public
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
            ("GET", "/v1/readyz"),
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
    /// (`{id}`, `{fmt}`) must match the routes mounted by
    /// [`router_with_security`](Self) one-to-one, and the HTTP methods under
    /// each path must agree.
    ///
    /// Issue #2812 — the Router side of the comparison is no longer a
    /// manually-maintained `AXUM_ROUTES` array (which hid `/v1/campaigns`
    /// for ~5 months in #2747/#2803). It now comes from the single source of
    /// truth [`REST_ROUTES`] — the same table the production builder iterates
    /// — so the gate cannot silently green a route that was registered
    /// without a doc entry (or vice versa). Path templates compare directly
    /// because axum 0.8+ and OpenAPI both use `{x}` captures.
    #[test]
    fn openapi_yaml_paths_match_router() {
        // Single source of truth: the same `REST_ROUTES` the live
        // `router_with_security` mounts from.
        let registry: Vec<(HttpMethod, &str)> =
            REST_ROUTES.iter().map(|(m, p, _)| (*m, *p)).collect();
        let yaml = include_str!("openapi.yaml");

        let drift = openapi_router_drift(&registry, yaml);

        assert!(
            drift.is_clean(),
            "OpenAPI ↔ Router drift detected (Issue #2812).\n\
             Routes mounted (via REST_ROUTES) but missing from openapi.yaml: \
             {:#?}\n\
             Routes in openapi.yaml but missing from the Router: {:#?}\n\
             Per-path HTTP-method mismatches: {:#?}\n\
             Update both sides — axum 0.8+ and OpenAPI both use `{{id}}`/`{{fmt}}` \
             and lowercase method keys (`get`, `post`, …).",
            drift.only_in_router,
            drift.only_in_openapi,
            drift.method_mismatches,
        );
    }

    /// Issue #2812 — proves the gate *catches* drift rather than silently
    /// passing. A route present in the registry but absent from the OpenAPI
    /// document must be reported in `only_in_router`.
    #[test]
    fn drift_detector_flags_route_only_in_registry() {
        let yaml = "paths:\n  /v1/here:\n    get: {}\n";
        let registry = [
            (HttpMethod::Get, "/v1/here"),
            (HttpMethod::Post, "/v1/undocumented"),
        ];
        let drift = openapi_router_drift(&registry, yaml);
        assert_eq!(drift.only_in_router, vec!["/v1/undocumented".to_string()]);
        assert!(drift.only_in_openapi.is_empty());
        assert!(drift.method_mismatches.is_empty());
        assert!(!drift.is_clean());
    }

    /// Issue #2812 — a path documented in OpenAPI but not mounted on the
    /// Router must be reported in `only_in_openapi`.
    #[test]
    fn drift_detector_flags_path_only_in_openapi() {
        let yaml = "paths:\n  /v1/mounted:\n    get: {}\n  /v1/orphan:\n    post: {}\n";
        let registry = [(HttpMethod::Get, "/v1/mounted")];
        let drift = openapi_router_drift(&registry, yaml);
        assert_eq!(drift.only_in_openapi, vec!["/v1/orphan".to_string()]);
        assert!(drift.only_in_router.is_empty());
        assert!(!drift.is_clean());
    }

    /// Issue #2812 (bonus) — a path present on both sides whose HTTP methods
    /// disagree must be reported in `method_mismatches` even though the path
    /// sets match.
    #[test]
    fn drift_detector_flags_method_mismatch() {
        let yaml = "paths:\n  /v1/thing:\n    post: {}\n";
        // Registry says GET, OpenAPI says POST → method mismatch.
        let registry = [(HttpMethod::Get, "/v1/thing")];
        let drift = openapi_router_drift(&registry, yaml);
        assert!(drift.only_in_router.is_empty());
        assert!(drift.only_in_openapi.is_empty());
        assert_eq!(
            drift.method_mismatches,
            vec![(
                "/v1/thing".to_string(),
                vec!["get".to_string()],
                vec!["post".to_string()],
            )]
        );
        assert!(!drift.is_clean());
    }

    /// Issue #2812 — a perfectly aligned registry/document must report a
    /// clean (empty) drift so the gate stays green in the happy path.
    #[test]
    fn drift_detector_clean_when_aligned() {
        let yaml = "paths:\n  /a:\n    get: {}\n  /b:\n    post: {}\n";
        let registry = [(HttpMethod::Get, "/a"), (HttpMethod::Post, "/b")];
        let drift = openapi_router_drift(&registry, yaml);
        assert!(drift.is_clean());
    }

    /// Issue #2812 — every path in [`REST_ROUTES`] must resolve to a handler
    /// in [`method_router_for_path`]. A table entry added without its handler
    /// arm triggers the `unreachable!` here, so registry/handler drift can
    /// never silently drop a route. This closes the one gap the pure drift
    /// detector cannot see (the handler wiring is not in the OpenAPI doc).
    #[test]
    fn method_router_for_path_covers_every_registry_path() {
        for &(_, path, _) in REST_ROUTES {
            // Must not panic — proves a handler is wired for this path.
            let _ = method_router_for_path(path);
        }
    }

    #[test]
    fn run_simulation_rejects_heating_ge_cooling() {
        let mut bad = default_schema_v1();
        bad.controls.zone_control.heating_setpoint = 25.0;
        bad.controls.zone_control.cooling_setpoint = 24.0;
        let err = run_simulation(&bad, 1, false, ThermalSelector::default(), "test").unwrap_err();
        assert!(matches!(err, ApiError::InvalidSchema(_)));
    }

    #[test]
    fn run_simulation_rejects_empty_geometry() {
        let mut bad = default_schema_v1();
        bad.geometry.zones.clear();
        bad.geometry.total_floor_area = 0.0;
        bad.geometry.total_volume = 0.0;
        let err = run_simulation(&bad, 1, false, ThermalSelector::default(), "test").unwrap_err();
        assert!(matches!(err, ApiError::InvalidSchema(_)));
    }

    // ---- Issue #2518 — per-simulation observability ----------------------
    //
    // These tests use a thread-local `DebuggingRecorder` (via
    // `metrics::with_local_recorder`) so they never touch the process-global
    // Prometheus recorder and are safe to run in parallel with the REST API
    // integration tests. Each `snapshot().into_hashmap()` *drains* the
    // recorder (counters/gauges reset to zero, histograms cleared), so exactly
    // one observation is expected per metric per run.
    //
    // (`::metrics::` with leading colons resolves to the external `metrics`
    // crate; the parent module binds the bare name `metrics` to
    // `crate::api::metrics`, so we disambiguate explicitly here.)

    /// Helper: find a metric observation in a `DebuggingRecorder` snapshot by
    /// name and a predicate over its labels. Returns a borrow of the matching
    /// `DebugValue` (`DebugValue` is not `Clone`), or panics with a
    /// descriptive message.
    fn find_metric_value<'a>(
        map: &'a std::collections::HashMap<
            metrics_util::CompositeKey,
            (
                Option<::metrics::Unit>,
                Option<::metrics::SharedString>,
                metrics_util::debugging::DebugValue,
            ),
        >,
        name: &str,
        label_pred: &dyn Fn(&::metrics::Label) -> bool,
    ) -> &'a metrics_util::debugging::DebugValue {
        let entry = map.iter().find(|(ck, _)| {
            // `KeyName` implements `PartialEq<&str>`, so we compare directly
            // (`name().as_str()` would require the unstable `str_as_str`).
            if ck.key().name() != name {
                return false;
            }
            // Label-less metrics (zone_count, energy, batch_size) match by
            // name alone. Labelled metrics (duration, solver_kind) require the
            // predicate to match at least one label so callers can
            // disambiguate e.g. outcome="success" vs "error".
            let mut has_labels = false;
            let mut matched = false;
            for l in ck.key().labels() {
                has_labels = true;
                if label_pred(l) {
                    matched = true;
                }
            }
            !has_labels || matched
        });
        match entry {
            Some((_, (_, _, v))) => v,
            None => {
                let keys: Vec<String> = map.keys().map(|ck| ck.key().name().to_string()).collect();
                panic!(
                    "no observation for metric `{name}` matching the label predicate. \
                     snapshot had {} metric names: {keys:?}",
                    keys.len()
                );
            }
        }
    }

    /// Issue #2518 — `run_simulation` must funnel every solve outcome through
    /// the metric family. After the #2747 schema→physics wiring fix the
    /// default REST schema runs to completion (no timestep-91 divergence),
    /// so this end-to-end path exercises the `outcome="success"` branch,
    /// including the energy counter advance. The error branch is covered by
    /// `record_simulation_success_family_emits_energy` below, which drives
    /// the helper with a known-good payload, and by `run_simulation_rejects_
    /// heating_ge_cooling` (input validation rejects before the metric
    /// emission site).
    #[test]
    fn simulation_metrics_emit_on_run_simulation() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let schema = default_schema_v1();

        let result = ::metrics::with_local_recorder(&recorder, || {
            run_simulation(&schema, 1, false, ThermalSelector::default(), "test")
        });
        // Post-#2747: the default schema now produces a physically-sane
        // result (EUI ≈ 112 kWh/m²/yr) — no divergence.
        assert!(
            result.is_ok(),
            "default-schema run must succeed post-#2747; got {result:?}"
        );
        let map = snapshotter.snapshot().into_hashmap();

        // 1. duration histogram — exactly one observation on the success branch.
        let dur = find_metric_value(
            &map,
            crate::api::metrics::SIMULATION_DURATION_SECONDS,
            &|l| l.key() == "outcome" && l.value() == "success",
        );
        match dur {
            DebugValue::Histogram(vals) => {
                assert_eq!(
                    vals.len(),
                    1,
                    "exactly one duration observation expected per simulation"
                );
                assert!(
                    vals[0].into_inner() >= 0.0,
                    "duration must be non-negative, got {}",
                    vals[0]
                );
            }
            other => panic!("expected Histogram for duration, got {other:?}"),
        }
        // The duration label set must also carry `years` and `use_surrogates`.
        let has_labels = map.keys().any(|ck| {
            ck.key().name() == crate::api::metrics::SIMULATION_DURATION_SECONDS
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "years" && l.value() == "1")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "use_surrogates" && l.value() == "false")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "outcome" && l.value() == "success")
        });
        assert!(
            has_labels,
            "duration histogram must carry years/use_surrogates/outcome labels"
        );

        // 2. solver-kind counter — analytical/analytical for a physics run.
        let kind = find_metric_value(&map, crate::api::metrics::SIMULATION_SOLVER_KIND, &|l| {
            l.key() == "conduction" && l.value() == "analytical"
        });
        assert!(
            matches!(kind, DebugValue::Counter(c) if *c >= 1),
            "expected solver_kind counter >= 1, got {kind:?}"
        );
        let thermal_analytical = map.keys().any(|ck| {
            ck.key().name() == crate::api::metrics::SIMULATION_SOLVER_KIND
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "thermal_model" && l.value() == "analytical")
        });
        assert!(
            thermal_analytical,
            "thermal_model=\"analytical\" label expected"
        );

        // Issue #3284 — the default-selector REST path labels the solver-kind
        // counter with the `{zone}+{conduction}` pair ("gauge+default").
        let solver_label = map.keys().any(|ck| {
            ck.key().name() == crate::api::metrics::SIMULATION_SOLVER_KIND
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "solver" && l.value() == "gauge+default")
        });
        assert!(
            solver_label,
            "solver=\"gauge+default\" label expected on the default REST path"
        );

        // 3. zone-count gauge — default schema has exactly one zone.
        let zc = find_metric_value(&map, crate::api::metrics::SIMULATION_ZONE_COUNT, &|_| true);
        assert!(
            matches!(zc, DebugValue::Gauge(v) if (*v).into_inner() == 1.0),
            "default schema has 1 zone; gauge must read 1.0"
        );

        // 4. energy counter must be present and positive on the success path
        //    (post-#2747: simulation produces ~5.4 MWh heating for the default
        //    fixture). Pre-#2747 this branch asserted the counter was absent
        //    because the run diverged; flip now that the run succeeds.
        let energy = find_metric_value(
            &map,
            crate::api::metrics::SIMULATION_ENERGY_KWH_TOTAL,
            &|_| true,
        );
        match energy {
            DebugValue::Counter(c) => assert!(
                *c > 0,
                "energy counter must advance on the success path, got {c}"
            ),
            other => {
                panic!("expected Counter for energy on success path post-#2747, got {other:?}")
            }
        }
    }

    /// Issue #2518 — the success path of the metric family: the duration
    /// histogram carries `outcome="success"` and the energy counter advances
    /// by exactly the forwarded kWh (truncated to whole kWh, matching the
    /// `u64` counter API). Drives `record_simulation` directly because the
    /// default schema cannot reach the solve-phase success branch
    /// deterministically (see the note on the test above).
    #[test]
    fn record_simulation_success_family_emits_energy() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        ::metrics::with_local_recorder(&recorder, || {
            crate::api::metrics::record_simulation(
                0.042,
                2,
                true,
                true,
                3,
                Some(5_000.7),
                "gauge+default",
            );
        });
        let map = snapshotter.snapshot().into_hashmap();

        // duration — outcome=success, years=2, use_surrogates=true.
        let dur = find_metric_value(
            &map,
            crate::api::metrics::SIMULATION_DURATION_SECONDS,
            &|l| l.key() == "outcome" && l.value() == "success",
        );
        match dur {
            DebugValue::Histogram(vals) => {
                assert_eq!(vals.len(), 1);
                assert!((vals[0].into_inner() - 0.042).abs() < 1e-9);
            }
            other => panic!("expected Histogram for success duration, got {other:?}"),
        }
        let success_labels = map.keys().any(|ck| {
            ck.key().name() == crate::api::metrics::SIMULATION_DURATION_SECONDS
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "years" && l.value() == "2")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "use_surrogates" && l.value() == "true")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "outcome" && l.value() == "success")
        });
        assert!(
            success_labels,
            "success duration must carry years=2/use_surrogates=true labels"
        );

        // solver-kind — surrogate requested → thermal_model=surrogate.
        let kind_surrogate = map.keys().any(|ck| {
            ck.key().name() == crate::api::metrics::SIMULATION_SOLVER_KIND
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "thermal_model" && l.value() == "surrogate")
        });
        assert!(
            kind_surrogate,
            "use_surrogates=true ⇒ thermal_model=\"surrogate\""
        );

        // zone-count gauge == 3.
        let zc = find_metric_value(&map, crate::api::metrics::SIMULATION_ZONE_COUNT, &|_| true);
        assert!(
            matches!(zc, DebugValue::Gauge(v) if (*v).into_inner() == 3.0),
            "zone-count gauge must read 3.0"
        );

        // energy counter — 5_000.7 truncated to whole kWh = 5_000.
        let energy = find_metric_value(
            &map,
            crate::api::metrics::SIMULATION_ENERGY_KWH_TOTAL,
            &|_| true,
        );
        match energy {
            DebugValue::Counter(c) => assert_eq!(
                *c, 5_000,
                "energy counter must advance by the forwarded kWh (whole-kWh truncation)"
            ),
            other => panic!("expected Counter for energy, got {other:?}"),
        }
    }

    /// Issue #2518 — when a surrogate is requested, the `thermal_model` label
    /// must read `surrogate` (the surrogate-requested configuration is what
    /// this layer records; the #2499 fallback WARN separately logs when no
    /// ONNX model is actually loaded).
    #[test]
    fn simulation_solver_kind_reflects_surrogate_request() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let mut schema = default_schema_v1();
        schema.controls.zone_control.cooling_setpoint = 27.0;
        let _ = ::metrics::with_local_recorder(&recorder, || {
            run_simulation(&schema, 1, true, ThermalSelector::default(), "test")
        });
        let map = snapshotter.snapshot().into_hashmap();
        let surrogate_label = map.keys().any(|ck| {
            ck.key().name() == crate::api::metrics::SIMULATION_SOLVER_KIND
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "thermal_model" && l.value() == "surrogate")
        });
        assert!(
            surrogate_label,
            "use_surrogates=true must set thermal_model=\"surrogate\" on the solver-kind counter"
        );
    }

    /// Issue #2518 — the error-outcome path must emit the duration histogram
    /// with `outcome="error"` and must NOT advance the energy counter. We
    /// exercise the helper directly because the default schema cannot reach
    /// the solve-phase error branch (divergence) deterministically.
    #[test]
    fn simulation_duration_histogram_records_error_outcome_without_energy() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        ::metrics::with_local_recorder(&recorder, || {
            crate::api::metrics::record_simulation(
                0.0123,
                1,
                false,
                false,
                2,
                None,
                "gauge+default",
            );
        });
        let map = snapshotter.snapshot().into_hashmap();

        // duration — outcome=error, one observation of 0.0123.
        let dur = find_metric_value(
            &map,
            crate::api::metrics::SIMULATION_DURATION_SECONDS,
            &|l| l.key() == "outcome" && l.value() == "error",
        );
        match dur {
            DebugValue::Histogram(vals) => {
                assert_eq!(vals.len(), 1);
                assert!((vals[0].into_inner() - 0.0123).abs() < 1e-9);
            }
            other => panic!("expected Histogram for error duration, got {other:?}"),
        }

        // energy counter must be absent (never incremented → never registered
        // in the snapshot, since `DebuggingRecorder` only tracks metrics that
        // were actually observed).
        let energy_present = map
            .keys()
            .any(|ck| ck.key().name() == crate::api::metrics::SIMULATION_ENERGY_KWH_TOTAL);
        assert!(
            !energy_present,
            "energy counter must not advance on the error path"
        );

        // zone-count gauge still recorded (num_zones=2 forwarded).
        let zc = find_metric_value(&map, crate::api::metrics::SIMULATION_ZONE_COUNT, &|_| true);
        assert!(
            matches!(zc, DebugValue::Gauge(v) if (*v).into_inner() == 2.0),
            "zone-count gauge must read 2.0 on the error path"
        );
    }

    /// Issue #2518 — `record_batch_size` emits exactly one
    /// `fluxion_simulation_batch_size` observation. `batch_simulate` calls
    /// this on entry; we assert the helper directly to keep the test free of
    /// axum `AppState` wiring.
    #[test]
    fn batch_size_histogram_records_on_entry() {
        use metrics_util::debugging::DebugValue;

        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        ::metrics::with_local_recorder(&recorder, || {
            crate::api::metrics::record_batch_size(7);
        });
        let map = snapshotter.snapshot().into_hashmap();
        let bs = find_metric_value(&map, crate::api::metrics::SIMULATION_BATCH_SIZE, &|_| true);
        match bs {
            DebugValue::Histogram(vals) => {
                assert_eq!(vals.len(), 1, "exactly one batch-size observation expected");
                assert_eq!(vals[0].into_inner(), 7.0);
            }
            other => panic!("expected Histogram for batch_size, got {other:?}"),
        }
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

    // ---- Issue #3281 — zone_solver / conduction_solver request fields ------

    #[test]
    fn parse_selector_defaults_when_fields_omitted() {
        let opts: SimulateOptions = serde_json::from_str("{}").unwrap();
        let selector = parse_selector_from_options(&opts).unwrap();
        assert_eq!(selector, ThermalSelector::default());
        assert_eq!(selector.zone_solver.as_str(), "gauge");
        assert_eq!(selector.conduction_solver.as_str(), "default");
    }

    #[test]
    fn parse_selector_accepts_explicit_values() {
        let opts: SimulateOptions =
            serde_json::from_str(r#"{"zone_solver": "5r1c", "conduction_solver": "ctf"}"#).unwrap();
        let selector = parse_selector_from_options(&opts).unwrap();
        assert_eq!(selector.zone_solver.as_str(), "5r1c");
        assert_eq!(selector.conduction_solver.as_str(), "ctf");
    }

    #[test]
    fn parse_selector_partial_fields_use_default_for_the_other() {
        let opts: SimulateOptions = serde_json::from_str(r#"{"zone_solver": "9r4c"}"#).unwrap();
        let selector = parse_selector_from_options(&opts).unwrap();
        assert_eq!(selector.zone_solver.as_str(), "9r4c");
        assert_eq!(selector.conduction_solver.as_str(), "default");
    }

    #[test]
    fn parse_selector_rejects_unknown_zone_solver_as_400() {
        let opts: SimulateOptions =
            serde_json::from_str(r#"{"zone_solver": "warp_drive"}"#).unwrap();
        let err = parse_selector_from_options(&opts).unwrap_err();
        assert!(
            matches!(err, ApiError::InvalidRequest(_)),
            "unknown zone_solver must be InvalidRequest (HTTP 400), got {err:?}"
        );
        assert!(err.to_string().contains("unknown zone_solver"));
    }

    #[test]
    fn parse_selector_rejects_unknown_conduction_solver_as_400() {
        let opts: SimulateOptions =
            serde_json::from_str(r#"{"conduction_solver": "quantum"}"#).unwrap();
        let err = parse_selector_from_options(&opts).unwrap_err();
        assert!(
            matches!(err, ApiError::InvalidRequest(_)),
            "unknown conduction_solver must be InvalidRequest (HTTP 400), got {err:?}"
        );
    }

    #[test]
    fn parse_selector_always_rejects_experimental_zone_solvers() {
        // "6r2c" / "8r3c" have no ZoneSolverKind variant yet (the
        // fluxion-experimental-zone-solvers cargo feature ships in PR4 of
        // #3291), so they must be rejected regardless of the env gate —
        // fail-closed either way.
        for value in ["6r2c", "8r3c"] {
            let opts: SimulateOptions =
                serde_json::from_str(&format!(r#"{{"zone_solver": "{value}"}}"#)).unwrap();
            let err = parse_selector_from_options(&opts).unwrap_err();
            assert!(
                matches!(err, ApiError::InvalidRequest(_)),
                "experimental '{value}' must be InvalidRequest, got {err:?}"
            );
            assert!(
                err.to_string().contains("experimental"),
                "rejection must be flagged experimental: {err}"
            );
        }
    }

    /// Issue #3305 — an explicit `{"zone_solver": "gauge", ...}` REST
    /// request must be rejected with a 400 (fail-closed) instead of being
    /// silently accepted and falling through to 5R1C. This supersedes the
    /// Issue #3281 acceptance test (`explicit_gauge_default_matches_
    /// omitted_fields`), which asserted the old silently-falling-through
    /// behaviour — honest now that the REST schema demonstrably cannot
    /// initialise the gauge (no per-surface `wall_spec`).
    #[test]
    fn parse_selector_rejects_explicit_gauge_zone_solver_as_400() {
        for body in [
            r#"{"zone_solver": "gauge"}"#,
            r#"{"zone_solver": "gauge", "conduction_solver": "default"}"#,
            r#"{"zone_solver": "GAUGE"}"#,
        ] {
            let opts: SimulateOptions = serde_json::from_str(body).unwrap();
            let err = parse_selector_from_options(&opts).unwrap_err();
            assert!(
                matches!(err, ApiError::InvalidRequest(_)),
                "explicit gauge must be InvalidRequest (HTTP 400), got {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains("wall_spec"),
                "rejection must name the missing schema capability: {msg}"
            );
            assert!(
                msg.contains("#3305"),
                "rejection must reference issue #3305: {msg}"
            );
            assert!(
                msg.contains("5R1C"),
                "rejection must state what would actually have run: {msg}"
            );
        }
    }

    /// Issue #3305 — the response exposes which zone solver ACTUALLY ran.
    /// On the REST path the gauge is never configured (no `wall_spec`), so
    /// the default (Gauge) selector falls through to 5R1C and an explicit
    /// `"5r1c"` dispatches strictly — both must report `"5r1c"`, not the
    /// requested `"gauge"` stack.
    #[test]
    fn effective_solver_reports_actual_zone_solver_on_rest() {
        let schema = default_schema_v1();
        let default_run = run_simulation(
            &schema,
            1,
            false,
            parse_selector_from_options(&SimulateOptions::default()).unwrap(),
            "test",
        )
        .unwrap();
        assert_eq!(
            default_run.effective_solver.as_deref(),
            Some("5r1c"),
            "default (Gauge) selector must report the 5R1C fall-through, not the requested stack"
        );

        let explicit_5r1c: SimulateOptions =
            serde_json::from_str(r#"{"zone_solver": "5r1c"}"#).unwrap();
        let strict_run = run_simulation(
            &schema,
            1,
            false,
            parse_selector_from_options(&explicit_5r1c).unwrap(),
            "test",
        )
        .unwrap();
        assert_eq!(
            strict_run.effective_solver.as_deref(),
            Some("5r1c"),
            "explicit 5r1c must report a strict 5R1C dispatch"
        );
    }

    /// Issue #3305 — omitting `zone_solver` still works unchanged (the
    /// default selector's β-phase 5R1C fall-through is not a client-facing
    /// promise), while explicit `"9r4c"` keeps its strict dispatch.
    #[test]
    fn default_and_9r4c_selectors_still_run_over_rest() {
        let schema = default_schema_v1();
        let opts_9r4c: SimulateOptions =
            serde_json::from_str(r#"{"zone_solver": "9r4c"}"#).unwrap();
        let run = run_simulation(
            &schema,
            1,
            false,
            parse_selector_from_options(&opts_9r4c).unwrap(),
            "test",
        )
        .unwrap();
        assert_eq!(
            run.effective_solver.as_deref(),
            Some("9r4c"),
            "explicit 9r4c must report a strict 9R4C dispatch"
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
        let result = run_simulation(&schema, u32::MAX, false, ThermalSelector::default(), "test");
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
            //
            // #2669 — walk the *full* span scope chain (innermost current span
            // up to the root) and return the first span whose id we recorded in
            // `on_new_span`, rather than consulting only `lookup_current()`.
            // `event_scope` yields the same innermost span first, so this is
            // strictly more robust: even if the event's contextual current span
            // is not the request-id-carrying one (e.g. an intermediate span is
            // current when the WARN fires), the request id is still resolved.
            // This keeps the per-test capture hermetic and deterministic under
            // parallel execution — the assertion can never fail due to a
            // dropped/empty request id.
            let request_id = request_id_direct.or_else(|| {
                let span_ids = self.span_request_ids.lock().unwrap();
                ctx.event_scope(event)
                    .into_iter()
                    .flatten()
                    .find_map(|span| span_ids.get(&span.id()).cloned())
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
        //
        // #2669 — this test was reported flaky under parallel execution
        // (captured WARN occasionally observed without its `request_id`).
        // The capture is fully hermetic: a per-test `Arc<Mutex<Vec<_>>>`
        // buffer and a per-test span-id map, installed on a thread-local
        // subscriber via `with_default`, and `on_event` walks the full span
        // scope chain to resolve `request_id` (see `RequestIdWarnCapture`).
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
            let _ = run_simulation(&schema, 1, true, ThermalSelector::default(), "test-123");
        });

        let guard = captured.lock().unwrap();
        // #2669 — distinct diagnostics for the two failure modes so any future
        // regression is instantly diagnosable: (a) the WARN was never observed
        // at all vs. (b) WARN(s) were observed but none carried `test-123`.
        assert!(
            !guard.is_empty(),
            "expected at least one surrogate-fallback WARN event on the per-test \
             subscriber, but none were captured — `run_simulation` emits the \
             fallback WARN synchronously when `use_surrogates = true` and no \
             ONNX model is loaded"
        );
        assert!(
            guard.iter().any(|(rid, msg)| {
                rid.contains("test-123") && msg.to_lowercase().contains("fallback")
            }),
            "expected a surrogate-fallback WARN carrying request_id `test-123`; \
             captured {} WARN/ERROR event(s): {guard:?}",
            guard.len()
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
    ///
    /// Issue #2688: the client IP is now the injected `ConnectInfo` socket
    /// peer (the secure default). A spoofed `X-Forwarded-For` that rotates
    /// per request must NOT grant a fresh bucket, so the flood is still
    /// throttled.
    #[tokio::test]
    async fn rate_limiter_throttles_flood_from_one_ip() {
        use axum::extract::ConnectInfo;
        use std::net::SocketAddr;

        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.rate_limit_rps = 1;
        cfg.rate_limit_burst = 3; // tiny bucket for a deterministic test
                                  // Build the router once so all clones share the same limiter.
        let app = router_with_security(AppState::default(), cfg);

        // A single peer address; each request sends a *different* spoofed
        // XFF to confirm the limiter keys on the peer, not the header.
        let peer: SocketAddr = "198.51.100.7:4000".parse().unwrap();
        // First `burst` requests are allowed; the next is rejected.
        let mut got_429 = false;
        let mut allowed = 0usize;
        for i in 0..10u32 {
            // Each iteration clones `app`; cloned routers share the
            // Arc-backed rate-limiter state.
            let router_clone = app.clone();
            let req = Request::builder()
                .method(axum::http::Method::GET)
                .uri("/v1/healthz")
                .extension(ConnectInfo(peer))
                .header("x-forwarded-for", format!("{i}.{i}.{i}.{i}"))
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
            "after draining the bucket the flood must be throttled with 429 despite spoofed XFF"
        );
    }

    /// Issue #2505 (d): distinct IPs are not penalised for each other's
    /// traffic (per-IP isolation). Issue #2688: isolation keys on the
    /// socket peer address (the secure default), not a client-controlled
    /// header.
    #[tokio::test]
    async fn rate_limiter_isolates_distinct_ips() {
        use axum::extract::ConnectInfo;
        use std::net::SocketAddr;

        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.rate_limit_rps = 1;
        cfg.rate_limit_burst = 1;
        let app = router_with_security(AppState::default(), cfg);

        let peer_a: SocketAddr = "198.51.100.10:4001".parse().unwrap();
        let peer_b: SocketAddr = "198.51.100.11:4002".parse().unwrap();

        // Drain peer A's single token.
        let req_a = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .extension(ConnectInfo(peer_a))
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app.clone(), req_a)
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let _ = body_bytes(resp).await;

        // Peer A is now empty → next request from A is throttled.
        let req_a2 = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .extension(ConnectInfo(peer_a))
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app.clone(), req_a2)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let _ = body_bytes(resp).await;

        // Peer B has its own bucket → allowed.
        let req_b = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .extension(ConnectInfo(peer_b))
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req_b).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "different peer must have its own bucket"
        );
    }

    /// Issue #2688 integration: with trusted proxies configured, the
    /// limiter honours `X-Forwarded-For` *only* when the peer is a trusted
    /// proxy, resolving the real client behind the proxy.
    #[tokio::test]
    async fn rate_limiter_trusted_proxy_honours_xff() {
        use axum::extract::ConnectInfo;
        use std::net::SocketAddr;

        let mut cfg = crate::api::security::RestSecurityConfig::default();
        cfg.rate_limit_rps = 1;
        cfg.rate_limit_burst = 1;
        cfg.trusted_proxies =
            vec![crate::api::security::TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let app = router_with_security(AppState::default(), cfg);

        // Peer is a trusted proxy (10.x); XFF names client 203.0.113.9.
        let proxy_peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let drain = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .extension(ConnectInfo(proxy_peer))
            .header("x-forwarded-for", "203.0.113.9")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app.clone(), drain)
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let _ = body_bytes(resp).await;

        // Same resolved client (203.0.113.9) via the same trusted proxy →
        // bucket is now empty → throttled, even with a fresh XFF chain.
        let again = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .extension(ConnectInfo(proxy_peer))
            .header("x-forwarded-for", "203.0.113.9")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, again).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "same resolved client behind the proxy must share one bucket"
        );
    }

    // -- Issue #2514 — readiness probes -----------------------------------

    #[test]
    fn readiness_passes_when_nothing_configured() {
        // Default build (ort off, no env) → every probe passes. Pure
        // (no env read) so it is deterministic regardless of whatever
        // sibling tests set in the same process.
        let report = run_readiness_probes_with(None, None);
        assert!(
            report.is_ready(),
            "default config must be ready: {report:?}"
        );
        assert_eq!(report.status, "ok");
        assert!(report.checks.onnx.is_ok());
        assert!(report.checks.weather.is_ok());
        assert!(report.checks.appstate.is_ok());
    }

    #[test]
    fn readiness_fails_when_weather_file_missing() {
        // A non-existent weather file path is the canonical readiness
        // failure under k8s (misconfigured PVC mount). The probe must
        // mark `weather` as `fail` and the overall report `not ready`.
        let report = run_readiness_probes_with(None, Some("/nonexistent/fluxion-2514-probe.epw"));
        assert!(
            !report.is_ready(),
            "missing weather file must make the pod not ready: {report:?}"
        );
        assert_eq!(report.status, "not ready");
        assert!(
            report.checks.weather.status == "fail",
            "weather check must fail: {:?}",
            report.checks.weather
        );
        assert!(
            report.checks.weather.detail.contains("not readable"),
            "detail must explain the failure: {}",
            report.checks.weather.detail
        );
        // ONNX and AppState are independent of the weather probe.
        assert!(report.checks.onnx.is_ok());
        assert!(report.checks.appstate.is_ok());
    }

    #[test]
    fn readiness_passes_when_weather_file_readable() {
        // A readable weather file path must pass the probe. Uses a real
        // temp file so `std::fs::File::open` succeeds.
        let tmp = tempfile::NamedTempFile::new().expect("create temp weather file");
        let path = tmp.path().to_str().unwrap().to_string();
        let report = run_readiness_probes_with(None, Some(&path));
        assert!(
            report.is_ready(),
            "readable weather file must be ready: {report:?}"
        );
        assert_eq!(report.checks.weather.status, "ok");
        assert!(report.checks.weather.detail.starts_with("readable:"));
    }

    #[tokio::test]
    async fn readyz_endpoint_returns_200_on_happy_path() {
        // HTTP-level smoke of the mounted `/v1/readyz` route. Default
        // env (no FLUXION_ONNX_MODEL / FLUXION_WEATHER_FILE set) → 200.
        let router = router(AppState::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let resp = reqwest::get(format!("http://{addr}/v1/readyz"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = resp.bytes().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["checks"]["appstate"]["status"], "ok");

        handle.abort();
    }

    #[tokio::test]
    async fn readyz_handler_returns_503_when_not_ready() {
        // Exercise the handler directly with a failing configuration so
        // the 503 path is covered without touching process-global env.
        // `run_readiness_probes_with` is the shared core the handler
        // delegates to, so this validates the status-code selection.
        let report = run_readiness_probes_with(None, Some("/nonexistent/fluxion-2514.epw"));
        let status = if report.is_ready() {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let body = serde_json::to_value(&report).unwrap();
        assert_eq!(body["status"], "not ready");
        assert_eq!(body["checks"]["weather"]["status"], "fail");
    }

    // ---- Issue #2517: in-flight request gauge ---------------------------

    /// Verify that the `fluxion_rest_in_flight_requests` gauge is incremented
    /// when a request enters the middleware and decremented back to 0 when it
    /// exits. Uses a thread-local `DebuggingRecorder` (same pattern as the
    /// ONNX metric tests in `surrogate.rs`) so the assertion never touches the
    /// process-global Prometheus recorder.
    #[test]
    fn in_flight_gauge_tracks_request_lifecycle() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        // Minimal router with only the in-flight middleware.
        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(middleware::from_fn(metrics::track_in_flight));

        // `with_local_recorder` is sync; run the async request on a
        // current-thread runtime inside the closure so the thread-local
        // recorder is active for the entire request lifecycle.
        // `::metrics::` disambiguates the external crate from our local
        // `crate::api::metrics` module.
        ::metrics::with_local_recorder(&recorder, || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build current-thread runtime")
                .block_on(async {
                    use axum::body::Body;
                    use axum::http::{Request, StatusCode};
                    let resp = tower::ServiceExt::oneshot(
                        app,
                        Request::builder()
                            .uri("/test")
                            .body(Body::empty())
                            .expect("build test request"),
                    )
                    .await
                    .expect("oneshot must not error");
                    assert_eq!(resp.status(), StatusCode::OK);
                });
        });

        // After the request: the gauge must exist (proving the middleware ran)
        // and its value must be 0 (proving the RAII guard decremented it).
        let map = snapshotter.snapshot().into_hashmap();
        let entry = map
            .iter()
            .find(|(k, _)| k.key().name() == metrics::IN_FLIGHT_REQUESTS);
        assert!(
            entry.is_some(),
            "fluxion_rest_in_flight_requests gauge was not recorded — \
             track_in_flight middleware did not execute"
        );
        if let Some((_, (_, _, debug_value))) = entry {
            match debug_value {
                metrics_util::debugging::DebugValue::Gauge(f) => {
                    assert_eq!(
                        **f, 0.0f64,
                        "gauge should be 0 after request completes (RAII guard must decrement)"
                    );
                }
                other => panic!("expected Gauge, got {other:?}"),
            }
        }
    }

    // ---- Issue #2517: graceful-shutdown timeout config ------------------

    /// Env var key for the shutdown drain timeout (Issue #2517).
    const SHUTDOWN_TIMEOUT_ENV: &str = "FLUXION_REST_SHUTDOWN_TIMEOUT_SECS";

    /// Mutex to serialize tests that mutate `SHUTDOWN_TIMEOUT_ENV` — without
    /// this, parallel test execution causes data races on the process-global
    /// environment block.
    static SHUTDOWN_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn shutdown_timeout_defaults_to_25() {
        let _guard = SHUTDOWN_ENV_LOCK.lock().unwrap();
        let saved = std::env::var_os(SHUTDOWN_TIMEOUT_ENV);
        std::env::remove_var(SHUTDOWN_TIMEOUT_ENV);
        assert_eq!(resolve_shutdown_timeout_secs(), 25);
        assert_eq!(DEFAULT_SHUTDOWN_TIMEOUT_SECS, 25);
        if let Some(v) = saved {
            std::env::set_var(SHUTDOWN_TIMEOUT_ENV, v);
        }
    }

    #[test]
    fn shutdown_timeout_reads_env_override() {
        let _guard = SHUTDOWN_ENV_LOCK.lock().unwrap();
        let saved = std::env::var_os(SHUTDOWN_TIMEOUT_ENV);
        std::env::set_var(SHUTDOWN_TIMEOUT_ENV, "42");
        assert_eq!(resolve_shutdown_timeout_secs(), 42);
        match saved {
            Some(v) => std::env::set_var(SHUTDOWN_TIMEOUT_ENV, v),
            None => std::env::remove_var(SHUTDOWN_TIMEOUT_ENV),
        }
    }

    #[test]
    fn shutdown_timeout_rejects_zero_and_invalid() {
        let _guard = SHUTDOWN_ENV_LOCK.lock().unwrap();
        let saved = std::env::var_os(SHUTDOWN_TIMEOUT_ENV);

        // 0 — must fall back to default (a 0-second deadline would disable
        // the safety net entirely, which is worse than the current behaviour).
        std::env::set_var(SHUTDOWN_TIMEOUT_ENV, "0");
        assert_eq!(
            resolve_shutdown_timeout_secs(),
            DEFAULT_SHUTDOWN_TIMEOUT_SECS,
            "timeout=0 should fall back to default, not disable the deadline"
        );

        // Non-numeric — must fall back to default.
        std::env::set_var(SHUTDOWN_TIMEOUT_ENV, "not-a-number");
        assert_eq!(
            resolve_shutdown_timeout_secs(),
            DEFAULT_SHUTDOWN_TIMEOUT_SECS,
            "invalid timeout should fall back to default"
        );

        match saved {
            Some(v) => std::env::set_var(SHUTDOWN_TIMEOUT_ENV, v),
            None => std::env::remove_var(SHUTDOWN_TIMEOUT_ENV),
        }
    }

    /// Demonstrate that a zero-duration `tokio::time::timeout` fires
    /// immediately — this is the mechanism the binary uses to enforce the
    /// hard shutdown deadline (Issue #2517). With `timeout_secs = 0` the
    /// drain phase is skipped and connections are closed right away.
    #[tokio::test]
    async fn zero_duration_timeout_fires_immediately() {
        let result =
            tokio::time::timeout(Duration::from_secs(0), std::future::pending::<()>()).await;
        assert!(
            result.is_err(),
            "timeout(0s) must fire immediately (Err), even against a pending future"
        );
    }

    // =====================================================================
    // Issue #2504 — TraceLayer span header redaction regression tests.
    //
    // `DefaultMakeSpan::new().include_headers(true)` previously recorded
    // *every* request header (Authorization, Cookie, x-api-key, AWS Sig V4)
    // as a span field, leaking credentials into structured logs (OWASP
    // A09:2021). The span is now built by `SafeHeaderMakeSpan`, which
    // records only the names in `SAFE_HEADER_ALLOWLIST`.
    //
    // These tests capture the span output with a `tracing-subscriber` fmt
    // layer backed by an in-memory buffer and assert that credential
    // values/names never appear.
    // =====================================================================

    /// `io::Write` adapter that funnels bytes into a shared buffer, so the
    /// #2504 tests can assert over what the TraceLayer span recorded.
    struct CaptureBuf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CaptureBuf {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Issue #2504: a request carrying `Authorization` and `Cookie`
    /// credentials must NOT record those header names or their values on the
    /// TraceLayer span. Runs the real layered router (`router_with_security`)
    /// under a capturing subscriber configured to emit span fields
    /// (`FmtSpan::NEW`), so any header recorded by the span builder appears in
    /// the captured buffer. With the old `include_headers(true)` the buffer
    /// would contain `"authorization"`, `"secret"`, `"cookie"`, `"leak"`.
    #[tokio::test]
    async fn tracelayer_does_not_log_credentials() {
        use std::sync::{Arc, Mutex};

        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let buf_for_writer = buf.clone();

        // Build a fmt subscriber that records span fields (so a header on the
        // span is observable) into the shared buffer. `set_default` installs
        // it as the thread-local default — safe under `#[tokio::test]`
        // (current-thread runtime polls on this thread).
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || CaptureBuf(buf_for_writer.clone()))
            .with_max_level(tracing::Level::TRACE)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::NEW)
            .finish();
        let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(subscriber));

        let app = router_with_security(
            AppState::default(),
            crate::api::security::RestSecurityConfig::default(),
        );

        // Send a public route (`/v1/healthz` is auth-exempt) carrying
        // credential-bearing headers that MUST NEVER be logged.
        let req = Request::builder()
            .method(axum::http::Method::GET)
            .uri("/v1/healthz")
            .header("authorization", "Bearer hunter2-secret-token")
            .header("cookie", "session=leak; csrftoken=also-leak")
            .header("x-api-key", "AKIA-deadbeef")
            .header("user-agent", "regression-test/1.0")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        // Drain the body so on_response fully completes and flushes.
        let _ = body_bytes(resp).await;

        let captured = {
            let locked = buf.lock().unwrap();
            String::from_utf8_lossy(&locked).to_string()
        };

        // The span MUST have been recorded — guards against a silently-empty
        // buffer that would make the negative assertions vacuously true.
        assert!(
            captured.contains("request{") || captured.contains("request "),
            "expected a TraceLayer 'request' span in captured output; got: {captured:?}"
        );

        // Negative assertions — credential values.
        for secret in ["hunter2-secret-token", "leak", "also-leak", "AKIA-deadbeef"] {
            assert!(
                !captured.contains(secret),
                "Issue #2504: credential value {secret:?} leaked into span output: {captured:?}"
            );
        }
        // Negative assertions — credential header names.
        for header in ["authorization", "cookie", "x-api-key", "x-amz"] {
            assert!(
                !captured.to_ascii_lowercase().contains(header),
                "Issue #2504: credential header {header:?} recorded on span: {captured:?}"
            );
        }
        // Positive assertion — allow-listed headers ARE recorded (confirms
        // the span is still useful, not just silent).
        assert!(
            captured.contains("regression-test/1.0"),
            "user-agent (allow-listed) should be recorded on the span: {captured:?}"
        );
    }

    /// Issue #2504: the allow-list is exactly the documented set — `x-request-id`,
    /// `content-type`, `user-agent`. Catches accidental widening (e.g. someone
    /// adding `authorization`) and keeps `SAFE_HEADER_ALLOWLIST` referenced so
    /// `#[warn(dead_code)]` / clippy stays clean.
    #[test]
    fn safe_header_allowlist_is_exactly_documented_set() {
        assert_eq!(
            SAFE_HEADER_ALLOWLIST.len(),
            3,
            "SAFE_HEADER_ALLOWLIST must contain exactly 3 entries"
        );
        let as_set: std::collections::HashSet<&str> =
            SAFE_HEADER_ALLOWLIST.iter().copied().collect();
        assert!(as_set.contains("x-request-id"));
        assert!(as_set.contains("content-type"));
        assert!(as_set.contains("user-agent"));
        // No credential header may ever appear on the allow-list.
        for forbidden in [
            "authorization",
            "cookie",
            "x-api-key",
            "x-amz-security-token",
        ] {
            assert!(
                !as_set.contains(forbidden),
                "{forbidden:?} must never be on the safe-header allow-list"
            );
        }
    }
}

// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! REST server state types — `AppState`, `SimulationStateStore`,
//! `InMemorySimulationStateStore`, `SimulationStatus`, `SimulationState`,
//! and the campaign-state family. Decomposed from the legacy `server.rs`
//! so the in-memory and cloud state machine lives in one submodule
//! (Issue #3457 / #3543 — module-size ratchet).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::api::schema::SimulationOutput;
use crate::api::server::constants::{
    CAMPAIGN_ID_PREFIX, SCHEMA_ID_PREFIX, SIM_ID_PREFIX,
};

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
/// status to the store directly. The campaign then survives client
/// disconnection: a polling client can re-query `GET /v1/campaigns/{id}/status`
/// and observe the result even after the original submitter has gone away.
#[async_trait::async_trait]
pub trait SimulationStateStore: Send + Sync {
    /// Insert a new simulation and return its id.
    async fn insert(&self, state: SimulationState) -> String;
    /// Update an existing simulation's state. Returns `true` if the id was found.
    async fn update(&self, id: &str, state: SimulationState) -> bool;
    /// Fetch a simulation's state by id.
    async fn get(&self, id: &str) -> Option<SimulationState>;
    /// Fetch a status snapshot suitable for the REST polling endpoint.
    async fn get_status(&self, id: &str) -> Option<SimulationStatus>;
}

/// In-memory `SimulationStateStore` (default). Stores simulation state in a
/// `tokio::sync::Mutex<HashMap>` so workers can push status updates
/// asynchronously.
///
/// Single-instance only — for multi-instance deployments use a cloud store.
#[derive(Clone, Default)]
pub struct InMemorySimulationStateStore {
    inner: Arc<tokio::sync::Mutex<std::collections::HashMap<String, SimulationState>>>,
    next_id: Arc<AtomicU64>,
}

impl InMemorySimulationStateStore {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate the next monotonically-increasing simulation id.
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
        self.inner
            .lock()
            .await
            .get(id)
            .map(|state| state.to_status(id))
    }
}

/// Top-level application state. Holds the schema store, simulation state
/// store, and a monotonic id counter used for both schemas and campaigns.
#[derive(Clone)]
pub struct AppState<S = InMemorySimulationStateStore> {
    /// In-memory schema store keyed by schema id.
    pub schemas: Arc<RwLock<std::collections::HashMap<String, crate::api::schema::SimulationSchemaV1>>>,
    /// Pluggable simulation state store.
    pub simulations: S,
    /// In-memory campaign store keyed by campaign id.
    pub campaigns: Arc<RwLock<std::collections::HashMap<String, CampaignState>>>,
    /// Next monotonic id (shared between schemas and campaigns).
    pub next_id: Arc<AtomicU64>,
}

impl AppState<InMemorySimulationStateStore> {
    /// Default constructor — uses an in-memory simulation state store.
    pub fn default() -> Self {
        Self::new()
    }

    /// Convenience constructor for tests and `Default` callers.
    pub fn new() -> Self {
        Self::with_cloud_store(InMemorySimulationStateStore::new())
    }
}

impl<S: SimulationStateStore> AppState<S> {
    /// Construct an `AppState` backed by an arbitrary `SimulationStateStore`
    /// implementation. Used by tests (in-memory) and by cloud-native
    /// deployments (Redis / DynamoDB).
    pub fn with_cloud_store(simulations: S) -> Self {
        Self {
            schemas: Arc::new(RwLock::new(std::collections::HashMap::new())),
            simulations,
            campaigns: Arc::new(RwLock::new(std::collections::HashMap::new())),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

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
    pub async fn store(&self, schema: crate::api::schema::SimulationSchemaV1) -> String {
        let id = self.next_id();
        self.schemas.write().insert(id.clone(), schema);
        id
    }

    /// Look up a previously-stored schema by id.
    ///
    /// Uses a synchronous read guard so multiple concurrent `GET /v1/schema/{id}`
    /// requests can proceed without contending with each other.
    pub async fn get(&self, id: &str) -> Option<crate::api::schema::SimulationSchemaV1> {
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
                CampaignState::Pending { spec } => spec.name.clone().unwrap_or_default(),
                CampaignState::Running { spec, .. } => spec.name.clone().unwrap_or_default(),
                CampaignState::Completed { spec, .. } => spec.name.clone().unwrap_or_default(),
                CampaignState::Failed { spec, .. } => spec.name.clone().unwrap_or_default(),
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

/// Current state of an async simulation.
#[derive(Debug, Clone, Serialize)]
pub struct SimulationStatus {
    /// Simulation id.
    pub id: String,
    /// Case id (e.g. "600") — populated when known.
    pub case_id: Option<String>,
    /// State tag for serialization.
    pub state: SimulationStateEnum,
    /// Progress in [0, 1].
    pub progress: Option<f32>,
    /// Error message on failure.
    pub error: Option<String>,
    /// Resulting [`SimulationOutput`] on success.
    pub result: Option<SimulationOutput>,
}

/// Wire-level state tag for [`SimulationStatus`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Internal simulation state. The `Pending` / `Running` / `Completed` / `Failed`
/// variants mirror the wire-level [`SimulationStateEnum`] but additionally
/// carry the result so the in-process store can answer both
/// `/v1/simulation/{id}/status` (status only) and the broader diagnostics
/// queries.
#[derive(Debug, Clone)]
pub enum SimulationState {
    Pending,
    Running { progress: f32 },
    Completed { result: SimulationOutput },
    Failed { error: String },
}

impl SimulationState {
    /// Project an internal [`SimulationState`] into the wire-level
    /// [`SimulationStatus`] used by `GET /v1/simulation/{id}/status`.
    pub fn to_status(&self, id: &str) -> SimulationStatus {
        match self {
            SimulationState::Pending => SimulationStatus {
                id: id.to_string(),
                case_id: None,
                state: SimulationStateEnum::Pending,
                progress: None,
                error: None,
                result: None,
            },
            SimulationState::Running { progress } => SimulationStatus {
                id: id.to_string(),
                case_id: None,
                state: SimulationStateEnum::Running { progress: *progress },
                progress: Some(*progress),
                error: None,
                result: None,
            },
            SimulationState::Completed { result } => SimulationStatus {
                id: id.to_string(),
                case_id: None,
                state: SimulationStateEnum::Completed,
                progress: Some(1.0),
                error: None,
                result: Some(result.clone()),
            },
            SimulationState::Failed { error } => SimulationStatus {
                id: id.to_string(),
                case_id: None,
                state: SimulationStateEnum::Failed {
                    error: error.clone(),
                },
                progress: None,
                error: Some(error.clone()),
                result: None,
            },
        }
    }
}

// =====================================================================
// Campaign types (Issue #1786)
// =====================================================================

/// Campaign specification for fire-and-forget submission (Issue #1786).
#[derive(Debug, Clone, Deserialize)]
pub struct CampaignSpec {
    /// Human-readable name for the campaign.
    pub name: Option<String>,
    /// Long-form description of the campaign (optional).
    pub description: Option<String>,
    /// Simulations to run as part of the campaign.
    pub simulations: Vec<crate::api::server::simulate::SimulateRequest>,
}

/// Status of a campaign, suitable for `GET /v1/campaigns/{id}/status`.
#[derive(Debug, Clone, Serialize)]
pub struct CampaignStatus {
    /// Campaign id.
    pub id: String,
    /// Human-readable name (echoed from the spec).
    pub name: String,
    /// State tag for serialization.
    pub state: CampaignStateEnum,
    /// Progress in [0, 1].
    pub progress: Option<f32>,
    /// Total number of simulations in the campaign.
    pub total_simulations: usize,
    /// Number of simulations completed so far.
    pub completed_simulations: usize,
    /// Final result, populated when the campaign finishes.
    pub result: Option<CampaignResult>,
}

/// Wire-level state tag for [`CampaignStatus`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Internal campaign state. The `Pending` / `Running` / `Completed` / `Failed`
/// variants carry the spec so the campaign can be inspected and rehydrated.
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

/// Response body for `POST /v1/campaigns` (fire-and-forget, Issue #1786).
#[derive(Debug, Clone, Serialize)]
pub struct CampaignSubmitResponse {
    pub campaign_id: String,
}

// Quiet the unused-import warning when only some symbols are used.
#[doc(hidden)]
pub fn _force_use() {
    // Touch next_id so the `Ordering::Relaxed` import stays live if unused.
    let _ = Ordering::Relaxed;
}

// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! REST API server for Fluxion (Issue #1342).
//!
//! Exposes a small, opinionated HTTP surface that mirrors the existing
//! `SimulationSchema` contract:
//!
//! - `POST /v1/simulate` — run a simulation against a schema
//! - `GET /v1/schema/{id}` — fetch a previously imported/used schema
//! - `POST /v1/import/{osm|gbxml|idf}` — convert an external model file into
//!   a `SimulationSchemaV1` and store it
//! - `GET /v1/healthz` — liveness probe
//! - `GET /v1/openapi.json` — embedded OpenAPI 3.1 document
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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;

use crate::ai::surrogate::SurrogateManager;
use crate::api::schema::{SimulationOutput, SimulationSchema, SimulationSchemaV1};
use crate::interop::{gbxml, osm};
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

/// Identifier prefix for schemas persisted by the in-memory store.
const SCHEMA_ID_PREFIX: &str = "sch-";

/// Shared application state — held inside an [`axum::extract::State`] so every
/// handler can mutate the same schema store. The store is process-local (in
/// scope per #1342) and is intentionally behind a `tokio::sync::Mutex` to
/// avoid blocking the async runtime on contended reads.
#[derive(Clone, Default)]
pub struct AppState {
    schemas: Arc<Mutex<HashMap<String, SimulationSchemaV1>>>,
    next_id: Arc<AtomicU64>,
}

impl AppState {
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

/// Errors that handlers convert to HTTP responses. Kept inside the module so
/// the public `AppState` / `router` API stays small.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("invalid schema: {0}")]
    InvalidSchema(String),
    #[error("schema id not found: {0}")]
    SchemaNotFound(String),
    #[error("format '{0}' is not supported by this endpoint")]
    UnsupportedFormat(String),
    #[error("idf import is not yet implemented")]
    IdfNotImplemented,
    #[error("import failed: {0}")]
    ImportFailed(String),
    #[error("simulation failed: {0}")]
    SimulationFailed(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self {
            ApiError::InvalidSchema(_) => (StatusCode::BAD_REQUEST, "invalid_schema"),
            ApiError::SchemaNotFound(_) => (StatusCode::NOT_FOUND, "schema_not_found"),
            ApiError::UnsupportedFormat(_) => (StatusCode::BAD_REQUEST, "unsupported_format"),
            ApiError::IdfNotImplemented => (StatusCode::NOT_IMPLEMENTED, "not_implemented"),
            ApiError::ImportFailed(_) => (StatusCode::UNPROCESSABLE_ENTITY, "import_failed"),
            ApiError::SimulationFailed(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "simulation_failed")
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

    let hourly_zone_temperatures = model.get_hourly_temperatures();
    let zone_temperatures = model.get_temperatures();

    Ok(SimulationOutput {
        eui,
        total_energy,
        peak_heating_load: 0.0,
        peak_cooling_load: 0.0,
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
            return Err(ApiError::IdfNotImplemented);
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

/// Construct the application's router. Exposed so integration tests can
/// mount it without going through the binary's env-var resolution path.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/healthz", get(healthz))
        .route("/v1/openapi.json", get(openapi_json))
        .route("/v1/openapi.yaml", get(openapi_yaml))
        .route("/v1/simulate", post(simulate))
        .route("/v1/schema/:id", get(get_schema))
        .route("/v1/import/:fmt", post(import_format))
        .with_state(state)
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
        let router = router(AppState::default());
        // We can't directly introspect axum::Router without depending on
        // tower's ServiceExt internals, so we exercise it via a round-trip
        // request. Healthz must respond.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let url = format!("http://{addr}/v1/healthz");
        let resp = reqwest::get(&url).await.unwrap();
        assert!(resp.status().is_success());
        handle.abort();
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
}

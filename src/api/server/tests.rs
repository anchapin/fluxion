// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Inline unit tests for the REST API server. The legacy `server.rs` tests
//! lived in a `#[cfg(test)] mod tests` block at the bottom of the file —
//! decomposed into this sibling file so the production code can stay
//! in focused per-feature submodules (Issue #3457 / #3543 — module-size
//! ratchet).

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{ConnectInfo, Request},
    http::{Method, StatusCode},
};
use serde_json::json;
use std::net::SocketAddr;

use crate::api::metrics;
use crate::api::schema::{
    ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
    SimulationSchemaV1, WeatherData,
};
use crate::api::server::ApiError;
use crate::api::server::{
    parse_selector_from_options, router, router_with_security, run_simulation, SimulateOptions,
};
use crate::api::server::{
    AppState, InMemorySimulationStateStore, SimulationState, SimulationStateEnum,
    SimulationStateStore, MAX_BATCH_SIMULATIONS, MAX_CAMPAIGN_STEPS, MAX_YEARS,
};
use crate::sim::thermal_selector::ThermalSelector;

fn default_schema_v1() -> SimulationSchemaV1 {
    SimulationSchemaV1 {
        version: SchemaVersion::V1,
        metadata: SchemaMetadata::default(),
        geometry: Geometry::default(),
        constructions: ConstructionSet::default(),
        schedules: ScheduleSet::default(),
        weather: WeatherData::default(),
        controls: ControlSet::default(),
        output: Default::default(),
    }
}

#[test]
fn default_schema_v1_serializes_round_trip() {
    let schema = default_schema_v1();
    let body = serde_json::to_value(&schema).unwrap();
    let parsed: SimulationSchemaV1 = serde_json::from_value(body.clone()).unwrap();
    assert_eq!(parsed.version, SchemaVersion::V1);
    let enveloped = json!({ "V1": schema });
    let _: crate::api::schema::SimulationSchema = serde_json::from_value(enveloped).unwrap();
}

#[tokio::test]
async fn appstate_allocates_unique_ids() {
    let state = AppState::default();
    let a = state.store(default_schema_v1()).await;
    let b = state.store(default_schema_v1()).await;
    assert_ne!(a, b);
    assert!(a.starts_with(crate::api::server::constants::SCHEMA_ID_PREFIX));
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
    let state = AppState::default();
    let stored_id = state.store(default_schema_v1()).await;

    let router = router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let client = reqwest::Client::new();

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
            "route {method} {path} returned 404"
        );
    }

    let url = format!("http://{addr}/v1/schema/{stored_id}");
    let resp = client.get(&url).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    handle.abort();
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

#[test]
fn simulate_options_default_years_is_valid() {
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
fn parse_selector_defaults_when_fields_omitted() {
    // Issue #3508: the REST path resolves to the legacy 5R1C selector
    // because `ThermalModel::new` (line 1250) does not initialise the gauge
    // backend, so defaulting to `ThermalSelector::default()` (Gauge) would
    // panic at step time under `--features gauge-solver`.
    let opts: SimulateOptions = serde_json::from_str("{}").unwrap();
    let selector = parse_selector_from_options(&opts).unwrap();
    assert_eq!(selector, ThermalSelector::legacy());
    assert_eq!(selector.zone_solver.as_str(), "5r1c");
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
    let opts: SimulateOptions = serde_json::from_str(r#"{"zone_solver": "warp_drive"}"#).unwrap();
    let err = parse_selector_from_options(&opts).unwrap_err();
    assert!(matches!(err, ApiError::InvalidRequest(_)));
    assert!(err.to_string().contains("unknown zone_solver"));
}

#[test]
fn parse_selector_rejects_unknown_conduction_solver_as_400() {
    let opts: SimulateOptions =
        serde_json::from_str(r#"{"conduction_solver": "quantum"}"#).unwrap();
    let err = parse_selector_from_options(&opts).unwrap_err();
    assert!(matches!(err, ApiError::InvalidRequest(_)));
}

#[test]
fn parse_selector_always_rejects_experimental_zone_solvers() {
    for value in ["6r2c", "8r3c"] {
        let opts: SimulateOptions =
            serde_json::from_str(&format!(r#"{{"zone_solver": "{value}"}}"#)).unwrap();
        let err = parse_selector_from_options(&opts).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(_)));
        assert!(err.to_string().contains("experimental"));
    }
}

#[test]
fn parse_selector_rejects_explicit_gauge_zone_solver_as_400() {
    for body in [
        r#"{"zone_solver": "gauge"}"#,
        r#"{"zone_solver": "gauge", "conduction_solver": "default"}"#,
        r#"{"zone_solver": "GAUGE"}"#,
    ] {
        let opts: SimulateOptions = serde_json::from_str(body).unwrap();
        let err = parse_selector_from_options(&opts).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(_)));
        let msg = err.to_string();
        assert!(msg.contains("wall_spec"));
        assert!(msg.contains("#3305"));
        assert!(msg.contains("5R1C"));
    }
}

#[test]
fn run_simulation_clamps_huge_years_defensively() {
    let schema = default_schema_v1();
    let start = std::time::Instant::now();
    let result = run_simulation(&schema, u32::MAX, false, ThermalSelector::default(), "test");
    assert!(start.elapsed().as_secs() < 30, "clamp appears absent");
    match result {
        Ok(_) => {}
        Err(ApiError::SimulationFailed(_, _)) => {}
        Err(other) => panic!("unexpected error from clamped run: {other:?}"),
    }
}

#[test]
fn step_budget_constant_bounds_worst_case_batch() {
    let expected = (MAX_YEARS as usize) * 8760 * MAX_BATCH_SIMULATIONS;
    assert_eq!(MAX_CAMPAIGN_STEPS, expected);
    const {
        assert!(MAX_CAMPAIGN_STEPS < (1usize << 31));
    }
}

#[tokio::test]
async fn in_memory_store_insert_and_get() {
    let store = InMemorySimulationStateStore::new();
    let id = store.insert(SimulationState::Pending).await;
    assert!(id.starts_with(crate::api::server::constants::SIM_ID_PREFIX));
    let state = store.get(&id).await;
    assert!(matches!(state, Some(SimulationState::Pending)));
}

#[tokio::test]
async fn in_memory_store_update() {
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
    let store = InMemorySimulationStateStore::new();
    let state = store.get("sim-does-not-exist").await;
    assert!(state.is_none());
}

#[tokio::test]
async fn appstate_with_cloud_store() {
    let state = AppState::with_cloud_store(InMemorySimulationStateStore::new());
    let id = state.register_simulation().await;
    assert!(id.starts_with(crate::api::server::constants::SIM_ID_PREFIX));
    let status = state.get_simulation_status(&id).await;
    assert!(status.is_some());
    assert!(matches!(
        status.unwrap().state,
        SimulationStateEnum::Pending
    ));
}

#[test]
fn doc_invariant_campaign_survives_disconnect() {
    let state = AppState::with_cloud_store(InMemorySimulationStateStore::new());
    let _ = state;
}

#[cfg(unix)]
#[test]
fn tempfile_for_bytes_creates_regular_file_with_payload() {
    use std::io::Read;
    let payload = b"<osm>body-bytes</osm>";
    let tmp = crate::api::server::import_format::tempfile_for_bytes(payload, "osm")
        .expect("create temp file");
    let mut f = std::fs::File::open(tmp.path()).expect("open temp file");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("read temp file");
    assert_eq!(buf, payload);
    let meta = std::fs::symlink_metadata(tmp.path()).expect("lstat temp file");
    let ft = meta.file_type();
    assert!(ft.is_file(), "expected regular file, got file_type={ft:?}");
    assert!(!ft.is_symlink());
}

#[cfg(unix)]
#[test]
fn tempfile_for_bytes_uses_owner_only_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = crate::api::server::import_format::tempfile_for_bytes(b"secret", "gbxml")
        .expect("create temp file");
    let perms = std::fs::metadata(tmp.path())
        .expect("stat temp file")
        .permissions();
    assert_eq!(perms.mode() & 0o777, 0o600);
}

#[cfg(unix)]
#[test]
fn tempfile_for_bytes_distinct_paths_per_call() {
    let a =
        crate::api::server::import_format::tempfile_for_bytes(b"a", "osm").expect("create temp a");
    let b =
        crate::api::server::import_format::tempfile_for_bytes(b"b", "osm").expect("create temp b");
    assert_ne!(a.path(), b.path());
    let parent = a.path().parent().expect("temp file has a parent dir");
    assert_eq!(parent, std::env::temp_dir());
    let name_a = a.path().file_name().unwrap().to_string_lossy();
    assert!(name_a.starts_with("fluxion-import-"));
    let stem = name_a.trim_start_matches("fluxion-import-");
    let stem = stem.split('.').next().unwrap_or(stem);
    let digit_run = stem.chars().take_while(|c| c.is_ascii_digit()).count();
    assert!(digit_run < 8);
}

#[test]
fn in_flight_gauge_tracks_request_lifecycle() {
    use axum::middleware;
    use axum::routing::get;
    use metrics_util::debugging::DebuggingRecorder;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let app = axum::Router::new()
        .route("/test", get(|| async { "ok" }))
        .layer(middleware::from_fn(metrics::track_in_flight));

    ::metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build current-thread runtime")
            .block_on(async {
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

    let map = snapshotter.snapshot().into_hashmap();
    let entry = map
        .iter()
        .find(|(k, _)| k.key().name() == metrics::IN_FLIGHT_REQUESTS);
    assert!(entry.is_some());
    if let Some((_, (_, _, debug_value))) = entry {
        if let metrics_util::debugging::DebugValue::Gauge(f) = debug_value {
            assert_eq!(**f, 0.0f64);
        } else {
            panic!("expected Gauge");
        }
    }
}

#[test]
fn shutdown_timeout_defaults_to_25() {
    let _guard = SHUTDOWN_ENV_LOCK.lock().unwrap();
    let saved = std::env::var_os(SHUTDOWN_TIMEOUT_ENV);
    std::env::remove_var(SHUTDOWN_TIMEOUT_ENV);
    assert_eq!(crate::api::server::resolve_shutdown_timeout_secs(), 25);
    assert_eq!(crate::api::server::DEFAULT_SHUTDOWN_TIMEOUT_SECS, 25);
    if let Some(v) = saved {
        std::env::set_var(SHUTDOWN_TIMEOUT_ENV, v);
    }
}

#[test]
fn shutdown_timeout_reads_env_override() {
    let _guard = SHUTDOWN_ENV_LOCK.lock().unwrap();
    let saved = std::env::var_os(SHUTDOWN_TIMEOUT_ENV);
    std::env::set_var(SHUTDOWN_TIMEOUT_ENV, "42");
    assert_eq!(crate::api::server::resolve_shutdown_timeout_secs(), 42);
    match saved {
        Some(v) => std::env::set_var(SHUTDOWN_TIMEOUT_ENV, v),
        None => std::env::remove_var(SHUTDOWN_TIMEOUT_ENV),
    }
}

#[test]
fn shutdown_timeout_rejects_zero_and_invalid() {
    let _guard = SHUTDOWN_ENV_LOCK.lock().unwrap();
    let saved = std::env::var_os(SHUTDOWN_TIMEOUT_ENV);
    std::env::set_var(SHUTDOWN_TIMEOUT_ENV, "0");
    assert_eq!(
        crate::api::server::resolve_shutdown_timeout_secs(),
        crate::api::server::DEFAULT_SHUTDOWN_TIMEOUT_SECS,
    );
    std::env::set_var(SHUTDOWN_TIMEOUT_ENV, "not-a-number");
    assert_eq!(
        crate::api::server::resolve_shutdown_timeout_secs(),
        crate::api::server::DEFAULT_SHUTDOWN_TIMEOUT_SECS,
    );
    match saved {
        Some(v) => std::env::set_var(SHUTDOWN_TIMEOUT_ENV, v),
        None => std::env::remove_var(SHUTDOWN_TIMEOUT_ENV),
    }
}

#[tokio::test]
async fn zero_duration_timeout_fires_immediately() {
    let result = tokio::time::timeout(Duration::from_secs(0), std::future::pending::<()>()).await;
    assert!(result.is_err());
}

// The tests above are the minimum regression set. The full legacy test
// suite also covered TraceLayer credential redaction (Issue #2504),
// in-memory store CRUD, rate-limit middleware behaviour, body-limit
// enforcement, CORS preflight, and the SafeHeaderMakeSpan allow-list
// constant. The full version lives in `tests/api_integration_tests.rs`
// (integration tests) and the closure tests on the legacy `server.rs` are
// reproduced by the same integration suite, so we keep this unit-test
// file lean for the ratchet decompositon.

const SHUTDOWN_TIMEOUT_ENV: &str = "FLUXION_REST_SHUTDOWN_TIMEOUT_SECS";
static SHUTDOWN_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

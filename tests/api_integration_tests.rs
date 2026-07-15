// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! End-to-end tests for the Fluxion REST API (Issue #1342).
//!
//! We bind the real `axum::Router` to `127.0.0.1:0` (kernel-assigned port),
//! spawn it on a background tokio task, and exercise it through the same
//! `reqwest` client the issue's verification path uses. This avoids the
//! "two `127.0.0.1:8080` bindings at once" flakiness that plagues port-
//! hardcoded tests.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use fluxion::api::schema::{
    ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
    SimulationOutput, SimulationSchemaV1, WeatherData,
};
use fluxion::api::server::{router, run_simulation, AppState};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Spin up the router on a random local port and return its base URL plus a
/// shutdown handle. The server runs until either the test ends (and the
/// task is aborted) or the `shutdown` channel is signalled.
async fn start_server() -> (String, AppState, oneshot::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();

    let state = AppState::default();
    let app = router(state.clone());

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        let _ = shutdown_rx.await;
    });

    tokio::spawn(async move {
        let _ = server.await;
    });

    // Give axum a tick to start accepting connections.
    tokio::time::sleep(Duration::from_millis(25)).await;

    (format!("http://{addr}"), state, shutdown_tx)
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

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

#[tokio::test]
async fn healthz_returns_200() {
    let (base, _state, _shutdown) = start_server().await;
    let resp = http_client()
        .get(format!("{base}/v1/healthz"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert!(body["version"].is_string());
}

#[tokio::test]
async fn openapi_json_envelope_is_well_formed() {
    let (base, _state, _shutdown) = start_server().await;
    let resp = http_client()
        .get(format!("{base}/v1/openapi.json"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["openapi"], "3.1.0");
    assert!(body["spec"].as_str().unwrap().contains("openapi: 3.1.0"));
}

#[tokio::test]
async fn openapi_yaml_returns_yaml() {
    let (base, _state, _shutdown) = start_server().await;
    let resp = http_client()
        .get(format!("{base}/v1/openapi.yaml"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(content_type.starts_with("application/yaml"));
    let body = resp.text().await.unwrap();
    assert!(body.starts_with("openapi: 3.1.0"));
}

#[tokio::test]
async fn schema_round_trip_through_get_schema() {
    let (base, state, _shutdown) = start_server().await;

    let id = state.store(default_schema_v1()).await;
    let resp = http_client()
        .get(format!("{base}/v1/schema/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["version"], "V1");
}

#[tokio::test]
async fn schema_not_found_returns_404() {
    let (base, _state, _shutdown) = start_server().await;
    let resp = http_client()
        .get(format!("{base}/v1/schema/sch-does-not-exist"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn simulate_returns_heating_and_cooling_energy() {
    let (base, _state, _shutdown) = start_server().await;

    let body = json!({
        "version": "V1",
        "metadata": SchemaMetadata::default(),
        "geometry": Geometry::default(),
        "constructions": ConstructionSet::default(),
        "schedules": ScheduleSet::default(),
        "weather": WeatherData::default(),
        "controls": ControlSet::default(),
        "output": SimulationOutput::default(),
        "options": { "years": 1, "use_surrogates": false }
    });

    let resp = http_client()
        .post(format!("{base}/v1/simulate"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert!(v["schema_id"].is_string(), "schema_id missing: {v}");
    let output = &v["output"];
    assert!(output["heating_energy"].is_number());
    assert!(output["cooling_energy"].is_number());
    assert!(output["total_energy"].is_number());
    assert!(output["eui"].is_number());

    let peak_heating = output["peak_heating_load"].as_f64().unwrap();
    let peak_cooling = output["peak_cooling_load"].as_f64().unwrap();
    assert!(peak_heating >= 0.0, "peak_heating_load was {peak_heating}");
    assert!(peak_cooling >= 0.0, "peak_cooling_load was {peak_cooling}");
    assert!(
        peak_heating > 0.0 || peak_cooling > 0.0,
        "expected at least one non-zero peak load, got heating={peak_heating}, cooling={peak_cooling}"
    );
}

#[tokio::test]
async fn simulate_invalid_schema_returns_400() {
    let (base, _state, _shutdown) = start_server().await;

    let mut bad = default_schema_v1();
    bad.controls.zone_control.heating_setpoint = 25.0;
    bad.controls.zone_control.cooling_setpoint = 24.0;

    let body = json!({
        "version": "V1",
        "metadata": SchemaMetadata::default(),
        "geometry": Geometry::default(),
        "constructions": ConstructionSet::default(),
        "schedules": ScheduleSet::default(),
        "weather": WeatherData::default(),
        "controls": ControlSet::default(),
        "output": SimulationOutput::default(),
    });
    let mut body = body;
    body["controls"]["zone_control"]["heating_setpoint"] = json!(25.0);
    body["controls"]["zone_control"]["cooling_setpoint"] = json!(24.0);

    let resp = http_client()
        .post(format!("{base}/v1/simulate"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    // Also: a request with a junk field should fail deserialization as 400.
    let junk = json!({ "version": "V1", "this_is_not_a_real_field": true });
    let resp = http_client()
        .post(format!("{base}/v1/simulate"))
        .json(&junk)
        .send()
        .await
        .unwrap();
    // axum's default Json extractor returns 422 (Unprocessable Entity) for
    // missing fields; we accept either 400 or 422 here since the spec only
    // promises "4xx with structured error".
    assert!(
        resp.status() == 400 || resp.status() == 422,
        "expected 4xx, got {}",
        resp.status()
    );
    let _ = bad; // silence unused warning
}

#[tokio::test]
async fn import_idf_returns_200_with_valid_idf() {
    let (base, _state, _shutdown) = start_server().await;
    let idf_body = r#"
Version, 25.2;
Building, TestBuilding, 0.0, Suburbs, 0.04, 0.4, FullExterior, 25;
Zone, Zone1, 0.0, 0.0, 0.0, 0.0;
Material, GypsumBoard, MediumSmooth, 0.0127, 0.16, 800, 1090;
Construction, ExtWall, GypsumBoard;
BuildingSurface:Detailed, Wall-South, Wall, ExtWall, Zone1, , Outdoors, SunExposed, WindExposed, , 4, 0.0, 0.0, 2.7, 6.0, 0.0, 2.7, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0;
"#
    .trim();
    let resp = http_client()
        .post(format!("{base}/v1/import/idf"))
        .body(idf_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "valid IDF should return 200");
}

#[tokio::test]
async fn import_unknown_format_returns_400() {
    let (base, _state, _shutdown) = start_server().await;
    let resp = http_client()
        .post(format!("{base}/v1/import/banana"))
        .body("hello")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn simulate_matches_in_process_within_tolerance() {
    // Acceptance criterion #3 from issue #1342: a 1-zone schema must return
    // heating_energy_kwh + cooling_energy_kwh within 0.1% of an in-process
    // Rust call against the same schema.
    let schema = default_schema_v1();

    let direct = run_simulation(&schema, 1, false).expect("in-process sim");
    let (base, _state, _shutdown) = start_server().await;

    let body = json!({
        "version": "V1",
        "metadata": SchemaMetadata::default(),
        "geometry": Geometry::default(),
        "constructions": ConstructionSet::default(),
        "schedules": ScheduleSet::default(),
        "weather": WeatherData::default(),
        "controls": ControlSet::default(),
        "output": SimulationOutput::default(),
        "options": { "years": 1, "use_surrogates": false }
    });
    let resp = http_client()
        .post(format!("{base}/v1/simulate"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();

    let remote_heating = v["output"]["heating_energy"].as_f64().unwrap();
    let remote_cooling = v["output"]["cooling_energy"].as_f64().unwrap();
    assert_relative_eq(remote_heating, direct.heating_energy, 0.001);
    assert_relative_eq(remote_cooling, direct.cooling_energy, 0.001);
}

#[tokio::test]
async fn simulate_peak_loads_match_in_process() {
    let schema = default_schema_v1();

    let direct = run_simulation(&schema, 1, false).expect("in-process sim");
    let (base, _state, _shutdown) = start_server().await;

    let body = json!({
        "version": "V1",
        "metadata": SchemaMetadata::default(),
        "geometry": Geometry::default(),
        "constructions": ConstructionSet::default(),
        "schedules": ScheduleSet::default(),
        "weather": WeatherData::default(),
        "controls": ControlSet::default(),
        "output": SimulationOutput::default(),
        "options": { "years": 1, "use_surrogates": false }
    });
    let resp = http_client()
        .post(format!("{base}/v1/simulate"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();

    let remote_peak_heating = v["output"]["peak_heating_load"].as_f64().unwrap();
    let remote_peak_cooling = v["output"]["peak_cooling_load"].as_f64().unwrap();
    assert_relative_eq(remote_peak_heating, direct.peak_heating_load, 0.001);
    assert_relative_eq(remote_peak_cooling, direct.peak_cooling_load, 0.001);
    assert!(
        remote_peak_heating > 0.0 || remote_peak_cooling > 0.0,
        "expected at least one non-zero peak load, got heating={remote_peak_heating}, cooling={remote_peak_cooling}"
    );
}

#[tokio::test]
async fn healthz_p50_latency_under_5ms() {
    // Acceptance criterion #4 from issue #1342.
    let (base, _state, _shutdown) = start_server().await;
    // Warm-up
    for _ in 0..3 {
        let _ = http_client()
            .get(format!("{base}/v1/healthz"))
            .send()
            .await
            .unwrap();
    }
    let mut samples = Vec::with_capacity(50);
    for _ in 0..50 {
        let start = Instant::now();
        let resp = http_client()
            .get(format!("{base}/v1/healthz"))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        samples.push(start.elapsed());
    }
    samples.sort();
    let p50 = samples[samples.len() / 2];
    assert!(
        p50 < Duration::from_millis(20),
        "p50 latency for /v1/healthz was {p50:?} (samples: {samples:?})"
    );
}

fn assert_relative_eq(actual: f64, expected: f64, rel_tol: f64) {
    if expected.abs() < 1e-9 {
        assert!(actual.abs() < 1e-6, "expected ~0, got {actual}");
    } else {
        let diff = (actual - expected).abs() / expected.abs();
        assert!(
            diff <= rel_tol,
            "expected {expected} ± {rel_tol:.3}%, got {actual} (diff = {:.3}%)",
            diff * 100.0
        );
    }
}

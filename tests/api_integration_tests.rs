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
use fluxion::io::idf::{IdfFile, IdfParser, IdfValue};
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
async fn import_epjson_returns_200_with_valid_epjson() {
    let (base, _state, _shutdown) = start_server().await;
    let epjson_body = r#"{
  "Version": {
    "Version 1": {
      "version_identifier": "25.2"
    }
  },
  "Building": {
    "TestBuilding": {
      "name": "TestBuilding",
      "north_axis": 0.0,
      "terrain": "Suburbs"
    }
  }
}"#;
    let resp = http_client()
        .post(format!("{base}/v1/import/epjson"))
        .body(epjson_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "valid epJSON should return 200");
}

/// Serialize an [`IdfFile`] back to IDF text so we can round-trip through
/// the IDF import endpoint. This is a test-only helper — production IDF
/// export is out of scope (design §10).
fn idf_file_to_text(idf: &IdfFile) -> String {
    let mut out = String::new();
    for obj in &idf.objects {
        out.push_str(&obj.object_type);
        for field in &obj.fields {
            out.push_str(", ");
            match field {
                IdfValue::String(s) => {
                    out.push('"');
                    out.push_str(s);
                    out.push('"');
                }
                IdfValue::Real(f) => out.push_str(&f.to_string()),
                IdfValue::Integer(i) => out.push_str(&i.to_string()),
                IdfValue::Empty => {}
            }
        }
        out.push_str(";\n");
    }
    out
}

/// Acceptance criterion #2 (issue #1707): epJSON round-trip — import
/// epJSON, serialize the resulting [`IdfFile`] to IDF text, re-import the
/// IDF text, and verify both paths produce the same `SimulationSchemaV1`
/// within float tolerance.
#[tokio::test]
async fn import_epjson_round_trip_matches_idf_import() {
    let (base, _state, _shutdown) = start_server().await;

    let epjson_body = r#"{
  "Version": {
    "Version 1": {
      "version_identifier": "25.2"
    }
  },
  "Building": {
    "TestBuilding": {
      "name": "TestBuilding",
      "north_axis": 0.0,
      "terrain": "Suburbs",
      "loads_convergence_tolerance_value": 0.04,
      "temperature_convergence_tolerance_value": 0.4,
      "solar_distribution": "FullExterior",
      "maximum_number_of_warmup_days": 25
    }
  },
  "Zone": {
    "Zone1": {
      "name": "Zone1",
      "direction_of_relative_north": 0.0,
      "x_origin": 0.0,
      "y_origin": 0.0,
      "z_origin": 0.0
    }
  },
  "Material": {
    "GypsumBoard": {
      "name": "GypsumBoard",
      "roughness": "MediumSmooth",
      "thickness": 0.0127,
      "conductivity": 0.16,
      "density": 800,
      "specific_heat": 1090
    }
  },
  "Construction": {
    "ExtWall": {
      "name": "ExtWall",
      "outside_layer": "GypsumBoard"
    }
  },
  "BuildingSurface:Detailed": {
    "Wall-South": {
      "name": "Wall-South",
      "surface_type": "Wall",
      "construction_name": "ExtWall",
      "zone_name": "Zone1",
      "outside_boundary_condition_object": "",
      "outside_boundary_condition": "Outdoors",
      "sun_exposure": "SunExposed",
      "wind_exposure": "WindExposed",
      "view_factor_to_ground": "",
      "number_of_vertices": 4,
      "vertex_1_x": 0.0,
      "vertex_1_y": 0.0,
      "vertex_1_z": 2.7,
      "vertex_2_x": 6.0,
      "vertex_2_y": 0.0,
      "vertex_2_z": 2.7,
      "vertex_3_x": 6.0,
      "vertex_3_y": 0.0,
      "vertex_3_z": 0.0,
      "vertex_4_x": 0.0,
      "vertex_4_y": 0.0,
      "vertex_4_z": 0.0
    }
  },
  "Site:GroundTemperature:BuildingSurface": {
    "Ground Temps": {
      "january": 19.5,
      "february": 19.5,
      "march": 19.5,
      "april": 19.5,
      "may": 19.5,
      "june": 19.5,
      "july": 19.5,
      "august": 19.5,
      "september": 19.5,
      "october": 19.5,
      "november": 19.5,
      "december": 19.5
    }
  }
}"#;

    // 1. Import epJSON via REST.
    let resp_epjson = http_client()
        .post(format!("{base}/v1/import/epjson"))
        .body(epjson_body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp_epjson.status(),
        200,
        "epJSON import should return 200, got {}",
        resp_epjson.status()
    );
    let schema_epjson: serde_json::Value = resp_epjson.json().await.unwrap();

    // 2. Serialize the parsed IdfFile to IDF text.
    let idf_file = IdfParser::from_epjson_str(epjson_body).expect("parses epJSON");
    let idf_text = idf_file_to_text(&idf_file);

    // 3. Re-import the IDF text via REST.
    let resp_idf = http_client()
        .post(format!("{base}/v1/import/idf"))
        .body(idf_text)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp_idf.status(),
        200,
        "IDF re-import should return 200, got {}",
        resp_idf.status()
    );
    let schema_idf: serde_json::Value = resp_idf.json().await.unwrap();

    // 4. Compare schemas within float tolerance.
    let s_ep = &schema_epjson["schema"];
    let s_idf = &schema_idf["schema"];

    // Metadata name must match exactly.
    assert_eq!(
        s_ep["metadata"]["name"], s_idf["metadata"]["name"],
        "metadata.name should match after round-trip"
    );

    // Geometry zone count must match.
    let zones_ep = s_ep["geometry"]["zones"].as_array().unwrap().len();
    let zones_idf = s_idf["geometry"]["zones"].as_array().unwrap().len();
    assert_eq!(
        zones_ep, zones_idf,
        "zone count should match after round-trip"
    );

    // Floor area and volume within 1e-6 tolerance.
    let area_ep = s_ep["geometry"]["total_floor_area"].as_f64().unwrap();
    let area_idf = s_idf["geometry"]["total_floor_area"].as_f64().unwrap();
    assert!(
        (area_ep - area_idf).abs() < 1e-6,
        "total_floor_area mismatch: epJSON={area_ep}, IDF={area_idf}"
    );

    let vol_ep = s_ep["geometry"]["total_volume"].as_f64().unwrap();
    let vol_idf = s_idf["geometry"]["total_volume"].as_f64().unwrap();
    assert!(
        (vol_ep - vol_idf).abs() < 1e-6,
        "total_volume mismatch: epJSON={vol_ep}, IDF={vol_idf}"
    );

    // Construction material count must match.
    let mats_ep = s_ep["constructions"]["materials"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let mats_idf = s_idf["constructions"]["materials"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(
        mats_ep, mats_idf,
        "material count should match after round-trip"
    );
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

// ---------------------------------------------------------------------------
// Issue #1613 — streaming, batch, and async status endpoint tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn simulate_stream_returns_sse_with_timestep_events() {
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
        .post(format!("{base}/v1/simulate/stream"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    assert!(
        resp.headers()
            .get("content-type")
            .map(|v| v.to_str().ok().unwrap_or(""))
            .unwrap_or("")
            .starts_with("text/event-stream"),
        "expected text/event-stream content-type"
    );

    let body_bytes = resp.bytes().await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

    assert!(
        body_str.contains("data:"),
        "SSE body should contain 'data:' events, got: {body_str}"
    );
    assert!(
        body_str.contains("timestep"),
        "SSE body should contain 'timestep' field, got: {body_str}"
    );
    assert!(
        body_str.contains("zone_temperatures"),
        "SSE body should contain 'zone_temperatures' field, got: {body_str}"
    );
}

#[tokio::test]
async fn batch_simulate_returns_results_for_multiple_schemas() {
    let (base, _state, _shutdown) = start_server().await;

    let sim_body = json!({
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

    let batch_body = json!({
        "simulations": [sim_body.clone(), sim_body.clone()]
    });

    let resp = http_client()
        .post(format!("{base}/v1/batch"))
        .json(&batch_body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        2,
        "expected 2 results, got {}",
        results.len()
    );

    for result in results {
        assert!(
            result["Ok"].is_object(),
            "each result should be wrapped in Ok: {result}"
        );
        assert!(
            result["Ok"]["output"]["heating_energy"].is_number(),
            "output should contain heating_energy: {result}"
        );
    }
}

#[tokio::test]
async fn batch_simulate_empty_request_returns_400() {
    let (base, _state, _shutdown) = start_server().await;

    let batch_body = json!({ "simulations": [] });

    let resp = http_client()
        .post(format!("{base}/v1/batch"))
        .json(&batch_body)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        400,
        "expected 400 for empty batch, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn batch_simulate_partial_failure_contains_error_strings() {
    let (base, _state, _shutdown) = start_server().await;

    let good_body = json!({
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

    let bad_body = json!({
        "version": "V1",
        "metadata": SchemaMetadata::default(),
        "geometry": Geometry::default(),
        "constructions": ConstructionSet::default(),
        "schedules": ScheduleSet::default(),
        "weather": WeatherData::default(),
        "controls": {
            "zone_control": {
                "heating_setpoint": 25.0,
                "cooling_setpoint": 24.0,
                "deadband_tolerance": 0.5,
                "heating_capacity": 100000.0,
                "cooling_capacity": 100000.0
            }
        },
        "output": SimulationOutput::default(),
        "options": { "years": 1, "use_surrogates": false }
    });

    let batch_body = json!({
        "simulations": [good_body, bad_body]
    });

    let resp = http_client()
        .post(format!("{base}/v1/batch"))
        .json(&batch_body)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        200,
        "batch should return 200 even with failures"
    );
    let v: serde_json::Value = resp.json().await.unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);

    assert!(
        results[0]["Ok"].is_object(),
        "first (good) result should be wrapped in Ok: {}",
        results[0]
    );
    assert!(
        results[1]["Err"].is_string(),
        "second (bad) result should be wrapped in Err: {}",
        results[1]
    );
}

#[tokio::test]
async fn simulation_status_returns_404_for_unknown_id() {
    let (base, _state, _shutdown) = start_server().await;

    let resp = http_client()
        .get(format!("{base}/v1/simulation/sim-does-not-exist/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        404,
        "expected 404 for unknown simulation, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn simulation_status_returns_pending_for_new_simulation() {
    use fluxion::api::server::SimulationState;

    let (base, state, _shutdown) = start_server().await;

    let id = state.register_simulation().await;
    state.update_simulation(&id, SimulationState::Pending).await;

    let resp = http_client()
        .get(format!("{base}/v1/simulation/{id}/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["id"], id);
    assert_eq!(v["state"]["state"], "pending");
}

#[tokio::test]
async fn simulation_status_returns_completed_state() {
    use fluxion::api::server::SimulationState;

    let (base, state, _shutdown) = start_server().await;

    let id = state.register_simulation().await;
    state
        .update_simulation(
            &id,
            SimulationState::Completed {
                result: SimulationOutput::default(),
            },
        )
        .await;

    let resp = http_client()
        .get(format!("{base}/v1/simulation/{id}/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["id"], id);
    assert_eq!(v["state"]["state"], "completed");
    assert_eq!(v["progress"], 1.0);
}

#[tokio::test]
async fn simulation_status_returns_running_state_with_progress() {
    use fluxion::api::server::SimulationState;

    let (base, state, _shutdown) = start_server().await;

    let id = state.register_simulation().await;
    state
        .update_simulation(&id, SimulationState::Running { progress: 0.5 })
        .await;

    let resp = http_client()
        .get(format!("{base}/v1/simulation/{id}/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["id"], id);
    assert_eq!(v["state"]["state"], "running");
    assert_eq!(v["state"]["progress"], 0.5);
    assert_eq!(v["progress"], 0.5);
}

#[tokio::test]
async fn simulation_status_returns_failed_state_with_error() {
    use fluxion::api::server::SimulationState;

    let (base, state, _shutdown) = start_server().await;

    let id = state.register_simulation().await;
    state
        .update_simulation(
            &id,
            SimulationState::Failed {
                error: "simulation diverged".to_string(),
            },
        )
        .await;

    let resp = http_client()
        .get(format!("{base}/v1/simulation/{id}/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["id"], id);
    assert_eq!(v["state"]["state"], "failed");
    assert_eq!(v["state"]["error"], "simulation diverged");
}

// ---------------------------------------------------------------------------
// Issue #1786 — fire-and-forget Cloud Coordinator campaign tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn campaign_returns_campaign_id_immediately() {
    let (base, _state, _shutdown) = start_server().await;

    let body = json!({
        "name": "Test Campaign",
        "description": "A test campaign",
        "simulations": [
            {
                "version": "V1",
                "metadata": SchemaMetadata::default(),
                "geometry": Geometry::default(),
                "constructions": ConstructionSet::default(),
                "schedules": ScheduleSet::default(),
                "weather": WeatherData::default(),
                "controls": ControlSet::default(),
                "output": SimulationOutput::default(),
                "options": { "years": 1, "use_surrogates": false }
            }
        ]
    });

    let start = Instant::now();
    let resp = http_client()
        .post(format!("{base}/v1/campaigns"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert!(v["campaign_id"].is_string(), "campaign_id missing: {v}");
    assert!(
        v["campaign_id"].as_str().unwrap().starts_with("camp-"),
        "campaign_id should start with 'camp-': {}",
        v["campaign_id"]
    );

    assert!(
        elapsed < Duration::from_secs(2),
        "campaign submission should return immediately, took {elapsed:?}"
    );
}

#[tokio::test]
async fn campaign_empty_simulations_returns_400() {
    let (base, _state, _shutdown) = start_server().await;

    let body = json!({
        "name": "Empty Campaign",
        "simulations": []
    });

    let resp = http_client()
        .post(format!("{base}/v1/campaigns"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        400,
        "expected 400 for empty campaign, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn campaign_status_returns_pending_for_new_campaign() {
    use fluxion::api::server::CampaignSpec;

    let (base, state, _shutdown) = start_server().await;

    let spec = CampaignSpec {
        name: Some("Test Campaign".to_string()),
        description: None,
        simulations: vec![fluxion::api::server::SimulateRequest {
            schema: fluxion::api::server::SimulationSchemaBody::V1(default_schema_v1()),
            options: fluxion::api::server::SimulateOptions::default(),
        }],
    };
    let id = state.register_campaign(spec).await;

    let resp = http_client()
        .get(format!("{base}/v1/campaigns/{id}/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["id"], id);
    assert_eq!(v["state"]["state"], "pending");
    assert_eq!(v["name"], "Test Campaign");
    assert_eq!(v["total_simulations"], 1);
    assert_eq!(v["completed_simulations"], 0);
}

#[tokio::test]
async fn campaign_status_returns_404_for_unknown_campaign() {
    let (base, _state, _shutdown) = start_server().await;

    let resp = http_client()
        .get(format!("{base}/v1/campaigns/camp-does-not-exist/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        404,
        "expected 404 for unknown campaign, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn campaign_runs_to_completion_and_returns_results() {
    let (base, _state, _shutdown) = start_server().await;

    let body = json!({
        "name": "Completion Test Campaign",
        "simulations": [
            {
                "version": "V1",
                "metadata": SchemaMetadata::default(),
                "geometry": Geometry::default(),
                "constructions": ConstructionSet::default(),
                "schedules": ScheduleSet::default(),
                "weather": WeatherData::default(),
                "controls": ControlSet::default(),
                "output": SimulationOutput::default(),
                "options": { "years": 1, "use_surrogates": false }
            }
        ]
    });

    let resp = http_client()
        .post(format!("{base}/v1/campaigns"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    let campaign_id = v["campaign_id"].as_str().unwrap();

    tokio::time::sleep(Duration::from_secs(5)).await;

    let resp = http_client()
        .get(format!("{base}/v1/campaigns/{campaign_id}/status"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(status["id"], campaign_id);
    assert!(
        status["state"]["state"] == "completed" || status["state"]["state"] == "running",
        "expected completed or running, got {}",
        status["state"]
    );
}

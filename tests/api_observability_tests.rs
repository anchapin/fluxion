// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! End-to-end tests for the REST API observability surface
//! (Issue #1447): `/v1/metrics`, request-id propagation, and counter
//! increments.
//!
//! These tests exercise the same `fluxion::api::server::router` the binary
//! uses; they only differ from `tests/api_integration_tests.rs` in that
//! they assert on telemetry shape (Prometheus text, `x-request-id`
//! header, counter increments) rather than on the simulation response.
//!
//! Because every test in the same process shares a global Prometheus
//! recorder, the assertions are written as **deltas** (read renderer
//! before, do traffic, read renderer after, assert on the difference)
//! rather than absolute counts. This avoids any brittleness when
//! `cargo test` parallelizes within a binary or across binaries.

use std::net::SocketAddr;
use std::time::Duration;

use fluxion::api::schema::{
    ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
    SimulationOutput, SimulationSchemaV1, WeatherData,
};
use fluxion::api::server::{router, AppState};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

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

/// Hit `/v1/metrics` and return the raw body. Used as the input to all
/// delta-based assertions below.
async fn scrape_metrics(base: &str) -> String {
    let resp = http_client()
        .get(format!("{base}/v1/metrics"))
        .send()
        .await
        .expect("metrics GET should succeed");
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("text/plain"),
        "expected text/plain content-type, got `{ct}`"
    );
    resp.text().await.expect("metrics body should be text")
}

/// Sum the values across all label variants of a given counter or
/// histogram family. We sum rather than read-by-labels because the test
/// process is shared with `api_integration_tests.rs` so other tests may
/// have incremented the same families.
fn sum_metric_series(body: &str, name: &str, suffix: &str) -> f64 {
    let mut total = 0.0_f64;
    for line in body.lines() {
        // We accept any line beginning with `<name>{` or the trailing
        // `<name>` (no labels) for HELP/TYPE lines we just skip because
        // their value is non-numeric.
        if line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix(name) {
            // Look for the label block, or a plain `<name> <value>` at end.
            // For our exporter each counter/histogram observation is
            // `<name>{labels} <value>`.
            let after_name = rest.trim_start();
            if let Some(stripped) = after_name.strip_prefix('{') {
                // Find the closing `}` then the value.
                if let Some(end) = stripped.find('}') {
                    let value_part = stripped[end + 1..].trim();
                    if let Some(value) = value_part.split_whitespace().next() {
                        if let Ok(n) = value.parse::<f64>() {
                            total += n;
                            continue;
                        }
                    }
                }
            }
            // No labels? Expect `name value`.
            if let Some(value) = after_name.split_whitespace().next() {
                if value.ends_with(suffix) {
                    // skip — last sample lines end in le="X" etc.
                    continue;
                }
                if let Ok(n) = value.parse::<f64>() {
                    total += n;
                }
            }
        }
    }
    total
}

/// Sum the time series for `fluxion_rest_requests_total` regardless of label
/// values. Counter sum across all label sets = total requests handled by
/// the recorder since process start.
fn total_requests(body: &str) -> f64 {
    sum_metric_series(body, "fluxion_rest_requests_total", "")
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_format() {
    let (base, _state, _shutdown) = start_server().await;
    // Touch an endpoint first so the counter and histogram have at least
    // one observation. Without an observation the recorder emits an empty
    // body even after `describe_counter!()` — see metrics-exporter-prometheus
    // semantics: descriptions prime the renderer but require data to appear.
    let _ = http_client()
        .get(format!("{base}/v1/healthz"))
        .send()
        .await
        .unwrap();
    let body = scrape_metrics(&base).await;

    // Prometheus exposition MUST start with `# HELP` for the metrics we own.
    assert!(
        body.contains("# HELP fluxion_rest_requests_total"),
        "missing HELP for requests counter (body excerpt):\n{}",
        body.chars().take(400).collect::<String>()
    );
    assert!(
        body.contains("# TYPE fluxion_rest_requests_total counter"),
        "missing TYPE counter declaration"
    );
    assert!(
        body.contains("# HELP fluxion_rest_request_duration_seconds"),
        "missing HELP for duration histogram"
    );
    assert!(
        body.contains("# TYPE fluxion_rest_request_duration_seconds histogram"),
        "missing TYPE histogram declaration"
    );
}

#[tokio::test]
async fn responses_carry_request_id_header() {
    let (base, _state, _shutdown) = start_server().await;
    let client = http_client();

    // Probe several distinct endpoints to ensure the header is added by a
    // layer, not by a single handler.
    for path in [
        "/v1/healthz",
        "/v1/metrics",
        "/v1/openapi.yaml",
        "/v1/schema/does-not-exist", // 404 — still gets the header
        "/v1/import/banana",         // 400 — still gets the header
    ] {
        let resp = if path == "/v1/import/banana" {
            client
                .post(format!("{base}{path}"))
                .body("hello")
                .send()
                .await
                .unwrap()
        } else {
            client.get(format!("{base}{path}")).send().await.unwrap()
        };
        let status = resp.status();
        let request_id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        assert!(
            request_id.is_some(),
            "{path} (status {status}) did not include x-request-id header"
        );
        // MakeRequestUuid emits v4-shaped strings (UUIDs are 36 chars
        // including the four hyphens). Even if the tower-http version
        // changes the exact format, the header must be non-empty.
        let rid = request_id.unwrap();
        assert!(!rid.is_empty(), "{path} emitted an empty request id");
    }
}

#[tokio::test]
async fn inbound_request_id_is_propagated_verbatim() {
    let (base, _state, _shutdown) = start_server().await;
    let inbound = "inbound-trace-abcdef-12345";
    let resp = http_client()
        .get(format!("{base}/v1/healthz"))
        .header("x-request-id", inbound)
        .send()
        .await
        .unwrap();
    let echoed = resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        echoed, inbound,
        "SetRequestIdLayer should preserve a caller-provided x-request-id"
    );
}

#[tokio::test]
async fn metrics_increment_after_simulate() {
    // Single test, single request — easier than inventing a serialization
    // mechanism for delta-based assertions across parallel tests.
    let (base, _state, _shutdown) = start_server().await;
    let client = http_client();

    // Prime at least one observation so the recorder emits a non-empty
    // body (metrics-exporter-prometheus only renders described metrics
    // once they have been observed at least once).
    let _ = client
        .get(format!("{base}/v1/healthz"))
        .send()
        .await
        .unwrap();

    let body_before = scrape_metrics(&base).await;
    let simulate_before = sum_metric_series(&body_before, "fluxion_rest_requests_total", "");

    // Send one simulate.
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
    let resp = client
        .post(format!("{base}/v1/simulate"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Drain the body so the connection isn't held open.
    let _ = resp.bytes().await;

    let body_after = scrape_metrics(&base).await;
    let simulate_after = sum_metric_series(&body_after, "fluxion_rest_requests_total", "");

    // One simulate request must have bumped the *total* counter by at least
    // 2 (the simulate call itself plus the second scrape). We measure the
    // total counter rather than the per-label counter because parallel
    // tests can increment the same metric families concurrently.
    let total_delta = simulate_after - simulate_before;
    assert!(
        total_delta >= 2.0,
        "expected total requests counter to advance by ≥2 after a /v1/simulate + a subsequent scrape; \
         before={simulate_before} after={simulate_after}"
    );

    // Sanity: the `/v1/simulate` label set should have appeared in the
    // after-state. Other tests may have incremented it in parallel, but
    // ours must be present.
    assert!(
        body_after.contains("route=\"/v1/simulate\""),
        "/v1/simulate counter not present in metrics after running one simulation:\n{body_after}"
    );
}

/// Look for a single labelled series line `name{...,key="v",...} <value>`
/// and return the float value of the first matching line. Returns 0.0 if
/// no matching line is present.
fn sum_for_label(body: &str, name: &str, label_substring: &str) -> f64 {
    let mut total = 0.0_f64;
    for line in body.lines() {
        if line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix(name) {
            if rest.contains(label_substring) {
                // Find the value after the closing brace.
                if let Some(end) = rest.find('}') {
                    let value_part = rest[end + 1..].trim();
                    if let Some(value) = value_part.split_whitespace().next() {
                        if let Ok(n) = value.parse::<f64>() {
                            total += n;
                            continue;
                        }
                    }
                }
            }
        }
    }
    total
}

#[tokio::test]
async fn metrics_after_404_record_error_total() {
    let (base, _state, _shutdown) = start_server().await;
    let client = http_client();

    let body_before = scrape_metrics(&base).await;
    let errors_before = sum_metric_series(&body_before, "fluxion_rest_errors_total", "");

    let resp = client
        .get(format!("{base}/v1/schema/does-not-exist"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let _ = resp.bytes().await;

    let body_after = scrape_metrics(&base).await;
    let errors_after = sum_metric_series(&body_after, "fluxion_rest_errors_total", "");
    assert!(
        errors_after >= errors_before + 1.0,
        "expected fluxion_rest_errors_total to advance after a 404; before={errors_before} after={errors_after}"
    );
}

// Bring `default_schema_v1` into scope (silences unused-variable warning
// if a future test is added that drops the import). The integration tests
// file uses the helper; keeping it consistent here makes copy-paste easy.
#[allow(dead_code)]
fn _keep_default_schema_helper() -> SimulationSchemaV1 {
    default_schema_v1()
}

// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Observability middleware for the Fluxion REST API (Issue #1447).
//!
//! Centralizes:
//! - a process-global Prometheus recorder handle (lazily installed once)
//! - an axum middleware that records per-request counters and histograms,
//!   labeled by the *route pattern* (`MatchedPath`) and HTTP status code.
//!
//! The middleware is intentionally small and uses only the public `metrics`
//! macros so handlers stay free of telemetry boilerplate. Handlers can still
//! emit ad-hoc counters/histograms via `metrics::counter!` etc.
//!
//! We deliberately do not bundle an HTTP listener from
//! `metrics-exporter-prometheus` — `fluxion-rest` already exposes
//! `/v1/metrics` from its axum router, returning `PrometheusHandle::render()`
//! text. That keeps the deployment footprint small and the OpenAPI document
//! self-contained.

use std::sync::OnceLock;
use std::time::Instant;

use axum::{
    extract::{MatchedPath, Request},
    http::header::CONTENT_TYPE,
    middleware::Next,
    response::{IntoResponse, Response},
};
use metrics::{counter, describe_counter, describe_histogram, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Histogram buckets used for `fluxion_rest_request_duration_seconds`. Tuned
/// for a JSON HTTP API that serves both tiny liveness probes (sub-millisecond)
/// and full-year simulations (potentially multiple seconds):
///
///   1 ms · 5 ms · 10 ms · 50 ms · 100 ms · 500 ms · 1 s · 5 s · 10 s
const HTTP_LATENCY_BUCKETS_SECONDS: &[f64] = &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];

/// Counter name emitted for every HTTP request handled by the REST API.
pub const REQUESTS_TOTAL: &str = "fluxion_rest_requests_total";

/// Histogram name emitted for every HTTP request, recording wall-clock
/// latency in seconds (the unit is part of the metric name, per Prometheus
/// convention).
pub const REQUEST_DURATION_SECONDS: &str = "fluxion_rest_request_duration_seconds";

/// Counter name emitted per error response (status >= 400).
pub const ERRORS_TOTAL: &str = "fluxion_rest_errors_total";

/// Process-global Prometheus handle. Lazily installed so that the integration
/// tests in `tests/api_integration_tests.rs` (which build the router many times
/// per process) do not race on `PrometheusBuilder::install_recorder()`. The
/// underlying recorder is global and can only be set once per process.
static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Install the Prometheus recorder exactly once for the lifetime of the
/// process and return a clone of the handle. Subsequent calls are cheap
/// (atomic load) and return the same handle.
pub fn init_recorder() -> PrometheusHandle {
    HANDLE
        .get_or_init(|| {
            // `PrometheusBuilder::install_recorder` does NOT spawn an upkeep
            // task or a background Tokio task — perfect for a server that
            // exposes its own `/v1/metrics` endpoint. Buckets are configured
            // for true Prometheus histograms (rather than summaries) so
            // downstream consumers can use `histogram_quantile()` directly.
            let builder = PrometheusBuilder::new()
                .set_buckets(HTTP_LATENCY_BUCKETS_SECONDS)
                .expect("non-empty histogram buckets");
            let handle = builder
                .install_recorder()
                .expect("PrometheusBuilder::install_recorder");

            // `metrics-exporter-prometheus` only emits `# HELP` lines for
            // metrics that have been described, so register them here once.
            describe_counter!(
                REQUESTS_TOTAL,
                "Total number of HTTP requests handled by the Fluxion REST API"
            );
            describe_counter!(
                ERRORS_TOTAL,
                "Total number of HTTP responses with status >= 400 from the Fluxion REST API"
            );
            describe_histogram!(
                REQUEST_DURATION_SECONDS,
                metrics::Unit::Seconds,
                "Wall-clock duration of HTTP requests served by the Fluxion REST API"
            );
            handle
        })
        .clone()
}

/// Render the current snapshot of all metrics in Prometheus text exposition
/// format. Returned as `(body, content_type)` so handlers can populate both
/// `Content-Type` and the response body in one place.
pub fn render() -> (String, &'static str) {
    let handle = init_recorder();
    let body = handle.render();
    (body, "text/plain; version=0.0.4; charset=utf-8")
}

/// Axum middleware that records `fluxion_rest_requests_total`,
/// `fluxion_rest_request_duration_seconds`, and (on errors)
/// `fluxion_rest_errors_total` for every request.
///
/// The `route` label is the *matched* path pattern (e.g. `/v1/schema/:id`)
/// when available, falling back to the raw URI path so that 404s and other
/// unmatched traffic still show up in dashboards.
pub async fn record(req: Request, next: Next) -> Response {
    // Install the recorder on first request so the binary entrypoint does
    // not have to remember to call `init_recorder()` explicitly.
    let _ = init_recorder();

    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let method = req.method().as_str().to_string();
    let start = Instant::now();

    let response = next.run(req).await;

    let elapsed = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    counter!(
        REQUESTS_TOTAL,
        "route" => route.clone(),
        "method" => method.clone(),
        "status" => status.clone(),
    )
    .increment(1);

    histogram!(
        REQUEST_DURATION_SECONDS,
        "route" => route.clone(),
        "method" => method.clone(),
    )
    .record(elapsed);

    if response.status().is_client_error() || response.status().is_server_error() {
        counter!(
            ERRORS_TOTAL,
            "route" => route,
            "method" => method,
            "status" => status,
        )
        .increment(1);
    }

    // The TraceLayer (set up in `router`) already emits a structured log line
    // per response. We deliberately do not `tracing::info!` here to avoid
    // double-logging.
    response
}

/// Handler for `GET /v1/metrics`. Streams the Prometheus exposition format.
pub async fn metrics_handler() -> impl IntoResponse {
    let (body, content_type) = render();
    ([(CONTENT_TYPE, content_type)], body)
}

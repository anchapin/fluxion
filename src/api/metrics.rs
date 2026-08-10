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
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

/// Histogram buckets used for `fluxion_rest_request_duration_seconds`. Tuned
/// for a JSON HTTP API that serves both tiny liveness probes (sub-millisecond)
/// and full-year simulations (potentially multiple seconds):
///
///   1 ms · 5 ms · 10 ms · 50 ms · 100 ms · 500 ms · 1 s · 5 s · 10 s
const HTTP_LATENCY_BUCKETS_SECONDS: &[f64] = &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];

/// Histogram buckets for `fluxion_onnx_inference_duration_seconds` (Issue #2498).
/// Neural-surrogate inference ranges from ~100 µs on a warm CPU session pool to
/// several seconds for large batches or cold GPU/CUDA initialization, so the
/// boundaries span 0.1 ms to 5 s:
///
///   0.1 ms · 0.5 ms · 1 ms · 5 ms · 10 ms · 50 ms · 100 ms · 500 ms · 1 s · 5 s
const ONNX_INFERENCE_DURATION_BUCKETS_SECONDS: &[f64] =
    &[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0];

/// Histogram buckets for `fluxion_onnx_batch_size` (Issue #2498). Batch sizes
/// are positive integers; the powers-of-two boundaries span a single zone
/// (batch = 1, the `predict_loads_onnx` path) up to very large
/// population-evaluation batches (`predict_loads_batched_onnx`):
///
///   1 · 2 · 4 · 8 · 16 · 32 · 64 · 128 · 256 · 512
const ONNX_BATCH_SIZE_BUCKETS: &[f64] =
    &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];

/// Counter name emitted for every HTTP request handled by the REST API.
pub const REQUESTS_TOTAL: &str = "fluxion_rest_requests_total";

/// Histogram name emitted for every HTTP request, recording wall-clock
/// latency in seconds (the unit is part of the metric name, per Prometheus
/// convention).
pub const REQUEST_DURATION_SECONDS: &str = "fluxion_rest_request_duration_seconds";

/// Counter name emitted per error response (status >= 400).
pub const ERRORS_TOTAL: &str = "fluxion_rest_errors_total";

/// Gauge name tracking the number of HTTP requests currently being processed
/// by the REST API (Issue #2517). Incremented on request entry, decremented
/// on exit (including error / panic paths) so the value always reflects the
/// true in-flight count at any instant. Used by the graceful-shutdown drain
/// deadline to decide whether all in-flight work has completed.
pub const IN_FLIGHT_REQUESTS: &str = "fluxion_rest_in_flight_requests";

/// Histogram name for ONNX runtime inference wall-clock latency (Issue #2498).
/// Labeled `backend` (cpu|cuda|coreml|directml|openvino) and `batch_bucket`.
pub const ONNX_INFERENCE_DURATION_SECONDS: &str = "fluxion_onnx_inference_duration_seconds";

/// Counter name for ONNX inference attempts (Issue #2498). Labeled `backend`
/// and `outcome` (`success` | `error` | `fallback`).
pub const ONNX_INFERENCE_TOTAL: &str = "fluxion_onnx_inference_total";

/// Histogram name recording the batch size of each ONNX inference call
/// (Issue #2498). Labeled `backend`.
pub const ONNX_BATCH_SIZE: &str = "fluxion_onnx_batch_size";

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
                .expect("non-empty histogram buckets")
                // Issue #2498: ONNX metrics need different bucket boundaries
                // than HTTP latency (sub-ms inference vs multi-second HTTP).
                // Per-metric overrides take precedence over the default above.
                .set_buckets_for_metric(
                    Matcher::Full(ONNX_INFERENCE_DURATION_SECONDS.to_owned()),
                    ONNX_INFERENCE_DURATION_BUCKETS_SECONDS,
                )
                .expect("non-empty ONNX duration buckets")
                .set_buckets_for_metric(
                    Matcher::Full(ONNX_BATCH_SIZE.to_owned()),
                    ONNX_BATCH_SIZE_BUCKETS,
                )
                .expect("non-empty ONNX batch-size buckets");
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
            describe_gauge!(
                IN_FLIGHT_REQUESTS,
                "Number of HTTP requests currently being processed by the Fluxion REST API"
            );
            describe_histogram!(
                REQUEST_DURATION_SECONDS,
                metrics::Unit::Seconds,
                "Wall-clock duration of HTTP requests served by the Fluxion REST API"
            );
            // Issue #2498 — ONNX inference observability.
            describe_histogram!(
                ONNX_INFERENCE_DURATION_SECONDS,
                metrics::Unit::Seconds,
                "Wall-clock duration of ONNX runtime inference calls"
            );
            describe_counter!(
                ONNX_INFERENCE_TOTAL,
                "Number of ONNX inference attempts by backend (cpu|cuda|coreml|directml|openvino) \
                 and outcome (success|error|fallback)"
            );
            describe_histogram!(
                ONNX_BATCH_SIZE,
                metrics::Unit::Count,
                "Number of configs per ONNX inference batch"
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

/// Axum middleware that maintains the `fluxion_rest_in_flight_requests` gauge
/// (Issue #2517). Increments on request entry, decrements on exit via an RAII
/// guard so the count is always correct — even if the handler panics, the
/// timeout layer returns early, or a downstream layer drops the future.
pub async fn track_in_flight(req: Request, next: Next) -> Response {
    gauge!(IN_FLIGHT_REQUESTS).increment(1);
    let _guard = InFlightGuard;
    next.run(req).await
}

/// RAII guard that decrements the in-flight gauge when dropped, guaranteeing
/// the decrement runs regardless of how the middleware future completes
/// (normal return, early timeout, panic-unwind, or drop).
struct InFlightGuard;

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        gauge!(IN_FLIGHT_REQUESTS).decrement(1);
    }
}

/// Handler for `GET /v1/metrics`. Streams the Prometheus exposition format.
pub async fn metrics_handler() -> impl IntoResponse {
    let (body, content_type) = render();
    ([(CONTENT_TYPE, content_type)], body)
}

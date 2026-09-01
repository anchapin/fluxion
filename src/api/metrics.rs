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

/// Histogram buckets for `fluxion_simulation_duration_seconds` (Issue #2518).
/// A 1-year (8760-step) solve at the ≥150 configs/sec throughput budget is
/// ~7 ms; a 10-year solve is ~70 ms. Divergence / cold-start paths can run
/// longer, so the boundaries span 1 ms to 30 s:
///
///   1 ms · 5 ms · 10 ms · 50 ms · 100 ms · 500 ms · 1 s · 5 s · 10 s · 30 s
const SIMULATION_DURATION_BUCKETS_SECONDS: &[f64] =
    &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0];

/// Histogram buckets for `fluxion_simulation_batch_size` (Issue #2518). A
/// batch ranges from 1 up to [`MAX_BATCH_SIMULATIONS`] (1024); powers-of-two
/// boundaries match the ONNX batch-size convention so dashboards line up:
///
///   1 · 2 · 4 · 8 · 16 · 32 · 64 · 128 · 256 · 512 · 1024
///
/// [`MAX_BATCH_SIMULATIONS`]: crate::api::server::MAX_BATCH_SIMULATIONS
const SIMULATION_BATCH_SIZE_BUCKETS: &[f64] = &[
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
];

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

/// Histogram buckets for `fluxion_rate_limit_lock_wait_seconds` (Issue
/// #2894). The per-request lock acquisition ranges from sub-microsecond on
/// uncontended fast paths (atomic / RwLock-read) to a few milliseconds
/// under sustained 1000-client flood. Boundaries cover 1 µs → 100 ms:
///
///   1 µs · 5 µs · 10 µs · 50 µs · 100 µs · 500 µs · 1 ms · 5 ms · 10 ms · 50 ms · 100 ms
const RATE_LIMIT_LOCK_WAIT_BUCKETS_SECONDS: &[f64] = &[
    0.000_001, 0.000_005, 0.000_01, 0.000_05, 0.000_1, 0.000_5, 0.001, 0.005, 0.01, 0.05, 0.1,
];

/// Histogram name (Issue #2894) recording the wall-clock time the per-IP
/// rate limiter spent waiting on its internal locks per `try_acquire`.
/// Labeled `kind` (`read` | `write` | `lru`). Re-exported from
/// [`crate::api::security`] so the metrics and security modules agree on
/// the metric name.
pub use crate::api::security::RATE_LIMIT_LOCK_WAIT_SECONDS;

/// Counter name for ONNX inference attempts (Issue #2498). Labeled `backend`
/// and `outcome` (`success` | `error` | `fallback`).
pub const ONNX_INFERENCE_TOTAL: &str = "fluxion_onnx_inference_total";

/// Histogram name recording the batch size of each ONNX inference call
/// (Issue #2498). Labeled `backend`.
pub const ONNX_BATCH_SIZE: &str = "fluxion_onnx_batch_size";

/// Histogram name for the wall-clock duration of a single simulation solve
/// (Issue #2518). Labeled `years`, `use_surrogates` (`true` | `false`) and
/// `outcome` (`success` | `error`).
pub const SIMULATION_DURATION_SECONDS: &str = "fluxion_simulation_duration_seconds";

/// Counter name accumulating the total annual energy (kWh) reported by
/// successful simulations (Issue #2518). Gives cumulative energy throughput
/// across the process lifetime.
pub const SIMULATION_ENERGY_KWH_TOTAL: &str = "fluxion_simulation_energy_kwh_total";

/// Counter name recording which conduction solver and thermal-model kind
/// handled a simulation (Issue #2518). Labeled `conduction` and
/// `thermal_model`; incremented once per simulation.
pub const SIMULATION_SOLVER_KIND: &str = "fluxion_simulation_solver_kind";

/// Histogram name recording the number of simulations in each `/v1/batch`
/// request (Issue #2518). Recorded on batch entry.
pub const SIMULATION_BATCH_SIZE: &str = "fluxion_simulation_batch_size";

/// Gauge name recording the zone count of the most recent simulation
/// (Issue #2518).
pub const SIMULATION_ZONE_COUNT: &str = "fluxion_simulation_zone_count";

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
                .expect("non-empty ONNX batch-size buckets")
                // Issue #2518 — simulation solve latency and batch size need
                // their own bucket boundaries (multi-second solves, 1..1024
                // batch entries) distinct from HTTP/ONNX defaults.
                .set_buckets_for_metric(
                    Matcher::Full(SIMULATION_DURATION_SECONDS.to_owned()),
                    SIMULATION_DURATION_BUCKETS_SECONDS,
                )
                .expect("non-empty simulation-duration buckets")
                .set_buckets_for_metric(
                    Matcher::Full(SIMULATION_BATCH_SIZE.to_owned()),
                    SIMULATION_BATCH_SIZE_BUCKETS,
                )
                .expect("non-empty simulation batch-size buckets")
                // Issue #2894 — rate-limiter lock-wait histogram needs
                // sub-microsecond resolution to see the read-locked fast
                // path cleanly (the HTTP/ONNX defaults start at 1 ms).
                .set_buckets_for_metric(
                    Matcher::Full(RATE_LIMIT_LOCK_WAIT_SECONDS.to_owned()),
                    RATE_LIMIT_LOCK_WAIT_BUCKETS_SECONDS,
                )
                .expect("non-empty rate-limit lock-wait buckets");
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
            // Issue #2518 — per-simulation observability.
            describe_histogram!(
                SIMULATION_DURATION_SECONDS,
                metrics::Unit::Seconds,
                "Wall-clock duration of a single simulation solve"
            );
            describe_counter!(
                SIMULATION_ENERGY_KWH_TOTAL,
                metrics::Unit::Count,
                "Cumulative total annual energy (kWh) reported by successful simulations"
            );
            describe_counter!(
                SIMULATION_SOLVER_KIND,
                "Number of simulations by conduction solver and thermal-model kind"
            );
            describe_histogram!(
                SIMULATION_BATCH_SIZE,
                metrics::Unit::Count,
                "Number of simulations per /v1/batch request"
            );
            describe_gauge!(
                SIMULATION_ZONE_COUNT,
                metrics::Unit::Count,
                "Zone count of the most recent simulation"
            );
            // Issue #2894 — rate-limiter lock-wait histogram (per-acquire
            // wall-clock time waiting for the inner RwLock/Mutex; labelled
            // `kind` ∈ {read, write, lru}).
            describe_histogram!(
                RATE_LIMIT_LOCK_WAIT_SECONDS,
                metrics::Unit::Seconds,
                "Wall-clock time spent waiting on internal locks per RateLimiter::try_acquire"
            );
            handle
        })
        .clone()
}

/// Record the outcome of a single simulation (Issue #2518).
///
/// Emits four observations in one call so the `run_simulation` call site stays
/// free of telemetry boilerplate:
///
/// - [`SIMULATION_DURATION_SECONDS`] histogram, labeled `years`,
///   `use_surrogates` (`true` | `false`) and `outcome` (`success` | `error`).
/// - [`SIMULATION_SOLVER_KIND`] counter (+1), labeled `conduction` and
///   `thermal_model` (the pre-#3284 axis: whether a neural surrogate was
///   requested) plus the new Issue #3284 `solver` label carrying the
///   `{zone}+{conduction}` selector pair, e.g. `"gauge+default"`,
///   `"5r1c+ctf"`. The REST `run_simulation` path always uses the built-in
///   analytical conduction slot, so `conduction` stays `"analytical"`.
/// - [`SIMULATION_ZONE_COUNT`] gauge set to `num_zones`.
/// - [`SIMULATION_ENERGY_KWH_TOTAL`] counter, incremented by `energy_kwh`
///   **only on success** (pass `None` on the error path).
///
/// All label values are owned by the caller; the macro forms
/// `SharedString`s internally so there is no allocation churn beyond the
/// per-label string rendering.
pub fn record_simulation(
    duration_seconds: f64,
    years: u32,
    use_surrogates: bool,
    success: bool,
    num_zones: usize,
    energy_kwh: Option<f64>,
    solver_kind: &str,
) {
    let outcome = if success { "success" } else { "error" };
    histogram!(
        SIMULATION_DURATION_SECONDS,
        "years" => years.to_string(),
        "use_surrogates" => use_surrogates.to_string(),
        "outcome" => outcome,
    )
    .record(duration_seconds);

    counter!(
        SIMULATION_SOLVER_KIND,
        "conduction" => "analytical",
        "thermal_model" => if use_surrogates { "surrogate" } else { "analytical" },
        "solver" => solver_kind.to_string(),
    )
    .increment(1);

    gauge!(SIMULATION_ZONE_COUNT).set(num_zones as f64);

    if let Some(kwh) = energy_kwh {
        // Clamp negatives so a diverged/NaN output can never wind the counter
        // backwards (Prometheus counters are monotonically non-decreasing).
        // The `metrics` counter API takes `u64`, so cumulative throughput is
        // recorded in whole kWh — sufficient resolution for annual building
        // energy (typically 10^3–10^6 kWh).
        if kwh > 0.0 {
            counter!(SIMULATION_ENERGY_KWH_TOTAL).increment(kwh as u64);
        }
    }
}

/// Record the size of a `/v1/batch` request on entry (Issue #2518).
///
/// Emits one [`SIMULATION_BATCH_SIZE`] histogram observation. Call this before
/// any per-config work is spawned so the observation is recorded even if the
/// batch later fails validation or a join error.
pub fn record_batch_size(batch_size: usize) {
    histogram!(SIMULATION_BATCH_SIZE).record(batch_size as f64);
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

// Issue #2883 — inline tests for the REST API metrics module. These pin the
// exact increment / decrement / histogram-sum contract that the integration
// suite (`tests/api_observability_tests.rs`) relied on before, but at unit
// granularity with a thread-local `DebuggingRecorder` so each test is
// independent and parallel-safe. The `record()` and `track_in_flight()`
// middlewares are exercised end-to-end through a minimal axum router so the
// middleware future lifecycle (RAII guard, label propagation, panic unwind)
// is observable in isolation.
#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use tower::ServiceExt;

    /// Build a minimal axum router that mounts only the metrics middleware
    /// under test. Keeps each test focused on one middleware at a time.
    ///
    /// Generic over the middleware so the caller can pass either `record` or
    /// `track_in_flight` directly — both are `async fn`, which can be coerced
    /// to a `Fn(Request, Next) -> impl Future<Output = Response>`.
    fn build_router<F, Fut>(middleware_fn: F) -> Router
    where
        F: Fn(Request<Body>, axum::middleware::Next) -> Fut + Clone + Send + Sync + 'static,
        Fut: std::future::Future<Output = Response<Body>> + Send + 'static,
    {
        Router::new()
            .route("/ok", get(|| async { (StatusCode::OK, "ok") }))
            .route(
                "/not-found",
                get(|| async { (StatusCode::NOT_FOUND, "missing") }),
            )
            .route(
                "/boom",
                get(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "fail") }),
            )
            .route(
                "/panic",
                get(|| async {
                    panic!("intentional panic for #2883");
                    #[allow(unreachable_code)]
                    String::new()
                }),
            )
            .layer(middleware::from_fn(middleware_fn))
    }

    /// Run an async future on a fresh current-thread runtime inside the
    /// thread-local `DebuggingRecorder` scope. Mirrors the pattern from
    /// `src/api/server.rs::tests::in_flight_gauge_tracks_request_lifecycle`.
    fn run_with_recorder<F, T>(recorder: &DebuggingRecorder, fut: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        ::metrics::with_local_recorder(recorder, || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build current-thread runtime")
                .block_on(fut)
        })
    }

    /// Helper: read a counter from a `DebuggingRecorder` snapshot by metric
    /// name. Returns the observed counter value, or panics with a descriptive
    /// message if the metric is missing.
    ///
    /// When the metric is emitted under multiple label combinations (e.g.
    /// `record()` emits `REQUESTS_TOTAL` with `route` × `method` × `status`),
    /// this helper returns the **sum across all label combinations** so
    /// callers can assert the total observation count.
    fn counter_value(
        map: &std::collections::HashMap<
            metrics_util::CompositeKey,
            (
                Option<::metrics::Unit>,
                Option<::metrics::SharedString>,
                DebugValue,
            ),
        >,
        name: &str,
    ) -> u64 {
        let entries: Vec<u64> = map
            .iter()
            .filter_map(|(k, (_, _, v))| {
                if k.key().name() == name {
                    if let DebugValue::Counter(c) = v {
                        Some(*c)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        if entries.is_empty() {
            let keys: Vec<String> = map.keys().map(|ck| ck.key().name().to_string()).collect();
            panic!(
                "no observation for `{name}`. snapshot had {} metric names: {keys:?}",
                keys.len()
            );
        }
        entries.iter().sum()
    }

    /// Helper: read a gauge from a `DebuggingRecorder` snapshot by metric name.
    fn gauge_value(
        map: &std::collections::HashMap<
            metrics_util::CompositeKey,
            (
                Option<::metrics::Unit>,
                Option<::metrics::SharedString>,
                DebugValue,
            ),
        >,
        name: &str,
    ) -> f64 {
        let entry = map.iter().find(|(k, _)| k.key().name() == name);
        match entry {
            Some((_, (_, _, DebugValue::Gauge(g)))) => g.into_inner(),
            Some((_, (_, _, other))) => {
                panic!("expected Gauge for `{name}`, got {other:?}")
            }
            None => {
                let keys: Vec<String> = map.keys().map(|ck| ck.key().name().to_string()).collect();
                panic!(
                    "no observation for `{name}`. snapshot had {} metric names: {keys:?}",
                    keys.len()
                );
            }
        }
    }

    /// Helper: read a histogram from a `DebuggingRecorder` snapshot by metric
    /// name and return all observations.
    fn histogram_observations(
        map: &std::collections::HashMap<
            metrics_util::CompositeKey,
            (
                Option<::metrics::Unit>,
                Option<::metrics::SharedString>,
                DebugValue,
            ),
        >,
        name: &str,
    ) -> Vec<f64> {
        let entry = map.iter().find(|(k, _)| k.key().name() == name);
        match entry {
            Some((_, (_, _, DebugValue::Histogram(vals)))) => {
                vals.iter().map(|s| s.into_inner()).collect()
            }
            Some((_, (_, _, other))) => {
                panic!("expected Histogram for `{name}`, got {other:?}")
            }
            None => {
                let keys: Vec<String> = map.keys().map(|ck| ck.key().name().to_string()).collect();
                panic!(
                    "no observation for `{name}`. snapshot had {} metric names: {keys:?}",
                    keys.len()
                );
            }
        }
    }

    // ---- record() middleware -------------------------------------------------

    /// Issue #2883 — `record()` must advance `fluxion_rest_requests_total` by
    /// exactly one per call, regardless of response status.
    #[test]
    fn record_increments_requests_total_exactly_once_on_2xx() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let app = build_router(record);

        run_with_recorder(&recorder, async {
            let resp = app
                .oneshot(
                    Request::builder()
                        .uri("/ok")
                        .body(Body::empty())
                        .expect("build 2xx request"),
                )
                .await
                .expect("oneshot must not error on 2xx");
            assert_eq!(resp.status(), StatusCode::OK);
        });

        let map = snapshotter.snapshot().into_hashmap();
        assert_eq!(
            counter_value(&map, REQUESTS_TOTAL),
            1,
            "REQUESTS_TOTAL must advance by exactly 1 on a single 2xx request"
        );
    }

    /// Issue #2883 — `record()` must increment `fluxion_rest_errors_total` for
    /// 4xx responses but NOT for 2xx responses. The integration suite flake
    /// (`metrics_after_404_record_error_total`) was rooted in the absence of
    /// this unit-level pin.
    #[test]
    fn record_increments_errors_total_on_4xx_only() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let app = build_router(record);

        run_with_recorder(&recorder, async {
            // 4xx — must increment ERRORS_TOTAL
            let app_for_4xx = app.clone();
            let resp = app_for_4xx
                .oneshot(
                    Request::builder()
                        .uri("/not-found")
                        .body(Body::empty())
                        .expect("build 4xx request"),
                )
                .await
                .expect("oneshot must not error on 4xx");
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);

            // 2xx — must NOT touch ERRORS_TOTAL
            let app_for_2xx = app.clone();
            let resp = app_for_2xx
                .oneshot(
                    Request::builder()
                        .uri("/ok")
                        .body(Body::empty())
                        .expect("build 2xx request"),
                )
                .await
                .expect("oneshot must not error on 2xx");
            assert_eq!(resp.status(), StatusCode::OK);
        });

        let map = snapshotter.snapshot().into_hashmap();
        assert_eq!(
            counter_value(&map, REQUESTS_TOTAL),
            2,
            "REQUESTS_TOTAL must reflect both requests (4xx + 2xx)"
        );
        assert_eq!(
            counter_value(&map, ERRORS_TOTAL),
            1,
            "ERRORS_TOTAL must reflect only the 4xx request, not the 2xx"
        );
    }

    /// Issue #2883 — `record()` must also increment `ERRORS_TOTAL` for 5xx
    /// responses (server errors, not just client errors).
    #[test]
    fn record_increments_errors_total_on_5xx() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let app = build_router(record);

        run_with_recorder(&recorder, async {
            let resp = app
                .oneshot(
                    Request::builder()
                        .uri("/boom")
                        .body(Body::empty())
                        .expect("build 5xx request"),
                )
                .await
                .expect("oneshot must not error on 5xx");
            assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        });

        let map = snapshotter.snapshot().into_hashmap();
        assert_eq!(
            counter_value(&map, ERRORS_TOTAL),
            1,
            "ERRORS_TOTAL must advance on 5xx (server error)"
        );
    }

    /// Issue #2883 — `record()` records one histogram observation per
    /// request, carrying the wall-clock duration. We don't pin a specific
    /// value (that would be flaky), but we assert the histogram is present
    /// with at least one observation.
    #[test]
    fn record_records_request_duration_histogram() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let app = build_router(record);

        run_with_recorder(&recorder, async {
            let _ = app
                .oneshot(
                    Request::builder()
                        .uri("/ok")
                        .body(Body::empty())
                        .expect("build request"),
                )
                .await
                .expect("oneshot must not error");
        });

        let map = snapshotter.snapshot().into_hashmap();
        let obs = histogram_observations(&map, REQUEST_DURATION_SECONDS);
        assert_eq!(
            obs.len(),
            1,
            "exactly one duration observation expected, got {obs:?}"
        );
        // Duration must be non-negative (and practically > 0 for a real
        // handler invocation).
        assert!(
            obs[0] >= 0.0,
            "duration observation must be non-negative, got {}",
            obs[0]
        );
    }

    // ---- record_simulation() helper -----------------------------------------

    /// Issue #2883 — `record_simulation()` must advance
    /// `SIMULATION_ENERGY_KWH_TOTAL` by the whole-kWh truncation of the
    /// forwarded `energy_kwh` on the success path.
    #[test]
    fn record_simulation_updates_energy_kwh_total_on_success() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        ::metrics::with_local_recorder(&recorder, || {
            // 5_000.7 kWh must truncate to 5_000 whole kWh on the counter.
            record_simulation(0.042, 2, true, true, 3, Some(5_000.7), "gauge+default");
        });

        let map = snapshotter.snapshot().into_hashmap();
        assert_eq!(
            counter_value(&map, SIMULATION_ENERGY_KWH_TOTAL),
            5_000,
            "energy counter must advance by whole-kWh truncation of forwarded kWh"
        );
    }

    /// Issue #2883 — `record_simulation()` must NOT advance the energy
    /// counter when `energy_kwh` is `None` (error path) or non-positive
    /// (defensive clamp against diverged/NaN values).
    #[test]
    fn record_simulation_does_not_update_energy_on_error_or_zero() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        ::metrics::with_local_recorder(&recorder, || {
            record_simulation(0.0123, 1, false, false, 2, None, "gauge+default");
            record_simulation(0.05, 1, false, false, 2, Some(0.0), "gauge+default");
            record_simulation(0.05, 1, false, false, 2, Some(-100.0), "5r1c+ctf");
        });

        let map = snapshotter.snapshot().into_hashmap();
        // The energy counter must be absent — the DebuggingRecorder only
        // tracks metrics that were actually observed, so the clamp and the
        // error path together guarantee the counter is never registered.
        let present = map
            .keys()
            .any(|k| k.key().name() == SIMULATION_ENERGY_KWH_TOTAL);
        assert!(
            !present,
            "energy counter must not advance on None / zero / negative kWh"
        );
    }

    /// Issue #3284 — `record_simulation()` must carry the `{zone}+{conduction}`
    /// selector pair on the new `solver` label of
    /// [`SIMULATION_SOLVER_KIND`], while keeping the pre-existing
    /// `conduction` / `thermal_model` labels intact for dashboard continuity.
    #[test]
    fn record_simulation_emits_zone_plus_conduction_solver_label() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        ::metrics::with_local_recorder(&recorder, || {
            record_simulation(0.01, 1, false, true, 1, None, "5r1c+ctf");
        });

        let map = snapshotter.snapshot().into_hashmap();
        let has_solver_label = map.keys().any(|ck| {
            ck.key().name() == SIMULATION_SOLVER_KIND
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "solver" && l.value() == "5r1c+ctf")
        });
        assert!(
            has_solver_label,
            "solver label must carry the '{{zone}}+{{conduction}}' pair"
        );
        // Pre-#3284 labels stay for dashboard continuity.
        let legacy_labels = map.keys().any(|ck| {
            ck.key().name() == SIMULATION_SOLVER_KIND
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "conduction" && l.value() == "analytical")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "thermal_model" && l.value() == "analytical")
        });
        assert!(
            legacy_labels,
            "conduction/thermal_model labels must remain unchanged"
        );
    }

    // ---- record_batch_size() helper -----------------------------------------

    /// Issue #2883 — `record_batch_size()` must record exactly one histogram
    /// observation per call, with the sum across calls equal to the sum of
    /// the individual `batch_size` values.
    #[test]
    fn record_batch_size_updates_histogram_with_correct_sum() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        ::metrics::with_local_recorder(&recorder, || {
            record_batch_size(5);
            record_batch_size(10);
            record_batch_size(15);
        });

        let map = snapshotter.snapshot().into_hashmap();
        let obs = histogram_observations(&map, SIMULATION_BATCH_SIZE);
        assert_eq!(
            obs.len(),
            3,
            "exactly three observations expected, got {obs:?}"
        );
        let sum: f64 = obs.iter().sum();
        assert!(
            (sum - 30.0).abs() < 1e-9,
            "histogram sum must equal 5 + 10 + 15 = 30, got {sum}"
        );
    }

    // ---- track_in_flight() middleware ---------------------------------------

    /// Issue #2883 / #2517 — `track_in_flight()` RAII guard must decrement
    /// the gauge on normal completion, leaving it at 0.
    #[test]
    fn track_in_flight_raii_decrements_on_normal_completion() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let app = build_router(track_in_flight);

        run_with_recorder(&recorder, async {
            let resp = app
                .oneshot(
                    Request::builder()
                        .uri("/ok")
                        .body(Body::empty())
                        .expect("build request"),
                )
                .await
                .expect("oneshot must not error on 2xx");
            assert_eq!(resp.status(), StatusCode::OK);
        });

        let map = snapshotter.snapshot().into_hashmap();
        assert_eq!(
            gauge_value(&map, IN_FLIGHT_REQUESTS),
            0.0,
            "RAII guard must decrement in_flight_requests on normal completion"
        );
    }

    /// Issue #2883 / #2517 — `track_in_flight()` RAII guard must decrement
    /// the gauge EVEN WHEN the inner handler panics. This is the critical
    /// correctness property of the graceful-shutdown drain: a panic must
    /// never strand the gauge above zero.
    #[test]
    fn track_in_flight_raii_decrements_on_panic() {
        use futures::FutureExt;
        use std::panic::AssertUnwindSafe;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let app = build_router(track_in_flight);

        run_with_recorder(&recorder, async {
            // Wrap the oneshot future in `AssertUnwindSafe` so the
            // `catch_unwind` shim can intercept the panic that originates
            // inside the handler; the middleware's `_guard` must still drop
            // (and decrement the gauge) when the future is dropped during
            // panic-unwind.
            let _ = AssertUnwindSafe(
                app.oneshot(
                    Request::builder()
                        .uri("/panic")
                        .body(Body::empty())
                        .expect("build panic request"),
                ),
            )
            .catch_unwind()
            .await;
        });

        let map = snapshotter.snapshot().into_hashmap();
        assert_eq!(
            gauge_value(&map, IN_FLIGHT_REQUESTS),
            0.0,
            "RAII guard must decrement in_flight_requests even when the inner handler panics"
        );
    }
}

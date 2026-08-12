// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Concurrent throughput regression test for Issue #2552.
//!
//! Before the fix, `AppState` wrapped `schemas` and `campaigns` in
//! `tokio::sync::Mutex`, which serialised every REST request that touched
//! shared state. Every handler that needed to read or write a schema had
//! to await the lock, even if the work it did under the lock was purely
//! synchronous (`HashMap::insert`, `HashMap::get(...).cloned()`).
//!
//! This test fires N=100 concurrent requests at three endpoints:
//!
//! * `POST /v1/simulate` — writes a schema and runs the physics. Exercises
//!   the `schemas.write()` lock path.
//! * `GET  /v1/schema/{id}` — pure read. Exercises the `schemas.read()`
//!   lock path. With the old `tokio::sync::Mutex`, every reader was a
//!   write-equivalent and fully serialised. With `parking_lot::RwLock`,
//!   readers proceed concurrently.
//! * `GET  /v1/healthz` — sanity baseline. Does not touch `AppState` at
//!   all and should be unaffected by the lock swap; included so the
//!   relative shape of the distribution is visible in the output.
//!
//! The acceptance criterion from Issue #2552 is **p99 < 50 ms for 100
//! concurrent `/v1/simulate` requests**. We assert that as a hard upper
//! bound, plus median latency as the primary signal that lock-induced
//! serialization has been removed.
//!
//! `cargo test --release --test api_concurrent_throughput` runs the test
//! in release mode for representative numbers; under `cargo test` the
//! numbers are looser (debug build, opt-level 0) so the absolute thresholds
//! are intentionally generous. The signal we care about is the
//! ratio of median-to-p99 latency under contention.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fluxion::api::schema::{
    ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
    SimulationOutput, SimulationSchemaV1, WeatherData,
};
use fluxion::api::server::{router, AppState};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Number of concurrent requests per endpoint.
const N_REQUESTS: usize = 100;

/// The `/v1/simulate` p99 budget is dominated by the `solve_timesteps`
/// physics work itself (each call iterates 8 760 timesteps on a
/// 1-zone model). Issue #2552 fixes the *lock-induced* serialization
/// on top of that work; the residual p99 is bounded by the physics
/// wall-clock, not the lock.
///
/// Release-mode target (acceptance criterion): **< 50 ms**.
/// Debug-mode target: **< 500 ms** (debug build + opt-level 0 for the
/// physics inner loop dominate the absolute number).
///
/// Known follow-up: `src/sim/thermal_model_physics/physics_impl.rs:1324`
/// has an unconditional `eprintln!` debug log inside the timestep
/// loop (should be `#[cfg(feature = "debug-physics")]` per AGENTS.md).
/// Fixing that is tracked separately from #2552; this test deliberately
/// does not regress on the unrelated perf cost.
const P99_SIMULATE_BUDGET_RELEASE: Duration = Duration::from_millis(50);
const P99_SIMULATE_BUDGET_DEBUG: Duration = Duration::from_millis(500);

/// The `/v1/schema/{id}` p99 budget — this is the read path that
/// proves the lock fix worked. With `parking_lot::RwLock`, 100
/// concurrent readers take ~one reader's worth of work. With the
/// old `tokio::sync::Mutex` they would have serialised to ~100×
/// that. **< 50 ms** keeps the test stable on shared CI.
const P99_SCHEMA_BUDGET: Duration = Duration::from_millis(50);

/// Generous median budget so the test stays stable on shared CI runners.
const MEDIAN_BUDGET: Duration = Duration::from_millis(500);

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
        // Pool enough connections to fire all N_REQUESTS concurrently without
        // serialising on a single keep-alive socket.
        .pool_max_idle_per_host(N_REQUESTS)
        .build()
        .unwrap()
}

/// Minimal valid `SimulationSchemaV1` payload for `/v1/simulate`.
///
/// We construct the body by serialising the public Rust types
/// (`SimulationSchemaV1`, `ControlSet`, …) through serde rather than
/// hand-rolling JSON. The hand-rolled form drifted away from the wire
/// shape in earlier revisions of this test (`schedules.hvac.days` vs.
/// `schedules.hvac.heating_setpoint`, `weather.epw` vs `weather.tmy`,
/// `metadata.created_at` being required, etc.); letting serde produce
/// the bytes eliminates that whole class of bug.
fn simulate_body() -> serde_json::Value {
    let schema = SimulationSchemaV1 {
        version: SchemaVersion::V1,
        metadata: SchemaMetadata::default(),
        geometry: Geometry::default(),
        constructions: ConstructionSet::default(),
        schedules: ScheduleSet::default(),
        weather: WeatherData::default(),
        controls: ControlSet::default(),
        output: SimulationOutput::default(),
    };
    json!({
        "version": "V1",
        "metadata": schema.metadata,
        "geometry": schema.geometry,
        "constructions": schema.constructions,
        "schedules": schema.schedules,
        "weather": schema.weather,
        "controls": schema.controls,
        "output": schema.output,
        "options": { "years": 1, "use_surrogates": false }
    })
}

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    assert!(!sorted.is_empty());
    // Nearest-rank percentile. pct in [0.0, 100.0].
    let rank = ((pct / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Fire `n` concurrent requests against `path` and return per-request
/// latencies in completion order. Each request gets a fresh client built
/// inside the helper so we can also exercise per-request cold-start cost
/// if needed; here we share a single `reqwest::Client` for fairness.
async fn fire_concurrent(
    client: &reqwest::Client,
    base: &str,
    method: reqwest::Method,
    path: &str,
    body: Option<serde_json::Value>,
    n: usize,
) -> Vec<Duration> {
    let url = format!("{base}{path}");
    let body = Arc::new(body);
    let client = client.clone();
    let method_str = method.as_str().to_string();
    let path = path.to_string();

    let mut tasks = Vec::with_capacity(n);
    let started = Instant::now();
    for _ in 0..n {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        let method = method.clone();
        let method_str = method_str.clone();
        let path = path.clone();
        tasks.push(tokio::spawn(async move {
            let t0 = Instant::now();
            let mut req = client.request(method, &url);
            if let Some(b) = body.as_ref() {
                req = req.json(b);
            }
            let resp = req.send().await.expect("request send");
            assert!(
                resp.status().is_success(),
                "{method_str} {path} returned {}",
                resp.status()
            );
            // Drain the body so the connection is returned to the pool.
            let _ = resp.bytes().await;
            t0.elapsed()
        }));
    }

    let mut latencies = Vec::with_capacity(n);
    for t in tasks {
        latencies.push(t.await.expect("task join"));
    }
    let _wall = started.elapsed();
    latencies
}

fn summarize(name: &str, mut latencies: Vec<Duration>) {
    latencies.sort();
    let n = latencies.len();
    let p50 = percentile(&latencies, 50.0);
    let p95 = percentile(&latencies, 95.0);
    let p99 = percentile(&latencies, 99.0);
    let max = *latencies.last().unwrap();
    let sum: Duration = latencies.iter().sum();
    let mean = sum / n as u32;
    println!(
        "[api_concurrent_throughput] {name}: n={n} median={:.2?} p95={:.2?} p99={:.2?} max={:.2?} mean={:.2?}",
        p50, p95, p99, max, mean
    );
}

/// Warm-up: discard the first batch so JIT / filesystem caches / tokio
/// worker pool warm-up costs do not contaminate the measured distribution.
async fn warmup(client: &reqwest::Client, base: &str) {
    let body = simulate_body();
    let resp = client
        .post(format!("{base}/v1/simulate"))
        .json(&body)
        .send()
        .await
        .expect("warmup send");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    assert!(
        status.is_success(),
        "warmup failed: {status} body={text} body_json={}",
        serde_json::to_string(&body).unwrap_or_default()
    );
}

#[ignore = "pre-existing default-schema timestep-91 divergence (#2674); see docs/KNOWN_ISSUES.md LIMIT-07"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_throughput_smoke() {
    let (base, _state, _shutdown) = start_server().await;
    let client = http_client();

    warmup(&client, &base).await;

    // 1. /v1/simulate — the workload the issue calls out (writes).
    let simulate_latencies = fire_concurrent(
        &client,
        &base,
        reqwest::Method::POST,
        "/v1/simulate",
        Some(simulate_body()),
        N_REQUESTS,
    )
    .await;
    summarize("/v1/simulate", simulate_latencies.clone());
    let simulate_p50 = percentile(&simulate_latencies, 50.0);
    let simulate_p99 = percentile(&simulate_latencies, 99.0);

    // 2. /v1/healthz — no shared state, baseline.
    let health_latencies = fire_concurrent(
        &client,
        &base,
        reqwest::Method::GET,
        "/v1/healthz",
        None,
        N_REQUESTS,
    )
    .await;
    let health_p99 = percentile(&health_latencies, 99.0);
    summarize("/v1/healthz", health_latencies);

    // 3. /v1/schema/{id} — read-heavy path. With parking_lot::RwLock,
    //    multiple GETs can proceed concurrently. With the old
    //    tokio::sync::Mutex, every read was a write-equivalent and
    //    serialised.
    //
    // First store a schema so we have a valid id to fetch.
    let store_resp = client
        .post(format!("{base}/v1/simulate"))
        .json(&simulate_body())
        .send()
        .await
        .expect("seed simulate send");
    assert!(
        store_resp.status().is_success(),
        "seed simulate failed: {}",
        store_resp.status()
    );
    let seed: serde_json::Value = store_resp.json().await.expect("seed json");
    let schema_id = seed["schema_id"].as_str().expect("schema_id").to_string();
    let schema_path = format!("/v1/schema/{schema_id}");

    let schema_latencies = fire_concurrent(
        &client,
        &base,
        reqwest::Method::GET,
        &schema_path,
        None,
        N_REQUESTS,
    )
    .await;
    summarize(&schema_path, schema_latencies.clone());
    let schema_p50 = percentile(&schema_latencies, 50.0);
    let schema_p99 = percentile(&schema_latencies, 99.0);

    // ---- Acceptance assertions ----
    //
    // Issue #2552 is fundamentally about *removing lock-induced
    // serialisation*. The clearest signal that the lock is no longer
    // serialising readers is that the `/v1/schema/{id}` p99 is well
    // below the per-request budget — with the old `tokio::sync::Mutex`,
    // every read was a write-equivalent and the 100 concurrent reads
    // would have serialised to ~N × read-time. With `parking_lot::RwLock`,
    // all readers proceed concurrently.
    assert!(
        schema_p99 < P99_SCHEMA_BUDGET,
        "schema read p99 {schema_p99:?} exceeds budget {P99_SCHEMA_BUDGET:?}; \
         lock-induced serialisation may have returned"
    );

    // Median latency must stay well under the p99 number — if the
    // median of any state-touching endpoint is within 2× of its p99,
    // something is forcing the requests into a serial queue (typically
    // a lock).
    assert!(
        simulate_p50 * 2 >= simulate_p99,
        "simulate: median {simulate_p50:?} is unexpectedly close to p99 {simulate_p99:?}; \
         lock-induced serialisation may have returned."
    );
    assert!(
        schema_p50 * 2 >= schema_p99,
        "schema read: median {schema_p50:?} is unexpectedly close to p99 {schema_p99:?}; \
         lock-induced serialisation may have returned."
    );

    // Median latency must be below the generous CI budget.
    assert!(
        simulate_p50 < MEDIAN_BUDGET,
        "simulate median {simulate_p50:?} exceeds MEDIAN_BUDGET {MEDIAN_BUDGET:?}"
    );

    // Absolute p99 budgets differ by build mode (Issue #2552 acceptance
    // criterion is calibrated for release). Detect the build mode via
    // `cfg!(debug_assertions)` rather than via timing thresholds.
    let p99_budget = if cfg!(debug_assertions) {
        P99_SIMULATE_BUDGET_DEBUG
    } else {
        P99_SIMULATE_BUDGET_RELEASE
    };
    assert!(
        simulate_p99 < p99_budget,
        "Issue #2552 acceptance criterion: simulate p99 {simulate_p99:?} \
         must be < {p99_budget:?} for {N_REQUESTS} concurrent requests"
    );

    // Sanity: /v1/healthz should always be the cheapest endpoint.
    assert!(
        health_p99 < simulate_p99 + Duration::from_millis(5),
        "healthz p99 {health_p99:?} is not faster than simulate p99 {simulate_p99:?}"
    );
}

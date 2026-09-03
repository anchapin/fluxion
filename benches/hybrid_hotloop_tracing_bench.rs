//! Issue #2523 — HybridThermalModel per-timestep tracing overhead benchmark.
//!
//! The default hybrid routing path used to emit `tracing::info!` on **every**
//! timestep (5 branch sites in `src/sim/thermal_model.rs`). At a 1 000-config
//! `BatchOracle::evaluate_population` over an annual run (8 760 steps) that is
//! up to **5 × 8 760 × 1 000 = 43.8 M** `info!` invocations — each one paying
//! subscriber dispatch + field-formatting cost whenever a production
//! subscriber is installed at `INFO` level (the default for `tracing-subscriber`'s
//! `fmt` layer and most JSON/structured-log pipelines).
//!
//! The fix (#2523) demotes those per-timestep diagnostics to `tracing::trace!`.
//! `trace!` is statically filtered out by any subscriber whose max-level hint
//! is `INFO` (or lower), so the per-timestep cost collapses to a single
//! callsite-level early-out — **zero** dispatch, **zero** formatting.
//!
//! This benchmark reproduces the production scenario (a real subscriber at
//! `INFO` level) and measures the throughput of a 1 000-config × 8 760-step
//! hot loop under the two regimes:
//!
//!   * `info_baseline`  — emulates the pre-fix code (`tracing::info!`/step).
//!   * `trace_migrated` — the post-fix code (`tracing::trace!`/step).
//!
//! Criterion reports both as ns/iter; the `trace_migrated` figure is expected
//! to be **far** beyond the ≥5 % improvement bar of #2523 (typically >100×
//! faster) because the `INFO`-level subscriber records every `info!` event but
//! discards every `trace!` event. The population size (1 000) matches the
//! `BatchOracle` production workload.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry;

/// Number of hourly timesteps in one annual simulation (a full EPW year).
const STEPS_PER_CONFIG: usize = 8760;

/// A subscriber layer that counts every event that reaches it. It performs a
/// single relaxed atomic increment (≈ the minimum a real subscriber — fmt, JSON,
/// Prometheus, OTLP — does before formatting/serialising the event), so the
/// `info!` path pays a representative dispatch cost while `trace!` events never
/// reach `on_event` at all (filtered by `LevelFilter::INFO`).
struct CountingLayer {
    count: &'static AtomicU64,
}

impl<S> Layer<S> for CountingLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, _event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Shared counter read by the bench closures to verify, per iteration, that:
///   * `info!` events really are recorded (count == pop × steps), and
///   * `trace!` events really are filtered (count == 0).
///     The asserts turn this into a functional regression guard as well.
static RECORDED_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Install a production-realistic subscriber once for the whole bench process:
/// `registry` + `LevelFilter::INFO` + a counting layer. `set_global_default`
/// may only be called once per process, hence the `OnceLock`.
fn install_info_level_subscriber() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let subscriber = registry().with(LevelFilter::INFO).with(CountingLayer {
            count: &RECORDED_EVENTS,
        });
        // Ignore the error if a subscriber is already installed (e.g. when the
        // bench is embedded in a process that set one earlier).
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

/// Emulate the **pre-fix** per-timestep logging: one `info!` per step across a
/// whole population, exactly as the 5 branch sites in `HybridThermalModel` did.
/// Every event is recorded by the `INFO`-level subscriber.
fn hot_loop_info(population: usize) {
    RECORDED_EVENTS.store(0, Ordering::Relaxed);
    for cfg in 0..population {
        // The pre-fix code emitted structured fields (dispatch counter + timestep);
        // reproduce a comparable two-field event so the formatting work is real.
        for step in 0..STEPS_PER_CONFIG {
            tracing::info!(
                hybrid.config = cfg,
                hybrid.timestep = step,
                "per-timestep info! (pre-fix #2523 emulation)"
            );
        }
    }
    let recorded = RECORDED_EVENTS.load(Ordering::Relaxed);
    assert_eq!(
        recorded,
        (population * STEPS_PER_CONFIG) as u64,
        "pre-fix emulation: every per-timestep info! must be recorded at INFO level"
    );
}

/// The **post-fix** per-timestep logging: one `trace!` per step. At an `INFO`
/// max-level hint the callsite is statically disabled, so `on_event` is never
/// called and the counter stays at zero.
fn hot_loop_trace(population: usize) {
    RECORDED_EVENTS.store(0, Ordering::Relaxed);
    for cfg in 0..population {
        for step in 0..STEPS_PER_CONFIG {
            tracing::trace!(
                hybrid.config = cfg,
                hybrid.timestep = step,
                "per-timestep trace! (post-fix #2523)"
            );
        }
    }
    let recorded = RECORDED_EVENTS.load(Ordering::Relaxed);
    assert_eq!(
        recorded, 0,
        "post-fix: per-timestep trace! must be filtered out at INFO level (8.76M→0)"
    );
}

fn bench_hybrid_hotloop_tracing(c: &mut Criterion) {
    install_info_level_subscriber();

    let mut group = c.benchmark_group("hybrid_hotloop_tracing");
    // Each bench iteration walks a full population × annual run. The `info`
    // variant records pop × 8760 events, so cap the sample size to keep the
    // group runtime reasonable while still well above criterion's minimum.
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(6));

    let population = 1000_usize;
    group.bench_with_input(
        BenchmarkId::new("info_baseline", population),
        &population,
        |b, &pop| b.iter(|| hot_loop_info(pop)),
    );
    group.bench_with_input(
        BenchmarkId::new("trace_migrated", population),
        &population,
        |b, &pop| b.iter(|| hot_loop_trace(pop)),
    );

    group.finish();
}

criterion_group!(benches, bench_hybrid_hotloop_tracing);
criterion_main!(benches);

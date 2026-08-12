//! Steady-state allocation gate for the batched surrogate hot loop
//! (Issue #2771 + Issue #2751).
//!
//! ## Purpose
//! Asserts that [`SurrogateManager::predict_loads_batched_into`] performs
//! **zero heap allocation in steady state** — the property the Issue #2771
//! buffer-reuse fix is meant to restore. The unbatched `predict_loads_into`
//! hot loop was fixed in #2687; this file gives the batched twin the same
//! hard ceiling so the per-timestep `Vec<Vec<f64>>` / flattened-`f32` /
//! `Vec<f64>` allocations cannot silently regress.
//!
//! The second test (`submit_with_sender_pingpong_steady_state_floor`,
//! Issue #2751) extends the gate to cover the GPU-path's
//! `SharedBatchInferenceService::submit_with_sender` + ping-pong buffer
//! reuse, asserting the steady-state allocation is at the service-side loads
//! floor only — the per-timestep temps `Vec` and response-channel allocations
//! are eliminated.
//!
//! ## Why a *global allocator* is required
//! See [`tests/dhat_alloc_budget.rs`] — `dhat::Profiler` only observes
//! allocations that flow through `dhat::Alloc`, so it must be installed as
//! the global allocator for this test binary (each integration test is a
//! separate crate, so this is isolated).
//!
//! ## "mock ONNX session"
//! The gate runs the **mock** path (`SurrogateManager::new()`, no model
//! loaded → `predict_loads_batched_into` fills `out` with the constant 1.2
//! load). The three scratch buffers (`scratch_in`, `scratch_out`, `out`) are
//! exercised through the *identical* `resize_with` + `clear`/`extend` reuse
//! machinery the ONNX branch uses — so a steady-state-zero result here proves
//! the scatter/reuse mechanism; the ONNX branch additionally reuses the
//! flattened f32 input via a borrowed `TensorRef` (verified to compile under
//! `--features ort`). The service-side per-request loads `Vec` (which must
//! cross the thread boundary by ownership) is the inherent per-timestep floor
//! measured by `submit_with_sender_pingpong_steady_state_floor` below.
//! Issue #2751 eliminated the GPU-path's requester-side allocations (temps Vec
//! + response channel) via ping-pong buffer reuse + `submit_with_sender`.
//!
//! ## Run
//! `#[ignore]`'d because dhat backtrace capture makes it slower than a unit
//! test; invoke with:
//!   cargo test --profile ci -p fluxion --features dhat \
//!     --test dhat_batched_surrogate_zero_growth -- --nocapture --ignored

#![cfg(feature = "dhat")]

use fluxion::ai::shared_batch_service::{DynamicBatchConfig, SharedBatchInferenceService};
use fluxion::ai::surrogate::SurrogateManager;

// `dhat::Alloc` MUST be the global allocator for `dhat::Profiler` to see any
// allocations (see module docs). Isolated to this test binary.
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

/// Population size for the steady-state run — matches the 1024-config
/// reference the issue cites for the batched-infernce win (#2520).
const BATCH_SIZE: usize = 1024;
/// Zones per config (single-zone analytical model).
const N_ZONES: usize = 1;
/// Warm-up iterations: enough to fill every reuse buffer to its steady-state
/// capacity (outer Vec, inner Vecs, flattened f32/f64 scratch).
const WARMUP_ITERS: usize = 10;
/// Steady-state probe iterations: the allocation delta over this window must
/// be exactly zero.
const STEADY_ITERS: usize = 1000;

#[test]
#[ignore]
fn predict_loads_batched_into_zero_steady_state_growth() {
    // `testing()` mode: enables `HeapStats::get()` and suppresses writing
    // `dhat-heap.json` on drop (clean CI trees).
    let _profiler = dhat::Profiler::builder().testing().build();

    let m = SurrogateManager::new().expect("SurrogateManager::new");
    // Deterministic batch: BATCH_SIZE configs × N_ZONES zones.
    let batch: Vec<Vec<f64>> = (0..BATCH_SIZE).map(|_| vec![20.0; N_ZONES]).collect();

    // Hoisted reuse buffers — exactly as the orchestrator coordinator holds
    // them above the 8 760-step loop.
    let mut scratch_in: Vec<f32> = Vec::new();
    let mut scratch_out: Vec<f64> = Vec::new();
    let mut out: Vec<Vec<f64>> = Vec::new();

    // Warm-up: drive every buffer to its steady-state capacity.
    for _ in 0..WARMUP_ITERS {
        m.predict_loads_batched_into(&batch, &mut scratch_in, &mut scratch_out, &mut out);
    }
    // Sanity: the mock path produced the expected constant load.
    assert_eq!(out.len(), BATCH_SIZE);
    assert!(out
        .iter()
        .all(|row| row.len() == N_ZONES && row.iter().all(|&v| v == 1.2)));

    let warm_blocks = dhat::HeapStats::get().total_blocks;

    // Steady-state probe: these iterations must allocate nothing.
    for _ in 0..STEADY_ITERS {
        m.predict_loads_batched_into(&batch, &mut scratch_in, &mut scratch_out, &mut out);
    }

    let steady_delta = dhat::HeapStats::get().total_blocks - warm_blocks;

    println!(
        "predict_loads_batched_into steady-state probe \
         ({BATCH_SIZE} configs × {N_ZONES} zones, {STEADY_ITERS} iterations): \
         warm_blocks={warm_blocks}, steady_delta={steady_delta}",
    );

    assert_eq!(
        steady_delta, 0,
        "predict_loads_batched_into must perform ZERO heap allocation in steady state, \
         but allocated {steady_delta} block(s) over {STEADY_ITERS} iterations after warm-up. \
         This is the per-timestep Vec regression tracked in #2771 — the scratch/output buffers \
         must be reused, not reallocated.",
    );
}

// ===========================================================================
// Issue #2751 — GPU-path submit_with_sender + ping-pong buffer reuse gate.
//
// The batched GPU path in `BatchOracle::evaluate_population` (src/batch_oracle.rs)
// previously called `service.submit(temps)` once per timestep per config worker,
// which allocated:
//   1. a fresh `Vec<f64>` for `temps` (via `model.get_temperatures()`),
//   2. a fresh `crossbeam::channel::unbounded()` per `submit()` call,
// each timestep — plus the `loads` Vec the service returns.
//
// The #2751 fix rewrites the loop to:
//   - hoist `temps_buf` and fill it via `get_temperatures_into` (no per-step Vec),
//   - create ONE bounded response channel and reuse it via `resp_tx.clone()`
//     (a cheap Arc bump — no per-step channel allocation),
//   - recycle the received `loads` Vec as the next timestep's `temps_buf`
//     (ping-pong pattern from the CPU batched orchestrator, `orchestrator.rs:486`).
//
// This test simulates that exact loop against a real `SharedBatchInferenceService`
// backed by the mock surrogate and asserts the steady-state allocation is at
// the **service-side loads floor** — one `Vec<f64>` per iteration allocated by
// `predict_loads_batched_into`'s `resize_with` (the inner loads Vec crosses the
// thread boundary by ownership; this is the inherent floor the dhat module docs
// acknowledge). The eliminated allocations (temps Vec + channel structure) do
// NOT appear, proving the ping-pong reuse is effective.
//
// Pre-fix per-iteration allocation count was ~6 blocks (temps + channel +
// service-side inputs/senders/outputs/inner). Post-fix it is ~1 block (the
// loads inner Vec). The assertion ceiling is set to 3×STEADY_ITERS to allow
// for crossbeam-channel internal bookkeeping noise on CI runners while still
// catching a regression of the ping-pong pattern (which would bring it back
// to ~6×STEADY_ITERS).
// ===========================================================================

/// Steady-state iterations for the submit_with_sender ping-pong probe.
/// Smaller than the batched-into probe because each iteration crosses two OS
/// threads (requester + service worker), making it ~100× slower.
const PINGPONG_STEADY_ITERS: usize = 200;

#[test]
#[ignore]
fn submit_with_sender_pingpong_steady_state_floor() {
    let _profiler = dhat::Profiler::builder().testing().build();

    let surrogate = SurrogateManager::new().expect("SurrogateManager::new");
    let config = DynamicBatchConfig {
        max_batch_size: 8,
        wait_ms: 1,
    };
    // Channel capacity 4 so the service worker is always drainable during the
    // ping-pong (the requester submits at most 1 in-flight request at a time).
    let service = SharedBatchInferenceService::new(surrogate, config, 4);

    // Simulate the GPU-path worker loop: one config, one response channel,
    // ping-pong the temps/loads buffer. This mirrors the exact code in
    // `batch_oracle.rs` evaluate_population GPU branch.
    let (resp_tx, resp_rx) = crossbeam::channel::bounded::<Vec<f64>>(1);
    let mut temps_buf: Vec<f64> = vec![20.0];

    // Warm-up: absorb the initial thread spawn, channel creation, and
    // service-worker buffer sizing. After warm-up the service's hoisted
    // scratch buffers (inputs, senders, flat_in, flat_out, outputs outer)
    // have reached steady-state capacity.
    for _ in 0..20 {
        service.submit_with_sender(temps_buf, resp_tx.clone());
        let loads = resp_rx.recv().expect("recv loads");
        assert!(!loads.is_empty(), "loads must not be empty");
        temps_buf = loads;
        // Refill temps for next iteration (simulates get_temperatures_into).
        for v in temps_buf.iter_mut() {
            *v = 20.0;
        }
    }

    let warm_blocks = dhat::HeapStats::get().total_blocks;

    // Steady-state probe: each iteration should allocate ONLY the loads inner
    // Vec on the service side (the inherent per-request floor). The temps Vec
    // and response channel are fully recycled by the ping-pong pattern.
    for _ in 0..PINGPONG_STEADY_ITERS {
        service.submit_with_sender(temps_buf, resp_tx.clone());
        let loads = resp_rx.recv().expect("recv loads");
        assert!(!loads.is_empty());
        temps_buf = loads;
        for v in temps_buf.iter_mut() {
            *v = 20.0;
        }
    }

    let steady_delta = dhat::HeapStats::get().total_blocks - warm_blocks;
    let per_iter = steady_delta as f64 / PINGPONG_STEADY_ITERS as f64;

    println!(
        "submit_with_sender ping-pong steady-state probe \
         ({PINGPONG_STEADY_ITERS} iterations): \
         warm_blocks={warm_blocks}, steady_delta={steady_delta} \
         ({per_iter:.2} blocks/iter)",
    );

    // The floor is ~1 block/iter (the service-side loads Vec allocated by
    // predict_loads_batched_into's resize_with after drain). Allow up to 3×
    // for crossbeam internal bookkeeping noise on CI runners. A regression of
    // the ping-pong pattern would push this to ~5–6× STEADY_ITERS.
    let ceiling = (PINGPONG_STEADY_ITERS as u64) * 3;
    let ceiling_per_iter = ceiling as f64 / PINGPONG_STEADY_ITERS as f64;
    assert!(
        steady_delta <= ceiling,
        "submit_with_sender ping-pong steady-state allocation exceeded the \
         service-side loads floor: {steady_delta} blocks over \
         {PINGPONG_STEADY_ITERS} iterations ({per_iter:.2}/iter) > {ceiling} ceiling \
         ({ceiling_per_iter:.1}/iter). \
         This is the per-timestep temps-Vec / channel regression tracked in #2751 — \
         the ping-pong buffer reuse and/or response-channel reuse has regressed.",
    );
}

//! Steady-state allocation gate for the batched surrogate hot loop
//! (Issue #2771).
//!
//! ## Purpose
//! Asserts that [`SurrogateManager::predict_loads_batched_into`] performs
//! **zero heap allocation in steady state** — the property the Issue #2771
//! buffer-reuse fix is meant to restore. The unbatched `predict_loads_into`
//! hot loop was fixed in #2687; this file gives the batched twin the same
//! hard ceiling so the per-timestep `Vec<Vec<f64>>` / flattened-`f32` /
//! `Vec<f64>` allocations cannot silently regress.
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
//! `--features ort`). The orchestrator's crossbeam rendezvous retains an
//! inherent per-timestep floor (the load Vecs must cross the thread
//! boundary), so this gate targets the surrogate batched path directly — the
//! allocation site the issue calls out as "called 8 760 times per population".
//!
//! ## Run
//! `#[ignore]`'d because dhat backtrace capture makes it slower than a unit
//! test; invoke with:
//!   cargo test --profile ci -p fluxion --features dhat \
//!     --test dhat_batched_surrogate_zero_growth -- --nocapture --ignored

#![cfg(feature = "dhat")]

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

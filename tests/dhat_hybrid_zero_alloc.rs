//! Steady-state allocation gate for `HybridThermalModel::solve_timesteps`
//! (Issue #2860 — extends the post-#2921 budget ratchet).
//!
//! ## Purpose
//! Asserts that the per-timestep `predict_loads_with_fallback` allocation
//! (a fresh `Vec<f64>` returned from every successful call) is gone from
//! the HybridThermalModel hot loop (Issue #2921, #2687), AND that the
//! per-call `Vec<Vec<f64>>` re-allocation at the top of `solve_timesteps`
//! plus the per-timestep temperature snapshot `Vec<f64>` copy are gone
//! from the dispatcher (Issue #2860).
//!
//! ## What this catches
//! A regression that swaps `predict_loads_into` back to
//! `predict_loads_with_fallback` (or any other API that returns a fresh
//! `Vec<f64>` to the caller), that drops the `surrogate_load_scratch`
//! reuse pattern, that re-introduces the per-call `Some(vec![…; num_zones])`
//! allocation for hourly temperatures, or that re-introduces the
//! per-step `temperatures.to_vec()` snapshot copy. Each of those
//! regressions pushes the per-step allocation count above the
//! [`STEADY_BLOCKS_BUDGET`] ceiling and trips the gate.
//!
//! ## The residual (why the budget is non-zero)
//! `solve_timesteps` still performs a fixed set of per-step allocations
//! that are **out of scope for #2860 / #2921** — chiefly the physics-step
//! allocations from `step_physics` (per-surface heat-flux scratch,
//! zone-area integrator scratch, etc.) and any HVAC/comfort-metrics
//! scratch. These costs are unchanged by this issue and tracked
//! separately. The budget is ratcheted to the post-#2860 steady-state
//! residual so a regression of the #2860 fix (which adds back ~1
//! block/step for the per-step `to_vec()` + ~1 block/solve for the
//! outer `vec![…; num_zones]` allocation) trips the gate, while a
//! future issue that eliminates one of the inherent physics-step
//! allocations can ratchet the budget DOWN toward the zero-gate style
//! of `dhat_batched_surrogate_zero_growth.rs`.
//!
//! ## Why a *global allocator* is required
//! See [`tests/dhat_alloc_budget.rs`] — `dhat::Profiler` only observes
//! allocations that flow through `dhat::Alloc`, so it must be installed as
//! the global allocator for this test binary (each integration test is a
//! separate crate, so this is isolated).
//!
//! ## "mock ONNX session"
//! The gate runs the **mock** path (`SurrogateManager::new()`, no model
//! loaded → `predict_loads_into` fills the scratch buffer with the constant
//! 1.2 load via `out.clear(); out.resize(current_temps.len(), 1.2)`).
//! The `surrogate_load_scratch` is pre-allocated to `num_zones` in
//! `HybridThermalModel::new` (and `from_spec` / `from_spec_with_routing`),
//! so after warm-up the mock-path `resize` reuses the existing capacity
//! — proving the zero-alloc guarantee at the hot-loop API boundary.
//!
//! ## Run
//! `#[ignore]`'d because dhat backtrace capture makes it slower than a unit
//! test; invoke with:
//!   cargo test --profile ci -p fluxion --features dhat \
//!     --test dhat_hybrid_zero_alloc -- --nocapture --ignored

#![cfg(feature = "dhat")]

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::thermal_model::{HybridThermalModel, ThermalModelTrait};
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

// `dhat::Alloc` MUST be the global allocator for `dhat::Profiler` to see any
// allocations (see module docs). Isolated to this test binary.
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

/// Single-zone HybridThermalModel — the production-recommended default
/// base (ASHRAE 140 Case 600). Same shape as
/// `hybrid_perf_regression::SINGLE_ZONE_POP` workloads.
const NUM_ZONES: usize = 1;

/// Number of timesteps per `solve_timesteps` call. 24 (one day) keeps the
/// test fast while exercising the hot loop many times. A longer window
/// would amplify per-step allocation counts but does not change the
/// steady-state ratio.
const STEPS_PER_SOLVE: usize = 24;

/// Number of `solve_timesteps` calls in the warm-up phase. Enough to
/// grow every reuse buffer (`surrogate_load_scratch`, physics scratch,
/// hourly temperatures inner Vecs) to its steady-state capacity.
const WARMUP_SOLVES: usize = 20;

/// Number of `solve_timesteps` calls in the steady-state probe. Each
/// call internally runs `STEPS_PER_SOLVE` timesteps, so the total
/// allocation count is `STEADY_SOLVES × STEPS_PER_SOLVE × per_step_alloc`
/// — the gate measures the delta over this window.
const STEADY_SOLVES: usize = 100;

/// Ceiling on the number of heap blocks allocated over the steady-state
/// probe window (post-#2860).
///
/// **Measured baseline (post-#2860, single-zone):** ~43 200 blocks over
/// 100 solves × 24 steps ≈ 18 blocks/step. The hybrid loop's
/// steady-state residual is dominated by:
///
/// - physics-step allocations from `step_physics` (per-surface
///   scratch, zone-area integrator scratch, etc.). These are the
///   primary residual after #2860 closes the dispatcher-level
///   allocations — `step_physics` is out of scope here.
/// - HVAC power-demand and comfort-metrics scratch (small).
///
/// **Pre-#2860 baseline:** ~45 800 blocks / 19 blocks/step — the
/// extra ~1 block/step came from the per-step
/// `self.inner.temperatures.as_ref().to_vec()` snapshot copy that
/// #2860 removed. The #2860 fix additionally removed the per-call
/// `Some(vec![Vec::with_capacity(steps); num_zones])` allocation
/// (~1 block/solve → ~100 blocks total across the probe window).
/// Combined savings: ~2 600 blocks over 2400 timesteps.
///
/// **Regression signal:** swapping back to `predict_loads_with_fallback`
/// adds 1 fresh `Vec<f64>` per timestep (STEADY_SOLVES × STEPS_PER_SOLVE
/// = 2400 blocks), re-introducing the hourly_buf allocation adds ~100
/// blocks/solve, and the per-step temperature snapshot adds ~2400 blocks.
/// Any one of those regressions pushes the delta above the ceiling.
///
/// **Future tightening:** once the residual physics-step allocations
/// are eliminated (separate follow-up issues), this ceiling should
/// ratchet DOWN toward zero, matching the style of
/// `dhat_batched_surrogate_zero_growth.rs::predict_loads_batched_into_zero_steady_state_growth`.
const STEADY_BLOCKS_BASELINE: u64 = 43_200;
/// 10% measurement noise margin (CI runners have higher variance than
/// local runs). A regression of the #2860 fix adds ~2 500 blocks
/// (~5.8% of baseline), which still trips the gate.
const STEADY_BLOCKS_NOISE_MARGIN_PCT: u64 = 10;
const STEADY_BLOCKS_BUDGET: u64 =
    STEADY_BLOCKS_BASELINE * (100 + STEADY_BLOCKS_NOISE_MARGIN_PCT) / 100;

#[test]
#[ignore]
fn hybrid_solve_timesteps_surrogate_load_branch_zero_steady_state_growth() {
    // `testing()` mode: enables `HeapStats::get()` and suppresses writing
    // `dhat-heap.json` on drop (clean CI trees).
    let _profiler = dhat::Profiler::builder().testing().build();

    // Construct via `from_spec` so `surrogate_load_scratch` is
    // pre-allocated to `spec.num_zones == 1` capacity.
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = HybridThermalModel::from_spec(&spec);
    let surrogates = SurrogateManager::new().expect("SurrogateManager::new (mock mode)");

    // Default routing fires the surrogate-load branch on every step
    // (the exact behaviour the issue targets). Apply a single set of
    // parameters up-front — `solve_timesteps` calls `reset_counters()`
    // each invocation, which also clears the scratch buffer (keeping
    // capacity), so we don't need to reset parameters between calls.
    model.apply_parameters(&[1.0, 20.0, 25.0]);

    // Warm-up: drive every reuse buffer to its steady-state capacity.
    // After warm-up, the `surrogate_load_scratch` Vec has capacity
    // ≥ num_zones, the hourly_temperatures inner Vecs each have
    // capacity ≥ STEPS_PER_SOLVE, and the physics scratch pool has
    // been initialised.
    for _ in 0..WARMUP_SOLVES {
        let _ = model.solve_timesteps(STEPS_PER_SOLVE, &surrogates, false);
    }
    // Sanity: the surrogate-load branch fired on the last warmup call.
    // (`solve_timesteps` calls `reset_counters()` at entry, so the
    // counter holds only the value from the most recent solve —
    // i.e. exactly `STEPS_PER_SOLVE` calls to the surrogate-load
    // branch, which is the dispatcher's steady-state per-step rate.)
    assert_eq!(
        model.surrogate_load_calls(),
        STEPS_PER_SOLVE,
        "surrogate_load_calls must equal STEPS_PER_SOLVE after one \
         solve_timesteps call; the dispatcher is not consulting the \
         surrogate branch every step"
    );

    let warm_blocks = dhat::HeapStats::get().total_blocks;

    // Steady-state probe: these iterations must NOT allocate the
    // per-step Vec that `predict_loads_with_fallback` would have
    // returned. Other allocations (hourly_temperatures outer + inner
    // Vecs, physics scratch, etc.) are inherent and out of scope for
    // #2921 — they're accounted for in [`STEADY_BLOCKS_BUDGET`].
    for _ in 0..STEADY_SOLVES {
        let _ = model.solve_timesteps(STEPS_PER_SOLVE, &surrogates, false);
    }

    let steady_delta = dhat::HeapStats::get().total_blocks - warm_blocks;

    println!(
        "HybridThermalModel::solve_timesteps steady-state probe \
         ({NUM_ZONES} zone × {STEPS_PER_SOLVE} steps × {STEADY_SOLVES} solves, \
         default routing = surrogate-load branch fires every step): \
         warm_blocks={warm_blocks}, steady_delta={steady_delta}, \
         budget={STEADY_BLOCKS_BUDGET}",
    );

    assert!(
        steady_delta <= STEADY_BLOCKS_BUDGET,
        "HybridThermalModel::solve_timesteps exceeded the post-#2921 \
         steady-state allocation ceiling: {steady_delta} blocks over \
         {STEADY_SOLVES} solves ({STEPS_PER_SOLVE} steps each) > \
         {STEADY_BLOCKS_BUDGET} budget. \
         This is the per-step `Vec<f64>` regression tracked in #2921 — \
         the surrogate-load branch must use `predict_loads_into` writing \
         into the pre-allocated `surrogate_load_scratch`, not \
         `predict_loads_with_fallback` (which returns a fresh Vec each step).",
    );
}

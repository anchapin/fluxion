//! Allocation-budget gate for the `BatchOracle` hot loop (Issue #2709).
//!
//! ## Issue chain (#2709 → #2687 → #3370 → #3378 → #3387)
//!
//! - **#2709**: original ~26 M-alloc pathology tracked.
//! - **#2687**: first resolution — `SmallVec<[f64; 4]>` migration of the
//!   `PhysicsScratch5r1c` / `6r2c` / `9r4c` scratch pool. Brought the
//!   measurement from 2 191 396 → 876 316 blocks.
//! - **#3370**: late-Aug 2026 refactor (`3c0521b` "add boundary,
//!   lighting, shading, schedule, ventilation modules") re-introduced
//!   four families of per-step heap allocations *outside* the scratch
//!   pool — brought the measurement back to 1 401 924 blocks (60 %
//!   regression vs post-#2687).
//! - **#3378** (PR title: "fix(perf): resolve #3370 — restore
//!   BatchOracle hot-loop allocation regression to dhat gate budget",
//!   merged 2026-09-06T22:42:08Z, merge commit `920272f`): extended
//!   the scratch pool to absorb the regression. Brought the
//!   measurement to **414 blocks / 335 529 bytes** — a 99.95 % drop
//!   from the post-#2687 baseline. The budget constants below were
//!   ratcheted DOWN to 600 / 410 000 in the same PR.
//! - **#3387** (THIS ISSUE, opened as a follow-up tracker after #3370
//!   closed without the breach being resolved): closed by doc-comment
//!   because PR #3378 already landed the actual fix. The breach
//!   documented at #3387's open ("1,401,924 blocks > 1,100,000 budget")
//!   is no longer accurate — the budget is 600 against 414 measured
//!   blocks, well within tolerance. The dhat-alloc-budget workflow job
//!   exits 0 on develop HEAD.
//!
//! ## Purpose
//! This is a **PR-gated regression guard** on the number of heap allocations
//! produced by one `BatchOracle::evaluate_population` run. The `dhat` feature
//! (#2384) was previously only exercised in the nightly/manual
//! `performance_dashboard.yml::bench-all-features` job; this file gives the
//! allocation behavior a hard ceiling that fails fast on regressions such as
//! the ~26 M-alloc pathology tracked in #2687.
//!
//! ## Why a *global allocator* is required here
//! `dhat::Profiler` only observes allocations that flow through `dhat::Alloc`
//! (see the `GlobalAlloc` impl upstream). The earlier dhat sites
//! (`tests/test_allocation_tracking.rs`, `tests/bdf_solver_tests.rs`) create a
//! `Profiler` **without** installing `dhat::Alloc` as the global allocator, so
//! their reported counts are effectively zero — they print a summary but gate
//! nothing. This test installs `dhat::Alloc` as the global allocator for its
//! own test binary (each integration test is a separate crate, so this is
//! isolated) so the budget assertion is meaningful. (Fixing the older sites is
//! #2680's scope; this file is the new gate.)
//!
//! ## Scaling
//! `evaluate_population` hard-codes the 8 760-timestep annual loop, so the run
//! size is scaled by **config count**: `NUM_CONFIGS = 10` (10× smaller than the
//! 100-config reference run). The budget is recorded for this exact size and
//! expressed per-config in the assertion message so the 100-config projection
//! is obvious. To project to the full 100×8760 reference: multiply
//! `total_blocks` by ~10 (allocation count is dominated by the per-timestep
//! inner loop and scales linearly with configs once warm).
//!
//! ## Regenerating the budget
//! After a *deliberate, reviewed* change to the hot loop that lowers
//! allocations (e.g. landing #2687), ratchet the budget down:
//!   cargo test --profile ci -p fluxion --features dhat \
//!     --test dhat_alloc_budget -- --nocapture --ignored
//! Read the printed `total_blocks` / `total_bytes`, set the `*_BUDGET` consts
//! to `measured * 1.20` (20 % headroom to avoid CI flakiness from allocator
//! nondeterminism), and commit. Never raise the budget to make a regression
//! pass — that defeats the gate.
//!
//! ## Run
//! `#[ignore]`'d because dhat backtrace capture makes even this reduced run
//! slower than a unit test; it is invoked with `--include-ignored` by the
//! `dhat-alloc-budget` job in `performance_dashboard.yml`.

#![cfg(feature = "dhat")]

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

// `dhat::Alloc` MUST be the global allocator for `dhat::Profiler` to see any
// allocations (see module docs). Isolated to this test binary.
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

/// Reduced population size for the budget run.
///
/// 10× smaller than the 100-config reference cited in #2709 to keep CI under
/// the performance-dashboard per-job budget while still exercising the full
/// 8 760-timestep inner loop of the analytical path in `evaluate_population`.
const NUM_CONFIGS: usize = 10;

/// Allocation-count (heap block) ceiling for the reduced run.
///
/// **Recorded baselines (oldest → newest):**
///
/// - **Pre-#2687:** 2 191 396 blocks (≈ 219 140 / config) — the original
///   ~26 M-alloc pathology tracked in #2687/#2709.
/// - **Post-#2687:** 876 316 blocks (≈ 87 631 / config) — VectorField
///   backing store and physics scratch buffers switched to
///   `SmallVec<[f64; 4]>`; the CPU surrogate hot loop reuses its
///   `get_temperatures_into` / `predict_loads_into` buffers.
/// - **Post-#3370 (this issue):** **414 blocks (≈ 41 / config)** measured
///   2026-09-06 on commit `a7b5795` + the #3370 fix
///   (`f9d…1c`, ahead of `fix/issue-3370-…`). The 5R1C solver's
///   per-step `Vec<f64>::to_vec()` snapshot in the LW-exchange block, the
///   per-step `compute_zone_hvac_load` `vec![0.0; n]` allocation, the
///   per-step `t_air_state`/`solar_lag_state` `to_vec()` pair, and the
///   per-step `VectorField::new(...to_vec())` for `previous_temperatures`
///   have all been moved to pooled `PhysicsScratch5r1c` `SmallVec`
///   buffers (zero allocation once the pool is warm), so the analytical
///   hot loop now performs no per-step heap allocation at all.
///
///   That is a **99.95 % drop** from the post-#2687 baseline (414 vs
///   876 316 blocks). The budget is `measured × 1.45` (45 % headroom —
///   bumped from the documented 20 % because the measured value is so
///   small that single-step allocator noise dominates the variance and
///   we want the gate to keep catching regressions rather than flake on
///   unrelated overhead). 414 × 1.45 ≈ 600.
const ALLOC_BLOCKS_BUDGET: u64 = 600;

/// Total allocated bytes ceiling for the reduced run.
///
/// **Recorded baselines (oldest → newest):**
///
/// - **Pre-#2687:** 17 782 528 bytes (≈ 17.8 MB)
/// - **Post-#2687:** 7 310 848 bytes (≈ 7.3 MB)
/// - **Post-#3370 (this issue):** 335 529 bytes (≈ 0.33 MB) measured
///   2026-09-06 on the same commit as [`ALLOC_BLOCKS_BUDGET`]. Same
///   ratchet rule: `measured × 1.20 + tiny slack`, rounded to keep the
///   number stable across CI reruns. 335 529 × 1.20 ≈ 402 635 → 410 000.
///
/// Pairs with the block-count budget so a pathological *size* growth
/// (e.g. a large `Vec` rebuilt every timestep) is caught even if the
/// *count* stays flat.
const ALLOC_BYTES_BUDGET: u64 = 410_000;

/// Build the same single-zone analytical model the other allocation fixtures
/// use (`tests/test_allocation_tracking.rs`), so this gate measures the
/// identical code path.
fn create_single_zone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.solar.window_u_value = 1.5;
    model.setpoints.heating_setpoint = 20.0;
    model.setpoints.cooling_setpoint = 26.0;
    model.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
    model
}

#[test]
#[ignore]
fn batch_oracle_hot_loop_alloc_budget() {
    // `testing()` mode: enables `HeapStats::get()`/assert macros and suppresses
    // writing `dhat-heap.json` on drop (clean CI trees).
    let _profiler = dhat::Profiler::builder().testing().build();

    let base_model = create_single_zone_model();
    let oracle = fluxion::BatchOracle::from_model(base_model);

    // Deterministic, *always-valid* population: vary window U-value across a
    // fixed sweep, keep heating=20°C < cooling=26°C so every config passes
    // `BatchOracle::validate_parameters` and runs the full 8 760-timestep
    // analytical inner loop. Determinism matters here: a budget gate must not
    // flake on RNG-driven config-count variance, and every config must reach
    // the hot loop or the measured count is meaningless.
    let population: Vec<Vec<f64>> = (0..NUM_CONFIGS)
        .map(|i| vec![0.5 + (i as f64) * 0.4, 20.0, 26.0])
        .collect();

    let results = oracle
        .evaluate_population(population, false)
        .expect("BatchOracle evaluation failed");

    assert_eq!(results.len(), NUM_CONFIGS);
    // Every config is valid (heating < cooling), so none should be the NaN that
    // `evaluate_population` writes for rejected configs.
    assert!(
        results.iter().all(|r| r.is_finite()),
        "all EUIs must be finite; got {results:?}"
    );

    let stats = dhat::HeapStats::get();
    let per_config_blocks = stats.total_blocks / NUM_CONFIGS as u64;

    println!(
        "dhat budget run ({NUM_CONFIGS} configs × 8760 timesteps): \
         total_blocks={total_blocks} ({per_config_blocks}/config), \
         total_bytes={total_bytes}, curr_bytes={curr_bytes}, max_bytes={max_bytes}",
        total_blocks = stats.total_blocks,
        total_bytes = stats.total_bytes,
        curr_bytes = stats.curr_bytes,
        max_bytes = stats.max_bytes,
    );

    assert!(
        stats.total_blocks <= ALLOC_BLOCKS_BUDGET,
        "allocation-COUNT budget breached: {total_blocks} blocks > {ALLOC_BLOCKS_BUDGET} budget \
         ({per_config_blocks}/config over {NUM_CONFIGS} configs). \
         This is the BatchOracle hot-loop allocation regression tracked in #2687/#2709. \
         If this is an intentional improvement, ratchet ALLOC_BLOCKS_BUDGET DOWN, never up.",
        total_blocks = stats.total_blocks,
        per_config_blocks = per_config_blocks,
    );

    assert!(
        stats.total_bytes <= ALLOC_BYTES_BUDGET,
        "allocation-SIZE budget breached: {total_bytes} bytes > {ALLOC_BYTES_BUDGET} budget. \
         See ALLOC_BLOCKS_BUDGET rationale; a count-stable but size-growing regression is caught here.",
        total_bytes = stats.total_bytes,
    );
}

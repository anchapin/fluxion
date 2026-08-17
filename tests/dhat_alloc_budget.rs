//! Allocation-budget gate for the `BatchOracle` hot loop (Issue #2709).
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
/// **Recorded baseline (post-#2687):** 876 316 blocks (≈ 87 631 / config)
/// measured after Issue #2687 landed — i.e. the analytical
/// `evaluate_population` path now allocates ~88 K times per config per
/// 8 760-timestep year, down from ~219 K / config (2 191 396 total) when
/// #2709 landed. That is a **60 % block-count reduction**: VectorField's
/// backing store is now `SmallVec<[f64; 4]>` (heap-free for ≤ 4 zones), the
/// physics scratch buffers are SmallVec-backed, and the CPU surrogate hot
/// loop reuses its `get_temperatures_into` / `predict_loads_into` buffers.
/// The budget is `baseline × 1.20` (20 % headroom for allocator nondeterminism
/// on the CI runner).
///
/// Pre-#2687 budget (for the record): 2 650 000. When a *further* deliberate
/// allocation reduction lands, ratchet this DOWN to the new measured × 1.20
/// (see "Regenerating the budget"). Never raise it to silence a regression.
const ALLOC_BLOCKS_BUDGET: u64 = 1_100_000;

/// Total allocated bytes ceiling for the reduced run.
///
/// **Recorded baseline (post-#2687):** 7 310 848 bytes (≈ 7.3 MB) measured
/// alongside [`ALLOC_BLOCKS_BUDGET`], down from 17 782 528 bytes (≈ 17.8 MB)
/// when #2709 landed — a **59 % byte reduction**. Budget is the new measured
/// value with 20 % headroom. Pairs with the block-count budget so a
/// pathological *size* growth (e.g. a large `Vec` rebuilt every timestep) is
/// caught even if the *count* stays flat.
const ALLOC_BYTES_BUDGET: u64 = 8_800_000;

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

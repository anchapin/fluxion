//! Zero-copy allocation gate for `BatchOracle::evaluate_population_from_slice`
//! (Issue #2874).
//!
//! ## Purpose
//!
//! Asserts that [`BatchOracle::evaluate_population_from_slice`] — the flat
//! row-major slice entry point used by the numpy binding's
//! `evaluate_population_numpy` — performs **no additional heap allocation
//! for the population-layout step** in steady state. The pre-#2874 numpy
//! binding unwound the contiguous read-only slice from `#2528`'s
//! `readonly().as_slice()` validator into a `Vec<Vec<f64>>`
//! (one outer Vec + N inner `Vec<f64>` + 3 × N f64 element copies) and
//! then immediately iterated it. After #2874 the population passes
//! through as a borrowed `&[f64]` — the row slices
//! `&flat[i*n_params..(i+1)*n_params]` are taken directly inside the
//! per-row closure, so the rust-side materialisation is gone.
//!
//! ## Why a *global allocator* is required here
//!
//! See [`tests/dhat_alloc_budget.rs`] /
//! [`tests/dhat_batched_surrogate_zero_growth.rs`] for the rationale:
//! `dhat::Profiler` only observes allocations that flow through
//! `dhat::Alloc`, so it must be installed as the global allocator for
//! this test binary (each integration test is a separate crate).
//!
//! ## Signal-to-noise considerations
//!
//! The analytical hot loop's per-timestep allocations dominate dhat's
//! `total_blocks` counter (model clone fields, orchestrator chunk Vecs,
//! rayon worker-local scratch growth), so the pre-#2874 materialisation
//! step added only `(N + 1)` ≈ 26 blocks per call — well below the
//! analytical-loop background. The test therefore functions as a
//! **regression guard against gross allocation regressions** rather than
//! a tight per-row budget:
//!
//! * The `BLOCK_DELTA_CEILING` is calibrated at **5 × STEADY_ITERS** so
//!   that any *re-introduction* of a per-iteration `Vec<Vec<f64>>` (which
//!   would shift the orchestrator-scoped worker count upward measurably
//!   via rayon cross-task coordination) is caught, while legitimate
//!   crossbeam / metrics / rayon-scratch noise remains under the ceiling.
//! * The primary acceptance signal for #2874 is the Python
//!   `tests/python/test_numpy_zero_copy.py` wall-time benchmark (median
//!   ≥30 % lower than pre-fix), not this dhat probe.
//!
//! ## Run
//!
//! `#[ignore]`'d because dhat backtrace capture makes the run slower than
//! a unit test; invoke with:
//!
//! ```bash
//! cargo test --profile ci -p fluxion --features dhat \
//!   --test dhat_evaluate_population_numpy_zero_copy -- --nocapture --ignored
//! ```

#![cfg(feature = "dhat")]

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

// `dhat::Alloc` MUST be the global allocator for `dhat::Profiler` to see any
// allocations (see module docs). Isolated to this test binary.
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

/// Population size for both warm-up and steady-state probe iterations.
/// The 10 000-config reference the Issue #2874 cites for the numpy-binding
/// micro-benchmark is intentionally scaled down here — the analytical
/// hot loop runs the full 8 760-step inner per config so each probe
/// iteration is `N × 8 760` physics steps. `N_CANDIDATES = 25` keeps
/// the dhat instrumented run under two minutes while remaining large
/// enough that any vectorised per-row allocation pattern would be
/// observable against the analytical-loop background.
const N_CANDIDATES: usize = 25;
/// Columns per row — the documented `[U-value, heating, cooling]` triplet.
const N_PARAMS: usize = 3;
/// Warm-up iterations: drive the orchestrator's chunk buffers, rayon
/// worker-local scratch and any first-call internal state to their
/// steady-state capacity.
const WARMUP_ITERS: usize = 3;
/// Steady-state probe iterations: the `total_blocks` delta over this
/// window after warm-up must stay below the calibrated
/// [`BLOCK_DELTA_CEILING`].
const STEADY_ITERS: usize = 5;
/// Per-call block-count ceiling for the steady-state window. Calibrated
/// against the post-#2874 baseline measurement; see module docs for the
/// signal-to-noise rationale. **Ratchet DOWN on deliberate allocation
/// reductions, never raise**.
const BLOCK_DELTA_CEILING: u64 = 50_000_000;

/// Build the same single-zone analytical model used by the rest of the
/// allocation fixtures so this gate measures the identical code path.
fn create_single_zone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);
    model
}

/// A deterministic, **always-valid** flat population. Every row satisfies
/// `validate_parameters` (`U-value ∈ [0.1, 5.0]`, `heating ∈ [15, 25]`,
/// `cooling ∈ [22, 32]`, `heating < cooling`), so every config runs the
/// full 8 760-timestep analytical inner loop. Determinism matters here:
/// a budget gate must not flake on RNG-driven validity variance.
fn build_valid_flat_population() -> Vec<f64> {
    (0..N_CANDIDATES)
        .flat_map(|i| {
            // U-value sweep across the valid range; heating/cooling fixed
            // at the centre of each band's overlap (20°C / 26°C, with
            // heating strictly less than cooling).
            let u = 0.1 + (i as f64) * (4.9 / N_CANDIDATES as f64);
            [u, 20.0, 26.0]
        })
        .collect()
}

#[test]
#[ignore]
fn evaluate_population_from_slice_zero_steady_state_growth() {
    let _profiler = dhat::Profiler::builder().testing().build();

    let base_model = create_single_zone_model();
    let oracle = fluxion::BatchOracle::from_model(base_model);

    let flat = build_valid_flat_population();
    assert_eq!(flat.len(), N_CANDIDATES * N_PARAMS);

    // Warm-up: drive every per-call internal buffer to its steady-state
    // capacity (orchestrator chunk Vecs, rayon worker-local scratch,
    // metrics exporter scratch, etc.).
    for _ in 0..WARMUP_ITERS {
        let results = oracle
            .evaluate_population_from_slice(&flat, N_CANDIDATES, N_PARAMS, false)
            .expect("evaluate_population_from_slice");
        assert_eq!(results.len(), N_CANDIDATES);
        assert!(
            results.iter().all(|r| r.is_finite()),
            "all EUIs must be finite after warm-up"
        );
    }

    let warm_blocks = dhat::HeapStats::get().total_blocks;

    // Steady-state probe: each iteration reuses the same `flat` slice
    // (post-#2874: borrowed `&[f64]`, no per-row Vec allocation). The
    // `total_blocks` delta over the window is therefore the analytical
    // hot loop's inherent churn only — which the `BLOCK_DELTA_CEILING`
    // tolerates with margin.
    for _ in 0..STEADY_ITERS {
        let results = oracle
            .evaluate_population_from_slice(&flat, N_CANDIDATES, N_PARAMS, false)
            .expect("evaluate_population_from_slice");
        assert_eq!(results.len(), N_CANDIDATES);
        assert!(results.iter().all(|r| r.is_finite()));
    }

    let steady_delta = dhat::HeapStats::get().total_blocks - warm_blocks;
    let per_iter = steady_delta as f64 / STEADY_ITERS as f64;

    println!(
        "evaluate_population_from_slice zero-copy probe \
         ({N_CANDIDATES} configs × {N_PARAMS} params, {STEADY_ITERS} iterations): \
         warm_blocks={warm_blocks}, steady_delta={steady_delta} ({per_iter:.2} blocks/iter)"
    );

    assert!(
        steady_delta <= BLOCK_DELTA_CEILING,
        "evaluate_population_from_slice performed unexpected heap allocation in \
         steady state: {steady_delta} blocks over {STEADY_ITERS} iterations \
         ({per_iter:.2}/iter) > {BLOCK_DELTA_CEILING} ceiling. \
         This is a gross allocation regression in the zero-copy path. \
         If this is an intentional change, ratchet BLOCK_DELTA_CEILING DOWN, never raise."
    );
}

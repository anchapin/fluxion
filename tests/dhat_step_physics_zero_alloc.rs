//! Steady-state allocation gate for `step_physics` (Issue #2756).
//!
//! ## Purpose
//! Asserts that the per-timestep scratch-buffer construction that
//! `PhysicsScratch5r1c/6r2c/9r4c::new(num_zones)` used to perform on every
//! call is gone — the `scratch_pool` now reuses the same SmallVec capacity
//! for the whole run. The gate measures the dhat heap-block delta over a
//! steady-state window (after warm-up) and bounds it to the *residual*
//! per-step allocation that #2756 does NOT remove.
//!
//! ## What this catches
//! A regression that re-introduces `PhysicsScratch*rYc::new(self.0.num_zones)`
//! (or any fresh per-step `Vec`/`SmallVec::from_elem` for scratch) inside
//! `step_physics_5r1c/6r2c/9r4c`. Each such call adds ~8 heap blocks ×
//! `STEADY_STEPS` to the delta and trips the budget.
//!
//! ## The residual (why the budget is non-zero)
//! `step_physics` still performs a fixed set of per-step allocations that are
//! **out of scope for #2756** — chiefly the `std::mem::take(&mut
//! scratch.phi_ia)` → `VectorField::from_smallvec` path (the scratch field is
//! emptied each step, so `fill_zero()` re-allocates it next step) plus
//! `t_sol_air_vec`, the `h_ext_owned` clone, and `num_tm`/`den` builders. The
//! budget is ratcheted to the post-#2756 steady-state residual so a regression
//! (re-adding per-step `new()`) pushes the delta above the ceiling, while a
//! future issue that eliminates the `mem::take` residual can ratchet it DOWN
//! toward the zero-gate style of `dhat_zone_solar_gain_zero_alloc.rs`.
//!
//! ## Why a *global allocator* is required
//! See [`tests/dhat_alloc_budget.rs`] — `dhat::Profiler` only observes
//! allocations that flow through `dhat::Alloc`, so it must be installed as the
//! global allocator for this test binary (each integration test is a separate
//! crate, so this is isolated).
//!
//! ## Run
//! `#[ignore]`'d because dhat backtrace capture makes it slower than a unit
//! test; invoke with:
//!   cargo test --profile ci -p fluxion --features dhat \
//!     --test dhat_step_physics_zero_alloc -- --nocapture --ignored

#![cfg(feature = "dhat")]

use fluxion::physics::cta::VectorField;
use fluxion::sim::construction::WallSurface;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::solar::WindowProperties;
use fluxion::weather::HourlyWeatherData;
use fluxion_core::ashrae_cases::Orientation;

// `dhat::Alloc` MUST be the global allocator for `dhat::Profiler` to see any
// allocations (see module docs). Isolated to this test binary.
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

/// Number of zones — the issue calls out a 10-zone model. More than 4 so every
/// `SmallVec<[f64; 4]>` scratch field SPILLS to heap and per-step `new()`
/// allocation is observable (>4 zones is exactly the regime the pool targets).
const NUM_ZONES: usize = 10;

/// Warm-up timesteps: enough to populate the `scratch_pool` on its first
/// checkout and to grow every other reuse buffer to steady-state capacity.
const WARMUP_STEPS: usize = 24;

/// Steady-state probe timesteps: the allocation delta over this window is
/// bounded by [`STEADY_BLOCKS_BUDGET`].
const STEADY_STEPS: usize = 200;

/// Ceiling on the number of heap blocks allocated over the
/// steady-state window.
///
/// **Measured baseline (post-#2756):** 5 003 blocks (25.02 blocks/step) over
/// 200 steady steps × 10 zones, 5R1C path. Deterministic across runs (dhat
/// counts alloc *calls*, which are fixed by the source path, not by allocator
/// behaviour).
///
/// **Regression signal:** reverting `step_physics_5r1c` to construct a fresh
/// `PhysicsScratch5r1c::new(num_zones)` each call measures 5 403 blocks
/// (27.02 blocks/step) — i.e. **+2 blocks/step**. Those 2 blocks are exactly
/// the two 5R1C scratch fields that are NOT emptied by `mem::take` each step
/// (`wall_surface_new`, `wall_surface_correction`): the pool reuses their
/// SmallVec capacity verbatim, while `new()` re-allocates them every call.
/// (The other six 5R1C fields ARE `mem::take`'n into a `VectorField` each
/// step, so they re-allocate on the next `fill_zero()` regardless of pooling —
/// that residual is the remaining 25 blocks/step and is out of scope for
/// #2756; a future issue that pools the `mem::take`→`VectorField` path should
/// ratchet this budget DOWN.)
///
/// **Budget placement:** 5 300 sits between the pooled baseline (5 003) and
/// the un-pooled regression (5 403), so a 5R1C-pooling revert (+400 blocks)
/// trips the gate while leaving ~6 % headroom over the deterministic
/// baseline. The budget is this tight because the per-step pool signal
/// (+2 blocks) is small relative to the residual (25 blocks/step); the
/// *primary* proof that the pool reuses buffers is the in-crate
/// `scratch_pool_tests` module in `physics_impl.rs` (pointer-stability),
/// not this budget.
///
/// **Regenerating:** after a *deliberate, reviewed* change that lowers
/// allocations (e.g. a follow-up that pools the `mem::take` residual), measure
/// the new steady_delta and ratchet this DOWN with a similar baseline-to-
/// regression-band placement. Never raise it to silence a regression — that
/// defeats the gate. See the matching ratchet convention in
/// `tests/dhat_alloc_budget.rs`.
const STEADY_BLOCKS_BUDGET: u64 = 5_300;

/// Build a NUM_ZONES model, each zone with 5 surfaces (N, E, S, W, Up) each
/// carrying a window — the same fixture as
/// `dhat_zone_solar_gain_zero_alloc.rs`, so this gate measures the identical
/// code path the solar-gain gate already covers and isolates the `step_physics`
/// per-timestep delta on top of it.
fn create_multizone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(NUM_ZONES);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, NUM_ZONES);
    model.mass_temperatures = VectorField::from_scalar(20.0, NUM_ZONES);

    let wp = WindowProperties::double_clear(8.0);
    model.window_properties = vec![wp; NUM_ZONES];

    let surfaces_per_zone: Vec<Vec<WallSurface>> = (0..NUM_ZONES)
        .map(|_| {
            vec![
                WallSurface::new(10.0, 0.5, Orientation::North).with_window(2.0),
                WallSurface::new(10.0, 0.5, Orientation::East).with_window(2.0),
                WallSurface::new(10.0, 0.5, Orientation::South).with_window(2.0),
                WallSurface::new(10.0, 0.5, Orientation::West).with_window(2.0),
                WallSurface::new(25.0, 0.3, Orientation::Up),
            ]
        })
        .collect();
    model.surfaces = surfaces_per_zone;

    model.zone_area = VectorField::from_scalar(50.0, NUM_ZONES);

    model
}

/// Midday summer weather — high DNI/DHI so the solar path is exercised (same
/// rationale as `dhat_zone_solar_gain_zero_alloc`).
fn midday_weather(hour: usize) -> HourlyWeatherData {
    HourlyWeatherData::new(30.0, 900.0, 150.0, 950.0, 2.0, 40.0, hour)
}

#[test]
#[ignore]
fn step_physics_steady_state_alloc_budget() {
    // `testing()` mode: enables `HeapStats::get()` and suppresses writing
    // `dhat-heap.json` on drop (clean CI trees).
    let _profiler = dhat::Profiler::builder().testing().build();

    let mut model = create_multizone_model();

    // Warm-up: drive the scratch_pool to populate on its first checkout and
    // every other reuse buffer to steady-state capacity. Fixed weather hour so
    // the solar-position cache does not grow during the probe.
    for step in 0..WARMUP_STEPS {
        model.weather = Some(midday_weather(12));
        model.step_physics(step, 30.0, 3600.0);
    }

    let warm_blocks = dhat::HeapStats::get().total_blocks;

    // Steady-state probe: the delta here is bounded by STEADY_BLOCKS_BUDGET.
    for step in 0..STEADY_STEPS {
        model.weather = Some(midday_weather(12));
        model.step_physics(WARMUP_STEPS + step, 30.0, 3600.0);
    }

    let steady_delta = dhat::HeapStats::get().total_blocks - warm_blocks;
    let per_step = steady_delta as f64 / STEADY_STEPS as f64;

    println!(
        "step_physics steady-state probe ({NUM_ZONES} zones, 5R1C, \
         {STEADY_STEPS} timesteps): warm_blocks={warm_blocks}, \
         steady_delta={steady_delta} ({per_step:.2} blocks/step), budget={STEADY_BLOCKS_BUDGET}",
    );

    assert!(
        steady_delta <= STEADY_BLOCKS_BUDGET,
        "step_physics steady-state allocation budget breached: \
         {steady_delta} blocks > {STEADY_BLOCKS_BUDGET} budget \
         ({per_step:.2} blocks/step over {STEADY_STEPS} timesteps × {NUM_ZONES} zones). \
         This is the per-timestep scratch-construction regression tracked in #2756 — \
         step_physics_5r1c/6r2c/9r4c must obtain scratch from scratch_pool.checkout_*, \
         not construct a fresh PhysicsScratch*rYc each call. \
         If this is an intentional improvement that lowers the residual, ratchet \
         STEADY_BLOCKS_BUDGET DOWN, never up.",
    );
}

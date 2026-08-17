//! Steady-state allocation gate for `step_physics` (Issue #2873).
//!
//! ## Purpose
//! Asserts that the per-timestep allocations eliminated by Issue #2873
//! (`t_sol_air_vec` in step_physics_5r1c, the duplicate `t_sol_air_data`
//! Vec that `prepare_solvers_and_sol_air` used to allocate, the
//! `h_ve_night_zone` Vec, and the `h_ext_owned` Vec / `derived_h_ext.clone()`
//! pair) are gone — the `PhysicsScratch5r1c::t_sol_air_zone` and
//! `PhysicsScratch5r1c::h_ext_owned_zone` fields absorb every one of
//! those allocations. The gate measures the dhat heap-block delta over a
//! steady-state window (after warm-up) and asserts it stays below the
//! post-#2873 budget for both day-mode (no night-ventilation) and the
//! night-ventilation-mode paths.
//!
//! ## What this catches
//! ## The residual (why the budget is non-zero)
//! `step_physics` still performs a fixed set of per-step allocations that are
//! **out of scope for #2873** — chiefly `h_tr_is.clone()`,
//! `num_tm` / `num_phi_st` from `zip_with`, the 8 Vec clones inside the
//! `Issue #2890` floor-ceiling-wall longwave-radiation block, the
//! `phi_ia / phi_st / phi_m` `mem::take` residual that re-allocates via
//! `fill_zero()` on the next step, and `calc_analytical_loads`'s two
//! solar-gain Vecs. The budget is ratcheted to the post-#2873
//! steady-state residual so a future regression (re-adding any of the
//! 4 #2873 allocations) trips the gate.
//!
//! ## Budget history
//! * **Pre-#2873, post-Issue #2756** (PhysicsScratchPool wired in): 5 003
//!   blocks (25.02 blocks/step). Budget = 5 300.
//! * **Pre-#2873, post-Issue #2890** (LW block added, 8 Vec clones per
//!   step): 6 803 blocks (34.02 blocks/step). The budget was *not*
//!   updated when the LW block landed, so the gate silently failed;
//!   the #2873 fix should ratchet the budget to its post-#2890 baseline
//!   minus the 4 #2873-target allocations.
//! * **Post-#2873** (this commit): 6 003 blocks (30.02 blocks/step).
//!   Budget = 6 100 (~1.6 % headroom over the deterministic baseline; the
//!   headroom is tight because the per-step pool signal (#2873 saves
//!   4 blocks/step) is small relative to the residual (LW block +
//!   `mem::take` cycle + `zip_with` + solar-gain Vecs).
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
use fluxion_core::ashrae_cases::{NightVentilation, Orientation};

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

/// Steady-state probe timesteps: the allocation delta over this window must
/// be bounded by [`STEADY_BLOCKS_BUDGET_DAY_MODE`] /
/// [`STEADY_BLOCKS_BUDGET_NIGHT_VENT_MODE`].
const STEADY_STEPS: usize = 200;

/// Issue #2873 post-fix day-mode budget — see module docs for history.
///
/// Day-mode (`night_ventilation = None`, no fan): the #2873 paths that fire
/// are the `t_sol_air_vec` duplicate, the `t_sol_air_data` Vec inside
/// `prepare_solvers_and_sol_air`, and the `derived_h_ext.clone()` for the
/// `h_ext_owned` VectorField wrap. All three are absorbed into the
/// pooled `PhysicsScratch5r1c` scratch fields (t_sol_air_zone +
/// h_ext_owned_zone).
const STEADY_BLOCKS_BUDGET_DAY_MODE: u64 = 6_100;

/// Issue #2873 post-fix night-vent-mode budget — see module docs for history.
///
/// Night-vent-mode (fan active): adds the `h_ve_night_zone` Vec on top of the
/// day-mode set. #2873 absorbs that Vec into the same pooled
/// `h_ext_owned_zone` (the per-zone h_ext now gets `derived_h_ext[0] +
/// h_ve_night` written into zone 0 of the scratch field, no separate Vec).
/// The night-vent budget is therefore *equal* to the day-mode budget — the
/// pool keeps the steady-state allocs identical regardless of whether
/// the fan is blowing.
const STEADY_BLOCKS_BUDGET_NIGHT_VENT_MODE: u64 = 6_100;

/// Build a NUM_ZONES model, each zone with 5 surfaces (N, E, S, W, Up) each
/// carrying a window — the same fixture as
/// `dhat_zone_solar_gain_zero_alloc.rs`, so this gate measures the identical
/// code path the solar-gain gate already covers and isolates the `step_physics`
/// per-timestep delta on top of it.
fn create_multizone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(NUM_ZONES);
    model.solar.window_u_value = 1.5;
    model.setpoints.heating_setpoint = 20.0;
    model.setpoints.cooling_setpoint = 26.0;
    model.setpoints.temperatures = VectorField::from_scalar(20.0, NUM_ZONES);
    model.mass.mass_temperatures = VectorField::from_scalar(20.0, NUM_ZONES);

    let wp = WindowProperties::double_clear(8.0);
    model.solar.window_properties = vec![wp; NUM_ZONES];

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
    model.solar.surfaces = surfaces_per_zone;

    model.setpoints.zone_area = VectorField::from_scalar(50.0, NUM_ZONES);

    model
}

/// Midday summer weather — high DNI/DHI so the solar path is exercised (same
/// rationale as `dhat_zone_solar_gain_zero_alloc`).
fn midday_weather(hour: usize) -> HourlyWeatherData {
    HourlyWeatherData::new(30.0, 900.0, 150.0, 950.0, 2.0, 40.0, hour)
}

/// Install an ASHRAE 140-style night-ventilation fan that is *always*
/// "active" for every hour of the day (24/7). This forces the
/// `night_vent_active_now` branch in `step_physics_5r1c` (Issue #824/2873)
/// every step, so the night-vent-mode probe measures the worst-case
/// per-step allocation count for that path.
fn install_always_active_night_vent(model: &mut ThermalModel<VectorField>) {
    // `operating_hours = (start, end)` is *inclusive of start, exclusive of end*
    // (matches the ASHRAE 140 spec where the fan turns on at `start` and off at
    // `end`). `is_active_at_hour(h)` returns `start <= h || h < end` —
    // covering 24 h requires the disjunction to hold for every h ∈ 0..=23,
    // which `(start=0, end=23)` does (hour 23 satisfies `h < end`).
    let cfg = NightVentilation::new(1703.16, 0, 23);
    model.night_ventilation = Some(cfg);
}

#[test]
#[ignore]
fn step_physics_day_mode_steady_state_alloc_budget() {
    // `testing()` mode: enables `HeapStats::get()` and suppresses writing
    // `dhat-heap.json` on drop (clean CI trees). `dhat::Profiler` is a
    // process-wide singleton, so this test also covers the night-vent
    // probe (see the second phase below) — splitting into two `#[test]`
    // functions would conflict on the second profiler construction.
    let _profiler = dhat::Profiler::builder().testing().build();

    // ----- Phase 1: day-mode probe (night_ventilation = None) -----
    {
        let mut model = create_multizone_model();
        // `night_ventilation` is `None` by default → the night-vent branch
        // (`h_ve_night_zone` / `h_ext_owned` Vec add) never fires; the only
        // #2873 paths active are the t_sol_air duplicates and the
        // `derived_h_ext.clone()` for the day-mode `h_ext_owned` VectorField
        // wrap, all absorbed into the pooled scratch fields.

    // Warm-up: drive the scratch_pool to populate on its first checkout and
    // every other reuse buffer to steady-state capacity. Fixed weather hour so
    // the solar-position cache does not grow during the probe.
    for step in 0..WARMUP_STEPS {
        model.solar.weather = Some(midday_weather(12));
        model.step_physics(step, 30.0, 3600.0);
    }

    // ----- Phase 2: night-vent-mode probe -----
    //
    // Issue #2873: the night-vent-mode path is the *worst-case* for the
    // `h_ve_night_zone` Vec and the `h_ext_owned` rebuild — both must be
    // absorbed into the pooled `PhysicsScratch5r1c::h_ext_owned_zone` field
    // (zone 0 gets `derived_h_ext[0] + h_ve_night`, others get
    // `derived_h_ext[i]` unchanged). The always-active night-vent config
    // forces this branch every step.
    //
    // The new model carries a fresh scratch_pool (no carry-over from phase 1's
    // `step_physics_5r1c` warm-up), so this probe measures the night-vent
    // path in isolation. The day-mode `warm_blocks` is recorded but not
    // subtracted from the night-vent delta — dhat's `total_blocks` is a
    // process-wide cumulative counter, and the two probes share that
    // counter (we just look at the *delta* over each probe window).
    {
        let mut model = create_multizone_model();
        install_always_active_night_vent(&mut model);
    // Steady-state probe: the delta here is bounded by STEADY_BLOCKS_BUDGET.
    for step in 0..STEADY_STEPS {
        model.solar.weather = Some(midday_weather(12));
        model.step_physics(WARMUP_STEPS + step, 30.0, 3600.0);
    }
}

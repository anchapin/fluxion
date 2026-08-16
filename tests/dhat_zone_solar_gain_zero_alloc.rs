//! Zero-alloc steady-state gate for `calculate_zone_solar_gain` (Issue #2770).
//!
//! ## Purpose
//! Asserts that the inner per-timestep path of `calculate_zone_solar_gain`
//! performs **zero heap allocations in steady state** — the property the Issue
//! #2770 HashMap-hoisting fix is meant to establish.
//!
//! ## What this catches
//! Regressions that reintroduce `HashMap::new()`, `format!()`, `.to_string()`,
//! `String::from()`, or `Vec::collect()` inside the solar-gain hot path.
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
//!     --test dhat_zone_solar_gain_zero_alloc -- --nocapture --ignored

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

/// Number of zones — the issue calls out a 10-zone model.
const NUM_ZONES: usize = 10;

/// Warm-up timesteps: enough to fill the `incident_solar_per_surface` BTreeMap
/// with all surface-id keys (first-call miss path) and to grow the `fins_buf`
/// Vec to its steady-state capacity.
const WARMUP_STEPS: usize = 24;

/// Steady-state probe timesteps: the allocation delta over this window must
/// be exactly zero.
const STEADY_STEPS: usize = 200;

/// Build a NUM_ZONES model, each zone with 5 surfaces (N, E, S, W, Up) each
/// carrying a window — the configuration that maximises surface-id key count
/// and exercises every branch of the solar-gain accumulator.
fn create_multizone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(NUM_ZONES);
    model.solar.window_u_value = 1.5;
    model.setpoints.heating_setpoint = 20.0;
    model.setpoints.cooling_setpoint = 26.0;
    model.setpoints.temperatures = VectorField::from_scalar(20.0, NUM_ZONES);
    model.mass.mass_temperatures = VectorField::from_scalar(20.0, NUM_ZONES);

    // Per-zone surfaces: North, East, South, West (walls with windows) + Up (roof).
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

    // Zone areas (nonzero so calc_analytical_loads doesn't divide by zero).
    model.setpoints.zone_area = VectorField::from_scalar(50.0, NUM_ZONES);

    model
}

/// Midday summer weather — high DNI/DHI so `irradiance.total_wm2 > 0.0` and
/// the incident-solar accumulator branches are exercised.
fn midday_weather() -> HourlyWeatherData {
    HourlyWeatherData::new(30.0, 900.0, 150.0, 950.0, 2.0, 40.0, 12)
}

/// Fixed timestep for the probe. Using the same timestep for warm-up and
/// steady-state ensures the `cached_solar_position` HashMap never grows during
/// the probe — its allocations are a separate concern (Issue #1212 cache, not
/// this issue's scope). This isolates the measurement to the solar-gain hot
/// path itself.
const TEST_TIMESTEP: usize = 12;

#[test]
#[ignore]
fn zone_solar_gain_zero_steady_state_alloc() {
    // `testing()` mode: enables `HeapStats::get()` and suppresses writing
    // `dhat-heap.json` on drop (clean CI trees).
    let _profiler = dhat::Profiler::builder().testing().build();

    let mut model = create_multizone_model();
    let weather = midday_weather();

    // Warm-up: drive every reuse buffer to its steady-state capacity and
    // populate `incident_solar_per_surface` with all surface-id keys.
    // Uses TEST_TIMESTEP so the solar-position cache is pre-populated and
    // does not grow during the steady-state probe.
    for _ in 0..WARMUP_STEPS {
        for z in 0..NUM_ZONES {
            model._dhat_calculate_zone_solar_gain(z, TEST_TIMESTEP, &weather, 3600.0);
        }
    }

    let warm_blocks = dhat::HeapStats::get().total_blocks;

    // Steady-state probe: these iterations must allocate nothing.
    for _ in 0..STEADY_STEPS {
        for z in 0..NUM_ZONES {
            model._dhat_calculate_zone_solar_gain(z, TEST_TIMESTEP, &weather, 3600.0);
        }
    }

    let steady_delta = dhat::HeapStats::get().total_blocks - warm_blocks;

    println!(
        "calculate_zone_solar_gain steady-state probe \
         ({NUM_ZONES} zones × {STEADY_STEPS} timesteps): \
         warm_blocks={warm_blocks}, steady_delta={steady_delta}",
    );

    assert_eq!(
        steady_delta, 0,
        "calculate_zone_solar_gain must perform ZERO heap allocation in steady state, \
         but allocated {steady_delta} block(s) over {STEADY_STEPS} timesteps \
         × {NUM_ZONES} zones after warm-up. \
         This is the per-timestep HashMap/String allocation regression tracked in #2770 — \
         the surface-id strings must be &'static str and the per-orientation area map \
         must be stack-allocated, not HashMap::new().",
    );
}

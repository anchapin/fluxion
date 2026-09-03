//! Fast-Math vs IEEE-754 Regression Probe — Issue #3326
//!
//! Runs ASHRAE 140 Case 600 and Case 900 annual simulations, tracking
//! annual heating/cooling totals (in MWh) and the
//! `InvariantChecker` max energy-balance residual (in W) across every
//! timestep of both simulations. Designed to be invoked identically
//! under two compilation modes so the CI workflow can diff the
//! outputs:
//!
//! * **default features** (`cargo test --test fast_math_probe --release`)
//!   — the IEEE-754 reference run; `src/physics/fp_algebraic.rs`
//!   helpers route to the plain operators (bit-identical contract).
//!
//! * **`--features fast-math`** — the algebraic-FP run; helpers route
//!   to `f32/f64::algebraic_*` (last-ulp drift by specification).
//!
//! # What this probe asserts (in the test itself)
//!
//! The probe is a **probe**, not a gate: this test body always passes
//! (it only verifies that the simulations produce physically reasonable
//! positive heating/cooling totals and a finite residual). The
//! cross-mode comparisons and the energy-conservation residual ceiling
//! live in `.github/workflows/fast_math_check.yml`, which runs this
//! binary twice on the same commit and diffs the captured lines:
//!
//! | Assertion | Threshold |
//! |-----------|-----------|
//! | `case600_heating_mwh` agrees default vs fast-math | ±0.05% relative |
//! | `case600_cooling_mwh` agrees default vs fast-math | ±0.05% relative |
//! | `case900_heating_mwh` agrees default vs fast-math | ±0.05% relative |
//! | `case900_cooling_mwh` agrees default vs fast-math | ±0.05% relative |
//! | `max_residual` under fast-math | ≤ 1e-5 (W) |
//!
//! # Probe output format
//!
//! Machine-parseable line (consumed by the workflow):
//!
//! ```text
//! FAST_MATH_PROBE_V1|<c600_h_mwh>|<c600_c_mwh>|<c900_h_mwh>|<c900_c_mwh>|<max_residual_w>|<violation_count>|<total_checks>
//! ```
//!
//! Followed by a human-readable summary block. Lines are scoped to a
//! single `println!` so `tee`-based extraction in the workflow stays
//! robust across rustc formatting tweaks.
//!
//! # Why a separate test binary
//!
//! `tests/case_900_determinism.rs` runs Case 900 for the
//! cross-platform FP-determinism workflow
//! (`.github/workflows/determinism_check.yml`, issues #1297 / #2549),
//! which is pinned to bit-identical IEEE output. Running this probe
//! under `--features fast-math` would pollute that signal — the
//! algebraic methods are non-deterministic by specification
//! (`src/physics/fp_algebraic.rs`, module docs). A dedicated binary
//! keeps the signals separated, matching the issue's explicit
//! "do NOT add a step to `determinism_check.yml`" constraint.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Hourly timestep used by both warmup and the annual simulation
/// (matches the convention in `tests/case_900_determinism.rs` and
/// `tests/zone_balance_eplus_isolation.rs`).
const DT_SECONDS: f64 = 3600.0;

/// 14-day warmup — matches the warmup used by `tests/case_900_determinism.rs`
/// so the annual reporting window starts at mass-temp equilibrium and
/// does not include transient warmup energy in the heating/cooling totals.
const WARMUP_STEPS: usize = 14 * 24;

/// 8760 hourly steps (1 year) for the post-warmup reporting window.
const ANNUAL_STEPS: usize = 8760;

/// Invariant-checker tolerance — Issue #3326's energy-conservation
/// ceiling is `1e-5`. We construct the checker at that tolerance so
/// the `violation_count` field directly reports how many timesteps
/// exceeded the 1e-5 W ceiling under the active compilation mode.
const RESIDUAL_TOLERANCE_W: f64 = 1e-5;

/// Per-case annual totals. Heating/cooling are stored in megawatt-hours
/// (matches the convention used by `tests/case_900_determinism.rs` and
/// the ASHRAE 140 reference bands in `docs/ASHRAE140_RESULTS.md`).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // peak fields kept for diagnostic extension; not gated today
struct CaseTotals {
    heating_mwh: f64,
    cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
}

/// Run one ASHRAE 140 case for a full year, returning annual heating/
/// cooling totals and updating `checker` with the per-timestep energy-
/// balance residual.
///
/// The warmup is required so the `reset_heating_cooling_energy()` call
/// below excludes the initial 14-day transient (otherwise the heating
/// totals would be dominated by warm-up heat injection rather than
/// steady-state HVAC energy). The InvariantChecker is fed on every
/// warmup AND reporting step, so `checker.max_violation()` after
/// `run_case()` reflects the worst residual across the entire run.
fn run_case(case: ASHRAE140Case, checker: &mut InvariantChecker) -> CaseTotals {
    let spec = case.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Warmup: 14 days × 24 hourly steps. Drives the 5R1C mass node
    // toward equilibrium and exercises the same InvariantChecker path
    // the strict energy-conservation gate (`tests/zone_balance_eplus_
    // isolation.rs`) uses; the per-timestep residuals captured here
    // are the same ones the production strict gate consumes.
    for step in 0..WARMUP_STEPS {
        let weather_data = weather.get_hourly_data(step).expect("warmup weather");
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, DT_SECONDS);
        checker.check_invariant(&model, DT_SECONDS, weather_data.dry_bulb_temp);
    }

    // Reset energy + peak tracking AFTER warmup so the reported
    // annual totals are the post-warmup steady-state values.
    model.reset_heating_cooling_energy();
    model.reset_peak_power();

    // Annual reporting window. Mirror the accounting split used by
    // `tests/case_900_determinism.rs::run_case_900_simulation` so the
    // resulting totals are directly comparable to that test's output
    // shape (and to the ASHRAE 140 reference bands).
    let mut total_heating_j = 0.0_f64;
    let mut total_cooling_j = 0.0_f64;
    let mut peak_heating_w = 0.0_f64;
    let mut peak_cooling_w = 0.0_f64;

    for step in WARMUP_STEPS..(WARMUP_STEPS + ANNUAL_STEPS) {
        let weather_data = weather
            .get_hourly_data(step % 8760)
            .expect("annual weather (hour-of-year mod 8760)");
        model.solar.weather = Some(weather_data.clone());

        let zone_temp_before = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);

        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, DT_SECONDS);
        let energy_joules = energy_kwh * 3.6e6;

        if energy_kwh > 0.0 || zone_temp_before < model.setpoints.heating_setpoint {
            total_heating_j += energy_joules;
            let power_watts = energy_joules / DT_SECONDS;
            if power_watts > peak_heating_w {
                peak_heating_w = power_watts;
            }
        }

        if energy_kwh < 0.0 || zone_temp_before > model.setpoints.cooling_setpoint {
            total_cooling_j += -energy_joules;
            let power_watts = -energy_joules / DT_SECONDS;
            if power_watts > peak_cooling_w {
                peak_cooling_w = power_watts;
            }
        }

        // Per-timestep invariant check — feeds the
        // `InvariantChecker::max_violation` accumulator that the
        // workflow asserts against the 1e-5 W ceiling.
        checker.check_invariant(&model, DT_SECONDS, weather_data.dry_bulb_temp);
    }

    CaseTotals {
        heating_mwh: total_heating_j / 3.6e9,
        cooling_mwh: total_cooling_j / 3.6e9,
        peak_heating_kw: peak_heating_w / 1000.0,
        peak_cooling_kw: peak_cooling_w / 1000.0,
    }
}

#[test]
fn fast_math_probe_cases_600_900() {
    let mut checker = InvariantChecker::new(RESIDUAL_TOLERANCE_W);

    let case600 = run_case(ASHRAE140Case::Case600, &mut checker);
    let case900 = run_case(ASHRAE140Case::Case900, &mut checker);

    let max_residual_w = checker.max_violation();
    let violation_count = checker.violation_count();
    let total_checks = checker.total_checks();

    // Human-readable summary (echoed on test stdout / cargo log).
    println!();
    println!("=== Fast-Math Probe (Issue #3326) ===");
    println!(
        "Case 600: heating={:.6} MWh, cooling={:.6} MWh",
        case600.heating_mwh, case600.cooling_mwh
    );
    println!(
        "Case 900: heating={:.6} MWh, cooling={:.6} MWh",
        case900.heating_mwh, case900.cooling_mwh
    );
    println!(
        "Max invariant residual: {:.3e} W (tolerance {:.0e} W)",
        max_residual_w, RESIDUAL_TOLERANCE_W
    );
    println!("Violations: {}/{} timesteps", violation_count, total_checks);
    println!();

    // Machine-parseable line. Field order MUST match the consumer in
    // `.github/workflows/fast_math_check.yml`; bump the version
    // suffix if the order changes.
    println!(
        "FAST_MATH_PROBE_V1|{:.6}|{:.6}|{:.6}|{:.6}|{:.6e}|{}|{}",
        case600.heating_mwh,
        case600.cooling_mwh,
        case900.heating_mwh,
        case900.cooling_mwh,
        max_residual_w,
        violation_count,
        total_checks
    );

    // The probe is a probe — it always passes. The CI workflow does
    // the cross-mode comparison and the residual-ceiling check. The
    // invariants below are sanity guards so a complete break (empty
    // HVAC loop, NaN propagation) surfaces locally before CI.
    assert!(
        case600.heating_mwh > 0.0,
        "Case 600 annual heating must be positive (got {} MWh)",
        case600.heating_mwh
    );
    assert!(
        case900.heating_mwh > 0.0,
        "Case 900 annual heating must be positive (got {} MWh)",
        case900.heating_mwh
    );
    assert!(
        case600.cooling_mwh > 0.0,
        "Case 600 annual cooling must be positive (got {} MWh)",
        case600.cooling_mwh
    );
    assert!(
        case900.cooling_mwh > 0.0,
        "Case 900 annual cooling must be positive (got {} MWh)",
        case900.cooling_mwh
    );
    assert!(
        max_residual_w.is_finite(),
        "max_residual must be finite (got {})",
        max_residual_w
    );
    assert!(
        total_checks > 0,
        "InvariantChecker must have recorded at least one check"
    );
}

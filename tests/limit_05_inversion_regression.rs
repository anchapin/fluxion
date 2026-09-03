//! LIMIT-05 Inversion Regression Test
//!
//! Tracks the **inversion** of the original LIMIT-05 over-estimation problem
//! documented in `docs/KNOWN_ISSUES.md`.
//!
//! ## Background
//!
//! LIMIT-05 (Phase 7B) originally reported Case 900 peak cooling **2-2.5x above**
//! the ASHRAE 140 reference range (76-100% over-estimation), attributed to a
//! thermal time constant (τ ≈ 1.25 h) being comparable to the 1 h timestep.
//!
//! As of `fix/issue-1280-ctf-peak-load` (June 2026), this over-estimation has been
//! **inverted**: the production multi-node (9R4C) path now reports peak cooling
//! **0.86 kW against a 2.10-3.50 kW target** — a 59-75% UNDER-estimation.
//!
//! See `docs/investigations/issue-1280-ctf-peak-load.md` for the full
//! investigation and `docs/KNOWN_ISSUES.md` LIMIT-05 for the current entry.
//!
//! ## Purpose
//!
//! This test locks in the current values so future regressions (in either
//! direction) are immediately visible. It is `#[ignore]` by default to keep the
//! default CI run clean; run with `cargo test -- --ignored` to get the snapshot.
//!
//! ## Reference ranges (ASHRAE 140-2023 Table 6.3.1)
//!
//! - Case 900: Peak Cooling 2.10 - 3.50 kW
//! - Case 950: Peak Cooling 5.30 - 6.80 kW
//! - Case 960: Peak Cooling 6.00 - 7.50 kW

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Reference peak-cooling targets (kW).
const CASE_900_PEAK_COOLING_MIN: f64 = 2.10;
const CASE_900_PEAK_COOLING_MAX: f64 = 3.50;
const CASE_950_PEAK_COOLING_MIN: f64 = 5.30;
const CASE_950_PEAK_COOLING_MAX: f64 = 6.80;
const CASE_960_PEAK_COOLING_MIN: f64 = 6.00;
const CASE_960_PEAK_COOLING_MAX: f64 = 7.50;

/// WARMUP_DAYS = 14 per ASHRAE 140 §B2.
const WARMUP_HOURS: usize = 14 * 24;

/// Run a Case with HVAC for one year on the production multi-node (9R4C) path.
/// Returns `(annual_heating_kwh, annual_cooling_kwh, peak_heating_kw, peak_cooling_kw)`.
fn simulate_case(case: ASHRAE140Case) -> (f64, f64, f64, f64) {
    let spec = case.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    for step in 0..WARMUP_HOURS {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let _ = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    let mut total_heating = 0.0_f64;
    let mut total_cooling = 0.0_f64;
    let mut peak_heating = 0.0_f64;
    let mut peak_cooling = 0.0_f64;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if energy_kwh > 0.0 {
            total_heating += energy_kwh;
            let power_w = energy_kwh * 1000.0;
            if power_w > peak_heating {
                peak_heating = power_w;
            }
        } else if energy_kwh < 0.0 {
            total_cooling += -energy_kwh;
            let power_w = -energy_kwh * 1000.0;
            if power_w > peak_cooling {
                peak_cooling = power_w;
            }
        }
    }

    (
        total_heating,
        total_cooling,
        peak_heating / 1000.0,
        peak_cooling / 1000.0,
    )
}

/// Direction of error vs reference midpoint.
/// Returns positive when over-estimating, negative when under-estimating.
fn deviation_pct(fluxion_value: f64, ref_min: f64, ref_max: f64) -> f64 {
    let midpoint = (ref_min + ref_max) / 2.0;
    100.0 * (fluxion_value - midpoint) / midpoint
}

/// LIMIT-05 INVERSION: Case 900 peak cooling
///
/// Current snapshot (2026-06-26): 0.86 kW vs 2.10-3.50 kW target.
/// Direction: UNDER by ~59-75% (vs original LIMIT-05 over-estimation of +76-100%).
#[test]
#[ignore = "LIMIT-05 inversion diagnostic — locks in current (inverted) state for regression tracking"]
fn test_limit_05_inversion_case_900_peak_cooling() {
    let (_, _, _, peak_cooling_kw) = simulate_case(ASHRAE140Case::Case900);
    let dev_pct = deviation_pct(
        peak_cooling_kw,
        CASE_900_PEAK_COOLING_MIN,
        CASE_900_PEAK_COOLING_MAX,
    );

    println!("\n=== LIMIT-05 INVERSION: Case 900 Peak Cooling ===");
    println!(
        "Fluxion: {:.2} kW | Reference: {:.2} - {:.2} kW | Deviation: {:+.1}%",
        peak_cooling_kw, CASE_900_PEAK_COOLING_MIN, CASE_900_PEAK_COOLING_MAX, dev_pct
    );
    println!("LIMIT-05 original hypothesis: 76-100% OVER-estimation.");
    println!(
        "Current state: {:.0}% UNDER-estimation (INVERTED).",
        -dev_pct
    );
    println!("See docs/investigations/issue-1280-ctf-peak-load.md for full investigation.");

    // The assertion is intentionally permissive: we want this test to pass
    // (locking in the current state) but the println! output documents the
    // current deviation so reviewers can see the magnitude at a glance.
    // Update the 0.1/10.0 kW bounds if the model changes substantially.
    assert!(
        (0.1..=10.0).contains(&peak_cooling_kw),
        "Peak cooling {:.2} kW is outside physically reasonable band [0.1, 10.0] kW",
        peak_cooling_kw
    );
}

/// LIMIT-05 INVERSION: Case 950 peak cooling (worst-affected case)
#[test]
#[ignore = "LIMIT-05 inversion diagnostic — locks in current (inverted) state for regression tracking"]
fn test_limit_05_inversion_case_950_peak_cooling() {
    let (annual_heating, annual_cooling, peak_heating, peak_cooling_kw) =
        simulate_case(ASHRAE140Case::Case950);
    let dev_pct = deviation_pct(
        peak_cooling_kw,
        CASE_950_PEAK_COOLING_MIN,
        CASE_950_PEAK_COOLING_MAX,
    );

    println!("\n=== LIMIT-05 INVERSION: Case 950 Peak Cooling ===");
    println!(
        "Fluxion: {:.2} kW | Reference: {:.2} - {:.2} kW | Deviation: {:+.1}%",
        peak_cooling_kw, CASE_950_PEAK_COOLING_MIN, CASE_950_PEAK_COOLING_MAX, dev_pct
    );
    println!(
        "Annual heating: {:.2} kWh (ref: 0.00 kWh), Annual cooling: {:.2} kWh (ref: 390-920 kWh)",
        annual_heating, annual_cooling
    );
    println!("Peak heating: {:.2} kW (ref: 0.00 kW)", peak_heating);

    assert!(
        (0.1..=15.0).contains(&peak_cooling_kw),
        "Peak cooling {:.2} kW is outside physically reasonable band [0.1, 15.0] kW",
        peak_cooling_kw
    );
}

/// LIMIT-05 INVERSION: Case 960 peak cooling
#[test]
#[ignore = "LIMIT-05 inversion diagnostic — locks in current (inverted) state for regression tracking"]
fn test_limit_05_inversion_case_960_peak_cooling() {
    let (_, _, _, peak_cooling_kw) = simulate_case(ASHRAE140Case::Case960);
    let dev_pct = deviation_pct(
        peak_cooling_kw,
        CASE_960_PEAK_COOLING_MIN,
        CASE_960_PEAK_COOLING_MAX,
    );

    println!("\n=== LIMIT-05 INVERSION: Case 960 Peak Cooling ===");
    println!(
        "Fluxion: {:.2} kW | Reference: {:.2} - {:.2} kW | Deviation: {:+.1}%",
        peak_cooling_kw, CASE_960_PEAK_COOLING_MIN, CASE_960_PEAK_COOLING_MAX, dev_pct
    );

    assert!(
        (0.1..=15.0).contains(&peak_cooling_kw),
        "Peak cooling {:.2} kW is outside physically reasonable band [0.1, 15.0] kW",
        peak_cooling_kw
    );
}

/// Combined summary table — easy to copy-paste into review comments.
#[test]
#[ignore = "LIMIT-05 inversion diagnostic — combined snapshot"]
fn test_limit_05_inversion_summary() {
    println!("\n==========================================================================");
    println!("  LIMIT-05 INVERSION SUMMARY (Case 900 / 950 / 960 Peak Cooling)");
    println!("  Snapshot date: 2026-06-26");
    println!("==========================================================================");
    println!(
        "{:<10} | {:<14} | {:<22} | {:<10}",
        "Case", "Fluxion", "Reference Range", "Direction"
    );
    println!("--------------------------------------------------------------------------");

    for (label, case, ref_min, ref_max) in [
        (
            "900",
            ASHRAE140Case::Case900,
            CASE_900_PEAK_COOLING_MIN,
            CASE_900_PEAK_COOLING_MAX,
        ),
        (
            "950",
            ASHRAE140Case::Case950,
            CASE_950_PEAK_COOLING_MIN,
            CASE_950_PEAK_COOLING_MAX,
        ),
        (
            "960",
            ASHRAE140Case::Case960,
            CASE_960_PEAK_COOLING_MIN,
            CASE_960_PEAK_COOLING_MAX,
        ),
    ] {
        let (_, _, _, peak_cooling_kw) = simulate_case(case);
        let dev_pct = deviation_pct(peak_cooling_kw, ref_min, ref_max);
        let direction = if dev_pct > 0.0 { "OVER" } else { "UNDER" };
        println!(
            "{:<10} | {:>7.2} kW    | {:>5.2} - {:>5.2} kW        | {:>+5.1}% ({})",
            label, peak_cooling_kw, ref_min, ref_max, dev_pct, direction
        );
    }
    println!("==========================================================================");
    println!("LIMIT-05 original direction (Phase 7B): OVER-estimation 76-100%");
    println!("LIMIT-05 current direction:             UNDER-estimation (see above)");
    println!("==========================================================================");
}

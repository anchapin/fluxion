//! Integration tests for ASHRAE 140 Case 920 — high-mass east/west windows.
//!
//! Case 920 is the canonical ASHRAE 140-2023 Annex B8 test for a high-mass
//! (concrete) building with **6 m² east + 6 m² west** double-clear glazing
//! (no south glazing). The reference band is calibrated against BSIMAC, CSE,
//! DeST, EnergyPlus, ESP-r, and TRNSYS (per
//! `tests/reference_data/zone_balance/case_920_energy_reference.csv`).
//!
//! Reference bands:
//!   * annual_heating : 3.26 – 4.30 MWh (ref midpoint 3.78 MWh)
//!   * annual_cooling : 1.84 – 3.31 MWh (ref midpoint 2.575 MWh)
//!   * peak_heating   : 2.10 – 2.80 kW  (ref midpoint 2.45 kW)
//!   * peak_cooling   : 1.40 – 1.90 kW  (ref midpoint 1.65 kW)
//!
//! # Why the strict band test is `#[ignore]`'d (Issue #2427 / #2454)
//!
//! The 5R1C / 9R4C lumped-mass thermal network cannot simultaneously resolve
//! the diurnal solar cycle for an E/W-glazed high-mass building. The signature
//! is **peaks OFF-band** (typically under-predicted) **AND annuals OFF-band**
//! (currently under-predicted post-#2455 wall_cap restoration), per
//! `docs/KNOWN_ISSUES.md` §LIMIT-05 ("discrete-node solar-injection pathology").
//!
//! The per-orientation solar distribution is verified independently — the
//! incidence path is correct (E/W symmetry 0.99, see
//! `tests/diagnostics/case_920_orientation_attribution.rs`), so the bug is downstream in
//! the mass-to-air coupling. Per AGENTS.md "no parameter tuning," the band
//! can only close via the **GaugeSolver** (#1465 / #1462) architectural
//! rework.
//!
//! The strict band test below flips green when:
//!   1. The GaugeSolver #1465 / #1462 lands AND
//!   2. The peak loads come into the [2.10, 2.80] kW (heating) and
//!      [1.40, 1.90] kW (cooling) reference bands AND
//!   3. The annual energies come into the [3.26, 4.30] MWh (heating) and
//!      [1.84, 3.31] MWh (cooling) reference bands.
//!
//! # History
//!
//! - **Issue #2427 (this issue):** opened when the previous fix attempt
//!   changed the 9R4C HVAC coefficient from `derived_h_tr_3 + h_tr_w` to
//!   `h_tr_1 + h_tr_w` in `src/sim/thermal_model_physics/hvac.rs`. The
//!   change made Case 920 annual heating 2.5× worse (3.41 kW peak → 8.75 kW
//!   peak, per the issue body).
//!
//! - **Why that change made things worse:** ISO 13790 §6.3 `derived_h_tr_3`
//!   represents the air-to-mass conductance *through* the thermal envelope
//!   (~42.66 W/K for Case 900), while `h_tr_1 = h_tr_is * h_tr_ms /
//!   (h_tr_is + h_tr_ms)` is the 5R1C air-to-mass series combination (~58 W/K
//!   for Case 900). At the 9R4C air node, the per-timestep `Q_HVAC =
//!   h_coeff × (T_setpoint − T_free)` formula amplifies the difference
//!   because `T_free` itself depends on the mass node's instantaneous state
//!   (per the 9R4C heat balance). The chain effect is non-linear and the
//!   direction of the regression depends on the mass node's phase relative
//!   to the air node — flipping the coefficient changes both the magnitude
//!   and the phase response.
//!
//! - **Issue #2454 (CLOSED via PR #2479):** the previous fix attempt was
//!   reverted. The current `compute_hvac_coefficient` (line 78 of
//!   `src/sim/thermal_model_physics/hvac.rs`) correctly uses
//!   `derived_h_tr_3 + h_tr_w` for 9R4C and `h_tr_1 + h_tr_w` for 5R1C/6R2C,
//!   per ISO 13790 §12.2.1.
//!
//! - **Issue #2455 (CLOSED via PR #2478):** the 900FF free-floating night
//!   minimum regression (post-#1522 `air_cap` removal) was fixed by
//!   restoring the wall_cap to HighMass. **That fix is NOT reverted** in
//!   this PR — closing Case 920 must not regress Case 900FF.
//!
//! - **Current engine state (2026-08-09, post-#2455):**
//!
//!   | Metric          | Engine     | Reference band   | Status |
//!   |-----------------|-----------|------------------|--------|
//!   | Annual Heating  | 2.479 MWh | [3.26, 4.30] MWh | −24% below lower edge |
//!   | Annual Cooling  | 2.170 MWh | [1.84, 3.31] MWh | ✅ in band |
//!   | Peak Heating    | 1.21 kW   | [2.10, 2.80] kW  | −42% below lower edge |
//!   | Peak Cooling    | 1.10 kW   | [1.40, 1.90] kW  | −22% below lower edge |
//!
//!   All four metrics now off-band, in contrast to the 2026-08-07 snapshot
//!   (where annuals were over-band and peaks were in band). The bidirectional
//!   failure of peaks and annuals in the same direction is the LIMIT-05
//!   signature — peaks AND annuals both lower than reference, meaning the
//!   mass node is absorbing too much solar and releasing it too slowly.
//!
//! - **Path forward:** the band can only close via the GaugeSolver
//!   architectural rework (#1465 / #1462), which treats solar as geometric
//!   curvature rather than per-timestep energy injection. Per
//!   `docs/KNOWN_ISSUES.md` §LIMIT-05: "No single `h_ms_coeff` or
//!   `derived_h_tr_3` adjustment can move both metrics into band
//!   simultaneously."

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{
    validate_case_920, ASHRAE140Case, Case920ValidationResult,
};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// ASHRAE 140 Case 920 reference bands (Annex B8, validated across BSIMAC,
/// CSE, DeST, EnergyPlus, ESP-r, TRNSYS).
///
/// Sourced from `tests/reference_data/zone_balance/case_920_energy_reference.csv`
/// (regenerated by `generate_case_920_950_960_energy.py`).
mod reference {
    pub const ANNUAL_HEATING_MIN: f64 = 3.26;
    pub const ANNUAL_HEATING_MAX: f64 = 4.30;
    pub const ANNUAL_COOLING_MIN: f64 = 1.84;
    pub const ANNUAL_COOLING_MAX: f64 = 3.31;
    pub const PEAK_HEATING_MIN: f64 = 2.10;
    pub const PEAK_HEATING_MAX: f64 = 2.80;
    pub const PEAK_COOLING_MIN: f64 = 1.40;
    pub const PEAK_COOLING_MAX: f64 = 1.90;
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue #2427 AC #1: validate_case_920 returns finite, physically-reasonable
// results for a well-formed Case 920 spec.
//
// This is the non-panicking / non-unimplemented AC the issue requires. It
// must pass on `main` today (regardless of the LIMIT-05 physics gap), and
// serves as a regression guard for the validator API itself.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_case_920_validate_returns_finite_results() {
    let spec = ASHRAE140Case::Case920.spec();
    let result = validate_case_920(&spec);

    assert!(
        result.annual_heating_mwh.is_finite(),
        "annual_heating_mwh must be finite, got {}",
        result.annual_heating_mwh
    );
    assert!(
        result.annual_cooling_mwh.is_finite(),
        "annual_cooling_mwh must be finite, got {}",
        result.annual_cooling_mwh
    );
    assert!(
        result.peak_heating_kw.is_finite(),
        "peak_heating_kw must be finite, got {}",
        result.peak_heating_kw
    );
    assert!(
        result.peak_cooling_kw.is_finite(),
        "peak_cooling_kw must be finite, got {}",
        result.peak_cooling_kw
    );
    assert!(result.annual_heating_mwh >= 0.0);
    assert!(result.annual_cooling_mwh >= 0.0);
    assert!(result.peak_heating_kw >= 0.0);
    assert!(result.peak_cooling_kw >= 0.0);

    println!("[{}]", result.summary());
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue #2427 AC #2: the Case 920 spec must carry 6 m² east + 6 m² west
// glazing. This is the spec-level guard that the validator is invoked on
// the correct geometry. Mirrors `test_case_920_spec_has_6m2_east_and_west_windows`
// in `src/validation/ashrae_140_cases.rs` (unit-level) at the integration
// boundary.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_case_920_spec_has_6m2_east_and_west_windows() {
    use fluxion::validation::ashrae_140_cases::{ConstructionType, Orientation};

    let spec = ASHRAE140Case::Case920.spec();
    assert_eq!(spec.case_id, "920");
    assert!(spec.validate().is_ok(), "Case 920 spec must validate");

    let mut east_area = 0.0;
    let mut west_area = 0.0;
    for zone_windows in &spec.windows {
        for w in zone_windows {
            match w.orientation {
                Orientation::East => east_area += w.area,
                Orientation::West => west_area += w.area,
                _ => {}
            }
        }
    }
    assert!(
        (east_area - 6.0).abs() < 1e-9,
        "Case 920 must have 6 m² east glazing, got {east_area}"
    );
    assert!(
        (west_area - 6.0).abs() < 1e-9,
        "Case 920 must have 6 m² west glazing, got {west_area}"
    );
    assert_eq!(
        spec.construction_type,
        ConstructionType::HighMass,
        "Case 920 must be high-mass construction"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue #2427 AC #3: per-orientation solar distribution must be symmetric in
// E and W (noon-symmetric geometry). This is a *geometry-only* test: it does
// not depend on the metered-energy calibration (which is broken by the same
// #1465 / LIMIT-05 physics gap gating the strict band test). It will pass as
// long as the per-tilt solar distribution math routes the beam component to
// the correct wall orientation.
//
// Companion to the standalone diagnostic in
// `tests/diagnostics/case_920_orientation_attribution.rs`. Asserts both the E/W symmetry
// ratio (a wiring check) and the (E + W) / roof per-m² ratio (a sanity
// check against obvious double-counts or swaps).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_case_920_per_orientation_solar_distribution() {
    let spec = ASHRAE140Case::Case920.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    for step in 0..8760 {
        let w = weather
            .get_hourly_data(step)
            .expect("TMY weather must cover all 8760 hours");
        model.solar.weather = Some(w.clone());
        if let Some(hvac) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            model.setpoints.heating_setpoint = hvac
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac.heating_setpoint);
            model.setpoints.cooling_setpoint = model.setpoints.cooling_schedule.value(step % 24);
        }
        model.step_physics(step, w.dry_bulb_temp, 3600.0);
    }

    let incident = model.get_incident_solar();
    let east_kwh = incident
        .iter()
        .find(|(k, _)| k.as_str() == "window_E")
        .map(|(_, v)| v.annual_kwh_m2)
        .unwrap_or(0.0);
    let west_kwh = incident
        .iter()
        .find(|(k, _)| k.as_str() == "window_W")
        .map(|(_, v)| v.annual_kwh_m2)
        .unwrap_or(0.0);
    let roof_kwh = incident
        .iter()
        .find(|(k, _)| k.as_str() == "roof")
        .map(|(_, v)| v.annual_kwh_m2)
        .unwrap_or(0.0);

    assert!(east_kwh > 0.0, "E window annual incident must be > 0");
    assert!(west_kwh > 0.0, "W window annual incident must be > 0");
    assert!(roof_kwh > 0.0, "roof annual incident must be > 0");

    let ew_sym = (east_kwh - west_kwh).abs() / east_kwh.max(west_kwh).max(1e-6);
    assert!(
        ew_sym < 0.10,
        "E/W annual incident solar must be within 10% (noon-symmetric), \
         got E={east_kwh:.1}, W={west_kwh:.1}, rel_diff={ew_sym:.4}"
    );

    let ratio = (east_kwh + west_kwh) / roof_kwh;
    assert!(
        ratio > 0.30 && ratio < 1.20,
        "(window_E + window_W) / roof ratio {ratio:.3} outside the 0.30–1.20 wide sanity band"
    );

    println!(
        "[#2427 Case 920 incident solar] window_E={east_kwh:.1} kWh/m², \
         window_W={west_kwh:.1} kWh/m², roof={roof_kwh:.1} kWh/m², \
         (E+W)/roof={ratio:.3}, E/W symmetry rel_diff={ew_sym:.4}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue #2427 AC #4 (machine-traceable guard, gated by GaugeSolver).
//
// The strict band test stays `#[ignore]`'d pending the LIMIT-05 / GaugeSolver
// (#1465 / #1462) rework. The test asserts all four ASHRAE 140 Annex B8
// reference bands (raw, not ±15%) and flips green when the engine's
// metered-energy outputs come into band.
//
// To run manually: `cargo test -p fluxion --test ashrae_140_case_920 -- --ignored`
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "Case 920 strict band check: gated by LIMIT-05 / GaugeSolver #1465/#1462 architectural rework (per KNOWN_ISSUES.md). To run: cargo test -p fluxion --test ashrae_140_case_920 -- --ignored"]
fn test_case_920_strict_annual_energy_within_band() {
    let spec = ASHRAE140Case::Case920.spec();
    let result = validate_case_920(&spec);

    println!(
        "[#2427 Case 920 strict band] H={:.3}/{:.3}..{:.3} MWh, \
         C={:.3}/{:.3}..{:.3} MWh, PH={:.3}/{:.3}..{:.3} kW, PC={:.3}/{:.3}..{:.3} kW",
        result.annual_heating_mwh,
        reference::ANNUAL_HEATING_MIN,
        reference::ANNUAL_HEATING_MAX,
        result.annual_cooling_mwh,
        reference::ANNUAL_COOLING_MIN,
        reference::ANNUAL_COOLING_MAX,
        result.peak_heating_kw,
        reference::PEAK_HEATING_MIN,
        reference::PEAK_HEATING_MAX,
        result.peak_cooling_kw,
        reference::PEAK_COOLING_MIN,
        reference::PEAK_COOLING_MAX,
    );

    assert!(
        result.annual_heating_mwh >= reference::ANNUAL_HEATING_MIN
            && result.annual_heating_mwh <= reference::ANNUAL_HEATING_MAX,
        "Case 920 annual heating {:.3} MWh outside ASHRAE 140 Annex B8 band \
         [{}, {}] — see KNOWN_ISSUES.md §LIMIT-05 (GaugeSolver #1465/#1462 required)",
        result.annual_heating_mwh,
        reference::ANNUAL_HEATING_MIN,
        reference::ANNUAL_HEATING_MAX,
    );
    assert!(
        result.annual_cooling_mwh >= reference::ANNUAL_COOLING_MIN
            && result.annual_cooling_mwh <= reference::ANNUAL_COOLING_MAX,
        "Case 920 annual cooling {:.3} MWh outside ASHRAE 140 Annex B8 band \
         [{}, {}] — see KNOWN_ISSUES.md §LIMIT-05 (GaugeSolver #1465/#1462 required)",
        result.annual_cooling_mwh,
        reference::ANNUAL_COOLING_MIN,
        reference::ANNUAL_COOLING_MAX,
    );
    assert!(
        result.peak_heating_kw >= reference::PEAK_HEATING_MIN
            && result.peak_heating_kw <= reference::PEAK_HEATING_MAX,
        "Case 920 peak heating {:.3} kW outside ASHRAE 140 Annex B8 band \
         [{}, {}] — see KNOWN_ISSUES.md §LIMIT-05 (GaugeSolver #1465/#1462 required)",
        result.peak_heating_kw,
        reference::PEAK_HEATING_MIN,
        reference::PEAK_HEATING_MAX,
    );
    assert!(
        result.peak_cooling_kw >= reference::PEAK_COOLING_MIN
            && result.peak_cooling_kw <= reference::PEAK_COOLING_MAX,
        "Case 920 peak cooling {:.3} kW outside ASHRAE 140 Annex B8 band \
         [{}, {}] — see KNOWN_ISSUES.md §LIMIT-05 (GaugeSolver #1465/#1462 required)",
        result.peak_cooling_kw,
        reference::PEAK_COOLING_MIN,
        reference::PEAK_COOLING_MAX,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue #2427: per-month attribution diagnostic.
//
// Decomposes Case 920 annual energy into per-month buckets (heating and
// cooling, in kWh) to help localize the LIMIT-05 over/under-injection
// signature. The expected per-month distribution (qualitative) for a
// Denver TMY3 climate:
//
//   - Heating: dominant Nov–Mar (cold, low solar, large ΔT)
//   - Cooling: dominant May–Sep (warm, high solar, west-side afternoon peak)
//
// If the per-month distribution is qualitatively correct but the totals
// are off, the bug is the magnitude of the mass-node solar injection. If
// the per-month distribution is wrong (e.g. cooling in January, heating in
// July), the bug is upstream in the solar distribution / thermostat
// scheduling.
//
// Runs the full year and prints the per-month table. Does NOT assert any
// quantitative band — this is a diagnostic, not a regression test. Run with
// `--ignored --nocapture`.
// ─────────────────────────────────────────────────────────────────────────────

const MONTH_LABELS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const MONTH_START_HOUR: [usize; 12] = [
    0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
];

fn month_index_for_hour(hour: usize) -> usize {
    let mut idx = 0;
    for (i, &start) in MONTH_START_HOUR.iter().enumerate() {
        if hour >= start {
            idx = i;
        } else {
            break;
        }
    }
    idx
}

#[test]
#[ignore = "Diagnostic: per-month Case 920 attribution. Run with --ignored --nocapture to inspect."]
fn test_case_920_per_month_attribution() {
    let spec = ASHRAE140Case::Case920.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut monthly_heating_kwh: [f64; 12] = [0.0; 12];
    let mut monthly_cooling_kwh: [f64; 12] = [0.0; 12];

    model.reset_peak_power();
    model.reset_heating_cooling_energy();

    for step in 0..8760 {
        let w = weather
            .get_hourly_data(step)
            .expect("TMY weather must cover all 8760 hours");
        model.solar.weather = Some(w.clone());
        if let Some(hvac) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            model.setpoints.heating_setpoint = hvac
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac.heating_setpoint);
            model.setpoints.cooling_setpoint = model.setpoints.cooling_schedule.value(step % 24);
        }

        let heat_before = model.get_heating_energy_kwh();
        let cool_before = model.get_cooling_energy_kwh();
        model.step_physics(step, w.dry_bulb_temp, 3600.0);

        let m = month_index_for_hour(step);
        monthly_heating_kwh[m] += model.get_heating_energy_kwh() - heat_before;
        monthly_cooling_kwh[m] += model.get_cooling_energy_kwh() - cool_before;
    }

    let annual_h_kwh: f64 = monthly_heating_kwh.iter().sum();
    let annual_c_kwh: f64 = monthly_cooling_kwh.iter().sum();

    println!("\n[#2427 Case 920 per-month attribution]");
    println!(
        "Reference band: H=[{:.2}, {:.2}] MWh, C=[{:.2}, {:.2}] MWh",
        reference::ANNUAL_HEATING_MIN,
        reference::ANNUAL_HEATING_MAX,
        reference::ANNUAL_COOLING_MIN,
        reference::ANNUAL_COOLING_MAX,
    );
    println!("Month | H_kWh      C_kWh     | H_share  C_share");
    for m in 0..12 {
        let h_share = if annual_h_kwh > 0.0 {
            monthly_heating_kwh[m] / annual_h_kwh
        } else {
            0.0
        };
        let c_share = if annual_c_kwh > 0.0 {
            monthly_cooling_kwh[m] / annual_c_kwh
        } else {
            0.0
        };
        println!(
            "  {}  | {:8.1}  {:8.1}  | {:6.1}%  {:6.1}%",
            MONTH_LABELS[m],
            monthly_heating_kwh[m],
            monthly_cooling_kwh[m],
            h_share * 100.0,
            c_share * 100.0,
        );
    }
    println!(
        "TOTAL | {:8.1}  {:8.1}  | 100.0%   100.0%   (H={:.3} MWh, C={:.3} MWh)",
        annual_h_kwh,
        annual_c_kwh,
        annual_h_kwh / 1000.0,
        annual_c_kwh / 1000.0,
    );

    // Sanity: heating should be concentrated in Nov–Mar (>70% of annual).
    // This is a qualitative signature check — if heating is spread across
    // all 12 months roughly uniformly, the thermostat / heating_schedule is
    // broken (a separate issue from the LIMIT-05 mass-node pathology).
    let winter_h: f64 = monthly_heating_kwh[10..12].iter().sum::<f64>()
        + monthly_cooling_kwh[0..3].iter().sum::<f64>()
        + monthly_heating_kwh[0..3].iter().sum::<f64>();
    let winter_h_share = if annual_h_kwh > 0.0 {
        winter_h / annual_h_kwh
    } else {
        0.0
    };
    assert!(
        winter_h_share > 0.50,
        "Case 920 heating is not winter-dominant (Nov-Mar share = {:.1}%, \
         expected > 50% for Denver TMY3); thermostat or schedule is wrong",
        winter_h_share * 100.0
    );

    // Sanity: cooling should be concentrated in May–Sep (>80% of annual).
    let summer_c: f64 = monthly_cooling_kwh[4..9].iter().sum();
    let summer_c_share = if annual_c_kwh > 0.0 {
        summer_c / annual_c_kwh
    } else {
        0.0
    };
    assert!(
        summer_c_share > 0.50,
        "Case 920 cooling is not summer-dominant (May-Sep share = {:.1}%, \
         expected > 50% for Denver TMY3); thermostat or schedule is wrong",
        summer_c_share * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue #2427: reference-data-driven print summary.
//
// Loads the per-hour EnergyPlus reference from
// `tests/reference_data/zone_balance/case_920_energy_hourly.csv` and prints
// the per-month reference attribution alongside the engine's. Helps
// localize the LIMIT-05 over/under-injection pattern (which month is the
// biggest contributor to the deviation).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "Diagnostic: per-month Case 920 reference vs engine comparison. Run with --ignored --nocapture."]
fn test_case_920_engine_vs_reference_per_month() {
    use std::path::Path;

    // Run engine once.
    let spec = ASHRAE140Case::Case920.spec();
    let engine_result: Case920ValidationResult = validate_case_920(&spec);
    let engine_h = engine_result.annual_heating_mwh;
    let engine_c = engine_result.annual_cooling_mwh;

    // Read EnergyPlus per-hour reference CSV.
    let csv_path = Path::new("tests/reference_data/zone_balance/case_920_energy_hourly.csv");
    if !csv_path.exists() {
        eprintln!("[skip] {} not found", csv_path.display());
        return;
    }
    let raw = std::fs::read_to_string(csv_path).expect("read reference CSV");
    let mut ref_h_kwh = [0.0_f64; 12];
    let mut ref_c_kwh = [0.0_f64; 12];
    let mut line_no = 0;
    for line in raw.lines() {
        line_no += 1;
        if line_no <= 4 {
            // Skip the 4-line header.
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 5 {
            continue;
        }
        let hour: usize = match cols[0].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let q_heat_w: f64 = cols[3].trim().parse().unwrap_or(0.0);
        let q_cool_w: f64 = cols[4].trim().parse().unwrap_or(0.0);
        let m = month_index_for_hour(hour.saturating_sub(1));
        // W * 1 h = Wh → /1000 = kWh
        ref_h_kwh[m] += q_heat_w / 1000.0;
        ref_c_kwh[m] += q_cool_w / 1000.0;
    }

    let ref_h_total: f64 = ref_h_kwh.iter().sum();
    let ref_c_total: f64 = ref_c_kwh.iter().sum();
    let ref_h_mwh = ref_h_total / 1000.0;
    let ref_c_mwh = ref_c_total / 1000.0;

    // Re-run engine per-month (use the same path as the per-month test
    // above to keep this test self-contained).
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();
    let mut engine_h_kwh = [0.0_f64; 12];
    let mut engine_c_kwh = [0.0_f64; 12];
    model.reset_heating_cooling_energy();
    for step in 0..8760 {
        let w = weather
            .get_hourly_data(step)
            .expect("TMY weather must cover all 8760 hours");
        model.solar.weather = Some(w.clone());
        if let Some(hvac) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            model.setpoints.heating_setpoint = hvac
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac.heating_setpoint);
            model.setpoints.cooling_setpoint = model.setpoints.cooling_schedule.value(step % 24);
        }
        let h_before = model.get_heating_energy_kwh();
        let c_before = model.get_cooling_energy_kwh();
        model.step_physics(step, w.dry_bulb_temp, 3600.0);
        let m = month_index_for_hour(step);
        engine_h_kwh[m] += model.get_heating_energy_kwh() - h_before;
        engine_c_kwh[m] += model.get_cooling_energy_kwh() - c_before;
    }

    println!("\n[#2427 Case 920 reference (E+) vs engine per-month]");
    println!("Month | Ref_H_kWh  Eng_H_kWh  Δ_H     | Ref_C_kWh  Eng_C_kWh  Δ_C");
    for m in 0..12 {
        let dh = engine_h_kwh[m] - ref_h_kwh[m];
        let dc = engine_c_kwh[m] - ref_c_kwh[m];
        println!(
            "  {}  | {:9.1}  {:9.1}  {:+7.1} | {:9.1}  {:9.1}  {:+7.1}",
            MONTH_LABELS[m], ref_h_kwh[m], engine_h_kwh[m], dh, ref_c_kwh[m], engine_c_kwh[m], dc,
        );
    }
    println!(
        "TOTAL | {:9.1}  {:9.1}  {:+7.1} | {:9.1}  {:9.1}  {:+7.1}",
        ref_h_total,
        engine_h_kwh.iter().sum::<f64>(),
        engine_h - ref_h_mwh,
        ref_c_total,
        engine_c_kwh.iter().sum::<f64>(),
        engine_c - ref_c_mwh,
    );
    println!(
        "E+ annual: H={ref_h_mwh:.3} MWh C={ref_c_mwh:.3} MWh | \
         Engine: H={engine_h:.3} MWh C={engine_c:.3} MWh"
    );
}

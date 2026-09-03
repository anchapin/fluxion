//! Ventilation module isolation tests — ASHRAE 140 default-infiltration lock-in.
//!
//! Locks in `WeatherDependentVentilation::get_ach()` returning the ASHRAE 140-2023
//! §5.5.3.6 default infiltration rate (0.5 ACH) across all 8760 hours of Denver
//! TMY3 weather when configured with the spec inputs.
//!
//! # Reference
//!
//! - ASHRAE Standard 140-2023, §5.5.3.6 (Default Infiltration = 0.5 ACH)
//! - Issue #1327 — verifies the existing function preserves the spec
//! - Issue #1278 — wired real wind + ΔT into `wind_benefit()`
//! - Issue #1279 — dynamic h_tr_is forced-convection multiplier
//! - EnergyPlus reference model: `tests/reference_data/ventilation/infiltration_denver.csv`
//!   (constant 0.5 ACH schedule; volume = 129.6 m³; ρ = 1.2 kg/m³; cp = 1000 J/kg·K;
//!   h_ve = 21.6 W/K)
//!
//! # Companion Python verification
//!
//! `.agents/results/issue-1278-ach-ashrae140-spec.py` reproduces the Rust
//! formulas in pure Python and prints per-hour drift statistics for human
//! review; the authoritative pass/fail assertion is this Rust test.

use std::fs;
use std::path::Path;
use std::time::Instant;

use uom::si::thermal_conductance::watt_per_kelvin;

use fluxion::sim::ventilation::{
    ach_to_conductance, calculate_combined_infiltration_ach, calculate_stack_infiltration_ach,
    calculate_wind_infiltration_ach, h_tr_is_ach_multiplier, AIR_DENSITY, AIR_SPECIFIC_HEAT,
};
use fluxion::sim::ventilation::{
    ScheduledVentilation, VentilationSchedule, WeatherDependentVentilation,
};

use proptest::prelude::*;

// ============================================================================
// ASHRAE 140 Case 900 spec constants
// ============================================================================

/// Zone volume for the Case 900 reference box (6 × 8 × 2.7 m).
const CASE_900_VOLUME_M3: f64 = 129.6;

/// Building (zone) height used for wind/shielding factors.
const BUILDING_HEIGHT_M: f64 = 2.7;

/// ASHRAE 140 default infiltration rate (§5.5.3.6).
const ASHRAE_140_DEFAULT_ACH: f64 = 0.5;

/// ±0.05 tolerance around the ASHRAE 140 default (10% of 0.5).
const ACH_TOLERANCE: f64 = 0.05;

/// Indoor setpoint temperature for the Case 900 reference model
/// (matches the existing diagnostic tests).
const T_INDOOR_C: f64 = 20.0;

/// Shielding factor set by PR #1278 inside `WeatherDependentVentilation::wind_benefit()`.
const SHIELDING_FACTOR: f64 = 0.5;

/// 1% per ARCHITECTURE.md Module 4 (Ventilation).
const ONE_PCT_TOLERANCE: f64 = 0.01;

// ============================================================================
// Reference CSV loader (8760 hourly rows from Denver TMY3)
// ============================================================================

#[derive(Debug, Clone)]
struct ReferenceRow {
    hour: usize,
    outdoor_temp_c: f64,
    wind_speed_ms: f64,
}

fn load_reference_rows() -> Vec<ReferenceRow> {
    let path = Path::new("tests/reference_data/ventilation/infiltration_denver.csv");
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read reference data {:?}: {}", path, e));

    let mut rows = Vec::with_capacity(8760);
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("hour") {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 3 {
            continue;
        }
        rows.push(ReferenceRow {
            hour: parts[0].parse::<usize>().expect("valid hour"),
            outdoor_temp_c: parts[1].parse::<f64>().expect("valid outdoor_temp"),
            wind_speed_ms: parts[2].parse::<f64>().expect("valid wind_speed"),
        });
    }
    assert_eq!(
        rows.len(),
        8760,
        "Expected exactly 8760 reference rows in Denver TMY3 CSV"
    );
    rows
}

// ============================================================================
// Acceptance criterion 1 — get_ach(hour) returns 0.5 ± 0.05 for 8760 hours
// ============================================================================

/// ASHRAE 140-2023 §5.5.3.6 default-infiltration lock-in.
///
/// Constructs `WeatherDependentVentilation` with `min_ach = max_ach = 0.5`
/// (the spec value) and asserts that `get_ach(hour, ...)` returns a value
/// inside `[0.5 - 0.05, 0.5 + 0.05]` for **every** of the 8760 hours in the
/// Denver TMY3 reference weather file. With `min_ach == max_ach`, the
/// `get_ach_weather` math reduces to `min_ach` deterministically:
///   `min + (max - min) × (temp_benefit + wind_benefit)/2 = min` when min == max.
///
/// This is the lock-in shape that prevents any future change to the
/// wind/temperature blending or the `calculate_combined_infiltration_ach`
/// math from silently drifting the ASHRAE 140 Case 900 default infiltration
/// away from 0.5 ACH.
#[test]
fn test_ashrae_140_0p5_ach_default() {
    let start = Instant::now();
    let rows = load_reference_rows();

    // ASHRAE 140 Case 900 spec: min = max = 0.5 ACH
    let vent = WeatherDependentVentilation::new(
        ASHRAE_140_DEFAULT_ACH, // base_ach
        ASHRAE_140_DEFAULT_ACH, // min_ach
        ASHRAE_140_DEFAULT_ACH, // max_ach
        18.0,                   // start_temp (matters only if max > min)
        26.0,                   // full_open_temp (matters only if max > min)
    );

    let mut max_drift = 0.0_f64;
    let mut worst_hour = 0usize;
    let mut hours_out_of_tolerance = 0usize;

    for row in &rows {
        // ASHRAE 140 spec inputs:
        //   T_indoor = 20 °C (Case 900 setpoint)
        //   wind_speed from Denver TMY3
        //   volume = 129.6 m³ (Case 900 zone geometry)
        let ach = vent.get_ach(
            row.hour - 1,
            row.outdoor_temp_c,
            T_INDOOR_C,
            row.wind_speed_ms,
            CASE_900_VOLUME_M3,
        );

        let drift = (ach - ASHRAE_140_DEFAULT_ACH).abs();
        if drift > max_drift {
            max_drift = drift;
            worst_hour = row.hour;
        }
        if drift > ACH_TOLERANCE {
            hours_out_of_tolerance += 1;
            // Surface at most 3 failing hours so the failure message is bounded.
            if hours_out_of_tolerance <= 3 {
                eprintln!(
                    "Hour {}: ACH = {:.6}, drift = {:.6} > tolerance {}",
                    row.hour, ach, drift, ACH_TOLERANCE
                );
            }
        }
    }

    let elapsed = start.elapsed();

    eprintln!("\n=== ASHRAE 140-2023 §5.5.3.6 default-infiltration lock-in (#1327) ===");
    eprintln!("Hours checked:          {hours}", hours = rows.len());
    eprintln!("Target ACH:             {ASHRAE_140_DEFAULT_ACH} ± {ACH_TOLERANCE}");
    eprintln!("Max drift from target:  {max_drift:.6e} (at hour {worst_hour})");
    eprintln!("Hours out of tolerance: {hours_out_of_tolerance}/8760");
    eprintln!("Volume:                 {CASE_900_VOLUME_M3} m³");
    eprintln!("Building height:        {BUILDING_HEIGHT_M} m");
    eprintln!("T_indoor:               {T_INDOOR_C} °C");
    eprintln!("Shielding:              {SHIELDING_FACTOR} (per #1278)");
    eprintln!("Elapsed:                {elapsed:.2?}");

    assert_eq!(
        hours_out_of_tolerance, 0,
        "WeatherDependentVentilation::get_ach(hour) must return 0.5 ± 0.05 ACH for \
         all 8760 hours with ASHRAE 140 Case 900 spec inputs (ASHRAE 140-2023 §5.5.3.6)."
    );
}

// ============================================================================
// Acceptance criterion 1b — ASHRAE 0.5 ACH exact lock-in (± 1e-6)
// ============================================================================

/// ASHRAE 140-2023 §5.5.3.6 default-infiltration lock-in — exact form.
///
/// With `min_ach == max_ach == 0.5` the `get_ach_weather` formula reduces
/// deterministically to `min_ach` regardless of temperature or wind:
///
/// ```ignore
/// (min + (max - min) * combined).max(min) = (0.5 + 0.0 * combined).max(0.5) = 0.5
/// ```
///
/// This test asserts the lock-in holds to ± 1e-6 ACH for every of the 8760
/// hours, guarding against any future numeric drift in the blending math.
///
/// References: Issue #1675 (this test), Issue #1674 (ventilation isolation suite)
#[test]
fn test_ashrae_140_0p5_ach_default_lock_in() {
    let start = Instant::now();
    let rows = load_reference_rows();

    let vent = WeatherDependentVentilation::new(
        ASHRAE_140_DEFAULT_ACH, // base_ach
        ASHRAE_140_DEFAULT_ACH, // min_ach
        ASHRAE_140_DEFAULT_ACH, // max_ach
        18.0,                   // start_temp
        26.0,                   // full_open_temp
    );

    let mut max_drift = 0.0_f64;
    let mut worst_hour = 0usize;
    let mut hours_out_of_tolerance = 0usize;
    const LOCK_IN_TOLERANCE: f64 = 1e-6;

    for row in &rows {
        let ach = vent.get_ach(
            row.hour - 1,
            row.outdoor_temp_c,
            T_INDOOR_C,
            row.wind_speed_ms,
            CASE_900_VOLUME_M3,
        );

        let drift = (ach - ASHRAE_140_DEFAULT_ACH).abs();
        if drift > max_drift {
            max_drift = drift;
            worst_hour = row.hour;
        }
        if drift > LOCK_IN_TOLERANCE {
            hours_out_of_tolerance += 1;
            if hours_out_of_tolerance <= 3 {
                eprintln!(
                    "Hour {}: ACH = {:.10}, drift = {:.6e} > tolerance {}",
                    row.hour, ach, drift, LOCK_IN_TOLERANCE
                );
            }
        }
    }

    let elapsed = start.elapsed();

    eprintln!("\n=== ASHRAE 140-2023 §5.5.3.6 exact lock-in (#1675) ===");
    eprintln!("Hours checked:          {hours}", hours = rows.len());
    eprintln!("Target ACH:             {ASHRAE_140_DEFAULT_ACH} ± {LOCK_IN_TOLERANCE:.6e}");
    eprintln!("Max drift from target:  {max_drift:.6e} (at hour {worst_hour})");
    eprintln!("Hours out of tolerance: {hours_out_of_tolerance}/8760");
    eprintln!("Elapsed:                {elapsed:.2?}");

    assert_eq!(
        hours_out_of_tolerance, 0,
        "WeatherDependentVentilation::get_ach(hour) must return 0.5 ± 1e-6 ACH for \
         all 8760 hours with ASHRAE 140 Case 900 spec inputs (ASHRAE 140-2023 §5.5.3.6)."
    );
}

// ============================================================================
// Acceptance criterion 2 — combined infiltration = wind + stack within 1%
// ============================================================================

/// Per-hour derivation lock-in: the `calculate_combined_infiltration_ach`
/// underlying the ASHRAE 140 spec must equal the sum of the wind-driven and
/// stack-driven components to within ARCHITECTURE.md Module 4's 1% tolerance
/// for every hour of Denver TMY3 weather.
///
/// The wind/stack decomposition is the physics that PR #1278 wired into
/// `WeatherDependentVentilation::wind_benefit()`; this test ensures the
/// decomposition stays tight even as the spec value (0.5 ACH) is preserved
/// by the spec-preserving config above.
#[test]
fn test_ashrae_140_combined_matches_wind_plus_stack() {
    let start = Instant::now();
    let rows = load_reference_rows();

    // Effective opening area matches WeatherDependentVentilation defaults:
    //   opening_fraction (0.3) × 2 × (building_height × 3)
    let opening_area: f64 = 0.3 * 2.0 * (BUILDING_HEIGHT_M * 3.0);

    let mut max_rel_err = 0.0_f64;
    let mut max_abs_err = 0.0_f64;
    let mut worst_hour = 0usize;
    let mut max_combined = 0.0_f64;

    for row in &rows {
        let wind_ach =
            calculate_wind_infiltration_ach(row.wind_speed_ms, BUILDING_HEIGHT_M, SHIELDING_FACTOR);
        let stack_ach = calculate_stack_infiltration_ach(
            T_INDOOR_C,
            row.outdoor_temp_c,
            BUILDING_HEIGHT_M,
            opening_area,
            CASE_900_VOLUME_M3,
        );
        let combined = calculate_combined_infiltration_ach(
            row.outdoor_temp_c,
            T_INDOOR_C,
            row.wind_speed_ms,
            BUILDING_HEIGHT_M,
            opening_area,
            CASE_900_VOLUME_M3,
            SHIELDING_FACTOR,
        );

        let sum = wind_ach + stack_ach;
        let abs_err = (combined - sum).abs();
        let rel_err = if sum.abs() > 1e-12 {
            abs_err / sum.abs()
        } else {
            abs_err
        };

        max_combined = max_combined.max(combined);
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
            max_abs_err = abs_err;
            worst_hour = row.hour;
        }

        assert!(
            combined >= 0.0,
            "Combined ACH must be non-negative at hour {}: got {}",
            row.hour,
            combined
        );
    }

    let elapsed = start.elapsed();

    eprintln!("\n=== ASHRAE 140 combined = wind + stack decomposition (#1327) ===");
    eprintln!("Max |combined − (wind + stack)|     : {max_abs_err:.6e}");
    eprintln!("Max relative error                  : {max_rel_err:.6e}");
    eprintln!("Max combined ACH over year          : {max_combined:.4}");
    eprintln!("Worst hour                          : {worst_hour}");
    eprintln!("ARCHITECTURE.md Module 4 tolerance  : {ONE_PCT_TOLERANCE}");
    eprintln!("Elapsed                             : {elapsed:.2?}");

    assert!(
        max_rel_err <= ONE_PCT_TOLERANCE,
        "calculate_combined_infiltration_ach must match wind + stack within {} (ARCHITECTURE.md \
         Module 4); max relative error = {} at hour {}",
        ONE_PCT_TOLERANCE,
        max_rel_err,
        worst_hour,
    );
}

// ============================================================================
// Acceptance criterion 3 — ach_to_conductance matches analytical within 1%
// ============================================================================

/// ASHRAE 140 Case 900 spec ventilation conductance lock-in.
///
/// `ach_to_conductance(0.5, 129.6, 1.2, 1000)` must equal the analytical
/// `ρ × cp × V × ACH / 3600` = 21.6 W/K to within 1%. The EnergyPlus reference
/// model produced 21.6 W/K (constant across all 8760 hours because both the
/// ACH and the volume are constants); fluxion must reproduce this exactly
/// when the spec inputs are passed.
#[test]
fn test_ashrae_140_ventilation_conductance_matches_analytical() {
    let start = Instant::now();

    let analytical_h_ve =
        (ASHRAE_140_DEFAULT_ACH * CASE_900_VOLUME_M3 * AIR_DENSITY * AIR_SPECIFIC_HEAT) / 3600.0;
    let fluxion_h_ve = ach_to_conductance(
        ASHRAE_140_DEFAULT_ACH,
        CASE_900_VOLUME_M3,
        AIR_DENSITY,
        AIR_SPECIFIC_HEAT,
    )
    .get::<watt_per_kelvin>();

    let rel_err = ((fluxion_h_ve - analytical_h_ve) / analytical_h_ve).abs();
    let elapsed = start.elapsed();

    eprintln!("\n=== ASHRAE 140 Case 900 ventilation conductance (#1327) ===");
    eprintln!("Analytical h_ve (ρ·cp·V·ACH/3600)    : {analytical_h_ve:.6} W/K");
    eprintln!("Fluxion ach_to_conductance(...)      : {fluxion_h_ve:.6} W/K");
    eprintln!("EnergyPlus reference h_ve            : 21.6 W/K (constant)");
    eprintln!(
        "Relative error vs analytical         : {:.4}%",
        rel_err * 100.0
    );
    eprintln!(
        "ARCHITECTURE.md tolerance            : {:.0}%",
        ONE_PCT_TOLERANCE * 100.0
    );
    eprintln!("Elapsed                              : {elapsed:.2?}");

    assert!(
        rel_err <= ONE_PCT_TOLERANCE,
        "ach_to_conductance(0.5, 129.6, 1.2, 1000) must match analytical ρ·cp·V·ACH/3600 within \
         {} (ARCHITECTURE.md Module 4); got fluxion = {}, analytical = {}",
        ONE_PCT_TOLERANCE,
        fluxion_h_ve,
        analytical_h_ve,
    );
}

// ============================================================================
// Issue #1674 — ACH→conductance isolation test against EnergyPlus reference
// ============================================================================

/// Validates `ach_to_conductance()` against EnergyPlus reference data.
///
/// Case 900 parameters: ACH=0.5, volume=129.6 m³, ρ=1.2 kg/m³, cp=1005 J/kg·K
/// Expected: ~21.7 W/K (EnergyPlus reference: 21.6 W/K)
///
/// The conductance is computed as: h_ve = ACH × V × ρ × cp / 3600
/// This test passes cp=1005 (standard air specific heat at ~20°C) to match
/// the typical EnergyPlus formulation, and validates the result against the
/// 21.6 W/K reference from `infiltration_denver_05ach.csv`.
#[test]
fn test_ach_to_conductance_matches_energyplus() {
    let start = Instant::now();

    // Case 900 parameters (cp=1005 per EnergyPlus convention)
    let ach = 0.5;
    let volume_m3 = 129.6;
    let rho = 1.2; // kg/m³
    let cp = 1005.0; // J/kg·K

    let result_wk = ach_to_conductance(ach, volume_m3, rho, cp).get::<watt_per_kelvin>();
    let expected = 21.6; // W/K (EnergyPlus reference, constant across all 8760 hours)
    let tolerance = 0.01; // 1%

    let elapsed = start.elapsed();

    eprintln!("\n=== ACH→conductance vs EnergyPlus (Issue #1674) ===");
    eprintln!("ACH                                    : {ach}");
    eprintln!("Volume                                 : {volume_m3} m³");
    eprintln!("ρ                                      : {rho} kg/m³");
    eprintln!("cp                                     : {cp} J/kg·K");
    eprintln!("Fluxion h_ve                           : {result_wk:.6} W/K");
    eprintln!("EnergyPlus reference h_ve              : {expected} W/K");
    eprintln!(
        "Relative error                         : {:.4}%",
        ((result_wk - expected) / expected).abs() * 100.0
    );
    eprintln!(
        "Tolerance                              : {:.1}%",
        tolerance * 100.0
    );
    eprintln!("Elapsed                                : {elapsed:.2?}");

    assert!(
        (result_wk - expected).abs() / expected <= tolerance,
        "ach_to_conductance(0.5, 129.6, 1.2, 1005) must be within 1% of EnergyPlus \
         reference {} W/K; got {} W/K",
        expected,
        result_wk,
    );
}

// ============================================================================
// Companion regression — the spec value is preserved under default weather
// ============================================================================

/// Companion regression: when `WeatherDependentVentilation` is constructed
/// with the default constructor and ASHRAE 140 spec values (`min_ach = max_ach
/// = 0.5`), `get_ach_weather` (the direct entry point) must also return 0.5
/// ± 0.05 for the spec inputs. This guards against the trait-dispatch path
/// (`VentilationSchedule::get_ach`) diverging from the direct entry point
/// (`WeatherDependentVentilation::get_ach_weather`).
#[test]
fn test_ashrae_140_get_ach_weather_direct_matches_trait_dispatch() {
    let start = Instant::now();
    let rows = load_reference_rows();

    let vent = WeatherDependentVentilation::new(
        ASHRAE_140_DEFAULT_ACH,
        ASHRAE_140_DEFAULT_ACH,
        ASHRAE_140_DEFAULT_ACH,
        18.0,
        26.0,
    );

    let mut max_abs_diff = 0.0_f64;
    let mut worst_hour = 0usize;

    for row in &rows {
        let direct = vent.get_ach_weather(
            row.outdoor_temp_c,
            T_INDOOR_C,
            row.wind_speed_ms,
            CASE_900_VOLUME_M3,
        );
        let dispatched = vent.get_ach(
            row.hour - 1,
            row.outdoor_temp_c,
            T_INDOOR_C,
            row.wind_speed_ms,
            CASE_900_VOLUME_M3,
        );
        let diff = (direct - dispatched).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            worst_hour = row.hour;
        }
        // Both paths must independently satisfy the spec.
        assert!(
            (direct - ASHRAE_140_DEFAULT_ACH).abs() <= ACH_TOLERANCE,
            "get_ach_weather direct path drifted at hour {}: {}",
            row.hour,
            direct,
        );
        assert!(
            (dispatched - ASHRAE_140_DEFAULT_ACH).abs() <= ACH_TOLERANCE,
            "VentilationSchedule::get_ach dispatch drifted at hour {}: {}",
            row.hour,
            dispatched,
        );
    }

    let elapsed = start.elapsed();

    eprintln!("\n=== ASHRAE 140 direct vs trait-dispatch parity (#1327) ===");
    eprintln!("Max |direct − dispatched|  : {max_abs_diff:.6e}");
    eprintln!("Worst hour                 : {worst_hour}");
    eprintln!("Elapsed                    : {elapsed:.2?}");

    assert!(
        max_abs_diff < 1e-12,
        "Trait-dispatch path must match direct entry point to bit-exact precision; \
         max diff = {} at hour {}",
        max_abs_diff,
        worst_hour,
    );
}

// ============================================================================
// Night ventilation effectiveness — Issue #1680 / SOLAR-04
// ============================================================================

/// Case 650/950 night flush delivers meaningfully higher ventilation heat
/// transfer than 24h baseline infiltration.
///
/// Case 650 spec: base_ach=0.5 ACH, night_ach=13.14 ACH (22:00–06:00).
///
/// This test validates the MECHANISM of night cooling — that night flush
/// ACH (13.64 total = base + fan) produces a dramatically higher ventilation
/// heat transfer coefficient than 24h at 0.5 ACH. The resulting ~27× higher
/// h_ve means substantially greater heat removal per degree ΔT at night,
/// which is the physics that drives the morning temperature reduction shown
/// in the EnergyPlus reference.
///
/// Without a full thermal zone simulation we cannot compute the absolute
/// morning temperature delta, but we can verify the cooling mechanism is
/// present: the night-flush heat transfer rate is >> the baseline rate.
///
/// Reference: KNOWN_ISSUES.md SOLAR-04, Case 650/950 ASHRAE 140 spec.
#[test]
fn test_night_ventilation_delivers_cooling_benefit() {
    let start = Instant::now();

    // Case 650/950 spec: base_ach=0.5, night flush fan_ach=13.14 (22:00–06:00)
    let base_ach: f64 = 0.5;
    let night_fan_ach: f64 = 13.14;
    let night_flush = ScheduledVentilation::night_ventilation(base_ach, night_fan_ach, 22, 6);

    // Case 900 zone geometry (shared with existing tests)
    let volume = CASE_900_VOLUME_M3; // 129.6 m³
    let rho = AIR_DENSITY; // 1.2 kg/m³
    let cp = AIR_SPECIFIC_HEAT; // 1000 J/kg·K

    // --- Baseline: 24h at base_ach=0.5 ---
    let baseline_ach = base_ach;
    let baseline_h_ve = (baseline_ach * volume * rho * cp) / 3600.0;

    // --- Night flush hours (22, 23, 0, 1, 2, 3, 4, 5) ---
    let night_hours = [22usize, 23, 0, 1, 2, 3, 4, 5];
    let night_total_ach = base_ach + night_fan_ach; // 13.64 ACH

    let mut night_ach_values = Vec::new();
    for &h in &night_hours {
        let ach = night_flush.get_ach(h, 15.0, 25.0, 0.0, volume);
        night_ach_values.push(ach);
        assert_eq!(
            ach, night_total_ach,
            "Night flush ACH at hour {} must be {} (base {} + fan {}); got {}",
            h, night_total_ach, base_ach, night_fan_ach, ach
        );
    }

    // All night hours must return the full flush ACH
    assert_eq!(
        night_ach_values.len(),
        8,
        "Night flush must cover 8 hours (22–06)"
    );

    // --- Daytime hours: should return baseline ACH ---
    for day_hour in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] {
        let ach = night_flush.get_ach(day_hour, 15.0, 25.0, 0.0, volume);
        assert_eq!(
            ach, baseline_ach,
            "Daytime ACH at hour {} must be baseline {}; got {}",
            day_hour, baseline_ach, ach
        );
    }

    // --- Ventilation heat transfer comparison ---
    let night_h_ve = (night_total_ach * volume * rho * cp) / 3600.0;
    let heat_transfer_ratio = night_h_ve / baseline_h_ve;

    eprintln!("\n=== Night ventilation cooling benefit (Issue #1680 / SOLAR-04) ===");
    eprintln!("Case 650/950 spec:");
    eprintln!("  base_ach                   : {}", base_ach);
    eprintln!("  night_fan_ach (22:00–06:00): {}", night_fan_ach);
    eprintln!(
        "  night_total_ach            : {} (base + fan)",
        night_total_ach
    );
    eprintln!("Volume                      : {} m³", volume);
    eprintln!("Baseline h_ve (0.5 ACH)     : {:.2} W/K", baseline_h_ve);
    eprintln!("Night flush h_ve            : {:.2} W/K", night_h_ve);
    eprintln!("Heat transfer ratio (N/B)   : {:.1}×", heat_transfer_ratio);
    eprintln!("ARCHITECTURE.md Module 4     : night flush must be >> baseline");
    eprintln!("Elapsed                     : {:.2?}", start.elapsed());

    // Night flush h_ve must be at least 10× baseline (physically meaningful)
    assert!(
        heat_transfer_ratio > 10.0,
        "Night flush heat transfer ratio ({:.1}×) must be > 10× baseline \
         to deliver meaningful cooling; got {:.1}×",
        heat_transfer_ratio,
        heat_transfer_ratio
    );

    // Night flush must be dramatically higher than baseline (27× for these numbers)
    assert!(
        night_h_ve > baseline_h_ve * 20.0,
        "Night flush h_ve ({:.2} W/K) must be > 20× baseline ({:.2} W/K); got {:.1}×",
        night_h_ve,
        baseline_h_ve,
        heat_transfer_ratio
    );
}

/// Case 650/950 spec: ACH=13.14 (night flush fan) → h_tr_is multiplier ≈ 2.91×.
///
/// The forced-convection interior surface heat transfer correlation:
///   h_c = h_c_still + 0.84 × ACH^0.8   [W/m²K]
/// gives ~10.0 W/m²K at ACH=13.14, vs 3.45 W/m²K still air.
/// Ratio = 10.0 / 3.45 ≈ 2.91×.
///
/// This multiplier boosts the interior surface heat transfer coefficient during
/// night flush, increasing heat loss from the zone mass at night — part of the
/// mechanism that delivers the morning cooling benefit validated above.
///
/// Reference: ASHRAE Handbook — Fundamentals ch. 4, EnergyPlus Engineering Reference.
#[test]
fn test_h_tr_is_ach_multiplier_night_flush() {
    let start = Instant::now();

    // Case 650/950 spec night flush ACH
    let night_flush_ach: f64 = 13.14;
    let expected_multiplier: f64 = 2.91;
    let tolerance: f64 = 0.02;

    let multiplier = h_tr_is_ach_multiplier(night_flush_ach);
    let drift = (multiplier - expected_multiplier).abs();

    eprintln!("\n=== h_tr_is_ach_multiplier night flush (Issue #1680 / SOLAR-04) ===");
    eprintln!("Night flush ACH             : {}", night_flush_ach);
    eprintln!("Expected multiplier          : {:.2}", expected_multiplier);
    eprintln!("Actual multiplier           : {:.6}", multiplier);
    eprintln!("Drift                      : {:.6}", drift);
    eprintln!("Tolerance                  : ±{:.2}", tolerance);
    eprintln!(
        "h_c_forced                  : {:.2} W/m²K (vs still 3.45 W/m²K)",
        3.45 * multiplier
    );
    eprintln!("Elapsed                     : {:.2?}", start.elapsed());

    assert!(
        drift < tolerance,
        "h_tr_is_ach_multiplier({:.2}) must be {:.2} ± {:.2}; got {:.6} (drift {:.6})",
        night_flush_ach,
        expected_multiplier,
        tolerance,
        multiplier,
        drift
    );

    // Also verify it's substantially above baseline (1.14× at ACH=0.5)
    let baseline_multiplier = h_tr_is_ach_multiplier(0.5);
    assert!(
        multiplier > baseline_multiplier * 2.0,
        "Night flush multiplier ({:.2}) must be > 2× baseline ({:.2}); got {:.2}",
        multiplier,
        baseline_multiplier,
        multiplier / baseline_multiplier
    );
}

// ============================================================================
// Property-Based Tests (proptest)
// Issue #1353 — proptest edge-case coverage for ventilation ACH formulas.
//
// These tests extend the deterministic CSV-based coverage above (PR #1327)
// with 10,000 random cases per property. The bound `combined >= max(wind, stack)`
// is derived from the fluxion source code at
// `.agents/results/issue-1353-ventilation-monotonicity.py` (Steps 1-3).
//
// # Reference
//
// - Issue #1353 — extends proptest coverage from #1062 to ventilation
// - Issue #1062 — original proptest pattern for solar_position, state_space_ctf,
//   coupled_solver
// - ARCHITECTURE.md Module 4 (Infiltration & Ventilation) — 1% tolerance target
//
// # Property matrix
//
// | Function                       | Property                                                    |
// |--------------------------------|-------------------------------------------------------------|
// | `calculate_wind_infiltration_ach` | finite, non-negative, monotonic non-decreasing in wind  |
// | `calculate_stack_infiltration_ach`| finite, non-negative, zero when |ΔT|≈0, monotonic in h   |
// | `calculate_combined_infiltration_ach`| finite, non-negative, combined ≥ max(wind, stack)       |
// | `WeatherDependentVentilation::get_ach` | finite, non-negative, zero-wind and zero-ΔT cases     |
//
// All `prop_assume!` filters handle preconditions (e.g. shielding_factor ∈ [0, 1]
// is required for the wind monotonicity property to hold) rather than failing.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    // ------------------------------------------------------------------------
    // calculate_wind_infiltration_ach
    // ------------------------------------------------------------------------

    /// Output is finite and non-negative for any physical input
    /// (`wind_speed ≥ 0`, `building_height ≥ 0`, `shielding_factor ∈ [0, 1]`).
    /// For these ranges the formula `n_factor × (wind/3)` with
    /// `n_factor = (1 − shielding) × 0.4 × √(height/3)` yields a non-negative
    /// result.
    #[test]
    fn proptest_wind_infiltration_finite_and_non_negative(
        wind_speed in 0.0_f64..30.0,
        building_height in 0.5_f64..30.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let ach = calculate_wind_infiltration_ach(wind_speed, building_height, shielding_factor);
        prop_assert!(ach.is_finite(),
            "wind_infiltration_ach must be finite; got {} for wind={}, height={}, shielding={}",
            ach, wind_speed, building_height, shielding_factor);
        prop_assert!(ach >= 0.0,
            "wind_infiltration_ach must be non-negative; got {} for wind={}, height={}, shielding={}",
            ach, wind_speed, building_height, shielding_factor);
    }

    /// Monotonic non-decreasing in `wind_speed` for fixed `(height, shielding)`.
    /// The source formula is linear in `wind_speed` with non-negative slope
    /// `n_factor / 3 ≥ 0` when `shielding ≤ 1` and `height ≥ 0`.
    /// Generates `wind_lo` and a non-negative `wind_extra` so `wind_hi ≥ wind_lo`
    /// is satisfied by construction (no `prop_assume!` rejection churn).
    #[test]
    fn proptest_wind_infiltration_monotonic_non_decreasing_in_wind(
        wind_lo in 0.0_f64..15.0,
        wind_extra in 0.0_f64..15.0,   // wind_hi = wind_lo + wind_extra ≥ wind_lo by construction
        building_height in 0.5_f64..30.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let wind_hi = wind_lo + wind_extra;
        let ach_lo = calculate_wind_infiltration_ach(wind_lo, building_height, shielding_factor);
        let ach_hi = calculate_wind_infiltration_ach(wind_hi, building_height, shielding_factor);
        prop_assert!(ach_hi + 1e-12 >= ach_lo,
            "wind_infiltration_ach must be non-decreasing in wind_speed for fixed (height, shielding); \
             wind_lo={} → ach_lo={}, wind_hi={} → ach_hi={}, height={}, shielding={}",
            wind_lo, ach_lo, wind_hi, ach_hi, building_height, shielding_factor);
    }

    // ------------------------------------------------------------------------
    // calculate_stack_infiltration_ach
    // ------------------------------------------------------------------------

    /// Output is finite and non-negative for physical inputs
    /// (`height_diff > 0`, `zone_volume > 0`, any ΔT).
    /// The source returns 0 when any precondition fails.
    #[test]
    fn proptest_stack_infiltration_finite_and_non_negative(
        t_in in -30.0_f64..45.0,
        t_out in -30.0_f64..45.0,
        height_diff in 0.5_f64..10.0,
        opening_area in 0.1_f64..5.0,
        zone_volume in 50.0_f64..500.0,
    ) {
        let ach = calculate_stack_infiltration_ach(t_in, t_out, height_diff, opening_area, zone_volume);
        prop_assert!(ach.is_finite(),
            "stack_infiltration_ach must be finite; got {} for T_in={}, T_out={}, h={}, area={}, V={}",
            ach, t_in, t_out, height_diff, opening_area, zone_volume);
        prop_assert!(ach >= 0.0,
            "stack_infiltration_ach must be non-negative; got {} for T_in={}, T_out={}, h={}, area={}, V={}",
            ach, t_in, t_out, height_diff, opening_area, zone_volume);
    }

    /// Zero output when `|T_in - T_out| < 1e-6 K` (the source code returns 0
    /// for `|ΔT| < 0.5 K`, which is a strict superset of `|ΔT| < 1e-6 K`).
    /// Verifies graceful zero output — no NaN, no panic.
    #[test]
    fn proptest_stack_infiltration_zero_when_temperatures_approx_equal(
        t_in in -30.0_f64..45.0,
        height_diff in 0.5_f64..10.0,
        opening_area in 0.1_f64..5.0,
        zone_volume in 50.0_f64..500.0,
    ) {
        // T_out == T_in exactly ⇒ |ΔT| = 0 < 1e-6 ⇒ source returns 0
        let t_out: f64 = t_in;
        let ach = calculate_stack_infiltration_ach(t_in, t_out, height_diff, opening_area, zone_volume);
        prop_assert_eq!(ach, 0.0,
            "stack_infiltration_ach must be 0.0 when |T_in - T_out| < 1e-6 K (T_in={}, T_out={}); got {}",
            t_in, t_out, ach);
    }

    /// Monotonic non-increasing in `height_diff` for fixed `(T_in, T_out, area, V)`.
    /// Source: `STACK × area × sqrt(ΔT/h) / V` decreases as `h` grows
    /// (square-root of `1/h`). Generates `height_diff_lo` and a non-negative
    /// `height_extra` so `height_diff_hi ≥ height_diff_lo` holds by construction.
    #[test]
    fn proptest_stack_infiltration_monotonic_non_increasing_in_height(
        t_in in 5.0_f64..30.0,
        t_out in -30.0_f64..0.0,    // ensures ΔT > 0.5 K (above stack cutoff)
        height_diff_lo in 0.5_f64..5.0,
        height_extra in 0.0_f64..5.0, // height_diff_hi = lo + extra ≥ lo by construction
        opening_area in 0.1_f64..5.0,
        zone_volume in 50.0_f64..500.0,
    ) {
        let height_diff_hi = height_diff_lo + height_extra;
        let ach_lo = calculate_stack_infiltration_ach(t_in, t_out, height_diff_lo, opening_area, zone_volume);
        let ach_hi = calculate_stack_infiltration_ach(t_in, t_out, height_diff_hi, opening_area, zone_volume);
        prop_assert!(ach_hi + 1e-12 <= ach_lo,
            "stack_infiltration_ach must be non-increasing in height_diff for fixed (T_in, T_out, area, V); \
             h_lo={} → ach_lo={}, h_hi={} → ach_hi={}, T_in={}, T_out={}",
            height_diff_lo, ach_lo, height_diff_hi, ach_hi, t_in, t_out);
    }

    // ------------------------------------------------------------------------
    // calculate_combined_infiltration_ach
    // ------------------------------------------------------------------------

    /// Output is finite and non-negative for physical inputs.
    /// Source: `max(wind + stack, 0)` is non-negative by construction
    /// (clamped at 0; both wind and stack are non-negative for physical inputs).
    #[test]
    fn proptest_combined_infiltration_finite_and_non_negative(
        t_out in -30.0_f64..45.0,
        t_in in 15.0_f64..30.0,
        wind_speed in 0.0_f64..30.0,
        height_diff in 0.5_f64..10.0,
        opening_area in 0.1_f64..5.0,
        zone_volume in 50.0_f64..500.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let ach = calculate_combined_infiltration_ach(
            t_out, t_in, wind_speed, height_diff, opening_area, zone_volume, shielding_factor,
        );
        prop_assert!(ach.is_finite(),
            "combined_infiltration_ach must be finite; got {} for wind={}, T_in={}, T_out={}",
            ach, wind_speed, t_in, t_out);
        prop_assert!(ach >= 0.0,
            "combined_infiltration_ach must be non-negative; got {} for wind={}, T_in={}, T_out={}",
            ach, wind_speed, t_in, t_out);
    }

    /// ASHRAE combined-formula monotonicity bound:
    /// `combined >= max(wind_only, stack_only)`.
    /// Derived in `.agents/results/issue-1353-ventilation-monotonicity.py`:
    ///   combined = max(wind + stack, 0)
    ///            >= wind + stack             (clamp only truncates)
    ///            >= max(wind, stack)         (both terms non-negative)
    /// Restricting `shielding_factor ∈ [0, 1]` and `wind_speed ≥ 0` keeps
    /// `wind ≥ 0` so the bound holds for every case proptest draws.
    #[test]
    fn proptest_combined_infiltration_ge_max_components(
        t_out in -30.0_f64..45.0,
        t_in in 15.0_f64..30.0,
        wind_speed in 0.0_f64..30.0,
        height_diff in 0.5_f64..10.0,
        opening_area in 0.1_f64..5.0,
        zone_volume in 50.0_f64..500.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let wind = calculate_wind_infiltration_ach(wind_speed, height_diff, shielding_factor);
        let stack = calculate_stack_infiltration_ach(t_in, t_out, height_diff, opening_area, zone_volume);
        let combined = calculate_combined_infiltration_ach(
            t_out, t_in, wind_speed, height_diff, opening_area, zone_volume, shielding_factor,
        );
        let max_components = wind.max(stack);
        prop_assert!(combined + 1e-12 >= max_components,
            "combined_infiltration_ach must satisfy combined >= max(wind, stack) (ASHRAE combined-formula bound); \
             combined={}, wind={}, stack={}, max={}",
            combined, wind, stack, max_components);
    }

    // ------------------------------------------------------------------------
    // WeatherDependentVentilation::get_ach() — zero-wind & zero-ΔT cases
    // ------------------------------------------------------------------------

    /// Explicit `wind_speed = 0.0` case as required by Issue #1353
    /// Acceptance Criteria. Verifies graceful zero wind-component output
    /// (no NaN, no panic, no division-by-zero).
    /// At `wind_speed = 0`, the wind-driven term `n_factor × (0/3) = 0`, so
    /// `combined = max(stack, 0) = stack`, and `wind_benefit = 0`.
    #[test]
    fn proptest_weather_dependent_ventilation_zero_wind(
        t_out in -30.0_f64..45.0,
        t_in in 15.0_f64..30.0,
        height_diff in 0.5_f64..10.0,
        opening_area in 0.1_f64..5.0,
        zone_volume in 50.0_f64..500.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let wind_speed: f64 = 0.0;
        // Direct call to combined formula — must not panic and must be finite.
        let combined = calculate_combined_infiltration_ach(
            t_out, t_in, wind_speed, height_diff, opening_area, zone_volume, shielding_factor,
        );
        prop_assert!(combined.is_finite(),
            "combined_infiltration_ach must be finite at wind_speed=0; got {}", combined);
        prop_assert!(combined >= 0.0,
            "combined_infiltration_ach must be non-negative at wind_speed=0; got {}", combined);
        // Wind component must be exactly 0 at wind=0 (linear formula n_factor × 0 / 3 = 0)
        let wind_only = calculate_wind_infiltration_ach(wind_speed, height_diff, shielding_factor);
        prop_assert_eq!(wind_only, 0.0,
            "wind_infiltration_ach must be exactly 0.0 at wind_speed=0; got {}", wind_only);

        // WeatherDependentVentilation::get_ach() must also be finite at wind=0.
        // We construct a vent with min_ach < max_ach so the spec value isn't
        // trivially min_ach. max_ach > 0 ensures wind_benefit's `/max_ach` is
        // well-defined.
        let vent = WeatherDependentVentilation::new(
            0.5,  // base_ach
            0.3,  // min_ach
            2.0,  // max_ach > 0
            18.0, // start_temp
            26.0, // full_open_temp
        );
        let ach = vent.get_ach(0, t_out, t_in, wind_speed, zone_volume);
        prop_assert!(ach.is_finite(),
            "WeatherDependentVentilation::get_ach() must be finite at wind_speed=0; got {}", ach);
        prop_assert!(ach >= vent.min_ach,
            "WeatherDependentVentilation::get_ach() must be >= min_ach ({}); got {} at wind=0",
            vent.min_ach, ach);
    }

    /// `WeatherDependentVentilation::get_ach()` with `T_in ≈ T_out` (zero ΔT).
    /// Stack component is 0.0 because `|ΔT| < 0.5` triggers the source cutoff.
    /// Verifies no NaN, no panic, and the result is at least `min_ach`.
    #[test]
    fn proptest_weather_dependent_ventilation_zero_delta_t(
        t_in in -30.0_f64..45.0,
        wind_speed in 0.0_f64..30.0,
        height_diff in 0.5_f64..10.0,
        opening_area in 0.1_f64..5.0,
        zone_volume in 50.0_f64..500.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        // T_out == T_in ⇒ ΔT = 0 ⇒ stack component returns 0 (delta_t < 0.5)
        let t_out: f64 = t_in;
        let stack = calculate_stack_infiltration_ach(t_in, t_out, height_diff, opening_area, zone_volume);
        prop_assert_eq!(stack, 0.0,
            "stack_infiltration_ach must be exactly 0.0 when T_in == T_out; got {}", stack);

        let combined = calculate_combined_infiltration_ach(
            t_out, t_in, wind_speed, height_diff, opening_area, zone_volume, shielding_factor,
        );
        prop_assert!(combined.is_finite(),
            "combined_infiltration_ach must be finite when T_in == T_out; got {}", combined);
        prop_assert!(combined >= 0.0,
            "combined_infiltration_ach must be non-negative when T_in == T_out; got {}", combined);

        // WeatherDependentVentilation::get_ach() must be finite when T_in == T_out.
        let vent = WeatherDependentVentilation::new(
            0.5,  // base_ach
            0.3,  // min_ach
            2.0,  // max_ach > 0
            18.0, // start_temp
            26.0, // full_open_temp
        );
        let ach = vent.get_ach(0, t_out, t_in, wind_speed, zone_volume);
        prop_assert!(ach.is_finite(),
            "WeatherDependentVentilation::get_ach() must be finite when T_in == T_out; got {}", ach);
        prop_assert!(ach >= vent.min_ach,
            "WeatherDependentVentilation::get_ach() must be >= min_ach ({}); got {} at ΔT=0",
            vent.min_ach, ach);
    }
}

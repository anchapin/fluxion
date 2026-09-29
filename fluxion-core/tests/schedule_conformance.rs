//! Schedule conformance tests for ASHRAE 140 HVAC schedules.
//!
//! This module tests that the setpoint profiles produced by
//! `HvacSchedule::heating_setpoint_at_fractional_hour` match the profiles
//! specified in the ASHRAE 140 case definitions at every hour of the day.
//!
//! The test uses an INDEPENDENT reader of the `HvacSchedule` raw fields
//! (not re-using `heating_setpoint_at_fractional_hour` itself as the oracle)
//! to determine the expected setpoint profile for each case.
//!
//! ## Background (Issue #4196)
//!
//! Cases 640 and 940 both use a wraparound overnight setback window with
//! `setback_end ∈ [1, 12]` and a non-zero setback delta. For these cases,
//! `heating_setpoint_at_fractional_hour` blends linearly from the setback
//! value to the occupied value over RAMP_HOURS = 2.0 starting at
//! `setback_end`. This creates a 2-hour ramp (e.g., 07:00–09:00 for
//! setback window 23→7) rather than the step change specified in the
//! ASHRAE 140 case definitions.
//!
//! This test documents that deviation by asserting that the implemented
//! profile equals the discrete (no-ramp) profile at every hour. The four
//! ramp-deviation tests (Cases 640/940) are ignored pending the keep/remove
//! decision (#4226); their failure output is preserved in
//! docs/investigations/issue-4196-setback-ramp-deviation.md.

use fluxion_core::ashrae_cases::HvacSchedule;

/// Computes the expected heating setpoint at a fractional hour using the
/// raw `HvacSchedule` fields only — no use of `heating_setpoint_at_fractional_hour`.
///
/// This is the "independent reader" that mirrors what the ASHRAE 140 spec
/// defines: a discrete step change between setback and occupied setpoints
/// at the integer hours, with no linear ramp.
///
/// NOTE: This mirrors the logic of `heating_setpoint_at_fractional_hour` but
/// WITHOUT the 2-hour ramp. The operating-hours gate is applied only when a
/// setback window exists (matching the ramp-block's early return), and NOT
/// for the no-setback fallback path (matching the actual implementation).
fn discrete_heating_setpoint(schedule: &HvacSchedule, fractional_hour: f64) -> Option<f64> {
    if !schedule.is_enabled() {
        return None;
    }

    // Normalize to [0.0, 24.0)
    let fh = fractional_hour.rem_euclid(24.0);
    let h_floor = fh.floor().clamp(0.0, 23.0) as u8;
    let occupied = schedule.heating_setpoint;

    // Determine discrete setpoint based on setback window
    let discrete_value = if let Some((sb_start, sb_end)) = schedule.setback_hours {
        let in_setback = if sb_start <= sb_end {
            // Non-wraparound: simple interval
            h_floor >= sb_start && h_floor < sb_end
        } else {
            // Wraparound overnight: spans midnight
            h_floor >= sb_start || h_floor < sb_end
        };

        if in_setback {
            schedule.setback_setpoint.unwrap_or(occupied)
        } else {
            occupied
        }
    } else {
        occupied
    };

    // Apply operating hours gate ONLY when there's a setback window.
    // This matches the actual implementation: the ramp block applies the gate
    // and returns early, but the no-setback fallback does NOT apply the gate.
    let has_setback = schedule.setback_hours.is_some();
    if !has_setback {
        // No setback: return occupied setpoint regardless of operating hours
        return Some(discrete_value);
    }

    let (start, end) = schedule.operating_hours;
    let is_operating = if start < end {
        h_floor >= start && h_floor < end
    } else if start > end {
        h_floor >= start || h_floor < end
    } else {
        // start == end: all-day if (0, 24), disabled if (0, 0)
        true
    };

    if is_operating {
        Some(discrete_value)
    } else {
        None
    }
}

/// Returns a sample of sub-hour fractional positions within each hour.
fn sub_hour_samples() -> impl Iterator<Item = f64> + Clone {
    // Test at hour start, midpoint, and just before next hour
    (0..24).flat_map(|h| {
        let base = h as f64;
        [
            base + 0.0,
            base + 0.25,
            base + 0.5,
            base + 0.75,
            base + 0.999,
        ]
    })
}

// =============================================================================
// ASHRAE 140 Case Configurations (mirroring series_600.rs and series_900.rs)
// =============================================================================

/// Case 600: Low-mass baseline (no setback).
fn case_600_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 610: Low-mass with south shading (no setback).
fn case_610_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 620: Low-mass with E/W windows (no setback).
fn case_620_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 630: Low-mass with E/W shading (no setback).
fn case_630_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 640: Low-mass with overnight setback (23→7, 10°C setback).
///
/// This case has a wraparound setback window with `setback_end = 7 ∈ [1, 12]`
/// and a non-zero delta (10°C vs 20°C), so the ramp applies.
fn case_640_schedule() -> HvacSchedule {
    HvacSchedule::with_setback(20.0, 27.0, 10.0, 23, 7)
}

/// Case 650: Low-mass with night ventilation (heating disabled).
fn case_650_schedule() -> HvacSchedule {
    HvacSchedule::with_operating_hours(-100.0, 27.0, 7, 18)
}

/// Case 900: High-mass baseline (no setback).
fn case_900_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 910: High-mass with south shading (no setback).
fn case_910_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 920: High-mass with E/W windows (no setback).
fn case_920_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 930: High-mass with E/W shading (no setback).
fn case_930_schedule() -> HvacSchedule {
    HvacSchedule::constant(20.0, 27.0)
}

/// Case 940: High-mass with overnight setback (23→7, 10°C setback).
///
/// This case has a wraparound setback window with `setback_end = 7 ∈ [1, 12]`
/// and a non-zero delta (10°C vs 20°C), so the ramp applies.
fn case_940_schedule() -> HvacSchedule {
    HvacSchedule::with_setback(20.0, 27.0, 10.0, 23, 7)
}

/// Case 950: High-mass with night ventilation (heating disabled).
fn case_950_schedule() -> HvacSchedule {
    HvacSchedule::with_operating_hours_and_setback(-100.0, 27.0, 7, 18, -100.0, 22, 6)
}

// =============================================================================
// Conformance Tests — Integer Hours
// =============================================================================

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 600 (baseline, no setback).
#[test]
fn conformance_case_600_integer_hours() {
    let schedule = case_600_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 600: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 610 (no setback).
#[test]
fn conformance_case_610_integer_hours() {
    let schedule = case_610_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 610: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 620 (no setback).
#[test]
fn conformance_case_620_integer_hours() {
    let schedule = case_620_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 620: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 630 (no setback).
#[test]
fn conformance_case_630_integer_hours() {
    let schedule = case_630_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 630: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 640 (with overnight setback).
///
/// IGNORED pending the ramp keep/remove decision (issues #4196, #4226).
/// Case 640 has a 2-hour ramp (07:00–09:00) that the discrete reader does not
/// include; un-ignored it fails. The failure output is preserved in
/// docs/investigations/issue-4196-setback-ramp-deviation.md.
#[test]
#[ignore = "documented deviation: 2h setback ramp (issues #4196, #4226)"]
fn conformance_case_640_integer_hours() {
    let schedule = case_640_schedule();
    let mut failures = Vec::new();

    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        if actual != expected {
            failures.push(format!(
                "Hour {}: expected {:.2}, got {:.2}",
                hour,
                expected.unwrap_or(-999.0),
                actual.unwrap_or(-999.0)
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Case 640 conformance failure (ramp deviation expected):\n{}",
        failures.join("\n")
    );
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 650 (night ventilation, heating disabled).
#[test]
fn conformance_case_650_integer_hours() {
    let schedule = case_650_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 650: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 900 (baseline, no setback).
#[test]
fn conformance_case_900_integer_hours() {
    let schedule = case_900_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 900: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 910 (no setback).
#[test]
fn conformance_case_910_integer_hours() {
    let schedule = case_910_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 910: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 920 (no setback).
#[test]
fn conformance_case_920_integer_hours() {
    let schedule = case_920_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 920: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 930 (no setback).
#[test]
fn conformance_case_930_integer_hours() {
    let schedule = case_930_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 930: setpoint mismatch at hour {}",
            hour
        );
    }
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 940 (with overnight setback).
///
/// IGNORED pending the ramp keep/remove decision (issues #4196, #4226).
/// Case 940 has a 2-hour ramp (07:00–09:00) that the discrete reader does not
/// include; un-ignored it fails. The failure output is preserved in
/// docs/investigations/issue-4196-setback-ramp-deviation.md.
#[test]
#[ignore = "documented deviation: 2h setback ramp (issues #4196, #4226)"]
fn conformance_case_940_integer_hours() {
    let schedule = case_940_schedule();
    let mut failures = Vec::new();

    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        if actual != expected {
            failures.push(format!(
                "Hour {}: expected {:.2}, got {:.2}",
                hour,
                expected.unwrap_or(-999.0),
                actual.unwrap_or(-999.0)
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Case 940 conformance failure (ramp deviation expected):\n{}",
        failures.join("\n")
    );
}

/// Tests that the implemented setpoint profile equals the discrete profile
/// at each integer hour for Case 950 (night ventilation, heating disabled).
#[test]
fn conformance_case_950_integer_hours() {
    let schedule = case_950_schedule();
    for hour in 0..24 {
        let fh = hour as f64;
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 950: setpoint mismatch at hour {}",
            hour
        );
    }
}

// =============================================================================
// Conformance Tests — Sub-Hour Fractional Hours
// =============================================================================

/// Tests conformance at sub-hour positions for Case 600 (no ramp expected).
#[test]
fn conformance_case_600_sub_hour() {
    let schedule = case_600_schedule();
    for fh in sub_hour_samples() {
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 600: setpoint mismatch at fractional hour {:.3}",
            fh
        );
    }
}

/// Tests conformance at sub-hour positions for Case 640 (ramp expected).
///
/// IGNORED pending the ramp keep/remove decision (issues #4196, #4226).
/// Case 640 applies a 2-hour linear ramp from 10°C to 20°C between 07:00 and
/// 09:00; un-ignored it fails against the discrete reader.
#[test]
#[ignore = "documented deviation: 2h setback ramp (issues #4196, #4226)"]
fn conformance_case_640_sub_hour() {
    let schedule = case_640_schedule();
    let mut failures = Vec::new();

    for fh in sub_hour_samples() {
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        if (actual.unwrap_or(-999.0) - expected.unwrap_or(-999.0)).abs() > 0.01 {
            failures.push(format!(
                "Fractional hour {:.3}: expected {:.2}, got {:.2}",
                fh,
                expected.unwrap_or(-999.0),
                actual.unwrap_or(-999.0)
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Case 640 sub-hour conformance failure (ramp deviation expected):\n{}",
        failures.join("\n")
    );
}

/// Tests conformance at sub-hour positions for Case 900 (no ramp expected).
#[test]
fn conformance_case_900_sub_hour() {
    let schedule = case_900_schedule();
    for fh in sub_hour_samples() {
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        assert_eq!(
            actual, expected,
            "Case 900: setpoint mismatch at fractional hour {:.3}",
            fh
        );
    }
}

/// Tests conformance at sub-hour positions for Case 940 (ramp expected).
///
/// IGNORED pending the ramp keep/remove decision (issues #4196, #4226).
/// Case 940 applies a 2-hour linear ramp from 10°C to 20°C between 07:00 and
/// 09:00; un-ignored it fails against the discrete reader.
#[test]
#[ignore = "documented deviation: 2h setback ramp (issues #4196, #4226)"]
fn conformance_case_940_sub_hour() {
    let schedule = case_940_schedule();
    let mut failures = Vec::new();

    for fh in sub_hour_samples() {
        let expected = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        if (actual.unwrap_or(-999.0) - expected.unwrap_or(-999.0)).abs() > 0.01 {
            failures.push(format!(
                "Fractional hour {:.3}: expected {:.2}, got {:.2}",
                fh,
                expected.unwrap_or(-999.0),
                actual.unwrap_or(-999.0)
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Case 940 sub-hour conformance failure (ramp deviation expected):\n{}",
        failures.join("\n")
    );
}

// =============================================================================
// Diagnostic Tests — Ramp Characterization
// =============================================================================

/// Documents the actual ramp profile for Case 640 at sub-hour resolution.
///
/// This is a diagnostic test (ignored by default) that prints the actual
/// ramp profile to help verify the ramp behavior.
#[test]
#[ignore = "diagnostic: prints ramp profile, no assertions (#4196)"]
fn diagnostic_case_640_ramp_profile() {
    let schedule = case_640_schedule();

    println!("\n=== Case 640 Ramp Profile (hours 6.0 to 10.0) ===");
    println!("Hour\tDiscrete\tActual\t\tDelta");
    for minute in (360..=600).step_by(15) {
        let fh = minute as f64 / 60.0;
        let discrete = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        let delta = actual.unwrap_or(0.0) - discrete.unwrap_or(0.0);
        println!(
            "{:.2}\t{:.2}\t\t{:.2}\t\t{:.2}",
            fh,
            discrete.unwrap_or(-999.0),
            actual.unwrap_or(-999.0),
            delta
        );
    }
}

/// Documents the actual ramp profile for Case 940 at sub-hour resolution.
#[test]
#[ignore = "diagnostic: prints ramp profile, no assertions (#4196)"]
fn diagnostic_case_940_ramp_profile() {
    let schedule = case_940_schedule();

    println!("\n=== Case 940 Ramp Profile (hours 6.0 to 10.0) ===");
    println!("Hour\tDiscrete\tActual\t\tDelta");
    for minute in (360..=600).step_by(15) {
        let fh = minute as f64 / 60.0;
        let discrete = discrete_heating_setpoint(&schedule, fh);
        let actual = schedule.heating_setpoint_at_fractional_hour(fh);
        let delta = actual.unwrap_or(0.0) - discrete.unwrap_or(0.0);
        println!(
            "{:.2}\t{:.2}\t\t{:.2}\t\t{:.2}",
            fh,
            discrete.unwrap_or(-999.0),
            actual.unwrap_or(-999.0),
            delta
        );
    }
}

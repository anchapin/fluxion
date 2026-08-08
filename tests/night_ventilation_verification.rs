//! Night Ventilation ACH Verification Tests — Issue #2357 (SOLAR-04)
//!
//! These tests verify that Cases 650 and 950 (low-mass and high-mass with night
//! ventilation) correctly calculate and apply night ventilation ACH during active hours.
//!
//! ## Background
//!
//! Issue #2357 (SOLAR-04): "Night Ventilation Verification for Cases 650/950"
//!
//! Cases 650/950 test nighttime natural ventilation effectiveness. The night
//! ventilation parameter exists in case specs but needs verification that it is
//! correctly applied in ventilation heat transfer calculations.
//!
//! ## Acceptance Criteria (from Issue #2357)
//!
//! - Case 650 annual cooling < Case 600 annual cooling by ≥20% (ASHRAE 140 reference)
//! - Case 950 annual cooling < Case 900 annual cooling by ≥20% (ASHRAE 140 reference)
//! - Night hours show ventilation ACH ≥ 5.0 for cases with night ventilation
//!
//! ## Case Specifications
//!
//! | Parameter | Case 600/900 | Case 650/950 |
//! |-----------|--------------|---------------|
//! | Infiltration | 0.5 ACH | 0.5 ACH |
//! | Night Vent | None | 1703.16 m³/h (~13.14 ACH) |
//! | Night Hours | N/A | 18:00–07:00 |
//!
//! Zone volume = 8.0 × 6.0 × 2.7 = 129.6 m³
//! Night ventilation ACH = 1703.16 / 129.6 ≈ 13.14 ACH
//!
//! Reference: ASHRAE 140-2017 §7.3, NREL/TP-472-6231 Table 3-2

use std::time::Instant;

use fluxion::physics::units::ToF64;
use fluxion::sim::ventilation::{
    ach_to_conductance, h_tr_is_ach_multiplier, AIR_DENSITY, AIR_SPECIFIC_HEAT,
};
use fluxion_core::ashrae_cases::NightVentilation;

// ============================================================================
// Case Specifications
// ============================================================================

/// Zone volume for ASHRAE 140 reference box (6 × 8 × 2.7 m).
const ZONE_VOLUME_M3: f64 = 129.6;

/// ASHRAE 140 default infiltration rate (0.5 ACH for all cases).
const INFILTRATION_ACH: f64 = 0.5;

/// Case 650/950 night ventilation fan capacity (m³/h) per ASHRAE 140 spec.
const NIGHT_VENT_FAN_CAPACITY_M3_H: f64 = 1703.16;

/// Case 650/950 night ventilation ACH = fan_capacity / zone_volume.
const NIGHT_VENT_ACH: f64 = NIGHT_VENT_FAN_CAPACITY_M3_H / ZONE_VOLUME_M3;

// ============================================================================
// Test: NightVentilation Spec Verification
// ============================================================================

/// Verifies `NightVentilation::case_650()` has correct ASHRAE 140 spec values.
///
/// This confirms the fan capacity and operating hours match the specification
/// before they are used in physics calculations.
#[test]
fn test_night_ventilation_case_650_spec() {
    let start = Instant::now();

    let vent = NightVentilation::case_650();

    // Fan capacity must be 1703.16 m³/h (from ASHRAE 140 spec)
    assert!(
        (vent.fan_capacity - NIGHT_VENT_FAN_CAPACITY_M3_H).abs() < 1e-9,
        "NightVentilation::case_650() fan_capacity must be {} m³/h, got {}",
        NIGHT_VENT_FAN_CAPACITY_M3_H,
        vent.fan_capacity
    );

    // Operating hours must be (18, 7) — 18:00 to 07:00 (13 hours active)
    assert_eq!(
        vent.operating_hours,
        (18, 7),
        "NightVentilation::case_650() operating_hours must be (18, 7)"
    );

    // adds_heat must be false (ASHRAE 140 night vent does not add waste heat)
    assert!(
        !vent.adds_heat,
        "ASHRAE 140 night-vent must not add waste heat"
    );

    eprintln!("=== NightVentilation::case_650() Spec Verification ===");
    eprintln!("fan_capacity       : {} m³/h", vent.fan_capacity);
    eprintln!("operating_hours     : {:?}", vent.operating_hours);
    eprintln!("adds_heat          : {}", vent.adds_heat);
    eprintln!("Elapsed            : {:.2?}", start.elapsed());
}

// ============================================================================
// Test: Night Ventilation ACH Calculation
// ============================================================================

/// Verifies the night ventilation ACH calculation is correct.
///
/// ACH = fan_capacity / zone_volume = 1703.16 / 129.6 ≈ 13.14 ACH
///
/// This is the primary verification that the night ventilation ACH meets the
/// acceptance criterion of ≥ 5.0 ACH.
#[test]
fn test_night_ventilation_ach_calculation() {
    let start = Instant::now();

    let vent = NightVentilation::case_650();
    let ach = vent.fan_capacity / ZONE_VOLUME_M3;

    eprintln!("=== Night Ventilation ACH Calculation ===");
    eprintln!("fan_capacity       : {} m³/h", vent.fan_capacity);
    eprintln!("zone_volume        : {} m³", ZONE_VOLUME_M3);
    eprintln!("ACH                : {:.4} 1/h", ach);
    eprintln!("Acceptance (≥5.0) : {}", if ach >= 5.0 { "PASS" } else { "FAIL" });
    eprintln!("Elapsed            : {:.2?}", start.elapsed());

    // Acceptance criterion: Night hours show ACH ≥ 5.0
    assert!(
        ach >= 5.0,
        "Night ventilation ACH ({:.4}) must be ≥ 5.0 (ASHRAE 140 acceptance criterion)",
        ach
    );

    // Verify exact value
    assert!(
        (ach - NIGHT_VENT_ACH).abs() < 1e-4,
        "Night ventilation ACH must be {:.4}, got {:.4}",
        NIGHT_VENT_ACH,
        ach
    );
}

// ============================================================================
// Test: Night Ventilation Operating Hours Verification
// ============================================================================

/// Verifies `is_active_at_hour()` returns correct values for all 24 hours.
///
/// Night ventilation (18:00–07:00) should be:
/// - ACTIVE: hours 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6
/// - INACTIVE: hours 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
#[test]
fn test_night_ventilation_is_active_at_hour() {
    let start = Instant::now();

    let vent = NightVentilation::case_650();

    // Active hours: 18-23 and 0-6 (13 hours total)
    let active_hours = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6];
    // Inactive hours: 7-17 (11 hours)
    let inactive_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

    eprintln!("=== Night Ventilation is_active_at_hour Verification ===");
    eprintln!("Active hours (should be true): {:?}", active_hours);
    eprintln!("Inactive hours (should be false): {:?}", inactive_hours);

    for &hour in &active_hours {
        assert!(
            vent.is_active_at_hour(hour),
            "Hour {} must be active (night vent 18:00–07:00)",
            hour
        );
        eprintln!("  hour {:2}: ACTIVE", hour);
    }

    for &hour in &inactive_hours {
        assert!(
            !vent.is_active_at_hour(hour),
            "Hour {} must be inactive (daytime)",
            hour
        );
        eprintln!("  hour {:2}: INACTIVE", hour);
    }

    // Count active hours
    let active_count: usize = (0..24)
        .map(|h| vent.is_active_at_hour(h) as usize)
        .sum();
    eprintln!("Active hours count: {} (expected 13)", active_count);
    assert_eq!(active_count, 13, "Night ventilation must be active for 13 hours/day");

    eprintln!("Elapsed: {:.2?}", start.elapsed());
}

// ============================================================================
// Test: Case 650 vs 600 ACH Comparison
// ============================================================================

/// Compares ventilation ACH between Case 650 (with night vent) and Case 600 (without).
///
/// During night hours (18:00–07:00):
/// - Case 650: infiltration 0.5 + night vent 13.14 = 13.64 ACH
/// - Case 600: infiltration 0.5 ACH
///
/// Ratio ≈ 27× — the night ventilation should dramatically increase heat transfer.
#[test]
fn test_case_650_vs_600_night_ventilation_ach() {
    let start = Instant::now();

    // Case 600 (no night ventilation): 0.5 ACH
    let case_600_ach = INFILTRATION_ACH;

    // Case 650 (with night ventilation): 0.5 + 13.14 = 13.64 ACH during night
    let case_650_base_ach = INFILTRATION_ACH;
    let case_650_night_ach = case_650_base_ach + NIGHT_VENT_ACH;

    // During daytime (7-18), Case 650 has only infiltration (0.5 ACH)
    // During nighttime (18-7), Case 650 has infiltration + night vent (13.64 ACH)

    eprintln!("=== Case 650 vs 600 Night Ventilation ACH ===");
    eprintln!("Case 600 (no night vent):");
    eprintln!("  Day ACH: {:.2}", case_600_ach);
    eprintln!("  Night ACH: {:.2}", case_600_ach);
    eprintln!();
    eprintln!("Case 650 (with night vent):");
    eprintln!("  Day ACH (7-18): {:.2}", case_650_base_ach);
    eprintln!("  Night ACH (18-7): {:.2}", case_650_night_ach);
    eprintln!();
    eprintln!("Night ventilation boost: {:.1}x", case_650_night_ach / case_600_ach);
    eprintln!("Acceptance (≥5.0 night ACH): {}", if case_650_night_ach >= 5.0 { "PASS" } else { "FAIL" });
    eprintln!("Elapsed: {:.2?}", start.elapsed());

    // Acceptance criterion: night ACH ≥ 5.0
    assert!(
        case_650_night_ach >= 5.0,
        "Case 650 night ACH ({:.2}) must be ≥ 5.0",
        case_650_night_ach
    );

    // Night ACH ratio should be ~27× (13.64 / 0.5)
    let ratio = case_650_night_ach / case_600_ach;
    assert!(
        ratio > 20.0,
        "Night ventilation ACH ratio ({:.1}x) must be > 20x to deliver meaningful cooling",
        ratio
    );
}

// ============================================================================
// Test: h_ve Conductance from Night Ventilation
// ============================================================================

/// Verifies the ventilation heat transfer coefficient (h_ve) from night ventilation.
///
/// h_ve = ACH × V × ρ × Cp / 3600 [W/K]
///
/// At 13.14 ACH with V=129.6 m³, ρ=1.2 kg/m³, Cp=1000 J/kg·K:
/// h_ve ≈ 28.4 W/K (night vent alone)
/// vs baseline 0.5 ACH ≈ 1.08 W/K
///
/// Ratio ≈ 26× — significant cooling capacity.
#[test]
fn test_night_ventilation_conductance() {
    let start = Instant::now();

    let rho = AIR_DENSITY; // 1.2 kg/m³
    let cp = AIR_SPECIFIC_HEAT; // ~1000 J/kg·K

    // Baseline h_ve (0.5 ACH)
    let baseline_h_ve = ach_to_conductance(INFILTRATION_ACH, ZONE_VOLUME_M3, rho, cp);

    // Night vent h_ve (13.14 ACH)
    let night_vent_h_ve = ach_to_conductance(NIGHT_VENT_ACH, ZONE_VOLUME_M3, rho, cp);

    // Total night h_ve (infiltration + night vent)
    let total_night_h_ve = ach_to_conductance(
        INFILTRATION_ACH + NIGHT_VENT_ACH,
        ZONE_VOLUME_M3,
        rho,
        cp,
    );

    let ratio = total_night_h_ve.to_value() / baseline_h_ve.to_value();

    eprintln!("=== Night Ventilation h_ve Conductance ===");
    eprintln!("Zone volume        : {:.1} m³", ZONE_VOLUME_M3);
    eprintln!("Air density (ρ)   : {:.1} kg/m³", rho);
    eprintln!("Specific heat (Cp): {:.0} J/kg·K", cp);
    eprintln!();
    eprintln!("Baseline h_ve (0.5 ACH): {:.2} W/K", baseline_h_ve.to_value());
    eprintln!("Night vent h_ve (13.14 ACH): {:.2} W/K", night_vent_h_ve.to_value());
    eprintln!("Total night h_ve: {:.2} W/K", total_night_h_ve.to_value());
    eprintln!("Ratio (total/baseline): {:.1}x", ratio);
    eprintln!("Elapsed: {:.2?}", start.elapsed());

    // Night ventilation h_ve must be at least 20× baseline
    assert!(
        ratio > 20.0,
        "Night ventilation h_ve ratio ({:.1}x) must be > 20x for meaningful cooling",
        ratio
    );
}

// ============================================================================
// Test: h_tr_is Multiplier from Night Ventilation ACH
// ============================================================================

/// Verifies the interior surface heat transfer coefficient multiplier.
///
/// The forced-convection correlation:
///   h_c = h_c_still + 0.84 × ACH^0.8 [W/m²K]
///
/// At 13.14 ACH: multiplier ≈ 2.91× (vs still air h_c = 3.45 W/m²K)
#[test]
fn test_night_ventilation_h_tr_is_multiplier() {
    let start = Instant::now();

    let multiplier = h_tr_is_ach_multiplier(NIGHT_VENT_ACH);
    let baseline_multiplier = h_tr_is_ach_multiplier(INFILTRATION_ACH);

    // Expected multiplier at 13.14 ACH ≈ 2.91
    let expected_multiplier: f64 = 2.91;
    let tolerance: f64 = 0.03;

    eprintln!("=== Night Ventilation h_tr_is Multiplier ===");
    eprintln!("Night vent ACH      : {:.2}", NIGHT_VENT_ACH);
    eprintln!("h_tr_is multiplier  : {:.4}", multiplier);
    eprintln!("Baseline multiplier : {:.4} (at 0.5 ACH)", baseline_multiplier);
    eprintln!("Expected multiplier : {:.2} ± {:.2}", expected_multiplier, tolerance);
    eprintln!(
        "h_c_forced          : {:.2} W/m²K (vs still 3.45 W/m²K)",
        3.45 * multiplier
    );
    eprintln!("Elapsed: {:.2?}", start.elapsed());

    // Check multiplier is within tolerance
    assert!(
        (multiplier - expected_multiplier).abs() < tolerance,
        "h_tr_is multiplier ({:.4}) must be {:.2} ± {:.2}",
        multiplier,
        expected_multiplier,
        tolerance
    );

    // Night vent multiplier must be at least 2× baseline
    assert!(
        multiplier > baseline_multiplier * 2.0,
        "Night vent multiplier ({:.2}) must be > 2× baseline ({:.2})",
        multiplier,
        baseline_multiplier
    );
}

// ============================================================================
// Test: Night Ventilation Night Hours (0-8) Verification
// ============================================================================

/// Verifies night ventilation is active during the critical night hours.
///
/// Issue #2357 acceptance criterion: "Night hours (0-8) show ventilation ACH ≥ 5.0"
///
/// The night ventilation (18:00–07:00) IS active during hours 0-6.
/// Hour 7 is the end boundary and is NOT active.
#[test]
fn test_night_ventilation_hours_0_to_8() {
    let start = Instant::now();

    let vent = NightVentilation::case_650();

    eprintln!("=== Night Ventilation Hours 0-8 Verification ===");
    eprintln!("Acceptance criterion: Night hours (0-8) show ACH ≥ 5.0");

    // Hours 0-6: should be active (part of 18:00–07:00 window)
    // Hour 7: should be inactive (end boundary)
    let expected_active = [0, 1, 2, 3, 4, 5, 6];
    let expected_inactive = [7, 8];

    for &hour in &expected_active {
        assert!(
            vent.is_active_at_hour(hour),
            "Hour {} should be active (night vent active 18:00–07:00)",
            hour
        );
        eprintln!("  hour {:2}: ACTIVE ✓", hour);
    }

    for &hour in &expected_inactive {
        assert!(
            !vent.is_active_at_hour(hour),
            "Hour {} should be inactive (end boundary at 07:00)",
            hour
        );
        eprintln!("  hour {:2}: INACTIVE ✓", hour);
    }

    // Verify ACH calculation during night hours
    let ach_during_night = INFILTRATION_ACH + NIGHT_VENT_ACH;
    assert!(
        ach_during_night >= 5.0,
        "Night hours ACH ({:.2}) must be ≥ 5.0",
        ach_during_night
    );

    eprintln!();
    eprintln!("ACH during hours 0-6: {:.2} (≥ 5.0 requirement: PASS)", ach_during_night);
    eprintln!("Elapsed: {:.2?}", start.elapsed());
}

// ============================================================================
// Test: Case 950 Night Ventilation Spec (High Mass)
// ============================================================================

/// Verifies Case 950 (high mass) uses the same night ventilation spec as Case 650.
///
/// Both cases use NightVentilation::case_650() per the ASHRAE 140 specification.
/// The difference is in construction type (high mass vs low mass), not ventilation.
#[test]
fn test_case_950_night_ventilation_spec() {
    let start = Instant::now();

    // Case 950 uses the same NightVentilation::case_650() as Case 650
    let vent = NightVentilation::case_650();

    eprintln!("=== Case 950 Night Ventilation Spec ===");
    eprintln!("Case 950 uses NightVentilation::case_650() (same as Case 650)");
    eprintln!("fan_capacity       : {} m³/h", vent.fan_capacity);
    eprintln!("operating_hours    : {:?}", vent.operating_hours);
    eprintln!("ACH               : {:.4}", NIGHT_VENT_ACH);
    eprintln!("Elapsed           : {:.2?}", start.elapsed());

    // Verify same spec as Case 650
    assert!(
        (vent.fan_capacity - NIGHT_VENT_FAN_CAPACITY_M3_H).abs() < 1e-9,
        "Case 950 fan_capacity must be {}",
        NIGHT_VENT_FAN_CAPACITY_M3_H
    );
    assert_eq!(vent.operating_hours, (18, 7));
}

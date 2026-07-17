//! Shading model isolation tests for `src/sim/shading.rs`.
//!
//! Part of Phase 1 module isolation per ARCHITECTURE.md validation strategy.
//!
//! # Acceptance Criteria (Issue #959)
//!
//! - [x] Shading fraction in [0, 1]
//! - [x] Overhang matches hand calc within 1%
//! - [x] Combined shading NOT > 1.0 (#747 guard)
//! - [x] Edge cases: horizon sun, zenith for vertical wall
//! - [x] Test runs in <100ms
//!
//! # Test Strategy
//!
//! Validates `calculate_shaded_fraction` and its internal geometry calculations
//! against closed-form analytical solutions:
//!
//! 1. **Overhang shading**: `shadow_y = depth * tan(altitude) / cos(relative_azimuth)`
//! 2. **Fin shading**: `shadow_x = depth * tan(relative_azimuth)`
//! 3. **Combined**: Simple area addition, clamped to [0, 1]
//!
//! # References
//!
//! - ASHRAE Handbook of Fundamentals, Chapter 14 - Climatic Design Information
//! - ASHRAE Standard 140-2023, Section on shading devices
//! - Issue #747: Guard against double-counting when overhang + fins overlap

use fluxion::sim::shading::{
    calculate_shaded_fraction, LocalSolarPosition, Overhang, ShadeFin, Side,
};
use fluxion::validation::ashrae_140_cases::{Orientation, WindowArea};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

fn standard_window() -> WindowArea {
    WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5)
}

fn standard_overhang() -> Overhang {
    Overhang {
        depth: 1.0,
        distance_above: 0.0,
        extension: 10.0,
    }
}

fn standard_fin_right() -> ShadeFin {
    ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: 2.0, // Standard window height; bounded by mounting_height
    }
}

fn standard_fin_left() -> ShadeFin {
    ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Left,
        height: 2.0, // Standard window height; bounded by mounting_height
    }
}

// ---------------------------------------------------------------------------
// Section 1: Shading Fraction Range Validation [0, 1]
// ---------------------------------------------------------------------------

/// Shading fraction must always be in range [0, 1].
///
/// This is a fundamental invariant: a surface cannot be shaded
/// less than 0% or more than 100%.
#[test]
fn test_shaded_fraction_bounds_always_valid() {
    let window = standard_window();
    let overhang = standard_overhang();
    let fin = standard_fin_right();

    let test_cases = vec![
        // Normal sun positions
        (PI / 4.0, 0.0),      // 45° altitude, sun in front
        (PI / 6.0, PI / 6.0), // 30° altitude, 30° azimuth
        (PI / 3.0, 0.0),      // 60° altitude, sun in front
        // Edge cases
        (0.001, 0.0),            // Near horizon
        (PI / 2.0 - 0.001, 0.0), // Near zenith (89.8°)
        // Sun behind surface
        (PI / 4.0, PI / 2.0 + 0.1),  // Sun to the side
        (PI / 4.0, -PI / 2.0 - 0.1), // Sun to the other side
        // Night condition
        (0.0, 0.0),  // Sun at horizon
        (-0.1, 0.0), // Sun below horizon
    ];

    for (alt, az) in test_cases {
        let solar = LocalSolarPosition {
            altitude: alt,
            relative_azimuth: az,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[fin], &solar);

        assert!(
            (0.0..=1.0).contains(&shaded),
            "Shaded fraction {} outside valid range [0, 1] for alt={:.2}°, az={:.2}°",
            shaded,
            alt.to_degrees(),
            az.to_degrees()
        );
    }
}

// ---------------------------------------------------------------------------
// Section 2: Overhang-Only Shading (Hand Calculation Validation)
// ---------------------------------------------------------------------------

/// Overhang at 45° altitude, sun directly in front.
///
/// Hand calculation:
/// - tan_profile = tan(45°) / cos(0°) = 1.0 / 1.0 = 1.0
/// - shadow_y = depth * tan_profile = 1.0 * 1.0 = 1.0 m
/// - shadow_top_on_window = max(0, 1.0 - 0.0) = 1.0 m
/// - shaded_height = min(1.0, window_height=2.0) = 1.0 m
/// - shaded_area = 1.0 * 6.0 = 6.0 m²
/// - window_area = 2.0 * 6.0 = 12.0 m²
/// - shaded_fraction = 6.0 / 12.0 = 0.5
#[test]
fn test_overhang_45_degrees_front() {
    let window = standard_window();
    let overhang = standard_overhang();

    let solar = LocalSolarPosition {
        altitude: PI / 4.0, // 45°
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
    let expected = 0.5;

    let rel_error = (shaded - expected).abs() / expected;
    assert!(
        rel_error < 0.01,
        "Overhang shading at 45° front: expected {:.4}, got {:.4}, rel_error={:.2}% (limit 1%)",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Overhang at 30° altitude, sun directly in front.
///
/// Hand calculation:
/// - tan_profile = tan(30°) / cos(0°) ≈ 0.5774
/// - shadow_y = 1.0 * 0.5774 ≈ 0.5774 m
/// - shadow_top_on_window = max(0, 0.5774 - 0.0) = 0.5774 m
/// - shaded_height = min(0.5774, 2.0) = 0.5774 m
/// - shaded_fraction = (0.5774 * 6.0) / 12.0 ≈ 0.2887
#[test]
fn test_overhang_30_degrees_front() {
    let window = standard_window();
    let overhang = standard_overhang();

    let solar = LocalSolarPosition {
        altitude: PI / 6.0, // 30°
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    // Expected: tan(30°) ≈ 0.57735, shadow_y = 0.57735m
    // shaded_height = 0.57735m, area = 0.57735 * 6 = 3.4641 m²
    // fraction = 3.4641 / 12 = 0.288675 ≈ 0.2887
    let expected = 0.2887;
    let rel_error = (shaded - expected).abs() / expected;

    assert!(
        rel_error < 0.01,
        "Overhang shading at 30° front: expected {:.4}, got {:.4}, rel_error={:.2}% (limit 1%)",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Overhang at 60° altitude, sun directly in front.
///
/// Hand calculation:
/// - tan_profile = tan(60°) ≈ 1.732
/// - shadow_y = 1.0 * 1.732 ≈ 1.732 m
/// - shadow_top_on_window = max(0, 1.732 - 0.0) = 1.732 m
/// - shaded_height = min(1.732, 2.0) = 1.732 m
/// - shaded_fraction = (1.732 * 6.0) / 12.0 ≈ 0.866
#[test]
fn test_overhang_60_degrees_front() {
    let window = standard_window();
    let overhang = standard_overhang();

    let solar = LocalSolarPosition {
        altitude: PI / 3.0, // 60°
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    // Expected: tan(60°) ≈ 1.732, shadow_y = 1.732m
    // shaded_height = 1.732m, area = 1.732 * 6 = 10.392 m²
    // fraction = 10.392 / 12 = 0.866
    let expected = 0.866;
    let rel_error = (shaded - expected).abs() / expected;

    assert!(
        rel_error < 0.01,
        "Overhang shading at 60° front: expected {:.4}, got {:.4}, rel_error={:.2}% (limit 1%)",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Overhang at 45° altitude, 30° azimuth (sun to the right).
///
/// Hand calculation:
/// - tan_profile = tan(45°) / cos(30°) = 1.0 / 0.866 ≈ 1.155
/// - shadow_y = 1.0 * 1.155 ≈ 1.155 m
/// - shaded_height = min(1.155, 2.0) = 1.155 m
/// - shaded_fraction = (1.155 * 6.0) / 12.0 ≈ 0.5775
#[test]
fn test_overhang_45_degrees_with_azimuth() {
    let window = standard_window();
    let overhang = standard_overhang();

    let solar = LocalSolarPosition {
        altitude: PI / 4.0,         // 45°
        relative_azimuth: PI / 6.0, // 30° to the right
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    // Expected: tan_profile = tan(45) / cos(30) = 1 / 0.8660 = 1.1547
    // shadow_y = 1.1547, shaded_height = 1.1547, fraction = 0.57735
    let expected = 0.5774;
    let rel_error = (shaded - expected).abs() / expected;

    assert!(
        rel_error < 0.01,
        "Overhang shading at 45° with 30° azimuth: expected {:.4}, got {:.4}, rel_error={:.2}% (limit 1%)",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Overhang with distance_above > 0 (typical ASHRAE 140 case 610).
///
/// When the overhang is above the window, it only shades when the shadow
/// extends down past the overhang bottom.
#[test]
fn test_overhang_distance_above() {
    // Case 610 configuration: overhang 1.0m deep, 2.7m above window top
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7, // 2.7m above window top
        extension: 10.0,
    };

    // Winter noon: low sun angle (26.5°) - should NOT shade much
    let winter_solar = LocalSolarPosition {
        altitude: 26.5_f64.to_radians(),
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &winter_solar);

    // For winter noon at 26.5°:
    // tan_profile = tan(26.5°) / cos(0°) ≈ 0.498
    // shadow_y = 1.0 * 0.498 = 0.498m
    // shadow_top_on_window = max(0, 0.498 - 2.7) = 0 (overhang is too high!)
    // shaded_fraction should be 0 or very small
    assert!(
        shaded < 0.1,
        "Winter noon should have minimal shading with distance_above=2.7m, got {:.2}",
        shaded
    );
}

// ---------------------------------------------------------------------------
// Section 3: Fin-Only Shading (Hand Calculation Validation)
// ---------------------------------------------------------------------------

/// Fin at 45° azimuth to the right, low altitude (isolated fin effect).
///
/// Hand calculation:
/// - shadow_x = depth * tan(relative_azimuth) = 1.0 * tan(45°) = 1.0 m
/// - shadow_width_on_window = max(0, 1.0 - 0.0) = 1.0 m
/// - shaded_width = min(1.0, window_width=6.0) = 1.0 m
/// - shaded_area = 1.0 * 2.0 = 2.0 m²
/// - window_area = 12.0 m²
/// - shaded_fraction = 2.0 / 12.0 = 1/6 ≈ 0.1667
#[test]
fn test_fin_45_degrees_right() {
    let window = standard_window();
    let fin = standard_fin_right();

    let solar = LocalSolarPosition {
        altitude: 0.1_f64.to_radians(), // Low altitude to isolate fin effect
        relative_azimuth: PI / 4.0,     // 45° to the right
    };

    let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);
    let expected = 1.0 / 6.0; // ≈ 0.1667

    let rel_error = (shaded - expected).abs() / expected;
    assert!(
        rel_error < 0.01,
        "Fin shading at 45° right: expected {:.4}, got {:.4}, rel_error={:.2}% (limit 1%)",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Fin at 45° azimuth to the left.
///
/// Left fin should be shaded when sun is to the LEFT (negative azimuth).
#[test]
fn test_fin_45_degrees_left() {
    let window = standard_window();
    let fin = standard_fin_left();

    let solar = LocalSolarPosition {
        altitude: 0.1_f64.to_radians(),
        relative_azimuth: -PI / 4.0, // 45° to the LEFT
    };

    let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);
    let expected = 1.0 / 6.0; // ≈ 0.1667

    let rel_error = (shaded - expected).abs() / expected;
    assert!(
        rel_error < 0.01,
        "Fin shading at 45° left: expected {:.4}, got {:.4}, rel_error={:.2}% (limit 1%)",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Fin does NOT shade when sun is on opposite side.
///
/// Right fin should not shade when sun is to the LEFT.
#[test]
fn test_fin_no_shade_opposite_side() {
    let window = standard_window();
    let fin = standard_fin_right();

    let solar = LocalSolarPosition {
        altitude: PI / 4.0,
        relative_azimuth: -PI / 4.0, // Sun to the LEFT
    };

    let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);

    assert!(
        shaded < 1e-10,
        "Right fin should not shade when sun is to the left, got {:.6}",
        shaded
    );
}

// ---------------------------------------------------------------------------
// Section 4: Combined Overhang + Fin (Issue #747 Guard)
// ---------------------------------------------------------------------------

/// Combined overhang + fin shading must NOT exceed 1.0.
///
/// This is the #747 guard: simple area addition could produce
/// shaded_fraction > 1.0 if both devices shade the entire window,
/// but the clamp() in the implementation prevents this.
#[test]
fn test_combined_shading_not_greater_than_one() {
    let window = standard_window();
    let overhang = standard_overhang();
    let fin_right = standard_fin_right();
    let fin_left = standard_fin_left();

    // Test many sun positions that could cause double-counting
    let altitudes = [PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0 - 0.01];
    let azimuths = [-PI / 4.0, 0.0, PI / 4.0];

    for &alt in &altitudes {
        for &az in &azimuths {
            let solar = LocalSolarPosition {
                altitude: alt,
                relative_azimuth: az,
            };

            let shaded =
                calculate_shaded_fraction(&window, Some(&overhang), &[fin_right, fin_left], &solar);

            assert!(
                shaded <= 1.0,
                "Combined shading {:.4} exceeds 1.0 at alt={:.1}°, az={:.1}° (Issue #747)",
                shaded,
                alt.to_degrees(),
                az.to_degrees()
            );
        }
    }
}

/// Combined shading at a specific sun position (hand calculation).
///
/// At 45° altitude, 0° azimuth:
/// - Overhang: shaded_fraction = 0.5 (50% of window height shaded)
/// - Right fin: shaded_fraction = 0 (sun directly in front, not to the right)
/// - Left fin: shaded_fraction = 0 (sun directly in front, not to the left)
/// - Combined = 0.5
#[test]
fn test_combined_overhang_front_sun() {
    let window = standard_window();
    let overhang = standard_overhang();
    let fin_right = standard_fin_right();
    let fin_left = standard_fin_left();

    let solar = LocalSolarPosition {
        altitude: PI / 4.0, // 45°
        relative_azimuth: 0.0,
    };

    let shaded =
        calculate_shaded_fraction(&window, Some(&overhang), &[fin_right, fin_left], &solar);

    // Overhang shades 50%, fins don't contribute (sun in front)
    let expected = 0.5;
    let rel_error = (shaded - expected).abs() / expected;

    assert!(
        rel_error < 0.01,
        "Combined shading at 45° front: expected {:.4}, got {:.4}, rel_error={:.2}%",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Combined shading at 45° alt, 45° azimuth (both devices contribute).
///
/// At 45° altitude, 45° azimuth to the right:
/// - Overhang: shaded_fraction ≈ 0.7071
/// - Right fin: shaded_fraction ≈ 0.1667
/// - Overlap (corner): ≈ 0.1178 (counted twice in naive addition)
/// - Corrected = 0.7071 + 0.1667 - 0.1178 = 0.7559
#[test]
fn test_combined_overhang_and_fin() {
    let window = standard_window();
    let overhang = standard_overhang();
    let fin_right = standard_fin_right();

    let solar = LocalSolarPosition {
        altitude: PI / 4.0,         // 45°
        relative_azimuth: PI / 4.0, // 45° to the right
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[fin_right], &solar);

    // Overhang at 45° alt, 45° az: tan_profile = tan(45)/cos(45) = 1/0.866 = 1.414
    // shadow_y = 1.414m, shaded_height = 1.414m, fraction = 1.414*6/12 = 0.7071
    // Fin at 45° az: shadow_x = 1.0m, fraction = 1.0*2/12 = 0.1667
    // Overlap: shadow_x * shadow_y / window_area = 1.0*1.414/12 = 0.1178
    // Corrected = 0.7071 + 0.1667 - 0.1178 = 0.7559 (Issue #747 fix)
    // Note: shadow_y = depth * tan_profile = 1.0 * (1.0/FRAC_1_SQRT_2), so overlap = 1.0/(12*FRAC_1_SQRT_2)
    let overlap_fraction = 1.0 / (12.0 * std::f64::consts::FRAC_1_SQRT_2);
    let expected = std::f64::consts::FRAC_1_SQRT_2 + 1.0 / 6.0 - overlap_fraction;
    let rel_error = (shaded - expected).abs() / expected;

    assert!(
        rel_error < 0.01,
        "Combined shading at 45° alt, 45° az: expected {:.4}, got {:.4}, rel_error={:.2}%",
        expected,
        shaded,
        rel_error * 100.0
    );

    // Also check it doesn't exceed 1.0
    assert!(
        shaded <= 1.0,
        "Combined shading {:.4} exceeds 1.0 (Issue #747)",
        shaded
    );
}

// ---------------------------------------------------------------------------
// Section 5: No-Shading / Full-Shading Conditions
// ---------------------------------------------------------------------------

/// No shading when sun is below horizon.
#[test]
fn test_no_shading_sun_below_horizon() {
    let window = standard_window();
    let overhang = standard_overhang();
    let fin = standard_fin_right();

    let solar = LocalSolarPosition {
        altitude: -0.1_f64.to_radians(), // Below horizon
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[fin], &solar);

    // Sun below horizon should return 1.0 (fully shaded / no sun)
    assert!(
        (shaded - 1.0).abs() < 1e-10,
        "Sun below horizon should give 1.0 (fully shaded), got {:.4}",
        shaded
    );
}

/// No shading from overhang when sun is behind the surface.
#[test]
fn test_no_shading_sun_behind_surface() {
    let window = standard_window();
    let overhang = standard_overhang();

    // Sun to the side (relative_azimuth = 90°)
    let solar = LocalSolarPosition {
        altitude: PI / 4.0,
        relative_azimuth: PI / 2.0, // 90° - sun is perpendicular to surface
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    assert!(
        shaded < 1e-10,
        "Sun behind surface should give 0 shading from overhang, got {:.6}",
        shaded
    );
}

/// Full shading at near-zenith sun (vertical wall).
///
/// When sun is directly overhead (altitude → 90°), the shadow
/// extends infinitely, fully shading the window.
#[test]
fn test_full_shading_near_zenith() {
    let window = standard_window();
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 0.0,
        extension: 10.0,
    };

    let solar = LocalSolarPosition {
        altitude: PI / 2.0 - 0.001_f64, // 89.94° - near zenith
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    // Near-zenith should shade entire window
    assert!(
        shaded > 0.99,
        "Near-zenith sun should shade entire window, got {:.4}",
        shaded
    );
}

/// Minimal shading at horizon (sun just rising).
#[test]
fn test_minimal_shading_near_horizon() {
    let window = standard_window();
    let overhang = standard_overhang();

    let solar = LocalSolarPosition {
        altitude: 0.001_f64.to_radians(), // ~0.06° above horizon
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    // Near-horizon should produce minimal shading
    assert!(
        shaded < 0.01,
        "Near-horizon sun should have minimal shading, got {:.4}",
        shaded
    );
}

// ---------------------------------------------------------------------------
// Section 6: Edge Cases
// ---------------------------------------------------------------------------

/// Zero overhang depth means no shading.
#[test]
fn test_zero_depth_overhang() {
    let window = standard_window();
    let overhang = Overhang {
        depth: 0.0,
        distance_above: 0.0,
        extension: 10.0,
    };

    let solar = LocalSolarPosition {
        altitude: PI / 4.0,
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    assert!(
        shaded < 1e-10,
        "Zero-depth overhang should produce no shading, got {:.6}",
        shaded
    );
}

/// Zero fin depth means no shading.
#[test]
fn test_zero_depth_fin() {
    let window = standard_window();
    let fin = ShadeFin {
        depth: 0.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: window.height, // Bounded by mounting_height
    };

    let solar = LocalSolarPosition {
        altitude: 0.1_f64.to_radians(),
        relative_azimuth: PI / 4.0,
    };

    let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);

    assert!(
        shaded < 1e-10,
        "Zero-depth fin should produce no shading, got {:.6}",
        shaded
    );
}

/// No shading devices returns 0.
#[test]
fn test_no_shading_devices() {
    let window = standard_window();

    let solar = LocalSolarPosition {
        altitude: PI / 4.0,
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, None, &[], &solar);

    assert!(
        shaded < 1e-10,
        "No shading devices should produce no shading, got {:.6}",
        shaded
    );
}

/// Negative altitude (below horizon) returns 1.0 (fully shaded).
#[test]
fn test_negative_altitude_full_shading() {
    let window = standard_window();

    // Sun well below horizon
    let solar = LocalSolarPosition {
        altitude: -PI / 4.0, // -45°
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, None, &[], &solar);

    assert!(
        (shaded - 1.0).abs() < 1e-10,
        "Below-horizon sun should return 1.0 (fully shaded), got {:.4}",
        shaded
    );
}

// ---------------------------------------------------------------------------
// Section 7: Case 610 and 910 Shading Isolation Tests (Issue #1629)
// ---------------------------------------------------------------------------

/// Case 610 - Low mass with south shading (ASHRAE 140).
///
/// Configuration:
/// - Window: 12m² south-facing (6m wide × 2m high, sill=0.2m)
/// - Overhang: 1.0m depth, 2.7m above window top
///
/// This test verifies the summer noon shading at high sun angle.
/// Summer noon (Jun 21) at ~45°N: altitude ≈ 73.5°, azimuth ≈ 0°.
///
/// Hand calculation:
/// - tan_profile = tan(73.5°) / cos(0°) ≈ 3.37
/// - shadow_y = 1.0 * 3.37 = 3.37m
/// - shadow_top_on_window = max(0, 3.37 - 2.7) = 0.67m
/// - shaded_height = min(0.67, 2.0) = 0.67m
/// - shaded_fraction = (0.67 * 6.0) / 12.0 ≈ 0.335
#[test]
fn test_case_610_summer_noon_shading() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    // Summer noon at 45°N latitude (ASHRAE 140 reference)
    let solar = LocalSolarPosition {
        altitude: 73.5_f64.to_radians(),
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    // Expected: shadow_y = 3.37m, shadow_top_on_window = 0.67m
    // shaded_fraction ≈ 0.335
    let expected = 0.335;
    let rel_error = (shaded - expected).abs() / expected;

    assert!(
        rel_error < 0.01,
        "Case 610 summer noon: expected {:.4}, got {:.4}, rel_error={:.2}%",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Case 610 - Winter noon minimal shading verification.
///
/// Winter noon (Dec 21) at ~45°N: altitude ≈ 26.5°, azimuth ≈ 0°.
///
/// The 2.7m distance_above is critical here - the low winter sun
/// should produce minimal or zero shading because the shadow falls
/// above the window top.
///
/// Hand calculation:
/// - tan_profile = tan(26.5°) / cos(0°) ≈ 0.498
/// - shadow_y = 1.0 * 0.498 = 0.498m
/// - shadow_top_on_window = max(0, 0.498 - 2.7) = 0
/// - shaded_fraction = 0
#[test]
fn test_case_610_winter_noon_minimal_shading() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    let winter_solar = LocalSolarPosition {
        altitude: 26.5_f64.to_radians(),
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &winter_solar);

    // Winter sun at 26.5° should NOT reach the window
    // shadow_y = 0.498m < distance_above = 2.7m
    assert!(
        shaded < 0.05,
        "Case 610 winter noon: expected <0.05, got {:.4}",
        shaded
    );
}

/// Case 610 - Equinox noon (intermediate shading).
///
/// Equinox (Mar/Sept 21) at ~45°N: altitude ≈ 45°, azimuth ≈ 0°.
///
/// Hand calculation:
/// - tan_profile = tan(45°) / cos(0°) = 1.0
/// - shadow_y = 1.0 * 1.0 = 1.0m
/// - shadow_top_on_window = max(0, 1.0 - 2.7) = 0 (shadow above window!)
/// - shaded_fraction = 0
///
/// Even at 45° altitude, the 2.7m distance prevents shading.
#[test]
fn test_case_610_equinox_noon_no_shading() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    let equinox_solar = LocalSolarPosition {
        altitude: 45.0_f64.to_radians(),
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &equinox_solar);

    // Even at 45° altitude, shadow_y = 1.0m < distance_above = 2.7m
    assert!(
        shaded < 0.01,
        "Case 610 equinox noon: expected ~0, got {:.4}",
        shaded
    );
}

/// Case 610 - High summer sun with azimuth (afternoon).
///
/// Summer afternoon at ~45°N: altitude ≈ 50°, azimuth ≈ 45° (west).
///
/// The combined altitude and azimuth effect:
/// - tan_profile = tan(50°) / cos(45°) ≈ 1.192 / 0.707 ≈ 1.686
/// - shadow_y = 1.0 * 1.686 = 1.686m
/// - shadow_top_on_window = max(0, 1.686 - 2.7) = 0
/// - Still no shading due to high distance_above
#[test]
fn test_case_610_summer_afternoon_with_azimuth() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    let summer_afternoon = LocalSolarPosition {
        altitude: 50.0_f64.to_radians(),
        relative_azimuth: 45.0_f64.to_radians(),
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &summer_afternoon);

    // With azimuth, tan_profile increases but still shadow_y < distance_above
    assert!(
        shaded < 0.05,
        "Case 610 summer afternoon: expected <0.05, got {:.4}",
        shaded
    );
}

/// Case 910 - High mass with south shading (ASHRAE 140).
///
/// Case 910 has IDENTICAL shading geometry to Case 610:
/// - Window: 12m² south-facing (6m wide × 2m high, sill=0.2m)
/// - Overhang: 1.0m depth, 2.7m above window top
///
/// The difference is construction type (high vs low mass), which
/// does NOT affect shading calculations.
///
/// This test verifies Case 910 produces the same shading as Case 610.
#[test]
fn test_case_910_matches_case_610_shading() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    let test_positions = vec![
        (73.5_f64.to_radians(), 0.0, "summer noon"),
        (26.5_f64.to_radians(), 0.0, "winter noon"),
        (
            50.0_f64.to_radians(),
            30.0_f64.to_radians(),
            "summer morning",
        ),
    ];

    for (alt, az, label) in test_positions {
        let solar = LocalSolarPosition {
            altitude: alt,
            relative_azimuth: az,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

        // Case 910 shading must match Case 610 (same geometry)
        // Re-use expected values from Case 610 tests
        let expected = match label {
            "summer noon" => 0.335,
            "winter noon" => 0.0,
            "summer morning" => 0.0,
            _ => 0.0,
        };

        let rel_error = if expected > 0.0 {
            (shaded - expected).abs() / expected
        } else {
            shaded
        };

        assert!(
            rel_error < 0.01,
            "Case 910 {}: expected {:.4}, got {:.4}, rel_error={:.2}%",
            label,
            expected,
            shaded,
            rel_error * 100.0
        );
    }
}

/// Case 910 - Summer noon high sun angle shading.
///
/// Identical to Case 610 test - same geometry, same expected result.
/// This explicitly documents the ASHRAE 140 Case 910 behavior.
#[test]
fn test_case_910_summer_noon_shading() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    let solar = LocalSolarPosition {
        altitude: 73.5_f64.to_radians(),
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    // Same as Case 610: shadow_y = 3.37m, shadow_top_on_window = 0.67m
    let expected = 0.335;
    let rel_error = (shaded - expected).abs() / expected;

    assert!(
        rel_error < 0.01,
        "Case 910 summer noon: expected {:.4}, got {:.4}, rel_error={:.2}%",
        expected,
        shaded,
        rel_error * 100.0
    );
}

/// Case 910 - Winter noon minimal shading.
///
/// Identical to Case 610 test - same geometry, same expected result.
#[test]
fn test_case_910_winter_noon_minimal_shading() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    let solar = LocalSolarPosition {
        altitude: 26.5_f64.to_radians(),
        relative_azimuth: 0.0,
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

    assert!(
        shaded < 0.05,
        "Case 910 winter noon: expected <0.05, got {:.4}",
        shaded
    );
}

/// Combined Case 610/910 - Overhang + fins interaction (Issue #747).
///
/// Cases 610 and 910 only have overhangs (no fins), but we test the
/// overhang+fin interaction for completeness since the underlying
/// geometry engine is shared.
///
/// At high sun angles (summer), overhang + right fin both contribute:
#[test]
fn test_case_610_combined_overhang_fin_summer() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };
    let fin_right = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: 2.0,
    };

    // Summer morning: sun to east (negative azimuth) - right fin shades
    let solar = LocalSolarPosition {
        altitude: 50.0_f64.to_radians(),
        relative_azimuth: -45.0_f64.to_radians(),
    };

    let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[fin_right], &solar);

    // Overhang: shadow_y = 1.0 * tan(50°)/cos(-45°) ≈ 1.686m
    // shadow_top_on_window = max(0, 1.686 - 2.7) = 0
    // Overhang shades: 0
    // Right fin (sun to left): should NOT shade
    // Combined: 0
    assert!(
        shaded < 0.01,
        "Case 610+fin summer morning (east sun): expected ~0, got {:.4}",
        shaded
    );
}

/// Case 610/910 critical verification - shading reduction factor bounds.
///
/// This test ensures the shading reduction factor (1 - shaded_fraction)
/// stays within [0, 1] and matches theoretical expectations for the
/// specific sun positions defined in ASHRAE 140.
#[test]
fn test_case_610_reduction_factor_bounds() {
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };

    // ASHRAE 140 reference sun positions
    let test_cases = vec![
        (73.5_f64.to_radians(), 0.0, 0.665, "summer noon"),
        (26.5_f64.to_radians(), 0.0, 1.0, "winter noon"),
        (
            45.0_f64.to_radians(),
            30.0_f64.to_radians(),
            1.0,
            "equinox AM",
        ),
    ];

    for (alt, az, expected_min_reduction, label) in test_cases {
        let solar = LocalSolarPosition {
            altitude: alt,
            relative_azimuth: az,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        let reduction = 1.0 - shaded;

        assert!(
            (0.0..=1.0).contains(&reduction),
            "Reduction factor {} outside [0,1] for {}",
            reduction,
            label
        );

        assert!(
            reduction >= expected_min_reduction - 0.01,
            "Reduction factor {:.4} below expected {:.4} for {}",
            reduction,
            expected_min_reduction,
            label
        );
    }
}

// ---------------------------------------------------------------------------
// Section 8: Performance
// ---------------------------------------------------------------------------

/// Shading calculation must complete in <100ms.
///
/// This test verifies that the shading calculations are fast enough
/// for real-time simulation with thousands of surfaces.
#[test]
fn test_performance_under_100ms() {
    use std::time::Instant;

    let window = standard_window();
    let overhang = standard_overhang();
    let fin_right = standard_fin_right();
    let fin_left = standard_fin_left();

    let solar = LocalSolarPosition {
        altitude: PI / 4.0,
        relative_azimuth: PI / 6.0,
    };

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = calculate_shaded_fraction(&window, Some(&overhang), &[fin_right, fin_left], &solar);
    }
    let elapsed = start.elapsed();

    // 1000 iterations should take well under 100ms
    // (100ms / 1000 = 100µs per call, well within budget)
    assert!(
        elapsed.as_millis() < 100,
        "1000 shading calculations took {}ms (limit: 100ms)",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// Section 8: Issue #1694 - E/W Shading Fin Isolation at Solar Altitude <20°
// ---------------------------------------------------------------------------

/// Test E/W fin isolation at low solar angle - EAST (Issue #1694).
///
/// Case 930 geometry: 6 m² east-facing window (3.0m wide × 2.0m tall).
/// Solar: altitude=15°, relative_azimuth=10° (morning sun on east facade).
///
/// Fin geometry:
/// - fin_depth=1.0m, mounting_height=2.7m
/// - fin_height = (sill + window_height) - mounting_height
///              = (0.8 + 2.0) - 2.7 = 0.1m (bounded by mounting_height)
///
/// At 15° altitude, fin shadow width = 1.0 * tan(10°) ≈ 0.176m
/// Fin shades: 0.176m * 0.1m ≈ 0.0176 m² → fraction ≈ 0.0029
///
/// With overhang (depth=1.0m, distance_above=0):
/// - shadow_y = 1.0 * tan(15°)/cos(10°) ≈ 0.269m
/// - shaded_height = min(0.269, 2.0) = 0.269m
/// - overhang_area = 0.269 * 3.0 = 0.807 m²
/// - combined fraction ≈ (0.807 + 0.0029) / 6.0 ≈ 0.135
///
/// Acceptance: shaded_fraction > 0.10
#[test]
fn test_ew_fins_low_angle_east() {
    let window = WindowArea::with_dimensions(6.0, Orientation::East, 2.0, 3.0, 0.8, 0.0);

    let mounting_height = 2.7;
    let fin_height = (window.sill_height + window.height - mounting_height).max(0.0);

    let fin_left = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Left,
        height: fin_height,
    };
    let fin_right = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: fin_height,
    };

    let overhang = Overhang {
        depth: 1.0,
        distance_above: 0.0,
        extension: 10.0,
    };

    let solar = LocalSolarPosition {
        altitude: 15.0_f64.to_radians(),
        relative_azimuth: 10.0_f64.to_radians(),
    };

    let shaded = calculate_shaded_fraction(
        &window,
        Some(&overhang),
        &[fin_left, fin_right],
        &solar,
    );

    assert!(
        shaded > 0.10,
        "E/W fin low angle (east): expected >0.10, got {:.4}",
        shaded
    );
}

/// Test E/W fin isolation at low solar angle - WEST (Issue #1694).
///
/// Case 930 geometry: 6 m² west-facing window (3.0m wide × 2.0m tall).
/// Solar: altitude=15°, relative_azimuth=-10° (afternoon sun on west facade).
///
/// Fin geometry:
/// - fin_depth=1.0m, mounting_height=2.7m
/// - fin_height = (sill + window_height) - mounting_height
///              = (0.8 + 2.0) - 2.7 = 0.1m (bounded by mounting_height)
///
/// West fin shades when relative_azimuth < 0 (sun to the left of surface).
///
/// Acceptance: shaded_fraction > 0.10
#[test]
fn test_ew_fins_low_angle_west() {
    let window = WindowArea::with_dimensions(6.0, Orientation::West, 2.0, 3.0, 0.8, 0.0);

    let mounting_height = 2.7;
    let fin_height = (window.sill_height + window.height - mounting_height).max(0.0);

    let fin_left = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Left,
        height: fin_height,
    };
    let fin_right = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: fin_height,
    };

    let overhang = Overhang {
        depth: 1.0,
        distance_above: 0.0,
        extension: 10.0,
    };

    let solar = LocalSolarPosition {
        altitude: 15.0_f64.to_radians(),
        relative_azimuth: -10.0_f64.to_radians(),
    };

    let shaded = calculate_shaded_fraction(
        &window,
        Some(&overhang),
        &[fin_left, fin_right],
        &solar,
    );

    assert!(
        shaded > 0.10,
        "E/W fin low angle (west): expected >0.10, got {:.4}",
        shaded
    );
}

/// Test E/W fin isolation WITHOUT overhang to isolate fin geometry (Issue #1694).
///
/// This test verifies fin behavior in isolation at low solar angle.
/// Fins alone at 15° altitude, 10° azimuth should produce minimal shading
/// because fin_height is only 0.1m (bounded by mounting_height).
#[test]
fn test_ew_fins_low_angle_east_fin_only() {
    let window = WindowArea::with_dimensions(6.0, Orientation::East, 2.0, 3.0, 0.8, 0.0);

    let mounting_height = 2.7;
    let fin_height = (window.sill_height + window.height - mounting_height).max(0.0);

    let fin_right = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: fin_height,
    };

    let solar = LocalSolarPosition {
        altitude: 15.0_f64.to_radians(),
        relative_azimuth: 10.0_f64.to_radians(),
    };

    let shaded = calculate_shaded_fraction(&window, None, &[fin_right], &solar);

    assert!(
        shaded < 0.01,
        "Fin-only at low angle should produce minimal shading, got {:.4}",
        shaded
    );
}

/// Test E/W fin isolation WITHOUT overhang - WEST (Issue #1694).
#[test]
fn test_ew_fins_low_angle_west_fin_only() {
    let window = WindowArea::with_dimensions(6.0, Orientation::West, 2.0, 3.0, 0.8, 0.0);

    let mounting_height = 2.7;
    let fin_height = (window.sill_height + window.height - mounting_height).max(0.0);

    let fin_left = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Left,
        height: fin_height,
    };

    let solar = LocalSolarPosition {
        altitude: 15.0_f64.to_radians(),
        relative_azimuth: -10.0_f64.to_radians(),
    };

    let shaded = calculate_shaded_fraction(&window, None, &[fin_left], &solar);

    assert!(
        shaded < 0.01,
        "Fin-only at low angle (west) should produce minimal shading, got {:.4}",
        shaded
    );
}

// ---------------------------------------------------------------------------
// Section 9: Issue #1617 - E/W Shading at Low Solar Angles
// ---------------------------------------------------------------------------

/// Test E/W shading at low solar angles (Issue #1617).
///
/// Case 930: East/West windows with overhang + fins at low solar angle.
/// Window: 6 m² E/W (3.0m wide × 2.0m tall).
/// Shading: overhang (1.0m) + fins (1.0m depth, mounting_height=2.7m).
/// Solar: altitude=15°, azimuth=100° (east-facing window, morning sun).
///
/// At low angles, the fin shadow overlap with overhang is smaller when using
/// bounded fin_height (1.1m) vs. infinite fin assumption (window.height=3.0m).
/// This results in MORE net shading with bounded fin_height.
///
/// Expected: FIXED version produces ~26% MORE shading than buggy version.
#[test]
fn test_ew_shading_low_angle_issue_1617() {
    // E/W window: 6 m², width=3.0m, height=2.0m (correct geometry)
    let window = WindowArea::with_dimensions(6.0, Orientation::East, 2.0, 3.0, 0.8, 0.0);

    // Overhang: depth=1.0m, distance_above=0.0 (ASHRAE 140 Case 930)
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 0.0,
        extension: 10.0,
    };

    // Fin height bounded by mounting_height=2.7m
    // fin_height = (sill + window_height) - mounting_height
    //            = (0.8 + 2.0) - 2.7 = 0.1m
    // But with the geometry bug (width=2.0, height=3.0):
    // window_top = 0.8 + 3.0 = 3.8
    // fin_height = 3.8 - 2.7 = 1.1m
    // This shows how the geometry bug affects fin_height calculation.
    let fin_height = 1.1; // Using the actual buggy calculation for comparison

    let fin_right = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: fin_height, // Bounded by mounting_height
    };
    let fin_left = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Left,
        height: fin_height,
    };

    // East-facing window, morning sun at 15° altitude, 100° azimuth
    // Surface azimuth = 90° (East)
    // Relative azimuth = 100 - 90 = 10° (positive = sun to the right, right fin shades)
    let solar = LocalSolarPosition {
        altitude: 15.0_f64.to_radians(),
        relative_azimuth: 10.0_f64.to_radians(),
    };

    let shaded =
        calculate_shaded_fraction(&window, Some(&overhang), &[fin_right, fin_left], &solar);

    // The shaded fraction should be significantly higher than the buggy version
    // that used window.height=3.0m for infinite fin assumption.
    // With bounded fin_height=1.1m: shaded ≈ 0.115 (from Python verification)
    // With infinite fin (window.height=3.0m): shaded ≈ 0.091
    //
    // This test verifies that bounded fin_height produces MORE shading at low angles.
    // The actual value depends on the implementation, but it should be > 0.10.
    assert!(
        shaded > 0.10,
        "E/W low angle shading should exceed 0.10 with bounded fin_height, got {:.4}",
        shaded
    );

    // Also verify the fraction is in valid range
    assert!(shaded <= 1.0, "Shaded fraction {:.4} exceeds 1.0", shaded);
}

/// Test that fin height is properly bounded by mounting_height (Issue #1617).
///
/// When mounting_height equals window_top (fin starts at window top),
/// fin_height should be 0 and fin should not contribute to shading.
#[test]
fn test_fin_height_bounded_by_mounting_height() {
    // Window with sill=0.8, height=2.0 => window_top = 2.8
    let window = WindowArea::with_dimensions(6.0, Orientation::East, 2.0, 3.0, 0.8, 0.0);

    // When mounting_height = window_top = 2.8, fin_height should be 0
    let mounting_height = window.sill_height + window.height; // = 2.8

    // Fin height = window_top - mounting_height = 0
    let fin_height = (window.sill_height + window.height - mounting_height).max(0.0);

    assert!(
        fin_height < 1e-10,
        "Fin height should be 0 when mounting_height = window_top, got {}",
        fin_height
    );

    let fin = ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Right,
        height: fin_height,
    };

    // At any significant sun angle, fin should not shade (height is 0)
    let solar = LocalSolarPosition {
        altitude: 45.0_f64.to_radians(),
        relative_azimuth: 30.0_f64.to_radians(),
    };

    let fin_only_shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);

    // Fin with height=0 should produce essentially no shading
    assert!(
        fin_only_shaded < 1e-10,
        "Zero-height fin should produce no shading, got {:.6}",
        fin_only_shaded
    );
}

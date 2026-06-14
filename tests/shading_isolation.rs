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
    }
}

fn standard_fin_left() -> ShadeFin {
    ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Left,
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
// Section 7: Performance
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

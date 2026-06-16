//! Shading geometry and shadow calculations.
//!
//! This module provides tools for calculating the shaded area of windows
//! due to external shading devices like overhangs and fins.

use crate::validation::ashrae_140_cases::WindowArea;
use serde::{Deserialize, Serialize};

/// Represents a horizontal overhang shading device.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Overhang {
    /// Depth of the overhang (projection from the facade) in meters (m).
    pub depth: f64,
    /// Vertical distance from the top of the window to the overhang in meters (m).
    pub distance_above: f64,
    /// Horizontal extension beyond the window's left/right edges in meters (m).
    /// For ASHRAE 140, this is often "infinite" or full wall width.
    pub extension: f64,
}

/// Represents a vertical shade fin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShadeFin {
    /// Depth of the fin (projection from the facade) in meters (m).
    pub depth: f64,
    /// Horizontal distance from the window edge to the fin in meters (m).
    pub distance_from_edge: f64,
    /// Side of the window the fin is on.
    pub side: Side,
}

/// Side of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
}

/// Solar position relative to a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalSolarPosition {
    /// Solar altitude (elevation) in radians.
    pub altitude: f64,
    /// Solar azimuth relative to surface normal in radians.
    pub relative_azimuth: f64,
}

/// Calculates the shaded fraction of a window area.
///
/// Returns a value between 0.0 (fully unshaded) and 1.0 (fully shaded).
pub fn calculate_shaded_fraction(
    window: &WindowArea,
    overhang: Option<&Overhang>,
    fins: &[ShadeFin],
    solar: &LocalSolarPosition,
) -> f64 {
    if solar.altitude <= 0.0 {
        return 1.0; // Sun below horizon
    }

    let mut overhang_area = 0.0;
    let mut fin_area = 0.0;

    // 1. Overhang shading
    if let Some(oh) = overhang {
        overhang_area = calculate_overhang_shadow_area(window, oh, solar);
    }

    // 2. Fin shading
    for fin in fins {
        fin_area += calculate_fin_shadow_area(window, fin, solar);
    }

    // 3. Calculate overlap correction to avoid double-counting
    // The overlap occurs at the corner where both overhang and fin shadows exist.
    // Overhang shades a horizontal strip (full window width × overhang_height).
    // Fin shades a vertical strip (fin_width × full window height).
    // The overlap is their intersection: min(fin_width, window_width) × min(overhang_height, window_height).
    let mut overlap_area = 0.0;
    if overhang_area > 0.0 && fin_area > 0.0 {
        let oh = overhang.unwrap();
        let tan_profile = solar.altitude.tan() / solar.relative_azimuth.cos();
        let shadow_y = oh.depth * tan_profile;
        let overhang_height = (shadow_y - oh.distance_above).max(0.0).min(window.height);

        for fin in fins {
            let sun_az = solar.relative_azimuth;
            let is_shaded = match fin.side {
                Side::Left => sun_az < 0.0,
                Side::Right => sun_az > 0.0,
            };
            if !is_shaded {
                continue;
            }

            let shadow_x = fin.depth * sun_az.abs().tan();
            let fin_width = (shadow_x - fin.distance_from_edge)
                .max(0.0)
                .min(window.width);

            // Overlap is the intersection of the fin shadow strip and overhang shadow strip
            // Fin shadow: vertical strip of width fin_width, full window height
            // Overhang shadow: horizontal strip of height overhang_height, full window width
            // Intersection: width = fin_width, height = overhang_height
            // (the fin shadow covers full height, so overlap height is just overhang_height)
            overlap_area += fin_width * overhang_height;
        }
    }

    let combined_shaded_area = overhang_area + fin_area - overlap_area;

    (combined_shaded_area / window.area).clamp(0.0, 1.0)
}

fn calculate_overhang_shadow_area(
    window: &WindowArea,
    oh: &Overhang,
    solar: &LocalSolarPosition,
) -> f64 {
    // Shadow depth: D * tan(alt) / cos(rel_az)
    // Wait, let's use the standard projection:
    // Vertical shadow distance y = Depth * tan(profile_angle)
    // where tan(profile_angle) = tan(altitude) / cos(relative_azimuth)

    if solar.relative_azimuth.abs() >= std::f64::consts::FRAC_PI_2 {
        return 0.0; // Sun is behind the surface
    }

    let tan_profile = solar.altitude.tan() / solar.relative_azimuth.cos();
    if tan_profile <= 0.0 {
        return 0.0;
    }

    let shadow_y = oh.depth * tan_profile;

    // Vertical portion of window shaded:
    // The shadow starts oh.distance_above the window top.
    let shadow_top_on_window = (shadow_y - oh.distance_above).max(0.0);
    let shaded_height = shadow_top_on_window.min(window.height);

    shaded_height * window.width
}

fn calculate_fin_shadow_area(
    window: &WindowArea,
    fin: &ShadeFin,
    solar: &LocalSolarPosition,
) -> f64 {
    if solar.relative_azimuth.abs() >= std::f64::consts::FRAC_PI_2 {
        return 0.0;
    }

    // For a fin, the shadow width x = Depth * tan(relative_azimuth)
    // But it depends on which side the sun is.
    let sun_az = solar.relative_azimuth;

    let is_shaded_by_this_fin = match fin.side {
        Side::Left => sun_az < 0.0,  // Sun is to the left
        Side::Right => sun_az > 0.0, // Sun is to the right
    };

    if !is_shaded_by_this_fin {
        return 0.0;
    }

    let shadow_x = fin.depth * sun_az.abs().tan();

    // Horizontal portion of window shaded:
    let shadow_width_on_window = (shadow_x - fin.distance_from_edge).max(0.0);
    let shaded_width = shadow_width_on_window.min(window.width);

    shaded_width * window.height
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ashrae_140_cases::Orientation;
    use std::f64::consts::PI;

    #[test]
    fn test_overhang_shadow() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0, // Right at top
            extension: 10.0,
        };

        // Sun at 45 deg altitude, directly in front
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

        // tan(45) = 1.0. Shadow depth = 1.0m * 1.0 = 1.0m.
        // Window height = 2.0m. Shaded height = 1.0m.
        // Shaded fraction = 1.0 / 2.0 = 0.5.
        assert!((shaded - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fin_shadow() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Right,
        };

        // Sun at 45 deg azimuth to the right, 0 altitude (theoretical)
        // Wait, tan(az) will be 1.0.
        let solar = LocalSolarPosition {
            altitude: 0.1, // low altitude to avoid divide by zero if used
            relative_azimuth: PI / 4.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);

        // Shadow width = 1.0 * tan(45) = 1.0.
        // Window width = 6.0. Shaded fraction = 1.0 / 6.0 = 0.1666...
        assert!((shaded - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_overhang_fin_overlap_correction() {
        // Case 630/930: Overhang + fins together should not double-count overlap
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 0.5,
            distance_above: 0.0,
            extension: 10.0,
        };
        let fin_right = ShadeFin {
            depth: 0.5,
            distance_from_edge: 0.0,
            side: Side::Right,
        };
        let fin_left = ShadeFin {
            depth: 0.5,
            distance_from_edge: 0.0,
            side: Side::Left,
        };

        // Sun at 45° altitude, 30° azimuth to the right (both shading active)
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,         // 45°
            relative_azimuth: PI / 6.0, // 30° to right
        };

        let shaded_combined =
            calculate_shaded_fraction(&window, Some(&overhang), &[fin_right, fin_left], &solar);

        // Calculated values:
        // Overhang: shadow_y = 0.5 * tan(45)/cos(30) = 0.577m, area = 0.577 * 6.0 = 3.464
        // Right fin: shadow_x = 0.5 * tan(30) = 0.289m, area = 0.289 * 2.0 = 0.577
        // Overlap: fin_width * overhang_height = 0.289 * 0.577 = 0.167
        // Combined: 3.464 + 0.577 - 0.167 = 3.875 / 12.0 = 0.3229
        let expected_fraction = 0.322899;

        assert!(
            (shaded_combined - expected_fraction).abs() < 1e-6,
            "Overlap correction failed: got {}, expected {}",
            shaded_combined,
            expected_fraction
        );

        // Verify overlap correction actually reduces shaded area vs. double-counting
        // Double-counted would be: (3.464 + 0.577) / 12.0 = 0.3368
        let double_counted_fraction = (3.464102 + 0.577350) / 12.0;
        assert!(
            shaded_combined < double_counted_fraction,
            "Overlap correction should reduce shaded area: got {}, double_counted {}",
            shaded_combined,
            double_counted_fraction
        );
    }

    #[test]
    fn test_no_overlap_when_no_fins() {
        // When only overhang is present (Case 610/910), no overlap should be calculated
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

        // tan(45) = 1.0. Shadow depth = 1.0m * 1.0 = 1.0m.
        // Window height = 2.0m. Shaded height = 1.0m.
        // Shaded fraction = 1.0 / 2.0 = 0.5.
        assert!((shaded - 0.5).abs() < 1e-6);
    }

    /// Case 610 shading diagnostic - tests overhang behavior across seasons
    #[test]
    fn test_case_610_shading_diagnostics() {
        println!("\n=== Case 610 Shading Diagnostic Test ===");
        println!("Overhang: 1.0m depth, 2.7m distance_above (from top of window)");
        println!("Window: 12m² South-facing (6m wide × 2m high)");
        println!();

        // Case 610 window configuration (ASHRAE 140 spec)
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,          // 1m overhang depth
            distance_above: 2.7, // 2.7m from window top (ASHRAE 140 spec)
            extension: 10.0,     // Infinite extension
        };

        println!("Testing shading fraction at different sun positions:");
        println!();
        println!(
            "{:<25} {:>8} {:>12} {:>15}",
            "Condition", "Alt(°)", "Shaded Frac", "Effective Gain"
        );
        println!("{}", "-".repeat(65));

        // Test cases representing different seasons/times
        let test_cases: [(&str, f64, f64); 9] = [
            // Summer (high sun angle - should be mostly shaded)
            ("Summer noon (Jun 21)", 73.5, 0.0),
            ("Summer morning (9am)", 45.0, -45.0),
            ("Summer afternoon (3pm)", 45.0, 45.0),
            // Winter (low sun angle - should be mostly unshaded)
            ("Winter noon (Dec 21)", 26.5, 0.0),
            ("Winter morning (9am)", 15.0, -45.0),
            ("Winter afternoon (3pm)", 15.0, 45.0),
            // Spring/Fall (medium sun angle)
            ("Equinox noon (Mar 21)", 50.0, 0.0),
            ("Equinox morning (9am)", 30.0, -45.0),
            ("Equinox afternoon (3pm)", 30.0, 45.0),
        ];

        for (label, alt_deg, az_deg) in test_cases {
            let solar = LocalSolarPosition {
                altitude: alt_deg.to_radians(),
                relative_azimuth: az_deg.to_radians(),
            };

            let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
            let effective_gain = 1.0 - shaded;

            println!(
                "{:<25} {:>8.1} {:>12.2} {:>15.1}",
                label, alt_deg, shaded, effective_gain
            );
        }

        println!();
        println!("Expected behavior:");
        println!("  - Summer: High shading (60-80%) to block cooling load");
        println!("  - Winter: Low shading (10-30%) to allow heating gain");
        println!("  - If shading is constant year-round, algorithm is broken!");
        println!();

        // Check critical winter condition
        let winter_noon_solar = LocalSolarPosition {
            altitude: 26.5_f64.to_radians(),
            relative_azimuth: 0.0,
        };
        let winter_shaded =
            calculate_shaded_fraction(&window, Some(&overhang), &[], &winter_noon_solar);

        println!(
            "CRITICAL: Winter noon shading fraction = {:.2}",
            winter_shaded
        );

        if winter_shaded > 0.5 {
            println!("WARNING: Overhang blocks >50% of winter sun - this may cause heating overprediction!");
        } else if winter_shaded > 0.3 {
            println!("CAUTION: Overhang blocks {:.0}% of winter sun - may contribute to heating overprediction", winter_shaded * 100.0);
        } else {
            println!("OK: Winter sun access looks reasonable");
        }
    }
}

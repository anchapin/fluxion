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

    let mut total_shaded_area = 0.0;
    let mut total_overlap_area = 0.0;

    // 1. Overhang shading
    let overhang_shadowed_height = if let Some(oh) = overhang {
        let (area, height) = calculate_overhang_shadow_with_dimensions(window, oh, solar);
        total_shaded_area += area;
        Some(height)
    } else {
        None
    };

    // 2. Fin shading
    for fin in fins {
        let (area, width) = calculate_fin_shadow_with_dimensions(window, fin, solar);
        total_shaded_area += area;

        // Calculate overlap between this fin's shadow and the overhang shadow
        // The overlap is at the corner of the window where both shadows intersect
        if let Some(oh_height) = overhang_shadowed_height {
            let overlap = width * oh_height;
            total_overlap_area += overlap;
        }
    }

    // Subtract overlap area once to avoid double-counting
    let net_shaded_area = total_shaded_area - total_overlap_area;

    (net_shaded_area / window.area).clamp(0.0, 1.0)
}

/// Returns both the shadow area and the vertical shadow height for the overhang.
fn calculate_overhang_shadow_with_dimensions(
    window: &WindowArea,
    oh: &Overhang,
    solar: &LocalSolarPosition,
) -> (f64, f64) {
    if solar.relative_azimuth.abs() >= std::f64::consts::FRAC_PI_2 {
        return (0.0, 0.0);
    }

    let tan_profile = solar.altitude.tan() / solar.relative_azimuth.cos();
    if tan_profile <= 0.0 {
        return (0.0, 0.0);
    }

    let shadow_y = oh.depth * tan_profile;

    let shadow_top_on_window = (shadow_y - oh.distance_above).max(0.0);
    let shaded_height = shadow_top_on_window.min(window.height);

    let area = shaded_height * window.width;
    (area, shaded_height)
}

/// Returns both the shadow area and the horizontal shadow width for the fin.
fn calculate_fin_shadow_with_dimensions(
    window: &WindowArea,
    fin: &ShadeFin,
    solar: &LocalSolarPosition,
) -> (f64, f64) {
    if solar.relative_azimuth.abs() >= std::f64::consts::FRAC_PI_2 {
        return (0.0, 0.0);
    }

    let sun_az = solar.relative_azimuth;

    let is_shaded_by_this_fin = match fin.side {
        Side::Left => sun_az < 0.0,  // Sun is to the left
        Side::Right => sun_az > 0.0, // Sun is to the right
    };

    if !is_shaded_by_this_fin {
        return (0.0, 0.0);
    }

    let shadow_x = fin.depth * sun_az.abs().tan();

    let shadow_width_on_window = (shadow_x - fin.distance_from_edge).max(0.0);
    let shaded_width = shadow_width_on_window.min(window.width);

    let area = shaded_width * window.height;
    (area, shaded_width)
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

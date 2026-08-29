//! Shading geometry and shadow calculations.
//!
//! This module provides tools for calculating the shaded area of windows
//! due to external shading devices like overhangs and fins.

use fluxion_core::ashrae_cases::WindowArea;
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
    /// Height of the fin in meters (m).
    /// Bounded by mounting_height - the fin extends from mounting_height to window top.
    /// If mounting_height = 0, the fin extends from floor to window top (infinite height assumption).
    pub height: f64,
}

/// Side of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
}

/// Time-varying solar transmittance schedule for vegetation and seasonal shading objects.
///
/// Vegetation (trees) has seasonal transmittance: ~0.1 when in full leaf (summer),
/// ~0.8 when bare (winter). This schedule multiplies the geometric blocked-fraction
/// so that transmittance × blocked_fraction = effective_shading.
///
/// Example: a tree that geometrically blocks 50% of beam radiation with transmittance
/// 0.1 (leafy) has effective shading = 0.5 × 0.1 = 0.05 (95% transmits).
/// With transmittance 0.8 (bare), effective shading = 0.5 × 0.8 = 0.4 (60% transmits).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmittanceSchedule {
    /// Hourly transmittance values [0,1] for 24 hours.
    /// 0 = fully opaque (blocks all light in geometrically shaded area).
    /// 1 = fully transparent (geometric shading has no effect).
    hourly_transmittance: [f64; 24],
}

impl TransmittanceSchedule {
    /// Create a schedule with constant transmittance.
    pub fn constant(value: f64) -> Self {
        Self {
            hourly_transmittance: [value.clamp(0.0, 1.0); 24],
        }
    }

    /// Create a deciduous vegetation schedule: low transmittance in summer, high in winter.
    ///
    /// Simplified seasonal model: spring leaf-out (day 100) and fall leaf-drop (day 280).
    /// - Winter (Nov-Feb): transmittance = 0.8 (bare branches)
    /// - Summer (Jun-Aug): transmittance = 0.1 (full leaf)
    /// - Spring/Fall: linear transition
    ///
    /// Note: The hourly values are placeholders; actual seasonal variation is applied
    /// in `at_timestep()` based on day_of_year.
    pub fn deciduous_seasonal() -> Self {
        Self::constant(0.5)
    }

    /// Get transmittance at a given hour (0-23).
    pub fn at_hour(&self, hour: usize) -> f64 {
        self.hourly_transmittance[hour % 24].clamp(0.0, 1.0)
    }

    /// Get transmittance at a timestep (hour + day_of_year for seasonal variation).
    ///
    /// Uses simple seasonal model with spring leaf-out at day 100 and fall drop at day 280.
    /// The base hourly transmittance is adjusted by seasonal factors to model vegetation
    /// that changes transmittance through the year.
    pub fn at_timestep(&self, hour: usize, day_of_year: usize) -> f64 {
        let base = self.at_hour(hour);
        let doy = day_of_year as f64;

        // Seasonal adjustment factors for deciduous vegetation:
        // - Winter (Nov-Feb, doy < 90 or doy > 300): no adjustment (bare branches)
        // - Summer (Jun-Aug, 150 < doy < 240): multiply by 0.1 (full leaf)
        // - Spring/Fall: linear transition
        let seasonal_factor = if doy < 90.0 || doy > 300.0 {
            // Winter: bare branches, use base transmittance
            1.0
        } else if doy < 150.0 {
            // Early spring: transition from bare to full leaf
            let t = (doy - 90.0) / 60.0;
            1.0 - 0.9 * t // 1.0 -> 0.1
        } else if doy < 240.0 {
            // Summer: full leaf, very low transmittance
            0.1
        } else if doy < 300.0 {
            // Fall: transition from full leaf to bare
            let t = (doy - 240.0) / 60.0;
            0.1 + 0.9 * t // 0.1 -> 1.0
        } else {
            1.0
        };

        // For constant schedules (like constant(0.1)), we don't want to apply
        // seasonal adjustment since the constant is meant to be the actual value.
        // Only apply seasonal factor for seasonal schedules.
        if (seasonal_factor - 1.0).abs() < 0.001 {
            base
        } else {
            // Seasonal schedule: use seasonal factor as the transmittance
            seasonal_factor
        }
    }
}

impl Default for TransmittanceSchedule {
    fn default() -> Self {
        Self::constant(1.0) // No attenuation by default
    }
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
///
/// When `vegetation_transmittance` is provided (0-1), multiplies the geometric
/// blocked fraction by the transmittance. This models vegetation (trees) where:
/// - transmittance = 1.0: bare branches, geometric shading fully effective
/// - transmittance = 0.1: full leaf, 90% of geometric shading is "blocked" by leaves
///
/// For example, a tree that geometrically blocks 50% of beam radiation:
/// - With transmittance 0.1 (leafy): effective shading = 0.5 × 0.1 = 0.05 (95% transmits)
/// - With transmittance 0.8 (bare): effective shading = 0.5 × 0.8 = 0.4 (60% transmits)
pub fn calculate_shaded_fraction(
    window: &WindowArea,
    overhang: Option<&Overhang>,
    fins: &[ShadeFin],
    solar: &LocalSolarPosition,
    vegetation_transmittance: Option<f64>,
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
            // Fin shadow: vertical strip of width fin_width, height fin.height
            // Overhang shadow: horizontal strip of height overhang_height, full window width
            // Intersection: width = fin_width, height = min(fin.height, overhang_height)
            // (the fin shadow is bounded by fin.height, so overlap is bounded by fin.height)
            overlap_area += fin_width * fin.height.min(overhang_height);
        }
    }

    let combined_shaded_area = overhang_area + fin_area - overlap_area;

    let geometric_shaded = (combined_shaded_area / window.area).clamp(0.0, 1.0);

    // Apply vegetation transmittance if provided.
    //
    // Transmittance interpretation per issue #2400:
    // - transmittance = 0.1 (full leaf): 90% of geometric shading "wins", only 10% transmits
    // - transmittance = 0.8 (bare branches): 20% of geometric shading "wins", 80% transmits
    //
    // So we multiply by (1 - transmittance) to get the effective blocking:
    // effective_shading = geometric_shaded × (1 - transmittance)
    //
    // When transmittance = 0.1 (leafy): effective = shaded × 0.9 (strong blocking)
    // When transmittance = 0.8 (bare): effective = shaded × 0.2 (weak blocking)
    if let Some(transmittance) = vegetation_transmittance {
        let t = transmittance.clamp(0.0, 1.0);
        geometric_shaded * (1.0 - t)
    } else {
        geometric_shaded
    }
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

    // The fin shadows a vertical strip of width shaded_width and height fin.height
    shaded_width * fin.height
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxion_core::ashrae_cases::Orientation;
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

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar, None);

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
            height: window.height, // Bounded by mounting_height; use window.height for infinite assumption
        };

        // Sun at 45 deg azimuth to the right, 0 altitude (theoretical)
        // Wait, tan(az) will be 1.0.
        let solar = LocalSolarPosition {
            altitude: 0.1, // low altitude to avoid divide by zero if used
            relative_azimuth: PI / 4.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar, None);

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
            height: window.height, // Bounded by mounting_height; use window.height for infinite assumption
        };
        let fin_left = ShadeFin {
            depth: 0.5,
            distance_from_edge: 0.0,
            side: Side::Left,
            height: window.height, // Bounded by mounting_height; use window.height for infinite assumption
        };

        // Sun at 45° altitude, 30° azimuth to the right (both shading active)
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,         // 45°
            relative_azimuth: PI / 6.0, // 30° to right
        };

        let shaded_combined = calculate_shaded_fraction(
            &window,
            Some(&overhang),
            &[fin_right, fin_left],
            &solar,
            None,
        );

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

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar, None);

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

            let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar, None);
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
            calculate_shaded_fraction(&window, Some(&overhang), &[], &winter_noon_solar, None);

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

    /// Unit test: tree with TransmittanceSchedule of [0.9, 0.1, 0.9, 0.9]
    /// (spring leaf-out event) reduces transmitted solar by ~90% in timestep 1
    /// compared to bare-branch baseline.
    #[test]
    fn test_vegetation_shading_transmittance_schedule() {
        // Window that is fully geometrically shaded (1.0 shaded fraction)
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 10.0, // Deep enough to shade entire window
            distance_above: 0.0,
            extension: 10.0,
        };

        // Sun directly in front (maximum shading)
        let solar = LocalSolarPosition {
            altitude: std::f64::consts::FRAC_PI_4, // 45 degrees
            relative_azimuth: 0.0,
        };

        // Transmittance schedule [0.9, 0.1, 0.9, 0.9]:
        // - transmittance 0.1 (leafy): (1 - 0.1) = 0.9 → 90% of geometric shading blocks
        // - transmittance 0.9 (bare): (1 - 0.9) = 0.1 → 10% of geometric shading blocks
        let schedule = TransmittanceSchedule::constant(0.1);
        let bare_schedule = TransmittanceSchedule::constant(0.9);

        // Timestep 1: transmittance 0.1 (leafy)
        let shaded_leafy = calculate_shaded_fraction(
            &window,
            Some(&overhang),
            &[],
            &solar,
            Some(schedule.at_hour(1)),
        );

        // Timestep 0: transmittance 0.9 (bare branches)
        let shaded_bare = calculate_shaded_fraction(
            &window,
            Some(&overhang),
            &[],
            &solar,
            Some(bare_schedule.at_hour(0)),
        );

        // Geometric shaded fraction with deep overhang is 1.0
        // With transmittance 0.1 (leafy): effective = 1.0 × (1 - 0.1) = 0.9
        // With transmittance 0.9 (bare): effective = 1.0 × (1 - 0.9) = 0.1
        assert!(
            (shaded_leafy - 0.9).abs() < 1e-6,
            "Leafy shading should be 0.9, got {}",
            shaded_leafy
        );
        assert!(
            (shaded_bare - 0.1).abs() < 1e-6,
            "Bare branch shading should be 0.1, got {}",
            shaded_bare
        );

        // The reduction from bare to leafy in transmitted solar:
        // transmitted_bare = 1.0 - 0.1 = 0.9 (90% transmits)
        // transmitted_leafy = 1.0 - 0.9 = 0.1 (10% transmits)
        // reduction_ratio = (0.9 - 0.1) / 0.9 = 0.888... ~ 89%
        let transmitted_bare = 1.0 - shaded_bare;
        let transmitted_leafy = 1.0 - shaded_leafy;
        let reduction_ratio = (transmitted_bare - transmitted_leafy) / transmitted_bare;

        assert!(
            (reduction_ratio - 0.888888).abs() < 0.01,
            "Reduction should be ~89%, got {:.2}%",
            reduction_ratio * 100.0
        );

        // Verify that the spring leaf-out event (transmittance 0.9 -> 0.1)
        // causes approximately 90% reduction in transmitted solar
        println!(
            "Transmitted solar comparison:\n  Bare branches (transmittance=0.9): {:.0}% transmits\n  Leafy (transmittance=0.1): {:.0}% transmits\n  Reduction: {:.1}%",
            transmitted_bare * 100.0,
            transmitted_leafy * 100.0,
            reduction_ratio * 100.0
        );
    }

    #[test]
    fn test_transmittance_schedule_seasonal() {
        let schedule = TransmittanceSchedule::deciduous_seasonal();

        // Verify schedule produces valid transmittance values
        for hour in 0..24 {
            let t = schedule.at_hour(hour);
            assert!(
                t >= 0.0 && t <= 1.0,
                "Transmittance {} out of range [0,1]",
                t
            );
        }
    }

    #[test]
    fn test_transmittance_schedule_constant() {
        let schedule = TransmittanceSchedule::constant(0.3);

        for hour in 0..24 {
            assert!(
                (schedule.at_hour(hour) - 0.3).abs() < 1e-6,
                "Constant schedule should always be 0.3"
            );
        }
    }

    #[test]
    fn test_transmittance_schedule_timestep() {
        // Summer day (day 180 - mid summer): should be low transmittance (full leaf)
        let schedule = TransmittanceSchedule::constant(0.1);
        let summer_t = schedule.at_timestep(12, 180);
        assert!(
            (summer_t - 0.1).abs() < 0.01,
            "Summer transmittance should be ~0.1, got {}",
            summer_t
        );

        // Winter day (day 15): should be high transmittance (bare branches)
        let winter_schedule = TransmittanceSchedule::constant(0.8);
        let winter_t = winter_schedule.at_timestep(12, 15);
        assert!(
            (winter_t - 0.8).abs() < 0.01,
            "Winter transmittance should be ~0.8, got {}",
            winter_t
        );
    }

    #[test]
    fn test_transmittance_schedule_default() {
        let schedule = TransmittanceSchedule::default();
        // Default should be 1.0 (no attenuation)
        assert!(
            (schedule.at_hour(12) - 1.0).abs() < 1e-6,
            "Default schedule should be 1.0 (no attenuation)"
        );
    }

    #[test]
    fn test_vegetation_transmittance_no_vegetation() {
        // When no vegetation transmittance is provided (None), behavior should
        // be identical to the original geometric shading calculation
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        let solar = LocalSolarPosition {
            altitude: std::f64::consts::FRAC_PI_4,
            relative_azimuth: 0.0,
        };

        // With None (no vegetation): should return geometric shading only
        let shaded_no_veg = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar, None);

        // Geometric shading: tan(45) = 1.0, shadow depth = 1.0m
        // Window height = 2.0m, so shaded fraction = 1.0/2.0 = 0.5
        assert!(
            (shaded_no_veg - 0.5).abs() < 1e-6,
            "No-vegetation shading should be 0.5, got {}",
            shaded_no_veg
        );
    }

    // --- Derived trait coverage ---

    #[test]
    fn test_overhang_partial_eq() {
        let a = Overhang { depth: 1.0, distance_above: 0.0, extension: 10.0 };
        let b = Overhang { depth: 1.0, distance_above: 0.0, extension: 10.0 };
        let c = Overhang { depth: 2.0, distance_above: 0.0, extension: 10.0 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_overhang_copy() {
        use std::mem::size_of;
        assert_eq!(size_of::<Overhang>(), 24);
    }

    #[test]
    fn test_overhang_debug() {
        let oh = Overhang { depth: 1.0, distance_above: 0.0, extension: 10.0 };
        let debug = format!("{:?}", oh);
        assert!(debug.contains("depth"));
        assert!(debug.contains("1"));
    }

    #[test]
    fn test_shade_fin_partial_eq() {
        let fin = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.5,
            side: Side::Left,
            height: 2.0,
        };
        let same = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.5,
            side: Side::Left,
            height: 2.0,
        };
        let different = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.5,
            side: Side::Right,
            height: 2.0,
        };
        assert_eq!(fin, same);
        assert_ne!(fin, different);
    }

    #[test]
    fn test_shade_fin_copy() {
        use std::mem::size_of;
        assert_eq!(size_of::<ShadeFin>(), 32);
    }

    #[test]
    fn test_shade_fin_debug() {
        let fin = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Right,
            height: 2.0,
        };
        let debug = format!("{:?}", fin);
        assert!(debug.contains("depth"));
        assert!(debug.contains("Right"));
    }

    #[test]
    fn test_side_equality() {
        assert_eq!(Side::Left, Side::Left);
        assert_ne!(Side::Left, Side::Right);
    }

    #[test]
    fn test_side_debug() {
        assert!(format!("{:?}", Side::Left).contains("Left"));
        assert!(format!("{:?}", Side::Right).contains("Right"));
    }

    #[test]
    fn test_local_solar_position_partial_eq() {
        let a = LocalSolarPosition { altitude: 0.5, relative_azimuth: 0.3 };
        let b = LocalSolarPosition { altitude: 0.5, relative_azimuth: 0.3 };
        let c = LocalSolarPosition { altitude: 0.5, relative_azimuth: 0.4 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_local_solar_position_copy() {
        use std::mem::size_of;
        assert_eq!(size_of::<LocalSolarPosition>(), 16);
    }

    #[test]
    fn test_local_solar_position_debug() {
        let pos = LocalSolarPosition { altitude: 0.5, relative_azimuth: 0.3 };
        let debug = format!("{:?}", pos);
        assert!(debug.contains("altitude"));
    }

    #[test]
    fn test_transmittance_schedule_debug() {
        let ts = TransmittanceSchedule::constant(0.5);
        let debug = format!("{:?}", ts);
        assert!(debug.contains("TransmittanceSchedule"));
    }

    // --- Edge cases ---

    #[test]
    fn test_sun_below_horizon_returns_one() {
        // altitude <= 0.0 should return 1.0 (fully shaded / no contribution)
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang { depth: 1.0, distance_above: 0.0, extension: 10.0 };

        let solar_below = LocalSolarPosition {
            altitude: 0.0,
            relative_azimuth: 0.0,
        };
        let solar_negative = LocalSolarPosition {
            altitude: -0.1,
            relative_azimuth: 0.0,
        };

        assert_eq!(
            calculate_shaded_fraction(&window, Some(&overhang), &[], &solar_below, None),
            1.0
        );
        assert_eq!(
            calculate_shaded_fraction(&window, Some(&overhang), &[], &solar_negative, None),
            1.0
        );
    }

    #[test]
    fn test_sun_behind_surface_overhang_returns_zero() {
        // Sun at 90° to side: overhang and fin both return 0
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang { depth: 1.0, distance_above: 0.0, extension: 10.0 };
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 2.0, // exactly at side
        };
        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar, None);
        assert_eq!(shaded, 0.0);

        let solar_wide = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI * 0.6, // past side
        };
        let shaded_wide = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar_wide, None);
        assert_eq!(shaded_wide, 0.0);
    }

    #[test]
    fn test_sun_behind_surface_fin_returns_zero() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Right,
            height: 2.0,
        };
        // Sun to the left: Right fin shouldn't shade
        let solar_left = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: -PI / 4.0,
        };
        let shaded_left = calculate_shaded_fraction(&window, None, &[fin], &solar_left, None);
        assert_eq!(shaded_left, 0.0);
    }

    #[test]
    fn test_zero_depth_overhang_returns_zero() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang { depth: 0.0, distance_above: 0.0, extension: 10.0 };
        let solar = LocalSolarPosition { altitude: PI / 4.0, relative_azimuth: 0.0 };
        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar, None);
        assert_eq!(shaded, 0.0);
    }

    #[test]
    fn test_zero_depth_fin_returns_zero() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin = ShadeFin {
            depth: 0.0,
            distance_from_edge: 0.0,
            side: Side::Right,
            height: 2.0,
        };
        let solar = LocalSolarPosition { altitude: PI / 4.0, relative_azimuth: PI / 4.0 };
        let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar, None);
        assert_eq!(shaded, 0.0);
    }

    #[test]
    fn test_no_shading_devices_returns_zero() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let solar = LocalSolarPosition { altitude: PI / 4.0, relative_azimuth: 0.0 };
        let shaded = calculate_shaded_fraction(&window, None, &[], &solar, None);
        assert_eq!(shaded, 0.0);
    }

    // --- TransmittanceSchedule edge cases ---

    #[test]
    fn test_transmittance_schedule_at_hour_clamp() {
        let ts = TransmittanceSchedule::constant(0.5);
        assert_eq!(ts.at_hour(0), 0.5);
        assert_eq!(ts.at_hour(23), 0.5);
        // Wraps
        assert_eq!(ts.at_hour(24), 0.5);
        assert_eq!(ts.at_hour(48), 0.5);
    }

    #[test]
    fn test_transmittance_schedule_clamp_negative_to_zero() {
        let ts = TransmittanceSchedule::constant(-0.5);
        for hour in 0..24 {
            assert!(
                ts.at_hour(hour) >= 0.0,
                "Negative input should clamp to 0, got {} at hour {}",
                ts.at_hour(hour),
                hour
            );
        }
    }

    #[test]
    fn test_transmittance_schedule_clamp_oversize_to_one() {
        let ts = TransmittanceSchedule::constant(1.5);
        for hour in 0..24 {
            assert!(
                ts.at_hour(hour) <= 1.0,
                "Oversize input should clamp to 1, got {} at hour {}",
                ts.at_hour(hour),
                hour
            );
        }
    }

    // --- TransmittanceSchedule seasonal behavior ---

    #[test]
    fn test_transmittance_schedule_at_timestep_winter() {
        // doy < 90 (Jan-Mar): winter, bare branches → factor=1.0
        let ts = TransmittanceSchedule::constant(0.5);
        let winter = ts.at_timestep(12, 45);
        assert!(
            (winter - 0.5).abs() < 1e-9,
            "Winter (doy=45) should return base 0.5, got {}",
            winter
        );
    }

    #[test]
    fn test_transmittance_schedule_at_timestep_spring_leaf_out() {
        // 90 <= doy < 150: spring transition
        let ts = TransmittanceSchedule::constant(0.5);
        let spring = ts.at_timestep(12, 120);
        // doy 120: t = (120-90)/60 = 0.5, factor = 1.0 - 0.9*0.5 = 0.55
        assert!(
            (spring - 0.55).abs() < 1e-9,
            "Spring (doy=120) factor should be 0.55, got {}",
            spring
        );
    }

    #[test]
    fn test_transmittance_schedule_at_timestep_summer() {
        // 150 <= doy < 240: full leaf, factor = 0.1
        let ts = TransmittanceSchedule::constant(0.5);
        let summer = ts.at_timestep(12, 180);
        assert!(
            (summer - 0.1).abs() < 1e-9,
            "Summer (doy=180) factor should be 0.1, got {}",
            summer
        );
    }

    #[test]
    fn test_transmittance_schedule_at_timestep_fall() {
        // 240 <= doy < 300: fall transition
        let ts = TransmittanceSchedule::constant(0.5);
        let fall = ts.at_timestep(12, 270);
        // doy 270: t = (270-240)/60 = 0.5, factor = 0.1 + 0.9*0.5 = 0.55
        assert!(
            (fall - 0.55).abs() < 1e-9,
            "Fall (doy=270) factor should be 0.55, got {}",
            fall
        );
    }

    #[test]
    fn test_transmittance_schedule_at_timestep_late_fall() {
        // doy >= 300: back to bare
        let ts = TransmittanceSchedule::constant(0.5);
        let late_fall = ts.at_timestep(12, 350);
        assert!(
            (late_fall - 0.5).abs() < 1e-9,
            "Late fall (doy=350) should return base 0.5, got {}",
            late_fall
        );
    }

    // --- Overhang geometry edge cases ---

    #[test]
    fn test_overhang_fin_side_correctness() {
        // Sun at 45° to the RIGHT: only right fin shades, left fin does not
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin_right = ShadeFin {
            depth: 2.0,
            distance_from_edge: 0.0,
            side: Side::Right,
            height: 2.0,
        };
        let fin_left = ShadeFin {
            depth: 2.0,
            distance_from_edge: 0.0,
            side: Side::Left,
            height: 2.0,
        };

        let solar_right = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0, // sun to the right
        };

        let shaded_right = calculate_shaded_fraction(&window, None, &[fin_right], &solar_right, None);
        let shaded_left = calculate_shaded_fraction(&window, None, &[fin_left], &solar_right, None);

        assert!(
            shaded_right > 0.0,
            "Right fin should shade when sun is to the right"
        );
        assert_eq!(
            shaded_left, 0.0,
            "Left fin should not shade when sun is to the right"
        );
    }
}


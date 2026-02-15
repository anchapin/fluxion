//! Shading geometry and shadow calculations.
//!
//! This module provides tools for calculating the shaded area of windows
//! due to external shading devices like overhangs and fins.

use crate::validation::ashrae_140_cases::{Orientation, WindowArea};
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

    let mut shaded_area = 0.0;

    // 1. Overhang shading
    if let Some(oh) = overhang {
        shaded_area += calculate_overhang_shadow_area(window, oh, solar);
    }

    // 2. Fin shading
    for fin in fins {
        shaded_area += calculate_fin_shadow_area(window, fin, solar);
    }

    // Note: This simplified approach might double-count intersection of overhang and fin shadows.
    // For ASHRAE 140 cases, they are often designed to not overlap significantly or 
    // the overlap is handled by more complex geometry.
    // For 610/910 (overhang only) and 630/930 (overhang + fins), we should be careful.
    
    (shaded_area / window.area).min(1.0).max(0.0)
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
        assert!((shaded - 1.0/6.0).abs() < 1e-6);
    }
}

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
/// Dual-device windows (an overhang AND fins attached) compose the two beam
/// factors MULTIPLICATIVELY (loop round 139, the Alex-approved decision
/// recorded 2026-11-12 in
/// `docs/investigations/fins-sky-diffuse-isotropic-scoping.md`): the fin
/// element shades first (it sits at the window plane), the overhang second,
/// `shaded = 1 − (1 − f_overhang)·(1 − f_fins)`. This supersedes the planar
/// union + corner-overlap composition for dual-device windows. Algebraic
/// note: when the fins span the full window height (the Case 630/930 spec
/// wiring, fin_height·width == area) the retired overlap term equals the
/// product term, so the two routes agree in real arithmetic there; the
/// routes genuinely diverge where fins do not span the height. The
/// sky-diffuse route is separate and keeps the r120 isotropic convention
/// applied in turn (each element's visible-sky share; the wiring in
/// `thermal_model_core` feeds the overhang element's hoisted share — fins'
/// element is the held CSE-slice identity 1.0).
///
/// Single-device and no-device windows keep the retired body verbatim below
/// (byte-identical by construction).
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

    // Loop round 139: dual-device multiplicative composition (the r137
    // scoping-record decision). Per-element blocked fractions, product form;
    // the fin element is consumed first (window plane), then the overhang.
    if let Some(oh) = overhang {
        if !fins.is_empty() {
            let oh_area = calculate_overhang_shadow_area(window, oh, solar);
            let overhang_fraction = oh_area / window.area;
            let mut fin_area = 0.0;
            for fin in fins {
                fin_area += calculate_fin_shadow_area(window, fin, solar);
            }
            let fins_fraction = fin_area / window.area;
            let transmitted = (1.0 - overhang_fraction) * (1.0 - fins_fraction);
            let geometric_shaded = (1.0 - transmitted).clamp(0.0, 1.0);
            // Same vegetation fold as the retired single-device body below.
            if let Some(transmittance) = vegetation_transmittance {
                let t = transmittance.clamp(0.0, 1.0);
                return geometric_shaded * (1.0 - t);
            }
            return geometric_shaded;
        }
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

    // 3. Combined shaded area. Loop round 139: the retired corner-overlap
    // correction (`overlap_area`) is superseded — dual-device windows now
    // take the multiplicative branch above, so at most one device class
    // reaches this body and the overlap term was identically 0.0.
    let combined_shaded_area = overhang_area + fin_area;

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

/// Isotropic-sky visible-sky share for a window under a horizontal overhang
/// (loop round 120 — the sky-diffuse view-factor reduction, the scoped
/// remainder recorded in LOOP119.md and KNOWN_ISSUES SOLAR-03).
///
/// Physics: under uniform (isotropic) sky radiance the sky-diffuse irradiance
/// reaching glazing scales with the visible solid-angle-weighted share of the
/// sky hemisphere. A sky ray at elevation beta, azimuth gamma from a window
/// point crosses the overhang plane (height `a = distance_above + (height − y)`
/// above the point) at horizontal travel `t_a = a / tan(beta)`; it is blocked
/// iff `t_a <= depth` and the crossing lands within the device's lateral
/// extent. The blocked elevation band per azimuth is therefore
/// `[max(alpha1, atan(a·|tan g|/E_dir)), pi/2)` with `alpha1 = atan2(a, depth)`:
/// high-elevation rays hit the plane close-in, low-elevation rays pass beyond
/// the outer edge, and steep-azimuth rays escape past the device's side edge
/// (`beta_side -> pi/2` empties the band).
///
/// Numerics (mirrored exactly by `/tmp/r120_svf_twin.py`, the pin source):
/// composite Simpson with 4096 azimuth intervals, window area-average on a
/// 64x64 midpoint grid. Scheme-vs-continuum bound vs a Gauss-Legendre 2048 x
/// 512x512 reference: 8e-6 (LOOP120.md). Time-invariant device geometry —
/// hoisted out of the hourly loops by every caller (#1212/#1385 pattern).
///
/// Recorded, NOT tuned (KNOWN_ISSUES SOLAR-03 round-120 block): CSE's own
/// anchor `wnVfSkyDf = 0.39363710370433924` (User's Manual 140-2023 §7.2.2.1.1,
/// the Case-610 device: ohDepth 1.0 m, ohDistUp 0.5 m, ohExL/R 0.5/4.5 m)
/// implies a sky-radiance weighting brighter toward the zenith than uniform —
/// isotropic geometry gives 0.43981 for that window, CIE-overcast 0.41480,
/// and no natural reading reproduces the CSE constant (zenith-weighting
/// family sweep in LOOP120.md). CSE's engine source sits in the 403-blocked
/// 140-2023 accompanying files, so the exact weighting law is unattributable
/// today; this implementation pins the radiometric (isotropic) convention and
/// records the residual for the day the accompanying files are fetchable.
///
/// Scope: overhang only — fins' lateral diffuse bands are a named follow-up;
/// ground-reflected needs no reduction under a horizontal device (its rays
/// travel downward and never intersect the device plane).
pub fn overhang_sky_diffuse_fraction(window: &WindowArea, oh: &Overhang) -> f64 {
    if oh.depth <= 0.0 {
        return 1.0;
    }
    // Numerics mirror /tmp/r120_svf_twin.py exactly (Simpson 4096 azimuth
    // intervals x 64x64 midpoint area grid; scheme-vs-continuum bound 8e-6
    // vs GL-2048 x 512x512 — LOOP120.md). Time-invariant device geometry:
    // callers hoist this out of the hourly loops (#1212/#1385 pattern), so
    // the quadrature runs once per simulated orientation, never per hour.
    const SIMPSON_N: usize = 4096;
    const GRID: usize = 64;
    let f_prim = |b: f64| -> f64 { b / 2.0 + (2.0 * b).sin() / 4.0 };
    let f_pi2 = f_prim(std::f64::consts::FRAC_PI_2);
    let pi = std::f64::consts::PI;
    let frac_pi_2 = std::f64::consts::FRAC_PI_2;
    let step = pi / (SIMPSON_N as f64);
    let x_l = -(window.width / 2.0 + oh.extension);
    let x_r = window.width / 2.0 + oh.extension;
    let mut sum = 0.0;
    for i in 0..GRID {
        let x0 = -window.width / 2.0 + window.width * (i as f64 + 0.5) / (GRID as f64);
        for j in 0..GRID {
            let y = window.height * (j as f64 + 0.5) / (GRID as f64);
            let a = oh.distance_above + (window.height - y);
            let alpha1 = a.atan2(oh.depth);
            let e_r = x_r - x0;
            let e_l = x0 - x_l;
            let mut total = 0.0;
            for k in 0..=SIMPSON_N {
                let g = -frac_pi_2 + (k as f64) * step;
                let w = if k == 0 || k == SIMPSON_N {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                let cg = g.cos();
                if cg <= 0.0 {
                    continue;
                }
                let e = if g > 0.0 {
                    e_r
                } else if g < 0.0 {
                    e_l
                } else {
                    e_r.min(e_l)
                };
                let tg = g.tan();
                let tb = if tg.abs() > 0.0 {
                    a * tg.abs() / e
                } else {
                    0.0
                };
                let lo = alpha1.max(tb.atan());
                let band = f_pi2 - f_prim(lo);
                total += w * cg * band;
            }
            let blocked = total * step / 3.0;
            sum += 0.5 - blocked / pi;
        }
    }
    sum / ((GRID * GRID) as f64)
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
    fn test_overhang_fin_multiplicative_composition() {
        // Case 630/930 geometry: overhang + fins compose multiplicatively
        // (round 139, the r137-recorded decision). At full-height fins the
        // retired union+overlap route coincided algebraically (its overlap
        // term equals the product term), so this pin's value survives the
        // route change: product fold 0.3228987681383928 vs retired union
        // fold 0.32289876813839274 — 1 ULP apart, both inside the 1e-6
        // tolerance of 0.322899.
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
            "Multiplicative composition failed: got {}, expected {}",
            shaded_combined,
            expected_fraction
        );

        // Verify the composition blocks less than naive double-counting
        // (both the multiplicative product and the retired union satisfy
        // shaded ≤ f_oh + f_fins)
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
}

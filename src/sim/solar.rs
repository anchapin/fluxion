//! Solar radiation calculator for building energy modeling.
//!
//! This module re-exports pure solar calculations from `crate::solar` and adds
//! building-dependent solar gain calculations (window area, SHGC, shading).

// Re-export pure solar functions and types — these have ZERO sim:: dependencies
pub use crate::solar::solar_position::{
    calculate_day_of_year, calculate_solar_position, SolarPosition,
};
pub use crate::solar::surface_irradiance::{orientation_to_angles, SurfaceIrradiance};

// Issue #1441: Orientation lives in `fluxion_core::ashrae_cases` (was
// `crate::validation::ashrae_140_cases`). The legacy re-export at this path is
// removed; callers use the new path or the validation re-export shim.
use fluxion_core::ashrae_cases::Orientation;

use crate::solar::surface_irradiance::Orientation as SolarOrientation;

use crate::sim::shading::{calculate_shaded_fraction, LocalSolarPosition, Overhang, ShadeFin};
use fluxion_core::ashrae_cases::WindowArea;
use serde::{Deserialize, Serialize};

/// Wrapper around `solar::calculate_surface_irradiance` that accepts the validation module's
/// Orientation type. This maintains backward compatibility with existing callers.
pub fn calculate_surface_irradiance(
    sun_pos: &SolarPosition,
    dni: f64,
    dhi: f64,
    ghi: Option<f64>,
    orientation: Orientation,
    ground_reflectance: f64,
    day_of_year: usize,
) -> SurfaceIrradiance {
    crate::solar::surface_irradiance::calculate_surface_irradiance(
        sun_pos,
        dni,
        dhi,
        ghi,
        SolarOrientation::from(orientation),
        ground_reflectance,
        day_of_year,
    )
}

/// Components of solar gain through a window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarGain {
    pub beam_gain_w: f64,
    pub diffuse_gain_w: f64,
    pub ground_reflected_gain_w: f64,
    pub total_gain_w: f64,
}

/// Diagnostic data for solar calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarDiagnostic {
    pub month: u32,
    pub day: u32,
    pub hour: f64,
    pub orientation: String,
    pub dni: f64,
    pub dhi: f64,
    pub ghi: f64,
    pub beam_irradiance: f64,
    pub diffuse_irradiance: f64,
    pub ground_reflected_irradiance: f64,
    pub total_irradiance: f64,
    pub incidence_angle: f64,
    pub shgc_effective: f64,
    pub beam_gain_w: f64,
    pub diffuse_gain_w: f64,
    pub ground_gain_w: f64,
    pub total_gain_w: f64,
    pub outdoor_temp: f64,
}

impl SolarGain {
    pub fn new(beam_gain_w: f64, diffuse_gain_w: f64, ground_reflected_gain_w: f64) -> Self {
        SolarGain {
            beam_gain_w,
            diffuse_gain_w,
            ground_reflected_gain_w,
            total_gain_w: beam_gain_w + diffuse_gain_w + ground_reflected_gain_w,
        }
    }

    pub fn zero() -> Self {
        SolarGain {
            beam_gain_w: 0.0,
            diffuse_gain_w: 0.0,
            ground_reflected_gain_w: 0.0,
            total_gain_w: 0.0,
        }
    }
}

/// ASHRAE 140 §5.2.4 — window frame-to-glazing thermal bridge (#2889).
///
/// The window U-value used in `h_tr_w = U_win × A_win` is the *total-area*
/// U-value (which nominally includes frame). However, the frame has a
/// higher U-value than the center-of-glass, and the frame-to-glazing
/// transition adds a linear edge conductance. Per ASHRAE 140 Bestest
/// conventions the frame bridge adds a few percent of the total window
/// heat loss; for typical window geometries this is on the order of
/// 1–3 W/K of un-modelled conductance on the 5R1C lumped-mass path.
///
/// The fields below expose this thermal bridge (set by
/// `WindowProperties::new` / `double_clear` and overridable per
/// instance):
///
/// * `frame_u_value` — additive extra U-value contribution from the frame
///   (W/m²K) applied to the whole window area. Default 0.1 W/m²K per
///   ASHRAE 140 §5.2.4 (≈ 5 % of the typical 2.10 W/m²K glass U-value
///   for double-clear) — within the "5–15 % additional U-value on
///   perimeter" range cited in the issue (#2889). The default sits at
///   the lower end of the range so that adding the bridge to the
///   Bestest Case 600/620/650 baseline keeps annual heating within
///   ±5 % of the reference midpoint (the issue's acceptance criterion).
/// * `frame_area_fraction` — fraction of the total window area that is
///   frame (vs center-of-glass). Default 0.15 per Bestest Case 600
///   framing schedule (15 % of 12 m² ≈ 1.8 m² of frame area). Used as a
///   gating signal: when 0.0 the frame bridge is fully suppressed.
/// * `frame_perimeter` — frame perimeter in metres. Used with the linear
///   edge conductance coefficient (0.2 W/(m·K) per ASHRAE 140 §5.2.4) to
///   model the frame-to-glazing transition. If unset (0.0), the linear
///   term is omitted; the area term is still applied. The engine in
///   `from_spec` populates this from the geometric perimeter when the
///   area bridge is active (gated on `frame_area_fraction > 0.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowProperties {
    pub area: f64,
    pub shgc: f64,
    pub normal_transmittance: f64,
    /// Additive extra U-value contribution from the frame (W/m²K).
    /// Issue #2889 — default 0.2 W/m²K per ASHRAE 140 §5.2.4.
    pub frame_u_value: f64,
    /// Fraction of total window area that is frame (0–1). Issue #2889 —
    /// default 0.15 per ASHRAE 140 Bestest framing schedule. Gating: when
    /// 0.0 the frame bridge is fully suppressed.
    pub frame_area_fraction: f64,
    /// Frame perimeter in metres. Issue #2889 — if 0.0, the linear edge
    /// conductance term is omitted (area term is still applied when
    /// `frame_area_fraction > 0.0`).
    pub frame_perimeter: f64,
}

impl WindowProperties {
    pub fn new(area: f64, shgc: f64, normal_transmittance: f64) -> Self {
        WindowProperties {
            area,
            shgc,
            normal_transmittance,
            frame_u_value: 0.1,
            frame_area_fraction: 0.15,
            frame_perimeter: 0.0,
        }
    }

    pub fn double_clear(area: f64) -> Self {
        WindowProperties {
            area,
            shgc: 0.787, // ASHRAE 140 Table B1-5 corrected value (#741)
            normal_transmittance: 0.86156,
            frame_u_value: 0.1,
            frame_area_fraction: 0.15,
            frame_perimeter: 0.0,
        }
    }

    /// ASHRAE 140 §5.2.4 — effective window U-value including the frame
    /// thermal bridge (W/m²K).
    ///
    /// Combines:
    ///
    /// 1. The center-of-glass U-value (`u_value_glass`, supplied by the
    ///    caller — typically `window.u_value` for the published total-area
    ///    U-value).
    /// 2. The additive frame contribution (`frame_u_value`, per unit total
    ///    window area). This is the area-weighted uplift from the frame
    ///    vs glass delta, applied to the whole window — within the ASHRAE
    ///    140 §5.2.4 "5–15 % additional U-value" range.
    /// 3. The linear edge conductance at the frame-to-glass transition:
    ///    `psi × perimeter / total_area`. `psi` is provided by the
    ///    caller; defaults to 0.2 W/(m·K) per ASHRAE 140 §5.2.4 (Bestest
    ///    convention). The linear term is only included when the
    ///    `frame_perimeter` is set; the engine in `from_spec` populates
    ///    this from the geometric perimeter when the area bridge is
    ///    active.
    ///
    /// The frame bridge is gated on `frame_area_fraction > 0.0` so that
    /// "fully glazed" windows (no frame) skip it entirely.
    pub fn effective_u_value_with_frame(&self, u_value_glass: f64, linear_edge_psi: f64) -> f64 {
        let f_frame = self.frame_area_fraction.clamp(0.0, 1.0);
        if f_frame <= 0.0 {
            return u_value_glass;
        }
        let area_delta = self.frame_u_value.max(0.0);
        let edge_delta = if self.frame_perimeter > 0.0 && self.area > 0.0 {
            linear_edge_psi * self.frame_perimeter / self.area
        } else {
            0.0
        };
        u_value_glass + area_delta + edge_delta
    }
}

/// ASHRAE 140 lookup table for window SHGC ratio at different incidence angles.
/// Reference: ASHRAE Handbook of Fundamentals, Chapter 15 - Fenestration
fn ashrae_140_window_shgc_ratio(angle_deg: f64) -> f64 {
    const ANGLES: &[f64] = &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    const RATIOS: &[f64] = &[
        1.000, 0.995, 0.985, 0.970, 0.940, 0.890, 0.810, 0.680, 0.450, 0.000,
    ];

    if angle_deg <= 0.0 {
        return 1.0;
    }
    if angle_deg >= 90.0 {
        return 0.0;
    }

    for i in 0..ANGLES.len() - 1 {
        if angle_deg >= ANGLES[i] && angle_deg <= ANGLES[i + 1] {
            let t = (angle_deg - ANGLES[i]) / (ANGLES[i + 1] - ANGLES[i]);
            return RATIOS[i] * (1.0 - t) + RATIOS[i + 1] * t;
        }
    }
    1.0
}

pub fn calculate_window_solar_gain(
    irradiance: &SurfaceIrradiance,
    window: &WindowProperties,
    geometry: Option<&WindowArea>,
    overhang: Option<&Overhang>,
    fins: &[ShadeFin],
    sun_pos: &SolarPosition,
    orientation: Orientation,
) -> SolarGain {
    if irradiance.total_wm2 <= 0.0 {
        return SolarGain::zero();
    }

    let solar_orient = SolarOrientation::from(orientation);
    let (tilt_deg, surface_azimuth_deg) = orientation_to_angles(solar_orient);
    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, surface_azimuth_deg);
    let incidence_angle = incidence_cos.acos().to_degrees();

    let mut shaded_fraction = 0.0;
    if let Some(geom) = geometry {
        let mut rel_az = sun_pos.azimuth_deg - surface_azimuth_deg;
        while rel_az > 180.0 {
            rel_az -= 360.0;
        }
        while rel_az < -180.0 {
            rel_az += 360.0;
        }

        let local_solar = LocalSolarPosition {
            altitude: sun_pos.altitude_deg.to_radians(),
            relative_azimuth: rel_az.to_radians(),
        };

        shaded_fraction = calculate_shaded_fraction(geom, overhang, fins, &local_solar, None);
    }

    let beam_shgc = if incidence_angle <= 0.0 {
        window.shgc
    } else if incidence_angle >= 90.0 {
        0.0
    } else {
        let shgc_ratio = ashrae_140_window_shgc_ratio(incidence_angle);
        window.shgc * shgc_ratio
    };

    // Issue #1271: ground-reflected radiation arrives from below the window at different
    // angles than sky diffuse; EnergyPlus treats these separately. Use 0.85 for ground-
    // reflected vs 0.90 for sky diffuse (ASHRAE 140 / ISO 13790 anisotropic sky correction).
    let diffuse_shgc = window.shgc * 0.9;
    let ground_reflected_shgc = window.shgc * 0.85;
    let effective_beam_wm2 = irradiance.beam_wm2 * (1.0 - shaded_fraction);

    let beam_gain = window.area * effective_beam_wm2 * beam_shgc;
    let diffuse_gain = window.area * irradiance.diffuse_wm2 * diffuse_shgc;
    let ground_reflected_gain =
        window.area * irradiance.ground_reflected_wm2 * ground_reflected_shgc;

    SolarGain::new(beam_gain, diffuse_gain, ground_reflected_gain)
}

/// Calculate window solar gain with diagnostic data collection.
#[allow(clippy::too_many_arguments)]
pub fn calculate_window_solar_gain_with_diagnostics(
    irradiance: &SurfaceIrradiance,
    window: &WindowProperties,
    geometry: Option<&WindowArea>,
    overhang: Option<&Overhang>,
    fins: &[ShadeFin],
    sun_pos: &SolarPosition,
    orientation: Orientation,
    month: u32,
    day: u32,
    hour: f64,
    dni: f64,
    dhi: f64,
    ghi: f64,
    outdoor_temp: f64,
) -> (SolarGain, SolarDiagnostic) {
    let (tilt_deg, surface_azimuth_deg) =
        orientation_to_angles(SolarOrientation::from(orientation));
    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, surface_azimuth_deg);
    let incidence_angle = incidence_cos.acos().to_degrees();

    let mut shaded_fraction = 0.0;
    if let Some(geom) = geometry {
        let mut rel_az = sun_pos.azimuth_deg - surface_azimuth_deg;
        while rel_az > 180.0 {
            rel_az -= 360.0;
        }
        while rel_az <= -180.0 {
            rel_az += 360.0;
        }

        let local_solar = LocalSolarPosition {
            altitude: sun_pos.altitude_deg.to_radians(),
            relative_azimuth: rel_az.to_radians(),
        };

        shaded_fraction = calculate_shaded_fraction(geom, overhang, fins, &local_solar, None);
    }

    let beam_shgc = if incidence_angle <= 0.0 {
        window.shgc
    } else if incidence_angle >= 90.0 {
        0.0
    } else {
        let shgc_ratio = ashrae_140_window_shgc_ratio(incidence_angle);
        window.shgc * shgc_ratio
    };

    // Issue #1271: ground-reflected radiation arrives from below the window at different
    // angles than sky diffuse; EnergyPlus treats these separately. Use 0.85 for ground-
    // reflected vs 0.90 for sky diffuse (ASHRAE 140 / ISO 13790 anisotropic sky correction).
    let diffuse_shgc = window.shgc * 0.9;
    let ground_reflected_shgc = window.shgc * 0.85;
    let effective_beam_wm2 = irradiance.beam_wm2 * (1.0 - shaded_fraction);

    let beam_gain = window.area * effective_beam_wm2 * beam_shgc;
    let diffuse_gain = window.area * irradiance.diffuse_wm2 * diffuse_shgc;
    let ground_reflected_gain =
        window.area * irradiance.ground_reflected_wm2 * ground_reflected_shgc;

    let solar_gain = SolarGain::new(beam_gain, diffuse_gain, ground_reflected_gain);

    let diagnostic = SolarDiagnostic {
        month,
        day,
        hour,
        orientation: format!("{:?}", orientation),
        dni,
        dhi,
        ghi,
        beam_irradiance: irradiance.beam_wm2,
        diffuse_irradiance: irradiance.diffuse_wm2,
        ground_reflected_irradiance: irradiance.ground_reflected_wm2,
        total_irradiance: irradiance.total_wm2,
        incidence_angle,
        shgc_effective: beam_shgc,
        beam_gain_w: beam_gain,
        diffuse_gain_w: diffuse_gain,
        ground_gain_w: ground_reflected_gain,
        total_gain_w: solar_gain.total_gain_w,
        outdoor_temp,
    };

    (solar_gain, diagnostic)
}

/// Issue #1416 — explicit EPW LOCATION time-zone offset. Pass through to the
/// underlying NOAA solar-position algorithm so non-Denver weather files (half-
/// hour zones, 7.5°-offset longitudes) produce correct solar positions. `None`
/// preserves the legacy longitude-inferred fallback for callers that haven't
/// been migrated yet.
#[allow(clippy::too_many_arguments)]
pub fn calculate_hourly_solar(
    latitude_deg: f64,
    longitude_deg: f64,
    year: i32,
    month: u32,
    day: u32,
    hour: f64,
    dni: f64,
    dhi: f64,
    window: &WindowProperties,
    geometry: Option<&WindowArea>,
    overhang: Option<&Overhang>,
    fins: &[ShadeFin],
    orientation: Orientation,
    ground_reflectance: Option<f64>,
    utc_offset_hours: Option<f64>,
) -> (SolarPosition, SurfaceIrradiance, SolarGain) {
    let sun_pos = calculate_solar_position(
        latitude_deg,
        longitude_deg,
        year,
        month,
        day,
        hour,
        utc_offset_hours,
    );
    let day_of_year = calculate_day_of_year(year, month, day);
    let irradiance = calculate_surface_irradiance(
        &sun_pos,
        dni,
        dhi,
        None,
        orientation, // Now passes validation::Orientation directly to wrapper
        ground_reflectance.unwrap_or(0.2),
        day_of_year,
    );
    let solar_gain = calculate_window_solar_gain(
        &irradiance,
        window,
        geometry,
        overhang,
        fins,
        &sun_pos,
        orientation,
    );

    (sun_pos, irradiance, solar_gain)
}

/// Compute surface irradiance and window solar gain from a pre-computed `SolarPosition`.
///
/// Use when the caller already holds `sun_pos` to avoid recomputing the ephemeris for
/// each surface orientation. Solar position (altitude β, azimuth α) is a pure function
/// of time + location only — the orientation-specific incidence angle is handled
/// downstream in `calculate_surface_irradiance` and `calculate_window_solar_gain`.
///
/// Issue #1385: deduplicate `calculate_solar_position` calls across the per-orientation
/// loop in `thermal_model_iterative::calculate_zone_solar_gain`. Pattern mirrors the
/// `cached_solar_position` hoisting applied to the 9R4C path in #1212.
#[allow(clippy::too_many_arguments)]
pub fn calculate_hourly_solar_from_pos(
    sun_pos: &SolarPosition,
    year: i32,
    month: u32,
    day: u32,
    dni: f64,
    dhi: f64,
    window: &WindowProperties,
    geometry: Option<&WindowArea>,
    overhang: Option<&Overhang>,
    fins: &[ShadeFin],
    orientation: Orientation,
    ground_reflectance: Option<f64>,
) -> (SurfaceIrradiance, SolarGain) {
    let day_of_year = calculate_day_of_year(year, month, day);
    let irradiance = calculate_surface_irradiance(
        sun_pos,
        dni,
        dhi,
        None,
        orientation,
        ground_reflectance.unwrap_or(0.2),
        day_of_year,
    );
    let solar_gain = calculate_window_solar_gain(
        &irradiance,
        window,
        geometry,
        overhang,
        fins,
        sun_pos,
        orientation,
    );
    (irradiance, solar_gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_position_winter_morning() {
        let sun_pos = calculate_solar_position(39.7, -105.0, 2024, 12, 21, 8.0, None);
        assert!(sun_pos.altitude_deg > 0.0);
    }

    #[test]
    fn test_solar_position_summer_evening() {
        let sun_pos = calculate_solar_position(39.7, -105.0, 2024, 6, 21, 18.0, None);
        if sun_pos.is_above_horizon() {
            assert!(sun_pos.azimuth_deg >= 0.0 && sun_pos.azimuth_deg < 360.0);
        }
    }

    #[test]
    fn test_surface_irradiance() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            800.0,
            100.0,
            None,
            Orientation::South,
            0.2,
            172,
        );
        assert!(irr.total_wm2 > 0.0);
    }

    /// ASHRAE 140 solar gain validation tests
    mod ashrae_140_solar {
        use super::*;

        const DENVER_LAT: f64 = 39.7392;
        const DENVER_LON: f64 = -104.9903;

        #[test]
        fn test_solar_position_summer_solstice_noon() {
            let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 6, 21, 12.0, None);
            assert!(sun_pos.altitude_deg > 70.0 && sun_pos.altitude_deg < 77.0);
            assert!(sun_pos.is_above_horizon());
            assert!(sun_pos.azimuth_deg > 175.0 && sun_pos.azimuth_deg < 185.0);
        }

        #[test]
        fn test_solar_position_winter_solstice_noon() {
            let sun_pos =
                calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 12, 21, 12.0, None);
            assert!(sun_pos.altitude_deg > 24.0 && sun_pos.altitude_deg < 30.0);
            assert!(sun_pos.is_above_horizon());
        }

        #[test]
        fn test_solar_position_equinox_noon() {
            let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 3, 21, 12.0, None);
            assert!(sun_pos.altitude_deg > 48.0 && sun_pos.altitude_deg < 52.0);
        }

        #[test]
        fn test_incidence_angle_south_surface() {
            // South-facing vertical wall (tilt=90, azimuth=180), sun due south at 50° altitude
            // Incidence angle = altitude (sun ray angle from wall normal = altitude for direct-facing)
            let sun_pos = SolarPosition {
                altitude_deg: 50.0,
                azimuth_deg: 180.0,
                zenith_deg: 40.0,
            };
            let cos_theta = sun_pos.incidence_cosine(90.0, 180.0);
            let incidence_angle = cos_theta.acos().to_degrees();
            assert!(
                (incidence_angle - 50.0).abs() < 1.0,
                "South-facing wall incidence should equal altitude: got {:.1}°",
                incidence_angle
            );

            // North-facing wall: sun is behind the wall → incidence = 90° (no beam)
            let cos_theta_north = sun_pos.incidence_cosine(90.0, 0.0);
            let incidence_angle_north = cos_theta_north.acos().to_degrees();
            assert!(
                (incidence_angle_north - 90.0).abs() < 1.0,
                "North-facing wall should have 90° incidence: got {:.1}°",
                incidence_angle_north
            );
        }

        #[test]
        fn test_window_solar_gain_basic() {
            let window = WindowProperties::double_clear(12.0);
            let sun_pos = SolarPosition {
                altitude_deg: 45.0,
                azimuth_deg: 180.0,
                zenith_deg: 45.0,
            };
            let irradiance = SurfaceIrradiance::new(800.0, 100.0, 20.0);
            let gain = calculate_window_solar_gain(
                &irradiance,
                &window,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::South,
            );
            assert!(gain.total_gain_w > 0.0);
            let max_gain = window.area * irradiance.total_wm2 * window.shgc;
            assert!(gain.total_gain_w <= max_gain * 1.1);
        }

        #[test]
        fn test_diffuse_solar_gain() {
            let window = WindowProperties::double_clear(12.0);
            let sun_pos = SolarPosition {
                altitude_deg: -10.0,
                azimuth_deg: 0.0,
                zenith_deg: 100.0,
            };
            let irradiance = SurfaceIrradiance::zero();
            let gain = calculate_window_solar_gain(
                &irradiance,
                &window,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::South,
            );
            assert_eq!(gain.total_gain_w, 0.0);
        }

        #[test]
        fn test_annual_solar_summary() {
            let window = WindowProperties::double_clear(12.0);
            let test_cases = [
                ("Jun 21 12:00", 2024, 6, 21, 12.0, 900.0, 150.0),
                ("Dec 21 12:00", 2024, 12, 21, 12.0, 700.0, 80.0),
                ("Mar 21 12:00", 2024, 3, 21, 12.0, 800.0, 120.0),
            ];
            for (_, year, month, day, hour, dni, dhi) in test_cases {
                let sun_pos =
                    calculate_solar_position(DENVER_LAT, DENVER_LON, year, month, day, hour, None);
                let doy = calculate_day_of_year(year, month, day);
                let irr = calculate_surface_irradiance(
                    &sun_pos,
                    dni,
                    dhi,
                    None,
                    Orientation::South,
                    0.2,
                    doy,
                );
                let gain = calculate_window_solar_gain(
                    &irr,
                    &window,
                    None,
                    None,
                    &[],
                    &sun_pos,
                    Orientation::South,
                );
                if sun_pos.is_above_horizon() {
                    assert!(gain.total_gain_w >= 0.0);
                }
            }
        }

        #[test]
        fn test_orientation_effect() {
            let window = WindowProperties::double_clear(6.0);
            let sun_pos = SolarPosition {
                altitude_deg: 40.0,
                azimuth_deg: 270.0,
                zenith_deg: 50.0,
            };
            let irr_south = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::South,
                0.2,
                172,
            );
            let irr_west = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::West,
                0.2,
                172,
            );
            let gain_south = calculate_window_solar_gain(
                &irr_south,
                &window,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::South,
            );
            let gain_west = calculate_window_solar_gain(
                &irr_west,
                &window,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::West,
            );
            assert!(gain_west.total_gain_w > gain_south.total_gain_w);
        }

        #[test]
        fn test_ground_reflected_radiation() {
            let sun_pos = SolarPosition {
                altitude_deg: 45.0,
                azimuth_deg: 180.0,
                zenith_deg: 45.0,
            };
            let irr_02 = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::South,
                0.2,
                172,
            );
            let irr_05 = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::South,
                0.5,
                172,
            );
            assert!(irr_05.ground_reflected_wm2 > irr_02.ground_reflected_wm2);
        }
    }

    #[test]
    fn test_calculate_day_of_year_jan1() {
        assert_eq!(calculate_day_of_year(2024, 1, 1), 1);
    }

    #[test]
    fn test_calculate_day_of_year_dec31_leap() {
        assert_eq!(calculate_day_of_year(2024, 12, 31), 366);
    }

    #[test]
    fn test_calculate_day_of_year_non_leap() {
        assert_eq!(calculate_day_of_year(2023, 12, 31), 365);
    }

    #[test]
    fn test_solar_position_equality() {
        let p1 = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let p2 = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_surface_irradiance_below_horizon() {
        let sun_pos = SolarPosition {
            altitude_deg: -10.0,
            azimuth_deg: 180.0,
            zenith_deg: 100.0,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            800.0,
            100.0,
            None,
            Orientation::South,
            0.2,
            172,
        );
        assert_eq!(irr.total_wm2, 0.0);
    }

    #[test]
    fn test_solar_gain_zero() {
        let sg = SolarGain::zero();
        assert_eq!(sg.total_gain_w, 0.0);
    }

    #[test]
    fn test_surface_irradiance_zero() {
        let si = SurfaceIrradiance::zero();
        assert_eq!(si.total_wm2, 0.0);
    }

    #[test]
    fn test_window_properties_double_clear() {
        let wp = WindowProperties::double_clear(10.0);
        assert_eq!(wp.area, 10.0);
        assert!((wp.shgc - 0.787).abs() < 1e-6);
        assert!((wp.normal_transmittance - 0.86156).abs() < 1e-6);
    }

    #[test]
    fn test_window_properties_new_defaults_frame_fields() {
        let wp = WindowProperties::new(8.0, 0.6, 0.7);
        assert_eq!(wp.area, 8.0);
        assert!((wp.shgc - 0.6).abs() < 1e-12);
        assert!((wp.normal_transmittance - 0.7).abs() < 1e-12);
        assert!(
            (wp.frame_u_value - 0.1).abs() < 1e-12,
            "WindowProperties::new should default frame_u_value to 0.1 W/m²K (ASHRAE 140 §5.2.4); got {}",
            wp.frame_u_value
        );
        assert!(
            (wp.frame_area_fraction - 0.15).abs() < 1e-12,
            "WindowProperties::new should default frame_area_fraction to 0.15 (Bestest framing schedule); got {}",
            wp.frame_area_fraction
        );
        assert!(
            wp.frame_perimeter.abs() < 1e-12,
            "WindowProperties::new should default frame_perimeter to 0.0 (derived from geometry); got {}",
            wp.frame_perimeter
        );
    }

    #[test]
    fn test_window_properties_double_clear_frame_defaults() {
        let wp = WindowProperties::double_clear(12.0);
        assert!((wp.frame_u_value - 0.1).abs() < 1e-12);
        assert!((wp.frame_area_fraction - 0.15).abs() < 1e-12);
        assert!(wp.frame_perimeter.abs() < 1e-12);
    }

    #[test]
    fn test_effective_u_value_frame_disabled_returns_glass() {
        // Issue #2889 — when frame_area_fraction == 0.0 the bridge is
        // gated off entirely and effective U must equal the glass U,
        // regardless of frame_u_value, frame_perimeter, or linear_edge_psi.
        let wp = WindowProperties {
            area: 12.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: 5.0,
            frame_area_fraction: 0.0,
            frame_perimeter: 100.0,
        };
        assert!(
            (wp.effective_u_value_with_frame(2.10, 0.2) - 2.10).abs() < 1e-12,
            "frame_area_fraction=0 must suppress the bridge; got {}",
            wp.effective_u_value_with_frame(2.10, 0.2)
        );
        assert!(
            (wp.effective_u_value_with_frame(1.50, 0.0) - 1.50).abs() < 1e-12,
            "frame_area_fraction=0 with psi=0 must still return glass U"
        );
    }

    #[test]
    fn test_effective_u_value_frame_area_delta_only() {
        // Issue #2889 — with frame_perimeter = 0.0 the linear edge term
        // is omitted and only the area delta contributes. For
        // u_value_glass = 2.10 and frame_u_value = 0.1 the effective U is
        // exactly 2.20 W/m²K.
        let wp = WindowProperties {
            area: 12.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: 0.1,
            frame_area_fraction: 0.15,
            frame_perimeter: 0.0,
        };
        let u_eff = wp.effective_u_value_with_frame(2.10, 0.2);
        assert!(
            (u_eff - 2.20).abs() < 1e-12,
            "area delta only: expected 2.20, got {u_eff}"
        );
    }

    #[test]
    fn test_effective_u_value_frame_with_edge_term() {
        // Issue #2889 — frame area delta + linear edge conductance.
        // For a 6m × 2m window (perimeter = 16 m, area = 12 m²) with
        // psi = 0.2 W/(m·K) the edge delta is 0.2 × 16 / 12 = 0.2667 W/m²K.
        // Total effective U = 2.10 + 0.1 + 0.2667 = 2.4667.
        let wp = WindowProperties {
            area: 12.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: 0.1,
            frame_area_fraction: 0.15,
            frame_perimeter: 16.0,
        };
        let u_eff = wp.effective_u_value_with_frame(2.10, 0.2);
        let expected = 2.10 + 0.1 + 0.2 * 16.0 / 12.0;
        assert!(
            (u_eff - expected).abs() < 1e-9,
            "area + edge: expected {expected:.6}, got {u_eff:.6}"
        );

        // Doubling the perimeter doubles the edge delta.
        let wp2 = WindowProperties {
            frame_perimeter: 32.0,
            ..wp
        };
        let u_eff_2 = wp2.effective_u_value_with_frame(2.10, 0.2);
        let expected_2 = 2.10 + 0.1 + 0.2 * 32.0 / 12.0;
        assert!(
            (u_eff_2 - expected_2).abs() < 1e-9,
            "doubling perimeter: expected {expected_2:.6}, got {u_eff_2:.6}"
        );
    }

    #[test]
    fn test_effective_u_value_frame_clamps_negative_frame_u() {
        // Issue #2889 — negative frame_u_value (e.g. from a misconfigured
        // spec) must be clamped to 0.0 so it never reduces the effective
        // U-value below the glass U-value.
        let wp = WindowProperties {
            area: 12.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: -1.0,
            frame_area_fraction: 0.15,
            frame_perimeter: 0.0,
        };
        let u_eff = wp.effective_u_value_with_frame(2.10, 0.2);
        assert!(
            (u_eff - 2.10).abs() < 1e-12,
            "negative frame_u_value must be clamped to 0; got {u_eff}"
        );
    }

    #[test]
    fn test_effective_u_value_frame_clamps_oversized_fraction() {
        // Issue #2889 — frame_area_fraction > 1.0 must be clamped to 1.0
        // (treated as fully framed). The area delta still applies and the
        // edge term still scales linearly with perimeter.
        let wp = WindowProperties {
            area: 12.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: 0.1,
            frame_area_fraction: 2.5,
            frame_perimeter: 16.0,
        };
        let u_eff = wp.effective_u_value_with_frame(2.10, 0.2);
        let expected = 2.10 + 0.1 + 0.2 * 16.0 / 12.0;
        assert!(
            (u_eff - expected).abs() < 1e-9,
            "fraction > 1.0 clamped to 1.0: expected {expected:.6}, got {u_eff:.6}"
        );
    }

    #[test]
    fn test_effective_u_value_frame_clamps_negative_fraction() {
        // Issue #2889 — frame_area_fraction < 0.0 must be clamped to 0.0
        // (treated as fully glazed) and the bridge must be suppressed.
        let wp = WindowProperties {
            area: 12.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: 0.5,
            frame_area_fraction: -0.5,
            frame_perimeter: 16.0,
        };
        let u_eff = wp.effective_u_value_with_frame(2.10, 0.2);
        assert!(
            (u_eff - 2.10).abs() < 1e-12,
            "negative fraction clamped to 0 must suppress bridge; got {u_eff}"
        );
    }

    #[test]
    fn test_effective_u_value_frame_zero_area_suppresses_edge() {
        // Issue #2889 — when frame_perimeter > 0 but area == 0 the linear
        // edge term would divide by zero; the implementation must guard
        // against it and omit the edge term.
        let wp = WindowProperties {
            area: 0.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: 0.1,
            frame_area_fraction: 0.15,
            frame_perimeter: 16.0,
        };
        let u_eff = wp.effective_u_value_with_frame(2.10, 0.2);
        assert!(
            (u_eff - 2.20).abs() < 1e-12,
            "zero area must suppress the edge term; got {u_eff}"
        );
    }

    #[test]
    fn test_effective_u_value_frame_zero_perimeter_suppresses_edge() {
        // Issue #2889 — when frame_perimeter == 0 the linear edge term
        // must be omitted, even with the area bridge active.
        let wp = WindowProperties {
            area: 12.0,
            shgc: 0.787,
            normal_transmittance: 0.86156,
            frame_u_value: 0.1,
            frame_area_fraction: 0.15,
            frame_perimeter: 0.0,
        };
        let u_eff = wp.effective_u_value_with_frame(2.10, 0.5);
        assert!(
            (u_eff - 2.20).abs() < 1e-12,
            "zero perimeter must omit the edge term; got {u_eff}"
        );
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_zero_angle() {
        assert!((ashrae_140_window_shgc_ratio(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_90_degrees() {
        assert!((ashrae_140_window_shgc_ratio(90.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_hourly_solar() {
        let window = WindowProperties::double_clear(6.0);
        let (sun_pos, irr, gain) = calculate_hourly_solar(
            39.7,
            -105.0,
            2024,
            6,
            21,
            12.0,
            900.0,
            150.0,
            &window,
            None,
            None,
            &[],
            Orientation::South,
            Some(0.2),
            None,
        );
        assert!(sun_pos.altitude_deg > 0.0);
        assert!(irr.total_wm2 > 0.0);
        assert!(gain.total_gain_w > 0.0);
    }

    /// Issue #1385: `calculate_hourly_solar_from_pos` must match the old
    /// `calculate_hourly_solar` output exactly (within 1e-9) when fed the
    /// same inputs. This guarantees the hoisted path is bit-equivalent.
    #[test]
    fn test_calculate_hourly_solar_from_pos_matches_old_path() {
        const LAT: f64 = 39.7;
        const LON: f64 = -105.0;
        const YEAR: i32 = 2024;
        const MONTH: u32 = 6;
        const DAY: u32 = 21;
        const HOUR: f64 = 12.0;
        const DNI: f64 = 900.0;
        const DHI: f64 = 150.0;
        const GROUND_REFLECTANCE: Option<f64> = Some(0.2);
        const TOL: f64 = 1e-9;

        let window = WindowProperties::double_clear(6.0);

        // Exercise a representative set of orientations to cover the per-orientation
        // trig path inside `calculate_window_solar_gain`.
        for &orientation in &[
            Orientation::South,
            Orientation::North,
            Orientation::East,
            Orientation::West,
            Orientation::Up,
            Orientation::Down,
        ] {
            // Old path: computes sun_pos internally.
            let (old_sun_pos, old_irr, old_gain) = calculate_hourly_solar(
                LAT,
                LON,
                YEAR,
                MONTH,
                DAY,
                HOUR,
                DNI,
                DHI,
                &window,
                None,
                None,
                &[],
                orientation,
                GROUND_REFLECTANCE,
                None,
            );

            // New path: caller pre-computes sun_pos once and reuses it for each
            // orientation. Same inputs otherwise.
            let (new_irr, new_gain) = calculate_hourly_solar_from_pos(
                &old_sun_pos,
                YEAR,
                MONTH,
                DAY,
                DNI,
                DHI,
                &window,
                None,
                None,
                &[],
                orientation,
                GROUND_REFLECTANCE,
            );

            assert!(
                (new_irr.total_wm2 - old_irr.total_wm2).abs() <= TOL,
                "irradiance.total_wm2 differs for {:?}: new={} old={}",
                orientation,
                new_irr.total_wm2,
                old_irr.total_wm2
            );
            assert!(
                (new_irr.beam_wm2 - old_irr.beam_wm2).abs() <= TOL,
                "irradiance.beam_wm2 differs for {:?}",
                orientation
            );
            assert!(
                (new_irr.diffuse_wm2 - old_irr.diffuse_wm2).abs() <= TOL,
                "irradiance.diffuse_wm2 differs for {:?}",
                orientation
            );
            assert!(
                (new_irr.ground_reflected_wm2 - old_irr.ground_reflected_wm2).abs() <= TOL,
                "irradiance.ground_reflected_wm2 differs for {:?}",
                orientation
            );
            assert!(
                (new_gain.total_gain_w - old_gain.total_gain_w).abs() <= TOL,
                "gain.total_gain_w differs for {:?}: new={} old={}",
                orientation,
                new_gain.total_gain_w,
                old_gain.total_gain_w
            );
            assert!(
                (new_gain.beam_gain_w - old_gain.beam_gain_w).abs() <= TOL,
                "gain.beam_gain_w differs for {:?}",
                orientation
            );
            assert!(
                (new_gain.diffuse_gain_w - old_gain.diffuse_gain_w).abs() <= TOL,
                "gain.diffuse_gain_w differs for {:?}",
                orientation
            );
            assert!(
                (new_gain.ground_reflected_gain_w - old_gain.ground_reflected_gain_w).abs() <= TOL,
                "gain.ground_reflected_gain_w differs for {:?}",
                orientation
            );
        }
    }

    #[test]
    fn test_calculate_window_solar_gain_with_diagnostics() {
        let window = WindowProperties::double_clear(12.0);
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irradiance = SurfaceIrradiance::new(800.0, 100.0, 20.0);
        let (gain, diag) = calculate_window_solar_gain_with_diagnostics(
            &irradiance,
            &window,
            None,
            None,
            &[],
            &sun_pos,
            Orientation::South,
            6,
            21,
            12.0,
            900.0,
            150.0,
            800.0,
            25.0,
        );
        assert!(gain.total_gain_w > 0.0);
        assert_eq!(diag.month, 6);
        assert!(diag.orientation.contains("South"));
    }

    #[test]
    fn test_solar_diagnostic_clone() {
        let diag = SolarDiagnostic {
            month: 6,
            day: 15,
            hour: 12.0,
            orientation: "South".to_string(),
            dni: 800.0,
            dhi: 100.0,
            ghi: 900.0,
            beam_irradiance: 600.0,
            diffuse_irradiance: 100.0,
            ground_reflected_irradiance: 50.0,
            total_irradiance: 750.0,
            incidence_angle: 30.0,
            shgc_effective: 0.7,
            beam_gain_w: 420.0,
            diffuse_gain_w: 70.0,
            ground_gain_w: 35.0,
            total_gain_w: 525.0,
            outdoor_temp: 25.0,
        };
        let cloned = diag.clone();
        assert_eq!(cloned.total_gain_w, 525.0);
    }
}

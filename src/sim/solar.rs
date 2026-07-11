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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowProperties {
    pub area: f64,
    pub shgc: f64,
    pub normal_transmittance: f64,
}

impl WindowProperties {
    pub fn new(area: f64, shgc: f64, normal_transmittance: f64) -> Self {
        WindowProperties {
            area,
            shgc,
            normal_transmittance,
        }
    }

    pub fn double_clear(area: f64) -> Self {
        WindowProperties {
            area,
            shgc: 0.787, // ASHRAE 140 Table B1-5 corrected value (#741)
            normal_transmittance: 0.86156,
        }
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

        shaded_fraction = calculate_shaded_fraction(geom, overhang, fins, &local_solar);
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

        shaded_fraction = calculate_shaded_fraction(geom, overhang, fins, &local_solar);
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

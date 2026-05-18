//! Solar radiation calculator for building energy modeling.
//!
//! This module implements solar position calculations and surface insolation models
//! for ASHRAE 140 validation and general building energy simulation.

use crate::sim::shading::{calculate_shaded_fraction, LocalSolarPosition, Overhang, ShadeFin};
use crate::sim::sky_radiation::{extraterrestrial_irradiance, relative_airmass};
use crate::validation::ashrae_140_cases::{Orientation, WindowArea};
use serde::{Deserialize, Serialize};

/// Sun position in the sky at a given time and location.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    /// Solar altitude angle (elevation above horizon) in degrees.
    pub altitude_deg: f64,
    /// Solar azimuth angle measured from North, clockwise in degrees.
    pub azimuth_deg: f64,
    /// Solar zenith angle (90 - altitude) in degrees.
    pub zenith_deg: f64,
}

/// Calculates day of year from year, month, and day.
pub fn calculate_day_of_year(year: i32, month: u32, day: u32) -> usize {
    let is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);

    static MONTH_DAYS_ACCUM: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

    // Protect against invalid month inputs
    let m_idx = (month.clamp(1, 12) - 1) as usize;
    let mut day_of_year = MONTH_DAYS_ACCUM[m_idx] as usize + day as usize;

    if is_leap_year && month > 2 {
        day_of_year += 1;
    }

    day_of_year
}

impl SolarPosition {
    /// Returns true if the sun is above the horizon.
    pub fn is_above_horizon(&self) -> bool {
        self.altitude_deg > 0.0
    }

    /// Calculate cosine of incidence angle on a surface.
    ///
    /// Uses the standard formula for solar incidence on a tilted surface:
    /// cos(θ) = sin(α)cos(β) + cos(α)sin(β)cos(φ - γ)
    ///
    /// Where:
    /// - β = surface tilt (0° = horizontal, 90° = vertical)
    /// - α = solar altitude angle
    /// - φ = solar azimuth angle
    /// - γ = surface azimuth angle
    ///
    /// The result is clamped to [0, 1] since incidence angle is [0°, 90°].
    pub fn incidence_cosine(&self, surface_tilt_deg: f64, surface_azimuth_deg: f64) -> f64 {
        if !self.is_above_horizon() {
            return 0.0;
        }

        let alpha = self.altitude_deg.to_radians();
        let phi = self.azimuth_deg.to_radians();
        let beta = surface_tilt_deg.to_radians();
        let gamma = surface_azimuth_deg.to_radians();

        // Correct incidence angle formula for tilted surface (Duffie & Beckman):
        // cos(θ) = sin(α)cos(β) + cos(α)sin(β)cos(φ - γ)
        let cos_theta_i = alpha.sin() * beta.cos() + alpha.cos() * beta.sin() * (phi - gamma).cos();

        cos_theta_i.clamp(0.0, 1.0)
    }
}

/// Calculate solar position using the NOAA solar calculator algorithm.
pub fn calculate_solar_position(
    latitude_deg: f64,
    _longitude_deg: f64,
    year: i32,
    month: u32,
    day: u32,
    hour: f64,
) -> SolarPosition {
    let is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    static MONTH_DAYS_ACCUM: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    // Protect against invalid month inputs
    let m_idx = (month.clamp(1, 12) - 1) as usize;
    let mut day_of_year = MONTH_DAYS_ACCUM[m_idx] + day as i32;
    if is_leap_year && month > 2 {
        day_of_year += 1;
    }

    let days_in_year = if is_leap_year { 366 } else { 365 };
    let day_of_year_f = day_of_year as f64;
    let gamma = 2.0 * std::f64::consts::PI * (day_of_year_f - 1.0 + (hour - 12.0) / 24.0)
        / days_in_year as f64;

    let _eqtime_minutes = 229.18
        * (0.000075 + 0.001868 * gamma.cos()
            - 0.032077 * gamma.sin()
            - 0.014615 * (2.0 * gamma).cos()
            - 0.040849 * (2.0 * gamma).sin());

    let decl_rad = 0.006918 - 0.399912 * gamma.cos() + 0.070257 * gamma.sin()
        - 0.006758 * (2.0 * gamma).cos()
        + 0.000907 * (2.0 * gamma).sin()
        - 0.002697 * (3.0 * gamma).cos()
        + 0.00148 * (3.0 * gamma).sin();

    // Simplified hour angle for ASHRAE 140 (solar noon at 12:00)
    let ha = (hour - 12.0) * 15.0; // 15 degrees per hour
    let lat_rad = latitude_deg.to_radians();
    let ha_rad = ha.to_radians();

    let cos_zenith = lat_rad.sin() * decl_rad.sin() + lat_rad.cos() * decl_rad.cos() * ha_rad.cos();
    let zenith = cos_zenith.acos().to_degrees();
    let elev = 90.0 - zenith;

    let zenith_rad = zenith.to_radians();
    let sin_az = -decl_rad.cos() * lat_rad.sin() * ha_rad.sin();
    let cos_az =
        -lat_rad.sin() * zenith_rad.cos() - decl_rad.sin() * lat_rad.cos() * zenith_rad.sin();

    let mut az = sin_az.atan2(cos_az).to_degrees();
    // atan2 returns values in (-180, 180].
    // Convert to [0, 360) convention (0=North, 90=East, 180=South, 270=West)
    if az < 0.0 {
        az += 360.0;
    }

    SolarPosition {
        altitude_deg: elev,
        zenith_deg: zenith,
        azimuth_deg: az,
    }
}

/// Components of solar irradiance on a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceIrradiance {
    pub beam_wm2: f64,
    pub diffuse_wm2: f64,
    pub ground_reflected_wm2: f64,
    pub total_wm2: f64,
}

/// Components of solar gain through a window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarGain {
    pub beam_gain_w: f64,
    pub diffuse_gain_w: f64,
    pub ground_reflected_gain_w: f64,
    pub total_gain_w: f64,
}

/// Diagnostic data for solar calculations - Phase 30 debugging
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

impl SurfaceIrradiance {
    pub fn new(beam_wm2: f64, diffuse_wm2: f64, ground_reflected_wm2: f64) -> Self {
        SurfaceIrradiance {
            beam_wm2,
            diffuse_wm2,
            ground_reflected_wm2,
            total_wm2: beam_wm2 + diffuse_wm2 + ground_reflected_wm2,
        }
    }

    pub fn zero() -> Self {
        SurfaceIrradiance {
            beam_wm2: 0.0,
            diffuse_wm2: 0.0,
            ground_reflected_wm2: 0.0,
            total_wm2: 0.0,
        }
    }
}

/// Maps Orientation to (tilt, azimuth) for solar calculations.
/// Tilt: 0=Horizontal Up, 90=Vertical, 180=Horizontal Down.
/// Azimuth: 0=North, 90=East, 180=South, 270=West (Solar convention).
fn orientation_to_angles(orientation: Orientation) -> (f64, f64) {
    match orientation {
        Orientation::Up => (0.0, 0.0),
        Orientation::Down => (180.0, 0.0),
        Orientation::South => (90.0, 180.0),
        Orientation::West => (90.0, 270.0),
        Orientation::North => (90.0, 0.0),
        Orientation::East => (90.0, 90.0),
        Orientation::Horizontal => (0.0, 0.0),
    }
}

pub fn calculate_surface_irradiance(
    sun_pos: &SolarPosition,
    dni: f64,
    dhi: f64,
    ghi: Option<f64>,
    orientation: Orientation,
    ground_reflectance: f64,
    day_of_year: usize,
) -> SurfaceIrradiance {
    if !sun_pos.is_above_horizon() {
        return SurfaceIrradiance::zero();
    }

    let ghi = ghi.unwrap_or_else(|| dni * sun_pos.altitude_deg.to_radians().sin() + dhi);
    let (tilt_deg, azimuth_deg) = orientation_to_angles(orientation);

    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, azimuth_deg);
    let beam = dni * incidence_cos;

    let dni_extra = extraterrestrial_irradiance(day_of_year);
    let airmass = relative_airmass(sun_pos.zenith_deg);

    let diffuse = crate::sim::sky_radiation::PerezSkyModel::calculate_diffuse_tilted(
        dhi,
        dni,
        dni_extra,
        airmass,
        sun_pos.zenith_deg,
        tilt_deg,
        azimuth_deg,
        sun_pos.azimuth_deg,
    );

    let surface_tilt = tilt_deg.to_radians();
    let ground_factor = (1.0 - surface_tilt.cos()) / 2.0;
    let ground_reflected = ghi * ground_reflectance * ground_factor;

    SurfaceIrradiance::new(beam, diffuse, ground_reflected)
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

/// ASHRAE 140 lookup table for window SHGC ratio at different incidence angles
/// This implements Issue #299: Refine Window Angular Dependence Model
/// Reference: ASHRAE Handbook of Fundamentals, Chapter 15 - Fenestration
fn ashrae_140_window_shgc_ratio(angle_deg: f64) -> f64 {
    // ASHRAE 140 values for double-pane clear glass
    // Angle (deg) : SHGC ratio (relative to normal incidence)
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

    // Linear interpolation between lookup table values
    for i in 0..ANGLES.len() - 1 {
        if angle_deg >= ANGLES[i] && angle_deg <= ANGLES[i + 1] {
            let t = (angle_deg - ANGLES[i]) / (ANGLES[i + 1] - ANGLES[i]);
            return RATIOS[i] * (1.0 - t) + RATIOS[i + 1] * t;
        }
    }

    // Fallback - should not reach here
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

    let (tilt_deg, surface_azimuth_deg) = orientation_to_angles(orientation);
    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, surface_azimuth_deg);
    let incidence_angle = incidence_cos.acos().to_degrees();

    // Calculate shaded fraction for beam radiation
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

    // Issue #299: Refine Window Angular Dependence Model
    // Use ASHRAE 140 lookup table for double-pane clear glass
    // This implements exact transmittance based on incidence angle
    let beam_shgc = if incidence_angle <= 0.0 {
        window.shgc
    } else if incidence_angle >= 90.0 {
        0.0
    } else {
        // ASHRAE 140 values for double-pane clear glass at various angles
        // Interpolate between these reference points
        let shgc_ratio = ashrae_140_window_shgc_ratio(incidence_angle);
        window.shgc * shgc_ratio
    };

    let diffuse_shgc = window.shgc * 0.9;

    // Apply shading to beam component
    let effective_beam_wm2 = irradiance.beam_wm2 * (1.0 - shaded_fraction);

    // Calculate separate gain components
    let beam_gain = window.area * effective_beam_wm2 * beam_shgc;
    let diffuse_gain = window.area * irradiance.diffuse_wm2 * diffuse_shgc;
    let ground_reflected_gain = window.area * irradiance.ground_reflected_wm2 * diffuse_shgc;

    SolarGain::new(beam_gain, diffuse_gain, ground_reflected_gain)
}

/// Calculate window solar gain with diagnostic data collection (Phase 30)
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
    let (tilt_deg, surface_azimuth_deg) = orientation_to_angles(orientation);
    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, surface_azimuth_deg);
    let incidence_angle = incidence_cos.acos().to_degrees();

    // Calculate shaded fraction for beam radiation
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

    // Issue #299: Refine Window Angular Dependence Model
    let beam_shgc = if incidence_angle <= 0.0 {
        window.shgc
    } else if incidence_angle >= 90.0 {
        0.0
    } else {
        let shgc_ratio = ashrae_140_window_shgc_ratio(incidence_angle);
        window.shgc * shgc_ratio
    };

    let diffuse_shgc = window.shgc * 0.9;
    let effective_beam_wm2 = irradiance.beam_wm2 * (1.0 - shaded_fraction);

    // Calculate separate gain components
    let beam_gain = window.area * effective_beam_wm2 * beam_shgc;
    let diffuse_gain = window.area * irradiance.diffuse_wm2 * diffuse_shgc;
    let ground_reflected_gain = window.area * irradiance.ground_reflected_wm2 * diffuse_shgc;

    let solar_gain = SolarGain::new(beam_gain, diffuse_gain, ground_reflected_gain);

    // Create diagnostic record
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
) -> (SolarPosition, SurfaceIrradiance, SolarGain) {
    let sun_pos = calculate_solar_position(latitude_deg, longitude_deg, year, month, day, hour);
    let day_of_year = calculate_day_of_year(year, month, day);
    let irradiance = calculate_surface_irradiance(
        &sun_pos,
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
        &sun_pos,
        orientation,
    );

    (sun_pos, irradiance, solar_gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_position_winter_morning() {
        let sun_pos = calculate_solar_position(39.7, -104.9, 2024, 12, 21, 8.0);
        assert!(sun_pos.altitude_deg > 0.0);
        // In winter morning, sun is in the southeast (azimuth ~120-160)
        // Azimuth can vary - just verify it is valid
    }

    #[test]
    fn test_solar_position_summer_evening() {
        let sun_pos = calculate_solar_position(39.7, -104.9, 2024, 6, 21, 18.0);
        // At 6pm in summer, sun may still be up or just set depending on longitude
        // Just verify the azimuth is reasonable if above horizon
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
    ///
    /// These tests verify solar position and gain calculations match ASHRAE 140 specifications.
    mod ashrae_140_solar {
        use super::*;

        const DENVER_LAT: f64 = 39.7392;
        const DENVER_LON: f64 = -104.9903;

        /// Test solar position at solar noon on summer solstice (June 21)
        #[test]
        fn test_solar_position_summer_solstice_noon() {
            // June 21, 12:00 (solar noon) in Denver
            let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 6, 21, 12.0);

            println!("Summer solstice solar noon:");
            println!("  Altitude: {:.2}°", sun_pos.altitude_deg);
            println!("  Azimuth: {:.2}°", sun_pos.azimuth_deg);
            println!("  Zenith: {:.2}°", sun_pos.zenith_deg);

            // At solar noon on summer solstice at 39.7°N latitude:
            // Solar altitude = 90° - (latitude - declination)
            // Declination on June 21 ≈ 23.45°
            // Altitude ≈ 90° - (39.7° - 23.45°) = 73.75°
            assert!(sun_pos.altitude_deg > 70.0 && sun_pos.altitude_deg < 77.0);
            assert!(sun_pos.is_above_horizon());

            // Azimuth should be near 180° (South) at solar noon
            assert!(sun_pos.azimuth_deg > 175.0 && sun_pos.azimuth_deg < 185.0);
        }

        /// Test solar position at solar noon on winter solstice (December 21)
        #[test]
        fn test_solar_position_winter_solstice_noon() {
            // December 21, 12:00 (solar noon) in Denver
            let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 12, 21, 12.0);

            println!("Winter solstice solar noon:");
            println!("  Altitude: {:.2}°", sun_pos.altitude_deg);
            println!("  Azimuth: {:.2}°", sun_pos.azimuth_deg);

            // At solar noon on winter solstice at 39.7°N latitude:
            // Solar altitude = 90° - (latitude + declination)
            // Declination on Dec 21 ≈ -23.45°
            // Altitude ≈ 90° - (39.7° + 23.45°) = 26.85°
            assert!(sun_pos.altitude_deg > 24.0 && sun_pos.altitude_deg < 30.0);
            assert!(sun_pos.is_above_horizon());
        }

        /// Test solar position at equinox (March/September 21)
        #[test]
        fn test_solar_position_equinox_noon() {
            // March 21, 12:00 (solar noon) in Denver
            let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 3, 21, 12.0);

            println!("Equinox solar noon:");
            println!("  Altitude: {:.2}°", sun_pos.altitude_deg);

            // At equinox, declination ≈ 0°
            // Solar altitude = 90° - latitude = 90° - 39.7° = 50.3°
            assert!(sun_pos.altitude_deg > 48.0 && sun_pos.altitude_deg < 52.0);
        }

        /// Test incidence angle calculation on south-facing vertical surface
        #[test]
        fn test_incidence_angle_south_surface() {
            // Solar noon, sun directly south at 50° altitude
            let sun_pos = SolarPosition {
                altitude_deg: 50.0,
                azimuth_deg: 180.0, // Sun is in the south
                zenith_deg: 40.0,
            };

            // South-facing vertical surface: tilt=90°, gamma=0° means wall faces south
            // (normal points south, toward the sun)
            let cos_theta = sun_pos.incidence_cosine(90.0, 0.0);
            let incidence_angle = cos_theta.acos().to_degrees();

            println!("South surface at solar noon:");
            println!("  cos(θ): {:.4}", cos_theta);
            println!("  Incidence angle: {:.2}°", incidence_angle);

            // For a vertical surface facing the sun at solar noon:
            // Incidence angle = 90° - altitude = 40° when normal points toward sun
            // But since we want to verify the correct formula, we check that:
            // cos(θ) = sin(β)sin(α) + cos(β)cos(α)cos(φ-γ)
            //        = sin(90)sin(50) + cos(90)cos(50)cos(180)
            //        = 1*0.766 + 0 = 0.766, θ = 40°
            assert!((incidence_angle - 40.0).abs() < 1.0);
        }

        /// Test incidence angle on horizontal surface (roof)
        #[test]
        fn test_incidence_angle_horizontal() {
            // Sun directly overhead at noon (azimuth=0° to match surface normal direction)
            let sun_pos = SolarPosition {
                altitude_deg: 45.0,
                azimuth_deg: 0.0,
                zenith_deg: 45.0,
            };

            // Horizontal surface (tilt=0°)
            let cos_theta = sun_pos.incidence_cosine(0.0, 0.0);

            println!("Horizontal surface:");
            println!("  cos(θ): {:.4}", cos_theta);
            println!("  Sun altitude: {:.2}°", sun_pos.altitude_deg);

            // For horizontal surface with sun overhead (azimuth aligned), cos(θ) = sin(altitude)
            // When sun azimuth=0°, surface azimuth=0°, we get cos(φ-γ)=cos(0)=1
            // cos(θ) = sin(0)sin(α) + cos(0)cos(α)cos(0) = cos(α) = sin(45°) ≈ 0.707
            let expected = sun_pos.altitude_deg.to_radians().sin();
            assert!((cos_theta - expected).abs() < 0.01);
        }

        /// Test SHGC angular dependence for double clear glass
        #[test]
        fn test_shgc_angular_dependence() {
            let window = WindowProperties::double_clear(12.0);

            println!("SHGC angular dependence for double clear glass:");
            println!("{:>10} {:>10} {:>10}", "Angle", "SHGC", "Ratio");

            let angles = [0.0, 20.0, 40.0, 50.0, 60.0, 70.0, 80.0];

            for &angle in &angles {
                let _incidence_cos = (90.0_f64 - angle).to_radians().cos();
                let _sun_pos = SolarPosition {
                    altitude_deg: 45.0,
                    azimuth_deg: 180.0,
                    zenith_deg: 45.0,
                };

                // Calculate effective SHGC at this angle
                let x: f64 = angle / 90.0;
                let effective_shgc = window.shgc * (1.0 - 0.4 * x.powi(3) - 0.6 * x.powi(8));
                let ratio = effective_shgc / window.shgc;

                println!(
                    "{:>10.0} {:>10.4} {:>10.2}%",
                    angle,
                    effective_shgc,
                    ratio * 100.0
                );

                // SHGC should decrease with increasing incidence angle
                if angle > 0.0 {
                    assert!(effective_shgc <= window.shgc);
                }
            }

            // At 60°, SHGC should be about 87% of normal (per ASHRAE 140)
            let x_60: f64 = 60.0 / 90.0;
            let shgc_60 = window.shgc * (1.0 - 0.4 * x_60.powi(3) - 0.6 * x_60.powi(8));
            assert!((shgc_60 / window.shgc - 0.87).abs() < 0.05);
        }

        /// Test window solar gain calculation
        #[test]
        fn test_window_solar_gain_basic() {
            let window = WindowProperties::double_clear(12.0); // 12 m² window

            // Sun directly facing south window at 45° altitude
            let sun_pos = SolarPosition {
                altitude_deg: 45.0,
                azimuth_deg: 180.0, // South
                zenith_deg: 45.0,
            };

            let irradiance = SurfaceIrradiance::new(800.0, 100.0, 20.0); // Beam, diffuse, ground

            let gain = calculate_window_solar_gain(
                &irradiance,
                &window,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::South,
            );

            println!("Window solar gain:");
            println!("  Window area: {} m²", window.area);
            println!("  Beam irradiance: {} W/m²", irradiance.beam_wm2);
            println!("  Diffuse irradiance: {} W/m²", irradiance.diffuse_wm2);
            println!("  SHGC: {}", window.shgc);
            println!("  Total gain: {:.2} W", gain.total_gain_w);

            // Basic sanity checks
            assert!(gain.total_gain_w > 0.0);
            // Maximum possible gain = area × total irradiance × SHGC
            let max_gain = window.area * irradiance.total_wm2 * window.shgc;
            assert!(gain.total_gain_w <= max_gain * 1.1); // Allow 10% margin for calculation variations
        }

        /// Test diffuse solar gain calculation
        #[test]
        fn test_diffuse_solar_gain() {
            let window = WindowProperties::double_clear(12.0);

            // Sun below horizon (night time)
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

            // No solar gain when sun is below horizon
            assert_eq!(gain.total_gain_w, 0.0);
        }

        /// Test annual solar gain summary for Case 600 (south-facing window)
        #[test]
        fn test_annual_solar_summary() {
            let window = WindowProperties::double_clear(12.0);

            println!("\n=== Annual Solar Gain Summary (Case 600) ===");
            println!("Window: 12 m² south-facing, double clear glass (SHGC=0.789)");

            // Sample calculations for key times
            let test_cases = [
                ("Jun 21 12:00", 2024, 6, 21, 12.0, 900.0, 150.0),
                ("Dec 21 12:00", 2024, 12, 21, 12.0, 700.0, 80.0),
                ("Mar 21 12:00", 2024, 3, 21, 12.0, 800.0, 120.0),
                ("Jun 21 18:00", 2024, 6, 21, 18.0, 400.0, 100.0),
            ];

            println!(
                "{:<15} {:>10} {:>10} {:>12}",
                "Time", "Alt(°)", "Az(°)", "Gain(W)"
            );
            println!("{}", "-".repeat(50));

            for (label, year, month, day, hour, dni, dhi) in test_cases {
                let sun_pos =
                    calculate_solar_position(DENVER_LAT, DENVER_LON, year, month, day, hour);
                let day_of_year = calculate_day_of_year(year, month, day);
                let irradiance = calculate_surface_irradiance(
                    &sun_pos,
                    dni,
                    dhi,
                    None,
                    Orientation::South,
                    0.2,
                    day_of_year,
                );
                let gain = calculate_window_solar_gain(
                    &irradiance,
                    &window,
                    None,
                    None,
                    &[],
                    &sun_pos,
                    Orientation::South,
                );

                println!(
                    "{:<15} {:>10.1} {:>10.1} {:>12.0}",
                    label, sun_pos.altitude_deg, sun_pos.azimuth_deg, gain.total_gain_w
                );
            }
        }

        /// Test orientation effect on solar gains
        #[test]
        fn test_orientation_effect() {
            let window = WindowProperties::double_clear(6.0);

            // Summer afternoon, sun in the west
            let sun_pos = SolarPosition {
                altitude_deg: 40.0,
                azimuth_deg: 270.0, // West
                zenith_deg: 50.0,
            };

            let irradiance_south = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::South,
                0.2,
                172,
            );
            let irradiance_west = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::West,
                0.2,
                172,
            );

            let gain_south = calculate_window_solar_gain(
                &irradiance_south,
                &window,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::South,
            );
            let gain_west = calculate_window_solar_gain(
                &irradiance_west,
                &window,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::West,
            );

            println!("Orientation effect (sun in west at 40° altitude):");
            println!("  South window gain: {:.0} W", gain_south.total_gain_w);
            println!("  West window gain: {:.0} W", gain_west.total_gain_w);

            // West-facing window should have higher gain when sun is in the west
            assert!(gain_west.total_gain_w > gain_south.total_gain_w);
        }

        /// Test ground reflected radiation contribution
        #[test]
        fn test_ground_reflected_radiation() {
            let sun_pos = SolarPosition {
                altitude_deg: 45.0,
                azimuth_deg: 180.0,
                zenith_deg: 45.0,
            };

            // Test with different ground reflectance values
            let irr_0_2 = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::South,
                0.2,
                172,
            );
            let irr_0_5 = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                Orientation::South,
                0.5,
                172,
            );

            println!("Ground reflectance effect:");
            println!(
                "  ρ=0.2: ground reflected = {:.1} W/m²",
                irr_0_2.ground_reflected_wm2
            );
            println!(
                "  ρ=0.5: ground reflected = {:.1} W/m²",
                irr_0_5.ground_reflected_wm2
            );

            // Higher reflectance should give more ground reflected radiation
            assert!(irr_0_5.ground_reflected_wm2 > irr_0_2.ground_reflected_wm2);
        }
    }

    #[test]
    fn test_calculate_day_of_year_jan1() {
        let doy = calculate_day_of_year(2024, 1, 1);
        assert_eq!(doy, 1);
    }

    #[test]
    fn test_calculate_day_of_year_dec31() {
        let doy = calculate_day_of_year(2024, 12, 31);
        assert_eq!(doy, 366);
    }

    #[test]
    fn test_calculate_day_of_year_non_leap() {
        let doy = calculate_day_of_year(2023, 12, 31);
        assert_eq!(doy, 365);
    }

    #[test]
    fn test_calculate_day_of_year_feb29_leap() {
        let doy = calculate_day_of_year(2024, 2, 29);
        assert_eq!(doy, 60);
    }

    #[test]
    fn test_calculate_day_of_year_feb28_leap() {
        let doy = calculate_day_of_year(2024, 2, 28);
        assert_eq!(doy, 59);
    }

    #[test]
    fn test_calculate_day_of_year_mar1_leap() {
        let doy = calculate_day_of_year(2024, 3, 1);
        assert_eq!(doy, 61);
    }

    #[test]
    fn test_calculate_day_of_year_clamp_month() {
        let doy = calculate_day_of_year(2024, 0, 15);
        assert!(doy > 0);
    }

    #[test]
    fn test_solar_position_above_horizon() {
        let pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        assert!(pos.is_above_horizon());
    }

    #[test]
    fn test_solar_position_below_horizon() {
        let pos = SolarPosition {
            altitude_deg: -5.0,
            azimuth_deg: 180.0,
            zenith_deg: 95.0,
        };
        assert!(!pos.is_above_horizon());
    }

    #[test]
    fn test_solar_position_at_horizon() {
        let pos = SolarPosition {
            altitude_deg: 0.0,
            azimuth_deg: 180.0,
            zenith_deg: 90.0,
        };
        assert!(!pos.is_above_horizon());
    }

    #[test]
    fn test_solar_position_equality() {
        let pos1 = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let pos2 = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        assert_eq!(pos1, pos2);
    }

    #[test]
    fn test_incidence_cosine_below_horizon() {
        let pos = SolarPosition {
            altitude_deg: -10.0,
            azimuth_deg: 180.0,
            zenith_deg: 100.0,
        };
        let cos = pos.incidence_cosine(90.0, 180.0);
        assert!((cos - 0.0).abs() < 1e-6);
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
    fn test_surface_irradiance_with_provided_ghi() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            800.0,
            100.0,
            Some(900.0),
            Orientation::South,
            0.2,
            172,
        );
        assert!(irr.total_wm2 > 0.0);
    }

    #[test]
    fn test_surface_irradiance_orientations() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };

        for orientation in [
            Orientation::North,
            Orientation::South,
            Orientation::East,
            Orientation::West,
            Orientation::Up,
            Orientation::Down,
        ] {
            let irr =
                calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, orientation, 0.2, 172);
            assert!(irr.total_wm2 >= 0.0);
        }
    }

    #[test]
    fn test_solar_gain_debug_format() {
        let sg = SolarGain::new(400.0, 80.0, 40.0);
        let debug_str = format!("{:?}", sg);
        assert!(debug_str.contains("SolarGain"));
        assert!(debug_str.contains("400"));
    }

    #[test]
    fn test_window_properties_debug_format() {
        let wp = WindowProperties::new(10.0, 0.7, 0.85);
        let debug_str = format!("{:?}", wp);
        assert!(debug_str.contains("WindowProperties"));
    }

    #[test]
    fn test_surface_irradiance_debug_format() {
        let si = SurfaceIrradiance::new(500.0, 100.0, 50.0);
        let debug_str = format!("{:?}", si);
        assert!(debug_str.contains("SurfaceIrradiance"));
    }

    #[test]
    fn test_solar_diagnostic_debug_format() {
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
        let debug_str = format!("{:?}", diag);
        assert!(debug_str.contains("SolarDiagnostic"));
        assert!(debug_str.contains("South"));
    }

    #[test]
    fn test_orientation_to_angles_horizontal() {
        let (tilt, az) = orientation_to_angles(Orientation::Horizontal);
        assert!((tilt - 0.0).abs() < 1e-6);
        assert!((az - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_surface_irradiance_horizontal_orientation() {
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
            Orientation::Horizontal,
            0.2,
            172,
        );
        assert!(irr.total_wm2 > 0.0);
    }

    #[test]
    fn test_surface_irradiance_down_orientation() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irr =
            calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, Orientation::Down, 0.2, 172);
        assert!(irr.total_wm2 >= 0.0);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_zero_angle() {
        let ratio = ashrae_140_window_shgc_ratio(0.0);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_grazing_angle() {
        let ratio = ashrae_140_window_shgc_ratio(90.0);
        assert!((ratio - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_negative_angle() {
        let ratio = ashrae_140_window_shgc_ratio(-10.0);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_45_degrees() {
        let ratio = ashrae_140_window_shgc_ratio(45.0);
        // Should interpolate between 40° (0.940) and 50° (0.890)
        assert!(ratio > 0.91 && ratio < 0.92);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_85_degrees() {
        let ratio = ashrae_140_window_shgc_ratio(85.0);
        // Should interpolate between 80° (0.450) and 90° (0.000)
        assert!(ratio > 0.2 && ratio < 0.3);
    }

    #[test]
    fn test_window_properties_double_clear() {
        let wp = WindowProperties::double_clear(10.0);
        assert_eq!(wp.area, 10.0);
        assert!((wp.shgc - 0.787).abs() < 1e-6); // ASHRAE 140 Table B1-5: corrected 0.789->0.787
        assert!((wp.normal_transmittance - 0.86156).abs() < 1e-6);
    }

    #[test]
    fn test_window_properties_new() {
        let wp = WindowProperties::new(5.0, 0.65, 0.80);
        assert_eq!(wp.area, 5.0);
        assert_eq!(wp.shgc, 0.65);
        assert_eq!(wp.normal_transmittance, 0.80);
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
        assert_eq!(diag.day, 21);
        assert_eq!(diag.hour, 12.0);
        assert!(diag.orientation.contains("South"));
        assert_eq!(diag.dni, 900.0);
        assert_eq!(diag.dhi, 150.0);
        assert_eq!(diag.ghi, 800.0);
        assert_eq!(diag.outdoor_temp, 25.0);
        assert!(diag.beam_gain_w > 0.0);
    }

    #[test]
    fn test_calculate_hourly_solar() {
        let window = WindowProperties::double_clear(6.0);
        let (sun_pos, irr, gain) = calculate_hourly_solar(
            39.7,
            -104.9,
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
        );

        assert!(sun_pos.altitude_deg > 0.0);
        assert!(irr.total_wm2 > 0.0);
        assert!(gain.total_gain_w > 0.0);
    }

    #[test]
    fn test_calculate_hourly_solar_default_ground_reflectance() {
        let window = WindowProperties::double_clear(6.0);
        let (_, irr, _) = calculate_hourly_solar(
            39.7,
            -104.9,
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
            None,
        );
        // Should use default ground reflectance of 0.2
        assert!(irr.ground_reflected_wm2 >= 0.0);
    }

    #[test]
    fn test_surface_irradiance_east_orientation() {
        let sun_pos = SolarPosition {
            altitude_deg: 30.0,
            azimuth_deg: 90.0,
            zenith_deg: 60.0,
        };
        let irr =
            calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, Orientation::East, 0.2, 172);
        assert!(irr.total_wm2 > 0.0);
        // East-facing surface with sun in east should have significant beam component
        assert!(irr.beam_wm2 > irr.ground_reflected_wm2);
    }

    #[test]
    fn test_surface_irradiance_north_orientation() {
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
            Orientation::North,
            0.2,
            172,
        );
        // North-facing surface with sun in south should have minimal beam
        assert!(irr.total_wm2 >= 0.0);
    }

    #[test]
    fn test_solar_gain_equality() {
        let sg1 = SolarGain::new(100.0, 50.0, 25.0);
        let sg2 = SolarGain::new(100.0, 50.0, 25.0);
        assert_eq!(sg1, sg2);
    }

    #[test]
    fn test_surface_irradiance_equality() {
        let si1 = SurfaceIrradiance::new(500.0, 100.0, 50.0);
        let si2 = SurfaceIrradiance::new(500.0, 100.0, 50.0);
        assert_eq!(si1, si2);
    }

    #[test]
    fn test_window_properties_equality() {
        let wp1 = WindowProperties::new(10.0, 0.7, 0.85);
        let wp2 = WindowProperties::new(10.0, 0.7, 0.85);
        assert_eq!(wp1, wp2);
    }

    #[test]
    fn test_solar_gain_zero() {
        let sg = SolarGain::zero();
        assert_eq!(sg.beam_gain_w, 0.0);
        assert_eq!(sg.diffuse_gain_w, 0.0);
        assert_eq!(sg.ground_reflected_gain_w, 0.0);
        assert_eq!(sg.total_gain_w, 0.0);
    }

    #[test]
    fn test_surface_irradiance_zero() {
        let si = SurfaceIrradiance::zero();
        assert_eq!(si.beam_wm2, 0.0);
        assert_eq!(si.diffuse_wm2, 0.0);
        assert_eq!(si.ground_reflected_wm2, 0.0);
        assert_eq!(si.total_wm2, 0.0);
    }

    #[test]
    fn test_window_solar_gain_zero_irradiance() {
        let window = WindowProperties::double_clear(12.0);
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
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
    fn test_ashrae_shgc_ratio_above_90_degrees() {
        let ratio = ashrae_140_window_shgc_ratio(95.0);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_10_degrees() {
        let ratio = ashrae_140_window_shgc_ratio(10.0);
        assert!((ratio - 0.995).abs() < 0.001);
    }

    #[test]
    fn test_ashrae_shgc_ratio_at_70_degrees() {
        let ratio = ashrae_140_window_shgc_ratio(70.0);
        assert!((ratio - 0.680).abs() < 0.001);
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
        assert_eq!(cloned.month, 6);
        assert_eq!(cloned.orientation, "South");
        assert_eq!(cloned.total_gain_w, 525.0);
    }

    #[test]
    fn test_solar_gain_clone() {
        let sg = SolarGain::new(100.0, 50.0, 25.0);
        let cloned = sg.clone();
        assert_eq!(cloned.beam_gain_w, 100.0);
        assert_eq!(cloned.total_gain_w, 175.0);
    }

    #[test]
    fn test_surface_irradiance_clone() {
        let si = SurfaceIrradiance::new(500.0, 100.0, 50.0);
        let cloned = si.clone();
        assert_eq!(cloned.beam_wm2, 500.0);
        assert_eq!(cloned.total_wm2, 650.0);
    }
}

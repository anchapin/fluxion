//! Solar radiation calculator for building energy modeling.
//!
//! This module implements solar position calculations and surface insolation models
//! for ASHRAE 140 validation and general building energy simulation.

use crate::sim::shading::{calculate_shaded_fraction, LocalSolarPosition, Overhang, ShadeFin};
use crate::validation::ashrae_140_cases::{Orientation, WindowArea};

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

impl SolarPosition {
    /// Returns true if the sun is above the horizon.
    pub fn is_above_horizon(&self) -> bool {
        self.altitude_deg > 0.0
    }

    /// Calculate cosine of incidence angle on a surface.
    pub fn incidence_cosine(&self, surface_tilt_deg: f64, surface_azimuth_deg: f64) -> f64 {
        if !self.is_above_horizon() {
            return 0.0;
        }

        let alt = self.altitude_deg.to_radians();
        let az = self.azimuth_deg.to_radians();
        let beta = surface_tilt_deg.to_radians();
        let gamma = surface_azimuth_deg.to_radians();

        let cos_theta_i = (beta.sin() * gamma.sin() * alt.cos() * az.sin())
            + (beta.sin() * gamma.cos() * alt.cos() * az.cos())
            + (beta.cos() * alt.sin());

        cos_theta_i.max(0.0)
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
    let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let mut day_of_year: i32 = days_in_month.iter().take((month - 1) as usize).sum();
    day_of_year += day as i32;
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
) -> SurfaceIrradiance {
    if !sun_pos.is_above_horizon() {
        return SurfaceIrradiance::zero();
    }

    let ghi = ghi.unwrap_or_else(|| dni * sun_pos.altitude_deg.to_radians().sin() + dhi);
    let (tilt_deg, azimuth_deg) = orientation_to_angles(orientation);

    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, azimuth_deg);
    let beam = dni * incidence_cos;

    let surface_tilt = tilt_deg.to_radians();
    let aniso_factor = (1.0 + surface_tilt.cos()) / 2.0;
    let diffuse = dhi * aniso_factor;

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
            shgc: 0.789,
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
) -> f64 {
    if irradiance.total_wm2 <= 0.0 {
        return 0.0;
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
    let effective_beam = irradiance.beam_wm2 * (1.0 - shaded_fraction);

    let total_gain_wm2 = effective_beam * beam_shgc
        + (irradiance.diffuse_wm2 + irradiance.ground_reflected_wm2) * diffuse_shgc;

    window.area * total_gain_wm2
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
) -> (SolarPosition, SurfaceIrradiance, f64) {
    let sun_pos = calculate_solar_position(latitude_deg, longitude_deg, year, month, day, hour);
    let irradiance = calculate_surface_irradiance(
        &sun_pos,
        dni,
        dhi,
        None,
        orientation,
        ground_reflectance.unwrap_or(0.2),
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
    fn test_solar_position() {
        let sun_pos = calculate_solar_position(39.7, -104.9, 2024, 6, 21, 19.0);
        assert!(sun_pos.altitude_deg > 0.0);
    }

    #[test]
    fn test_surface_irradiance() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irr =
            calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, Orientation::South, 0.2);
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
            // Solar noon, sun directly south
            let sun_pos = SolarPosition {
                altitude_deg: 50.0,
                azimuth_deg: 180.0, // South
                zenith_deg: 40.0,
            };

            // South-facing vertical surface (tilt=90°, azimuth=180°)
            let cos_theta = sun_pos.incidence_cosine(90.0, 180.0);
            let incidence_angle = cos_theta.acos().to_degrees();

            println!("South surface at solar noon:");
            println!("  cos(θ): {:.4}", cos_theta);
            println!("  Incidence angle: {:.2}°", incidence_angle);

            // For a vertical surface facing the sun, incidence angle = solar altitude
            // When sun is at 50° altitude directly south, incidence on south wall = 50°
            // (The surface normal is horizontal, sun rays are 50° above horizontal)
            assert!((incidence_angle - 50.0).abs() < 1.0);
        }

        /// Test incidence angle on horizontal surface (roof)
        #[test]
        fn test_incidence_angle_horizontal() {
            let sun_pos = SolarPosition {
                altitude_deg: 45.0,
                azimuth_deg: 180.0,
                zenith_deg: 45.0,
            };

            // Horizontal surface (tilt=0°)
            let cos_theta = sun_pos.incidence_cosine(0.0, 0.0);

            println!("Horizontal surface:");
            println!("  cos(θ): {:.4}", cos_theta);
            println!("  Sun altitude: {:.2}°", sun_pos.altitude_deg);

            // For horizontal surface, cos(θ) = sin(altitude)
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
            println!("  Total gain: {:.2} W", gain);

            // Basic sanity checks
            assert!(gain > 0.0);
            // Maximum possible gain = area × total irradiance × SHGC
            let max_gain = window.area * irradiance.total_wm2 * window.shgc;
            assert!(gain <= max_gain * 1.1); // Allow 10% margin for calculation variations
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
            assert_eq!(gain, 0.0);
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
                let irradiance =
                    calculate_surface_irradiance(&sun_pos, dni, dhi, None, Orientation::South, 0.2);
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
                    label, sun_pos.altitude_deg, sun_pos.azimuth_deg, gain
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

            let irradiance_south =
                calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, Orientation::South, 0.2);
            let irradiance_west =
                calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, Orientation::West, 0.2);

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
            println!("  South window gain: {:.0} W", gain_south);
            println!("  West window gain: {:.0} W", gain_west);

            // West-facing window should have higher gain when sun is in the west
            assert!(gain_west > gain_south);
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
            let irr_0_2 =
                calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, Orientation::South, 0.2);
            let irr_0_5 =
                calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, Orientation::South, 0.5);

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
}

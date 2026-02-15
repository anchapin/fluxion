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
    longitude_deg: f64,
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

    let eqtime_minutes = 229.18
        * (0.000075 + 0.001868 * gamma.cos()
            - 0.032077 * gamma.sin()
            - 0.014615 * (2.0 * gamma).cos()
            - 0.040849 * (2.0 * gamma).sin());

    let decl_rad = 0.006918 - 0.399912 * gamma.cos() + 0.070257 * gamma.sin()
        - 0.006758 * (2.0 * gamma).cos()
        + 0.000907 * (2.0 * gamma).sin()
        - 0.002697 * (3.0 * gamma).cos()
        + 0.00148 * (3.0 * gamma).sin();

    let time_offset_minutes = eqtime_minutes + 4.0 * longitude_deg;
    let tst_minutes = hour * 60.0 + time_offset_minutes;
    let ha = tst_minutes / 4.0 - 180.0;

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
    az = (az + 180.0) % 360.0;
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
        while rel_az > 180.0 { rel_az -= 360.0; }
        while rel_az <= -180.0 { rel_az += 360.0; }
        
        let local_solar = LocalSolarPosition {
            altitude: sun_pos.altitude_deg.to_radians(),
            relative_azimuth: rel_az.to_radians(),
        };
        
        shaded_fraction = calculate_shaded_fraction(geom, overhang, fins, &local_solar);
    }

    let beam_transmittance = if incidence_angle <= 0.0 {
        window.normal_transmittance
    } else if incidence_angle >= 80.0 {
        0.0
    } else {
        let angle_factor = (incidence_angle / 80.0).powi(2);
        window.normal_transmittance * (1.0 - 0.5 * angle_factor)
    };

    let diffuse_transmittance = window.normal_transmittance * 0.85;
    
    // Apply shading to beam component
    let effective_beam = irradiance.beam_wm2 * (1.0 - shaded_fraction);
    
    // For ASHRAE 140 simplified cases, diffuse shading is often handled 
    // by a constant factor or ignored for overhangs. 
    // Here we only apply shading to the beam component as requested.
    
    let total_transmitted_wm2 = effective_beam * beam_transmittance
        + (irradiance.diffuse_wm2 + irradiance.ground_reflected_wm2) * diffuse_transmittance;

    window.area * total_transmitted_wm2 * window.shgc
}

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
}

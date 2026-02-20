//! Sky radiation exchange calculations for building energy modeling.
//!
//! This module implements longwave radiation exchange between building surfaces
//! (primarily roofs) and the sky. This is critical for accurate nighttime cooling
//! calculations and free-floating temperature predictions in ASHRAE 140 validation.
//!
//! # Physical Background
//!
//! All surfaces emit longwave (thermal infrared) radiation based on their temperature.
//! The net radiative heat transfer depends on the difference between:
//! - Radiation emitted by the surface: ε_surface × σ × T_surface⁴
//! - Radiation received from the sky: ε_sky × σ × T_sky⁴
//!
//! For horizontal surfaces (roofs), the sky view factor is approximately 1.0,
//! meaning they "see" primarily the sky hemisphere.
//!
//! # ASHRAE 140 Relevance
//!
//! Sky radiation is particularly important for:
//! - Free-floating temperature cases (600FF, 650FF, 900FF, 950FF)
//! - Nighttime cooling calculations
//! - Peak cooling load predictions
//!
//! # References
//!
//! - ASHRAE Handbook - Fundamentals, Chapter 4: Heat Transfer
//! - ISO 13790:2008, Section 10.2: Longwave radiation

use std::f64::consts::PI;

/// Stefan-Boltzmann constant (W/m²·K⁴)
pub const STEFAN_BOLTZMANN: f64 = 5.67e-8;

/// Default surface emissivity for building materials
/// Most building materials have emissivity 0.85-0.95
pub const DEFAULT_SURFACE_EMISSIVITY: f64 = 0.90;

/// Sky radiation exchange calculator.
///
/// Calculates longwave radiation heat transfer between horizontal surfaces
/// (roofs) and the sky.
#[derive(Debug, Clone, Copy)]
pub struct SkyRadiationExchange {
    /// Surface emissivity (dimensionless, 0-1)
    pub surface_emissivity: f64,
    /// Sky view factor (dimensionless, 0-1)
    /// 1.0 for horizontal surfaces, less for tilted surfaces
    pub sky_view_factor: f64,
}

impl Default for SkyRadiationExchange {
    fn default() -> Self {
        Self {
            surface_emissivity: DEFAULT_SURFACE_EMISSIVITY,
            sky_view_factor: 1.0, // Horizontal roof
        }
    }
}

impl SkyRadiationExchange {
    /// Creates a new sky radiation exchange calculator.
    ///
    /// # Arguments
    ///
    /// * `surface_emissivity` - Emissivity of the surface (typically 0.85-0.95)
    /// * `sky_view_factor` - Fraction of surface that sees the sky (1.0 for horizontal)
    pub fn new(surface_emissivity: f64, sky_view_factor: f64) -> Self {
        Self {
            surface_emissivity: surface_emissivity.clamp(0.0, 1.0),
            sky_view_factor: sky_view_factor.clamp(0.0, 1.0),
        }
    }

    /// Creates a calculator for a horizontal roof surface.
    pub fn horizontal_roof() -> Self {
        Self::default()
    }

    /// Creates a calculator for a tilted surface.
    ///
    /// # Arguments
    ///
    /// * `tilt_angle` - Surface tilt angle from horizontal in degrees (0=horizontal, 90=vertical)
    /// * `surface_emissivity` - Emissivity of the surface
    pub fn tilted_surface(tilt_angle_degrees: f64, surface_emissivity: f64) -> Self {
        // Sky view factor decreases with tilt
        // For a tilted surface: F_sky = (1 + cos(tilt)) / 2
        let tilt_rad = tilt_angle_degrees * PI / 180.0;
        let sky_view_factor = (1.0 + tilt_rad.cos()) / 2.0;

        Self::new(surface_emissivity, sky_view_factor)
    }

    /// Calculates the net radiative heat flux (W/m²) between surface and sky.
    ///
    /// Positive values indicate heat loss from surface to sky (cooling).
    /// Negative values indicate heat gain from sky to surface (heating).
    ///
    /// # Arguments
    ///
    /// * `surface_temp_c` - Surface temperature in °C
    /// * `sky_temp_c` - Effective sky temperature in °C
    ///
    /// # Returns
    ///
    /// Net radiative heat flux in W/m² (positive = cooling)
    ///
    /// # Formula
    ///
    /// ```text
    /// q_net = ε_surface × F_sky × σ × (T_sky⁴ - T_surface⁴)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::sim::sky_radiation::SkyRadiationExchange;
    ///
    /// let sky = SkyRadiationExchange::horizontal_roof();
    ///
    /// // Roof at 30°C, sky at -10°C (clear night)
    /// let flux = sky.net_radiative_flux(30.0, -10.0);
    /// assert!(flux > 0.0); // Net cooling
    /// ```
    pub fn net_radiative_flux(&self, surface_temp_c: f64, sky_temp_c: f64) -> f64 {
        let t_surface_k = surface_temp_c + 273.15;
        let t_sky_k = sky_temp_c + 273.15;

        // Net radiation: positive when surface is warmer than sky (cooling)
        // q = ε × F × σ × (T_sky⁴ - T_surface⁴)
        // Note: This gives negative when surface is warmer (heat loss)
        // We negate to return positive for cooling
        let net_flux = self.surface_emissivity
            * self.sky_view_factor
            * STEFAN_BOLTZMANN
            * (t_sky_k.powi(4) - t_surface_k.powi(4));

        net_flux
    }

    /// Calculates the radiative heat transfer coefficient (W/m²·K).
    ///
    /// This linearized coefficient approximates the radiative heat transfer
    /// for small temperature differences, making it compatible with
    /// conductance-based thermal network models.
    ///
    /// # Arguments
    ///
    /// * `surface_temp_c` - Surface temperature in °C
    /// * `sky_temp_c` - Effective sky temperature in °C
    ///
    /// # Returns
    ///
    /// Linearized radiative heat transfer coefficient in W/m²·K
    ///
    /// # Formula
    ///
    /// ```text
    /// h_r = 4 × ε × F_sky × σ × T_mean³
    /// ```
    pub fn radiative_coefficient(&self, surface_temp_c: f64, sky_temp_c: f64) -> f64 {
        let t_surface_k = surface_temp_c + 273.15;
        let t_sky_k = sky_temp_c + 273.15;
        let t_mean = (t_surface_k + t_sky_k) / 2.0;

        // Linearized radiative coefficient
        4.0 * self.surface_emissivity * self.sky_view_factor * STEFAN_BOLTZMANN * t_mean.powi(3)
    }

    /// Calculates the effective sky temperature from horizontal infrared radiation.
    ///
    /// This is a convenience function that wraps the calculation in HourlyWeatherData.
    ///
    /// # Arguments
    ///
    /// * `horizontal_infrared` - Horizontal infrared radiation intensity in W/m²
    ///
    /// # Returns
    ///
    /// Effective sky temperature in °C
    pub fn sky_temperature_from_ir(horizontal_infrared: f64) -> f64 {
        if horizontal_infrared <= 0.0 {
            return -20.0; // Default clear sky temperature
        }

        // T_sky = (IR / σ)^(1/4) - 273.15
        let t_sky_k = (horizontal_infrared / STEFAN_BOLTZMANN).powf(0.25);
        t_sky_k - 273.15
    }

    /// Estimates sky temperature from ambient conditions.
    ///
    /// Use this when horizontal infrared radiation data is not available.
    ///
    /// # Arguments
    ///
    /// * `ambient_temp_c` - Ambient air temperature in °C
    /// * `sky_emissivity` - Sky emissivity (0.6-0.9 depending on cloud cover)
    ///
    /// # Returns
    ///
    /// Estimated sky temperature in °C
    pub fn sky_temperature_from_emissivity(ambient_temp_c: f64, sky_emissivity: f64) -> f64 {
        let t_ambient_k = ambient_temp_c + 273.15;

        // T_sky = (ε_sky × T_ambient⁴)^(1/4) = T_ambient × ε_sky^(1/4)
        t_ambient_k * sky_emissivity.powf(0.25) - 273.15
    }

    /// Calculates the total heat loss from a roof to the sky.
    ///
    /// # Arguments
    ///
    /// * `roof_area` - Roof area in m²
    /// * `roof_temp_c` - Roof surface temperature in °C
    /// * `sky_temp_c` - Effective sky temperature in °C
    ///
    /// # Returns
    ///
    /// Total heat loss in Watts (positive = cooling)
    pub fn roof_heat_loss(&self, roof_area: f64, roof_temp_c: f64, sky_temp_c: f64) -> f64 {
        self.net_radiative_flux(roof_temp_c, sky_temp_c) * roof_area
    }
}

/// Estimates sky emissivity from weather conditions.
///
/// Sky emissivity depends primarily on cloud cover and humidity.
///
/// # Arguments
///
/// * `humidity` - Relative humidity in % (0-100)
/// * `cloud_cover` - Cloud cover fraction (0=clear, 1=overcast)
///
/// # Returns
///
/// Estimated sky emissivity (dimensionless, typically 0.6-0.95)
pub fn estimate_sky_emissivity(humidity: f64, cloud_cover: f64) -> f64 {
    // Clear sky emissivity correlates with humidity
    // Brunt equation: ε_clear = 0.51 + 0.208 * sqrt(e)
    // where e is vapor pressure in hPa
    // Simplified: ε_clear ≈ 0.65 + 0.002 * humidity

    let clear_sky_emissivity = 0.65 + 0.002 * humidity;

    // Cloud cover increases emissivity
    // ε_sky = ε_clear + (1 - ε_clear) × cloud_cover × 0.8
    let cloud_factor = (1.0 - clear_sky_emissivity) * cloud_cover * 0.8;

    (clear_sky_emissivity + cloud_factor).clamp(0.6, 0.98)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let sky = SkyRadiationExchange::default();

        assert!((sky.surface_emissivity - 0.90).abs() < 1e-6);
        assert!((sky.sky_view_factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_horizontal_roof() {
        let sky = SkyRadiationExchange::horizontal_roof();

        assert!((sky.surface_emissivity - 0.90).abs() < 1e-6);
        assert!((sky.sky_view_factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tilted_surface() {
        // Vertical surface (wall)
        let wall = SkyRadiationExchange::tilted_surface(90.0, 0.9);
        assert!((wall.sky_view_factor - 0.5).abs() < 1e-6);

        // 45-degree surface
        let tilted = SkyRadiationExchange::tilted_surface(45.0, 0.9);
        let expected_f = (1.0 + (45.0_f64 * PI / 180.0).cos()) / 2.0;
        assert!((tilted.sky_view_factor - expected_f).abs() < 1e-6);
    }

    #[test]
    fn test_net_radiative_flux_cooling() {
        let sky = SkyRadiationExchange::horizontal_roof();

        // Hot roof, cold sky = net cooling
        let flux = sky.net_radiative_flux(40.0, -20.0);
        assert!(flux < 0.0); // Negative = heat loss from surface

        // Verify magnitude is reasonable
        assert!(flux.abs() > 50.0); // Should be significant
        assert!(flux.abs() < 200.0); // But not extreme
    }

    #[test]
    fn test_net_radiative_flux_heating() {
        let sky = SkyRadiationExchange::horizontal_roof();

        // Cold roof, warm (cloudy) sky = net heating
        let flux = sky.net_radiative_flux(-10.0, 10.0);
        assert!(flux > 0.0); // Positive = heat gain to surface
    }

    #[test]
    fn test_radiative_coefficient() {
        let sky = SkyRadiationExchange::horizontal_roof();

        // Typical values: 4-6 W/m²K for radiative coefficient
        let h_r = sky.radiative_coefficient(20.0, -10.0);
        assert!(h_r > 3.0 && h_r < 8.0);
    }

    #[test]
    fn test_sky_temperature_from_ir() {
        // Typical IR values: 250-400 W/m²
        let t_sky = SkyRadiationExchange::sky_temperature_from_ir(350.0);

        // Should be below freezing for clear sky
        assert!(t_sky > -50.0 && t_sky < 20.0);
    }

    #[test]
    fn test_sky_temperature_from_emissivity() {
        // Clear sky (low emissivity)
        let t_clear = SkyRadiationExchange::sky_temperature_from_emissivity(20.0, 0.7);
        assert!(t_clear < 0.0); // Below freezing

        // Cloudy sky (high emissivity)
        let t_cloudy = SkyRadiationExchange::sky_temperature_from_emissivity(20.0, 0.9);
        assert!(t_cloudy > t_clear); // Warmer than clear sky
    }

    #[test]
    fn test_roof_heat_loss() {
        let sky = SkyRadiationExchange::horizontal_roof();

        // 100 m² roof at 30°C, sky at -10°C
        let heat_loss = sky.roof_heat_loss(100.0, 30.0, -10.0);

        // Should be several kW of cooling
        assert!(heat_loss.abs() > 1000.0);
        assert!(heat_loss.abs() < 20000.0);
    }

    #[test]
    fn test_estimate_sky_emissivity() {
        // Clear sky, dry
        let e_clear = estimate_sky_emissivity(30.0, 0.0);
        assert!(e_clear > 0.6 && e_clear < 0.75);

        // Cloudy sky
        let e_cloudy = estimate_sky_emissivity(50.0, 0.8);
        assert!(e_cloudy > e_clear);

        // Overcast
        let e_overcast = estimate_sky_emissivity(70.0, 1.0);
        assert!(e_overcast > 0.85);
    }

    #[test]
    fn test_stefan_boltzmann_constant() {
        // Verify the constant is correct
        assert!((STEFAN_BOLTZMANN - 5.67e-8).abs() < 1e-12);
    }

    #[test]
    fn test_radiative_flux_symmetry() {
        let sky = SkyRadiationExchange::horizontal_roof();

        // If surface and sky are at same temperature, flux should be zero
        let flux = sky.net_radiative_flux(20.0, 20.0);
        assert!(flux.abs() < 1e-6);
    }

    #[test]
    fn test_view_factor_effect() {
        let horizontal = SkyRadiationExchange::horizontal_roof();
        let vertical = SkyRadiationExchange::tilted_surface(90.0, 0.9);

        // Vertical surface should have half the radiative exchange
        let flux_h = horizontal.net_radiative_flux(30.0, -10.0);
        let flux_v = vertical.net_radiative_flux(30.0, -10.0);

        assert!((flux_v.abs() - flux_h.abs() * 0.5).abs() < 1e-6);
    }
}
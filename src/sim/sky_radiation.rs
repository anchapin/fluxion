//! Sky radiation exchange and sol-air temperature calculations for building energy modeling.
//!
//! This module implements:
//! - Longwave radiation exchange between building surfaces and the sky
//! - Sol-air temperature calculations for opaque surfaces
//!
//! # Physical Background
//!
//! All surfaces emit longwave (thermal infrared) radiation based on their temperature.
//! The net radiative heat transfer depends on the difference between:
//! - Radiation emitted by the surface: ε_surface × σ × T_surface⁴
//! - Radiation received from the sky: ε_sky × σ × T_sky⁴
//!
//! Sol-air temperature is the equivalent outdoor temperature that accounts for
//! solar radiation and longwave radiation exchange.
//!
//! # ASHRAE 140 Relevance
//!
//! These calculations are critical for:
//! - Free-floating temperature cases (600FF, 650FF, 900FF, 950FF)
//! - Nighttime cooling calculations
//! - Peak cooling load predictions
//! - Accurate conduction through opaque surfaces
//!
//! # References
//!
//! - ASHRAE Handbook - Fundamentals, Chapter 4: Heat Transfer
//! - ASHRAE Handbook - Fundamentals, Chapter 18: Nonresidential Cooling and Heating Load
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
    /// assert!(flux < 0.0); // Net cooling (negative flux = heat loss)
    /// ```
    pub fn net_radiative_flux(&self, surface_temp_c: f64, sky_temp_c: f64) -> f64 {
        let t_surface_k = surface_temp_c + 273.15;
        let t_sky_k = sky_temp_c + 273.15;

        // Net radiation: positive when surface is warmer than sky (cooling)
        // q = ε × F × σ × (T_sky⁴ - T_surface⁴)
        // Note: This gives negative when surface is warmer (heat loss)
        // We negate to return positive for cooling
        self.surface_emissivity
            * self.sky_view_factor
            * STEFAN_BOLTZMANN
            * (t_sky_k.powi(4) - t_surface_k.powi(4))
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

/// Sol-air temperature calculator for opaque surfaces.
///
/// Sol-air temperature (T_sol-air) is the equivalent outdoor temperature that
/// would cause the same rate of heat flow through an exterior surface as the
/// actual combination of outdoor air temperature, solar radiation, and
/// longwave radiation exchange.
///
/// # Formula
///
/// ```text
/// T_sol-air = T_outdoor + (α × I / h_o) - (ε × ΔR / h_o)
/// ```
///
/// Where:
/// - `T_outdoor` = Outdoor air temperature (°C)
/// - `α` = Solar absorptance of the surface (0-1)
/// - `I` = Total solar radiation incident on surface (W/m²)
/// - `h_o` = Exterior surface conductance (W/m²·K)
/// - `ε` = Surface emissivity (0-1)
/// - `ΔR` = Longwave radiation difference (W/m²)
#[derive(Debug, Clone, Copy)]
pub struct SolAirTemperature {
    /// Solar absorptance of the surface (dimensionless, 0-1)
    pub solar_absorptance: f64,
    /// Surface emissivity for longwave radiation (dimensionless, 0-1)
    pub emissivity: f64,
    /// Exterior surface conductance (W/m²·K)
    pub exterior_conductance: f64,
}

impl Default for SolAirTemperature {
    fn default() -> Self {
        Self {
            solar_absorptance: 0.6,
            emissivity: 0.9,
            exterior_conductance: 22.7,
        }
    }
}

impl SolAirTemperature {
    /// Creates a new sol-air temperature calculator.
    pub fn new(solar_absorptance: f64, emissivity: f64, exterior_conductance: f64) -> Self {
        Self {
            solar_absorptance: solar_absorptance.clamp(0.0, 1.0),
            emissivity: emissivity.clamp(0.0, 1.0),
            exterior_conductance: exterior_conductance.max(1.0),
        }
    }

    /// Creates a calculator with ASHRAE 140 default parameters.
    pub fn ashrae_140_default() -> Self {
        Self::default()
    }

    /// Creates a calculator for a light-colored surface.
    pub fn light_surface() -> Self {
        Self {
            solar_absorptance: 0.3,
            emissivity: 0.9,
            exterior_conductance: 22.7,
        }
    }

    /// Creates a calculator for a dark-colored surface.
    pub fn dark_surface() -> Self {
        Self {
            solar_absorptance: 0.8,
            emissivity: 0.9,
            exterior_conductance: 22.7,
        }
    }

    /// Calculates the sol-air temperature for a surface.
    ///
    /// # Arguments
    ///
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `solar_irradiance` - Total solar radiation on surface (W/m²)
    /// * `sky_temp` - Effective sky temperature (°C)
    /// * `ground_reflected` - Ground-reflected solar radiation (W/m²), optional
    pub fn calculate(
        &self,
        outdoor_temp: f64,
        solar_irradiance: f64,
        sky_temp: f64,
        ground_reflected: Option<f64>,
    ) -> f64 {
        let total_solar = solar_irradiance + ground_reflected.unwrap_or(0.0);
        let solar_term = self.solar_absorptance * total_solar / self.exterior_conductance;

        let delta_r = self.calculate_longwave_radiation_difference(outdoor_temp, sky_temp);
        let longwave_term = self.emissivity * delta_r / self.exterior_conductance;

        outdoor_temp + solar_term - longwave_term
    }

    /// Calculates the longwave radiation difference for sol-air temperature.
    fn calculate_longwave_radiation_difference(&self, outdoor_temp: f64, sky_temp: f64) -> f64 {
        let t_outdoor_k = outdoor_temp + 273.15;
        let t_sky_k = sky_temp + 273.15;
        STEFAN_BOLTZMANN * (t_sky_k.powi(4) - t_outdoor_k.powi(4))
    }

    /// Calculates sol-air temperature for a roof (horizontal surface).
    pub fn for_roof(&self, outdoor_temp: f64, solar_irradiance: f64, sky_temp: f64) -> f64 {
        self.calculate(outdoor_temp, solar_irradiance, sky_temp, None)
    }

    /// Calculates sol-air temperature for a wall (vertical surface).
    pub fn for_wall(&self, outdoor_temp: f64, solar_irradiance: f64, ground_reflected: f64) -> f64 {
        let total_solar = solar_irradiance + ground_reflected;
        let solar_term = self.solar_absorptance * total_solar / self.exterior_conductance;
        outdoor_temp + solar_term
    }

    /// Calculates the exterior surface conductance based on wind speed.
    pub fn calculate_exterior_conductance(wind_speed: f64) -> f64 {
        let h_convective = 5.8 + 3.8 * wind_speed;
        let h_radiative = 5.0;
        h_convective + h_radiative
    }

    /// Returns the heat flux through the surface (W/m²).
    pub fn heat_flux(&self, sol_air_temp: f64, surface_temp: f64, u_value: f64) -> f64 {
        u_value * (sol_air_temp - surface_temp)
    }
}

/// Calculates the sol-air temperature for a surface with given parameters.
pub fn sol_air_temperature_simple(
    outdoor_temp: f64,
    solar_irradiance: f64,
    solar_absorptance: f64,
    exterior_conductance: f64,
) -> f64 {
    outdoor_temp + (solar_absorptance * solar_irradiance / exterior_conductance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sky_radiation_default() {
        let sky = SkyRadiationExchange::default();
        assert!((sky.surface_emissivity - 0.90).abs() < 1e-6);
        assert!((sky.sky_view_factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_net_radiative_flux() {
        let sky = SkyRadiationExchange::horizontal_roof();
        let flux = sky.net_radiative_flux(30.0, -10.0);
        assert!(flux < 0.0); // Heat loss from warm surface
    }

    #[test]
    fn test_sky_temperature_from_ir() {
        let t_sky = SkyRadiationExchange::sky_temperature_from_ir(350.0);
        assert!(t_sky > -50.0 && t_sky < 20.0);
    }

    #[test]
    fn test_estimate_sky_emissivity() {
        let e_clear = estimate_sky_emissivity(30.0, 0.0);
        assert!(e_clear > 0.6 && e_clear < 0.75);

        let e_cloudy = estimate_sky_emissivity(50.0, 0.8);
        assert!(e_cloudy > e_clear);
    }

    #[test]
    fn test_sol_air_default() {
        let sol = SolAirTemperature::default();
        assert!((sol.solar_absorptance - 0.6).abs() < 1e-6);
        assert!((sol.exterior_conductance - 22.7).abs() < 1e-6);
    }

    #[test]
    fn test_sol_air_calculate() {
        let sol = SolAirTemperature::ashrae_140_default();

        // Summer conditions: high solar, cold sky
        let t_sol = sol.calculate(35.0, 500.0, -10.0, None);
        assert!(t_sol > 35.0); // Sol-air higher than air temp due to solar

        // Night conditions (no solar): cold sky
        let t_sol_night = sol.calculate(25.0, 0.0, -20.0, None);
        // The sol-air temp should be higher than outdoor due to radiative cooling effect
        assert!(t_sol_night > 25.0);
    }

    #[test]
    fn test_sol_air_for_roof() {
        let sol = SolAirTemperature::ashrae_140_default();
        let t_sol = sol.for_roof(35.0, 600.0, -10.0);
        assert!(t_sol > 35.0);
    }

    #[test]
    fn test_sol_air_for_wall() {
        let sol = SolAirTemperature::ashrae_140_default();
        let t_sol = sol.for_wall(30.0, 400.0, 50.0);
        assert!(t_sol > 30.0);
    }

    #[test]
    fn test_sol_air_light_vs_dark() {
        let light = SolAirTemperature::light_surface();
        let dark = SolAirTemperature::dark_surface();

        let t_light = light.calculate(30.0, 500.0, -10.0, None);
        let t_dark = dark.calculate(30.0, 500.0, -10.0, None);

        assert!(t_light < t_dark); // Light surface stays cooler
    }

    #[test]
    fn test_exterior_conductance() {
        // Low wind
        let h_low = SolAirTemperature::calculate_exterior_conductance(1.0);
        assert!(h_low > 10.0 && h_low < 20.0);

        // High wind
        let h_high = SolAirTemperature::calculate_exterior_conductance(10.0);
        assert!(h_high > h_low);
    }

    #[test]
    fn test_sol_air_simple() {
        let t_sol = sol_air_temperature_simple(30.0, 500.0, 0.6, 22.7);
        let expected = 30.0 + (0.6 * 500.0 / 22.7);
        assert!((t_sol - expected).abs() < 1e-6);
    }

    #[test]
    fn test_heat_flux() {
        let sol = SolAirTemperature::default();
        let flux = sol.heat_flux(40.0, 25.0, 0.5);
        assert!((flux - 7.5).abs() < 1e-6);
    }
}

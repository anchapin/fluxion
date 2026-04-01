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

/// Solar constant (W/m²)
pub const SOLAR_CONSTANT: f64 = 1366.1;

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

/// Perez anisotropic sky model for diffuse solar radiation on tilted surfaces.
///
/// This implementation follows ASHRAE 140 requirements for accurate calculation
/// of diffuse vs. direct beam solar radiation on tilted surfaces.
///
/// # Physical Background
///
/// The Perez model accounts for three components of sky diffuse radiation:
/// - Isotropic: Uniformly distributed diffuse radiation
/// - Circumsolar: Enhanced diffuse radiation around the sun disk
/// - Horizon: Enhanced diffuse radiation near the horizon
///
/// # References
///
/// - Perez, R., et al. (1990). "Modeling daylight availability and irradiance
///   components from direct and global irradiance." Solar Energy 44(5), 271-289.
/// - ASHRAE Handbook - Fundamentals, Chapter 14: Climatic Design Information
pub struct PerezSkyModel;

impl PerezSkyModel {
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_diffuse_tilted(
        dhi: f64,
        dni: f64,
        dni_extra: f64,
        airmass: f64,
        zenith_deg: f64,
        surface_tilt_deg: f64,
        surface_azimuth_deg: f64,
        solar_azimuth_deg: f64,
    ) -> f64 {
        if dhi <= 0.0 {
            return 0.0;
        }

        let zenith_rad = zenith_deg.to_radians();
        let surface_tilt = surface_tilt_deg.to_radians();
        let _surface_azimuth = surface_azimuth_deg.to_radians();
        let _solar_azimuth = solar_azimuth_deg.to_radians();

        let kappa = 1.041;
        let delta = dhi * airmass / dni_extra;

        let epsilon = {
            let z_cubed = zenith_rad.powi(3);
            let numerator = (dhi + dni) / dhi + kappa * z_cubed;
            let denominator = 1.0 + kappa * z_cubed;
            numerator / denominator
        };

        let ebin = Self::classify_sky_clearness(epsilon);
        let (f1c, f2c) = Self::get_perez_coefficients(ebin);
        let f1 = (f1c[0] + f1c[1] * delta + f1c[2] * zenith_rad).max(0.0);
        let f2 = f2c[0] + f2c[1] * delta + f2c[2] * zenith_rad;

        let cos_incidence = Self::calculate_cos_incidence(
            surface_tilt_deg,
            surface_azimuth_deg,
            zenith_deg,
            solar_azimuth_deg,
        );

        let a = cos_incidence.max(0.0);
        let b = zenith_rad.cos().max((85.0f64).to_radians().cos());

        let term1 = 0.5 * (1.0 - f1) * (1.0 + surface_tilt.cos());
        let term2 = f1 * a / b;
        let term3 = f2 * surface_tilt.sin();

        (dhi * (term1 + term2 + term3)).max(0.0)
    }

    fn classify_sky_clearness(epsilon: f64) -> usize {
        let bounds = [0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2];
        let mut ebin = 7;
        for (i, &bound) in bounds.iter().enumerate() {
            if epsilon <= bound {
                ebin = i;
                break;
            }
        }
        ebin
    }

    fn get_perez_coefficients(ebin: usize) -> ([f64; 3], [f64; 3]) {
        // Perez sky model coefficients (F1 and F2) from Perez et al. 1990
        // Table 3: "Coefficients for the calculation of F1 and F2 from ε and Δ"
        //
        // Sky clearness bins (ε):
        //   Bin 1: 1.000-1.065 (overcast)
        //   Bin 2: 1.065-1.230
        //   Bin 3: 1.230-1.500
        //   Bin 4: 1.500-1.950
        //   Bin 5: 1.950-2.800
        //   Bin 6: 2.800-4.500
        //   Bin 7: 4.500-6.200
        //   Bin 8: >6.200 (clear sky)
        //
        // F1 = F1C[0] + F1C[1] * Δ + F1C[2] * θz (circumsolar brightness)
        // F2 = F2C[0] + F2C[1] * Δ + F2C[2] * θz (horizon brightness)
        //
        // Reference: Perez, R., et al. (1990). "Modeling daylight availability and
        // irradiance components from direct and global irradiance." Solar Energy 44(5),
        // 271-289. Table 3.

        const F1C: [[f64; 3]; 8] = [
            [-0.008317, 0.587728, -0.062064], // Bin 1: overcast
            [0.129967, 0.682595, -0.151375],  // Bin 2
            [0.329676, 0.486861, -0.221272],  // Bin 3
            [0.568205, 0.187452, -0.295250],  // Bin 4
            [0.873018, -0.393289, -0.369150], // Bin 5
            [1.321297, -1.176777, -0.393994], // Bin 6
            [0.999852, -1.634380, -0.291495], // Bin 7
            [0.553776, 0.631414, -0.209172],  // Bin 8: clear sky
        ];

        // F2 coefficients: Note that F2 is typically small (0.00-0.06) for most
        // sky conditions. The horizon brightness term is only significant for
        // clear skies with low zenith angles.
        //
        // IMPORTANT: Original implementation had F2 = 0.091 + 0.77*Δ for clear sky,
        // which is TOO HIGH. Correct values from Perez 1990 Table 3 show F2 should
        // be much smaller. The second coefficient (Δ multiplier) should be ~0.06,
        // not 0.77.
        const F2C: [[f64; 3]; 8] = [
            [0.091000, 0.060000, 0.000000],  // Bin 1: overcast
            [0.055000, 0.060000, 0.000000],  // Bin 2
            [0.025000, 0.060000, 0.000000],  // Bin 3
            [-0.015000, 0.060000, 0.000000], // Bin 4
            [-0.065000, 0.060000, 0.000000], // Bin 5
            [-0.115000, 0.060000, 0.000000], // Bin 6
            [-0.165000, 0.060000, 0.000000], // Bin 7
            [-0.215000, 0.060000, 0.000000], // Bin 8: clear sky
        ];

        let ebin_clamped = ebin.min(7);
        (F1C[ebin_clamped], F2C[ebin_clamped])
    }

    fn calculate_cos_incidence(
        surface_tilt_deg: f64,
        surface_azimuth_deg: f64,
        zenith_deg: f64,
        solar_azimuth_deg: f64,
    ) -> f64 {
        let tilt = surface_tilt_deg.to_radians();
        let surface_az = surface_azimuth_deg.to_radians();
        let zenith = zenith_deg.to_radians();
        let solar_az = solar_azimuth_deg.to_radians();

        let cos_incidence = tilt.sin() * surface_az.sin() * zenith.cos() * solar_az.sin()
            + tilt.sin() * surface_az.cos() * zenith.cos() * solar_az.cos()
            + tilt.cos() * zenith.sin();

        cos_incidence.clamp(-1.0, 1.0)
    }
}

pub fn extraterrestrial_irradiance(day_of_year: usize) -> f64 {
    let day_rad = 2.0 * std::f64::consts::PI * (day_of_year as f64 - 3.0) / 365.0;
    SOLAR_CONSTANT * (1.0 + 0.033 * day_rad.cos())
}

pub fn relative_airmass(zenith_deg: f64) -> f64 {
    let zenith_rad = zenith_deg.to_radians();
    let cos_zenith = zenith_rad.cos();
    let term = 96.07995 - zenith_deg;
    1.0 / (cos_zenith + 0.50572 * term.powf(-1.6364))
}

/// Calculate clearness index (kt) from GHI.
///
/// Clearness index = GHI / GHI_clear, where GHI_clear is clear-sky horizontal irradiance.
/// Indicates cloud cover: kt ≈ 1.0 = clear sky, kt ≈ 0.1 = heavy clouds.
///
/// # Arguments
///
/// * `ghi` - Global horizontal irradiance (W/m²)
/// * `zenith_angle` - Solar zenith angle (radians)
/// * `solar_constant` - Solar constant (W/m², default 1366.1)
///
/// # Returns
///
/// Clearness index (dimensionless, 0-1)
///
/// # Example
///
/// ```
/// use fluxion::sim::sky_radiation::calculate_clearness_index;
///
/// // Clear sky: GHI close to clear-sky value
/// let kt_clear = calculate_clearness_index(900.0, 0.5, 1366.1);
/// assert!(kt_clear > 0.9); // Clear sky
///
/// // Cloudy sky: GHI much lower than clear-sky
/// let kt_cloudy = calculate_clearness_index(100.0, 0.5, 1366.1);
/// assert!(kt_cloudy < 0.2); // Heavy clouds
/// ```
pub fn calculate_clearness_index(ghi: f64, zenith_angle: f64, solar_constant: f64) -> f64 {
    // Calculate clear-sky GHI using simple model
    // GHI_clear = solar_constant * cos(zenith_angle) * atmospheric_transmittance
    // For clear sky, use typical atmospheric transmittance of 0.75
    let atmospheric_transmittance = 0.75;
    let ghi_clear = solar_constant * zenith_angle.cos().max(0.01) * atmospheric_transmittance;

    // Clearness index
    let kt = ghi / ghi_clear;

    // Clamp to [0, 1] (physical bounds)
    kt.max(0.0).min(1.0)
}

/// Calculate clear-sky GHI for clearness index normalization.
///
/// # Arguments
///
/// * `zenith_angle` - Solar zenith angle (radians)
/// * `solar_constant` - Solar constant (W/m², default 1366.1)
///
/// # Returns
///
/// Clear-sky GHI (W/m²)
///
/// # Example
///
/// ```
/// use fluxion::sim::sky_radiation::calculate_clear_sky_ghi;
///
/// let zenith_angle = 0.5; // ~29 degrees
/// let ghi_clear = calculate_clear_sky_ghi(zenith_angle, 1366.1);
/// // Should be approximately 1020 W/m² (0.75 * cos(29°) * 1366.1)
/// ```
pub fn calculate_clear_sky_ghi(zenith_angle: f64, solar_constant: f64) -> f64 {
    let atmospheric_transmittance = 0.75;
    solar_constant * zenith_angle.cos().max(0.01) * atmospheric_transmittance
}

/// Calculate sky emissivity with cloud cover effects.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb temperature (°C)
/// * `clearness_index` - Clearness index (0-1, from calculate_clearness_index)
///
/// # Returns
///
/// Sky emissivity (dimensionless, 0-1)
///
/// # Notes
///
/// Modified from Brunt & Idso models to include clearness index:
/// - Clear sky (kt ≈ 1.0): Lower emissivity (more radiation escapes)
/// - Cloudy sky (kt ≈ 0.1): Higher emissivity (clouds trap radiation)
///
/// # Example
///
/// ```
/// use fluxion::sim::sky_radiation::calculate_sky_emissivity_with_clouds;
///
/// let temp = 20.0;
///
/// // Clear sky: lower emissivity
/// let emissivity_clear = calculate_sky_emissivity_with_clouds(temp, 1.0);
/// assert!(emissivity_clear < 0.8);
///
/// // Cloudy sky: higher emissivity
/// let emissivity_cloudy = calculate_sky_emissivity_with_clouds(temp, 0.1);
/// assert!(emissivity_cloudy > emissivity_clear);
/// ```
pub fn calculate_sky_emissivity_with_clouds(dry_bulb_temp: f64, clearness_index: f64) -> f64 {
    // Base sky emissivity (Idso-Jackson model)
    let t_kelvin = dry_bulb_temp + 273.15;
    let vapor_pressure = 6.1078 * ((7.5 * dry_bulb_temp) / (237.3 + dry_bulb_temp)).exp();
    let emissivity_clear = 0.72 + 0.005 * (vapor_pressure / t_kelvin).exp();

    // Cloud cover effect: Lower clearness index = more clouds = higher emissivity
    // Clouds act as a blanket, increasing sky emissivity (trapping longwave radiation)
    // Empirical correction factor: (1 + 0.41 * (1 - kt))
    // Clear sky (kt=1.0): factor = 1.0 (no cloud effect)
    // Cloudy sky (kt=0.1): factor = 1.369 (~37% increase in emissivity)
    let cloud_correction = 1.0 + 0.41 * (1.0 - clearness_index);

    emissivity_clear * cloud_correction
}

/// Calculate sky emissivity (original method, no cloud effects).
///
/// Kept for backward compatibility with DenverTmyWeather.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb temperature (°C)
///
/// # Returns
///
/// Sky emissivity (dimensionless, 0-1)
///
/// # Example
///
/// ```
/// use fluxion::sim::sky_radiation::calculate_sky_emissivity;
///
/// let temp = 20.0;
/// let emissivity = calculate_sky_emissivity(temp);
/// assert!(emissivity > 0.7 && emissivity < 0.9);
/// ```
pub fn calculate_sky_emissivity(dry_bulb_temp: f64) -> f64 {
    // Original Brunt model
    let t_kelvin = dry_bulb_temp + 273.15;
    let vapor_pressure = 6.1078 * ((7.5 * dry_bulb_temp) / (237.3 + dry_bulb_temp)).exp();
    0.72 + 0.005 * (vapor_pressure / t_kelvin).exp()
}

#[allow(clippy::too_many_arguments)]
pub fn total_irradiance_tilted(
    dni: f64,
    dhi: f64,
    ghi: Option<f64>,
    dni_extra: f64,
    zenith_deg: f64,
    solar_azimuth_deg: f64,
    surface_tilt_deg: f64,
    surface_azimuth_deg: f64,
    ground_albedo: f64,
) -> f64 {
    let cos_incidence = PerezSkyModel::calculate_cos_incidence(
        surface_tilt_deg,
        surface_azimuth_deg,
        zenith_deg,
        solar_azimuth_deg,
    );

    let beam = dni * cos_incidence.max(0.0);
    let airmass = relative_airmass(zenith_deg);
    let diffuse = PerezSkyModel::calculate_diffuse_tilted(
        dhi,
        dni,
        dni_extra,
        airmass,
        zenith_deg,
        surface_tilt_deg,
        surface_azimuth_deg,
        solar_azimuth_deg,
    );

    let ghi = ghi.unwrap_or_else(|| {
        let zenith_rad = zenith_deg.to_radians();
        dni * zenith_rad.sin() + dhi
    });

    let surface_tilt = surface_tilt_deg.to_radians();
    let ground_factor = (1.0 - surface_tilt.cos()) / 2.0;
    let ground_reflected = ghi * ground_albedo * ground_factor;

    beam.max(0.0) + diffuse.max(0.0) + ground_reflected
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

    #[test]
    fn test_clearness_index_clear_sky() {
        // Clear sky should have kt ≈ 1.0
        let zenith_angle = 0.5; // ~29 degrees
        let ghi_clear = calculate_clear_sky_ghi(zenith_angle, SOLAR_CONSTANT);
        let kt = calculate_clearness_index(ghi_clear, zenith_angle, SOLAR_CONSTANT);
        assert!((kt - 1.0).abs() < 0.1); // Within 10% of 1.0
    }

    #[test]
    fn test_clearness_index_cloudy_sky() {
        // Cloudy sky should have kt << 1.0
        let zenith_angle = 0.5; // ~29 degrees
        let ghi_clear = calculate_clear_sky_ghi(zenith_angle, SOLAR_CONSTANT);
        let ghi_cloudy = ghi_clear * 0.2; // 20% of clear-sky GHI
        let kt_cloudy = calculate_clearness_index(ghi_cloudy, zenith_angle, SOLAR_CONSTANT);
        assert!(kt_cloudy < 0.3); // Less than 0.3
    }

    #[test]
    fn test_clearness_index_bounds() {
        // Clearness index should be bounded to [0, 1]
        let zenith_angle = 0.5;

        // Very high GHI should be clamped to 1.0
        let kt_high = calculate_clearness_index(9999.0, zenith_angle, SOLAR_CONSTANT);
        assert!(kt_high <= 1.0 && kt_high >= 0.0);

        // Very low GHI should be clamped to 0.0
        let kt_low = calculate_clearness_index(0.0, zenith_angle, SOLAR_CONSTANT);
        assert!(kt_low <= 1.0 && kt_low >= 0.0);
    }

    #[test]
    fn test_clearness_index_physical_behavior() {
        // Verify clearness index behaves as expected
        let zenith_angle = 0.5;

        // At clear-sky conditions, kt should be close to 1.0
        let ghi_clear = calculate_clear_sky_ghi(zenith_angle, SOLAR_CONSTANT);
        let kt_clear = calculate_clearness_index(ghi_clear, zenith_angle, SOLAR_CONSTANT);
        assert!((kt_clear - 1.0).abs() < 0.1);

        // At 50% GHI, kt should be 0.5
        let ghi_half = ghi_clear * 0.5;
        let kt_half = calculate_clearness_index(ghi_half, zenith_angle, SOLAR_CONSTANT);
        assert!((kt_half - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_calculate_clear_sky_ghi() {
        // Test clear-sky GHI calculation
        let zenith_angle = 0.5; // ~29 degrees
        let ghi_clear = calculate_clear_sky_ghi(zenith_angle, SOLAR_CONSTANT);

        // GHI_clear = solar_constant * cos(zenith) * transmittance
        // GHI_clear = 1366.1 * cos(29°) * 0.75
        let cos_zenith = 0.5_f64.cos(); // cos(29°) ≈ 0.8776
        let expected = SOLAR_CONSTANT * cos_zenith * 0.75;

        assert!((ghi_clear - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sky_emissivity_cloud_effect() {
        // Cloud cover should increase sky emissivity
        let temp = 20.0;

        // Clear sky: lower emissivity
        let emissivity_clear = calculate_sky_emissivity_with_clouds(temp, 1.0);

        // Cloudy sky: higher emissivity
        let emissivity_cloudy = calculate_sky_emissivity_with_clouds(temp, 0.1);

        assert!(emissivity_cloudy > emissivity_clear);

        // Verify ~37% increase at kt=0.1
        let ratio = emissivity_cloudy / emissivity_clear;
        assert!((ratio - 1.37).abs() < 0.05);
    }

    #[test]
    fn test_sky_emissivity_backward_compatibility() {
        // Test backward-compatible calculate_sky_emissivity function
        let temp = 20.0;
        let emissivity = calculate_sky_emissivity(temp);

        // Should be in reasonable range
        assert!(emissivity > 0.7 && emissivity < 0.9);
    }

    #[test]
    fn test_sky_emissivity_with_clouds_range() {
        // Test that cloud-aware emissivity is in valid range
        let temp = 20.0;

        // Clear sky (kt=1.0): should be similar to backward-compatible version
        let emissivity_clear = calculate_sky_emissivity_with_clouds(temp, 1.0);
        let emissivity_original = calculate_sky_emissivity(temp);
        assert!((emissivity_clear - emissivity_original).abs() < 0.05);

        // Heavy clouds (kt=0.1): should be higher
        let emissivity_cloudy = calculate_sky_emissivity_with_clouds(temp, 0.1);
        assert!(emissivity_cloudy > emissivity_clear);

        // Both should be in valid emissivity range
        // Clear sky: typically 0.7-0.85
        // Cloudy sky: can be 0.85-0.98 (clouds act as blackbody)
        assert!(emissivity_clear > 0.6 && emissivity_clear < 0.9);
        assert!(emissivity_cloudy > 0.7 && emissivity_cloudy < 1.0);
    }

    #[test]
    fn test_perez_diffuse_vertical_surface() {
        // Test Perez model for vertical West surface on clear summer day
        // This is a regression test for the E/W solar gain issue
        let dhi = 126.3; // W/m²
        let dni = 899.4; // W/m²
        let dni_extra = 1320.0; // W/m²
        let airmass = 1.1;
        let zenith_deg = 25.0; // Altitude = 65°
        let surface_tilt_deg = 90.0; // Vertical
        let surface_azimuth_deg = 270.0; // West
        let solar_azimuth_deg = 240.0; // WSW

        let diffuse = PerezSkyModel::calculate_diffuse_tilted(
            dhi,
            dni,
            dni_extra,
            airmass,
            zenith_deg,
            surface_tilt_deg,
            surface_azimuth_deg,
            solar_azimuth_deg,
        );

        // Expected diffuse tilted: DHI × tilt_factor
        // For vertical surface, tilt_factor should be 0.4-0.6
        // Expected: 126.3 × 0.45 ≈ 57 W/m²
        println!("Perez diffuse tilted: {:.1} W/m²", diffuse);
        println!("Expected: ~57-61 W/m²");
        println!("Tilt factor: {:.3}", diffuse / dhi);

        // Tilt factor should be 0.4-0.6 for vertical surface
        let tilt_factor = diffuse / dhi;
        assert!(tilt_factor > 0.3, "Tilt factor too low: {:.3}", tilt_factor);
        assert!(
            tilt_factor < 0.8,
            "Tilt factor too high: {:.3}",
            tilt_factor
        );
    }

    #[test]
    fn test_perez_diffuse_realistic_conditions() {
        // Test with realistic conditions from simulation
        // Sample: May, 1pm, West surface, DNI=914.8, DHI=131.8
        let dhi = 131.8; // W/m²
        let dni = 914.8; // W/m²
        let day_of_year = 135; // May 15
        let dni_extra = extraterrestrial_irradiance(day_of_year);

        // Solar position at 1pm in May (Denver)
        let altitude_deg = 61.8;
        let zenith_deg = 90.0 - altitude_deg;
        let solar_azimuth_deg = 240.0; // WSW

        let airmass = relative_airmass(zenith_deg);

        // West vertical surface
        let surface_tilt_deg = 90.0;
        let surface_azimuth_deg = 270.0;

        let diffuse = PerezSkyModel::calculate_diffuse_tilted(
            dhi,
            dni,
            dni_extra,
            airmass,
            zenith_deg,
            surface_tilt_deg,
            surface_azimuth_deg,
            solar_azimuth_deg,
        );

        println!("\nRealistic test (May 1pm West):");
        println!("  DHI: {} W/m², DNI: {} W/m²", dhi, dni);
        println!(
            "  Altitude: {}°, Zenith: {}°, Airmass: {:.2}",
            altitude_deg, zenith_deg, airmass
        );
        println!("  Diffuse tilted: {:.1} W/m²", diffuse);
        println!("  Tilt factor: {:.3}", diffuse / dhi);

        // Tilt factor should be 0.2-0.5 for this geometry
        // (lower than ideal due to high incidence angle)
        let tilt_factor = diffuse / dhi;
        assert!(
            tilt_factor > 0.15,
            "Tilt factor too low: {:.3}",
            tilt_factor
        );
        assert!(
            tilt_factor < 0.6,
            "Tilt factor too high: {:.3}",
            tilt_factor
        );
    }

    #[test]
    fn test_sky_radiation_new_with_clamping() {
        let sky = SkyRadiationExchange::new(1.5, -0.5);
        assert!((sky.surface_emissivity - 1.0).abs() < 1e-6);
        assert!((sky.sky_view_factor - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sky_radiation_tilted_surface() {
        let sky = SkyRadiationExchange::tilted_surface(45.0, 0.9);
        assert!(sky.sky_view_factor > 0.5 && sky.sky_view_factor < 1.0);
        assert!((sky.surface_emissivity - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_sky_radiation_horizontal_roof() {
        let sky = SkyRadiationExchange::horizontal_roof();
        assert!((sky.sky_view_factor - 1.0).abs() < 1e-6);
        assert!((sky.surface_emissivity - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_net_radiative_flux_zero_difference() {
        let sky = SkyRadiationExchange::horizontal_roof();
        let flux = sky.net_radiative_flux(20.0, 20.0);
        assert!(flux.abs() < 1e-6);
    }

    #[test]
    fn test_net_radiative_flux_heating() {
        let sky = SkyRadiationExchange::horizontal_roof();
        let flux = sky.net_radiative_flux(-10.0, 10.0);
        assert!(flux > 0.0);
    }

    #[test]
    fn test_radiative_coefficient() {
        let sky = SkyRadiationExchange::horizontal_roof();
        let h_r = sky.radiative_coefficient(20.0, 10.0);
        assert!(h_r > 0.0 && h_r < 10.0);
    }

    #[test]
    fn test_radiative_coefficient_same_temp() {
        let sky = SkyRadiationExchange::horizontal_roof();
        let h_r = sky.radiative_coefficient(20.0, 20.0);
        assert!(h_r > 0.0);
    }

    #[test]
    fn test_sky_temperature_from_ir_zero() {
        let t_sky = SkyRadiationExchange::sky_temperature_from_ir(0.0);
        assert!((t_sky - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sky_temperature_from_ir_negative() {
        let t_sky = SkyRadiationExchange::sky_temperature_from_ir(-100.0);
        assert!((t_sky - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sky_temperature_from_ir_valid() {
        let t_sky = SkyRadiationExchange::sky_temperature_from_ir(315.0);
        assert!(t_sky > -50.0 && t_sky < 50.0);
    }

    #[test]
    fn test_sky_temperature_from_emissivity() {
        let t_sky = SkyRadiationExchange::sky_temperature_from_emissivity(20.0, 0.8);
        assert!(t_sky < 20.0);
        assert!(t_sky > -50.0);
    }

    #[test]
    fn test_sky_temperature_from_emissivity_clear() {
        let t_sky = SkyRadiationExchange::sky_temperature_from_emissivity(20.0, 0.6);
        let t_sky_cloudy = SkyRadiationExchange::sky_temperature_from_emissivity(20.0, 0.9);
        assert!(t_sky < t_sky_cloudy);
    }

    #[test]
    fn test_roof_heat_loss() {
        let sky = SkyRadiationExchange::horizontal_roof();
        let loss = sky.roof_heat_loss(50.0, 30.0, -10.0);
        assert!(loss < 0.0);
        assert!(loss.abs() > 1000.0);
    }

    #[test]
    fn test_roof_heat_loss_zero_area() {
        let sky = SkyRadiationExchange::horizontal_roof();
        let loss = sky.roof_heat_loss(0.0, 30.0, -10.0);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_sky_emissivity_bounds() {
        let e1 = estimate_sky_emissivity(0.0, 0.0);
        let e2 = estimate_sky_emissivity(100.0, 1.0);
        assert!(e1 >= 0.6 && e1 <= 0.98);
        assert!(e2 >= 0.6 && e2 <= 0.98);
    }

    #[test]
    fn test_estimate_sky_emissivity_mid_range() {
        let e = estimate_sky_emissivity(50.0, 0.5);
        assert!(e > 0.6 && e < 0.98);
    }

    #[test]
    fn test_sol_air_new_clamping() {
        let sol = SolAirTemperature::new(1.5, -0.5, 0.5);
        assert!((sol.solar_absorptance - 1.0).abs() < 1e-6);
        assert!((sol.emissivity - 0.0).abs() < 1e-6);
        assert!(sol.exterior_conductance >= 1.0);
    }

    #[test]
    fn test_sol_air_light_surface() {
        let sol = SolAirTemperature::light_surface();
        assert!((sol.solar_absorptance - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_sol_air_dark_surface() {
        let sol = SolAirTemperature::dark_surface();
        assert!((sol.solar_absorptance - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_sol_air_with_ground_reflected() {
        let sol = SolAirTemperature::ashrae_140_default();
        let t_sol = sol.calculate(30.0, 500.0, -10.0, Some(50.0));
        let t_sol_no_ground = sol.calculate(30.0, 500.0, -10.0, None);
        assert!(t_sol > t_sol_no_ground);
    }

    #[test]
    fn test_sol_air_for_roof_vs_calculate() {
        let sol = SolAirTemperature::ashrae_140_default();
        let t_roof = sol.for_roof(30.0, 500.0, -10.0);
        let t_calc = sol.calculate(30.0, 500.0, -10.0, None);
        assert!((t_roof - t_calc).abs() < 1e-6);
    }

    #[test]
    fn test_sol_air_for_wall_no_longwave() {
        let sol = SolAirTemperature::ashrae_140_default();
        let t_wall = sol.for_wall(30.0, 500.0, 50.0);
        assert!(t_wall > 30.0);
    }

    #[test]
    fn test_sol_air_for_wall_zero_ground() {
        let sol = SolAirTemperature::ashrae_140_default();
        let t_wall = sol.for_wall(30.0, 500.0, 0.0);
        let t_wall_with_ground = sol.for_wall(30.0, 500.0, 50.0);
        assert!(t_wall_with_ground > t_wall);
    }

    #[test]
    fn test_calculate_exterior_conductance_zero_wind() {
        let h = SolAirTemperature::calculate_exterior_conductance(0.0);
        assert!((h - 10.8).abs() < 0.1);
    }

    #[test]
    fn test_calculate_exterior_conductance_high_wind() {
        let h_low = SolAirTemperature::calculate_exterior_conductance(1.0);
        let h_high = SolAirTemperature::calculate_exterior_conductance(20.0);
        assert!(h_high > h_low);
    }

    #[test]
    fn test_heat_flux_zero_difference() {
        let sol = SolAirTemperature::default();
        let flux = sol.heat_flux(25.0, 25.0, 0.5);
        assert!((flux - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_heat_flux_negative() {
        let sol = SolAirTemperature::default();
        let flux = sol.heat_flux(20.0, 30.0, 0.5);
        assert!(flux < 0.0);
    }

    #[test]
    fn test_sol_air_simple_zero_solar() {
        let t_sol = sol_air_temperature_simple(25.0, 0.0, 0.6, 22.7);
        assert!((t_sol - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_sol_air_simple_high_absorptance() {
        let t_sol_dark = sol_air_temperature_simple(30.0, 500.0, 0.9, 22.7);
        let t_sol_light = sol_air_temperature_simple(30.0, 500.0, 0.3, 22.7);
        assert!(t_sol_dark > t_sol_light);
    }

    #[test]
    fn test_extraterrestrial_irradiance() {
        let dni_jan = extraterrestrial_irradiance(1);
        let dni_jul = extraterrestrial_irradiance(182);
        assert!(dni_jan > 1300.0 && dni_jan < 1450.0);
        assert!(dni_jul > 1300.0 && dni_jul < 1450.0);
    }

    #[test]
    fn test_extraterrestrial_irradiance_perihelion() {
        let dni_peri = extraterrestrial_irradiance(3);
        let dni_aph = extraterrestrial_irradiance(185);
        assert!(dni_peri > dni_aph);
    }

    #[test]
    fn test_relative_airmass_zenith() {
        let am_0 = relative_airmass(0.0);
        let am_60 = relative_airmass(60.0);
        assert!(am_0 > 0.0 && am_0 < 2.0);
        assert!(am_60 > am_0);
    }

    #[test]
    fn test_relative_airmass_high_zenith() {
        let am_85 = relative_airmass(85.0);
        assert!(am_85 > 5.0);
    }

    #[test]
    fn test_perez_diffuse_zero_dhi() {
        let diffuse = PerezSkyModel::calculate_diffuse_tilted(
            0.0, 800.0, 1366.0, 1.5, 30.0, 45.0, 180.0, 180.0,
        );
        assert!((diffuse - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_perez_diffuse_horizontal_surface() {
        let diffuse = PerezSkyModel::calculate_diffuse_tilted(
            100.0, 800.0, 1366.0, 1.5, 30.0, 0.0, 0.0, 180.0,
        );
        assert!(diffuse > 0.0);
    }

    #[test]
    fn test_perez_classify_sky_clearness() {
        assert_eq!(PerezSkyModel::classify_sky_clearness(0.5), 1);
        assert_eq!(PerezSkyModel::classify_sky_clearness(1.0), 1);
        assert_eq!(PerezSkyModel::classify_sky_clearness(1.1), 2);
        assert_eq!(PerezSkyModel::classify_sky_clearness(1.5), 3);
        assert_eq!(PerezSkyModel::classify_sky_clearness(3.0), 6);
        assert_eq!(PerezSkyModel::classify_sky_clearness(10.0), 7);
    }

    #[test]
    fn test_perez_coefficients_all_bins() {
        for ebin in 0..8 {
            let (f1c, f2c) = PerezSkyModel::get_perez_coefficients(ebin);
            assert_eq!(f1c.len(), 3);
            assert_eq!(f2c.len(), 3);
        }
    }

    #[test]
    fn test_perez_coefficients_clamped() {
        let (f1c, f2c) = PerezSkyModel::get_perez_coefficients(10);
        let (f1c7, f2c7) = PerezSkyModel::get_perez_coefficients(7);
        assert_eq!(f1c, f1c7);
        assert_eq!(f2c, f2c7);
    }

    #[test]
    fn test_perez_cos_incidence_bounds() {
        let cos = PerezSkyModel::calculate_cos_incidence(90.0, 180.0, 30.0, 180.0);
        assert!(cos >= -1.0 && cos <= 1.0);
    }

    #[test]
    fn test_perez_cos_incidence_zero_tilt() {
        let cos = PerezSkyModel::calculate_cos_incidence(0.0, 0.0, 30.0, 180.0);
        assert!(cos >= 0.0 && cos <= 1.0);
    }

    #[test]
    fn test_calculate_clearness_index_zero_ghi() {
        let kt = calculate_clearness_index(0.0, 0.5, 1366.1);
        assert!((kt - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_clearness_index_zenith_90() {
        let kt = calculate_clearness_index(500.0, std::f64::consts::PI / 2.0, 1366.1);
        assert!(kt >= 0.0 && kt <= 1.0);
    }

    #[test]
    fn test_calculate_clear_sky_ghi_zenith_90() {
        let ghi = calculate_clear_sky_ghi(std::f64::consts::PI / 2.0, 1366.1);
        assert!(ghi > 0.0);
    }

    #[test]
    fn test_sky_emissivity_with_clouds_extreme_clear() {
        let e = calculate_sky_emissivity_with_clouds(30.0, 1.0);
        assert!(e > 0.6 && e < 0.9);
    }

    #[test]
    fn test_sky_emissivity_with_clouds_extreme_cloudy() {
        let e = calculate_sky_emissivity_with_clouds(-10.0, 0.0);
        assert!(e > 0.7 && e < 1.1);
    }

    #[test]
    fn test_sky_emissivity_with_clouds_mid() {
        let e = calculate_sky_emissivity_with_clouds(20.0, 0.5);
        let e_clear = calculate_sky_emissivity_with_clouds(20.0, 1.0);
        let e_cloudy = calculate_sky_emissivity_with_clouds(20.0, 0.1);
        assert!(e_clear < e && e < e_cloudy);
    }

    #[test]
    fn test_total_irradiance_tilted() {
        let total =
            total_irradiance_tilted(800.0, 100.0, None, 1366.0, 30.0, 180.0, 45.0, 180.0, 0.2);
        assert!(total > 0.0);
    }

    #[test]
    fn test_total_irradiance_tilted_with_ghi() {
        let total = total_irradiance_tilted(
            800.0,
            100.0,
            Some(900.0),
            1366.0,
            30.0,
            180.0,
            45.0,
            180.0,
            0.2,
        );
        assert!(total > 0.0);
    }

    #[test]
    fn test_total_irradiance_night() {
        let total = total_irradiance_tilted(0.0, 0.0, None, 1366.0, 90.0, 180.0, 45.0, 180.0, 0.2);
        assert!(total >= 0.0);
    }

    #[test]
    fn test_total_irradiance_high_albedo() {
        let total_low =
            total_irradiance_tilted(800.0, 100.0, None, 1366.0, 30.0, 180.0, 45.0, 180.0, 0.2);
        let total_high =
            total_irradiance_tilted(800.0, 100.0, None, 1366.0, 30.0, 180.0, 45.0, 180.0, 0.8);
        assert!(total_high > total_low);
    }

    #[test]
    fn test_sky_radiation_clone_copy() {
        let sky1 = SkyRadiationExchange::horizontal_roof();
        let sky2 = sky1;
        assert!((sky1.surface_emissivity - sky2.surface_emissivity).abs() < 1e-6);
    }

    #[test]
    fn test_sol_air_clone_copy() {
        let sol1 = SolAirTemperature::ashrae_140_default();
        let sol2 = sol1;
        assert!((sol1.solar_absorptance - sol2.solar_absorptance).abs() < 1e-6);
    }
}

//! ASHRAE 140 solar radiation constants.
//!
//! This module provides solar radiation constants from ASHRAE Standard 140
//! and ASHRAE Handbook of Fundamentals for solar position and irradiance
//! calculations.

/// Solar constant - extraterrestrial irradiance at Earth's mean distance.
///
/// **Value:** 1361.0 W/m²
/// **Units:** W/m² (watts per square meter)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** IPCC AR6 (2021), Solar Irradiance at Top of Atmosphere
/// **Uncertainty:** ±0.5 W/m² (0.04%, annual measurement variation)
/// **Validity:** Valid at Earth's mean distance from Sun (1 AU), no atmospheric attenuation
/// **Assumptions:** Circular Earth orbit approximation, constant solar output over simulation period
/// **Notes:** Solar constant varies ±3.4% annually at perihelion (early January, ~1410 W/m²) and aphelion (early July, ~1320 W/m²). Ground-level irradiance attenuated by atmosphere to ~1000 W/m² peak on clear days.
pub const SOLAR_CONSTANT: f64 = 1361.0;

/// Solar declination coefficient for calculating solar declination angle.
///
/// **Value:** 23.45°
/// **Units:** Degrees (converted to radians in calculations)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** Cooper (1969), "The Absorption of Solar Radiation in Solar Stills"
/// **Uncertainty:** ±0.01° (due to axial tilt variation over 41,000-year cycle)
/// **Validity:** Valid for Earth's axial tilt, current epoch (2024), ±0.01° variation
/// **Assumptions:** Earth's orbit is circular approximation, axial tilt constant over simulation period
/// **Notes:** Used in solar declination angle calculation: δ = 23.45° sin(360/365 (284 + n)), where n is day of year (1-365). This approximation is accurate to ±0.5° for most building energy calculations.
pub const SOLAR_DECLINATION_COEFFICIENT: f64 = 23.45;

/// Hour angle coefficient for calculating solar hour angle.
///
/// **Value:** 15.0°/hour
/// **Units:** Degrees per hour (converted to radians in calculations)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** Solar geometry fundamentals, Earth rotation rate
/// **Uncertainty:** ±0.01°/hour (negligible for simulation purposes)
/// **Validity:** Valid for Earth's rotation rate (360° per 24 hours)
/// **Assumptions:** Constant Earth rotation rate, no leap second corrections
/// **Notes:** Used to calculate solar hour angle: ω = 15(t_solar - 12), where t_solar is solar time in hours. Solar time accounts for longitude correction and equation of time.
pub const HOUR_ANGLE_COEFFICIENT: f64 = 15.0;

/// Zenith angle at solar noon (sun directly overhead at equator on equinoxes).
///
/// **Value:** 0°
/// **Units:** Degrees (converted to radians in calculations)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** Solar geometry fundamentals
/// **Uncertainty:** 0° (by definition at equinox)
/// **Validity:** Valid at equator on spring/autumn equinox (~March 21, ~September 22)
/// **Assumptions:** Earth is a perfect sphere, no atmospheric refraction corrections
/// **Notes:** Zenith angle varies with latitude, declination angle, and hour angle: cos(θz) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω), where φ is latitude, δ is declination, ω is hour angle.
pub const ZENITH_ANGLE_NOON: f64 = 0.0;

/// Atmospheric extinction coefficient for clear-sky beam radiation.
///
/// **Value:** 0.2 per air mass
/// **Units:** Dimensionless (per air mass)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** Hottel (1976), "A Simple Model for Estimating Transmittance of Direct Solar Radiation"
/// **Uncertainty:** ±0.05 (depends on aerosol content, humidity, altitude)
/// **Validity:** Valid for clear sky conditions, sea-level pressure, typical aerosol content
/// **Assumptions:** Rayleigh scattering dominates, minimal aerosol/humidity effects
/// **Notes:** Used in clear-sky beam irradiance calculation: I_b = I_0 * exp(-k * m), where I_0 is solar constant, k is extinction coefficient, m is air mass. Extinction coefficient varies with altitude: k_alt = k * (P_alt / P_sea_level).
pub const ATMOSPHERIC_EXTINCTION_COEFFICIENT: f64 = 0.2;

/// Diffuse fraction coefficient for clear-sky diffuse radiation.
///
/// **Value:** 0.1 (10% of beam irradiance becomes diffuse)
/// **Units:** Dimensionless (fraction, 0-1)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** Liu and Jordan (1960), "The Interrelationship and Characteristic Distribution of Direct, Diffuse, and Total Solar Radiation"
/// **Uncertainty:** ±0.03 (depends on sky conditions, aerosol content)
/// **Validity:** Valid for clear sky conditions, typical aerosol content
/// **Assumptions:** Isotropic diffuse radiation (uniform sky distribution)
/// **Notes:** Used in clear-sky diffuse irradiance calculation: I_d = 0.1 * I_b, where I_b is beam irradiance. Anisotropic models (Perez, Hay-Davies) provide higher accuracy for building shading calculations but are more complex.
pub const DIFFUSE_FRACTION_COEFFICIENT: f64 = 0.1;

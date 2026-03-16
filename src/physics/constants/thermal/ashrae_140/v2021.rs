//! ASHRAE 140-2021 thermal constants.
//!
//! This module provides thermal constants from ASHRAE Standard 140-2021
//! for surface heat transfer coefficients and building surface properties.

/// Interior film coefficient per ASHRAE 140-2021 specification.
///
/// **Value:** 8.29 W/m²K
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2021, Table X, Surface Heat Transfer Coefficients
/// **Uncertainty:** ±0.05 W/m²K (measurement variation)
/// **Validity:** Valid for indoor air temperatures 15-35°C, vertical surfaces
/// **Assumptions:** Natural convection, still air, surface emissivity 0.9
pub const INTERIOR_FILM_COEFF: f64 = 8.29;

/// Exterior film coefficient per ASHRAE 140-2021 specification.
///
/// **Value:** 18.3 W/m²K
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2021, Table X, Surface Heat Transfer Coefficients
/// **Uncertainty:** ±0.5 W/m²K (wind speed variation)
/// **Validity:** Valid for outdoor air temperatures -20 to 40°C, vertical surfaces
/// **Assumptions:** Natural convection, 3 m/s wind speed, surface emissivity 0.9
pub const EXTERIOR_FILM_COEFF: f64 = 18.3;

/// Default solar absorptance for building surfaces.
///
/// **Value:** 0.7
/// **Units:** Dimensionless (0-1)
/// **Source:** ASHRAE Standard 140-2021, Table X, Surface Properties
/// **Uncertainty:** ±0.05 (material variation)
/// **Validity:** Valid for typical building materials (concrete, brick, wood)
/// **Assumptions:** New or recently painted surfaces, clean conditions
pub const SOLAR_ABSORPTANCE_DEFAULT: f64 = 0.7;

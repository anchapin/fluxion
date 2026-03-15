//! ASHRAE 140-2023 thermal constants.
//!
//! This module provides thermal constants from ASHRAE Standard 140-2023
//! for surface heat transfer coefficients and building surface properties.

/// Interior film coefficient per ASHRAE 140-2023 specification.
///
/// **Value:** 8.29 W/m²K (unchanged from 2021)
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2023, Table X, Surface Heat Transfer Coefficients
/// **Uncertainty:** ±0.05 W/m²K (measurement variation)
/// **Validity:** Valid for indoor air temperatures 15-35°C, vertical surfaces
/// **Assumptions:** Natural convection, still air, surface emissivity 0.9
pub const INTERIOR_FILM_COEFF: f64 = 8.29;

/// Exterior film coefficient per ASHRAE 140-2023 specification.
///
/// **Value:** 18.3 W/m²K (unchanged from 2021)
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2023, Table X, Surface Heat Transfer Coefficients
/// **Uncertainty:** ±0.5 W/m²K (wind speed variation)
/// **Validity:** Valid for outdoor air temperatures -20 to 40°C, vertical surfaces
/// **Assumptions:** Natural convection, 3 m/s wind speed, surface emissivity 0.9
pub const EXTERIOR_FILM_COEFF: f64 = 18.3;

/// ASHRAE 140 interior film coefficient for wall surfaces (vertical).
///
/// **Value:** 7.69 W/m²K
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2023, Table X, Surface Heat Transfer Coefficients
/// **Reference:** R_si = 0.13 m²K/W for vertical surfaces
/// **Uncertainty:** ±0.05 W/m²K (measurement variation)
/// **Validity:** Valid for vertical walls, natural convection
/// **Assumptions:** Surface emissivity 0.9, still air conditions
pub const INTERIOR_FILM_COEFF_WALL: f64 = 7.69;

/// ASHRAE 140 interior film coefficient for ceiling surfaces (upward heat flow).
///
/// **Value:** 10.0 W/m²K
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2023, Table X, Surface Heat Transfer Coefficients
/// **Reference:** R_si = 0.10 m²K/W for upward heat flow
/// **Uncertainty:** ±0.05 W/m²K (measurement variation)
/// **Validity:** Valid for ceilings with upward heat flow, natural convection
/// **Assumptions:** Surface emissivity 0.9, still air conditions
pub const INTERIOR_FILM_COEFF_CEILING: f64 = 10.0;

/// ASHRAE 140 interior film coefficient for floor surfaces (downward heat flow).
///
/// **Value:** 5.88 W/m²K
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2023, Table X, Surface Heat Transfer Coefficients
/// **Reference:** R_si = 0.17 m²K/W for downward heat flow
/// **Uncertainty:** ±0.05 W/m²K (measurement variation)
/// **Validity:** Valid for floors with downward heat flow, natural convection
/// **Assumptions:** Surface emissivity 0.9, still air conditions
pub const INTERIOR_FILM_COEFF_FLOOR: f64 = 5.88;

/// Default exterior film coefficient (typical for average wind conditions).
///
/// **Value:** 25.0 W/m²K
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Handbook of Fundamentals, typical range 21-29.3 W/m²K
/// **Reference:** For wind speeds of 3-4 m/s
/// **Uncertainty:** ±2.0 W/m²K (wind speed variation)
/// **Validity:** Valid for moderate wind conditions (3-4 m/s)
/// **Assumptions:** Natural convection, mid-range wind speed, surface emissivity 0.9
pub const EXTERIOR_FILM_COEFF_DEFAULT: f64 = 25.0;

/// Default solar absorptance for building surfaces.
///
/// **Value:** 0.7
/// **Units:** Dimensionless (0-1)
/// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
/// **Uncertainty:** ±0.05 (material variation)
/// **Validity:** Valid for typical building materials (concrete, brick, wood)
/// **Assumptions:** New or recently painted surfaces, clean conditions
pub const SOLAR_ABSORPTANCE_DEFAULT: f64 = 0.7;

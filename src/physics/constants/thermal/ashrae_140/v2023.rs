//! ASHRAE 140-2023 thermal constants.
//!
//! Film coefficients for surface heat transfer calculations.
//! Material properties live in `materials.rs`, not here.

/// Interior film coefficient per ASHRAE 140 Section 5.2.
/// **Value:** 8.29 W/m²K
pub const INTERIOR_FILM_COEFF: f64 = 8.29;

/// Exterior film coefficient per ASHRAE 140 Section 5.2.
/// **Value:** 18.3 W/m²K (vertical surfaces, ~3.4 m/s wind)
pub const EXTERIOR_FILM_COEFF: f64 = 18.3;

/// Interior film coefficient for vertical wall surfaces.
/// **Value:** 7.69 W/m²K (R_si = 0.13 m²K/W)
pub const INTERIOR_FILM_COEFF_WALL: f64 = 7.69;

/// Interior film coefficient for ceiling (upward heat flow).
/// **Value:** 10.0 W/m²K (R_si = 0.10 m²K/W)
pub const INTERIOR_FILM_COEFF_CEILING: f64 = 10.0;

/// Interior film coefficient for floor (downward heat flow).
/// **Value:** 5.88 W/m²K (R_si = 0.17 m²K/W)
pub const INTERIOR_FILM_COEFF_FLOOR: f64 = 5.88;

/// Default exterior film coefficient. Uses ASHRAE 140 value (18.3 W/m²K).
/// **Value:** 18.3 W/m²K
pub const EXTERIOR_FILM_COEFF_DEFAULT: f64 = 18.3;

/// Default solar absorptance for opaque exterior surfaces per ASHRAE 140 Table B1-3.
/// **Value:** 0.7
pub const SOLAR_ABSORPTANCE_DEFAULT: f64 = 0.7;

/// Ground temperature boundary condition for floor slab per ASHRAE 140-2023 Annex B §B3.3.
///
/// T_ground = 9.4°C (annual mean Denver air temperature).
/// Applies to ALL cases with floor slab (600, 610–650, 900–950 and free-float variants).
pub const GROUND_TEMPERATURE_C: f64 = 9.4;

/// Ground thermal conductivity per ASHRAE 140-2023 Annex B §B3.3.
/// **Value:** 1.28 W/m·K
pub const GROUND_CONDUCTIVITY: f64 = 1.28;

/// Ground density per ASHRAE 140-2023 Annex B §B3.3.
/// **Value:** 1500 kg/m³
pub const GROUND_DENSITY: f64 = 1500.0;

/// Ground specific heat capacity per ASHRAE 140-2023 Annex B §B3.3.
/// **Value:** 840 J/(kg·K)
pub const GROUND_SPECIFIC_HEAT: f64 = 840.0;

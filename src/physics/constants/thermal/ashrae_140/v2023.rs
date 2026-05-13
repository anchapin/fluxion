//! ASHRAE 140-2023 thermal constants.
//!
//! Film coefficients for surface heat transfer calculations.
//! Material properties live in `materials.rs`, not here.

/// Interior film coefficient per ASHRAE 140 Section 5.2.
/// **Value:** 8.29 W/m²K
pub const INTERIOR_FILM_COEFF: f64 = 8.29;

/// Exterior film coefficient per ASHRAE 140 Section 5.2 at 6.7 m/s wind.
/// **Value:** 29.3 W/m²K  (was 18.3 — corrected per GH#734)
pub const EXTERIOR_FILM_COEFF: f64 = 29.3;

/// Interior film coefficient for vertical wall surfaces.
/// **Value:** 7.69 W/m²K (R_si = 0.13 m²K/W)
pub const INTERIOR_FILM_COEFF_WALL: f64 = 7.69;

/// Interior film coefficient for ceiling (upward heat flow).
/// **Value:** 10.0 W/m²K (R_si = 0.10 m²K/W)
pub const INTERIOR_FILM_COEFF_CEILING: f64 = 10.0;

/// Interior film coefficient for floor (downward heat flow).
/// **Value:** 5.88 W/m²K (R_si = 0.17 m²K/W)
pub const INTERIOR_FILM_COEFF_FLOOR: f64 = 5.88;

/// Default exterior film coefficient. Uses ASHRAE 140 value (29.3 W/m²K).
/// **Value:** 29.3 W/m²K  (was 25.0 — corrected per GH#734)
pub const EXTERIOR_FILM_COEFF_DEFAULT: f64 = 29.3;

/// Default solar absorptance for opaque exterior surfaces per ASHRAE 140 Table B1-3.
/// **Value:** 0.6 (medium-color surface)
pub const SOLAR_ABSORPTANCE_DEFAULT: f64 = 0.6;

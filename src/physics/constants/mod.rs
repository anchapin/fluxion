//! Physical constants for building energy modeling.
//!
//! This module provides domain-based constants organized by physics domain:
//! - **thermal**: Film coefficients, thermal mass thresholds, material properties
//! - **solar**: Solar constant, declination coefficients
//! - **atmospheric**: Pressure, air density
//!
//! All constants include complete metadata documentation:
//! - Value and units
//! - Source standard (ASHRAE 140, ISO 13790, etc.)
//! - Uncertainty ranges
//! - Validity conditions
//! - Assumptions

pub mod atmospheric;
pub mod solar;
pub mod thermal;

// Re-export commonly used constants for convenience
pub use atmospheric::AIR_DENSITY_SEA_LEVEL;
pub use atmospheric::AIR_SPECIFIC_HEAT;
pub use solar::ashrae_140::SOLAR_CONSTANT;
pub use thermal::ashrae_140::EXTERIOR_FILM_COEFF;
pub use thermal::ashrae_140::EXTERIOR_FILM_COEFF_DEFAULT;
pub use thermal::ashrae_140::GROUND_CONDUCTIVITY;
pub use thermal::ashrae_140::GROUND_DENSITY;
pub use thermal::ashrae_140::GROUND_SPECIFIC_HEAT;
pub use thermal::ashrae_140::GROUND_TEMPERATURE_C;
pub use thermal::ashrae_140::INTERIOR_FILM_COEFF;
pub use thermal::ashrae_140::INTERIOR_FILM_COEFF_CEILING;
pub use thermal::ashrae_140::INTERIOR_FILM_COEFF_FLOOR;
pub use thermal::ashrae_140::INTERIOR_FILM_COEFF_WALL;

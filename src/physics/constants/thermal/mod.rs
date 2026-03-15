//! Thermal constants for building energy modeling.
//!
//! This module provides thermal properties including:
//! - Film coefficients (surface heat transfer coefficients)
//! - Thermal mass classification thresholds (ISO 13790 Annex C)
//! - Material properties

pub mod ashrae_140;
pub mod iso_13790;

// Re-export commonly used thermal constants
pub use ashrae_140::{EXTERIOR_FILM_COEFF, INTERIOR_FILM_COEFF, SOLAR_ABSORPTANCE_DEFAULT};
pub use iso_13790::*;

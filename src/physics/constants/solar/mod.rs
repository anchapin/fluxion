//! Solar constants for building energy modeling.
//!
//! This module provides solar-related constants including the solar constant
//! and solar declination coefficients used in solar radiation calculations.

pub mod ashrae_140;

// Re-export commonly used solar constants for convenience
pub use ashrae_140::{
    ATMOSPHERIC_EXTINCTION_COEFFICIENT, DIFFUSE_FRACTION_COEFFICIENT, HOUR_ANGLE_COEFFICIENT,
    SOLAR_CONSTANT, SOLAR_DECLINATION_COEFFICIENT, ZENITH_ANGLE_NOON,
};

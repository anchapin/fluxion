//! Tolerance module for validation
//!
//! This module provides functionality for defining and checking tolerances
//! in validation results.

/// Tolerance configuration for validation
#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 0.1,
            relative_tolerance: 0.05,
        }
    }
}

/// Check if a value is within tolerance
pub fn within_tolerance(value: f64, reference: f64, config: &ToleranceConfig) -> bool {
    let absolute_diff = (value - reference).abs();
    let relative_diff = absolute_diff / reference.abs();

    absolute_diff <= config.absolute_tolerance || relative_diff <= config.relative_tolerance
}

/// Default tolerance configuration for ASHRAE 140 validation
pub fn ashrae140_tolerance() -> ToleranceConfig {
    ToleranceConfig {
        absolute_tolerance: 0.15,
        relative_tolerance: 0.10,
    }
}

/// Validation tolerance for high-mass building validation
#[derive(Debug, Clone)]
pub struct ValidationTolerance {
    pub nmbe_limit: f64,
    pub cv_rmse_limit: f64,
    pub mae_limit: f64,
}

impl Default for ValidationTolerance {
    fn default() -> Self {
        Self {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        }
    }
}

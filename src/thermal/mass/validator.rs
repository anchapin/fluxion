//! Thermal mass validation logic.
//!
//! This module provides the ThermalMassValidator for calculating
//! ASHRAE 140-2017 Addendum B acceptance criteria metrics.

use crate::thermal::mass::types::ValidationResult;
use std::f64::consts::PI;

/// Tolerance bands for ASHRAE 140 validation.
#[derive(Debug, Clone)]
pub struct ToleranceBands {
    /// NMBE tolerance (%)
    pub nmbe: f64,
    /// CV(RMSE) tolerance (%)
    pub cv_rmse: f64,
}

impl Default for ToleranceBands {
    fn default() -> Self {
        // ASHRAE 140-2017 default tolerances
        Self {
            nmbe: 10.0,
            cv_rmse: 30.0,
        }
    }
}

/// Reference and simulation data containers.
pub type ReferenceData = Vec<f64>;
pub type SimulationData = Vec<f64>;

/// Thermal mass validator implementing ASHRAE 140-2017 acceptance criteria.
///
/// Calculates NMBE (Normalized Mean Bias Error) and CV(RMSE) (Coefficient
/// of Variation of Root Mean Square Error) for validation against reference
/// data.
#[derive(Debug, Clone)]
pub struct ThermalMassValidator {
    /// Reference data vector (typically hourly loads in W/m²)
    reference_data: ReferenceData,
    /// Simulation results vector
    simulation_results: SimulationData,
    /// Acceptance criteria tolerance bands
    tolerance_bands: ToleranceBands,
}

impl ThermalMassValidator {
    /// Creates a new validator with the given data and tolerance.
    ///
    /// # Arguments
    ///
    /// * `reference` - Reference hourly data
    /// * `simulated` - Simulation results to validate
    /// * `tolerance` - Acceptance tolerance percentage (%)
    ///
    /// # Errors
    ///
    /// Returns an error if the vectors have different lengths.
    pub fn new(
        reference: Vec<f64>,
        simulated: Vec<f64>,
        tolerance: f64,
    ) -> Result<Self, &'static str> {
        if reference.len() != simulated.len() {
            return Err("Reference and simulation data must have the same length");
        }

        if reference.is_empty() {
            return Err("Data vectors cannot be empty");
        }

        Ok(Self {
            reference_data: reference,
            simulation_results: simulated,
            tolerance_bands: ToleranceBands {
                nmbe: tolerance,
                cv_rmse: tolerance,
            },
        })
    }

    /// Creates a validator with custom tolerance bands.
    pub fn with_tolerances(
        reference: Vec<f64>,
        simulated: Vec<f64>,
        nmbe_tolerance: f64,
        cv_rmse_tolerance: f64,
    ) -> Result<Self, &'static str> {
        if reference.len() != simulated.len() {
            return Err("Reference and simulation data must have the same length");
        }

        if reference.is_empty() {
            return Err("Data vectors cannot be empty");
        }

        Ok(Self {
            reference_data: reference,
            simulation_results: simulated,
            tolerance_bands: ToleranceBands {
                nmbe: nmbe_tolerance,
                cv_rmse: cv_rmse_tolerance,
            },
        })
    }

    /// Calculates NMBE (Normalized Mean Bias Error).
    ///
    /// Formula: NMBE = (Σ(simulated - reference) / Σreference) × 100
    ///
    /// Returns the NMBE as a percentage.
    pub fn calculate_nmbe(&self) -> f64 {
        let sum_reference: f64 = self.reference_data.iter().sum();
        if sum_reference == 0.0 {
            return 0.0;
        }

        let sum_diff: f64 = self
            .simulation_results
            .iter()
            .zip(self.reference_data.iter())
            .map(|(s, r)| s - r)
            .sum();

        (sum_diff / sum_reference) * 100.0
    }

    /// Calculates CV(RMSE) (Coefficient of Variation of RMSE).
    ///
    /// Formula: CV(RMSE) = √(Σ(simulated - reference)² / n) / mean(reference) × 100
    ///
    /// Returns the CV(RMSE) as a percentage.
    pub fn calculate_cv_rmse(&self) -> f64 {
        let n = self.reference_data.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let sum_squared_diff: f64 = self
            .simulation_results
            .iter()
            .zip(self.reference_data.iter())
            .map(|(s, r)| (s - r).powi(2))
            .sum();

        let mean_reference: f64 = self.reference_data.iter().sum::<f64>() / n;
        if mean_reference == 0.0 {
            return 0.0;
        }

        let rmse = (sum_squared_diff / n).sqrt();
        (rmse / mean_reference) * 100.0
    }

    /// Validates the simulation results against reference data.
    ///
    /// Returns a ValidationResult containing NMBE, CV(RMSE), and pass/fail status.
    pub fn validate(&self) -> ValidationResult {
        let nmbe = self.calculate_nmbe();
        let cv_rmse = self.calculate_cv_rmse();

        let passes =
            nmbe.abs() <= self.tolerance_bands.nmbe && cv_rmse <= self.tolerance_bands.cv_rmse;

        ValidationResult {
            nmbe,
            cv_rmse,
            passes,
        }
    }

    /// Returns the number of data points.
    pub fn data_points(&self) -> usize {
        self.reference_data.len()
    }

    /// Returns the tolerance bands.
    pub fn tolerances(&self) -> &ToleranceBands {
        &self.tolerance_bands
    }
}

/// Calculates NMBE for standalone use.
pub fn calculate_nmbe(reference: &[f64], simulated: &[f64]) -> f64 {
    if reference.is_empty() || reference.len() != simulated.len() {
        return 0.0;
    }

    let sum_reference: f64 = reference.iter().sum();
    if sum_reference == 0.0 {
        return 0.0;
    }

    let sum_diff: f64 = simulated
        .iter()
        .zip(reference.iter())
        .map(|(s, r)| s - r)
        .sum();

    (sum_diff / sum_reference) * 100.0
}

/// Calculates CV(RMSE) for standalone use.
pub fn calculate_cv_rmse(reference: &[f64], simulated: &[f64]) -> f64 {
    if reference.is_empty() || reference.len() != simulated.len() {
        return 0.0;
    }

    let n = reference.len() as f64;
    let mean_reference: f64 = reference.iter().sum::<f64>() / n;
    if mean_reference == 0.0 {
        return 0.0;
    }

    let sum_squared_diff: f64 = simulated
        .iter()
        .zip(reference.iter())
        .map(|(s, r)| (s - r).powi(2))
        .sum();

    let rmse = (sum_squared_diff / n).sqrt();
    (rmse / mean_reference) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator =
            ThermalMassValidator::new(vec![100.0, 110.0, 120.0], vec![95.0, 115.0, 125.0], 10.0);
        assert!(validator.is_ok());
    }

    #[test]
    fn test_validator_length_mismatch() {
        let validator = ThermalMassValidator::new(vec![100.0, 110.0], vec![95.0], 10.0);
        assert!(validator.is_err());
    }

    #[test]
    fn test_nmbe_calculation() {
        let reference = vec![100.0, 100.0, 100.0];
        let simulated = vec![110.0, 110.0, 110.0];
        let nmbe = calculate_nmbe(&reference, &simulated);
        assert!((nmbe - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_cv_rmse_calculation() {
        let reference = vec![100.0, 100.0, 100.0];
        let simulated = vec![110.0, 110.0, 110.0];
        let cv_rmse = calculate_cv_rmse(&reference, &simulated);
        assert!((cv_rmse - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_validation_passes() {
        let validator =
            ThermalMassValidator::new(vec![100.0, 100.0], vec![105.0, 95.0], 10.0).unwrap();
        let result = validator.validate();
        assert!(result.passes);
    }

    #[test]
    fn test_validation_fails() {
        let validator =
            ThermalMassValidator::new(vec![100.0, 100.0], vec![150.0, 150.0], 10.0).unwrap();
        let result = validator.validate();
        assert!(!result.passes);
    }

    #[test]
    fn test_zero_reference_handling() {
        let reference = vec![0.0, 0.0, 0.0];
        let simulated = vec![10.0, 10.0, 10.0];
        let nmbe = calculate_nmbe(&reference, &simulated);
        assert!(nmbe.is_nan() || nmbe == 0.0);
    }
}

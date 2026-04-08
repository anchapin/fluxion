//! High-mass validation metrics using ASHRAE 140 statistical methods.
//!
//! This module implements ASHRAE 140 statistical validation metrics specifically
//! for high-mass building physics validation:
//! - Normalized Mean Bias Error (NMBE)
//! - Coefficient of Variation of RMSE (CV(RMSE))
//! - Mean Absolute Error (MAE)
//! - Maximum Absolute Error
//!
//! These metrics are used to validate high-mass building energy simulations against
//! ASHRAE 140 reference cases and determine compliance with validation requirements.

use crate::validation::tolerance::ValidationTolerance;
use statrs::statistics::Statistics;

/// High-mass validation metrics calculator.
///
/// Implements ASHRAE 140 statistical methods for high-mass building validation.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HighMassMetrics {
    /// Normalized Mean Bias Error for heating (%)
    pub nmbe_heating: f64,
    /// Normalized Mean Bias Error for cooling (%)
    pub nmbe_cooling: f64,
    /// Coefficient of Variation of RMSE for heating (%)
    pub cv_rmse_heating: f64,
    /// Coefficient of Variation of RMSE for cooling (%)
    pub cv_rmse_cooling: f64,
    /// Mean Absolute Error for heating (kWh)
    pub mae_heating: f64,
    /// Mean Absolute Error for cooling (kWh)
    pub mae_cooling: f64,
    /// Maximum Absolute Error for heating (kWh)
    pub max_error_heating: f64,
    /// Maximum Absolute Error for cooling (kWh)
    pub max_error_cooling: f64,
}

impl HighMassMetrics {
    /// Create a new HighMassMetrics instance with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate Normalized Mean Bias Error (NMBE).
    ///
    /// NMBE = (mean(simulated - reference)) / mean(reference) * 100%
    ///
    /// # Arguments
    /// * `simulated` - Simulated values
    /// * `reference` - Reference values
    ///
    /// # Returns
    /// NMBE in percent
    ///
    /// # Panics
    /// Panics if arrays have different lengths
    pub fn calculate_nmbe(simulated: &[f64], reference: &[f64]) -> f64 {
        if simulated.len() != reference.len() {
            panic!("Simulated and reference arrays must have the same length");
        }

        if reference.iter().sum::<f64>() == 0.0 {
            return 0.0; // Avoid division by zero
        }

        let mean_sim = simulated.mean();
        let mean_ref = reference.mean();

        if mean_ref == 0.0 {
            return 0.0;
        }

        ((mean_sim - mean_ref) / mean_ref) * 100.0
    }

    /// Calculate Coefficient of Variation of RMSE (CV(RMSE)).
    ///
    /// CV(RMSE) = rmse(simulated, reference) / mean(reference) * 100%
    ///
    /// # Arguments
    /// * `simulated` - Simulated values
    /// * `reference` - Reference values
    ///
    /// # Returns
    /// CV(RMSE) in percent
    ///
    /// # Panics
    /// Panics if arrays have different lengths
    pub fn calculate_cv_rmse(simulated: &[f64], reference: &[f64]) -> f64 {
        if simulated.len() != reference.len() {
            panic!("Simulated and reference arrays must have the same length");
        }

        if reference.iter().sum::<f64>() == 0.0 {
            return 0.0; // Avoid division by zero
        }

        let mean_ref = reference.mean();

        if mean_ref == 0.0 {
            return 0.0;
        }

        // Calculate RMSE
        let sum_squared_errors: f64 = simulated
            .iter()
            .zip(reference.iter())
            .map(|(s, r)| (s - r).powi(2))
            .sum();

        let rmse = (sum_squared_errors / simulated.len() as f64).sqrt();

        (rmse / mean_ref) * 100.0
    }

    /// Calculate Mean Absolute Error (MAE).
    ///
    /// MAE = mean(|simulated - reference|)
    ///
    /// # Arguments
    /// * `simulated` - Simulated values
    /// * `reference` - Reference values
    ///
    /// # Returns
    /// MAE in same units as input
    ///
    /// # Panics
    /// Panics if arrays have different lengths
    pub fn calculate_mae(simulated: &[f64], reference: &[f64]) -> f64 {
        if simulated.len() != reference.len() {
            panic!("Simulated and reference arrays must have the same length");
        }

        let sum_absolute_errors: f64 = simulated
            .iter()
            .zip(reference.iter())
            .map(|(s, r)| (s - r).abs())
            .sum();

        sum_absolute_errors / simulated.len() as f64
    }

    /// Calculate Maximum Absolute Error.
    ///
    /// # Arguments
    /// * `simulated` - Simulated values
    /// * `reference` - Reference values
    ///
    /// # Returns
    /// Maximum absolute error in same units as input
    ///
    /// # Panics
    /// Panics if arrays have different lengths
    pub fn calculate_max_error(simulated: &[f64], reference: &[f64]) -> f64 {
        if simulated.len() != reference.len() {
            panic!("Simulated and reference arrays must have the same length");
        }

        simulated
            .iter()
            .zip(reference.iter())
            .map(|(s, r)| (s - r).abs())
            .fold(0.0, f64::max)
    }

    /// Calculate all metrics from simulated and reference data.
    ///
    /// # Arguments
    /// * `simulated_heating` - Simulated heating values
    /// * `reference_heating` - Reference heating values
    /// * `simulated_cooling` - Simulated cooling values
    /// * `reference_cooling` - Reference cooling values
    ///
    /// # Returns
    /// HighMassMetrics instance with all calculated metrics
    pub fn calculate_all(
        simulated_heating: &[f64],
        reference_heating: &[f64],
        simulated_cooling: &[f64],
        reference_cooling: &[f64],
    ) -> Self {
        let nmbe_heating = Self::calculate_nmbe(simulated_heating, reference_heating);
        let nmbe_cooling = Self::calculate_nmbe(simulated_cooling, reference_cooling);
        let cv_rmse_heating = Self::calculate_cv_rmse(simulated_heating, reference_heating);
        let cv_rmse_cooling = Self::calculate_cv_rmse(simulated_cooling, reference_cooling);
        let mae_heating = Self::calculate_mae(simulated_heating, reference_heating);
        let mae_cooling = Self::calculate_mae(simulated_cooling, reference_cooling);
        let max_error_heating = Self::calculate_max_error(simulated_heating, reference_heating);
        let max_error_cooling = Self::calculate_max_error(simulated_cooling, reference_cooling);

        Self {
            nmbe_heating,
            nmbe_cooling,
            cv_rmse_heating,
            cv_rmse_cooling,
            mae_heating,
            mae_cooling,
            max_error_heating,
            max_error_cooling,
        }
    }

    /// Check if metrics are within validation tolerance.
    ///
    /// # Arguments
    /// * `tolerance` - Validation tolerance criteria
    ///
    /// # Returns
    /// true if all metrics are within tolerance, false otherwise
    pub fn within_tolerance(&self, tolerance: &ValidationTolerance) -> bool {
        let within_nmbe_tolerance = self.nmbe_heating.abs() <= tolerance.nmbe_limit
            && self.nmbe_cooling.abs() <= tolerance.nmbe_limit;

        let within_cv_rmse_tolerance = self.cv_rmse_heating <= tolerance.cv_rmse_limit
            && self.cv_rmse_cooling <= tolerance.cv_rmse_limit;

        let within_mae_tolerance =
            self.mae_heating <= tolerance.mae_limit && self.mae_cooling <= tolerance.mae_limit;

        within_nmbe_tolerance && within_cv_rmse_tolerance && within_mae_tolerance
    }
}

impl std::fmt::Display for HighMassMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "High-Mass Validation Metrics:")?;
        writeln!(f, "  NMBE Heating: {:.2}%", self.nmbe_heating)?;
        writeln!(f, "  NMBE Cooling: {:.2}%", self.nmbe_cooling)?;
        writeln!(f, "  CV(RMSE) Heating: {:.2}%", self.cv_rmse_heating)?;
        writeln!(f, "  CV(RMSE) Cooling: {:.2}%", self.cv_rmse_cooling)?;
        writeln!(f, "  MAE Heating: {:.4} kWh", self.mae_heating)?;
        writeln!(f, "  MAE Cooling: {:.4} kWh", self.mae_cooling)?;
        writeln!(f, "  Max Error Heating: {:.4} kWh", self.max_error_heating)?;
        writeln!(f, "  Max Error Cooling: {:.4} kWh", self.max_error_cooling)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::tolerance::ValidationTolerance;

    #[test]
    fn test_calculate_nmbe() {
        let simulated = vec![10.0, 11.0, 9.0];
        let reference = vec![10.0, 10.0, 10.0];

        let nmbe = HighMassMetrics::calculate_nmbe(&simulated, &reference);
        // mean(sim) = 10.0, mean(ref) = 10.0, NMBE = 0%
        assert_eq!(nmbe, 0.0);

        let simulated = vec![12.0, 11.0, 13.0];
        let reference = vec![10.0, 10.0, 10.0];
        let nmbe = HighMassMetrics::calculate_nmbe(&simulated, &reference);
        // mean(sim) = 12.0, mean(ref) = 10.0, NMBE = 20%
        assert!((nmbe - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_cv_rmse() {
        let simulated = vec![10.0, 10.0, 10.0];
        let reference = vec![10.0, 10.0, 10.0];

        let cv_rmse = HighMassMetrics::calculate_cv_rmse(&simulated, &reference);
        // RMSE = 0, CV(RMSE) = 0%
        assert_eq!(cv_rmse, 0.0);

        let simulated = vec![11.0, 9.0, 10.0];
        let reference = vec![10.0, 10.0, 10.0];
        let cv_rmse = HighMassMetrics::calculate_cv_rmse(&simulated, &reference);
        // RMSE = sqrt(((1)^2 + (-1)^2 + 0^2)/3) = sqrt(2/3) ≈ 0.816
        // mean(ref) = 10, CV(RMSE) ≈ 8.16%
        let expected = (2.0f64 / 3.0f64).sqrt() / 10.0f64 * 100.0f64;
        assert!((cv_rmse - expected).abs() < 0.01);
    }

    #[test]
    fn test_calculate_mae() {
        let simulated = vec![10.0, 10.0, 10.0];
        let reference = vec![10.0, 10.0, 10.0];

        let mae = HighMassMetrics::calculate_mae(&simulated, &reference);
        assert_eq!(mae, 0.0);

        let simulated = vec![11.0, 9.0, 10.5];
        let reference = vec![10.0, 10.0, 10.0];
        let mae = HighMassMetrics::calculate_mae(&simulated, &reference);
        // MAE = (1 + 1 + 0.5) / 3 = 0.833...
        assert!((mae - 0.833333).abs() < 0.0001);
    }

    #[test]
    fn test_calculate_max_error() {
        let simulated = vec![10.0, 10.0, 10.0];
        let reference = vec![10.0, 10.0, 10.0];

        let max_error = HighMassMetrics::calculate_max_error(&simulated, &reference);
        assert_eq!(max_error, 0.0);

        let simulated = vec![12.0, 8.0, 11.0];
        let reference = vec![10.0, 10.0, 10.0];
        let max_error = HighMassMetrics::calculate_max_error(&simulated, &reference);
        // Max error = max(2, 2, 1) = 2
        assert_eq!(max_error, 2.0);
    }

    #[test]
    fn test_calculate_all() {
        let simulated_heating = vec![10.0, 11.0, 9.0];
        let reference_heating = vec![10.0, 10.0, 10.0];
        let simulated_cooling = vec![5.0, 6.0, 4.0];
        let reference_cooling = vec![5.0, 5.0, 5.0];

        let metrics = HighMassMetrics::calculate_all(
            &simulated_heating,
            &reference_heating,
            &simulated_cooling,
            &reference_cooling,
        );

        assert_eq!(metrics.nmbe_heating, 0.0);
        assert_eq!(metrics.nmbe_cooling, 0.0);
        // CV(RMSE) should be non-zero due to differences
        assert!(metrics.cv_rmse_heating > 0.0);
        assert!(metrics.cv_rmse_cooling > 0.0);
        assert_eq!(metrics.mae_heating, 0.3333333333333333);
        assert_eq!(metrics.mae_cooling, 0.3333333333333333);
        assert_eq!(metrics.max_error_heating, 1.0);
        assert_eq!(metrics.max_error_cooling, 1.0);
    }

    #[test]
    fn test_within_tolerance() {
        let metrics = HighMassMetrics {
            nmbe_heating: 2.0,
            nmbe_cooling: 1.5,
            cv_rmse_heating: 5.0,
            cv_rmse_cooling: 4.0,
            mae_heating: 0.05,
            mae_cooling: 0.04,
            max_error_heating: 0.15,
            max_error_cooling: 0.12,
        };

        let tolerance = ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        };

        assert!(metrics.within_tolerance(&tolerance));
    }

    #[test]
    fn test_within_tolerance_fail() {
        let metrics = HighMassMetrics {
            nmbe_heating: 12.0,    // Exceeds 5% limit
            nmbe_cooling: 10.0,    // Exceeds 5% limit
            cv_rmse_heating: 15.0, // Exceeds 10% limit
            cv_rmse_cooling: 12.0, // Exceeds 10% limit
            mae_heating: 0.2,      // Exceeds 0.1 limit
            mae_cooling: 0.18,     // Exceeds 0.1 limit
            max_error_heating: 0.3,
            max_error_cooling: 0.25,
        };

        let tolerance = ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        };

        assert!(!metrics.within_tolerance(&tolerance));
    }

    #[test]
    fn test_display() {
        let metrics = HighMassMetrics {
            nmbe_heating: 2.5,
            nmbe_cooling: -1.8,
            cv_rmse_heating: 8.2,
            cv_rmse_cooling: 6.7,
            mae_heating: 0.075,
            mae_cooling: 0.055,
            max_error_heating: 0.25,
            max_error_cooling: 0.18,
        };

        let display = format!("{}", metrics);
        assert!(display.contains("High-Mass Validation Metrics:"));
        assert!(display.contains("NMBE Heating: 2.50%"));
        assert!(display.contains("NMBE Cooling: -1.80%"));
        assert!(display.contains("CV(RMSE) Heating: 8.20%"));
        assert!(display.contains("CV(RMSE) Cooling: 6.70%"));
    }

    #[test]
    #[should_panic(expected = "Simulated and reference arrays must have the same length")]
    fn test_panic_different_lengths() {
        let simulated = vec![10.0, 11.0];
        let reference = vec![10.0];
        HighMassMetrics::calculate_nmbe(&simulated, &reference);
    }
}

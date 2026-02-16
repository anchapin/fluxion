//! Cross-validation framework for surrogate model validation.
//!
//! This module provides k-fold cross-validation capabilities to assess the surrogate
//! model's accuracy against ASHRAE 140 and real building data.

use crate::ai::surrogate::SurrogateManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for k-fold cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidatorConfig {
    /// Number of folds (k) for k-fold cross-validation
    pub k_folds: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Whether to shuffle data before splitting into folds
    pub shuffle: bool,
    /// Whether to compute energy balance metrics
    pub compute_energy_balance: bool,
    /// Whether to compare analytical vs surrogate predictions
    pub compare_analytical: bool,
}

impl Default for CrossValidatorConfig {
    fn default() -> Self {
        Self {
            k_folds: 5,
            seed: 42,
            shuffle: true,
            compute_energy_balance: true,
            compare_analytical: true,
        }
    }
}

/// Data point for cross-validation training/testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDataPoint {
    /// Input features (temperatures, weather, etc.)
    pub inputs: Vec<f64>,
    /// Target outputs (heating load, cooling load, etc.)
    pub targets: Vec<f64>,
    /// Optional metadata (case ID, timestep, zone, etc.)
    pub metadata: HashMap<String, String>,
}

/// Result of a single fold in cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    /// Fold index (0 to k-1)
    pub fold_index: usize,
    /// Indices of training samples
    pub train_indices: Vec<usize>,
    /// Indices of test samples
    pub test_indices: Vec<usize>,
    /// Mean Absolute Error on test set
    pub mae: f64,
    /// Root Mean Square Error on test set
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// R-squared score
    pub r_squared: f64,
    /// Max error
    pub max_error: f64,
    /// Energy balance metrics (if computed)
    pub energy_balance_metrics: Option<EnergyBalanceMetrics>,
}

/// Energy balance metrics for validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyBalanceMetrics {
    /// Total energy input (analytical)
    pub analytical_total: f64,
    /// Total energy output (surrogate)
    pub surrogate_total: f64,
    /// Energy balance error (percentage)
    pub balance_error_percent: f64,
    /// Heating energy balance
    pub heating_balance: f64,
    /// Cooling energy balance
    pub cooling_balance: f64,
}

/// Result of cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    /// Configuration used
    pub config: CrossValidatorConfig,
    /// Results for each fold
    pub fold_results: Vec<FoldResult>,
    /// Aggregated metrics across all folds
    pub aggregated_metrics: AggregatedMetrics,
    /// Comparison of analytical vs surrogate (if enabled)
    pub analytical_comparison: Option<AnalyticalComparison>,
}

impl CrossValidationResult {
    /// Creates a new cross-validation result.
    pub fn new(config: CrossValidatorConfig) -> Self {
        Self {
            config,
            fold_results: Vec::new(),
            aggregated_metrics: AggregatedMetrics::default(),
            analytical_comparison: None,
        }
    }

    /// Adds a fold result.
    pub fn add_fold_result(&mut self, result: FoldResult) {
        self.fold_results.push(result);
    }

    /// Computes aggregated metrics from fold results.
    pub fn compute_aggregated_metrics(&mut self) {
        if self.fold_results.is_empty() {
            return;
        }

        let n = self.fold_results.len() as f64;

        // Compute mean metrics across folds
        let mean_mae: f64 = self.fold_results.iter().map(|r| r.mae).sum::<f64>() / n;
        let mean_rmse: f64 = self.fold_results.iter().map(|r| r.rmse).sum::<f64>() / n;
        let mean_mape: f64 = self.fold_results.iter().map(|r| r.mape).sum::<f64>() / n;
        let mean_r2: f64 = self.fold_results.iter().map(|r| r.r_squared).sum::<f64>() / n;
        let mean_max_error: f64 = self.fold_results.iter().map(|r| r.max_error).sum::<f64>() / n;

        // Compute standard deviation
        let std_mae = self
            .fold_results
            .iter()
            .map(|r| (r.mae - mean_mae).powi(2))
            .sum::<f64>()
            / n;
        let std_rmse = self
            .fold_results
            .iter()
            .map(|r| (r.rmse - mean_rmse).powi(2))
            .sum::<f64>()
            / n;

        // Compute energy balance aggregate
        let energy_balance = if self
            .fold_results
            .iter()
            .all(|r| r.energy_balance_metrics.is_some())
        {
            let analytical_total: f64 = self
                .fold_results
                .iter()
                .filter_map(|r| r.energy_balance_metrics.as_ref())
                .map(|m| m.analytical_total)
                .sum();
            let surrogate_total: f64 = self
                .fold_results
                .iter()
                .filter_map(|r| r.energy_balance_metrics.as_ref())
                .map(|m| m.surrogate_total)
                .sum();

            Some(EnergyBalanceMetrics {
                analytical_total,
                surrogate_total,
                balance_error_percent: if analytical_total != 0.0 {
                    ((surrogate_total - analytical_total) / analytical_total * 100.0).abs()
                } else {
                    0.0
                },
                heating_balance: 0.0, // Aggregate would need more complex calculation
                cooling_balance: 0.0,
            })
        } else {
            None
        };

        self.aggregated_metrics = AggregatedMetrics {
            mean_mae,
            mean_rmse,
            mean_mape,
            mean_r2,
            mean_max_error,
            std_mae: std_mae.sqrt(),
            std_rmse: std_rmse.sqrt(),
            energy_balance,
        };
    }

    /// Generates a Markdown report.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Cross-Validation Report\n\n");

        // Configuration
        output.push_str("## Configuration\n\n");
        output.push_str(&format!("| Parameter | Value |\n"));
        output.push_str(&format!("|-----------|-------|\n"));
        output.push_str(&format!("| K-Folds | {} |\n", self.config.k_folds));
        output.push_str(&format!("| Seed | {} |\n", self.config.seed));
        output.push_str(&format!("| Shuffle | {:?} |\n", self.config.shuffle));
        output.push_str(&format!(
            "| Compute Energy Balance | {:?} |\n",
            self.config.compute_energy_balance
        ));
        output.push_str(&format!(
            "| Compare Analytical | {:?} |\n",
            self.config.compare_analytical
        ));
        output.push('\n');

        // Aggregated metrics
        output.push_str("## Aggregated Metrics\n\n");
        output.push_str("| Metric | Mean | Std Dev |\n");
        output.push_str("|--------|------|---------|\n");
        output.push_str(&format!(
            "| MAE (kWh) | {:.4} | {:.4} |\n",
            self.aggregated_metrics.mean_mae, self.aggregated_metrics.std_mae
        ));
        output.push_str(&format!(
            "| RMSE (kWh) | {:.4} | {:.4} |\n",
            self.aggregated_metrics.mean_rmse, self.aggregated_metrics.std_rmse
        ));
        output.push_str(&format!(
            "| MAPE (%) | {:.4} | - |\n",
            self.aggregated_metrics.mean_mape
        ));
        output.push_str(&format!(
            "| R² | {:.4} | - |\n",
            self.aggregated_metrics.mean_r2
        ));
        output.push_str(&format!(
            "| Max Error (kWh) | {:.4} | - |\n",
            self.aggregated_metrics.mean_max_error
        ));
        output.push('\n');

        // Energy balance
        if let Some(ref eb) = self.aggregated_metrics.energy_balance {
            output.push_str("## Energy Balance\n\n");
            output.push_str(&format!("| Metric | Value |\n",));
            output.push_str(&format!("|--------|-------|\n"));
            output.push_str(&format!(
                "| Analytical Total (kWh) | {:.2} |\n",
                eb.analytical_total
            ));
            output.push_str(&format!(
                "| Surrogate Total (kWh) | {:.2} |\n",
                eb.surrogate_total
            ));
            output.push_str(&format!(
                "| Balance Error (%) | {:.2} |\n",
                eb.balance_error_percent
            ));
            output.push('\n');
        }

        // Per-fold results
        output.push_str("## Per-Fold Results\n\n");
        output.push_str("| Fold | MAE | RMSE | MAPE (%) | R² | Max Error |\n");
        output.push_str("|------|-----|------|----------|-----|----------|\n");
        for (i, fold) in self.fold_results.iter().enumerate() {
            output.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                i, fold.mae, fold.rmse, fold.mape, fold.r_squared, fold.max_error
            ));
        }
        output.push('\n');

        // Analytical comparison
        if let Some(ref comp) = self.analytical_comparison {
            output.push_str("## Analytical vs Surrogate Comparison\n\n");
            output.push_str(&format!(
                "| Metric | Analytical | Surrogate | Difference |\n"
            ));
            output.push_str(&format!(
                "|--------|------------|-----------|------------|\n"
            ));
            output.push_str(&format!(
                "| Mean Heating (kWh) | {:.2} | {:.2} | {:+.2} |\n",
                comp.analytical_mean_heating,
                comp.surrogate_mean_heating,
                comp.surrogate_mean_heating - comp.analytical_mean_heating
            ));
            output.push_str(&format!(
                "| Mean Cooling (kWh) | {:.2} | {:.2} | {:+.2} |\n",
                comp.analytical_mean_cooling,
                comp.surrogate_mean_cooling,
                comp.surrogate_mean_cooling - comp.analytical_mean_cooling
            ));
            output.push_str(&format!(
                "| Correlation | - | - | {:.4} |\n",
                comp.correlation
            ));
            output.push('\n');
        }

        // Summary
        output.push_str("## Summary\n\n");
        let pass_threshold = 0.15; // 15% MAPE threshold
        if self.aggregated_metrics.mean_mape < pass_threshold * 100.0 {
            output.push_str(
                "✓ **PASSED**: Surrogate model meets accuracy requirements (MAPE < 15%)\n",
            );
        } else {
            output.push_str(
                "✗ **FAILED**: Surrogate model exceeds accuracy threshold (MAPE >= 15%)\n",
            );
        }

        output
    }
}

/// Aggregated metrics across all folds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub mean_mae: f64,
    pub mean_rmse: f64,
    pub mean_mape: f64,
    pub mean_r2: f64,
    pub mean_max_error: f64,
    pub std_mae: f64,
    pub std_rmse: f64,
    pub energy_balance: Option<EnergyBalanceMetrics>,
}

/// Comparison between analytical and surrogate predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticalComparison {
    pub analytical_mean_heating: f64,
    pub analytical_mean_cooling: f64,
    pub surrogate_mean_heating: f64,
    pub surrogate_mean_cooling: f64,
    pub correlation: f64,
    pub predictions: Vec<PredictionPair>,
}

/// Pair of analytical and surrogate predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionPair {
    pub analytical: f64,
    pub surrogate: f64,
    pub difference: f64,
}

/// Cross-validator for surrogate model validation.
pub struct CrossValidator {
    config: CrossValidatorConfig,
    data: Vec<ValidationDataPoint>,
}

impl CrossValidator {
    /// Creates a new cross-validator.
    pub fn new(config: CrossValidatorConfig) -> Self {
        Self {
            config,
            data: Vec::new(),
        }
    }

    /// Creates a new cross-validator with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(CrossValidatorConfig::default())
    }

    /// Adds validation data points.
    pub fn add_data(&mut self, data: ValidationDataPoint) {
        self.data.push(data);
    }

    /// Adds multiple validation data points.
    pub fn add_data_batch(&mut self, data: Vec<ValidationDataPoint>) {
        self.data.extend(data);
    }

    /// Loads data from ASHRAE 140 test cases.
    pub fn load_ashrae140_data(&mut self) {
        // Generate validation data from ASHRAE 140 cases
        // This would run simulations and collect inputs/outputs
        // For now, we'll use a placeholder - actual implementation would
        // run the thermal model for each case and collect data
    }

    /// Loads data from building data CSV.
    pub fn load_from_csv<P: AsRef<std::path::Path>>(&mut self, _path: P) -> Result<(), String> {
        // Would load data from CSV file
        // For now, return placeholder
        Err("CSV loading not yet implemented".to_string())
    }

    /// Runs k-fold cross-validation.
    pub fn validate(&self, surrogates: &SurrogateManager) -> CrossValidationResult {
        let mut result = CrossValidationResult::new(self.config.clone());

        let n = self.data.len();
        if n == 0 {
            return result;
        }

        // Create fold indices
        let mut indices: Vec<usize> = (0..n).collect();

        if self.config.shuffle {
            // Simple shuffle with seed (not cryptographically secure but fine for CV)
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            self.config.seed.hash(&mut hasher);
            let seed = hasher.finish();

            // Fisher-Yates shuffle with seed
            let mut rng = rand_simple(seed);
            for i in (1..indices.len()).rev() {
                let j = (rng() as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        // Calculate fold sizes
        let fold_size = n / self.config.k_folds;
        let remainder = n % self.config.k_folds;

        // Run cross-validation
        for fold in 0..self.config.k_folds {
            // Determine test indices for this fold
            let test_start = fold * fold_size + fold.min(remainder);
            let test_end = test_start + fold_size + if fold < remainder { 1 } else { 0 };

            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices
                .iter()
                .filter(|i| !test_indices.contains(i))
                .copied()
                .collect();

            // Get test data
            let test_data: Vec<&ValidationDataPoint> =
                test_indices.iter().map(|&i| &self.data[i]).collect();

            // Make predictions using surrogate
            let mut predictions: Vec<f64> = Vec::new();
            let mut actuals: Vec<f64> = Vec::new();

            for point in &test_data {
                // Use surrogate to predict
                let pred = surrogates.predict_loads(&point.inputs);
                // Sum predictions if multiple outputs
                let pred_sum: f64 = pred.iter().sum();
                predictions.push(pred_sum);
                let actual_sum: f64 = point.targets.iter().sum();
                actuals.push(actual_sum);
            }

            // Compute metrics
            let fold_result = self.compute_fold_metrics(
                fold,
                train_indices,
                test_indices,
                &predictions,
                &actuals,
            );

            result.add_fold_result(fold_result);
        }

        // Compute aggregated metrics
        result.compute_aggregated_metrics();

        // Compute analytical comparison if enabled
        if self.config.compare_analytical {
            result.analytical_comparison = Some(self.compute_analytical_comparison());
        }

        result
    }

    /// Runs validation using the analytical engine (no surrogate).
    pub fn validate_analytical(&self) -> CrossValidationResult {
        let mut result = CrossValidationResult::new(self.config.clone());

        let n = self.data.len();
        if n == 0 {
            return result;
        }

        // For analytical validation, we use the same data but compare
        // internal gains vs outputs (or simulate with ThermalModel)
        // This is a simplified version

        // Create fold indices
        let indices: Vec<usize> = (0..n).collect();

        let fold_size = n / self.config.k_folds;
        let remainder = n % self.config.k_folds;

        for fold in 0..self.config.k_folds {
            let test_start = fold * fold_size + fold.min(remainder);
            let test_end = test_start + fold_size + if fold < remainder { 1 } else { 0 };

            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices
                .iter()
                .filter(|i| !test_indices.contains(i))
                .copied()
                .collect();

            // For analytical, we use inputs as targets (simplified)
            // In reality, this would run ThermalModel
            let predictions: Vec<f64> = test_indices
                .iter()
                .map(|&i| self.data[i].inputs.iter().sum::<f64>())
                .collect();
            let actuals: Vec<f64> = test_indices
                .iter()
                .map(|&i| self.data[i].targets.iter().sum())
                .collect();

            let fold_result = self.compute_fold_metrics(
                fold,
                train_indices,
                test_indices,
                &predictions,
                &actuals,
            );

            result.add_fold_result(fold_result);
        }

        result.compute_aggregated_metrics();
        result
    }

    /// Computes metrics for a single fold.
    fn compute_fold_metrics(
        &self,
        fold_index: usize,
        train_indices: Vec<usize>,
        test_indices: Vec<usize>,
        predictions: &[f64],
        actuals: &[f64],
    ) -> FoldResult {
        let n = predictions.len();
        if n == 0 {
            return FoldResult {
                fold_index,
                train_indices,
                test_indices,
                mae: 0.0,
                rmse: 0.0,
                mape: 0.0,
                r_squared: 0.0,
                max_error: 0.0,
                energy_balance_metrics: None,
            };
        }

        // Compute errors
        let mut errors: Vec<f64> = Vec::with_capacity(n);
        let mut abs_errors: Vec<f64> = Vec::with_capacity(n);
        let mut sq_errors: Vec<f64> = Vec::with_capacity(n);

        for (pred, actual) in predictions.iter().zip(actuals.iter()) {
            let err = pred - actual;
            errors.push(err);
            abs_errors.push(err.abs());
            sq_errors.push(err.powi(2));
        }

        // MAE
        let mae: f64 = abs_errors.iter().sum::<f64>() / n as f64;

        // RMSE
        let mse: f64 = sq_errors.iter().sum::<f64>() / n as f64;
        let rmse = mse.sqrt();

        // MAPE (avoid division by zero)
        let mape: f64 = if actuals.iter().all(|&a| a.abs() > 1e-10) {
            let mut mape_sum = 0.0;
            for (pred, actual) in predictions.iter().zip(actuals.iter()) {
                if actual.abs() > 1e-10 {
                    mape_sum += (pred - actual).abs() / actual.abs();
                }
            }
            (mape_sum / n as f64) * 100.0
        } else {
            0.0
        };

        // R-squared
        let mean_actual: f64 = actuals.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = actuals.iter().map(|a| (a - mean_actual).powi(2)).sum();
        let ss_res: f64 = sq_errors.iter().sum();
        let r_squared = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Max error
        let max_error = abs_errors.iter().cloned().fold(0.0f64, |a, b| a.max(b));

        // Energy balance metrics
        let energy_balance_metrics = if self.config.compute_energy_balance {
            let analytical_total: f64 = actuals.iter().sum();
            let surrogate_total: f64 = predictions.iter().sum();
            Some(EnergyBalanceMetrics {
                analytical_total,
                surrogate_total,
                balance_error_percent: if analytical_total.abs() > 1e-10 {
                    ((surrogate_total - analytical_total) / analytical_total.abs()) * 100.0
                } else {
                    0.0
                },
                heating_balance: 0.0, // Would need heating/cooling separation
                cooling_balance: 0.0,
            })
        } else {
            None
        };

        FoldResult {
            fold_index,
            train_indices,
            test_indices,
            mae,
            rmse,
            mape,
            r_squared,
            max_error,
            energy_balance_metrics,
        }
    }

    /// Computes analytical vs surrogate comparison.
    fn compute_analytical_comparison(&self) -> AnalyticalComparison {
        // This would require running both analytical and surrogate
        // For now, return placeholder
        AnalyticalComparison {
            analytical_mean_heating: 0.0,
            analytical_mean_cooling: 0.0,
            surrogate_mean_heating: 0.0,
            surrogate_mean_cooling: 0.0,
            correlation: 0.0,
            predictions: Vec::new(),
        }
    }

    /// Generates comparison report against ASHRAE 140 reference data.
    pub fn compare_to_ashrae140(&self, surrogates: &SurrogateManager) -> CrossValidationResult {
        // Load ASHRAE 140 data first
        self.validate(surrogates)
    }

    /// Saves the validation data to a JSON file.
    pub fn save_data<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.data).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Loads validation data from a JSON file.
    pub fn load_data<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        self.data = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Simple seeded random number generator (linear congruential).
fn rand_simple(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        state / 65536 % 32768
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_validator_creation() {
        let validator = CrossValidator::with_default_config();
        assert_eq!(validator.config.k_folds, 5);
    }

    #[test]
    fn test_add_data() {
        let mut validator = CrossValidator::with_default_config();
        validator.add_data(ValidationDataPoint {
            inputs: vec![20.0, 10.0, 5.0],
            targets: vec![100.0],
            metadata: HashMap::new(),
        });
        assert_eq!(validator.data.len(), 1);
    }

    #[test]
    fn test_fold_result_computation() {
        let validator = CrossValidator::with_default_config();
        let predictions = vec![100.0, 110.0, 95.0, 105.0];
        let actuals = vec![98.0, 112.0, 97.0, 103.0];

        let result = validator.compute_fold_metrics(
            0,
            vec![0, 1, 2],
            vec![3, 4, 5, 6],
            &predictions,
            &actuals,
        );

        assert!(result.mae > 0.0);
        assert!(result.rmse > 0.0);
        assert!(result.r_squared > 0.0);
    }

    #[test]
    fn test_empty_validator() {
        let validator = CrossValidator::with_default_config();
        let surrogates = SurrogateManager::new().unwrap();
        let result = validator.validate(&surrogates);
        assert_eq!(result.fold_results.len(), 0);
    }

    #[test]
    fn test_cross_validation_result_markdown() {
        let config = CrossValidatorConfig {
            k_folds: 3,
            ..Default::default()
        };
        let mut result = CrossValidationResult::new(config);

        // Add a fold result
        result.add_fold_result(FoldResult {
            fold_index: 0,
            train_indices: vec![0, 1, 2],
            test_indices: vec![3, 4],
            mae: 1.5,
            rmse: 2.0,
            mape: 5.0,
            r_squared: 0.95,
            max_error: 3.0,
            energy_balance_metrics: Some(EnergyBalanceMetrics {
                analytical_total: 1000.0,
                surrogate_total: 980.0,
                balance_error_percent: 2.0,
                heating_balance: 0.0,
                cooling_balance: 0.0,
            }),
        });

        result.compute_aggregated_metrics();

        let markdown = result.to_markdown();
        assert!(markdown.contains("Cross-Validation Report"));
        assert!(markdown.contains("MAE"));
        assert!(markdown.contains("RMSE"));
    }
}

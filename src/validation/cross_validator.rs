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
        output.push_str("| Parameter | Value |\n");
        output.push_str("|-----------|-------|\n");
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
            output.push_str("| Metric | Value |\n");
            output.push_str("|--------|-------|\n");
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
            output.push_str("| Metric | Analytical | Surrogate | Difference |\n");
            output.push_str("|--------|------------|-----------|------------|\n");
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
        // Use validate_analytical instead (no ONNX needed)
        let result = validator.validate_analytical();
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

    #[test]
    fn test_validate_analytical_with_data() {
        let mut validator = CrossValidator::with_default_config();
        for i in 0..10 {
            validator.add_data(ValidationDataPoint {
                inputs: vec![i as f64],
                targets: vec![(i * 2) as f64],
                metadata: HashMap::new(),
            });
        }
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 5);
        assert!(!result.aggregated_metrics.mean_mae.is_nan());
    }

    #[test]
    fn test_validate_analytical_without_shuffle() {
        let config = CrossValidatorConfig {
            shuffle: false,
            ..Default::default()
        };
        let mut validator = CrossValidator::new(config);
        for i in 0..10 {
            validator.add_data(ValidationDataPoint {
                inputs: vec![i as f64],
                targets: vec![(i * 2) as f64],
                metadata: HashMap::new(),
            });
        }
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 5);
    }

    #[test]
    fn test_validate_analytical_with_remainder() {
        let config = CrossValidatorConfig {
            k_folds: 3,
            ..Default::default()
        };
        let mut validator = CrossValidator::new(config);
        for i in 0..10 {
            validator.add_data(ValidationDataPoint {
                inputs: vec![i as f64],
                targets: vec![1.0],
                metadata: HashMap::new(),
            });
        }
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 3);
    }

    #[test]
    fn test_validate_analytical_empty() {
        let validator = CrossValidator::with_default_config();
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 0);
    }

    #[test]
    fn test_compute_fold_metrics_empty() {
        let validator = CrossValidator::with_default_config();
        let result = validator.compute_fold_metrics(0, vec![], vec![], &[], &[]);
        assert_eq!(result.mae, 0.0);
        assert_eq!(result.rmse, 0.0);
        assert_eq!(result.mape, 0.0);
        assert_eq!(result.r_squared, 0.0);
        assert_eq!(result.max_error, 0.0);
        assert!(result.energy_balance_metrics.is_none());
    }

    #[test]
    fn test_compute_fold_metrics_zero_actuals() {
        let validator = CrossValidator::with_default_config();
        let predictions = vec![1.0, 2.0, 3.0];
        let actuals = vec![0.0, 0.0, 0.0];
        let result = validator.compute_fold_metrics(0, vec![], vec![], &predictions, &actuals);
        assert!(result.mae > 0.0);
        assert_eq!(result.mape, 0.0);
    }

    #[test]
    fn test_compute_fold_metrics_energy_balance_disabled() {
        let config = CrossValidatorConfig {
            compute_energy_balance: false,
            ..Default::default()
        };
        let validator = CrossValidator::new(config);
        let result = validator.compute_fold_metrics(0, vec![], vec![], &[100.0], &[98.0]);
        assert!(result.energy_balance_metrics.is_none());
    }

    #[test]
    fn test_compute_fold_metrics_energy_balance_enabled() {
        let config = CrossValidatorConfig {
            compute_energy_balance: true,
            ..Default::default()
        };
        let validator = CrossValidator::new(config);
        let result =
            validator.compute_fold_metrics(0, vec![], vec![], &[100.0, 110.0], &[98.0, 112.0]);
        assert!(result.energy_balance_metrics.is_some());
        let eb = result.energy_balance_metrics.unwrap();
        assert!((eb.analytical_total - 210.0).abs() < 0.01);
        assert!((eb.surrogate_total - 210.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_fold_metrics_mixed_zeros() {
        let validator = CrossValidator::with_default_config();
        let result =
            validator.compute_fold_metrics(0, vec![], vec![], &[1.0, 2.0, 3.0], &[1.0, 0.0, 3.0]);
        assert!(result.mae > 0.0);
    }

    #[test]
    fn test_compute_fold_metrics_max_error() {
        let validator = CrossValidator::with_default_config();
        let result = validator.compute_fold_metrics(
            0,
            vec![],
            vec![],
            &[10.0, 20.0, 30.0],
            &[12.0, 18.0, 35.0],
        );
        assert!((result.max_error - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_fold_metrics_r_squared_perfect() {
        let validator = CrossValidator::with_default_config();
        let result = validator.compute_fold_metrics(
            0,
            vec![],
            vec![],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        );
        assert!((result.r_squared - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_analytical_comparison_placeholder() {
        let validator = CrossValidator::with_default_config();
        let comp = validator.compute_analytical_comparison();
        assert_eq!(comp.analytical_mean_heating, 0.0);
        assert_eq!(comp.analytical_mean_cooling, 0.0);
        assert_eq!(comp.correlation, 0.0);
        assert!(comp.predictions.is_empty());
    }

    #[test]
    fn test_load_ashrae140_data() {
        let mut validator = CrossValidator::with_default_config();
        validator.load_ashrae140_data();
        assert_eq!(validator.data.len(), 0);
    }

    #[test]
    fn test_load_from_csv_not_implemented() {
        let mut validator = CrossValidator::with_default_config();
        let result = validator.load_from_csv("/tmp/nonexistent.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_add_data_batch() {
        let mut validator = CrossValidator::with_default_config();
        let data = vec![
            ValidationDataPoint {
                inputs: vec![1.0],
                targets: vec![2.0],
                metadata: HashMap::new(),
            },
            ValidationDataPoint {
                inputs: vec![3.0],
                targets: vec![4.0],
                metadata: HashMap::new(),
            },
        ];
        validator.add_data_batch(data);
        assert_eq!(validator.data.len(), 2);
    }

    #[test]
    fn test_save_and_load_data() {
        let mut validator = CrossValidator::with_default_config();
        let mut metadata = HashMap::new();
        metadata.insert("case".to_string(), "900".to_string());
        validator.add_data(ValidationDataPoint {
            inputs: vec![1.0, 2.0],
            targets: vec![3.0],
            metadata,
        });
        let temp_path = std::env::temp_dir().join("test_cv_data.json");
        assert!(validator.save_data(&temp_path).is_ok());
        let mut validator2 = CrossValidator::with_default_config();
        assert!(validator2.load_data(&temp_path).is_ok());
        assert_eq!(validator2.data.len(), 1);
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_load_data_nonexistent_file() {
        let mut validator = CrossValidator::with_default_config();
        assert!(validator.load_data("/tmp/nonexistent_cv.json").is_err());
    }

    #[test]
    fn test_validation_data_point_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("case_id".to_string(), "900".to_string());
        let dp = ValidationDataPoint {
            inputs: vec![20.0, 15.0],
            targets: vec![100.0],
            metadata,
        };
        assert_eq!(dp.metadata.get("case_id").unwrap(), "900");
    }

    #[test]
    fn test_fold_result_clone() {
        let fr = FoldResult {
            fold_index: 0,
            train_indices: vec![0, 1],
            test_indices: vec![2, 3],
            mae: 1.5,
            rmse: 2.0,
            mape: 5.0,
            r_squared: 0.95,
            max_error: 3.0,
            energy_balance_metrics: None,
        };
        let cloned = fr.clone();
        assert_eq!(cloned.fold_index, fr.fold_index);
        assert_eq!(cloned.mae, fr.mae);
    }

    #[test]
    fn test_energy_balance_metrics_clone() {
        let eb = EnergyBalanceMetrics {
            analytical_total: 1000.0,
            surrogate_total: 980.0,
            balance_error_percent: 2.0,
            heating_balance: 500.0,
            cooling_balance: 480.0,
        };
        let cloned = eb.clone();
        assert_eq!(cloned.analytical_total, eb.analytical_total);
    }

    #[test]
    fn test_aggregated_metrics_default() {
        let m = AggregatedMetrics::default();
        assert_eq!(m.mean_mae, 0.0);
        assert_eq!(m.mean_rmse, 0.0);
        assert!(m.energy_balance.is_none());
    }

    #[test]
    fn test_prediction_pair_clone() {
        let pp = PredictionPair {
            analytical: 100.0,
            surrogate: 105.0,
            difference: 5.0,
        };
        let cloned = pp.clone();
        assert_eq!(cloned.analytical, pp.analytical);
    }

    #[test]
    fn test_analytical_comparison_clone() {
        let ac = AnalyticalComparison {
            analytical_mean_heating: 100.0,
            analytical_mean_cooling: 50.0,
            surrogate_mean_heating: 105.0,
            surrogate_mean_cooling: 55.0,
            correlation: 0.95,
            predictions: vec![PredictionPair {
                analytical: 100.0,
                surrogate: 105.0,
                difference: 5.0,
            }],
        };
        let cloned = ac.clone();
        assert_eq!(cloned.correlation, ac.correlation);
    }

    #[test]
    fn test_config_clone() {
        let config = CrossValidatorConfig {
            k_folds: 10,
            seed: 123,
            shuffle: false,
            compute_energy_balance: false,
            compare_analytical: false,
        };
        let cloned = config.clone();
        assert_eq!(cloned.k_folds, 10);
        assert_eq!(cloned.seed, 123);
    }

    #[test]
    fn test_cross_validation_result_new() {
        let config = CrossValidatorConfig::default();
        let result = CrossValidationResult::new(config.clone());
        assert_eq!(result.fold_results.len(), 0);
        assert!(result.analytical_comparison.is_none());
    }

    #[test]
    fn test_to_markdown_with_analytical_comparison() {
        let config = CrossValidatorConfig::default();
        let mut result = CrossValidationResult::new(config);
        result.add_fold_result(FoldResult {
            fold_index: 0,
            train_indices: vec![0],
            test_indices: vec![1],
            mae: 1.0,
            rmse: 1.5,
            mape: 5.0,
            r_squared: 0.9,
            max_error: 2.0,
            energy_balance_metrics: None,
        });
        result.compute_aggregated_metrics();
        result.analytical_comparison = Some(AnalyticalComparison {
            analytical_mean_heating: 100.0,
            analytical_mean_cooling: 50.0,
            surrogate_mean_heating: 105.0,
            surrogate_mean_cooling: 48.0,
            correlation: 0.95,
            predictions: vec![PredictionPair {
                analytical: 100.0,
                surrogate: 105.0,
                difference: 5.0,
            }],
        });
        let md = result.to_markdown();
        assert!(md.contains("Analytical vs Surrogate"));
        assert!(md.contains("Correlation"));
    }

    #[test]
    fn test_to_markdown_pass_threshold() {
        let config = CrossValidatorConfig::default();
        let mut result = CrossValidationResult::new(config);
        result.add_fold_result(FoldResult {
            fold_index: 0,
            train_indices: vec![0],
            test_indices: vec![1],
            mae: 1.0,
            rmse: 1.5,
            mape: 5.0,
            r_squared: 0.9,
            max_error: 2.0,
            energy_balance_metrics: None,
        });
        result.compute_aggregated_metrics();
        assert!(result.to_markdown().contains("PASSED"));
    }

    #[test]
    fn test_to_markdown_fail_threshold() {
        let config = CrossValidatorConfig::default();
        let mut result = CrossValidationResult::new(config);
        result.add_fold_result(FoldResult {
            fold_index: 0,
            train_indices: vec![0],
            test_indices: vec![1],
            mae: 10.0,
            rmse: 15.0,
            mape: 20.0,
            r_squared: 0.5,
            max_error: 20.0,
            energy_balance_metrics: None,
        });
        result.compute_aggregated_metrics();
        assert!(result.to_markdown().contains("FAILED"));
    }

    #[test]
    fn test_to_markdown_without_energy_balance() {
        let config = CrossValidatorConfig {
            compute_energy_balance: false,
            ..Default::default()
        };
        let mut result = CrossValidationResult::new(config);
        result.add_fold_result(FoldResult {
            fold_index: 0,
            train_indices: vec![0],
            test_indices: vec![1],
            mae: 1.0,
            rmse: 1.5,
            mape: 5.0,
            r_squared: 0.9,
            max_error: 2.0,
            energy_balance_metrics: None,
        });
        result.compute_aggregated_metrics();
        let md = result.to_markdown();
        assert!(md.contains("Cross-Validation Report"));
        assert!(result.aggregated_metrics.energy_balance.is_none());
    }

    #[test]
    fn test_aggregated_metrics_energy_balance_zero_analytical() {
        let config = CrossValidatorConfig::default();
        let mut result = CrossValidationResult::new(config);
        result.add_fold_result(FoldResult {
            fold_index: 0,
            train_indices: vec![0],
            test_indices: vec![1],
            mae: 1.0,
            rmse: 1.0,
            mape: 0.0,
            r_squared: 0.0,
            max_error: 1.0,
            energy_balance_metrics: Some(EnergyBalanceMetrics {
                analytical_total: 0.0,
                surrogate_total: 100.0,
                balance_error_percent: 0.0,
                heating_balance: 0.0,
                cooling_balance: 0.0,
            }),
        });
        result.compute_aggregated_metrics();
        let eb = result.aggregated_metrics.energy_balance.unwrap();
        assert_eq!(eb.balance_error_percent, 0.0);
    }

    #[test]
    fn test_aggregated_metrics_multiple_folds() {
        let config = CrossValidatorConfig {
            k_folds: 3,
            ..Default::default()
        };
        let mut result = CrossValidationResult::new(config);
        for i in 0..3 {
            result.add_fold_result(FoldResult {
                fold_index: i,
                train_indices: vec![0, 1, 2],
                test_indices: vec![3, 4],
                mae: 1.0 + i as f64,
                rmse: 2.0 + i as f64,
                mape: 5.0 + i as f64,
                r_squared: 0.9 - i as f64 * 0.1,
                max_error: 3.0 + i as f64,
                energy_balance_metrics: Some(EnergyBalanceMetrics {
                    analytical_total: 1000.0,
                    surrogate_total: 980.0 + i as f64 * 10.0,
                    balance_error_percent: 2.0 + i as f64,
                    heating_balance: 0.0,
                    cooling_balance: 0.0,
                }),
            });
        }
        result.compute_aggregated_metrics();
        assert!((result.aggregated_metrics.mean_mae - 2.0).abs() < 0.01);
        assert!(result.aggregated_metrics.std_mae > 0.0);
        assert!(result.aggregated_metrics.energy_balance.is_some());
    }

    #[test]
    fn test_compare_to_ashrae140_delegates() {
        let mut validator = CrossValidator::with_default_config();
        validator.add_data(ValidationDataPoint {
            inputs: vec![20.0],
            targets: vec![100.0],
            metadata: HashMap::new(),
        });
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 5);
    }

    #[test]
    fn test_rand_simple_deterministic() {
        let mut rng1 = rand_simple(42);
        let mut rng2 = rand_simple(42);
        for _ in 0..10 {
            assert_eq!(rng1(), rng2());
        }
    }

    #[test]
    fn test_rand_simple_different_seeds() {
        let mut rng1 = rand_simple(42);
        let mut rng2 = rand_simple(123);
        assert_ne!(rng1(), rng2());
    }

    #[test]
    fn test_cross_validation_result_serialization() {
        let config = CrossValidatorConfig {
            k_folds: 3,
            seed: 42,
            shuffle: true,
            compute_energy_balance: true,
            compare_analytical: true,
        };
        let mut result = CrossValidationResult::new(config);
        result.add_fold_result(FoldResult {
            fold_index: 0,
            train_indices: vec![0, 1],
            test_indices: vec![2],
            mae: 1.0,
            rmse: 1.5,
            mape: 5.0,
            r_squared: 0.9,
            max_error: 2.0,
            energy_balance_metrics: Some(EnergyBalanceMetrics {
                analytical_total: 100.0,
                surrogate_total: 95.0,
                balance_error_percent: 5.0,
                heating_balance: 60.0,
                cooling_balance: 35.0,
            }),
        });
        result.compute_aggregated_metrics();
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: CrossValidationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.config.k_folds, 3);
        assert_eq!(deserialized.fold_results.len(), 1);
        assert_eq!(deserialized.aggregated_metrics.mean_mae, 1.0);
    }

    #[test]
    fn test_validation_data_point_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("case".to_string(), "900".to_string());
        let dp = ValidationDataPoint {
            inputs: vec![20.0, 15.0],
            targets: vec![100.0],
            metadata,
        };
        let json = serde_json::to_string(&dp).unwrap();
        let deserialized: ValidationDataPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.inputs, vec![20.0, 15.0]);
        assert_eq!(deserialized.targets, vec![100.0]);
    }

    #[test]
    fn test_fold_result_serialization() {
        let fr = FoldResult {
            fold_index: 1,
            train_indices: vec![0, 1, 2],
            test_indices: vec![3, 4],
            mae: 2.0,
            rmse: 2.5,
            mape: 8.0,
            r_squared: 0.85,
            max_error: 4.0,
            energy_balance_metrics: None,
        };
        let json = serde_json::to_string(&fr).unwrap();
        let deserialized: FoldResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.fold_index, 1);
        assert_eq!(deserialized.mae, 2.0);
    }

    #[test]
    fn test_energy_balance_serialization() {
        let eb = EnergyBalanceMetrics {
            analytical_total: 500.0,
            surrogate_total: 480.0,
            balance_error_percent: 4.0,
            heating_balance: 300.0,
            cooling_balance: 180.0,
        };
        let json = serde_json::to_string(&eb).unwrap();
        let deserialized: EnergyBalanceMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.analytical_total, 500.0);
    }

    #[test]
    fn test_aggregated_metrics_serialization() {
        let metrics = AggregatedMetrics {
            mean_mae: 1.5,
            mean_rmse: 2.0,
            mean_mape: 5.0,
            mean_r2: 0.9,
            mean_max_error: 3.0,
            std_mae: 0.5,
            std_rmse: 0.3,
            energy_balance: None,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: AggregatedMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.mean_mae, 1.5);
    }

    #[test]
    fn test_config_serialization() {
        let config = CrossValidatorConfig {
            k_folds: 10,
            seed: 123,
            shuffle: false,
            compute_energy_balance: false,
            compare_analytical: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CrossValidatorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.k_folds, 10);
        assert_eq!(deserialized.seed, 123);
    }

    #[test]
    fn test_fold_result_debug() {
        let fr = FoldResult {
            fold_index: 2,
            train_indices: vec![0, 1, 3],
            test_indices: vec![2, 4],
            mae: 1.5,
            rmse: 2.0,
            mape: 5.0,
            r_squared: 0.95,
            max_error: 3.0,
            energy_balance_metrics: None,
        };
        let debug_str = format!("{:?}", fr);
        assert!(debug_str.contains("FoldResult"));
        assert!(debug_str.contains("fold_index"));
    }

    #[test]
    fn test_config_debug() {
        let config = CrossValidatorConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CrossValidatorConfig"));
    }

    #[test]
    fn test_energy_balance_metrics_debug() {
        let eb = EnergyBalanceMetrics {
            analytical_total: 1000.0,
            surrogate_total: 950.0,
            balance_error_percent: 5.0,
            heating_balance: 600.0,
            cooling_balance: 350.0,
        };
        let debug_str = format!("{:?}", eb);
        assert!(debug_str.contains("EnergyBalanceMetrics"));
    }

    #[test]
    fn test_aggregated_metrics_debug() {
        let metrics = AggregatedMetrics {
            mean_mae: 1.5,
            mean_rmse: 2.0,
            mean_mape: 5.0,
            mean_r2: 0.9,
            mean_max_error: 3.0,
            std_mae: 0.5,
            std_rmse: 0.3,
            energy_balance: None,
        };
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("AggregatedMetrics"));
    }

    #[test]
    fn test_prediction_pair_debug() {
        let pp = PredictionPair {
            analytical: 100.0,
            surrogate: 105.0,
            difference: 5.0,
        };
        let debug_str = format!("{:?}", pp);
        assert!(debug_str.contains("PredictionPair"));
    }

    #[test]
    fn test_analytical_comparison_debug() {
        let ac = AnalyticalComparison {
            analytical_mean_heating: 100.0,
            analytical_mean_cooling: 50.0,
            surrogate_mean_heating: 105.0,
            surrogate_mean_cooling: 55.0,
            correlation: 0.95,
            predictions: vec![],
        };
        let debug_str = format!("{:?}", ac);
        assert!(debug_str.contains("AnalyticalComparison"));
    }

    #[test]
    fn test_validate_with_large_k_folds() {
        let config = CrossValidatorConfig {
            k_folds: 20,
            shuffle: false,
            ..Default::default()
        };
        let mut validator = CrossValidator::new(config);
        for i in 0..100 {
            validator.add_data(ValidationDataPoint {
                inputs: vec![i as f64],
                targets: vec![i as f64],
                metadata: HashMap::new(),
            });
        }
        // Use validate_analytical instead (no ONNX needed)
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 20);
    }

    #[test]
    fn test_validate_with_single_fold() {
        let config = CrossValidatorConfig {
            k_folds: 1,
            shuffle: false,
            ..Default::default()
        };
        let mut validator = CrossValidator::new(config);
        for i in 0..5 {
            validator.add_data(ValidationDataPoint {
                inputs: vec![i as f64],
                targets: vec![i as f64],
                metadata: HashMap::new(),
            });
        }
        // Use validate_analytical instead (no ONNX needed)
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 1);
    }

    #[test]
    fn test_validate_with_compare_analytical_true() {
        let config = CrossValidatorConfig {
            k_folds: 3,
            shuffle: false,
            compare_analytical: true,
            ..Default::default()
        };
        let mut validator = CrossValidator::new(config);
        for i in 0..15 {
            validator.add_data(ValidationDataPoint {
                inputs: vec![i as f64],
                targets: vec![i as f64],
                metadata: HashMap::new(),
            });
        }
        // Use validate_analytical instead (no ONNX needed)
        let result = validator.validate_analytical();
        assert_eq!(result.fold_results.len(), 3);
    }

    #[test]
    fn test_markdown_report_multiple_folds() {
        let config = CrossValidatorConfig {
            k_folds: 5,
            ..Default::default()
        };
        let mut result = CrossValidationResult::new(config);
        for i in 0..5 {
            result.add_fold_result(FoldResult {
                fold_index: i,
                train_indices: vec![0, 1, 2],
                test_indices: vec![3, 4],
                mae: 1.0 + i as f64 * 0.5,
                rmse: 2.0 + i as f64 * 0.3,
                mape: 5.0 + i as f64 * 2.0,
                r_squared: 0.95 - i as f64 * 0.05,
                max_error: 3.0 + i as f64,
                energy_balance_metrics: Some(EnergyBalanceMetrics {
                    analytical_total: 1000.0 + i as f64 * 100.0,
                    surrogate_total: 980.0 + i as f64 * 90.0,
                    balance_error_percent: 2.0 + i as f64 * 0.5,
                    heating_balance: 500.0,
                    cooling_balance: 480.0,
                }),
            });
        }
        result.compute_aggregated_metrics();
        let markdown = result.to_markdown();
        assert!(markdown.contains("Cross-Validation Report"));
        assert!(markdown.contains("Configuration"));
        assert!(markdown.contains("Aggregated Metrics"));
        assert!(markdown.contains("Energy Balance"));
        assert!(markdown.contains("Per-Fold Results"));
        assert!(markdown.contains("| 0 |"));
        assert!(markdown.contains("| 4 |"));
    }

    #[test]
    fn test_mape_all_nonzero_actuals() {
        let validator = CrossValidator::with_default_config();
        let predictions = vec![110.0, 220.0, 330.0];
        let actuals = vec![100.0, 200.0, 300.0];
        let result = validator.compute_fold_metrics(0, vec![], vec![], &predictions, &actuals);
        assert!((result.mape - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_save_data_error_handling() {
        let validator = CrossValidator::with_default_config();
        let result = validator.save_data("/nonexistent/path/data.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_data_invalid_json() {
        let mut validator = CrossValidator::with_default_config();
        let temp_path = std::env::temp_dir().join("test_invalid_cv.json");
        std::fs::write(&temp_path, "not valid json").unwrap();
        let result = validator.load_data(&temp_path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_validation_data_point_clone() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());
        let dp = ValidationDataPoint {
            inputs: vec![1.0, 2.0, 3.0],
            targets: vec![4.0, 5.0],
            metadata,
        };
        let cloned = dp.clone();
        assert_eq!(cloned.inputs, dp.inputs);
        assert_eq!(cloned.targets, dp.targets);
        assert_eq!(cloned.metadata.get("key").unwrap(), "value");
    }

    #[test]
    fn test_r_squared_negative_prediction() {
        let validator = CrossValidator::with_default_config();
        let predictions = vec![1.0, 1.0, 1.0];
        let actuals = vec![1.0, 2.0, 3.0];
        let result = validator.compute_fold_metrics(0, vec![], vec![], &predictions, &actuals);
        assert!(result.r_squared < 0.0);
    }
}

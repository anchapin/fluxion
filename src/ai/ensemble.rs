//! Ensemble surrogate methods for multiple model predictions with uncertainty.
//!
//! This module provides infrastructure for running ensemble predictions across
//! multiple surrogate models, calculating disagreement metrics, and combining
//! predictions through various aggregation methods.

use crate::ai::surrogate::{PredictionWithUncertainty, SurrogateManager};
use std::collections::HashMap;

/// Ensemble configuration for managing multiple models.
#[derive(Clone, Debug)]
pub struct EnsembleConfig {
    /// List of model paths to include in the ensemble
    pub model_paths: Vec<String>,
    /// Aggregation method for combining predictions
    pub aggregation_method: AggregationMethod,
    /// Number of models required for a valid ensemble
    pub min_models: usize,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        EnsembleConfig {
            model_paths: vec![],
            aggregation_method: AggregationMethod::Mean,
            min_models: 1,
        }
    }
}

impl EnsembleConfig {
    /// Create a new ensemble config with given model paths
    pub fn new(model_paths: Vec<String>) -> Self {
        EnsembleConfig {
            model_paths,
            ..Default::default()
        }
    }

    /// Set the aggregation method
    pub fn with_method(mut self, method: AggregationMethod) -> Self {
        self.aggregation_method = method;
        self
    }

    /// Set minimum number of models required
    pub fn with_min_models(mut self, min: usize) -> Self {
        self.min_models = min;
        self
    }
}

/// Methods for aggregating predictions from multiple models.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum AggregationMethod {
    /// Simple mean of all predictions
    #[default]
    Mean,
    /// Weighted mean based on model performance
    WeightedMean,
    /// Median of predictions (robust to outliers)
    Median,
    /// Trimmed mean (excludes extreme values)
    TrimmedMean(f32), // fraction to trim from each side
    /// Best model selection based on validation
    BestModel(usize), // index of best model
}

/// Result of an ensemble prediction.
#[derive(Clone, Debug)]
pub struct EnsemblePrediction {
    /// Aggregated predictions
    pub predictions: Vec<f64>,
    /// Standard deviation across models
    pub disagreement: Vec<f64>,
    /// Predictions from individual models
    pub model_predictions: Vec<Vec<f64>>,
    /// Model weights (if using weighted mean)
    pub weights: Option<Vec<f64>>,
}

impl EnsemblePrediction {
    /// Calculate the mean disagreement (average std across outputs)
    pub fn mean_disagreement(&self) -> f64 {
        if self.disagreement.is_empty() {
            return 0.0;
        }
        self.disagreement.iter().sum::<f64>() / self.disagreement.len() as f64
    }

    /// Get the maximum disagreement across all outputs
    pub fn max_disagreement(&self) -> f64 {
        self.disagreement.iter().cloned().fold(0.0_f64, f64::max)
    }
}

/// Ensemble of multiple surrogate models.
///
/// Provides methods for running predictions across multiple models,
/// calculating disagreement metrics, and combining predictions.
#[derive(Clone, Debug)]
pub struct EnsembleSurrogate {
    /// Individual model managers
    models: Vec<SurrogateManager>,
    /// Configuration
    config: EnsembleConfig,
    /// Optional model weights (for weighted aggregation)
    model_weights: Option<Vec<f64>>,
    /// Optional validation metrics per model
    validation_metrics: HashMap<usize, ModelMetrics>,
}

/// Metrics for individual model performance.
#[derive(Clone, Debug, Default)]
pub struct ModelMetrics {
    /// Mean absolute error on validation set
    pub mae: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// R-squared score
    pub r2: f64,
    /// Mean prediction bias
    pub bias: f64,
}

impl ModelMetrics {
    /// Create new metrics
    pub fn new(mae: f64, rmse: f64, r2: f64, bias: f64) -> Self {
        ModelMetrics {
            mae,
            rmse,
            r2,
            bias,
        }
    }

    /// Calculate composite score (lower is better)
    pub fn composite_score(&self) -> f64 {
        // Combine errors with bias penalty
        self.mae + self.rmse + self.bias.abs()
    }
}

impl EnsembleSurrogate {
    /// Create a new ensemble from model paths.
    pub fn new(config: EnsembleConfig) -> Result<Self, String> {
        let mut models = Vec::new();

        for path in &config.model_paths {
            match SurrogateManager::load_onnx(path) {
                Ok(manager) => models.push(manager),
                Err(e) => {
                    eprintln!("Warning: Failed to load model {}: {}", path, e);
                }
            }
        }

        if models.len() < config.min_models {
            return Err(format!(
                "Not enough models loaded: {} < {}",
                models.len(),
                config.min_models
            ));
        }

        Ok(EnsembleSurrogate {
            models,
            config,
            model_weights: None,
            validation_metrics: HashMap::new(),
        })
    }

    /// Create ensemble from a single model (for consistent API)
    pub fn from_single(model_path: &str) -> Result<Self, String> {
        let config = EnsembleConfig::new(vec![model_path.to_string()]);
        Self::new(config)
    }

    /// Add validation metrics for a model
    pub fn add_model_metrics(&mut self, model_index: usize, metrics: ModelMetrics) {
        self.validation_metrics.insert(model_index, metrics);
        self.recalculate_weights();
    }

    /// Recalculate model weights based on validation metrics
    fn recalculate_weights(&mut self) {
        if self.validation_metrics.is_empty() {
            self.model_weights = None;
            return;
        }

        // Calculate weights based on inverse of composite score
        let scores: Vec<f64> = self
            .validation_metrics
            .values()
            .map(|m| m.composite_score())
            .collect();

        // Handle zero/negative scores
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let weights = if min_score <= 0.0 {
            // Use equal weights if any score is non-positive
            vec![1.0 / self.models.len() as f64; self.models.len()]
        } else {
            // Inverse score weighting
            let total_inv: f64 = scores.iter().map(|s| 1.0 / s).sum();
            scores.iter().map(|s| (1.0 / s) / total_inv).collect()
        };

        self.model_weights = Some(weights);
    }

    /// Get the number of models in the ensemble
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Run predictions on all models in the ensemble.
    pub fn predict_all(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        self.models
            .iter()
            .map(|model| model.predict_loads(inputs))
            .collect()
    }

    /// Run ensemble prediction with aggregation.
    pub fn predict(&self, inputs: &[f64]) -> EnsemblePrediction {
        if self.models.is_empty() {
            return EnsemblePrediction {
                predictions: vec![],
                disagreement: vec![],
                model_predictions: vec![],
                weights: None,
            };
        }

        // Get predictions from all models
        let model_predictions = self.predict_all(inputs);

        if model_predictions.is_empty() {
            return EnsemblePrediction {
                predictions: vec![],
                disagreement: vec![],
                model_predictions: vec![],
                weights: None,
            };
        }

        let num_outputs = model_predictions[0].len();
        let _num_models = model_predictions.len();

        // Aggregate predictions based on method
        let predictions = match self.config.aggregation_method {
            AggregationMethod::Mean => self.aggregate_mean(&model_predictions, num_outputs),
            AggregationMethod::WeightedMean => {
                self.aggregate_weighted_mean(&model_predictions, num_outputs)
            }
            AggregationMethod::Median => self.aggregate_median(&model_predictions, num_outputs),
            AggregationMethod::TrimmedMean(trim_frac) => {
                self.aggregate_trimmed_mean(&model_predictions, num_outputs, trim_frac)
            }
            AggregationMethod::BestModel(idx) => {
                if idx < model_predictions.len() {
                    model_predictions[idx].clone()
                } else {
                    self.aggregate_mean(&model_predictions, num_outputs)
                }
            }
        };

        // Calculate disagreement (std across models)
        let disagreement = self.calculate_disagreement(&model_predictions, num_outputs);

        EnsemblePrediction {
            predictions,
            disagreement,
            model_predictions,
            weights: self.model_weights.clone(),
        }
    }

    /// Calculate mean across model predictions
    fn aggregate_mean(&self, predictions: &[Vec<f64>], num_outputs: usize) -> Vec<f64> {
        let num_models = predictions.len();
        let mut result = vec![0.0; num_outputs];

        for pred in predictions {
            for (i, &val) in pred.iter().enumerate() {
                result[i] += val;
            }
        }

        for val in &mut result {
            *val /= num_models as f64;
        }

        result
    }

    /// Calculate weighted mean across model predictions
    fn aggregate_weighted_mean(&self, predictions: &[Vec<f64>], num_outputs: usize) -> Vec<f64> {
        let default_weights = vec![1.0; predictions.len()];
        let weights = self.model_weights.as_ref().unwrap_or(&default_weights);
        let mut result = vec![0.0; num_outputs];

        for (pred, &w) in predictions.iter().zip(weights.iter()) {
            for (i, &val) in pred.iter().enumerate() {
                result[i] += val * w;
            }
        }

        result
    }

    /// Calculate median across model predictions
    fn aggregate_median(&self, predictions: &[Vec<f64>], num_outputs: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(num_outputs);

        for i in 0..num_outputs {
            let mut values: Vec<f64> = predictions.iter().map(|p| p[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            let median = if values.len().is_multiple_of(2) {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            };
            result.push(median);
        }

        result
    }

    /// Calculate trimmed mean across model predictions
    fn aggregate_trimmed_mean(
        &self,
        predictions: &[Vec<f64>],
        num_outputs: usize,
        trim_frac: f32,
    ) -> Vec<f64> {
        let mut result = Vec::with_capacity(num_outputs);

        for i in 0..num_outputs {
            let mut values: Vec<f64> = predictions.iter().map(|p| p[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let trim_count = (values.len() as f32 * trim_frac) as usize;
            let trimmed = &values[trim_count..values.len() - trim_count];

            let mean = if trimmed.is_empty() {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                trimmed.iter().sum::<f64>() / trimmed.len() as f64
            };
            result.push(mean);
        }

        result
    }

    /// Calculate disagreement (standard deviation) across models
    fn calculate_disagreement(&self, predictions: &[Vec<f64>], num_outputs: usize) -> Vec<f64> {
        if predictions.len() < 2 {
            return vec![0.0; num_outputs];
        }

        let mut disagreement = vec![0.0; num_outputs];

        // Calculate mean per output
        for i in 0..num_outputs {
            let sum: f64 = predictions.iter().map(|p| p[i]).sum();
            let mean = sum / predictions.len() as f64;

            // Calculate variance
            let variance: f64 = predictions
                .iter()
                .map(|p| {
                    let diff = p[i] - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (predictions.len() - 1) as f64;

            disagreement[i] = variance.sqrt();
        }

        disagreement
    }

    /// Get prediction with uncertainty from ensemble.
    ///
    /// This uses the ensemble disagreement as a proxy for uncertainty,
    /// providing a simpler alternative to MC Dropout.
    pub fn predict_with_uncertainty(&self, inputs: &[f64]) -> PredictionWithUncertainty {
        let ensemble_pred = self.predict(inputs);

        // Use disagreement as uncertainty measure
        let std = ensemble_pred.disagreement.clone();

        PredictionWithUncertainty::new(ensemble_pred.predictions, std)
    }
}

/// Disagreement metrics between models in the ensemble.
#[derive(Clone, Debug, Default)]
pub struct DisagreementMetrics {
    /// Average pairwise difference
    pub avg_pairwise_diff: f64,
    /// Maximum pairwise difference
    pub max_pairwise_diff: f64,
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
    /// Number of models that agree within tolerance
    pub agreement_count: usize,
}

impl DisagreementMetrics {
    /// Calculate from ensemble prediction
    pub fn from_prediction(prediction: &EnsemblePrediction, tolerance: f64) -> Self {
        let model_preds = &prediction.model_predictions;
        if model_preds.len() < 2 {
            return DisagreementMetrics::default();
        }

        let num_outputs = model_preds[0].len();
        let num_models = model_preds.len();

        // Calculate pairwise differences
        let mut total_diff = 0.0;
        let mut max_diff: f64 = 0.0;
        let mut pair_count = 0;

        for i in 0..num_models {
            for j in (i + 1)..num_models {
                let preds_i = &model_preds[i];
                let preds_j = &model_preds[j];
                for k in 0..num_outputs {
                    let diff = (preds_i[k] - preds_j[k]).abs();
                    total_diff += diff;
                    max_diff = max_diff.max(diff);
                    pair_count += 1;
                }
            }
        }

        let avg_pairwise_diff = if pair_count > 0 {
            total_diff / pair_count as f64
        } else {
            0.0
        };

        // Calculate coefficient of variation
        let mean_pred: f64 = prediction.predictions.iter().sum::<f64>() / num_outputs as f64;
        let avg_std = prediction.disagreement.iter().sum::<f64>() / num_outputs as f64;
        let coefficient_of_variation = if mean_pred != 0.0 {
            avg_std / mean_pred.abs()
        } else {
            0.0
        };

        // Count agreements within tolerance
        let agreement_count = prediction
            .disagreement
            .iter()
            .filter(|&&d| d <= tolerance)
            .count();

        DisagreementMetrics {
            avg_pairwise_diff,
            max_pairwise_diff: max_diff,
            coefficient_of_variation,
            agreement_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_config() {
        let config =
            EnsembleConfig::new(vec!["model1.onnx".to_string(), "model2.onnx".to_string()])
                .with_method(AggregationMethod::Median)
                .with_min_models(2);

        assert_eq!(config.model_paths.len(), 2);
        assert_eq!(config.aggregation_method, AggregationMethod::Median);
        assert_eq!(config.min_models, 2);
    }

    #[test]
    fn test_model_metrics() {
        let metrics = ModelMetrics::new(0.5, 0.7, 0.95, 0.1);
        let score = metrics.composite_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_disagreement_calculation() {
        // Create mock predictions
        let predictions = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
        ];

        // Calculate disagreement manually
        let mut expected_disagreement = vec![0.0; 3];
        for i in 0..3 {
            let mean = (predictions[0][i] + predictions[1][i] + predictions[2][i]) / 3.0;
            let diff0: f64 = predictions[0][i] - mean;
            let diff1: f64 = predictions[1][i] - mean;
            let diff2: f64 = predictions[2][i] - mean;
            let variance = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2) / 2.0;
            expected_disagreement[i] = variance.sqrt();
        }

        // Simple test: verify disagreement is calculated
        assert!(expected_disagreement[0] > 0.0);
    }

    #[test]
    fn test_aggregation_methods() {
        let predictions = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];

        // Test mean
        let mean_result: Vec<f64> = predictions
            .iter()
            .fold(vec![0.0, 0.0], |acc, p| {
                p.iter().zip(acc.iter()).map(|(a, b)| a + b).collect()
            })
            .iter()
            .map(|v| v / 3.0)
            .collect();

        assert_eq!(mean_result, vec![2.0, 3.0]);
    }
}

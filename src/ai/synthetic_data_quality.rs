//! Synthetic data quality validation for ML surrogate training data (Issue #1780).
//!
//! Gates dataset quality before training consumes it. Performs statistical checks:
//! - NaN/Inf detection
//! - Distribution bounds validation
//! - Outlier flagging
//! - Drift detection vs reference data
//! - Per-shard summary statistics

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct DataShardStats {
    pub shard_id: String,
    pub num_samples: usize,
    pub nan_count: HashMap<String, usize>,
    pub inf_count: HashMap<String, usize>,
    pub min_values: HashMap<String, f64>,
    pub max_values: HashMap<String, f64>,
    pub mean_values: HashMap<String, f64>,
    pub std_values: HashMap<String, f64>,
    pub outlier_count: HashMap<String, usize>,
    pub out_of_bounds_count: HashMap<String, usize>,
}

impl DataShardStats {
    pub fn new(shard_id: &str) -> Self {
        DataShardStats {
            shard_id: shard_id.to_string(),
            num_samples: 0,
            nan_count: HashMap::new(),
            inf_count: HashMap::new(),
            min_values: HashMap::new(),
            max_values: HashMap::new(),
            mean_values: HashMap::new(),
            std_values: HashMap::new(),
            outlier_count: HashMap::new(),
            out_of_bounds_count: HashMap::new(),
        }
    }

    pub fn merge(&mut self, other: &DataShardStats) {
        self.num_samples += other.num_samples;
        for (k, v) in &other.nan_count {
            *self.nan_count.entry(k.clone()).or_insert(0) += v;
        }
        for (k, v) in &other.inf_count {
            *self.inf_count.entry(k.clone()).or_insert(0) += v;
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldBounds {
    pub min: f64,
    pub max: f64,
}

impl FieldBounds {
    pub fn new(min: f64, max: f64) -> Self {
        FieldBounds { min, max }
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }
}

#[derive(Clone, Debug, Default)]
pub struct ValidationConfig {
    pub outlier_z_threshold: f64,
    pub drift_reference_window: usize,
    pub min_coverage_per_field: f64,
    pub fields: HashMap<String, FieldBounds>,
}

impl ValidationConfig {
    pub fn standard() -> Self {
        let mut fields = HashMap::new();
        fields.insert("exterior_temp".to_string(), FieldBounds::new(-50.0, 60.0));
        fields.insert("zone_temp".to_string(), FieldBounds::new(10.0, 40.0));
        fields.insert("solar_rad".to_string(), FieldBounds::new(0.0, 1200.0));
        fields.insert("humidity".to_string(), FieldBounds::new(0.0, 100.0));
        fields.insert("occupancy".to_string(), FieldBounds::new(0.0, 10.0));
        ValidationConfig {
            outlier_z_threshold: 3.0,
            drift_reference_window: 100,
            min_coverage_per_field: 0.95,
            fields,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QualityGrade {
    Pass,
    Warn,
    Fail,
}

#[derive(Clone, Debug)]
pub struct QualityCheckResult {
    pub grade: QualityGrade,
    pub nan_fields: Vec<String>,
    pub inf_fields: Vec<String>,
    pub out_of_bounds_fields: Vec<String>,
    pub outlier_fields: Vec<String>,
    pub drift_detected: bool,
    pub messages: Vec<String>,
}

impl QualityCheckResult {
    pub fn pass() -> Self {
        QualityCheckResult {
            grade: QualityGrade::Pass,
            nan_fields: Vec::new(),
            inf_fields: Vec::new(),
            out_of_bounds_fields: Vec::new(),
            outlier_fields: Vec::new(),
            drift_detected: false,
            messages: Vec::new(),
        }
    }

    pub fn is_pass(&self) -> bool {
        self.grade == QualityGrade::Pass
    }
}

pub struct SyntheticDataValidator {
    config: ValidationConfig,
    reference_stats: Option<DataShardStats>,
}

impl SyntheticDataValidator {
    pub fn new(config: ValidationConfig) -> Self {
        SyntheticDataValidator {
            config,
            reference_stats: None,
        }
    }

    pub fn with_reference_stats(mut self, stats: DataShardStats) -> Self {
        self.reference_stats = Some(stats);
        self
    }

    pub fn set_reference_stats(&mut self, stats: DataShardStats) {
        self.reference_stats = Some(stats);
    }

    fn compute_stats_for_field(values: &[f64]) -> (f64, f64, f64, f64) {
        if values.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }
        let n = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let mean = sum / n;
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max, mean, std)
    }

    pub fn compute_shard_stats(
        &self,
        shard_id: &str,
        field_data: &HashMap<String, Vec<f64>>,
    ) -> DataShardStats {
        let mut stats = DataShardStats::new(shard_id);
        stats.num_samples = field_data.values().next().map(|v| v.len()).unwrap_or(0);

        for (field, values) in field_data {
            let (min, max, mean, std) = Self::compute_stats_for_field(values);
            let nan_count = values.iter().filter(|v| v.is_nan()).count();
            let inf_count = values.iter().filter(|v| v.is_infinite()).count();
            stats.nan_count.insert(field.clone(), nan_count);
            stats.inf_count.insert(field.clone(), inf_count);
            stats.min_values.insert(field.clone(), min);
            stats.max_values.insert(field.clone(), max);
            stats.mean_values.insert(field.clone(), mean);
            stats.std_values.insert(field.clone(), std);

            if let Some(bounds) = self.config.fields.get(field) {
                let oob = values.iter().filter(|v| !bounds.contains(**v)).count();
                stats.out_of_bounds_count.insert(field.clone(), oob);
            }

            if std > 0.0 {
                let outlier_count = values
                    .iter()
                    .filter(|v| {
                        let z = (**v - mean).abs() / std;
                        z > self.config.outlier_z_threshold
                    })
                    .count();
                stats.outlier_count.insert(field.clone(), outlier_count);
            }
        }

        stats
    }

    pub fn validate_shard(&self, stats: &DataShardStats) -> QualityCheckResult {
        let mut result = QualityCheckResult::pass();
        let total = stats.num_samples.max(1);

        for (field, &nan_count) in &stats.nan_count {
            if nan_count > 0 {
                result
                    .nan_fields
                    .push(format!("{} ({} NaN/{})", field, nan_count, total));
                result
                    .messages
                    .push(format!("field '{}' has {} NaN values", field, nan_count));
            }
        }

        for (field, &inf_count) in &stats.inf_count {
            if inf_count > 0 {
                result
                    .inf_fields
                    .push(format!("{} ({} Inf/{})", field, inf_count, total));
                result
                    .messages
                    .push(format!("field '{}' has {} Inf values", field, inf_count));
            }
        }

        for (field, &oob_count) in &stats.out_of_bounds_count {
            let ratio = oob_count as f64 / total as f64;
            if ratio > 1.0 - self.config.min_coverage_per_field {
                result.out_of_bounds_fields.push(field.clone());
                result.messages.push(format!(
                    "field '{}' has {:.1}% out-of-bounds values",
                    field,
                    ratio * 100.0
                ));
            }
        }

        for (field, &outlier_count) in &stats.outlier_count {
            let ratio = outlier_count as f64 / total as f64;
            if ratio > 0.05 {
                result.outlier_fields.push(field.clone());
                result.messages.push(format!(
                    "field '{}' has {:.1}% outlier values",
                    field,
                    ratio * 100.0
                ));
            }
        }

        if let Some(ref reference) = self.reference_stats {
            self.detect_drift(stats, reference, &mut result);
        }

        if !result.nan_fields.is_empty() || !result.inf_fields.is_empty() {
            result.grade = QualityGrade::Fail;
        } else if !result.out_of_bounds_fields.is_empty()
            || !result.outlier_fields.is_empty()
            || result.drift_detected
        {
            result.grade = QualityGrade::Warn;
        }

        result
    }

    fn detect_drift(
        &self,
        current: &DataShardStats,
        reference: &DataShardStats,
        result: &mut QualityCheckResult,
    ) {
        for field in current.mean_values.keys() {
            let Some(&current_mean) = current.mean_values.get(field) else {
                continue;
            };
            let Some(&ref_mean) = reference.mean_values.get(field) else {
                continue;
            };
            let Some(&ref_std) = reference.std_values.get(field) else {
                continue;
            };

            if ref_std > 0.0 {
                let z_drift = (current_mean - ref_mean).abs() / ref_std;
                if z_drift > 2.0 {
                    result.drift_detected = true;
                    result.messages.push(format!(
                        "field '{}' shows drift: z={:.2} vs reference",
                        field, z_drift
                    ));
                }
            }
        }
    }

    pub fn validate_batch(
        &self,
        shards: &[DataShardStats],
        shard_ids: &[String],
    ) -> HashMap<String, QualityCheckResult> {
        let mut results = HashMap::new();
        for (shard_stats, shard_id) in shards.iter().zip(shard_ids.iter()) {
            let result = self.validate_shard(shard_stats);
            results.insert(shard_id.clone(), result);
        }
        results
    }

    pub fn aggregate_stats(shards: &[DataShardStats]) -> DataShardStats {
        let mut aggregated = DataShardStats::new("aggregated");
        if shards.is_empty() {
            return aggregated;
        }
        aggregated.num_samples = shards.iter().map(|s| s.num_samples).sum();
        let mut field_means: HashMap<String, Vec<f64>> = HashMap::new();
        let mut field_stds: HashMap<String, Vec<f64>> = HashMap::new();
        for shard in shards {
            for (field, &val) in &shard.mean_values {
                field_means.entry(field.clone()).or_default().push(val);
                if let Some(&std_val) = shard.std_values.get(field) {
                    field_stds.entry(field.clone()).or_default().push(std_val);
                }
            }
        }
        for (field, means) in &field_means {
            let (min, max, mean, _std) = Self::compute_stats_for_field(means);
            aggregated.min_values.insert(field.clone(), min);
            aggregated.max_values.insert(field.clone(), max);
            aggregated.mean_values.insert(field.clone(), mean);
        }
        for (field, stds) in &field_stds {
            let pooled_std =
                (stds.iter().map(|s| s.powi(2)).sum::<f64>() / stds.len() as f64).sqrt();
            aggregated.std_values.insert(field.clone(), pooled_std);
        }
        for shard in shards {
            for (field, &nan) in &shard.nan_count {
                *aggregated.nan_count.entry(field.clone()).or_insert(0) += nan;
            }
            for (field, &inf) in &shard.inf_count {
                *aggregated.inf_count.entry(field.clone()).or_insert(0) += inf;
            }
            for (field, &oob) in &shard.out_of_bounds_count {
                *aggregated
                    .out_of_bounds_count
                    .entry(field.clone())
                    .or_insert(0) += oob;
            }
            for (field, &outlier) in &shard.outlier_count {
                *aggregated.outlier_count.entry(field.clone()).or_insert(0) += outlier;
            }
        }
        aggregated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field_data() -> HashMap<String, Vec<f64>> {
        let mut data = HashMap::new();
        data.insert(
            "exterior_temp".to_string(),
            vec![20.0, 22.0, 21.0, 19.0, 23.0],
        );
        data.insert("zone_temp".to_string(), vec![22.0, 23.0, 21.5, 22.5, 23.5]);
        data.insert(
            "solar_rad".to_string(),
            vec![500.0, 600.0, 450.0, 550.0, 580.0],
        );
        data
    }

    #[test]
    fn test_field_bounds_contains() {
        let bounds = FieldBounds::new(0.0, 100.0);
        assert!(bounds.contains(50.0));
        assert!(bounds.contains(0.0));
        assert!(bounds.contains(100.0));
        assert!(!bounds.contains(-0.1));
        assert!(!bounds.contains(100.1));
    }

    #[test]
    fn test_compute_stats_for_field_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std) = SyntheticDataValidator::compute_stats_for_field(&values);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert!((mean - 3.0).abs() < 1e-10);
        assert!((std - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_for_field_empty() {
        let values: Vec<f64> = vec![];
        let (min, max, mean, std) = SyntheticDataValidator::compute_stats_for_field(&values);
        assert!(min.is_nan());
        assert!(max.is_nan());
        assert!(mean.is_nan());
        assert!(std.is_nan());
    }

    #[test]
    fn test_compute_shard_stats_basic() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let field_data = make_field_data();
        let stats = validator.compute_shard_stats("shard_1", &field_data);

        assert_eq!(stats.shard_id, "shard_1");
        assert_eq!(stats.num_samples, 5);
        assert_eq!(stats.nan_count.get("exterior_temp"), Some(&0));
        assert_eq!(stats.inf_count.get("solar_rad"), Some(&0));
        assert!(stats.mean_values.contains_key("exterior_temp"));
        assert!(stats.min_values.contains_key("solar_rad"));
        assert!(stats.max_values.contains_key("solar_rad"));
    }

    #[test]
    fn test_compute_shard_stats_detects_nan() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let mut field_data = make_field_data();
        field_data.insert("exterior_temp".to_string(), vec![20.0, f64::NAN, 22.0]);

        let stats = validator.compute_shard_stats("shard_nan", &field_data);
        assert_eq!(stats.nan_count.get("exterior_temp"), Some(&1));
    }

    #[test]
    fn test_compute_shard_stats_detects_inf() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let mut field_data = make_field_data();
        field_data.insert("solar_rad".to_string(), vec![f64::INFINITY, 600.0, 450.0]);

        let stats = validator.compute_shard_stats("shard_inf", &field_data);
        assert_eq!(stats.inf_count.get("solar_rad"), Some(&1));
    }

    #[test]
    fn test_compute_shard_stats_detects_out_of_bounds() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let mut field_data = make_field_data();
        field_data.insert("exterior_temp".to_string(), vec![20.0, 22.0, 100.0]);

        let stats = validator.compute_shard_stats("shard_oob", &field_data);
        assert_eq!(stats.out_of_bounds_count.get("exterior_temp"), Some(&1));
    }

    #[test]
    fn test_validate_shard_passes_clean_data() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let field_data = make_field_data();
        let stats = validator.compute_shard_stats("shard_1", &field_data);
        let result = validator.validate_shard(&stats);

        assert!(result.is_pass());
        assert!(result.nan_fields.is_empty());
        assert!(result.inf_fields.is_empty());
        assert!(result.out_of_bounds_fields.is_empty());
        assert!(result.outlier_fields.is_empty());
        assert!(!result.drift_detected);
    }

    #[test]
    fn test_validate_shard_fails_nan() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let mut field_data = make_field_data();
        field_data.insert("exterior_temp".to_string(), vec![f64::NAN, 22.0, 21.0]);

        let stats = validator.compute_shard_stats("shard_nan", &field_data);
        let result = validator.validate_shard(&stats);

        assert_eq!(result.grade, QualityGrade::Fail);
        assert!(!result.nan_fields.is_empty());
    }

    #[test]
    fn test_validate_shard_fails_inf() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let mut field_data = make_field_data();
        field_data.insert(
            "solar_rad".to_string(),
            vec![f64::NEG_INFINITY, 600.0, 450.0],
        );

        let stats = validator.compute_shard_stats("shard_inf", &field_data);
        let result = validator.validate_shard(&stats);

        assert_eq!(result.grade, QualityGrade::Fail);
        assert!(!result.inf_fields.is_empty());
    }

    #[test]
    fn test_validate_shard_warns_out_of_bounds() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let mut field_data = make_field_data();
        field_data.insert("exterior_temp".to_string(), vec![20.0, 22.0, 80.0]);

        let stats = validator.compute_shard_stats("shard_oob", &field_data);
        let result = validator.validate_shard(&stats);

        assert_eq!(result.grade, QualityGrade::Warn);
        assert!(!result.out_of_bounds_fields.is_empty());
    }

    #[test]
    fn test_validate_shard_warns_outliers() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        let mut field_data = make_field_data();
        field_data.insert("exterior_temp".to_string(), vec![20.0; 100]);
        field_data
            .entry("exterior_temp".to_string())
            .or_default()
            .push(100.0);

        let stats = validator.compute_shard_stats("shard_outlier", &field_data);
        let result = validator.validate_shard(&stats);

        assert_eq!(result.grade, QualityGrade::Warn);
    }

    #[test]
    fn test_drift_detection() {
        let config = ValidationConfig::standard();
        let mut validator = SyntheticDataValidator::new(config);

        let mut ref_data = make_field_data();
        ref_data.insert("exterior_temp".to_string(), vec![20.0; 100]);
        let ref_stats = validator.compute_shard_stats("reference", &ref_data);
        validator.set_reference_stats(ref_stats);

        let mut new_data = HashMap::new();
        new_data.insert("exterior_temp".to_string(), vec![50.0; 100]);
        new_data.insert("zone_temp".to_string(), vec![22.0; 100]);
        new_data.insert("solar_rad".to_string(), vec![500.0; 100]);

        let stats = validator.compute_shard_stats("new_data", &new_data);
        let result = validator.validate_shard(&stats);

        assert!(result.drift_detected);
        assert_eq!(result.grade, QualityGrade::Warn);
    }

    #[test]
    fn test_validate_batch() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);

        let shard_1 = validator.compute_shard_stats("shard_1", &make_field_data());
        let shard_2 = validator.compute_shard_stats("shard_2", &make_field_data());

        let results = validator.validate_batch(
            &[shard_1, shard_2],
            &["shard_1".to_string(), "shard_2".to_string()],
        );

        assert_eq!(results.len(), 2);
        assert!(results.get("shard_1").unwrap().is_pass());
        assert!(results.get("shard_2").unwrap().is_pass());
    }

    #[test]
    fn test_aggregate_stats() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);

        let field_data_1 = make_field_data();
        let field_data_2 = make_field_data();

        let stats_1 = validator.compute_shard_stats("s1", &field_data_1);
        let stats_2 = validator.compute_shard_stats("s2", &field_data_2);

        let aggregated = SyntheticDataValidator::aggregate_stats(&[stats_1, stats_2]);
        assert_eq!(aggregated.num_samples, 10);
        assert_eq!(aggregated.nan_count.get("exterior_temp"), Some(&0));
    }

    #[test]
    fn test_aggregate_stats_empty() {
        let aggregated = SyntheticDataValidator::aggregate_stats(&[]);
        assert_eq!(aggregated.num_samples, 0);
        assert!(aggregated.shard_id == "aggregated");
    }

    #[test]
    fn test_quality_check_result_pass() {
        let result = QualityCheckResult::pass();
        assert!(result.is_pass());
        assert_eq!(result.grade, QualityGrade::Pass);
    }

    #[test]
    fn test_validation_config_standard() {
        let config = ValidationConfig::standard();
        assert_eq!(config.outlier_z_threshold, 3.0);
        assert_eq!(config.drift_reference_window, 100);
        assert_eq!(config.min_coverage_per_field, 0.95);
        assert!(config.fields.contains_key("exterior_temp"));
        assert!(config.fields.contains_key("solar_rad"));
        assert_eq!(
            config.fields.get("exterior_temp"),
            Some(&FieldBounds::new(-50.0, 60.0))
        );
    }

    #[test]
    fn test_synthetic_data_validator_new() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);
        assert!(validator.reference_stats.is_none());
    }

    #[test]
    fn test_synthetic_data_validator_with_reference() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config.clone());
        let field_data = make_field_data();
        let stats = validator.compute_shard_stats("ref", &field_data);
        let validator_with_ref = validator.with_reference_stats(stats);
        assert!(validator_with_ref.reference_stats.is_some());
    }

    #[test]
    fn test_outlier_z_threshold_boundary() {
        let config = ValidationConfig::standard();
        let validator = SyntheticDataValidator::new(config);

        let mut field_data = make_field_data();
        field_data.insert("exterior_temp".to_string(), vec![20.0; 100]);
        let stats = validator.compute_shard_stats("shard", &field_data);
        let result = validator.validate_shard(&stats);
        assert_eq!(result.outlier_fields.len(), 0);
    }
}

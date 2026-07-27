//! Integration tests for synthetic data quality validation (Issue #1780).
//!
//! Tests the `SyntheticDataValidator` from `crate::ai::synthetic_data_quality`
//! against the acceptance criteria:
//! - Automated checks: no NaN/inf, expected distributions, coverage of sampled parameter ranges.
//! - Per-shard summary stats emitted alongside data.
//! - CI gate fails on quality regression.
//!
//! ## CI Gate Behavior
//!
//! The quality gate operates in two modes:
//! 1. **Fail on NaN/Inf** — any NaN or Inf value immediately sets grade to `Fail`.
//! 2. **Warn on quality issues** — out-of-bounds, outliers, or drift detected set grade to `Warn`.
//! 3. **Pass** — all checks clean sets grade to `Pass`.
//!
//! CI gate is considered failing when grade is `Fail` (quality regression detected).

use fluxion::ai::synthetic_data_quality::{
    DataShardStats, FieldBounds, QualityCheckResult, QualityGrade, SyntheticDataValidator,
    ValidationConfig,
};
use std::collections::HashMap;

fn make_standard_config() -> ValidationConfig {
    ValidationConfig::standard()
}

fn make_clean_field_data() -> HashMap<String, Vec<f64>> {
    let mut data = HashMap::new();
    data.insert(
        "exterior_temp".to_string(),
        vec![20.0, 22.0, 21.0, 19.0, 23.0, 18.5, 21.5],
    );
    data.insert(
        "zone_temp".to_string(),
        vec![22.0, 23.0, 21.5, 22.5, 23.5, 22.0, 21.8],
    );
    data.insert(
        "solar_rad".to_string(),
        vec![500.0, 600.0, 450.0, 550.0, 580.0, 420.0, 510.0],
    );
    data.insert(
        "humidity".to_string(),
        vec![50.0, 55.0, 48.0, 52.0, 58.0, 45.0, 53.0],
    );
    data.insert(
        "occupancy".to_string(),
        vec![0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.35],
    );
    data
}

fn make_nan_field_data() -> HashMap<String, Vec<f64>> {
    let mut data = make_clean_field_data();
    data.insert(
        "exterior_temp".to_string(),
        vec![20.0, f64::NAN, 22.0, 21.0, f64::NAN],
    );
    data
}

fn make_inf_field_data() -> HashMap<String, Vec<f64>> {
    let mut data = make_clean_field_data();
    data.insert(
        "solar_rad".to_string(),
        vec![f64::INFINITY, 600.0, 450.0, f64::NEG_INFINITY, 550.0],
    );
    data
}

fn make_out_of_bounds_field_data() -> HashMap<String, Vec<f64>> {
    let mut data = make_clean_field_data();
    data.insert(
        "exterior_temp".to_string(),
        vec![20.0, 22.0, 80.0, 21.0, 19.0],
    );
    data.insert(
        "solar_rad".to_string(),
        vec![500.0, 600.0, 1500.0, 550.0, 580.0],
    );
    data
}

fn make_drift_field_data() -> HashMap<String, Vec<f64>> {
    let mut data = HashMap::new();
    data.insert("exterior_temp".to_string(), vec![50.0; 50]);
    data.insert("zone_temp".to_string(), vec![22.0; 50]);
    data.insert("solar_rad".to_string(), vec![500.0; 50]);
    data.insert("humidity".to_string(), vec![50.0; 50]);
    data.insert("occupancy".to_string(), vec![0.3; 50]);
    data
}

// ---------------------------------------------------------------------------------------------------
// AC1: Automated checks — no NaN/Inf, expected distributions, coverage of sampled parameter ranges
// ---------------------------------------------------------------------------------------------------

#[test]
fn test_ci_gate_fails_on_nan() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_nan_field_data();
    let stats = validator.compute_shard_stats("nan_shard", &field_data);
    let result = validator.validate_shard(&stats);

    assert_eq!(
        result.grade,
        QualityGrade::Fail,
        "NaN values should cause CI gate to fail"
    );
    assert!(
        !result.nan_fields.is_empty(),
        "NaN fields should be reported"
    );
}

#[test]
fn test_ci_gate_fails_on_inf() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_inf_field_data();
    let stats = validator.compute_shard_stats("inf_shard", &field_data);
    let result = validator.validate_shard(&stats);

    assert_eq!(
        result.grade,
        QualityGrade::Fail,
        "Inf values should cause CI gate to fail"
    );
    assert!(
        !result.inf_fields.is_empty(),
        "Inf fields should be reported"
    );
}

#[test]
fn test_no_nan_inf_in_clean_data() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_clean_field_data();
    let stats = validator.compute_shard_stats("clean_shard", &field_data);
    let result = validator.validate_shard(&stats);

    assert_eq!(
        result.grade,
        QualityGrade::Pass,
        "Clean data should pass CI gate"
    );
    assert!(
        result.nan_fields.is_empty(),
        "No NaN fields expected in clean data"
    );
    assert!(
        result.inf_fields.is_empty(),
        "No Inf fields expected in clean data"
    );
}

#[test]
fn test_distribution_bounds_coverage() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_out_of_bounds_field_data();
    let stats = validator.compute_shard_stats("oob_shard", &field_data);
    let result = validator.validate_shard(&stats);

    assert_eq!(
        result.grade,
        QualityGrade::Warn,
        "Out-of-bounds values should warn (not fail for coverage > threshold)"
    );
    assert!(
        !result.out_of_bounds_fields.is_empty(),
        "Out-of-bounds fields should be flagged"
    );
}

#[test]
fn test_outlier_detection() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let mut field_data = HashMap::new();
    field_data.insert("exterior_temp".to_string(), vec![0.0; 18]);
    field_data
        .entry("exterior_temp".to_string())
        .or_default()
        .push(10000.0);
    field_data.insert("zone_temp".to_string(), vec![22.0; 19]);
    field_data.insert("solar_rad".to_string(), vec![500.0; 19]);
    field_data.insert("humidity".to_string(), vec![50.0; 19]);
    field_data.insert("occupancy".to_string(), vec![0.3; 19]);

    let stats = validator.compute_shard_stats("outlier_shard", &field_data);
    let result = validator.validate_shard(&stats);

    assert!(
        !result.outlier_fields.is_empty(),
        "Outlier fields should be reported"
    );
    assert_eq!(
        result.grade,
        QualityGrade::Warn,
        "Outliers (>5%% of samples beyond z=3) should cause a Warn grade"
    );
}

// ---------------------------------------------------------------------------------------------------
// AC2: Per-shard summary stats emitted alongside data
// ---------------------------------------------------------------------------------------------------

#[test]
fn test_per_shard_stats_emitted() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_clean_field_data();
    let stats = validator.compute_shard_stats("test_shard", &field_data);

    assert_eq!(stats.shard_id, "test_shard", "Shard ID should be recorded");
    assert_eq!(stats.num_samples, 7, "Sample count should be recorded");
    assert!(
        stats.min_values.contains_key("exterior_temp"),
        "Min values should be emitted per field"
    );
    assert!(
        stats.max_values.contains_key("exterior_temp"),
        "Max values should be emitted per field"
    );
    assert!(
        stats.mean_values.contains_key("exterior_temp"),
        "Mean values should be emitted per field"
    );
    assert!(
        stats.std_values.contains_key("exterior_temp"),
        "Std values should be emitted per field"
    );
}

#[test]
fn test_per_shard_nan_inf_counts_emitted() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);

    let mut nan_data = make_clean_field_data();
    nan_data.insert(
        "exterior_temp".to_string(),
        vec![20.0, f64::NAN, 22.0, f64::NAN],
    );
    nan_data.insert(
        "solar_rad".to_string(),
        vec![500.0, f64::INFINITY, 600.0, 450.0],
    );

    let stats = validator.compute_shard_stats("nan_inf_shard", &nan_data);

    assert_eq!(
        stats.nan_count.get("exterior_temp"),
        Some(&2),
        "NaN count should be recorded per field"
    );
    assert_eq!(
        stats.inf_count.get("solar_rad"),
        Some(&1),
        "Inf count should be recorded per field"
    );
}

#[test]
fn test_aggregate_stats_combines_shards() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);

    let field_data_1 = make_clean_field_data();
    let field_data_2 = make_clean_field_data();

    let stats_1 = validator.compute_shard_stats("shard_1", &field_data_1);
    let stats_2 = validator.compute_shard_stats("shard_2", &field_data_2);

    let aggregated = SyntheticDataValidator::aggregate_stats(&[stats_1, stats_2]);

    assert_eq!(
        aggregated.num_samples, 14,
        "Aggregated sample count should sum across shards"
    );
    assert_eq!(
        aggregated.nan_count.get("exterior_temp"),
        Some(&0),
        "NaN counts should be summed in aggregation"
    );
}

#[test]
fn test_shard_stats_have_all_required_fields() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_clean_field_data();
    let stats = validator.compute_shard_stats("full_shard", &field_data);

    assert!(
        stats.min_values.len() >= 5,
        "Should have min for all configured fields"
    );
    assert!(
        stats.max_values.len() >= 5,
        "Should have max for all configured fields"
    );
    assert!(
        stats.mean_values.len() >= 5,
        "Should have mean for all configured fields"
    );
    assert!(
        stats.std_values.len() >= 5,
        "Should have std for all configured fields"
    );
    assert!(
        stats.nan_count.len() >= 5,
        "Should have nan_count for all configured fields"
    );
    assert!(
        stats.inf_count.len() >= 5,
        "Should have inf_count for all configured fields"
    );
    assert!(
        stats.out_of_bounds_count.len() >= 5,
        "Should have out_of_bounds_count for all configured fields"
    );
    assert!(
        stats.outlier_count.len() >= 5,
        "Should have outlier_count for all configured fields"
    );
}

// ---------------------------------------------------------------------------------------------------
// AC3: CI gate fails on quality regression
// ---------------------------------------------------------------------------------------------------

#[test]
fn test_ci_gate_fails_quality_regression_nan() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_nan_field_data();
    let stats = validator.compute_shard_stats("regression_nan", &field_data);
    let result = validator.validate_shard(&stats);

    assert_eq!(
        result.grade,
        QualityGrade::Fail,
        "CI gate must fail on NaN regression"
    );
}

#[test]
fn test_ci_gate_fails_quality_regression_inf() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_inf_field_data();
    let stats = validator.compute_shard_stats("regression_inf", &field_data);
    let result = validator.validate_shard(&stats);

    assert_eq!(
        result.grade,
        QualityGrade::Fail,
        "CI gate must fail on Inf regression"
    );
}

#[test]
fn test_ci_gate_warns_on_drift() {
    let config = make_standard_config();
    let mut validator = SyntheticDataValidator::new(config);

    let ref_data = make_clean_field_data();
    let ref_stats = validator.compute_shard_stats("reference", &ref_data);
    validator.set_reference_stats(ref_stats);

    let new_data = make_drift_field_data();
    let new_stats = validator.compute_shard_stats("drifted", &new_data);
    let result = validator.validate_shard(&new_stats);

    assert!(
        result.drift_detected,
        "Drift should be detected vs reference data"
    );
    assert_eq!(
        result.grade,
        QualityGrade::Warn,
        "Drift should cause Warn grade (not Fail)"
    );
}

#[test]
fn test_ci_gate_passes_clean_batch() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);

    let shard_1 = validator.compute_shard_stats("batch_shard_1", &make_clean_field_data());
    let shard_2 = validator.compute_shard_stats("batch_shard_2", &make_clean_field_data());

    let results = validator.validate_batch(
        &[shard_1, shard_2],
        &["batch_shard_1".to_string(), "batch_shard_2".to_string()],
    );

    for (_shard_id, result) in results {
        assert_eq!(
            result.grade,
            QualityGrade::Pass,
            "All shards in clean batch should pass CI gate"
        );
    }
}

#[test]
fn test_ci_gate_batch_fails_on_single_nan_shard() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);

    let shard_clean = validator.compute_shard_stats("clean", &make_clean_field_data());
    let shard_nan = validator.compute_shard_stats("nan_shard", &make_nan_field_data());

    let results = validator.validate_batch(
        &[shard_clean, shard_nan],
        &["clean".to_string(), "nan_shard".to_string()],
    );

    assert_eq!(
        results.get("nan_shard").unwrap().grade,
        QualityGrade::Fail,
        "NaN shard should cause CI gate failure"
    );
    assert_eq!(
        results.get("clean").unwrap().grade,
        QualityGrade::Pass,
        "Clean shard should still pass"
    );
}

// ---------------------------------------------------------------------------------------------------
// Edge cases and regression guards
// ---------------------------------------------------------------------------------------------------

#[test]
fn test_empty_field_data_handled() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);

    let mut empty_data = HashMap::new();
    empty_data.insert("exterior_temp".to_string(), vec![]);

    let stats = validator.compute_shard_stats("empty_shard", &empty_data);
    assert_eq!(
        stats.num_samples, 0,
        "Empty field data should produce zero sample count"
    );
}

#[test]
fn test_single_value_field_stats() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);

    let mut single_data = HashMap::new();
    single_data.insert("exterior_temp".to_string(), vec![42.0]);

    let stats = validator.compute_shard_stats("single_shard", &single_data);

    assert_eq!(stats.num_samples, 1);
    assert_eq!(stats.min_values.get("exterior_temp"), Some(&42.0));
    assert_eq!(stats.max_values.get("exterior_temp"), Some(&42.0));
    assert_eq!(stats.mean_values.get("exterior_temp"), Some(&42.0));
    assert!(
        stats
            .std_values
            .get("exterior_temp")
            .map(|&v| v == 0.0)
            .unwrap_or(false),
        "Single value should produce zero std (zero variance)"
    );
}

#[test]
fn test_quality_check_result_pass_helper() {
    let result = QualityCheckResult::pass();
    assert!(result.is_pass());
    assert_eq!(result.grade, QualityGrade::Pass);
    assert!(result.nan_fields.is_empty());
    assert!(result.inf_fields.is_empty());
    assert!(result.out_of_bounds_fields.is_empty());
    assert!(result.outlier_fields.is_empty());
    assert!(!result.drift_detected);
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
fn test_validation_config_standard_fields() {
    let config = ValidationConfig::standard();
    assert_eq!(config.outlier_z_threshold, 3.0);
    assert_eq!(config.drift_reference_window, 100);
    assert_eq!(config.min_coverage_per_field, 0.95);
    assert_eq!(
        config.fields.get("exterior_temp"),
        Some(&FieldBounds::new(-50.0, 60.0))
    );
    assert_eq!(
        config.fields.get("zone_temp"),
        Some(&FieldBounds::new(10.0, 40.0))
    );
    assert_eq!(
        config.fields.get("solar_rad"),
        Some(&FieldBounds::new(0.0, 1200.0))
    );
    assert_eq!(
        config.fields.get("humidity"),
        Some(&FieldBounds::new(0.0, 100.0))
    );
    assert_eq!(
        config.fields.get("occupancy"),
        Some(&FieldBounds::new(0.0, 10.0))
    );
}

#[test]
fn test_aggregate_stats_empty_shards() {
    let aggregated = SyntheticDataValidator::aggregate_stats(&[]);
    assert_eq!(aggregated.num_samples, 0);
    assert_eq!(aggregated.shard_id, "aggregated");
}

#[test]
fn test_data_shard_stats_new() {
    let stats = DataShardStats::new("test_id");
    assert_eq!(stats.shard_id, "test_id");
    assert_eq!(stats.num_samples, 0);
    assert!(stats.nan_count.is_empty());
    assert!(stats.inf_count.is_empty());
}

#[test]
fn test_validator_with_reference_stats() {
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let field_data = make_clean_field_data();
    let stats = validator.compute_shard_stats("ref", &field_data);
    let validator_with_ref = validator.with_reference_stats(stats);
    let mut ref_data = HashMap::new();
    ref_data.insert("exterior_temp".to_string(), vec![20.0; 10]);
    ref_data.insert("zone_temp".to_string(), vec![22.0; 10]);
    ref_data.insert("solar_rad".to_string(), vec![500.0; 10]);
    ref_data.insert("humidity".to_string(), vec![50.0; 10]);
    ref_data.insert("occupancy".to_string(), vec![0.3; 10]);
    let new_stats = validator_with_ref.compute_shard_stats("new", &ref_data);
    let result = validator_with_ref.validate_shard(&new_stats);
    assert!(
        !result.drift_detected,
        "No drift expected when reference stats match the same distribution"
    );
}

#[test]
fn test_multiple_drift_fields_detected() {
    let config = make_standard_config();
    let mut validator = SyntheticDataValidator::new(config);

    let mut ref_data = HashMap::new();
    let ref_temp: Vec<f64> = (0..50)
        .map(|i| if i % 2 == 0 { 20.0 } else { 22.0 })
        .collect();
    ref_data.insert("exterior_temp".to_string(), ref_temp);
    ref_data.insert("zone_temp".to_string(), vec![22.0; 50]);
    ref_data.insert("solar_rad".to_string(), vec![500.0; 50]);
    ref_data.insert("humidity".to_string(), vec![50.0; 50]);
    ref_data.insert("occupancy".to_string(), vec![0.3; 50]);

    let ref_stats = validator.compute_shard_stats("reference", &ref_data);
    assert_eq!(
        ref_stats.num_samples, 50,
        "Reference stats should have 50 samples"
    );
    assert!(
        ref_stats
            .std_values
            .get("exterior_temp")
            .copied()
            .unwrap_or(0.0)
            > 0.0,
        "Reference std should be > 0 for drift detection"
    );
    validator.set_reference_stats(ref_stats);

    let mut new_data = HashMap::new();
    new_data.insert("exterior_temp".to_string(), vec![60.0; 50]);
    new_data.insert("zone_temp".to_string(), vec![22.0; 50]);
    new_data.insert("solar_rad".to_string(), vec![500.0; 50]);
    new_data.insert("humidity".to_string(), vec![50.0; 50]);
    new_data.insert("occupancy".to_string(), vec![0.3; 50]);

    let stats = validator.compute_shard_stats("drifted", &new_data);
    assert_eq!(
        stats.num_samples, 50,
        "New data stats should have 50 samples"
    );
    let result = validator.validate_shard(&stats);

    assert!(
        result.drift_detected,
        "Drift should be detected when mean differs significantly from reference"
    );
}

#[test]
fn test_compute_stats_via_shard_stats() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut field_data = HashMap::new();
    field_data.insert("test_field".to_string(), values);
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let stats = validator.compute_shard_stats("stats_test", &field_data);
    assert_eq!(stats.min_values.get("test_field"), Some(&1.0));
    assert_eq!(stats.max_values.get("test_field"), Some(&5.0));
    assert!((stats.mean_values.get("test_field").copied().unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_compute_stats_empty_field() {
    let values: Vec<f64> = vec![];
    let mut field_data = HashMap::new();
    field_data.insert("empty_field".to_string(), values);
    let config = make_standard_config();
    let validator = SyntheticDataValidator::new(config);
    let stats = validator.compute_shard_stats("empty_test", &field_data);
    assert_eq!(stats.num_samples, 0);
    assert_eq!(stats.nan_count.get("empty_field"), Some(&0));
}

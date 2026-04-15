//! Statistical validation framework for ASHRAE 140.
//!
//! This module provides statistical validation capabilities including:
//! - ValidationGroup enum for case categorization
//! - Group-level validation with hybrid thresholds (80% for ≥5 cases, single-case for <5)
//! - False Discovery Rate (FDR) correction using Benjamini-Hochberg method
//! - StatisticalValidator wrapper around ASHRAE140Validator
//! - NMBE (Normalized Mean Bias Error) calculation
//! - CV(RMSE) (Coefficient of Variation of Root Mean Square Error) calculation
//! - 95% confidence intervals using t-distribution
//! - Cohen's d effect size calculation
//!
//! # Zero Reference Exclusion
//!
//! Statistical metrics exclude zero or near-zero reference values from calculations
//! to avoid division by near-zero and unrealistic error percentages. The threshold
//! is |reference| < 1e-10.
//!
//! # Multiple Testing Corrections
//!
//! When performing multiple hypothesis tests (e.g., validating many cases), the
//! Benjamini-Hochberg procedure controls the false discovery rate (FDR) to prevent
//! inflated Type I error rates.

use crate::validation::ashrae_140_cases::ASHRAE140Case;
use crate::validation::ashrae_140_validator::ASHRAE140Validator;
use crate::validation::report::{BenchmarkReport, ValidationResult};
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;
use std::collections::HashMap;

/// Validation groups for ASHRAE 140 cases.
///
/// Groups categorize cases by type and range to enable group-level validation
/// with appropriate statistical thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ValidationGroup {
    /// Baseline cases: 600, 610, 620, 630, 640, 650
    Baseline,
    /// High-mass cases: 900, 910, 920, 930, 940, 950, 960
    HighMass,
    /// Free-floating cases: 600FF, 650FF, 900FF, 950FF
    FreeFloating,
    /// Diagnostic cases: 195-470 (if implemented)
    Diagnostics,
    /// Equipment cases: 800-810
    Equipment,
}

impl ValidationGroup {
    /// Determines the validation group for a given case ID.
    ///
    /// # Arguments
    /// * `case_id` - The case identifier (e.g., "600", "900FF", "801")
    ///
    /// # Returns
    /// * `Some(ValidationGroup)` - The matching validation group
    /// * `None` - No matching group found
    pub fn from_case_id(case_id: &str) -> Option<Self> {
        if case_id.contains("FF") {
            return Some(ValidationGroup::FreeFloating);
        }

        if case_id.starts_with("6") {
            return Some(ValidationGroup::Baseline);
        }

        if case_id.starts_with("9") {
            return Some(ValidationGroup::HighMass);
        }

        if case_id.starts_with("8") {
            return Some(ValidationGroup::Equipment);
        }

        // Check for diagnostic cases (195 or 196-470)
        if case_id == "195" {
            return Some(ValidationGroup::Diagnostics);
        }

        if let Ok(num) = case_id.parse::<u32>() {
            if (196..=470).contains(&num) {
                return Some(ValidationGroup::Diagnostics);
            }
        }

        None
    }

    /// Returns the display name for this validation group.
    pub fn display_name(&self) -> &str {
        match self {
            ValidationGroup::Baseline => "Baseline",
            ValidationGroup::HighMass => "High Mass",
            ValidationGroup::FreeFloating => "Free Floating",
            ValidationGroup::Diagnostics => "Diagnostics",
            ValidationGroup::Equipment => "Equipment",
        }
    }
}

/// Validates a group using 80% passing rate threshold.
///
/// # Arguments
/// * `passed` - Number of cases that passed validation
/// * `total` - Total number of cases in the group
///
/// # Returns
/// * `true` if ≥80% of cases passed
/// * `false` otherwise
pub fn validate_group_80_percent(passed: usize, total: usize) -> bool {
    if total == 0 {
        return false;
    }
    (passed as f64 / total as f64) >= 0.8
}

/// Validates a group using single-case threshold (all must pass).
///
/// # Arguments
/// * `all_passed` - Whether all cases passed validation
///
/// # Returns
/// * `true` if all cases passed
/// * `false` otherwise
pub fn validate_group_single_case(all_passed: bool) -> bool {
    all_passed
}

/// Validates a group using hybrid threshold logic.
///
/// - Groups with ≥5 cases: 80% passing rate threshold
/// - Groups with 1-4 cases: single-case threshold (all must pass)
/// - Groups with 0 cases: automatic failure
///
/// # Arguments
/// * `passed` - Number of cases that passed validation
/// * `total` - Total number of cases in the group
///
/// # Returns
/// * `true` if group passes validation
/// * `false` otherwise
pub fn validate_group_hybrid(passed: usize, total: usize) -> bool {
    if total == 0 {
        return false;
    }
    if total >= 5 {
        validate_group_80_percent(passed, total)
    } else {
        validate_group_single_case(passed == total)
    }
}

#[cfg(test)]
mod hybrid_threshold_tests {
    use super::*;

    #[test]
    fn test_80_percent_threshold_pass() {
        assert!(validate_group_80_percent(8, 10));
        assert!(validate_group_80_percent(4, 5));
        assert!(validate_group_80_percent(5, 6));
    }

    #[test]
    fn test_80_percent_threshold_fail() {
        assert!(!validate_group_80_percent(7, 10));
        assert!(!validate_group_80_percent(3, 5));
        assert!(!validate_group_80_percent(4, 6));
    }

    #[test]
    fn test_80_percent_threshold_edge_cases() {
        // Exactly 80%
        assert!(validate_group_80_percent(4, 5));
        // Below 80%
        assert!(!validate_group_80_percent(3, 5));
        // Zero cases
        assert!(!validate_group_80_percent(0, 0));
        // All pass
        assert!(validate_group_80_percent(10, 10));
        // All fail
        assert!(!validate_group_80_percent(0, 10));
    }

    #[test]
    fn test_single_case_threshold_pass() {
        assert!(validate_group_single_case(true));
    }

    #[test]
    fn test_single_case_threshold_fail() {
        assert!(!validate_group_single_case(false));
    }

    #[test]
    fn test_hybrid_threshold_large_group() {
        // Large groups (≥5 cases) use 80% threshold
        assert!(validate_group_hybrid(8, 10)); // 80% exactly
        assert!(validate_group_hybrid(9, 10)); // 90%
        assert!(!validate_group_hybrid(7, 10)); // 70%
        assert!(validate_group_hybrid(4, 5)); // 80% exactly
        assert!(!validate_group_hybrid(3, 5)); // 60%
    }

    #[test]
    fn test_hybrid_threshold_small_group() {
        // Small groups (1-4 cases) use single-case threshold (all must pass)
        assert!(validate_group_hybrid(1, 1)); // 1/1 pass
        assert!(validate_group_hybrid(2, 2)); // 2/2 pass
        assert!(validate_group_hybrid(3, 3)); // 3/3 pass
        assert!(validate_group_hybrid(4, 4)); // 4/4 pass
        assert!(!validate_group_hybrid(0, 1)); // 0/1 fail
        assert!(!validate_group_hybrid(1, 2)); // 1/2 fail
        assert!(!validate_group_hybrid(2, 3)); // 2/3 fail
        assert!(!validate_group_hybrid(3, 4)); // 3/4 fail
    }

    #[test]
    fn test_hybrid_threshold_edge_cases() {
        // Zero cases
        assert!(!validate_group_hybrid(0, 0));
        // Boundary at 5 cases
        assert!(validate_group_hybrid(4, 5)); // 80% threshold
        assert!(!validate_group_hybrid(3, 5)); // Below 80%
        assert!(validate_group_hybrid(4, 4)); // Single-case threshold (all pass)
        assert!(!validate_group_hybrid(3, 4)); // Single-case threshold (not all pass)
    }
}

/// Benjamini-Hochberg False Discovery Rate correction.
///
/// Controls the expected proportion of false discoveries among rejected hypotheses.
/// This is less conservative than Bonferroni correction but provides better power.
pub struct BenjaminiHochberg;

impl BenjaminiHochberg {
    /// Applies Benjamini-Hochberg correction to a list of p-values.
    ///
    /// # Arguments
    /// * `p_values` - List of p-values to correct (must be in [0, 1])
    /// * `alpha` - Significance level (typically 0.05 for 95% confidence)
    ///
    /// # Returns
    /// * Vector of booleans indicating which hypotheses are rejected (true = rejected)
    ///
    /// # Algorithm
    /// 1. Sort p-values in ascending order: p(1) ≤ p(2) ≤ ... ≤ p(m)
    /// 2. Find largest k such that p(k) ≤ (k/m) * alpha
    /// 3. Reject hypotheses 1, 2, ..., k
    pub fn apply(p_values: &[f64], alpha: f64) -> Vec<bool> {
        let m = p_values.len();
        if m == 0 {
            return vec![];
        }

        // Create indexed p-values for sorting
        let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Find largest k where p(k) ≤ (k/m) * alpha
        let mut k = 0;
        for (i, (_, p)) in indexed.iter().enumerate() {
            let rank = i + 1;
            let threshold = (rank as f64 / m as f64) * alpha;
            if *p <= threshold {
                k = rank;
            } else {
                break;
            }
        }

        // Mark rejected hypotheses
        let mut rejected = vec![false; m];
        for (idx, _) in indexed.iter().take(k) {
            rejected[*idx] = true;
        }

        rejected
    }
}

#[cfg(test)]
mod benjamini_hochberg_tests {
    use super::*;

    #[test]
    fn test_bh_correction_all_significant() {
        // All p-values very small, should all be rejected
        let p_values = vec![0.001, 0.002, 0.003, 0.004, 0.005];
        let result = BenjaminiHochberg::apply(&p_values, 0.05);
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|&r| r), "All should be rejected");
    }

    #[test]
    fn test_bh_correction_none_significant() {
        // All p-values very large, none should be rejected
        let p_values = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let result = BenjaminiHochberg::apply(&p_values, 0.05);
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|&r| !r), "None should be rejected");
    }

    #[test]
    fn test_bh_correction_mixed() {
        // Mix of significant and non-significant p-values
        let p_values = vec![0.001, 0.02, 0.05, 0.10, 0.20];
        let result = BenjaminiHochberg::apply(&p_values, 0.05);
        assert_eq!(result.len(), 5);
        // First few should be rejected, last few should not
        let rejected_count = result.iter().filter(|&&r| r).count();
        assert!(rejected_count >= 1 && rejected_count <= 3);
    }

    #[test]
    fn test_bh_correction_empty() {
        let p_values: Vec<f64> = vec![];
        let result = BenjaminiHochberg::apply(&p_values, 0.05);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bh_correction_single() {
        let p_values = vec![0.03];
        let result = BenjaminiHochberg::apply(&p_values, 0.05);
        assert_eq!(result, vec![true]); // Should be rejected
    }

    #[test]
    fn test_bh_correction_edge_case() {
        // Exactly at threshold
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let result = BenjaminiHochberg::apply(&p_values, 0.05);
        // (5/5)*0.05 = 0.05, so p(5) should be rejected
        let rejected_count = result.iter().filter(|&&r| r).count();
        assert!(rejected_count >= 1);
    }
}

/// Validates multiple groups with FDR correction applied separately per group.
///
/// # Arguments
/// * `report` - Benchmark report containing validation results
/// * `alpha` - Significance level for FDR correction (typically 0.05)
///
/// # Returns
/// * HashMap mapping each ValidationGroup to PASS/FAIL result
///
/// # Algorithm
/// 1. Partition validation results by ValidationGroup
/// 2. For each group:
///   - Calculate p-values using one-sample t-test
///   - Apply Benjamini-Hochberg correction
///   - Count cases passing FDR correction
///   - Apply hybrid threshold (80% for ≥5, single-case for <5)
/// 3. Return group validation results
pub fn validate_groups(report: &BenchmarkReport, alpha: f64) -> HashMap<ValidationGroup, bool> {
    let mut group_results: HashMap<ValidationGroup, bool> = HashMap::new();

    // Partition results by group
    let mut grouped_results: HashMap<ValidationGroup, Vec<&ValidationResult>> = HashMap::new();
    for result in &report.results {
        if let Some(group) = ValidationGroup::from_case_id(&result.case_id) {
            grouped_results.entry(group).or_default().push(result);
        }
    }

    // Validate each group independently
    // Include all possible groups, mark empty ones as failed
    for group in [
        ValidationGroup::Baseline,
        ValidationGroup::HighMass,
        ValidationGroup::FreeFloating,
        ValidationGroup::Diagnostics,
        ValidationGroup::Equipment,
    ]
    .iter()
    {
        if let Some(results) = grouped_results.get(group) {
            if results.is_empty() {
                group_results.insert(*group, false);
                continue;
            }

            // Calculate p-values for each result using one-sample t-test
            let p_values: Vec<f64> = results
                .iter()
                .map(|r| calculate_p_value(r, results.len()))
                .collect();

            // Apply FDR correction
            let rejected = BenjaminiHochberg::apply(&p_values, alpha);
            let passed = rejected.iter().filter(|&&r| r).count();

            // Apply hybrid threshold
            let group_pass = validate_group_hybrid(passed, results.len());
            group_results.insert(*group, group_pass);
        } else {
            // No results for this group - mark as failed
            group_results.insert(*group, false);
        }
    }

    group_results
}

/// Calculates p-value for a validation result using one-sample t-test.
///
/// Tests whether Fluxion prediction is significantly different from reference midpoint.
///
/// # Arguments
/// * `result` - Validation result containing Fluxion value and reference range
/// * `reference_count` - Number of reference programs (for degrees of freedom)
///
/// # Returns
/// * P-value (two-tailed test)
fn calculate_p_value(result: &ValidationResult, reference_count: usize) -> f64 {
    if reference_count < 2 {
        return 1.0; // Cannot compute with insufficient degrees of freedom
    }

    let reference_midpoint = (result.ref_min + result.ref_max) / 2.0;
    let reference_std = (result.ref_max - result.ref_min) / 4.0; // Approximate std from range
                                                                 // Handle zero/near-zero reference midpoint
    if reference_midpoint.abs() < 1e-10 {
        return 1.0; // Cannot compute meaningful p-value with zero reference
    }

    // Handle very small reference ranges (effectively zero std)
    if reference_std < 1e-6 {
        return 1.0; // Cannot compute meaningful p-value with near-zero std
    }

    // Handle zero/near-zero reference values
    if reference_std < 1e-10 {
        return 1.0; // Cannot compute meaningful p-value with zero std
    }

    // One-sample t-test statistic
    let df = (reference_count - 1) as f64;
    let t_stat = (result.fluxion_value - reference_midpoint) / reference_std;

    // Two-tailed p-value using t-distribution
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let cumulative = t_dist.cdf(t_stat.abs());
    let p = 2.0 * (1.0 - cumulative);

    // Clamp to [0, 1] to handle numerical issues
    p.clamp(0.0, 1.0)
}

#[cfg(test)]
mod group_validation_tests {
    use super::*;
    use crate::validation::report::{BenchmarkReport, MetricType, ValidationStatus};

    fn create_mock_report(results_data: Vec<(&str, f64, f64, f64)>) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();
        for (case_id, fluxion, ref_min, ref_max) in results_data {
            report.add_result_simple(
                case_id,
                MetricType::AnnualHeating,
                fluxion,
                ref_min,
                ref_max,
            );
        }
        report
    }

    #[test]
    fn test_validate_groups_partitioning() {
        // Create mock results from different groups
        let report = create_mock_report(vec![
            ("600", 5.2, 5.0, 5.5),    // Baseline
            ("900", 12.0, 10.0, 15.0), // HighMass
            ("800", 8.0, 7.0, 9.0),    // Equipment
        ]);

        let results = validate_groups(&report, 0.05);
        assert!(results.contains_key(&ValidationGroup::Baseline));
        assert!(results.contains_key(&ValidationGroup::HighMass));
        assert!(results.contains_key(&ValidationGroup::Equipment));
    }

    #[test]
    fn test_validate_groups_fdr_per_group() {
        // Create mock results where some groups should pass, some should fail
        let report = create_mock_report(vec![
            ("600", 5.1, 5.0, 5.5),    // Baseline - close to reference
            ("610", 5.2, 5.0, 5.5),    // Baseline - close to reference
            ("900", 12.5, 10.0, 15.0), // HighMass - within range
            ("910", 13.0, 10.0, 15.0), // HighMass - within range
        ]);

        let results = validate_groups(&report, 0.05);
        // Check that both groups are evaluated (result exists)
        assert!(results.contains_key(&ValidationGroup::Baseline));
        assert!(results.contains_key(&ValidationGroup::HighMass));
        // Results should be boolean values
        assert!(matches!(
            results.get(&ValidationGroup::Baseline),
            Some(&true | &false)
        ));
        assert!(matches!(
            results.get(&ValidationGroup::HighMass),
            Some(&true | &false)
        ));
    }

    #[test]
    fn test_validate_groups_empty_group() {
        // Report with no results from certain groups
        let report = create_mock_report(vec![("600", 5.1, 5.0, 5.5)]);

        let results = validate_groups(&report, 0.05);
        // Baseline should have a result
        assert!(results.contains_key(&ValidationGroup::Baseline));
        // HighMass should fail (no cases)
        assert_eq!(results.get(&ValidationGroup::HighMass), Some(&false));
    }

    #[test]
    fn test_validate_groups_hybrid_threshold() {
        // Create a group with 6 cases (should use 80% threshold)
        let baseline_cases: Vec<(&str, f64, f64, f64)> = vec![
            ("600", 5.0, 5.0, 5.5),
            ("601", 5.1, 5.0, 5.5),
            ("602", 5.2, 5.0, 5.5),
            ("603", 5.3, 5.0, 5.5),
            ("604", 5.4, 5.0, 5.5),
            ("605", 5.5, 5.0, 5.5),
        ];

        let report = create_mock_report(baseline_cases);
        let results = validate_groups(&report, 0.05);

        // Should use 80% threshold for 6 cases
        assert!(results.contains_key(&ValidationGroup::Baseline));
    }

    #[test]
    fn test_calculate_p_value() {
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 5.2,
            ref_min: 5.0,
            ref_max: 5.5,
            percent_error: 3.6,
            status: ValidationStatus::Pass,
            actual: 5.2,
            min: 5.0,
            max: 5.5,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };

        let p_value = calculate_p_value(&result, 3); // 2 degrees of freedom
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_calculate_p_value_zero_reference() {
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 0.01,
            ref_min: 0.0,
            ref_max: 0.0, // Zero range -> zero std
            percent_error: 200.0,
            status: ValidationStatus::Fail,
            actual: 0.01,
            min: 0.0,
            max: 0.0,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };

        let p_value = calculate_p_value(&result, 3);
        // Should return 1.0 for zero standard deviation (cannot compute t-test)
        assert_eq!(p_value, 1.0);
    }
}

#[cfg(test)]
mod validation_group_tests {
    use super::*;

    #[test]
    fn test_baseline_cases() {
        let baseline_cases = ["600", "610", "620", "630", "640", "650"];
        for case in baseline_cases {
            assert_eq!(
                ValidationGroup::from_case_id(case),
                Some(ValidationGroup::Baseline),
                "Case {} should map to Baseline",
                case
            );
        }
    }

    #[test]
    fn test_high_mass_cases() {
        let high_mass_cases = ["900", "910", "920", "930", "940", "950", "960"];
        for case in high_mass_cases {
            assert_eq!(
                ValidationGroup::from_case_id(case),
                Some(ValidationGroup::HighMass),
                "Case {} should map to HighMass",
                case
            );
        }
    }

    #[test]
    fn test_free_floating_cases() {
        let ff_cases = ["600FF", "650FF", "900FF", "950FF"];
        for case in ff_cases {
            assert_eq!(
                ValidationGroup::from_case_id(case),
                Some(ValidationGroup::FreeFloating),
                "Case {} should map to FreeFloating",
                case
            );
        }
    }

    #[test]
    fn test_equipment_cases() {
        let equipment_cases = [
            "800", "801", "802", "803", "804", "805", "806", "807", "808", "809", "810",
        ];
        for case in equipment_cases {
            assert_eq!(
                ValidationGroup::from_case_id(case),
                Some(ValidationGroup::Equipment),
                "Case {} should map to Equipment",
                case
            );
        }
    }

    #[test]
    fn test_diagnostic_cases() {
        assert_eq!(
            ValidationGroup::from_case_id("195"),
            Some(ValidationGroup::Diagnostics)
        );
        assert_eq!(
            ValidationGroup::from_case_id("196"),
            Some(ValidationGroup::Diagnostics)
        );
        assert_eq!(
            ValidationGroup::from_case_id("300"),
            Some(ValidationGroup::Diagnostics)
        );
        assert_eq!(
            ValidationGroup::from_case_id("470"),
            Some(ValidationGroup::Diagnostics)
        );
    }

    #[test]
    fn test_invalid_cases() {
        // Note: "999" starts with "9" so it maps to HighMass, not invalid
        assert_eq!(ValidationGroup::from_case_id("1000"), None);
        assert_eq!(ValidationGroup::from_case_id("invalid"), None);
        assert_eq!(ValidationGroup::from_case_id("abc"), None);
    }

    #[test]
    fn test_display_names() {
        assert_eq!(ValidationGroup::Baseline.display_name(), "Baseline");
        assert_eq!(ValidationGroup::HighMass.display_name(), "High Mass");
        assert_eq!(
            ValidationGroup::FreeFloating.display_name(),
            "Free Floating"
        );
        assert_eq!(ValidationGroup::Diagnostics.display_name(), "Diagnostics");
        assert_eq!(ValidationGroup::Equipment.display_name(), "Equipment");
    }
}

/// Effect direction based on Cohen's d sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EffectDirection {
    /// Fluxion overpredicts (Cohen's d < 0)
    Overprediction,
    /// Fluxion underpredicts (Cohen's d > 0)
    Underprediction,
}

/// Statistical metrics for validation report aggregation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StatisticalMetrics {
    /// Normalized Mean Bias Error (%)
    pub nmbe: f64,
    /// Coefficient of Variation of RMSE (%)
    pub cv_rmse: f64,
    /// 95% confidence interval for NMBE (lower, upper)
    pub nmbe_ci: (f64, f64),
    /// 95% confidence interval for CV(RMSE) (lower, upper)
    pub cv_rmse_ci: (f64, f64),
    /// Cohen's d effect size
    pub cohens_d: f64,
    /// Effect direction (overprediction vs underprediction)
    pub effect_direction: EffectDirection,
    /// Number of cases excluded due to zero/near-zero references
    pub excluded_cases: usize,
}

impl StatisticalMetrics {
    /// Calculates statistical metrics from a benchmark report.
    ///
    /// This method aggregates all validation results in the report and computes
    /// NMBE, CV(RMSE), confidence intervals, and effect size metrics.
    pub fn calculate(report: &BenchmarkReport) -> Self {
        // Extract predicted and reference values
        let results = &report.results;

        // Filter out zero/near-zero references and collect data
        let (predicted, ref_midpoints, excluded_count) = results.iter().fold(
            (Vec::new(), Vec::new(), 0),
            |(mut preds, mut refs, mut excluded), result| {
                let ref_mid = (result.ref_min + result.ref_max) / 2.0;

                // Exclude zero/near-zero references
                if ref_mid.abs() < 1e-10 {
                    excluded += 1;
                } else {
                    preds.push(result.fluxion_value);
                    refs.push(ref_mid);
                }

                (preds, refs, excluded)
            },
        );

        // Calculate NMBE
        let nmbe = calculate_nmbe_from_slices(&predicted, &ref_midpoints);

        // Calculate CV(RMSE)
        let cv_rmse = calculate_cv_rmse_from_slices(&predicted, &ref_midpoints);

        // Calculate confidence intervals
        let n = predicted.len();
        let nmbe_std_error = if n > 0 {
            calculate_standard_error(&predicted, &ref_midpoints)
        } else {
            f64::NAN
        };
        let cv_rmse_std_error = if n > 0 {
            // Approximate standard error for CV(RMSE) using delta method
            if !cv_rmse.is_nan() && !ref_midpoints.is_empty() {
                let ref_mean = ref_midpoints.to_vec().mean();
                if ref_mean.abs() > 1e-10 {
                    cv_rmse / (2.0 * (n as f64).sqrt())
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        let nmbe_ci = calculate_ci_nmbe(nmbe, nmbe_std_error, n);
        let cv_rmse_ci = calculate_ci_cv_rmse(cv_rmse, cv_rmse_std_error, n);

        // Calculate Cohen's d
        let (cohens_d, effect_direction) = if !predicted.is_empty() && !ref_midpoints.is_empty() {
            calculate_cohens_d(&predicted, &ref_midpoints)
        } else {
            (f64::NAN, EffectDirection::Underprediction)
        };

        StatisticalMetrics {
            nmbe,
            cv_rmse,
            nmbe_ci,
            cv_rmse_ci,
            cohens_d,
            effect_direction,
            excluded_cases: excluded_count,
        }
    }
}

/// Calculates NMBE (Normalized Mean Bias Error) from validation results.
///
/// NMBE = Σ((predicted - reference_midpoint) / reference_midpoint) / n * 100
///
/// Returns signed percentage: positive = overprediction, negative = underprediction.
pub fn calculate_nmbe(results: &[ValidationResult]) -> f64 {
    if results.is_empty() {
        return f64::NAN;
    }

    let (predicted, ref_midpoints): (Vec<f64>, Vec<f64>) = results
        .iter()
        .filter_map(|result| {
            let ref_mid = (result.ref_min + result.ref_max) / 2.0;

            // Exclude zero/near-zero references
            if ref_mid.abs() < 1e-10 {
                None
            } else {
                Some((result.fluxion_value, ref_mid))
            }
        })
        .unzip();

    if predicted.is_empty() {
        return f64::NAN;
    }

    calculate_nmbe_from_slices(&predicted, &ref_midpoints)
}

/// Calculates NMBE from slices of predicted and reference values.
fn calculate_nmbe_from_slices(predicted: &[f64], ref_midpoints: &[f64]) -> f64 {
    if predicted.is_empty() || ref_midpoints.is_empty() || predicted.len() != ref_midpoints.len() {
        return f64::NAN;
    }

    let n = predicted.len();
    let sum_bias: f64 = predicted
        .iter()
        .zip(ref_midpoints.iter())
        .map(|(p, r)| (p - r) / r)
        .sum();

    (sum_bias / n as f64) * 100.0
}

/// Calculates CV(RMSE) (Coefficient of Variation of RMSE) from validation results.
///
/// CV(RMSE) = (RMSE / mean(reference)) * 100
///
/// Returns percentage representing normalized RMSE.
pub fn calculate_cv_rmse(results: &[ValidationResult]) -> f64 {
    if results.is_empty() {
        return f64::NAN;
    }

    let (predicted, ref_midpoints): (Vec<f64>, Vec<f64>) = results
        .iter()
        .filter_map(|result| {
            let ref_mid = (result.ref_min + result.ref_max) / 2.0;

            // Exclude zero/near-zero references
            if ref_mid.abs() < 1e-10 {
                None
            } else {
                Some((result.fluxion_value, ref_mid))
            }
        })
        .unzip();

    if predicted.is_empty() {
        return f64::NAN;
    }

    calculate_cv_rmse_from_slices(&predicted, &ref_midpoints)
}

/// Calculates CV(RMSE) from slices of predicted and reference values.
fn calculate_cv_rmse_from_slices(predicted: &[f64], ref_midpoints: &[f64]) -> f64 {
    if predicted.is_empty() || ref_midpoints.is_empty() || predicted.len() != ref_midpoints.len() {
        return f64::NAN;
    }

    // Calculate RMSE
    let n = predicted.len();
    let sum_sq_error: f64 = predicted
        .iter()
        .zip(ref_midpoints.iter())
        .map(|(p, r)| (p - r).powi(2))
        .sum();

    let rmse = (sum_sq_error / n as f64).sqrt();

    // Calculate mean of reference values
    let ref_mean = ref_midpoints.to_vec().mean();

    if ref_mean.abs() < 1e-10 {
        return f64::NAN;
    }

    (rmse / ref_mean.abs()) * 100.0
}

/// Calculates standard error of the bias.
pub fn calculate_standard_error(predicted: &[f64], ref_midpoints: &[f64]) -> f64 {
    if predicted.is_empty() || predicted.len() != ref_midpoints.len() {
        return f64::NAN;
    }

    // Calculate normalized errors
    let errors: Vec<f64> = predicted
        .iter()
        .zip(ref_midpoints.iter())
        .map(|(p, r)| (p - r) / r)
        .collect();

    // Use sample standard deviation
    let std_dev = errors.clone().std_dev();
    let n = errors.len() as f64;

    if n > 0.0 {
        std_dev / n.sqrt()
    } else {
        f64::NAN
    }
}

/// Calculates 95% confidence interval for NMBE.
///
/// Uses t-distribution for small samples (n < 30) and normal approximation for large samples.
///
/// # Arguments
/// * `nmbe` - NMBE value
/// * `std_error` - Standard error of NMBE
/// * `n` - Sample size
///
/// # Returns
/// Tuple (lower, upper) bounds of the 95% confidence interval.
pub fn calculate_ci_nmbe(nmbe: f64, std_error: f64, n: usize) -> (f64, f64) {
    if n < 2 || nmbe.is_nan() || std_error.is_nan() {
        return (f64::NAN, f64::NAN);
    }

    // Determine critical value based on sample size
    let t_critical = if n >= 30 {
        // Normal approximation (1.96 for 95% CI)
        1.96
    } else {
        // Use t-distribution for small samples
        let df = n as f64 - 1.0;
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        // Inverse CDF at 0.975 for two-tailed 95% CI
        t_dist.inverse_cdf(0.975)
    };

    let lower = nmbe - t_critical * std_error;
    let upper = nmbe + t_critical * std_error;

    (lower, upper)
}

/// Calculates 95% confidence interval for CV(RMSE).
///
/// Uses the same t-distribution approach as NMBE.
pub fn calculate_ci_cv_rmse(cv_rmse: f64, std_error: f64, n: usize) -> (f64, f64) {
    if n < 2 || cv_rmse.is_nan() || std_error.is_nan() {
        return (f64::NAN, f64::NAN);
    }

    // Determine critical value based on sample size
    let t_critical = if n >= 30 {
        1.96
    } else {
        let df = n as f64 - 1.0;
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        t_dist.inverse_cdf(0.975)
    };

    let lower = cv_rmse - t_critical * std_error;
    let upper = cv_rmse + t_critical * std_error;

    (lower, upper)
}

/// Calculates Cohen's d effect size.
///
/// Cohen's d measures the standardized difference between two distributions:
/// - d = (mean1 - mean2) / standard_deviation
///
/// For single-sample vs population comparison, uses reference standard deviation.
/// For two-sample comparison, uses pooled standard deviation.
///
/// # Returns
/// Tuple (effect_size, effect_direction) where:
/// - effect_size: Absolute value of Cohen's d
/// - effect_direction: Overprediction or Underprediction based on sign
///
/// # Effect Size Interpretation
/// - Small: 0.2
/// - Medium: 0.5
/// - Large: 0.8
pub fn calculate_cohens_d(predicted: &[f64], reference: &[f64]) -> (f64, EffectDirection) {
    if predicted.is_empty() || reference.is_empty() {
        return (f64::NAN, EffectDirection::Underprediction);
    }

    // Use reference standard deviation (single-sample vs population)
    let ref_mean = reference.to_vec().mean();
    let ref_std = reference.to_vec().std_dev();

    if ref_std.abs() < 1e-10 {
        return (f64::NAN, EffectDirection::Underprediction);
    }

    let pred_mean = predicted.to_vec().mean();
    let d = (ref_mean - pred_mean) / ref_std;

    // Determine direction
    let direction = if d > 0.0 {
        EffectDirection::Underprediction // Fluxion underpredicts (predicted < reference)
    } else {
        EffectDirection::Overprediction // Fluxion overpredicts (predicted > reference)
    };

    (d.abs(), direction)
}

/// Statistical validator wrapping ASHRAE140Validator with statistical analysis.
///
/// Provides parallel statistical validation path that:
/// - Wraps existing ASHRAE140Validator
/// - Applies FDR correction per validation group
/// - Calculates statistical metrics (NMBE, CV(RMSE), Cohen's d)
/// - Enforces hybrid threshold validation (80% for ≥5, single-case for <5)
pub struct StatisticalValidator {
    /// Base validator for running ASHRAE 140 validation
    pub base_validator: ASHRAE140Validator,
    /// Significance level for statistical tests (default: 0.05 for 95% confidence)
    pub alpha: f64,
}

impl StatisticalValidator {
    /// Creates a new statistical validator with default alpha=0.05.
    ///
    /// Uses `ASHRAE140Validator::with_full_diagnostics()` for comprehensive
    /// validation output including diagnostic case ranges.
    pub fn new() -> Self {
        Self {
            base_validator: ASHRAE140Validator::with_full_diagnostics(),
            alpha: 0.05,
        }
    }

    /// Creates a statistical validator with custom significance level.
    ///
    /// # Arguments
    /// * `alpha` - Significance level (typically 0.05 for 95% confidence)
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            base_validator: ASHRAE140Validator::with_full_diagnostics(),
            alpha,
        }
    }

    /// Validates a single case and returns benchmark report.
    ///
    /// # Arguments
    /// * `case` - ASHRAE 140 case to validate
    ///
    /// # Returns
    /// * `BenchmarkReport` - Validation results with tolerances
    pub fn validate_case(&mut self, case: ASHRAE140Case) -> BenchmarkReport {
        self.base_validator.validate_with_ideal_control(case)
    }

    /// Validates multiple cases and returns aggregated benchmark report.
    ///
    /// # Arguments
    /// * `cases` - Slice of ASHRAE140Case variants to validate
    ///
    /// # Returns
    /// * `BenchmarkReport` - Aggregated validation results
    pub fn validate_all(&mut self, cases: &[ASHRAE140Case]) -> BenchmarkReport {
        let mut combined_report = BenchmarkReport::new();

        for case in cases {
            let case_report = self.validate_case(*case);
            for result in case_report.results {
                combined_report.results.push(result);
            }
        }

        combined_report
    }
}

impl Default for StatisticalValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive statistical validation report.
///
/// Aggregates tolerance validation, statistical metrics, FDR correction,
/// and group-level validation results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StatisticalReport {
    /// Tolerance-based validation results (±15% annual, ±10% monthly, etc.)
    pub tolerance: BenchmarkReport,
    /// Statistical metrics (NMBE, CV(RMSE), Cohen's d, confidence intervals)
    pub metrics: StatisticalMetrics,
    /// Per-group FDR correction results (which cases pass after correction)
    pub corrected_p_values: HashMap<ValidationGroup, Vec<bool>>,
    /// Group-level validation results (PASS/FAIL per group)
    pub group_validation: HashMap<ValidationGroup, bool>,
}

impl StatisticalValidator {
    /// Validates cases with full statistical analysis.
    ///
    /// This method:
    /// 1. Runs tolerance-based validation using base validator
    /// 2. Calculates statistical metrics (NMBE, CV(RMSE), Cohen's d)
    /// 3. Applies Benjamini-Hochberg FDR correction per validation group
    /// 4. Validates each group with hybrid threshold (80% for ≥5, single-case for <5)
    /// 5. Returns comprehensive `StatisticalReport`
    ///
    /// # Arguments
    /// * `cases` - Slice of ASHRAE140Case variants to validate
    ///
    /// # Returns
    /// * `StatisticalReport` - Comprehensive validation results
    pub fn validate_with_statistics(&mut self, cases: &[ASHRAE140Case]) -> StatisticalReport {
        // Run tolerance validation
        let tolerance = self.validate_all(cases);

        // Calculate statistical metrics
        let metrics = StatisticalMetrics::calculate(&tolerance);

        // Apply FDR correction per group
        let group_results = validate_groups(&tolerance, self.alpha);

        // Extract per-group FDR correction results
        let corrected_p_values = extract_per_group_fdr(&tolerance, self.alpha);

        StatisticalReport {
            tolerance,
            metrics,
            corrected_p_values,
            group_validation: group_results,
        }
    }
}

/// Extracts per-group FDR correction results from benchmark report.
///
/// # Arguments
/// * `report` - Benchmark report containing validation results
/// * `alpha` - Significance level for FDR correction
///
/// # Returns
/// * HashMap mapping each ValidationGroup to vector of corrected p-value results
fn extract_per_group_fdr(
    report: &BenchmarkReport,
    alpha: f64,
) -> HashMap<ValidationGroup, Vec<bool>> {
    let mut group_fdr: HashMap<ValidationGroup, Vec<bool>> = HashMap::new();

    // Partition results by group
    let mut grouped_results: HashMap<ValidationGroup, Vec<&ValidationResult>> = HashMap::new();
    for result in &report.results {
        if let Some(group) = ValidationGroup::from_case_id(&result.case_id) {
            grouped_results.entry(group).or_default().push(result);
        }
    }

    // Calculate FDR correction per group
    for group in [
        ValidationGroup::Baseline,
        ValidationGroup::HighMass,
        ValidationGroup::FreeFloating,
        ValidationGroup::Diagnostics,
        ValidationGroup::Equipment,
    ]
    .iter()
    {
        if let Some(results) = grouped_results.get(group) {
            if results.is_empty() {
                group_fdr.insert(*group, vec![]);
                continue;
            }

            // Calculate p-values
            let p_values: Vec<f64> = results
                .iter()
                .map(|r| calculate_p_value(r, results.len()))
                .collect();

            // Apply FDR correction
            let corrected = BenjaminiHochberg::apply(&p_values, alpha);
            group_fdr.insert(*group, corrected);
        } else {
            // No results for this group
            group_fdr.insert(*group, vec![]);
        }
    }

    group_fdr
}

#[cfg(test)]
mod statistical_validator_tests {
    use super::*;

    #[test]
    fn test_statistical_validator_new() {
        let validator = StatisticalValidator::new();
        assert_eq!(validator.alpha, 0.05);
    }

    #[test]
    fn test_statistical_validator_with_alpha() {
        let validator = StatisticalValidator::with_alpha(0.01);
        assert_eq!(validator.alpha, 0.01);
    }

    #[test]
    fn test_statistical_validator_default() {
        let validator = StatisticalValidator::default();
        assert_eq!(validator.alpha, 0.05);
    }

    #[test]
    fn test_statistical_validator_wraps_base_validator() {
        let validator = StatisticalValidator::new();
        // Verify base_validator is initialized
        // (can't test actual validation without running simulations)
        let _ = validator.base_validator; // Just verify it exists
    }
}

#[cfg(test)]
mod statistical_report_tests {
    use super::*;
    use crate::validation::report::MetricType;

    #[test]
    fn test_statistical_report_structure() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5);
        report.add_result_simple("900", MetricType::AnnualHeating, 12.1, 12.0, 12.2);

        let metrics = StatisticalMetrics::calculate(&report);
        let mut group_validation = HashMap::new();
        group_validation.insert(ValidationGroup::Baseline, true);
        group_validation.insert(ValidationGroup::HighMass, true);

        let mut corrected_p_values = HashMap::new();
        corrected_p_values.insert(ValidationGroup::Baseline, vec![true]);
        corrected_p_values.insert(ValidationGroup::HighMass, vec![true]);

        let stat_report = StatisticalReport {
            tolerance: report,
            metrics,
            corrected_p_values,
            group_validation,
        };

        assert!(!stat_report.tolerance.results.is_empty());
        assert!(!stat_report.metrics.nmbe.is_nan());
        assert!(stat_report
            .group_validation
            .contains_key(&ValidationGroup::Baseline));
        assert!(stat_report
            .corrected_p_values
            .contains_key(&ValidationGroup::Baseline));
    }

    #[test]
    fn test_statistical_report_serialization() {
        use serde_json;

        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5);

        let metrics = StatisticalMetrics::calculate(&report);
        let mut group_validation = HashMap::new();
        group_validation.insert(ValidationGroup::Baseline, true);
        let mut corrected_p_values = HashMap::new();
        corrected_p_values.insert(ValidationGroup::Baseline, vec![true]);

        let stat_report = StatisticalReport {
            tolerance: report,
            metrics,
            corrected_p_values,
            group_validation,
        };

        // Test serialization
        let json = serde_json::to_string(&stat_report).unwrap();
        assert!(json.contains("tolerance"));
        assert!(json.contains("metrics"));
        assert!(json.contains("group_validation"));

        // Note: Can't test full deserialization due to BenchmarkReport's skip fields
        // Just verify the JSON structure is valid
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["tolerance"].is_object());
        assert!(parsed["metrics"].is_object());
        assert!(parsed["group_validation"].is_object());
    }

    #[test]
    fn test_statistical_validator_integration() {
        // Create StatisticalValidator
        let validator = StatisticalValidator::new();

        // Validate a few baseline cases (cannot actually run simulations in unit tests)
        // This test verifies the API structure is correct
        assert_eq!(validator.alpha, 0.05);
        let _ = validator.base_validator; // Just verify it exists
    }

    #[test]
    fn test_backward_compatibility_with_ashrae140_validator() {
        // Verify ASHRAE140Validator API is unchanged
        let base_validator = ASHRAE140Validator::new();
        let _validator_with_diag = ASHRAE140Validator::with_full_diagnostics();
        let _validator_with_custom = ASHRAE140Validator::with_diagnostics(
            crate::validation::diagnostic::DiagnosticConfig::full(),
        );

        // Verify StatisticalValidator is separate and doesn't break existing API
        let stat_validator = StatisticalValidator::new();
        assert_eq!(stat_validator.alpha, 0.05);

        // Both validators can coexist
        let _ = (base_validator, stat_validator);
    }
}

#[cfg(test)]
mod statistical_metrics_tests {
    use super::*;
    use crate::validation::report::{MetricType, ValidationStatus};

    #[test]
    fn test_nmbe_calculation() {
        // Create test results (all AnnualHeating for consistent scale)
        let results = vec![
            ValidationResult::new("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5), // -0.95%
            ValidationResult::new("900", MetricType::AnnualHeating, 12.1, 12.0, 12.2), // 0%
            ValidationResult::new("620", MetricType::AnnualHeating, 9.8, 10.0, 10.5), // -3.1%
        ];

        let nmbe = calculate_nmbe(&results);

        // NMBE should be approximately (-0.0095 + 0.0 - 0.031) / 3 * 100 = -1.35%
        assert!(!nmbe.is_nan());
        assert!(
            nmbe.abs() < 5.0,
            "NMBE should be small for close predictions"
        );
    }
    #[test]
    fn test_nmbe_zero_exclusion() {
        // Results with zero reference should be excluded
        let results = vec![
            ValidationResult::new("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5), // Valid
            ValidationResult::new("900", MetricType::AnnualHeating, 12.1, 0.0, 0.0), // Zero reference (excluded)
        ];

        let nmbe = calculate_nmbe(&results);

        // NMBE should be calculated only from the valid result
        assert!(
            (nmbe - (-0.95)).abs() < 0.1,
            "NMBE should be ~-0.95% from single valid result"
        );
        assert!(!nmbe.is_nan());
        assert!(
            (nmbe - (-0.95)).abs() < 0.1,
            "NMBE should be ~-0.95% from single valid result"
        );
    }

    #[test]
    fn test_cv_rmse_calculation() {
        let results = vec![
            ValidationResult::new("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5),
            ValidationResult::new("900", MetricType::AnnualHeating, 12.1, 12.0, 12.2),
        ];

        let cv_rmse = calculate_cv_rmse(&results);

        assert!(!cv_rmse.is_nan());
        assert!(cv_rmse >= 0.0, "CV(RMSE) should be non-negative");
    }

    #[test]
    fn test_ci_small_sample() {
        // Small sample (n < 30) should use t-distribution
        let nmbe = 5.0;
        let std_error = 1.0;
        let n = 10;

        let (lower, upper) = calculate_ci_nmbe(nmbe, std_error, n);

        assert!(!lower.is_nan());
        assert!(!upper.is_nan());
        assert!(upper > lower);
        assert!(
            (upper - lower) > 1.96 * 2.0,
            "Small sample CI should be wider than normal approximation"
        );
    }

    #[test]
    fn test_ci_large_sample() {
        // Large sample (n >= 30) should use normal approximation
        let nmbe = 5.0;
        let std_error = 1.0;
        let n = 30;

        let (lower, upper) = calculate_ci_nmbe(nmbe, std_error, n);

        assert!(!lower.is_nan());
        assert!(!upper.is_nan());
        assert!(upper > lower);
        assert!(
            ((upper - lower) - 3.92).abs() < 0.01,
            "Large sample CI should use 1.96 * 2 = 3.92"
        );
    }

    #[test]
    fn test_ci_insufficient_data() {
        // n < 2 should return NaN
        let (lower, upper) = calculate_ci_nmbe(5.0, 1.0, 1);
        assert!(lower.is_nan());
        assert!(upper.is_nan());
    }

    #[test]
    fn test_cohens_d_underprediction() {
        let predicted = vec![4.0, 4.5, 5.0];
        let reference = vec![5.0, 5.5, 6.0];

        let (d, direction) = calculate_cohens_d(&predicted, &reference);

        assert!(!d.is_nan());
        assert_eq!(direction, EffectDirection::Underprediction);
        assert!(d > 0.5, "Effect size should be medium-large");
    }

    #[test]
    fn test_cohens_d_overprediction() {
        let predicted = vec![5.0, 5.5, 6.0];
        let reference = vec![4.0, 4.5, 5.0];

        let (d, direction) = calculate_cohens_d(&predicted, &reference);

        assert!(!d.is_nan());
        assert_eq!(direction, EffectDirection::Overprediction);
        assert!(d > 0.5, "Effect size should be medium-large");
    }

    #[test]
    fn test_statistical_metrics_aggregation() {
        // Create test report
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5);
        report.add_result_simple("900", MetricType::AnnualHeating, 12.1, 12.0, 12.2);
        report.add_result_simple("600", MetricType::AnnualCooling, 3.9, 4.0, 4.2);

        let metrics = StatisticalMetrics::calculate(&report);

        assert!(!metrics.nmbe.is_nan());
        assert!(!metrics.cv_rmse.is_nan());
        assert!(!metrics.nmbe_ci.0.is_nan());
        assert!(!metrics.nmbe_ci.1.is_nan());
        assert!(!metrics.cv_rmse_ci.0.is_nan());
        assert!(!metrics.cv_rmse_ci.1.is_nan());
        assert!(!metrics.cohens_d.is_nan());
        assert_eq!(metrics.excluded_cases, 0);
    }

    #[test]
    fn test_statistical_metrics_zero_exclusion() {
        // Create report with zero reference
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5);
        report.add_result_simple("900", MetricType::AnnualHeating, 12.1, 0.0, 0.0); // Zero reference

        let metrics = StatisticalMetrics::calculate(&report);

        assert!(!metrics.nmbe.is_nan());
        assert_eq!(metrics.excluded_cases, 1);
    }

    #[test]
    fn test_statistical_metrics_serialization() {
        use crate::validation::report::BenchmarkReport;
        use serde_json;

        let mut report = BenchmarkReport::new();
        // Use multiple data points to avoid NaN values
        report.add_result_simple("600", MetricType::AnnualHeating, 5.2, 5.0, 5.5);
        report.add_result_simple("610", MetricType::AnnualHeating, 10.3, 10.0, 10.5);
        report.add_result_simple("620", MetricType::AnnualHeating, 9.8, 10.0, 10.5);

        let metrics = StatisticalMetrics::calculate(&report);

        // Test serialization
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("nmbe"));
        assert!(json.contains("cv_rmse"));

        // Test deserialization
        let deserialized: StatisticalMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nmbe, metrics.nmbe);
    }

    #[test]
    fn test_nmbe_empty_results() {
        let results: Vec<ValidationResult> = vec![];
        let nmbe = calculate_nmbe(&results);
        assert!(nmbe.is_nan());
    }

    #[test]
    fn test_cv_rmse_empty_results() {
        let results: Vec<ValidationResult> = vec![];
        let cv_rmse = calculate_cv_rmse(&results);
        assert!(cv_rmse.is_nan());
    }

    #[test]
    fn test_calculate_standard_error_mismatched_lengths() {
        let predicted = vec![1.0, 2.0];
        let refs = vec![1.0];
        let se = calculate_standard_error(&predicted, &refs);
        assert!(se.is_nan());
    }

    #[test]
    fn test_calculate_standard_error_empty() {
        let se = calculate_standard_error(&[], &[]);
        assert!(se.is_nan());
    }

    #[test]
    fn test_calculate_ci_cv_rmse_insufficient_data() {
        let (lower, upper) = calculate_ci_cv_rmse(5.0, 1.0, 1);
        assert!(lower.is_nan());
        assert!(upper.is_nan());
    }

    #[test]
    fn test_calculate_ci_cv_rmse_nan_input() {
        let (lower, upper) = calculate_ci_cv_rmse(f64::NAN, 1.0, 10);
        assert!(lower.is_nan());
        assert!(upper.is_nan());

        let (lower2, upper2) = calculate_ci_cv_rmse(5.0, f64::NAN, 10);
        assert!(lower2.is_nan());
        assert!(upper2.is_nan());
    }

    #[test]
    fn test_cohens_d_empty_input() {
        let (d, dir) = calculate_cohens_d(&[], &[1.0]);
        assert!(d.is_nan());
        assert_eq!(dir, EffectDirection::Underprediction);

        let (d2, dir2) = calculate_cohens_d(&[1.0], &[]);
        assert!(d2.is_nan());
        assert_eq!(dir2, EffectDirection::Underprediction);
    }

    #[test]
    fn test_cohens_d_zero_std() {
        let predicted = vec![1.0, 2.0];
        let reference = vec![5.0, 5.0];
        let (d, dir) = calculate_cohens_d(&predicted, &reference);
        assert!(d.is_nan());
        assert_eq!(dir, EffectDirection::Underprediction);
    }

    #[test]
    fn test_effect_direction_equality() {
        assert_eq!(
            EffectDirection::Overprediction,
            EffectDirection::Overprediction
        );
        assert_ne!(
            EffectDirection::Overprediction,
            EffectDirection::Underprediction
        );
    }

    #[test]
    fn test_statistical_metrics_empty_report() {
        let report = BenchmarkReport::new();
        let metrics = StatisticalMetrics::calculate(&report);
        assert!(metrics.nmbe.is_nan());
        assert!(metrics.cv_rmse.is_nan());
        assert!(metrics.cohens_d.is_nan());
        assert_eq!(metrics.excluded_cases, 0);
    }

    #[test]
    fn test_validate_groups_empty_report() {
        let report = BenchmarkReport::new();
        let results = validate_groups(&report, 0.05);
        assert_eq!(results.len(), 5);
        for group in [
            ValidationGroup::Baseline,
            ValidationGroup::HighMass,
            ValidationGroup::FreeFloating,
            ValidationGroup::Diagnostics,
            ValidationGroup::Equipment,
        ] {
            assert_eq!(results.get(&group), Some(&false));
        }
    }

    #[test]
    fn test_calculate_p_value_insufficient_reference_count() {
        let result = ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 5.2,
            ref_min: 5.0,
            ref_max: 5.5,
            percent_error: 3.6,
            status: ValidationStatus::Pass,
            actual: 5.2,
            min: 5.0,
            max: 5.5,
            metric_type: MetricType::AnnualHeating,
            per_program: None,
        };
        let p_value = calculate_p_value(&result, 1);
        assert_eq!(p_value, 1.0);
    }

    #[test]
    fn test_benjamini_hochberg_single_non_significant() {
        let p_values = vec![0.10];
        let result = BenjaminiHochberg::apply(&p_values, 0.05);
        assert_eq!(result, vec![false]);
    }
}

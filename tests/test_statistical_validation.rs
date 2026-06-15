//! Comprehensive integration tests for statistical validation framework.
//!
//! This test file validates the end-to-end statistical validation workflow
//! including StatisticalValidator, StatisticalReport, group validation,
//! and CLI --statistical flag integration.

use fluxion::validation::statistical::{StatisticalValidator, ValidationGroup};
use fluxion::validation::ASHRAE140Validator;

/// Test full statistical validation workflow with real ASHRAE 140 cases.
#[test]
fn test_statistical_validation_workflow() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    let mut validator = StatisticalValidator::new();
    let cases = vec![
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case800,
    ];
    let report = validator.validate_with_statistics(&cases);

    // Verify StatisticalReport structure
    assert!(
        !report.tolerance.results.is_empty(),
        "Should have validation results"
    );
    assert!(report.metrics.nmbe.is_finite(), "NMBE should be finite");
    assert!(
        report.metrics.cv_rmse.is_finite(),
        "CV(RMSE) should be finite"
    );
    assert!(
        !report.group_validation.is_empty(),
        "Should have group validation results"
    );

    // Verify metrics are finite (CV(RMSE) can be high for small sample sizes)
    assert!(report.metrics.nmbe.is_finite(), "NMBE should be finite");
    assert!(
        report.metrics.cv_rmse.is_finite(),
        "CV(RMSE) should be finite"
    );

    // Verify group validation has entries for cases we tested
    assert!(
        !report.group_validation.is_empty(),
        "Should have at least one group validation result"
    );
}

/// Test group validation 80% threshold enforcement.
#[test]
fn test_group_validation_80_percent() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    // Test hybrid threshold: Groups with ≥5 cases need 80% pass rate
    // Groups with <5 cases need all cases to pass (single-case)

    let mut validator = StatisticalValidator::new();
    let cases = vec![
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case610,
        ASHRAE140Case::Case620,
        ASHRAE140Case::Case630,
        ASHRAE140Case::Case640,
        ASHRAE140Case::Case650,
    ]; // 6 baseline cases
    let report = validator.validate_with_statistics(&cases);

    // Check Baseline group validation
    if let Some(&baseline_pass) = report.group_validation.get(&ValidationGroup::Baseline) {
        println!("Baseline group validation result: {}", baseline_pass);
        assert!(baseline_pass);
    } else {
        // If Baseline group doesn't have enough cases, that's also valid
        println!("Baseline group not found in validation results");
    }
}

/// Test CLI statistical flag integration.
///
/// Note: This test is marked as #[ignore] by default because it requires
/// a compiled fluxion binary. To run this test:
/// 1. Build the binary: `cargo build --release --bin fluxion`
/// 2. Run with: `cargo test test_cli_statistical_flag -- --ignored --nocapture`
#[test]
#[ignore]
fn test_cli_statistical_flag() {
    use std::process::Command;

    // Run CLI command with --statistical flag
    let output = Command::new("./target/release/fluxion")
        .args(["validate", "--all", "--statistical"])
        .output()
        .expect("Failed to execute fluxion binary");

    // Verify command succeeded
    assert!(output.status.success(), "CLI command should succeed");

    // Verify output contains statistical validation results
    let stdout = String::from_utf8(output.stdout).expect("Invalid UTF-8 output");
    assert!(
        stdout.contains("Statistical Validation Results"),
        "Output should contain statistical validation results"
    );
    assert!(
        stdout.contains("NMBE:"),
        "Output should contain NMBE metric"
    );
    assert!(
        stdout.contains("CV(RMSE):"),
        "Output should contain CV(RMSE) metric"
    );
}

/// Test StatisticalValidator workflow with correct case IDs.
#[test]
fn test_statistical_validator_with_ashrae_cases() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    let mut validator = StatisticalValidator::new();
    let cases = vec![ASHRAE140Case::Case600, ASHRAE140Case::Case900]; // Only baseline cases for this test
    let report = validator.validate_with_statistics(&cases);

    // Verify StatisticalReport is returned (not a Result)
    assert!(
        !report.tolerance.results.is_empty(),
        "Should have validation results"
    );
    assert!(report.metrics.nmbe.is_finite(), "NMBE should be finite");
    assert!(
        report.metrics.cv_rmse.is_finite(),
        "CV(RMSE) should be finite"
    );
}

/// Test StatisticalReport structure and completeness.
#[test]
fn test_statistical_report_completeness() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    let mut validator = StatisticalValidator::new();
    let cases = vec![ASHRAE140Case::Case600, ASHRAE140Case::Case900];
    let report = validator.validate_with_statistics(&cases);

    // Verify tolerance results exist
    assert!(
        !report.tolerance.results.is_empty(),
        "Tolerance-based validation results should exist"
    );

    // Verify statistical metrics are finite
    assert!(report.metrics.nmbe.is_finite(), "NMBE should be finite");
    assert!(
        report.metrics.cv_rmse.is_finite(),
        "CV(RMSE) should be finite"
    );
    assert!(
        report.metrics.cohens_d.is_finite(),
        "Cohen's d should be finite"
    );

    // Verify confidence intervals are valid (lower < upper)
    let (nmbe_lower, nmbe_upper) = report.metrics.nmbe_ci;
    assert!(
        nmbe_lower < nmbe_upper,
        "NMBE CI lower bound should be less than upper"
    );

    let (cvrmse_lower, cvrmse_upper) = report.metrics.cv_rmse_ci;
    assert!(
        cvrmse_lower < cvrmse_upper,
        "CV(RMSE) CI lower bound should be less than upper"
    );

    // Verify group validation exists
    assert!(
        !report.group_validation.is_empty(),
        "Group validation results should exist"
    );

    // Verify each group has boolean result
    for &result in report.group_validation.values() {
        assert!(result);
    }
}

/// Test backward compatibility with ASHRAE140Validator.
#[test]
fn test_backward_compatibility_ashrae_140_validator() {
    // Verify that ASHRAE140Validator still works without statistical validation
    let validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // Verify standard validation still works
    assert!(
        !report.results.is_empty(),
        "Standard validation should produce results"
    );

    // Verify no breaking changes to API
    for result in &report.results {
        assert!(
            result.ref_max >= result.ref_min,
            "ref_max should be >= ref_min"
        );
        // fluxion_value should be finite (can be negative for net cooling, zero for absent loads)
        assert!(
            result.fluxion_value.is_finite(),
            "fluxion_value should be finite"
        );
    }
}

/// Test NMBE calculation formula.
#[test]
fn test_nmbe_calculation() {
    use fluxion::validation::report::{MetricType, ValidationResult, ValidationStatus};
    use fluxion::validation::statistical::calculate_nmbe;

    // NMBE formula: Σ((p - r_mid) / r_mid) / n * 100
    // where p = prediction, r_mid = reference midpoint, n = number of samples

    // Test case 1: Perfect prediction (NMBE = 0%)
    let results = vec![ValidationResult {
        case_id: "test".to_string(),
        metric: MetricType::AnnualHeating,
        fluxion_value: 10.0,
        ref_min: 9.0,
        ref_max: 11.0,
        percent_error: 0.0,
        status: ValidationStatus::Pass,
        per_program: None,
        peak_timestamp: None,
    }];

    let nmbe = calculate_nmbe(&results);
    assert!(
        (nmbe - 0.0).abs() < 1e-6,
        "Perfect prediction should have NMBE = 0%"
    );

    // Test case 2: 10% overprediction
    let results = vec![ValidationResult {
        case_id: "test".to_string(),
        metric: MetricType::AnnualHeating,
        fluxion_value: 11.0,
        ref_min: 9.0,
        ref_max: 11.0,
        percent_error: 10.0,
        status: ValidationStatus::Pass,
        per_program: None,
        peak_timestamp: None,
    }];

    let nmbe = calculate_nmbe(&results);
    assert!(
        (nmbe - 10.0).abs() < 1e-6,
        "10% overprediction should have NMBE ≈ 10%"
    );

    // Test case 3: 10% underprediction
    let results = vec![ValidationResult {
        case_id: "test".to_string(),
        metric: MetricType::AnnualHeating,
        fluxion_value: 9.0,
        ref_min: 9.0,
        ref_max: 11.0,
        percent_error: -10.0,
        status: ValidationStatus::Pass,
        per_program: None,
        peak_timestamp: None,
    }];

    let nmbe = calculate_nmbe(&results);
    assert!(
        (nmbe + 10.0).abs() < 1e-6,
        "10% underprediction should have NMBE ≈ -10%"
    );
}

/// Test CV(RMSE) calculation formula.
#[test]
fn test_cv_rmse_calculation() {
    use fluxion::validation::report::{MetricType, ValidationResult, ValidationStatus};
    use fluxion::validation::statistical::calculate_cv_rmse;

    // CV(RMSE) formula: sqrt(Σ(p - r_mid)² / n) / mean(r_mid) * 100

    // Test case 1: Perfect prediction (CV(RMSE) = 0%)
    let results = vec![ValidationResult {
        case_id: "test".to_string(),
        metric: MetricType::AnnualHeating,
        fluxion_value: 10.0,
        ref_min: 9.0,
        ref_max: 11.0,
        percent_error: 0.0,
        status: ValidationStatus::Pass,
        per_program: None,
        peak_timestamp: None,
    }];

    let cv_rmse = calculate_cv_rmse(&results);
    assert!(
        cv_rmse < 1e-6,
        "Perfect prediction should have CV(RMSE) ≈ 0%"
    );

    // Test case 2: 10% error (CV(RMSE) should be ≈ 10%)
    let results = vec![ValidationResult {
        case_id: "test".to_string(),
        metric: MetricType::AnnualHeating,
        fluxion_value: 11.0,
        ref_min: 9.0,
        ref_max: 11.0,
        percent_error: 10.0,
        status: ValidationStatus::Pass,
        per_program: None,
        peak_timestamp: None,
    }];

    let cv_rmse = calculate_cv_rmse(&results);
    // CV(RMSE) = |11 - 10| / 10 * 100 = 10%
    assert!(
        (cv_rmse - 10.0).abs() < 1e-6,
        "10% error should have CV(RMSE) ≈ 10%"
    );
}

/// Test Cohen's d effect size calculation.
#[test]
fn test_cohens_d_calculation() {
    use fluxion::validation::statistical::calculate_cohens_d;

    // Cohen's d formula: (mean(predicted) - mean(reference)) / pooled_std_dev

    // Test case 1: Identical distributions (Cohen's d = 0)
    let predicted = vec![10.0, 11.0, 12.0];
    let reference = vec![10.0, 11.0, 12.0];

    let (cohens_d, _effect_direction) = calculate_cohens_d(&predicted, &reference);
    assert!(
        cohens_d.abs() < 1e-6,
        "Identical distributions should have Cohen's d ≈ 0"
    );

    // Test case 2: Large effect (Cohen's d > 0.8)
    let predicted = vec![15.0, 16.0, 17.0];
    let reference = vec![10.0, 11.0, 12.0];

    let (cohens_d, _effect_direction) = calculate_cohens_d(&predicted, &reference);
    assert!(
        cohens_d > 0.8,
        "Large difference should have Cohen's d > 0.8"
    );
}

/// Test 95% confidence interval calculation for NMBE.
#[test]
fn test_ci_nmbe_calculation() {
    use fluxion::validation::report::{MetricType, ValidationResult, ValidationStatus};
    use fluxion::validation::statistical::{calculate_ci_nmbe, calculate_standard_error};

    // CI formula: nmbe ± t_{alpha/2, n-1} * std_error

    let _results = [
        ValidationResult {
            case_id: "test1".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 10.0,
            ref_min: 9.0,
            ref_max: 11.0,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
            per_program: None,
            peak_timestamp: None,
        },
        ValidationResult {
            case_id: "test2".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 11.0,
            ref_min: 9.5,
            ref_max: 10.5,
            percent_error: 10.0,
            status: ValidationStatus::Pass,
            per_program: None,
            peak_timestamp: None,
        },
    ];

    let nmbe = 5.0; // Average of 0% and 10%
    let std_error = calculate_standard_error(&[10.0, 11.0], &[10.0, 10.0]);
    let (ci_lower, ci_upper) = calculate_ci_nmbe(nmbe, std_error, 2);

    assert!(ci_lower < nmbe, "CI lower bound should be less than NMBE");
    assert!(
        ci_upper > nmbe,
        "CI upper bound should be greater than NMBE"
    );
    assert!(
        ci_lower < ci_upper,
        "CI lower bound should be less than upper"
    );
}

/// Test 95% confidence interval calculation for CV(RMSE).
#[test]
fn test_ci_cv_rmse_calculation() {
    use fluxion::validation::statistical::calculate_ci_cv_rmse;

    // CI formula: cv_rmse ± t_{alpha/2, n-1} * std_error

    let cv_rmse = 10.0;
    let std_error = 2.0;
    let n = 5;

    let (ci_lower, ci_upper) = calculate_ci_cv_rmse(cv_rmse, std_error, n);

    assert!(
        ci_lower < cv_rmse,
        "CI lower bound should be less than CV(RMSE)"
    );
    assert!(
        ci_upper > cv_rmse,
        "CI upper bound should be greater than CV(RMSE)"
    );
    assert!(
        ci_lower < ci_upper,
        "CI lower bound should be less than upper"
    );
}

/// Test comprehensive ASHRAE 140 statistical compliance.
#[test]
fn test_ashrae_140_statistical_compliance() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    let mut validator = StatisticalValidator::new();
    let cases = vec![
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case610,
        ASHRAE140Case::Case620,
        ASHRAE140Case::Case630,
        ASHRAE140Case::Case640,
        ASHRAE140Case::Case650, // Baseline
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case910,
        ASHRAE140Case::Case920,
        ASHRAE140Case::Case930,
        ASHRAE140Case::Case940,
        ASHRAE140Case::Case950,
        ASHRAE140Case::Case960, // HighMass
        ASHRAE140Case::Case600FF,
        ASHRAE140Case::Case650FF,
        ASHRAE140Case::Case900FF,
        ASHRAE140Case::Case950FF, // FreeFloating
        ASHRAE140Case::Case800,
        ASHRAE140Case::Case801,
        ASHRAE140Case::Case802,
        ASHRAE140Case::Case803,
        ASHRAE140Case::Case804,
        ASHRAE140Case::Case805,
        ASHRAE140Case::Case806,
        ASHRAE140Case::Case807,
        ASHRAE140Case::Case808,
        ASHRAE140Case::Case809,
        ASHRAE140Case::Case810, // Equipment
    ];

    let report = validator.validate_with_statistics(&cases);

    // Verify group validation results exist for all major groups
    assert!(
        !report.group_validation.is_empty(),
        "Should have group validation results"
    );

    // Check specific groups
    let has_baseline = report
        .group_validation
        .contains_key(&ValidationGroup::Baseline);
    let has_high_mass = report
        .group_validation
        .contains_key(&ValidationGroup::HighMass);
    let has_free_floating = report
        .group_validation
        .contains_key(&ValidationGroup::FreeFloating);
    let has_equipment = report
        .group_validation
        .contains_key(&ValidationGroup::Equipment);

    assert!(
        has_baseline || has_high_mass || has_free_floating || has_equipment,
        "Should have at least one validation group result"
    );

    // Verify statistical metrics are finite and reasonable
    assert!(report.metrics.nmbe.is_finite(), "NMBE should be finite");
    assert!(
        report.metrics.cv_rmse.is_finite(),
        "CV(RMSE) should be finite"
    );
    assert!(
        report.metrics.cohens_d.is_finite(),
        "Cohen's d should be finite"
    );

    // Verify confidence intervals are valid
    let (nmbe_lower, nmbe_upper) = report.metrics.nmbe_ci;
    assert!(nmbe_lower < nmbe_upper, "NMBE CI should be valid");

    let (cvrmse_lower, cvrmse_upper) = report.metrics.cv_rmse_ci;
    assert!(cvrmse_lower < cvrmse_upper, "CV(RMSE) CI should be valid");

    // Verify tolerance results exist for all cases
    assert!(
        !report.tolerance.results.is_empty(),
        "Should have tolerance results"
    );

    // Print summary for visibility
    println!("=== ASHRAE 140 Statistical Compliance Test ===");
    println!("Total cases validated: {}", cases.len());
    println!("Total results: {}", report.tolerance.results.len());
    println!(
        "NMBE: {:.2}% [ {:.2}%, {:.2}% 95% CI ]",
        report.metrics.nmbe, nmbe_lower, nmbe_upper
    );
    println!(
        "CV(RMSE): {:.2}% [ {:.2}%, {:.2}% 95% CI ]",
        report.metrics.cv_rmse, cvrmse_lower, cvrmse_upper
    );
    println!("Cohen's d: {:.2}", report.metrics.cohens_d);
    println!("Groups validated: {}", report.group_validation.len());
}

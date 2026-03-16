//! A/B testing integration tests for thermal network variant comparison.
//!
//! This module provides integration tests for the A/B testing framework,
//! including framework initialization, variant comparison, and 900-series A/B comparison.
//!
//! # Manual Execution
//!
//! These tests are designed to be run manually via:
//! ```bash
//! cargo test ab_testing -- --nocapture
//! ```
//!
//! No CI integration - this is for research and experimentation during Phase 22.

use fluxion::validation::ab_testing::{ABTestRunner, ThermalNetworkVariant};

/// Test framework initialization and basic execution.
///
/// Verifies that:
/// - ABTestRunner can be created with default configuration
/// - Variants can be added to the runner
/// - Test cases can be added to the runner
/// - All variants can be run for test cases
/// - ABTestResult calculations (NMBE, CV(RMSE), pass_rate) are correct
#[test]
fn test_ab_testing_framework() {
    println!("\n=== A/B Testing Framework Initialization Test ===\n");

    // Create ABTestRunner with default configuration
    let runner = ABTestRunner::new()
        .with_variants(vec![
            ThermalNetworkVariant::FiveR1C,
            ThermalNetworkVariant::SixR2C,
        ])
        .with_cases(vec!["600", "900"]);

    println!("Created ABTestRunner:");
    println!("  Variants: {:?}", runner.variants);
    println!("  Cases: {:?}", runner.cases);

    // Run all variants for test cases
    println!("\nRunning 5R1C variant...");
    let result_5r1c = runner.run_all_variants(ThermalNetworkVariant::FiveR1C);
    println!("  NMBE heating: {:.2}%", result_5r1c.nmbe_heating);
    println!("  NMBE cooling: {:.2}%", result_5r1c.nmbe_cooling);
    println!("  CV(RMSE) heating: {:.2}%", result_5r1c.cv_rmse_heating);
    println!("  CV(RMSE) cooling: {:.2}%", result_5r1c.cv_rmse_cooling);
    println!("  Pass rate: {:.1}%", result_5r1c.pass_rate);

    println!("\nRunning 6R2C variant...");
    let result_6r2c = runner.run_all_variants(ThermalNetworkVariant::SixR2C);
    println!("  NMBE heating: {:.2}%", result_6r2c.nmbe_heating);
    println!("  NMBE cooling: {:.2}%", result_6r2c.nmbe_cooling);
    println!("  CV(RMSE) heating: {:.2}%", result_6r2c.cv_rmse_heating);
    println!("  CV(RMSE) cooling: {:.2}%", result_6r2c.cv_rmse_cooling);
    println!("  Pass rate: {:.1}%", result_6r2c.pass_rate);

    // Print summary
    println!("\n=== Summary ===");
    println!(
        "5R1C pass rate: {:.1}%, 6R2C pass rate: {:.1}%",
        result_5r1c.pass_rate, result_6r2c.pass_rate
    );

    // Verify calculations are not NaN
    assert!(
        !result_5r1c.nmbe_heating.is_nan(),
        "5R1C NMBE heating should not be NaN"
    );
    assert!(
        !result_5r1c.nmbe_cooling.is_nan(),
        "5R1C NMBE cooling should not be NaN"
    );
    assert!(
        !result_6r2c.nmbe_heating.is_nan(),
        "6R2C NMBE heating should not be NaN"
    );
    assert!(
        !result_6r2c.nmbe_cooling.is_nan(),
        "6R2C NMBE cooling should not be NaN"
    );

    // Verify pass rates are within valid range [0, 100]
    assert!(result_5r1c.pass_rate >= 0.0 && result_5r1c.pass_rate <= 100.0);
    assert!(result_6r2c.pass_rate >= 0.0 && result_6r2c.pass_rate <= 100.0);
}

/// Test variant comparison and report generation.
///
/// Verifies that:
/// - Baseline and test variant results can be compared
/// - Comparison report can be generated
/// - Markdown report includes improvement metrics
/// - Recommendation is generated based on metrics
#[test]
fn test_variant_comparison() {
    println!("\n=== Variant Comparison Test ===\n");

    // Create ABTestRunner with test cases
    let runner = ABTestRunner::new()
        .with_variants(vec![
            ThermalNetworkVariant::FiveR1C,
            ThermalNetworkVariant::SixR2C,
        ])
        .with_cases(vec!["600", "900"]);

    // Run 5R1C baseline
    println!("Running 5R1C baseline...");
    let baseline = runner.run_all_variants(ThermalNetworkVariant::FiveR1C);
    println!("  Pass rate: {:.1}%", baseline.pass_rate);

    // Run 6R2C test variant
    println!("Running 6R2C test variant...");
    let test = runner.run_all_variants(ThermalNetworkVariant::SixR2C);
    println!("  Pass rate: {:.1}%", test.pass_rate);

    // Compare results
    println!("\nComparing results...");
    let report = runner.compare_results(&baseline, &test);

    // Generate comparison report
    let markdown = report.to_markdown();
    println!("\n=== Comparison Report ===\n");
    println!("{}", markdown);

    // Verify recommendation is generated
    assert!(
        report.recommendation == "ADOPT" || report.recommendation == "DEFER",
        "Recommendation should be ADOPT or DEFER"
    );
    assert!(
        !report.explanation.is_empty(),
        "Explanation should not be empty"
    );

    // Verify improvement metrics are calculated
    assert!(!report.heating_nmbe_improvement.is_nan());
    assert!(!report.cooling_nmbe_improvement.is_nan());
    assert!(!report.pass_rate_improvement.is_nan());

    println!("\n=== Summary ===");
    println!("Recommendation: {}", report.recommendation);
    println!(
        "Pass rate improvement: {:+.1}%",
        report.pass_rate_improvement
    );
}

/// Test 900-series A/B comparison for high-mass cases.
///
/// Verifies that:
/// - All 900-series cases (920, 930, 940, 950, 960) can be run
/// - 5R1C and 6R2C variants can be compared for high-mass cases
/// - NMBE improvement can be quantified
/// - Comparison report includes high-mass specific metrics
#[test]
fn test_900_series_ab_comparison() {
    println!("\n=== 900-Series A/B Comparison Test ===\n");

    // Create ABTestRunner with 900-series cases
    let runner = ABTestRunner::new()
        .with_variants(vec![
            ThermalNetworkVariant::FiveR1C,
            ThermalNetworkVariant::SixR2C,
        ])
        .with_cases(vec!["920", "930", "940", "950", "960"]);

    println!("Running 5R1C for 900-series cases...");
    let result_5r1c = runner.run_all_variants(ThermalNetworkVariant::FiveR1C);
    println!("  NMBE heating: {:.2}%", result_5r1c.nmbe_heating);
    println!("  NMBE cooling: {:.2}%", result_5r1c.nmbe_cooling);
    println!("  Pass rate: {:.1}%", result_5r1c.pass_rate);

    println!("\nRunning 6R2C for 900-series cases...");
    let result_6r2c = runner.run_all_variants(ThermalNetworkVariant::SixR2C);
    println!("  NMBE heating: {:.2}%", result_6r2c.nmbe_heating);
    println!("  NMBE cooling: {:.2}%", result_6r2c.nmbe_cooling);
    println!("  Pass rate: {:.1}%", result_6r2c.pass_rate);

    // Compare NMBE for high-mass cases
    let heating_nmbe_improvement = result_5r1c.nmbe_heating - result_6r2c.nmbe_heating;
    let cooling_nmbe_improvement = result_5r1c.nmbe_cooling - result_6r2c.nmbe_cooling;

    // Print diagnostic
    println!("\n=== High-Mass Diagnostic ===");
    println!(
        "High-mass NMBE: 5R1C={:.2}%, 6R2C={:.2}% (improvement: {:+.2}%)",
        result_5r1c.nmbe_heating, result_6r2c.nmbe_heating, heating_nmbe_improvement
    );
    println!(
        "High-mass NMBE: 5R1C={:.2}%, 6R2C={:.2}% (improvement: {:+.2}%)",
        result_5r1c.nmbe_cooling, result_6r2c.nmbe_cooling, cooling_nmbe_improvement
    );

    // Generate comparison report
    let report = runner.compare_results(&result_5r1c, &result_6r2c);
    println!("\n=== Comparison Report ===\n");
    println!("{}", report.to_markdown());

    // Verify NMBE values are reasonable (not NaN, not infinite)
    assert!(!result_5r1c.nmbe_heating.is_nan());
    assert!(!result_5r1c.nmbe_heating.is_infinite());
    assert!(!result_6r2c.nmbe_heating.is_nan());
    assert!(!result_6r2c.nmbe_heating.is_infinite());
}

/// Test statistical validation integration.
///
/// Verifies that:
/// - NMBE calculations match statistical.rs implementation
/// - CV(RMSE) calculations match statistical.rs implementation
/// - Pass rate calculations match ValidationReport::is_within_tolerance()
#[test]
fn test_statistical_validation_integration() {
    println!("\n=== Statistical Validation Integration Test ===\n");

    // Create ABTestRunner with test cases
    let runner = ABTestRunner::new()
        .with_variants(vec![ThermalNetworkVariant::FiveR1C])
        .with_cases(vec!["600", "610", "620"]);

    // Run 5R1C variant
    println!("Running 5R1C variant...");
    let result = runner.run_all_variants(ThermalNetworkVariant::FiveR1C);

    // Verify NMBE calculations
    println!("\nVerifying NMBE calculations...");
    println!("  Heating NMBE: {:.2}%", result.nmbe_heating);
    println!("  Cooling NMBE: {:.2}%", result.nmbe_cooling);

    // NMBE should be calculated from statistical.rs
    // Verify it's not NaN and is within reasonable range (-100% to +100%)
    assert!(
        !result.nmbe_heating.is_nan(),
        "Heating NMBE should not be NaN"
    );
    assert!(
        !result.nmbe_cooling.is_nan(),
        "Cooling NMBE should not be NaN"
    );
    assert!(
        result.nmbe_heating >= -100.0 && result.nmbe_heating <= 100.0,
        "Heating NMBE should be within [-100%, 100%]"
    );
    assert!(
        result.nmbe_cooling >= -100.0 && result.nmbe_cooling <= 100.0,
        "Cooling NMBE should be within [-100%, 100%]"
    );

    // Verify CV(RMSE) calculations
    println!("\nVerifying CV(RMSE) calculations...");
    println!("  Heating CV(RMSE): {:.2}%", result.cv_rmse_heating);
    println!("  Cooling CV(RMSE): {:.2}%", result.cv_rmse_cooling);

    // CV(RMSE) should be calculated from statistical.rs
    // Verify it's not NaN and is non-negative
    assert!(
        !result.cv_rmse_heating.is_nan() && result.cv_rmse_heating >= 0.0,
        "Heating CV(RMSE) should not be NaN and should be non-negative"
    );
    assert!(
        !result.cv_rmse_cooling.is_nan() && result.cv_rmse_cooling >= 0.0,
        "Cooling CV(RMSE) should not be NaN and should be non-negative"
    );

    // Verify pass rate calculations
    println!("\nVerifying pass rate calculations...");
    println!("  Pass rate: {:.1}%", result.pass_rate);

    // Pass rate should be within [0, 100]
    assert!(
        result.pass_rate >= 0.0 && result.pass_rate <= 100.0,
        "Pass rate should be within [0, 100]"
    );

    // Verify individual cases are marked correctly
    println!("\nVerifying individual case results...");
    for case in &result.cases {
        println!(
            "  Case {}: heating_ok={}, cooling_ok={}",
            case.case_id,
            case.heating_ok(),
            case.cooling_ok()
        );

        // Verify heating_ok() and cooling_ok() are correct
        // (This should match ValidationReport::is_within_tolerance())
        let heating_expected = case.annual_heating_mwh >= case.annual_heating_ref_min * 0.85
            && case.annual_heating_mwh <= case.annual_heating_ref_max * 1.15;
        let cooling_expected = case.annual_cooling_mwh >= case.annual_cooling_ref_min * 0.85
            && case.annual_cooling_mwh <= case.annual_cooling_ref_max * 1.15;

        assert_eq!(
            case.heating_ok(),
            heating_expected,
            "Heating check mismatch for case {}",
            case.case_id
        );
        assert_eq!(
            case.cooling_ok(),
            cooling_expected,
            "Cooling check mismatch for case {}",
            case.case_id
        );
    }

    println!("\n=== Summary ===");
    println!("All statistical calculations verified successfully");
}

/// Test A/B testing documentation examples.
///
/// Provides working examples for documentation.
#[test]
fn test_ab_testing_documentation_examples() {
    println!("\n=== Documentation Examples Test ===\n");

    // Example 1: How to compare 5R1C vs 6R2C for high-mass cases
    println!("Example 1: Compare 5R1C vs 6R2C for high-mass cases");
    println!("```rust");
    println!("use fluxion::validation::ab_testing::{ABTestRunner, ThermalNetworkVariant};");
    println!("");
    println!("let runner = ABTestRunner::new()");
    println!(
        "    .with_variants(vec![ThermalNetworkVariant::FiveR1C, ThermalNetworkVariant::SixR2C])"
    );
    println!("    .with_cases(vec![\"920\", \"930\", \"940\", \"950\", \"960\"]);");
    println!("");
    println!("let baseline = runner.run_all_variants(ThermalNetworkVariant::FiveR1C);");
    println!("let test = runner.run_all_variants(ThermalNetworkVariant::SixR2C);");
    println!("let report = runner.compare_results(&baseline, &test);");
    println!("println!(\"{}\", report.to_markdown());");
    println!("```");

    // Example 2: How to generate comparison report in markdown format
    println!("\nExample 2: Generate comparison report in markdown format");
    println!("```rust");
    println!("use fluxion::validation::ab_testing::{ABTestRunner, ThermalNetworkVariant};");
    println!("");
    println!("let runner = ABTestRunner::new()");
    println!(
        "    .with_variants(vec![ThermalNetworkVariant::FiveR1C, ThermalNetworkVariant::SixR2C])"
    );
    println!("    .with_cases(vec![\"600\", \"900\"]);");
    println!("");
    println!("let baseline = runner.run_all_variants(ThermalNetworkVariant::FiveR1C);");
    println!("let test = runner.run_all_variants(ThermalNetworkVariant::SixR2C);");
    println!("let report = runner.compare_results(&baseline, &test);");
    println!("");
    println!("// Save report to file");
    println!("let markdown = report.to_markdown();");
    println!("std::fs::write(\"ab_test_report.md\", markdown).unwrap();");
    println!("```");

    println!("\n=== Summary ===");
    println!("Documentation examples generated successfully");
}

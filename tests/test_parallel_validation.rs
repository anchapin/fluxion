//! Test infrastructure for Phase 6 performance optimization verification.
//! This test verifies BATCH-01: Parallel execution of ASHRAE validation cases.

use fluxion::validation::ASHRAE140Validator;
use std::time::Instant;

#[test]
fn test_parallel_validation_execution() {
    // This test verifies that all 18+ ASHRAE 140 cases can be validated in parallel
    // and complete within a reasonable time threshold (< 5 minutes for initial version,
    // will be optimized further in Plan 06-02).
    //
    // BATCH-01 Requirement: All 18+ ASHRAE 140 cases executed in parallel with rayon

    let mut validator = ASHRAE140Validator::new();

    // Record start time
    let start = Instant::now();

    // Run full validation - this should execute cases in parallel ( rayon::par_iter )
    let report = validator.validate_analytical_engine();

    // Record duration
    let duration = start.elapsed();
    let duration_secs = duration.as_secs() as f64 + duration.subsec_nanos() as f64 / 1e9;

    println!("Validation completed in {:.2} seconds", duration_secs);
    println!("Pass rate: {:.1}%", report.pass_rate());
    println!("MAE: {:.2}%", report.mae());
    println!("Max deviation: {:.2}%", report.max_deviation());

    // Verify that we have results for all expected cases
    // We should have at least 18 cases, each with multiple metrics (heating, cooling, peak, free-float)
    assert!(
        !report.results.is_empty(),
        "Validation results should not be empty"
    );
    assert!(
        report.results.len() >= 18,
        "Should have at least 18 validation results"
    );

    // Verify that the report contains benchmark data
    assert!(
        !report.benchmark_data.is_empty(),
        "Benchmark data should be populated"
    );

    // Timing assertion: Initial threshold < 300 seconds (5 minutes)
    // This will be tightened in later optimization plans (target < 5 minutes for all cases)
    assert!(
        duration_secs < 300.0,
        "Validation took too long: {:.2}s, expected < 300s",
        duration_secs
    );

    // Verify that we have results for all expected case IDs
    let expected_case_ids = [
        "600", "610", "620", "630", "640", "650", "600FF", "650FF", "900", "910", "920", "930",
        "940", "950", "900FF", "950FF", "960", "195",
    ];

    for case_id in &expected_case_ids {
        let case_results: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.case_id == *case_id)
            .collect();
        assert!(
            !case_results.is_empty(),
            "Case {} should have validation results",
            case_id
        );
    }

    // Verify that pass rate is a valid percentage
    let pass_rate = report.pass_rate();
    assert!(
        pass_rate >= 0.0 && pass_rate <= 100.0,
        "Pass rate should be 0-100%"
    );

    // Basic sanity check: MAE and max deviation should be non-negative
    assert!(report.mae() >= 0.0, "MAE should be non-negative");
    assert!(
        report.max_deviation() >= 0.0,
        "Max deviation should be non-negative"
    );
}

#[test]
fn test_result_aggregation_basics() {
    // Sanity check that the report aggregation functions work correctly
    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // All aggregation methods should return valid numbers
    let pass_rate = report.pass_rate();
    let mae = report.mae();
    let max_dev = report.max_deviation();

    // These should not panic and should be finite
    assert!(pass_rate.is_finite(), "Pass rate should be finite");
    assert!(mae.is_finite(), "MAE should be finite");
    assert!(max_dev.is_finite(), "Max deviation should be finite");
}

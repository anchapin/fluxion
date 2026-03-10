//! Test infrastructure for BenchmarkReport performance metrics and aggregation.
//! This test validates Phase 6 verification requirements for BenchmarkReport.

use fluxion::validation::benchmark::{get_all_benchmark_data, get_benchmark_data};
use fluxion::validation::report::{
    BenchmarkReport, MetricType, ValidationResult, ValidationStatus,
};

#[test]
fn test_benchmark_report_with_real_data() {
    // Load real benchmark data and create a simple report to ensure APIs work
    let benchmark_data = get_all_benchmark_data();
    assert!(!benchmark_data.is_empty());
    assert!(benchmark_data.contains_key("600"));
    assert!(benchmark_data.contains_key("900"));
    assert!(benchmark_data.contains_key("960"));
    assert!(benchmark_data.contains_key("195"));

    // Verify Case 600 data
    let case_600 = get_benchmark_data("600").unwrap();
    assert!(case_600.annual_heating_min > 0.0);
    assert!(case_600.annual_heating_max > case_600.annual_heating_min);
}

#[test]
fn test_benchmark_report_aggregation_with_synthetic_data() {
    // Create a report with synthetic results and verify aggregation
    let mut report = BenchmarkReport::new();

    // Add a mix of pass, warning, fail results
    let test_cases = vec![
        // Pass
        ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 6.5,
            ref_min: 5.5,
            ref_max: 7.5,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
        },
        // Pass
        ValidationResult {
            case_id: "600".to_string(),
            metric: MetricType::AnnualCooling,
            fluxion_value: 9.25,
            ref_min: 8.0,
            ref_max: 10.5,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
        },
        // Warning (within range but high deviation)
        ValidationResult {
            case_id: "900".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 2.5,
            ref_min: 1.17,
            ref_max: 2.04,
            percent_error: 50.0,
            status: ValidationStatus::Warning,
        },
        // Fail
        ValidationResult {
            case_id: "900".to_string(),
            metric: MetricType::AnnualCooling,
            fluxion_value: 6.0,
            ref_min: 2.13,
            ref_max: 3.67,
            percent_error: 100.0,
            status: ValidationStatus::Fail,
        },
    ];

    for result in test_cases {
        report.add_result(result);
    }

    // Verify aggregation
    assert_eq!(report.results.len(), 4);
    let pass_count = report.results.iter().filter(|r| r.passed()).count();
    assert_eq!(pass_count, 2);
    assert_eq!(report.warning_count(), 1);
    assert_eq!(report.fail_count(), 1);
    assert!((report.pass_rate() - 50.0).abs() < 0.01);

    // MAE should be computed correctly
    let mae = report.mae();
    assert!(mae.is_finite() && mae >= 0.0);

    // Max deviation
    let max_dev = report.max_deviation();
    assert!((max_dev - 100.0).abs() < 0.01);
}

// Note: Phase 6 will add performance metrics (timing, throughput) to BenchmarkReport.
// Future tests (after implementation) will verify these fields:
// - Total validation time
// - Per-case timing
// - Throughput (cases/second)
// - Peak memory usage (if tracked)
// These tests will be added in Plans 06-02 through 06-05.

#[test]
#[ignore] // Placeholder for future performance metrics validation
fn test_benchmark_report_performance_metrics() {
    // After Plan 06-03: Verify that BenchmarkReport includes timing information
    // let report = BenchmarkReport::new();
    // assert!(report.total_execution_time_ms > 0);
    // assert!(report.per_case_timings.is_some());
    // This test will be completed when the feature is implemented.
}

#[test]
#[ignore] // Placeholder for throughput measurement
fn test_validation_throughput_requirement() {
    // After Plan 06-02: Verify that complete validation suite executes in < 5 minutes (300 sec)
    // This translates to throughput of 18+ cases / 300 sec = ~0.06 cases/sec minimum,
    // but our target is much higher (> 10 cases/sec on 8-core).
    // let mut validator = ASHRAE140Validator::new();
    // let report = validator.validate_analytical_engine();
    // assert!(report.total_execution_time_ms < 300000);
    // This test will be completed when timing instrumentation is added.
}

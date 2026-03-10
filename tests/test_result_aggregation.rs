//! Unit tests for BenchmarkReport aggregation logic.
//! This test verifies BATCH-02: Aggregated validation results collection and summarization.

use fluxion::validation::report::{
    BenchmarkReport, MetricType, ValidationResult, ValidationStatus,
};

#[test]
fn test_pass_rate_empty() {
    // Edge case: empty results should return 100% pass rate
    let report = BenchmarkReport {
        results: vec![],
        benchmark_data: std::collections::HashMap::new(),
    };
    assert_eq!(report.pass_rate(), 100.0);
}

#[test]
fn test_pass_rate_all_passed() {
    // All results passed
    let report = BenchmarkReport {
        results: vec![
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualHeating,
                fluxion_value: 6.0,
                ref_min: 5.5,
                ref_max: 7.5,
                percent_error: 0.0,
                status: ValidationStatus::Pass,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualCooling,
                fluxion_value: 9.0,
                ref_min: 8.0,
                ref_max: 10.5,
                percent_error: 0.0,
                status: ValidationStatus::Pass,
            },
        ],
        benchmark_data: std::collections::HashMap::new(),
    };
    assert_eq!(report.pass_rate(), 100.0);
}

#[test]
fn test_pass_rate_all_failed() {
    // All results failed
    let report = BenchmarkReport {
        results: vec![
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualHeating,
                fluxion_value: 20.0, // Way outside range
                ref_min: 5.5,
                ref_max: 7.5,
                percent_error: 200.0,
                status: ValidationStatus::Fail,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualCooling,
                fluxion_value: 30.0, // Way outside range
                ref_min: 8.0,
                ref_max: 10.5,
                percent_error: 250.0,
                status: ValidationStatus::Fail,
            },
        ],
        benchmark_data: std::collections::HashMap::new(),
    };
    assert_eq!(report.pass_rate(), 0.0);
}

#[test]
fn test_pass_rate_mixed() {
    // Mixed pass/fail/warning
    let report = BenchmarkReport {
        results: vec![
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualHeating,
                fluxion_value: 6.0,
                ref_min: 5.5,
                ref_max: 7.5,
                percent_error: 0.0,
                status: ValidationStatus::Pass,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualCooling,
                fluxion_value: 9.0,
                ref_min: 8.0,
                ref_max: 10.5,
                percent_error: 0.0,
                status: ValidationStatus::Pass,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::PeakHeating,
                fluxion_value: 20.0,
                ref_min: 2.8,
                ref_max: 3.8,
                percent_error: 500.0,
                status: ValidationStatus::Fail,
            },
        ],
        benchmark_data: std::collections::HashMap::new(),
    };
    // 2 passed out of 3 = 66.67%
    assert!((report.pass_rate() - 66.67).abs() < 0.01);
}

#[test]
fn test_mae_empty() {
    let report = BenchmarkReport {
        results: vec![],
        benchmark_data: std::collections::HashMap::new(),
    };
    assert_eq!(report.mae(), 0.0);
}

#[test]
fn test_mae_simple() {
    // Test MAE calculation with known values
    let report = BenchmarkReport {
        results: vec![
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualHeating,
                fluxion_value: 5.0,
                ref_min: 5.0,
                ref_max: 10.0,
                percent_error: ((5.0 - 7.5) / 7.5) * 100.0, // -33.33%
                status: ValidationStatus::Fail,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualCooling,
                fluxion_value: 12.0,
                ref_min: 10.0,
                ref_max: 15.0,
                percent_error: ((12.0 - 12.5) / 12.5) * 100.0, // -4.0%
                status: ValidationStatus::Pass,
            },
        ],
        benchmark_data: std::collections::HashMap::new(),
    };
    let mae = report.mae();
    // MAE = (33.33% + 4.0%) / 2 = 18.665%
    assert!((mae - 18.665).abs() < 0.01);
}

#[test]
fn test_max_deviation_empty() {
    let report = BenchmarkReport {
        results: vec![],
        benchmark_data: std::collections::HashMap::new(),
    };
    // For empty results, max_deviation uses fold(0.0, max), so returns 0.0
    assert_eq!(report.max_deviation(), 0.0);
}

#[test]
fn test_max_deviation_simple() {
    let report = BenchmarkReport {
        results: vec![
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualHeating,
                fluxion_value: 5.0,
                ref_min: 5.0,
                ref_max: 10.0,
                percent_error: -33.33,
                status: ValidationStatus::Fail,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualCooling,
                fluxion_value: 12.0,
                ref_min: 10.0,
                ref_max: 15.0,
                percent_error: -4.0,
                status: ValidationStatus::Pass,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::PeakHeating,
                fluxion_value: 3.5,
                ref_min: 2.8,
                ref_max: 3.8,
                percent_error: 15.0, // positive
                status: ValidationStatus::Warning,
            },
        ],
        benchmark_data: std::collections::HashMap::new(),
    };
    let max_dev = report.max_deviation();
    // Max of abs(33.33, 4.0, 15.0) = 33.33
    assert!((max_dev - 33.33).abs() < 0.01);
}

#[test]
fn test_fail_count() {
    let report = BenchmarkReport {
        results: vec![
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualHeating,
                fluxion_value: 20.0,
                ref_min: 5.5,
                ref_max: 7.5,
                percent_error: 200.0,
                status: ValidationStatus::Fail,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualCooling,
                fluxion_value: 9.0,
                ref_min: 8.0,
                ref_max: 10.5,
                percent_error: -5.0,
                status: ValidationStatus::Pass,
            },
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::PeakHeating,
                fluxion_value: 3.5,
                ref_min: 2.8,
                ref_max: 3.8,
                percent_error: 10.0,
                status: ValidationStatus::Warning,
            },
        ],
        benchmark_data: std::collections::HashMap::new(),
    };
    assert_eq!(report.fail_count(), 1);
    assert_eq!(report.warning_count(), 1);
    let pass_count = report.results.iter().filter(|r| r.passed()).count();
    assert_eq!(pass_count, 1);
}

#[test]
fn test_worst_cases() {
    let report = BenchmarkReport {
        results: vec![
            ValidationResult {
                case_id: "600".to_string(),
                metric: MetricType::AnnualHeating,
                fluxion_value: 20.0,
                ref_min: 5.5,
                ref_max: 7.5,
                percent_error: 200.0,
                status: ValidationStatus::Fail,
            },
            ValidationResult {
                case_id: "900".to_string(),
                metric: MetricType::AnnualCooling,
                fluxion_value: 2.0,
                ref_min: 2.13,
                ref_max: 3.67,
                percent_error: -15.0,
                status: ValidationStatus::Warning,
            },
            ValidationResult {
                case_id: "195".to_string(),
                metric: MetricType::PeakHeating,
                fluxion_value: 3.0,
                ref_min: 1.4,
                ref_max: 2.2,
                percent_error: 50.0,
                status: ValidationStatus::Fail,
            },
        ],
        benchmark_data: std::collections::HashMap::new(),
    };
    let worst = report.worst_cases(2);
    assert_eq!(worst.len(), 2);
    // Should be sorted by descending absolute percent_error
    assert!(worst[0].percent_error.abs() >= worst[1].percent_error.abs());
}

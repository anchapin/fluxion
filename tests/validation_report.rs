//! Validation report tests for src/validation/report.rs

use fluxion::validation::report::{
    compute_status, BenchmarkData, BenchmarkReport, MetricType, ValidationResult, ValidationStatus,
    ValidationSuite,
};

#[test]
fn test_compute_status_within_range() {
    let status = compute_status(50.0, 40.0, 60.0);
    assert!(matches!(status, ValidationStatus::Pass));
}

#[test]
fn test_compute_status_below_range() {
    let status = compute_status(30.0, 40.0, 60.0);
    assert!(matches!(status, ValidationStatus::Fail));
}

#[test]
fn test_compute_status_above_range() {
    let status = compute_status(70.0, 40.0, 60.0);
    assert!(matches!(status, ValidationStatus::Fail));
}

#[test]
fn test_compute_status_zero_range() {
    let status = compute_status(0.0, 0.0, 0.0);
    assert!(matches!(status, ValidationStatus::Pass));
}

#[test]
fn test_compute_status_negative_values() {
    let status = compute_status(-50.0, -60.0, -40.0);
    assert!(
        matches!(status, ValidationStatus::Pass) || matches!(status, ValidationStatus::Warning)
    );
}

#[test]
fn test_metric_type_units() {
    assert_eq!(MetricType::AnnualHeating.units(), "MWh");
    assert_eq!(MetricType::AnnualCooling.units(), "MWh");
    assert_eq!(MetricType::PeakHeating.units(), "kW");
    assert_eq!(MetricType::PeakCooling.units(), "kW");
    assert_eq!(MetricType::MinFreeFloat.units(), "°C");
    assert_eq!(MetricType::MaxFreeFloat.units(), "°C");
}

#[test]
fn test_validation_result_new_pass() {
    let result = ValidationResult::new("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    assert!(result.is_pass());
    assert!(!result.is_warning());
    assert!(!result.is_fail());
}

#[test]
fn test_validation_result_new_fail() {
    let result = ValidationResult::new("case_600", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    assert!(!result.is_pass());
    assert!(result.is_fail());
}

#[test]
fn test_validation_result_is_within_range() {
    let result = ValidationResult::new("case", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    assert!(result.is_within_range());
}

#[test]
fn test_validation_result_passed_failed() {
    let pass = ValidationResult::new("case", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    let fail = ValidationResult::new("case", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    assert!(pass.passed());
    assert!(!pass.failed());
    assert!(fail.failed());
    assert!(!fail.passed());
}

#[test]
fn test_validation_result_case_id() {
    let result =
        ValidationResult::new("test_case_123", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    assert_eq!(result.case_id, "test_case_123");
}

#[test]
fn test_benchmark_data_new() {
    let bench = BenchmarkData::new();
    assert!(bench.get_range(MetricType::AnnualHeating).is_none());
}

#[test]
fn test_benchmark_data_get_range_with_values() {
    let bench = BenchmarkData {
        annual_heating_min: 40.0,
        annual_heating_max: 60.0,
        ..BenchmarkData::new()
    };
    let range = bench.get_range(MetricType::AnnualHeating);
    assert!(range.is_some());
    let (min, max) = range.unwrap();
    assert!((min - 40.0).abs() < 0.01);
    assert!((max - 60.0).abs() < 0.01);
}

#[test]
fn test_benchmark_data_midpoint() {
    let bench = BenchmarkData {
        annual_heating_min: 40.0,
        annual_heating_max: 60.0,
        ..BenchmarkData::new()
    };
    let midpoint = bench.midpoint(MetricType::AnnualHeating);
    assert!(midpoint.is_some());
    assert!((midpoint.unwrap() - 50.0).abs() < 0.01);
}

#[test]
fn test_benchmark_data_default() {
    let bench = BenchmarkData::default();
    assert!(bench.get_range(MetricType::AnnualHeating).is_none());
}

#[test]
fn test_validation_suite_new() {
    let suite = ValidationSuite::new();
    assert_eq!(suite.len(), 0);
    assert!(suite.is_empty());
}

#[test]
fn test_validation_suite_add_result() {
    let mut suite = ValidationSuite::new();
    let result = ValidationResult::new("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result(result);
    assert_eq!(suite.len(), 1);
    assert!(!suite.is_empty());
}

#[test]
fn test_validation_suite_add_result_simple() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    assert_eq!(suite.len(), 1);
}

#[test]
fn test_validation_suite_pass_fail_count() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    assert_eq!(suite.pass_count(), 1);
    assert_eq!(suite.fail_count(), 1);
}

#[test]
fn test_validation_suite_calculate_pass_rate() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case3", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    let rate = suite.calculate_pass_rate();
    assert!((rate - 66.67).abs() < 1.0);
}

#[test]
fn test_validation_suite_calculate_fail_rate() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    let rate = suite.calculate_fail_rate();
    assert!((rate - 100.0).abs() < 1.0);
}

#[test]
fn test_validation_suite_calculate_mae() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 60.0, 40.0, 60.0);
    let mae = suite.calculate_mae();
    assert!(mae > 0.0);
}

#[test]
fn test_validation_suite_calculate_rmse() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 55.0, 40.0, 60.0);
    let rmse = suite.calculate_rmse();
    assert!(rmse >= 0.0);
}

#[test]
fn test_validation_suite_calculate_max_deviation() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    let max_dev = suite.calculate_max_deviation();
    assert!(max_dev > 0.0);
}

#[test]
fn test_validation_suite_worst_cases() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 100.0, 40.0, 60.0);
    suite.add_result_simple("case3", MetricType::AnnualHeating, 55.0, 40.0, 60.0);
    let worst = suite.worst_cases(2);
    assert_eq!(worst.len(), 2);
    assert_eq!(worst[0].case_id, "case2");
}

#[test]
fn test_validation_suite_get_case_results() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case1", MetricType::AnnualCooling, 30.0, 20.0, 40.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    let results = suite.get_case_results("case1");
    assert_eq!(results.len(), 2);
}

#[test]
fn test_validation_suite_get_metric_results() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 55.0, 40.0, 60.0);
    suite.add_result_simple("case1", MetricType::AnnualCooling, 30.0, 20.0, 40.0);
    let results = suite.get_metric_results(MetricType::AnnualHeating);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_validation_suite_clear() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    assert_eq!(suite.len(), 1);
    suite.clear();
    assert_eq!(suite.len(), 0);
    assert!(suite.is_empty());
}

#[test]
fn test_validation_suite_summary_by_case() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case1", MetricType::AnnualCooling, 30.0, 20.0, 40.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    let summary = suite.summary_by_case();
    assert!(summary.contains_key("case1"));
    assert!(summary.contains_key("case2"));
}

#[test]
fn test_validation_suite_summary_by_metric() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 55.0, 40.0, 60.0);
    suite.add_result_simple("case1", MetricType::AnnualCooling, 30.0, 20.0, 40.0);
    let summary = suite.summary_by_metric();
    assert!(summary.contains_key(&MetricType::AnnualHeating));
    assert!(summary.contains_key(&MetricType::AnnualCooling));
}

#[test]
fn test_validation_suite_case_pass_rate() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case1", MetricType::AnnualCooling, 70.0, 20.0, 40.0);
    let rate = suite
        .calculate_case_pass_rate("case1")
        .expect("case1 should have results");
    assert!((rate - 50.0).abs() < 1.0);
}

#[test]
fn test_benchmark_report_new() {
    let report = BenchmarkReport::new();
    assert_eq!(report.pass_rate(), 100.0);
    assert_eq!(report.fail_count(), 0);
    assert_eq!(report.warning_count(), 0);
}

#[test]
fn test_benchmark_report_add_result() {
    let mut report = BenchmarkReport::new();
    let result = ValidationResult::new("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    report.add_result(result);
    assert_eq!(report.pass_rate(), 100.0);
}

#[test]
fn test_benchmark_report_add_result_simple() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    assert_eq!(report.pass_rate(), 100.0);
}

#[test]
fn test_benchmark_report_add_benchmark_data() {
    let mut report = BenchmarkReport::new();
    let bench = BenchmarkData {
        annual_heating_min: 40.0,
        annual_heating_max: 60.0,
        ..BenchmarkData::new()
    };
    report.add_benchmark_data("case_600", bench);
}

#[test]
fn test_benchmark_report_mae() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    report.add_result_simple("case2", MetricType::AnnualHeating, 55.0, 40.0, 60.0);
    let mae = report.mae();
    assert!(mae >= 0.0);
}

#[test]
fn test_benchmark_report_max_deviation() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    report.add_result_simple("case2", MetricType::AnnualHeating, 100.0, 40.0, 60.0);
    let max_dev = report.max_deviation();
    assert!(max_dev > 0.0);
}

#[test]
fn test_benchmark_report_worst_cases() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    report.add_result_simple("case2", MetricType::AnnualHeating, 100.0, 40.0, 60.0);
    report.add_result_simple("case3", MetricType::AnnualHeating, 55.0, 40.0, 60.0);
    let worst = report.worst_cases(2);
    assert_eq!(worst.len(), 2);
}

#[test]
fn test_benchmark_report_to_json() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    let json = report.to_json();
    assert!(!json.is_empty());
    assert!(json.contains("case_600"));
}

#[test]
fn test_benchmark_report_to_csv() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    let csv = report.to_csv();
    assert!(!csv.is_empty());
    assert!(csv.contains("case_600"));
}

#[test]
fn test_benchmark_report_to_markdown() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    let md = report.to_markdown();
    assert!(md.contains("case_600"));
}

#[test]
fn test_benchmark_report_duration() {
    let mut report = BenchmarkReport::new();
    report.set_start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    report.set_end();
    let duration = report.duration_seconds();
    assert!(duration > 0.0);
}

#[test]
fn test_benchmark_report_print_summary() {
    let mut report = BenchmarkReport::new();
    report.add_result_simple("case_600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    report.print_summary();
}

#[test]
fn test_validation_result_zero_reference_range() {
    let result = ValidationResult::new("case", MetricType::AnnualHeating, 0.0, 0.0, 0.0);
    assert!(result.is_pass());
}

#[test]
fn test_validation_suite_empty_statistics() {
    let suite = ValidationSuite::new();
    assert_eq!(suite.calculate_pass_rate(), 100.0);
    assert_eq!(suite.calculate_fail_rate(), 0.0);
    assert_eq!(suite.calculate_mae(), 0.0);
    assert_eq!(suite.calculate_rmse(), 0.0);
    assert_eq!(suite.calculate_max_deviation(), 0.0);
}

#[test]
fn test_validation_suite_all_pass() {
    let mut suite = ValidationSuite::new();
    for i in 0..10 {
        suite.add_result_simple(
            &format!("case{}", i),
            MetricType::AnnualHeating,
            50.0,
            40.0,
            60.0,
        );
    }
    assert_eq!(suite.pass_count(), 10);
    assert_eq!(suite.fail_count(), 0);
    assert_eq!(suite.calculate_pass_rate(), 100.0);
}

#[test]
fn test_validation_suite_all_fail() {
    let mut suite = ValidationSuite::new();
    for i in 0..10 {
        suite.add_result_simple(
            &format!("case{}", i),
            MetricType::AnnualHeating,
            100.0,
            40.0,
            60.0,
        );
    }
    assert_eq!(suite.pass_count(), 0);
    assert_eq!(suite.fail_count(), 10);
    assert_eq!(suite.calculate_fail_rate(), 100.0);
}

#[test]
fn test_benchmark_report_empty_outputs() {
    let report = BenchmarkReport::new();
    assert!(!report.to_json().is_empty());
    assert!(!report.to_csv().is_empty());
    assert!(!report.to_markdown().is_empty());
}

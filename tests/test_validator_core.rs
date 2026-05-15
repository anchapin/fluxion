//! Unit tests for ASHRAE 140 validator core logic.
//!
//! Tests validation status computation, tolerance calculations,
//! multi-reference enrichment, and edge cases.

use fluxion::validation::report::{
    BenchmarkData, BenchmarkReport, MetricType, ValidationResult, ValidationStatus,
};

#[cfg(test)]
mod validator_unit_tests {
    use super::*;

    // ========================================================================
    // ValidationStatus Tests
    // ========================================================================

    #[test]
    fn test_validation_status_display_names() {
        assert_eq!(ValidationStatus::Pass.display_name(), "PASS");
        assert_eq!(ValidationStatus::Warning.display_name(), "WARN");
        assert_eq!(ValidationStatus::Fail.display_name(), "FAIL");
    }

    #[test]
    fn test_validation_status_icon() {
        assert_eq!(ValidationStatus::Pass.icon(), "✅");
        assert_eq!(ValidationStatus::Warning.icon(), "⚠️");
        assert_eq!(ValidationStatus::Fail.icon(), "❌");
    }

    #[test]
    fn test_validation_status_color() {
        assert_eq!(ValidationStatus::Pass.color(), "green");
        assert_eq!(ValidationStatus::Warning.color(), "yellow");
        assert_eq!(ValidationStatus::Fail.color(), "red");
    }

    // ========================================================================
    // MetricType Tests
    // ========================================================================

    #[test]
    fn test_metric_type_display_names() {
        assert_eq!(
            MetricType::AnnualHeating.display_name(),
            "Annual Heating Energy (MWh)"
        );
        assert_eq!(
            MetricType::AnnualCooling.display_name(),
            "Annual Cooling Energy (MWh)"
        );
        assert_eq!(
            MetricType::PeakHeating.display_name(),
            "Peak Heating Load (kW)"
        );
        assert_eq!(
            MetricType::PeakCooling.display_name(),
            "Peak Cooling Load (kW)"
        );
        // IncidentSolar per ASHRAE 140-2023 Section 8.2.3
        assert_eq!(
            MetricType::IncidentSolar {
                surface_id: "S".to_string(),
                orientation: Orientation::South,
            }
            .display_name(),
            "Incident Solar Radiation (kWh/m²)"
        );
    }

    #[test]
    fn test_metric_type_units() {
        assert_eq!(MetricType::AnnualHeating.units(), "MWh");
        assert_eq!(MetricType::AnnualCooling.units(), "MWh");
        assert_eq!(MetricType::PeakHeating.units(), "kW");
        assert_eq!(MetricType::PeakCooling.units(), "kW");
        // IncidentSolar per ASHRAE 140-2023 Section 8.2.3
        assert_eq!(
            MetricType::IncidentSolar {
                surface_id: "roof".to_string(),
                orientation: Orientation::Up,
            }
            .units(),
            "kWh/m²"
        );
    }

    // ========================================================================
    // compute_status Tests
    // ========================================================================

    #[test]
    fn test_compute_status_within_range() {
        // Value exactly in middle of range
        let status = fluxion::validation::report::compute_status(5.0, 0.0, 10.0);
        assert_eq!(status, ValidationStatus::Pass);
    }

    #[test]
    fn test_compute_status_at_lower_bound() {
        let status = fluxion::validation::report::compute_status(0.0, 0.0, 10.0);
        assert_eq!(status, ValidationStatus::Warning);
    }

    #[test]
    fn test_compute_status_at_upper_bound() {
        let status = fluxion::validation::report::compute_status(10.0, 0.0, 10.0);
        assert_eq!(status, ValidationStatus::Warning);
    }

    #[test]
    fn test_compute_status_below_range() {
        let status = fluxion::validation::report::compute_status(-1.0, 0.0, 10.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_above_range() {
        let status = fluxion::validation::report::compute_status(11.0, 0.0, 10.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_slightly_below_range() {
        // Just below lower bound - should still be warning or fail
        let status = fluxion::validation::report::compute_status(-0.5, 0.0, 10.0);
        assert!(status == ValidationStatus::Warning || status == ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_slightly_above_range() {
        // Just above upper bound - should still be warning or fail
        let status = fluxion::validation::report::compute_status(10.5, 0.0, 10.0);
        assert!(status == ValidationStatus::Warning || status == ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_zero_range() {
        // Zero reference range (e.g., zero energy cases)
        let status = fluxion::validation::report::compute_status(0.0, 0.0, 0.0);
        assert_eq!(status, ValidationStatus::Pass);
    }

    #[test]
    fn test_compute_status_zero_value_in_range() {
        let status = fluxion::validation::report::compute_status(0.0, -1.0, 1.0);
        assert_eq!(status, ValidationStatus::Pass);
    }

    // ========================================================================
    // ValidationResult Tests
    // ========================================================================

    #[test]
    fn test_validation_result_pass() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        assert!(result.is_pass());
        assert!(!result.is_warning());
        assert!(!result.is_fail());
    }

    #[test]
    fn test_validation_result_fail_high() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 10.0, 5.5, 7.5);
        assert!(!result.is_pass());
        assert!(!result.is_warning());
        assert!(result.is_fail());
    }

    #[test]
    fn test_validation_result_fail_low() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 3.0, 5.5, 7.5);
        assert!(!result.is_pass());
        assert!(!result.is_warning());
        assert!(result.is_fail());
    }

    #[test]
    fn test_validation_result_deviation_percent() {
        // Value 10% above reference midpoint (6.5)
        let result = ValidationResult::new(
            "600",
            MetricType::AnnualHeating,
            7.15, // 6.5 * 1.1
            5.5,
            7.5,
        );
        let deviation = result.deviation_percent();
        assert!(
            (deviation - 10.0).abs() < 0.5,
            "Deviation should be ~10%, got {}",
            deviation
        );
    }

    #[test]
    fn test_validation_result_within_range() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        assert!(result.is_within_range());
    }

    #[test]
    fn test_validation_result_outside_range() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 8.0, 5.5, 7.5);
        assert!(!result.is_within_range());
    }

    #[test]
    fn test_validation_result_aliases() {
        let pass_result = ValidationResult::new("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        assert!(pass_result.passed());
        assert!(!pass_result.warning());
        assert!(!pass_result.failed());

        let fail_result = ValidationResult::new("600", MetricType::AnnualHeating, 10.0, 5.5, 7.5);
        assert!(!fail_result.passed());
        assert!(!fail_result.warning());
        assert!(fail_result.failed());
    }

    // ========================================================================
    // BenchmarkData Tests
    // ========================================================================

    #[test]
    fn test_benchmark_data_get_range() {
        let mut data = BenchmarkData::new();
        data.annual_heating_min = 5.5;
        data.annual_heating_max = 7.5;

        let range = data.get_range(MetricType::AnnualHeating);
        assert!(range.is_some());
        let (min, max) = range.unwrap();
        assert!((min - 5.5).abs() < 0.01);
        assert!((max - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_data_get_range_missing() {
        let data = BenchmarkData::new();
        let range = data.get_range(MetricType::AnnualHeating);
        assert!(range.is_none());
    }

    #[test]
    fn test_benchmark_data_midpoint() {
        let mut data = BenchmarkData::new();
        data.annual_heating_min = 5.5;
        data.annual_heating_max = 7.5;

        let midpoint = data.midpoint(MetricType::AnnualHeating);
        assert!(midpoint.is_some());
        assert!((midpoint.unwrap() - 6.5).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_data_midpoint_missing() {
        let data = BenchmarkData::new();
        let midpoint = data.midpoint(MetricType::AnnualHeating);
        assert!(midpoint.is_none());
    }

    // ========================================================================
    // BenchmarkReport Tests
    // ========================================================================

    #[test]
    fn test_benchmark_report_new() {
        let report = BenchmarkReport::new();
        assert_eq!(report.results.len(), 0);
        assert_eq!(report.benchmark_data.len(), 0);
    }

    #[test]
    fn test_benchmark_report_delta_analysis() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        report.add_result_simple("610", MetricType::AnnualHeating, 7.0, 5.8, 7.8);

        let delta = report.delta_analysis("600");
        assert!(delta.contains_key("610"));
        // Delta should be positive (7.0 - 6.5 = 0.5)
        assert!(delta["610"] > 0.0);
    }

    #[test]
    fn test_benchmark_report_delta_analysis_missing_baseline() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("610", MetricType::AnnualHeating, 7.0, 5.8, 7.8);

        let delta = report.delta_analysis("600");
        // Should handle missing baseline gracefully
        assert!(delta.is_empty() || !delta.contains_key("610"));
    }

    #[test]
    fn test_benchmark_report_to_json() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);

        let json = report.to_json();
        assert!(json.contains("600"));
        assert!(json.contains("annual_heating"));
    }

    // ========================================================================
    // Edge Cases and Boundary Conditions
    // ========================================================================

    #[test]
    fn test_validation_result_very_large_value() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 1000.0, 5.5, 7.5);
        assert!(result.is_fail());
        assert!(result.deviation_percent() > 100.0);
    }

    #[test]
    fn test_validation_result_very_small_value() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 0.001, 5.5, 7.5);
        assert!(result.is_fail());
    }

    #[test]
    fn test_validation_result_negative_value() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, -1.0, 5.5, 7.5);
        assert!(result.is_fail());
    }

    #[test]
    fn test_validation_result_nan_handling() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, f64::NAN, 5.5, 7.5);
        // NaN should fail comparison checks
        assert!(!result.is_within_range());
    }

    #[test]
    fn test_validation_result_infinity_handling() {
        let result =
            ValidationResult::new("600", MetricType::AnnualHeating, f64::INFINITY, 5.5, 7.5);
        assert!(result.is_fail());
    }

    #[test]
    fn test_compute_status_very_narrow_range() {
        // Very narrow reference range
        let status = fluxion::validation::report::compute_status(5.0001, 5.0, 5.001);
        assert_eq!(status, ValidationStatus::Pass);
    }

    #[test]
    fn test_compute_status_very_wide_range() {
        // Very wide reference range
        let status = fluxion::validation::report::compute_status(500.0, 0.0, 1000.0);
        assert_eq!(status, ValidationStatus::Pass);
    }

    #[test]
    fn test_benchmark_report_multiple_cases() {
        let mut report = BenchmarkReport::new();
        for i in 600..=650 {
            let case_id = format!("{}", i);
            report.add_result_simple(&case_id, MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        }
        assert_eq!(report.results.len(), 51);
        assert_eq!(report.pass_rate(), 100.0);
    }

    #[test]
    fn test_benchmark_report_multiple_metrics() {
        let mut report = BenchmarkReport::new();
        let metrics = vec![
            MetricType::AnnualHeating,
            MetricType::AnnualCooling,
            MetricType::PeakHeating,
            MetricType::PeakCooling,
        ];
        for metric in metrics {
            report.add_result_simple("600", metric, 6.5, 5.5, 7.5);
        }
        assert_eq!(report.results.len(), 4);
    }
}

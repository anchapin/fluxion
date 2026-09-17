mod tests {
    use crate::validation::report::*;
    use crate::validation::statistical::ValidationGroup;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_validation_result_new_methods() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert!(result.is_within_range());
        assert!((result.deviation_percent() - (-0.0999)).abs() < 0.1);

        let fail = ValidationResult::new("600", MetricType::AnnualHeating, 3.0, 4.30, 5.71);
        assert!(!fail.is_within_range());
    }

    #[test]
    fn test_delta_report() {
        let mut report = DeltaReport::new();
        report.add_delta(DeltaResult {
            case_id: "610".to_string(),
            baseline_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_delta: 0.1,
            reference_delta: 0.08,
            deviation_percent: 25.0,
        });

        let md = report.to_markdown();
        assert!(md.contains("610 vs 600"));
        assert!(md.contains("Annual Heating"));
        assert!(md.contains("25.00%"));
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(
            MetricType::AnnualHeating.display_name(),
            "Annual Heating Energy (kWh)"
        );
        assert_eq!(MetricType::AnnualCooling.units(), "kWh");
        assert_eq!(MetricType::PeakHeating.units(), "kW");
    }

    #[test]
    fn test_validation_status_display() {
        assert_eq!(ValidationStatus::Pass.to_string(), "PASS");
        assert_eq!(ValidationStatus::Warning.to_string(), "WARN");
        assert_eq!(ValidationStatus::Fail.to_string(), "FAIL");
    }

    #[test]
    fn test_validation_result_pass() {
        // Case 600: Heating range 4.30-5.71 MWh
        // Midpoint: 5.005
        // 5% tolerance: [4.085, 5.9955]
        // Fluxion value 5.0 should pass
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert_eq!(result.status, ValidationStatus::Pass);
        assert!(result.passed());
        assert!(!result.warning());
        assert!(!result.failed());
    }

    #[test]
    fn test_validation_result_warning() {
        // Case 600: Heating range 4.30-5.71 MWh
        // Midpoint: 5.005
        // 4.30 is within range but has >2% deviation from midpoint
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 4.31, 4.30, 5.71);
        assert_eq!(result.status, ValidationStatus::Warning);
        assert!(!result.passed());
        assert!(result.warning());
        assert!(!result.failed());
    }

    #[test]
    fn test_validation_result_fail() {
        // Case 600: Heating range 4.30-5.71 MWh
        // 4.0 is outside 5% tolerance (below 4.085)
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 4.0, 4.30, 5.71);
        assert_eq!(result.status, ValidationStatus::Fail);
        assert!(!result.passed());
        assert!(!result.warning());
        assert!(result.failed());
    }

    #[test]
    fn test_validation_result_percent_error() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.50, 4.30, 5.71);
        // Midpoint: 5.005, Error: (5.50 - 5.005) / 5.005 * 100 ≈ 9.89%
        assert!((result.percent_error - 9.89).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_data_range() {
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };

        let range = data.get_range(MetricType::AnnualHeating);
        assert_eq!(range, Some((4.30, 5.71)));

        let range = data.get_range(MetricType::AnnualCooling);
        assert_eq!(range, None); // Not set
    }

    #[test]
    fn test_benchmark_data_midpoint() {
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };

        let midpoint = data.midpoint(MetricType::AnnualHeating);
        assert_eq!(midpoint, Some(5.005));
    }

    #[test]
    fn test_validation_report_basic() {
        let mut report = BenchmarkReport::new();

        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        report.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04);

        assert_eq!(report.results.len(), 3);
        assert!(report.pass_rate() > 0.0);
        assert!(report.mae() >= 0.0);
    }

    #[test]
    fn test_validation_report_markdown() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);

        let markdown = report.to_markdown();
        assert!(markdown.contains("# ASHRAE 140 Validation Report"));
        assert!(markdown.contains("## Summary"));
        assert!(markdown.contains("600"));
    }

    #[test]
    fn test_validation_report_csv() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);

        let csv = report.to_csv();
        assert!(csv.contains("Case,Metric,Fluxion,Ref Min,Ref Max"));
        assert!(csv.contains("600,Annual Heating"));
    }

    #[test]
    fn test_validation_suite_basic() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);

        assert_eq!(suite.len(), 2);
        assert!(!suite.is_empty());
        assert_eq!(suite.pass_count(), 2);
        assert_eq!(suite.fail_count(), 0);
    }

    #[test]
    fn test_validation_suite_pass_rate() {
        let mut suite = ValidationSuite::new();

        // Add mix of pass, warning, fail
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.0, 6.14, 8.45); // Fail
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.17, 1.17, 2.04); // Warning

        let pass_rate = suite.calculate_pass_rate();
        assert!((pass_rate - 33.33).abs() < 0.1); // 1 out of 3 = 33.33%

        let warning_rate = suite.calculate_warning_rate();
        assert!((warning_rate - 33.33).abs() < 0.1); // 1 out of 3

        let fail_rate = suite.calculate_fail_rate();
        assert!((fail_rate - 33.33).abs() < 0.1); // 1 out of 3
    }

    #[test]
    fn test_validation_suite_mae() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // ~0%
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 6.14, 8.45); // ~5%

        let mae = suite.calculate_mae();
        assert!((0.0..=10.0).contains(&mae));
    }

    #[test]
    fn test_validation_suite_rmse() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 6.14, 8.45);

        let rmse = suite.calculate_rmse();
        assert!(rmse >= 0.0);
    }

    #[test]
    fn test_validation_suite_max_deviation() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // ~0%
        suite.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45); // ~25%

        let max_dev = suite.calculate_max_deviation();
        assert!(max_dev >= 20.0);
    }

    #[test]
    fn test_validation_suite_worst_cases() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45);
        suite.add_result_simple("900", MetricType::AnnualHeating, 0.5, 1.17, 2.04);

        let worst = suite.worst_cases(2);
        assert_eq!(worst.len(), 2);

        // Check that worst case has highest deviation
        let first_dev = worst[0].percent_error.abs();
        let second_dev = worst[1].percent_error.abs();
        assert!(first_dev >= second_dev);
    }

    #[test]
    fn test_validation_suite_get_case_results() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04);

        let case_600_results = suite.get_case_results("600");
        assert_eq!(case_600_results.len(), 2);

        let case_900_results = suite.get_case_results("900");
        assert_eq!(case_900_results.len(), 1);
    }

    #[test]
    fn test_validation_suite_get_metric_results() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04);

        let heating_results = suite.get_metric_results(MetricType::AnnualHeating);
        assert_eq!(heating_results.len(), 2);

        let cooling_results = suite.get_metric_results(MetricType::AnnualCooling);
        assert_eq!(cooling_results.len(), 1);
    }

    #[test]
    fn test_validation_suite_case_pass_rate() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.0, 6.14, 8.45); // Fail

        let pass_rate = suite.calculate_case_pass_rate("600");
        assert_eq!(pass_rate, Some(50.0));

        let no_data = suite.calculate_case_pass_rate("INVALID");
        assert_eq!(no_data, None);
    }

    #[test]
    fn test_validation_suite_summary_by_case() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.31, 6.14, 8.45); // Fail
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04); // Pass

        let summary = suite.summary_by_case();

        let case_600 = summary.get("600").unwrap();
        assert_eq!(case_600, &(1, 0, 1)); // 1 pass, 0 warnings, 1 fail

        let case_900 = summary.get("900").unwrap();
        assert_eq!(case_900, &(1, 0, 0)); // 1 pass, 0 warnings, 0 fails
    }

    #[test]
    fn test_validation_suite_summary_by_metric() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.0, 6.14, 8.45); // Fail
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04); // Pass

        let summary = suite.summary_by_metric();

        let heating = summary.get(&MetricType::AnnualHeating).unwrap();
        assert_eq!(heating, &(2, 0, 0)); // 2 pass, 0 warnings, 0 fails

        let cooling = summary.get(&MetricType::AnnualCooling).unwrap();
        assert_eq!(cooling, &(0, 0, 1)); // 0 pass, 0 warnings, 1 fail
    }

    #[test]
    fn test_validation_suite_generate_report() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);

        let report = suite.generate_report();

        assert_eq!(report.results.len(), 1);
        assert!(!report.benchmark_data.is_empty());
    }

    #[test]
    fn test_validation_suite_clear() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert_eq!(suite.len(), 1);

        suite.clear();
        assert_eq!(suite.len(), 0);
        assert!(suite.is_empty());
    }

    #[test]
    fn test_validation_suite_mean_deviation() {
        let mut suite = ValidationSuite::new();

        // Use values that are more symmetric to get mean close to 0
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.5, 4.30, 5.71); // +9.89%
        suite.add_result_simple("600", MetricType::AnnualCooling, 6.57, 6.14, 8.45); // -10%

        let mean_dev = suite.calculate_mean_deviation();
        // Should be close to 0 (positive and negative cancel out)
        assert!(mean_dev.abs() < 1.0);
    }

    #[test]
    fn test_validation_suite_empty() {
        let suite = ValidationSuite::new();

        assert_eq!(suite.len(), 0);
        assert!(suite.is_empty());
        assert_eq!(suite.calculate_pass_rate(), 100.0); // Empty suite defaults to 100%
        assert_eq!(suite.calculate_mae(), 0.0);
    }

    #[test]
    fn test_append_history() {
        use std::fs;
        use std::thread::sleep;
        use std::time::Duration;
        use tempfile::tempdir;

        // Create a report with some results
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_benchmark_data(
            "600",
            BenchmarkData {
                annual_heating_min: 4.30,
                annual_heating_max: 5.71,
                ..Default::default()
            },
        );

        // Simulate validation timing
        report.set_start();
        sleep(Duration::from_millis(10));
        report.set_end();

        // Setup temporary directory guard to isolate file operations
        struct DirGuard(PathBuf);
        impl Drop for DirGuard {
            fn drop(&mut self) {
                // Restore original directory on drop, panic-safe
                let _ = std::env::set_current_dir(&self.0);
            }
        }

        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = tempdir().unwrap();
        let _guard = DirGuard(original_dir.clone());
        std::env::set_current_dir(temp_dir.path()).unwrap();

        // Call append_history
        report.append_history();

        // Verify file creation
        let log_path = temp_dir
            .path()
            .join("target")
            .join("performance_history.jsonl");
        assert!(log_path.exists(), "Performance history file should exist");

        // Read and verify content
        let content = fs::read_to_string(&log_path).expect("Should read log file");
        let mut valid_lines = 0;
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let json: serde_json::Value = serde_json::from_str(line).expect("Valid JSON line");
            assert!(json.get("timestamp").is_some());
            assert!(json.get("mae").is_some());
            assert!(json.get("max_deviation").is_some());
            assert!(json.get("pass_rate").is_some());
            assert!(json.get("validation_time_seconds").is_some());
            assert!(json.get("throughput").is_some());
            assert!(json.get("git_sha").is_some());
            valid_lines += 1;
        }
        assert_eq!(valid_lines, 1);
    }

    #[test]
    fn test_benchmark_report_statistical_fields() {
        use crate::validation::statistical::{EffectDirection, StatisticalMetrics};

        let mut report = BenchmarkReport::new();

        // Test 1: BenchmarkReport can hold optional StatisticalMetrics
        let metrics = StatisticalMetrics {
            nmbe: 2.3,
            cv_rmse: 8.7,
            nmbe_ci: (1.5, 3.1),
            cv_rmse_ci: (7.2, 10.2),
            cohens_d: 0.42,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        };
        report.statistical_metrics = Some(metrics.clone());

        assert!(report.statistical_metrics.is_some());
        let retrieved = report.statistical_metrics.as_ref().unwrap();
        assert_eq!(retrieved.nmbe, 2.3);
        assert_eq!(retrieved.cv_rmse, 8.7);

        // Test 2: BenchmarkReport can hold p-values and BH correction
        report.statistical_p_values = Some(vec![0.023, 0.089, 0.156]);
        report.statistical_corrected = Some(vec![true, false, false]);

        assert!(report.statistical_p_values.is_some());
        assert_eq!(report.statistical_p_values.as_ref().unwrap().len(), 3);
        assert!(report.statistical_corrected.is_some());
        assert!(report.statistical_corrected.as_ref().unwrap()[0]);

        // Test 3: BenchmarkReport can hold group validation results
        let mut group_results = std::collections::HashMap::new();
        group_results.insert(ValidationGroup::Baseline, true);
        group_results.insert(ValidationGroup::HighMass, false);
        report.group_validation = Some(group_results);

        assert!(report.group_validation.is_some());
        let groups = report.group_validation.as_ref().unwrap();
        assert_eq!(groups.get(&ValidationGroup::Baseline), Some(&true));
        assert_eq!(groups.get(&ValidationGroup::HighMass), Some(&false));

        // Test 4: Report without statistical fields (backward compatibility)
        let bare_report = BenchmarkReport::new();
        assert!(bare_report.statistical_metrics.is_none());
        assert!(bare_report.statistical_p_values.is_none());
        assert!(bare_report.statistical_corrected.is_none());
        assert!(bare_report.group_validation.is_none());

        // Test 5: Serialization with optional fields
        let json = serde_json::to_string(&report).expect("Should serialize");
        assert!(json.contains("statistical_metrics"));
        assert!(json.contains("nmbe"));

        let bare_json = serde_json::to_string(&bare_report).expect("Should serialize bare report");
        assert!(!bare_json.contains("statistical_metrics"));
    }

    #[test]
    fn test_validation_result_no_modification_needed() {
        // Test 2: ValidationResult doesn't need modification (per-case stats separate)
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 6.0, 5.5, 7.0);
        assert_eq!(result.case_id, "600");
        assert_eq!(result.metric, MetricType::AnnualHeating);
        assert_eq!(result.fluxion_value, 6.0);
        assert!(result.is_within_range());
    }

    #[test]
    fn test_benchmark_report_serialization_with_statistical_fields() {
        use crate::validation::statistical::{EffectDirection, StatisticalMetrics};

        let mut report = BenchmarkReport::new();
        report.add_result(ValidationResult::new(
            "600",
            MetricType::AnnualHeating,
            6.0,
            5.5,
            7.0,
        ));

        // Add statistical metrics
        report.statistical_metrics = Some(StatisticalMetrics {
            nmbe: 1.5,
            cv_rmse: 5.2,
            nmbe_ci: (0.8, 2.2),
            cv_rmse_ci: (4.1, 6.3),
            cohens_d: 0.28,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        });

        // Add p-values and correction
        report.statistical_p_values = Some(vec![0.03, 0.12, 0.45]);
        report.statistical_corrected = Some(vec![true, false, false]);

        // Add group validation
        let mut groups = std::collections::HashMap::new();
        groups.insert(ValidationGroup::Baseline, true);
        groups.insert(ValidationGroup::HighMass, false);
        report.group_validation = Some(groups);

        // Test JSON serialization
        let json = serde_json::to_string_pretty(&report).expect("Should serialize");
        assert!(json.contains("statistical_metrics"));
        assert!(json.contains("statistical_p_values"));
        assert!(json.contains("statistical_corrected"));
        assert!(json.contains("group_validation"));
        assert!(json.contains("\"nmbe\": 1.5"));
        assert!(json.contains("\"cv_rmse\": 5.2"));

        // Test deserialization
        let deserialized: BenchmarkReport =
            serde_json::from_str(&json).expect("Should deserialize");
        assert!(deserialized.statistical_metrics.is_some());
        assert_eq!(deserialized.statistical_metrics.as_ref().unwrap().nmbe, 1.5);
        assert_eq!(deserialized.statistical_p_values.as_ref().unwrap().len(), 3);
        assert_eq!(
            deserialized.statistical_corrected.as_ref().unwrap().len(),
            3
        );
        assert!(deserialized.group_validation.is_some());

        // Test CSV export (optional fields should be handled gracefully)
        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join("test_statistical_export.csv");
        let csv_content = report.to_csv();
        fs::write(&csv_path, csv_content).expect("Should write CSV file");
        assert!(csv_path.exists());

        // Clean up
        let _ = std::fs::remove_file(csv_path);
    }

    #[test]
    fn test_compute_status_pass() {
        let status = compute_status(5.0, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Pass);
    }

    #[test]
    fn test_compute_status_warning_within_range() {
        let status = compute_status(4.01, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Warning);
    }

    #[test]
    fn test_compute_status_warning_tolerance_band() {
        let status = compute_status(6.2, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Warning);
    }

    #[test]
    fn test_compute_status_fail_below() {
        let status = compute_status(3.0, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_fail_above() {
        let status = compute_status(7.0, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_zero_ref_mid() {
        let status = compute_status(0.5, 0.0, 0.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_validation_status_color_and_icon() {
        assert_eq!(ValidationStatus::Pass.color(), "green");
        assert_eq!(ValidationStatus::Warning.color(), "orange");
        assert_eq!(ValidationStatus::Fail.color(), "red");
        assert_eq!(ValidationStatus::Pass.icon(), "✓");
        assert_eq!(ValidationStatus::Warning.icon(), "⚠");
        assert_eq!(ValidationStatus::Fail.icon(), "✗");
        assert_eq!(ValidationStatus::Pass.display_name(), "PASS");
        assert_eq!(ValidationStatus::Warning.display_name(), "WARN");
        assert_eq!(ValidationStatus::Fail.display_name(), "FAIL");
    }

    #[test]
    fn test_reference_program_display() {
        assert_eq!(format!("{}", ReferenceProgram::EnergyPlus), "EnergyPlus");
        assert_eq!(format!("{}", ReferenceProgram::EspR), "ESP-r");
        assert_eq!(format!("{}", ReferenceProgram::TRNSYS), "TRNSYS");
        assert_eq!(format!("{}", ReferenceProgram::DOE2), "DOE2");
    }

    #[test]
    fn test_benchmark_data_new_and_default() {
        let data = BenchmarkData::new();
        assert_eq!(data.annual_heating_min, 0.0);
        assert_eq!(data.annual_cooling_max, 0.0);
        assert_eq!(data.peak_heating_min, 0.0);
        assert_eq!(data.peak_cooling_max, 0.0);
        assert_eq!(data.min_free_float_min, 0.0);
        assert_eq!(data.max_free_float_max, 0.0);
        let default_data = BenchmarkData::default();
        assert_eq!(default_data.annual_heating_min, 0.0);
    }

    #[test]
    fn test_benchmark_data_all_ranges() {
        let data = BenchmarkData {
            annual_heating_min: 1.0,
            annual_heating_max: 2.0,
            annual_cooling_min: 3.0,
            annual_cooling_max: 4.0,
            peak_heating_min: 5.0,
            peak_heating_max: 6.0,
            peak_cooling_min: 7.0,
            peak_cooling_max: 8.0,
            min_free_float_min: 9.0,
            min_free_float_max: 10.0,
            max_free_float_min: 11.0,
            max_free_float_max: 12.0,
        };
        assert_eq!(data.get_range(MetricType::AnnualHeating), Some((1.0, 2.0)));
        assert_eq!(data.get_range(MetricType::AnnualCooling), Some((3.0, 4.0)));
        assert_eq!(data.get_range(MetricType::PeakHeating), Some((5.0, 6.0)));
        assert_eq!(data.get_range(MetricType::PeakCooling), Some((7.0, 8.0)));
        assert_eq!(data.get_range(MetricType::MinFreeFloat), Some((9.0, 10.0)));
        assert_eq!(data.get_range(MetricType::MaxFreeFloat), Some((11.0, 12.0)));
        assert_eq!(data.midpoint(MetricType::AnnualHeating), Some(1.5));
    }

    #[test]
    fn test_validation_result_is_pass_warning_fail() {
        let pass = ValidationResult::new("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert!(pass.is_pass());
        assert!(!pass.is_warning());
        assert!(!pass.is_fail());

        let fail = ValidationResult::new("600", MetricType::AnnualHeating, 1.0, 4.30, 5.71);
        assert!(!fail.is_pass());
        assert!(!fail.is_warning());
        assert!(fail.is_fail());
    }

    #[test]
    fn test_validation_result_deviation_string() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        let dev = result.deviation_string();
        assert!(dev.contains("%"));
    }

    #[test]
    fn test_interpretation_default() {
        let interp = Interpretation::default();
        assert!(interp.root_cause_hypotheses.is_empty());
        assert!(interp.parameter_sensitivity.is_empty());
        assert!(interp.recommended_next_steps.is_empty());
        assert!(interp.what_if_scenarios.is_empty());
        assert!(interp.references.is_empty());
    }

    #[test]
    fn test_benchmark_report_to_json() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let json = report.to_json();
        assert!(json.contains("results"));
        assert!(json.contains("600"));
    }

    #[test]
    fn test_benchmark_report_add_result_simple() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].case_id, "600");
        assert_eq!(report.results[0].fluxion_value, 5.0);
    }

    #[test]
    fn test_benchmark_report_add_benchmark_data() {
        let mut report = BenchmarkReport::new();
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };
        report.add_benchmark_data("600", data);
        assert!(report.benchmark_data.contains_key("600"));
    }

    #[test]
    fn test_benchmark_report_delta_analysis() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("610", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        report.add_result_simple("610", MetricType::AnnualCooling, 6.5, 6.14, 8.45);

        let deltas = report.delta_analysis("600");
        assert!(!deltas.is_empty());
        assert!(deltas.contains_key("610 - Annual Heating Energy (kWh)"));
        assert!((deltas["610 - Annual Heating Energy (kWh)"] - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_report_delta_analysis_no_baseline_match() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("610", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        let deltas = report.delta_analysis("600");
        assert!(deltas.is_empty());
    }

    #[test]
    fn test_benchmark_report_pass_rate_empty() {
        let report = BenchmarkReport::new();
        assert_eq!(report.pass_rate(), 100.0);
    }

    #[test]
    fn test_benchmark_report_fail_count_and_warning_count() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 1.0, 6.14, 8.45);
        assert_eq!(report.fail_count(), 1);
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn test_benchmark_report_mae_empty() {
        let report = BenchmarkReport::new();
        assert_eq!(report.mae(), 0.0);
    }

    #[test]
    fn test_benchmark_report_max_deviation() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45);
        let max_dev = report.max_deviation();
        assert!(max_dev > 20.0);
    }

    #[test]
    fn test_benchmark_report_worst_cases() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45);
        report.add_result_simple("900", MetricType::AnnualHeating, 0.5, 1.17, 2.04);
        let worst = report.worst_cases(2);
        assert_eq!(worst.len(), 2);
        assert!(worst[0].percent_error.abs() >= worst[1].percent_error.abs());
    }

    #[test]
    fn test_benchmark_report_worst_cases_empty() {
        let report = BenchmarkReport::new();
        let worst = report.worst_cases(5);
        assert!(worst.is_empty());
    }

    #[test]
    fn test_benchmark_report_duration_and_throughput() {
        let mut report = BenchmarkReport::new();
        report.add_benchmark_data("600", BenchmarkData::new());
        assert_eq!(report.duration_seconds(), 0.0);
        assert_eq!(report.cases_per_second(), 0.0);
        report.set_start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        report.set_end();
        assert!(report.duration_seconds() > 0.0);
        assert!(report.cases_per_second() > 0.0);
    }

    #[test]
    fn test_benchmark_report_to_html() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let html = report.to_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("ASHRAE 140 Validation Report"));
        assert!(html.contains("600"));
        assert!(html.contains("class=\"pass\""));
    }

    #[test]
    fn test_benchmark_report_to_html_with_delta() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("610", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        report.add_benchmark_data("600", BenchmarkData::new());
        let html = report.to_html();
        assert!(html.contains("Delta Analysis"));
    }

    #[test]
    fn test_benchmark_report_to_html_with_worst() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 15.0, 6.14, 8.45);
        let html = report.to_html();
        assert!(html.contains("Worst Performing Cases"));
        assert!(html.contains("class=\"fail\""));
    }

    #[test]
    fn test_benchmark_report_to_csv() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("900", MetricType::AnnualCooling, 3.0, 2.13, 3.67);
        let csv = report.to_csv();
        assert!(csv.contains("Case,Metric,Fluxion,Ref Min,Ref Max,Percent Error,Status"));
        assert!(csv.contains("600"));
        assert!(csv.contains("900"));
        assert!(csv.contains("Annual Heating"));
        assert!(csv.contains("Annual Cooling"));
    }

    #[test]
    fn test_benchmark_report_save_to_file_markdown() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.md");
        assert!(report.save_to_file(&path).is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("ASHRAE 140 Validation Report"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_html() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.html");
        assert!(report.save_to_file(&path).is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_csv() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.csv");
        assert!(report.save_to_file(&path).is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Case,Metric"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_txt() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.txt");
        assert!(report.save_to_file(&path).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_unsupported() {
        let report = BenchmarkReport::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.xml");
        let result = report.save_to_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_benchmark_report_to_markdown_with_interpretations() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 1.0, 4.30, 5.71);
        let mut interp = Interpretation::default();
        interp
            .root_cause_hypotheses
            .push("Test hypothesis".to_string());
        interp
            .parameter_sensitivity
            .push("Test sensitivity".to_string());
        interp.recommended_next_steps.push("Test step".to_string());
        interp.what_if_scenarios.push("Test scenario".to_string());
        interp.references.push("Test reference".to_string());
        report.interpretations.insert("600".to_string(), interp);
        let md = report.to_markdown();
        assert!(md.contains("Interpretation Guidance"));
        assert!(md.contains("Test hypothesis"));
        assert!(md.contains("Test sensitivity"));
        assert!(md.contains("Test step"));
        assert!(md.contains("Test scenario"));
        assert!(md.contains("Test reference"));
    }

    #[test]
    fn test_benchmark_report_add_result_with_multi_case_not_found() {
        use crate::validation::multi_reference::MultiReferenceDB;
        let mut report = BenchmarkReport::new();
        let db = MultiReferenceDB {
            version: "test".to_string(),
            source: None,
            cases: std::collections::HashMap::new(),
        };
        report.add_result_with_multi("NONEXISTENT", MetricType::AnnualHeating, 5.0, &db);
        assert_eq!(report.results.len(), 0);
    }

    #[test]
    fn test_benchmark_report_enrich_with_multi_reference_empty() {
        use crate::validation::multi_reference::MultiReferenceDB;
        let mut report = BenchmarkReport::new();
        let db = MultiReferenceDB {
            version: "test".to_string(),
            source: None,
            cases: std::collections::HashMap::new(),
        };
        report.enrich_with_multi_reference(&db);
        assert!(report.results.is_empty());
    }

    #[test]
    fn test_benchmark_report_enrich_with_multi_reference_free_float_unchanged() {
        use crate::validation::multi_reference::MultiReferenceDB;
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600FF", MetricType::MinFreeFloat, -10.0, -18.8, -15.6);
        let db = MultiReferenceDB {
            version: "test".to_string(),
            source: None,
            cases: std::collections::HashMap::new(),
        };
        report.enrich_with_multi_reference(&db);
        assert_eq!(report.results.len(), 1);
        assert!(report.results[0].per_program.is_none());
    }

    #[test]
    fn test_validation_suite_with_ashrae140_data() {
        let suite = ValidationSuite::with_ashrae140_data();
        assert!(!suite.benchmark_data.is_empty());
    }

    #[test]
    fn test_validation_suite_add_benchmark_data() {
        let mut suite = ValidationSuite::new();
        suite.add_benchmark_data("600", BenchmarkData::new());
        assert!(suite.benchmark_data.contains_key("600"));
    }

    #[test]
    fn test_validation_suite_generate_report_with_benchmark_data() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };
        suite.add_benchmark_data("600", data);
        let report = suite.generate_report();
        assert!(report.benchmark_data.contains_key("600"));
    }

    #[test]
    fn test_validation_suite_print_detailed_summary() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        suite.print_detailed_summary();
    }

    #[test]
    fn test_validation_suite_generate_interpretations() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("900", MetricType::AnnualHeating, 5.0, 1.17, 2.04);
        suite.generate_interpretations();
        assert!(!suite.interpretations.is_empty());
        assert!(suite.interpretations.contains_key("900"));
    }

    #[test]
    fn test_validation_suite_generate_interpretations_no_failures() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.generate_interpretations();
        assert!(suite.interpretations.is_empty());
    }

    #[test]
    fn test_validation_suite_generate_interpretations_case_960() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("960", MetricType::AnnualCooling, 5.0, 1.55, 2.78);
        suite.generate_interpretations();
        assert!(suite.interpretations.contains_key("960"));
        let interp = suite.interpretations.get("960").unwrap();
        assert!(!interp.root_cause_hypotheses.is_empty());
    }

    #[test]
    fn test_validation_suite_generate_interpretations_unknown_case() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("XXX", MetricType::AnnualHeating, 5.0, 1.0, 2.0);
        suite.generate_interpretations();
        assert!(suite.interpretations.contains_key("XXX"));
    }
}

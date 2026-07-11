//! Unit tests for validation report module.
//!
//! Tests report generation, systematic issue detection,
//! baseline comparison, and edge cases.

use fluxion::validation::report::{BenchmarkReport, MetricType};
use fluxion::validation::reporter::{BaselineMetrics, SystematicIssue, ValidationReportGenerator};
use std::path::PathBuf;

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod reporter_unit_tests {
    use super::*;

    // ========================================================================
    // ValidationReportGenerator Tests
    // ========================================================================

    #[test]
    fn test_report_generator_new() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test_report.md"));
        assert_eq!(generator.output_path, PathBuf::from("/tmp/test_report.md"));
    }

    #[test]
    fn test_render_markdown_basic_structure() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
        let report = create_sample_report();

        let markdown = generator.render_markdown(&report, None, None);
        assert!(markdown.is_ok());
        let content = markdown.unwrap();

        // Check required sections exist
        assert!(content.contains("# ASHRAE Standard 140 Validation Results"));
        assert!(content.contains("## Summary"));
        assert!(content.contains("## Performance Summary"));
        assert!(content.contains("## Detailed Results"));
        assert!(content.contains("## Systematic Issues"));
        assert!(content.contains("## Phase Progress"));
        assert!(content.contains("## Legend"));
    }

    #[test]
    fn test_render_markdown_summary_values() {
        let _generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
        let mut report = BenchmarkReport::new();

        // Add passing results
        report.add_result_simple("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        report.add_result_simple("600", MetricType::AnnualCooling, 9.0, 8.0, 10.5);

        // Add failing result
        report.add_result_simple("900", MetricType::AnnualHeating, 15.0, 5.5, 7.5);

        let max_dev = report.max_deviation();
        assert!(
            max_dev > 0.0,
            "Max deviation should be positive, got {}",
            max_dev
        );
    }

    #[test]
    fn test_report_duration_seconds() {
        let report = BenchmarkReport::new();
        let duration = report.duration_seconds();
        assert!(duration >= 0.0, "Duration should be non-negative");
    }

    #[test]
    fn test_report_cases_per_second() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        report.add_result_simple("610", MetricType::AnnualHeating, 7.0, 5.8, 7.8);

        let cps = report.cases_per_second();
        assert!(cps >= 0.0, "Cases per second should be non-negative");
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_render_markdown_empty_report() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
        let report = BenchmarkReport::new();

        let markdown = generator.render_markdown(&report, None, None);
        assert!(markdown.is_ok());
        let content = markdown.unwrap();
        assert!(content.contains("# ASHRAE Standard 140 Validation Results"));
    }

    #[test]
    fn test_render_markdown_single_result() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);

        let markdown = generator.render_markdown(&report, None, None).unwrap();
        assert!(markdown.contains("600"));
    }

    #[test]
    fn test_render_markdown_all_metrics_for_case() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
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

        let markdown = generator.render_markdown(&report, None, None).unwrap();
        assert!(markdown.contains("Annual Heating"));
        assert!(markdown.contains("Annual Cooling"));
        assert!(markdown.contains("Peak Heating"));
        assert!(markdown.contains("Peak Cooling"));
    }

    #[test]
    fn test_render_markdown_many_cases() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
        let mut report = BenchmarkReport::new();

        for i in 600..=650 {
            let case_id = format!("{}", i);
            report.add_result_simple(&case_id, MetricType::AnnualHeating, 6.5, 5.5, 7.5);
        }

        let markdown = generator.render_markdown(&report, None, None).unwrap();
        assert!(markdown.contains("Total Results"));
        assert!(markdown.contains("51")); // 51 cases added
    }

    #[test]
    fn test_baseline_comparison_improvement() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);

        let baseline = BaselineMetrics {
            mae: 10.0, // Worse baseline
            max_deviation: 25.0,
            pass_rate: 0.50,
            validation_time_seconds: 200.0,
        };

        let markdown = generator
            .render_markdown(&report, None, Some(&baseline))
            .unwrap();

        // Should show improvement indicators
        assert!(markdown.contains("Performance Comparison"));
    }

    #[test]
    fn test_baseline_comparison_regression() {
        let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/test.md"));
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 10.0, 5.5, 7.5); // FAIL

        let baseline = BaselineMetrics {
            mae: 2.0, // Better baseline
            max_deviation: 5.0,
            pass_rate: 0.95,
            validation_time_seconds: 80.0,
        };

        let markdown = generator
            .render_markdown(&report, None, Some(&baseline))
            .unwrap();

        assert!(markdown.contains("Performance Comparison"));
    }

    #[test]
    fn test_systematic_issue_all_variants() {
        let issues = [
            SystematicIssue::SolarGains,
            SystematicIssue::ThermalMass,
            SystematicIssue::InterZoneTransfer,
            SystematicIssue::HvacLoad,
            SystematicIssue::WeatherData,
            SystematicIssue::ModelLimitation,
            SystematicIssue::Unknown,
        ];

        // All should be distinct
        let unique_count = issues
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(unique_count, issues.len());
    }

    /// Property test for the data-driven classifier (issue #1423).
    ///
    /// Asserts:
    /// * **Determinism** — classifying the same report twice yields identical
    ///   `{case-metric -> issue}` maps.
    /// * **Stability** — no tuned parameters: the Unknown bucket stays small
    ///   (<=8 of the representative failure set), so the tree is not silently
    ///   dropping failures it should be able to categorise.
    /// * **Coverage** — spans Case 195, Cases 600-650, the 900-series, 960, and
    ///   the 940-960 free-float permutations (per acceptance criteria).
    #[test]
    fn classifier_property_test() {
        let mut report = BenchmarkReport::new();

        // --- Case 195 (low-mass): energy + peak metrics ----------------------
        report.add_result_simple("195", MetricType::AnnualCooling, 3.0, 5.0, 7.0); // under -> SolarGains
        report.add_result_simple("195", MetricType::PeakCooling, 3.0, 5.0, 7.0); // under -> SolarGains
        report.add_result_simple("195", MetricType::PeakHeating, 9.0, 5.0, 7.0); // over low-mass -> HvacLoad

        // --- Cases 600-650: annual energy + peak cooling ---------------------
        for case in &["600", "610", "620", "630", "640", "650"] {
            report.add_result_simple(case, MetricType::AnnualCooling, 3.0, 5.0, 7.0); // under -> SolarGains
            report.add_result_simple(case, MetricType::PeakCooling, 3.0, 5.0, 7.0); // under -> SolarGains
        }

        // --- 900-series: annual energy + peak loads --------------------------
        for case in &["900", "910", "920", "930", "940", "950"] {
            report.add_result_simple(case, MetricType::AnnualHeating, 5.0, 1.17, 2.04); // over >=30% -> ModelLimitation
            report.add_result_simple(case, MetricType::PeakCooling, 9.0, 5.0, 7.0); // over high-mass -> ThermalMass
            report.add_result_simple(case, MetricType::PeakHeating, 9.0, 5.0, 7.0); // high-mass -> ThermalMass
        }

        // --- Case 960 --------------------------------------------------------
        report.add_result_simple("960", MetricType::AnnualCooling, 5.0, 1.6, 2.8); // -> InterZoneTransfer
        report.add_result_simple("960", MetricType::PeakCooling, 3.0, 5.0, 7.0); // under -> SolarGains

        // --- 940/950/960 free-float permutations -----------------------------
        report.add_result_simple("900FF", MetricType::MinFreeFloat, 30.0, 40.0, 50.0); // high-mass FF -> ThermalMass
        report.add_result_simple("950FF", MetricType::MaxFreeFloat, 60.0, 40.0, 50.0); // high-mass FF -> ThermalMass
        report.add_result_simple("600FF", MetricType::MaxFreeFloat, 60.0, 40.0, 50.0); // low-mass FF -> SolarGains

        // --- Determinism: two independent runs must agree --------------------
        let map_a = ValidationReportGenerator::classify_systematic_issues(&report);
        let map_b = ValidationReportGenerator::classify_systematic_issues(&report);
        assert_eq!(map_a, map_b, "classifier must be deterministic");
        assert!(!map_a.is_empty(), "expected classified failures");

        // --- Stability: Unknown bucket must be small (<=8) -------------------
        let unknown_count = map_a.values().filter(|v| **v == SystematicIssue::Unknown).count();
        let total = map_a.len();
        assert!(
            unknown_count <= 8,
            "Unknown bucket too large: {} of {} (target <=8)",
            unknown_count,
            total
        );

        // --- Category coverage: each branch of the tree fires at least once --
        use std::collections::HashSet;
        let seen: HashSet<_> = map_a.values().collect();
        assert!(seen.contains(&SystematicIssue::InterZoneTransfer), "missing InterZoneTransfer");
        assert!(seen.contains(&SystematicIssue::ModelLimitation), "missing ModelLimitation");
        assert!(seen.contains(&SystematicIssue::SolarGains), "missing SolarGains");
        assert!(seen.contains(&SystematicIssue::ThermalMass), "missing ThermalMass");

        // --- Every failed metric received a Known-or-Unknown entry -----------
        let failed_count = report.results.iter().filter(|r| r.failed()).count();
        assert_eq!(
            map_a.len(),
            failed_count,
            "every failed result must be classified exactly once"
        );
    }
}

// ========================================================================
// Helper Functions
// ========================================================================

fn create_sample_report() -> BenchmarkReport {
    let mut report = BenchmarkReport::new();

    // Baseline cases
    report.add_result_simple("600", MetricType::AnnualHeating, 6.5, 5.5, 7.5);
    report.add_result_simple("600", MetricType::AnnualCooling, 9.0, 8.0, 10.5);
    report.add_result_simple("610", MetricType::AnnualHeating, 7.0, 5.8, 7.8);
    report.add_result_simple("610", MetricType::AnnualCooling, 5.5, 3.9, 6.1);

    // High-mass cases
    report.add_result_simple("900", MetricType::AnnualHeating, 1.8, 1.2, 2.0);
    report.add_result_simple("900", MetricType::AnnualCooling, 3.0, 2.1, 3.7);
    report.add_result_simple("920", MetricType::AnnualHeating, 4.0, 3.3, 4.3);
    report.add_result_simple("920", MetricType::AnnualCooling, 2.5, 1.8, 3.3);

    // Free-floating cases (no HVAC energy)
    report.add_result_simple("600FF", MetricType::AnnualHeating, 0.0, 0.0, 0.0);
    report.add_result_simple("900FF", MetricType::AnnualHeating, 0.0, 0.0, 0.0);

    // Special cases
    report.add_result_simple("960", MetricType::AnnualHeating, 2.0, 1.7, 2.5);
    report.add_result_simple("960", MetricType::AnnualCooling, 2.0, 1.6, 2.8);

    report
}

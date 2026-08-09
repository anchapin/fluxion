//! Integration tests for `fluxion::validation::report::ValidationSuite`.
//!
//! Replaces a near-empty stub (4 trivial `ValidationResult::passed/failed`
//! checks) with substantive coverage of the validation suite's case-coverage
//! aggregation: pass/warn/fail counting, MAE / RMSE / max-deviation math,
//! case-level and metric-level summary grouping, BenchmarkReport generation,
//! and `clear()`. See GitHub issue #2564.
//!
//! All numbers are chosen so that the expected outcomes are mechanically
//! verifiable (no dependency on simulated physics): midpoint, percent-error,
//! tolerance-band, and aggregation math are all exercised directly.

use fluxion::validation::ashrae_140_cases::Orientation;
use fluxion::validation::report::{
    MetricType, ValidationResult, ValidationStatus, ValidationSuite,
};
use fluxion::validation::{ValidationConfig, ValidationMode};

#[test]
fn suite_new_is_empty() {
    let suite = ValidationSuite::new();
    assert_eq!(suite.len(), 0);
    assert!(suite.is_empty());
    assert_eq!(suite.pass_count(), 0);
    assert_eq!(suite.warning_count(), 0);
    assert_eq!(suite.fail_count(), 0);
}

#[test]
fn suite_new_with_config_does_not_populate_results() {
    let config = ValidationConfig::ashrae140(900);
    let suite = ValidationSuite::new_with_config(config);
    assert!(suite.is_empty());
}

#[test]
fn suite_ashrae_140_case_600_annual_metrics_pass() {
    // Case 600 (light mass, baseline) reference range for AnnualHeating/AnnualCooling.
    // A value at the midpoint must register as PASS with percent_error = 0.
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("600", MetricType::AnnualHeating, 5.2, 4.8, 5.6);
    suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 7.0, 8.0);

    assert_eq!(suite.len(), 2);
    assert_eq!(suite.pass_count(), 2);
    assert_eq!(suite.calculate_pass_rate(), 100.0);

    let six_hundred_results = suite.get_case_results("600");
    assert_eq!(six_hundred_results.len(), 2);
    assert_eq!(
        suite
            .calculate_case_pass_rate("600")
            .expect("case 600 must have results"),
        100.0
    );
}

#[test]
fn suite_detects_fail_outside_tolerance_band() {
    // Reference [4.0, 6.0] has midpoint 5.0 and tolerance band [3.8, 6.3].
    // 6.5 sits just outside the band (within 5% slack) so the
    // `ValidationResult` itself reports FAIL.
    let result = ValidationResult::new("600", MetricType::AnnualHeating, 6.5, 4.0, 6.0);
    assert_eq!(result.status, ValidationStatus::Fail);
    assert!(!result.passed());
    assert!(!result.warning());
    assert!(result.failed());

    let mut suite = ValidationSuite::new();
    suite.add_result(result);
    assert_eq!(suite.fail_count(), 1);
    assert_eq!(suite.pass_count(), 0);
    assert_eq!(suite.calculate_fail_rate(), 100.0);
}

#[test]
fn suite_detects_warning_in_band_but_high_deviation() {
    // Reference [4.0, 6.0] (midpoint 5.0).
    // Value 5.6 is inside [4.0, 6.0] but percent error from midpoint is
    // |5.6 - 5.0| / 5.0 = 12%, which exceeds the 10% threshold and must
    // register as Warning per `compute_status` semantics.
    let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.6, 4.0, 6.0);
    assert_eq!(result.status, ValidationStatus::Warning);
    assert!(!result.passed());
    assert!(!result.failed());
    assert!(result.warning());

    let mut suite = ValidationSuite::new();
    suite.add_result(result);
    assert_eq!(suite.warning_count(), 1);
    assert_eq!(suite.calculate_warning_rate(), 100.0);
}

#[test]
fn suite_detects_warning_in_tolerance_band_outside_range() {
    // Reference [4.0, 6.0]; tolerance band [3.8, 6.3].
    // Value 6.2 is outside [4.0, 6.0] but inside the tolerance band → Warning.
    let result = ValidationResult::new("900", MetricType::AnnualCooling, 6.2, 4.0, 6.0);
    assert_eq!(result.status, ValidationStatus::Warning);
    assert!(result.warning());

    let mut suite = ValidationSuite::new();
    suite.add_result(result);
    assert_eq!(suite.warning_count(), 1);
}

#[test]
fn suite_mae_is_mean_of_absolute_percent_errors() {
    // Two values: 50.0 vs midpoint 50.0 → 0% error,
    //             60.0 vs midpoint 50.0 → 20% error.
    // MAE = (0 + 20) / 2 = 10.
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("A", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("B", MetricType::AnnualHeating, 60.0, 40.0, 60.0);

    let mae = suite.calculate_mae();
    assert!((mae - 10.0).abs() < 1e-9, "expected MAE=10.0, got {mae}");
    assert_eq!(suite.mae(), mae);
}

#[test]
fn suite_rmse_is_sqrt_mean_of_squared_errors() {
    // Two values, both 5% off midpoint 50 → squared errors 25 and 25,
    // RMSE = sqrt(25) = 5.
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("A", MetricType::AnnualHeating, 52.5, 40.0, 60.0);
    suite.add_result_simple("B", MetricType::AnnualHeating, 47.5, 40.0, 60.0);

    let rmse = suite.calculate_rmse();
    assert!((rmse - 5.0).abs() < 1e-9, "expected RMSE=5.0, got {rmse}");
}

#[test]
fn suite_max_deviation_picks_worst_case() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("low", MetricType::AnnualHeating, 50.0, 40.0, 60.0); // 0%
    suite.add_result_simple("mid", MetricType::AnnualHeating, 70.0, 40.0, 60.0); // 40%
    suite.add_result_simple("high", MetricType::AnnualHeating, 50.0, 40.0, 60.0); // 0%

    let max_dev = suite.calculate_max_deviation();
    assert!(
        (max_dev - 40.0).abs() < 1e-9,
        "expected max_deviation=40.0, got {max_dev}"
    );
    assert_eq!(suite.max_deviation(), max_dev);
}

#[test]
fn suite_worst_cases_returns_top_n_sorted() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("a", MetricType::AnnualHeating, 50.0, 40.0, 60.0); // 0%
    suite.add_result_simple("b", MetricType::AnnualHeating, 70.0, 40.0, 60.0); // 40%
    suite.add_result_simple("c", MetricType::AnnualHeating, 55.0, 40.0, 60.0); // 10%
    suite.add_result_simple("d", MetricType::AnnualHeating, 100.0, 40.0, 60.0); // 100%

    let top2 = suite.worst_cases(2);
    assert_eq!(top2.len(), 2);
    assert_eq!(top2[0].case_id, "d");
    assert_eq!(top2[1].case_id, "b");
}

#[test]
fn suite_summary_by_case_groups_pass_warn_fail() {
    // case1: 1 pass, 0 warn, 0 fail
    // case2: 0 pass, 1 warn, 1 fail
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("case1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("case2", MetricType::AnnualHeating, 56.0, 40.0, 60.0); // warning
    suite.add_result_simple("case2", MetricType::AnnualCooling, 80.0, 40.0, 60.0); // fail

    let summary = suite.summary_by_case();
    assert_eq!(summary.get("case1"), Some(&(1, 0, 0)));
    assert_eq!(summary.get("case2"), Some(&(0, 1, 1)));
}

#[test]
fn suite_summary_by_metric_groups_pass_warn_fail() {
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("c1", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    suite.add_result_simple("c2", MetricType::AnnualHeating, 70.0, 40.0, 60.0);
    // Value at midpoint → pass. (60.0 in [40, 60] is technically within
    // range but 20% off midpoint → Warning; we want a deterministic
    // pass here.)
    suite.add_result_simple("c3", MetricType::AnnualCooling, 50.0, 40.0, 60.0);

    let summary = suite.summary_by_metric();
    let heating = summary
        .get(&MetricType::AnnualHeating)
        .expect("heating metric must be present");
    assert_eq!(heating, &(1, 0, 1));
    let cooling = summary
        .get(&MetricType::AnnualCooling)
        .expect("cooling metric must be present");
    assert_eq!(cooling, &(1, 0, 0));
}

#[test]
fn suite_clear_empties_results() {
    let mut suite = ValidationSuite::new_with_config(ValidationConfig::ashrae140(940));
    suite.add_result_simple("600", MetricType::AnnualHeating, 50.0, 40.0, 60.0);
    assert_eq!(suite.len(), 1);
    suite.clear();
    assert_eq!(suite.len(), 0);
    assert!(suite.is_empty());
}

#[test]
fn suite_generate_report_populates_results_and_benchmark_data() {
    // With no benchmark data pre-populated, generate_report must populate
    // BenchmarkData from the ValidationResult fields.
    let mut suite = ValidationSuite::new();
    suite.add_result_simple("600", MetricType::AnnualHeating, 5.2, 4.8, 5.6);
    suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 7.0, 8.0);
    suite.add_result_simple("900", MetricType::AnnualHeating, 1.7, 1.4, 2.0);

    let report = suite.generate_report();
    assert_eq!(report.results.len(), 3);
    assert!(report.benchmark_data.contains_key("600"));
    assert!(report.benchmark_data.contains_key("900"));

    let six_hundred = &report.benchmark_data["600"];
    assert!((six_hundred.annual_heating_min - 4.8).abs() < 1e-9);
    assert!((six_hundred.annual_heating_max - 5.6).abs() < 1e-9);
    assert!((six_hundred.annual_cooling_min - 7.0).abs() < 1e-9);
    assert!((six_hundred.annual_cooling_max - 8.0).abs() < 1e-9);
}

#[test]
fn suite_full_case_coverage_exercise_for_600_and_900() {
    // Exercise a full multi-metric case-coverage scenario across two
    // ASHRAE 140 series (600 low-mass and 900 high-mass) with a mix of
    // pass, warning, and fail outcomes.
    let mut suite = ValidationSuite::new_with_config(ValidationConfig::ashrae140(900));

    // Case 600: all pass
    suite.add_result_simple("600", MetricType::AnnualHeating, 5.2, 4.8, 5.6);
    suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 7.0, 8.0);
    suite.add_result_simple("600", MetricType::PeakHeating, 4.5, 4.0, 5.0);
    suite.add_result_simple("600", MetricType::PeakCooling, 6.0, 5.5, 6.5);

    // Case 900: high-mass; three passes, zero warnings, one fail.
    // PeakCooling 11.0 is outside [9.5, 10.5] and outside tolerance
    // [9.025, 11.025] → FAIL.
    suite.add_result_simple("900", MetricType::AnnualHeating, 1.7, 1.4, 2.0);
    suite.add_result_simple("900", MetricType::AnnualCooling, 2.9, 2.4, 3.4);
    suite.add_result_simple("900", MetricType::PeakHeating, 8.5, 8.0, 9.0);
    suite.add_result_simple("900", MetricType::PeakCooling, 11.2, 9.5, 10.5); // fail (>5% above max)

    assert_eq!(suite.len(), 8);
    assert_eq!(suite.pass_count(), 7);
    assert_eq!(suite.warning_count(), 0);
    assert_eq!(suite.fail_count(), 1);
    assert_eq!(suite.calculate_pass_rate(), 87.5);

    let cases = suite.summary_by_case();
    assert_eq!(cases.get("600"), Some(&(4, 0, 0)));
    assert_eq!(cases.get("900"), Some(&(3, 0, 1)));

    let report = suite.generate_report();
    assert_eq!(report.results.len(), 8);
    assert_eq!(report.fail_count(), 1);
    assert!((report.pass_rate() - 87.5).abs() < 1e-9);

    let worst = suite.worst_cases(1);
    assert_eq!(worst.len(), 1);
    assert_eq!(worst[0].case_id, "900");
    assert!(worst[0].failed());
}

#[test]
fn suite_full_case_coverage_with_mixed_pass_warn_fail() {
    // Same shape as above but force one Warning so the warn / fail
    // counts are non-zero and independently verifiable.
    let mut suite = ValidationSuite::new();

    // Case 600: pass / pass / pass / pass
    suite.add_result_simple("600", MetricType::AnnualHeating, 5.2, 4.8, 5.6);
    suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 7.0, 8.0);
    suite.add_result_simple("600", MetricType::PeakHeating, 4.5, 4.0, 5.0);
    suite.add_result_simple("600", MetricType::PeakCooling, 6.0, 5.5, 6.5);

    // Case 900: pass / pass / warn / fail.
    // PeakHeating 9.4 is outside [8.0, 9.0] but inside tolerance band
    // [7.6, 9.45] → Warning. PeakCooling 11.2 is outside tolerance
    // [9.025, 11.025] → Fail.
    suite.add_result_simple("900", MetricType::AnnualHeating, 1.7, 1.4, 2.0);
    suite.add_result_simple("900", MetricType::AnnualCooling, 2.9, 2.4, 3.4);
    suite.add_result_simple("900", MetricType::PeakHeating, 9.4, 8.0, 9.0); // warn (in tolerance band, outside range)
    suite.add_result_simple("900", MetricType::PeakCooling, 11.2, 9.5, 10.5); // fail

    assert_eq!(suite.len(), 8);
    assert_eq!(suite.pass_count(), 6);
    assert_eq!(suite.warning_count(), 1);
    assert_eq!(suite.fail_count(), 1);
    assert_eq!(suite.calculate_pass_rate(), 75.0);

    let cases = suite.summary_by_case();
    assert_eq!(cases.get("600"), Some(&(4, 0, 0)));
    assert_eq!(cases.get("900"), Some(&(2, 1, 1)));
}

#[test]
fn suite_handles_incident_solar_metric_variant() {
    // IncidentSolar is a struct variant — must hash/compare correctly in
    // summary_by_metric and not collide with the plain AnnualHeating key.
    let mut suite = ValidationSuite::new();
    suite.add_result_simple(
        "600",
        MetricType::IncidentSolar {
            surface_id: "S".to_string(),
            orientation: Orientation::South,
        },
        750.0,
        700.0,
        800.0,
    );
    suite.add_result_simple("600", MetricType::AnnualHeating, 5.2, 4.8, 5.6);

    let summary = suite.summary_by_metric();
    assert!(summary.contains_key(&MetricType::AnnualHeating));
    assert!(summary.contains_key(&MetricType::IncidentSolar {
        surface_id: "S".to_string(),
        orientation: Orientation::South,
    }));
    assert_eq!(summary.len(), 2);
}

#[test]
fn suite_empty_suite_returns_safe_statistical_defaults() {
    let suite = ValidationSuite::new();
    assert_eq!(suite.calculate_pass_rate(), 100.0);
    assert_eq!(suite.calculate_fail_rate(), 0.0);
    assert_eq!(suite.calculate_warning_rate(), 0.0);
    assert_eq!(suite.calculate_mae(), 0.0);
    assert_eq!(suite.calculate_rmse(), 0.0);
    assert_eq!(suite.calculate_max_deviation(), 0.0);
    assert_eq!(suite.calculate_mean_deviation(), 0.0);
    assert_eq!(suite.pass_count(), 0);
    assert_eq!(suite.fail_count(), 0);
}

#[test]
fn suite_validation_mode_round_trips_via_config() {
    // Cross-check that the ValidationMode discriminator on the config
    // round-trips through suite construction without altering suite
    // contents.
    let cfg_standard = ValidationConfig::standard();
    assert!(matches!(cfg_standard.mode, ValidationMode::Standard));

    let cfg_ashrae = ValidationConfig::ashrae140(600);
    assert!(matches!(cfg_ashrae.mode, ValidationMode::ASHRAE140(600)));

    let cfg_perf = ValidationConfig {
        mode: ValidationMode::PerformanceOnly,
        performance_thresholds: fluxion::validation::PerformanceThresholds {
            max_timestep_duration_ms: 50.0,
            max_memory_usage_bytes: 10_000_000,
        },
    };
    assert!(matches!(cfg_perf.mode, ValidationMode::PerformanceOnly));

    // All three configs must build a valid empty suite.
    assert!(ValidationSuite::new_with_config(cfg_standard).is_empty());
    assert!(ValidationSuite::new_with_config(cfg_ashrae).is_empty());
    assert!(ValidationSuite::new_with_config(cfg_perf).is_empty());
}

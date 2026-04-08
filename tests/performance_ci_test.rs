use crate::validation::performance::ci::{
    BenchmarkResult, CiPerformanceReport, CiPerformanceValidator,
};
use chrono::Utc;

#[test]
fn test_ci_performance_validation() {
    let validator = CiPerformanceValidator::new(None);
    let result = validator.validate_no_regression();
    assert!(
        result.is_ok(),
        "Performance validation should pass: {:?}",
        result
    );
}

#[test]
fn test_ci_report_generation() {
    let validator = CiPerformanceValidator::new(None);
    let report = validator.generate_ci_report().unwrap();

    assert!(
        report.regressions.is_empty(),
        "No regressions should be detected in CI report"
    );
    assert!(
        !report.benchmarks.is_empty(),
        "CI report should contain benchmark results"
    );
}

#[test]
fn test_performance_baseline_comparison() {
    // Create a temporary baseline file
    let baseline = CiPerformanceReport {
        timestamp: Utc::now(),
        benchmarks: vec![BenchmarkResult {
            name: "test_benchmark".to_string(),
            duration_ms: 100.0,
        }],
        regressions: vec![],
        improvements: vec![],
    };

    let baseline_path = "test_baseline.json";
    let json = serde_json::to_string(&baseline).unwrap();
    std::fs::write(baseline_path, json).unwrap();

    let validator = CiPerformanceValidator::new(Some(baseline_path.to_string()));
    let current = validator.generate_ci_report().unwrap();

    // Clean up
    std::fs::remove_file(baseline_path).unwrap();

    // Verify comparison works
    assert_eq!(current.benchmarks.len(), baseline.benchmarks.len());
}

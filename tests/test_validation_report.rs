use fluxion::validation::report::{BenchmarkData, BenchmarkReport, MetricType};
use fluxion::validation::reporter::ValidationReportGenerator;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[test]
fn test_validation_report_includes_performance_summary() {
    // Build a minimal report with one case
    let mut report = BenchmarkReport::new();
    report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.0, 6.0);
    report.benchmark_data.insert(
        "600".to_string(),
        BenchmarkData {
            annual_heating_min: 4.0,
            annual_heating_max: 6.0,
            annual_cooling_min: 0.0,
            annual_cooling_max: 0.0,
            peak_heating_min: 0.0,
            peak_heating_max: 0.0,
            peak_cooling_min: 0.0,
            peak_cooling_max: 0.0,
            min_free_float_min: 0.0,
            min_free_float_max: 0.0,
            max_free_float_min: 0.0,
            max_free_float_max: 0.0,
        },
    );

    // Set timing with known duration
    let start = Instant::now();
    let duration = Duration::from_millis(10);
    let end = start + duration;
    report.start_time = Some(start);
    report.end_time = Some(end);

    let generator = ValidationReportGenerator::new(PathBuf::from("/tmp/dummy"));
    let markdown = generator
        .render_markdown(&report, None, None)
        .expect("Should render markdown");

    assert!(
        markdown.contains("## Performance Summary"),
        "Report should contain Performance Summary section"
    );
    assert!(
        markdown.contains("Total Validation Duration"),
        "Report should contain Total Validation Duration metric"
    );
    assert!(
        markdown.contains("Throughput"),
        "Report should contain Throughput metric"
    );
    assert!(
        markdown.contains("cases/sec"),
        "Report should contain cases/sec unit"
    );
    assert!(
        markdown.contains("Total Cases"),
        "Report should contain Total Cases metric"
    );

    // Verify the numeric values appear (with formatting)
    let duration_secs = duration.as_secs_f64();
    assert!(
        markdown.contains(&format!("{:.2} seconds", duration_secs)),
        "Report should display duration in seconds"
    );

    let expected_cps = 1.0 / duration_secs;
    assert!(
        markdown.contains(&format!("{:.2} cases/sec", expected_cps)),
        "Report should display throughput with two decimals"
    );

    assert!(
        markdown.contains("Total Cases | 1 |"),
        "Report should show 1 total case"
    );
}

// Simple test to verify multi-zone reporting functionality
use fluxion::validation::report::{BenchmarkReport, MetricType, ValidationStatus};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut report = BenchmarkReport::new();

    // Add some test results
    report.add_result_simple("960", MetricType::AnnualHeating, 12.5, 10.54, 14.26);
    report.add_result_simple("960", MetricType::AnnualCooling, 8.5, 7.39, 10.00);
    report.add_result_simple("970", MetricType::AnnualHeating, 15.0, 12.75, 17.25);
    report.add_result_simple("970", MetricType::AnnualCooling, 12.0, 10.20, 14.80);

    // Test multi-zone reporting methods
    let markdown_report = report.generate_multi_zone_markdown_report();
    println!(
        "Markdown report length: {} characters",
        markdown_report.len()
    );

    let csv_report = report.generate_multi_zone_csv_report();
    println!("CSV report length: {} characters", csv_report.len());

    let json_report = report.generate_multi_zone_json_report();
    println!("JSON report length: {} characters", json_report.len());

    let comparison_table = report.generate_comparison_table();
    println!(
        "Comparison table length: {} characters",
        comparison_table.len()
    );

    // Test summary generation
    let summary = report.generate_multi_zone_summary();
    println!(
        "Summary: {} tests, {:.1}% pass rate",
        summary.total_tests, summary.pass_rate
    );

    println!("✅ Multi-zone reporting functionality test passed!");

    Ok(())
}

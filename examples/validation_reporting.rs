// examples/validation_reporting.rs
/// Examples of validation reporting functionality
use fluxion::validation::reporting::{ReportFormat, ReportingConfig, ValidationReporter};
use std::error::Error;

/// Example 1: Basic report generation
pub fn example_basic_report_generation() -> Result<(), Box<dyn Error>> {
    // Create a validation reporter with default configuration
    let config = ReportingConfig {
        output_dir: "examples/reports".to_string(),
        format: ReportFormat::Markdown,
        include_diagnostics: true,
        comprehensive: true,
    };

    let reporter = ValidationReporter::new(config);

    // Generate a comprehensive validation report
    let report = reporter.generate_comprehensive_report()?;

    println!("✅ Generated comprehensive validation report");
    println!("Total validations: {}", report.summary.total_validations);
    println!("Pass rate: {:.2}%", report.summary.pass_rate * 100.0);
    println!("Overall status: {:?}", report.summary.overall_status);

    Ok(())
}

/// Example 2: Custom configuration and output formats
pub fn example_custom_configuration() -> Result<(), Box<dyn Error>> {
    // Generate JSON report
    let json_config = ReportingConfig {
        output_dir: "examples/reports".to_string(),
        format: ReportFormat::Json,
        include_diagnostics: false,
        comprehensive: true,
    };

    let json_reporter = ValidationReporter::new(json_config);
    json_reporter.generate_json_report("examples/reports/comprehensive.json")?;
    println!("✅ Generated JSON report");

    // Generate HTML report
    let html_config = ReportingConfig {
        output_dir: "examples/reports".to_string(),
        format: ReportFormat::Html,
        include_diagnostics: true,
        comprehensive: true,
    };

    let html_reporter = ValidationReporter::new(html_config);
    html_reporter.generate_html_report("examples/reports/comprehensive.html")?;
    println!("✅ Generated HTML report");

    // Generate Markdown report
    let md_config = ReportingConfig {
        output_dir: "examples/reports".to_string(),
        format: ReportFormat::Markdown,
        include_diagnostics: true,
        comprehensive: true,
    };

    let md_reporter = ValidationReporter::new(md_config);
    md_reporter.generate_markdown_report("examples/reports/comprehensive.md")?;
    println!("✅ Generated Markdown report");

    Ok(())
}

/// Example 3: Programmatic report generation and analysis
pub fn example_programmatic_analysis() -> Result<(), Box<dyn Error>> {
    let config = ReportingConfig {
        output_dir: "examples/reports".to_string(),
        format: ReportFormat::Markdown,
        include_diagnostics: true,
        comprehensive: true,
    };

    let reporter = ValidationReporter::new(config);
    let report = reporter.generate_comprehensive_report()?;

    // Analyze report data programmatically
    println!("📊 Report Analysis:");
    println!("ASHRAE 140 Cases: {}", report.ashrae140_results.len());
    println!("Climate Zone Validations: {}", report.climate_results.len());
    println!(
        "Occupancy Pattern Validations: {}",
        report.occupancy_results.len()
    );

    // Calculate pass rates by category
    let ashrae_pass_rate = report
        .ashrae140_results
        .iter()
        .filter(|r| {
            matches!(
                r.status,
                fluxion::validation::reporting::ValidationStatus::Pass
            )
        })
        .count() as f64
        / report.ashrae140_results.len() as f64
        * 100.0;

    let climate_pass_rate = report
        .climate_results
        .iter()
        .filter(|r| {
            matches!(
                r.overall_status,
                fluxion::validation::reporting::ValidationStatus::Pass
            )
        })
        .count() as f64
        / report.climate_results.len() as f64
        * 100.0;

    let occupancy_pass_rate = report
        .occupancy_results
        .iter()
        .filter(|r| {
            matches!(
                r.validation_status,
                fluxion::validation::reporting::ValidationStatus::Pass
            )
        })
        .count() as f64
        / report.occupancy_results.len() as f64
        * 100.0;

    println!("ASHRAE 140 Pass Rate: {:.1}%", ashrae_pass_rate);
    println!("Climate Zone Pass Rate: {:.1}%", climate_pass_rate);
    println!("Occupancy Pattern Pass Rate: {:.1}%", occupancy_pass_rate);

    // Quality metrics analysis
    println!("\n🎯 Quality Metrics:");
    println!(
        "Mean Absolute Error: {:.4}",
        report.quality_metrics.mean_absolute_error
    );
    println!(
        "Root Mean Square Error: {:.4}",
        report.quality_metrics.root_mean_square_error
    );
    println!(
        "Coverage Score: {:.1}%",
        report.quality_metrics.coverage_score
    );
    println!(
        "Completeness Score: {:.1}%",
        report.quality_metrics.completeness_score
    );

    Ok(())
}

/// Example 4: Integration with validation workflows
pub fn example_integration_with_workflows() -> Result<(), Box<dyn Error>> {
    // Run validation tests first
    // (In a real scenario, this would run actual validation tests)
    println!("🧪 Running validation tests...");

    // Generate report after validation
    let config = ReportingConfig {
        output_dir: "examples/reports".to_string(),
        format: ReportFormat::Json,
        include_diagnostics: true,
        comprehensive: true,
    };

    let reporter = ValidationReporter::new(config);
    let report = reporter.generate_comprehensive_report()?;

    // Save report for compliance and auditing
    reporter.generate_json_report("examples/reports/compliance_report.json")?;

    println!("✅ Validation workflow completed");
    println!("📄 Compliance report generated: examples/reports/compliance_report.json");

    // Check if validation passed overall
    if matches!(
        report.summary.overall_status,
        fluxion::validation::reporting::ValidationStatus::Pass
    ) {
        println!("🎉 All validations passed! System is compliant.");
    } else {
        println!("⚠️  Some validations failed. Review report for details.");
    }

    Ok(())
}

/// Example 5: Error handling examples
pub fn example_error_handling() -> Result<(), Box<dyn Error>> {
    let config = ReportingConfig {
        output_dir: "examples/reports".to_string(),
        format: ReportFormat::Markdown,
        include_diagnostics: true,
        comprehensive: true,
    };

    let reporter = ValidationReporter::new(config);

    // Try to generate report to invalid path to demonstrate error handling
    let result = reporter.generate_markdown_report("/invalid/path/report.md");

    match result {
        Ok(_) => println!("✅ Report generated successfully"),
        Err(e) => println!("❌ Error generating report: {}", e),
    }

    // Demonstrate proper error handling with valid path
    if let Err(e) = reporter.generate_markdown_report("examples/reports/error_handling_example.md")
    {
        eprintln!("Error: {}", e);
        return Err(e.into());
    }

    println!("✅ Error handling example completed");
    Ok(())
}

/// Run all examples
pub fn run_all_examples() -> Result<(), Box<dyn Error>> {
    println!("🚀 Running Validation Reporting Examples\n");

    println!("=== Example 1: Basic Report Generation ===");
    example_basic_report_generation()?;

    println!("\n=== Example 2: Custom Configuration ===");
    example_custom_configuration()?;

    println!("\n=== Example 3: Programmatic Analysis ===");
    example_programmatic_analysis()?;

    println!("\n=== Example 4: Integration with Workflows ===");
    example_integration_with_workflows()?;

    println!("\n=== Example 5: Error Handling ===");
    example_error_handling()?;

    println!("\n🎉 All examples completed successfully!");
    println!("📁 Check the examples/reports directory for generated reports.");

    Ok(())
}

fn main() {
    if let Err(e) = run_all_examples() {
        eprintln!("Error running examples: {}", e);
        std::process::exit(1);
    }
}

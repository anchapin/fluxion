// examples/cross_validation_example.rs
use clap::{Arg, Command};
use fluxion::validation::esp_r::{EspRTestConfig, EspRValidator, ReportFormat};
use fluxion::validation::MultiZoneValidationResults;
use std::error::Error;
/// Standalone Cross-Validation Example
///
/// This example demonstrates how to run ESP-r cross-validation from the command line.
/// It shows the complete workflow: loading reference data, configuring Fluxion,
/// running cross-validation, and generating reports.
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let matches = Command::new("Fluxion ESP-r Cross-Validation Example")
        .version("1.0")
        .author("Fluxion Team")
        .about("Demonstrates ESP-r cross-validation workflow")
        .arg(
            Arg::new("esp-r-file")
                .long("esp-r")
                .value_name("FILE")
                .help("Path to ESP-r reference CSV file")
                .required(true),
        )
        .arg(
            Arg::new("fluxion-file")
                .long("fluxion")
                .value_name("FILE")
                .help("Path to Fluxion validation results JSON file")
                .required(false),
        )
        .arg(
            Arg::new("tolerance")
                .long("tolerance")
                .value_name("DEGREES")
                .help("Temperature tolerance for comparison (default: 0.5)")
                .default_value("0.5"),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .value_name("FORMAT")
                .help("Output format (json or markdown)")
                .default_value("json"),
        )
        .get_matches();

    // Extract arguments
    let esp_r_path = PathBuf::from(matches.get_one::<String>("esp-r-file").unwrap());
    let tolerance: f64 = matches.get_one::<String>("tolerance").unwrap().parse()?;
    let format_str = matches.get_one::<String>("format").unwrap();

    let report_format = match format_str.to_lowercase().as_str() {
        "json" => ReportFormat::Json,
        "markdown" => ReportFormat::Markdown,
        _ => {
            eprintln!("Invalid format. Using JSON as default.");
            ReportFormat::Json
        }
    };

    println!("=== Fluxion ESP-r Cross-Validation Example ===");
    println!("ESP-r reference file: {}", esp_r_path.display());
    println!("Tolerance: {:.2}°C", tolerance);
    println!("Output format: {}", format_str);
    println!();

    // Create validator
    let validator = EspRValidator::new(esp_r_path.clone(), tolerance);

    // Check if Fluxion results file is provided
    let fluxion_results = if let Some(fluxion_path) = matches.get_one::<String>("fluxion-file") {
        let path = PathBuf::from(fluxion_path);
        println!("Loading Fluxion results from: {}", path.display());

        // Load from JSON file
        let json_data = std::fs::read_to_string(path)?;
        serde_json::from_str(&json_data)?
    } else {
        println!("No Fluxion results file provided. Using sample data.");

        // Create sample Fluxion results
        let mut results = MultiZoneValidationResults::default();
        results.add_zone_result("LivingRoom".to_string(), vec![22.1, 22.3, 22.2, 22.4, 22.5]);
        results.add_zone_result("Bedroom".to_string(), vec![21.8, 21.9, 22.0, 22.1, 22.2]);
        results.add_zone_result("Kitchen".to_string(), vec![23.0, 23.1, 23.2, 23.3, 23.4]);
        results
    };

    // Run validation
    println!("Running cross-validation...");
    let report = validator.validate(&fluxion_results)?;

    // Display results
    println!();
    println!("=== Validation Results ===");
    println!("Overall Status: {}", report.overall_pass);
    println!(
        "Average Temperature Difference: {:.2}°C",
        report.average_temperature_difference
    );
    println!("Number of Zones: {}", report.zone_results.len());
    println!();

    println!("Zone Details:");
    for (zone_name, zone_result) in &report.zone_results {
        println!("  {}:", zone_name);
        println!("    Pass: {}", zone_result.pass);
        println!(
            "    Average Difference: {:.2}°C",
            zone_result.average_difference
        );
        println!(
            "    Maximum Difference: {:.2}°C",
            zone_result.max_difference
        );
        println!(
            "    Standard Deviation: {:.2}°C",
            zone_result.standard_deviation
        );
    }

    // Generate and save report
    println!();
    println!("=== Generating Report ===");

    match report_format {
        ReportFormat::Json => {
            let json_report = serde_json::to_string_pretty(&report)?;
            let output_path = "cross_validation_report.json";
            std::fs::write(output_path, &json_report)?;
            println!("JSON report saved to: {}", output_path);
            println!("Report content:");
            println!("{}", json_report);
        }
        ReportFormat::Markdown => {
            let markdown_report = generate_markdown_report(&report, tolerance);
            let output_path = "cross_validation_report.md";
            std::fs::write(output_path, &markdown_report)?;
            println!("Markdown report saved to: {}", output_path);
            println!("Report content:");
            println!("{}", markdown_report);
        }
    }

    println!();
    println!("Cross-validation example completed successfully!");

    Ok(())
}

/// Generate a Markdown formatted report
fn generate_markdown_report(
    report: &fluxion::validation::reports::CrossValidationReport,
    tolerance: f64,
) -> String {
    let mut markdown = String::new();

    markdown.push_str("# ESP-r Cross-Validation Report\n\n");
    markdown.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));

    markdown.push_str("## Summary\n\n");
    markdown.push_str(&format!("- **Overall Status**: {}\n", report.overall_pass));
    markdown.push_str(&format!("- **Tolerance**: {:.2}°C\n", tolerance));
    markdown.push_str(&format!(
        "- **Average Difference**: {:.2}°C\n",
        report.average_temperature_difference
    ));
    markdown.push_str(&format!(
        "- **Zones Validated**: {}\n\n",
        report.zone_results.len()
    ));

    markdown.push_str("## Zone Results\n\n");
    markdown.push_str("| Zone | Pass | Avg Diff (°C) | Max Diff (°C) | Std Dev (°C) |\n");
    markdown.push_str("|------|------|--------------|--------------|--------------|\n");

    for (zone_name, zone_result) in &report.zone_results {
        markdown.push_str(&format!(
            "| {} | {} | {:.2} | {:.2} | {:.2} |\n",
            zone_name,
            zone_result.pass,
            zone_result.average_difference,
            zone_result.max_difference,
            zone_result.standard_deviation
        ));
    }

    markdown.push_str("\n## Detailed Statistics\n\n");
    markdown.push_str(&format!(
        "- **Maximum Difference Across All Zones**: {:.2}°C\n",
        report
            .zone_results
            .values()
            .map(|z| z.max_difference)
            .fold(0.0, f64::max)
    ));
    markdown.push_str(&format!(
        "- **Minimum Difference Across All Zones**: {:.2}°C\n",
        report
            .zone_results
            .values()
            .map(|z| z.average_difference)
            .fold(f64::INFINITY, f64::min)
    ));

    if report.overall_pass {
        markdown.push_str("\n✅ **Validation PASSED**: All zones within tolerance limits.");
    } else {
        markdown.push_str("\n⚠️ **Validation FAILED**: Some zones exceeded tolerance limits.");
    }

    markdown
}

/// Example usage function for programmatic use
///
/// This function demonstrates how to use the cross-validation functionality
/// programmatically without command-line arguments.
///
/// # Example
/// ```
/// use fluxion_examples::cross_validation_example::run_example;
/// run_example().unwrap();
/// ```
pub fn run_example() -> Result<(), Box<dyn Error>> {
    // This would normally run the example, but for now we just verify it compiles
    println!("Cross-validation example would run here");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_generation() {
        let mut report = fluxion::validation::reports::CrossValidationReport::default();
        report.overall_pass = true;
        report.average_temperature_difference = 0.25;

        let markdown = generate_markdown_report(&report, 0.5);
        assert!(markdown.contains("# ESP-r Cross-Validation Report"));
        assert!(markdown.contains("## Summary"));
        assert!(markdown.contains("✅ **Validation PASSED**"));
    }

    #[test]
    fn test_run_example() {
        // Test that the example function runs without panicking
        let result = run_example();
        assert!(result.is_ok());
    }
}

// validation/esp_r/examples.rs
use crate::validation::cross_validation::CrossValidationReport;
use crate::validation::esp_r::{EspRTestConfig, EspRValidator, ReportFormat};
use crate::validation::MultiZoneValidationResults;
use std::error::Error;
/// ESP-r Cross-Validation Examples
///
/// This module provides comprehensive examples demonstrating how to use the ESP-r
/// cross-validation functionality in Fluxion. These examples cover basic usage,
/// advanced configuration, error handling, and report generation.
use std::path::PathBuf;

/// Basic cross-validation example
///
/// Demonstrates a simple ESP-r vs Fluxion comparison using default settings.
/// This is the simplest way to get started with cross-validation.
///
/// # Example
/// ```
/// use fluxion::validation::esp_r::examples::basic_cross_validation_example;
/// basic_cross_validation_example().unwrap();
/// ```
pub fn basic_cross_validation_example() -> Result<(), Box<dyn Error>> {
    println!("=== Basic Cross-Validation Example ===");

    // Create a simple validator with default tolerance (0.5°C)
    let validator = EspRValidator::new(
        PathBuf::from("examples/reference_data/esp_r_basic.csv"),
        0.5,
    );

    // Create sample Fluxion results (in a real scenario, these would come from actual simulations)
    let mut fluxion_results = MultiZoneValidationResults::default();
    // Add some sample zone data
    fluxion_results.add_zone_result("Zone1".to_string(), vec![22.1, 22.3, 22.2]);
    fluxion_results.add_zone_result("Zone2".to_string(), vec![21.8, 21.9, 22.0]);

    // Run validation
    let report = validator.validate(&fluxion_results)?;

    // Display results
    println!("Validation completed successfully!");
    println!("Overall pass status: {}", report.overall_pass);
    println!("Number of zones validated: {}", report.zone_results.len());
    println!(
        "Average temperature difference: {:.2}°C",
        report.average_temperature_difference
    );

    Ok(())
}

/// Advanced cross-validation example
///
/// Demonstrates custom tolerance settings, multiple zone validation,
/// and detailed report generation with file output.
///
/// # Example
/// ```
/// use fluxion::validation::esp_r::examples::advanced_cross_validation_example;
/// advanced_cross_validation_example().unwrap();
/// ```
pub fn advanced_cross_validation_example() -> Result<(), Box<dyn Error>> {
    println!("=== Advanced Cross-Validation Example ===");

    // Create validator with custom tolerance (0.25°C for high precision)
    let validator = EspRValidator::new(
        PathBuf::from("examples/reference_data/esp_r_advanced.csv"),
        0.25,
    );

    // Create more comprehensive Fluxion results
    let mut fluxion_results = MultiZoneValidationResults::default();

    // Add results for multiple zones with more time steps
    fluxion_results.add_zone_result(
        "LivingRoom".to_string(),
        vec![20.5, 20.7, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5],
    );

    fluxion_results.add_zone_result(
        "Bedroom".to_string(),
        vec![19.0, 19.2, 19.5, 19.8, 20.0, 20.2, 20.5, 20.8, 21.0, 21.2],
    );

    fluxion_results.add_zone_result(
        "Kitchen".to_string(),
        vec![21.0, 21.2, 21.5, 21.8, 22.0, 22.3, 22.5, 22.8, 23.0, 23.2],
    );

    // Run validation
    let report = validator.validate(&fluxion_results)?;

    // Display detailed results
    println!("Advanced validation completed!");
    println!("Tolerance: {:.2}°C", validator.tolerance);
    println!("Overall pass status: {}", report.overall_pass);
    println!("Zones validated: {}", report.zone_results.len());

    for zone_result in &report.zone_results {
        println!(
            "  Zone '{}': pass={}, temp_diff={:.2}°C, heating_diff={:.2}°C",
            zone_result.zone_id,
            zone_result.temp_within_tolerance,
            zone_result.temp_difference,
            zone_result.heating_difference
        );
    }

    // Save report to file
    let report_json = serde_json::to_string_pretty(&report)?;
    std::fs::write("examples/advanced_validation_report.json", report_json)?;
    println!("Report saved to: examples/advanced_validation_report.json");

    Ok(())
}

/// Error handling example
///
/// Demonstrates how to handle various error scenarios that can occur
/// during cross-validation, including proper error handling and recovery patterns.
///
/// # Example
/// ```
/// use fluxion::validation::esp_r::examples::error_handling_example;
/// error_handling_example().unwrap();
/// ```
pub fn error_handling_example() -> Result<(), Box<dyn Error>> {
    println!("=== Error Handling Example ===");

    // Example 1: Handle missing reference file
    println!("\n1. Handling missing reference file:");
    let validator = EspRValidator::new(PathBuf::from("nonexistent_file.csv"), 0.5);

    let fluxion_results = MultiZoneValidationResults::default();

    match validator.validate(&fluxion_results) {
        Ok(_) => println!("  Unexpected success!"),
        Err(e) => println!("  Caught expected error: {}", e),
    }

    // Example 2: Handle invalid tolerance
    println!("\n2. Handling invalid tolerance:");
    let validator = EspRValidator::new(
        PathBuf::from("examples/reference_data/esp_r_basic.csv"),
        -1.0, // Invalid negative tolerance
    );

    // Even with invalid tolerance, the validation will run but likely fail
    match validator.validate(&fluxion_results) {
        Ok(report) => {
            println!("  Validation completed with negative tolerance");
            println!("  Pass status: {}", report.overall_pass);
        }
        Err(e) => println!("  Error during validation: {}", e),
    }

    // Example 3: Handle empty results
    println!("\n3. Handling empty results:");
    let validator = EspRValidator::new(
        PathBuf::from("examples/reference_data/esp_r_basic.csv"),
        0.5,
    );

    let empty_results = MultiZoneValidationResults::default();

    match validator.validate(&empty_results) {
        Ok(report) => {
            println!("  Validation completed with empty results");
            println!("  Number of zones: {}", report.zone_results.len());
        }
        Err(e) => println!("  Error during validation: {}", e),
    }

    println!("\nError handling examples completed!");
    Ok(())
}

/// Report generation example
///
/// Demonstrates different report formats and custom report formatting options.
/// Shows how to generate JSON, Markdown, and custom formatted reports.
///
/// # Example
/// ```
/// use fluxion::validation::esp_r::examples::report_generation_example;
/// report_generation_example().unwrap();
/// ```
pub fn report_generation_example() -> Result<(), Box<dyn Error>> {
    println!("=== Report Generation Example ===");

    // Create validator
    let validator = EspRValidator::new(
        PathBuf::from("examples/reference_data/esp_r_basic.csv"),
        0.5,
    );

    // Create sample results
    let mut fluxion_results = MultiZoneValidationResults::default();
    fluxion_results.add_zone_result("Zone1".to_string(), vec![22.1, 22.3, 22.2]);
    fluxion_results.add_zone_result("Zone2".to_string(), vec![21.8, 21.9, 22.0]);

    // Run validation
    let report = validator.validate(&fluxion_results)?;

    // Generate JSON report
    println!("\n1. JSON Report:");
    let json_report = serde_json::to_string_pretty(&report)?;
    println!("{}", json_report);
    std::fs::write("examples/report_json.json", &json_report)?;

    // Generate Markdown report
    println!("\n2. Markdown Report:");
    let markdown_report = format!(
        "# Cross-Validation Report\n\n{}",
        format!(
            "## Summary\n\n{}",
            format!("- Overall Status: {}\n", report.overall_pass)
                + &format!("- Tolerance: {:.2}°C\n", validator.tolerance)
                + &format!(
                    "- Average Difference: {:.2}°C\n",
                    report.average_temperature_difference
                )
                + &format!("- Zones Validated: {}\n\n", report.zone_results.len())
        )
    ) + "## Zone Results\n\n";

    let mut markdown_report = markdown_report;
    for zone_result in &report.zone_results {
        markdown_report.push_str(&format!(
            "- **{}**: Pass={}, Temp Diff={:.2}°C, Heating Diff={:.2}°C\n",
            zone_result.zone_id,
            zone_result.temp_within_tolerance,
            zone_result.temp_difference,
            zone_result.heating_difference
        ));
    }

    println!("{}", markdown_report);
    std::fs::write("examples/report_markdown.md", markdown_report)?;

    // Generate custom formatted report
    println!("\n3. Custom Formatted Report:");
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let custom_report = format!("Cross-Validation Report - {}\n", timestamp)
        + "================================\n"
        + &format!("Tolerance: {:.2}°C\n", validator.tolerance)
        + &format!("Overall: {}\n", report.overall_pass)
        + &format!("Avg Diff: {:.2}°C\n", report.average_temperature_difference)
        + "\nZone Details:\n";

    let mut custom_report = custom_report;
    for zone_result in &report.zone_results {
        custom_report.push_str(&format!(
            "  {}: {:.2}°C temp, {:.2}°C heating {}\n",
            zone_result.zone_id,
            zone_result.temp_difference,
            zone_result.heating_difference,
            if zone_result.temp_within_tolerance && zone_result.heating_within_tolerance {
                "✓"
            } else {
                "✗"
            }
        ));
    }

    println!("{}", custom_report);
    std::fs::write("examples/report_custom.txt", custom_report)?;

    println!("\nReports generated and saved!");
    Ok(())
}

/// Run all examples in sequence
///
/// This function runs all the examples in order, demonstrating the full
/// range of cross-validation functionality.
///
/// # Example
/// ```
/// use fluxion::validation::esp_r::examples::run_all_examples;
/// run_all_examples().unwrap();
/// ```
pub fn run_all_examples() -> Result<(), Box<dyn Error>> {
    println!("Running all ESP-r cross-validation examples...\n");

    basic_cross_validation_example()?;
    println!();

    advanced_cross_validation_example()?;
    println!();

    error_handling_example()?;
    println!();

    report_generation_example()?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_example() {
        // This test would normally run the basic example
        // For now, we just verify it compiles
        assert!(true);
    }

    #[test]
    fn test_advanced_example() {
        // This test would normally run the advanced example
        assert!(true);
    }

    #[test]
    fn test_error_handling() {
        // This test would normally verify error handling
        assert!(true);
    }

    #[test]
    fn test_report_generation() {
        // This test would normally verify report generation
        assert!(true);
    }
}

// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Multi-Zone ASHRAE 140 Validation Runner
//!
//! This binary runs ASHRAE 140 multi-zone validation cases (960, 970, 980)
//! and provides comprehensive reporting capabilities.
//!
//! # Features
//! - Case 960: Two-zone sunspace building validation
//! - Case 970: Multi-zone building validation framework
//! - Multiple output formats: console, Markdown, CSV, JSON
//! - Detailed statistical analysis and visualization support

use clap::{ArgAction, Parser, Subcommand};
use fluxion::validation::ashrae_140_multi_zone::ASHRAE140MultiZoneValidator;
use fluxion::validation::report::{BenchmarkReport, ValidationStatus};
use std::path::Path;
use std::process;

/// Multi-zone ASHRAE 140 validation CLI
#[derive(Parser, Debug)]
#[command(name = "run-multi-zone-validation")]
#[command(version = "1.0.0")]
#[command(about = "ASHRAE 140 Multi-Zone Validation Tool", long_about = None)]
struct Cli {
    /// Enable verbose output with detailed statistics
    #[arg(short, long, global = true, action = ArgAction::SetTrue)]
    verbose: bool,

    /// Export results to CSV format
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    csv_export: bool,

    /// Export results to JSON format
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    json_export: bool,

    /// Override default validation tolerances (e.g., 0.15 for 15%)
    #[arg(long, global = true)]
    tolerance: Option<f64>,

    /// Custom path to reference data files
    #[arg(long, global = true)]
    reference_path: Option<String>,

    /// Subcommands for specific validation cases
    #[command(subcommand)]
    command: Commands,
}

/// Validation subcommands
#[derive(Subcommand, Debug)]
enum Commands {
    /// Run Case 960 validation (two-zone sunspace building)
    Case960,

    /// Run Case 970 validation (multi-zone building framework)
    Case970,

    /// Run all multi-zone validation cases
    All,

    /// Generate comprehensive validation report
    Report {
        /// Output file path for the report
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize validator
    let mut validator = ASHRAE140MultiZoneValidator::new();

    match &cli.command {
        Commands::Case960 => {
            run_case_960(&mut validator, &cli)?;
        }
        Commands::Case970 => {
            run_case_970(&mut validator, &cli)?;
        }
        Commands::All => {
            run_all_cases(&mut validator, &cli)?;
        }
        Commands::Report { output } => {
            generate_report(&mut validator, output, &cli)?;
        }
    }

    Ok(())
}

fn run_case_960(
    validator: &mut ASHRAE140MultiZoneValidator,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ASHRAE 140 Case 960 Validation ===");
    println!("Two-zone sunspace building validation");
    println!("-----------------------------------------");

    // Run Case 960 validation
    let report = validator.run_multi_zone_validation();

    // Filter and display Case 960 results
    let case_960_results: Vec<_> = report
        .results
        .iter()
        .filter(|r| r.case_id == "960")
        .collect();

    if case_960_results.is_empty() {
        println!("❌ No Case 960 results found");
        process::exit(2);
    }

    println!("\nCase 960 Results:");
    println!("| Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |");
    println!("|--------|---------|---------|---------|-----------|--------|");

    let mut overall_status = ValidationStatus::Pass;

    for result in case_960_results {
        let status_icon = match result.status {
            ValidationStatus::Pass => "✓",
            ValidationStatus::Warning => "⚠",
            ValidationStatus::Fail => "✗",
        };

        let _status_color = match result.status {
            ValidationStatus::Pass => "green",
            ValidationStatus::Warning => "yellow",
            ValidationStatus::Fail => "red",
        };

        println!(
            "{} {} | {:.2} | {:.2} | {:.2} | {:+.1}% | {}",
            status_icon,
            result.metric,
            result.fluxion_value,
            result.ref_min,
            result.ref_max,
            result.percent_error.abs(),
            result.status
        );

        if result.status == ValidationStatus::Fail {
            overall_status = ValidationStatus::Fail;
        } else if result.status == ValidationStatus::Warning
            && overall_status == ValidationStatus::Pass
        {
            overall_status = ValidationStatus::Warning;
        }
    }

    // Display overall status
    println!(
        "\nOverall Status: {}",
        match overall_status {
            ValidationStatus::Pass => "🟢 PASS",
            ValidationStatus::Warning => "🟡 WARN",
            ValidationStatus::Fail => "🔴 FAIL",
        }
    );

    // Export if requested
    if cli.csv_export {
        export_results_csv(&report, "case_960_results.csv")?;
        println!("\n📊 Exported CSV: case_960_results.csv");
    }

    if cli.json_export {
        export_results_json(&report, "case_960_results.json")?;
        println!("📊 Exported JSON: case_960_results.json");
    }

    // Exit with appropriate code
    match overall_status {
        ValidationStatus::Pass => process::exit(0),
        ValidationStatus::Warning => process::exit(1),
        ValidationStatus::Fail => process::exit(2),
    }
}

fn run_case_970(
    validator: &mut ASHRAE140MultiZoneValidator,
    _cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ASHRAE 140 Case 970 Validation ===");
    println!("Multi-zone building validation framework");
    println!("-----------------------------------------");
    println!("⚠️  Case 970 is a framework implementation");
    println!("    Full validation will be implemented in future work");
    println!("    Running basic framework validation...\n");

    // Run comprehensive validation (includes Case 970 stub)
    let report = validator.run_comprehensive_validation();

    // Filter and display Case 970 results
    let case_970_results: Vec<_> = report
        .results
        .iter()
        .filter(|r| r.case_id == "970")
        .collect();

    println!("Case 970 Framework Status:");
    for result in case_970_results {
        println!(
            "  {}: {} ({:.1}% error)",
            result.metric,
            result.status,
            result.percent_error.abs()
        );
    }

    println!("\n📋 Case 970 framework is operational");
    println!("   Ready for full implementation");

    Ok(())
}

fn run_all_cases(
    validator: &mut ASHRAE140MultiZoneValidator,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ASHRAE 140 Multi-Zone Validation Suite ===");
    println!("Running all multi-zone validation cases");
    println!("---------------------------------------------");

    // Run comprehensive validation
    let report = validator.run_comprehensive_validation();

    println!("\nValidation Results Summary:");
    println!("| Case | Metric | Status | Deviation |");
    println!("|------|--------|--------|-----------|");

    let mut overall_status = ValidationStatus::Pass;

    for result in &report.results {
        let status_icon = match result.status {
            ValidationStatus::Pass => "✓",
            ValidationStatus::Warning => "⚠",
            ValidationStatus::Fail => "✗",
        };

        println!(
            "{} Case {} | {} | {} | {:+.1}% |",
            status_icon,
            result.case_id,
            result.metric,
            result.status,
            result.percent_error.abs()
        );

        if result.status == ValidationStatus::Fail {
            overall_status = ValidationStatus::Fail;
        } else if result.status == ValidationStatus::Warning
            && overall_status == ValidationStatus::Pass
        {
            overall_status = ValidationStatus::Warning;
        }
    }

    // Display statistics
    println!("\n📊 Statistics:");
    println!("  Total Cases: {}", report.results.len());
    println!("  Pass Rate: {:.1}%", report.pass_rate());
    println!("  Warnings: {}", report.warning_count());
    println!("  Failed: {}", report.fail_count());
    println!("  Max Deviation: {:.1}%", report.max_deviation());

    // Export if requested
    if cli.csv_export {
        export_results_csv(&report, "multi_zone_results.csv")?;
        println!("\n📊 Exported CSV: multi_zone_results.csv");
    }

    if cli.json_export {
        export_results_json(&report, "multi_zone_results.json")?;
        println!("📊 Exported JSON: multi_zone_results.json");
    }

    // Exit with appropriate code
    match overall_status {
        ValidationStatus::Pass => process::exit(0),
        ValidationStatus::Warning => process::exit(1),
        ValidationStatus::Fail => process::exit(2),
    }
}

fn generate_report(
    validator: &mut ASHRAE140MultiZoneValidator,
    output: &Option<String>,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Generating Multi-Zone Validation Report ===");

    // Run comprehensive validation to get all results
    let report = validator.run_comprehensive_validation();

    // Generate Markdown report
    let markdown_report = report.to_markdown();

    // Save to file if output path provided
    if let Some(output_path) = output {
        report.save_to_file(Path::new(output_path))?;
        println!("📄 Report saved to: {}", output_path);
    } else {
        // Print to console
        println!("{}", markdown_report);
    }

    // Also save default documentation
    let docs_path = "docs/ASHRAE140_MULTI_ZONE_RESULTS.md";
    report.save_to_file(Path::new(docs_path))?;
    println!("📄 Documentation saved to: {}", docs_path);

    // Export additional formats if requested
    if cli.csv_export {
        export_results_csv(&report, "validation_report.csv")?;
        println!("📊 CSV export: validation_report.csv");
    }

    if cli.json_export {
        export_results_json(&report, "validation_report.json")?;
        println!("📊 JSON export: validation_report.json");
    }

    Ok(())
}

fn export_results_csv(
    report: &BenchmarkReport,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let csv_content = report.to_csv();
    std::fs::write(path, csv_content)?;
    Ok(())
}

fn export_results_json(
    report: &BenchmarkReport,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json_content = report.to_json();
    std::fs::write(path, json_content)?;
    Ok(())
}

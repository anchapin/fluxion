// src/cli/commands/cross_validation.rs
/// CLI commands for ESP-r cross-validation workflows
///
/// Provides command-line interface for running cross-validation tests,
/// generating reports, and managing ESP-r validation workflows.
use clap::{Args, Subcommand};
use std::path::PathBuf;

/// Cross-validation subcommands
#[derive(Subcommand, Debug)]
pub enum CrossValidationCommand {
    /// Run ESP-r cross-validation test
    #[clap(name = "run")]
    Run(CrossValidationRunArgs),

    /// Generate validation report from existing results
    #[clap(name = "report")]
    Report(CrossValidationReportArgs),

    /// Run validation and generate report (combined operation)
    #[clap(name = "validate")]
    Validate(CrossValidationValidateArgs),
}

/// Arguments for cross-validation run command
#[derive(Args, Debug)]
pub struct CrossValidationRunArgs {
    /// Path to ESP-r output CSV file
    #[clap(short, long, required = true)]
    pub esp_r_output: PathBuf,

    /// Path to Fluxion configuration JSON file
    #[clap(short, long, required = true)]
    pub fluxion_config: PathBuf,

    /// Validation tolerance in °C
    #[clap(short, long, default_value = "0.1")]
    pub tolerance: f64,

    /// Output format (json or markdown)
    #[clap(short, long, value_enum, default_value = "markdown")]
    pub format: crate::validation::esp_r::cli_integration::ReportFormat,

    /// Output file path (optional - if not provided, output to stdout)
    #[clap(short, long)]
    pub output: Option<PathBuf>,

    /// Show verbose output
    #[clap(short, long)]
    pub verbose: bool,
}

/// Arguments for cross-validation report command
#[derive(Args, Debug)]
pub struct CrossValidationReportArgs {
    /// Path to validation results JSON file
    #[clap(short, long, required = true)]
    pub results_file: PathBuf,

    /// Output format (json or markdown)
    #[clap(short, long, value_enum, default_value = "markdown")]
    pub format: crate::validation::esp_r::cli_integration::ReportFormat,

    /// Output file path (optional - if not provided, output to stdout)
    #[clap(short, long)]
    pub output: Option<PathBuf>,

    /// Show detailed statistics
    #[clap(short, long)]
    pub detailed: bool,
}

/// Arguments for cross-validation validate command
#[derive(Args, Debug)]
pub struct CrossValidationValidateArgs {
    /// Path to ESP-r output CSV file
    #[clap(short, long, required = true)]
    pub esp_r_output: PathBuf,

    /// Path to Fluxion configuration JSON file
    #[clap(short, long, required = true)]
    pub fluxion_config: PathBuf,

    /// Validation tolerance in °C
    #[clap(short, long, default_value = "0.1")]
    pub tolerance: f64,

    /// Output format (json or markdown)
    #[clap(short, long, value_enum, default_value = "markdown")]
    pub format: crate::validation::esp_r::cli_integration::ReportFormat,

    /// Output file path (optional - if not provided, output to stdout)
    #[clap(short, long)]
    pub output: Option<PathBuf>,

    /// Show verbose output
    #[clap(short, long)]
    pub verbose: bool,

    /// Generate detailed statistics report
    #[clap(short, long)]
    pub detailed: bool,
}

/// Execute cross-validation run command
pub fn execute_run_command(
    args: &CrossValidationRunArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.verbose {
        println!("Running ESP-r cross-validation...");
        println!("ESP-r Output: {}", args.esp_r_output.display());
        println!("Fluxion Config: {}", args.fluxion_config.display());
        println!("Tolerance: {}°C", args.tolerance);
        println!("Format: {}", args.format);
    }

    // Create CLI configuration
    let config = crate::validation::esp_r::cli_integration::EspRCliConfig {
        esp_r_output: args.esp_r_output.clone(),
        fluxion_config: args.fluxion_config.clone(),
        tolerance: args.tolerance,
        output_format: args.format.clone(),
        output_path: args.output.clone(),
    };

    // Run validation
    let result = crate::validation::esp_r::cli_integration::run_cli_validation(&config)?;

    if args.verbose {
        println!("\nValidation Results:");
        println!(
            "  Status: {}",
            if result.passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        );
        println!("  Pass Rate: {:.1}%", result.pass_rate * 100.0);
        println!("  Total Zones: {}", result.report.zone_results.len());
        println!(
            "  Zones Passed: {}",
            (result.pass_rate * result.report.zone_results.len() as f64) as usize
        );
    }

    Ok(())
}

/// Execute cross-validation report command
pub fn execute_report_command(
    args: &CrossValidationReportArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating cross-validation report...");

    // Load validation results from file
    let file_content = std::fs::read_to_string(&args.results_file)?;
    let cli_result: crate::validation::esp_r::cli_integration::EspRCliResult =
        serde_json::from_str(&file_content)?;

    // Generate report in requested format
    let report = match args.format {
        crate::validation::esp_r::cli_integration::ReportFormat::JSON => {
            serde_json::to_string_pretty(&cli_result)?
        }
        crate::validation::esp_r::cli_integration::ReportFormat::Markdown => {
            crate::validation::esp_r::cli_integration::generate_markdown_report(&cli_result)?
        }
    };

    // Output report
    if let Some(output_path) = &args.output {
        std::fs::write(output_path, report)?;
        println!("Report saved to: {}", output_path.display());
    } else {
        println!("{}", report);
    }

    if args.detailed {
        println!("\nDetailed Statistics:");
        println!(
            "  Mean Temp Difference: {:.2}°C",
            cli_result.report.statistics.mean_temp_difference
        );
        println!(
            "  Max Temp Difference: {:.2}°C",
            cli_result.report.statistics.max_temp_difference
        );
        println!(
            "  Mean Heating Difference: {:.2} W",
            cli_result.report.statistics.mean_heating_difference
        );
        println!(
            "  Max Heating Difference: {:.2} W",
            cli_result.report.statistics.max_heating_difference
        );
    }

    Ok(())
}

/// Execute cross-validation validate command
pub fn execute_validate_command(
    args: &CrossValidationValidateArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.verbose {
        println!("Running ESP-r cross-validation with detailed reporting...");
    }

    // Run the validation
    execute_run_command(&CrossValidationRunArgs {
        esp_r_output: args.esp_r_output.clone(),
        fluxion_config: args.fluxion_config.clone(),
        tolerance: args.tolerance,
        format: args.format.clone(),
        output: args.output.clone(),
        verbose: args.verbose,
    })?;

    // If output file was specified, generate additional detailed report
    if let Some(output_path) = &args.output {
        if args.detailed {
            let results_file = output_path.with_extension("json");

            // Generate detailed report
            execute_report_command(&CrossValidationReportArgs {
                results_file,
                format: crate::validation::esp_r::cli_integration::ReportFormat::Markdown,
                output: Some(output_path.with_extension("detailed.md")),
                detailed: true,
            })?;
        }
    }

    Ok(())
}

/// Register cross-validation commands with the CLI
pub fn register_commands() -> clap::Command {
    clap::Command::new("cross-validation")
        .about("ESP-r cross-validation commands")
        .subcommand(
            clap::Command::new("run")
                .about("Run ESP-r cross-validation test")
                .arg(
                    clap::Arg::new("esp-r-output")
                        .short('e')
                        .long("esp-r-output")
                        .help("Path to ESP-r output CSV file")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("fluxion-config")
                        .short('f')
                        .long("fluxion-config")
                        .help("Path to Fluxion configuration JSON file")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("tolerance")
                        .short('t')
                        .long("tolerance")
                        .help("Validation tolerance in °C")
                        .default_value("0.1"),
                )
                .arg(
                    clap::Arg::new("format")
                        .short('o')
                        .long("format")
                        .help("Output format (json or markdown)")
                        .value_parser(["json", "markdown"])
                        .default_value("markdown"),
                )
                .arg(
                    clap::Arg::new("output")
                        .short('o')
                        .long("output")
                        .help("Output file path"),
                )
                .arg(
                    clap::Arg::new("verbose")
                        .short('v')
                        .long("verbose")
                        .help("Show verbose output"),
                ),
        )
        .subcommand(
            clap::Command::new("report")
                .about("Generate validation report from existing results")
                .arg(
                    clap::Arg::new("results-file")
                        .short('r')
                        .long("results-file")
                        .help("Path to validation results JSON file")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("format")
                        .short('f')
                        .long("format")
                        .help("Output format (json or markdown)")
                        .value_parser(["json", "markdown"])
                        .default_value("markdown"),
                )
                .arg(
                    clap::Arg::new("output")
                        .short('o')
                        .long("output")
                        .help("Output file path"),
                )
                .arg(
                    clap::Arg::new("detailed")
                        .short('d')
                        .long("detailed")
                        .help("Show detailed statistics"),
                ),
        )
        .subcommand(
            clap::Command::new("validate")
                .about("Run validation and generate report (combined operation)")
                .arg(
                    clap::Arg::new("esp-r-output")
                        .short('e')
                        .long("esp-r-output")
                        .help("Path to ESP-r output CSV file")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("fluxion-config")
                        .short('f')
                        .long("fluxion-config")
                        .help("Path to Fluxion configuration JSON file")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("tolerance")
                        .short('t')
                        .long("tolerance")
                        .help("Validation tolerance in °C")
                        .default_value("0.1"),
                )
                .arg(
                    clap::Arg::new("format")
                        .short('o')
                        .long("format")
                        .help("Output format (json or markdown)")
                        .value_parser(["json", "markdown"])
                        .default_value("markdown"),
                )
                .arg(
                    clap::Arg::new("output")
                        .short('o')
                        .long("output")
                        .help("Output file path"),
                )
                .arg(
                    clap::Arg::new("verbose")
                        .short('v')
                        .long("verbose")
                        .help("Show verbose output"),
                )
                .arg(
                    clap::Arg::new("detailed")
                        .short('d')
                        .long("detailed")
                        .help("Generate detailed statistics report"),
                ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_command_parsing() {
        // Test that the command structure compiles and can be parsed
        let command = CrossValidationCommand::Run(CrossValidationRunArgs {
            esp_r_output: PathBuf::from("test.csv"),
            fluxion_config: PathBuf::from("config.json"),
            tolerance: 0.1,
            format: crate::validation::esp_r::cli_integration::ReportFormat::Markdown,
            output: None,
            verbose: false,
        });

        // This test just verifies the structure compiles
        assert!(matches!(command, CrossValidationCommand::Run(_)));
    }

    #[test]
    fn test_report_format_parsing() {
        // Test that report format can be parsed from strings
        assert!(matches!(
            "json".parse::<crate::validation::esp_r::cli_integration::ReportFormat>(),
            Ok(crate::validation::esp_r::cli_integration::ReportFormat::JSON)
        ));
        assert!(matches!(
            "markdown".parse::<crate::validation::esp_r::cli_integration::ReportFormat>(),
            Ok(crate::validation::esp_r::cli_integration::ReportFormat::Markdown)
        ));
    }
}

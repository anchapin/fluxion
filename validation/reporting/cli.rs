// validation/reporting/cli.rs
/// CLI commands for validation reporting
use clap::{Args, Subcommand};
use std::path::PathBuf;

/// Reporting subcommands
#[derive(Subcommand, Debug)]
pub enum ReportingCommand {
    /// Generate comprehensive validation report
    #[clap(name = "generate")]
    Generate(ReportGenerateArgs),
}

/// Arguments for report generate command
#[derive(Args, Debug)]
pub struct ReportGenerateArgs {
    /// Report format (json, html, or markdown)
    #[clap(short, long, value_enum, default_value = "markdown")]
    pub format: ReportFormat,

    /// Output file path
    #[clap(short, long, required = true)]
    pub output: PathBuf,

    /// Include comprehensive data from all validation modules
    #[clap(short, long)]
    pub comprehensive: bool,

    /// Include detailed diagnostics in report
    #[clap(short, long)]
    pub diagnostics: bool,
}

/// Supported report formats
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum ReportFormat {
    Json,
    Html,
    Markdown,
}

impl From<ReportFormat> for super::ReportFormat {
    fn from(format: ReportFormat) -> Self {
        match format {
            ReportFormat::Json => super::ReportFormat::Json,
            ReportFormat::Html => super::ReportFormat::Html,
            ReportFormat::Markdown => super::ReportFormat::Markdown,
        }
    }
}

/// Execute report generate command
pub fn execute_report_generate_command(args: &ReportGenerateArgs) -> Result<(), String> {
    let config = super::ReportingConfig {
        output_dir: args
            .output
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "validation/reports".to_string()),
        format: args.format.clone().into(),
        include_diagnostics: args.diagnostics,
        comprehensive: args.comprehensive,
    };

    let reporter = super::ValidationReporter::new(config);

    match args.format {
        ReportFormat::Json => reporter.generate_json_report(&args.output.to_string_lossy()),
        ReportFormat::Html => reporter.generate_html_report(&args.output.to_string_lossy()),
        ReportFormat::Markdown => reporter.generate_markdown_report(&args.output.to_string_lossy()),
    }
}

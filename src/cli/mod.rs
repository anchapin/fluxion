// CLI module for Fluxion
// This module provides the main CLI interface and subcommand structure

pub mod commands;
pub mod hvac_commands;
pub mod multi_zone;
pub mod performance;
pub mod validation;

pub use commands::cross_validation::*;
pub use commands::cross_validation::{
    execute_report_command, execute_run_command, execute_validate_command,
};
pub use multi_zone::*;
pub use performance::PerformanceCommand;
pub use validation::ValidationSubcommand;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Main CLI structure
#[derive(Parser)]
#[command(name = "fluxion")]
#[command(about = "Fluxion Building Energy Modeling CLI", long_about = None)]
#[command(
    after_help = "Examples:\n  fluxion validation run 800\n  fluxion validation run-series 800-810\n  fluxion validation cross-validate 800 --tool energyplus --reference-file results/case_800.csv"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Top-level CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Multi-zone building energy modeling commands
    #[command(subcommand)]
    MultiZone(MultiZoneCommand),

    /// Building energy model validation commands
    #[command(subcommand)]
    Validation(ValidationSubcommand),

    /// Performance testing and validation commands
    #[command(subcommand)]
    Performance(performance::PerformanceCommand),

    /// ESP-r cross-validation commands
    #[command(subcommand)]
    CrossValidation(CrossValidationCommand),
}

/// Parse and execute CLI commands
pub fn run_cli() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();

    match cli.command {
        Commands::MultiZone(subcommand) => match subcommand {
            MultiZoneCommand::Simulate(cmd) => multi_zone::execute_simulate_command(&cmd),
            MultiZoneCommand::Hvac(cmd) => multi_zone::execute_hvac_command(cmd),
            MultiZoneCommand::Validate(cmd) => multi_zone::execute_validate_command(&cmd),
            MultiZoneCommand::Performance(cmd) => multi_zone::execute_performance_command(&cmd),
        },
        Commands::Validation(cmd) => {
            validation::handle_validation_command(&cmd)?;
            Ok(())
        }
        Commands::Performance(cmd) => {
            performance::handle_performance_command(&cmd).map_err(|e| anyhow::anyhow!(e))?;
            Ok(())
        }
        Commands::CrossValidation(cmd) => {
            match cmd {
                CrossValidationCommand::Run(args) => {
                    execute_run_command(&args).map_err(|e| anyhow::anyhow!(e.to_string()))?
                }
                CrossValidationCommand::Report(args) => {
                    execute_report_command(&args).map_err(|e| anyhow::anyhow!(e.to_string()))?
                }
                CrossValidationCommand::Validate(args) => {
                    execute_validate_command(&args).map_err(|e| anyhow::anyhow!(e.to_string()))?
                }
            };
            Ok(())
        }
    }
}

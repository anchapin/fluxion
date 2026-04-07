// CLI module for Fluxion
// This module provides the main CLI interface and subcommand structure

pub mod multi_zone;

pub use multi_zone::*;

use clap::{Parser, Subcommand};

/// Main CLI structure
#[derive(Parser)]
#[command(name = "fluxion")]
#[command(about = "Fluxion Building Energy Modeling CLI", long_about = None)]
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
    // Other commands would be added here
    // Validate, Simulate, etc.
}

/// Parse and execute CLI commands
pub fn run_cli() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();

    match cli.command {
        Commands::MultiZone(subcommand) => match subcommand {
            MultiZoneCommand::Simulate(cmd) => multi_zone::execute_simulate_command(&cmd),
            MultiZoneCommand::Validate(cmd) => multi_zone::execute_validate_command(&cmd),
            MultiZoneCommand::Performance(cmd) => multi_zone::execute_performance_command(&cmd),
        },
    }
}

use clap::{Parser, Subcommand};
use fluxion::validation::ASHRAE140Validator;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "fluxion")]
#[command(about = "Fluxion Building Energy Modeling CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validates the engine against ASHRAE Standard 140
    Validate {
        /// Run all validation cases
        #[arg(short, long)]
        all: bool,

        /// Run a specific case (e.g., "600")
        #[arg(short, long)]
        case: Option<String>,

        /// Output format
        #[arg(short, long, default_value = "markdown")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output_file: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Validate { all, case, format, output_file } => {
            let mut validator = ASHRAE140Validator::new();
            
            // For now, we always run the analytical engine validation
            let report = validator.validate_analytical_engine();
            
            let output = match format.as_str() {
                "markdown" => report.to_markdown(),
                "csv" => report.to_csv(),
                "json" => report.to_json(),
                "html" => report.to_html(),
                _ => anyhow::bail!("Unsupported format: {}", format),
            };

            if let Some(path) = output_file {
                std::fs::write(&path, output)?;
                println!("Report saved to {:?}", path);
            } else {
                println!("{}", output);
            }
        }
    }

    Ok(())
}

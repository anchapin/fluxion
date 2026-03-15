//! CSV Export Tool for ASHRAE 140 Diagnostics
//!
//! This standalone CLI tool runs ASHRAE 140 validation cases and exports
//! hourly time series data to CSV files for external analysis.
//!
//! ## Usage
//!
//! ```bash
//! # Export default cases (900, 960) to output/csv/
//! cargo run --bin export_csv --
//!
//! # Export specific cases
//! cargo run --bin export_csv -- --cases 600,650,900
//!
//! # Use semicolon delimiter (European format)
//! cargo run --bin export_csv -- --delimiter ';'
//!
//! # Specify output directory
//! cargo run --bin export_csv -- --output-dir results/csv
//! ```
//!
//! Output structure:
//! ```text
//! output/csv/
//! ├── 900/
//! │   ├── case_900_zone0.csv
//! │   ├── case_900_zone1.csv (if multi-zone)
//! │   └── metadata.json
//! ├── 960/
//! │   ├── case_960_zone0.csv
//! │   ├── case_960_zone1.csv
//! │   └── metadata.json
//! └── ...
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use fluxion::validation::{
    ashrae_140_cases::ASHRAE140Case, export::CsvExporter, ASHRAE140Validator,
};
use std::path::PathBuf;

/// Command-line arguments for the CSV export tool.
#[derive(Parser, Debug)]
#[command(
    name = "export_csv",
    about = "Export ASHRAE 140 simulation diagnostics to CSV",
    long_about = None
)]
struct Args {
    /// Comma-separated list of case IDs to export (e.g., "900,960,600")
    #[arg(short, long, default_value = "900,960")]
    cases: String,

    /// Output directory for CSV files (default: "output/csv")
    #[arg(short, long, default_value = "output/csv")]
    output_dir: String,

    /// CSV field delimiter (default: ',')
    #[arg(short, long, default_value = ",")]
    delimiter: char,
}

/// Convert a case ID string to the corresponding ASHRAE140Case enum variant.
///
/// Supports all standard ASHRAE 140 cases.
fn parse_case_id(id: &str) -> Option<ASHRAE140Case> {
    match id {
        "600" => Some(ASHRAE140Case::Case600),
        "610" => Some(ASHRAE140Case::Case610),
        "620" => Some(ASHRAE140Case::Case620),
        "630" => Some(ASHRAE140Case::Case630),
        "640" => Some(ASHRAE140Case::Case640),
        "650" => Some(ASHRAE140Case::Case650),
        "600FF" => Some(ASHRAE140Case::Case600FF),
        "650FF" => Some(ASHRAE140Case::Case650FF),
        "900" => Some(ASHRAE140Case::Case900),
        "910" => Some(ASHRAE140Case::Case910),
        "920" => Some(ASHRAE140Case::Case920),
        "930" => Some(ASHRAE140Case::Case930),
        "940" => Some(ASHRAE140Case::Case940),
        "950" => Some(ASHRAE140Case::Case950),
        "900FF" => Some(ASHRAE140Case::Case900FF),
        "950FF" => Some(ASHRAE140Case::Case950FF),
        "960" => Some(ASHRAE140Case::Case960),
        "195" => Some(ASHRAE140Case::Case195),
        _ => None,
    }
}

fn main() -> Result<()> {
    // Parse command-line arguments
    let args = Args::parse();

    // Parse the list of case IDs
    let case_ids: Vec<&str> = args.cases.split(',').map(|s| s.trim()).collect();

    if case_ids.is_empty() {
        anyhow::bail!("No cases specified. Use --cases to provide at least one case ID.");
    }

    // Create validator with full diagnostics enabled
    let mut validator = ASHRAE140Validator::with_full_diagnostics();

    // Create CSV exporter
    let output_dir = PathBuf::from(&args.output_dir);
    let exporter = CsvExporter::new(output_dir, args.delimiter);

    // Process each case
    for case_id_str in case_ids {
        let case = match parse_case_id(case_id_str) {
            Some(c) => c,
            None => {
                eprintln!("Warning: Unknown case ID '{}', skipping.", case_id_str);
                continue;
            }
        };

        println!("=== Validating and exporting case {} ===", case_id_str);

        // Run validation with diagnostics
        let (report, collector) = validator.validate_single_case_with_diagnostics(case);

        // Get case specification
        let spec = case.spec();

        // Export diagnostics CSV files (one per zone)
        exporter
            .export_diagnostics(case_id_str, &collector, &spec)
            .with_context(|| format!("Failed to export diagnostics for case {}", case_id_str))?;

        // Export metadata JSON
        exporter
            .export_metadata(case_id_str, &spec, &report, &collector)
            .with_context(|| format!("Failed to export metadata for case {}", case_id_str))?;

        println!(
            "Exported case {} to {}/{}",
            case_id_str, args.output_dir, case_id_str
        );
    }

    println!("\nAll exports completed successfully.");
    Ok(())
}

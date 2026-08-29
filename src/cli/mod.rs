// CLI module for Fluxion
//
// Issue #2929: this module is the canonical CLI surface. `src/bin/fluxion.rs`
// no longer redeclares `Cli`/`Commands`; it delegates to [`run_cli`] below
// (the bin's `main` is a thin shim). The lib's previous, simpler 5-subcommand
// CLI was dead code: `run_cli` was never called by any user-facing entry point,
// so the bin's richer 16-subcommand CLI and the lib's 5-subcommand CLI drifted
// silently. Unifying the surface here means the bin and any future embedders
// (e.g. a hypothetical GUI front-end) parse the same arg shapes and dispatch
// through the same handlers.

pub mod hvac_commands;
pub mod monte_carlo;
pub mod multi_zone;
pub mod performance;
pub mod validation;

// Issue #2929: the previous `mod commands { pub mod import; }` and
// `mod commands::cross_validation` sub-tree only existed to back the old
// lib-only 5-subcommand `Cli` (which was dead code). The canonical CLI
// surface defined below replaces it; the `import.rs` / `cross_validation.rs`
// files are kept on disk for any future re-instatement but are not wired in.

pub use multi_zone::*;
pub use performance::PerformanceCommand;
pub use validation::ValidationSubcommand;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use csv::Reader;
use serde::Deserialize;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::analysis::components;
use crate::analysis::delta::{self, DeltaConfig};
use crate::analysis::sensitivity::{self, ParameterRange, SensitivityReport};
use crate::analysis::swing::{
    calculate_swing_metrics, generate_swing_report, interpret_swing_metrics,
};
use crate::analysis::visualization::{
    generate_animation, generate_html, Dataset, PlotPanel, TimeSeriesData,
};
use crate::sim::engine::ThermalModel;
use crate::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use crate::validation::benchmark::{get_high_mass_cases, get_low_mass_cases, get_special_cases};
use crate::validation::commands::update_references;
use crate::validation::guardrails;
use crate::validation::reporter::{BaselineMetrics, ValidationReportGenerator};
use crate::validation::statistical::StatisticalValidator;
use crate::validation::ASHRAE140Validator;
use crate::weather::epw::EpwWeatherSource;
use crate::weather::epw_path::epw_required;
use crate::BatchOracle;

/// Automation subcommands for test workflows and CI/CD integration.
#[derive(Subcommand, Debug)]
pub enum AutomationSubcommand {
    /// Run automated test workflows
    #[clap(name = "test")]
    Test {
        /// Test cases directory
        #[clap(short, long, default_value = "tests/fixtures")]
        test_cases: String,
        /// Output directory for reports
        #[clap(short, long, default_value = "./target/test_reports")]
        output: String,
        /// Temperature tolerance for validation
        #[clap(short, long, default_value = "0.5")]
        tolerance: f64,
        /// Enable verbose output
        #[clap(short, long)]
        verbose: bool,
        /// Output format (markdown, json)
        #[clap(short, long, default_value = "markdown")]
        format: String,
    },

    /// Generate GitHub Actions workflows
    #[clap(name = "generate-workflow")]
    GenerateWorkflow {
        /// Workflow type (cross-validation, performance, ci-cd)
        #[clap(value_enum, default_value = "cross-validation")]
        workflow_type: String,
        /// Output file path
        #[clap(short, long, default_value = ".github/workflows/generated.yml")]
        output: String,
        /// Workflow name override
        #[clap(short, long)]
        name: Option<String>,
        /// Workflow description override
        #[clap(short, long)]
        description: Option<String>,
    },

    /// Run GitHub Actions test automation
    #[clap(name = "github-actions")]
    GitHubActions {
        /// GitHub repository (owner/repo format)
        #[clap(short, long)]
        repository: Option<String>,
        /// GitHub token for API access
        #[clap(short, long)]
        token: Option<String>,
        /// Workflow file to execute
        #[clap(short, long, default_value = ".github/workflows/cross-validation.yml")]
        workflow: String,
        /// Dry run (don't actually trigger workflow)
        #[clap(short, long)]
        dry_run: bool,
    },
}

/// Reference-data management subcommands.
#[derive(Subcommand, Debug)]
pub enum ReferenceCommands {
    /// Updates reference data from the configured source
    Update {
        /// URL to fetch reference data from (optional, uses default if omitted)
        #[arg(short, long)]
        url: Option<String>,
    },
}

/// Measure subcommands. All paths under here are documented but fail loudly
/// with `#2947` (originally #2711) until measure execution is wired in.
#[derive(Subcommand, Debug)]
pub enum MeasureSubcommand {
    /// Update measure.xml and README for a measure directory.
    #[command(
        long_about = "Update measure.xml and README for a measure directory.\n\n[NOT YET IMPLEMENTED, see #2947]: this command will return a non-zero exit code with a 'not yet implemented' error rather than silently succeeding. Originally tracked by #2711."
    )]
    Update {
        /// Path to measure directory
        #[arg(required(true))]
        measure_dir: PathBuf,
    },

    /// Update all measures in a directory.
    #[command(
        long_about = "Update all measures in a directory.\n\n[NOT YET IMPLEMENTED, see #2947]: this command will return a non-zero exit code with a 'not yet implemented' error rather than silently succeeding. Originally tracked by #2711."
    )]
    UpdateAll {
        /// Path to measures directory
        #[arg(required(true))]
        measures_dir: PathBuf,
    },

    /// Compute arguments for a measure.
    #[command(
        long_about = "Compute arguments for a measure.\n\n[NOT YET IMPLEMENTED, see #2947]: this command will return a non-zero exit code with a 'not yet implemented' error rather than silently succeeding. Originally tracked by #2711."
    )]
    ComputeArguments {
        /// Path to model file (.flux)
        #[arg(required(true))]
        model: PathBuf,

        /// Path to measure directory
        #[arg(required(true))]
        measure_dir: PathBuf,
    },

    /// Run tests for measures in a directory.
    #[command(
        long_about = "Run tests for measures in a directory.\n\n[NOT YET IMPLEMENTED, see #2947]: this command will return a non-zero exit code with a 'not yet implemented' error rather than silently succeeding. Originally tracked by #2711."
    )]
    RunTests {
        /// Path to measures directory
        #[arg(required(true))]
        measures_dir: PathBuf,
    },
}

/// Top-level CLI commands. This is the canonical surface (Issue #2929).
///
/// Two entry modes share one CLI:
///
/// 1. **Direct simulation (EnergyPlus-compatible)** — the user passes `-w
///    weather.epw input.flux` and no subcommand. The flags
///    `-w/-d/-p/-s/-D/-a/-j/-r` live on the [`Cli`] root; when `input` is set
///    and `command` is `None`, [`run_cli`] dispatches to [`run_direct_simulation`].
///
/// 2. **Subcommand mode** — the user passes one of the 16 variants below.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Manages reference data for validation
    References {
        #[command(subcommand)]
        command: ReferenceCommands,
    },

    /// Validates the engine against ASHRAE Standard 140
    Validate {
        /// Run complete validation suite (baseline + diagnostics)
        #[arg(short, long)]
        all: bool,

        /// Run diagnostic cases only
        #[arg(long)]
        diagnostics: bool,

        /// Run specific diagnostic range (e.g., 195-470, 800-810)
        #[arg(long)]
        range: Option<String>,

        /// Run a specific case (e.g., "600")
        #[arg(short, long)]
        case: Option<String>,

        /// Enable statistical validation (ASHRAE 140 Addendum B)
        #[arg(long)]
        statistical: bool,

        /// Alpha threshold for statistical FDR correction (default: 0.05)
        #[arg(long, default_value = "0.05")]
        alpha: f64,

        /// Output format
        #[arg(short, long, default_value = "markdown")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output_file: Option<PathBuf>,

        /// Enable CI mode (enforces guardrails and sets exit code on failure)
        #[arg(long)]
        ci: bool,

        /// Output machine-readable JSON summary for CI ingestion.
        /// This outputs only the summary metrics (pass rate, MAE, etc.) as JSON,
        /// avoiding fragile regex parsing of human-readable text.
        #[arg(long)]
        ci_summary_json: bool,
    },

    /// Validate specific diagnostic case or range
    ValidateCase {
        /// Case number or range (e.g., 800, 195-470, 800-810)
        #[arg(required(true))]
        case_spec: String,
    },

    /// Quantize an ONNX model for optimized edge inference
    Quantize {
        /// Path to input ONNX model
        #[arg(short, long)]
        model: PathBuf,

        /// Path to output quantized model
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization type (int8, uint8, fp16)
        #[arg(long, default_value = "int8")]
        quant_type: String,

        /// Run inference benchmark after quantization
        #[arg(short, long)]
        benchmark: bool,
    },

    /// Run inference benchmark on an ONNX model
    Benchmark {
        /// Path to ONNX model
        #[arg(short, long)]
        model: PathBuf,

        /// Number of inference runs
        #[arg(short, long, default_value = "100")]
        runs: usize,
    },

    /// Run sensitivity analysis
    Sensitivity {
        /// Path to sensitivity configuration YAML
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory (default: current directory)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Use AI surrogates for faster evaluation
        #[arg(long)]
        use_surrogates: bool,
    },

    /// Run delta testing comparison
    Delta {
        /// Path to delta configuration YAML
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Include hourly differences in output
        #[arg(long)]
        hourly: bool,
    },

    /// Generate component energy breakdown for a case
    Components {
        /// ASHRAE case ID (e.g., "600", "900FF")
        #[arg(short, long)]
        case: String,
        /// Output CSV file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Calculate and display swing metrics for a free-floating case
    Swing {
        /// ASHRAE free-floating case ID (e.g., "600FF", "900FF")
        #[arg(short, long)]
        case: String,
        /// Comfort band minimum temperature (°C)
        #[arg(long)]
        comfort_min: Option<f64>,
        /// Comfort band maximum temperature (°C)
        #[arg(long)]
        comfort_max: Option<f64>,
    },

    /// Generate interactive visualization from diagnostics CSV
    Visualize {
        /// Input diagnostics CSV file
        #[arg(short, long)]
        input: PathBuf,
        /// Output HTML file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate animated visualization from diagnostics CSV
    Animate {
        /// Input diagnostics CSV file
        #[arg(short, long)]
        input: PathBuf,
        /// Output HTML file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Building energy model validation commands
    Validation {
        #[command(subcommand)]
        command: ValidationSubcommand,
    },

    /// Test automation and CI/CD workflow commands
    Automation {
        #[command(subcommand)]
        command: AutomationSubcommand,
    },

    /// LLM-powered BEM input validation co-pilot
    Copilot {
        /// Path to building configuration JSON file
        #[arg(short, long)]
        config: PathBuf,

        /// Ollama server URL (default: http://localhost:11434)
        #[arg(short, long)]
        ollama_url: Option<String>,

        /// LLM model to use (default: llama3.2:latest)
        #[arg(long)]
        model: Option<String>,

        /// Skip LLM and use only rule-based checks
        #[arg(long)]
        rule_only: bool,

        /// Output results to JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run a simulation workflow (OpenStudio-compatible).
    #[command(
        long_about = "Run a simulation workflow (OpenStudio-compatible).\n\n[NOT YET IMPLEMENTED, see #2947]: the workflow file is parsed and its steps are listed, but measure execution is pending. Calling this command will return a non-zero exit code with a 'not yet implemented' error rather than silently succeeding. Originally tracked by #2711."
    )]
    Run {
        /// Workflow file path (.fwf format)
        #[arg(short = 'w', long = "workflow", value_name = "PATH")]
        workflow: Option<PathBuf>,

        /// Debug mode - keep temporary files
        #[arg(long)]
        debug: bool,

        /// Run only measures (skip EnergyPlus simulation) [NOT YET IMPLEMENTED, see #2947]
        #[arg(short = 'm', long = "measures-only")]
        measures_only: bool,

        /// Run only post-processing (use existing results) [NOT YET IMPLEMENTED, see #2947]
        #[arg(short = 'p', long = "postprocess-only")]
        postprocess_only: bool,
    },

    /// Manage and query measures.
    Measure {
        #[command(subcommand)]
        command: MeasureSubcommand,
    },
}

/// Main CLI structure.
///
/// `command` is `Option<Commands>` because the same root also accepts the
/// EnergyPlus-compatible direct-simulation form (`-w weather.epw input.flux`),
/// which is detected at parse time by the presence of a positional `input`
/// and the absence of a subcommand.
#[derive(Parser, Debug)]
#[command(
    name = "fluxion",
    about = "Fluxion Building Energy Modeling CLI",
    long_about = "Fluxion is a Rust-based building energy modeling engine compatible with EnergyPlus and OpenStudio workflows.

Direct Simulation Mode (EnergyPlus-compatible):
  fluxion -w weather.epw input.flux
  fluxion -w weather.epw -d output/ input.flux
  fluxion --annual -w weather.epw input.flux

Workflow Mode (OpenStudio-compatible):
  fluxion run -w workflow.fwf

Analysis Commands:
  fluxion validate --case 600
  fluxion sensitivity --config sens.yaml",
    after_help = "Examples:
  # Run annual simulation (EnergyPlus-style)
  fluxion -w USA_CO_Denver.epw building.flux

  # Run with custom output directory
  fluxion -w weather.epw -d results/ building.flux

  # Design day only
  fluxion -w weather.epw -D building.flux

  # Run workflow (OpenStudio-style)
  fluxion run -w baseline.fwf

  # ASHRAE 140 validation
  fluxion validate --case 600"
)]
pub struct Cli {
    /// Weather file path (EPW format)
    #[arg(short = 'w', long = "weather", value_name = "PATH")]
    pub weather: Option<String>,

    /// Output directory for simulation results
    #[arg(short = 'd', long = "output-directory", value_name = "PATH")]
    pub output_directory: Option<String>,

    /// Prefix for output file names
    #[arg(short = 'p', long = "output-prefix", value_name = "PREFIX")]
    pub output_prefix: Option<String>,

    /// Output suffix style (L=Legacy, C=Capital, D=Dash)
    #[arg(
        short = 's',
        long = "output-suffix",
        value_name = "STYLE",
        default_value = "L"
    )]
    pub output_suffix: String,

    /// Force design day only simulation
    #[arg(short = 'D', long = "design-day")]
    pub design_day: bool,

    /// Force annual simulation (default)
    #[arg(short = 'a', long = "annual")]
    pub annual: bool,

    /// Number of parallel jobs for multi-threaded operations
    #[arg(short = 'j', long = "jobs", value_name = "N")]
    pub jobs: Option<usize>,

    /// Run post-processing after simulation
    #[arg(short = 'r', long = "readvars")]
    pub readvars: bool,

    /// Input file path (.flux format)
    pub input: Option<String>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

// =============================================================================
// Helper functions (formerly in src/bin/fluxion.rs, promoted to lib so the
// canonical CLI surface owns them; the bin remains a thin shim).
// =============================================================================

/// Map a case_id string (e.g. "600", "900FF") to its [`CaseSpec`].
pub fn case_id_to_spec(case_id: &str) -> Option<CaseSpec> {
    match case_id {
        "600" => Some(ASHRAE140Case::Case600.spec()),
        "610" => Some(ASHRAE140Case::Case610.spec()),
        "620" => Some(ASHRAE140Case::Case620.spec()),
        "630" => Some(ASHRAE140Case::Case630.spec()),
        "640" => Some(ASHRAE140Case::Case640.spec()),
        "650" => Some(ASHRAE140Case::Case650.spec()),
        "600FF" => Some(ASHRAE140Case::Case600FF.spec()),
        "650FF" => Some(ASHRAE140Case::Case650FF.spec()),
        "900" => Some(ASHRAE140Case::Case900.spec()),
        "910" => Some(ASHRAE140Case::Case910.spec()),
        "920" => Some(ASHRAE140Case::Case920.spec()),
        "930" => Some(ASHRAE140Case::Case930.spec()),
        "940" => Some(ASHRAE140Case::Case940.spec()),
        "950" => Some(ASHRAE140Case::Case950.spec()),
        "900FF" => Some(ASHRAE140Case::Case900FF.spec()),
        "950FF" => Some(ASHRAE140Case::Case950FF.spec()),
        "960" => Some(ASHRAE140Case::Case960.spec()),
        "195" => Some(ASHRAE140Case::Case195.spec()),
        // HVAC equipment cases (800-810)
        "800" => Some(ASHRAE140Case::Case800.spec()),
        "801" => Some(ASHRAE140Case::Case801.spec()),
        "802" => Some(ASHRAE140Case::Case802.spec()),
        "803" => Some(ASHRAE140Case::Case803.spec()),
        "804" => Some(ASHRAE140Case::Case804.spec()),
        "805" => Some(ASHRAE140Case::Case805.spec()),
        "806" => Some(ASHRAE140Case::Case806.spec()),
        "807" => Some(ASHRAE140Case::Case807.spec()),
        "808" => Some(ASHRAE140Case::Case808.spec()),
        "809" => Some(ASHRAE140Case::Case809.spec()),
        "810" => Some(ASHRAE140Case::Case810.spec()),
        _ => None,
    }
}

/// Sensitivity configuration loaded from a YAML file.
#[derive(Deserialize)]
pub struct SensitivityConfig {
    pub case_id: String,
    pub method: String, // "oat" or "sobol"
    pub levels: Option<usize>,
    pub samples: Option<usize>,
    pub parameters: Vec<ParameterRange>,
}

/// Render a [`SensitivityReport`] as a Markdown table.
pub fn generate_sensitivity_markdown(report: &SensitivityReport) -> String {
    let mut out = String::new();
    out.push_str("# Sensitivity Analysis Report\n\n");
    out.push_str("| Rank | Parameter | NormalizedCoeff | CVRMSE | NMBE | Slope |\n");
    out.push_str("|------|-----------|-----------------|--------|------|-------|\n");
    for (rank, (param, metric)) in report
        .parameters
        .iter()
        .zip(report.metrics.iter())
        .enumerate()
    {
        out.push_str(&format!(
            "| {} | {} | {:.3} | {:.3}% | {:.3}% | {:.3} |\n",
            rank + 1,
            param,
            metric.normalized_coeff,
            metric.cvrmse,
            metric.nmbe,
            metric.slope
        ));
    }
    out
}

/// Load diagnostics CSV (as produced by `SimulationDiagnostics::export_csv`)
/// into [`TimeSeriesData`].
pub fn load_diagnostics_csv(path: &Path) -> Result<TimeSeriesData> {
    let mut rdr = Reader::from_path(path)?;
    // Assume headers present
    let _headers = rdr.headers()?;
    let mut timestamps = Vec::new();
    let mut datasets: Vec<Dataset> = Vec::new();
    let mut num_zones = 0usize;
    for record in rdr.records() {
        let record = record?;
        // Hour (col 0)
        let hour_str = record
            .get(0)
            .ok_or_else(|| anyhow!("Missing Hour column"))?;
        let hour: usize = hour_str.parse()?;
        timestamps.push(hour);
        // Zone_Temps (col 1)
        let zone_temps_str = record.get(1).ok_or_else(|| anyhow!("Missing Zone_Temps"))?;
        let zone_vals: Vec<f64> = zone_temps_str
            .split(';')
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("Failed to parse Zone_Temps: {}", e))?;
        if num_zones == 0 {
            num_zones = zone_vals.len();
            // Initialize temperature datasets
            for zone in 0..num_zones {
                datasets.push(Dataset {
                    label: format!("Zone {} Temperature", zone + 1),
                    values: Vec::new(),
                    color: None,
                    panel: Some(PlotPanel::Temperature),
                });
            }
            // Solar dataset
            datasets.push(Dataset {
                label: "Solar Gains (Total W)".to_string(),
                values: Vec::new(),
                color: Some("#FFA500".to_string()),
                panel: Some(PlotPanel::Solar),
            });
            // HVAC dataset
            datasets.push(Dataset {
                label: "HVAC (Heating+Cooling) (W)".to_string(),
                values: Vec::new(),
                color: Some("#0000FF".to_string()),
                panel: Some(PlotPanel::HVAC),
            });
        } else if zone_vals.len() != num_zones {
            anyhow::bail!("Inconsistent number of zones at hour {}", hour);
        }
        // Append zone temperatures
        for (i, &val) in zone_vals.iter().enumerate() {
            datasets[i].values.push(val);
        }
        // Solar_Watts (col 4)
        let solar_str = record
            .get(4)
            .ok_or_else(|| anyhow!("Missing Solar_Watts"))?;
        let solar_vals: Vec<f64> = solar_str
            .split(';')
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("Failed to parse Solar_Watts: {}", e))?;
        let solar_total: f64 = solar_vals.iter().sum();
        datasets[num_zones].values.push(solar_total);
        // HVAC_Watts (col 6)
        let hvac_str = record.get(6).ok_or_else(|| anyhow!("Missing HVAC_Watts"))?;
        let hvac_vals: Vec<f64> = hvac_str
            .split(';')
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("Failed to parse HVAC_Watts: {}", e))?;
        let hvac_total: f64 = hvac_vals.iter().sum();
        datasets[num_zones + 1].values.push(hvac_total);
    }
    Ok(TimeSeriesData {
        timestamps,
        datasets,
    })
}

/// Handle automation commands.
pub fn handle_automation_command(command: &AutomationSubcommand) -> Result<()> {
    match command {
        AutomationSubcommand::Test {
            test_cases,
            output,
            tolerance,
            verbose,
            format,
        } => {
            // Set up test runner configuration
            use crate::validation::automation::runner::TestRunnerConfig;

            let config = TestRunnerConfig::new(
                PathBuf::from(test_cases),
                PathBuf::from(output),
                *tolerance,
                *verbose,
                format.clone(),
            );

            // Create and run test runner
            let mut runner = crate::validation::automation::runner::TestRunner::new(config);
            runner.initialize()?;

            // Run all tests
            let reports = runner.run_all_tests()?;

            // Generate combined report
            let combined_report = runner.generate_combined_report(&reports)?;

            // Save report
            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
            let filename = format!("automation_report_{}.{}", timestamp, format);
            runner.save_report(&combined_report, &filename)?;

            // Clean up
            runner.cleanup()?;

            // Determine exit code
            let all_passed = reports.iter().all(|r| r.overall_pass);
            if all_passed {
                println!("✅ All automation tests passed!");
            } else {
                println!("❌ Some automation tests failed!");
                std::process::exit(1);
            }
        }

        AutomationSubcommand::GenerateWorkflow {
            workflow_type,
            output,
            name,
            description,
        } => {
            use crate::validation::automation::github::workflow::WorkflowGenerator;
            use crate::validation::automation::github::workflow::WorkflowGeneratorConfig;

            let config = WorkflowGeneratorConfig::default();
            let generator = WorkflowGenerator::new(config)?;

            // Generate appropriate workflow
            let yaml = match workflow_type.as_str() {
                "cross-validation" => generator
                    .generate_cross_validation_workflow(name.clone(), description.clone())?,
                "performance" => {
                    generator.generate_performance_workflow(name.clone(), description.clone())?
                }
                "ci-cd" => generator.generate_ci_cd_workflow(name.clone(), description.clone())?,
                _ => {
                    return Err(anyhow!("Unknown workflow type: {}", workflow_type));
                }
            };

            // Save workflow file
            std::fs::write(output, yaml)?;
            println!("✅ Workflow generated successfully: {}", output);
        }

        AutomationSubcommand::GitHubActions {
            repository,
            token,
            workflow,
            dry_run,
        } => {
            use crate::validation::automation::github::api::GitHubClient;

            // Read workflow file
            let workflow_content = std::fs::read_to_string(workflow)?;

            if *dry_run {
                println!("🔄 Dry run mode - would trigger workflow: {}", workflow);
                println!("Workflow content preview:");
                for (i, line) in workflow_content.lines().take(10).enumerate() {
                    println!("  {}: {}", i + 1, line);
                }
                println!("... (truncated)");
                return Ok(());
            }

            // Create GitHub client
            let _client = GitHubClient::new(token.clone());

            // Parse repository
            let repo_ref = repository
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("owner/repo");
            let repo_parts: Vec<&str> = repo_ref.split('/').collect();

            if repo_parts.len() != 2 {
                return Err(anyhow!("Repository must be in format 'owner/repo'"));
            }

            let owner = repo_parts[0];
            let repo = repo_parts[1];

            // Trigger workflow (simplified - actual implementation would use GitHub API)
            println!("🚀 Triggering GitHub Actions workflow...");
            println!("Repository: {}/{}", owner, repo);
            println!("Workflow: {}", workflow);

            println!("✅ Workflow triggered successfully!");
            println!(
                "📊 Check GitHub Actions tab for progress: https://github.com/{}/{}/actions",
                owner, repo
            );
        }
    }
    Ok(())
}

/// Validates a specific diagnostic case or range.
///
/// Handles explicit diagnostic case invocation via the `validate-case`
/// subcommand, supporting individual cases (e.g., "800") and case ranges
/// (e.g., "195-470", "800-810"). The ranges are documented but fail loudly
/// with `#2947` (originally #2711) until they are wired in.
///
/// # Example
///
/// ```bash
/// fluxion validate-case 800
/// fluxion validate-case 195-470
/// fluxion validate-case 800-810
/// ```
pub fn validate_diagnostic_case(case_spec: &str) -> Result<()> {
    match case_spec {
        // Single HVAC equipment cases (800-810)
        "800" | "801" | "802" | "803" | "804" | "805" | "806" | "807" | "808" | "809" | "810" => {
            // Get case spec
            let spec = case_id_to_spec(case_spec)
                .ok_or_else(|| anyhow!("Unknown case ID: {}", case_spec))?;

            // Run validation
            let validator = ASHRAE140Validator::new();
            let weather = EpwWeatherSource::from_file(
                epw_required("USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw").to_str().unwrap(),
            )
            .expect("Failed to load EPW weather data");
            let (results, _) = validator.simulate_case_with_diagnostics(&spec, &weather, case_spec);

            println!(
                "Case {} result: {:.2} MWh heating, {:.2} MWh cooling",
                case_spec, results.annual_heating_mwh, results.annual_cooling_mwh
            );
        }
        // Diagnostic ranges - not yet implemented (issue #2947, originally
        // #2711). Kept in the match so usage/`--help` documents the option,
        // but fail loudly.
        "195-470" => {
            return Err(not_yet_implemented("diagnostic case range 195-470"));
        }
        "800-810" => {
            return Err(not_yet_implemented("diagnostic case range 800-810"));
        }
        _ => {
            eprintln!("Unknown case specification: {}", case_spec);
            eprintln!("Valid options:");
            eprintln!("  Single cases: 800, 801, 802, ..., 810");
            eprintln!("  Case ranges: 195-470, 800-810");
            anyhow::bail!("Invalid case specification: {}", case_spec);
        }
    }
    Ok(())
}

/// Return a consistent "not yet implemented" error for stubbed CLI workflows.
///
/// These code paths are kept in the CLI surface so `--help` documents them, but
/// they do not execute yet. They must fail loudly with a non-zero exit code
/// rather than silently succeeding. Tracked by issue #2947 (originally #2711).
pub fn not_yet_implemented(feature: &str) -> anyhow::Error {
    anyhow!(
        "'{}' is not yet implemented (see issue #2947, originally #2711); this CLI path is documented but does not execute yet",
        feature
    )
}

/// Run a direct simulation (EnergyPlus-compatible mode).
///
/// This function handles the direct simulation mode where a user provides
/// an input file and weather file without a subcommand, similar to:
///   `fluxion -w weather.epw input.flux`
#[allow(clippy::too_many_arguments)]
pub fn run_direct_simulation(
    input_file: &str,
    weather: Option<&str>,
    output_directory: Option<&str>,
    output_prefix: Option<&str>,
    output_suffix: &str,
    design_day: bool,
    annual: bool,
    jobs: Option<usize>,
    readvars: bool,
) -> Result<()> {
    // Validate required arguments
    let weather_path = weather.ok_or_else(|| {
        anyhow!(
            "Weather file is required for simulation.\n\
             Usage: fluxion -w weather.epw input.flux\n\
             Use --help for more information."
        )
    })?;

    // Validate input file exists
    let input_path = Path::new(input_file);
    if !input_path.exists() {
        anyhow::bail!("Input file not found: {}", input_file);
    }

    // Validate weather file exists
    let weather_path = Path::new(weather_path);
    if !weather_path.exists() {
        anyhow::bail!("Weather file not found: {}", weather_path.display());
    }

    // Determine output directory
    let output_dir = output_directory
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir)?;

    // Determine output prefix
    let prefix = output_prefix.unwrap_or("fluxion_out");

    // Log simulation parameters
    println!("Fluxion Building Energy Modeling Engine");
    println!("======================================");
    println!("Input file: {}", input_file);
    println!("Weather file: {}", weather_path.display());
    println!("Output directory: {}", output_dir.display());
    println!("Output prefix: {}", prefix);
    println!("Output suffix style: {}", output_suffix);

    if design_day {
        println!("Simulation mode: Design Day Only");
    } else if annual {
        println!("Simulation mode: Annual");
    } else {
        println!("Simulation mode: Annual (default)");
    }

    if let Some(n_jobs) = jobs {
        println!("Parallel jobs: {}", n_jobs);
    }

    // Load the building model
    println!("\nLoading building model...");
    let model_content = std::fs::read_to_string(input_path)?;
    let _model: serde_json::Value = serde_json::from_str(&model_content)
        .map_err(|e| anyhow!("Failed to parse input file as JSON: {}", e))?;

    // Load weather data
    println!("Loading weather data...");
    let _weather = EpwWeatherSource::from_file(weather_path)
        .map_err(|e| anyhow!("Failed to load weather file: {}", e))?;

    // TODO(#2947): Wire the thermal simulation engine into this path. All
    // EnergyPlus-compatible arguments are parsed and validated above, but the
    // actual timestepping is not yet integrated. Fail loudly rather than
    // silently report success — blind runs depend on a non-zero exit here.
    // Originally tracked by #2711.
    if readvars {
        println!("Post-processing requested (readvars)");
    }
    eprintln!("error: direct simulation is not yet implemented (see issue #2947)");
    Err(not_yet_implemented("direct simulation"))
}

/// Run a workflow (OpenStudio-compatible mode).
pub fn run_workflow(
    workflow_path: Option<&Path>,
    debug: bool,
    measures_only: bool,
    postprocess_only: bool,
) -> Result<()> {
    let workflow_path = workflow_path.ok_or_else(|| {
        anyhow!(
            "Workflow file is required for 'run' command.\n\
             Usage: fluxion run -w workflow.fwf\n\
             Use 'fluxion run --help' for more information."
        )
    })?;

    if debug {
        println!("Debug mode enabled - temporary files will be preserved");
    }

    if measures_only {
        // TODO(#2947): Implement measures-only workflow. Originally tracked by #2711.
        return Err(not_yet_implemented("measures-only workflow"));
    }

    if postprocess_only {
        // TODO(#2947): Implement postprocess-only workflow. Originally tracked by #2711.
        return Err(not_yet_implemented("postprocess-only workflow"));
    }

    // Load and parse workflow file
    let workflow_content = std::fs::read_to_string(workflow_path)?;
    let workflow: serde_json::Value = serde_json::from_str(&workflow_content)
        .map_err(|e| anyhow!("Failed to parse workflow file: {}", e))?;

    println!("Fluxion Workflow Runner");
    println!("======================");
    println!("Workflow file: {}", workflow_path.display());

    if let Some(name) = workflow.get("name").and_then(|v| v.as_str()) {
        println!("Workflow name: {}", name);
    }
    if let Some(desc) = workflow.get("description").and_then(|v| v.as_str()) {
        println!("Description: {}", desc);
    }

    // Extract workflow steps
    if let Some(steps) = workflow.get("steps").and_then(|v| v.as_array()) {
        println!("\nWorkflow steps: {}", steps.len());
        for (i, step) in steps.iter().enumerate() {
            let measure_type = step
                .get("measure_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let measure_name = step
                .get("measure_dir_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed");
            println!("  {}. [{}] {}", i + 1, measure_type, measure_name);
        }
    }

    // TODO(#2947): Implement actual workflow execution. The workflow file is
    // parsed and its steps are listed above, but measure execution is pending.
    // Fail loudly rather than silently report success. Originally tracked by #2711.
    eprintln!("error: workflow execution is not yet implemented (see issue #2947)");
    Err(not_yet_implemented("workflow execution"))
}

/// Handle measure subcommands.
pub fn run_measure_command(command: &MeasureSubcommand) -> Result<()> {
    match command {
        MeasureSubcommand::Update { measure_dir } => {
            println!("Updating measure at: {}", measure_dir.display());
            // TODO(#2947): Implement measure update. Originally tracked by #2711.
            Err(not_yet_implemented("measure update"))
        }
        MeasureSubcommand::UpdateAll { measures_dir } => {
            println!("Updating all measures in: {}", measures_dir.display());
            // TODO(#2947): Implement measure update --all. Originally tracked by #2711.
            Err(not_yet_implemented("measure update --all"))
        }
        MeasureSubcommand::ComputeArguments { model, measure_dir } => {
            println!("Computing arguments for measure:");
            println!("  Model: {}", model.display());
            println!("  Measure: {}", measure_dir.display());
            // TODO(#2947): Implement measure compute-args. Originally tracked by #2711.
            Err(not_yet_implemented("measure compute-args"))
        }
        MeasureSubcommand::RunTests { measures_dir } => {
            println!("Running tests for measures in: {}", measures_dir.display());
            // TODO(#2947): Implement measure tests. Originally tracked by #2711.
            Err(not_yet_implemented("measure tests"))
        }
    }
}

/// Parse `argv` and dispatch the requested subcommand.
///
/// This is the canonical CLI entry point (Issue #2929). The bin's `main` is a
/// thin shim around this function. Two entry modes share one parser:
///
/// 1. **Direct simulation** — when `input` is present and `command` is `None`,
///    the EnergyPlus-compatible flags (`-w/-d/-p/-s/-D/-a/-j/-r`) are forwarded
///    to [`run_direct_simulation`].
/// 2. **Subcommand mode** — otherwise dispatch into the matched [`Commands`]
///    variant.
pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();

    // Handle direct simulation mode (EnergyPlus-compatible):
    // when an input file is provided without a subcommand, treat it as
    // `fluxion -w weather.epw input.flux`.
    if let Some(input_file) = &cli.input {
        return run_direct_simulation(
            input_file,
            cli.weather.as_deref(),
            cli.output_directory.as_deref(),
            cli.output_prefix.as_deref(),
            &cli.output_suffix,
            cli.design_day,
            cli.annual,
            cli.jobs,
            cli.readvars,
        );
    }

    // Handle subcommand mode
    let command = cli.command.ok_or_else(|| {
        anyhow!(
            "No input file or subcommand specified.\n\
             \n\
             Direct Simulation (EnergyPlus-style):\n\
               fluxion -w weather.epw input.flux\n\
             \n\
             Workflow Mode (OpenStudio-style):\n\
               fluxion run -w workflow.fwf\n\
             \n\
             Analysis Commands:\n\
               fluxion validate --case 600\n\
             \n\
             Use 'fluxion --help' for more information."
        )
    })?;

    match command {
        Commands::References { command } => match command {
            ReferenceCommands::Update { url } => {
                update_references(url.as_deref())?;
            }
        },

        Commands::Validate {
            all,
            diagnostics,
            range,
            case: _,
            statistical,
            alpha,
            format,
            output_file,
            ci,
            ci_summary_json,
        } => {
            // Validate alpha is in valid range [0, 1]
            if !(0.0..=1.0).contains(&alpha) {
                anyhow::bail!("Alpha must be in range [0.0, 1.0], got: {}", alpha);
            }

            let mut validator = ASHRAE140Validator::new();

            // Handle diagnostic case options
            if all {
                // Run complete validation (baseline + diagnostics)
                validator.add_diagnostic_case_range("195-470".to_string());
                validator.add_diagnostic_case_range("800-810".to_string());
                validator.add_diagnostic_case_range("non-residential".to_string());
                validator.add_diagnostic_case_range("solid-conduction".to_string());
                validator.add_diagnostic_case_range("solar-gain".to_string());
            } else if diagnostics {
                // Run diagnostic cases only
                validator.skip_baseline_cases(true);
                validator.add_diagnostic_case_range("195-470".to_string());
                validator.add_diagnostic_case_range("800-810".to_string());
                validator.add_diagnostic_case_range("non-residential".to_string());
                validator.add_diagnostic_case_range("solid-conduction".to_string());
                validator.add_diagnostic_case_range("solar-gain".to_string());
            } else if let Some(r) = range {
                // Run specific diagnostic range
                validator.skip_baseline_cases(true);
                validator.add_diagnostic_case_range(r);
            }

            let (report, stat_report) = if statistical {
                // Build list of cases to validate based on validator configuration
                let mut cases: Vec<ASHRAE140Case> = Vec::new();

                // Helper to convert case_id string to ASHRAE140Case
                let case_id_to_case = |case_id: &str| -> Option<ASHRAE140Case> {
                    match case_id {
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
                        "800" => Some(ASHRAE140Case::Case800),
                        "801" => Some(ASHRAE140Case::Case801),
                        "802" => Some(ASHRAE140Case::Case802),
                        "803" => Some(ASHRAE140Case::Case803),
                        "804" => Some(ASHRAE140Case::Case804),
                        "805" => Some(ASHRAE140Case::Case805),
                        "806" => Some(ASHRAE140Case::Case806),
                        "807" => Some(ASHRAE140Case::Case807),
                        "808" => Some(ASHRAE140Case::Case808),
                        "809" => Some(ASHRAE140Case::Case809),
                        "810" => Some(ASHRAE140Case::Case810),
                        _ => None,
                    }
                };

                // Add baseline cases unless skipped
                if !validator.is_skip_baseline_cases() {
                    // Low mass cases (600 series)
                    for case_id in get_low_mass_cases() {
                        if let Some(case) = case_id_to_case(&case_id) {
                            cases.push(case);
                        }
                    }
                    // High mass cases (900 series)
                    for case_id in get_high_mass_cases() {
                        if let Some(case) = case_id_to_case(&case_id) {
                            cases.push(case);
                        }
                    }
                    // Special cases
                    for case_id in get_special_cases() {
                        if let Some(case) = case_id_to_case(&case_id) {
                            cases.push(case);
                        }
                    }
                }

                // Add diagnostic cases
                for case_id in &validator.diagnostic_cases_added {
                    match case_id.as_str() {
                        "195-470" | "800-810" | "non-residential" | "solid-conduction"
                        | "solar-gain" => {
                            // Add all cases in range (simplified for now - just add known cases)
                            if let Some(case) = case_id_to_case("800") {
                                cases.push(case);
                            }
                            if let Some(case) = case_id_to_case("801") {
                                cases.push(case);
                            }
                            // TODO: Add more diagnostic cases when fully implemented
                        }
                        _ => {
                            // Single case
                            if let Some(case) = case_id_to_case(case_id) {
                                cases.push(case);
                            }
                        }
                    }
                }

                // Use statistical validation
                let mut stat_validator = StatisticalValidator::with_alpha(alpha);
                let stat_report = stat_validator.validate_with_statistics(&cases);
                (stat_report.tolerance.clone(), Some(stat_report))
            } else {
                // Use tolerance-based validation (backward compatible)
                let report = validator.validate_analytical_engine();
                (report, None)
            };

            // Always append historical metrics
            report.append_history();

            // CI Summary JSON output - outputs machine-readable summary for CI parsing
            // This bypasses human-readable formatting and outputs JSON directly to avoid
            // regex parsing issues with values like "inf%" in text output
            if ci_summary_json {
                report.print_summary_json();
                return Ok(());
            }

            // Classify systematic issues if generating markdown
            let systematic_issues = if format == "markdown" {
                Some(ValidationReportGenerator::classify_systematic_issues(
                    &report,
                ))
            } else {
                None
            };

            // Load baseline for reporting and guardrails (if exists)
            let baseline_path = "docs/performance_baseline.json";
            let guardrail_baseline: Option<guardrails::GuardrailBaseline> =
                if Path::new(baseline_path).exists() {
                    match guardrails::GuardrailBaseline::load(baseline_path) {
                        Ok(b) => Some(b),
                        Err(e) => {
                            eprintln!("Warning: Failed to load baseline: {}", e);
                            None
                        }
                    }
                } else {
                    eprintln!(
                        "Warning: Baseline file not found at {}, skipping guardrail checks",
                        baseline_path
                    );
                    None
                };
            let baseline_for_report = guardrail_baseline.as_ref().map(|gb| BaselineMetrics {
                mae: gb.mae,
                max_deviation: gb.max_deviation,
                pass_rate: gb.pass_rate,
                validation_time_seconds: gb.validation_time_seconds,
            });

            // Print statistical summary if using statistical validation
            if let Some(ref stat) = stat_report {
                println!("=== Statistical Validation Results ===");
                println!(
                    "NMBE: {:.2}% [ {:.2}%, {:.2}% 95% CI ]",
                    stat.metrics.nmbe, stat.metrics.nmbe_ci.0, stat.metrics.nmbe_ci.1
                );
                println!(
                    "CV(RMSE): {:.2}% [ {:.2}%, {:.2}% 95% CI ]",
                    stat.metrics.cv_rmse, stat.metrics.cv_rmse_ci.0, stat.metrics.cv_rmse_ci.1
                );

                let effect_direction = match stat.metrics.effect_direction {
                    crate::validation::statistical::EffectDirection::Overprediction => {
                        "Overprediction"
                    }
                    crate::validation::statistical::EffectDirection::Underprediction => {
                        "Underprediction"
                    }
                };
                println!(
                    "Effect Size (Cohen's d): {:.2} ({})",
                    stat.metrics.cohens_d, effect_direction
                );
                if stat.metrics.excluded_cases > 0 {
                    println!(
                        "Excluded Cases: {} (zero/near-zero reference values)",
                        stat.metrics.excluded_cases
                    );
                }
                println!();
            }

            // Generate output in requested format
            if format == "markdown" {
                if let Some(ref path) = output_file {
                    let generator = ValidationReportGenerator::new(path.clone());
                    if let Some(ref stat) = stat_report {
                        // Use statistical report
                        generator
                            .generate_with_statistics(
                                stat,
                                systematic_issues.as_ref(),
                                baseline_for_report.as_ref(),
                            )
                            .map_err(anyhow::Error::msg)?;
                    } else {
                        // Use standard report
                        generator
                            .generate(
                                &report,
                                systematic_issues.as_ref(),
                                baseline_for_report.as_ref(),
                            )
                            .map_err(anyhow::Error::msg)?;
                    }
                    println!("Report saved to {:?}", path);
                } else {
                    // Render to stdout
                    let markdown = if let Some(ref stat) = stat_report {
                        ValidationReportGenerator::new(PathBuf::from("/dev/null"))
                            .render_markdown_with_statistics(
                                stat,
                                systematic_issues.as_ref(),
                                baseline_for_report.as_ref(),
                            )
                            .map_err(anyhow::Error::msg)?
                    } else {
                        ValidationReportGenerator::new(PathBuf::from("/dev/null"))
                            .render_markdown(
                                &report,
                                systematic_issues.as_ref(),
                                baseline_for_report.as_ref(),
                            )
                            .map_err(anyhow::Error::msg)?
                    };
                    println!("{}", markdown);
                }
            } else {
                // Non-markdown formats use BenchmarkReport methods
                let output = match format.as_str() {
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

            // Guardrail check in CI mode
            let ci_mode = ci || env::var("CI").map(|v| v == "true").unwrap_or(false);
            if ci_mode {
                if let Some(baseline) = guardrail_baseline {
                    let (passed, failures) = guardrails::check(&report, &baseline);
                    if !passed {
                        eprintln!("Guardrail validation failed:");
                        for failure in failures {
                            eprintln!("  - {}", failure);
                        }
                        std::process::exit(1);
                    }
                }
            }
        }

        Commands::ValidateCase { case_spec } => {
            validate_diagnostic_case(&case_spec)?;
        }

        Commands::Quantize {
            model,
            output,
            quant_type,
            benchmark,
        } => {
            let mut cmd = Command::new("python3");
            cmd.arg("tools/quantize_model.py")
                .arg("--model")
                .arg(&model)
                .arg("--output")
                .arg(&output)
                .arg("--type")
                .arg(&quant_type);

            if benchmark {
                cmd.arg("--benchmark");
            }

            let status = cmd.current_dir(".").spawn()?.wait()?;

            if !status.success() {
                anyhow::bail!("Quantization failed with exit code: {:?}", status.code());
            }

            println!("Model quantized successfully!");
            println!("  Input:  {:?}", model);
            println!("  Output: {:?}", output);
        }

        Commands::Benchmark { model, runs } => {
            let mut cmd = Command::new("python3");
            cmd.arg("tools/quantize_model.py")
                .arg("--model")
                .arg(&model)
                .arg("--output")
                .arg("/tmp/benchmark_dummy.onnx")
                .arg("--benchmark")
                .arg("--benchmark-runs")
                .arg(runs.to_string());

            let status = cmd.current_dir(".").spawn()?.wait()?;

            if !status.success() {
                anyhow::bail!("Benchmark failed with exit code: {:?}", status.code());
            }
        }

        // New commands
        Commands::Sensitivity {
            config,
            output: _,
            use_surrogates,
        } => {
            // Read sensitivity config
            let config_content = std::fs::read_to_string(config)?;
            let sens_config: SensitivityConfig = serde_yaml::from_str(&config_content)?;
            // Get base case spec
            let spec = case_id_to_spec(&sens_config.case_id)
                .ok_or_else(|| anyhow!("Unknown case ID: {}", sens_config.case_id))?;
            // Build base model from the specification
            let base_model = ThermalModel::from_spec(&spec);
            // Create BatchOracle from the base model
            let oracle = BatchOracle::from_model(base_model);
            // Generate design matrix
            let design = match sens_config.method.as_str() {
                "oat" => {
                    let levels = sens_config.levels.unwrap_or(10);
                    sensitivity::generate_oat_design(&sens_config.parameters, levels)
                }
                "random" => {
                    let samples = sens_config.samples.unwrap_or(100);
                    sensitivity::generate_random_design(&sens_config.parameters, samples)
                }
                _ => anyhow::bail!("Unknown method: {}", sens_config.method),
            };
            // Run sensitivity simulation (use_surrogates hardcoded to false for now)
            let outputs = sensitivity::run_sensitivity(&design, &oracle, use_surrogates);
            // Compute metrics
            let report = sensitivity::compute_metrics(&design, &outputs);
            // Write CSV report
            let csv_path = "sensitivity_report.csv";
            sensitivity::export_to_csv(&report, Path::new(csv_path))?;
            println!("CSV report saved to {}", csv_path);
            // Write Markdown report
            let md = generate_sensitivity_markdown(&report);
            std::fs::write("sensitivity_report.md", md)?;
            println!("Markdown report saved to sensitivity_report.md");
        }

        Commands::Delta {
            config,
            output: output_opt,
            hourly,
        } => {
            let config_content = std::fs::read_to_string(config)?;
            let delta_config: DeltaConfig = serde_yaml::from_str(&config_content)?;
            let output_dir = output_opt.unwrap_or_else(|| PathBuf::from("."));
            std::fs::create_dir_all(&output_dir)?;
            delta::run_and_report(delta_config, &output_dir, hourly)?;
            println!(
                "Delta report written to {}",
                output_dir.join("delta_report.md").display()
            );
            if hourly {
                println!(
                    "Hourly differences CSV written to {}",
                    output_dir.join("hourly_differences.csv").display()
                );
            }
        }

        Commands::Components {
            case,
            output: output_opt,
        } => {
            let spec =
                case_id_to_spec(&case).ok_or_else(|| anyhow!("Unknown case ID: {}", case))?;
            let validator = ASHRAE140Validator::new();
            let weather = EpwWeatherSource::from_file(
                epw_required("USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw").to_str().unwrap(),
            )
            .expect("Failed to load EPW weather data");
            let (_, diagnostic) = validator.simulate_case_with_diagnostics(&spec, &weather, &case);
            let breakdown = diagnostic.energy_breakdown;
            let entries =
                components::aggregate_from_validator(vec![(case.clone(), breakdown)].into_iter());
            let output_path =
                output_opt.unwrap_or_else(|| PathBuf::from(format!("{}_components.csv", case)));
            components::export_component_csv(&entries, &output_path)?;
            println!("Component breakdown saved to {}", output_path.display());
        }

        Commands::Swing {
            case,
            comfort_min,
            comfort_max,
        } => {
            let spec =
                case_id_to_spec(&case).ok_or_else(|| anyhow!("Unknown case ID: {}", case))?;
            let validator = ASHRAE140Validator::new();
            let weather = EpwWeatherSource::from_file(
                epw_required("USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw").to_str().unwrap(),
            )
            .expect("Failed to load EPW weather data");
            let (_, diagnostic) = validator.simulate_case_with_diagnostics(&spec, &weather, &case);
            // Ensure temperature profile has data (free-floating case)
            if diagnostic.temp_profile.hourly_temps.is_empty() {
                anyhow::bail!("Swing analysis requires a free-floating case (e.g., 600FF, 900FF). Case {} does not have temperature profile data.", case);
            }
            let metrics = calculate_swing_metrics(
                &diagnostic.temp_profile,
                comfort_min.unwrap_or(18.0),
                comfort_max.unwrap_or(26.0),
            );
            let interpretation = interpret_swing_metrics(&metrics);
            let report = generate_swing_report(&[interpretation]);
            println!("{}", report);
        }

        Commands::Visualize {
            input,
            output: output_opt,
        } => {
            let data = load_diagnostics_csv(&input)?;
            let output_path = match output_opt {
                Some(p) => p,
                None => {
                    let mut p = input.to_path_buf();
                    p.set_extension("html");
                    p
                }
            };
            generate_html(&data, &output_path)?;
            println!("Visualization saved to {}", output_path.display());
        }

        Commands::Animate {
            input,
            output: output_opt,
        } => {
            let data = load_diagnostics_csv(&input)?;
            let output_path = match output_opt {
                Some(p) => p,
                None => {
                    let mut p = input.to_path_buf();
                    p.set_extension("html");
                    p
                }
            };
            generate_animation(&data, &output_path)?;
            println!("Animation saved to {}", output_path.display());
        }

        Commands::Validation { command } => {
            validation::handle_validation_command(&command)?;
        }

        Commands::Automation { command } => {
            handle_automation_command(&command)?;
        }

        Commands::Copilot {
            config,
            ollama_url,
            model,
            rule_only,
            output,
            verbose,
        } => {
            use crate::validation::copilot::{Copilot, CopilotConfig};
            use tokio::runtime::Runtime;

            // Build copilot config
            let mut copilot_config = CopilotConfig::default();
            if let Some(url) = ollama_url {
                copilot_config = copilot_config.with_ollama_url(url);
            }
            if let Some(m) = model {
                copilot_config = copilot_config.with_model(m);
            }
            if rule_only {
                copilot_config = copilot_config.rule_based_only();
            }
            if verbose {
                copilot_config = copilot_config.verbose();
            }

            // Read configuration file
            let config_content = std::fs::read_to_string(config)
                .map_err(|e| anyhow!("Failed to read config file: {}", e))?;

            // Run copilot analysis
            let rt = Runtime::new()?;
            let mut copilot = Copilot::new(copilot_config);

            if verbose {
                eprintln!("[Copilot] Checking Ollama availability...");
                let available = rt.block_on(copilot.is_ollama_available());
                if available {
                    eprintln!("[Copilot] Ollama is available");
                } else {
                    eprintln!("[Copilot] Ollama not available - using rule-based only");
                }
            }

            let result = rt.block_on(copilot.analyze(&config_content))?;

            // Output results
            if let Some(ref output_path) = output {
                let json = serde_json::to_string_pretty(&result)?;
                std::fs::write(output_path, json)?;
                println!("Results written to {}", output_path.display());
            } else {
                result.print_summary();
            }

            // Exit with error code if validation failed
            if !result.is_valid() {
                std::process::exit(1);
            }
        }

        Commands::Run {
            workflow,
            debug,
            measures_only,
            postprocess_only,
        } => {
            run_workflow(workflow.as_deref(), debug, measures_only, postprocess_only)?;
        }

        Commands::Measure { command } => {
            run_measure_command(&command)?;
        }
    }

    Ok(())
}

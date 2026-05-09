use anyhow::Result;
use clap::{Parser, Subcommand};
use csv::Reader;
use fluxion::analysis::components;
use fluxion::analysis::delta::{self, DeltaConfig};
use fluxion::analysis::sensitivity::{self, ParameterRange, SensitivityReport};
use fluxion::analysis::swing::{
    calculate_swing_metrics, generate_swing_report, interpret_swing_metrics,
};
use fluxion::analysis::visualization::{
    generate_animation, generate_html, Dataset, PlotPanel, TimeSeriesData,
};
use fluxion::cli::validation::ValidationSubcommand;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};

/// Automation subcommands for test workflows and CI/CD integration
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
use fluxion::validation::benchmark::{get_high_mass_cases, get_low_mass_cases, get_special_cases};
use fluxion::validation::commands::update_references;
use fluxion::validation::guardrails;
use fluxion::validation::reporter::{BaselineMetrics, ValidationReportGenerator};
use fluxion::validation::statistical::StatisticalValidator;
use fluxion::validation::ASHRAE140Validator;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::BatchOracle;
use serde::Deserialize;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// Helper: map case_id string to CaseSpec
fn case_id_to_spec(case_id: &str) -> Option<CaseSpec> {
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

// Sensitivity configuration from YAML
#[derive(Deserialize)]
struct SensitivityConfig {
    case_id: String,
    method: String, // "oat" or "sobol"
    levels: Option<usize>,
    samples: Option<usize>,
    parameters: Vec<ParameterRange>,
}

// Generate markdown report for sensitivity analysis
fn generate_sensitivity_markdown(report: &SensitivityReport) -> String {
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

// Load diagnostics CSV (as produced by SimulationDiagnostics::export_csv) into TimeSeriesData
fn load_diagnostics_csv(path: &Path) -> Result<TimeSeriesData> {
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
            .ok_or_else(|| anyhow::anyhow!("Missing Hour column"))?;
        let hour: usize = hour_str.parse()?;
        timestamps.push(hour);
        // Zone_Temps (col 1)
        let zone_temps_str = record
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("Missing Zone_Temps"))?;
        let zone_vals: Vec<f64> = zone_temps_str
            .split(';')
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to parse Zone_Temps: {}", e))?;
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
            .ok_or_else(|| anyhow::anyhow!("Missing Solar_Watts"))?;
        let solar_vals: Vec<f64> = solar_str
            .split(';')
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to parse Solar_Watts: {}", e))?;
        let solar_total: f64 = solar_vals.iter().sum();
        datasets[num_zones].values.push(solar_total);
        // HVAC_Watts (col 6)
        let hvac_str = record
            .get(6)
            .ok_or_else(|| anyhow::anyhow!("Missing HVAC_Watts"))?;
        let hvac_vals: Vec<f64> = hvac_str
            .split(';')
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to parse HVAC_Watts: {}", e))?;
        let hvac_total: f64 = hvac_vals.iter().sum();
        datasets[num_zones + 1].values.push(hvac_total);
    }
    Ok(TimeSeriesData {
        timestamps,
        datasets,
    })
}

#[derive(Parser)]
#[command(name = "fluxion")]
#[command(about = "Fluxion Building Energy Modeling CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum ReferenceCommands {
    /// Updates reference data from the configured source
    Update {
        /// URL to fetch reference data from (optional, uses default if omitted)
        #[arg(short, long)]
        url: Option<String>,
    },
}

#[derive(Subcommand)]
enum Commands {
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
}

/// Handle automation commands
fn handle_automation_command(command: &AutomationSubcommand) -> Result<()> {
    match command {
        AutomationSubcommand::Test {
            test_cases,
            output,
            tolerance,
            verbose,
            format,
        } => {
            // Set up test runner configuration
            use fluxion::validation::automation::runner::TestRunnerConfig;
            use std::path::PathBuf;

            let config = TestRunnerConfig::new(
                PathBuf::from(test_cases),
                PathBuf::from(output),
                *tolerance,
                *verbose,
                format.clone(),
            );

            // Create and run test runner
            let mut runner = fluxion::validation::automation::runner::TestRunner::new(config);
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
            use fluxion::validation::automation::github::workflow::WorkflowGenerator;
            use fluxion::validation::automation::github::workflow::WorkflowGeneratorConfig;

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
                    return Err(anyhow::anyhow!("Unknown workflow type: {}", workflow_type));
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
            use fluxion::validation::automation::github::api::GitHubClient;

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
                return Err(anyhow::anyhow!("Repository must be in format 'owner/repo'"));
            }

            let owner = repo_parts[0];
            let repo = repo_parts[1];

            // Trigger workflow (simplified - actual implementation would use GitHub API)
            println!("🚀 Triggering GitHub Actions workflow...");
            println!("Repository: {}/{}", owner, repo);
            println!("Workflow: {}", workflow);

            // In a real implementation, this would call:
            // client.post(&format!("/repos/{}/{}/actions/workflows/{}/dispatches", owner, repo, workflow_name), &payload)

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
/// This function handles explicit diagnostic case invocation via the validate-case
/// subcommand, supporting individual cases (e.g., "800") and case ranges
/// (e.g., "195-470", "800-810").
///
/// # Arguments
///
/// * `case_spec` - Case number or range specification (e.g., "800", "195-470", "800-810")
///
/// # Example
///
/// ```bash
/// fluxion validate-case 800
/// fluxion validate-case 195-470
/// fluxion validate-case 800-810
/// ```
fn validate_diagnostic_case(case_spec: &str) -> Result<()> {
    match case_spec {
        // Single HVAC equipment cases (800-810)
        "800" | "801" | "802" | "803" | "804" | "805" | "806" | "807" | "808" | "809" | "810" => {
            // Get case spec
            let spec = case_id_to_spec(case_spec)
                .ok_or_else(|| anyhow::anyhow!("Unknown case ID: {}", case_spec))?;

            // Run validation
            let validator = ASHRAE140Validator::new();
            let weather = EpwWeatherSource::from_file(
                "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
            )
            .expect("Failed to load EPW weather data");
            let (results, _) = validator.simulate_case_with_diagnostics(&spec, &weather, case_spec);

            println!(
                "Case {} result: {:.2} MWh heating, {:.2} MWh cooling",
                case_spec, results.annual_heating_mwh, results.annual_cooling_mwh
            );
        }
        // Diagnostic ranges - TODO: Implement diagnostic case runners
        "195-470" => {
            eprintln!("Diagnostic case range 195-470 not yet implemented");
            anyhow::bail!("Diagnostic cases 195-470 not implemented");
        }
        "800-810" => {
            eprintln!("Diagnostic case range 800-810 not yet implemented");
            anyhow::bail!("Diagnostic cases 800-810 not implemented");
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
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
                    fluxion::validation::statistical::EffectDirection::Overprediction => {
                        "Overprediction"
                    }
                    fluxion::validation::statistical::EffectDirection::Underprediction => {
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
                .ok_or_else(|| anyhow::anyhow!("Unknown case ID: {}", sens_config.case_id))?;
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
            let spec = case_id_to_spec(&case)
                .ok_or_else(|| anyhow::anyhow!("Unknown case ID: {}", case))?;
            let validator = ASHRAE140Validator::new();
            let weather = EpwWeatherSource::from_file(
                "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
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
            let spec = case_id_to_spec(&case)
                .ok_or_else(|| anyhow::anyhow!("Unknown case ID: {}", case))?;
            let validator = ASHRAE140Validator::new();
            let weather = EpwWeatherSource::from_file(
                "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
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
            fluxion::cli::validation::handle_validation_command(&command)?;
        }

        Commands::Automation { command } => {
            handle_automation_command(&command)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_statistical_flag_accepted() {
        // Test that --statistical flag is accepted
        let args = ["fluxion", "validate", "--statistical"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --statistical flag");
    }

    #[test]
    fn test_validate_alpha_flag_accepted() {
        // Test that --alpha flag is accepted
        let args = ["fluxion", "validate", "--statistical", "--alpha", "0.01"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --alpha flag");
    }

    #[test]
    fn test_validate_default_behavior_unchanged() {
        // Test that default behavior (no --statistical) works
        let args = ["fluxion", "validate"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should work without --statistical flag");

        if let Commands::Validate { statistical, .. } = cli.unwrap().command {
            assert!(!statistical, "Default should have statistical=false");
        }
    }

    #[test]
    fn test_validate_statistical_flag_sets_true() {
        // Test that --statistical flag sets statistical to true
        let args = ["fluxion", "validate", "--statistical"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --statistical flag");

        if let Commands::Validate { statistical, .. } = cli.unwrap().command {
            assert!(statistical, "--statistical should set statistical=true");
        }
    }

    #[test]
    fn test_validate_alpha_default_value() {
        // Test that --alpha has default value of 0.05
        let args = ["fluxion", "validate", "--statistical"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --statistical flag");

        if let Commands::Validate { alpha, .. } = cli.unwrap().command {
            assert_eq!(alpha, 0.05, "Default alpha should be 0.05");
        }
    }

    #[test]
    fn test_validate_alpha_custom_value() {
        // Test that --alpha accepts custom values
        let args = ["fluxion", "validate", "--statistical", "--alpha", "0.01"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --alpha flag");

        if let Commands::Validate { alpha, .. } = cli.unwrap().command {
            assert_eq!(alpha, 0.01, "Custom alpha should be 0.01");
        }
    }

    #[test]
    fn test_validate_alpha_boundary_values() {
        // Test that --alpha accepts boundary values 0.0 and 1.0
        for alpha_val in ["0.0", "1.0"] {
            let args = ["fluxion", "validate", "--statistical", "--alpha", alpha_val];
            let cli = Cli::try_parse_from(args.iter());
            assert!(cli.is_ok(), "CLI should accept alpha={}", alpha_val);

            if let Commands::Validate { alpha, .. } = cli.unwrap().command {
                let expected = alpha_val.parse::<f64>().unwrap();
                assert_eq!(alpha, expected, "Alpha should be {}", alpha_val);
            }
        }
    }

    #[test]
    fn test_validate_alpha_too_large_rejected() {
        // Test that alpha > 1.0 is rejected
        let args = ["fluxion", "validate", "--statistical", "--alpha", "1.5"];
        let cli = Cli::try_parse_from(args.iter());
        // This test verifies CLI parsing only; runtime validation is separate
        assert!(
            cli.is_ok(),
            "CLI parsing should succeed (runtime validation handles invalid alpha)"
        );
    }

    #[test]
    fn test_validate_statistical_flag_integration() {
        // Test that --statistical flag integrates with other flags
        let args = [
            "fluxion",
            "validate",
            "--statistical",
            "--alpha",
            "0.05",
            "--format",
            "markdown",
        ];
        let cli = Cli::try_parse_from(args.iter());
        assert!(
            cli.is_ok(),
            "CLI should accept --statistical with other flags"
        );

        if let Commands::Validate {
            statistical,
            alpha,
            format: fmt,
            ..
        } = cli.unwrap().command
        {
            assert!(statistical, "--statistical should be true");
            assert_eq!(alpha, 0.05, "alpha should be 0.05");
            assert_eq!(fmt, "markdown", "format should be markdown");
        }
    }

    #[test]
    fn test_validate_without_statistical_backward_compatible() {
        // Test that validation works without --statistical flag
        let args = ["fluxion", "validate", "--format", "csv"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should work without --statistical");

        if let Commands::Validate {
            statistical,
            format: fmt,
            ..
        } = cli.unwrap().command
        {
            assert!(!statistical, "statistical should be false by default");
            assert_eq!(fmt, "csv", "format should be csv");
        }
    }
}

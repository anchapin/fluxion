use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use csv::Reader;
use fluxion::analysis::components;
use fluxion::analysis::delta::{self, DeltaConfig};
use fluxion::analysis::sensitivity::{self, ParameterRange, SensitivityReport};
use fluxion::analysis::swing::{
    self, calculate_swing_metrics, generate_swing_report, interpret_swing_metrics,
    SwingInterpretation, SwingMetrics,
};
use fluxion::analysis::visualization::{
    self, generate_animation, generate_html, Dataset, PlotPanel, TimeSeriesData,
};
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use fluxion::validation::commands::update_references;
use fluxion::validation::diagnostic::TemperatureProfile;
use fluxion::validation::guardrails;
use fluxion::validation::reporter::{BaselineMetrics, ValidationReportGenerator};
use fluxion::validation::ASHRAE140Validator;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::BatchOracle;
use serde::Deserialize;
use serde_yaml;
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

        /// Enable CI mode (enforces guardrails and sets exit code on failure)
        #[arg(short, long)]
        ci: bool,
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
            all: _,
            case: _,
            format,
            output_file,
            ci,
        } => {
            let mut validator = ASHRAE140Validator::new();
            let report = validator.validate_analytical_engine();

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

            // Generate output in requested format
            if format == "markdown" {
                if let Some(ref path) = output_file {
                    let generator = ValidationReportGenerator::new(path.clone());
                    generator
                        .generate(
                            &report,
                            systematic_issues.as_ref(),
                            baseline_for_report.as_ref(),
                        )
                        .map_err(anyhow::Error::msg)?;
                    println!("Report saved to {:?}", path);
                } else {
                    // Render to stdout
                    let markdown = ValidationReportGenerator::new(PathBuf::from("/dev/null"))
                        .render_markdown(
                            &report,
                            systematic_issues.as_ref(),
                            baseline_for_report.as_ref(),
                        )
                        .map_err(anyhow::Error::msg)?;
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
                "sobol" => {
                    let samples = sens_config.samples.unwrap_or(100);
                    sensitivity::generate_sobol_design(&sens_config.parameters, samples)
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
            let weather = DenverTmyWeather::new();
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
            let weather = DenverTmyWeather::new();
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
    }

    Ok(())
}

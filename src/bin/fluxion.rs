use anyhow::Error;
use clap::{Parser, Subcommand};
use fluxion::validation::ASHRAE140Validator;
use fluxion::validation::guardrails;
use fluxion::validation::reporter::{BaselineMetrics, SystematicIssueMap, ValidationReportGenerator};
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

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
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
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
                Some(ValidationReportGenerator::classify_systematic_issues(&report))
            } else {
                None
            };

            // Load baseline for reporting and guardrails (if exists)
            let baseline_path = "docs/performance_baseline.json";
            let guardrail_baseline: Option<guardrails::GuardrailBaseline> = if Path::new(baseline_path).exists() {
                match guardrails::GuardrailBaseline::load(baseline_path) {
                    Ok(b) => Some(b),
                    Err(e) => {
                        eprintln!("Warning: Failed to load baseline: {}", e);
                        None
                    }
                }
            } else {
                eprintln!("Warning: Baseline file not found at {}, skipping guardrail checks", baseline_path);
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
                        .generate(&report, systematic_issues.as_ref(), baseline_for_report.as_ref())
                        .map_err(Error::msg)?;
                    println!("Report saved to {:?}", path);
                } else {
                    // Render to stdout
                    let markdown = ValidationReportGenerator::new(PathBuf::from("/dev/null"))
                        .render_markdown(&report, systematic_issues.as_ref(), baseline_for_report.as_ref())
                        .map_err(Error::msg)?;
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
            // Call Python quantization script
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
            // Call Python benchmark
            let mut cmd = Command::new("python3");
            cmd.arg("tools/quantize_model.py")
                .arg("--model")
                .arg(&model)
                .arg("--output")
                .arg("/tmp/benchmark_dummy.onnx") // Dummy output
                .arg("--benchmark")
                .arg("--benchmark-runs")
                .arg(runs.to_string());

            let status = cmd.current_dir(".").spawn()?.wait()?;

            if !status.success() {
                anyhow::bail!("Benchmark failed with exit code: {:?}", status.code());
            }
        }
    }

    Ok(())
}

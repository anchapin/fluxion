use clap::{Parser, Subcommand};
use fluxion::validation::ASHRAE140Validator;
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
        } => {
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

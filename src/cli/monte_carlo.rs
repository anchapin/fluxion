// CLI commands for Monte Carlo parameter sweeps via declarative deltas (Issue #1813).
//
// Implements the OSimFlow worker entrypoint that accepts a base model file and a
// delta file, samples parameter distributions, and runs an annual simulation per
// draw across rayon threads — all in-process, without invoking Python.

use crate::analysis::monte_carlo::{
    self, run_sweep, summarize, MonteCarloDelta, MonteCarloResult, SweepStatistics,
};
use crate::validation::ashrae_140_cases::CaseSpec;
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Monte Carlo sweep subcommands.
#[derive(Debug, Subcommand)]
pub enum MonteCarloCommand {
    /// Run a Monte Carlo sweep: load base model + delta file, sample, simulate.
    #[command(name = "sweep")]
    Sweep(SweepArgs),

    /// Dry-run: sample the delta file and emit the draws without simulating.
    /// Useful for verifying distributions and seeds before a long sweep.
    #[command(name = "dry-run")]
    DryRun(DryRunArgs),
}

/// Arguments for `fluxion monte-carlo sweep`.
///
/// Matches the OSimFlow worker contract from Issue #1813: a base model file plus
/// a delta file. The worker applies the delta in-memory via the Rust
/// `apply_sample` API and executes the simulation without invoking Python.
#[derive(Debug, Args)]
pub struct SweepArgs {
    /// Path to the base model file (serialized `CaseSpec` in YAML or JSON).
    #[arg(long = "base-model")]
    pub base_model: PathBuf,

    /// Path to the delta file specifying parameter distributions (YAML or JSON).
    #[arg(long = "delta-file")]
    pub delta_file: PathBuf,

    /// Output directory for per-draw results and the summary.
    #[arg(short, long, default_value = "./monte_carlo_results")]
    pub output: PathBuf,

    /// Override the number of samples from the command line.
    /// When set, takes precedence over the delta file's `samples` field.
    #[arg(short = 'n', long)]
    pub samples: Option<usize>,

    /// Override the RNG seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Collect and persist hourly diagnostics per draw (large output).
    #[arg(long)]
    pub hourly: bool,

    /// Also emit a per-draw JSON result file (delta_000000.json, ...).
    /// The combined results.jsonl is always written.
    #[arg(long)]
    pub per_draw_files: bool,

    /// Run sequentially on a single thread (useful for profiling / debugging
    /// the worker startup-time benchmark from #1813 acceptance criterion 5).
    #[arg(long)]
    pub sequential: bool,
}

/// Arguments for `fluxion monte-carlo dry-run`.
#[derive(Debug, Args)]
pub struct DryRunArgs {
    /// Path to the delta file specifying parameter distributions.
    #[arg(long = "delta-file")]
    pub delta_file: PathBuf,

    /// Override the number of samples.
    #[arg(short = 'n', long)]
    pub samples: Option<usize>,

    /// Override the RNG seed.
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Handle a `fluxion monte-carlo` subcommand.
pub fn handle_monte_carlo_command(command: &MonteCarloCommand) -> Result<()> {
    match command {
        MonteCarloCommand::Sweep(args) => handle_sweep(args),
        MonteCarloCommand::DryRun(args) => handle_dry_run(args),
    }
}

/// Combined sweep output written to `<output>/summary.json`.
#[derive(Debug, Serialize)]
struct SweepSummary {
    base_model: PathBuf,
    delta_file: PathBuf,
    samples_requested: usize,
    #[serde(flatten)]
    stats: SweepStatistics,
}

/// Per-draw result written to `<output>/results.jsonl` (one JSON object per line).
#[derive(Debug, Serialize)]
struct ResultLine {
    index: usize,
    inputs: std::collections::HashMap<String, f64>,
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl From<&MonteCarloResult> for ResultLine {
    fn from(r: &MonteCarloResult) -> Self {
        ResultLine {
            index: r.index,
            inputs: r.inputs.clone(),
            annual_heating_mwh: r.annual_heating_mwh,
            annual_cooling_mwh: r.annual_cooling_mwh,
            peak_heating_kw: r.peak_heating_kw,
            peak_cooling_kw: r.peak_cooling_kw,
            error: r.error.clone(),
        }
    }
}

fn handle_sweep(args: &SweepArgs) -> Result<()> {
    let base = load_case_spec(&args.base_model)?;
    let mut delta = MonteCarloDelta::from_file(&args.delta_file)?;
    apply_overrides(&mut delta, args.samples, args.seed);
    delta.validate()?;

    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("failed to create output dir {}", args.output.display()))?;

    let started = std::time::Instant::now();
    let results = if args.sequential {
        run_sweep_sequential(&base, &delta, args.hourly)?
    } else {
        run_sweep(&base, &delta, args.hourly)?
    };
    let elapsed = started.elapsed();

    // Write combined results.jsonl
    let results_path = args.output.join("results.jsonl");
    let combined = serialize_results_jsonl(&results);
    std::fs::write(&results_path, &combined)
        .with_context(|| format!("failed to write {}", results_path.display()))?;

    // Optional per-draw files (acceptance criterion: 10 deltas → 10 result files).
    if args.per_draw_files {
        write_per_draw_files(&args.output, &results)?;
    }

    // Write summary.json with statistics + benchmark timing.
    let stats = summarize(&results);
    let summary = SweepSummary {
        base_model: args.base_model.clone(),
        delta_file: args.delta_file.clone(),
        samples_requested: delta.samples,
        stats,
    };
    let mut summary_map = serde_json::to_value(&summary)?;
    if let Some(obj) = summary_map.as_object_mut() {
        obj.insert(
            "wall_seconds".to_string(),
            serde_json::json!(elapsed.as_secs_f64()),
        );
        obj.insert(
            "per_draw_ms".to_string(),
            serde_json::json!(if delta.samples > 0 {
                elapsed.as_secs_f64() * 1000.0 / delta.samples as f64
            } else {
                0.0
            }),
        );
        obj.insert(
            "parallelism".to_string(),
            serde_json::json!(if args.sequential {
                "sequential"
            } else {
                "rayon"
            }),
        );
    }
    let summary_path = args.output.join("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary_map)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;

    println!(
        "Monte Carlo sweep complete: {} draws ({} failed) in {:.2}s → {}",
        delta.samples,
        summary.stats.failures,
        elapsed.as_secs_f64(),
        args.output.display()
    );
    Ok(())
}

fn handle_dry_run(args: &DryRunArgs) -> Result<()> {
    let mut delta = MonteCarloDelta::from_file(&args.delta_file)?;
    apply_overrides(&mut delta, args.samples, args.seed);
    delta.validate()?;
    let samples = monte_carlo::sample_parameters(&delta)?;
    for s in samples {
        println!("{}", serde_json::to_string(&s)?);
    }
    Ok(())
}

fn run_sweep_sequential(
    base: &CaseSpec,
    delta: &MonteCarloDelta,
    collect_hourly: bool,
) -> Result<Vec<MonteCarloResult>> {
    use crate::analysis::delta::run_simulation;
    use crate::analysis::monte_carlo::{apply_sample, sample_parameters};
    delta.validate()?;
    let samples = sample_parameters(delta)?;
    let warm_up = delta.warm_up_years;
    let mut results = Vec::with_capacity(samples.len());
    for sample in samples {
        match apply_sample(base, &sample) {
            Ok(spec) => match run_simulation(&spec, collect_hourly, warm_up) {
                Ok(sim) => results.push(MonteCarloResult::from_sim(
                    sample.index,
                    sample.values.clone(),
                    sim,
                )),
                Err(e) => results.push(MonteCarloResult::from_err(
                    sample.index,
                    sample.values.clone(),
                    e,
                )),
            },
            Err(e) => results.push(MonteCarloResult::from_err(
                sample.index,
                sample.values.clone(),
                e,
            )),
        }
    }
    Ok(results)
}

fn apply_overrides(delta: &mut MonteCarloDelta, samples: Option<usize>, seed: Option<u64>) {
    if let Some(n) = samples {
        delta.samples = n;
    }
    if let Some(s) = seed {
        delta.seed = s;
    }
}

fn serialize_results_jsonl(results: &[MonteCarloResult]) -> String {
    let mut out = String::new();
    for r in results {
        let line: ResultLine = r.into();
        if let Ok(s) = serde_json::to_string(&line) {
            out.push_str(&s);
            out.push('\n');
        }
    }
    out
}

fn write_per_draw_files(output: &Path, results: &[MonteCarloResult]) -> Result<()> {
    for r in results {
        let name = format!("delta_{:06}.json", r.index);
        let path = output.join(name);
        let line: ResultLine = r.into();
        std::fs::write(&path, serde_json::to_string_pretty(&line)?)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

/// Load a `CaseSpec` from a YAML or JSON file (extension selects the format).
pub fn load_case_spec(path: &Path) -> Result<CaseSpec> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read base model {}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "json" {
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse JSON base model {}", path.display()))
    } else {
        serde_yaml::from_str(&text)
            .with_context(|| format!("failed to parse YAML base model {}", path.display()))
    }
}

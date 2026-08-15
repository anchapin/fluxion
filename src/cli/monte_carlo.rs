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

#[cfg(test)]
mod tests {
    //! Inline unit tests for the Monte Carlo CLI worker (Issue #2897).
    //!
    //! Coverage split:
    //! * clap argument wiring for `sweep` / `dry-run` (required args, defaults,
    //!   type rejection).
    //! * `apply_overrides` precedence (CLI beats delta file).
    //! * `run_sweep_sequential` determinism for a fixed seed.
    //! * `serialize_results_jsonl` / `write_per_draw_files` output contracts.
    //! * `load_case_spec` format selection and error paths.

    use super::*;
    use crate::analysis::monte_carlo::Distribution;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;
    use clap::Parser;
    use std::collections::HashMap;

    /// Test-only wrapper so `MonteCarloCommand` is exercised through real clap.
    #[derive(Debug, Parser)]
    #[command(name = "fluxion-monte-carlo-test")]
    struct TestCli {
        #[command(subcommand)]
        cmd: MonteCarloCommand,
    }

    fn parse(args: &[&str]) -> MonteCarloCommand {
        TestCli::try_parse_from(args)
            .expect("args should parse")
            .cmd
    }

    /// A two-draw, zero-warm-up delta with only `Fixed` distributions so the
    /// sequential sweep stays cheap while still exercising the full
    /// sample → patch → simulate → collect pipeline.
    fn tiny_delta(seed: u64) -> MonteCarloDelta {
        let mut parameters = HashMap::new();
        parameters.insert(
            "infiltration_ach".to_string(),
            Distribution::Uniform { min: 0.4, max: 0.6 },
        );
        MonteCarloDelta {
            samples: 2,
            seed,
            warm_up_years: 0,
            parameters,
        }
    }

    fn result(index: usize, heating: f64, error: Option<&str>) -> MonteCarloResult {
        let mut inputs = HashMap::new();
        inputs.insert("infiltration_ach".to_string(), 0.5);
        MonteCarloResult {
            index,
            inputs,
            annual_heating_mwh: heating,
            annual_cooling_mwh: heating / 2.0,
            peak_heating_kw: 3.0,
            peak_cooling_kw: 1.5,
            error: error.map(|e| e.to_string()),
        }
    }

    // ---------------------------------------------------------------
    // Argument validation — sweep
    // ---------------------------------------------------------------

    #[test]
    fn sweep_requires_base_model_and_delta_file() {
        assert!(
            TestCli::try_parse_from(["mc", "sweep"]).is_err(),
            "sweep requires --base-model and --delta-file"
        );
        assert!(
            TestCli::try_parse_from(["mc", "sweep", "--base-model", "base.yaml"]).is_err(),
            "sweep requires --delta-file"
        );
        assert!(
            TestCli::try_parse_from(["mc", "sweep", "--delta-file", "d.yaml"]).is_err(),
            "sweep requires --base-model"
        );
    }

    #[test]
    fn sweep_defaults_output_and_leaves_overrides_unset() {
        match parse(&[
            "mc",
            "sweep",
            "--base-model",
            "base.yaml",
            "--delta-file",
            "delta.yaml",
        ]) {
            MonteCarloCommand::Sweep(args) => {
                assert_eq!(args.base_model, PathBuf::from("base.yaml"));
                assert_eq!(args.delta_file, PathBuf::from("delta.yaml"));
                assert_eq!(args.output, PathBuf::from("./monte_carlo_results"));
                assert_eq!(args.samples, None, "no -n means the delta file decides");
                assert_eq!(args.seed, None, "no --seed means the delta file decides");
                assert!(!args.hourly);
                assert!(!args.per_draw_files);
                assert!(!args.sequential);
            }
            other => panic!("expected Sweep, got {other:?}"),
        }
    }

    #[test]
    fn sweep_parses_all_flags_including_short_sample_count() {
        match parse(&[
            "mc",
            "sweep",
            "--base-model",
            "b.json",
            "--delta-file",
            "d.json",
            "-o",
            "/tmp/mc",
            "-n",
            "16",
            "--seed",
            "1813",
            "--hourly",
            "--per-draw-files",
            "--sequential",
        ]) {
            MonteCarloCommand::Sweep(args) => {
                assert_eq!(args.output, PathBuf::from("/tmp/mc"));
                assert_eq!(args.samples, Some(16));
                assert_eq!(args.seed, Some(1813));
                assert!(args.hourly);
                assert!(args.per_draw_files);
                assert!(args.sequential);
            }
            other => panic!("expected Sweep, got {other:?}"),
        }
    }

    #[test]
    fn sweep_rejects_non_numeric_samples_and_seed() {
        let base = [
            "mc",
            "sweep",
            "--base-model",
            "b.yaml",
            "--delta-file",
            "d.yaml",
        ];
        let mut bad_samples = base.to_vec();
        bad_samples.extend(["-n", "many"]);
        assert!(
            TestCli::try_parse_from(bad_samples).is_err(),
            "-n must be a usize"
        );

        let mut bad_seed = base.to_vec();
        bad_seed.extend(["--seed", "-1"]);
        assert!(
            TestCli::try_parse_from(bad_seed).is_err(),
            "--seed must be a u64"
        );
    }

    #[test]
    fn dry_run_requires_delta_file_only() {
        match parse(&["mc", "dry-run", "--delta-file", "d.yaml"]) {
            MonteCarloCommand::DryRun(args) => {
                assert_eq!(args.delta_file, PathBuf::from("d.yaml"));
                assert_eq!(args.samples, None);
                assert_eq!(args.seed, None);
            }
            other => panic!("expected DryRun, got {other:?}"),
        }
        assert!(
            TestCli::try_parse_from(["mc", "dry-run"]).is_err(),
            "dry-run requires --delta-file"
        );
        // dry-run must NOT accept --base-model (it never simulates).
        assert!(
            TestCli::try_parse_from([
                "mc",
                "dry-run",
                "--delta-file",
                "d.yaml",
                "--base-model",
                "b.yaml"
            ])
            .is_err(),
            "dry-run has no --base-model"
        );
    }

    #[test]
    fn unknown_monte_carlo_subcommand_is_rejected() {
        assert!(TestCli::try_parse_from(["mc", "sweeeep"]).is_err());
        assert!(
            TestCli::try_parse_from(["mc"]).is_err(),
            "a subcommand is required"
        );
    }

    // ---------------------------------------------------------------
    // apply_overrides
    // ---------------------------------------------------------------

    #[test]
    fn apply_overrides_none_leaves_delta_untouched() {
        let mut delta = tiny_delta(7);
        let (samples, seed) = (delta.samples, delta.seed);
        apply_overrides(&mut delta, None, None);
        assert_eq!(delta.samples, samples);
        assert_eq!(delta.seed, seed);
    }

    #[test]
    fn apply_overrides_cli_values_take_precedence_over_delta_file() {
        let mut delta = tiny_delta(7);
        apply_overrides(&mut delta, Some(64), Some(99));
        assert_eq!(delta.samples, 64, "-n must override the delta file");
        assert_eq!(delta.seed, 99, "--seed must override the delta file");

        // Partial overrides only touch the field supplied.
        let mut delta = tiny_delta(7);
        apply_overrides(&mut delta, Some(5), None);
        assert_eq!(delta.samples, 5);
        assert_eq!(delta.seed, 7);
        let mut delta = tiny_delta(7);
        apply_overrides(&mut delta, None, Some(11));
        assert_eq!(delta.samples, 2);
        assert_eq!(delta.seed, 11);
    }

    #[test]
    fn apply_overrides_zero_samples_still_fails_validation() {
        // `-n 0` is accepted by clap but must be rejected by validate() rather
        // than producing an empty sweep with a bogus summary.
        let mut delta = tiny_delta(7);
        apply_overrides(&mut delta, Some(0), None);
        assert_eq!(delta.samples, 0);
        let err = delta.validate().unwrap_err().to_string();
        assert!(err.contains("samples"), "error must mention samples: {err}");
    }

    // ---------------------------------------------------------------
    // run_sweep_sequential — determinism for a fixed seed
    // ---------------------------------------------------------------

    #[test]
    fn run_sweep_sequential_is_deterministic_for_fixed_seed() {
        let base = ASHRAE140Case::Case600.spec();
        let delta = tiny_delta(4242);

        let first = run_sweep_sequential(&base, &delta, false).expect("first sweep");
        let second = run_sweep_sequential(&base, &delta, false).expect("second sweep");

        assert_eq!(first.len(), delta.samples);
        assert_eq!(second.len(), delta.samples);
        // Guard against a vacuous pass: the sweep must actually have simulated
        // something, otherwise two all-zero result sets would compare equal.
        assert!(
            first.iter().all(|r| r.error.is_none()),
            "no draw may fail: {:?}",
            first
                .iter()
                .filter_map(|r| r.error.as_ref())
                .collect::<Vec<_>>()
        );
        assert!(
            first.iter().any(|r| r.annual_heating_mwh > 0.0),
            "sweep must produce non-zero annual heating"
        );
        // Byte-for-byte identical JSONL output is the strongest determinism
        // contract the worker can offer to OSimFlow.
        assert_eq!(
            serialize_results_jsonl(&first),
            serialize_results_jsonl(&second),
            "same seed must produce identical sequential sweep output"
        );
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.index, b.index);
            assert_eq!(a.inputs, b.inputs);
            assert_eq!(
                a.annual_heating_mwh.to_bits(),
                b.annual_heating_mwh.to_bits()
            );
        }
    }

    #[test]
    fn run_sweep_sequential_preserves_draw_order_and_finite_outputs() {
        let base = ASHRAE140Case::Case600.spec();
        let delta = tiny_delta(1);
        let results = run_sweep_sequential(&base, &delta, false).expect("sweep");
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.index, i, "sequential sweep must preserve draw order");
            assert!(r.error.is_none(), "draw {i} failed: {:?}", r.error);
            assert!(r.annual_heating_mwh.is_finite());
            assert!(r.annual_cooling_mwh.is_finite());
        }
    }

    #[test]
    fn run_sweep_sequential_rejects_invalid_delta_before_simulating() {
        let base = ASHRAE140Case::Case600.spec();
        let mut delta = tiny_delta(1);
        delta.samples = 0;
        assert!(
            run_sweep_sequential(&base, &delta, false).is_err(),
            "an invalid delta must fail fast, not run a zero-draw sweep"
        );
    }

    // ---------------------------------------------------------------
    // Output serialization
    // ---------------------------------------------------------------

    #[test]
    fn serialize_results_jsonl_emits_one_line_per_draw() {
        let results = vec![result(0, 10.0, None), result(1, 12.0, None)];
        let jsonl = serialize_results_jsonl(&results);
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(jsonl.ends_with('\n'), "JSONL must be newline-terminated");
        for (i, line) in lines.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(line).expect("each line is JSON");
            assert_eq!(v["index"].as_u64(), Some(i as u64));
            assert!(v.get("error").is_none(), "successful draws omit `error`");
        }
    }

    #[test]
    fn serialize_results_jsonl_keeps_error_field_for_failed_draws() {
        let results = vec![result(0, f64::NAN, Some("solver diverged"))];
        let jsonl = serialize_results_jsonl(&results);
        assert!(
            jsonl.contains("solver diverged"),
            "failed draws must carry their error: {jsonl}"
        );
    }

    #[test]
    fn serialize_results_jsonl_of_empty_slice_is_empty() {
        assert!(serialize_results_jsonl(&[]).is_empty());
    }

    #[test]
    fn write_per_draw_files_uses_zero_padded_six_digit_names() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results = vec![
            result(0, 1.0, None),
            result(7, 2.0, None),
            result(123456, 3.0, None),
        ];
        write_per_draw_files(tmp.path(), &results).expect("write per-draw files");
        for (index, name) in [
            (0_usize, "delta_000000.json"),
            (7, "delta_000007.json"),
            (123456, "delta_123456.json"),
        ] {
            let path = tmp.path().join(name);
            assert!(path.is_file(), "expected {name} for draw {index}");
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).expect("read"))
                    .expect("valid JSON");
            assert_eq!(v["index"].as_u64(), Some(index as u64));
        }
    }

    #[test]
    fn result_line_from_monte_carlo_result_copies_every_metric() {
        let r = result(3, 42.0, Some("boom"));
        let line: ResultLine = (&r).into();
        assert_eq!(line.index, 3);
        assert_eq!(line.inputs, r.inputs);
        assert!((line.annual_heating_mwh - 42.0).abs() < f64::EPSILON);
        assert!((line.annual_cooling_mwh - 21.0).abs() < f64::EPSILON);
        assert!((line.peak_heating_kw - 3.0).abs() < f64::EPSILON);
        assert!((line.peak_cooling_kw - 1.5).abs() < f64::EPSILON);
        assert_eq!(line.error.as_deref(), Some("boom"));
    }

    // ---------------------------------------------------------------
    // load_case_spec
    // ---------------------------------------------------------------

    #[test]
    fn load_case_spec_reads_yaml_and_json_round_trips() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let spec = ASHRAE140Case::Case600.spec();

        let yaml_path = tmp.path().join("base.yaml");
        std::fs::write(&yaml_path, serde_yaml::to_string(&spec).expect("to yaml")).unwrap();
        let from_yaml = load_case_spec(&yaml_path).expect("load yaml");
        assert_eq!(from_yaml.case_id, spec.case_id);
        assert_eq!(from_yaml.num_zones, spec.num_zones);

        let json_path = tmp.path().join("base.json");
        std::fs::write(&json_path, serde_json::to_string(&spec).expect("to json")).unwrap();
        let from_json = load_case_spec(&json_path).expect("load json");
        assert_eq!(from_json.case_id, spec.case_id);
    }

    #[test]
    fn load_case_spec_treats_unknown_extension_as_yaml() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let spec = ASHRAE140Case::Case600.spec();
        // No extension at all — the loader must fall back to YAML, and YAML is
        // a superset of JSON so a JSON payload must also parse.
        let path = tmp.path().join("base");
        std::fs::write(&path, serde_json::to_string(&spec).expect("to json")).unwrap();
        assert_eq!(
            load_case_spec(&path).expect("fallback parse").case_id,
            spec.case_id
        );
    }

    #[test]
    fn load_case_spec_errors_name_the_offending_path() {
        let missing = Path::new("/definitely/not/here/base.yaml");
        let err = load_case_spec(missing).unwrap_err().to_string();
        assert!(
            err.contains("base.yaml"),
            "read error must name the file: {err}"
        );

        let tmp = tempfile::tempdir().expect("tempdir");
        let bad = tmp.path().join("bad.json");
        std::fs::write(&bad, "{ not valid json").unwrap();
        let err = load_case_spec(&bad).unwrap_err().to_string();
        assert!(
            err.contains("bad.json"),
            "parse error must name the file: {err}"
        );
    }

    // ---------------------------------------------------------------
    // handle_monte_carlo_command dispatch
    // ---------------------------------------------------------------

    #[test]
    fn handle_monte_carlo_command_dry_run_end_to_end() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let delta_path = tmp.path().join("delta.yaml");
        std::fs::write(
            &delta_path,
            "samples: 3\nseed: 5\nwarm_up_years: 0\nparameters:\n  infiltration_ach:\n    distribution: uniform\n    min: 0.4\n    max: 0.6\n",
        )
        .unwrap();

        let cmd = MonteCarloCommand::DryRun(DryRunArgs {
            delta_file: delta_path.clone(),
            samples: Some(4),
            seed: Some(9),
        });
        // Dry-run never simulates, so this is a pure sample-and-print path.
        handle_monte_carlo_command(&cmd).expect("dry-run should succeed");

        // The override must be what actually drives the draw count.
        let mut delta = MonteCarloDelta::from_file(&delta_path).expect("load delta");
        apply_overrides(&mut delta, Some(4), Some(9));
        assert_eq!(delta.samples, 4);
        assert_eq!(delta.seed, 9);
        assert_eq!(
            monte_carlo::sample_parameters(&delta)
                .expect("sample")
                .len(),
            4
        );
    }

    #[test]
    fn handle_monte_carlo_command_dry_run_propagates_missing_delta_file() {
        let cmd = MonteCarloCommand::DryRun(DryRunArgs {
            delta_file: PathBuf::from("/definitely/not/here/delta.yaml"),
            samples: None,
            seed: None,
        });
        assert!(
            handle_monte_carlo_command(&cmd).is_err(),
            "a missing delta file must surface as an error"
        );
    }
}

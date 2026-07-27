//! End-to-end smoke test for Monte Carlo sweeps via declarative deltas (Issue #1813).
//!
//! Exercises the OSimFlow worker entrypoint contract: a base model file plus a
//! delta file produce N per-draw result files. Uses 10 draws (acceptance
//! criterion: "10 deltas → 10 result files") with `warm_up_years: 0` to keep the
//! test fast while still running the full patch → simulate pipeline.

use fluxion::analysis::monte_carlo::{self, MonteCarloDelta};
use fluxion::cli::monte_carlo::{
    handle_monte_carlo_command, DryRunArgs, MonteCarloCommand, SweepArgs,
};
use fluxion::validation::{ASHRAE140Case, CaseSpec};
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

const N_DRAWS: usize = 10;
const DELTA_YAML: &str = "\
samples: 10
seed: 1813
warm_up_years: 0
parameters:
  infiltration_ach:
    distribution: uniform
    min: 0.3
    max: 1.5
  window_properties.u_value:
    distribution: normal
    mean: 3.0
    std: 0.3
";

fn write_base_model(dir: &std::path::Path) -> PathBuf {
    let spec: CaseSpec = ASHRAE140Case::Case600.spec();
    let yaml = serde_yaml::to_string(&spec).expect("serialize base CaseSpec");
    let path = dir.join("base_model.yaml");
    std::fs::write(&path, yaml).unwrap();
    path
}

fn write_delta_file(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("delta.yaml");
    std::fs::write(&path, DELTA_YAML).unwrap();
    path
}

#[test]
fn dry_run_emits_n_samples() {
    let dir = TempDir::new().unwrap();
    let delta_path = write_delta_file(dir.path());
    // Override samples via the CLI to verify the override plumbing.
    let cmd = MonteCarloCommand::DryRun(DryRunArgs {
        delta_file: delta_path,
        samples: Some(N_DRAWS),
        seed: Some(1813),
    });
    // Dry-run prints to stdout; just verify it doesn't error.
    handle_monte_carlo_command(&cmd).expect("dry-run should succeed");
}

#[test]
fn sweep_produces_per_draw_result_files() {
    let dir = TempDir::new().unwrap();
    let base_path = write_base_model(dir.path());
    let delta_path = write_delta_file(dir.path());
    let out_dir = dir.path().join("mc_out");

    let cmd = MonteCarloCommand::Sweep(SweepArgs {
        base_model: base_path,
        delta_file: delta_path,
        output: out_dir.clone(),
        samples: Some(N_DRAWS),
        seed: None,
        hourly: false,
        per_draw_files: true,
        sequential: true, // deterministic ordering for the file-index assertions
    });
    handle_monte_carlo_command(&cmd).expect("sweep should succeed");

    // Acceptance criterion: 10 deltas → 10 result files.
    for i in 0..N_DRAWS {
        let f = out_dir.join(format!("delta_{i:06}.json"));
        assert!(f.exists(), "missing per-draw result file {}", f.display());
    }

    // results.jsonl should contain exactly N lines.
    let jsonl = std::fs::read_to_string(out_dir.join("results.jsonl")).unwrap();
    let lines: Vec<&str> = jsonl.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), N_DRAWS, "results.jsonl line count");

    // summary.json must be present and report count == N with no failures.
    let summary_text = std::fs::read_to_string(out_dir.join("summary.json")).unwrap();
    let summary: serde_json::Value = serde_json::from_str(&summary_text).unwrap();
    assert_eq!(summary["count"], N_DRAWS);
    assert_eq!(summary["failures"], 0);
    assert!(summary["wall_seconds"].as_f64().unwrap() >= 0.0);
    assert_eq!(summary["parallelism"], "sequential");
}

#[test]
fn each_result_reflects_its_patched_parameter() {
    // Verify that distinct infiltration_ach draws produce distinct heating loads,
    // i.e. each result file reflects the patched parameter rather than a stale
    // copy of the base model.
    let dir = TempDir::new().unwrap();
    let base_path = write_base_model(dir.path());
    let delta_path = write_delta_file(dir.path());
    let out_dir = dir.path().join("mc_out2");

    let cmd = MonteCarloCommand::Sweep(SweepArgs {
        base_model: base_path,
        delta_file: delta_path,
        output: out_dir.clone(),
        samples: Some(N_DRAWS),
        seed: None,
        hourly: false,
        per_draw_files: true,
        sequential: true,
    });
    handle_monte_carlo_command(&cmd).unwrap();

    let mut inputs: Vec<f64> = Vec::new();
    let mut heating: Vec<f64> = Vec::new();
    for i in 0..N_DRAWS {
        let text = std::fs::read_to_string(out_dir.join(format!("delta_{i:06}.json"))).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let ach = v["inputs"]["infiltration_ach"]
            .as_f64()
            .expect("infiltration_ach input present");
        inputs.push(ach);
        heating.push(v["annual_heating_mwh"].as_f64().unwrap());
    }
    // The infiltration_ach draws should not all be identical (uniform sampling).
    // f64 isn't Hash, so compare via rounded string keys.
    let unique_inputs: std::collections::HashSet<String> =
        inputs.iter().map(|v| format!("{v:.6}")).collect();
    assert!(
        unique_inputs.len() > 1,
        "expected varied infiltration_ach draws, got {inputs:?}"
    );
    // Higher infiltration should generally correlate with higher heating load;
    // check at least that heating values are not all identical (physics responds).
    let unique_heating: std::collections::HashSet<String> =
        heating.iter().map(|v| format!("{v:.6}")).collect();
    assert!(
        unique_heating.len() > 1,
        "heating loads identical across draws — patch not applied"
    );
}

#[test]
fn library_run_sweep_matches_cli_sample_count() {
    // Cross-check: the library sweep produces the same number of results as the
    // CLI, and each MonteCarloResult carries its input map.
    let base = ASHRAE140Case::Case600.spec();
    let mut delta = MonteCarloDelta {
        samples: 8,
        seed: 42,
        warm_up_years: 0,
        parameters: HashMap::new(),
    };
    delta.parameters.insert(
        "infiltration_ach".to_string(),
        monte_carlo::Distribution::Uniform { min: 0.4, max: 1.2 },
    );
    let results = monte_carlo::run_sweep(&base, &delta, false).unwrap();
    assert_eq!(results.len(), 8);
    assert!(results
        .iter()
        .all(|r| r.inputs.contains_key("infiltration_ach")));
    assert!(results.iter().all(|r| r.error.is_none()));
}

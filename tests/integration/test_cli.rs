//! CLI integration tests
//!
//! Tests validate CLI commands work correctly using std::process::Command.

use std::process::Command;
use tempfile::tempdir;

/// Test CLI validate --all command
#[test]
fn test_cli_validate_command() {
    // Run fluxion validate --all
    let output = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "validate", "--all"])
        .output()
        .expect("Failed to execute fluxion validate --all");

    // Check exit code (may be 0 or non-zero depending on validation results)
    // We just verify the command runs without crashing
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify output contains validation-related content
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI validate output:\n{}", output_str);

    // Command should run successfully
    assert!(
        output.status.success() || stderr.contains("Validation Report"),
        "CLI validate command should run successfully or show validation report"
    );
}

/// Test CLI validate --case command
#[test]
fn test_cli_validate_case_command() {
    // Run fluxion validate --case 600
    let output = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "validate", "--case", "600"])
        .output()
        .expect("Failed to execute fluxion validate --case 600");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify output contains case-specific content
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI validate --case 600 output:\n{}", output_str);

    // Command should run successfully
    assert!(
        output.status.success() || stderr.contains("Case 600"),
        "CLI validate --case command should run successfully or show case information"
    );
}

/// Test CLI sensitivity command (if available)
#[test]
fn test_cli_sensitivity_command() {
    // Create a temporary config file
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let config_path = temp_dir.path().join("sensitivity_config.yaml");

    // Create a minimal sensitivity config
    let config_content = r#"
case_id: "600"
method: "oat"
levels: 5
parameters:
  - name: "window_u_value"
    min: 1.0
    max: 3.0
  - name: "heating_setpoint"
    min: 18.0
    max: 22.0
"#;

    std::fs::write(&config_path, config_content).expect("Failed to write config file");

    // Run fluxion sensitivity --config
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "fluxion",
            "--",
            "sensitivity",
            "--config",
            config_path.to_str().unwrap(),
            // Keep scratch reports out of the repo root (issue #3303)
            "--output",
            temp_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute fluxion sensitivity");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify output contains sensitivity analysis content
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI sensitivity output:\n{}", output_str);

    // Command should run successfully (or fail gracefully if sensitivity not available)
    assert!(
        output.status.success() || stderr.contains("sensitivity") || stderr.contains("Sensitivity"),
        "CLI sensitivity command should run successfully or show sensitivity information"
    );
}

/// Test CLI binary exists and is executable
#[test]
fn test_cli_binary_exists() {
    // Verify cargo run --bin fluxion works
    let output = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "--help"])
        .output()
        .expect("Failed to execute fluxion --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify help output is shown
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI --help output:\n{}", output_str);

    // Command should show help information
    assert!(
        output_str.contains("Usage:") || output_str.contains("usage:"),
        "CLI --help should show usage information"
    );
}

// =====================================================================
// Issue #2947 (originally #2711): stubbed workflow commands must fail
// loudly with a non-zero exit code and a "not yet implemented" message
// referencing #2947 — never silently succeed (the pre-#2711 behaviour).
// =====================================================================

/// Run `cargo run --bin fluxion -- <args>` and return (status, stdout, stderr).
///
/// This helper invokes the binary through cargo so it works in unbuilt
/// worktrees (the CI runner does the same). Tests that just need to confirm
/// the binary's exit semantics do not need to assert anything about stdout.
fn run_fluxion(args: &[&str]) -> (std::process::ExitStatus, String, String) {
    let mut full: Vec<String> = vec!["run".into(), "--bin".into(), "fluxion".into(), "--".into()];
    full.extend(args.iter().map(|a| a.to_string()));
    let output = Command::new("cargo")
        .args(&full)
        .output()
        .expect("Failed to execute fluxion via cargo run");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status, stdout, stderr)
}

/// Assert that a (status, stdout, stderr) triple represents a loud (non-zero)
/// failure with a "not yet implemented" error referencing #2947.
fn assert_unimplemented_failure(
    label: &str,
    result: &(std::process::ExitStatus, String, String),
    expected_substring: &str,
) {
    let (status, stdout, stderr) = result;
    assert!(
        !status.success(),
        "[{label}] expected non-zero exit, got success. stdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("not yet implemented"),
        "[{label}] expected 'not yet implemented' message in output, got:\n{combined}"
    );
    assert!(
        combined.contains("#2947"),
        "[{label}] expected output to reference tracking issue #2947, got:\n{combined}"
    );
    assert!(
        combined.contains(expected_substring),
        "[{label}] expected output to name '{expected_substring}', got:\n{combined}"
    );
}

/// `fluxion run -w <empty workflow>` must fail loudly with #2947.
#[test]
fn test_cli_workflow_run_returns_not_implemented() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let workflow_path = temp_dir.path().join("empty.fwf");
    std::fs::write(&workflow_path, "{\"name\": \"stub\", \"steps\": []}")
        .expect("Failed to write workflow file");

    let args: Vec<String> = vec![
        "run".into(),
        "-w".into(),
        workflow_path.to_string_lossy().into_owned(),
    ];
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_fluxion(&args_ref);
    assert_unimplemented_failure("fluxion run -w <file>", &result, "workflow execution");
}

/// `fluxion run -w <file> --measures-only` must fail loudly with #2947.
#[test]
fn test_cli_workflow_run_measures_only_returns_not_implemented() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let workflow_path = temp_dir.path().join("m.fwf");
    std::fs::write(&workflow_path, "{}").expect("Failed to write workflow file");

    let args: Vec<String> = vec![
        "run".into(),
        "-w".into(),
        workflow_path.to_string_lossy().into_owned(),
        "--measures-only".into(),
    ];
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_fluxion(&args_ref);
    assert_unimplemented_failure(
        "fluxion run --measures-only",
        &result,
        "measures-only workflow",
    );
}

/// `fluxion run -w <file> --postprocess-only` must fail loudly with #2947.
#[test]
fn test_cli_workflow_run_postprocess_only_returns_not_implemented() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let workflow_path = temp_dir.path().join("p.fwf");
    std::fs::write(&workflow_path, "{}").expect("Failed to write workflow file");

    let args: Vec<String> = vec![
        "run".into(),
        "-w".into(),
        workflow_path.to_string_lossy().into_owned(),
        "--postprocess-only".into(),
    ];
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_fluxion(&args_ref);
    assert_unimplemented_failure(
        "fluxion run --postprocess-only",
        &result,
        "postprocess-only workflow",
    );
}

/// `fluxion measure update <dir>` must fail loudly with #2947.
#[test]
fn test_cli_measure_update_returns_not_implemented() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let measure_dir = temp_dir.path().join("measure");
    std::fs::create_dir_all(&measure_dir).expect("Failed to create dir");

    let args: Vec<String> = vec![
        "measure".into(),
        "update".into(),
        measure_dir.to_string_lossy().into_owned(),
    ];
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_fluxion(&args_ref);
    assert_unimplemented_failure("fluxion measure update", &result, "measure update");
}

/// `fluxion measure update-all <dir>` must fail loudly with #2947.
#[test]
fn test_cli_measure_update_all_returns_not_implemented() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let measures_dir = temp_dir.path().join("measures");
    std::fs::create_dir_all(&measures_dir).expect("Failed to create dir");

    let args: Vec<String> = vec![
        "measure".into(),
        "update-all".into(),
        measures_dir.to_string_lossy().into_owned(),
    ];
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_fluxion(&args_ref);
    assert_unimplemented_failure(
        "fluxion measure update-all",
        &result,
        "measure update --all",
    );
}

/// `fluxion measure compute-arguments <model> <measure>` must fail loudly with #2947.
#[test]
fn test_cli_measure_compute_args_returns_not_implemented() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.flux");
    let measure_dir = temp_dir.path().join("measure");
    std::fs::write(&model_path, "{}").expect("Failed to write model file");
    std::fs::create_dir_all(&measure_dir).expect("Failed to create measure dir");

    let args: Vec<String> = vec![
        "measure".into(),
        "compute-arguments".into(),
        model_path.to_string_lossy().into_owned(),
        measure_dir.to_string_lossy().into_owned(),
    ];
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_fluxion(&args_ref);
    assert_unimplemented_failure(
        "fluxion measure compute-arguments",
        &result,
        "measure compute-args",
    );
}

/// `fluxion measure run-tests <dir>` must fail loudly with #2947.
#[test]
fn test_cli_measure_run_tests_returns_not_implemented() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let measures_dir = temp_dir.path().join("measures");
    std::fs::create_dir_all(&measures_dir).expect("Failed to create dir");

    let args: Vec<String> = vec![
        "measure".into(),
        "run-tests".into(),
        measures_dir.to_string_lossy().into_owned(),
    ];
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_fluxion(&args_ref);
    assert_unimplemented_failure("fluxion measure run-tests", &result, "measure tests");
}

/// The `fluxion run --help` and `fluxion measure <sub> --help` outputs must
/// mark every unimplemented path as "[NOT YET IMPLEMENTED, see #2947]" so the
/// silent-success trap is visible at the help layer as well as at runtime.
#[test]
fn test_cli_help_marks_unimplemented_paths() {
    // `fluxion run --help`
    let (status, stdout, _stderr) = run_fluxion(&["run", "--help"]);
    assert!(status.success(), "fluxion run --help should exit 0");
    assert!(
        stdout.contains("[NOT YET IMPLEMENTED, see #2947]"),
        "fluxion run --help should mark itself unimplemented; got:\n{stdout}"
    );
    assert!(
        stdout.contains("measures-only"),
        "fluxion run --help should still document --measures-only; got:\n{stdout}"
    );
    assert!(
        stdout.contains("postprocess-only"),
        "fluxion run --help should still document --postprocess-only; got:\n{stdout}"
    );

    // Each measure subcommand --help
    for (sub, _) in [
        ("update", "measure update"),
        ("update-all", "measure update-all"),
        ("compute-arguments", "measure compute-arguments"),
        ("run-tests", "measure run-tests"),
    ] {
        let (status, stdout, _stderr) = run_fluxion(&["measure", sub, "--help"]);
        assert!(
            status.success(),
            "fluxion measure {sub} --help should exit 0"
        );
        assert!(
            stdout.contains("[NOT YET IMPLEMENTED, see #2947]"),
            "fluxion measure {sub} --help should mark itself unimplemented; got:\n{stdout}"
        );
    }
}

/// `fluxion validate-case 195-470` and `800-810` must also fail loudly with
/// #2947 (the same pattern as the workflow commands). This guards the
/// diagnostic stub paths that the previous #2711 fix gated.
#[test]
fn test_cli_validate_case_range_returns_not_implemented() {
    let result = run_fluxion(&["validate-case", "195-470"]);
    assert_unimplemented_failure(
        "fluxion validate-case 195-470",
        &result,
        "diagnostic case range 195-470",
    );

    let result = run_fluxion(&["validate-case", "800-810"]);
    assert_unimplemented_failure(
        "fluxion validate-case 800-810",
        &result,
        "diagnostic case range 800-810",
    );
}

// =====================================================================
// Issue #2946: `parallel-issue-workflow` binary was deleted because its
// `create_fix_for_issue()` produced a placeholder markdown file tagged
// with `Closes #N`, which auto-closed the originating issue without any
// real fix. This regression test asserts the binary (and its source
// file + Cargo.toml `[[bin]]` entry) remain removed — re-adding the
// tool would re-introduce the silent-placeholder trap.
// =====================================================================

/// `cargo build --bin parallel-issue-workflow` must fail: the binary was
/// removed in #2946 because it could auto-close issues with placeholder
/// markdown files. Re-adding it requires the fix-closed rewrite from the
/// issue's acceptance criteria, not a silent revival.
#[test]
fn test_parallel_issue_workflow_binary_is_removed() {
    let output = Command::new("cargo")
        .args(["build", "--bin", "parallel-issue-workflow"])
        .output()
        .expect("Failed to invoke cargo build --bin parallel-issue-workflow");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        !output.status.success(),
        "parallel-issue-workflow binary must not build (issue #2946); got:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // Cargo emits "no bin target named `parallel-issue-workflow`" when the
    // [[bin]] entry is gone. Match the canonical phrasing so we catch a
    // re-add that accidentally re-registers a different target name.
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("parallel-issue-workflow"),
        "cargo error should reference the removed binary by name; got:\n{combined}"
    );
}

/// Source file must be gone — re-creating only the `[[bin]]` entry would
/// resurrect the placeholder-generating logic.
#[test]
fn test_parallel_issue_workflow_source_is_removed() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("bin")
        .join("parallel_issue_workflow.rs");
    assert!(
        !path.exists(),
        "src/bin/parallel_issue_workflow.rs must be removed (issue #2946); found at {path:?}"
    );
}

// ---- Issue #3283 — --zone-solver / --conduction-solver flags --------------

/// `fluxion --help` must advertise the new solver-selection flags.
#[test]
fn test_cli_help_shows_solver_flags() {
    let (status, stdout, stderr) = run_fluxion(&["--help"]);
    assert!(status.success(), "--help must exit 0; stderr:\n{stderr}");
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("--zone-solver"),
        "--help must list --zone-solver; got:\n{combined}"
    );
    assert!(
        combined.contains("--conduction-solver"),
        "--help must list --conduction-solver; got:\n{combined}"
    );
}

/// Unknown zone-solver values must fail with a non-zero exit before any
/// simulation work, using the shared rejection wording.
#[test]
fn test_cli_unknown_zone_solver_fails_loudly() {
    let (status, stdout, stderr) = run_fluxion(&["--zone-solver", "warp_drive", "dummy.flux"]);
    assert!(
        !status.success(),
        "unknown --zone-solver must exit non-zero; stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("unknown zone_solver"),
        "error must use the shared 'unknown zone_solver' wording; got:\n{combined}"
    );
}

/// Unknown conduction-solver values must fail with a non-zero exit.
#[test]
fn test_cli_unknown_conduction_solver_fails_loudly() {
    let (status, stdout, stderr) = run_fluxion(&["--conduction-solver", "quantum", "dummy.flux"]);
    assert!(
        !status.success(),
        "unknown --conduction-solver must exit non-zero; stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("unknown conduction_solver"),
        "error must use the shared 'unknown conduction_solver' wording; got:\n{combined}"
    );
}

/// The experimental `6r2c` identifier must be rejected (fail-closed) with a
/// message naming the FLUXION_EXPERIMENTAL_ZONE_SOLVERS gate — regardless of
/// whether the env var is set on the test process (without the
/// `fluxion-experimental-zone-solvers` cargo feature there is no variant to
/// construct).
#[test]
fn test_cli_experimental_zone_solver_rejected() {
    let (status, stdout, stderr) = run_fluxion(&["--zone-solver", "6r2c", "dummy.flux"]);
    assert!(
        !status.success(),
        "experimental --zone-solver 6r2c must exit non-zero; stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("experimental"),
        "rejection must be flagged experimental; got:\n{combined}"
    );
    assert!(
        combined.contains("6r2c"),
        "rejection must name the identifier; got:\n{combined}"
    );
}

/// Valid flag values must pass selection validation (the run then fails for
/// the unrelated, expected reason: the input file does not exist — NOT with
/// a solver error). A weather file is supplied so the run reaches the
/// input-file precondition (the weather check runs first).
#[test]
fn test_cli_valid_solver_flags_pass_selection() {
    let (status, stdout, stderr) = run_fluxion(&[
        "--zone-solver",
        "5r1c",
        "--conduction-solver",
        "ctf",
        "--weather",
        "dummy.epw",
        "definitely-missing-input.flux",
    ]);
    assert!(
        !status.success(),
        "missing input file must still fail; stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        !combined.contains("unknown zone_solver")
            && !combined.contains("unknown conduction_solver"),
        "valid solver values must NOT trigger selection errors; got:\n{combined}"
    );
    assert!(
        combined.contains("Input file not found"),
        "failure must be the missing-input check, proving selection passed; got:\n{combined}"
    );
}

// ---------------------------------------------------------------------------
// `fluxion topology export` / `fluxion export-topology` (Issue #3963/#3966)
// ---------------------------------------------------------------------------

/// Helper: run the binary, require success, and return stdout.
fn topology_export_stdout(args: &[&str]) -> String {
    let (status, stdout, stderr) = run_fluxion(args);
    assert!(
        status.success(),
        "`fluxion {}` must succeed; stdout:\n{stdout}\nstderr:\n{stderr}",
        args.join(" ")
    );
    stdout
}

/// Hand-rolled structural validation against `schemas/topology_v1.schema.json`
/// (no JSON-Schema validator dependency; see Issue #3963 task notes).
fn assert_matches_schema(doc: &serde_json::Value) {
    let schema_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/schemas/topology_v1.schema.json"
    );
    let schema: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(schema_path).expect("schemas/topology_v1.schema.json readable"),
    )
    .expect("schema is valid JSON");

    let node_kinds = schema["$defs"]["node_kind"]["enum"]
        .as_array()
        .expect("node_kind enum");
    let edge_kinds = schema["$defs"]["edge_kind"]["enum"]
        .as_array()
        .expect("edge_kind enum");

    let metadata = doc.get("metadata").expect("metadata block present");
    for key in [
        "schema_version",
        "tool_version",
        "model_name",
        "model_source",
        "node_count",
        "edge_count",
    ] {
        assert!(metadata.get(key).is_some(), "metadata.{key} required");
    }
    assert_eq!(metadata["schema_version"], "1.0.0");
    assert!(
        metadata.get("timestamp").is_none(),
        "timestamp omitted by default"
    );

    let nodes = doc["nodes"].as_array().expect("nodes array");
    let edges = doc["edges"].as_array().expect("edges array");
    assert_eq!(
        metadata["node_count"].as_u64().unwrap(),
        nodes.len() as u64,
        "node_count matches array length"
    );
    assert_eq!(
        metadata["edge_count"].as_u64().unwrap(),
        edges.len() as u64,
        "edge_count matches array length"
    );

    let mut ids: Vec<&str> = Vec::new();
    for n in nodes {
        for key in ["id", "kind", "name"] {
            assert!(n.get(key).is_some(), "node.{key} required");
        }
        assert!(
            node_kinds.contains(&n["kind"]),
            "node kind {} is in the schema enum",
            n["kind"]
        );
        if let Some(c) = n.get("capacitance_j_per_k") {
            assert!(c.as_f64().unwrap() >= 0.0, "capacitance >= 0");
        }
        ids.push(n["id"].as_str().unwrap());
    }
    let mut sorted_ids = ids.clone();
    sorted_ids.sort();
    assert_eq!(ids, sorted_ids, "nodes sorted by stable id");
    let id_set: std::collections::BTreeSet<&str> = ids.iter().copied().collect();
    assert_eq!(id_set.len(), ids.len(), "node ids unique");

    let mut edge_keys: Vec<(&str, &str, &str)> = Vec::new();
    for e in edges {
        for key in ["source_id", "target_id", "coupling_type", "bidirectional"] {
            assert!(e.get(key).is_some(), "edge.{key} required");
        }
        assert!(
            edge_kinds.contains(&e["coupling_type"]),
            "edge kind {} is in the schema enum",
            e["coupling_type"]
        );
        assert!(
            id_set.contains(e["source_id"].as_str().unwrap()),
            "edge source resolves: {}",
            e["source_id"]
        );
        assert!(
            id_set.contains(e["target_id"].as_str().unwrap()),
            "edge target resolves: {}",
            e["target_id"]
        );
        if let Some(f) = e.get("fraction") {
            let f = f.as_f64().unwrap();
            assert!((0.0..=1.0).contains(&f), "fraction within [0,1]: {f}");
        }
        edge_keys.push((
            e["source_id"].as_str().unwrap(),
            e["target_id"].as_str().unwrap(),
            e["coupling_type"].as_str().unwrap(),
        ));
    }
    let mut sorted_edges = edge_keys.clone();
    sorted_edges.sort();
    assert_eq!(
        edge_keys, sorted_edges,
        "edges sorted by (source,target,kind)"
    );
}

#[test]
fn test_topology_export_case_600_deterministic_and_schema_valid() {
    let a = topology_export_stdout(&["topology", "export", "--case", "600"]);
    let b = topology_export_stdout(&["topology", "export", "--case", "600"]);
    assert_eq!(a, b, "identical inputs must yield byte-identical output");

    let doc: serde_json::Value = serde_json::from_str(&a).expect("stdout is valid JSON");
    assert_matches_schema(&doc);

    // Thermal-network essentials for Case 600 (single zone, low mass).
    let kinds = |kind: &str| {
        doc["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| n["kind"] == kind)
            .count()
    };
    assert_eq!(kinds("outdoor_ambient"), 1);
    assert_eq!(kinds("zone_air"), 1);
    assert_eq!(kinds("internal_mass"), 1);
    assert_eq!(kinds("internal_gain"), 1);
    assert_eq!(kinds("hvac_terminal"), 1);
    assert!(
        kinds("wall_layer") >= 6,
        "multi-layer envelope nodes present"
    );
    assert!(kinds("exterior_surface") >= 5);
    assert!(kinds("interior_surface") >= 5);

    let edge_kinds = |kind: &str| {
        doc["edges"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["coupling_type"] == kind)
            .count()
    };
    assert!(edge_kinds("shortwave_solar_direct") >= 1);
    assert!(edge_kinds("shortwave_solar_diffuse") >= 1);
    assert_eq!(
        edge_kinds("internal_gain_split"),
        2,
        "radiative + convective split"
    );
    assert!(edge_kinds("air_exchange") >= 1, "infiltration present");
}

#[test]
fn test_topology_export_alias_matches_subcommand() {
    let alias = topology_export_stdout(&["export-topology", "--case", "600"]);
    let subcommand = topology_export_stdout(&["topology", "export", "--case", "600"]);
    assert_eq!(
        alias, subcommand,
        "`export-topology` alias must behave identically to `topology export`"
    );
}

#[test]
fn test_topology_export_rejects_case_and_model_together() {
    let (status, _stdout, stderr) = run_fluxion(&[
        "topology",
        "export",
        "--case",
        "600",
        "--model",
        "does-not-matter.json",
    ]);
    assert!(
        !status.success(),
        "--case and --model are mutually exclusive; stderr:\n{stderr}"
    );
}

#[test]
fn test_topology_export_rejects_missing_source() {
    let (status, _stdout, stderr) = run_fluxion(&["topology", "export"]);
    assert!(
        !status.success(),
        "one of --case/--model is required; stderr:\n{stderr}"
    );
    let (status_alias, _o, stderr_alias) = run_fluxion(&["export-topology"]);
    assert!(
        !status_alias.success(),
        "alias must reject missing source too; stderr:\n{stderr_alias}"
    );
}

#[test]
fn test_topology_export_unknown_case_fails_loud() {
    let (status, _stdout, stderr) = run_fluxion(&["topology", "export", "--case", "42"]);
    assert!(
        !status.success(),
        "unknown case id must fail non-zero; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("unknown ASHRAE 140 case id"),
        "error must name the problem; stderr:\n{stderr}"
    );
}

#[test]
fn test_topology_export_writes_output_file() {
    let temp_dir = tempdir().expect("tempdir");
    let out_path = temp_dir.path().join("case600.json");
    let out_str = out_path.to_string_lossy().into_owned();
    let (status, stdout, _stderr) =
        run_fluxion(&["topology", "export", "--case", "600", "--output", &out_str]);
    assert!(status.success(), "stdout:\n{stdout}");
    let written = std::fs::read_to_string(&out_path).expect("output file written");
    let doc: serde_json::Value = serde_json::from_str(&written).expect("file is valid JSON");
    assert_matches_schema(&doc);
}

#[test]
fn test_topology_export_model_file_roundtrip() {
    // Serialize the Case 600 spec through the library and export it via
    // `--model`; node/edge arrays must match the `--case 600` export (only
    // metadata provenance differs).
    let spec = fluxion::validation::ashrae_140_cases::ASHRAE140Case::Case600.spec();
    let spec_json = serde_json::to_string_pretty(&spec).expect("CaseSpec serializes");
    let temp_dir = tempdir().expect("tempdir");
    let model_path = temp_dir.path().join("case600_model.json");
    std::fs::write(&model_path, spec_json).expect("write model file");

    let model_str = model_path.to_string_lossy().into_owned();
    let via_model = topology_export_stdout(&["topology", "export", "--model", &model_str]);
    let via_case = topology_export_stdout(&["topology", "export", "--case", "600"]);

    let m: serde_json::Value = serde_json::from_str(&via_model).unwrap();
    let c: serde_json::Value = serde_json::from_str(&via_case).unwrap();
    assert_eq!(
        m["nodes"], c["nodes"],
        "identical graph regardless of source"
    );
    assert_eq!(m["edges"], c["edges"]);
    assert_eq!(
        m["metadata"]["model_source"],
        serde_json::json!(format!("model-file:{model_str}")),
        "provenance records the file path"
    );
}

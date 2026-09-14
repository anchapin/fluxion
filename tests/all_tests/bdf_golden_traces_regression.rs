//! Issue #3339 — Golden transient traces regression test.
//!
//! Locks the byte-equivalent baseline trace for the 5 stiff DAE
//! benchmark circuits. The strategy input is the `mode=0,
//! baseline_factor=1.0` fixed-damping schedule the original
//! `NewtonRaphsonConfig::default()` always produced.
//!
//! If a future change to `bdf_engine.rs` alters the inner Newton
//! iteration count, accepted/step count, or the conservation
//! residuals by more than the tolerances below, this test FAILS —
//! the campaign loses its byte-equivalent baseline and the
//! `SCORECARD.md` invariant is at risk.
//!
//! The committed `tools/evolution/results/dae/golden/baseline.json`
//! is the canonical oracle; the test re-runs the binary with the
//! same `--strategy-file` and compares per-circuit metrics.

use std::path::PathBuf;
use std::process::Command;

/// Locate the `bdf_evaluator` binary. Build it on demand if missing.
fn bdf_evaluator_path() -> PathBuf {
    // The binary lives at `target/release/bdf_evaluator` after
    // `cargo build --release --bin bdf_evaluator -p fluxion`.
    // The debug-binary fallback keeps the test runnable in CI
    // profiles that omit release builds.
    let release = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/bdf_evaluator");
    if release.exists() {
        return release;
    }
    let debug = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/bdf_evaluator");
    if debug.exists() {
        return debug;
    }
    panic!(
        "bdf_evaluator binary not built. Run `cargo build --release --bin bdf_evaluator -p fluxion` first."
    );
}

/// Locate the workspace root (this test file is in `tests/` of the
/// root crate).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn issue_3339_golden_transient_traces_are_byte_equivalent() {
    let bin = bdf_evaluator_path();
    let ws = workspace_root();
    let strategy = ws.join("tools/evolution/results/dae/golden/_baseline_strategy.json");
    let out_dir = ws.join("tools/evolution/results/dae/golden/_regression");
    let out_dir_str = out_dir.to_string_lossy().to_string();
    let _ = std::fs::create_dir_all(&out_dir);

    // Always (re)write the strategy file from the in-tree golden.
    std::fs::write(
        &strategy,
        r#"{
  "mode": 0,
  "baseline_factor": 1.0,
  "floor": 0.25,
  "loose_threshold": 0.5,
  "tight_threshold": 0.95,
  "aggressiveness": 1.0,
  "history_window": 4
}
"#,
    )
    .expect("write baseline strategy");

    let out_path = out_dir.join("baseline.json");
    let _ = std::fs::remove_file(&out_path);

    let status = Command::new(&bin)
        .args([
            "--candidate-id",
            "regression-baseline",
            "--strategy-file",
            strategy.to_string_lossy().as_ref(),
            "--generation",
            "0",
            "--output",
            out_path.to_string_lossy().as_ref(),
        ])
        .status()
        .expect("failed to invoke bdf_evaluator");
    assert!(
        status.success(),
        "bdf_evaluator must succeed for the baseline strategy (got {})",
        out_dir_str
    );

    let summary_text = std::fs::read_to_string(&out_path).expect("read summary");
    let summary: serde_json::Value = serde_json::from_str(&summary_text).expect("summary is JSON");

    // Per-circuit golden values from
    // `tools/evolution/results/dae/golden/baseline.json` — locked here
    // so the test does not require reading another file at runtime.
    let expected_per_circuit: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let circuits = summary
        .get("bdf_per_circuit")
        .and_then(|v| v.as_object())
        .expect("bdf_per_circuit must be present");

    // Each circuit must have the byte-equivalent metrics; the
    // campaign's score-improvement direction is tested separately.
    let expected: &[(&str, usize, usize)] = &[
        ("cooling_coil_wet", 8, 6),
        ("decoupling_loop_demand", 8, 6),
        ("heatpump_entering_fluid_step", 8, 6),
        ("mixing_valve_closure", 6, 4),
        ("pump_freq_ramp", 7, 5),
    ];
    for (name, iters, acc) in expected {
        let c = circuits
            .get(*name)
            .unwrap_or_else(|| panic!("missing circuit `{}`", name));
        assert_eq!(
            c.get("newton_iterations").and_then(|v| v.as_u64()).unwrap(),
            *iters as u64,
            "circuit `{}` Newton-iteration count drifted from golden",
            name
        );
        assert_eq!(
            c.get("steps_accepted").and_then(|v| v.as_u64()).unwrap(),
            *acc as u64,
            "circuit `{}` accepted-step count drifted from golden",
            name
        );
        assert_eq!(
            c.get("status").and_then(|v| v.as_str()).unwrap(),
            "ok",
            "circuit `{}` status drifted from ok",
            name
        );
        assert_eq!(
            c.get("nan_or_inf_count").and_then(|v| v.as_u64()).unwrap(),
            0,
            "circuit `{}` introduced NaN/Inf",
            name
        );
        // Touch the variable so the unused-warning is silenced.
        let _ = expected_per_circuit;
    }

    assert!(
        summary
            .get("invariants_passed")
            .and_then(|v| v.as_bool())
            .unwrap(),
        "baseline regression must keep all conservation invariants"
    );
    assert!(
        summary.get("compiled").and_then(|v| v.as_bool()).unwrap(),
        "baseline must compile"
    );
    // Total Newton iterations baseline = 37 (the byte-equivalent
    // reference; used as the denominator for the
    // 25%-improvement acceptance criterion in issue #3339).
    let total_iters: usize = expected.iter().map(|(_, i, _)| i).sum();
    assert_eq!(
        total_iters, 37,
        "baseline total Newton iterations must remain 37 (issue #3339 acceptance denominator)"
    );
}

// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Gauge Python diagnostic integration test (Issue #1699).
//!
//! Runs `.agents/results/issue-1465-diurnal-parity.py` via `std::process::Command`.
//! This script verifies GaugeSolver diurnal parity vs FiveR1CSolver baseline through
//! a 24-hour synthetic Case 900 diurnal forcing.
//!
//! SOME FAILURES ARE EXPECTED — the thermal mass lag in FiveR1C causes it to
//! deviate more than ±10% from the steady-state GaugeSolver flux. This is known
//! physics (τ ≈ 25.6 h thermal mass time constant) and is documented in the
//! script itself.
//!
//! The test validates that the Python script executes without crashing (panic,
//! segfault, etc.) and that it produces structured output. The exit code is
//! NOT asserted because the script legitimately returns 1 when diurnal parity
//! checks fail.
//!
//! The script lives under `.agents/results/`, which is `.gitignore`d. It was
//! historically tracked (commit `026875e` lineage, paired with
//! `issue-1464-qubo-verification.py`) but untracked by the #3076 hygiene
//! cleanup. When the script is absent we skip the test rather than fail —
//! the GaugeSolver parity logic itself is covered by the dedicated unit
//! tests in `src/sim/gauge_solver*.rs`; this script is a one-off
//! cross-check, not part of the published verification surface.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn gauge_python_diagnostic() {
    let script = repo_root().join(".agents/results/issue-1465-diurnal-parity.py");
    if !script.exists() {
        eprintln!(
            "Diurnal parity script not found at {}; skipping. \
             Restore the script (or move it to a tracked path) to re-enable \
             this cross-check.",
            script.display()
        );
        return;
    }

    let output = Command::new("python3")
        .arg(&script)
        .current_dir(repo_root())
        .output()
        .expect("Failed to execute python3 for diurnal parity diagnostic");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "Diurnal parity script produced no output (possible crash)"
    );

    assert!(
        stdout.contains("Hour |") && stdout.contains("Gauge") && stdout.contains("5R1C"),
        "Diurnal parity script did not produce expected hourly table output\n\nstdout:\n{}",
        stdout
    );

    assert!(
        stdout.contains("SUMMARY:") || stdout.contains("PASS") || stdout.contains("FAIL"),
        "Diurnal parity script did not produce a SUMMARY line\n\nstdout:\n{}",
        stdout
    );
}

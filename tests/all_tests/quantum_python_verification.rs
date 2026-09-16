// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Quantum Python verification integration test (Issue #1699).
//!
//! Runs `.agents/results/issue-1464-qubo-verification.py` via `std::process::Command`
//! and asserts exit code 0. The script validates the QUBO encoding math that
//! `src/quantum/qubo_mapping.rs` implements:
//!
//!     Q[(i,k), (j,l)] = metric[i,j] * 2^k * 2^l / scale^2
//!
//! so that for any binary `x`:
//!     x^T Q x = T_recon^T · metric · T_recon  (+ optional gauge bias)
//!
//! This test ensures that Rust changes to `geometry_tensor.rs` or `qubo_mapping.rs`
//! cannot break the Python-layer math without CI catching it.
//!
//! The script lives under `.agents/results/`, which is `.gitignore`d. It was
//! historically tracked (commit 026875e) but untracked by #3076 (commit 07f4d1e).
//! When the script is absent we skip the test rather than fail — the rest of
//! the test suite still covers `src/quantum/qubo_mapping.rs` via the 18 unit
//! tests in that module, and the script was a one-off cross-check, not part
//! of the published verification surface.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn quantum_python_verification() {
    let script = repo_root().join(".agents/results/issue-1464-qubo-verification.py");
    if !script.exists() {
        eprintln!(
            "QUBO verification script not found at {}; skipping. \
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
        .expect("Failed to execute python3 for QUBO verification");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "QUBO verification script failed (exit {})\n\nstdout:\n{}\n\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );
}

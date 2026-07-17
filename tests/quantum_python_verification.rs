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

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn quantum_python_verification() {
    let script = repo_root().join(".agents/results/issue-1464-qubo-verification.py");
    assert!(
        script.exists(),
        "QUBO verification script not found at {}",
        script.display()
    );

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

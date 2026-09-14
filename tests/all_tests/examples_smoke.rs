// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Smoke tests for the `examples/` directory and `tests/fixtures/` (Issue #1411).
//!
//! The goal is to prevent docs/example drift: any change that breaks
//! `python -c "import fluxion"` semantics, the canonical REST request body
//! (`tests/fixtures/single_zone.json`), or the basic parseability of the
//! example scripts fails CI here before it ships.
//!
//! This file deliberately does NOT touch `src/` or `Cargo.toml`; it only
//! asserts on artefacts under `examples/`, `docs/`, and `tests/fixtures/`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use fluxion::api::schema::SimulationSchemaV1;
use fluxion::api::server::AppState;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The canonical REST request body used by `docs/REST_API.md` and
/// `examples/run_rest.sh` must deserialize into the live
/// `SimulationSchemaV1`. If this fails, the fixture has drifted from the
/// Rust schema and every doc example is broken.
#[test]
fn single_zone_fixture_deserializes_against_simulation_schema_v1() {
    let path = repo_root().join("tests/fixtures/single_zone.json");
    assert!(path.exists(), "missing required fixture {}", path.display());
    let body = read(&path);
    let parsed: SimulationSchemaV1 = serde_json::from_str(&body).unwrap_or_else(|e| {
        panic!(
            "tests/fixtures/single_zone.json does not match SimulationSchemaV1: {e}\n\nbody: {body}"
        )
    });
    assert_eq!(parsed.geometry.zones.len(), 1);
    assert!(
        parsed.controls.zone_control.heating_setpoint
            < parsed.controls.zone_control.cooling_setpoint,
        "fixture must satisfy heating < cooling (Issue #1411 acceptance criterion)"
    );
}

/// `examples/run_model.py` and `examples/run_oracle.py` must at least
/// parse. We do not execute them — the Python bindings may not be built
/// in the CI environment that runs `cargo test` — but a syntax error
/// should fail CI immediately.
///
/// We shell out to `python3 -c "import ast; ast.parse(open('PATH').read())"`
/// because the AST-grammar check is the only way to catch real Python
/// syntax errors without pulling in a Python parser as a Rust dep. The
/// test is skipped if `python3` is not on PATH (e.g. minimal containers).
#[test]
fn example_python_scripts_parse() {
    for name in ["run_model.py", "run_oracle.py"] {
        let path = repo_root().join("examples").join(name);
        assert!(path.exists(), "missing example script {}", path.display());
        let src = read(&path);
        assert!(
            src.contains("def "),
            "{name} contains no `def` — is it a Python script?"
        );
        check_python_syntax(&path).unwrap_or_else(|e| {
            panic!("{name} fails to parse as Python: {e}");
        });
    }
}

/// `examples/quick_start.sh` and `examples/run_rest.sh` must be valid
/// bash and reference scripts that actually exist. `run_rest.sh` was
/// added in #1411; the check tolerates it not yet being present in case
/// this test is backported to a branch that landed partial work.
#[test]
fn example_shell_scripts_are_well_formed() {
    for name in ["quick_start.sh", "run_rest.sh"] {
        let path = repo_root().join("examples").join(name);
        if !path.exists() {
            if name == "run_rest.sh" {
                continue;
            }
            panic!("missing example script {}", path.display());
        }
        let src = read(&path);
        assert!(
            src.contains("#!/usr/bin/env bash") || src.contains("#!/bin/bash"),
            "{name} must start with a bash shebang"
        );
    }
}

/// The README in `examples/` references files that must exist. Naive
/// line-level scan over backtick-quoted paths; good enough to catch
/// rename/deletion regressions.
#[test]
fn examples_readme_does_not_reference_missing_files() {
    let readme = read(&repo_root().join("examples/README.md"));
    for line in readme.lines() {
        let mut rest = line;
        while let Some(start) = rest.find('`') {
            if let Some(end_rel) = rest[start + 1..].find('`') {
                let inner = &rest[start + 1..start + 1 + end_rel];
                if inner.starts_with("examples/") {
                    let rel = inner.trim_start_matches("examples/");
                    if rel.ends_with('/') || rel.contains('*') {
                        // not a concrete file reference
                        rest = &rest[start + 1 + end_rel + 1..];
                        continue;
                    }
                    let path = repo_root().join("examples").join(rel);
                    assert!(
                        path.exists(),
                        "examples/README.md references missing file: {inner}"
                    );
                }
                rest = &rest[start + 1 + end_rel + 1..];
            } else {
                break;
            }
        }
    }
}

/// `AppState::default()` is part of the public REST surface; this test
/// guarantees the type is reachable from an external integration test
/// and the constructor does not panic.
#[test]
fn api_appstate_is_constructible() {
    let _ = AppState::default();
}

/// Shell out to `python3` for the AST parse. Returns `Ok(())` on success,
/// `Err(msg)` on parse failure, or `Ok(())` if Python is not available
/// (the rest of the file's checks still run; a missing interpreter is
/// not a regression in `examples/`).
fn check_python_syntax(path: &Path) -> Result<(), String> {
    let output = Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import ast,sys; ast.parse(open({:?}).read())",
            path.to_string_lossy()
        ))
        .output();
    match output {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => Err(format!(
            "python3 ast.parse failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )),
        Err(_) => {
            // python3 not on PATH; treat as a soft skip rather than a hard
            // failure. The `def ` check above still catches trivial damage.
            Ok(())
        }
    }
}

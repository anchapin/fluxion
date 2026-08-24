//! Runtime architectural boundary enforcement for `fluxion-core`.
//!
//! Verifies that `fluxion-core` (the dependency-light leaf crate) does NOT
//! import from any of the main crate's boundary modules: `sim`, `physics`,
//! `ai`, or `validation`. This test mirrors the Python scripts
//! (`scripts/check_ashrae_cases_cycle.py`, `scripts/check_physics_sim_cycle.py`)
//! but provides Rust-level CI coverage that cannot be bypassed locally.
//!
//! Issue: #3168

use std::process::Command;

const FORBIDDEN: &[&str] = &["sim", "physics", "ai", "validation"];

#[test]
fn fluxion_core_has_no_upward_crate_dependencies() {
    let metadata = cargo_metadata();
    let packages = metadata["packages"].as_array().expect("packages must be an array");

    let fluxion_core_pkg = packages
        .iter()
        .find(|p| p["name"].as_str() == Some("fluxion-core"))
        .expect("fluxion-core package must be present");

    let deps = fluxion_core_pkg["dependencies"]
        .as_array()
        .expect("dependencies must be an array");

    let forbidden_deps: Vec<_> = deps
        .iter()
        .filter(|d| {
            d["name"]
                .as_str()
                .map(|n| FORBIDDEN.contains(&n))
                .unwrap_or(false)
        })
        .collect();

    if !forbidden_deps.is_empty() {
        let names: Vec<_> = forbidden_deps
            .iter()
            .map(|d| d["name"].as_str().unwrap())
            .collect();
        panic!(
            "fluxion-core must not depend on {}, but found: {names:?}",
            FORBIDDEN.join(", ")
        );
    }
}

#[test]
fn fluxion_core_source_has_no_upward_crate_references() {
    let repo_root = repo_root();
    let fluxion_core_src = repo_root.join("fluxion-core").join("src");

    let mut offenders = Vec::new();

    collect_crate_references(&fluxion_core_src, &mut offenders);

    if !offenders.is_empty() {
        let msg = offenders
            .iter()
            .map(|(file, line, content)| format!("{file}:{line}: {content}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "fluxion-core source must not reference crate::sim|physics|ai|validation:\n{msg}"
        );
    }
}

fn cargo_metadata() -> serde_json::Value {
    let manifest_path = repo_root()
        .join("fluxion-core")
        .join("Cargo.toml");

    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--format-version=1")
        .arg("--no-deps")
        .arg(format!("--manifest-path={}", manifest_path.display()))
        .current_dir(repo_root())
        .output()
        .expect("cargo metadata must succeed");

    if !output.status.success() {
        panic!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout).expect("valid JSON from cargo metadata")
}

fn repo_root() -> std::path::PathBuf {
    let cargo_manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set");
    let manifest_path = std::path::PathBuf::from(cargo_manifest);

    manifest_path
        .parent()
        .expect("CARGO_MANIFEST_DIR should have a parent")
        .to_path_buf()
}

fn collect_crate_references(dir: &std::path::Path, offenders: &mut Vec<(String, usize, String)>) {
    let upward_prefixes: Vec<&str> = vec![
        "crate::sim::",
        "crate::physics::",
        "crate::ai::",
        "crate::validation::",
    ];

    if !dir.exists() {
        return;
    }

    walkdir_non_recursive(dir, offenders, &upward_prefixes);
}

fn walkdir_non_recursive(
    dir: &std::path::Path,
    offenders: &mut Vec<(String, usize, String)>,
    prefixes: &[&str],
) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                walkdir_non_recursive(&path, offenders, prefixes);
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                scan_file(&path, offenders, prefixes);
            }
        }
    }
}

fn scan_file(path: &std::path::Path, offenders: &mut Vec<(String, usize, String)>, prefixes: &[&str]) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return,
    };

    for (lineno, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*") {
            continue;
        }
        for prefix in prefixes {
            if trimmed.contains(prefix) {
                let rel = path
                    .strip_prefix(repo_root())
                    .unwrap_or(path)
                    .display()
                    .to_string();
                offenders.push((rel, lineno + 1, trimmed.to_string()));
                break;
            }
        }
    }
}

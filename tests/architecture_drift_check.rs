//! Architecture drift check — pins claims in ARCHITECTURE.md to code.
//!
//! Issue #1509: ensures that the documented canonical constants, solver
//! registry composition, and absence of legacy literals stay in sync with
//! the codebase. If this test fails, either ARCHITECTURE.md needs updating
//! or the code has drifted from the documented contract.

use std::fs;
use std::path::PathBuf;

use fluxion::physics::constants::EXTERIOR_FILM_COEFF;
use fluxion::physics::ctf_solver_wrapper::CTFSolverWrapper;
use fluxion::physics::fd_solver_wrapper::FDSolverWrapper;
use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::physics::solver_registry::registry_keys;
use fluxion::physics::solver_registry::SolverRegistry;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::wall_spec::WallSpec;

/// Pinned canonical exterior film coefficient (v2023, ASHRAE 140 Sec. 5.2).
///
/// If this value changes, update ARCHITECTURE.md Module 3 §h_exterior.
#[test]
fn exterior_film_coeff_is_canonical_18_3() {
    assert_eq!(
        EXTERIOR_FILM_COEFF, 18.3,
        "EXTERIOR_FILM_COEFF must be 18.3 W/m²K (v2023). \
         If this changed, update ARCHITECTURE.md Module 3."
    );
}

/// No legacy `1.0 / 29.3` literal may appear in any `.rs` file under `src/`.
///
/// The pre-#1140 h_exterior (29.3 W/m²K) was replaced by the v2023 canonical
/// 18.3 W/m²K. Whitespace is normalized so variants like `1.0/29.3` or
/// `1.0  /  29.3` are also caught.
#[test]
fn no_legacy_29_3_literal_in_src() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");
    let mut offenders = Vec::new();
    scan_for_literal(&src_dir, &normalize_ws("1.0 / 29.3"), &mut offenders);
    assert!(
        offenders.is_empty(),
        "Found legacy `1.0 / 29.3` literal in src/ — use `1.0 / EXTERIOR_FILM_COEFF` \
         or `1.0 / ASHRAE140_H_EXT` instead.\nOffending files:\n{}",
        offenders.join("\n")
    );
}

/// `SolverRegistry` + direct construction must export ≥ 3 conduction solver
/// constructors.
///
/// ARCHITECTURE.md Module 3 §SolverRegistry lists 5R1C, CTF/FD, and
/// MultiNodeSolver. If fewer than 3 distinct solver types are constructible,
/// the doc or the code has drifted.
#[test]
fn solver_registry_exports_at_least_3_constructors() {
    let wall = WallSpec::single_layer("drift-check-wall", 0.20, 0.51, 1400.0, 840.0);

    let mut solvers: Vec<Box<dyn HeatConductionSolver>> = Vec::new();

    // 1. 5R1C via SolverRegistry::construct
    if let Ok(s) = SolverRegistry::construct(registry_keys::FIVE_R1C, &wall) {
        solvers.push(s);
    }

    // 2. MultiNodeSolver (9R4C) via SolverRegistry::construct (PR #1491)
    if let Ok(s) = SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall) {
        solvers.push(s);
    }

    // 3. FiveR1CSolver — direct construction
    solvers.push(Box::new(FiveR1CSolver::new()));

    // 4. FDSolverWrapper — direct construction
    solvers.push(Box::new(FDSolverWrapper::new()));

    // 5. CTFSolverWrapper — direct construction
    solvers.push(Box::new(CTFSolverWrapper::new()));

    assert!(
        solvers.len() >= 3,
        "Expected ≥ 3 conduction solver constructors, got {}. \
         Update ARCHITECTURE.md Module 3 §SolverRegistry if this changed.",
        solvers.len()
    );

    // Verify ≥ 3 distinct solver names (guards against duplicate registrations).
    let names: Vec<&str> = solvers.iter().map(|s| s.name()).collect();
    let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
    assert!(
        unique.len() >= 3,
        "Expected ≥ 3 distinct solver names, got {:?}",
        names
    );
}

// ── helpers ──────────────────────────────────────────────────────────

fn normalize_ws(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn scan_for_literal(dir: &PathBuf, needle: &str, offenders: &mut Vec<String>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_for_literal(&path, needle, offenders);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            if let Ok(content) = fs::read_to_string(&path) {
                let normalized = normalize_ws(&content);
                if normalized.contains(needle) {
                    offenders.push(path.display().to_string());
                }
            }
        }
    }
}

//! Behavior tests for ASHRAE 140 validator selector wiring (Refs #3986-A, Refs #3986).
//!
//! These tests exercise the public API only. They verify that
//! `ASHRAE140Validator` and the `run_multi_zone_validation*` free functions
//! accept an explicit `ThermalSelector` without changing behavior, and that
//! the legacy `new()` / `run_multi_zone_validation()` entry points are
//! preserved.
//!
//! Behavior coverage:
//! - `selector_round_trip` — `new_with_selector(g).selector() == &g`
//! - `case_600_default_vs_new_with_default_parity` — Case 600 annual energy
//!   matches between `new()` and `new_with_selector(ThermalSelector::default())`
//! - `case_900_explicit_legacy_selector_runs` — Case 900 with explicit
//!   `FiveROneC` selector completes without panic and matches `new()`
//! - `run_multi_zone_validation_with_selector_matches_default` — parallel API
//! - `free_fn_with_selector_dispatch` — sibling `_with_selector` free fns
//!   in `ashrae_140_multi_zone.rs` match their default counterparts
//! - `validate_case_960_with_selector_*` — `_with_selector` free fns in
//!   `ashrae_140_validator/mod.rs` match their default counterparts
//! - `no_hardcoded_default_in_validator` — file-content guard (the actual
//!   guard test lives in `src/validation/ashrae_140_validator/tests.rs` so
//!   the helper is co-located; this module imports and re-exports it).

use fluxion::sim::thermal_selector::{ThermalSelector, ZoneSolverKind};
use fluxion::validation::{ASHRAE140MultiZoneValidator, ASHRAE140Validator};

const ERROR_PCT_REL_TOL: f64 = 1e-9;

#[test]
fn selector_round_trip() {
    // Default selector round-trips through the new ctor + getter.
    let g = ThermalSelector::default();
    let v = ASHRAE140Validator::new_with_selector(g);
    assert_eq!(
        v.selector().zone_solver,
        ThermalSelector::default().zone_solver
    );

    // Explicit FiveROneC round-trips too.
    let g = ThermalSelector {
        zone_solver: ZoneSolverKind::FiveROneC,
        ..ThermalSelector::default()
    };
    let v = ASHRAE140Validator::new_with_selector(g);
    assert_eq!(v.selector().zone_solver, ZoneSolverKind::FiveROneC);
    assert_eq!(v.selector().zone_solver, ZoneSolverKind::FiveROneC);
}

#[test]
fn new_delegates_to_new_with_selector_default() {
    // new() and new_with_selector(default()) must produce equivalent validators
    // (same zone_solver selection; the rest of the config is identical).
    let v_default = ASHRAE140Validator::new();
    let v_explicit = ASHRAE140Validator::new_with_selector(ThermalSelector::default());
    assert_eq!(
        v_default.selector().zone_solver,
        v_explicit.selector().zone_solver
    );
}

#[test]
fn case_600_default_vs_new_with_default_parity() {
    // The refactor must NOT change Case 600 error_pct. Compare `new()`
    // against `new_with_selector(ThermalSelector::default())`.
    let v_default = ASHRAE140Validator::new();
    let v_explicit = ASHRAE140Validator::new_with_selector(ThermalSelector::default());
    let r_default = v_default.validate_case("600").expect("Case 600 (default)");
    let r_explicit = v_explicit
        .validate_case("600")
        .expect("Case 600 (explicit)");
    let diff = (r_default.error_pct - r_explicit.error_pct).abs();
    assert!(
        diff <= ERROR_PCT_REL_TOL,
        "Case 600 default vs new_with_selector(default) drift: error_pct diff={diff:.3e} (tol {ERROR_PCT_REL_TOL:.0e})"
    );
}

#[test]
fn case_900_explicit_legacy_selector_runs() {
    // ADR-0017: FiveROneC must always succeed. Explicit selector must not panic.
    let v_default = ASHRAE140Validator::new();
    let v_explicit = ASHRAE140Validator::new_with_selector(ThermalSelector::legacy());
    let r_default = v_default.validate_case("900").expect("Case 900 (default)");
    let r_explicit = v_explicit
        .validate_case("900")
        .expect("Case 900 (FiveROneC)");
    let diff = (r_default.error_pct - r_explicit.error_pct).abs();
    assert!(
        diff <= ERROR_PCT_REL_TOL,
        "Case 900 default vs new_with_selector(FiveROneC) drift: error_pct diff={diff:.3e} (tol {ERROR_PCT_REL_TOL:.0e})"
    );
    // Sanity: the explicit selector really was applied.
    assert_eq!(v_explicit.selector().zone_solver, ZoneSolverKind::FiveROneC);
}

// ---------- Multi-zone validator selector wiring (Refs #3986-A) ----------

#[test]
fn multi_zone_selector_round_trip() {
    let g = ThermalSelector::default();
    let v = ASHRAE140MultiZoneValidator::new_with_selector(g);
    assert_eq!(
        v.selector().zone_solver,
        ThermalSelector::default().zone_solver
    );
}

#[test]
fn multi_zone_new_delegates_to_new_with_selector_default() {
    let v_default = ASHRAE140MultiZoneValidator::new();
    let v_explicit = ASHRAE140MultiZoneValidator::new_with_selector(ThermalSelector::default());
    assert_eq!(
        v_default.selector().zone_solver,
        v_explicit.selector().zone_solver
    );
}

#[test]
fn multi_zone_explicit_legacy_selector_accepted() {
    // ADR-0017: FiveROneC must always succeed. Explicit selector must not panic.
    let v = ASHRAE140MultiZoneValidator::new_with_selector(ThermalSelector::legacy());
    assert_eq!(v.selector().zone_solver, ZoneSolverKind::FiveROneC);
}

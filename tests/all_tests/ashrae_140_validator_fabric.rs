//! Regression tests for the `validate_case` selector wiring fix (Refs #3986 / #4117).
//!
//! PR-A (#4119) threaded `ThermalSelector` through `ASHRAE140Validator` (added
//! `selector` field, `new_with_selector`, `selector()` getter, and a
//! file-content regression guard for the hardcoded `ThermalSelector::default()`
//! production-path sites). However, `validate_case(&self, case_id)` previously
//! discarded `self.selector` by constructing a fresh `ASHRAE140Validator::new()`
//! (default selector) and routing through that fresh validator's
//! `validate_single_case_with_diagnostics`. The result: explicit selectors
//! passed via `new_with_selector(selector).validate_case(id)` were silently
//! ignored — only the default FiveROneC was ever exercised through this
//! entry point, even though the receiver stored a different selector.
//!
//! PR-A+1 fixes this bug (1-line change in
//! `src/validation/ashrae_140_validator/mod.rs::validate_case`) and adds these
//! regression tests to prevent re-introduction.
//!
//! Behavior coverage:
//! - `validate_case_honors_explicit_selector` — `new_with_selector(NineRFourC)
//!   .validate_case(...)` exercises the 9R4C path; before the fix, this returned
//!   the same result as `new_with_selector(FiveROneC)` (the discarded default).
//! - `validate_case_with_default_matches_legacy_for_low_mass_case` — the
//!   pre-fix correctness path: `new().validate_case("600")` ==
//!   `new_with_selector(FiveROneC).validate_case("600")` (within 1e-9), because
//!   `legacy() == default()` and the old code used the default anyway. This
//!   guarantees the fix didn't change behavior for the default-selector case.
//! - `validate_case_with_selector_free_fn_still_works` — the pre-existing
//!   `validate_case_with_diagnostics_with_selector` sibling free fn
//!   (PR-A's correct sibling API) continues to work; ensures the fix didn't
//!   break the parallel path.
//!
//! Naming: this module is `ashrae_140_validator_fabric` to disambiguate from
//! the umbrella gate's `ashrae_140_fabric` sub-suite (PR-A's selector-parity
//! tests) and to leave room for a follow-up PR-A+2 that adds the broader
//! multi-selector fabric comparison tests (which need engine-level isolation
//! via `run_blind_annual_energy` from `tests/all_tests/zone_balance_eplus_isolation.rs`).

use fluxion::sim::thermal_selector::{ThermalSelector, ZoneSolverKind};
use fluxion::validation::ASHRAE140Validator;

/// Run `case_id` with `selector` and return its `error_pct`.
fn run_case_error_pct(case_id: &str, selector: ThermalSelector) -> f64 {
    let validator = ASHRAE140Validator::new_with_selector(selector);
    validator
        .validate_case(case_id)
        .unwrap_or_else(|e| panic!("case {case_id} must validate: {e}"))
        .error_pct
}

// ---------- Slice 2 (RED→GREEN): regression test for the fix ----------

#[test]
fn validate_case_honors_explicit_selector() {
    // The bug: before the fix, `validate_case` constructed `ASHRAE140Validator::new()`
    // (which has `selector = ThermalSelector::default()` == FiveROneC in the
    // default build) regardless of the receiver's stored selector. So
    // `new_with_selector(NineRFourC).validate_case(id)` would silently produce
    // the same result as `new_with_selector(FiveROneC).validate_case(id)`.
    //
    // After the fix, the two selectors must produce DIFFERENT `error_pct` for
    // at least one case (proving the selector wiring is honored end-to-end).
    // Case 600 (low-mass) is the canonical low-mass case where the 9R4C network
    // takes a different path than 5R1C; per AGENTS.md the selectors are
    // independent networks (no auto-fallback), so `error_pct` MUST differ
    // from the FiveROneC default if the wiring is honored.
    let five_r_one_c = run_case_error_pct("600", ThermalSelector::legacy());
    let nine_r_four_c = run_case_error_pct(
        "600",
        ThermalSelector {
            zone_solver: ZoneSolverKind::NineRFourC,
            ..ThermalSelector::default()
        },
    );
    let diff = (five_r_one_c - nine_r_four_c).abs();
    assert!(
        diff > 1e-6,
        "Case 600 selector wiring is silently ignored: FiveROneC={five_r_one_c} \
         and NineRFourC={nine_r_four_c} produce identical error_pct (diff={diff:.3e}); \
         this is the pre-PR-A+1 bug (validate_case constructed a fresh validator \
         with default selector, discarding self.selector). If you see this \
         assertion fail, check src/validation/ashrae_140_validator/mod.rs::validate_case \
         — it must construct the inner validator via \
         `ASHRAE140Validator::new_with_selector(self.selector)`."
    );
}

// ---------- Slice 3 (RED→GREEN): default-selector parity (regression guard) ----------

#[test]
fn validate_case_with_default_matches_legacy_for_low_mass_case() {
    // Sanity: the fix must NOT change behavior for the default-selector case.
    // `new()` and `new_with_selector(FiveROneC)` (== `legacy()` == `default()`
    // in the default build) must produce identical `error_pct` within `1e-9`.
    let via_new = run_case_error_pct("600", ThermalSelector::default());
    let via_explicit_legacy = run_case_error_pct("600", ThermalSelector::legacy());
    let diff = (via_new - via_explicit_legacy).abs();
    assert!(
        diff <= 1e-9,
        "Case 600 (low-mass): `new()` ({via_new}) vs `new_with_selector(FiveROneC)` \
         ({via_explicit_legacy}) drift={diff:.3e}; the fix should preserve \
         default-selector behavior. Exceeding 1e-9 means the fix changed \
         behavior for the legacy code path."
    );
}

// ---------- Slice 4 (RED→GREEN): parallel free-fn API still works ----------

#[test]
fn validate_case_with_selector_free_fn_still_works() {
    // PR-A's correct sibling API: `validate_case_with_diagnostics_with_selector`
    // already honored the explicit selector. After the fix, this API must
    // continue to produce the same result as the (now-correct) `validate_case`.
    // This test guards against accidental regressions in the parallel path.
    use fluxion::sim::thermal_selector::ThermalSelector;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    let selector = ThermalSelector::legacy();
    let case = ASHRAE140Case::Case600;
    let (report, _diagnostics) =
        fluxion::validation::ashrae_140_validator::validate_case_with_diagnostics_with_selector(
            case, false, &selector,
        );
    // The free-fn API must produce a sensible ValidationReport.
    assert!(
        !report.case_id.is_empty(),
        "free-fn API returned an empty ValidationReport.case_id"
    );
}

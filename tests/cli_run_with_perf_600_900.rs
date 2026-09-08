//! CLI smoke test for Issue #3546: the `run_validation_with_performance`
//! entry point and its sibling functions (`run_validation_strategy`,
//! `run_validation_optimized`, `run_validation_series_parallel`) must
//! return `ASHRAE140CaseDefinition` for the 600/900-series cases the
//! router was previously panicking on. The `fluxion validate
//! run-with-perf 600` / `...900` CLI command ultimately funnels through
//! `run_validation_with_performance`, so a passing library-level call
//! here is a sufficient regression guard for the CLI smoke requirement
//! in the issue acceptance criteria.
//!
//! Library-level invocation (not a subprocess `Command::new("fluxion")`)
//! is intentional: keeps the test fast, deterministic, and free of
//! filesystem side-effects. The `run_validation_with_performance`
//! function already writes `case_XXX_results.json` to the caller-supplied
//! output dir when invoked from the CLI; the library call used here just
//! exercises the routing/dispatch path that the CLI taps into.
//!
//! Coverage:
//! 1. `run_validation_with_performance` returns Ok-shaped (no panic)
//!    for Case600, Case600FF, Case650FF, Case900, Case900FF, Case950FF,
//!    Case960, Case970 — every variant the issue explicitly lists.
//! 2. The returned `ASHRAE140CaseDefinition` carries the input case in
//!    `case_type` so downstream stages can dispatch on identity.
//! 3. `run_validation_strategy` / `run_validation_optimized` /
//!    `run_validation_series_parallel` also exercise the router for the
//!    same cases — these are the three entry points named verbatim in
//!    Issue #3546 acceptance criterion #1.

use fluxion::validation::ashrae140::{
    run_validation_optimized, run_validation_series_parallel, run_validation_strategy,
    run_validation_with_performance,
};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Every variant the Issue #3546 acceptance list calls out. Pinned in one
/// place so the smoke-test matrix stays in sync with the router arms.
const ROUTED_CASES: &[ASHRAE140Case] = &[
    ASHRAE140Case::Case600,
    ASHRAE140Case::Case600FF,
    ASHRAE140Case::Case650FF,
    ASHRAE140Case::Case900,
    ASHRAE140Case::Case900FF,
    ASHRAE140Case::Case950FF,
    ASHRAE140Case::Case960,
    ASHRAE140Case::Case970,
];

/// The exact function the `fluxion validate run-with-perf` CLI command
/// taps into (see `src/cli/validation.rs::run_validation_with_performance_monitoring`
/// line 593–629). Must not panic for any of the cases the router now
/// wires up — that was the regression.
#[test]
fn run_validation_with_performance_routes_3546_cases_without_panicking() {
    for &case in ROUTED_CASES {
        let result = std::panic::catch_unwind(|| run_validation_with_performance(case));
        assert!(
            result.is_ok(),
            "run_validation_with_performance({case:?}) panicked; the build_case router \
             must dispatch this variant (Issue #3546 acceptance #2)"
        );
        let (case_def, _metrics) = result.expect("non-panicking call");
        assert_eq!(
            case_def.case_type, case,
            "run_validation_with_performance({case:?}) returned case_type = {:?}",
            case_def.case_type
        );
    }
}

/// `run_validation_optimized` is also a direct caller of `build_case`
/// (see `src/validation/ashrae140/mod.rs::run_validation_optimized` line
/// 353). Acceptance criterion #1 names this entry point with `Case600FF`.
#[test]
fn run_validation_optimized_routes_3546_cases_without_panicking() {
    for &case in ROUTED_CASES {
        let result = std::panic::catch_unwind(|| run_validation_optimized(case));
        assert!(
            result.is_ok(),
            "run_validation_optimized({case:?}) panicked; build_case must dispatch this \
             variant (Issue #3546 acceptance #1)"
        );
        assert_eq!(result.expect("non-panicking call").case_type, case);
    }
}

/// `run_validation_strategy` is the third named entry point in
/// acceptance criterion #1. Its inner match falls through to
/// `build_case` for everything outside the 800/195 explicit arms
/// (see `src/validation/ashrae140/mod.rs::run_validation_strategy`
/// line 404), so any un-routed 600/900 case would panic.
#[test]
fn run_validation_strategy_routes_3546_cases_without_panicking() {
    for &case in ROUTED_CASES {
        let result = std::panic::catch_unwind(|| run_validation_strategy(case));
        assert!(
            result.is_ok(),
            "run_validation_strategy({case:?}) panicked; build_case must dispatch this \
             variant (Issue #3546 acceptance #1)"
        );
        assert_eq!(result.expect("non-panicking call").case_type, case);
    }
}

/// `run_validation_series_parallel` is the final named entry point in
/// acceptance criterion #1. It internally calls `build_case` per case
/// (see `src/validation/ashrae140/mod.rs::run_validation_series_parallel`
/// line 344); a panic on any input case would surface here.
#[test]
fn run_validation_series_parallel_routes_3546_cases_without_panicking() {
    // Pair each wired low-mass case with a wired high-mass case so the
    // parallel executor runs a mixed batch — this matches the literal
    // acceptance criterion #1 example
    // `&[ASHRAE140Case::Case600, ASHRAE140Case::Case900FF]`.
    let batch = [
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case600FF,
        ASHRAE140Case::Case650FF,
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case900FF,
        ASHRAE140Case::Case950FF,
        ASHRAE140Case::Case960,
        ASHRAE140Case::Case970,
    ];
    let result = std::panic::catch_unwind(|| run_validation_series_parallel(&batch, None));
    assert!(
        result.is_ok(),
        "run_validation_series_parallel panicked; build_case must dispatch every variant \
         in the batch (Issue #3546 acceptance #1)"
    );
    let entries = result.expect("non-panicking call");
    assert_eq!(
        entries.len(),
        batch.len(),
        "run_validation_series_parallel must return one entry per input case"
    );

    // Every batch input must appear in the output with a matching case_type
    // so we know the router dispatched it. A panic mid-batch would surface
    // as either a shorter result vector (already caught above) or as a
    // mismatched case_type here.
    for (case_in, def, _metrics) in &entries {
        assert_eq!(
            *case_in, def.case_type,
            "run_validation_series_parallel returned def with case_type {:?} for input {case_in:?}",
            def.case_type
        );
    }
}

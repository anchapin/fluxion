//! ASHRAE 140 Case 600-series (low-mass) factory methods.
//!
//! Issue #3555: extracted from `src/validation/ashrae_140_cases.rs::CaseBuilder`
//! to shrink the legacy monolith and align the in-tree case definitions with
//! the `crate::validation::ashrae140::cases` tree (which already houses
//! `series_195` and `series_800`). Each function here returns the same
//! `CaseSpec` value the corresponding `CaseBuilder` factory did before —
//! callers do not need to change.
//!
//! Issue #3546: added [`build_case`] thin shim so the `build_case` router in
//! `crate::validation::ashrae140::cases::mod` can dispatch the variants this
//! module owns to the `CaseSpec`-returning factories below. The shim does
//! NOT alter any case-definition logic — it only selects which factory runs
//! and bridges the `CaseSpec` to the legacy `ASHRAE140CaseDefinition` surface.

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Thin routing shim for the 600-series cases `build_case` knows how to
/// dispatch (Issue #3546). Only the variants explicitly required by the
/// issue are wired here — the rest fall through to the catch-all panic in
/// `crate::validation::ashrae140::cases::build_case` (and remain there on
/// purpose; they have no `run_validation_*` callers today).
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    let spec = match case {
        ASHRAE140Case::Case600 => case_600_baseline(),
        ASHRAE140Case::Case600FF => case_600ff(),
        ASHRAE140Case::Case650FF => case_650ff(),
        _ => panic!("Invalid case for series 600: {:?}", case),
    };
    super::spec_to_definition(case, spec)
}

/// Case 600 — low-mass baseline (8 m × 6 m × 2.7 m, 12 m² south double-clear
// Issue #4157 (round 52): the factory bodies moved to
// `fluxion_core::ashrae_case_series::series_600`; re-exported here so
// `crate::validation::ashrae140::cases::series_600::case_*` paths resolve unchanged.
pub use fluxion_core::ashrae_case_series::series_600::*;

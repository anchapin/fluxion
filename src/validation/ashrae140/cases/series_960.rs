//! ASHRAE 140 Case 960 (sunspace) factory method.
//!
//! Issue #3555: extracted from `src/validation/ashrae_140_cases.rs::CaseBuilder`
//! to shrink the legacy monolith.
//!
//! Issue #3546: added [`build_case`] thin shim so the `build_case` router in
//! `crate::validation::ashrae140::cases::mod` can dispatch Case960 to the
//! `CaseSpec`-returning factory below. The shim does NOT alter any
//! case-definition logic — it only bridges the `CaseSpec` to the legacy
//! `ASHRAE140CaseDefinition` surface.

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Thin routing shim for Case 960 (Issue #3546).
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    let spec = match case {
        ASHRAE140Case::Case960 => case_960_sunspace(),
        _ => panic!("Invalid case for series 960: {:?}", case),
    };
    super::spec_to_definition(case, spec)
}

/// Case 960 — Sunspace (2-zone building: back-zone + sunspace).
///
/// ```text
///   ┌──────────────┬───────────┐
///   │              │ Sunspace  │
///   │  Back-zone   │           │
///   │   8m × 6m    │ 8m × 2m   │
///   │   (south 12m²│  (south   │
///   │    window)   │   6m²)    │
///   │              │           │
///   └──────────────┴───────────┘
///   ↑ common wall (8m × 2.7m = 21.6 m², 200 mm concrete)
// Issue #4157 (round 52): the factory bodies moved to
// `fluxion_core::ashrae_case_series::series_960`; re-exported here so
// `crate::validation::ashrae140::cases::series_960::case_*` paths resolve unchanged.
pub use fluxion_core::ashrae_case_series::series_960::*;

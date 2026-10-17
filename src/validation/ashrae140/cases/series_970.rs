//! ASHRAE 140 Case 970 (5-zone multi-zone cross-coupling) factory method.
//!
//! Issue #3555: extracted from `src/validation/ashrae_140_cases.rs::CaseBuilder`
//! to shrink the legacy monolith.
//!
//! Issue #3546: added [`build_case`] thin shim so the `build_case` router in
//! `crate::validation::ashrae140::cases::mod` can dispatch Case970 to the
//! `CaseSpec`-returning factory below. The shim does NOT alter any
//! case-definition logic — it only bridges the `CaseSpec` to the legacy
//! `ASHRAE140CaseDefinition` surface.

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Thin routing shim for Case 970 (Issue #3546).
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    let spec = match case {
        ASHRAE140Case::Case970 => case_970_five_zone_cross_coupling(),
        _ => panic!("Invalid case for series 970: {:?}", case),
    };
    super::spec_to_definition(case, spec)
}

/// Case 970 — 5-zone multi-zone cross-coupling (ASHRAE 140-2017 §B6.7).
///
/// 8 m × 6 m × 2.7 m high-mass concrete building divided into 5 zones
/// by interior partitions (Issue #1446):
///
/// ```text
///   ┌──────────┬─────┐
///   │          │  Z1 │  ← north
///   │          ├─────┤
///   │   Z0     │  Z2 │
///   │ (west)   ├─────┤
///   │  4m×6m   │  Z3 │
///   │          ├─────┤
///   │          │  Z4 │  ← south
///   └──────────┴─────┘
///        ↑ 4m ↑
/// ```
///
/// Total conditioned floor area: 24 + 4 × 6 = 48 m² (8 m × 6 m).
/// Common walls between zone 0 and zones 1–4 each have area
/// 1.5 m × 2.7 m = 4.05 m²; common walls between adjacent east-strip
/// zones each have area 4 m × 2.7 m = 10.8 m². All interior partitions
/// use the same 200 mm concrete wall as Case 960.
///
/// The 12 m² south-window total from Case 600 is distributed across the
/// five zones: 6 m² on zone 0 (west half) and 1.5 m² on each of the
/// four east-strip zones. All five zones are conditioned (20 °C /
/// 27 °C) so the MultiZoneAirflowNetwork 5×5 conductance matrix is
/// exercised in both directions on every timestep.
///
/// Reference values: ASHRAE 140-2023 Annex B8-3 inter-program envelope,
// Issue #4157 (round 52): the factory bodies moved to
// `fluxion_core::ashrae_case_series::series_970`; re-exported here so
// `crate::validation::ashrae140::cases::series_970::case_*` paths resolve unchanged.
pub use fluxion_core::ashrae_case_series::series_970::*;

//! Issue #3546: regression coverage for the `build_case` router in
//! [`super`] (i.e. `crate::validation::ashrae140::cases::build_case`).
//!
//! Background: the `#3555` burn-down extracted the
//! `series_600/900/960/970.rs` factory modules but left the catch-all
//! router in `mod.rs` un-updated — so any call into `build_case` for
//! Case600/600FF/650FF/900/900FF/950FF/960/970 panicked with
//! "Case {} not implemented in this module".
//!
//! These tests pin:
//! 1. Every variant that the router dispatches today returns an
//!    `ASHRAE140CaseDefinition` without panicking.
//! 2. The legacy catch-all panic still fires for variants that have NOT
//!    been wired (Case500..510, Case699, Warehouse, the unwired 610/620/
//!    630/640/650/910/920/930/940/950 variants) — preserving the
//!    "un-routed variants panic loudly" contract required by the issue
//!    scope guard.
//! 3. The wired cases all populate `case_type` to the input variant so
//!    downstream consumers (`run_validation_*`) can rely on identity.

use super::build_case;
use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Every `ASHRAE140Case` variant that `build_case` is expected to dispatch
/// successfully after the Issue #3546 wiring. Kept as a single source of
/// truth so the routing matrix below stays in sync with the match arms in
/// `super::build_case`.
const WIRED_CASES: &[ASHRAE140Case] = &[
    // 800-series — series_800::build_case (pre-existing).
    ASHRAE140Case::Case800,
    ASHRAE140Case::Case801,
    ASHRAE140Case::Case802,
    ASHRAE140Case::Case803,
    ASHRAE140Case::Case804,
    ASHRAE140Case::Case805,
    ASHRAE140Case::Case806,
    ASHRAE140Case::Case807,
    ASHRAE140Case::Case808,
    ASHRAE140Case::Case809,
    ASHRAE140Case::Case810,
    // 195/200/250/300/350/400/470 + Office/Retail/School — series_195::build_case
    // (pre-existing).
    ASHRAE140Case::Case195,
    ASHRAE140Case::Case195HighMass,
    ASHRAE140Case::Case195NoLoads,
    ASHRAE140Case::Case195NoSolar,
    ASHRAE140Case::Case195ThermalBridge,
    ASHRAE140Case::Case195SHGC03,
    ASHRAE140Case::Case195SHGC06,
    ASHRAE140Case::Case195SHGC09,
    ASHRAE140Case::Case195Albedo01,
    ASHRAE140Case::Case195Albedo05,
    ASHRAE140Case::Case195Albedo09,
    ASHRAE140Case::Case196,
    ASHRAE140Case::Case197,
    ASHRAE140Case::Case198,
    ASHRAE140Case::Case200,
    ASHRAE140Case::Case250,
    ASHRAE140Case::Case300,
    ASHRAE140Case::Case350,
    ASHRAE140Case::Case400,
    ASHRAE140Case::Case470,
    ASHRAE140Case::Office,
    ASHRAE140Case::Retail,
    ASHRAE140Case::School,
    // 600-series — wired by Issue #3546 (series_600::build_case).
    ASHRAE140Case::Case600,
    ASHRAE140Case::Case600FF,
    ASHRAE140Case::Case650FF,
    // 900-series — wired by Issue #3546 (series_900::build_case).
    ASHRAE140Case::Case900,
    ASHRAE140Case::Case900FF,
    ASHRAE140Case::Case950FF,
    // 960/970 — wired by Issue #3546 (series_960/series_970::build_case).
    ASHRAE140Case::Case960,
    ASHRAE140Case::Case970,
];

/// Variants that intentionally still hit the catch-all panic in
/// `super::build_case` — wiring them is outside Issue #3546 scope. This list
/// documents the contract; if the scope expands, move them into
/// `WIRED_CASES` and add the matching arm in `super::build_case`.
const PANICKING_CASES: &[ASHRAE140Case] = &[
    // 500..510 + 699 still route to the Case600-baseline fallback in
    // `super::build_spec`, but `super::build_case` does NOT dispatch them
    // — they keep the legacy panic contract.
    ASHRAE140Case::Case500,
    ASHRAE140Case::Case501,
    ASHRAE140Case::Case502,
    ASHRAE140Case::Case503,
    ASHRAE140Case::Case504,
    ASHRAE140Case::Case505,
    ASHRAE140Case::Case506,
    ASHRAE140Case::Case507,
    ASHRAE140Case::Case508,
    ASHRAE140Case::Case509,
    ASHRAE140Case::Case510,
    ASHRAE140Case::Case699,
    // Warehouse is in the legacy `CaseBuilder` factory surface but NOT in
    // `build_case` — keeps the panic contract.
    ASHRAE140Case::Warehouse,
    // 600/900-series variants NOT explicitly required by Issue #3546.
    // They have no `run_validation_*` callers today; wiring them is
    // intentionally out of scope.
    ASHRAE140Case::Case610,
    ASHRAE140Case::Case620,
    ASHRAE140Case::Case630,
    ASHRAE140Case::Case640,
    ASHRAE140Case::Case650,
    ASHRAE140Case::Case910,
    ASHRAE140Case::Case920,
    ASHRAE140Case::Case930,
    ASHRAE140Case::Case940,
    ASHRAE140Case::Case950,
];

/// Exhaustively iterate every wired variant and assert `build_case` returns
/// a value without panicking. A regression in the router (e.g. someone
/// removing an arm) surfaces as a test failure here with `cargo test
/// --lib validation::ashrae140::cases -- --nocapture`.
#[test]
fn build_case_routes_every_wired_variant_without_panicking() {
    for &case in WIRED_CASES {
        let result = std::panic::catch_unwind(|| build_case(case));
        assert!(
            result.is_ok(),
            "build_case({case:?}) panicked; check super::build_case routing match"
        );
    }
}

/// Assert the legacy panic fallthrough still fires for variants outside
/// the wired set. This is the contract Issue #3546 explicitly preserves —
/// "Preserve the existing panic fallthrough for truly unknown cases
/// (don't remove the panic — that's the contract)."
#[test]
fn build_case_panics_for_unwired_variants() {
    for &case in PANICKING_CASES {
        let result = std::panic::catch_unwind(|| build_case(case));
        assert!(
            result.is_err(),
            "build_case({case:?}) returned Ok but the contract requires a panic for \
             un-wired variants. Either wire this case in super::build_case OR \
             extend PANICKING_CASES if this is the new behaviour."
        );
    }
}

/// The specific cases called out by Issue #3546 acceptance criterion #1
/// must each return an `ASHRAE140CaseDefinition` and populate `case_type`
/// to the input variant.
#[test]
fn build_case_returns_correct_identity_for_3546_cases() {
    let pinned_cases = [
        (ASHRAE140Case::Case600, ASHRAE140Case::Case600),
        (ASHRAE140Case::Case600FF, ASHRAE140Case::Case600FF),
        (ASHRAE140Case::Case650FF, ASHRAE140Case::Case650FF),
        (ASHRAE140Case::Case900, ASHRAE140Case::Case900),
        (ASHRAE140Case::Case900FF, ASHRAE140Case::Case900FF),
        (ASHRAE140Case::Case950FF, ASHRAE140Case::Case950FF),
        (ASHRAE140Case::Case960, ASHRAE140Case::Case960),
        (ASHRAE140Case::Case970, ASHRAE140Case::Case970),
    ];
    for (input, expected) in pinned_cases {
        let def: ASHRAE140CaseDefinition = build_case(input);
        assert_eq!(
            def.case_type, expected,
            "build_case({input:?}) returned case_type = {:?}, expected {expected:?}",
            def.case_type
        );
    }
}

/// `build_case(Case600)` must produce a definition whose building fields
/// come from the spec — `floor_area` must be the 8 m × 6 m = 48 m² ASHRAE
/// 140 Case 600 baseline, `construction_type` must be the legacy
/// `Lightweight` (from the spec's `LowMass`), and `infiltration_rate`
/// must equal the 0.5 ACH baseline.
#[test]
fn build_case_case600_carries_spec_derived_building_fields() {
    let def = build_case(ASHRAE140Case::Case600);
    assert_eq!(def.case_type, ASHRAE140Case::Case600);
    let building = &def.building;
    assert!(
        (building.floor_area - 48.0).abs() < 1e-6,
        "Case 600 floor_area should be 8 × 6 = 48 m², got {}",
        building.floor_area
    );
    assert!(
        (building.infiltration_rate - 0.5).abs() < 1e-6,
        "Case 600 infiltration_rate should be 0.5 ACH, got {}",
        building.infiltration_rate
    );
    assert_eq!(
        building.construction_type,
        crate::validation::ashrae140::ConstructionType::Lightweight,
        "spec.construction_type::LowMass should map to legacy Lightweight"
    );
}

/// `build_case(Case900)` must produce a definition with high-mass
/// construction and the matching 8 × 6 = 48 m² floor area (the
/// high-mass baseline is the same footprint as Case 600).
#[test]
fn build_case_case900_carries_high_mass_derived_fields() {
    let def = build_case(ASHRAE140Case::Case900);
    assert_eq!(def.case_type, ASHRAE140Case::Case900);
    assert_eq!(
        def.building.construction_type,
        crate::validation::ashrae140::ConstructionType::HighMass,
        "spec.construction_type::HighMass should map to legacy HighMass"
    );
    assert!(
        (def.building.floor_area - 48.0).abs() < 1e-6,
        "Case 900 floor_area should be 8 × 6 = 48 m², got {}",
        def.building.floor_area
    );
}

/// `build_case(Case970)` should preserve the 5-zone floor area:
/// zone 0 is 4 × 6 = 24 m² and zones 1–4 are each 4 × 1.5 = 6 m², total
/// 48 m². The `spec_to_definition` helper sums `geometry[].floor_area()`
/// across all zones, so a 5-zone case must still resolve to 48 m² (the
/// ASHRAE 140 Case 970 documented total).
#[test]
fn build_case_case970_sums_all_zones_into_floor_area() {
    let def = build_case(ASHRAE140Case::Case970);
    assert_eq!(def.case_type, ASHRAE140Case::Case970);
    assert!(
        (def.building.floor_area - 48.0).abs() < 1e-6,
        "Case 970 total floor area should be 48 m², got {}",
        def.building.floor_area
    );
}

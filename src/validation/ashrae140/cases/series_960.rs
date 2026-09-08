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

use crate::sim::construction::Assemblies;
use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;
use crate::validation::ashrae_140_cases::{
    CaseBuilder, CaseSpec, HvacSchedule, InternalLoads, Orientation,
};

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
/// ```
pub fn case_960_sunspace() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("960".to_string())
        .with_description("Sunspace - 2-zone building (back-zone + sunspace)".to_string())
        // Zone 0: Back-zone (8m x 6m x 2.7m)
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_zone_window(0, 12.0, Orientation::South) // Back-zone south window
        .with_hvac_setpoints(20.0, 27.0)
        // Zone 1: Sunspace (8m x 2m x 2.7m)
        .add_zone(8.0, 2.0, 2.7)
        .with_zone_hvac(1, HvacSchedule::free_floating())
        .with_zone_window(1, 6.0, Orientation::South) // Sunspace south window
        // Common Wall (8m x 2.7m = 21.6 m2)
        .with_common_wall(0, 1, 21.6, Assemblies::concrete_wall(0.200))
        .with_infiltration(0.5)
        .with_door_geometry(2.0, 1.5) // Door opening: height=2.0m, area=1.5m² (Plan 04-04)
        .with_num_zones(2)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 960 should validate")
}

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
    CaseBuilder, CaseSpec, ConstructionSpec, HvacSchedule, InternalLoads, Orientation, ZoneOrigin,
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
        // Global (sun-zone) construction: heavyweight, exactly Case 900
        // (§5.2.2.2.7 preamble: "the sun zone shall be of heavyweight
        // construction" and §5.2.2.2.7.3: sun-zone properties same as Case
        // 900). The back zone overrides this per zone below.
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        // Round 24 (§LIMIT-14): the back zone is LIGHTWEIGHT — "the geometric
        // and thermal properties of the back zone shall be exactly the same as
        // for Case 600" (§5.2.2.2.7.1a), i.e. the Case 600 low-mass
        // assemblies. The legacy single global construction made both zones
        // high-mass, contradicting the spec.
        .with_zone_constructions(vec![
            ConstructionSpec::new(
                Assemblies::low_mass_wall(),
                Assemblies::low_mass_roof(),
                Assemblies::insulated_floor(),
            ),
            ConstructionSpec::new(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            ),
        ])
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6)) // back zone only; sun zone = 0 W (§5.2.2.2.7.3g)
        // §5.2.2.2.7.1a: the back zone's south wall/windows are REPLACED by
        // the common wall — no back-zone window. (Verified inert as-built:
        // with zone origins the back-zone south facade is the shared
        // partition, which carries no exterior window; round-24 E3 experiment
        // bit-identical.)
        .with_hvac_setpoints(20.0, 27.0)
        // Zone 1: Sunspace (8m x 2m x 2.7m)
        .add_zone(8.0, 2.0, 2.7)
        .with_zone_hvac(1, HvacSchedule::free_floating())
        .with_zone_window(1, 12.0, Orientation::South) // Sun-zone south glazing: TWO 3x2 m windows = 12 m2 (140-2017 §5.2.2.2.7.3c, Fig. 5-96; round 22 §LIMIT-14)
        // Common Wall (8m x 2.7m = 21.6 m2)
        .with_common_wall(0, 1, 21.6, Assemblies::concrete_wall(0.200))
        .with_infiltration(0.5) // No door: §5.2.2.2.7 specifies no door anywhere in Case 960 and no
        // inter-zonal air exchange (§5.2.2.2.7.6). (The retired
        // with_door_geometry entry was verified inert as-built — round-24 E6
        // experiment bit-identical.)
        .with_num_zones(2)
        // Zone plan origins (round 21, §LIMIT-23b): the 21.6 m² (8 m × 2.7 m)
        // common wall forces the sunspace south of the back-zone, which also
        // puts both south windows on outdoor-exposed facades. The sunspace's
        // north wall is correctly derived as the shared partition.
        .with_zone_origins(vec![
            ZoneOrigin::new(0.0, 2.0), // zone 0: back-zone x 0-8, y 2-8
            ZoneOrigin::new(0.0, 0.0), // zone 1: sunspace  x 0-8, y 0-2
        ])
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 960 should validate")
}

//! ASHRAE 140 Case 970-series `CaseSpec` factories.
//!
//! Issue #4157 (round 52): factory bodies moved verbatim from
//! `fluxion::validation::ashrae140::cases::series_970` into `fluxion-core` so the
//! main-crate `CaseBuilder` impl no longer interleaves case-factory methods
//! delegating back into the validation crate. The main-crate module re-exports
//! these and keeps the `build_case` router shim plus engine-driving tests.

use crate::ashrae_case_builder::CaseBuilder;
use crate::ashrae_cases::*;
use crate::construction::Assemblies;

/// `tests/reference_data/zone_balance/case_970_energy_reference.csv`.
pub fn case_970_five_zone_cross_coupling() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("970".to_string())
        .with_description(
            "Case 970 - 5-zone multi-zone cross-coupling (ASHRAE 140-2017 §B6.7)".to_string(),
        )
        // Zone 0: West core (4 m × 6 m × 2.7 m = 24 m² floor area).
        .with_dimensions(4.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_zone_window(0, 6.0, Orientation::South) // West core south window
        .with_hvac_setpoints(20.0, 27.0)
        // Zones 1-4: East strip (4 m × 1.5 m × 2.7 m = 6 m² each).
        .add_zone(4.0, 1.5, 2.7)
        .with_zone_window(1, 1.5, Orientation::South)
        .add_zone(4.0, 1.5, 2.7)
        .with_zone_window(2, 1.5, Orientation::South)
        .add_zone(4.0, 1.5, 2.7)
        .with_zone_window(3, 1.5, Orientation::South)
        .add_zone(4.0, 1.5, 2.7)
        .with_zone_window(4, 1.5, Orientation::South)
        // Common walls: zone 0 ↔ each east-strip zone (1.5 m × 2.7 m = 4.05 m²).
        .with_common_wall(0, 1, 4.05, Assemblies::concrete_wall(0.200))
        .with_common_wall(0, 2, 4.05, Assemblies::concrete_wall(0.200))
        .with_common_wall(0, 3, 4.05, Assemblies::concrete_wall(0.200))
        .with_common_wall(0, 4, 4.05, Assemblies::concrete_wall(0.200))
        // Common walls between adjacent east-strip zones (4 m × 2.7 m = 10.8 m²).
        .with_common_wall(1, 2, 10.8, Assemblies::concrete_wall(0.200))
        .with_common_wall(2, 3, 10.8, Assemblies::concrete_wall(0.200))
        .with_common_wall(3, 4, 10.8, Assemblies::concrete_wall(0.200))
        .with_infiltration(0.414)
        .with_num_zones(5)
        // Zone plan origins (round 21, §LIMIT-23b): zone 0 west core at the
        // origin, east strip x ∈ [4, 8]; zones stacked along the 6 m depth
        // with zone 1 at the north edge and zone 4 at the south edge. These
        // let the engine derive the real 75.6 m² perimeter envelope from
        // adjacency instead of treating every partition face as exterior.
        .with_zone_origins(vec![
            ZoneOrigin::new(0.0, 0.0), // zone 0: west core   x 0-4, y 0-6
            ZoneOrigin::new(4.0, 4.5), // zone 1: north strip x 4-8, y 4.5-6
            ZoneOrigin::new(4.0, 3.0), // zone 2: x 4-8, y 3-4.5
            ZoneOrigin::new(4.0, 1.5), // zone 3: x 4-8, y 1.5-3
            ZoneOrigin::new(4.0, 0.0), // zone 4: south strip x 4-8, y 0-1.5
        ])
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 970 should validate")
}

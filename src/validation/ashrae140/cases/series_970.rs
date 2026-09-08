//! ASHRAE 140 Case 970 (5-zone multi-zone cross-coupling) factory method.
//!
//! Issue #3555: extracted from `src/validation/ashrae_140_cases.rs::CaseBuilder`
//! to shrink the legacy monolith.

use crate::sim::construction::Assemblies;
use crate::validation::ashrae_140_cases::{CaseBuilder, CaseSpec, InternalLoads, Orientation};

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
        .with_infiltration(0.5)
        .with_num_zones(5)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 970 should validate")
}

//! Plant-loop equipment models.
//!
//! Contains the core traits, fluid-property tables, and concrete equipment
//! models that participate in a plant loop:
//!
//! * [`plant_component`] — `PlantComponent` trait and shared types.
//! * [`fluid_properties`] — temperature-dependent water / glycol properties.
//! * [`cooling_tower`] — `CoolingTowerSingleSpeed` (ASHRAE HoF Ch. 40).
//! * [`pump`] — `PumpConstantSpeed` and `PumpVariableSpeed` (affinity laws).
//! * [`plant_loop`] — `PlantLoop` sequential-iterative solver.

pub mod cooling_tower;
pub mod fluid_properties;
pub mod plant_component;
pub mod plant_loop;
pub mod pump;

// Re-export the most-used types at the module level for ergonomic access.
pub use cooling_tower::CoolingTowerSingleSpeed;
pub use fluid_properties::{water_cp, water_density, WATER_CP_J_PER_KG_K, WATER_DENSITY_KG_PER_M3};
pub use plant_component::{FluidState, PlantComponent, PlantComponentResult, PlantMode};
pub use plant_loop::{check_energy_balance, PlantLoop, PlantLoopResult};
pub use pump::{Pump, PumpConstantSpeed, PumpVariableSpeed};

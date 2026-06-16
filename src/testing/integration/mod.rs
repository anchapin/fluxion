//! Integration testing framework for Fluxion
//!
//! Provides reusable fixtures, wiring validation, and E2E test infrastructure.

pub mod doe_buildings;

pub mod fixtures;

pub mod wiring;

pub mod scenarios;

// Re-export public types for convenience
pub use doe_buildings::{run_annual_simulation, DoeBuildingConfig, DoeBuildingType, MemoryStats};

pub use fixtures::{BuildingScenario, HvacType};

pub use wiring::WiringTracer;

pub use scenarios::{
    heat_pump_scenario, high_mass_scenario, low_mass_scenario, multi_zone_scenario, vav_scenario,
};

// DOE Commercial Reference Buildings
pub mod doe_reference;

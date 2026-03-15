//! Pre-built test scenarios for common integration test cases
//!
//! Provides ready-to-use scenarios for low-mass, high-mass, and multi-zone buildings.

use super::fixtures::{BuildingScenario, HvacType};

/// Create a low-mass building scenario (ASHRAE 140 Case 600-like)
pub fn low_mass_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(1)
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
}

/// Create a high-mass building scenario (ASHRAE 140 Case 900-like)
pub fn high_mass_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(1)
        .with_window_u_value(2.0)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
}

/// Create a multi-zone building scenario (ASHRAE 140 Case 960-like)
pub fn multi_zone_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(3)
        .with_window_u_value(2.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
}

/// Create a scenario with VAV HVAC
pub fn vav_scenario() -> BuildingScenario {
    low_mass_scenario().with_hvac(HvacType::VAV)
}

/// Create a scenario with Heat Pump HVAC
pub fn heat_pump_scenario() -> BuildingScenario {
    low_mass_scenario().with_hvac(HvacType::HeatPump)
}

// TODO: Add heating_setpoint and cooling_setpoint setters to BuildingScenario
impl BuildingScenario {
    fn with_heating_setpoint(mut self, sp: f64) -> Self {
        self.heating_setpoint = Some(sp);
        self
    }

    fn with_cooling_setpoint(mut self, sp: f64) -> Self {
        self.cooling_setpoint = Some(sp);
        self
    }
}

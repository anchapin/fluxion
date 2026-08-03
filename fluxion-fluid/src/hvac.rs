//! HVAC operational state types for thermal-electrical coupling.
//!
//! This module defines the core HVAC types used across fluxion for representing
//! heating, ventilation, and air conditioning operational states.

use uuid::Uuid;

/// Operating mode of the HVAC system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvacMode {
    Off = 0,
    Heating = 1,
    Cooling = 2,
}

/// HVAC operational state for a single building.
///
/// This represents the thermal demand and operating conditions of an HVAC system
/// serving a specific building, which can be converted to electrical load via COP.
#[derive(Debug, Clone)]
pub struct HvacState {
    /// Unique identifier of the building this HVAC serves
    pub building_id: Uuid,
    /// Thermal power demand (W) — positive for cooling, negative for heating
    pub thermal_power_w: f64,
    /// Indoor air temperature setpoint (°C)
    pub setpoint_c: f64,
    /// Ambient outdoor air temperature (°C)
    pub ambient_temperature_c: f64,
    /// Operating mode: 0=off, 1=heating, 2=cooling
    pub mode: HvacMode,
}

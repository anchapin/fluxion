//! Unit type aliases for the physics module.
//!
//! This module provides strongly-typed wrappers around `uom` quantities
//! to enforce unit safety in the physics layer. Using these types makes
//! it impossible to accidentally add incompatible units (e.g., W/K + W/m²K).
//!
//! # Example
//!
//! ```ignore
//! use crate::physics::units::*;
//!
//! fn calculate_heat_loss(flux: HeatFlux) { ... }
//!
//! // This would NOT compile - type mismatch:
//! let conductance: ThermalConductance = ...;
//! calculate_heat_loss(conductance); // Error: expected HeatFlux, found ThermalConductance
//! ```

use uom::si::f64::{
    Energy as UomEnergy, HeatFluxDensity as UomHeatFluxDensity, HeatTransfer as UomHeatTransfer,
    Power as UomPower, ThermalConductance as UomThermalConductance,
    ThermodynamicTemperature as UomThermodynamicTemperature, Time as UomTime,
};

/// Time duration for a simulation timestep [s]
pub type Time = UomTime;

/// Thermodynamic temperature [°C] (relative to 0°C, NOT absolute)
/// Note: Building physics commonly uses Celsius for temperature differences.
/// For absolute temperatures requiring Kelvin conversion, use `AbsoluteTemperature`.
pub type Temperature = UomThermodynamicTemperature;

/// Heat transfer coefficient [W/(m²·K)]
pub type HeatTransferCoefficient = UomHeatTransfer;

/// Heat flux (positive = heat flowing into zone) [W/m²]
pub type HeatFlux = UomHeatFluxDensity;

/// Thermal conductance [W/K]
pub type ThermalConductance = UomThermalConductance;

/// Power [W]
pub type Power = UomPower;

/// Energy [J]
pub type Energy = UomEnergy;

// =============================================================================
// Conversion helpers — use uom native API directly in solvers:
//   • Create: `Quantity::new::<Unit>(value)`
//   • Extract: `quantity.get::<Unit>()`
// =============================================================================

/// Trait for types that can be created from a raw f64 value in their native unit.
/// Used to sidestep the orphan rule (can't impl inherent methods on external types).
pub trait FromF64 {
    fn from_value(val: f64) -> Self;
}

/// Trait for types that can be converted to a raw f64 value in their native unit.
pub trait ToF64 {
    fn to_value(&self) -> f64;
}

impl FromF64 for Time {
    fn from_value(val: f64) -> Self {
        Self::new::<uom::si::time::second>(val)
    }
}

impl ToF64 for Time {
    fn to_value(&self) -> f64 {
        self.get::<uom::si::time::second>()
    }
}

impl FromF64 for Temperature {
    fn from_value(val: f64) -> Self {
        Self::new::<uom::si::thermodynamic_temperature::degree_celsius>(val)
    }
}

impl ToF64 for Temperature {
    fn to_value(&self) -> f64 {
        self.get::<uom::si::thermodynamic_temperature::degree_celsius>()
    }
}

impl FromF64 for HeatTransferCoefficient {
    fn from_value(val: f64) -> Self {
        Self::new::<uom::si::heat_transfer::watt_per_square_meter_kelvin>(val)
    }
}

impl ToF64 for HeatTransferCoefficient {
    fn to_value(&self) -> f64 {
        self.get::<uom::si::heat_transfer::watt_per_square_meter_kelvin>()
    }
}

impl FromF64 for HeatFlux {
    fn from_value(val: f64) -> Self {
        Self::new::<uom::si::heat_flux_density::watt_per_square_meter>(val)
    }
}

impl ToF64 for HeatFlux {
    fn to_value(&self) -> f64 {
        self.get::<uom::si::heat_flux_density::watt_per_square_meter>()
    }
}

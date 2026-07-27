//! `fluxion-grid` — Grid-coupled thermal modeling for Fluxion.
//!
//! Provides voltage feedback models for heat pump COP adjustment and
//! thermal-electrical coupling components.

mod error;

pub use error::GridModelError;

/// Voltage-per-unit of nominal frequency (Hz/Hz) type alias for clarity.
pub type VoltagePu = f64;
/// Frequency-per-unit of nominal frequency type alias.
pub type FrequencyPu = f64;

mod thermal_electrical_coupler;
pub use thermal_electrical_coupler::ThermalElectricalCoupler;

mod heat_pump_voltage_model;
pub use heat_pump_voltage_model::HeatPumpVoltageModel;

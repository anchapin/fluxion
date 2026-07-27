//! Plant component trait for loop-level simulation.
//!
//! All equipment that participates in a plant loop — chillers, boilers,
//! cooling towers, pumps, and heat exchangers — implements this trait so
//! the loop solver can treat them uniformly.

use serde::{Deserialize, Serialize};

/// Operating mode for plant-side equipment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlantMode {
    /// Heat-addition equipment (boiler).
    Heating,
    /// Heat-rejection equipment (chiller, cooling tower).
    Cooling,
    /// Off / standby.
    Off,
}

/// Fluid state at a connection point.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FluidState {
    /// Bulk fluid temperature (°C).
    pub temperature: f64,
    /// Volumetric flow rate (m³/s).
    pub flow_rate: f64,
}

/// Result of a single plant component evaluation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlantComponentResult {
    /// Fluid state leaving the component (outlet).
    pub outlet: FluidState,
    /// Electrical power consumed by the component (W).  Zero for passive
    /// devices such as pipe segments.
    pub electrical_power_w: f64,
    /// Heat transfer rate into the fluid (positive = heat added, negative = heat
    /// rejected) (W).
    pub heat_transfer_w: f64,
}

/// Trait for components that participate in a plant loop.
///
/// The loop solver calls [`PlantComponent::evaluate`] with the inlet
/// conditions and the loop-side setpoint, and the component returns outlet
/// conditions plus power consumption.
pub trait PlantComponent: Send + Sync {
    /// Human-readable identifier (e.g. "Chiller-1").
    fn id(&self) -> &str;

    /// Evaluate the component at the given inlet conditions.
    ///
    /// # Arguments
    /// * `inlet` — fluid state entering the component
    /// * `outdoor_temp` — ambient dry-bulb temperature (°C), needed by
    ///   cooling towers and chillers with air-cooled condensers
    /// * `dt` — timestep length (seconds)
    ///
    /// The implementation must compute the outlet temperature from its
    /// thermodynamic model and return it together with power draw.
    fn evaluate(&self, inlet: FluidState, outdoor_temp: f64, dt: f64) -> PlantComponentResult;
}

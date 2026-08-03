//! Bridge module for integrating with the main `fluxion` crate's `ThermalModelTrait`.
//!
//! This module is only available when the `fluxion` feature flag is enabled.
//! It provides a wrapper that allows `ThermalElectricalCoupler` to hold
//! `Arc<dyn ThermalModelTrait>` for joint thermal-electrical convergence.
//!
//! # Example
//!
//! ```ignore
//! use fluxion_grid::thermal_electrical_coupler::ThermalElectricalCoupler;
//! use fluxion_grid::fluxion_bridge::ThermalModelTraitBridge;
//!
//! let coupler = ThermalElectricalCoupler::new(3.0);
//! let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);
//! ```

use std::sync::Arc;

#[cfg(feature = "fluxion")]
use fluxion::ThermalModelTrait;

/// Bridge that holds both a `ThermalElectricalCoupler` and an `Arc<dyn ThermalModelTrait>`.
///
/// This enables joint thermal-electrical convergence where the grid-side coupler
/// can query the full thermal solver state rather than relying on scalar HVAC values.
#[cfg(feature = "fluxion")]
pub struct ThermalModelTraitBridge {
    coupler: crate::ThermalElectricalCoupler,
    thermal_model: Arc<dyn ThermalModelTrait>,
}

#[cfg(feature = "fluxion")]
impl ThermalModelTraitBridge {
    /// Create a new bridge with a coupler and thermal model.
    pub fn new(
        coupler: crate::ThermalElectricalCoupler,
        thermal_model: Arc<dyn ThermalModelTrait>,
    ) -> Self {
        Self {
            coupler,
            thermal_model,
        }
    }

    /// Get a reference to the thermal model.
    pub fn thermal_model(&self) -> &Arc<dyn ThermalModelTrait> {
        &self.thermal_model
    }

    /// Get a reference to the coupler.
    pub fn coupler(&self) -> &crate::ThermalElectricalCoupler {
        &self.coupler
    }

    /// Get a mutable reference to the coupler.
    pub fn coupler_mut(&mut self) -> &mut crate::ThermalElectricalCoupler {
        &mut self.coupler
    }

    /// Get HVAC power demand from the thermal model and convert to electrical load.
    ///
    /// This queries `hvac_power_demand` from `ThermalModelTrait` and passes
    /// the result through the `ThermalElectricalCoupler` COP conversion.
    pub fn hvac_power_to_electrical(&self, timestep: usize, outdoor_temp: f64) -> f64 {
        let thermal_power = self.thermal_model.hvac_power_demand(timestep, outdoor_temp);
        self.coupler.thermal_to_electrical_simple(thermal_power)
    }
}

/// Tag type indicating the fluxion feature is not enabled.
#[cfg(not(feature = "fluxion"))]
pub struct ThermalModelTraitBridge;

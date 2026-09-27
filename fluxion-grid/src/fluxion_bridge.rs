//! Bridge module for joint thermal-electrical convergence.
//!
//! This module is only available when the `fluxion-integration` feature flag is enabled.
//! It provides a wrapper that allows `ThermalElectricalCoupler` to hold
//! `Arc<dyn ThermalModelQuery>` for joint thermal-electrical convergence.
//!
//! # Dependency direction (Issue #4005)
//!
//! The bridge deliberately defines its own minimal [`ThermalModelQuery`] trait instead of
//! depending back on the main `fluxion` crate's `ThermalModelTrait`. Cargo rejects
//! optional-optional package cycles, so the dependency direction is strictly
//! `fluxion` (feature `grid`) → `fluxion-grid` — never the reverse. A caller that has a
//! thermal model implements `ThermalModelQuery` for it (the trait only needs the HVAC
//! power-demand query the coupler consumes) and hands it to the bridge.
//!
//! # Example
//!
//! ```ignore
//! use fluxion_grid::thermal_electrical_coupler::ThermalElectricalCoupler;
//! use fluxion_grid::fluxion_bridge::{ThermalModelQuery, ThermalModelTraitBridge};
//!
//! let coupler = ThermalElectricalCoupler::new(3.0);
//! let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);
//! ```

use std::sync::Arc;

/// Minimal thermal-model query surface the grid-side bridge needs.
///
/// Implement this for a thermal model to enable joint thermal-electrical convergence
/// via [`ThermalModelTraitBridge`]. It is intentionally narrow — only the HVAC
/// power-demand query the coupler consumes — so `fluxion-grid` never needs a
/// dependency edge back into the main `fluxion` crate (Issue #4005).
pub trait ThermalModelQuery: Send + Sync {
    /// HVAC thermal power demand (W) at a timestep for a given outdoor temperature.
    fn hvac_power_demand(&self, timestep: usize, outdoor_temp_c: f64) -> f64;
}

/// Bridge that holds both a `ThermalElectricalCoupler` and an `Arc<dyn ThermalModelQuery>`.
///
/// This enables joint thermal-electrical convergence where the grid-side coupler
/// can query the full thermal solver state rather than relying on scalar HVAC values.
#[cfg(feature = "fluxion-integration")]
pub struct ThermalModelTraitBridge {
    coupler: crate::ThermalElectricalCoupler,
    thermal_model: Arc<dyn ThermalModelQuery>,
}

#[cfg(feature = "fluxion-integration")]
impl ThermalModelTraitBridge {
    /// Create a new bridge with a coupler and thermal model.
    pub fn new(
        coupler: crate::ThermalElectricalCoupler,
        thermal_model: Arc<dyn ThermalModelQuery>,
    ) -> Self {
        Self {
            coupler,
            thermal_model,
        }
    }

    /// Get a reference to the thermal model.
    pub fn thermal_model(&self) -> &Arc<dyn ThermalModelQuery> {
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
    /// This queries `hvac_power_demand` from `ThermalModelQuery` and passes
    /// the result through the `ThermalElectricalCoupler` COP conversion.
    pub fn hvac_power_to_electrical(&self, timestep: usize, outdoor_temp: f64) -> f64 {
        let thermal_power = self.thermal_model.hvac_power_demand(timestep, outdoor_temp);
        self.coupler.thermal_to_electrical_simple(thermal_power)
    }
}

/// Tag type indicating the fluxion-integration feature is not enabled.
#[cfg(not(feature = "fluxion-integration"))]
pub struct ThermalModelTraitBridge;

#[cfg(feature = "fluxion-integration")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThermalElectricalCoupler;

    /// Mock thermal model implementing ThermalModelQuery for testing.
    #[cfg(feature = "fluxion-integration")]
    struct MockThermalModelForTest {
        fixed_hvac_power: f64,
    }

    #[cfg(feature = "fluxion-integration")]
    impl MockThermalModelForTest {
        fn new(hvac_power: f64) -> Self {
            Self {
                fixed_hvac_power: hvac_power,
            }
        }
    }

    #[cfg(feature = "fluxion-integration")]
    impl ThermalModelQuery for MockThermalModelForTest {
        fn hvac_power_demand(&self, _timestep: usize, _outdoor_temp_c: f64) -> f64 {
            self.fixed_hvac_power
        }
    }

    #[test]
    fn test_thermal_model_trait_bridge_creation() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(3000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model.clone());

        assert!((bridge.thermal_model().hvac_power_demand(0, 10.0) - 3000.0).abs() < 1e-9);
    }

    #[test]
    fn test_hvac_power_to_electrical_conversion() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(3000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);

        let electrical_power = bridge.hvac_power_to_electrical(0, 10.0);

        assert!((electrical_power - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_hvac_power_to_electrical_with_different_cop() {
        let coupler = ThermalElectricalCoupler::new(4.0);
        let mock_model = MockThermalModelForTest::new(4000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);

        let electrical_power = bridge.hvac_power_to_electrical(0, 10.0);

        assert!((electrical_power - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_coupler_reference() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(1500.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler.clone(), thermal_model);

        assert_eq!(bridge.coupler().cop, 3.0);
    }

    #[test]
    fn test_joint_convergence_with_mock_thermal() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(5000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);

        let electrical = bridge.hvac_power_to_electrical(0, 5.0);

        assert!(electrical > 0.0);
        assert!((electrical - 5000.0 / 3.0).abs() < 1e-6);
    }
}

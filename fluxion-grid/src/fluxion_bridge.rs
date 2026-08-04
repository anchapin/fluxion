//! Bridge module for integrating with the main `fluxion` crate's `ThermalModelTrait`.
//!
//! This module is only available when the `fluxion-integration` feature flag is enabled.
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

#[cfg(feature = "fluxion-integration")]
use fluxion::ThermalModelTrait;

/// Bridge that holds both a `ThermalElectricalCoupler` and an `Arc<dyn ThermalModelTrait>`.
///
/// This enables joint thermal-electrical convergence where the grid-side coupler
/// can query the full thermal solver state rather than relying on scalar HVAC values.
#[cfg(feature = "fluxion-integration")]
pub struct ThermalModelTraitBridge {
    coupler: crate::ThermalElectricalCoupler,
    thermal_model: Arc<dyn ThermalModelTrait>,
}

#[cfg(feature = "fluxion-integration")]
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

/// Tag type indicating the fluxion-integration feature is not enabled.
#[cfg(not(feature = "fluxion-integration"))]
pub struct ThermalModelTraitBridge;

#[cfg(feature = "fluxion-integration")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThermalElectricalCoupler;

    /// Mock thermal model implementing ThermalModelTrait for testing.
    #[cfg(feature = "fluxion-integration")]
    struct MockThermalModelForTest {
        num_zones: usize,
        temperatures: Vec<f64>,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        zone_area: f64,
        fixed_hvac_power: f64,
    }

    #[cfg(feature = "fluxion-integration")]
    impl MockThermalModelForTest {
        fn new(num_zones: usize, hvac_power: f64) -> Self {
            Self {
                num_zones,
                temperatures: vec![20.0; num_zones],
                heating_setpoint: 20.0,
                cooling_setpoint: 24.0,
                zone_area: 100.0,
                fixed_hvac_power: hvac_power,
            }
        }
    }

    #[cfg(feature = "fluxion-integration")]
    impl ThermalModelTrait for MockThermalModelForTest {
        fn num_zones(&self) -> usize {
            self.num_zones
        }

        fn get_temperatures(&self) -> Vec<f64> {
            self.temperatures.clone()
        }

        fn set_temperatures(&mut self, temperatures: &[f64]) {
            self.temperatures = temperatures.to_vec();
        }

        fn mode(&self) -> fluxion::sim::thermal_model::ThermalModelMode {
            fluxion::sim::thermal_model::ThermalModelMode::Physics
        }

        fn set_mode(&mut self, _mode: fluxion::sim::thermal_model::ThermalModelMode) {}

        fn solve_timesteps(
            &mut self,
            _steps: usize,
            _surrogates: &fluxion::ai::surrogate::SurrogateManager,
            _use_surrogates: bool,
        ) -> f64 {
            0.0
        }

        fn apply_parameters(&mut self, _params: &[f64]) {}

        fn zone_area(&self) -> f64 {
            self.zone_area
        }

        fn heating_setpoint(&self) -> f64 {
            self.heating_setpoint
        }

        fn cooling_setpoint(&self) -> f64 {
            self.cooling_setpoint
        }

        fn hvac_power_demand(&self, _timestep: usize, _outdoor_temp: f64) -> f64 {
            self.fixed_hvac_power
        }

        fn is_valid(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_thermal_model_trait_bridge_creation() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(3, 3000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model.clone());

        assert!(bridge.thermal_model().num_zones() == 3);
    }

    #[test]
    fn test_hvac_power_to_electrical_conversion() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(1, 3000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);

        let electrical_power = bridge.hvac_power_to_electrical(0, 10.0);

        assert!((electrical_power - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_hvac_power_to_electrical_with_different_cop() {
        let coupler = ThermalElectricalCoupler::new(4.0);
        let mock_model = MockThermalModelForTest::new(1, 4000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);

        let electrical_power = bridge.hvac_power_to_electrical(0, 10.0);

        assert!((electrical_power - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_coupler_reference() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(2, 1500.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler.clone(), thermal_model);

        assert_eq!(bridge.coupler().cop, 3.0);
    }

    #[test]
    fn test_joint_convergence_with_mock_thermal() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let mock_model = MockThermalModelForTest::new(1, 5000.0);
        let thermal_model = Arc::new(mock_model);
        let bridge = ThermalModelTraitBridge::new(coupler, thermal_model);

        let electrical = bridge.hvac_power_to_electrical(0, 5.0);

        assert!(electrical > 0.0);
        assert!((electrical - 5000.0 / 3.0).abs() < 1e-6);
    }
}

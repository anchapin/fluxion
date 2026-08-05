//! Mock Thermal Model - Fixed-return implementation for testing.
//!
//! This module provides a `MockThermalModel` that implements `ThermalModelTrait`
//! with configurable fixed values. It enables testing of downstream consumers
//! (engine, HVAC controller, parametric studies) without depending on the
//! physics engine.
//!
//! # Design Rationale (Issue #943)
//!
//! The zone solver (`Engine`) takes `Box<dyn ThermalModelTrait>`. Before this
//! module, every test that needed a `ThermalModelTrait` had to construct a
//! `PhysicsThermalModel` with a real thermal network — pulling in conduction
//! solvers, weather data, and surrogate managers. `MockThermalModel` breaks
//! that dependency chain.

use crate::ai::surrogate::SurrogateManager;
use crate::sim::thermal_model::{
    compute_pmv_ppd_and_adaptive, ThermalModelMode, ThermalModelTrait, ZoneComfortMetrics,
};

/// A mock thermal model for testing that returns configurable fixed values.
///
/// Implements `ThermalModelTrait` so it can be used anywhere a real model is
/// expected (`Box<dyn ThermalModelTrait>`).
///
/// # Example
///
/// ```
/// use fluxion::sim::thermal_model_mock::MockThermalModel;
/// use fluxion::sim::thermal_model::{ThermalModelTrait, ThermalModelMode};
///
/// let mut model = MockThermalModel::new(2)
///     .with_heating_setpoint(20.0)
///     .with_cooling_setpoint(24.0);
///
/// assert_eq!(model.num_zones(), 2);
/// model.set_temperatures(&[21.0, 22.0]);
/// assert_eq!(model.get_temperatures(), vec![21.0, 22.0]);
/// assert_eq!(model.heating_setpoint(), 20.0);
/// assert_eq!(model.cooling_setpoint(), 24.0);
/// assert!(model.is_valid());
/// ```
pub struct MockThermalModel {
    /// Number of thermal zones.
    num_zones: usize,
    /// Current zone temperatures [°C].
    temperatures: Vec<f64>,
    /// Model execution mode.
    mode: ThermalModelMode,
    /// Fixed result returned by `solve_timesteps` [kWh/m²/year].
    fixed_solve_result: f64,
    /// Heating setpoint [°C].
    heating_setpoint: f64,
    /// Cooling setpoint [°C].
    cooling_setpoint: f64,
    /// Zone floor area [m²].
    zone_area: f64,
    /// Whether the model reports itself as valid.
    valid: bool,
    /// Last applied parameter vector (for test assertions).
    last_applied_params: Vec<f64>,
    /// Fixed HVAC power demand [W].
    fixed_hvac_power: f64,
}

impl MockThermalModel {
    /// Create a new mock model with the given number of zones.
    ///
    /// Defaults:
    /// - temperatures: 22.0 °C for all zones
    /// - mode: Physics
    /// - solve result: 100.0 kWh/m²/year
    /// - heating setpoint: 20.0 °C
    /// - cooling setpoint: 26.0 °C
    /// - zone area: 100.0 m²
    /// - valid: true
    /// - HVAC power: 0.0 W
    pub fn new(num_zones: usize) -> Self {
        Self {
            num_zones,
            temperatures: vec![22.0; num_zones],
            mode: ThermalModelMode::Physics,
            fixed_solve_result: 100.0,
            heating_setpoint: 20.0,
            cooling_setpoint: 26.0,
            zone_area: 100.0,
            valid: true,
            last_applied_params: Vec::new(),
            fixed_hvac_power: 0.0,
        }
    }

    /// Set the fixed return value for `solve_timesteps`.
    pub fn with_solve_result(mut self, result: f64) -> Self {
        self.fixed_solve_result = result;
        self
    }

    /// Set the heating setpoint [°C].
    pub fn with_heating_setpoint(mut self, sp: f64) -> Self {
        self.heating_setpoint = sp;
        self
    }

    /// Set the cooling setpoint [°C].
    pub fn with_cooling_setpoint(mut self, sp: f64) -> Self {
        self.cooling_setpoint = sp;
        self
    }

    /// Set the zone floor area [m²].
    pub fn with_zone_area(mut self, area: f64) -> Self {
        self.zone_area = area;
        self
    }

    /// Set the validity flag.
    pub fn with_valid(mut self, valid: bool) -> Self {
        self.valid = valid;
        self
    }

    /// Set the fixed HVAC power demand [W].
    pub fn with_hvac_power(mut self, power: f64) -> Self {
        self.fixed_hvac_power = power;
        self
    }

    /// Get the last parameters passed to `apply_parameters` (for test assertions).
    pub fn last_applied_params(&self) -> &[f64] {
        &self.last_applied_params
    }
}

impl ThermalModelTrait for MockThermalModel {
    fn num_zones(&self) -> usize {
        self.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        self.temperatures.clone()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        self.temperatures = temperatures.to_vec();
    }

    fn mode(&self) -> ThermalModelMode {
        self.mode
    }

    fn set_mode(&mut self, mode: ThermalModelMode) {
        self.mode = mode;
    }

    fn solve_timesteps(
        &mut self,
        _steps: usize,
        _surrogates: &SurrogateManager,
        _use_surrogates: bool,
    ) -> f64 {
        self.fixed_solve_result
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.last_applied_params = params.to_vec();
        // Apply setpoints if provided (matching real model convention)
        if params.len() >= 3 {
            self.heating_setpoint = params[1];
            self.cooling_setpoint = params[2];
        }
    }

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
        self.valid
    }

    fn get_comfort_metrics(&self) -> Vec<ZoneComfortMetrics> {
        self.temperatures
            .iter()
            .map(|&t| compute_pmv_ppd_and_adaptive(t, 0.5, 0.1, 1.0, 0.5))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::thermal_model::ThermalModelTrait;

    fn make_surrogate_manager() -> SurrogateManager {
        SurrogateManager::default()
    }

    #[test]
    fn test_mock_model_creation() {
        let model = MockThermalModel::new(3);
        assert_eq!(model.num_zones(), 3);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(model.is_valid());
    }

    #[test]
    fn test_mock_model_default_temperatures() {
        let model = MockThermalModel::new(2);
        assert_eq!(model.get_temperatures(), vec![22.0, 22.0]);
    }

    #[test]
    fn test_mock_model_set_temperatures() {
        let mut model = MockThermalModel::new(2);
        model.set_temperatures(&[18.0, 25.0]);
        assert_eq!(model.get_temperatures(), vec![18.0, 25.0]);
    }

    #[test]
    fn test_mock_model_set_mode() {
        let mut model = MockThermalModel::new(1);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        model.set_mode(ThermalModelMode::Surrogate);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        model.set_mode(ThermalModelMode::Hybrid);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_mock_model_solve_timesteps_returns_fixed() {
        let mut model = MockThermalModel::new(1).with_solve_result(42.0);
        let mgr = make_surrogate_manager();
        let result = model.solve_timesteps(8760, &mgr, false);
        assert!((result - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_mock_model_setpoints() {
        let model = MockThermalModel::new(1)
            .with_heating_setpoint(18.0)
            .with_cooling_setpoint(28.0);
        assert!((model.heating_setpoint() - 18.0).abs() < 1e-10);
        assert!((model.cooling_setpoint() - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_mock_model_zone_area() {
        let model = MockThermalModel::new(1).with_zone_area(250.0);
        assert!((model.zone_area() - 250.0).abs() < 1e-10);
    }

    #[test]
    fn test_mock_model_validity() {
        let valid_model = MockThermalModel::new(1).with_valid(true);
        assert!(valid_model.is_valid());

        let invalid_model = MockThermalModel::new(1).with_valid(false);
        assert!(!invalid_model.is_valid());
    }

    #[test]
    fn test_mock_model_apply_parameters() {
        let mut model = MockThermalModel::new(1);
        model.apply_parameters(&[1.5, 19.0, 25.0]);
        assert_eq!(model.last_applied_params(), &[1.5, 19.0, 25.0]);
        // Setpoints should be applied from params
        assert!((model.heating_setpoint() - 19.0).abs() < 1e-10);
        assert!((model.cooling_setpoint() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_mock_model_hvac_power() {
        let model = MockThermalModel::new(1).with_hvac_power(500.0);
        assert!((model.hvac_power_demand(0, 10.0) - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_mock_model_as_trait_object() {
        let model: Box<dyn ThermalModelTrait> = Box::new(
            MockThermalModel::new(2)
                .with_heating_setpoint(19.0)
                .with_cooling_setpoint(27.0),
        );

        assert_eq!(model.num_zones(), 2);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!((model.heating_setpoint() - 19.0).abs() < 1e-10);
        assert!((model.cooling_setpoint() - 27.0).abs() < 1e-10);
        assert!(model.is_valid());
    }

    #[test]
    fn test_trait_object_polymorphism_physics_and_mock() {
        // Verify that both PhysicsThermalModel and MockThermalModel can be
        // used interchangeably behind Box<dyn ThermalModelTrait>
        let physics: Box<dyn ThermalModelTrait> =
            Box::new(crate::sim::thermal_model::PhysicsThermalModel::new(1));
        let mock: Box<dyn ThermalModelTrait> = Box::new(MockThermalModel::new(1));

        // Both implement the same trait interface
        assert_eq!(physics.num_zones(), mock.num_zones());
        assert!(physics.is_valid());
        assert!(mock.is_valid());
    }

    #[test]
    fn test_builder_produces_same_trait_object() {
        // Verify ThermalModelBuilder and MockThermalModel both produce
        // Box<dyn ThermalModelTrait>
        let built = crate::sim::thermal_model::ThermalModelBuilder::new()
            .num_zones(1)
            .build();
        let mocked: Box<dyn ThermalModelTrait> = Box::new(MockThermalModel::new(1));

        assert_eq!(built.num_zones(), mocked.num_zones());
    }
}

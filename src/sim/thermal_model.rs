//! Thermal Model Trait - Modular architecture for swapping physics/surrogate models.
//!
//! This module defines the core trait interface for thermal modeling, allowing
//! easy swapping between traditional physics-based approaches and AI surrogate models.

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use std::error::Error;

/// Result type for thermal model operations
pub type ThermalModelResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Defines the mode of thermal model execution
#[derive(Clone, Debug, Copy, PartialEq, Eq, Default)]
pub enum ThermalModelMode {
    /// Physics-based thermal model using analytical calculations
    #[default]
    Physics,
    /// Surrogate-based thermal model using neural network inference
    Surrogate,
    /// Hybrid mode: some components use surrogates, others use physics
    Hybrid,
}

/// Core trait for thermal model implementations.
///
/// This trait defines the interface for building energy modeling, allowing
/// different implementations (physics-based, surrogate-based, or hybrid) to be
/// swapped at runtime.
///
/// # Design Philosophy
/// - Easy addition of new surrogate models (ONNX-based)
/// - Fallback from surrogate to physics-based when needed
/// - Hybrid mode where some components use surrogates, others use physics
pub trait ThermalModelTrait: Send + Sync {
    /// Get the number of thermal zones in the model
    fn num_zones(&self) -> usize;

    /// Get current zone temperatures
    fn get_temperatures(&self) -> Vec<f64>;

    /// Set zone temperatures
    fn set_temperatures(&mut self, temperatures: &[f64]);

    /// Get the model execution mode
    fn mode(&self) -> ThermalModelMode;

    /// Set the model execution mode
    fn set_mode(&mut self, mode: ThermalModelMode);

    /// Solve thermal model for specified timesteps.
    ///
    /// # Arguments
    /// * `steps` - Number of hourly timesteps (typically 8760 for 1 year)
    /// * `surrogates` - Reference to SurrogateManager for load predictions
    /// * `use_surrogates` - If true, use neural surrogates; if false, use analytical calculations
    ///
    /// # Returns
    /// Cumulative annual energy use intensity (EUI) in kWh/m²/year.
    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_surrogates: bool,
    ) -> f64;

    /// Apply parameters from an optimization gene vector.
    ///
    /// # Arguments
    /// * `params` - Parameter vector:
    ///   - `params[0]`: Window U-value (W/m²K, range: 0.5-3.0)
    ///   - `params[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `params[2]`: Cooling setpoint (°C, range: 22-32)
    fn apply_parameters(&mut self, params: &[f64]);

    /// Get zone floor area in m²
    fn zone_area(&self) -> f64;

    /// Get current heating setpoint (°C)
    fn heating_setpoint(&self) -> f64;

    /// Get current cooling setpoint (°C)
    fn cooling_setpoint(&self) -> f64;

    /// Calculate HVAC power demand based on current conditions.
    ///
    /// Returns heating power (positive) or cooling power (negative) in Watts.
    fn hvac_power_demand(&self, timestep: usize, _outdoor_temp: f64) -> f64;

    /// Check if the model is valid for simulation
    fn is_valid(&self) -> bool;
}

/// Physics-based thermal model implementation.
///
/// This is the default implementation using analytical 5R1C thermal network calculations.
pub struct PhysicsThermalModel {
    inner: crate::sim::engine::ThermalModel<VectorField>,
    mode: ThermalModelMode,
}

impl PhysicsThermalModel {
    /// Create a new physics-based thermal model
    pub fn new(num_zones: usize) -> Self {
        PhysicsThermalModel {
            inner: crate::sim::engine::ThermalModel::new(num_zones),
            mode: ThermalModelMode::Physics,
        }
    }

    /// Create from an ASHRAE 140 case specification
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        PhysicsThermalModel {
            inner: crate::sim::engine::ThermalModel::from_spec(spec),
            mode: ThermalModelMode::Physics,
        }
    }

    /// Get mutable reference to inner thermal model
    pub fn inner_mut(&mut self) -> &mut crate::sim::engine::ThermalModel<VectorField> {
        &mut self.inner
    }

    /// Get reference to inner thermal model
    pub fn inner(&self) -> &crate::sim::engine::ThermalModel<VectorField> {
        &self.inner
    }
}

impl ThermalModelTrait for PhysicsThermalModel {
    fn num_zones(&self) -> usize {
        self.inner.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        self.inner.temperatures = VectorField::new(temperatures.to_vec());
    }

    fn mode(&self) -> ThermalModelMode {
        self.mode
    }

    fn set_mode(&mut self, mode: ThermalModelMode) {
        self.mode = mode;
    }

    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_surrogates: bool,
    ) -> f64 {
        // Use the mode to determine whether to use surrogates
        let actual_use_surrogates = use_surrogates || self.mode == ThermalModelMode::Surrogate;
        self.inner
            .solve_timesteps(steps, surrogates, actual_use_surrogates)
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        self.inner.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        self.inner.cooling_setpoint
    }

    fn hvac_power_demand(&self, timestep: usize, _outdoor_temp: f64) -> f64 {
        // Simplified HVAC demand calculation
        let temps = self.inner.temperatures.as_ref();
        if temps.is_empty() {
            return 0.0;
        }
        let t = temps[0];
        let heating_sp = self.inner.heating_schedule.value(timestep % 24);
        let cooling_sp = self.inner.cooling_schedule.value(timestep % 24);

        if t < heating_sp {
            // Heating needed
            (heating_sp - t) * 100.0 // Simplified
        } else if t > cooling_sp {
            // Cooling needed
            -(t - cooling_sp) * 100.0
        } else {
            0.0 // In deadband
        }
    }

    fn is_valid(&self) -> bool {
        self.inner.num_zones > 0 && self.zone_area() > 0.0
    }
}

/// Surrogate-based thermal model implementation.
///
/// This implementation uses neural network surrogates for faster inference.
pub struct SurrogateThermalModel {
    inner: crate::sim::engine::ThermalModel<VectorField>,
    mode: ThermalModelMode,
    fallback_to_physics: bool,
}

impl SurrogateThermalModel {
    /// Create a new surrogate-based thermal model
    pub fn new(num_zones: usize) -> Self {
        SurrogateThermalModel {
            inner: crate::sim::engine::ThermalModel::new(num_zones),
            mode: ThermalModelMode::Surrogate,
            fallback_to_physics: true, // Default to fallback on surrogate failure
        }
    }

    /// Create from an ASHRAE 140 case specification
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        SurrogateThermalModel {
            inner: crate::sim::engine::ThermalModel::from_spec(spec),
            mode: ThermalModelMode::Surrogate,
            fallback_to_physics: true,
        }
    }

    /// Enable or disable fallback to physics-based model on surrogate failure
    pub fn with_fallback(mut self, fallback: bool) -> Self {
        self.fallback_to_physics = fallback;
        self
    }

    /// Get mutable reference to inner thermal model
    pub fn inner_mut(&mut self) -> &mut crate::sim::engine::ThermalModel<VectorField> {
        &mut self.inner
    }

    /// Get reference to inner thermal model
    pub fn inner(&self) -> &crate::sim::engine::ThermalModel<VectorField> {
        &self.inner
    }
}

impl ThermalModelTrait for SurrogateThermalModel {
    fn num_zones(&self) -> usize {
        self.inner.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        self.inner.temperatures = VectorField::new(temperatures.to_vec());
    }

    fn mode(&self) -> ThermalModelMode {
        self.mode
    }

    fn set_mode(&mut self, mode: ThermalModelMode) {
        self.mode = mode;
    }

    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        _use_surrogates: bool,
    ) -> f64 {
        // Always use surrogates for this model type
        self.inner.solve_timesteps(steps, surrogates, true)
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        self.inner.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        self.inner.cooling_setpoint
    }

    fn hvac_power_demand(&self, timestep: usize, _outdoor_temp: f64) -> f64 {
        let temps = self.inner.temperatures.as_ref();
        if temps.is_empty() {
            return 0.0;
        }
        let t = temps[0];
        let heating_sp = self.inner.heating_schedule.value(timestep % 24);
        let cooling_sp = self.inner.cooling_schedule.value(timestep % 24);

        if t < heating_sp {
            (heating_sp - t) * 100.0
        } else if t > cooling_sp {
            -(t - cooling_sp) * 100.0
        } else {
            0.0
        }
    }

    fn is_valid(&self) -> bool {
        self.inner.num_zones > 0 && self.zone_area() > 0.0
    }
}

/// Unified thermal model that can switch between physics and surrogate modes at runtime.
///
/// This is the main entry point for users who want to easily switch between
/// physics-based and surrogate-based thermal modeling.
pub struct UnifiedThermalModel {
    inner: crate::sim::engine::ThermalModel<VectorField>,
    mode: ThermalModelMode,
    use_surrogates: bool,
}

impl UnifiedThermalModel {
    /// Create a new unified thermal model with default physics mode
    pub fn new(num_zones: usize) -> Self {
        UnifiedThermalModel {
            inner: crate::sim::engine::ThermalModel::new(num_zones),
            mode: ThermalModelMode::Physics,
            use_surrogates: false,
        }
    }

    /// Create from an ASHRAE 140 case specification
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        UnifiedThermalModel {
            inner: crate::sim::engine::ThermalModel::from_spec(spec),
            mode: ThermalModelMode::Physics,
            use_surrogates: false,
        }
    }

    /// Switch to physics-based mode
    pub fn use_physics(&mut self) {
        self.mode = ThermalModelMode::Physics;
        self.use_surrogates = false;
    }

    /// Switch to surrogate-based mode
    pub fn use_surrogates(&mut self) {
        self.mode = ThermalModelMode::Surrogate;
        self.use_surrogates = true;
    }

    /// Switch to hybrid mode (some components surrogates, some physics)
    pub fn use_hybrid(&mut self) {
        self.mode = ThermalModelMode::Hybrid;
    }

    /// Check if currently using surrogates
    pub fn is_using_surrogates(&self) -> bool {
        self.use_surrogates
    }

    /// Get reference to inner thermal model
    pub fn inner(&self) -> &crate::sim::engine::ThermalModel<VectorField> {
        &self.inner
    }

    /// Get mutable reference to inner thermal model
    pub fn inner_mut(&mut self) -> &mut crate::sim::engine::ThermalModel<VectorField> {
        &mut self.inner
    }
}

impl ThermalModelTrait for UnifiedThermalModel {
    fn num_zones(&self) -> usize {
        self.inner.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        self.inner.temperatures = VectorField::new(temperatures.to_vec());
    }

    fn mode(&self) -> ThermalModelMode {
        self.mode
    }

    fn set_mode(&mut self, mode: ThermalModelMode) {
        self.mode = mode;
        self.use_surrogates = mode == ThermalModelMode::Surrogate;
    }

    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        _use_surrogates: bool,
    ) -> f64 {
        // Use the internal mode flag
        self.inner
            .solve_timesteps(steps, surrogates, self.use_surrogates)
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        self.inner.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        self.inner.cooling_setpoint
    }

    fn hvac_power_demand(&self, timestep: usize, _outdoor_temp: f64) -> f64 {
        let temps = self.inner.temperatures.as_ref();
        if temps.is_empty() {
            return 0.0;
        }
        let t = temps[0];
        let heating_sp = self.inner.heating_schedule.value(timestep % 24);
        let cooling_sp = self.inner.cooling_schedule.value(timestep % 24);

        if t < heating_sp {
            (heating_sp - t) * 100.0
        } else if t > cooling_sp {
            -(t - cooling_sp) * 100.0
        } else {
            0.0
        }
    }

    fn is_valid(&self) -> bool {
        self.inner.num_zones > 0 && self.zone_area() > 0.0
    }
}

/// Builder for creating thermal models with custom configurations
pub struct ThermalModelBuilder {
    num_zones: usize,
    mode: ThermalModelMode,
    use_surrogates: bool,
    fallback_to_physics: bool,
    spec: Option<crate::validation::ashrae_140_cases::CaseSpec>,
}

impl ThermalModelBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        ThermalModelBuilder {
            num_zones: 1,
            mode: ThermalModelMode::Physics,
            use_surrogates: false,
            fallback_to_physics: true,
            spec: None,
        }
    }

    /// Set number of thermal zones
    pub fn num_zones(mut self, num_zones: usize) -> Self {
        self.num_zones = num_zones;
        self
    }

    /// Set the execution mode
    pub fn mode(mut self, mode: ThermalModelMode) -> Self {
        self.mode = mode;
        self.use_surrogates = mode == ThermalModelMode::Surrogate;
        self
    }

    /// Enable or disable surrogate usage
    pub fn use_surrogates(mut self, use_surrogates: bool) -> Self {
        self.use_surrogates = use_surrogates;
        if use_surrogates {
            self.mode = ThermalModelMode::Surrogate;
        }
        self
    }

    /// Enable fallback to physics on surrogate failure
    pub fn fallback_to_physics(mut self, fallback: bool) -> Self {
        self.fallback_to_physics = fallback;
        self
    }

    /// Set ASHRAE 140 case specification
    pub fn with_case_spec(mut self, spec: crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        self.spec = Some(spec);
        self
    }

    /// Build the thermal model based on configuration
    pub fn build(self) -> Box<dyn ThermalModelTrait> {
        match self.mode {
            ThermalModelMode::Physics => {
                if let Some(spec) = self.spec {
                    Box::new(PhysicsThermalModel::from_spec(&spec))
                } else {
                    Box::new(PhysicsThermalModel::new(self.num_zones))
                }
            }
            ThermalModelMode::Surrogate => {
                if let Some(spec) = self.spec {
                    Box::new(
                        SurrogateThermalModel::from_spec(&spec)
                            .with_fallback(self.fallback_to_physics),
                    )
                } else {
                    Box::new(
                        SurrogateThermalModel::new(self.num_zones)
                            .with_fallback(self.fallback_to_physics),
                    )
                }
            }
            ThermalModelMode::Hybrid => {
                if let Some(spec) = self.spec {
                    Box::new(UnifiedThermalModel::from_spec(&spec))
                } else {
                    Box::new(UnifiedThermalModel::new(self.num_zones))
                }
            }
        }
    }

    /// Build a UnifiedThermalModel (allows runtime switching)
    pub fn build_unified(self) -> UnifiedThermalModel {
        let mut model = if let Some(spec) = self.spec {
            UnifiedThermalModel::from_spec(&spec)
        } else {
            UnifiedThermalModel::new(self.num_zones)
        };

        // Set the mode based on configuration
        model.set_mode(self.mode);
        model
    }
}

impl Default for ThermalModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_model_creation() {
        let model = PhysicsThermalModel::new(10);
        assert_eq!(model.num_zones(), 10);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(model.is_valid());
    }

    #[test]
    fn test_surrogate_model_creation() {
        let model = SurrogateThermalModel::new(5);
        assert_eq!(model.num_zones(), 5);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_valid());
    }

    #[test]
    fn test_unified_model_switching() {
        let mut model = UnifiedThermalModel::new(1);

        // Initially in physics mode
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());

        // Switch to surrogates
        model.use_surrogates();
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_using_surrogates());

        // Switch back to physics
        model.use_physics();
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());
    }

    #[test]
    fn test_builder_physics_mode() {
        let model = ThermalModelBuilder::new()
            .num_zones(5)
            .mode(ThermalModelMode::Physics)
            .build();

        assert_eq!(model.num_zones(), 5);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
    }

    #[test]
    fn test_builder_surrogate_mode() {
        let model = ThermalModelBuilder::new()
            .num_zones(3)
            .use_surrogates(true)
            .build();

        assert_eq!(model.num_zones(), 3);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_builder_default() {
        let model = ThermalModelBuilder::new().build();
        assert_eq!(model.num_zones(), 1);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
    }

    #[test]
    fn test_builder_build_unified() {
        let model = ThermalModelBuilder::new()
            .num_zones(10)
            .mode(ThermalModelMode::Hybrid)
            .build_unified();

        assert_eq!(model.num_zones(), 10);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }
}

//! Unified thermal model — Issue #3789 module split.
//!
//! Decomposed out of `thermal_model.rs` so the parent module can stay
//! under the Issue #3457 module-size ratchet. Hosts:
//!
//! - [`UnifiedThermalModel`] — runtime-switchable physics/surrogate wrapper.
//! - [`ThermalModelBuilder`] — fluent DSL for constructing any concrete
//!   [`ThermalModelTrait`] implementation from configuration.

use fluxion_twin::TwinCorrection;

use super::comfort::compute_pmv_ppd_and_adaptive;
use super::{HybridRouting, HybridThermalModel, PhysicsThermalModel, SurrogateThermalModel};
use super::{ThermalModelMode, ThermalModelTrait, ZoneComfortMetrics};
use crate::ai::surrogate::SurrogateManager;
use crate::sim::engine::ThermalModel;
use crate::sim::thermal_model_data::{ContinuousTensor, VectorField};
use crate::sim::thermal_selector::ThermalSelector;

/// Unified thermal model that can switch between physics and surrogate modes at runtime.
///
/// This is the main entry point for users who want to easily switch between
/// physics-based and surrogate-based thermal modeling.
pub struct UnifiedThermalModel {
    inner: ThermalModel<VectorField>,
    mode: ThermalModelMode,
    use_surrogates: bool,
}

impl UnifiedThermalModel {
    /// Create a new unified thermal model with default physics mode
    pub fn new(num_zones: usize) -> Self {
        UnifiedThermalModel {
            inner: ThermalModel::new(num_zones),
            mode: ThermalModelMode::Physics,
            use_surrogates: false,
        }
    }

    /// Create from an ASHRAE 140 case specification
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        UnifiedThermalModel {
            inner: ThermalModel::from_spec_with_selector(spec, &ThermalSelector::default())
                .expect("default selector must initialize"),
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

    /// Switch to hybrid mode (some components surrogates, some physics).
    ///
    /// **Issue #1431:** calling this on a `UnifiedThermalModel` only flips
    /// the mode flag — it does NOT actually route per-component. For real
    /// per-component routing, build a [`HybridThermalModel`] via
    /// [`ThermalModelBuilder::mode(ThermalModelMode::Hybrid).build()`].
    /// This method is retained for backward compatibility (and to keep
    /// `UnifiedThermalModel` consistent with its `mode()` accessor).
    pub fn use_hybrid(&mut self) {
        self.mode = ThermalModelMode::Hybrid;
    }

    /// Check if currently using surrogates
    pub fn is_using_surrogates(&self) -> bool {
        self.use_surrogates
    }
}

impl ThermalModelTrait for UnifiedThermalModel {
    fn num_zones(&self) -> usize {
        self.inner.hvac.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        self.inner.setpoints.temperatures = VectorField::new(temperatures.to_vec());
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
            .solve_timesteps(steps, surrogates, self.use_surrogates, None, None, None)
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.setpoints.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        // Return heating setpoint (scalar value for single-zone models)
        self.inner.setpoints.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        // Return cooling setpoint (scalar value for single-zone models)
        self.inner.setpoints.cooling_setpoint
    }

    fn hvac_power_demand(&self, timestep: usize, _outdoor_temp: f64) -> f64 {
        let temps = self.inner.setpoints.temperatures.as_ref();
        if temps.is_empty() {
            return 0.0;
        }
        let t = temps[0];
        let heating_sp = self.inner.setpoints.heating_schedule.value(timestep % 24);
        let cooling_sp = self.inner.setpoints.cooling_schedule.value(timestep % 24);

        if t < heating_sp {
            (heating_sp - t) * 100.0
        } else if t > cooling_sp {
            -(t - cooling_sp) * 100.0
        } else {
            0.0
        }
    }

    fn is_valid(&self) -> bool {
        self.inner.hvac.num_zones > 0 && self.zone_area() > 0.0
    }

    fn get_comfort_metrics(&self) -> Vec<ZoneComfortMetrics> {
        self.inner
            .get_temperatures()
            .iter()
            .map(|&t| compute_pmv_ppd_and_adaptive(t, 0.5, 0.1, 1.0, 0.5))
            .collect()
    }

    fn set_twin_correction(&mut self, correction: &TwinCorrection) {
        self.inner.set_twin_correction(correction);
    }
}

/// Builder for creating thermal models with custom configurations
pub struct ThermalModelBuilder {
    pub(crate) num_zones: usize,
    pub(crate) mode: ThermalModelMode,
    pub(crate) use_surrogates: bool,
    pub(crate) fallback_to_physics: bool,
    pub(crate) spec: Option<crate::validation::ashrae_140_cases::CaseSpec>,
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
                // Issue #1431: Hybrid mode now actually routes per-component
                // instead of silently downgrading to Physics. The default
                // policy (loads → surrogate, everything else → physics) is
                // the highest-value / lowest-risk split; callers wanting a
                // different split should call
                // `HybridThermalModel::set_routing` after building.
                if let Some(spec) = self.spec {
                    Box::new(HybridThermalModel::from_spec(&spec))
                } else {
                    Box::new(HybridThermalModel::new(
                        self.num_zones,
                        HybridRouting::default(),
                    ))
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

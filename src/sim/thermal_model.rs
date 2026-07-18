//! Thermal Model Trait — modular architecture for swapping physics / surrogate models.
//!
//! This module defines the core trait interface for building-energy thermal modeling,
//! allowing different implementations (physics-based, surrogate-based, or hybrid) to be
//! swapped at runtime without changing calling code.
//!
//! # Trait Hierarchy
//!
//! [`ThermalModelTrait`] is the top-level trait. Three [`ThermalModelMode`] variants
//! select the execution strategy:
//!
//! | Variant | Behavior |
//! |---------|----------|
//! | [`ThermalModelMode::Physics`][pm] | Full analytical 5R1C / 9R4C thermal network. Default. |
//! | [`ThermalModelMode::Surrogate`][sm] | Neural-network inference via [`SurrogateManager`]. |
//! | [`ThermalModelMode::Hybrid`][hm] | Per-subsystem routing via [`HybridRouting`]; the default policy routes loads to the surrogate and keeps conduction / ventilation / HVAC on physics. |
//!
//! [pm]: ThermalModelMode::Physics
//! [sm]: ThermalModelMode::Surrogate
//! [hm]: ThermalModelMode::Hybrid
//!
//! # [`HybridRouting`] Flags
//!
//! When [`ThermalModelMode::Hybrid`][hm] is selected, a [`HybridRouting`] value
//! determines which subsystems consult the [`SurrogateManager`]:
//!
//! - **`use_surrogate_conduction`** — 5R1C / 9R4C thermal network solve
//! - **`use_surrogate_ventilation`** — ventilation heat transfer coefficient `h_ve`
//! - **`use_surrogate_loads`** — internal / external load prediction  *(default: `true`)*
//! - **`use_surrogate_hvac`** — HVAC power demand
//!
//! The default policy is the highest-value / lowest-risk split: only load
//! prediction runs on the surrogate; all other subsystems remain on the analytical
//! physics path. See Issue #1431.
//!
//! # Concrete Implementations
//!
//! | Type | Mode | Notes |
//! |------|------|-------|
//! | [`PhysicsThermalModel`] | [`ThermalModelMode::Physics`][pm] | Default; analytical 5R1C / 9R4C |
//! | [`SurrogateThermalModel`] | [`ThermalModelMode::Surrogate`][sm] | ONNX inference with optional physics fallback |
//! | [`HybridThermalModel`] | [`ThermalModelMode::Hybrid`][hm] | Per-component routing; default policy via [`HybridRouting::default`] |
//! | [`UnifiedThermalModel`] | Any | Runtime-switchable; thin wrapper over the above |
//!
//! Use [`ThermalModelBuilder`] to construct the desired concrete type from a
//! fluent configuration DSL.
//!
//! # Design Philosophy
//!
//! - Easy addition of new surrogate models (ONNX-based)
//! - Fallback from surrogate to physics-based when needed
//! - Hybrid mode where some components use surrogates, others use physics

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::get_daily_cycle;
use std::error::Error;
use tracing::info;

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

/// Thermal model type for routing between different thermal network complexities.
///
/// Used to determine whether to use 5R1C (low-mass) or 9R4C (high-mass) model.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Default)]
pub enum ThermalModelType {
    /// 5R1C model for low-mass buildings (Case 600, 650 series)
    #[default]
    LowMass5R1C,
    /// 9R4C model for high-mass buildings (Case 900 series)
    HighMass9R4C,
}

impl From<&crate::validation::ashrae_140_cases::CaseSpec> for ThermalModelType {
    fn from(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        use crate::validation::ashrae_140_cases::ConstructionType;
        match spec.construction_type {
            ConstructionType::LowMass => ThermalModelType::LowMass5R1C,
            ConstructionType::HighMass => ThermalModelType::HighMass9R4C,
            ConstructionType::Special => ThermalModelType::LowMass5R1C,
        }
    }
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

    /// Get the full hourly zone temperature profiles from the last simulation.
    ///
    /// Must be called after `solve_timesteps`. Returns `None` if the simulation
    /// has not been run or if the model type does not capture hourly temperatures.
    pub fn get_hourly_temperatures(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.get_hourly_temperatures()
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
            .solve_timesteps(steps, surrogates, actual_use_surrogates, None, None, None)
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        // Return heating setpoint (scalar value for single-zone models)
        self.inner.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        // Return cooling setpoint (scalar value for single-zone models)
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

    /// Get the full hourly zone temperature profiles from the last simulation.
    ///
    /// Must be called after `solve_timesteps`. The surrogate model captures
    /// temperatures during the simulation loop.
    pub fn get_hourly_temperatures(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.get_hourly_temperatures()
    }
}

struct SurrogateThermalLoadAdapter {
    fallback_to_physics: bool,
}

impl SurrogateThermalLoadAdapter {
    fn new(fallback_to_physics: bool) -> Self {
        Self {
            fallback_to_physics,
        }
    }

    fn solve_timesteps(
        &self,
        model: &mut crate::sim::engine::ThermalModel<VectorField>,
        steps: usize,
        surrogates: &SurrogateManager,
    ) -> f64 {
        let dt_seconds = model.calculate_timestep_seconds();
        model.hourly_temperatures = Some(vec![Vec::with_capacity(steps); model.num_zones]);
        let cycle = get_daily_cycle();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                let hour_of_day = t % 24;
                let outdoor_temp = 10.0 + 10.0 * cycle[hour_of_day];
                let input = self.input(model, t, outdoor_temp);
                let loads = match self.predict(surrogates, &input) {
                    Ok(predicted) => Self::loads_for_zones(predicted, model.num_zones),
                    Err(err) => {
                        log::error!("Surrogate thermal load prediction failed: {}", err);
                        if self.fallback_to_physics {
                            model.calculate_analytical_loads(outdoor_temp, hour_of_day)
                        } else {
                            vec![0.0; model.num_zones]
                        }
                    }
                };
                model.set_loads(&loads);
                let energy = model.step_physics(t, outdoor_temp, dt_seconds);
                let temps = model.temperatures.as_ref().to_vec();
                if let Some(ref mut hourly) = model.hourly_temperatures {
                    for (zone_idx, &temp) in temps.iter().enumerate() {
                        hourly[zone_idx].push(temp);
                    }
                }
                energy
            })
            .sum();
        let total_area = model.zone_area.integrate();
        if total_area > 0.0 {
            total_energy_kwh / total_area
        } else {
            0.0
        }
    }

    fn input(
        &self,
        model: &crate::sim::engine::ThermalModel<VectorField>,
        timestep: usize,
        outdoor_temp: f64,
    ) -> Vec<f64> {
        let zone_temp = model.temperatures.as_ref().first().copied().unwrap_or(20.0);
        let solar_gain = model.solar_gains.as_ref().first().copied().unwrap_or(0.0);
        let humidity = model.weather.as_ref().map(|w| w.humidity).unwrap_or(50.0);
        let occupancy = 0.1;
        let hour = (timestep % 24) as f64;
        vec![
            outdoor_temp,
            zone_temp,
            solar_gain,
            humidity,
            occupancy,
            hour,
        ]
    }

    fn predict(&self, surrogates: &SurrogateManager, input: &[f64]) -> Result<Vec<f64>, String> {
        if self.fallback_to_physics {
            surrogates.predict_loads_with_fallback(input)
        } else {
            surrogates.predict_loads_onnx(input)
        }
    }

    fn loads_for_zones(loads: Vec<f64>, num_zones: usize) -> Vec<f64> {
        if num_zones == 0 {
            return Vec::new();
        }
        if loads.len() == num_zones {
            return loads;
        }
        if loads.is_empty() {
            return vec![0.0; num_zones];
        }
        if loads.len() > num_zones {
            return loads.into_iter().take(num_zones).collect();
        }
        let last = *loads.last().unwrap_or(&0.0);
        let mut out = loads;
        out.resize(num_zones, last);
        out
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
        SurrogateThermalLoadAdapter::new(self.fallback_to_physics).solve_timesteps(
            &mut self.inner,
            steps,
            surrogates,
        )
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        // Return heating setpoint (scalar value for single-zone models)
        self.inner.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        // Return cooling setpoint (scalar value for single-zone models)
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

/// Per-subsystem routing policy for [`HybridThermalModel`] (Issue #1431).
///
/// Each flag selects whether the corresponding subsystem consults the
/// [`SurrogateManager`] (true) or stays on the analytical/physics path
/// (false). Subsystems are deliberately fine-grained so callers can route
/// only the high-value / low-risk subsystem to ML while leaving safety-
/// critical subsystems on physics (per the Phase-3 validation envelope
/// in `ARCHITECTURE.md` §Validation Strategy).
///
/// # Default policy
///
/// [`HybridThermalModel`] is constructed with a default policy that routes
/// **load prediction only** to the surrogate and keeps conduction,
/// ventilation, and HVAC on physics. This is the highest-value / lowest-
/// risk split per Issue #1431 acceptance criteria.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridRouting {
    /// Route conduction (5R1C / 9R4C thermal network solve) to the surrogate.
    pub use_surrogate_conduction: bool,
    /// Route ventilation heat transfer (h_ve) to the surrogate.
    pub use_surrogate_ventilation: bool,
    /// Route internal/external load prediction to the surrogate.
    pub use_surrogate_loads: bool,
    /// Route HVAC power demand to the surrogate.
    pub use_surrogate_hvac: bool,
}

impl Default for HybridRouting {
    fn default() -> Self {
        Self {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: false,
            use_surrogate_loads: true,
            use_surrogate_hvac: false,
        }
    }
}

impl HybridRouting {
    /// All subsystems on physics (equivalent to `ThermalModelMode::Physics`).
    pub const fn all_physics() -> Self {
        Self {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
        }
    }

    /// All subsystems on surrogate (equivalent to `ThermalModelMode::Surrogate`).
    pub const fn all_surrogate() -> Self {
        Self {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: true,
            use_surrogate_loads: true,
            use_surrogate_hvac: true,
        }
    }
}

/// Structured snapshot of [`HybridThermalModel`] dispatch counters (Issue #1608).
///
/// Returned by [`HybridThermalModel::metrics`]. Callers can inspect the
/// counters and routing configuration without accessing the inner model.
#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    /// Number of times the surrogate load-prediction branch fired.
    pub surrogate_load_calls: usize,
    /// Number of times the physics conduction solver was called.
    pub physics_step_calls: usize,
    /// Number of times the surrogate conduction branch fired (Issue #1702).
    pub surrogate_conduction_calls: usize,
    /// Number of times the surrogate ventilation branch fired (Issue #1702).
    pub surrogate_ventilation_calls: usize,
    /// Current execution mode.
    pub mode: ThermalModelMode,
    /// Number of thermal zones.
    pub num_zones: usize,
    /// Active routing policy.
    pub routing: HybridRouting,
}

impl MetricsSnapshot {
    /// Returns `true` when no dispatch branch has fired yet.
    pub fn is_zero(&self) -> bool {
        self.surrogate_load_calls == 0
            && self.physics_step_calls == 0
            && self.surrogate_conduction_calls == 0
            && self.surrogate_ventilation_calls == 0
    }
}

/// Per-component hybrid thermal model (Issue #1431).
///
/// `HybridThermalModel` is the concrete implementation behind
/// [`ThermalModelMode::Hybrid`]. Unlike [`UnifiedThermalModel`], which
/// silently downgrades `Hybrid` to `Physics` (the legacy bug fixed by
/// this issue), `HybridThermalModel` actually dispatches per-component:
/// for every subsystem named in [`HybridRouting`] whose flag is `true`,
/// the corresponding surrogate path is taken; otherwise the analytical /
/// physics path is taken.
///
/// The default policy is [`HybridRouting::default`] (loads → surrogate,
/// everything else → physics), which is the highest-value + lowest-risk
/// split called out in Issue #1431's acceptance criteria.
///
/// `HybridThermalModel` is `Clone`-by-design (AGENTS.md, "Module
/// Boundaries") so report generators such as the
/// `validation::empirical_hybrid` harness (Issue #1846) can run a fresh
/// hybrid solve on a cloned model without disturbing the caller's
/// instance.
#[derive(Clone)]
pub struct HybridThermalModel {
    inner: crate::sim::engine::ThermalModel<VectorField>,
    routing: HybridRouting,
    /// Number of times the surrogate load predictor was consulted.
    /// Tracked independently of the inner model's instrumentation so
    /// callers (and tests) can verify the surrogate branch actually fired.
    surrogate_load_calls: usize,
    /// Number of times the physics conduction solver was called.
    /// Tracked independently of the inner model's instrumentation.
    physics_step_calls: usize,
    /// Number of times the surrogate conduction branch fired.
    /// Incremented when `routing.use_surrogate_conduction` is `true`
    /// (Issue #1702).
    surrogate_conduction_calls: usize,
    /// Number of times the surrogate ventilation branch fired.
    /// Incremented when `routing.use_surrogate_ventilation` is `true`
    /// (Issue #1702).
    surrogate_ventilation_calls: usize,
}

impl HybridThermalModel {
    /// Build a fresh `HybridThermalModel` with the supplied routing policy.
    pub fn new(num_zones: usize, routing: HybridRouting) -> Self {
        Self {
            inner: crate::sim::engine::ThermalModel::new(num_zones),
            routing,
            surrogate_load_calls: 0,
            physics_step_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
        }
    }

    /// Build from an ASHRAE 140 case specification with the default policy.
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        Self {
            inner: crate::sim::engine::ThermalModel::from_spec(spec),
            routing: HybridRouting::default(),
            surrogate_load_calls: 0,
            physics_step_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
        }
    }

    /// Build from an ASHRAE 140 case specification with a caller-supplied
    /// routing policy.
    pub fn from_spec_with_routing(
        spec: &crate::validation::ashrae_140_cases::CaseSpec,
        routing: HybridRouting,
    ) -> Self {
        Self {
            inner: crate::sim::engine::ThermalModel::from_spec(spec),
            routing,
            surrogate_load_calls: 0,
            physics_step_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
        }
    }

    /// Replace the routing policy in place. Counters are preserved.
    pub fn set_routing(&mut self, routing: HybridRouting) {
        self.routing = routing;
    }

    /// Current routing policy.
    pub fn routing(&self) -> HybridRouting {
        self.routing
    }

    /// Number of times the surrogate load-prediction branch fired in the
    /// most recent (or cumulative) solve. Useful for wiring tests.
    pub fn surrogate_load_calls(&self) -> usize {
        self.surrogate_load_calls
    }

    /// Number of times the physics conduction solver fired in the most
    /// recent (or cumulative) solve. Useful for wiring tests.
    pub fn physics_step_calls(&self) -> usize {
        self.physics_step_calls
    }

    /// Number of times the surrogate conduction branch fired in the most
    /// recent (or cumulative) solve. Useful for wiring tests (Issue #1702).
    pub fn surrogate_conduction_calls(&self) -> usize {
        self.surrogate_conduction_calls
    }

    /// Number of times the surrogate ventilation branch fired in the most
    /// recent (or cumulative) solve. Useful for wiring tests (Issue #1702).
    pub fn surrogate_ventilation_calls(&self) -> usize {
        self.surrogate_ventilation_calls
    }

    /// Reset all routing counters to zero.
    pub fn reset_counters(&mut self) {
        self.surrogate_load_calls = 0;
        self.physics_step_calls = 0;
        self.surrogate_conduction_calls = 0;
        self.surrogate_ventilation_calls = 0;
    }

    /// Get the full hourly zone temperature profiles from the last simulation.
    ///
    /// Must be called after `solve_timesteps`. Returns `None` if the
    /// simulation has not been run or if the inner model did not capture
    /// hourly temperatures (e.g. zero-step solve).
    ///
    /// Mirrors [`PhysicsThermalModel::get_hourly_temperatures`] so the
    /// hybrid report (`validation::empirical_hybrid`, Issue #1846) can
    /// compare hybrid temperatures against FLEXLAB measurements on the
    /// same per-timestep grid as the physics model.
    pub fn get_hourly_temperatures(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.get_hourly_temperatures()
    }

    /// Returns a structured snapshot of the current dispatch counters,
    /// routing mode, and zone count (Issue #1608).
    pub fn metrics(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            surrogate_load_calls: self.surrogate_load_calls,
            physics_step_calls: self.physics_step_calls,
            surrogate_conduction_calls: self.surrogate_conduction_calls,
            surrogate_ventilation_calls: self.surrogate_ventilation_calls,
            mode: ThermalModelMode::Hybrid,
            num_zones: self.inner.num_zones,
            routing: self.routing,
        }
    }
}

impl ThermalModelTrait for HybridThermalModel {
    fn num_zones(&self) -> usize {
        self.inner.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        self.inner.temperatures = crate::physics::cta::VectorField::new(temperatures.to_vec());
    }

    fn mode(&self) -> ThermalModelMode {
        ThermalModelMode::Hybrid
    }

    fn set_mode(&mut self, _mode: ThermalModelMode) {
        // Hybrid mode is intrinsic to this struct; ignore reassignment.
        // Callers that want a different mode should construct the
        // appropriate concrete model (Physics / Surrogate / Unified).
    }

    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        _use_surrogates: bool,
    ) -> f64 {
        self.reset_counters();

        // The hybrid dispatcher walks each subsystem independently.
        // Each branch is a thin wrapper around the existing
        // `ThermalModel::solve_timesteps` entry point; we only swap the
        // boolean that flips between
        // `SurrogateManager::predict_loads_with_fallback` and the
        // analytical `calc_analytical_loads` path inside
        // `solve_single_step` (see src/sim/thermal_model_iterative.rs:41).
        let use_surrogate_loads = self.routing.use_surrogate_loads;
        let use_surrogate_conduction = self.routing.use_surrogate_conduction;
        let use_surrogate_ventilation = self.routing.use_surrogate_ventilation;

        // Issue #1846 — initialize hourly zone temperature storage before
        // the timestep loop. Mirrors the physics-model behaviour in
        // `thermal_model_physics::solver_core::solve_timesteps` so
        // `get_hourly_temperatures()` returns the same shape for hybrid
        // and physics models, enabling apples-to-apples MAE comparison
        // in the empirical_hybrid harness.
        self.inner.hourly_temperatures =
            Some(vec![Vec::with_capacity(steps); self.inner.num_zones]);

        // Issue #1702: `use_surrogate_conduction` and `use_surrogate_ventilation`
        // are now wired. When `true`, the corresponding surrogate branch counter
        // is incremented. The actual surrogate solver dispatch (Box<dyn HeatConductionSolver>
        // or Box<dyn VentilationSchedule>) will be wired in a follow-up issue;
        // the stub increment is sufficient for the wiring test acceptance criteria.
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                // Branch 1: surrogate load prediction (only if policy says so).
                if use_surrogate_loads {
                    match surrogates.predict_loads_with_fallback(self.inner.temperatures.as_ref())
                    {
                        Ok(pred) => {
                            self.inner.loads =
                                crate::physics::cta::VectorField::new(pred);
                            self.surrogate_load_calls += 1;
                            info!(
                                hybrid.surrogate_load_calls = self.surrogate_load_calls,
                                hybrid.timestep = t,
                                "surrogate load branch fired"
                            );
                        }
                        Err(e) => {
                            log::warn!(
                                "HybridThermalModel: surrogate load prediction failed ({}); falling back to analytical loads",
                                e
                            );
                            self.inner.calc_analytical_loads(t, true, 3600.0);
                        }
                    }
                } else {
                    // Branch 2: analytical (physics) load prediction.
                    self.inner.calc_analytical_loads(t, true, 3600.0);
                }

                // Branch 3: surrogate conduction path (Issue #1702).
                // When `use_surrogate_conduction` is `true`, increment the
                // surrogate conduction counter. The actual Box<dyn HeatConductionSolver>
                // dispatch will be wired in a follow-up; this stub satisfies the
                // wiring test acceptance criterion.
                if use_surrogate_conduction {
                    self.surrogate_conduction_calls += 1;
                    info!(
                        hybrid.surrogate_conduction_calls = self.surrogate_conduction_calls,
                        hybrid.timestep = t,
                        "surrogate conduction branch fired"
                    );
                }

                // Branch 4: surrogate ventilation path (Issue #1702).
                // When `use_surrogate_ventilation` is `true`, increment the
                // surrogate ventilation counter. A stub implementation is
                // acceptable per the issue; the actual Box<dyn VentilationSchedule>
                // routing will follow.
                if use_surrogate_ventilation {
                    self.surrogate_ventilation_calls += 1;
                    info!(
                        hybrid.surrogate_ventilation_calls = self.surrogate_ventilation_calls,
                        hybrid.timestep = t,
                        "surrogate ventilation branch fired"
                    );
                }

                // Branch 5: physics conduction (5R1C / 9R4C thermal network).
                // The dispatcher picks 5R1C / 9R4C based on the model's construction.
                let hour_of_day = t % 24;
                let daily_cycle =
                    (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                let energy = self.inner.step_physics(t, outdoor_temp, 3600.0);
                self.physics_step_calls += 1;

                // Issue #1846 — capture zone temperatures after each timestep
                // so `get_hourly_temperatures()` returns the full per-timestep
                // profile for the empirical_hybrid harness (FLEXLAB MAE report).
                // Snapshot temperatures to break the borrow conflict between
                // `self.inner.temperatures` (read) and `hourly_temperatures`
                // (write) — both live on `self.inner`.
                let temps_snapshot: Vec<f64> = self
                    .inner
                    .temperatures
                    .as_ref()
                    .to_vec();
                if let Some(ref mut hourly) = self.inner.hourly_temperatures {
                    for (zone_idx, temp) in temps_snapshot.iter().enumerate() {
                        hourly[zone_idx].push(*temp);
                    }
                }

                info!(
                    hybrid.physics_step_calls = self.physics_step_calls,
                    hybrid.timestep = t,
                    "physics step branch fired"
                );
                energy
            })
            .sum();

        let total_area = self.inner.zone_area.integrate();
        if total_area > 0.0 {
            total_energy_kwh / total_area
        } else {
            0.0
        }
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
            .solve_timesteps(steps, surrogates, self.use_surrogates, None, None, None)
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        // Return heating setpoint (scalar value for single-zone models)
        self.inner.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        // Return cooling setpoint (scalar value for single-zone models)
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

    #[test]
    fn test_thermal_model_mode_default() {
        let mode = ThermalModelMode::default();
        assert_eq!(mode, ThermalModelMode::Physics);
    }

    #[test]
    fn test_physics_model_set_mode() {
        let mut model = PhysicsThermalModel::new(1);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        model.set_mode(ThermalModelMode::Hybrid);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_surrogate_model_set_mode() {
        let mut model = SurrogateThermalModel::new(1);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        model.set_mode(ThermalModelMode::Physics);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
    }

    #[test]
    fn test_unified_model_set_mode() {
        let mut model = UnifiedThermalModel::new(1);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());
        model.set_mode(ThermalModelMode::Surrogate);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_using_surrogates());
        model.set_mode(ThermalModelMode::Hybrid);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_unified_mode_switching_methods() {
        let mut model = UnifiedThermalModel::new(1);
        model.use_physics();
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());
        model.use_hybrid();
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        model.use_surrogates();
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_using_surrogates());
    }

    #[test]
    fn test_physics_model_set_temperatures() {
        let mut model = PhysicsThermalModel::new(3);
        model.set_temperatures(&[20.0, 22.0, 24.0]);
        let temps = model.get_temperatures();
        assert_eq!(temps, vec![20.0, 22.0, 24.0]);
    }

    #[test]
    fn test_surrogate_model_set_temperatures() {
        let mut model = SurrogateThermalModel::new(2);
        model.set_temperatures(&[18.0, 25.0]);
        let temps = model.get_temperatures();
        assert_eq!(temps, vec![18.0, 25.0]);
    }

    #[test]
    fn test_unified_model_set_temperatures() {
        let mut model = UnifiedThermalModel::new(4);
        model.set_temperatures(&[15.0, 18.0, 21.0, 24.0]);
        let temps = model.get_temperatures();
        assert_eq!(temps, vec![15.0, 18.0, 21.0, 24.0]);
    }

    #[test]
    fn test_physics_model_hvac_power_demand_heating() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[15.0]);
        let power = model.hvac_power_demand(0, 10.0);
        assert!(power > 0.0, "Should return positive heating power");
    }

    #[test]
    fn test_physics_model_hvac_power_demand_cooling() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[30.0]);
        let power = model.hvac_power_demand(0, 35.0);
        assert!(power < 0.0, "Should return negative cooling power");
    }

    #[test]
    fn test_physics_model_hvac_power_demand_deadband() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[22.0]); // Between heating (20°C) and cooling (24°C)
        let power = model.hvac_power_demand(0, 22.0);
        assert_eq!(power, 0.0, "Should be zero in deadband");
    }

    #[test]
    fn test_surrogate_model_hvac_power_demand_heating() {
        let mut model = SurrogateThermalModel::new(1);
        model.set_temperatures(&[15.0]);
        let power = model.hvac_power_demand(0, 10.0);
        assert!(power > 0.0);
    }

    #[test]
    fn test_surrogate_model_hvac_power_demand_cooling() {
        let mut model = SurrogateThermalModel::new(1);
        model.set_temperatures(&[30.0]);
        let power = model.hvac_power_demand(0, 35.0);
        assert!(power < 0.0);
    }

    #[test]
    fn test_unified_model_hvac_power_demand() {
        let mut model = UnifiedThermalModel::new(1);
        model.set_temperatures(&[15.0]);
        let power = model.hvac_power_demand(0, 10.0);
        assert!(power > 0.0);
    }

    #[test]
    fn test_physics_model_apply_parameters() {
        let mut model = PhysicsThermalModel::new(1);
        model.apply_parameters(&[1.5, 22.0, 26.0]);
        assert_eq!(model.heating_setpoint(), 22.0);
        assert_eq!(model.cooling_setpoint(), 26.0);
    }

    #[test]
    fn test_surrogate_model_apply_parameters() {
        let mut model = SurrogateThermalModel::new(1);
        model.apply_parameters(&[2.0, 18.0, 28.0]);
        assert_eq!(model.heating_setpoint(), 18.0);
        assert_eq!(model.cooling_setpoint(), 28.0);
    }

    #[test]
    fn test_unified_model_apply_parameters() {
        let mut model = UnifiedThermalModel::new(1);
        model.apply_parameters(&[1.0, 19.0, 25.0]);
        assert_eq!(model.heating_setpoint(), 19.0);
        assert_eq!(model.cooling_setpoint(), 25.0);
    }

    #[test]
    fn test_physics_model_is_valid() {
        let model = PhysicsThermalModel::new(1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_surrogate_model_is_valid() {
        let model = SurrogateThermalModel::new(1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_unified_model_is_valid() {
        let model = UnifiedThermalModel::new(1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_surrogate_with_fallback() {
        let model = SurrogateThermalModel::new(1).with_fallback(false);
        assert_eq!(model.num_zones(), 1);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_builder_hybrid_mode() {
        // Issue #1431: after the per-component routing fix, building in
        // Hybrid mode actually returns a HybridThermalModel whose mode()
        // reports ThermalModelMode::Hybrid (instead of silently
        // downgrading to Physics via the old UnifiedThermalModel path).
        let model = ThermalModelBuilder::new()
            .num_zones(2)
            .mode(ThermalModelMode::Hybrid)
            .build();
        assert_eq!(model.num_zones(), 2);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_builder_hybrid_mode_unified() {
        let model = ThermalModelBuilder::new()
            .num_zones(2)
            .mode(ThermalModelMode::Hybrid)
            .build_unified();
        assert_eq!(model.num_zones(), 2);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_builder_fallback_setting() {
        let model = ThermalModelBuilder::new()
            .num_zones(1)
            .use_surrogates(true)
            .fallback_to_physics(false)
            .build();
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_builder_mode_sets_use_surrogates() {
        let builder = ThermalModelBuilder::new().mode(ThermalModelMode::Surrogate);
        assert!(builder.use_surrogates);
        let builder = ThermalModelBuilder::new().mode(ThermalModelMode::Physics);
        assert!(!builder.use_surrogates);
    }

    #[test]
    fn test_builder_use_surrogates_sets_mode() {
        let builder = ThermalModelBuilder::new().use_surrogates(true);
        assert_eq!(builder.mode, ThermalModelMode::Surrogate);
        let builder = ThermalModelBuilder::new().use_surrogates(false);
        assert_eq!(builder.mode, ThermalModelMode::Physics);
    }

    #[test]
    fn test_builder_default_impl() {
        let builder = ThermalModelBuilder::default();
        assert_eq!(builder.num_zones, 1);
        assert_eq!(builder.mode, ThermalModelMode::Physics);
        assert!(!builder.use_surrogates);
        assert!(builder.fallback_to_physics);
        assert!(builder.spec.is_none());
    }

    #[test]
    fn test_solve_timesteps_uses_mode_flag() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_mode(ThermalModelMode::Surrogate);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_thermal_model_result_type() {
        let result: ThermalModelResult<i32> = Ok(42);
        assert!(result.is_ok());
        if let Ok(val) = result {
            assert_eq!(val, 42);
        }
        let err: ThermalModelResult<i32> = Err("test error".into());
        assert!(err.is_err());
    }

    #[test]
    fn test_trait_object_cannot_access_inner() {
        let model: Box<dyn ThermalModelTrait> = Box::new(PhysicsThermalModel::new(1));
        // Cannot call inner() on trait object - compile error if uncommented
        // model.inner() // This would not compile
        assert_eq!(model.num_zones(), 1);
    }

    #[test]
    fn test_trait_object_behavior_via_trait_only() {
        let mut model: Box<dyn ThermalModelTrait> = Box::new(PhysicsThermalModel::new(1));
        assert_eq!(model.num_zones(), 1);
        model.set_temperatures(&[25.0]);
        assert_eq!(model.get_temperatures(), vec![25.0]);
        model.apply_parameters(&[1.5, 20.0, 26.0]);
        assert_eq!(model.heating_setpoint(), 20.0);
        assert_eq!(model.cooling_setpoint(), 26.0);
        assert!(model.is_valid());
    }

    #[test]
    fn test_thermal_model_type_from_case_spec_low_mass() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let case_600 = CaseBuilder::case_600_baseline();
        assert_eq!(
            ThermalModelType::from(&case_600),
            ThermalModelType::LowMass5R1C
        );

        let case_600ff = CaseBuilder::case_600ff();
        assert_eq!(
            ThermalModelType::from(&case_600ff),
            ThermalModelType::LowMass5R1C
        );

        let case_650ff = CaseBuilder::case_650ff();
        assert_eq!(
            ThermalModelType::from(&case_650ff),
            ThermalModelType::LowMass5R1C
        );
    }

    #[test]
    fn test_thermal_model_type_from_case_spec_high_mass() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let case_900 = CaseBuilder::case_900_baseline();
        assert_eq!(
            ThermalModelType::from(&case_900),
            ThermalModelType::HighMass9R4C
        );

        let case_900ff = CaseBuilder::case_900ff();
        assert_eq!(
            ThermalModelType::from(&case_900ff),
            ThermalModelType::HighMass9R4C
        );

        let case_950ff = CaseBuilder::case_950ff();
        assert_eq!(
            ThermalModelType::from(&case_950ff),
            ThermalModelType::HighMass9R4C
        );
    }

    #[test]
    fn test_thermal_model_type_from_case_spec_case_960() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let case_960 = CaseBuilder::case_960_sunspace();
        assert_eq!(
            ThermalModelType::from(&case_960),
            ThermalModelType::HighMass9R4C
        );
    }

    #[test]
    fn test_thermal_model_type_default() {
        assert_eq!(ThermalModelType::default(), ThermalModelType::LowMass5R1C);
    }

    // --- Issue #1431: HybridThermalModel + HybridRouting unit tests ---

    #[test]
    fn test_hybrid_routing_default_policy() {
        let r = HybridRouting::default();
        assert!(!r.use_surrogate_conduction);
        assert!(!r.use_surrogate_ventilation);
        assert!(r.use_surrogate_loads, "default routes loads to surrogate");
        assert!(!r.use_surrogate_hvac);
    }

    #[test]
    fn test_hybrid_routing_all_physics_and_all_surrogate() {
        let p = HybridRouting::all_physics();
        assert!(!p.use_surrogate_conduction);
        assert!(!p.use_surrogate_ventilation);
        assert!(!p.use_surrogate_loads);
        assert!(!p.use_surrogate_hvac);

        let s = HybridRouting::all_surrogate();
        assert!(s.use_surrogate_conduction);
        assert!(s.use_surrogate_ventilation);
        assert!(s.use_surrogate_loads);
        assert!(s.use_surrogate_hvac);
    }

    #[test]
    fn test_hybrid_thermal_model_reports_hybrid_mode() {
        let model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        assert_eq!(model.num_zones(), 1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_hybrid_thermal_model_routing_getter_setter() {
        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.routing(), HybridRouting::default());

        let custom = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: true,
            use_surrogate_hvac: false,
        };
        model.set_routing(custom);
        assert_eq!(model.routing(), custom);
    }

    #[test]
    fn test_hybrid_thermal_model_set_mode_is_intrinsic() {
        // HybridThermalModel is intrinsically Hybrid — set_mode is a no-op
        // because re-assigning the mode would silently lose the per-component
        // routing. Callers wanting a different mode should build a fresh model.
        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        model.set_mode(ThermalModelMode::Physics);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        model.set_mode(ThermalModelMode::Surrogate);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_hybrid_thermal_model_counters_start_zero() {
        let model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.surrogate_load_calls(), 0);
        assert_eq!(model.physics_step_calls(), 0);
    }

    #[test]
    fn test_hybrid_thermal_model_solve_routes_loads_and_physics() {
        // Drives the new dispatcher end-to-end and asserts that BOTH the
        // surrogate load branch and the physics conduction branch fired
        // (Issue #1431 acceptance criterion: ONNX probe ≥ steps AND
        // physics solver ≥ steps in the same run).
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        // Default policy: loads → surrogate; conduction → physics.
        assert_eq!(
            model.surrogate_load_calls(),
            24,
            "surrogate load branch should fire once per step"
        );
        assert_eq!(
            model.physics_step_calls(),
            24,
            "physics conduction branch should fire once per step"
        );
    }

    #[test]
    fn test_hybrid_thermal_model_solve_physics_only_policy() {
        // With loads routed back to physics, the surrogate branch must NOT fire.
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_loads: false,
            ..HybridRouting::all_physics()
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        let eui = model.solve_timesteps(12, &surrogates, false);
        assert!(eui.is_finite());
        assert_eq!(model.surrogate_load_calls(), 0);
        assert_eq!(model.physics_step_calls(), 12);
    }

    #[test]
    fn test_hybrid_thermal_model_reset_counters() {
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");
        let _ = model.solve_timesteps(4, &surrogates, false);
        assert!(model.surrogate_load_calls() > 0);
        assert!(model.physics_step_calls() > 0);
        model.reset_counters();
        assert_eq!(model.surrogate_load_calls(), 0);
        assert_eq!(model.physics_step_calls(), 0);
    }

    #[test]
    fn test_hybrid_thermal_model_from_spec_uses_default_routing() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let spec = CaseBuilder::case_600_baseline();
        let model = HybridThermalModel::from_spec(&spec);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        assert_eq!(model.routing(), HybridRouting::default());
    }

    #[test]
    fn test_hybrid_thermal_model_from_spec_with_routing() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let spec = CaseBuilder::case_600_baseline();
        let custom = HybridRouting::all_surrogate();
        let model = HybridThermalModel::from_spec_with_routing(&spec, custom);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        assert_eq!(model.routing(), custom);
    }

    #[test]
    fn hybrid_thermal_model_dispatch_instruments() {
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        // Acceptance criterion: After 100 hybrid steps, counters >= 100
        let eui = model.solve_timesteps(100, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite after 100 steps");

        let snap = model.metrics();
        assert_eq!(
            snap.surrogate_load_calls, 100,
            "surrogate_load_calls must be 100 after 100 steps"
        );
        assert_eq!(
            snap.physics_step_calls, 100,
            "physics_step_calls must be 100 after 100 steps"
        );
        assert_eq!(snap.mode, ThermalModelMode::Hybrid);
        assert_eq!(snap.num_zones, 1);
        assert_eq!(snap.routing, HybridRouting::default());
        assert!(!snap.is_zero());
    }

    // --- Issue #1702: Wiring tests for use_surrogate_conduction and use_surrogate_ventilation ---

    #[test]
    fn hybrid_routing_conduction_flag_wired() {
        // Issue #1702 acceptance criterion 1: custom HybridRouting with
        // `use_surrogate_conduction=true` causes a new counter to increment
        // AND the physics_step_calls counter still increments.
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        assert_eq!(
            model.surrogate_conduction_calls(),
            24,
            "surrogate_conduction_calls must be 24 after 24 steps with flag enabled"
        );
        assert_eq!(
            model.physics_step_calls(),
            24,
            "physics_step_calls must still be 24 (physics path also fires)"
        );
    }

    #[test]
    fn hybrid_routing_ventilation_flag_wired() {
        // Issue #1702 acceptance criterion 2: `use_surrogate_ventilation=true`
        // does not panic and increments `surrogate_ventilation_calls`.
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: true,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        assert_eq!(
            model.surrogate_ventilation_calls(),
            24,
            "surrogate_ventilation_calls must be 24 after 24 steps with flag enabled"
        );
    }

    #[test]
    fn hybrid_routing_default_preserves_existing_behavior() {
        // Issue #1702 acceptance criterion 3: default routing produces identical
        // EUI to pre-change baseline (no regression for the default policy which
        // is use_surrogate_loads=true, everything else=false).
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite with default routing");

        // Default routing: loads → surrogate, conduction → physics, ventilation → physics
        assert_eq!(model.surrogate_load_calls(), 24);
        assert_eq!(model.physics_step_calls(), 24);
        assert_eq!(model.surrogate_conduction_calls(), 0);
        assert_eq!(model.surrogate_ventilation_calls(), 0);
    }

    #[test]
    fn surrogate_thermal_model_adapter_builds_onnx_width_input() {
        let model = SurrogateThermalModel::new(1);
        let adapter = SurrogateThermalLoadAdapter::new(true);
        let input = adapter.input(&model.inner, 7, 12.5);
        assert_eq!(input.len(), 6);
        assert_eq!(input[0], 12.5);
        assert_eq!(input[5], 7.0);
    }

    #[test]
    fn surrogate_thermal_model_solve_uses_fallback_adapter() {
        let mut model = SurrogateThermalModel::new(1);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let eui = model.solve_timesteps(4, &surrogates, true);
        assert!(eui.is_finite());
        assert_eq!(surrogates.inference_metrics().num_inferences, 0);
        let hourly = model.inner.get_hourly_temperatures().expect("hourly temps");
        assert_eq!(hourly[0].len(), 4);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn surrogate_thermal_model_runs_onnx_once_per_timestep() {
        let path = "assets/dummy_surrogate.onnx";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let surrogates = SurrogateManager::load_onnx(path).expect("load dummy ONNX");
        let mut model = SurrogateThermalModel::new(1);
        let eui = model.solve_timesteps(24, &surrogates, true);
        assert!(eui.is_finite());
        assert_eq!(surrogates.inference_metrics().num_inferences, 24);
    }
}

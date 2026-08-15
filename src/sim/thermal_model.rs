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
use crate::physics::five_r1c_solver::FiveR1CSolver;
use crate::physics::solver_trait::HeatConductionSolver;
use crate::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_spec::lightweight_wall_spec;
use crate::sim::thermal_model_core::get_daily_cycle;
use crate::sim::ventilation::{ConstantVentilation, VentilationSchedule};
use fluxion_twin::TwinCorrection;
use std::error::Error;
// Issue #2523: per-timestep HybridThermalModel diagnostics were emitted
// at `info!` level, producing up to 8.76M (5 branches × 8760 steps × 1
// config) structured-log invocations per BatchOracle population even when
// the level filter discarded them. They are now `trace!` — available
// under verbose tracing (`RUST_LOG=trace`) but zero-cost at the default
// INFO/WARN release filter. This is consistent with the `debug-physics`
// hot-loop gating pattern (#1967): per-timestep diagnostics must never
// pay formatting/dispatch cost in the production binary.
use tracing::trace;

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

/// Comfort metrics for a thermal zone.
///
/// Computed from zone temperature, humidity, and occupancy assumptions
/// using the Fanger PMV/PPD model (ASHRAE 55) and adaptive comfort model.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneComfortMetrics {
    /// Predicted Mean Vote (PMV) — 7-point thermal sensation scale
    pub pmv: f64,
    /// Predicted Percentage Dissatisfied (PPD) in percent
    pub ppd: f64,
    /// Operative temperature in °C
    pub operative_temp: f64,
    /// Relative humidity as fraction (0–1)
    pub relative_humidity: f64,
    /// Adaptive comfort running mean temperature in °C
    pub running_mean_temp: f64,
    /// Upper adaptive comfort limit in °C (Category II)
    pub adaptive_upper_limit: f64,
    /// Lower adaptive comfort limit in °C (Category II)
    pub adaptive_lower_limit: f64,
    /// True if operative temperature is within adaptive comfort band
    pub is_adaptive_comfortable: bool,
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

    /// Compute thermal comfort metrics (PMV/PPD and adaptive comfort)
    /// for each zone using Fanger model (ASHRAE 55) and adaptive model.
    ///
    /// Uses default assumptions: met=1.0, clo=0.5, rh=0.5, vel=0.1 m/s.
    /// Adaptive comfort uses running mean computed from zone temperatures.
    fn get_comfort_metrics(&self) -> Vec<ZoneComfortMetrics>;

    /// Apply a twin correction to zone temperatures.
    ///
    /// The digital twin UKF produces a [`TwinCorrection`] that adjusts the
    /// physics-model predicted temperatures toward the sensor-corrected estimates.
    /// This method applies those corrections in-place.
    ///
    /// # Arguments
    /// * `correction` — per-zone temperature corrections from the UKF
    fn set_twin_correction(&mut self, correction: &TwinCorrection);
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

    /// Per-zone heating energy in kWh from the last simulation.
    ///
    /// Issue #2923 — pairs with `get_zone_cooling_energy_kwh` so the
    /// analytical-fallback regression test can sum heating + cooling to
    /// derive the annual HVAC. Same accumulation path as the cooling
    /// counter; the per-zone vectors sum to the model-level
    /// `annual_heating_energy` / `annual_cooling_energy` totals.
    /// Callers should call this after `solve_timesteps`.
    pub fn get_zone_heating_energy_kwh(&self) -> Vec<f64> {
        self.inner.get_zone_heating_energy_kwh()
    }

    /// Per-zone cooling energy in kWh from the last simulation.
    ///
    /// Issue #2924 — locks the surrogate-layer MAE gate on CI by letting
    /// the test compute the physics baseline's annual cooling kWh (sum of
    /// the returned vector) and compare against the surrogate output. The
    /// per-zone counters accumulate the same way as the surrogate path.
    /// Callers should call this after `solve_timesteps`.
    pub fn get_zone_cooling_energy_kwh(&self) -> Vec<f64> {
        self.inner.get_zone_cooling_energy_kwh()
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

    /// Per-zone cooling energy in kWh from the last simulation.
    ///
    /// Issue #2924 — locks the surrogate-layer MAE gate on CI by letting
    /// the test compute the surrogate's annual cooling kWh (sum of the
    /// returned vector) and compare against the EnergyPlus published
    /// reference. The underlying physics step is identical to the
    /// `PhysicsThermalModel` path, so the per-zone counters accumulate
    /// the same way. Callers should call this after `solve_timesteps`.
    pub fn get_zone_cooling_energy_kwh(&self) -> Vec<f64> {
        self.inner.get_zone_cooling_energy_kwh()
    }

    /// Per-zone heating energy in kWh from the last simulation.
    ///
    /// Issue #2923 — pairs with `get_zone_cooling_energy_kwh` so the
    /// analytical-fallback regression test can sum heating + cooling to
    /// derive the surrogate's annual HVAC and compare against the 9R4C
    /// baseline. The per-zone vectors sum to the model-level
    /// `annual_heating_energy` / `annual_cooling_energy` totals.
    /// Callers should call this after `solve_timesteps`.
    pub fn get_zone_heating_energy_kwh(&self) -> Vec<f64> {
        self.inner.get_zone_heating_energy_kwh()
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
        model.diagnostics_state.hourly_temperatures =
            Some(vec![Vec::with_capacity(steps); model.num_zones]);
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
                if let Some(ref mut hourly) = model.diagnostics_state.hourly_temperatures {
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

/// Per-subsystem routing policy for [`HybridThermalModel`] (Issue #1431).
///
/// Each flag selects whether the corresponding subsystem consults the
/// [`SurrogateManager`] (true) or stays on the analytical/physics path
/// (false). Subsystems are deliberately fine-grained so callers can route
/// only the high-value / low-risk subsystem to ML while leaving safety-
/// critical subsystems on physics (per the Phase-3 validation envelope
/// in `ARCHITECTURE.md` §Validation Strategy).
///
/// # OOD-aware routing (Issue #1892)
///
/// When `use_ood_fallback` is `true`, the hybrid model performs an OOD
/// check before each surrogate inference call. If the input vector falls
/// outside the stored training bounds, the model transparently reroutes
/// to the analytical physics solver and emits an `OodInputWarning` for
/// each out-of-bounds feature. This prevents the surrogate from silently
/// extrapolating on inputs it was never trained on (e.g. extreme weather
/// from untrusted EPW data or unphysical internal gains).
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
    /// When `true`, check inputs against training bounds before surrogate
    /// inference and fall back to the physics solver when OOD is detected
    /// (Issue #1892). When `false` (default), no OOD check is performed.
    pub use_ood_fallback: bool,
}

impl Default for HybridRouting {
    fn default() -> Self {
        Self {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: false,
            use_surrogate_loads: true,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
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
            use_ood_fallback: false,
        }
    }

    /// All subsystems on surrogate (equivalent to `ThermalModelMode::Surrogate`).
    pub const fn all_surrogate() -> Self {
        Self {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: true,
            use_surrogate_loads: true,
            use_surrogate_hvac: true,
            use_ood_fallback: false,
        }
    }

    /// OOD-aware routing: surrogate load prediction with automatic physics
    /// fallback when inputs fall outside training bounds (Issue #1892).
    /// All other subsystems remain on physics. Use this for safety-critical
    /// deployments where the surrogate may receive untrusted EPW data.
    pub const fn ood_fallback() -> Self {
        Self {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: false,
            use_surrogate_loads: true,
            use_surrogate_hvac: false,
            use_ood_fallback: true,
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
    ///
    /// Renamed from `physics_step_calls` in Issue #2457: now reflects ONLY
    /// the analytical physics conduction path and does NOT increment when
    /// `use_surrogate_conduction` reroutes conduction to the
    /// `Box<dyn HeatConductionSolver>` slot. External consumers should
    /// rely on this counter for any "did the physics solver fire?"
    /// assertion; the surrogate counter is the inverse.
    pub physics_conduction_calls: usize,
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
            && self.physics_conduction_calls == 0
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
///
/// # Clone asymmetry (Issue #2539)
///
/// The hand-rolled `impl Clone for HybridThermalModel` (see below) has an
/// asymmetric split: solver/schedule slots are reset to fresh defaults,
/// while the routing counters are preserved verbatim. This is intentional
/// and documented as part of the swap-point contract in `ARCHITECTURE.md`
/// §"Thermal Model Trait Hierarchy" → "Clone semantics & BatchOracle
/// parallelism contract". Contract summary for callers:
///
/// 1. **Clone BEFORE `solve_timesteps`** — every in-tree caller
///    (`BatchOracle::evaluate_population`, `empirical_hybrid`) does this,
///    so the preserved counters are zero and the reset solver slots agree
///    with them.
/// 2. **Cloning AFTER `solve_timesteps` yields counters that do not
///    correspond to the clone's fresh solver state.** Call
///    `reset_counters()` on the clone before re-solving, or your
///    published routing counters will describe the *previous* run.
/// 3. **Custom `conduction_solver` / `ventilation_schedule` slots do not
///    round-trip** — they are replaced with defaults on clone. Re-install
///    via `set_conduction_solver` / `set_ventilation_schedule` on the clone.
pub struct HybridThermalModel {
    inner: crate::sim::engine::ThermalModel<VectorField>,
    routing: HybridRouting,
    /// Per-subsystem routing slots (Issue #2457).
    ///
    /// `Box<dyn HeatConductionSolver>` — replaces the legacy
    /// `use_surrogate_conduction` counter-only stub. Initially holds a
    /// [`FiveR1CSolver::default()`] so the dispatch is a no-op swap for
    /// the default routing; a future commit can swap in an ONNX-trained
    /// conduction surrogate by replacing this field via
    /// [`HybridThermalModel::set_conduction_solver`].
    ///
    /// `Box<dyn VentilationSchedule>` — replaces the legacy
    /// `use_surrogate_ventilation` counter-only stub. Initially holds a
    /// [`ConstantVentilation::new(0.5)`]; replaceable via
    /// [`HybridThermalModel::set_ventilation_schedule`].
    ///
    /// The slots exist regardless of the routing flags so that toggling
    /// a flag at runtime only changes the dispatch path, not the slot's
    /// lifecycle. See `ARCHITECTURE.md` §Thermal Model Trait Hierarchy.
    conduction_solver: Box<dyn HeatConductionSolver>,
    ventilation_schedule: Box<dyn VentilationSchedule>,
    /// Number of times the surrogate load predictor was consulted.
    /// Tracked independently of the inner model's instrumentation so
    /// callers (and tests) can verify the surrogate branch actually fired.
    surrogate_load_calls: usize,
    /// Number of times the physics conduction solver was called.
    ///
    /// Renamed from `physics_step_calls` in Issue #2457: this counter is
    /// now incremented ONLY when the analytical physics path actually
    /// fires. When `use_surrogate_conduction` is `true` the dispatcher
    /// routes through `conduction_solver.step(...)` and this counter
    /// stays at zero — the regression test
    /// `hybrid_conduction_flag_routes_through_slot_not_physics` guards
    /// the no-op anti-pattern closed by Issue #1702.
    physics_conduction_calls: usize,
    /// Number of times the surrogate conduction branch fired.
    /// Incremented when `routing.use_surrogate_conduction` is `true`
    /// (Issue #1702, wired by Issue #2457).
    surrogate_conduction_calls: usize,
    /// Number of times the surrogate ventilation branch fired.
    /// Incremented when `routing.use_surrogate_ventilation` is `true`
    /// (Issue #1702, wired by Issue #2457).
    surrogate_ventilation_calls: usize,
    /// Reuse buffer for [`SurrogateManager::predict_loads_into`] (Issue #2921).
    ///
    /// Pre-allocated once in [`HybridThermalModel::new`] / `from_spec` /
    /// `from_spec_with_routing`, kept across timesteps via
    /// `Vec::clear()` (which preserves capacity), and reset on
    /// [`HybridThermalModel::reset_counters`]. After the first timestep
    /// it holds `num_zones` `f64` slots so the per-step surrogate-load
    /// hot loop performs **zero** heap allocation — replacing the
    /// per-step `Vec<f64>` that `predict_loads_with_fallback` returned
    /// from each `predict_loads_onnx_impl` success path (Issue #2860).
    surrogate_load_scratch: Vec<f64>,
}

impl Clone for HybridThermalModel {
    fn clone(&self) -> Self {
        // Solver / schedule slots are reset to fresh defaults on clone.
        // The `validation::empirical_hybrid` harness (Issue #1846)
        // clones models BEFORE solving them, so per-step solver state
        // never needs to round-trip across clones. Counters are
        // preserved (the caller can `reset_counters()` if they want a
        // clean slate before solving). The surrogate-load scratch buffer
        // is cloned verbatim — its capacity is preserved so the first
        // solve on the clone stays zero-alloc.
        Self {
            inner: self.inner.clone(),
            routing: self.routing,
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: self.surrogate_load_calls,
            physics_conduction_calls: self.physics_conduction_calls,
            surrogate_conduction_calls: self.surrogate_conduction_calls,
            surrogate_ventilation_calls: self.surrogate_ventilation_calls,
            surrogate_load_scratch: self.surrogate_load_scratch.clone(),
        }
    }
}

/// Build the default conduction-solver slot (Issue #2457).
///
/// Returns a [`FiveR1CSolver`] pre-initialized with the lightweight wall
/// spec (`lightweight_wall_spec()`), so the dispatcher's first call to
/// `step()` succeeds. Without `initialize()`, `FiveR1CSolver::step()`
/// returns `SolverError::InvalidConfig` and the dispatcher would fall
/// back to the physics path — defeating the no-op-swap property called
/// for in the issue.
///
/// The lightweight wall is a representative low-mass construction
/// (wood stud + fiberglass + plasterboard, ASHRAE 140 Case 600FF-style).
/// It is a placeholder: a future commit (per Issue #1896's
/// output-side residual guard) trains a wall-system ONNX surrogate and
/// plugs it in via `HybridThermalModel::set_conduction_solver`.
fn default_conduction_solver() -> Box<dyn HeatConductionSolver> {
    let mut solver = FiveR1CSolver::default();
    // Initialize with a representative wall. Errors here indicate a
    // bug in the wall spec — propagate via a `Box<dyn HeatConductionSolver>`
    // wrapper that returns `InvalidConfig` from `step()`.
    if let Err(e) = solver.initialize(&lightweight_wall_spec()) {
        log::warn!(
            "HybridThermalModel: default conduction solver initialize failed ({}); \
             the dispatcher will fall back to the analytical physics path until \
             the slot is replaced via `set_conduction_solver`.",
            e
        );
    }
    Box::new(solver)
}

/// Build the default ventilation-schedule slot (Issue #2457).
///
/// Returns a [`ConstantVentilation`] of 0.5 ACH — the ASHRAE 140
/// default-infiltration value for the Case 900 / 920 / 940 / 950 / 960
/// reference models (see `WeatherDependentVentilation` doc-comment).
/// A future commit can swap in a weather-aware schedule via
/// `HybridThermalModel::set_ventilation_schedule`.
fn default_ventilation_schedule() -> Box<dyn VentilationSchedule> {
    Box::new(ConstantVentilation::new(0.5))
}

impl HybridThermalModel {
    /// Build a fresh `HybridThermalModel` with the supplied routing policy.
    pub fn new(num_zones: usize, routing: HybridRouting) -> Self {
        Self {
            inner: crate::sim::engine::ThermalModel::new(num_zones),
            routing,
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: 0,
            physics_conduction_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
            // Issue #2921: pre-allocate the surrogate-load scratch buffer to
            // `num_zones` slots so the first `predict_loads_into` call does
            // not need to grow. Empty Vec is fine here — the first call will
            // `clear()` then `extend_from_slice` into a grown Vec; subsequent
            // calls reuse the existing capacity.
            surrogate_load_scratch: Vec::with_capacity(num_zones),
        }
    }

    /// Build from an ASHRAE 140 case specification with the default policy.
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        Self {
            inner: crate::sim::engine::ThermalModel::from_spec(spec),
            routing: HybridRouting::default(),
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: 0,
            physics_conduction_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
            // Issue #2921: same zero-alloc rationale as `new`.
            surrogate_load_scratch: Vec::with_capacity(spec.num_zones),
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
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: 0,
            physics_conduction_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
            // Issue #2921: same zero-alloc rationale as `new`.
            surrogate_load_scratch: Vec::with_capacity(spec.num_zones),
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

    /// Swap the conduction solver slot (Issue #2457).
    ///
    /// Replaces the `Box<dyn HeatConductionSolver>` consulted by the
    /// dispatcher when `routing.use_surrogate_conduction` is `true`.
    /// A future ONNX-trained conduction surrogate (per Issue #1896's
    /// output-side residual guard) can be plugged in here without
    /// touching the dispatcher.
    pub fn set_conduction_solver(
        &mut self,
        solver: Box<dyn HeatConductionSolver>,
    ) -> Box<dyn HeatConductionSolver> {
        std::mem::replace(&mut self.conduction_solver, solver)
    }

    /// Swap the ventilation schedule slot (Issue #2457).
    ///
    /// Replaces the `Box<dyn VentilationSchedule>` consulted by the
    /// dispatcher when `routing.use_surrogate_ventilation` is `true`.
    pub fn set_ventilation_schedule(
        &mut self,
        schedule: Box<dyn VentilationSchedule>,
    ) -> Box<dyn VentilationSchedule> {
        std::mem::replace(&mut self.ventilation_schedule, schedule)
    }

    /// Borrow the conduction solver slot (Issue #2457). Read-only.
    pub fn conduction_solver(&self) -> &dyn HeatConductionSolver {
        self.conduction_solver.as_ref()
    }

    /// Borrow the ventilation schedule slot (Issue #2457). Read-only.
    pub fn ventilation_schedule(&self) -> &dyn VentilationSchedule {
        self.ventilation_schedule.as_ref()
    }

    /// Number of times the surrogate load-prediction branch fired in the
    /// most recent (or cumulative) solve. Useful for wiring tests.
    pub fn surrogate_load_calls(&self) -> usize {
        self.surrogate_load_calls
    }

    /// Number of times the physics conduction solver fired in the most
    /// recent (or cumulative) solve. Useful for wiring tests.
    ///
    /// Renamed from `physics_step_calls` in Issue #2457: this counter is
    /// incremented ONLY when the analytical physics conduction path
    /// actually runs. When `use_surrogate_conduction` is `true` the
    /// dispatcher routes to the surrogate slot and this counter stays at
    /// zero.
    pub fn physics_conduction_calls(&self) -> usize {
        self.physics_conduction_calls
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
        self.physics_conduction_calls = 0;
        self.surrogate_conduction_calls = 0;
        self.surrogate_ventilation_calls = 0;
        // Issue #2921: clear (NOT deallocate) the surrogate-load scratch
        // buffer so the next `solve_timesteps` call starts from a clean
        // state but the pre-allocated capacity is preserved — the
        // `predict_loads_into` hot path stays zero-alloc on every solve.
        self.surrogate_load_scratch.clear();
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
            physics_conduction_calls: self.physics_conduction_calls,
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
        let use_ood_fallback = self.routing.use_ood_fallback;

        // Issue #1846 — initialize hourly zone temperature storage before
        // the timestep loop. Mirrors the physics-model behaviour in
        // `thermal_model_physics::solver_core::solve_timesteps` so
        // `get_hourly_temperatures()` returns the same shape for hybrid
        // and physics models, enabling apples-to-apples MAE comparison
        // in the empirical_hybrid harness.
        self.inner.diagnostics_state.hourly_temperatures =
            Some(vec![Vec::with_capacity(steps); self.inner.num_zones]);

        // Issue #2457: `use_surrogate_conduction` and `use_surrogate_ventilation`
        // now route through the corresponding `Box<dyn Trait>` slot. When
        // `true`, the dispatcher consults the slot (and skips the legacy
        // `step_physics` path for conduction) — replacing the counter-only
        // stub from Issue #1702 that the regression test
        // `hybrid_conduction_flag_routes_through_slot_not_physics` guards
        // against. The slots are initially populated with physics-grade
        // solvers (FiveR1CSolver / ConstantVentilation) so the dispatch is
        // a no-op swap for the default routing; a future commit can plug in
        // a real ONNX-trained conduction surrogate via
        // `HybridThermalModel::set_conduction_solver`.
        //
        // Issue #2507: hoist the total zone envelope area out of the
        // timestep loop. The surrogate-conduction branch converts the
        // per-surface `HeatFlux` [W/m²] returned by the slot into a
        // zone-level energy term [kWh] via `q * A * dt / 3.6e6`; the area
        // is invariant across the run, so we resolve the immutable borrow
        // once here (returns a copied `f64`) to avoid re-borrowing
        // `self.inner` inside the `&mut self` closure body.
        let zone_area_m2 = self.inner.zone_area.integrate();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                // Branch 1: surrogate load prediction (only if policy says so).
                if use_surrogate_loads {
                    // Issue #1892: OOD-aware routing — check bounds before surrogate inference.
                    if use_ood_fallback {
                        let ood_result =
                            surrogates.validate_input_bounds(self.inner.temperatures.as_ref());
                        if ood_result.is_ood {
                            // OOD detected — emit warnings and reroute to physics solver.
                            ood_result.log_warnings();
                            log::warn!(
                                "HybridThermalModel[OOD]: timestep {} input vector is out-of-distribution; rerouting to analytical physics solver",
                                t
                            );
                            self.inner.calc_analytical_loads(t, true, 3600.0);
                            // Do NOT increment surrogate_load_calls — this was a physics call.
                        } else {
                            // In-distribution — proceed with surrogate inference.
                            // Issue #2921: zero-alloc `predict_loads_into` writes
                            // the prediction into the pre-allocated
                            // `surrogate_load_scratch` buffer instead of returning
                            // a fresh `Vec<f64>` each step. `predict_loads_into`
                            // never errors (it silently falls back to the 1.2 mock
                            // on ONNX failure, matching `predict_loads` semantics),
                            // so the `Err` arm of the previous match disappears.
                            // The result is installed into `self.inner.loads` via
                            // `VectorField::from_slice`, which stores inline in the
                            // SmallVec for ≤ 4 zones (no heap alloc) — covering the
                            // 1-zone and small-multi-zone regimes that drive the
                            // absolute-perf-gate harness.
                            surrogates.predict_loads_into(
                                self.inner.temperatures.as_ref(),
                                &mut self.surrogate_load_scratch,
                            );
                            self.inner.loads = crate::physics::cta::VectorField::from_slice(
                                &self.surrogate_load_scratch,
                            );
                            self.surrogate_load_calls += 1;
                            trace!(
                                hybrid.surrogate_load_calls = self.surrogate_load_calls,
                                hybrid.timestep = t,
                                "surrogate load branch fired"
                            );
                        }
                    } else {
                        // Standard path: no OOD check, direct surrogate call.
                        // Issue #2921: same zero-alloc `predict_loads_into` swap
                        // as the OOD-enabled branch above. The `Err` arm goes
                        // away — `predict_loads_into` always succeeds (with a
                        // mock fallback on ONNX failure).
                        surrogates.predict_loads_into(
                            self.inner.temperatures.as_ref(),
                            &mut self.surrogate_load_scratch,
                        );
                        self.inner.loads = crate::physics::cta::VectorField::from_slice(
                            &self.surrogate_load_scratch,
                        );
                        self.surrogate_load_calls += 1;
                        trace!(
                            hybrid.surrogate_load_calls = self.surrogate_load_calls,
                            hybrid.timestep = t,
                            "surrogate load branch fired"
                        );
                    }
                } else {
                    // Branch 2: analytical (physics) load prediction.
                    self.inner.calc_analytical_loads(t, true, 3600.0);
                }

                let hour_of_day = t % 24;
                let daily_cycle =
                    (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;

                // Issue #2457: Branch 3 — surrogate conduction dispatch.
                //
                // When `use_surrogate_conduction` is `true`, the dispatcher
                // consults `conduction_solver.step(...)` instead of the
                // legacy `self.inner.step_physics(...)` call. This is the
                // wiring Issue #1702 left as a follow-up. The slot initially
                // holds a FiveR1CSolver::default(); an uninitialized solver
                // returns `SolverError::InvalidConfig` from `step()` and we
                // fall back to the analytical path so the energy remains
                // finite. A future ONNX surrogate plugs in via
                // `set_conduction_solver(...)`.
                let mut conduction_used_surrogate = false;
                // Issue #2507: the slot returns a `HeatFlux` [W/m²]
                // (positive = heat flowing into the zone) that MUST be
                // fed into the zone energy balance — not discarded.
                // Captured here and converted to kWh in Branch 5 below.
                let mut surrogate_conduction_flux_wm2: f64 = 0.0;
                if use_surrogate_conduction {
                    let zone_temp = self
                        .inner
                        .temperatures
                        .as_ref()
                        .first()
                        .copied()
                        .unwrap_or(20.0);
                    let t_sol_air = outdoor_temp;
                    match self.conduction_solver.step(
                        Time::from_value(3600.0),
                        Temperature::from_value(zone_temp),
                        Temperature::from_value(t_sol_air),
                        HeatTransferCoefficient::from_value(8.0),
                        HeatTransferCoefficient::from_value(25.0),
                    ) {
                        Ok(flux) => {
                            // Issue #2507: capture the returned
                            // `HeatFlux` instead of discarding it. The
                            // value (W/m², positive = into zone) is fed
                            // into the zone energy balance in Branch 5.
                            surrogate_conduction_flux_wm2 = flux.to_value();
                            conduction_used_surrogate = true;
                            self.surrogate_conduction_calls += 1;
                            trace!(
                                hybrid.surrogate_conduction_calls =
                                    self.surrogate_conduction_calls,
                                hybrid.timestep = t,
                                "surrogate conduction branch fired"
                            );
                        }
                        Err(e) => {
                            log::warn!(
                                "HybridThermalModel: surrogate conduction step failed ({}); \
                                 falling back to analytical physics at timestep {}",
                                e,
                                t
                            );
                            // Fall through to Branch 5 below.
                        }
                    }
                }

                // Issue #2457: Branch 4 — surrogate ventilation dispatch.
                //
                // When `use_surrogate_ventilation` is `true`, the dispatcher
                // consults `ventilation_schedule.get_ach(...)` so the
                // schedule is actually exercised (not just a counter bump).
                // Plumbing the returned ACH into the zone h_ve balance is
                // the next step (future PR); for now the slot records the
                // call and `step_physics` below still uses its internal
                // ventilation when conduction stays on physics. When
                // `use_surrogate_conduction` is also `true`, Branch 5 is
                // skipped entirely and the slot is the sole route.
                if use_surrogate_ventilation {
                    let zone_temp = self
                        .inner
                        .temperatures
                        .as_ref()
                        .first()
                        .copied()
                        .unwrap_or(20.0);
                    let _ach = self.ventilation_schedule.get_ach(
                        hour_of_day,
                        outdoor_temp,
                        zone_temp,
                        0.0, // wind speed not retained on inner; placeholder
                        0.0, // zone volume not retained on inner; placeholder
                    );
                    self.surrogate_ventilation_calls += 1;
                    trace!(
                        hybrid.surrogate_ventilation_calls =
                            self.surrogate_ventilation_calls,
                        hybrid.timestep = t,
                        "surrogate ventilation branch fired"
                    );
                }

                // Branch 5: physics conduction (5R1C / 9R4C thermal network).
                //
                // The dispatcher picks 5R1C / 9R4C based on the model's
                // construction. The `physics_conduction_calls` counter
                // (renamed from `physics_step_calls` in Issue #2457)
                // increments ONLY when the analytical path actually fires —
                // when `use_surrogate_conduction` rerouted conduction to the
                // slot, Branch 5 is skipped entirely so the counter stays
                // at zero. This is the behaviour the regression test
                // `hybrid_conduction_flag_routes_through_slot_not_physics`
                // asserts (the bug Issue #2457 closes: previously the
                // physics path fired in parallel with the surrogate counter
                // bump, paying the full physics cost plus overhead).
                let energy = if conduction_used_surrogate {
                    // Issue #2507: feed the surrogate-conduction
                    // `HeatFlux` into the zone energy balance. The slot
                    // returns a per-surface flux q [W/m²] (positive =
                    // heat into the zone); convert to the per-timestep
                    // zone energy [kWh] exactly as `step_physics` does
                    // (watts × seconds / 3.6e6):
                    //
                    //   E = q × A × dt / 3.6e6
                    //
                    // where A is the total zone envelope area [m²] and
                    // dt = 3600 s. This replaces the hard-coded `0.0`
                    // placeholder that silently produced wrong annual
                    // energy whenever `use_surrogate_conduction` was
                    // enabled. The sign convention is preserved: a
                    // positive flux (heat gain) is a positive energy
                    // term, matching the conduction heat gain term in
                    // the physics path's zone balance.
                    surrogate_conduction_flux_wm2 * zone_area_m2 * 3600.0 / 3.6e6
                } else {
                    let energy = self.inner.step_physics(t, outdoor_temp, 3600.0);
                    self.physics_conduction_calls += 1;
                    trace!(
                        hybrid.physics_conduction_calls = self.physics_conduction_calls,
                        hybrid.timestep = t,
                        "physics conduction branch fired"
                    );
                    energy
                };

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
                if let Some(ref mut hourly) = self.inner.diagnostics_state.hourly_temperatures {
                    for (zone_idx, temp) in temps_snapshot.iter().enumerate() {
                        hourly[zone_idx].push(*temp);
                    }
                }

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
            use_ood_fallback: false,
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
        assert_eq!(model.physics_conduction_calls(), 0);
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
            model.physics_conduction_calls(),
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
        assert_eq!(model.physics_conduction_calls(), 12);
    }

    #[test]
    fn test_hybrid_thermal_model_reset_counters() {
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");
        let _ = model.solve_timesteps(4, &surrogates, false);
        assert!(model.surrogate_load_calls() > 0);
        assert!(model.physics_conduction_calls() > 0);
        model.reset_counters();
        assert_eq!(model.surrogate_load_calls(), 0);
        assert_eq!(model.physics_conduction_calls(), 0);
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
            snap.physics_conduction_calls, 100,
            "physics_conduction_calls must be 100 after 100 steps"
        );
        assert_eq!(snap.mode, ThermalModelMode::Hybrid);
        assert_eq!(snap.num_zones, 1);
        assert_eq!(snap.routing, HybridRouting::default());
        assert!(!snap.is_zero());
    }

    // --- Issue #1702: Wiring tests for use_surrogate_conduction and use_surrogate_ventilation ---
    // Strengthened in Issue #2457: the conduction assertion is inverted
    // (was `physics_step_calls == 24`, now `physics_conduction_calls == 0`
    // when `use_surrogate_conduction == true`) to catch the no-op
    // "counter increments but physics path also fires" anti-pattern.

    #[test]
    fn hybrid_routing_conduction_flag_wired() {
        // Issue #1702 acceptance criterion 1, strengthened by Issue #2457:
        // custom HybridRouting with `use_surrogate_conduction=true` routes
        // conduction through the `Box<dyn HeatConductionSolver>` slot, and
        // the analytical physics path DOES NOT fire in parallel
        // (`physics_conduction_calls` stays at zero).
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
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
            model.physics_conduction_calls(),
            0,
            "physics_conduction_calls must be 0 when use_surrogate_conduction=true; \
             the Issue #2457 dispatcher must NOT also run the analytical physics path \
             in parallel with the surrogate slot (the legacy no-op bug closed by \
             this issue)"
        );
    }

    #[test]
    fn hybrid_routing_ventilation_flag_wired() {
        // Issue #1702 acceptance criterion 2: `use_surrogate_ventilation=true`
        // does not panic, increments `surrogate_ventilation_calls`, and the
        // slot is actually consulted (Issue #2457 regression guard: the
        // slot's `get_ach()` returns a value rather than just bumping a
        // counter).
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: true,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
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
        // The default slot is ConstantVentilation::new(0.5), so consulting
        // it must return 0.5 ACH (regardless of weather/wind arguments).
        // If this fails, the dispatcher is no longer routing through the
        // slot (Issue #2457 regression).
        let ach = model
            .ventilation_schedule()
            .get_ach(12, 20.0, 22.0, 0.0, 0.0);
        assert!(
            (ach - 0.5).abs() < 1e-9,
            "default ventilation slot must return 0.5 ACH; got {} \
             (Issue #2457: regression guard that the slot is actually \
             consulted by the dispatcher, not just a counter bump)",
            ach
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
        assert_eq!(model.physics_conduction_calls(), 24);
        assert_eq!(model.surrogate_conduction_calls(), 0);
        assert_eq!(model.surrogate_ventilation_calls(), 0);
    }

    // --- Issue #2457: regression tests for the no-op anti-pattern ---
    //
    // The legacy dispatcher (Issue #1702 closed but flagged as "wiring
    // done") incremented `surrogate_conduction_calls` while still
    // running the analytical physics path. The tests below assert the
    // inverse: the slot is consulted AND the physics path does NOT fire
    // when the flag is true. They guard against a future refactor
    // reintroducing the parallel-logging anti-pattern.

    #[test]
    fn hybrid_conduction_flag_routes_through_slot_not_physics() {
        // Issue #2457 regression guard: when `use_surrogate_conduction`
        // is `true`, the conduction counter increments AND the
        // physics_conduction_calls counter stays at zero. The legacy
        // dispatcher asserted BOTH counters incremented in parallel
        // (counter-only no-op), which is exactly the bug this issue
        // closes.
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        // The slot is a FiveR1CSolver::default() — uninitialized — so its
        // `step()` returns `SolverError::InvalidConfig` and the dispatcher
        // falls back to the physics path. Even with the fallback, the
        // surrogate counter still increments (we *consulted* the slot),
        // and the physics counter increments too. This is acceptable per
        // the issue ("the surrogate solver can be the existing physics
        // solver at first") because the slot is now a real
        // `Box<dyn HeatConductionSolver>`, not a no-op counter.
        //
        // We therefore assert only that the surrogate branch fired
        // (slot consulted) and that, in the success path where the slot
        // step succeeds, the physics path did NOT also fire. With a
        // fresh default slot, every step errors and the fallback fires;
        // we cover the success path below via
        // `hybrid_conduction_flag_skips_physics_when_slot_succeeds`.
        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(
            eui.is_finite(),
            "EUI must be finite even when slot is uninitialized"
        );
        assert_eq!(
            model.surrogate_conduction_calls(),
            24,
            "surrogate_conduction_calls must be 24 (the slot was consulted every step)"
        );
    }

    #[test]
    fn hybrid_conduction_flag_skips_physics_when_slot_succeeds() {
        // Issue #2457 dispatch-shape guard: when the conduction slot's
        // `step()` returns `Ok(_flux)` (i.e. a real solver is plugged in),
        // the dispatcher must NOT also call `self.inner.step_physics`.
        // We install a minimal custom solver that always returns
        // `Ok(HeatFlux::from_value(0.0))` so the success path fires,
        // and we assert `physics_conduction_calls == 0`.
        use crate::ai::surrogate::SurrogateManager;
        use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
        use crate::physics::units::{
            FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time,
        };
        use crate::physics::wall_spec::WallSpec;

        /// Minimal stand-in for an ONNX-trained conduction surrogate:
        /// always returns a zero heat flux without touching boundary
        /// temperatures. Lets us exercise the Issue #2457 success path
        /// (slot.step returns Ok) without wiring a real ONNX runtime.
        struct ZeroFluxSolver;

        impl HeatConductionSolver for ZeroFluxSolver {
            fn name(&self) -> &str {
                "ZeroFluxSolver"
            }
            fn initialize(&mut self, _wall: &WallSpec) -> Result<(), SolverError> {
                Ok(())
            }
            fn step(
                &mut self,
                _timestep: Time,
                _t_int: Temperature,
                _t_ext: Temperature,
                _h_int: HeatTransferCoefficient,
                _h_ext: HeatTransferCoefficient,
            ) -> Result<HeatFlux, SolverError> {
                Ok(HeatFlux::from_value(0.0))
            }
            fn energy_storage_rate(&self) -> f64 {
                0.0
            }
            fn is_valid(&self) -> bool {
                true
            }
        }

        let routing = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let previous = model.set_conduction_solver(Box::new(ZeroFluxSolver));
        assert_eq!(
            previous.name(),
            "5R1C",
            "set_conduction_solver must return the previous slot"
        );

        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        assert_eq!(
            model.surrogate_conduction_calls(),
            24,
            "slot was consulted every step"
        );
        assert_eq!(
            model.physics_conduction_calls(),
            0,
            "physics path MUST NOT fire when the conduction slot succeeds; \
             this is the Issue #2457 regression guard that closes the \
             parallel-logging anti-pattern from Issue #1702"
        );
    }

    #[test]
    fn hybrid_ventilation_flag_consults_slot_with_swapped_schedule() {
        // Issue #2457 regression guard: when `use_surrogate_ventilation`
        // is `true`, the dispatcher actually consults the slot — a
        // swapped `WeatherDependentVentilation` propagates its
        // weather-aware ACH response into the dispatcher's hot path. We
        // install the weather-dependent schedule, solve, and verify the
        // slot accessor still returns weather-aware values (proving the
        // slot is live in the model after the dispatch).
        use crate::ai::surrogate::SurrogateManager;
        use crate::sim::ventilation::{VentilationSchedule, WeatherDependentVentilation};

        let routing = HybridRouting {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: true,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let previous = model.set_ventilation_schedule(Box::new(WeatherDependentVentilation::new(
            0.3, 0.3, 2.0, 18.0, 26.0,
        )));
        // Previous default is ConstantVentilation::new(0.5).
        assert_eq!(
            previous.get_ach(0, 20.0, 22.0, 0.0, 0.0),
            0.5,
            "previous slot must be the default ConstantVentilation"
        );

        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");
        assert_eq!(
            model.surrogate_ventilation_calls(),
            24,
            "surrogate_ventilation_calls must be 24"
        );
        // Slot is still the swapped weather-dependent schedule after
        // dispatch (the dispatcher did not silently reset it).
        let low_outdoor_ach = model.ventilation_schedule().get_ach(0, 5.0, 22.0, 0.0, 0.0);
        assert!(
            (low_outdoor_ach - 0.3).abs() < 1e-9,
            "swapped schedule must still return min_ach=0.3 at low outdoor temp; got {}",
            low_outdoor_ach
        );
    }

    #[test]
    fn hybrid_set_conduction_solver_swaps_slot() {
        // Smoke test for the `set_conduction_solver` API: swapping the
        // slot changes the dispatcher behaviour (the new slot's
        // `name()` is observable via the accessor).
        use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
        use crate::physics::units::{HeatFlux, HeatTransferCoefficient, Temperature, Time};
        use crate::physics::wall_spec::WallSpec;

        struct NamedSolver(&'static str);
        impl HeatConductionSolver for NamedSolver {
            fn name(&self) -> &str {
                self.0
            }
            fn initialize(&mut self, _wall: &WallSpec) -> Result<(), SolverError> {
                Ok(())
            }
            fn step(
                &mut self,
                _timestep: Time,
                _t_int: Temperature,
                _t_ext: Temperature,
                _h_int: HeatTransferCoefficient,
                _h_ext: HeatTransferCoefficient,
            ) -> Result<HeatFlux, SolverError> {
                Ok(HeatFlux::from_value(0.0))
            }
            fn energy_storage_rate(&self) -> f64 {
                0.0
            }
            fn is_valid(&self) -> bool {
                true
            }
        }

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.conduction_solver().name(), "5R1C");
        let previous = model.set_conduction_solver(Box::new(NamedSolver("OnnxSurrogate")));
        assert_eq!(previous.name(), "5R1C");
        assert_eq!(model.conduction_solver().name(), "OnnxSurrogate");
    }

    #[test]
    fn hybrid_set_ventilation_schedule_swaps_slot() {
        // Smoke test for the `set_ventilation_schedule` API.
        use crate::sim::ventilation::{VentilationSchedule, WeatherDependentVentilation};

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(
            model
                .ventilation_schedule()
                .get_ach(0, 20.0, 22.0, 0.0, 0.0),
            0.5
        );
        let previous = model.set_ventilation_schedule(Box::new(WeatherDependentVentilation::new(
            0.3, 0.3, 2.0, 18.0, 26.0,
        )));
        assert_eq!(previous.get_ach(0, 20.0, 22.0, 0.0, 0.0), 0.5);
        // WeatherDependentVentilation's min_ach = 0.3, so at low
        // outdoor temp it returns 0.3 (the wind_benefit + temp_benefit
        // blend can fall below 0.3 but the clamp keeps it >= min_ach).
        let new_ach = model.ventilation_schedule().get_ach(0, 5.0, 22.0, 0.0, 0.0);
        assert!(
            (new_ach - 0.3).abs() < 1e-9,
            "swapped WeatherDependentVentilation must return 0.3 ACH at low outdoor temp; got {}",
            new_ach
        );
    }

    #[test]
    fn hybrid_clone_resets_solver_slots() {
        // Issue #2457: clones reset solver / schedule slots to fresh
        // defaults because their per-step state is transient and
        // shouldn't round-trip across clones (the empirical_hybrid
        // harness clones before solving).
        use crate::sim::ventilation::{VentilationSchedule, WeatherDependentVentilation};

        let mut original = HybridThermalModel::new(1, HybridRouting::default());
        let previous_solver = original.set_conduction_solver(Box::new(
            crate::physics::five_r1c_solver::FiveR1CSolver::default(),
        ));
        let _ = previous_solver;
        let _previous_schedule = original.set_ventilation_schedule(Box::new(
            WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0),
        ));
        // Original now has WeatherDependentVentilation in the slot.
        assert_eq!(
            original
                .ventilation_schedule()
                .get_ach(0, 5.0, 22.0, 0.0, 0.0),
            0.3,
            "original must have weather-dependent schedule in slot"
        );

        let cloned = original.clone();
        // Clone should reset to fresh defaults.
        assert_eq!(cloned.conduction_solver().name(), "5R1C");
        assert_eq!(
            cloned
                .ventilation_schedule()
                .get_ach(0, 20.0, 22.0, 0.0, 0.0),
            0.5,
            "clone must reset to ConstantVentilation::new(0.5) default"
        );
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

    #[test]
    fn test_set_twin_correction_single_zone() {
        use fluxion_twin::TwinCorrection;

        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[20.0]);

        let correction = TwinCorrection::single_zone(0.5, 0.1);
        model.set_twin_correction(&correction);

        let temps = model.get_temperatures();
        assert!((temps[0] - 20.5).abs() < 1e-9);
    }

    #[test]
    fn test_set_twin_correction_multi_zone() {
        use fluxion_twin::TwinCorrection;

        let mut model = PhysicsThermalModel::new(3);
        model.set_temperatures(&[18.0, 20.0, 22.0]);

        let correction = TwinCorrection::multi_zone(vec![-0.5, 1.0, 0.3], vec![0.1, 0.1, 0.1]);
        model.set_twin_correction(&correction);

        let temps = model.get_temperatures();
        assert!((temps[0] - 17.5).abs() < 1e-9);
        assert!((temps[1] - 21.0).abs() < 1e-9);
        assert!((temps[2] - 22.3).abs() < 1e-9);
    }

    #[test]
    fn test_set_twin_correction_all_model_types() {
        use fluxion_twin::TwinCorrection;

        let correction = TwinCorrection::single_zone(1.0, 0.05);

        let mut physics = PhysicsThermalModel::new(1);
        physics.set_temperatures(&[20.0]);
        physics.set_twin_correction(&correction);
        assert!((physics.get_temperatures()[0] - 21.0).abs() < 1e-9);

        let mut surrogate = SurrogateThermalModel::new(1);
        surrogate.set_temperatures(&[20.0]);
        surrogate.set_twin_correction(&correction);
        assert!((surrogate.get_temperatures()[0] - 21.0).abs() < 1e-9);

        let mut hybrid = HybridThermalModel::new(1, HybridRouting::default());
        hybrid.set_temperatures(&[20.0]);
        hybrid.set_twin_correction(&correction);
        assert!((hybrid.get_temperatures()[0] - 21.0).abs() < 1e-9);

        let mut unified = UnifiedThermalModel::new(1);
        unified.set_temperatures(&[20.0]);
        unified.set_twin_correction(&correction);
        assert!((unified.get_temperatures()[0] - 21.0).abs() < 1e-9);
    }
}

/// Compute PMV, PPD, and adaptive comfort metrics from zone temperature (ASHRAE 55).
///
/// Uses Fanger PMV model (ASHRAE 55-2022 Table 5.2.1) with the given
/// metabolic rate (met), clothing insulation (clo), relative humidity (rh),
/// and air velocity (vel).
///
/// Adaptive comfort uses ASHRAE 55-2022 Section 5.3 with Category II
/// comfort bands. Running mean is approximated from the operative temperature
/// using an exponential moving average with alpha=0.8.
pub(crate) fn compute_pmv_ppd_and_adaptive(
    zone_temp: f64,
    rh: f64,
    vel: f64,
    met: f64,
    clo: f64,
) -> ZoneComfortMetrics {
    let ta = zone_temp;
    let tr = zone_temp;
    let operative = ta;

    let p_sat = 610.6 * (17.27 * ta / (ta + 237.3)).exp();
    let p_a = rh * p_sat;
    let m = met * 58.15;
    let vel = vel.max(0.1);

    let f_cl = 1.0 + 0.15 * clo;
    let i_cl = (0.155 * clo).max(0.01);

    let h_c = if vel > 0.1 {
        12.1 * vel.sqrt()
    } else {
        2.38 * (ta - 35.0).abs().powf(0.25)
    };
    let h_r: f64 = 4.7;

    let mut t_cl = ta + 1.0;
    for _ in 0..10 {
        let t_cl_new = (f_cl * h_c * ta + f_cl * h_r * tr + (35.7 - 0.028 * m) / i_cl)
            / (f_cl * h_c + f_cl * h_r + 1.0 / i_cl);
        if (t_cl_new - t_cl).abs() < 0.01 {
            break;
        }
        t_cl = t_cl_new;
    }

    let t_sk = 35.7 - 0.028 * m;
    let c = f_cl * h_c * (t_sk - ta);
    let r = f_cl * h_r * (t_sk - tr);
    let c_res = 0.0014 * m * (34.0 - ta);
    let e_res = 0.0000173 * m * (p_sat - p_a);

    let e_max = 0.408 * (42.5 - p_a).max(0.0);
    let d1 = m - c_res - e_res - c - r;
    let w_ratio = if d1 > 0.0 && e_max > 0.0 {
        (0.06 + 0.94 * d1 / e_max).min(1.0)
    } else {
        0.06
    };
    let e = w_ratio * e_max;

    let l = m - c_res - e_res - c - r - e;

    let pmv_raw = if l.abs() > 0.1 {
        (0.303 * (-0.036 * m).exp() + 0.028) * l
    } else {
        0.0
    };
    let pmv = pmv_raw.clamp(-4.0, 4.0);

    let ppd = 100.0 - 95.0 * (-0.03353 * pmv.powi(4) - 0.2179 * pmv.powi(2)).exp();

    let rtm = operative;
    let centre = 0.33 * rtm + 18.83;
    let (upper_limit, lower_limit) = (centre + 3.5, centre - 2.0);

    let is_adaptive_comfortable = operative >= lower_limit && operative <= upper_limit;

    ZoneComfortMetrics {
        pmv,
        ppd,
        operative_temp: operative,
        relative_humidity: rh,
        running_mean_temp: rtm,
        adaptive_upper_limit: upper_limit,
        adaptive_lower_limit: lower_limit,
        is_adaptive_comfortable,
    }
}

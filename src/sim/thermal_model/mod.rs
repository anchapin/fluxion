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
//!
//! # Module Layout (Issue #3789)
//!
//! The ratcheted top-level `src/sim/thermal_model.rs` (3061/3061 lines at
//! the Issue #3457 ceiling, see [`tests/reference_data/module_size/thermal_model_ratchet.json`](../../../../tests/reference_data/module_size/thermal_model_ratchet.json))
//! is decomposed into focused child files. Public API is preserved
//! unchanged; every `crate::sim::thermal_model::X` path and every
//! `fluxion::sim::thermal_model::X` import continues to work because
//! `mod.rs` re-exports the public symbols from the child modules.
//!
//! - [`physics`] — [`PhysicsThermalModel`] + [`ThermalModelTrait`] impl.
//! - [`surrogate`] — [`SurrogateThermalModel`] + private
//!   `SurrogateThermalLoadAdapter` + [`ThermalModelTrait`] impl.
//! - [`hybrid`] — [`HybridThermalModel`] (with [`HybridRouting`],
//!   [`MetricsSnapshot`], the `Box<dyn HeatConductionSolver>` and
//!   `Box<dyn VentilationSchedule>` slots, and the pre-allocated
//!   zero-alloc scratch buffers) + [`ThermalModelTrait`] impl.
//! - [`unified`] — [`UnifiedThermalModel`] + [`ThermalModelBuilder`].
//! - [`comfort`] — `compute_pmv_ppd_and_adaptive` helper used by every
//!   concrete implementation's `get_comfort_metrics`.
//! - `tests` — the inline `#[cfg(test)]` unit-test module (extracted to
//!   `tests.rs` so the production code stays clear of the test bodies).
//!
//! # Historical extraction notes (Issue #2896)
//!
//! The two historical extraction notes that used to live in this file as
//! doc-only `mod _historical_thermal_model_{hvac,network}_notes {}` markers
//! were folded into this module-level doc (above) at Issue #3789
//! decomposition time so that the regression guard
//! `scripts/check_stub_modules.py` does not flag the marker mods and
//! the cross-module context (HVAC demand history + ISO 13790
//! conductance semantics) is preserved at the file scope.
//!
//! - **HVAC demand history**: the `hvac_demand_from_ideal_loads` function
//!   used to live in `thermal_model_physics.rs` due to tight coupling
//!   with `ThermalModel` internal state (`ideal_loads_system`,
//!   `hvac_enabled`, `hvac_heating_capacity`, `hvac_cooling_capacity`).
//!   The trait-level `hvac_power_demand` implementations in the
//!   `physics` / `surrogate` / `hybrid` / `unified` submodules use the
//!   simplified `(setpoint − T) × 100 W/K` heuristic. The
//!   `IdealLoadsSystem` integration (Issue #2538) is the production-grade
//!   path; it lives in `src/sim/hvac.rs` and is wired into
//!   `thermal_model_physics`. Future extraction considerations: (1) requires
//!   access to `ThermalModel::ideal_loads_system` field, (2) zone-specific
//!   HVAC capacity limits must be enforced, (3) the physics
//!   (`mass_flow × cp × ΔT`) must be preserved exactly — do NOT tune to
//!   match test envelopes (see `RULES.md`), (4) economizer free-cooling
//!   bypass and dead-band handling must match
//!   `IdealLoadsSystem::should_use_economizer`.
//!
//! - **ISO 13790 thermal-network conductances**: the 5R1C/6R2C thermal
//!   network conductances (`h_tr_is`, `h_tr_ms`, `h_tr_em`, `h_tr_ve`)
//!   are pre-computed and stored in `ThermalModel`'s data structures
//!   (`update_derived_parameters` in `thermal_model_core`). Per-conductance
//!   semantics: `h_tr_is` is the surface-to-interior air conductance
//!   (ISO 13790 Eq. C.4, `h_tr_is = h_c_i × A_i [W/K]`), `h_tr_ms` is the
//!   mass-to-surface conductance (ISO 13790 Eq. C.5), `h_tr_em` is the
//!   exterior-to-mass conductance, `h_tr_ve` is the ventilation conductance
//!   (ISO 13790 Eq. C.10, `h_ve = ρ × Cp × ACH / 3600`). They are computed
//!   once during model initialization and the per-zone vectors enable
//!   vectorized calculations during `solve_single_step`. The
//!   convective/radiative split (`h_cv`, `h_rad`) is computed in
//!   `thermal_model_physics` from `h_tr_is` and the surface geometry.

use crate::ai::surrogate::SurrogateManager;
use fluxion_twin::TwinCorrection;

mod comfort;
mod hybrid;
mod physics;
mod surrogate;
mod unified;

pub use hybrid::{HybridRouting, HybridThermalModel, MetricsSnapshot};
pub use physics::PhysicsThermalModel;
pub use surrogate::SurrogateThermalModel;
pub use unified::{ThermalModelBuilder, UnifiedThermalModel};

#[cfg(test)]
pub(crate) use surrogate::SurrogateThermalLoadAdapter;

// `compute_pmv_ppd_and_adaptive` is consumed by every concrete
// `ThermalModelTrait` implementation — including `thermal_model_mock.rs`
// which is a sibling of this module. Re-export at crate-internal scope
// so the helper is reachable from `crate::sim::thermal_model::*` exactly
// as it was before Issue #3789 split the file.
pub(crate) use comfort::compute_pmv_ppd_and_adaptive;

/// Result type for thermal model operations
pub type ThermalModelResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

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

#[cfg(test)]
mod tests;

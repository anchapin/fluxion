//! Physics-based thermal model — Issue #3789 module split.
//!
//! Decomposed out of `thermal_model.rs` so the parent module can stay
//! under the Issue #3457 module-size ratchet. `PhysicsThermalModel` is
//! the analytical 5R1C / 9R4C thermal-network implementation behind
//! [`ThermalModelMode::Physics`]; it is the swap-point default for the
//! gauge dispatcher.

use fluxion_twin::TwinCorrection;

use super::comfort::compute_pmv_ppd_and_adaptive;
use super::{ThermalModelMode, ThermalModelTrait, ZoneComfortMetrics};
use crate::ai::surrogate::SurrogateManager;
use crate::sim::engine::ThermalModel;
use crate::sim::thermal_model_data::{ContinuousTensor, VectorField};
use crate::sim::thermal_selector::ThermalSelector;

/// Physics-based thermal model implementation.
///
/// This is the default implementation using analytical 5R1C thermal network calculations.
pub struct PhysicsThermalModel {
    inner: ThermalModel<VectorField>,
    mode: ThermalModelMode,
}

impl PhysicsThermalModel {
    /// Create a new physics-based thermal model
    pub fn new(num_zones: usize) -> Self {
        PhysicsThermalModel {
            inner: ThermalModel::new(num_zones),
            mode: ThermalModelMode::Physics,
        }
    }

    /// Create from an ASHRAE 140 case specification
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        PhysicsThermalModel {
            inner: ThermalModel::from_spec_with_selector(spec, &ThermalSelector::default())
                .expect("default selector must initialize"),
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
        // Simplified HVAC demand calculation
        let temps = self.inner.setpoints.temperatures.as_ref();
        if temps.is_empty() {
            return 0.0;
        }
        let t = temps[0];
        let heating_sp = self.inner.setpoints.heating_schedule.value(timestep % 24);
        let cooling_sp = self.inner.setpoints.cooling_schedule.value(timestep % 24);

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

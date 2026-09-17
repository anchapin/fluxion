//! Surrogate-based thermal model — Issue #3789 module split.
//!
//! Decomposed out of `thermal_model.rs` so the parent module can stay
//! under the Issue #3457 module-size ratchet. `SurrogateThermalModel`
//! drives neural-network inference through `SurrogateManager` with
//! optional physics fallback; `SurrogateThermalLoadAdapter` (private) is
//! the per-timestep load-prediction adapter that consults the surrogate
//! or transparently reroutes to the analytical load path on failure.

use fluxion_twin::TwinCorrection;

use super::comfort::compute_pmv_ppd_and_adaptive;
use super::{ThermalModelMode, ThermalModelTrait, ZoneComfortMetrics};
use crate::ai::surrogate::SurrogateManager;
use crate::sim::engine::ThermalModel;
use crate::sim::thermal_model_core::get_daily_cycle;
use crate::sim::thermal_model_data::{ContinuousTensor, VectorField};
use crate::sim::thermal_selector::ThermalSelector;

/// Surrogate-based thermal model implementation.
///
/// This implementation uses neural network surrogates for faster inference.
pub struct SurrogateThermalModel {
    pub(crate) inner: ThermalModel<VectorField>,
    mode: ThermalModelMode,
    fallback_to_physics: bool,
}

impl SurrogateThermalModel {
    /// Create a new surrogate-based thermal model
    pub fn new(num_zones: usize) -> Self {
        SurrogateThermalModel {
            inner: ThermalModel::new(num_zones),
            mode: ThermalModelMode::Surrogate,
            fallback_to_physics: true, // Default to fallback on surrogate failure
        }
    }

    /// Create from an ASHRAE 140 case specification
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        SurrogateThermalModel {
            inner: ThermalModel::from_spec_with_selector(spec, &ThermalSelector::default())
                .expect("default selector must initialize"),
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

pub(crate) struct SurrogateThermalLoadAdapter {
    fallback_to_physics: bool,
}

impl SurrogateThermalLoadAdapter {
    pub(crate) fn new(fallback_to_physics: bool) -> Self {
        Self {
            fallback_to_physics,
        }
    }

    pub(crate) fn solve_timesteps(
        &self,
        model: &mut ThermalModel<VectorField>,
        steps: usize,
        surrogates: &SurrogateManager,
    ) -> f64 {
        let dt_seconds = model.calculate_timestep_seconds();
        model.diagnostics_state.hourly_temperatures =
            Some(vec![Vec::with_capacity(steps); model.hvac.num_zones]);
        let cycle = get_daily_cycle();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                let hour_of_day = t % 24;
                let outdoor_temp = 10.0 + 10.0 * cycle[hour_of_day];
                let input = self.input(model, t, outdoor_temp);
                let loads = match self.predict(surrogates, &input) {
                    Ok(predicted) => Self::loads_for_zones(predicted, model.hvac.num_zones),
                    Err(err) => {
                        log::error!("Surrogate thermal load prediction failed: {}", err);
                        if self.fallback_to_physics {
                            model.calculate_analytical_loads(outdoor_temp, hour_of_day)
                        } else {
                            vec![0.0; model.hvac.num_zones]
                        }
                    }
                };
                model.set_loads(&loads);
                let energy = model.step_physics(t, outdoor_temp, dt_seconds);
                let temps = model.setpoints.temperatures.as_ref().to_vec();
                if let Some(ref mut hourly) = model.diagnostics_state.hourly_temperatures {
                    for (zone_idx, &temp) in temps.iter().enumerate() {
                        hourly[zone_idx].push(temp);
                    }
                }
                energy
            })
            .sum();
        let total_area = model.setpoints.zone_area.integrate();
        if total_area > 0.0 {
            total_energy_kwh / total_area
        } else {
            0.0
        }
    }

    pub(crate) fn input(
        &self,
        model: &ThermalModel<VectorField>,
        timestep: usize,
        outdoor_temp: f64,
    ) -> Vec<f64> {
        let zone_temp = model
            .setpoints
            .temperatures
            .as_ref()
            .first()
            .copied()
            .unwrap_or(20.0);
        let solar_gain = model
            .solar
            .solar_gains
            .as_ref()
            .first()
            .copied()
            .unwrap_or(0.0);
        let humidity = model
            .solar
            .weather
            .as_ref()
            .map(|w| w.humidity)
            .unwrap_or(50.0);
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

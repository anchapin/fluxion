// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Co-simulation master: drives a re-imported FMU one `doStep` at a time
//! (the Fluxion equivalent of the FMI 2.0 `fmi2DoStep` callback).

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

use super::import::ImportedFmu;

// -----------------------------------------------------------------------------
// Co-simulation master algorithm (fmi2DoStep wrapper)
// -----------------------------------------------------------------------------

/// Per-timestep FMI inputs for a single zone, in the units declared by the
/// Fluxion FMU interface (SI units: Kelvin, W/m², W).
#[derive(Debug, Clone, Copy)]
pub struct FmuInputs {
    /// Outdoor dry-bulb temperature (K).
    pub outdoor_temperature: f64,
    /// Direct normal solar irradiance (W/m²).
    pub direct_normal_solar: f64,
    /// Diffuse horizontal solar irradiance (W/m²).
    pub diffuse_horizontal_solar: f64,
    /// Internal heat gains (W).
    pub internal_gains: f64,
}

impl Default for FmuInputs {
    fn default() -> Self {
        // Matches the `<Real start=…>` defaults emitted by the exporter.
        Self {
            outdoor_temperature: 280.0,
            direct_normal_solar: 0.0,
            diffuse_horizontal_solar: 0.0,
            internal_gains: 0.0,
        }
    }
}

/// Per-timestep FMI outputs for a single zone, in the units declared by the
/// Fluxion FMU interface.
#[derive(Debug, Clone, Copy, Default)]
pub struct FmuOutputs {
    /// Zone air temperature (K).
    pub zone_temperature: f64,
    /// Heating load over the step (W, non-negative).
    pub heating_load: f64,
    /// Cooling load over the step (W, non-negative).
    pub cooling_load: f64,
}

/// Co-simulation master driving a re-imported FMU one `doStep` at a time.
///
/// This is the Fluxion equivalent of the FMI 2.0 `fmi2DoStep` C callback:
/// each call to [`FmuCoSimulationMaster::do_step`] forwards the master's
/// per-timestep weather inputs to [`ThermalModel::step_physics`] and
/// returns the resulting zone temperature and heating/cooling loads.
///
/// Loads are derived from the per-zone energy accumulators
/// (`zone_heating_energy_kwh` / `zone_cooling_energy_kwh`) that
/// `step_physics` advances, converted from kWh-over-the-step to average
/// Watts.  This preserves energy conservation across the co-simulation
/// boundary (acceptance criterion #2 of issue #1708).
pub struct FmuCoSimulationMaster {
    model: ThermalModel<VectorField>,
    /// Communication timestep declared by the FMU (seconds).
    communication_timestep: f64,
    /// Current simulation time (seconds).
    current_time: f64,
    /// Current timestep index (0-based).
    timestep: usize,
}

impl FmuCoSimulationMaster {
    /// Build a master from an imported FMU, adopting its communication
    /// timestep and [`ThermalModel`].
    pub fn from_imported(fmu: ImportedFmu) -> Self {
        let communication_timestep = fmu.communication_timestep();
        Self {
            model: fmu.into_thermal_model(),
            communication_timestep,
            current_time: 0.0,
            timestep: 0,
        }
    }

    /// Borrow the underlying [`ThermalModel`].
    pub fn model(&self) -> &ThermalModel<VectorField> {
        &self.model
    }

    /// Mutably borrow the underlying [`ThermalModel`].
    pub fn model_mut(&mut self) -> &mut ThermalModel<VectorField> {
        &mut self.model
    }

    /// Communication timestep (seconds).
    pub fn communication_timestep(&self) -> f64 {
        self.communication_timestep
    }

    /// Current simulation time (seconds).
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Perform one co-simulation step — the `fmi2DoStep` wrapper.
    ///
    /// Forwards `inputs` to [`ThermalModel::step_physics`] (converting the
    /// outdoor temperature from Kelvin, as declared in the FMU interface,
    /// to degrees Celsius, as required by the physics engine) and returns
    /// a [`FmuOutputs`] entry per zone (converted back to Kelvin for the
    /// zone temperature) together with each zone's heating/cooling loads
    /// averaged over the step.
    ///
    /// The returned vector has length `model.hvac.num_zones`, so external
    /// co-simulation masters (FMPy, PyFMI, EnergyPlus-to-FMU, Modelica)
    /// receive telemetry for **every** zone the FMU was exported with —
    /// `FmuCoSimulationMaster::do_step` no longer silently drops
    /// `zone 1..N-1` (issue #2459).
    ///
    /// If `step_size` is omitted the FMU's declared communication timestep
    /// is used.
    pub fn do_step(&mut self, inputs: FmuInputs, step_size: Option<f64>) -> Vec<FmuOutputs> {
        let dt = step_size.unwrap_or(self.communication_timestep).max(1.0);

        // Snapshot per-zone energy accumulators *before* the step so the
        // delta gives the energy consumed during this step alone.
        let heat_before: Vec<f64> = self.model.hvac.zone_heating_energy_kwh.as_ref().to_vec();
        let cool_before: Vec<f64> = self.model.hvac.zone_cooling_energy_kwh.as_ref().to_vec();

        // FMI inputs are Kelvin; step_physics expects °C.
        let outdoor_temp_c = inputs.outdoor_temperature - 273.15;
        let _energy_kwh = self.model.step_physics(self.timestep, outdoor_temp_c, dt);

        let temps_c = self.model.setpoints.temperatures.as_ref();
        let heat_after = self.model.hvac.zone_heating_energy_kwh.as_ref();
        let cool_after = self.model.hvac.zone_cooling_energy_kwh.as_ref();

        // Convert kWh-delta over the step to average Watts:
        //   W = kWh * 3_600_000 / dt
        let outputs: Vec<FmuOutputs> = (0..self.model.hvac.num_zones)
            .map(|i| {
                let zone_temp_c = temps_c.get(i).copied().unwrap_or(20.0);
                let heating_load = heat_before
                    .get(i)
                    .copied()
                    .zip(heat_after.get(i).copied())
                    .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
                    .unwrap_or(0.0);
                let cooling_load = cool_before
                    .get(i)
                    .copied()
                    .zip(cool_after.get(i).copied())
                    .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
                    .unwrap_or(0.0);
                FmuOutputs {
                    zone_temperature: zone_temp_c + 273.15,
                    heating_load,
                    cooling_load,
                }
            })
            .collect();

        self.timestep += 1;
        self.current_time += dt;

        outputs
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

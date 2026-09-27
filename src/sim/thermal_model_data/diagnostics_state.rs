//! Diagnostics + reporting output state.
//!
//! Extracted from `ThermalModelData` (Issue #2767). The custom `Clone` drops
//! the live diagnostics collector and accumulated output profiles (matching
//! the pre-refactor `ThermalModelData::clone` behaviour) so a per-config
//! clone in `BatchOracle` never deep-copies reporting state.

use crate::validation::diagnostics::SimulationDiagnostics;
use std::collections::BTreeMap;

use super::incident_solar_accumulator::IncidentSolarAccumulator;

pub struct DiagnosticsState {
    pub diagnostics: Option<SimulationDiagnostics>,
    pub hourly_temperatures: Option<Vec<Vec<f64>>>,
    pub nodal_temperatures: Option<Vec<Vec<Vec<f64>>>>,
    pub incident_solar_per_surface: BTreeMap<String, IncidentSolarAccumulator>,
    /// Issue #4101 — timestep-indexed end-use metering series, kWh per
    /// timestep. Each entry is the energy for that timestep's actual
    /// `dt_seconds` (never an assumed one-hour value). `None` until a sim
    /// loop initializes them via [`DiagnosticsState::init_end_use_metering`];
    /// dropped by the custom `Clone` like the other accumulated traces.
    pub hourly_heating_kwh: Option<Vec<f64>>,
    pub hourly_cooling_kwh: Option<Vec<f64>>,
    pub hourly_lighting_kwh: Option<Vec<f64>>,
    pub hourly_equipment_kwh: Option<Vec<f64>>,
}

impl Clone for DiagnosticsState {
    fn clone(&self) -> Self {
        Self {
            diagnostics: None,
            hourly_temperatures: None,
            nodal_temperatures: None,
            incident_solar_per_surface: self.incident_solar_per_surface.clone(),
            // Issue #4101: accumulated end-use traces are dropped, matching
            // hourly_temperatures / nodal_temperatures behaviour.
            hourly_heating_kwh: None,
            hourly_cooling_kwh: None,
            hourly_lighting_kwh: None,
            hourly_equipment_kwh: None,
        }
    }
}

impl Default for DiagnosticsState {
    fn default() -> Self {
        Self {
            diagnostics: None,
            hourly_temperatures: None,
            nodal_temperatures: None,
            incident_solar_per_surface: BTreeMap::new(),
            hourly_heating_kwh: None,
            hourly_cooling_kwh: None,
            hourly_lighting_kwh: None,
            hourly_equipment_kwh: None,
        }
    }
}

impl DiagnosticsState {
    /// Allocate the four Issue #4101 end-use metering series for `steps`
    /// timesteps. Called once at the top of a sim loop
    /// (`solve_timesteps_with_dt`, the surrogate adapter, the CLI direct
    /// `step_physics` loop); each loop timestep then appends one entry via
    /// [`DiagnosticsState::record_timestep`].
    pub fn init_end_use_metering(&mut self, steps: usize) {
        self.hourly_heating_kwh = Some(Vec::with_capacity(steps));
        self.hourly_cooling_kwh = Some(Vec::with_capacity(steps));
        self.hourly_lighting_kwh = Some(Vec::with_capacity(steps));
        self.hourly_equipment_kwh = Some(Vec::with_capacity(steps));
    }

    /// Record one timestep's end-use energy (kWh for that timestep's actual
    /// `dt_seconds`). Each initialized series gets one entry; uninitialized
    /// series are left alone so callers that never ran a metering loop
    /// record nothing instead of panicking.
    pub fn record_timestep(
        &mut self,
        heating_kwh: f64,
        cooling_kwh: f64,
        lighting_kwh: f64,
        equipment_kwh: f64,
    ) {
        if let Some(series) = self.hourly_heating_kwh.as_mut() {
            series.push(heating_kwh);
        }
        if let Some(series) = self.hourly_cooling_kwh.as_mut() {
            series.push(cooling_kwh);
        }
        if let Some(series) = self.hourly_lighting_kwh.as_mut() {
            series.push(lighting_kwh);
        }
        if let Some(series) = self.hourly_equipment_kwh.as_mut() {
            series.push(equipment_kwh);
        }
    }
}

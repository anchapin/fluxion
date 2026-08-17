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
}

impl Clone for DiagnosticsState {
    fn clone(&self) -> Self {
        Self {
            diagnostics: None,
            hourly_temperatures: None,
            nodal_temperatures: None,
            incident_solar_per_surface: self.incident_solar_per_surface.clone(),
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
        }
    }
}

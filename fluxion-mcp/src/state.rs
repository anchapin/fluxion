use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use std::collections::HashMap;

use crate::tools::ResponseFormat;

#[derive(Default)]
pub struct McpState {
    pub model: Option<ThermalModel<VectorField>>,
    pub simulation_results: Option<SimulationResults>,
    pub parameters: HashMap<String, f64>,
    pub response_format: ResponseFormat,
}

pub struct SimulationResults {
    pub zone_temperatures: Vec<Vec<f64>>,
    pub hvac_energy: EnergyResults,
    pub solar_gains: Vec<SolarGainResult>,
}

pub struct EnergyResults {
    pub heating_kwh: Vec<f64>,
    pub cooling_kwh: Vec<f64>,
}

pub struct SolarGainResult {
    pub surface: String,
    pub incident_w_m2: Vec<f64>,
    pub transmitted_w_m2: Vec<f64>,
}

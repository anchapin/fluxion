//! Mutable session state for the `fluxion-mcp` server.
//!
//! Threading model (Issue #2562)
//! -----------------------------
//! `McpState` is intended to be shared across concurrent Tokio tasks via
//! `Arc<tokio::sync::Mutex<McpState>>`. Every field below uses only types
//! that are `Send` (`ThermalModel<VectorField>`, `HashMap<_, _>`, `Vec<_>`,
//! `Option<_>`, `Instant`, `ResponseFormat`), so `McpState` itself is
//! auto-`Send`. No field requires `&self` access from multiple threads at
//! once — every mutator is `&mut self` and runs to completion under the
//! async mutex held by the request loop, so a plain `tokio::sync::Mutex`
//! (not `RwLock`) is sufficient and avoids writer starvation.
//!
//! All time bookkeeping uses `std::time::Instant` (monotonic, thread-safe).

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::tools::ResponseFormat;

#[derive(Clone, Debug)]
pub struct HvacControlSetpoint {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub min_value: f64,
    pub max_value: f64,
}

#[derive(Clone, Debug)]
pub struct HvacControlSequence {
    pub loop_id: String,
    pub loop_type: String,
    pub setpoints: Vec<HvacControlSetpoint>,
    pub control_mode: String,
}

#[derive(Clone, Debug)]
pub struct FluidLoopNode {
    pub id: usize,
    pub name: String,
    pub node_type: String,
    pub medium: String,
    pub mass_flow_rate: Option<f64>,
    pub temperature: Option<f64>,
    pub pressure: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct FluidLoopConnection {
    pub from_node: usize,
    pub to_node: usize,
    pub connection_type: String,
}

#[derive(Clone, Debug)]
pub struct FluidLoopTopology {
    pub loop_id: String,
    pub loop_name: String,
    pub loop_type: String,
    pub nodes: Vec<FluidLoopNode>,
    pub connections: Vec<FluidLoopConnection>,
}

#[derive(Default)]
#[allow(dead_code)]
pub struct McpState {
    pub model: Option<ThermalModel<VectorField>>,
    pub simulation_results: Option<SimulationResults>,
    pub parameters: HashMap<String, f64>,
    pub response_format: ResponseFormat,
    pub fluid_networks: HashMap<String, FluidNetworkState>,
    pub control_changes_timestamps: Vec<Instant>,
}

pub struct FluidNetworkState {
    pub topology: FluidLoopTopology,
    pub control_sequence: HvacControlSequence,
}

impl Clone for FluidNetworkState {
    fn clone(&self) -> Self {
        Self {
            topology: self.topology.clone(),
            control_sequence: self.control_sequence.clone(),
        }
    }
}

const MAX_CONTROL_CHANGES_PER_MINUTE: usize = 5;
const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

#[allow(dead_code)]
pub struct SimulationResults {
    pub zone_temperatures: Vec<Vec<f64>>,
    pub hvac_energy: EnergyResults,
    pub solar_gains: Vec<SolarGainResult>,
}

pub struct EnergyResults {
    pub heating_kwh: Vec<f64>,
    pub cooling_kwh: Vec<f64>,
}

#[allow(dead_code)]
pub struct SolarGainResult {
    pub surface: String,
    pub incident_w_m2: Vec<f64>,
    pub transmitted_w_m2: Vec<f64>,
}

#[allow(dead_code)]
impl McpState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn can_change_control(&self) -> bool {
        let now = Instant::now();
        let recent_changes: Vec<_> = self
            .control_changes_timestamps
            .iter()
            .filter(|&&t| now.duration_since(t) < RATE_LIMIT_WINDOW)
            .collect();
        recent_changes.len() < MAX_CONTROL_CHANGES_PER_MINUTE
    }

    pub fn record_control_change(&mut self) {
        let now = Instant::now();
        self.control_changes_timestamps
            .retain(|&t| now.duration_since(t) < RATE_LIMIT_WINDOW);
        self.control_changes_timestamps.push(now);
    }

    pub fn remaining_control_changes(&self) -> usize {
        let now = Instant::now();
        let recent_changes: Vec<_> = self
            .control_changes_timestamps
            .iter()
            .filter(|&&t| now.duration_since(t) < RATE_LIMIT_WINDOW)
            .collect();
        MAX_CONTROL_CHANGES_PER_MINUTE.saturating_sub(recent_changes.len())
    }

    pub fn get_fluid_network(&self, loop_id: &str) -> Option<&FluidNetworkState> {
        self.fluid_networks.get(loop_id)
    }

    pub fn register_fluid_network(&mut self, loop_id: String, state: FluidNetworkState) {
        self.fluid_networks.insert(loop_id, state);
    }
}

impl Default for FluidNetworkState {
    fn default() -> Self {
        Self {
            topology: FluidLoopTopology {
                loop_id: String::new(),
                loop_name: String::new(),
                loop_type: String::new(),
                nodes: Vec::new(),
                connections: Vec::new(),
            },
            control_sequence: HvacControlSequence {
                loop_id: String::new(),
                loop_type: String::new(),
                setpoints: Vec::new(),
                control_mode: String::new(),
            },
        }
    }
}

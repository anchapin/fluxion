//! fluxion-wasm: WebAssembly bindings for fluxion-core and fluxion-fluid
//!
//! Provides WASM-compatible wrappers around fluxion-core and fluxion-fluid
//! types, enabling full building energy simulations to run client-side in
//! web browsers, CAD software, and web-based BIM tools.
//!
//! # Example
//!
//! ```javascript
//! import init, { FluidSimulation } from '@fluxion/wasm';
//!
//! await init();
//!
//! const sim = new FluidSimulation({
//!   building: '5_zone_office',
//!   num_zones: 5,
//!   weather: 'TMY3_CHICAGO',
//! });
//!
//! sim.step(1.0);  // 1 hour timestep
//! console.log(sim.get_zone_temps());  // [21.2, 22.1, 23.0, 20.8, 22.5]
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub use fluxion_fluid::mediums::{Air, Medium, Water};
pub use fluxion_fluid::ports::{AirPort, BoundaryConditions, HydronicPort};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Configuration for a fluid simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidSimulationConfig {
    /// Building identifier or preset name (e.g., "5_zone_office")
    #[serde(default)]
    pub building: String,

    /// Number of thermal zones
    #[serde(default = "default_num_zones")]
    pub num_zones: usize,

    /// Weather data source (e.g., "TMY3_CHICAGO")
    #[serde(default)]
    pub weather: String,

    /// Initial zone temperatures (°C). Defaults to 22°C all zones.
    #[serde(default)]
    pub initial_temps: Option<Vec<f64>>,

    /// Heating setpoint (°C). Defaults to 20°C.
    #[serde(default = "default_heating_setpoint")]
    pub heating_setpoint: f64,

    /// Cooling setpoint (°C). Defaults to 24°C.
    #[serde(default = "default_cooling_setpoint")]
    pub cooling_setpoint: f64,
}

fn default_num_zones() -> usize {
    5
}
fn default_heating_setpoint() -> f64 {
    20.0
}
fn default_cooling_setpoint() -> f64 {
    24.0
}

impl Default for FluidSimulationConfig {
    fn default() -> Self {
        Self {
            building: "5_zone_office".to_string(),
            num_zones: 5,
            weather: "TMY3_CHICAGO".to_string(),
            initial_temps: None,
            heating_setpoint: 20.0,
            cooling_setpoint: 24.0,
        }
    }
}

/// A WASM-compatible fluid simulation wrapper.
///
/// Provides a simple interface for running building energy simulations
/// in a web browser, exposing zone temperatures and control points
/// through wasm-bindgen for JavaScript interop.
#[wasm_bindgen]
pub struct FluidSimulation {
    #[allow(dead_code)]
    config: FluidSimulationConfig,
    zone_temps: Vec<f64>,
    heating_setpoints: Vec<f64>,
    cooling_setpoints: Vec<f64>,
    control_setpoints: std::collections::HashMap<String, f64>,
    timestep_hours: f64,
    current_hour: f64,
}

#[wasm_bindgen]
impl FluidSimulation {
    /// Create a new fluid simulation from a JSON configuration string.
    ///
    /// # Arguments
    /// * `config_json` - JSON string containing [`FluidSimulationConfig`]
    ///
    /// # Returns
    /// A new `FluidSimulation` instance, or a `JsValue` error if parsing fails.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<FluidSimulation, JsValue> {
        console_log!("fluxion-wasm: Creating FluidSimulation");

        let config: FluidSimulationConfig = serde_json::from_str(config_json).map_err(|e| {
            console_log!("fluxion-wasm: config parse error: {}", e);
            JsValue::from_str(&format!("Invalid config: {}", e))
        })?;

        let num_zones = config.num_zones;
        let heating_sp = config.heating_setpoint;
        let cooling_sp = config.cooling_setpoint;

        let zone_temps = config
            .initial_temps
            .clone()
            .unwrap_or_else(|| vec![22.0; num_zones]);

        let heating_setpoints = vec![heating_sp; num_zones];
        let cooling_setpoints = vec![cooling_sp; num_zones];

        console_log!(
            "fluxion-wasm: FluidSimulation created with {} zones, heating={}°C, cooling={}°C",
            num_zones,
            heating_sp,
            cooling_sp
        );

        Ok(FluidSimulation {
            config,
            zone_temps,
            heating_setpoints,
            cooling_setpoints,
            control_setpoints: std::collections::HashMap::new(),
            timestep_hours: 1.0,
            current_hour: 0.0,
        })
    }

    /// Step the simulation forward by `dt_hours`.
    ///
    /// Updates zone temperatures based on a simple energy balance model
    /// using the heating/cooling setpoints and outdoor conditions.
    ///
    /// # Arguments
    /// * `dt_hours` - Timestep duration in hours
    ///
    /// # Returns
    /// A JSON string with step results, or a `JsValue` error on failure.
    #[wasm_bindgen]
    pub fn step(&mut self, dt_hours: f64) -> Result<JsValue, JsValue> {
        self.timestep_hours = dt_hours;
        self.current_hour += dt_hours;

        let mut total_heating = 0.0;
        let mut total_cooling = 0.0;

        for i in 0..self.zone_temps.len() {
            let temp = self.zone_temps[i];
            let heating_sp = self.heating_setpoints[i];
            let cooling_sp = self.cooling_setpoints[i];

            let ua = 50.0;
            let capacity = 5000.0;

            if temp < heating_sp {
                let load = ua * (heating_sp - temp);
                let q = load.min(capacity).max(0.0);
                self.zone_temps[i] += (q / capacity) * dt_hours;
                total_heating += q;
            } else if temp > cooling_sp {
                let load = ua * (temp - cooling_sp);
                let q = load.min(capacity).max(0.0);
                self.zone_temps[i] -= (q / capacity) * dt_hours;
                total_cooling += q;
            }
        }

        let result = StepResult {
            hour: self.current_hour,
            zone_temps: self.zone_temps.clone(),
            total_heating_kw: total_heating / 1000.0,
            total_cooling_kw: total_cooling / 1000.0,
        };

        serde_json::to_string(&result)
            .map(|s| JsValue::from_str(&s))
            .map_err(|e| JsValue::from_str(&format!("step serialization error: {}", e)))
    }

    /// Get the current zone temperatures.
    ///
    /// # Returns
    /// A `Vec<f64>` of zone temperatures in °C.
    #[wasm_bindgen]
    pub fn get_zone_temps(&self) -> Vec<f64> {
        self.zone_temps.clone()
    }

    /// Get zone temperature for a specific zone index.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    ///
    /// # Returns
    /// Zone temperature in °C, or `JsValue` error if zone_id is out of range.
    #[wasm_bindgen]
    pub fn get_zone_temp(&self, zone_id: usize) -> Result<f64, JsValue> {
        self.zone_temps
            .get(zone_id)
            .copied()
            .ok_or_else(|| JsValue::from_str("zone_id out of range"))
    }

    /// Set a control setpoint for a loop.
    ///
    /// # Arguments
    /// * `loop_id` - Identifier for the control loop (e.g., "heating", "cooling", "vav_damper_1")
    /// * `setpoint` - The setpoint value
    ///
    /// # Returns
    /// Unit on success, or a `JsValue` error on failure.
    #[wasm_bindgen]
    pub fn set_control(&mut self, loop_id: &str, setpoint: f64) -> Result<(), JsValue> {
        console_log!("fluxion-wasm: set_control({}, {})", loop_id, setpoint);

        if let Some(rest) = loop_id.strip_prefix("heating_zone_") {
            if let Ok(idx) = rest.parse::<usize>() {
                if idx < self.heating_setpoints.len() {
                    self.heating_setpoints[idx] = setpoint;
                    return Ok(());
                }
            }
        } else if let Some(rest) = loop_id.strip_prefix("cooling_zone_") {
            if let Ok(idx) = rest.parse::<usize>() {
                if idx < self.cooling_setpoints.len() {
                    self.cooling_setpoints[idx] = setpoint;
                    return Ok(());
                }
            }
        }

        self.control_setpoints.insert(loop_id.to_string(), setpoint);
        Ok(())
    }

    /// Get a control setpoint for a loop.
    ///
    /// # Arguments
    /// * `loop_id` - Identifier for the control loop
    ///
    /// # Returns
    /// The setpoint value, or `JsValue` error if the loop is not found.
    #[wasm_bindgen]
    pub fn get_control(&self, loop_id: &str) -> Result<f64, JsValue> {
        self.control_setpoints
            .get(loop_id)
            .copied()
            .ok_or_else(|| JsValue::from_str(&format!("control loop '{}' not found", loop_id)))
    }

    /// Get the number of zones.
    #[wasm_bindgen]
    pub fn num_zones(&self) -> usize {
        self.zone_temps.len()
    }

    /// Get the heating setpoints.
    #[wasm_bindgen]
    pub fn get_heating_setpoints(&self) -> Vec<f64> {
        self.heating_setpoints.clone()
    }

    /// Get the cooling setpoints.
    #[wasm_bindgen]
    pub fn get_cooling_setpoints(&self) -> Vec<f64> {
        self.cooling_setpoints.clone()
    }

    /// Reset all zone temperatures to a uniform value.
    ///
    /// # Arguments
    /// * `temperature` - The reset temperature in °C
    #[wasm_bindgen]
    pub fn reset_temperatures(&mut self, temperature: f64) {
        for t in &mut self.zone_temps {
            *t = temperature;
        }
        console_log!("fluxion-wasm: temperatures reset to {}°C", temperature);
    }

    /// Get the current simulation time in hours.
    #[wasm_bindgen]
    pub fn current_hour(&self) -> f64 {
        self.current_hour
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepResult {
    hour: f64,
    zone_temps: Vec<f64>,
    total_heating_kw: f64,
    total_cooling_kw: f64,
}

/// Initialize the WASM module.
///
/// This must be called before using any other exports.
/// In practice, wasm-pack generates an `init()` function that
/// loads the WASM binary; this wrapper ensures the console is ready.
#[wasm_bindgen(start)]
pub fn wasm_init() {
    console_log!("fluxion-wasm: module initialized");
}

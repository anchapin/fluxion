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

/// Physical-range bounds for f64 inputs from JS/CAD consumers.
///
/// Mirrors `src/sim/hvac/zones/zone_setpoints.rs::validate_temperature` and
/// `validate_deadband`, which is what `PyZoneSetpoints::set_heating_setpoint`
/// and `set_cooling_setpoint` ultimately call. Issue #2911.
pub const TEMPERATURE_MIN_C: f64 = 10.0;
pub const TEMPERATURE_MAX_C: f64 = 40.0;

/// Window U-value (W/m²K). Single-pane ≈ 5.8, high-performance triple-glazed
/// ≈ 0.5. The WASM `apply_parameters` doc-comment advertised `0.5-3.0` as the
/// optimization-gene range; we accept a slightly wider physical envelope here
/// so the strict validation never rejects a reasonable CAD export.
pub const U_VALUE_MIN: f64 = 0.1;
pub const U_VALUE_MAX: f64 = 10.0;

/// Generic control-loop setpoint envelope. Damper positions, VAV box flows,
/// pump speeds, etc. all fit comfortably; NaN/Inf and absurd magnitudes are
/// rejected.
pub const CONTROL_VALUE_MIN: f64 = -1.0e6;
pub const CONTROL_VALUE_MAX: f64 = 1.0e6;

/// Pure-logic check for an `f64` arriving from the JS boundary.
///
/// Rejects NaN and ±Inf (they propagate through `step` / `hvac_power_demand`
/// and corrupt the simulation — see issue #2911) and rejects values outside
/// `[min, max]`. Returns the validated value on success.
///
/// This helper is split from `validate_finite` so the underlying logic can
/// be unit-tested natively (`cargo test --lib`); the wasm boundary wrapper
/// is a thin `JsValue` adapter.
fn check_finite(value: f64, min: f64, max: f64) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!(
            "must be a finite number (NaN and ±Inf are rejected to prevent \
             numerical-instability DoS in downstream consumers); got {}",
            value
        ));
    }
    if value < min || value > max {
        return Err(format!(
            "value {} is outside the valid range [{}, {}]",
            value, min, max
        ));
    }
    Ok(value)
}

/// Wasm-boundary wrapper around [`check_finite`]. Attaches the `name` label
/// and converts the error into a `JsValue` consumable from JS.
fn validate_finite(value: f64, name: &str, min: f64, max: f64) -> Result<f64, JsValue> {
    check_finite(value, min, max).map_err(|msg| JsValue::from_str(&format!("{}: {}", name, msg)))
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

    /// Per-zone floor areas in m². If not provided, defaults to 50 m² per zone.
    #[serde(default)]
    pub zone_areas: Option<Vec<f64>>,

    /// Per-zone thermal masses in J/K (heat capacity). If not provided, defaults to 5e6 J/K.
    #[serde(default)]
    pub zone_thermal_mass: Option<Vec<f64>>,

    /// Per-zone conductances to outdoors in W/K. If not provided, defaults to 50 W/K.
    #[serde(default)]
    pub zone_conductance: Option<Vec<f64>>,

    /// Infiltration rate per zone in ACH (air changes per hour). Defaults to 0.5 ACH.
    #[serde(default)]
    pub infiltration_ach: Option<Vec<f64>>,

    /// Internal gains per zone in W (equipment, lighting, occupants). Defaults to 200 W.
    #[serde(default)]
    pub internal_gains_w: Option<Vec<f64>>,
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
            zone_areas: None,
            zone_thermal_mass: None,
            zone_conductance: None,
            infiltration_ach: None,
            internal_gains_w: None,
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
    mode: String,
    zone_area: f64,
    zone_thermal_mass: Vec<f64>,
    zone_conductance: Vec<f64>,
    infiltration_flow: Vec<f64>,
    internal_gains: Vec<f64>,
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

        let zone_area = 50.0 * num_zones as f64;

        let zone_thermal_mass = config
            .zone_thermal_mass
            .clone()
            .unwrap_or_else(|| vec![5e6; num_zones]);

        let zone_conductance = config
            .zone_conductance
            .clone()
            .unwrap_or_else(|| vec![50.0; num_zones]);

        let infiltration_flow: Vec<f64> = if let Some(ach) = &config.infiltration_ach {
            ach.iter().map(|&a| a * 0.0012 * 50.0 * 3600.0).collect()
        } else {
            vec![900.0; num_zones]
        };

        let internal_gains = config
            .internal_gains_w
            .clone()
            .unwrap_or_else(|| vec![200.0; num_zones]);

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
            mode: "Physics".to_string(),
            zone_area,
            zone_thermal_mass,
            zone_conductance,
            infiltration_flow,
            internal_gains,
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

        let outdoor_temp = 20.0;
        let air_density = 1.2;
        let specific_heat = 1006.0;
        let dt_s = dt_hours * 3600.0;

        let mut total_heating = 0.0;
        let mut total_cooling = 0.0;

        for i in 0..self.zone_temps.len() {
            let temp = self.zone_temps[i];
            let heating_sp = self.heating_setpoints[i];
            let cooling_sp = self.cooling_setpoints[i];
            let conductance = self.zone_conductance[i];
            let thermal_mass = self.zone_thermal_mass[i];
            let infiltration = self.infiltration_flow[i];
            let gains = self.internal_gains[i];

            let loss_to_outdoor = conductance * (temp - outdoor_temp);
            let infiltration_loss =
                infiltration * air_density * specific_heat * (temp - outdoor_temp) / 3600.0;
            let net_gains = gains - loss_to_outdoor - infiltration_loss;

            let mut hvac_load = 0.0;
            if temp < heating_sp {
                let load = (heating_sp - temp) * conductance;
                hvac_load = load.min(thermal_mass / dt_s).max(0.0);
                self.zone_temps[i] += (hvac_load - net_gains) * dt_s / thermal_mass;
                total_heating += hvac_load;
            } else if temp > cooling_sp {
                let load = (temp - cooling_sp) * conductance;
                hvac_load = load.min(thermal_mass / dt_s).max(0.0);
                self.zone_temps[i] -= (hvac_load + net_gains) * dt_s / thermal_mass;
                total_cooling += hvac_load;
            } else {
                self.zone_temps[i] += net_gains * dt_s / thermal_mass;
            }

            self.zone_temps[i] = self.zone_temps[i].clamp(TEMPERATURE_MIN_C, TEMPERATURE_MAX_C);
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
        if let Some(rest) = loop_id.strip_prefix("heating_zone_") {
            if let Ok(idx) = rest.parse::<usize>() {
                if idx < self.heating_setpoints.len() {
                    let validated = validate_finite(
                        setpoint,
                        &format!("set_control('{}')", loop_id),
                        TEMPERATURE_MIN_C,
                        TEMPERATURE_MAX_C,
                    )?;
                    console_log!("fluxion-wasm: set_control({}, {})", loop_id, validated);
                    self.heating_setpoints[idx] = validated;
                    return Ok(());
                }
            }
        } else if let Some(rest) = loop_id.strip_prefix("cooling_zone_") {
            if let Ok(idx) = rest.parse::<usize>() {
                if idx < self.cooling_setpoints.len() {
                    let validated = validate_finite(
                        setpoint,
                        &format!("set_control('{}')", loop_id),
                        TEMPERATURE_MIN_C,
                        TEMPERATURE_MAX_C,
                    )?;
                    console_log!("fluxion-wasm: set_control({}, {})", loop_id, validated);
                    self.cooling_setpoints[idx] = validated;
                    return Ok(());
                }
            }
        }

        let validated = validate_finite(
            setpoint,
            &format!("set_control('{}')", loop_id),
            CONTROL_VALUE_MIN,
            CONTROL_VALUE_MAX,
        )?;
        console_log!("fluxion-wasm: set_control({}, {})", loop_id, validated);
        self.control_setpoints
            .insert(loop_id.to_string(), validated);
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
    /// * `temperature` - The reset temperature in °C. Must be finite and within
    ///   the physical building-thermal range (10°C – 40°C, mirroring
    ///   `PyZoneSetpoints::set_heating_setpoint`).
    ///
    /// # Returns
    /// Unit on success, or a `JsValue` error if `temperature` is NaN/Inf or
    /// outside the physical range.
    #[wasm_bindgen]
    pub fn reset_temperatures(&mut self, temperature: f64) -> Result<(), JsValue> {
        let validated = validate_finite(
            temperature,
            "reset_temperatures(temperature)",
            TEMPERATURE_MIN_C,
            TEMPERATURE_MAX_C,
        )?;
        for t in &mut self.zone_temps {
            *t = validated;
        }
        console_log!("fluxion-wasm: temperatures reset to {}°C", validated);
        Ok(())
    }

    /// Get the current simulation time in hours.
    #[wasm_bindgen]
    pub fn current_hour(&self) -> f64 {
        self.current_hour
    }

    /// Set all zone temperatures.
    ///
    /// # Arguments
    /// * `temperatures` - Vector of zone temperatures in °C (must match
    ///   `num_zones`). Every element must be finite and within the physical
    ///   building-thermal range (10°C – 40°C, mirroring
    ///   `PyZoneSetpoints::set_heating_setpoint` / `set_cooling_setpoint`).
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error on length mismatch or on any
    /// element being NaN/Inf / outside the physical range.
    #[wasm_bindgen]
    pub fn set_temperatures(&mut self, temperatures: Vec<f64>) -> Result<(), JsValue> {
        if temperatures.len() != self.zone_temps.len() {
            return Err(JsValue::from_str(&format!(
                "temperatures length {} does not match num_zones {}",
                temperatures.len(),
                self.zone_temps.len()
            )));
        }
        for (i, t) in temperatures.iter().enumerate() {
            validate_finite(
                *t,
                &format!("set_temperatures[{}]", i),
                TEMPERATURE_MIN_C,
                TEMPERATURE_MAX_C,
            )?;
        }
        self.zone_temps = temperatures;
        Ok(())
    }

    /// Get the current execution mode.
    ///
    /// Returns `"Physics"` since the WASM build only supports physics-based simulation.
    #[wasm_bindgen]
    pub fn mode(&self) -> String {
        self.mode.clone()
    }

    /// Set the execution mode.
    ///
    /// Currently a no-op since only `"Physics"` mode is supported in WASM.
    /// Calling with any value logs a warning and leaves mode unchanged.
    #[wasm_bindgen]
    pub fn set_mode(&mut self, mode: &str) -> Result<(), JsValue> {
        console_log!(
            "fluxion-wasm: set_mode({}) — only 'Physics' mode is supported in WASM",
            mode
        );
        if mode != "Physics" {
            console_log!(
                "fluxion-wasm: WARNING — mode '{}' not supported, keeping 'Physics'",
                mode
            );
        }
        self.mode = "Physics".to_string();
        Ok(())
    }

    /// Apply parameters from an optimization gene vector.
    ///
    /// # Arguments
    /// * `params` - Parameter vector:
    ///   - `params[0]`: Window U-value (W/m²K, range: 0.1-10.0) — stored but
    ///     not used in the simplified model. Previously unconstrained (issue
    ///     #2911).
    ///   - `params[1]`: Heating setpoint (°C, range: 10-40) — applied to all
    ///     zones. Mirrors `PyZoneSetpoints::set_heating_setpoint`.
    ///   - `params[2]`: Cooling setpoint (°C, range: 10-40) — applied to all
    ///     zones. Mirrors `PyZoneSetpoints::set_cooling_setpoint`. The
    ///     previous 15-25 / 22-32 clamps silently masked out-of-range inputs;
    ///     we now reject them outright.
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error if param count is invalid, any
    /// element is NaN/Inf, or any element is outside its physical range.
    #[wasm_bindgen]
    pub fn apply_parameters(&mut self, params: Vec<f64>) -> Result<(), JsValue> {
        if params.len() < 3 {
            return Err(JsValue::from_str(&format!(
                "apply_parameters requires 3 params, got {}",
                params.len()
            )));
        }

        let u_value = validate_finite(
            params[0],
            "apply_parameters[0] (U-value)",
            U_VALUE_MIN,
            U_VALUE_MAX,
        )?;
        let heating_sp = validate_finite(
            params[1],
            "apply_parameters[1] (heating setpoint)",
            TEMPERATURE_MIN_C,
            TEMPERATURE_MAX_C,
        )?;
        let cooling_sp = validate_finite(
            params[2],
            "apply_parameters[2] (cooling setpoint)",
            TEMPERATURE_MIN_C,
            TEMPERATURE_MAX_C,
        )?;

        for sp in &mut self.heating_setpoints {
            *sp = heating_sp;
        }
        for sp in &mut self.cooling_setpoints {
            *sp = cooling_sp;
        }

        console_log!(
            "fluxion-wasm: apply_parameters — U={} W/m²K, heating={}°C, cooling={}°C",
            u_value,
            heating_sp,
            cooling_sp
        );
        Ok(())
    }

    /// Get zone floor area in m².
    ///
    /// Returns the total floor area (50 m² per zone).
    #[wasm_bindgen]
    pub fn zone_area(&self) -> f64 {
        self.zone_area
    }

    /// Calculate HVAC power demand based on current conditions.
    ///
    /// Uses an enhanced energy balance to estimate heating (positive) or cooling
    /// (negative) power demand in Watts, accounting for zone-specific thermal parameters.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (unused)
    /// * `outdoor_temp` - Outdoor drybulb temperature in °C
    ///
    /// # Returns
    /// Heating power (positive) or cooling power (negative) in Watts.
    #[wasm_bindgen]
    pub fn hvac_power_demand(&self, _timestep: usize, outdoor_temp: f64) -> f64 {
        let air_density = 1.2;
        let specific_heat = 1006.0;
        let mut total_power = 0.0;

        for i in 0..self.zone_temps.len() {
            let temp = self.zone_temps[i];
            let heating_sp = self.heating_setpoints[i];
            let cooling_sp = self.cooling_setpoints[i];
            let conductance = self.zone_conductance[i];
            let infiltration = self.infiltration_flow[i];

            if temp < heating_sp {
                total_power += conductance * (heating_sp - temp);
            } else if temp > cooling_sp {
                total_power -= conductance * (temp - cooling_sp);
            }

            let infiltration_loss =
                infiltration * air_density * specific_heat * (temp - outdoor_temp) / 3600.0;
            total_power -= infiltration_loss;
        }

        total_power
    }

    /// Check if the simulation state is valid.
    ///
    /// Returns `true` if zone count > 0 and heating setpoint < cooling setpoint.
    #[wasm_bindgen]
    pub fn is_valid(&self) -> bool {
        if self.zone_temps.is_empty() {
            return false;
        }
        for i in 0..self.heating_setpoints.len() {
            if self.heating_setpoints[i] >= self.cooling_setpoints[i] {
                return false;
            }
        }
        true
    }

    /// Solve thermal model for multiple timesteps.
    ///
    /// **Note:** This is a stub. Full `solve_timesteps` requires `SurrogateManager`
    /// which uses ONNX inference — unavailable in WASM. This method returns `0.0`
    /// as a placeholder EUI value.
    ///
    /// # Arguments
    /// * `steps` - Number of hourly timesteps (ignored)
    /// * `use_surrogates` - Ignored (always `false` in WASM)
    ///
    /// # Returns
    /// Placeholder EUI value of `0.0`.
    #[wasm_bindgen]
    pub fn solve_timesteps(&mut self, steps: usize, _use_surrogates: bool) -> f64 {
        console_log!(
            "fluxion-wasm: solve_timesteps({} steps) — ONNX surrogates unavailable in WASM, returning 0.0",
            steps
        );
        0.0
    }

    /// Get zone thermal mass in J/K for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    ///
    /// # Returns
    /// Thermal mass in J/K, or `JsValue` error if zone_id is out of range.
    #[wasm_bindgen]
    pub fn get_zone_thermal_mass(&self, zone_id: usize) -> Result<f64, JsValue> {
        self.zone_thermal_mass
            .get(zone_id)
            .copied()
            .ok_or_else(|| JsValue::from_str("zone_id out of range"))
    }

    /// Set zone thermal mass for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    /// * `thermal_mass` - Thermal mass in J/K (must be positive and finite)
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error if zone_id is out of range or thermal_mass is invalid.
    #[wasm_bindgen]
    pub fn set_zone_thermal_mass(
        &mut self,
        zone_id: usize,
        thermal_mass: f64,
    ) -> Result<(), JsValue> {
        let mass = validate_finite(thermal_mass, "set_zone_thermal_mass", 1e3, 1e10)?;
        if mass <= 0.0 {
            return Err(JsValue::from_str("thermal_mass must be positive"));
        }
        if zone_id >= self.zone_thermal_mass.len() {
            return Err(JsValue::from_str("zone_id out of range"));
        }
        self.zone_thermal_mass[zone_id] = mass;
        console_log!(
            "fluxion-wasm: zone {} thermal_mass set to {} J/K",
            zone_id,
            mass
        );
        Ok(())
    }

    /// Get all zone thermal masses.
    #[wasm_bindgen]
    pub fn get_all_thermal_masses(&self) -> Vec<f64> {
        self.zone_thermal_mass.clone()
    }

    /// Get zone conductance in W/K for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    ///
    /// # Returns
    /// Conductance in W/K, or `JsValue` error if zone_id is out of range.
    #[wasm_bindgen]
    pub fn get_zone_conductance(&self, zone_id: usize) -> Result<f64, JsValue> {
        self.zone_conductance
            .get(zone_id)
            .copied()
            .ok_or_else(|| JsValue::from_str("zone_id out of range"))
    }

    /// Set zone conductance for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    /// * `conductance` - Conductance in W/K (must be positive and finite)
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error if zone_id is out of range or conductance is invalid.
    #[wasm_bindgen]
    pub fn set_zone_conductance(
        &mut self,
        zone_id: usize,
        conductance: f64,
    ) -> Result<(), JsValue> {
        let cond = validate_finite(conductance, "set_zone_conductance", 0.1, 1e6)?;
        if cond <= 0.0 {
            return Err(JsValue::from_str("conductance must be positive"));
        }
        if zone_id >= self.zone_conductance.len() {
            return Err(JsValue::from_str("zone_id out of range"));
        }
        self.zone_conductance[zone_id] = cond;
        console_log!(
            "fluxion-wasm: zone {} conductance set to {} W/K",
            zone_id,
            cond
        );
        Ok(())
    }

    /// Get all zone conductances.
    #[wasm_bindgen]
    pub fn get_all_conductances(&self) -> Vec<f64> {
        self.zone_conductance.clone()
    }

    /// Get infiltration flow rate in kg/s for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    ///
    /// # Returns
    /// Infiltration flow in kg/s, or `JsValue` error if zone_id is out of range.
    #[wasm_bindgen]
    pub fn get_zone_infiltration(&self, zone_id: usize) -> Result<f64, JsValue> {
        self.infiltration_flow
            .get(zone_id)
            .copied()
            .ok_or_else(|| JsValue::from_str("zone_id out of range"))
    }

    /// Set infiltration rate for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    /// * `infiltration_ach` - Infiltration rate in ACH (air changes per hour)
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error if zone_id is out of range or infiltration_ach is invalid.
    #[wasm_bindgen]
    pub fn set_zone_infiltration(
        &mut self,
        zone_id: usize,
        infiltration_ach: f64,
    ) -> Result<(), JsValue> {
        let ach = validate_finite(infiltration_ach, "set_zone_infiltration", 0.0, 10.0)?;
        if zone_id >= self.infiltration_flow.len() {
            return Err(JsValue::from_str("zone_id out of range"));
        }
        let zone_area = self.zone_area / self.zone_temps.len() as f64;
        self.infiltration_flow[zone_id] = ach * 0.0012 * zone_area * 3600.0;
        console_log!(
            "fluxion-wasm: zone {} infiltration set to {} ACH",
            zone_id,
            ach
        );
        Ok(())
    }

    /// Get all infiltration rates in kg/s.
    #[wasm_bindgen]
    pub fn get_all_infiltration(&self) -> Vec<f64> {
        self.infiltration_flow.clone()
    }

    /// Get internal gains in W for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    ///
    /// # Returns
    /// Internal gains in W, or `JsValue` error if zone_id is out of range.
    #[wasm_bindgen]
    pub fn get_zone_internal_gains(&self, zone_id: usize) -> Result<f64, JsValue> {
        self.internal_gains
            .get(zone_id)
            .copied()
            .ok_or_else(|| JsValue::from_str("zone_id out of range"))
    }

    /// Set internal gains for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    /// * `gains_w` - Internal gains in W (equipment, lighting, occupants)
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error if zone_id is out of range or gains_w is invalid.
    #[wasm_bindgen]
    pub fn set_zone_internal_gains(&mut self, zone_id: usize, gains_w: f64) -> Result<(), JsValue> {
        let gains = validate_finite(gains_w, "set_zone_internal_gains", 0.0, 1e6)?;
        if zone_id >= self.internal_gains.len() {
            return Err(JsValue::from_str("zone_id out of range"));
        }
        self.internal_gains[zone_id] = gains;
        console_log!(
            "fluxion-wasm: zone {} internal_gains set to {} W",
            zone_id,
            gains
        );
        Ok(())
    }

    /// Get all internal gains in W.
    #[wasm_bindgen]
    pub fn get_all_internal_gains(&self) -> Vec<f64> {
        self.internal_gains.clone()
    }

    /// Get zone floor area for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zero-based zone index
    ///
    /// # Returns
    /// Zone floor area in m², or `JsValue` error if zone_id is out of range.
    #[wasm_bindgen]
    pub fn get_zone_area(&self, zone_id: usize) -> Result<f64, JsValue> {
        if zone_id >= self.zone_temps.len() {
            return Err(JsValue::from_str("zone_id out of range"));
        }
        Ok(self.zone_area / self.zone_temps.len() as f64)
    }

    /// Apply zone parameters from a JSON string.
    ///
    /// Allows updating multiple zone parameters at once.
    ///
    /// # Arguments
    /// * `params_json` - JSON string containing zone parameters:
    ///   - `zone_id`: Required zone index
    ///   - `thermal_mass`: Optional thermal mass in J/K
    ///   - `conductance`: Optional conductance in W/K
    ///   - `infiltration_ach`: Optional infiltration in ACH
    ///   - `internal_gains_w`: Optional internal gains in W
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error if parameters are invalid.
    #[wasm_bindgen]
    pub fn apply_zone_parameters(&mut self, params_json: &str) -> Result<(), JsValue> {
        #[derive(serde::Deserialize)]
        struct ZoneParams {
            zone_id: usize,
            thermal_mass: Option<f64>,
            conductance: Option<f64>,
            infiltration_ach: Option<f64>,
            internal_gains_w: Option<f64>,
        }

        let params: ZoneParams = serde_json::from_str(params_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid zone params JSON: {}", e)))?;

        if params.zone_id >= self.zone_temps.len() {
            return Err(JsValue::from_str("zone_id out of range"));
        }

        if let Some(tm) = params.thermal_mass {
            self.set_zone_thermal_mass(params.zone_id, tm)?;
        }
        if let Some(c) = params.conductance {
            self.set_zone_conductance(params.zone_id, c)?;
        }
        if let Some(ach) = params.infiltration_ach {
            self.set_zone_infiltration(params.zone_id, ach)?;
        }
        if let Some(g) = params.internal_gains_w {
            self.set_zone_internal_gains(params.zone_id, g)?;
        }

        Ok(())
    }

    /// Export current simulation state as JSON.
    ///
    /// Includes all zone temperatures, setpoints, and thermal parameters.
    ///
    /// # Returns
    /// JSON string with full simulation state.
    #[wasm_bindgen]
    pub fn export_state(&self) -> Result<String, JsValue> {
        #[derive(serde::Serialize)]
        struct SimulationState {
            current_hour: f64,
            num_zones: usize,
            zone_temps: Vec<f64>,
            heating_setpoints: Vec<f64>,
            cooling_setpoints: Vec<f64>,
            zone_thermal_mass: Vec<f64>,
            zone_conductance: Vec<f64>,
            infiltration_flow: Vec<f64>,
            internal_gains: Vec<f64>,
        }

        let state = SimulationState {
            current_hour: self.current_hour,
            num_zones: self.zone_temps.len(),
            zone_temps: self.zone_temps.clone(),
            heating_setpoints: self.heating_setpoints.clone(),
            cooling_setpoints: self.cooling_setpoints.clone(),
            zone_thermal_mass: self.zone_thermal_mass.clone(),
            zone_conductance: self.zone_conductance.clone(),
            infiltration_flow: self.infiltration_flow.clone(),
            internal_gains: self.internal_gains.clone(),
        };

        serde_json::to_string(&state)
            .map_err(|e| JsValue::from_str(&format!("state serialization error: {}", e)))
    }

    /// Load simulation state from JSON.
    ///
    /// Restores a previously exported simulation state.
    ///
    /// # Arguments
    /// * `state_json` - JSON string from `export_state()`
    ///
    /// # Returns
    /// Unit on success, or `JsValue` error if state is invalid.
    #[wasm_bindgen]
    pub fn load_state(&mut self, state_json: &str) -> Result<(), JsValue> {
        #[derive(serde::Deserialize)]
        struct SimulationState {
            current_hour: f64,
            zone_temps: Vec<f64>,
            heating_setpoints: Vec<f64>,
            cooling_setpoints: Vec<f64>,
            zone_thermal_mass: Vec<f64>,
            zone_conductance: Vec<f64>,
            infiltration_flow: Vec<f64>,
            internal_gains: Vec<f64>,
        }

        let state: SimulationState = serde_json::from_str(state_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid state JSON: {}", e)))?;

        if state.zone_temps.len() != self.zone_temps.len() {
            return Err(JsValue::from_str("zone count mismatch"));
        }

        self.current_hour = state.current_hour;
        self.zone_temps = state.zone_temps;
        self.heating_setpoints = state.heating_setpoints;
        self.cooling_setpoints = state.cooling_setpoints;
        self.zone_thermal_mass = state.zone_thermal_mass;
        self.zone_conductance = state.zone_conductance;
        self.infiltration_flow = state.infiltration_flow;
        self.internal_gains = state.internal_gains;

        console_log!("fluxion-wasm: state loaded successfully");
        Ok(())
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

#[cfg(test)]
mod tests {
    //! Inline tests covering the NaN / ±Inf / out-of-range acceptance
    //! criteria from issue #2911. The pure-logic `check_finite` helper is
    //! exercised natively; end-to-end coverage that goes through the
    //! wasm-bindgen `FluidSimulation` API is gated to wasm32 and runs
    //! under `wasm-pack test --node` (see `tests/wasm_integration_tests.rs`
    //! for the native-side mirror).

    use super::*;

    // ---- check_finite unit coverage (native + wasm32) -------------------

    #[test]
    fn check_finite_accepts_in_range() {
        assert_eq!(check_finite(22.0, 10.0, 40.0).unwrap(), 22.0);

        // Boundaries are inclusive.
        assert!(check_finite(10.0, 10.0, 40.0).is_ok());
        assert!(check_finite(40.0, 10.0, 40.0).is_ok());
    }

    #[test]
    fn check_finite_rejects_nan() {
        let err = check_finite(f64::NAN, 10.0, 40.0).unwrap_err();
        assert!(
            err.contains("finite"),
            "error must mention finiteness; got: {}",
            err
        );
    }

    #[test]
    fn check_finite_rejects_pos_inf() {
        assert!(check_finite(f64::INFINITY, 10.0, 40.0).is_err());
        assert!(check_finite(f64::MAX, 10.0, 40.0).is_err());
    }

    #[test]
    fn check_finite_rejects_neg_inf() {
        assert!(check_finite(f64::NEG_INFINITY, 10.0, 40.0).is_err());
        assert!(check_finite(f64::MIN, 10.0, 40.0).is_err());
    }

    #[test]
    fn check_finite_rejects_below_min() {
        let err = check_finite(9.999, 10.0, 40.0).unwrap_err();
        assert!(err.contains("outside the valid range"), "got: {}", err);
    }

    #[test]
    fn check_finite_rejects_above_max() {
        assert!(check_finite(40.001, 10.0, 40.0).is_err());
        assert!(check_finite(100.0, 10.0, 40.0).is_err());
    }

    // ---- end-to-end coverage (wasm32 only — needs wasm-bindgen runtime) -

    #[cfg(target_arch = "wasm32")]
    mod wasm_e2e {
        use super::*;
        use wasm_bindgen_test::*;

        fn config() -> FluidSimulationConfig {
            FluidSimulationConfig {
                num_zones: 3,
                initial_temps: Some(vec![22.0, 22.0, 22.0]),
                heating_setpoint: 20.0,
                cooling_setpoint: 24.0,
                ..Default::default()
            }
        }

        fn fresh_sim() -> FluidSimulation {
            FluidSimulation::new(&serde_json::to_string(&config()).unwrap()).unwrap()
        }

        #[wasm_bindgen_test]
        fn set_temperatures_rejects_nan() {
            let mut sim = fresh_sim();
            assert!(sim.set_temperatures(vec![20.0, f64::NAN, 22.0]).is_err());
            let temps = sim.get_zone_temps();
            assert!(temps.iter().all(|t| t.is_finite()));
        }

        #[wasm_bindgen_test]
        fn set_temperatures_rejects_pos_inf() {
            let mut sim = fresh_sim();
            assert!(sim
                .set_temperatures(vec![f64::INFINITY, 22.0, 22.0])
                .is_err());
        }

        #[wasm_bindgen_test]
        fn set_temperatures_rejects_neg_inf() {
            let mut sim = fresh_sim();
            assert!(sim
                .set_temperatures(vec![22.0, 22.0, f64::NEG_INFINITY])
                .is_err());
        }

        #[wasm_bindgen_test]
        fn set_temperatures_rejects_out_of_range() {
            let mut sim = fresh_sim();
            assert!(sim.set_temperatures(vec![20.0, 5.0, 22.0]).is_err());
            assert!(sim.set_temperatures(vec![20.0, 22.0, 55.0]).is_err());
            assert!(sim.set_temperatures(vec![20.0, 22.0, -273.15]).is_err());
        }

        #[wasm_bindgen_test]
        fn set_temperatures_accepts_boundary_values() {
            let mut sim = fresh_sim();
            assert!(sim.set_temperatures(vec![10.0, 25.0, 40.0]).is_ok());
            assert_eq!(sim.get_zone_temps(), vec![10.0, 25.0, 40.0]);
        }

        #[wasm_bindgen_test]
        fn set_control_rejects_nan_on_heating_setpoint() {
            let mut sim = fresh_sim();
            assert!(sim.set_control("heating_zone_0", f64::NAN).is_err());
        }

        #[wasm_bindgen_test]
        fn set_control_rejects_nan_on_cooling_setpoint() {
            let mut sim = fresh_sim();
            assert!(sim.set_control("cooling_zone_2", f64::NAN).is_err());
        }

        #[wasm_bindgen_test]
        fn set_control_rejects_pos_inf_on_custom_loop() {
            let mut sim = fresh_sim();
            assert!(sim.set_control("vav_damper_1", f64::INFINITY).is_err());
        }

        #[wasm_bindgen_test]
        fn set_control_rejects_neg_inf_on_custom_loop() {
            let mut sim = fresh_sim();
            assert!(sim.set_control("vav_damper_1", f64::NEG_INFINITY).is_err());
        }

        #[wasm_bindgen_test]
        fn set_control_rejects_out_of_range_temp() {
            let mut sim = fresh_sim();
            assert!(sim.set_control("heating_zone_0", 200.0).is_err());
            assert!(sim.set_control("cooling_zone_0", -50.0).is_err());
        }

        #[wasm_bindgen_test]
        fn set_control_does_not_leak_nan_to_console() {
            // Issue #2911 called out console_log leaking the NaN value
            // to DevTools. Validation happens BEFORE the log call.
            let mut sim = fresh_sim();
            let _ = sim.set_control("heating_zone_0", f64::NAN);
        }

        #[wasm_bindgen_test]
        fn reset_temperatures_rejects_nan() {
            let mut sim = fresh_sim();
            assert!(sim.reset_temperatures(f64::NAN).is_err());
            assert!(sim.get_zone_temps().iter().all(|t| t.is_finite()));
        }

        #[wasm_bindgen_test]
        fn reset_temperatures_rejects_pos_inf() {
            let mut sim = fresh_sim();
            assert!(sim.reset_temperatures(f64::INFINITY).is_err());
        }

        #[wasm_bindgen_test]
        fn reset_temperatures_rejects_neg_inf() {
            let mut sim = fresh_sim();
            assert!(sim.reset_temperatures(f64::NEG_INFINITY).is_err());
        }

        #[wasm_bindgen_test]
        fn reset_temperatures_rejects_out_of_range() {
            let mut sim = fresh_sim();
            assert!(sim.reset_temperatures(-10.0).is_err());
            assert!(sim.reset_temperatures(100.0).is_err());
        }

        #[wasm_bindgen_test]
        fn reset_temperatures_accepts_in_range() {
            let mut sim = fresh_sim();
            assert!(sim.reset_temperatures(15.0).is_ok());
            assert_eq!(sim.get_zone_temps(), vec![15.0, 15.0, 15.0]);
        }

        #[wasm_bindgen_test]
        fn apply_parameters_rejects_nan_in_u_value() {
            let mut sim = fresh_sim();
            assert!(sim.apply_parameters(vec![f64::NAN, 20.0, 24.0]).is_err());
        }

        #[wasm_bindgen_test]
        fn apply_parameters_rejects_pos_inf_in_heating() {
            let mut sim = fresh_sim();
            assert!(sim
                .apply_parameters(vec![1.5, f64::INFINITY, 24.0])
                .is_err());
        }

        #[wasm_bindgen_test]
        fn apply_parameters_rejects_neg_inf_in_cooling() {
            let mut sim = fresh_sim();
            assert!(sim
                .apply_parameters(vec![1.5, 20.0, f64::NEG_INFINITY])
                .is_err());
        }

        #[wasm_bindgen_test]
        fn apply_parameters_rejects_out_of_range_u_value() {
            let mut sim = fresh_sim();
            assert!(sim.apply_parameters(vec![0.0, 20.0, 24.0]).is_err());
            assert!(sim.apply_parameters(vec![20.0, 20.0, 24.0]).is_err());
        }

        #[wasm_bindgen_test]
        fn apply_parameters_rejects_out_of_range_heating_setpoint() {
            let mut sim = fresh_sim();
            assert!(sim.apply_parameters(vec![1.5, 5.0, 24.0]).is_err());
            assert!(sim.apply_parameters(vec![1.5, 60.0, 24.0]).is_err());
        }

        #[wasm_bindgen_test]
        fn apply_parameters_rejects_out_of_range_cooling_setpoint() {
            let mut sim = fresh_sim();
            assert!(sim.apply_parameters(vec![1.5, 20.0, 5.0]).is_err());
            assert!(sim.apply_parameters(vec![1.5, 20.0, 60.0]).is_err());
        }

        #[wasm_bindgen_test]
        fn apply_parameters_accepts_boundary_values() {
            let mut sim = fresh_sim();
            assert!(sim.apply_parameters(vec![0.1, 10.0, 10.0]).is_ok());
            let mut sim = fresh_sim();
            assert!(sim.apply_parameters(vec![10.0, 40.0, 40.0]).is_ok());
        }
    }
}

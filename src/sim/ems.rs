//! Energy Management System (EMS) Module
//!
//! This module provides an Energy Management System framework for overriding HVAC setpoints,
//! schedules, and equipment status based on custom logic executed at each simulation timestep.
//!
//! # Architecture
//!
//! - **EmsGlobalVariable**: Named numeric values that programs can read/write
//! - **EmsSensor**: Read-only access to simulation state (zone temp, outdoor conditions)
//! - **EmsActuator**: Write access to override HVAC setpoints, schedules, equipment
//! - **EmsProgram**: Callable logic that runs at each timestep, can read sensors and write actuators
//! - **EmsManager**: Coordinates all EMS components and integrates with the simulation loop
//!
//! # P1 Applications
//!
//! - **Demand Response**: Override HVAC setpoints to shed load within 1 timestep
//! - **Optimal Start/Stop**: Adjust pre-conditioning based on occupancy
//! - **Custom Schedules**: Override schedules based on runtime conditions
//!
//! # Usage
//!
//! ```rust,ignore
//! use fluxion::sim::ems::{EmsManager, EmsSensorType, EmsActuatorType};
//!
//! let mut ems = EmsManager::new();
//!
//! // Add a sensor to read zone 0 temperature
//! ems.add_sensor("zone_temp_0", EmsSensorType::ZoneTemperature(0));
//!
//! // Add an actuator to override heating setpoint
//! ems.add_actuator("heating_setpoint", EmsActuatorType::HeatingSetpoint(0));
//!
//! // Create a program that reduces setpoint by 2°C during demand response
//! ems.add_program("demand_response_shed", |ems, _timestep| {
//!     if ems.get_global("dr_signal").unwrap_or(0.0) > 0.0 {
//!         let zone_temp = ems.read_sensor("zone_temp_0").unwrap_or(20.0);
//!         if zone_temp > 22.0 {
//!             ems.write_actuator("heating_setpoint", zone_temp - 2.0);
//!         }
//!     }
//! });
//!
//! // At each timestep, execute all programs
//! ems.execute_programs(timestep);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64;

/// Sensor type for reading simulation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmsSensorType {
    /// Zone air temperature (°C)
    ZoneTemperature(usize),
    /// Outdoor dry-bulb temperature (°C)
    OutdoorTemperature,
    /// Zone relative humidity (%)
    ZoneHumidity(usize),
    /// HVAC heating power (W)
    HeatingPower(usize),
    /// HVAC cooling power (W)
    CoolingPower(usize),
    /// Global variable value
    GlobalVariable(String),
}

/// Actuator type for overriding simulation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmsActuatorType {
    /// Override heating setpoint (°C)
    HeatingSetpoint(usize),
    /// Override cooling setpoint (°C)
    CoolingSetpoint(usize),
    /// Override HVAC enabled flag
    HvacEnabled(usize),
    /// Override ventilation schedule multiplier
    VentilationMultiplier(usize),
    /// Override lighting schedule multiplier
    LightingMultiplier(usize),
    /// Set global variable value
    GlobalVariable(String),
}

/// Program runtime state accessible during execution
pub struct EmsRuntime<'a> {
    pub timestep: usize,
    pub hour_of_day: usize,
    pub day_of_year: usize,
    pub global_variables: &'a HashMap<String, f64>,
    pub sensor_values: &'a HashMap<String, f64>,
    pub actuator_values: &'a mut HashMap<String, f64>,
}

impl<'a> EmsRuntime<'a> {
    pub fn new(
        timestep: usize,
        global_variables: &'a HashMap<String, f64>,
        sensor_values: &'a HashMap<String, f64>,
        actuator_values: &'a mut HashMap<String, f64>,
    ) -> Self {
        let hour_of_day = timestep % 24;
        let day_of_year = timestep / 24;
        Self {
            timestep,
            hour_of_day,
            day_of_year,
            global_variables,
            sensor_values,
            actuator_values,
        }
    }
}

/// EMS global variable for storing values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmsGlobalVariable {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub description: String,
}

impl EmsGlobalVariable {
    pub fn new(name: &str, initial_value: f64, unit: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            value: initial_value,
            unit: unit.to_string(),
            description: description.to_string(),
        }
    }
}

/// EMS sensor for reading simulation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmsSensor {
    pub name: String,
    pub sensor_type: EmsSensorType,
}

impl EmsSensor {
    pub fn new(name: &str, sensor_type: EmsSensorType) -> Self {
        Self {
            name: name.to_string(),
            sensor_type,
        }
    }
}

/// EMS actuator for overriding simulation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmsActuator {
    pub name: String,
    pub actuator_type: EmsActuatorType,
}

impl EmsActuator {
    pub fn new(name: &str, actuator_type: EmsActuatorType) -> Self {
        Self {
            name: name.to_string(),
            actuator_type,
        }
    }
}

/// EMS program that executes at each timestep
pub type EmsProgramFn = Box<dyn Fn(&mut EmsRuntime, usize) + Send + Sync>;

pub struct EmsProgram {
    pub name: String,
    pub program_fn: EmsProgramFn,
    pub enabled: bool,
}

impl EmsProgram {
    pub fn new<F>(name: &str, program_fn: F) -> Self
    where
        F: Fn(&mut EmsRuntime, usize) + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            program_fn: Box::new(program_fn),
            enabled: true,
        }
    }

    pub fn execute(&self, runtime: &mut EmsRuntime) {
        if self.enabled {
            (self.program_fn)(runtime, runtime.timestep);
        }
    }
}

/// EMS manager coordinating all EMS components
pub struct EmsManager {
    pub global_variables: HashMap<String, f64>,
    pub sensors: HashMap<String, EmsSensor>,
    pub actuators: HashMap<String, EmsActuator>,
    pub programs: Vec<EmsProgram>,
    sensor_values: HashMap<String, f64>,
    actuator_values: HashMap<String, f64>,
    actuator_overrides: HashMap<String, bool>,
}

impl Default for EmsManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EmsManager {
    pub fn new() -> Self {
        Self {
            global_variables: HashMap::new(),
            sensors: HashMap::new(),
            actuators: HashMap::new(),
            programs: Vec::new(),
            sensor_values: HashMap::new(),
            actuator_values: HashMap::new(),
            actuator_overrides: HashMap::new(),
        }
    }

    pub fn add_global(&mut self, name: &str, initial_value: f64) {
        self.global_variables
            .insert(name.to_string(), initial_value);
    }

    pub fn get_global(&self, name: &str) -> Option<f64> {
        self.global_variables.get(name).copied()
    }

    pub fn set_global(&mut self, name: &str, value: f64) {
        self.global_variables.insert(name.to_string(), value);
    }

    pub fn add_sensor(&mut self, name: &str, sensor_type: EmsSensorType) {
        self.sensors
            .insert(name.to_string(), EmsSensor::new(name, sensor_type));
    }

    pub fn read_sensor(&self, name: &str) -> Option<f64> {
        self.sensor_values.get(name).copied()
    }

    pub fn add_actuator(&mut self, name: &str, actuator_type: EmsActuatorType) {
        self.actuators
            .insert(name.to_string(), EmsActuator::new(name, actuator_type));
        self.actuator_overrides.insert(name.to_string(), false);
    }

    pub fn write_actuator(&mut self, name: &str, value: f64) {
        self.actuator_values.insert(name.to_string(), value);
        self.actuator_overrides.insert(name.to_string(), true);
    }

    pub fn read_actuator(&self, name: &str) -> Option<f64> {
        self.actuator_values.get(name).copied()
    }

    pub fn is_actuator_overridden(&self, name: &str) -> bool {
        self.actuator_overrides.get(name).copied().unwrap_or(false)
    }

    pub fn clear_actuator_override(&mut self, name: &str) {
        self.actuator_overrides.insert(name.to_string(), false);
    }

    pub fn clear_all_actuator_overrides(&mut self) {
        let keys: Vec<_> = self.actuator_overrides.keys().cloned().collect();
        for key in keys {
            self.actuator_overrides.insert(key, false);
        }
    }

    pub fn add_program<F>(&mut self, name: &str, program_fn: F)
    where
        F: Fn(&mut EmsRuntime, usize) + Send + Sync + 'static,
    {
        self.programs.push(EmsProgram::new(name, program_fn));
    }

    pub fn set_program_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(program) = self.programs.iter_mut().find(|p| p.name == name) {
            program.enabled = enabled;
        }
    }

    pub fn remove_program(&mut self, name: &str) {
        self.programs.retain(|p| p.name != name);
    }

    /// Update sensor values from simulation state
    pub fn update_sensors(
        &mut self,
        zone_temperatures: &[f64],
        outdoor_temp: f64,
        heating_power: &[f64],
        cooling_power: &[f64],
    ) {
        for (name, sensor) in &self.sensors {
            let value = match &sensor.sensor_type {
                EmsSensorType::ZoneTemperature(zone_idx) => zone_temperatures
                    .get(*zone_idx)
                    .copied()
                    .unwrap_or(f64::NAN),
                EmsSensorType::OutdoorTemperature => outdoor_temp,
                EmsSensorType::HeatingPower(zone_idx) => {
                    heating_power.get(*zone_idx).copied().unwrap_or(0.0)
                }
                EmsSensorType::CoolingPower(zone_idx) => {
                    cooling_power.get(*zone_idx).copied().unwrap_or(0.0)
                }
                EmsSensorType::GlobalVariable(var_name) => self
                    .global_variables
                    .get(var_name)
                    .copied()
                    .unwrap_or(f64::NAN),
                EmsSensorType::ZoneHumidity(_) => {
                    // Humidity not typically tracked in basic thermal model
                    f64::NAN
                }
            };
            self.sensor_values.insert(name.clone(), value);
        }
    }

    /// Execute all EMS programs for the current timestep
    pub fn execute_programs(&mut self, timestep: usize) {
        let mut runtime = EmsRuntime::new(
            timestep,
            &self.global_variables,
            &self.sensor_values,
            &mut self.actuator_values,
        );

        for program in &self.programs {
            program.execute(&mut runtime);
        }
    }

    /// Apply actuator overrides to thermal model
    pub fn apply_actuator_overrides(
        &self,
        heating_setpoints: &mut [f64],
        cooling_setpoints: &mut [f64],
        hvac_enabled: &mut [bool],
        ventilation_multipliers: &mut [f64],
        lighting_multipliers: &mut [f64],
    ) {
        let names: Vec<_> = self.actuators.keys().cloned().collect();
        for name in names {
            if !self.actuator_values.contains_key(&name) {
                continue;
            }

            let value = *self.actuator_values.get(&name).unwrap();

            let actuator = match self.actuators.get(&name) {
                Some(a) => a,
                None => continue,
            };

            match &actuator.actuator_type {
                EmsActuatorType::HeatingSetpoint(zone_idx) => {
                    if let Some(setpoint) = heating_setpoints.get_mut(*zone_idx) {
                        *setpoint = value;
                    }
                }
                EmsActuatorType::CoolingSetpoint(zone_idx) => {
                    if let Some(setpoint) = cooling_setpoints.get_mut(*zone_idx) {
                        *setpoint = value;
                    }
                }
                EmsActuatorType::HvacEnabled(zone_idx) => {
                    if let Some(enabled) = hvac_enabled.get_mut(*zone_idx) {
                        *enabled = value > 0.5;
                    }
                }
                EmsActuatorType::VentilationMultiplier(zone_idx) => {
                    if let Some(mult) = ventilation_multipliers.get_mut(*zone_idx) {
                        *mult = value;
                    }
                }
                EmsActuatorType::LightingMultiplier(zone_idx) => {
                    if let Some(mult) = lighting_multipliers.get_mut(*zone_idx) {
                        *mult = value;
                    }
                }
                EmsActuatorType::GlobalVariable(_) => {
                    // Update global variable from actuator value
                    // This is handled separately in apply_global_overrides
                }
            }
        }
    }

    /// Apply global variable overrides
    pub fn apply_global_overrides(&mut self) {
        for (name, actuator) in &self.actuators {
            if let EmsActuatorType::GlobalVariable(var_name) = &actuator.actuator_type {
                if let Some(value) = self.actuator_values.get(name) {
                    self.global_variables.insert(var_name.clone(), *value);
                }
            }
        }
    }

    /// Reset all actuator values after a timestep
    pub fn reset_actuators(&mut self) {
        self.actuator_values.clear();
        let keys: Vec<_> = self.actuator_overrides.keys().cloned().collect();
        for key in keys {
            self.actuator_overrides.insert(key, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ems_manager_global_variables() {
        let mut ems = EmsManager::new();
        ems.add_global("test_var", 10.0);
        assert_eq!(ems.get_global("test_var"), Some(10.0));
        ems.set_global("test_var", 20.0);
        assert_eq!(ems.get_global("test_var"), Some(20.0));
        assert_eq!(ems.get_global("nonexistent"), None);
    }

    #[test]
    fn test_ems_manager_sensors() {
        let mut ems = EmsManager::new();
        ems.add_sensor("zone_0_temp", EmsSensorType::ZoneTemperature(0));
        ems.add_sensor("outdoor_temp", EmsSensorType::OutdoorTemperature);

        // Update sensor values
        ems.update_sensors(&[22.0, 24.0], 15.0, &[1000.0, 2000.0], &[0.0, 0.0]);

        assert_eq!(ems.read_sensor("zone_0_temp"), Some(22.0));
        assert_eq!(ems.read_sensor("outdoor_temp"), Some(15.0));
        assert_eq!(ems.read_sensor("nonexistent"), None);
    }

    #[test]
    fn test_ems_manager_actuators() {
        let mut ems = EmsManager::new();
        ems.add_actuator("heating_0", EmsActuatorType::HeatingSetpoint(0));
        ems.add_actuator("cooling_0", EmsActuatorType::CoolingSetpoint(0));

        // Write actuator values
        ems.write_actuator("heating_0", 22.0);
        ems.write_actuator("cooling_0", 26.0);

        assert_eq!(ems.read_actuator("heating_0"), Some(22.0));
        assert_eq!(ems.read_actuator("cooling_0"), Some(26.0));
        assert!(ems.is_actuator_overridden("heating_0"));
        assert!(ems.is_actuator_overridden("cooling_0"));
        assert!(!ems.is_actuator_overridden("nonexistent"));

        // Clear overrides
        ems.clear_all_actuator_overrides();
        assert!(!ems.is_actuator_overridden("heating_0"));
    }

    #[test]
    fn test_ems_manager_programs() {
        let mut ems = EmsManager::new();
        ems.add_global("counter", 0.0);
        ems.add_actuator(
            "counter",
            EmsActuatorType::GlobalVariable("counter".to_string()),
        );

        ems.add_program("increment_counter", |runtime, _timestep| {
            let current = runtime
                .global_variables
                .get("counter")
                .copied()
                .unwrap_or(0.0);
            runtime
                .actuator_values
                .insert("counter".to_string(), current + 1.0);
        });

        // Execute programs
        ems.execute_programs(0);
        ems.apply_global_overrides();
        assert_eq!(ems.get_global("counter"), Some(1.0));

        ems.execute_programs(1);
        ems.apply_global_overrides();
        assert_eq!(ems.get_global("counter"), Some(2.0));
    }

    #[test]
    fn test_demand_response_program() {
        let mut ems = EmsManager::new();

        ems.add_global("dr_signal", 1.0); // Active demand response
        ems.add_sensor("zone_temp", EmsSensorType::ZoneTemperature(0));
        ems.add_actuator("heating", EmsActuatorType::HeatingSetpoint(0));

        let _heating_override = 20.0; // Default setpoint

        // Program that reduces heating setpoint by 2°C during demand response
        ems.add_program("demand_response", |runtime, _timestep| {
            if runtime
                .global_variables
                .get("dr_signal")
                .copied()
                .unwrap_or(0.0)
                > 0.0
            {
                let zone_temp = runtime
                    .sensor_values
                    .get("zone_temp")
                    .copied()
                    .unwrap_or(20.0);
                if zone_temp > 22.0 {
                    let new_setpoint = (zone_temp - 2.0).max(15.0);
                    runtime
                        .actuator_values
                        .insert("heating".to_string(), new_setpoint);
                }
            }
        });

        // Update sensors with zone temp of 24°C
        ems.update_sensors(&[24.0], 15.0, &[1000.0], &[0.0]);

        // Execute program
        ems.execute_programs(0);

        // Apply actuator overrides
        let mut heating_setpoints = vec![20.0];
        let cooling_setpoints = vec![27.0];
        let hvac_enabled = vec![true];
        let ventilation_multipliers = vec![1.0];
        let lighting_multipliers = vec![1.0];

        ems.apply_actuator_overrides(
            &mut heating_setpoints,
            &mut cooling_setpoints.clone(),
            &mut hvac_enabled.clone(),
            &mut ventilation_multipliers.clone(),
            &mut lighting_multipliers.clone(),
        );

        // With zone temp at 24°C and DR active, setpoint should be reduced to 22°C
        assert_eq!(heating_setpoints[0], 22.0);
    }

    #[test]
    fn test_actuator_apply() {
        let mut ems = EmsManager::new();
        ems.add_actuator("hvac_0", EmsActuatorType::HvacEnabled(0));

        ems.write_actuator("hvac_0", 0.0); // Disable HVAC

        let mut heating_setpoints = vec![20.0];
        let mut cooling_setpoints = vec![27.0];
        let mut hvac_enabled = vec![true];
        let mut ventilation_multipliers = vec![1.0];
        let mut lighting_multipliers = vec![1.0];

        ems.apply_actuator_overrides(
            &mut heating_setpoints,
            &mut cooling_setpoints,
            &mut hvac_enabled,
            &mut ventilation_multipliers,
            &mut lighting_multipliers,
        );

        assert!(!hvac_enabled[0]); // HVAC should be disabled
    }

    #[test]
    fn test_runtime_access() {
        let mut ems = EmsManager::new();
        ems.add_global("test", 42.0);

        // Create sensor and update values
        ems.add_sensor("temp", EmsSensorType::OutdoorTemperature);
        ems.update_sensors(&[], 25.0, &[], &[]);

        // Create runtime
        let sensor_values = ems.sensor_values.clone();
        let mut actuator_values = HashMap::new();
        let runtime = EmsRuntime::new(
            100, // timestep 100 = hour 4 of day 4
            &ems.global_variables,
            &sensor_values,
            &mut actuator_values,
        );

        assert_eq!(runtime.timestep, 100);
        assert_eq!(runtime.hour_of_day, 4);
        assert_eq!(runtime.day_of_year, 4);
        assert_eq!(runtime.global_variables.get("test"), Some(&42.0));
        assert_eq!(runtime.sensor_values.get("temp"), Some(&25.0));
    }

    #[test]
    fn test_timestep_hour_day_calculation() {
        // Test hour and day calculations for various timesteps
        let test_cases = vec![
            (0, 0, 0),       // Start
            (23, 23, 0),     // Last hour of day 0
            (24, 0, 1),      // First hour of day 1
            (168, 0, 7),     // Start of day 7
            (8759, 23, 364), // Last hour of year
        ];

        for (timestep, expected_hour, expected_day) in test_cases {
            let hour = timestep % 24;
            let day = timestep / 24;
            assert_eq!(hour, expected_hour, "timestep {}: hour mismatch", timestep);
            assert_eq!(day, expected_day, "timestep {}: day mismatch", timestep);
        }
    }
}

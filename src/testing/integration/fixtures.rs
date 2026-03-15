//! Test fixtures for integration tests
//!
//! Provides builder-pattern fixtures for constructing test scenarios
//! with sensible defaults for buildings, weather, and HVAC configurations.

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

/// HVAC equipment types for test scenarios
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HvacType {
    VAV,
    CAV,
    HeatPump,
    Chiller,
}

/// Builder for constructing test building scenarios
pub struct BuildingScenario {
    num_zones: usize,
    weather_path: Option<String>,
    hvac_type: Option<HvacType>,
    window_u_value: Option<f64>,
    heating_setpoint: Option<f64>,
    cooling_setpoint: Option<f64>,
}

impl BuildingScenario {
    /// Create a new builder with sensible defaults
    pub fn new() -> Self {
        Self {
            num_zones: 1,
            weather_path: None,
            hvac_type: None,
            window_u_value: None,
            heating_setpoint: None,
            cooling_setpoint: None,
        }
    }

    /// Set the number of zones
    pub fn with_zone_count(mut self, count: usize) -> Self {
        self.num_zones = count;
        self
    }

    /// Set the weather file path
    pub fn with_weather(mut self, path: &str) -> Self {
        self.weather_path = Some(path.to_string());
        self
    }

    /// Set the HVAC equipment type
    pub fn with_hvac(mut self, hvac_type: HvacType) -> Self {
        self.hvac_type = Some(hvac_type);
        self
    }

    /// Set the window U-value
    pub fn with_window_u_value(mut self, u_value: f64) -> Self {
        self.window_u_value = Some(u_value);
        self
    }

    /// Build and validate the scenario
    pub fn build(&self) -> Self {
        // TODO: Validate parameters and load weather file
        self.clone()
    }

    /// Create a ThermalModel from this scenario
    pub fn create_model(&self) -> ThermalModel<VectorField> {
        let mut model = ThermalModel::new(self.num_zones);

        if let Some(u) = self.window_u_value {
            model.window_u_value = u;
        }

        if let Some(sp) = self.heating_setpoint {
            model.heating_setpoint = sp;
        }

        if let Some(sp) = self.cooling_setpoint {
            model.cooling_setpoint = sp;
        }

        // TODO: Apply weather and HVAC configuration

        model
    }
}

impl Default for BuildingScenario {
    fn default() -> Self {
        Self::new()
    }
}

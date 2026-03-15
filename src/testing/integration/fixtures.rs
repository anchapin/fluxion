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
#[derive(Debug, Clone)]
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

    /// Set the heating setpoint
    pub fn with_heating_setpoint(mut self, sp: f64) -> Self {
        self.heating_setpoint = Some(sp);
        self
    }

    /// Set the cooling setpoint
    pub fn with_cooling_setpoint(mut self, sp: f64) -> Self {
        self.cooling_setpoint = Some(sp);
        self
    }

    /// Build and validate the scenario
    pub fn build(&self) -> Result<Self, String> {
        // Validate parameters
        if self.num_zones == 0 {
            return Err("num_zones must be > 0".to_string());
        }

        if let Some(u) = self.window_u_value {
            if u < 0.1 || u > 5.0 {
                return Err(format!("window_u_value must be in [0.1, 5.0], got {}", u));
            }
        }

        if let Some(sp) = self.heating_setpoint {
            if sp < 15.0 || sp > 30.0 {
                return Err(format!("heating_setpoint must be in [15, 30], got {}", sp));
            }
        }

        if let Some(sp) = self.cooling_setpoint {
            if sp < 15.0 || sp > 30.0 {
                return Err(format!("cooling_setpoint must be in [15, 30], got {}", sp));
            }
        }

        Ok(self.clone())
    }

    /// Create a ThermalModel from this scenario
    pub fn create_model(&self) -> ThermalModel<VectorField> {
        let mut model = ThermalModel::new(self.num_zones);

        // Apply window U-value or use default
        let u_value = self.window_u_value.unwrap_or(1.5);
        model.window_u_value = u_value;

        // Apply setpoints or use defaults
        let heating_sp = self.heating_setpoint.unwrap_or(20.0);
        let cooling_sp = self.cooling_setpoint.unwrap_or(26.0);
        model.heating_setpoint = heating_sp;
        model.cooling_setpoint = cooling_sp;

        // Initialize temperatures with sensible defaults
        model.temperatures = VectorField::from_scalar(heating_sp, self.num_zones);
        model.mass_temperatures = VectorField::from_scalar(20.0, self.num_zones);

        // Initialize other required fields
        model.loads = VectorField::from_scalar(0.0, self.num_zones);
        model.solar_gains = VectorField::from_scalar(0.0, self.num_zones);

        // Set default zone area and building parameters
        let zone_area = 100.0; // 100 m² per zone
        model.zone_area = VectorField::from_scalar(zone_area, self.num_zones);
        model.ceiling_height = VectorField::from_scalar(3.0, self.num_zones);
        model.air_density = VectorField::from_scalar(1.2, self.num_zones);
        model.heat_capacity = VectorField::from_scalar(1005.0, self.num_zones); // J/kg·K for air
        model.window_ratio = VectorField::from_scalar(0.3, self.num_zones);
        model.aspect_ratio = VectorField::from_scalar(1.5, self.num_zones);
        model.infiltration_rate = VectorField::from_scalar(0.5, self.num_zones); // 0.5 ACH

        // Set default building-wide parameters
        model.wall_u_value = 0.3; // W/m²K
        model.roof_u_value = 0.25; // W/m²K
        model.floor_u_value = 0.5; // W/m²K

        // Set HVAC capacity limits
        model.hvac_heating_capacity = zone_area * 100.0; // 100 W/m²
        model.hvac_cooling_capacity = zone_area * 100.0;

        // Set default case ID
        model.case_id = "test".to_string();

        model
    }
}

impl Default for BuildingScenario {
    fn default() -> Self {
        Self::new()
    }
}

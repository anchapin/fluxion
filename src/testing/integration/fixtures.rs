//! Test fixtures for integration tests
//!
//! Provides builder-pattern fixtures for constructing test scenarios
//! with sensible defaults for buildings, weather, and HVAC configurations.

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use std::sync::Arc;

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
    pub(crate) num_zones: usize,
    pub(crate) weather_path: Option<String>,
    pub(crate) hvac_type: Option<HvacType>,
    pub(crate) window_u_value: Option<f64>,
    pub(crate) heating_setpoint: Option<f64>,
    pub(crate) cooling_setpoint: Option<f64>,
    pub(crate) tracer: Option<Arc<super::wiring::WiringTracer>>,
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
            tracer: None,
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

    /// Set a wiring tracer for automatic call recording
    pub fn with_tracer(mut self, tracer: Arc<super::wiring::WiringTracer>) -> Self {
        self.tracer = Some(tracer);
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

        // Apply tracer if provided
        if let Some(ref tracer) = self.tracer {
            model.set_tracer(Arc::clone(tracer));
        }

        // Apply window U-value or use default
        let u_value = self.window_u_value.unwrap_or(1.5);
        model.solar.window_u_value = u_value;

        // Apply setpoints or use defaults
        let heating_sp = self.heating_setpoint.unwrap_or(20.0);
        let cooling_sp = self.cooling_setpoint.unwrap_or(26.0);
        model.setpoints.heating_setpoint = heating_sp;
        model.setpoints.cooling_setpoint = cooling_sp;

        // Initialize temperatures with sensible defaults
        model.setpoints.temperatures = VectorField::from_scalar(heating_sp, self.num_zones);
        model.mass.mass_temperatures = VectorField::from_scalar(20.0, self.num_zones);

        // Initialize other required fields
        model.setpoints.loads = VectorField::from_scalar(0.0, self.num_zones);
        model.solar.solar_gains = VectorField::from_scalar(0.0, self.num_zones);

        // Set default zone area and building parameters
        let zone_area = 100.0; // 100 m² per zone
        model.setpoints.zone_area = VectorField::from_scalar(zone_area, self.num_zones);
        model.setpoints.ceiling_height = VectorField::from_scalar(3.0, self.num_zones);
        model.setpoints.air_density = VectorField::from_scalar(1.2, self.num_zones);
        model.setpoints.heat_capacity = VectorField::from_scalar(1005.0, self.num_zones); // J/kg·K for air
        model.setpoints.window_ratio = VectorField::from_scalar(0.3, self.num_zones);
        model.setpoints.aspect_ratio = VectorField::from_scalar(1.5, self.num_zones);
        model.setpoints.infiltration_rate = VectorField::from_scalar(0.5, self.num_zones); // 0.5 ACH

        // Set default building-wide parameters
        model.setpoints.wall_u_value = 0.3; // W/m²K
        model.setpoints.roof_u_value = 0.25; // W/m²K
        model.setpoints.floor_u_value = 0.5; // W/m²K

        // Set HVAC capacity limits
        model.hvac.hvac_heating_capacity = zone_area * 100.0; // 100 W/m²
        model.hvac.hvac_cooling_capacity = zone_area * 100.0;

        // Set default case ID
        model.hvac.case_id = "test".to_string();

        // Compute thermal capacitance: air thermal mass + building thermal mass
        // Air thermal mass = volume * density * cp = 300 * 1.2 * 1005 = 361,800 J/K
        // Add building thermal mass for walls, furniture, etc. (approximately 3x air thermal mass)
        let volume = zone_area * 3.0; // 300 m³
        let air_thermal_mass = volume * 1.2 * 1005.0; // J/K
        let total_thermal_mass = air_thermal_mass * 4.0; // Add building thermal mass
        model.mass.thermal_capacitance =
            VectorField::from_scalar(total_thermal_mass, self.num_zones);

        // Initialize h_tr_ms (surface-to-mass conductance) - required for 5R1C model
        // h_tr_ms = 6.83 W/m²K * wall_area for typical construction
        let perimeter = 2.0 * ((zone_area * 1.5).sqrt() + (zone_area / (zone_area * 1.5).sqrt()));
        let wall_area = perimeter * 3.0 - zone_area * 0.3; // Approximate wall area minus windows
        let h_tr_ms = 6.83 * wall_area; // W/K
        model.conduction.h_tr_ms = VectorField::from_scalar(h_tr_ms, self.num_zones);

        // Update derived physics parameters
        model.update_derived_parameters();

        model
    }
}

impl Default for BuildingScenario {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_building_scenario_new_has_defaults() {
        let scenario = BuildingScenario::new();
        assert_eq!(scenario.num_zones, 1);
        assert!(scenario.weather_path.is_none());
        assert!(scenario.hvac_type.is_none());
        assert!(scenario.window_u_value.is_none());
        assert!(scenario.heating_setpoint.is_none());
        assert!(scenario.cooling_setpoint.is_none());
        assert!(scenario.tracer.is_none());
    }

    #[test]
    fn test_building_scenario_default_equals_new() {
        let default = BuildingScenario::default();
        let new = BuildingScenario::new();
        assert_eq!(default.num_zones, new.num_zones);
    }

    #[test]
    fn test_building_scenario_with_zone_count() {
        let scenario = BuildingScenario::new().with_zone_count(3);
        assert_eq!(scenario.num_zones, 3);
    }

    #[test]
    fn test_building_scenario_with_weather() {
        let scenario = BuildingScenario::new().with_weather("/path/to/weather.epw");
        assert_eq!(
            scenario.weather_path,
            Some("/path/to/weather.epw".to_string())
        );
    }

    #[test]
    fn test_building_scenario_with_hvac() {
        let scenario = BuildingScenario::new().with_hvac(HvacType::VAV);
        assert_eq!(scenario.hvac_type, Some(HvacType::VAV));
    }

    #[test]
    fn test_building_scenario_with_window_u_value() {
        let scenario = BuildingScenario::new().with_window_u_value(2.5);
        assert_eq!(scenario.window_u_value, Some(2.5));
    }

    #[test]
    fn test_building_scenario_with_heating_setpoint() {
        let scenario = BuildingScenario::new().with_heating_setpoint(18.0);
        assert_eq!(scenario.heating_setpoint, Some(18.0));
    }

    #[test]
    fn test_building_scenario_with_cooling_setpoint() {
        let scenario = BuildingScenario::new().with_cooling_setpoint(24.0);
        assert_eq!(scenario.cooling_setpoint, Some(24.0));
    }

    #[test]
    fn test_building_scenario_builder_chaining() {
        let scenario = BuildingScenario::new()
            .with_zone_count(2)
            .with_window_u_value(1.8)
            .with_heating_setpoint(19.0)
            .with_cooling_setpoint(25.0);

        assert_eq!(scenario.num_zones, 2);
        assert_eq!(scenario.window_u_value, Some(1.8));
        assert_eq!(scenario.heating_setpoint, Some(19.0));
        assert_eq!(scenario.cooling_setpoint, Some(25.0));
    }

    #[test]
    fn test_building_scenario_build_success_valid_params() {
        let scenario = BuildingScenario::new()
            .with_zone_count(1)
            .with_window_u_value(1.5)
            .with_heating_setpoint(20.0)
            .with_cooling_setpoint(26.0)
            .build();

        assert!(scenario.is_ok());
    }

    #[test]
    fn test_building_scenario_build_fails_zero_zones() {
        let result = BuildingScenario::new().with_zone_count(0).build();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("num_zones"));
    }

    #[test]
    fn test_building_scenario_build_fails_low_window_u_value() {
        let result = BuildingScenario::new().with_window_u_value(0.05).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_building_scenario_build_fails_high_window_u_value() {
        let result = BuildingScenario::new().with_window_u_value(6.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_building_scenario_build_fails_low_heating_setpoint() {
        let result = BuildingScenario::new().with_heating_setpoint(10.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_building_scenario_build_fails_high_heating_setpoint() {
        let result = BuildingScenario::new().with_heating_setpoint(35.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_building_scenario_build_fails_low_cooling_setpoint() {
        let result = BuildingScenario::new().with_cooling_setpoint(10.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_building_scenario_build_fails_high_cooling_setpoint() {
        let result = BuildingScenario::new().with_cooling_setpoint(35.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_building_scenario_build_boundary_values_window_u() {
        let result_min = BuildingScenario::new().with_window_u_value(0.1).build();
        assert!(result_min.is_ok());

        let result_max = BuildingScenario::new().with_window_u_value(5.0).build();
        assert!(result_max.is_ok());
    }

    #[test]
    fn test_building_scenario_build_boundary_values_setpoints() {
        let result_heat_min = BuildingScenario::new().with_heating_setpoint(15.0).build();
        assert!(result_heat_min.is_ok());

        let result_heat_max = BuildingScenario::new().with_heating_setpoint(30.0).build();
        assert!(result_heat_max.is_ok());

        let result_cool_min = BuildingScenario::new().with_cooling_setpoint(15.0).build();
        assert!(result_cool_min.is_ok());

        let result_cool_max = BuildingScenario::new().with_cooling_setpoint(30.0).build();
        assert!(result_cool_max.is_ok());
    }

    #[test]
    fn test_building_scenario_create_model_defaults() {
        let scenario = BuildingScenario::new().build().unwrap();
        let model = scenario.create_model();

        assert_eq!(model.solar.window_u_value, 1.5);
        assert_eq!(model.setpoints.heating_setpoint, 20.0);
        assert_eq!(model.setpoints.cooling_setpoint, 26.0);
        assert_eq!(model.hvac.case_id, "test");
    }

    #[test]
    fn test_building_scenario_create_model_custom_values() {
        let scenario = BuildingScenario::new()
            .with_zone_count(2)
            .with_window_u_value(2.0)
            .with_heating_setpoint(18.0)
            .with_cooling_setpoint(24.0)
            .build()
            .unwrap();

        let model = scenario.create_model();

        assert_eq!(model.solar.window_u_value, 2.0);
        assert_eq!(model.setpoints.heating_setpoint, 18.0);
        assert_eq!(model.setpoints.cooling_setpoint, 24.0);
    }

    #[test]
    fn test_building_scenario_create_model_initializes_temperatures() {
        let scenario = BuildingScenario::new()
            .with_zone_count(3)
            .with_heating_setpoint(19.0)
            .build()
            .unwrap();

        let model = scenario.create_model();

        assert_eq!(model.setpoints.temperatures.len(), 3);
        assert_eq!(model.mass.mass_temperatures.len(), 3);
        assert_eq!(model.setpoints.zone_area.len(), 3);
    }

    #[test]
    fn test_building_scenario_with_tracer() {
        use crate::testing::integration::wiring::WiringTracer;
        use std::sync::Arc;

        let tracer = Arc::new(WiringTracer::new());
        let scenario = BuildingScenario::new().with_tracer(tracer.clone());
        let _model = scenario.create_model();

        tracer.record_call("test_function");
        assert!(tracer.verify_called(&["test_function"]));
    }

    #[test]
    fn test_hvac_type_all_variants() {
        let types = [
            HvacType::VAV,
            HvacType::CAV,
            HvacType::HeatPump,
            HvacType::Chiller,
        ];

        for hvac_type in types {
            let scenario = BuildingScenario::new()
                .with_hvac(hvac_type)
                .build()
                .unwrap();
            assert_eq!(scenario.hvac_type, Some(hvac_type));
        }
    }
}

//! DOE Commercial Reference Building configurations
//!
//! Provides structurally representative configurations for:
//! - Small Office (500 m², 4 zones)
//! - Medium Office (5000 m², 12 zones)
//! - Stand-alone Retail (2500 m², 4 zones)
//!
//! These are simplified but structurally representative of DOE Prototype Buildings
//! for testing engine scalability with multi-zone commercial buildings.

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::testing::integration::fixtures::BuildingScenario;
use std::time::Instant;

/// DOE Building Types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DoeBuildingType {
    SmallOffice,
    MediumOffice,
    StandaloneRetail,
}

/// DOE Building configuration with representative zone structure
#[derive(Debug, Clone)]
pub struct DoeBuildingConfig {
    pub building_type: DoeBuildingType,
    pub num_zones: usize,
    pub total_floor_area_m2: f64,
    pub zone_area_m2: f64,
    pub ceiling_height_m: f64,
    pub window_to_wall_ratio: f64,
    pub wall_u_value_w_m2k: f64,
    pub roof_u_value_w_m2k: f64,
    pub window_u_value_w_m2k: f64,
    pub infiltration_rate_ach: f64,
    pub heating_setpoint_c: f64,
    pub cooling_setpoint_c: f64,
    pub internal_loads_w_m2: f64,
    pub hvac_heating_capacity_w_m2: f64,
    pub hvac_cooling_capacity_w_m2: f64,
}

impl DoeBuildingConfig {
    /// Small Office - 511 m², 4 zones, 1 floor
    /// Based on DOE Prototype Building: SmallOffice
    pub fn small_office() -> Self {
        Self {
            building_type: DoeBuildingType::SmallOffice,
            num_zones: 4,
            total_floor_area_m2: 511.0,
            zone_area_m2: 127.75,
            ceiling_height_m: 3.0,
            window_to_wall_ratio: 0.18,
            wall_u_value_w_m2k: 0.55,
            roof_u_value_w_m2k: 0.25,
            window_u_value_w_m2k: 2.78,
            infiltration_rate_ach: 0.5,
            heating_setpoint_c: 21.0,
            cooling_setpoint_c: 24.0,
            internal_loads_w_m2: 15.0,
            hvac_heating_capacity_w_m2: 80.0,
            hvac_cooling_capacity_w_m2: 80.0,
        }
    }

    /// Medium Office - 4982 m², 12 zones, 3 floors
    /// Based on DOE Prototype Building: MediumOffice
    pub fn medium_office() -> Self {
        Self {
            building_type: DoeBuildingType::MediumOffice,
            num_zones: 12,
            total_floor_area_m2: 4982.0,
            zone_area_m2: 415.17,
            ceiling_height_m: 2.7,
            window_to_wall_ratio: 0.33,
            wall_u_value_w_m2k: 0.55,
            roof_u_value_w_m2k: 0.25,
            window_u_value_w_m2k: 2.78,
            infiltration_rate_ach: 0.5,
            heating_setpoint_c: 21.0,
            cooling_setpoint_c: 24.0,
            internal_loads_w_m2: 16.0,
            hvac_heating_capacity_w_m2: 80.0,
            hvac_cooling_capacity_w_m2: 80.0,
        }
    }

    /// Stand-alone Retail - 2326 m², 4 zones, 1 floor
    /// Based on DOE Prototype Building: RetailStandalone
    pub fn standalone_retail() -> Self {
        Self {
            building_type: DoeBuildingType::StandaloneRetail,
            num_zones: 4,
            total_floor_area_m2: 2326.0,
            zone_area_m2: 581.5,
            ceiling_height_m: 4.5,
            window_to_wall_ratio: 0.07,
            wall_u_value_w_m2k: 0.55,
            roof_u_value_w_m2k: 0.25,
            window_u_value_w_m2k: 2.78,
            infiltration_rate_ach: 0.5,
            heating_setpoint_c: 20.0,
            cooling_setpoint_c: 25.0,
            internal_loads_w_m2: 14.0,
            hvac_heating_capacity_w_m2: 70.0,
            hvac_cooling_capacity_w_m2: 70.0,
        }
    }

    /// Create a BuildingScenario from this DOE configuration
    pub fn to_building_scenario(&self) -> BuildingScenario {
        let mut scenario = BuildingScenario::new();
        scenario = scenario.with_zone_count(self.num_zones);
        scenario = scenario.with_window_u_value(self.window_u_value_w_m2k);
        scenario = scenario.with_heating_setpoint(self.heating_setpoint_c);
        scenario = scenario.with_cooling_setpoint(self.cooling_setpoint_c);
        scenario
            .build()
            .expect("DOE building config validation failed")
    }

    /// Create a ThermalModel from this DOE configuration
    pub fn create_model(&self) -> ThermalModel<VectorField> {
        let mut model = ThermalModel::new(self.num_zones);

        // Zone properties
        model.setpoints.zone_area = VectorField::from_scalar(self.zone_area_m2, self.num_zones);
        model.setpoints.ceiling_height =
            VectorField::from_scalar(self.ceiling_height_m, self.num_zones);
        model.setpoints.window_ratio =
            VectorField::from_scalar(self.window_to_wall_ratio, self.num_zones);
        model.setpoints.aspect_ratio = VectorField::from_scalar(1.5, self.num_zones);
        model.setpoints.infiltration_rate =
            VectorField::from_scalar(self.infiltration_rate_ach, self.num_zones);

        // Building envelope
        model.setpoints.wall_u_value = self.wall_u_value_w_m2k;
        model.setpoints.roof_u_value = self.roof_u_value_w_m2k;
        model.solar.window_u_value = self.window_u_value_w_m2k;

        // Setpoints
        model.setpoints.heating_setpoint = self.heating_setpoint_c;
        model.setpoints.cooling_setpoint = self.cooling_setpoint_c;
        model.setpoints.heating_setpoints =
            VectorField::from_scalar(self.heating_setpoint_c, self.num_zones);
        model.setpoints.cooling_setpoints =
            VectorField::from_scalar(self.cooling_setpoint_c, self.num_zones);

        // HVAC capacities
        let total_capacity = self.total_floor_area_m2 * self.hvac_heating_capacity_w_m2;
        model.hvac.hvac_heating_capacity = total_capacity;
        model.hvac.hvac_cooling_capacity = total_capacity;

        // Air properties
        model.setpoints.air_density = VectorField::from_scalar(1.2, self.num_zones);
        model.setpoints.heat_capacity = VectorField::from_scalar(1005.0, self.num_zones);

        // Initial temperatures
        model.setpoints.temperatures =
            VectorField::from_scalar(self.heating_setpoint_c, self.num_zones);
        model.mass.mass_temperatures =
            VectorField::from_scalar(self.heating_setpoint_c, self.num_zones);

        // Internal loads
        model.setpoints.loads =
            VectorField::from_scalar(self.internal_loads_w_m2 * self.zone_area_m2, self.num_zones);
        model.solar.solar_gains = VectorField::from_scalar(0.0, self.num_zones);

        // Case ID
        let name = match self.building_type {
            DoeBuildingType::SmallOffice => "DOE_SmallOffice",
            DoeBuildingType::MediumOffice => "DOE_MediumOffice",
            DoeBuildingType::StandaloneRetail => "DOE_RetailStandalone",
        };
        model.hvac.case_id = name.to_string();

        model
    }
}

impl Default for DoeBuildingConfig {
    fn default() -> Self {
        Self::small_office()
    }
}

/// Memory usage statistics for simulation
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub model_node_count: usize,
}

/// Run annual simulation and collect performance metrics
pub fn run_annual_simulation(
    config: &DoeBuildingConfig,
) -> Result<(f64, MemoryStats, std::time::Duration), String> {
    let mut model = config.create_model();
    let node_count = config.num_zones; // Simplified node count

    let start = Instant::now();
    let energy = {
        let surrogates =
            crate::ai::surrogate::SurrogateManager::new().map_err(|e| e.to_string())?;
        model.solve_timesteps(8760, &surrogates, false, None, None, None)
    };
    let elapsed = start.elapsed();

    // Memory stats (simplified - real implementation would use dhat or similar)
    let stats = MemoryStats {
        allocated_bytes: 0,
        peak_bytes: 0,
        model_node_count: node_count,
    };

    Ok((energy, stats, elapsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_office_config() {
        let config = DoeBuildingConfig::small_office();
        assert_eq!(config.num_zones, 4);
        assert!((config.total_floor_area_m2 - 511.0).abs() < 1.0);
        assert!(config.window_u_value_w_m2k > 0.0);
    }

    #[test]
    fn test_medium_office_config() {
        let config = DoeBuildingConfig::medium_office();
        assert_eq!(config.num_zones, 12);
        assert!((config.total_floor_area_m2 - 4982.0).abs() < 10.0);
    }

    #[test]
    fn test_retail_config() {
        let config = DoeBuildingConfig::standalone_retail();
        assert_eq!(config.num_zones, 4);
        assert!((config.total_floor_area_m2 - 2326.0).abs() < 10.0);
    }

    #[test]
    fn test_small_office_creates_valid_model() {
        let config = DoeBuildingConfig::small_office();
        let model = config.create_model();
        assert_eq!(model.hvac.case_id, "DOE_SmallOffice");
        assert!(model.setpoints.heating_setpoint > 0.0);
        assert!(model.setpoints.cooling_setpoint > model.setpoints.heating_setpoint);
    }

    #[test]
    fn test_all_configs_create_valid_models() {
        let configs = [
            DoeBuildingConfig::small_office(),
            DoeBuildingConfig::medium_office(),
            DoeBuildingConfig::standalone_retail(),
        ];

        for config in configs {
            let model = config.create_model();
            assert!(
                model.solar.window_u_value > 0.0,
                "{}: window_u_value should be positive",
                model.hvac.case_id
            );
            assert!(
                model.setpoints.heating_setpoint > 0.0,
                "{}: heating_setpoint should be positive",
                model.hvac.case_id
            );
            assert!(
                model.setpoints.cooling_setpoint > model.setpoints.heating_setpoint,
                "{}: cooling_setpoint should be > heating_setpoint",
                model.hvac.case_id
            );
            assert_eq!(
                model.setpoints.temperatures.len(),
                config.num_zones,
                "{}: temperatures should have num_zones length",
                model.hvac.case_id
            );
        }
    }

    #[test]
    fn test_building_scenario_from_config() {
        let config = DoeBuildingConfig::small_office();
        let scenario = config.to_building_scenario();
        let built = scenario.build().expect("scenario should build");
        assert_eq!(built.num_zones, 4);
    }
}

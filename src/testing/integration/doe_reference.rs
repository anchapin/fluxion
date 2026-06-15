//! DOE Commercial Reference Buildings
//!
//! This module provides DOE Commercial Reference Building configurations
//! for integration testing. These are standardized building models developed
//! by the US Department of Energy for building energy simulation validation.
//!
//! Reference: https://www.energycodes.gov/development/commercial_prototype_building_models

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

/// DOE Commercial Reference Building types
/// These correspond to the DOE prototype building models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoeBuildingType {
    /// Small Office (511 m²)
    SmallOffice,
    /// Medium Office (4982 m²)
    MediumOffice,
    /// Large Office (4630 m²)
    LargeOffice,
    /// Retail Standalone (2310 m²)
    RetailStandalone,
    /// Retail Stripmall (2370 m²)
    RetailStripmall,
    /// Warehouse (5130 m²)
    Warehouse,
    /// Quick Service Restaurant (195 m²)
    QuickServiceRestaurant,
    /// Full Service Restaurant (565 m²)
    FullServiceRestaurant,
    /// Midrise Apartment (3330 m²)
    MidriseApartment,
    /// Highrise Apartment (7460 m²)
    HighriseApartment,
    /// Hospital (22400 m²)
    Hospital,
    /// Outpatient (6880 m²)
    Outpatient,
    /// SuperMarket (4180 m²)
    SuperMarket,
    /// Primary School (5650 m²)
    PrimarySchool,
    /// Secondary School (19570 m²)
    SecondarySchool,
}

impl DoeBuildingType {
    /// Get the expected zone count for this building type
    pub fn zone_count(&self) -> usize {
        match self {
            DoeBuildingType::SmallOffice => 1,
            DoeBuildingType::MediumOffice => 3,
            DoeBuildingType::LargeOffice => 10,
            DoeBuildingType::RetailStandalone => 1,
            DoeBuildingType::RetailStripmall => 4,
            DoeBuildingType::Warehouse => 1,
            DoeBuildingType::QuickServiceRestaurant => 1,
            DoeBuildingType::FullServiceRestaurant => 2,
            DoeBuildingType::MidriseApartment => 4,
            DoeBuildingType::HighriseApartment => 10,
            DoeBuildingType::Hospital => 10,
            DoeBuildingType::Outpatient => 5,
            DoeBuildingType::SuperMarket => 1,
            DoeBuildingType::PrimarySchool => 6,
            DoeBuildingType::SecondarySchool => 8,
        }
    }

    /// Get the floor area in m² for this building type
    pub fn floor_area_m2(&self) -> f64 {
        match self {
            DoeBuildingType::SmallOffice => 511.0,
            DoeBuildingType::MediumOffice => 4982.0,
            DoeBuildingType::LargeOffice => 4630.0,
            DoeBuildingType::RetailStandalone => 2310.0,
            DoeBuildingType::RetailStripmall => 2370.0,
            DoeBuildingType::Warehouse => 5130.0,
            DoeBuildingType::QuickServiceRestaurant => 195.0,
            DoeBuildingType::FullServiceRestaurant => 565.0,
            DoeBuildingType::MidriseApartment => 3330.0,
            DoeBuildingType::HighriseApartment => 7460.0,
            DoeBuildingType::Hospital => 22400.0,
            DoeBuildingType::Outpatient => 6880.0,
            DoeBuildingType::SuperMarket => 4180.0,
            DoeBuildingType::PrimarySchool => 5650.0,
            DoeBuildingType::SecondarySchool => 19570.0,
        }
    }

    /// Get the window-to-wall ratio for this building type
    pub fn window_to_wall_ratio(&self) -> f64 {
        match self {
            DoeBuildingType::SmallOffice => 0.15,
            DoeBuildingType::MediumOffice => 0.18,
            DoeBuildingType::LargeOffice => 0.30,
            DoeBuildingType::RetailStandalone => 0.10,
            DoeBuildingType::RetailStripmall => 0.12,
            DoeBuildingType::Warehouse => 0.02,
            DoeBuildingType::QuickServiceRestaurant => 0.10,
            DoeBuildingType::FullServiceRestaurant => 0.15,
            DoeBuildingType::MidriseApartment => 0.20,
            DoeBuildingType::HighriseApartment => 0.35,
            DoeBuildingType::Hospital => 0.25,
            DoeBuildingType::Outpatient => 0.20,
            DoeBuildingType::SuperMarket => 0.08,
            DoeBuildingType::PrimarySchool => 0.20,
            DoeBuildingType::SecondarySchool => 0.25,
        }
    }

    /// Get the lighting power density W/m² for this building type
    pub fn lighting_wpm2(&self) -> f64 {
        match self {
            DoeBuildingType::SmallOffice => 8.0,
            DoeBuildingType::MediumOffice => 9.1,
            DoeBuildingType::LargeOffice => 9.7,
            DoeBuildingType::RetailStandalone => 9.6,
            DoeBuildingType::RetailStripmall => 9.1,
            DoeBuildingType::Warehouse => 3.8,
            DoeBuildingType::QuickServiceRestaurant => 12.0,
            DoeBuildingType::FullServiceRestaurant => 11.0,
            DoeBuildingType::MidriseApartment => 6.5,
            DoeBuildingType::HighriseApartment => 6.5,
            DoeBuildingType::Hospital => 10.0,
            DoeBuildingType::Outpatient => 10.0,
            DoeBuildingType::SuperMarket => 12.0,
            DoeBuildingType::PrimarySchool => 9.0,
            DoeBuildingType::SecondarySchool => 9.0,
        }
    }

    /// Get the equipment power density W/m² for this building type
    pub fn equipment_wpm2(&self) -> f64 {
        match self {
            DoeBuildingType::SmallOffice => 7.5,
            DoeBuildingType::MediumOffice => 8.0,
            DoeBuildingType::LargeOffice => 8.4,
            DoeBuildingType::RetailStandalone => 6.0,
            DoeBuildingType::RetailStripmall => 6.0,
            DoeBuildingType::Warehouse => 0.5,
            DoeBuildingType::QuickServiceRestaurant => 10.0,
            DoeBuildingType::FullServiceRestaurant => 15.0,
            DoeBuildingType::MidriseApartment => 4.0,
            DoeBuildingType::HighriseApartment => 4.0,
            DoeBuildingType::Hospital => 12.0,
            DoeBuildingType::Outpatient => 10.0,
            DoeBuildingType::SuperMarket => 3.0,
            DoeBuildingType::PrimarySchool => 5.0,
            DoeBuildingType::SecondarySchool => 5.0,
        }
    }

    /// Get the infiltration rate in ACH for this building type
    pub fn infiltration_ach(&self) -> f64 {
        match self {
            DoeBuildingType::SmallOffice => 0.5,
            DoeBuildingType::MediumOffice => 0.3,
            DoeBuildingType::LargeOffice => 0.2,
            DoeBuildingType::RetailStandalone => 0.5,
            DoeBuildingType::RetailStripmall => 0.5,
            DoeBuildingType::Warehouse => 0.3,
            DoeBuildingType::QuickServiceRestaurant => 1.0,
            DoeBuildingType::FullServiceRestaurant => 0.8,
            DoeBuildingType::MidriseApartment => 0.4,
            DoeBuildingType::HighriseApartment => 0.3,
            DoeBuildingType::Hospital => 0.3,
            DoeBuildingType::Outpatient => 0.4,
            DoeBuildingType::SuperMarket => 0.5,
            DoeBuildingType::PrimarySchool => 0.4,
            DoeBuildingType::SecondarySchool => 0.4,
        }
    }

    /// Get the ASHRAE 90.1 climate zone for this building type
    pub fn climate_zone(&self) -> &'static str {
        "ASHRAE 169-2013-4A"
    }

    /// Create a ThermalModel configured for this DOE building type
    pub fn create_model(&self) -> ThermalModel<VectorField> {
        let num_zones = self.zone_count();
        let floor_area = self.floor_area_m2();
        let zone_area = floor_area / num_zones as f64;
        let wwr = self.window_to_wall_ratio();

        let mut model = ThermalModel::new(num_zones);

        // Set window properties
        model.window_u_value = 2.7; // W/m²K typical for DOE buildings
        model.window_ratio = VectorField::from_scalar(wwr, num_zones);

        // Set setpoints
        model.heating_setpoint = 21.0; // 21°C
        model.cooling_setpoint = 24.0; // 24°C

        // Initialize temperatures
        model.temperatures = VectorField::from_scalar(21.0, num_zones);
        model.mass_temperatures = VectorField::from_scalar(21.0, num_zones);

        // Set zone properties
        model.zone_area = VectorField::from_scalar(zone_area, num_zones);
        model.ceiling_height = VectorField::from_scalar(3.0, num_zones);
        model.air_density = VectorField::from_scalar(1.2, num_zones);
        model.heat_capacity = VectorField::from_scalar(1005.0, num_zones);
        model.aspect_ratio = VectorField::from_scalar(1.5, num_zones);
        model.infiltration_rate = VectorField::from_scalar(self.infiltration_ach(), num_zones);

        // Set building envelope properties
        model.wall_u_value = 0.3; // W/m²K
        model.roof_u_value = 0.25; // W/m²K
        model.floor_u_value = 0.35; // W/m²K

        // Set HVAC capacity
        model.hvac_heating_capacity = zone_area * 80.0; // W/m²
        model.hvac_cooling_capacity = zone_area * 100.0; // W/m²

        // Set case ID
        model.case_id = format!("DOE-{:?}", self);

        // Initialize loads
        model.loads = VectorField::from_scalar(0.0, num_zones);
        model.solar_gains = VectorField::from_scalar(0.0, num_zones);

        model
    }
}

// ============================================================================
// Scenario Builders
// ============================================================================

/// Create a DOE Small Office scenario
pub fn doe_small_office_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::SmallOffice.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(21.0)
        .with_cooling_setpoint(24.0)
        .build()
        .expect("DOE Small Office scenario validation failed")
}

/// Create a DOE Medium Office scenario
pub fn doe_medium_office_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::MediumOffice.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(21.0)
        .with_cooling_setpoint(24.0)
        .build()
        .expect("DOE Medium Office scenario validation failed")
}

/// Create a DOE Large Office scenario
pub fn doe_large_office_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::LargeOffice.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(21.0)
        .with_cooling_setpoint(24.0)
        .build()
        .expect("DOE Large Office scenario validation failed")
}

/// Create a DOE Retail Standalone scenario
pub fn doe_retail_standalone_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::RetailStandalone.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(21.0)
        .with_cooling_setpoint(24.0)
        .build()
        .expect("DOE Retail Standalone scenario validation failed")
}

/// Create a DOE Warehouse scenario
pub fn doe_warehouse_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::Warehouse.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(18.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("DOE Warehouse scenario validation failed")
}

/// Create a DOE Restaurant scenario
pub fn doe_restaurant_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::FullServiceRestaurant.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(21.0)
        .with_cooling_setpoint(24.0)
        .build()
        .expect("DOE Restaurant scenario validation failed")
}

/// Create a DOE Hospital scenario
pub fn doe_hospital_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::Hospital.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(22.0)
        .with_cooling_setpoint(24.0)
        .build()
        .expect("DOE Hospital scenario validation failed")
}

/// Create a DOE School scenario
pub fn doe_school_scenario() -> super::fixtures::BuildingScenario {
    super::fixtures::BuildingScenario::new()
        .with_zone_count(DoeBuildingType::PrimarySchool.zone_count())
        .with_window_u_value(2.7)
        .with_heating_setpoint(21.0)
        .with_cooling_setpoint(24.0)
        .build()
        .expect("DOE School scenario validation failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doe_building_type_properties() {
        let small_office = DoeBuildingType::SmallOffice;
        assert_eq!(small_office.zone_count(), 1);
        assert_eq!(small_office.floor_area_m2(), 511.0);
        assert_eq!(small_office.window_to_wall_ratio(), 0.15);

        let hospital = DoeBuildingType::Hospital;
        assert_eq!(hospital.zone_count(), 10);
        assert_eq!(hospital.floor_area_m2(), 22400.0);
        assert_eq!(hospital.window_to_wall_ratio(), 0.25);
    }

    #[test]
    fn test_doe_scenario_builders() {
        let scenarios = [
            ("small_office", doe_small_office_scenario()),
            ("medium_office", doe_medium_office_scenario()),
            ("large_office", doe_large_office_scenario()),
            ("retail_standalone", doe_retail_standalone_scenario()),
            ("warehouse", doe_warehouse_scenario()),
            ("restaurant", doe_restaurant_scenario()),
            ("hospital", doe_hospital_scenario()),
            ("school", doe_school_scenario()),
        ];

        for (name, scenario) in scenarios {
            let built = scenario.build();
            assert!(built.is_ok(), "{} scenario should build successfully", name);
        }
    }

    #[test]
    fn test_doe_small_office_model_creation() {
        let model = DoeBuildingType::SmallOffice.create_model();
        assert_eq!(model.temperatures.len(), 1);
        assert!(model.window_u_value > 0.0);
    }

    #[test]
    fn test_doe_hospital_model_creation() {
        let model = DoeBuildingType::Hospital.create_model();
        assert_eq!(model.temperatures.len(), 10);
        assert!(model.window_u_value > 0.0);
    }
}

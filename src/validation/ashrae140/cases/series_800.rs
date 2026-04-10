//! ASHRAE 140 Cases 800-810: HVAC Equipment Validation
//!
//! This module implements ASHRAE 140 Cases 800-810 which focus on
//! HVAC equipment validation including heat pumps, chillers, boilers,
//! and comprehensive HVAC system configurations.

#![allow(clippy::needless_update)]

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae140::AshraeZone;
use crate::validation::ashrae140::BuildingProperties;
use crate::validation::ashrae140::HVACSystem;
use crate::validation::ashrae140::SimulationParameters;
use crate::validation::ashrae140::WeatherData;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

// Additional types needed for HVAC configurations
pub enum ChillerType {
    Centrifugal,
    Reciprocating,
    Screw,
    Scroll,
}

pub enum CondenserType {
    AirCooled,
    WaterCooled,
    Evaporative,
}

pub enum BoilerType {
    GasFired,
    OilFired,
    Electric,
    Condensing,
}

/// Build an ASHRAE 140 case definition for Cases 800-810
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    match case {
        ASHRAE140Case::Case800 => build_case_800(),
        ASHRAE140Case::Case801 => build_case_801(),
        ASHRAE140Case::Case802 => build_case_802(),
        ASHRAE140Case::Case803 => build_case_803(),
        ASHRAE140Case::Case804 => build_case_804(),
        ASHRAE140Case::Case805 => build_case_805(),
        ASHRAE140Case::Case806 => build_case_806(),
        ASHRAE140Case::Case807 => build_case_807(),
        ASHRAE140Case::Case808 => build_case_808(),
        ASHRAE140Case::Case809 => build_case_809(),
        ASHRAE140Case::Case810 => build_case_810(),
        _ => panic!("Invalid case for series 800: {:?}", case),
    }
}

/// Case 800: Heat pump (single-stage, basic control)
fn build_case_800() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case800,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 232.0,   // m²
            u_value: 0.45,       // W/m²K - exterior walls
            thermal_mass: 120.0, // kJ/m²K - lightweight construction
            window_wall_ratio: 0.2,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::HeatPump,
            heating_capacity: 12000.0, // W
            cooling_capacity: 10000.0, // W
            cop_heating: 3.2,
            cop_cooling: 3.5,
            stages: 1,               // Single-stage
            min_outdoor_temp: -15.0, // °C
            max_outdoor_temp: 40.0,  // °C
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone5A),
        simulation_parameters: crate::validation::ashrae140::SimulationParameters {
            timestep: 3600,         // 1 hour
            total_hours: 8760,      // Full year
            setpoint_heating: 20.0, // °C
            setpoint_cooling: 24.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 801: Heat pump (two-stage, intermediate control)
fn build_case_801() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case801,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 232.0,   // m²
            u_value: 0.45,       // W/m²K
            thermal_mass: 120.0, // kJ/m²K
            window_wall_ratio: 0.2,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::HeatPump,
            heating_capacity: 15000.0, // W - higher capacity for two-stage
            cooling_capacity: 12000.0, // W
            cop_heating: 3.4,
            cop_cooling: 3.7,
            stages: 2,               // Two-stage
            stage1_capacity: 0.6,    // 60% of total capacity in stage 1
            stage2_capacity: 1.0,    // 100% of total capacity in stage 2
            min_outdoor_temp: -20.0, // °C
            max_outdoor_temp: 45.0,  // °C
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone5A),
        simulation_parameters: crate::validation::ashrae140::SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 20.0, // °C
            setpoint_cooling: 24.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 802: Heat pump (variable-speed, advanced control)
fn build_case_802() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case802,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 232.0,   // m²
            u_value: 0.40,       // W/m²K - better insulation
            thermal_mass: 130.0, // kJ/m²K
            window_wall_ratio: 0.25,
            infiltration_rate: 0.4, // ACH - tighter building
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::HeatPump,
            heating_capacity: 18000.0, // W - variable speed
            cooling_capacity: 15000.0, // W
            cop_heating: 3.8,
            cop_cooling: 4.0,
            stages: 0,               // Variable speed (continuous modulation)
            min_speed: 0.3,          // 30% minimum speed
            max_speed: 1.0,          // 100% maximum speed
            min_outdoor_temp: -25.0, // °C
            max_outdoor_temp: 50.0,  // °C
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone5A),
        simulation_parameters: crate::validation::ashrae140::SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 20.0, // °C
            setpoint_cooling: 24.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 803: Chiller plant (single chiller, basic control)
fn build_case_803() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case803,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 464.0,   // m² - larger building for chiller plant
            u_value: 0.35,       // W/m²K
            thermal_mass: 200.0, // kJ/m²K
            window_wall_ratio: 0.3,
            infiltration_rate: 0.6, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::Chiller,
            cooling_capacity: 50000.0, // W - single chiller
            chiller_type: Some("Centrifugal".to_string()),
            cop_cooling: 5.5,
            condenser_type: Some("AirCooled".to_string()),
            chiller_count: 1,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone4A),
        simulation_parameters: crate::validation::ashrae140::SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 21.0, // °C
            setpoint_cooling: 23.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 804: Chiller plant (multiple chillers, staging)
fn build_case_804() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case804,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 928.0,   // m² - large building for multiple chillers
            u_value: 0.32,       // W/m²K
            thermal_mass: 220.0, // kJ/m²K
            window_wall_ratio: 0.25,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::Chiller,
            cooling_capacity: 120000.0, // W - total capacity
            chiller_type: Some("Centrifugal".to_string()),
            cop_cooling: 5.8,
            condenser_type: Some("WaterCooled".to_string()),
            chiller_count: 3, // Three chillers
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 21.0, // °C
            setpoint_cooling: 23.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 805: Boiler plant (single boiler, basic control)
fn build_case_805() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case805,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 300.0,   // m²
            u_value: 0.40,       // W/m²K
            thermal_mass: 300.0, // kJ/m²K - high mass for boiler testing
            window_wall_ratio: 0.15,
            infiltration_rate: 0.4, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::Boiler,
            heating_capacity: 40000.0, // W - single boiler
            boiler_type: Some("Condensing".to_string()),
            efficiency: 0.95, // 95% efficiency
            boiler_count: 1,
            fuel_type: Some("natural_gas".to_string()),
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone6A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 22.0, // °C
            setpoint_cooling: 26.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 806: Boiler plant (multiple boilers, staging)
fn build_case_806() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case806,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 800.0,   // m² - large building for multiple boilers
            u_value: 0.38,       // W/m²K
            thermal_mass: 350.0, // kJ/m²K
            window_wall_ratio: 0.2,
            infiltration_rate: 0.45, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::Boiler,
            heating_capacity: 150000.0, // W - total capacity
            boiler_type: Some("Condensing".to_string()),
            efficiency: 0.93, // 93% efficiency
            boiler_count: 4,  // Four boilers
            fuel_type: Some("natural_gas".to_string()),
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone7),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 22.0, // °C
            setpoint_cooling: 26.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 807: Hybrid system (heat pump + boiler)
fn build_case_807() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case807,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 400.0,   // m²
            u_value: 0.36,       // W/m²K
            thermal_mass: 250.0, // kJ/m²K
            window_wall_ratio: 0.22,
            infiltration_rate: 0.48, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::Hybrid,
            heating_capacity: 30000.0, // W - total hybrid system
            cooling_capacity: 25000.0, // W
            cop_heating: 3.5,
            efficiency: 0.92,
            hybrid_switch_temp: -5.0, // °C - switch from heat pump to boiler
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone5A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 20.0, // °C
            setpoint_cooling: 24.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 808: VAV system with heat recovery
fn build_case_808() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case808,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 1200.0,  // m² - office building size
            u_value: 0.30,       // W/m²K
            thermal_mass: 220.0, // kJ/m²K
            window_wall_ratio: 0.35,
            infiltration_rate: 0.3, // ACH - tight building
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::VAV,
            heating_capacity: 80000.0,      // W
            cooling_capacity: 100000.0,     // W
            min_airflow_ratio: 0.3,         // 30% minimum airflow
            heat_recovery_efficiency: 0.75, // 75% heat recovery
            economizer_enabled: true,
            supply_air_temp: 13.0, // °C
            vav_enabled: true,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone4A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 21.0, // °C
            setpoint_cooling: 24.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 809: CAV system with economizer
fn build_case_809() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case809,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 800.0,   // m² - retail building size
            u_value: 0.38,       // W/m²K
            thermal_mass: 150.0, // kJ/m²K
            window_wall_ratio: 0.4,
            infiltration_rate: 0.4, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::CAV,
            heating_capacity: 60000.0, // W
            cooling_capacity: 75000.0, // W
            airflow_rate: 2.5,         // L/s/m²
            economizer_type: Some("differential_dry_bulb".to_string()),
            economizer_limit: 18.0, // °C
            supply_air_temp: 12.0,  // °C
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 20.0, // °C
            setpoint_cooling: 24.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 810: Comprehensive HVAC equipment
fn build_case_810() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case810,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 1500.0,  // m² - large commercial building
            u_value: 0.28,       // W/m²K - high performance envelope
            thermal_mass: 400.0, // kJ/m²K
            window_wall_ratio: 0.3,
            infiltration_rate: 0.25, // ACH - very tight building
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::Comprehensive,
            heating_capacity: 120000.0, // W
            cooling_capacity: 150000.0, // W
            cop_heating: 4.0,
            efficiency: 0.95,
            cop_cooling: 6.0,
            heat_recovery_efficiency: 0.8,
            economizer_enabled: true,
            vav_enabled: true,
            min_airflow_ratio: 0.2,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone4A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 21.0, // °C
            setpoint_cooling: 24.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

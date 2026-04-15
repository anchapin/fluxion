//! ASHRAE 140 Cases 195-470: Diagnostic Validation
//!
//! This module implements ASHRAE 140 Cases 195-470 which focus on
//! diagnostic validation including thermal mass variations, window-to-wall
//! ratio variations, and internal load variations.

#![allow(clippy::needless_update)]

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae140::AshraeZone;
use crate::validation::ashrae140::BuildingProperties;
use crate::validation::ashrae140::HVACSystem;
use crate::validation::ashrae140::SimulationParameters;
use crate::validation::ashrae140::WeatherData;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Build an ASHRAE 140 case definition for Cases 195-470
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    match case {
        // Solid conduction diagnostic variants (195 series)
        ASHRAE140Case::Case195 => build_case_195(),
        ASHRAE140Case::Case195HighMass => build_case_195_high_mass(),
        ASHRAE140Case::Case195NoLoads => build_case_195_no_loads(),
        ASHRAE140Case::Case195NoSolar => build_case_195_no_solar(),
        ASHRAE140Case::Case195ThermalBridge => build_case_195_thermal_bridge(),

        // Solar gain diagnostic variants (195 series)
        ASHRAE140Case::Case195SHGC03 => build_case_195_shgc03(),
        ASHRAE140Case::Case195SHGC06 => build_case_195_shgc06(),
        ASHRAE140Case::Case195SHGC09 => build_case_195_shgc09(),
        ASHRAE140Case::Case195Albedo01 => build_case_195_albedo01(),
        ASHRAE140Case::Case195Albedo05 => build_case_195_albedo05(),
        ASHRAE140Case::Case195Albedo09 => build_case_195_albedo09(),

        // Diagnostic cases (195-470 series)
        ASHRAE140Case::Case196 => build_case_196(),
        ASHRAE140Case::Case197 => build_case_197(),
        ASHRAE140Case::Case198 => build_case_198(),
        ASHRAE140Case::Case200 => build_case_200(),
        ASHRAE140Case::Case250 => build_case_250(),
        ASHRAE140Case::Case300 => build_case_300(),
        ASHRAE140Case::Case350 => build_case_350(),
        ASHRAE140Case::Case400 => build_case_400(),
        ASHRAE140Case::Case470 => build_case_470(),

        // Non-Residential Building Types
        ASHRAE140Case::Office => build_case_office(),
        ASHRAE140Case::Retail => build_case_retail(),
        ASHRAE140Case::School => build_case_school(),

        _ => panic!("Invalid case for series 195: {:?}", case),
    }
}

/// Case 195: Solid conduction - no windows, no infiltration, no loads
fn build_case_195() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-HM: High-mass walls
fn build_case_195_high_mass() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195HighMass,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-NL: No internal loads
fn build_case_195_no_loads() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195NoLoads,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-NS: No solar gains
fn build_case_195_no_solar() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195NoSolar,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-TB: Thermal bridge
fn build_case_195_thermal_bridge() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195ThermalBridge,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-SHGC0.3: Low SHGC variant
fn build_case_195_shgc03() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195SHGC03,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-SHGC0.6: Medium SHGC variant
fn build_case_195_shgc06() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195SHGC06,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-SHGC0.9: High SHGC variant
fn build_case_195_shgc09() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195SHGC09,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-ALB0.1: Low albedo variant
fn build_case_195_albedo01() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195Albedo01,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-ALB0.5: Medium albedo variant
fn build_case_195_albedo05() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195Albedo05,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 195-ALB0.9: High albedo variant
fn build_case_195_albedo09() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case195Albedo09,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        ..Default::default()
    }
}

/// Case 196: Lighting diagnostics
fn build_case_196() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case196,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 557.0,   // m²
            u_value: 0.38,       // W/m²K
            thermal_mass: 100.0, // kJ/m²K - lightweight
            window_wall_ratio: 0.3,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 12000.0, // W
            cooling_capacity: 10000.0, // W
            cop_heating: 3.0,
            cop_cooling: 3.2,
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

/// Case 197: Equipment diagnostics
fn build_case_197() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case197,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 557.0,   // m²
            u_value: 0.38,       // W/m²K
            thermal_mass: 100.0, // kJ/m²K
            window_wall_ratio: 0.3,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 12000.0, // W
            cooling_capacity: 10000.0, // W
            cop_heating: 3.0,
            cop_cooling: 3.2,
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

/// Case 198: Occupancy diagnostics
fn build_case_198() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case198,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 557.0,   // m²
            u_value: 0.38,       // W/m²K
            thermal_mass: 100.0, // kJ/m²K
            window_wall_ratio: 0.3,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 12000.0, // W
            cooling_capacity: 10000.0, // W
            cop_heating: 3.0,
            cop_cooling: 3.2,
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

/// Case 200: Combined internal loads
fn build_case_200() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case200,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 557.0,   // m²
            u_value: 0.38,       // W/m²K
            thermal_mass: 100.0, // kJ/m²K
            window_wall_ratio: 0.3,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 15000.0, // W - higher capacity for combined loads
            cooling_capacity: 12000.0, // W
            cop_heating: 3.2,
            cop_cooling: 3.4,
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

/// Case 250: Thermal mass diagnostics
fn build_case_250() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case250,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 557.0,   // m²
            u_value: 0.30,       // W/m²K - better insulation for high mass
            thermal_mass: 400.0, // kJ/m²K - high thermal mass
            window_wall_ratio: 0.2,
            infiltration_rate: 0.3, // ACH - tighter building
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 15000.0, // W
            cooling_capacity: 12000.0, // W
            cop_heating: 3.2,
            cop_cooling: 3.4,
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

/// Case 300: Night ventilation diagnostics
fn build_case_300() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case300,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 557.0,      // m²
            u_value: 0.38,          // W/m²K
            thermal_mass: 150.0,    // kJ/m²K - medium mass for night ventilation
            window_wall_ratio: 0.4, // Higher for night ventilation
            infiltration_rate: 0.8, // ACH - higher for ventilation
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 12000.0, // W
            cooling_capacity: 10000.0, // W
            cop_heating: 3.0,
            cop_cooling: 3.2,

            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 18.0, // °C - lower for night ventilation
            setpoint_cooling: 26.0, // °C - higher for night ventilation
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 350: Setback diagnostics
fn build_case_350() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case350,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0,   // m²
            u_value: 0.35,       // W/m²K
            thermal_mass: 250.0, // kJ/m²K - medium mass for setback
            window_wall_ratio: 0.3,
            infiltration_rate: 0.4, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 15000.0, // W
            cooling_capacity: 12000.0, // W
            cop_heating: 3.2,
            cop_cooling: 3.4,

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

/// Case 400: Free-floating diagnostics
fn build_case_400() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case400,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            floor_area: 557.0,   // m²
            u_value: 0.38,       // W/m²K
            thermal_mass: 100.0, // kJ/m²K
            window_wall_ratio: 0.3,
            infiltration_rate: 0.5, // ACH
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::None,
            heating_capacity: 0.0, // W - no HVAC
            cooling_capacity: 0.0, // W
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone3A),
        simulation_parameters: SimulationParameters {
            timestep: 3600, // 1 hour
            total_hours: 8760,
            setpoint_heating: 0.0, // °C - no setpoint
            setpoint_cooling: 0.0, // °C
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Case 470: Comprehensive diagnostics
fn build_case_470() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Case470,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 557.0,   // m²
            u_value: 0.28,       // W/m²K - high performance
            thermal_mass: 400.0, // kJ/m²K - high mass
            window_wall_ratio: 0.3,
            infiltration_rate: 0.3, // ACH - tight building
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::Comprehensive,
            heating_capacity: 20000.0, // W
            cooling_capacity: 18000.0, // W
            cop_heating: 4.0,
            cop_cooling: 4.2,
            heat_recovery_efficiency: 0.8,
            economizer_enabled: true,

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

/// Office building
fn build_case_office() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Office,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::VAV,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone4A),
        ..Default::default()
    }
}

/// Retail building
fn build_case_retail() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::Retail,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::MediumWeight,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::CAV,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone4A),
        ..Default::default()
    }
}

/// School building
fn build_case_school() -> ASHRAE140CaseDefinition {
    ASHRAE140CaseDefinition {
        case_type: ASHRAE140Case::School,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 557.0, // m²
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::VAV,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone4A),
        ..Default::default()
    }
}

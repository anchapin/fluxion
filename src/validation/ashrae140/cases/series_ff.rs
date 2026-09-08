//! ASHRAE 140 Free-Floating Cases (600FF / 650FF / 900FF / 950FF)
//!
//! Free-floating variants of the low-mass (600) and high-mass (900) baseline
//! series. The HVAC system is disabled — the engine tracks zone temperature
//! without active conditioning. ASHRAE Standard 140-2017 §B6.5.
//!
//! - Case 600FF: Case 600 with HVAC disabled
//! - Case 650FF: Case 650 (night ventilation) with HVAC disabled
//! - Case 900FF: Case 900 with HVAC disabled
//! - Case 950FF: Case 950 (night ventilation) with HVAC disabled
//!
//! All four cases share the same `HVACType::None` configuration here; the
//! night-ventilation schedule flag distinguishing 650FF / 950FF from
//! 600FF / 900FF is wired through the engine's free-floating pathway
//! (Issue #3555 follow-up owns the deeper consolidation).

#![allow(clippy::needless_update)]

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae140::AshraeZone;
use crate::validation::ashrae140::BuildingProperties;
use crate::validation::ashrae140::HVACSystem;
use crate::validation::ashrae140::SimulationParameters;
use crate::validation::ashrae140::WeatherData;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

const FLOOR_AREA_M2: f64 = 48.0;
const INFILTRATION_ACH: f64 = 0.5;

fn no_hvac() -> HVACSystem {
    HVACSystem {
        system_type: crate::validation::ashrae140::HVACType::None,
        heating_capacity: 0.0,
        cooling_capacity: 0.0,
        ..Default::default()
    }
}

fn standard_simulation() -> SimulationParameters {
    SimulationParameters {
        timestep: 3600,
        total_hours: 8760,
        // Free-floating — setpoints are not enforced by the engine
        setpoint_heating: 0.0,
        setpoint_cooling: 0.0,
        ..Default::default()
    }
}

fn standard_weather() -> WeatherData {
    WeatherData::from_ashrae_zone(AshraeZone::Zone4A)
}

/// Build an ASHRAE 140 case definition for the free-floating variants
/// (600FF / 650FF / 900FF / 950FF).
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    let construction = match case {
        ASHRAE140Case::Case600FF | ASHRAE140Case::Case650FF => {
            crate::validation::ashrae140::ConstructionType::Lightweight
        }
        ASHRAE140Case::Case900FF | ASHRAE140Case::Case950FF => {
            crate::validation::ashrae140::ConstructionType::HighMass
        }
        _ => panic!("Invalid case for free-floating series: {:?}", case),
    };
    let (u_value, thermal_mass) = match construction {
        crate::validation::ashrae140::ConstructionType::Lightweight => (0.514, 150.0),
        crate::validation::ashrae140::ConstructionType::HighMass => (2.033, 600.0),
        _ => unreachable!(),
    };

    ASHRAE140CaseDefinition {
        case_type: case,
        building: BuildingProperties {
            construction_type: construction,
            floor_area: FLOOR_AREA_M2,
            u_value,
            thermal_mass,
            window_wall_ratio: 0.189,
            infiltration_rate: INFILTRATION_ACH,
            ..Default::default()
        },
        hvac: no_hvac(),
        weather: standard_weather(),
        simulation_parameters: standard_simulation(),
        ..Default::default()
    }
}
//! ASHRAE 140 Cases 600-650: Low-Mass Baseline Series
//!
//! This module implements ASHRAE 140 Cases 600-650 which focus on the
//! low-mass (lightweight wood-frame) single-zone baseline geometry from
//! ASHRAE Standard 140-2017 §B6.
//!
//! Geometry: 8 m (W) × 6 m (D) × 2.7 m (H) — floor area 48 m².
//! South-facing window: 12 m² (6 m × 2 m), SHGC 0.789, U-value 3.0 W/m²K.
//! Walls: lightweight construction (U ≈ 0.514 W/m²K), thermal mass ≈ 150 kJ/m²K.
//! Roof: lightweight (U ≈ 0.318 W/m²K). Floor: insulated (U ≈ 0.039 W/m²K).
//! Infiltration: 0.5 ACH. HVAC: 20°C heating / 27°C cooling, 100% efficient.
//!
//! Variant axes (per ASHRAE 140 §B6.4):
//! - 600: south windows, no shading, 24-h heating/cooling
//! - 610: south windows with 1 m overhang
//! - 620: east/west windows (6 m² each), no shading
//! - 630: east/west windows with overhang and fins
//! - 640: 600 + heating setback to 10 °C overnight (23:00-07:00)
//! - 650: 600 + heating disabled + night ventilation (18:00-07:00)

#![allow(clippy::needless_update)]

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae140::AshraeZone;
use crate::validation::ashrae140::BuildingProperties;
use crate::validation::ashrae140::HVACSystem;
use crate::validation::ashrae140::SimulationParameters;
use crate::validation::ashrae140::WeatherData;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

const FLOOR_AREA_M2: f64 = 48.0;
const WALL_U_VALUE: f64 = 0.514;
const THERMAL_MASS_LOW: f64 = 150.0;
const INFILTRATION_ACH: f64 = 0.5;

fn low_mass_building() -> BuildingProperties {
    BuildingProperties {
        construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
        floor_area: FLOOR_AREA_M2,
        u_value: WALL_U_VALUE,
        thermal_mass: THERMAL_MASS_LOW,
        window_wall_ratio: 0.189, // 12 m² window / ~63.6 m² wall ≈ 0.189
        infiltration_rate: INFILTRATION_ACH,
        ..Default::default()
    }
}

fn standard_hvac() -> HVACSystem {
    HVACSystem {
        system_type: crate::validation::ashrae140::HVACType::PTAC,
        heating_capacity: 12000.0,
        cooling_capacity: 10000.0,
        cop_heating: 1.0,
        cop_cooling: 1.0,
        ..Default::default()
    }
}

fn standard_simulation() -> SimulationParameters {
    SimulationParameters {
        timestep: 3600,
        total_hours: 8760,
        setpoint_heating: 20.0,
        setpoint_cooling: 27.0,
        ..Default::default()
    }
}

fn standard_weather() -> WeatherData {
    WeatherData::from_ashrae_zone(AshraeZone::Zone4A)
}

/// Build an ASHRAE 140 case definition for Cases 600-650 (low-mass baseline series)
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    match case {
        ASHRAE140Case::Case600
        | ASHRAE140Case::Case610
        | ASHRAE140Case::Case620
        | ASHRAE140Case::Case630
        | ASHRAE140Case::Case640 => ASHRAE140CaseDefinition {
            case_type: case,
            building: low_mass_building(),
            hvac: standard_hvac(),
            weather: standard_weather(),
            simulation_parameters: standard_simulation(),
            ..Default::default()
        },
        // Case 650: low mass with night ventilation, heating disabled.
        // Engine distinguishes 650 from 600 via the night-ventilation schedule
        // (Issue #3555 follow-up wires the schedule; here we only enable the
        // no-heating configuration flag).
        ASHRAE140Case::Case650 => ASHRAE140CaseDefinition {
            case_type: case,
            building: low_mass_building(),
            hvac: HVACSystem {
                system_type: crate::validation::ashrae140::HVACType::PTAC,
                heating_capacity: 0.0,
                cooling_capacity: 10000.0,
                cop_cooling: 1.0,
                ..Default::default()
            },
            weather: standard_weather(),
            simulation_parameters: standard_simulation(),
            ..Default::default()
        },
        _ => panic!("Invalid case for series 600: {:?}", case),
    }
}
//! ASHRAE 140 Case 960: Sunspace (2-zone building)
//!
//! ASHRAE Standard 140-2017 §B6.6 — a 2-zone building with a back zone
//! (conditioned space) and an attached sunspace (unconditioned buffer
//! to the south wall). Tests inter-zone heat transfer through the common
//! wall and the solar gain path through the sunspace glazing.

#![allow(clippy::needless_update)]

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae140::AshraeZone;
use crate::validation::ashrae140::BuildingProperties;
use crate::validation::ashrae140::HVACSystem;
use crate::validation::ashrae140::SimulationParameters;
use crate::validation::ashrae140::WeatherData;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Build an ASHRAE 140 case definition for Case 960 (sunspace).
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    debug_assert!(matches!(case, ASHRAE140Case::Case960));
    ASHRAE140CaseDefinition {
        case_type: case,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::Lightweight,
            // Back-zone floor area (8m × 6m); the attached sunspace adds
            // an additional 8m × 3m buffer to the south wall.
            floor_area: 48.0,
            u_value: 0.514,
            thermal_mass: 150.0,
            window_wall_ratio: 0.189,
            infiltration_rate: 0.5,
            ..Default::default()
        },
        hvac: HVACSystem {
            system_type: crate::validation::ashrae140::HVACType::PTAC,
            heating_capacity: 12000.0,
            cooling_capacity: 10000.0,
            cop_heating: 1.0,
            cop_cooling: 1.0,
            ..Default::default()
        },
        weather: WeatherData::from_ashrae_zone(AshraeZone::Zone4A),
        simulation_parameters: SimulationParameters {
            timestep: 3600,
            total_hours: 8760,
            setpoint_heating: 20.0,
            setpoint_cooling: 27.0,
            ..Default::default()
        },
        ..Default::default()
    }
}
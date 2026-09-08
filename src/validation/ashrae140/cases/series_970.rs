//! ASHRAE 140 Case 970: 5-Zone Cross-Coupling
//!
//! ASHRAE Standard 140-2017 §B6.7 — 8 m × 6 m × 2.7 m high-mass concrete
//! building divided into 5 zones by interior partitions. The canonical
//! reference (tests/reference_data/zone_balance/case_970_energy_reference.csv)
//! is the ASHRAE 140-2023 Annex B8-3 inter-program envelope: annual
//! heating 10.54–14.26 MWh, annual cooling 7.39–10.00 MWh. Exercises
//! `sim::multi_zone_network::MultiZoneAirflowNetwork` on a 5×5 symmetric
//! conductance matrix (`tests/multi_zone_n_zone_network.rs`).

#![allow(clippy::needless_update)]

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae140::AshraeZone;
use crate::validation::ashrae140::BuildingProperties;
use crate::validation::ashrae140::HVACSystem;
use crate::validation::ashrae140::SimulationParameters;
use crate::validation::ashrae140::WeatherData;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Build an ASHRAE 140 case definition for Case 970 (5-zone cross-coupling).
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    debug_assert!(matches!(case, ASHRAE140Case::Case970));
    ASHRAE140CaseDefinition {
        case_type: case,
        building: BuildingProperties {
            construction_type: crate::validation::ashrae140::ConstructionType::HighMass,
            floor_area: 48.0,
            u_value: 2.033,
            thermal_mass: 600.0,
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
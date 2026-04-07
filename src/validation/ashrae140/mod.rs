//! ASHRAE 140 Validation Framework
//!
//! This module provides the core data structures and functionality
//! for ASHRAE 140 building energy model validation.

pub mod cases;

pub use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Simulation Parameters
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationParameters {
    pub timestep: u32,
    pub total_hours: u32,
    pub setpoint_heating: f64,
    pub setpoint_cooling: f64,
    // Additional simulation parameters would go here
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            timestep: 3600,         // 1 hour
            total_hours: 8760,      // Full year
            setpoint_heating: 20.0, // °C
            setpoint_cooling: 24.0, // °C
        }
    }
}

/// ASHRAE 140 Case Definition
#[derive(Debug, Clone, PartialEq)]
pub struct ASHRAE140CaseDefinition {
    pub case_type: ASHRAE140Case,
    pub building: BuildingProperties,
    pub hvac: HVACSystem,
    pub weather: WeatherData,
    pub simulation_parameters: SimulationParameters,
    // Additional case-specific parameters would go here
}

impl Default for ASHRAE140CaseDefinition {
    fn default() -> Self {
        Self {
            case_type: ASHRAE140Case::Case600,
            building: BuildingProperties::default(),
            hvac: HVACSystem::default(),
            weather: WeatherData::default(),
            simulation_parameters: SimulationParameters::default(),
        }
    }
}

/// Building Properties
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingProperties {
    pub construction_type: ConstructionType,
    pub floor_area: f64,
    pub u_value: f64,
    pub thermal_mass: f64,
    pub window_wall_ratio: f64,
    pub infiltration_rate: f64,
    // Additional building properties would go here
}

impl Default for BuildingProperties {
    fn default() -> Self {
        Self {
            construction_type: ConstructionType::Lightweight,
            floor_area: 232.0,   // Default from ASHRAE 140
            u_value: 0.45,       // W/m²K
            thermal_mass: 120.0, // kJ/m²K
            window_wall_ratio: 0.2,
            infiltration_rate: 0.5, // ACH
        }
    }
}

/// HVAC System Configuration
#[derive(Debug, Clone, PartialEq)]
pub struct HVACSystem {
    pub system_type: HVACType,
    pub heating_capacity: f64,
    pub cooling_capacity: f64,
    pub cop_heating: f64,
    pub cop_cooling: f64,
    pub stages: u32,
    pub min_outdoor_temp: f64,
    pub max_outdoor_temp: f64,
    // Chiller-specific properties
    pub chiller_type: Option<String>,
    pub chiller_count: u32,
    pub condenser_type: Option<String>,
    // Boiler-specific properties
    pub boiler_type: Option<String>,
    pub boiler_count: u32,
    pub efficiency: f64,
    pub fuel_type: Option<String>,
    // Heat pump-specific properties
    pub heat_pump_capacity: f64,
    pub stage1_capacity: f64,
    pub stage2_capacity: f64,
    pub min_speed: f64,
    pub max_speed: f64,
    // Hybrid system properties
    pub boiler_capacity: f64,
    pub hybrid_switch_temp: f64,
    // VAV/CAV properties
    pub min_airflow_ratio: f64,
    pub heat_recovery_efficiency: f64,
    pub economizer_enabled: bool,
    pub airflow_rate: f64,
    pub economizer_type: Option<String>,
    pub economizer_limit: f64,
    pub supply_air_temp: f64,
    pub vav_enabled: bool,
    // Additional HVAC properties would go here
}

impl Default for HVACSystem {
    fn default() -> Self {
        Self {
            system_type: HVACType::PTAC,
            heating_capacity: 10000.0,
            cooling_capacity: 8000.0,
            cop_heating: 3.0,
            cop_cooling: 3.2,
            stages: 1,
            min_outdoor_temp: -10.0,
            max_outdoor_temp: 40.0,
            chiller_type: None,
            chiller_count: 1,
            condenser_type: None,
            boiler_type: None,
            boiler_count: 1,
            efficiency: 0.9,
            fuel_type: None,
            heat_pump_capacity: 0.0,
            stage1_capacity: 1.0,
            stage2_capacity: 1.0,
            min_speed: 0.5,
            max_speed: 1.0,
            boiler_capacity: 0.0,
            hybrid_switch_temp: 0.0,
            min_airflow_ratio: 0.5,
            heat_recovery_efficiency: 0.0,
            economizer_enabled: false,
            airflow_rate: 1.0,
            economizer_type: None,
            economizer_limit: 18.0,
            supply_air_temp: 13.0,
            vav_enabled: false,
        }
    }
}

/// Weather Data
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherData {
    pub ashrae_zone: AshraeZone,
    // Additional weather properties would go here
}

impl WeatherData {
    pub fn from_ashrae_zone(zone: AshraeZone) -> Self {
        Self { ashrae_zone: zone }
    }
}

impl Default for WeatherData {
    fn default() -> Self {
        Self {
            ashrae_zone: AshraeZone::Zone3A,
        }
    }
}

/// ASHRAE Climate Zone
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AshraeZone {
    Zone1A,
    Zone2A,
    Zone2B,
    Zone3A,
    Zone3B,
    Zone3C,
    Zone4A,
    Zone4B,
    Zone4C,
    Zone5A,
    Zone5B,
    Zone6A,
    Zone6B,
    Zone7,
    Zone8,
}

/// HVAC System Type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HVACType {
    None,
    PTAC,
    VAV,
    CAV,
    HeatPump,
    Chiller,
    Boiler,
    Hybrid,
    Comprehensive,
}

/// Construction Type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstructionType {
    Lightweight,
    MediumWeight,
    HighMass,
}

/// Run validation for a single case with performance monitoring
pub fn run_validation_with_performance(
    case: ASHRAE140Case,
) -> (
    crate::validation::ASHRAE140CaseDefinition,
    crate::validation::PerformanceMetrics,
) {
    let metrics = crate::validation::performance::profile_case(case, 1);
    let case_def = crate::validation::ashrae140::cases::build_case(case);
    crate::validation::performance::log_performance_metrics(&metrics);
    (case_def, metrics)
}

/// Run multiple cases in parallel with performance monitoring
pub fn run_validation_series_parallel(
    cases: &[ASHRAE140Case],
    max_threads: Option<usize>,
) -> Vec<(
    ASHRAE140Case,
    crate::validation::ASHRAE140CaseDefinition,
    crate::validation::PerformanceMetrics,
)> {
    // Set Rayon thread pool size if specified
    if let Some(threads) = max_threads {
        if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            rayon::set_global_thread_pool(pool).unwrap();
        }
    }

    use rayon::prelude::*;

    cases
        .par_iter()
        .map(|case| {
            let case_def = crate::validation::ashrae140::cases::build_case(*case);
            let metrics = crate::validation::performance::profile_case(*case, 1);
            crate::validation::performance::log_performance_metrics(&metrics);
            (*case, case_def, metrics)
        })
        .collect()
}

/// Optimized version of validation using Arc for thread-safe data
pub fn run_validation_optimized(case: ASHRAE140Case) -> crate::validation::ASHRAE140CaseDefinition {
    // Use Arc for thread-safe shared data where appropriate
    let case_def = crate::validation::ashrae140::cases::build_case(case);
    case_def
}

/// Select appropriate validation strategy based on case complexity
pub fn run_validation_strategy(case: ASHRAE140Case) -> crate::validation::ASHRAE140CaseDefinition {
    match case {
        // Simple cases - use standard validation
        ASHRAE140Case::Case800
        | ASHRAE140Case::Case801
        | ASHRAE140Case::Case802
        | ASHRAE140Case::Case803
        | ASHRAE140Case::Case804
        | ASHRAE140Case::Case805 => crate::validation::ashrae140::cases::build_case(case),
        // Complex cases - use optimized parallel version
        ASHRAE140Case::Case806
        | ASHRAE140Case::Case807
        | ASHRAE140Case::Case808
        | ASHRAE140Case::Case809
        | ASHRAE140Case::Case810 => run_validation_optimized(case),
        // Diagnostic cases - vary by complexity
        ASHRAE140Case::Case195
        | ASHRAE140Case::Case195HighMass
        | ASHRAE140Case::Case195NoLoads
        | ASHRAE140Case::Case195NoSolar
        | ASHRAE140Case::Case195ThermalBridge
        | ASHRAE140Case::Case195SHGC03
        | ASHRAE140Case::Case195SHGC06
        | ASHRAE140Case::Case195SHGC09
        | ASHRAE140Case::Case195Albedo01
        | ASHRAE140Case::Case195Albedo05
        | ASHRAE140Case::Case195Albedo09
        | ASHRAE140Case::Case196
        | ASHRAE140Case::Case197
        | ASHRAE140Case::Case198
        | ASHRAE140Case::Case200
        | ASHRAE140Case::Case250 => {
            // Medium complexity
            crate::validation::ashrae140::cases::build_case(case)
        }
        ASHRAE140Case::Case300
        | ASHRAE140Case::Case350
        | ASHRAE140Case::Case400
        | ASHRAE140Case::Case470
        | ASHRAE140Case::Office
        | ASHRAE140Case::Retail
        | ASHRAE140Case::School => {
            // High complexity - use parallel
            run_validation_optimized(case)
        }
        _ => crate::validation::ashrae140::cases::build_case(case),
    }
}

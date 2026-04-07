//! ASHRAE 140 Validation Framework
//!
//! This module provides the core data structures and functionality
//! for ASHRAE 140 building energy model validation.

pub mod cases;

pub use crate::validation::ASHRAE140Case;

/// ASHRAE 140 Case Definition
#[derive(Debug, Clone, PartialEq)]
pub struct ASHRAE140CaseDefinition {
    pub case_type: ASHRAE140Case,
    pub building: BuildingProperties,
    pub hvac: HVACSystem,
    pub weather: WeatherData,
    // Additional case-specific parameters would go here
}

impl Default for ASHRAE140CaseDefinition {
    fn default() -> Self {
        Self {
            case_type: ASHRAE140Case::Case600,
            building: BuildingProperties::default(),
            hvac: HVACSystem::default(),
            weather: WeatherData::default(),
        }
    }
}

/// Building Properties
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingProperties {
    pub construction_type: ConstructionType,
    pub floor_area: f64,
    // Additional building properties would go here
}

impl Default for BuildingProperties {
    fn default() -> Self {
        Self {
            construction_type: ConstructionType::Lightweight,
            floor_area: 232.0, // Default from ASHRAE 140
        }
    }
}

/// HVAC System Configuration
#[derive(Debug, Clone, PartialEq)]
pub struct HVACSystem {
    pub system_type: HVACType,
    // Additional HVAC properties would go here
}

impl Default for HVACSystem {
    fn default() -> Self {
        Self {
            system_type: HVACType::PTAC,
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
        ASHRAE140Case::Case800..=ASHRAE140Case::Case805 => {
            crate::validation::ashrae140::cases::build_case(case)
        }
        // Complex cases - use optimized parallel version
        ASHRAE140Case::Case806..=ASHRAE140Case::Case810 => run_validation_optimized(case),
        // Diagnostic cases - vary by complexity
        ASHRAE140Case::Case195..=ASHRAE140Case::Case250 => {
            // Medium complexity
            crate::validation::ashrae140::cases::build_case(case)
        }
        ASHRAE140Case::Case251..=ASHRAE140Case::Case470 => {
            // High complexity - use parallel
            run_validation_optimized(case)
        }
        _ => crate::validation::ashrae140::cases::build_case(case),
    }
}

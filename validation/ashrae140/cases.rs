// validation/ashrae140/cases.rs
/// ASHRAE 140 case utilities for validation module
///
/// This module provides utility functions for working with ASHRAE 140 cases
/// and connects to the comprehensive case definitions in src/validation/ashrae_140_cases.rs
use crate::validation::ashrae_140_cases::ASHRAE140Case;

/// Get expanded cases (500-699 series) only
pub fn get_expanded_cases() -> Vec<ASHRAE140Case> {
    get_cases_by_series("500-699")
}

/// Check if a case is part of the expanded validation coverage
pub fn is_expanded_case(case: &ASHRAE140Case) -> bool {
    matches!(
        case,
        ASHRAE140Case::Case500
            | ASHRAE140Case::Case501
            | ASHRAE140Case::Case502
            | ASHRAE140Case::Case503
            | ASHRAE140Case::Case504
            | ASHRAE140Case::Case505
            | ASHRAE140Case::Case506
            | ASHRAE140Case::Case507
            | ASHRAE140Case::Case508
            | ASHRAE140Case::Case509
            | ASHRAE140Case::Case510
            | ASHRAE140Case::Case699
    )
}

impl ASHRAE140Case {
    /// Get case ID as string
    pub fn case_id(&self) -> String {
        match self {
            ASHRAE140Case::Case600 => "600".to_string(),
            ASHRAE140Case::Case610 => "610".to_string(),
            ASHRAE140Case::Case620 => "620".to_string(),
            ASHRAE140Case::Case630 => "630".to_string(),
            ASHRAE140Case::Case640 => "640".to_string(),
            ASHRAE140Case::Case650 => "650".to_string(),
            ASHRAE140Case::Case600FF => "600FF".to_string(),
            ASHRAE140Case::Case650FF => "650FF".to_string(),
            ASHRAE140Case::Case900 => "900".to_string(),
            ASHRAE140Case::Case910 => "910".to_string(),
            ASHRAE140Case::Case920 => "920".to_string(),
            ASHRAE140Case::Case930 => "930".to_string(),
            ASHRAE140Case::Case940 => "940".to_string(),
            ASHRAE140Case::Case950 => "950".to_string(),
            ASHRAE140Case::Case900FF => "900FF".to_string(),
            ASHRAE140Case::Case950FF => "950FF".to_string(),
            ASHRAE140Case::Case960 => "960".to_string(),
            ASHRAE140Case::Case195 => "195".to_string(),
            ASHRAE140Case::Case500 => "500".to_string(),
            ASHRAE140Case::Case501 => "501".to_string(),
            ASHRAE140Case::Case502 => "502".to_string(),
            ASHRAE140Case::Case503 => "503".to_string(),
            ASHRAE140Case::Case504 => "504".to_string(),
            ASHRAE140Case::Case505 => "505".to_string(),
            ASHRAE140Case::Case506 => "506".to_string(),
            ASHRAE140Case::Case507 => "507".to_string(),
            ASHRAE140Case::Case508 => "508".to_string(),
            ASHRAE140Case::Case509 => "509".to_string(),
            ASHRAE140Case::Case510 => "510".to_string(),
            ASHRAE140Case::Case699 => "699".to_string(),
        }
    }

    /// Get case description
    pub fn description(&self) -> String {
        match self {
            ASHRAE140Case::Case600 => "Low mass baseline".to_string(),
            ASHRAE140Case::Case610 => "Low mass with south shading".to_string(),
            ASHRAE140Case::Case620 => "Low mass with east/west windows".to_string(),
            ASHRAE140Case::Case630 => "Low mass with east/west shading".to_string(),
            ASHRAE140Case::Case640 => "Low mass with thermostat setback".to_string(),
            ASHRAE140Case::Case650 => "Low mass with night ventilation".to_string(),
            ASHRAE140Case::Case600FF => "Low mass free-floating".to_string(),
            ASHRAE140Case::Case650FF => "Low mass free-floating with night ventilation".to_string(),
            ASHRAE140Case::Case900 => "High mass baseline".to_string(),
            ASHRAE140Case::Case910 => "High mass with south shading".to_string(),
            ASHRAE140Case::Case920 => "High mass with east/west windows".to_string(),
            ASHRAE140Case::Case930 => "High mass with east/west shading".to_string(),
            ASHRAE140Case::Case940 => "High mass with thermostat setback".to_string(),
            ASHRAE140Case::Case950 => "High mass with night ventilation".to_string(),
            ASHRAE140Case::Case900FF => "High mass free-floating".to_string(),
            ASHRAE140Case::Case950FF => {
                "High mass free-floating with night ventilation".to_string()
            }
            ASHRAE140Case::Case960 => "Sunspace (2-zone building)".to_string(),
            ASHRAE140Case::Case195 => "Solid conduction".to_string(),
            ASHRAE140Case::Case500 => "Low mass baseline with alternative construction".to_string(),
            ASHRAE140Case::Case501 => "Low mass with north windows".to_string(),
            ASHRAE140Case::Case502 => "Low mass with double glazing".to_string(),
            ASHRAE140Case::Case503 => "Low mass with triple glazing".to_string(),
            ASHRAE140Case::Case504 => "Low mass with reduced infiltration".to_string(),
            ASHRAE140Case::Case505 => "Low mass with increased infiltration".to_string(),
            ASHRAE140Case::Case506 => "Low mass with alternative roof construction".to_string(),
            ASHRAE140Case::Case507 => "Low mass with alternative floor construction".to_string(),
            ASHRAE140Case::Case508 => "Low mass with reduced window area".to_string(),
            ASHRAE140Case::Case509 => "Low mass with increased window area".to_string(),
            ASHRAE140Case::Case510 => "Low mass with alternative orientation".to_string(),
            ASHRAE140Case::Case699 => "Low mass with comprehensive HVAC integration".to_string(),
        }
    }

    /// Check if case is free-floating
    pub fn is_free_floating(&self) -> bool {
        matches!(
            self,
            ASHRAE140Case::Case600FF
                | ASHRAE140Case::Case650FF
                | ASHRAE140Case::Case900FF
                | ASHRAE140Case::Case950FF
        )
    }
}

/// Get all ASHRAE 140 cases
pub fn get_all_cases() -> Vec<ASHRAE140Case> {
    vec![
        // Low mass cases (600 series)
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case610,
        ASHRAE140Case::Case620,
        ASHRAE140Case::Case630,
        ASHRAE140Case::Case640,
        ASHRAE140Case::Case650,
        ASHRAE140Case::Case600FF,
        ASHRAE140Case::Case650FF,
        // High mass cases (900 series)
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case910,
        ASHRAE140Case::Case920,
        ASHRAE140Case::Case930,
        ASHRAE140Case::Case940,
        ASHRAE140Case::Case950,
        ASHRAE140Case::Case900FF,
        ASHRAE140Case::Case950FF,
        // Special cases
        ASHRAE140Case::Case960,
        ASHRAE140Case::Case195,
        // Expanded cases (500-699 series)
        ASHRAE140Case::Case500,
        ASHRAE140Case::Case501,
        ASHRAE140Case::Case502,
        ASHRAE140Case::Case503,
        ASHRAE140Case::Case504,
        ASHRAE140Case::Case505,
        ASHRAE140Case::Case506,
        ASHRAE140Case::Case507,
        ASHRAE140Case::Case508,
        ASHRAE140Case::Case509,
        ASHRAE140Case::Case510,
        ASHRAE140Case::Case699,
    ]
}

/// Get cases by series
pub fn get_cases_by_series(series: &str) -> Vec<ASHRAE140Case> {
    match series {
        "600" => vec![
            ASHRAE140Case::Case600,
            ASHRAE140Case::Case610,
            ASHRAE140Case::Case620,
            ASHRAE140Case::Case630,
            ASHRAE140Case::Case640,
            ASHRAE140Case::Case650,
            ASHRAE140Case::Case600FF,
            ASHRAE140Case::Case650FF,
        ],
        "900" => vec![
            ASHRAE140Case::Case900,
            ASHRAE140Case::Case910,
            ASHRAE140Case::Case920,
            ASHRAE140Case::Case930,
            ASHRAE140Case::Case940,
            ASHRAE140Case::Case950,
            ASHRAE140Case::Case900FF,
            ASHRAE140Case::Case950FF,
        ],
        "500-699" => vec![
            ASHRAE140Case::Case500,
            ASHRAE140Case::Case501,
            ASHRAE140Case::Case502,
            ASHRAE140Case::Case503,
            ASHRAE140Case::Case504,
            ASHRAE140Case::Case505,
            ASHRAE140Case::Case506,
            ASHRAE140Case::Case507,
            ASHRAE140Case::Case508,
            ASHRAE140Case::Case509,
            ASHRAE140Case::Case510,
            ASHRAE140Case::Case699,
        ],
        _ => vec![],
    }
}

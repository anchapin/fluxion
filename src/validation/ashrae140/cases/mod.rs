//! ASHRAE 140 Case Implementation Modules
//!
//! This module provides the implementation for ASHRAE 140 test cases
//! organized by series for better maintainability.

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

pub mod series_195;
pub mod series_600;
pub mod series_800;
pub mod series_900;
pub mod series_960;
pub mod series_970;
pub mod series_ff;

/// Build an ASHRAE 140 case definition based on the case enum variant
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    match case {
        ASHRAE140Case::Case800
        | ASHRAE140Case::Case801
        | ASHRAE140Case::Case802
        | ASHRAE140Case::Case803
        | ASHRAE140Case::Case804
        | ASHRAE140Case::Case805
        | ASHRAE140Case::Case806
        | ASHRAE140Case::Case807
        | ASHRAE140Case::Case808
        | ASHRAE140Case::Case809
        | ASHRAE140Case::Case810 => series_800::build_case(case),
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
        | ASHRAE140Case::Case250
        | ASHRAE140Case::Case300
        | ASHRAE140Case::Case350
        | ASHRAE140Case::Case400
        | ASHRAE140Case::Case470
        | ASHRAE140Case::Office
        | ASHRAE140Case::Retail
        | ASHRAE140Case::School => series_195::build_case(case),
        // Low-mass baseline series (600-650)
        ASHRAE140Case::Case600
        | ASHRAE140Case::Case610
        | ASHRAE140Case::Case620
        | ASHRAE140Case::Case630
        | ASHRAE140Case::Case640
        | ASHRAE140Case::Case650 => series_600::build_case(case),
        // High-mass baseline series (900-950)
        ASHRAE140Case::Case900
        | ASHRAE140Case::Case910
        | ASHRAE140Case::Case920
        | ASHRAE140Case::Case930
        | ASHRAE140Case::Case940
        | ASHRAE140Case::Case950 => series_900::build_case(case),
        // Free-floating variants
        ASHRAE140Case::Case600FF
        | ASHRAE140Case::Case650FF
        | ASHRAE140Case::Case900FF
        | ASHRAE140Case::Case950FF => series_ff::build_case(case),
        // Special cases
        ASHRAE140Case::Case960 => series_960::build_case(case),
        ASHRAE140Case::Case970 => series_970::build_case(case),
        // Add other case ranges as needed
        _ => panic!("Case {} not implemented in this module", case.number()),
    }
}
//! ASHRAE 140 Case Implementation Modules
//!
//! This module provides the implementation for ASHRAE 140 test cases
//! organized by series for better maintainability.
//!
//! Issue #3555: `series_600`, `series_900`, `series_960`, and
//! `series_970` absorb the case-600/900/950FF/960/970 `CaseSpec` factories
//! that previously lived inline in `src/validation/ashrae_140_cases.rs`.

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

pub mod series_195;
pub mod series_600;
pub mod series_800;
pub mod series_900;
pub mod series_960;
pub mod series_970;

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
        // Add other case ranges as needed
        _ => panic!("Case {} not implemented in this module", case.number()),
    }
}

/// Build the full `CaseSpec` for any of the cases whose definitions now
/// live in the dedicated per-series modules. Used by
/// `ASHRAE140Case::spec()` in `src/validation/ashrae_140_cases.rs` after
/// Issue #3555 extracted those factories out of `CaseBuilder`.
pub fn build_spec(case: ASHRAE140Case) -> crate::validation::ashrae_140_cases::CaseSpec {
    use crate::validation::ashrae_140_cases::CaseBuilder;
    match case {
        // Issue #3555 burn-down: 600-series + 600FF/650FF extracted to series_600.rs.
        ASHRAE140Case::Case600 => series_600::case_600_baseline(),
        ASHRAE140Case::Case610 => series_600::case_610_south_shading(),
        ASHRAE140Case::Case620 => series_600::case_620_ew_windows(),
        ASHRAE140Case::Case630 => series_600::case_630_ew_shading(),
        ASHRAE140Case::Case640 => series_600::case_640_setback(),
        ASHRAE140Case::Case650 => series_600::case_650_night_vent(),
        ASHRAE140Case::Case600FF => series_600::case_600ff(),
        ASHRAE140Case::Case650FF => series_600::case_650ff(),
        // 900-series + 900FF/950FF extracted to series_900.rs.
        ASHRAE140Case::Case900 => series_900::case_900_baseline(),
        ASHRAE140Case::Case910 => series_900::case_910_south_shading(),
        ASHRAE140Case::Case920 => series_900::case_920_ew_windows(),
        ASHRAE140Case::Case930 => series_900::case_930_ew_shading(),
        ASHRAE140Case::Case940 => series_900::case_940_setback(),
        ASHRAE140Case::Case950 => series_900::case_950_night_vent(),
        ASHRAE140Case::Case900FF => series_900::case_900ff(),
        ASHRAE140Case::Case950FF => series_900::case_950ff(),
        // 960 (sunspace) extracted to series_960.rs.
        ASHRAE140Case::Case960 => series_960::case_960_sunspace(),
        // 970 (5-zone cross-coupling) extracted to series_970.rs.
        ASHRAE140Case::Case970 => series_970::case_970_five_zone_cross_coupling(),
        // 195-series + 196..470 + Office/Retail/School/Warehouse + 800-series
        // + 500..510 + 699 still live as `CaseBuilder` factory methods on the
        // legacy `ashrae_140_cases.rs::CaseBuilder`. Dispatch through there.
        ASHRAE140Case::Case195 => CaseBuilder::case_195_solid_conduction(),
        ASHRAE140Case::Case195HighMass => CaseBuilder::case_195_high_mass(),
        ASHRAE140Case::Case195NoLoads => CaseBuilder::case_195_no_loads(),
        ASHRAE140Case::Case195NoSolar => CaseBuilder::case_195_no_solar(),
        ASHRAE140Case::Case195ThermalBridge => CaseBuilder::case_195_thermal_bridge(),
        ASHRAE140Case::Case195SHGC03 => CaseBuilder::case_195_shgc_low(),
        ASHRAE140Case::Case195SHGC06 => CaseBuilder::case_195_shgc_medium(),
        ASHRAE140Case::Case195SHGC09 => CaseBuilder::case_195_shgc_high(),
        ASHRAE140Case::Case195Albedo01 => CaseBuilder::case_195_albedo_low(),
        ASHRAE140Case::Case195Albedo05 => CaseBuilder::case_195_albedo_medium(),
        ASHRAE140Case::Case195Albedo09 => CaseBuilder::case_195_albedo_high(),
        ASHRAE140Case::Case196 => CaseBuilder::case_196_lighting_diagnostics(),
        ASHRAE140Case::Case197 => CaseBuilder::case_197_equipment_diagnostics(),
        ASHRAE140Case::Case198 => CaseBuilder::case_198_occupancy_diagnostics(),
        ASHRAE140Case::Case200 => CaseBuilder::case_200_combined_internal_loads(),
        ASHRAE140Case::Case250 => CaseBuilder::case_250_thermal_mass_diagnostics(),
        ASHRAE140Case::Case300 => CaseBuilder::case_300_night_ventilation_diagnostics(),
        ASHRAE140Case::Case350 => CaseBuilder::case_350_setback_diagnostics(),
        ASHRAE140Case::Case400 => CaseBuilder::case_400_free_floating_diagnostics(),
        ASHRAE140Case::Case470 => CaseBuilder::case_470_comprehensive_diagnostics(),
        ASHRAE140Case::Office => CaseBuilder::office_building(),
        ASHRAE140Case::Retail => CaseBuilder::retail_building(),
        ASHRAE140Case::School => CaseBuilder::school_building(),
        ASHRAE140Case::Warehouse => CaseBuilder::warehouse_building(),
        ASHRAE140Case::Case800 => CaseBuilder::case_800_heat_pump_single_stage(),
        ASHRAE140Case::Case801 => CaseBuilder::case_801_heat_pump_two_stage(),
        ASHRAE140Case::Case802 => CaseBuilder::case_802_heat_pump_variable_speed(),
        ASHRAE140Case::Case803 => CaseBuilder::case_803_chiller_single(),
        ASHRAE140Case::Case804 => CaseBuilder::case_804_chiller_multiple(),
        ASHRAE140Case::Case805 => CaseBuilder::case_805_boiler_single(),
        ASHRAE140Case::Case806 => CaseBuilder::case_806_boiler_multiple(),
        ASHRAE140Case::Case807 => CaseBuilder::case_807_hybrid_heat_pump_boiler(),
        ASHRAE140Case::Case808 => CaseBuilder::case_808_vav_heat_recovery(),
        ASHRAE140Case::Case809 => CaseBuilder::case_809_cav_economizer(),
        ASHRAE140Case::Case810 => CaseBuilder::case_810_comprehensive_hvac(),
        // 500..510 + 699 lack dedicated factories — fall back to the
        // low-mass baseline (#1293).
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
        | ASHRAE140Case::Case699 => {
            let mut spec = CaseBuilder::case_600_baseline();
            spec.case_id = format!("{:?}", case);
            spec.description = format!("{:?} (fallback baseline — see issue #1293)", case);
            spec
        }
    }
}

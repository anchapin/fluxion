//! ASHRAE 140 Case 600-series (low-mass) factory methods.
//!
//! Issue #3555: extracted from `src/validation/ashrae_140_cases.rs::CaseBuilder`
//! to shrink the legacy monolith and align the in-tree case definitions with
//! the `crate::validation::ashrae140::cases` tree (which already houses
//! `series_195` and `series_800`). Each function here returns the same
//! `CaseSpec` value the corresponding `CaseBuilder` factory did before —
//! callers do not need to change.
//!
//! Issue #3546: added [`build_case`] thin shim so the `build_case` router in
//! `crate::validation::ashrae140::cases::mod` can dispatch the variants this
//! module owns to the `CaseSpec`-returning factories below. The shim does
//! NOT alter any case-definition logic — it only selects which factory runs
//! and bridges the `CaseSpec` to the legacy `ASHRAE140CaseDefinition` surface.

use crate::validation::ashrae140::ASHRAE140CaseDefinition;
use crate::validation::ashrae_140_cases::ASHRAE140Case;
use crate::validation::ashrae_140_cases::{
    CaseBuilder, CaseSpec, HvacSchedule, InternalLoads, NightVentilation, ShadingDevice, WindowSpec,
};

/// Thin routing shim for the 600-series cases `build_case` knows how to
/// dispatch (Issue #3546). Only the variants explicitly required by the
/// issue are wired here — the rest fall through to the catch-all panic in
/// `crate::validation::ashrae140::cases::build_case` (and remain there on
/// purpose; they have no `run_validation_*` callers today).
pub fn build_case(case: ASHRAE140Case) -> ASHRAE140CaseDefinition {
    let spec = match case {
        ASHRAE140Case::Case600 => case_600_baseline(),
        ASHRAE140Case::Case600FF => case_600ff(),
        ASHRAE140Case::Case650FF => case_650ff(),
        _ => panic!("Invalid case for series 600: {:?}", case),
    };
    super::spec_to_definition(case, spec)
}

/// Case 600 — low-mass baseline (8 m × 6 m × 2.7 m, 12 m² south double-clear
/// window, 0.5 ACH, 20°C / 27°C, Denver ground-coupled).
pub fn case_600_baseline() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("600".to_string())
        .with_description(
            "Low mass baseline - standard construction with south windows".to_string(),
        )
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 600 should validate")
}

/// Case 610 — low-mass with 1 m south overhang.
pub fn case_610_south_shading() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("610".to_string())
        .with_description("Low mass with south shading (1m overhang)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_shading(ShadingDevice::overhang(1.0, 2.7))
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 610 should validate")
}

/// Case 620 — low-mass with 6 m² east + 6 m² west windows.
pub fn case_620_ew_windows() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("620".to_string())
        .with_description("Low mass with east/west windows (6m² each)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_ew_windows(6.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 620 should validate")
}

/// Case 630 — low-mass with east/west overhang + fins.
pub fn case_630_ew_shading() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("630".to_string())
        .with_description("Low mass with east/west shading (overhang + fins)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_ew_windows(6.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_shading(ShadingDevice::overhang_and_fins(1.0, 1.0, 2.7))
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 630 should validate")
}

/// Case 640 — low-mass with overnight setback (10°C).
pub fn case_640_setback() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("640".to_string())
        .with_description("Low mass with thermostat setback (overnight)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setback(20.0, 27.0, 10.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 640 should validate")
}

/// Case 650 — low-mass with night ventilation (heating disabled).
pub fn case_650_night_vent() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("650".to_string())
        .with_description("Low mass with night ventilation (no heating)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac(HvacSchedule::with_operating_hours(-100.0, 27.0, 7, 18)) // Heating ALWAYS OFF
        .with_night_ventilation(NightVentilation::case_650())
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 650 should validate")
}

/// Case 600FF — low-mass free-floating (no HVAC, no internal loads).
///
/// Per ASHRAE 140, free-floating cases have NO internal loads.
pub fn case_600ff() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("600FF".to_string())
        .with_description("Low mass free-floating (no HVAC, no internal loads)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        // No internal loads for free-floating cases per ASHRAE 140
        .with_hvac(HvacSchedule::free_floating())
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 600FF should validate")
}

/// Case 650FF — low-mass free-floating with night ventilation.
///
/// Per ASHRAE 140, free-floating cases have NO internal loads.
pub fn case_650ff() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("650FF".to_string())
        .with_description(
            "Low mass free-floating with night ventilation (no internal loads)".to_string(),
        )
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        // No internal loads for free-floating cases per ASHRAE 140
        .with_hvac(HvacSchedule::free_floating())
        .with_night_ventilation(NightVentilation::case_650())
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 650FF should validate")
}

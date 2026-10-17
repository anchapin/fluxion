//! ASHRAE 140 Case 900-series `CaseSpec` factories.
//!
//! Issue #4157 (round 52): factory bodies moved verbatim from
//! `fluxion::validation::ashrae140::cases::series_900` into `fluxion-core` so the
//! main-crate `CaseBuilder` impl no longer interleaves case-factory methods
//! delegating back into the validation crate. The main-crate module re-exports
//! these and keeps the `build_case` router shim plus engine-driving tests.

use crate::ashrae_case_builder::CaseBuilder;
use crate::ashrae_cases::*;
use crate::construction::Assemblies;

/// window, 0.414 ACH (140 Table 5-42 adjusted; 0.5 nominal), 20°C / 27°C, Denver ground-coupled).
pub fn case_900_baseline() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("900".to_string())
        .with_description(
            "High mass baseline - concrete construction with south windows".to_string(),
        )
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_window_u_value_override(crate::ashrae_cases::ASHRAE140_WINDOW_U_VALUE)
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 900 should validate")
}

/// Case 910 — high-mass with 1 m south overhang.
pub fn case_910_south_shading() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("910".to_string())
        .with_description("High mass with south shading (1m overhang)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_window_u_value_override(crate::ashrae_cases::ASHRAE140_WINDOW_U_VALUE)
        .with_shading(ShadingDevice::overhang(1.0, 2.7))
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 910 should validate")
}

/// Case 920 — high-mass with 6 m² east + 6 m² west windows.
pub fn case_920_ew_windows() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("920".to_string())
        .with_description("High mass with east/west windows (6m² each)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_ew_windows(6.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_window_u_value_override(crate::ashrae_cases::ASHRAE140_WINDOW_U_VALUE)
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 920 should validate")
}

/// Case 930 — high-mass with east/west overhang + fins.
pub fn case_930_ew_shading() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("930".to_string())
        .with_description("High mass with east/west shading (overhang + fins)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_ew_windows(6.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_window_u_value_override(crate::ashrae_cases::ASHRAE140_WINDOW_U_VALUE)
        .with_shading(ShadingDevice::overhang_and_fins(1.0, 1.0, 2.7))
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 930 should validate")
}

/// Case 940 — high-mass with overnight setback (10°C).
pub fn case_940_setback() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("940".to_string())
        .with_description("High mass with thermostat setback (overnight)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_window_u_value_override(crate::ashrae_cases::ASHRAE140_WINDOW_U_VALUE)
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setback(20.0, 27.0, 10.0)
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 940 should validate")
}

/// Case 950 — high-mass with night ventilation (heating disabled).
///
/// Per issue #1347 (case_950): the spec wires a HvacSchedule with a
/// 22:00-06:00 setback window (8 h/day = 2920 hours/year, AC4) AND a
/// NightVentilation schedule (18:00-07:00, AC3). Heating is OFF by
/// spec (heating_sp = -100°C — "no heating" per ASHRAE 140 Case 950),
/// so the setback setpoint value is moot (it is overwritten by the
/// operating-hours disabled-region fill in `schedule.rs`). The
/// setback hours are a *marker* in the spec so the validator can
/// assert "HvacSchedule night-flush window = 8 h/day" without
/// changing simulation behavior.
pub fn case_950_night_vent() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("950".to_string())
        .with_description("High mass with night ventilation (no heating)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_window_u_value_override(crate::ashrae_cases::ASHRAE140_WINDOW_U_VALUE)
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac(HvacSchedule::with_operating_hours_and_setback(
            -100.0, 27.0, 7, 18, // operating hours (cooling 7-18, heating OFF always)
            -100.0, 22, 6, // setback window 22:00-06:00 (setpoint -100 → no heating)
        ))
        .with_night_ventilation(NightVentilation::case_650())
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 950 should validate")
}

/// Case 900FF — high-mass free-floating (no HVAC, no internal loads).
///
/// Per ASHRAE 140, free-floating cases have NO internal loads.
pub fn case_900ff() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("900FF".to_string())
        .with_description("High mass free-floating (no HVAC, no internal loads)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::single_clear_glass())
        // No internal loads for free-floating cases per ASHRAE 140
        .with_hvac(HvacSchedule::free_floating())
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 900FF should validate")
}

/// Case 950FF — high-mass free-floating with night ventilation.
///
/// Per ASHRAE 140, free-floating cases have NO internal loads.
pub fn case_950ff() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("950FF".to_string())
        .with_description(
            "High mass free-floating with night ventilation (no internal loads)".to_string(),
        )
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_window_u_value_override(crate::ashrae_cases::ASHRAE140_WINDOW_U_VALUE)
        // No internal loads for free-floating cases per ASHRAE 140
        .with_hvac(HvacSchedule::free_floating())
        .with_night_ventilation(NightVentilation::case_650())
        .with_infiltration(0.414)
        .with_num_zones(1)
        .with_ground_temperature(crate::physics_constants::ASHRAE140_GROUND_TEMPERATURE_C)
        .build()
        .expect("Case 950FF should validate")
}

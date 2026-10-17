//! Teacher-validation reachability — issue #4157 acceptance item 5 (slice 3).
//!
//! ADR-0017 D6 (issue #3986) names the teacher-validation suite the sole
//! flip authority for the DAE solver, but before #4157 slice 3 the zone
//! model (`ThermalModel::from_spec`) could only be constructed through
//! `fluxion::validation::ashrae_140_cases::CaseSpec` — a type that lived in
//! the validation layer. This test proves the DAE-teacher-adjacent zone
//! path is reachable **without touching the validation layer**: the entire
//! file imports `CaseSpec` and its component types from
//! `fluxion_core::ashrae_cases` (and `fluxion_core::construction`) only,
//! builds a `CaseSpec` struct literal (no `CaseBuilder`, which still lives
//! in the validation layer), and constructs + steps a
//! `fluxion::PhysicsThermalModel` from it.
//!
//! Guard-rail for the flip gate: if anyone re-couples the zone-model
//! input type to the validation layer (moving `CaseSpec` back, or making
//! `from_spec` require a validation-layer type), this file stops
//! compiling and the #3986 path loses its non-validation construction
//! route.
//!
//! This test asserts construction and simulation reachability, not
//! numerical conformance; the numerical teacher gates live in
//! `conduction_1052rp_analytical.rs` and `teacher_validation_pcm_box.rs`.

use fluxion::PhysicsThermalModel;
use fluxion::ThermalModelTrait;
use fluxion_core::ashrae_cases::{
    BuildingType, CaseSpec, ConstructionSpec, ConstructionType, GeometrySpec, GlassType,
    HvacSchedule, InternalLoads, NightVentilation, Orientation, ShadingDevice, ShadingType,
    WindowArea, WindowSpec,
};
use fluxion_core::construction::{Assemblies, Construction};

/// A minimal single-zone low-mass CaseSpec built **entirely from
/// fluxion-core types** — the non-validation construction path this file
/// exists to prove.
fn core_only_case_spec() -> CaseSpec {
    let geometry = GeometrySpec::new(8.0, 6.0, 2.7);
    CaseSpec {
        case_id: "core-only-teacher-path".to_string(),
        description: "fluxion_core-only zone model construction (#4157 slice 3)".to_string(),
        geometry: vec![geometry],
        construction_type: ConstructionType::LowMass,
        construction: ConstructionSpec::new(
            Assemblies::low_mass_wall(),
            Assemblies::low_mass_roof(),
            Assemblies::high_mass_floor(),
        ),
        windows: vec![vec![WindowArea::new(12.0, Orientation::South)]],
        window_properties: WindowSpec::double_clear_glass(),
        shading: None::<ShadingDevice>,
        internal_loads: vec![Some(InternalLoads::new(200.0, 0.4, 0.6))],
        hvac: vec![HvacSchedule::constant(20.0, 27.0)],
        night_ventilation: None::<NightVentilation>,
        common_walls: Vec::new(),
        infiltration_ach: 0.5,
        opaque_absorptance: 0.6,
        num_zones: 1,
        zone_origins: None,
        zone_constructions: None,
        weather_data: None,
        door_height: None,
        door_area: None,
        epw_path: None,
        hvac_equipment: None,
        ground_temperature_c: None,
        floor_u_value_override: None,
        window_u_value_override: None,
        building_type: BuildingType::Residential,
    }
}

#[test]
fn thermal_model_constructs_from_core_only_case_spec() {
    let spec = core_only_case_spec();
    assert_eq!(spec.validate(), Ok(()), "core-only spec must be valid");

    // The construction proof: `PhysicsThermalModel::from_spec` accepts the
    // fluxion_core::ashrae_cases::CaseSpec with no validation-layer type in
    // scope anywhere in this file.
    let model = PhysicsThermalModel::from_spec(&spec);
    assert_eq!(model.num_zones(), 1);
    assert!(model.is_valid(), "model must report valid after from_spec");
    assert_eq!(model.mode(), fluxion::ThermalModelMode::Physics);
}

#[test]
fn thermal_model_steps_from_core_only_case_spec() {
    let spec = core_only_case_spec();
    let mut model = PhysicsThermalModel::from_spec(&spec);

    // The DAE-teacher-adjacent conduction path is exercised by stepping the
    // model on the mock (no neural surrogate) manager — the same mock mode
    // the #3986 analytical suites use.
    let surrogates =
        fluxion::ai::surrogate::SurrogateManager::new().expect("SurrogateManager::new (mock mode)");
    let eui = model.solve_timesteps(8760, &surrogates, false);
    assert!(
        eui.is_finite(),
        "annual EUI from the core-only construction path must be finite, got {eui}"
    );

    // Reachability assertions only: this spec carries no `weather_data`, so
    // the annual integration is not physically meaningful (no boundary
    // conditions). The gate here is that the full solve path — network
    // assembly, conduction update, HVAC dispatch — executes end-to-end on a
    // spec built without the validation layer, without panicking, and
    // reports finite outputs. Physical conformance is pinned by
    // `conduction_1052rp_analytical.rs` and the strict gate.
    let temps = model.get_temperatures();
    assert_eq!(temps.len(), 1);
    assert!(
        temps[0].is_finite(),
        "zone air temperature must be finite after the annual run, got {}",
        temps[0]
    );
    assert!(
        model.get_hourly_temperatures().is_some(),
        "hourly temperature profile must be captured after solve_timesteps"
    );
}

// Compile-time assertion that the shading type used above resolves against
// the core leaf types (ShadingType is a leaf type moved by #1441); keeps the
// import list honest if fields are renamed.
const _: fn(f64, Orientation) -> WindowArea = WindowArea::new;
const _: () = {
    let _ = ShadingType::None;
};

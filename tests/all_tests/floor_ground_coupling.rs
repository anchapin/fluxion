//! Data-driven floor-to-ground coupling (PR #4074): the ASHRAE 140
//! Case 195 solid-conduction test specifies its floor U-value
//! (0.039 W/m²K) as spec data (`CaseSpec.floor_u_value_override`) instead
//! of a `case_id == "195"` hard-code. The old 900-series ×1.2 `h_tr_floor`
//! multiplier was dead code (900-series auto-promotes to 9R4C, which never
//! consumes the lumped `h_tr_floor`) and is removed: `h_tr_floor` is now
//! uniformly `floor_u_value × floor_area`.
//!
//! These tests live in the integration runner (not in
//! `src/sim/thermal_model_core/tests.rs`) because they reference
//! `crate::validation::ashrae_140_cases` — keeping them under `src/sim/`
//! adds sim→validation cycle edges that trip
//! `scripts/check_ashrae_cases_cycle.py` and
//! `scripts/check_cycle_downward_trend.py` (Issues #1441 / #2495 / #2768).

use fluxion::physics::cta::VectorField;
use fluxion::sim::construction::SurfaceType as SimSurfaceType;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseBuilder};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn case_195_spec_carries_floor_u_override() {
    let spec = CaseBuilder::case_195_solid_conduction();
    assert!(
        approx_eq(spec.floor_u_value_override.unwrap_or(-1.0), 0.039, 1e-12),
        "Case 195 spec must carry the 0.039 floor U override, got {:?}",
        spec.floor_u_value_override
    );
}

#[test]
fn case_600_spec_has_no_floor_u_override() {
    let spec = ASHRAE140Case::Case600.spec();
    assert!(
        spec.floor_u_value_override.is_none(),
        "Case 600 must use the construction-derived floor U-value"
    );
}

#[test]
fn from_spec_applies_floor_u_override_to_setpoints_and_h_tr_floor() {
    let spec = CaseBuilder::case_195_solid_conduction();
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    assert!(
        approx_eq(model.0.setpoints.floor_u_value, 0.039, 1e-12),
        "setpoints.floor_u_value must be the 0.039 override"
    );
    let zone_area = spec.geometry[0].floor_area();
    let expected = 0.039 * zone_area;
    assert!(
        approx_eq(model.0.conduction.h_tr_floor[0], expected, 1e-9),
        "h_tr_floor must be 0.039 × area, got {}",
        model.0.conduction.h_tr_floor[0]
    );
}

#[test]
fn from_spec_uses_construction_floor_u_without_override() {
    // Case 600: no override → construction U-value, no multiplier.
    let spec = ASHRAE140Case::Case600.spec();
    let expected_u = spec
        .construction
        .floor
        .u_value(Some(SimSurfaceType::Floor), None);
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    assert!(
        approx_eq(model.0.setpoints.floor_u_value, expected_u, 1e-12),
        "setpoints.floor_u_value must be the construction U-value"
    );
    let zone_area = spec.geometry[0].floor_area();
    assert!(
        approx_eq(
            model.0.conduction.h_tr_floor[0],
            expected_u * zone_area,
            1e-9
        ),
        "h_tr_floor must be exactly U × area (no multiplier)"
    );
}

#[test]
fn from_spec_900_series_has_no_floor_multiplier() {
    // Case 900: HighMass, would previously have matched the
    // `is_900_series_hvac` ×1.2 branch. Must now be exactly U × area.
    let spec = ASHRAE140Case::Case900.spec();
    assert!(spec.floor_u_value_override.is_none());
    let expected_u = spec
        .construction
        .floor
        .u_value(Some(SimSurfaceType::Floor), None);
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    let zone_area = spec.geometry[0].floor_area();
    assert!(
        approx_eq(
            model.0.conduction.h_tr_floor[0],
            expected_u * zone_area,
            1e-9
        ),
        "Case 900 h_tr_floor must be exactly U × area, got {}",
        model.0.conduction.h_tr_floor[0]
    );
}

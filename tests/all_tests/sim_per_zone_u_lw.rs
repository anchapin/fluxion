//! ROUND25 (§LIMIT-14 follow-up): per-zone U-values threaded to the 5R1C
//! longwave-exchange path. Lives in the integration suite because the cycle
//! guard (issue #1441/#2495) rejects any new `crate::validation` reference
//! under `src/sim/**`, and the test needs the validation-side CaseSpecs.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

/// ROUND25 (§LIMIT-14 follow-up): the per-zone construction split must
/// reach the 5R1C longwave-exchange path as per-zone U-value vectors.
/// A spec without `zone_constructions` broadcasts the global values
/// (bit-identical legacy behaviour); a split spec carries each zone's
/// own wall/roof/floor U.
#[test]
fn test_per_zone_u_values_threaded_to_lw_path() {
    let mut spec = fluxion::validation::ashrae_140_cases::ASHRAE140Case::Case960.spec();
    let lightweight = spec.construction.clone();
    let heavyweight = spec
        .zone_constructions
        .as_ref()
        .expect("Case 960 carries the split")[1]
        .clone();
    spec.zone_constructions = Some(vec![lightweight.clone(), heavyweight.clone()]);

    let selector = fluxion::sim::thermal_selector::ThermalSelector::default();
    let model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &selector)
        .expect("default selector must initialize");

    let u0 = spec.zone_constructions.as_ref().unwrap()[0]
        .wall
        .u_value(None, None);
    let u1 = spec.zone_constructions.as_ref().unwrap()[1]
        .wall
        .u_value(None, None);
    assert!(
        (model.setpoints.wall_u_values.as_ref()[0] - u0).abs() < 1e-12,
        "back zone must carry its own wall U"
    );
    assert!(
        (model.setpoints.wall_u_values.as_ref()[1] - u1).abs() < 1e-12,
        "sun zone must carry the heavyweight wall U"
    );
    // Legacy broadcast: with no split, the vectors equal the scalar.
    let mut plain = fluxion::validation::ashrae_140_cases::ASHRAE140Case::Case900.spec();
    plain.zone_constructions = None;
    let model_plain = ThermalModel::<VectorField>::from_spec_with_selector(&plain, &selector)
        .expect("default selector must initialize");
    for i in 0..plain.num_zones {
        assert!(
            (model_plain.setpoints.wall_u_values.as_ref()[i] - model_plain.setpoints.wall_u_value)
                .abs()
                < 1e-12
        );
        assert!(
            (model_plain.setpoints.roof_u_values.as_ref()[i] - model_plain.setpoints.roof_u_value)
                .abs()
                < 1e-12
        );
        assert!(
            (model_plain.setpoints.floor_u_values.as_ref()[i]
                - model_plain.setpoints.floor_u_value)
                .abs()
                < 1e-12
        );
    }
    // The spec-level floor override (Case 195) must win in the vector too.
    let spec_195 = fluxion::validation::ashrae_140_cases::ASHRAE140Case::Case195.spec();
    let model_195 = ThermalModel::<VectorField>::from_spec_with_selector(&spec_195, &selector)
        .expect("default selector must initialize");
    assert_eq!(
        model_195.setpoints.floor_u_values.as_ref()[0],
        model_195.setpoints.floor_u_value,
        "floor override must reach the per-zone vector"
    );
}

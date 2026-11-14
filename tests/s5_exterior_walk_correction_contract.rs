//! Loop round 143 — the S5 exterior `h_tr_em` walk correction: wiring the
//! #2613 convention (commit `28cecde7`, PR #2653) into the `from_spec`
//! builder's three inline walks (wall, roof, south wall).
//!
//! Finding (r143 scoping at tip `38e2372e`): the builder's inline
//! exterior-to-mass walks still carry the reversed-index visit that issue
//! #2613 fixed in `compute_r_exterior_to_mass` (`thermal_model_core/mod.rs`,
//! commit `28cecde7`): the loop enumerates `layers[idx]` in INTERIOR-first
//! order while comparing `reverse_idx = n-1-idx`, so the interior-most
//! layer's R folds into the exterior path and the true exterior layer never
//! contributes. The fixed `pub` helper has NO production caller — its
//! regression test (`test_compute_r_exterior_to_mass_case600_wall_physical_value`)
//! pins the corrected Case 600 wall resistance 0.943930 m²K/W vs the buggy
//! 0.954645 on the helper alone. The r141 S5 suite pinned the builder's
//! as-written values as engine truth with the route-as-finding note
//! (`tests/step_9r4c_free_float_contract.rs`, S5 section comment); this is
//! that finding, landed as an engine-semantics fix under the loop's
//! fidelity-first standing direction.
//!
//! Python replication (`/tmp/r143_walk_replication.py`, both walks
//! operation-for-operation on the real assemblies — `Materials`/`Assemblies`
//! literals from `fluxion-core/src/construction.rs`, film 1/18.3, dominant
//! insulation = highest R with LAST-max-on-ties): corrected R values,
//! bit-exact f64 from δ/k arithmetic —
//!   600-series wall: 0.9439305230288837  (as-written 0.9546448087431695)
//!   600-series roof: 1.587859094457455   (as-written 1.5146448087431694)
//!   900-series wall: 1.0194732401157183  (as-written 0.8796448087431693)
//!   900-series roof: 1.577859094457455   (as-written 1.5129412689201605)
//! `h_tr_em = area / R`; the contracts assert the engine values bit-exactly
//! against `area / R_corrected` recomposed from the spec's own layers, and
//! guard against the as-written (buggy) values. The builder-level per-surface
//! fields clear to `None` on the low-mass storage gate (the D5 block in
//! `thermal_model_core/mod.rs`), so W1/W2 pin the conditioned Case 900
//! builder fields, W3 pins the 900FF 9R4C solver nodes, and W4 pins the
//! south-wall site (stored for every model).
//!
//! Contrast twins that must stay green THROUGH the fix (walk-independent
//! forms, already pinned in `tests/step_9r4c_free_float_contract.rs`): the
//! lumped Eq. 64 `h_tr_em` (composed from plain-film u_values, S5c) and the
//! floor `h_tr_em` (the `u_value(None, None)` reciprocal form, S5c). Their
//! continued green is the non-interference evidence; the S5c wall/roof
//! relational pins re-bind to the corrected walk in the same PR.
//!
//! Battery expectation (recorded, not weighted, per the standing
//! bands-ignored direction): the low-mass 600-series family is expected
//! byte-identical (its per-class exterior drive is the R66 boundary-surface
//! aggregates, which do not consume these walks); the 9R4C high-mass family
//! moves (step_9r4c reads `conduction.h_tr_em_wall/roof` for the mass nodes
//! and `conduction.h_tr_em` at three further sites). Full re-record battery
//! with dated old→new provenance rides the same PR.

use fluxion::physics::cta::VectorField;
use fluxion::sim::thermal_model_core::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Layer R-value: δ / k (`ConstructionLayer::r_value`).
fn w143_r_value(layer: &fluxion_core::construction::ConstructionLayer) -> f64 {
    layer.thickness / layer.conductivity
}

/// Dominant-insulation index: highest R, LAST max on ties
/// (`find_dominant_insulation_layer_index` semantics).
fn w143_dominant_ins_idx(layers: &[fluxion_core::construction::ConstructionLayer]) -> usize {
    let mut best = 0usize;
    let mut best_r = w143_r_value(&layers[0]);
    for (j, layer) in layers.iter().enumerate().skip(1) {
        let r = w143_r_value(layer);
        if r >= best_r {
            best_r = r;
            best = j;
        }
    }
    best
}

/// The #2613-corrected exterior-to-mass resistance, transcribed
/// operation-for-operation from `compute_r_exterior_to_mass`
/// (`thermal_model_core/mod.rs:163`): exterior film, then the slice walked
/// exterior→interior — full R strictly exterior to the dominant insulation,
/// half R for the insulation layer, stop. Interior-side layers excluded.
fn w143_r_exterior_corrected(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    let ins = w143_dominant_ins_idx(layers);
    // EXTERIOR_FILM_COEFF_DEFAULT (the engine's own literal, 18.3 W/m²K).
    let mut r = 1.0 / 18.3;
    for idx in (0..layers.len()).rev() {
        let layer_r = w143_r_value(&layers[idx]);
        if idx > ins {
            r += layer_r;
        } else if idx == ins {
            r += layer_r / 2.0;
            break;
        } else {
            break;
        }
    }
    r
}

/// The as-written (pre-fix) walk, transcribed operation-for-operation from
/// the builder's inline loops: interior-first enumerate, `reverse_idx` pair —
/// folds layers[0] (interior-most) into the exterior path. The guards below
/// assert the engine no longer produces these values.
fn w143_r_exterior_as_written(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    let ins = w143_dominant_ins_idx(layers);
    let n = layers.len();
    let mut r = 1.0 / 18.3;
    for (idx, layer) in layers.iter().enumerate() {
        let reverse_idx = n - 1 - idx;
        let layer_r = w143_r_value(layer);
        if reverse_idx > ins {
            r += layer_r;
        } else if reverse_idx == ins {
            r += layer_r / 2.0;
            break;
        } else {
            break;
        }
    }
    r
}

/// Gross exterior wall area and opaque wall area recomposed from the spec
/// geometry, in the builder's own association order (the r141-proven shape:
/// ((south + west) + north) + east).
fn w143_wall_areas(spec: &fluxion_core::ashrae_cases::CaseSpec) -> (f64, f64) {
    use fluxion_core::ashrae_cases::Orientation;
    let g = &spec.geometry[0];
    let south = g.width * g.height;
    let west = g.depth * g.height;
    let north = g.width * g.height;
    let east = g.depth * g.height;
    let ext_area = ((south + west) + north) + east;
    let window = spec.window_area_by_zone_and_orientation(0, Orientation::South);
    let opaque_area = (ext_area - window).max(0.0);
    (ext_area, opaque_area)
}

/// Contract W1 — the builder wall walk: the per-surface `h_tr_em_wall`
/// equals `opaque_area / R_corrected` recomposed from the spec's own wall
/// layers, bit-exact, and does NOT read the as-written (reversed-index)
/// resistance — the #2613 guard at the builder level, where the fix never
/// reached. Observable on the high-mass model: the low-mass storage gate
/// clears `h_tr_em_wall/roof` to `None` (`thermal_model_core/mod.rs`, the
/// D5 storage block), so the low-mass cohort's non-interference rides the
/// battery's byte-identity instead.
#[test]
fn w143_builder_wall_walk_implements_the_2613_convention_high_mass() {
    let spec = ASHRAE140Case::Case900.spec();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let (_ext_area, opaque_area) = w143_wall_areas(&spec);
    let layers = &spec.construction.wall.layers;
    let r_fixed = w143_r_exterior_corrected(layers);
    let r_buggy = w143_r_exterior_as_written(layers);

    #[allow(clippy::float_cmp)]
    {
        assert_eq!(
            model.conduction.h_tr_em_wall.as_ref().expect("wall h_tr_em")[0],
            opaque_area / r_fixed,
            "the builder wall walk must implement the #2613 corrected convention (area/R with the true exterior layer)"
        );
        assert_ne!(
            model.conduction.h_tr_em_wall.as_ref().expect("wall h_tr_em")[0],
            opaque_area / r_buggy,
            "REGRESSION: the wall walk folded the interior layer into the exterior path again (the #2613 pre-fix value)"
        );
    }
}

/// Contract W2 — the builder roof walk: same convention over the spec roof
/// layers with the floor-area-as-roof-area form the builder uses.
#[test]
fn w143_builder_roof_walk_implements_the_2613_convention_high_mass() {
    let spec = ASHRAE140Case::Case900.spec();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let floor_area = spec.geometry[0].floor_area();
    let layers = &spec.construction.roof.layers;
    let r_fixed = w143_r_exterior_corrected(layers);
    let r_buggy = w143_r_exterior_as_written(layers);

    #[allow(clippy::float_cmp)]
    {
        assert_eq!(
            model
                .conduction
                .h_tr_em_roof
                .as_ref()
                .expect("roof h_tr_em")[0],
            floor_area / r_fixed,
            "the builder roof walk must implement the #2613 corrected convention"
        );
        assert_ne!(
            model.conduction.h_tr_em_roof.as_ref().expect("roof h_tr_em")[0],
            floor_area / r_buggy,
            "REGRESSION: the roof walk folded the interior layer into the exterior path again (the #2613 pre-fix value)"
        );
    }
}

/// Contract W3 — the high-mass wall and roof on the 9R4C solver nodes: the
/// per-surface mass-node conductances the 900FF fixture carries recompose
/// from the spec layers under the corrected convention. The foam-core
/// 900-series wall is where the correction moves the coupling most
/// (R 0.8796448087431693 → 1.0194732401157183 m²K/W, ~13.5 % less
/// exterior-to-mass conductance).
#[test]
fn w143_builder_walks_implement_the_2613_convention_high_mass_9r4c() {
    use fluxion::validation::ashrae140::cases::series_900::case_900ff;
    let spec = case_900ff();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let solver = &model.conduction.backend.multi_node_solvers[0];
    let (_ext_area, opaque_area) = w143_wall_areas(&spec);
    let floor_area = spec.geometry[0].floor_area();
    let wall_layers = &spec.construction.wall.layers;
    let roof_layers = &spec.construction.roof.layers;
    let r_wall_fixed = w143_r_exterior_corrected(wall_layers);
    let r_wall_buggy = w143_r_exterior_as_written(wall_layers);
    let r_roof_fixed = w143_r_exterior_corrected(roof_layers);
    let r_roof_buggy = w143_r_exterior_as_written(roof_layers);

    #[allow(clippy::float_cmp)]
    {
        assert_eq!(
            solver.mass.wall.h_tr_em,
            opaque_area / r_wall_fixed,
            "the 9R4C wall node's h_tr_em must recompose under the corrected walk; Python replication 1.0194732401157183 m²K/W"
        );
        assert_ne!(
            solver.mass.wall.h_tr_em,
            opaque_area / r_wall_buggy,
            "REGRESSION: the 9R4C wall node carries the as-written reversed-index value again"
        );
        assert_eq!(
            solver.mass.roof.h_tr_em,
            floor_area / r_roof_fixed,
            "the 9R4C roof node's h_tr_em must recompose under the corrected walk; Python replication 1.577859094457455 m²K/W"
        );
        assert_ne!(
            solver.mass.roof.h_tr_em,
            floor_area / r_roof_buggy,
            "REGRESSION: the 9R4C roof node carries the as-written reversed-index value again"
        );
    }
}

/// Contract W4 — the south-wall site (Issue #715): the stored
/// `h_tr_em_south` recomposes from the same corrected walk over the wall
/// layers with the south opaque area (south gross minus the south window).
#[test]
fn w143_builder_south_wall_site_implements_the_2613_convention() {
    use fluxion_core::ashrae_cases::Orientation;
    let spec = ASHRAE140Case::Case600.spec();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let g = &spec.geometry[0];
    let south_gross = g.width * g.height;
    let south_window = spec.window_area_by_zone_and_orientation(0, Orientation::South);
    let south_opaque = (south_gross - south_window).max(0.0);
    let r_fixed = w143_r_exterior_corrected(&spec.construction.wall.layers);
    let r_buggy = w143_r_exterior_as_written(&spec.construction.wall.layers);

    #[allow(clippy::float_cmp)]
    {
        assert_eq!(
            model.conduction.h_tr_em_south[0],
            south_opaque / r_fixed.max(0.001),
            "the #715 south-wall site must read the corrected walk; Python replication R 0.9439305230288837 m²K/W"
        );
        assert_ne!(
            model.conduction.h_tr_em_south[0],
            south_opaque / r_buggy.max(0.001),
            "REGRESSION: the south-wall site folded the interior layer into the exterior path again"
        );
    }
}

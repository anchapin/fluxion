//! Issue #3063 — wind-dependent `h_tr_em` setup for 5R1C.
//!
//! Lives in its own module to keep `mod.rs` under the module-size ratchet
//! (Issue #2878). The helper populates the four fields added to
//! `ConductionState` for the wind-dependent `h_tr_em` recomputation in
//! `step_5r1c`: `r_materials_wall`, `r_materials_roof` (zone-invariant
//! scalars from the construction layer stack) and `opaque_wall_area`,
//! `roof_area_zone` (per-zone vectors).
//!
//! The per-zone area derivation mirrors the inline logic in
//! `from_spec_with_selector` (lines 1169-1183): the four wall orientation
//! windows are summed via `window_area_by_zone_and_orientation`, then
//! subtracted from `wall_area`. Code duplication is intentional and
//! small; both the legacy default-init and `try_new` use `default_init`.

use crate::sim::thermal_model_data::VectorField;
use crate::validation::ashrae_140_cases::CaseSpec;
use fluxion_core::ashrae_cases::Orientation;

/// Populate the four Issue #3063 fields from a `CaseSpec`. Called by
/// `from_spec_with_selector` after the legacy `h_tr_em_vec` builder
/// finishes. The `num_zones` should match `h_tr_em_vec`'s length so
/// the per-zone vectors stay consistent.
pub fn populate_from_spec(
    conduction: &mut crate::sim::thermal_model_data::ConductionState<VectorField>,
    spec: &CaseSpec,
    num_zones: usize,
) {
    // Material R-values: zone-invariant, derived directly from the layer
    // stack (sum of d/k, no film terms baked in — see helper-test
    // `r_materials_plus_default_films_recovers_u_value_inverse`).
    conduction.r_materials_wall = spec.construction.wall.r_value_materials();
    conduction.r_materials_roof = spec.construction.roof.r_value_materials();
    let mut opaque_wall_areas: Vec<f64> = Vec::with_capacity(num_zones);
    let mut roof_areas: Vec<f64> = Vec::with_capacity(num_zones);
    for zone_idx in 0..num_zones {
        let geom = spec
            .geometry
            .get(zone_idx)
            .unwrap_or_else(|| &spec.geometry[0]);
        // Per-zone window area: mirror the 4-wall-orientation summation in
        // `from_spec_with_selector` (lines 1169-1183).
        let zone_window_area: f64 = [
            Orientation::South,
            Orientation::West,
            Orientation::North,
            Orientation::East,
        ]
        .iter()
        .map(|&orientation| spec.window_area_by_zone_and_orientation(zone_idx, orientation))
        .sum();
        opaque_wall_areas.push(geom.wall_area() - zone_window_area);
        roof_areas.push(geom.floor_area());
    }
    conduction.opaque_wall_area = VectorField::new(opaque_wall_areas);
    conduction.roof_area_zone = VectorField::new(roof_areas);
}

/// Default-init the four Issue #3063 fields for legacy paths that bypass
/// `from_spec_with_selector`. The recomputation in `step_5r1c` is never
/// reached for callers of these paths; leaving the fields at zero avoids
/// producing NaN with uninitialised material R-values.
pub fn default_init(
    conduction: &mut crate::sim::thermal_model_data::ConductionState<VectorField>,
    num_zones: usize,
) {
    conduction.r_materials_wall = 0.0;
    conduction.r_materials_roof = 0.0;
    conduction.opaque_wall_area = VectorField::from_scalar(0.0, num_zones);
    conduction.roof_area_zone = VectorField::from_scalar(0.0, num_zones);
}

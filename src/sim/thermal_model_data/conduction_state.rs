//! Conduction state — concrete solver backend + zone conductances + derived caches.
//!
//! Extracted from `ThermalModelData` (Issue #2878). Wraps the existing
//! [`ConductionBackend`] (CTF/FD/MultiNode/SolverManager/Gauge) plus every
//! per-zone `T`-typed conductance vector and derived-cache that the 5R1C / 6R2C
//! physics hot loops read. Per-config clone is cheap: `ConductionBackend`'s
//! heavy solver `Vec`s are dropped on clone (Issue #2767), and `T` clones are
//! the same allocation as the legacy flat `ThermalModelData::clone` path.

use super::ContinuousTensor;
use crate::sim::boundary::GroundTemperature;

use super::conduction_backend::ConductionBackend;

/// [Round 78 (#4512)] Per-zone R66/R67 class tuples carried per zone by
/// `step_physics_5r1c`: `(cls_g, cls_t, cls_area, cls_beam)`, each `[f64; 3]`
/// over the wall/roof/floor classes.
pub type R66ClassTuples = ([f64; 3], [f64; 3], [f64; 3], [f64; 3]);

pub struct ConductionState<T: ContinuousTensor<f64>> {
    // Concrete solver backend (CTF / FD / MultiNode / SolverManager / Gauge).
    pub backend: ConductionBackend,
    // Per-zone conductances (W/K).
    pub h_tr_w: T,
    pub h_ve: T,
    pub h_tr_floor: T,
    pub h_tr_iz: T,
    pub h_tr_iz_rad: T,
    pub surface_emissivity: T,
    /// Issue #2868 — exterior IR emittance (ASHRAE 140 §5.2 /
    /// `sky_radiation::ashrae_140_default`). `from_spec` overrides this with
    /// the construction's outermost-layer emissivity. Defaults to 0.9 so every
    /// case whose spec keeps the default emissivity is bit-identical to the
    /// pre-#2868 behaviour; Case 195 specifies 0.1 to suppress radiative
    /// exchange and isolate pure conduction.
    pub exterior_emissivity: T,
    pub h_tr_em: T,
    pub h_tr_ms: T,
    pub h_tr_is: T,
    /// h_tr_is excluding south wall contribution (for south wall bypass fix, Issue #715).
    pub h_tr_is_no_south: T,
    /// South wall's h_tr_em for series path computation (Issue #715).
    pub h_tr_em_south: T,
    /// Issue #3063 — opaque envelope material-only R-value for walls (m²·K/W),
    /// used by `step_5r1c` to recompute `h_tr_em` per-timestep with the
    /// wind-dependent exterior film coefficient. Pre-stored (rather than
    /// inverted from `u_value`) to avoid coupling to `u_value`'s internal
    /// default-film assumptions.
    pub r_materials_wall: f64,
    /// Issue #3063 — opaque envelope material-only R-value for roof (m²·K/W).
    pub r_materials_roof: f64,
    /// Issue #3063 — per-zone opaque (wall minus window) area (m²). The
    /// gross `setpoints.wall_area[i]` includes windows and overcounts the
    /// wall contribution to `h_tr_em`, so the per-zone opaque area is
    /// precomputed at construction time.
    pub opaque_wall_area: T,
    /// Issue #3063 — per-zone roof area (m²), equal to the floor area for
    /// the same zone (the legacy `from_spec_with_selector` uses
    /// `zone_floor_area` for the roof contribution).
    pub roof_area_zone: T,
    /// Per-surface thermal mass conductances for 9R4C model (Issue #715, Phase 6B).
    pub h_tr_ms_wall: Option<T>,
    pub h_tr_ms_roof: Option<T>,
    pub h_tr_ms_floor: Option<T>,
    pub h_tr_em_wall: Option<T>,
    pub h_tr_em_roof: Option<T>,
    pub h_tr_em_floor: Option<T>,
    /// Per-surface thermal capacitances for 9R4C model.
    pub cm_wall: Option<T>,
    pub cm_roof: Option<T>,
    pub cm_floor: Option<T>,
    pub cm_internal: Option<T>,
    /// Ground BC used by the FD/5R1C exterior soil coupling path.
    pub ground_temperature: Box<dyn GroundTemperature>,
    /// ISO 13790 §C.6: H_tr_1 = 1/(1/H_ve_adj + 1/H_tr_is) — combined ventilation + surface-to-air.
    pub derived_h_tr_1: T,
    /// ISO 13790 §C.7: H_tr_2 = H_tr_1 + H_tr_w — adds window conductance.
    pub derived_h_tr_2: T,
    /// ISO 13790 §C.8: H_tr_3 = 1/(1/H_tr_2 + 1/H_tr_ms) — combined air-to-mass (~40 W/K for Case 900).
    pub derived_h_tr_3: T,
    /// Optimization cache populated by `update_optimization_cache()`.
    pub derived_h_ext: T,
    pub derived_term_rest_1: T,
    pub derived_h_ms_is_prod: T,
    pub derived_den: T,
    pub derived_ground_coeff: T,
    /// Applied per-zone 5R1C exterior boundary from the most recent step —
    /// the wind-dependent aggregated `h_tr_em` [W/K] and sol-air [°C] the
    /// step actually used (#4113/#4166 semantics). The strict-energy
    /// invariant checker consumes these so its mass-node identity is
    /// reconstructed from the applied values rather than the legacy
    /// roof-style reconstruction that predates the wind-dependent film.
    /// `None` before the first 5R1C step (checker falls back).
    pub applied_5r1c_boundary: Option<(Vec<f64>, Vec<f64>)>,
    /// Applied per-zone 5R1C per-class exterior boundary from the most recent
    /// step (loop round 73, #1522/#4504). The D5-default per-surface mass
    /// advance consumes per-class aggregates (wall/roof exterior conductance
    /// and sol-air, per-class areas and incident-beam weights) plus the
    /// per-zone scalars (`h_air_mass`, blended `t_i`, `phi_m`, `phi_st`,
    /// night-vent mass conductance) that the class loop actually used. The
    /// strict-energy invariant checker consumes these so its per-class
    /// identity is reconstructed from the applied values instead of
    /// re-deriving the ~130-line boundary-surface aggregation — the same
    /// pattern as `applied_5r1c_boundary` (#4373). `None` before the first
    /// step or when the per-surface path did not run (checker falls back).
    pub applied_5r1c_class_boundary: Option<Vec<Option<Applied5R1CClassBoundary>>>,

    /// [Round 78 (#4512) — pooled per-class tuple scratch.] Reusable buffer
    /// for `step_physics_5r1c`'s per-zone R66/R67 class tuples, which the
    /// D5 default flip rebuilt with `vec![None; num_zones]` every step (1
    /// heap allocation per timestep, #4512). `None` until the first step.
    /// Engine-private scratch; contents never read before overwrite.
    pub r66_cls_workspace: Option<Vec<Option<R66ClassTuples>>>,
}

/// Per-zone record of [`ConductionState::applied_5r1c_class_boundary`]
/// (`None` for zones that did not run the per-class path).
#[derive(Debug, Clone)]
pub struct Applied5R1CClassBoundary {
    /// Per-class exterior conductance [W/K] (wall, roof, floor=0).
    pub cls_g: [f64; 3],
    /// Per-class sol-air temperature [°C] (floor: outdoor).
    pub cls_t: [f64; 3],
    /// Per-class opaque area [m²] (floor: slab area).
    pub cls_area: [f64; 3],
    /// Per-class incident irradiance×area [W] (Issue #864 convention).
    pub cls_beam: [f64; 3],
    /// Zone air↔mass conductance actually used (ISO H_tr,3 or the
    /// degenerate series fallback).
    pub h_air_mass: f64,
    /// Blended zone air temperature the mass advance consumed.
    pub t_i: f64,
    /// Zone mass-node heat gain [W].
    pub phi_m_zone: f64,
    /// Zone surface-node heat gain [W].
    pub phi_st_zone: f64,
    /// Night-vent mass conductance [W/K] (0 when inactive).
    pub h_vent_mass_zone: f64,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for ConductionState<T> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            h_tr_w: self.h_tr_w.clone(),
            h_ve: self.h_ve.clone(),
            h_tr_floor: self.h_tr_floor.clone(),
            h_tr_iz: self.h_tr_iz.clone(),
            h_tr_iz_rad: self.h_tr_iz_rad.clone(),
            surface_emissivity: self.surface_emissivity.clone(),
            exterior_emissivity: self.exterior_emissivity.clone(),
            h_tr_em: self.h_tr_em.clone(),
            h_tr_ms: self.h_tr_ms.clone(),
            h_tr_is: self.h_tr_is.clone(),
            h_tr_is_no_south: self.h_tr_is_no_south.clone(),
            h_tr_em_south: self.h_tr_em_south.clone(),
            r_materials_wall: self.r_materials_wall,
            r_materials_roof: self.r_materials_roof,
            opaque_wall_area: self.opaque_wall_area.clone(),
            roof_area_zone: self.roof_area_zone.clone(),
            h_tr_ms_wall: self.h_tr_ms_wall.clone(),
            h_tr_ms_roof: self.h_tr_ms_roof.clone(),
            h_tr_ms_floor: self.h_tr_ms_floor.clone(),
            h_tr_em_wall: self.h_tr_em_wall.clone(),
            h_tr_em_roof: self.h_tr_em_roof.clone(),
            h_tr_em_floor: self.h_tr_em_floor.clone(),
            cm_wall: self.cm_wall.clone(),
            cm_roof: self.cm_roof.clone(),
            cm_floor: self.cm_floor.clone(),
            cm_internal: self.cm_internal.clone(),
            ground_temperature: self.ground_temperature.clone_box(),
            derived_h_tr_1: self.derived_h_tr_1.clone(),
            derived_h_tr_2: self.derived_h_tr_2.clone(),
            derived_h_tr_3: self.derived_h_tr_3.clone(),
            derived_h_ext: self.derived_h_ext.clone(),
            derived_term_rest_1: self.derived_term_rest_1.clone(),
            derived_h_ms_is_prod: self.derived_h_ms_is_prod.clone(),
            derived_den: self.derived_den.clone(),
            derived_ground_coeff: self.derived_ground_coeff.clone(),
            applied_5r1c_boundary: self.applied_5r1c_boundary.clone(),
            applied_5r1c_class_boundary: self.applied_5r1c_class_boundary.clone(),
            r66_cls_workspace: self.r66_cls_workspace.clone(),
        }
    }
}

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
    pub h_tr_em: T,
    pub h_tr_ms: T,
    pub h_tr_is: T,
    /// h_tr_is excluding south wall contribution (for south wall bypass fix, Issue #715).
    pub h_tr_is_no_south: T,
    /// South wall's h_tr_em for series path computation (Issue #715).
    pub h_tr_em_south: T,
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
            h_tr_em: self.h_tr_em.clone(),
            h_tr_ms: self.h_tr_ms.clone(),
            h_tr_is: self.h_tr_is.clone(),
            h_tr_is_no_south: self.h_tr_is_no_south.clone(),
            h_tr_em_south: self.h_tr_em_south.clone(),
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
        }
    }
}

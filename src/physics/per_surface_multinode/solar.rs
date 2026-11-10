//! Solar distribution helpers for the per-surface multi-node formulation.
//!
//! Issue #3983 — the whole point of the per-surface formulation is that the
//! incident solar is **apportioned across the surfaces** by area and
//! absorptance instead of being injected into a single lumped mass node
//! behind a fixed split fraction. The three helpers here are the only places
//! where that apportionment is computed, so the rule is auditable in one
//! spot:
//!
//! * exterior absorption — weight `α_k · A_k` (absorptance × area);
//! * interior (glazing-transmitted) landing — weight `α_int,k · A_k`
//!   (interior-finish absorptance × area);
//! * per-surface absorbed flux — `α_k · I_k · A_k` (irradiance × area).

use super::SurfaceSpec;

/// Solar flux absorbed by the exterior face of a surface [W].
///
/// `Φ_abs,k = α_k · I_k · A_k` — exterior absorptance × incident
/// irradiance × area.
///
/// Negative irradiance or area yields 0 W (a surface that receives no sun
/// absorbs nothing).
pub fn absorbed_flux(spec: &SurfaceSpec, irradiance: f64) -> f64 {
    if irradiance <= 0.0 || spec.area <= 0.0 || spec.absorptance <= 0.0 {
        return 0.0;
    }
    spec.absorptance * irradiance * spec.area
}

/// Distribute a zone-level incident solar flux across the surfaces by
/// `absorptance × area` (ISO 13790-style per-surface distribution).
///
/// Returns the **absorbed** flux carried by each surface [W]; the returned
/// values sum to `total_flux` whenever at least one surface has a non-zero
/// `α · A` weight, and are all zero otherwise.
///
/// The weights are the ISO 13790 per-surface absorption weights: a surface
/// absorbs in proportion to its absorptance and its area, so a large,
/// dark roof outranks a small, light wall — as opposed to the lumped
/// formulation, where every surface's solar lands in one node behind a
/// single fixed split fraction.
pub fn distribute_by_area_absorptance(specs: &[SurfaceSpec], total_flux: f64) -> Vec<f64> {
    let weights: Vec<f64> = specs
        .iter()
        .map(|s| s.absorptance.max(0.0) * s.area.max(0.0))
        .collect();
    let total_weight: f64 = weights.iter().sum();
    if total_flux <= 0.0 || total_weight <= 0.0 {
        return vec![0.0; specs.len()];
    }
    weights
        .iter()
        .map(|w| total_flux * (w / total_weight))
        .collect()
}

/// Distribute glazing-transmitted solar across the interior faces of the
/// surfaces by `interior absorptance × area`.
///
/// Returns the solar flux landing on each surface's interior face [W]; the
/// returned values sum to `total_flux` whenever at least one surface has a
/// non-zero `α_int · A` weight, and are all zero otherwise.
pub fn distribute_transmitted(specs: &[SurfaceSpec], total_flux: f64) -> Vec<f64> {
    let weights: Vec<f64> = specs
        .iter()
        .map(|s| s.interior_absorptance.max(0.0) * s.area.max(0.0))
        .collect();
    let total_weight: f64 = weights.iter().sum();
    if total_flux <= 0.0 || total_weight <= 0.0 {
        return vec![0.0; specs.len()];
    }
    weights
        .iter()
        .map(|w| total_flux * (w / total_weight))
        .collect()
}

//! ISO 13790 thermal network conductance calculations.
//!
//! This module is intended to contain ISO 13790 network conductance calculations
//! for the 5R1C/6R2C thermal network models.
//!
//! ## Module Responsibilities (Future)
//! - h_tr_is: Surface-to-interior air conductance (ISO 13790 Eq. C.4)
//! - h_tr_ms: Mass-to-surface conductance (ISO 13790 Eq. C.5)
//! - h_tr_em: Exterior-to-mass conductance
//! - h_tr_ve: Ventilation conductance (ISO 13790 Eq. C.10)
//!
//! ## Current Status
//! The network conductances are currently pre-computed and stored in ThermalModel
//! data structures (h_tr_is, h_tr_ms, h_tr_em, h_tr_ve fields). The calculation
//! logic is in `thermal_model_core` and `thermal_model_physics`.
//!
//! This module is a marker for future extraction of:
//! 1. Conductance calculation functions (currently in update_derived_parameters)
//! 2. Ventilation conductance calculations (h_ve = ρ × Cp × ACH / 3600)
//! 3. Convective/radiative split calculations (h_cv, h_rad)
//!
//! ## Design Considerations
//! - Conductances depend on surface area, thermal conductivity, thickness
//! - h_tr values are typically computed once during model initialization
//! - The per-zone vectors (h_tr_is, etc.) enable vectorized calculations

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;

/// Calculate surface-to-interior air thermal conductance.
///
/// Per ISO 13790, the surface-to-interior air conductance represents
/// the convective heat transfer coefficient between the interior surface
/// and the zone air node.
///
/// h_tr_is = h_c_i × A_i  [W/K]
///
/// where h_c_i is the internal convective heat transfer coefficient
/// and A_i is the surface area.
impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    // Placeholder for future extraction of h_tr_is calculation
    // Currently the values are pre-computed in thermal_model_core
}

//! HVAC demand calculation for `ThermalModel`.
//!
//! This submodule hosts the HVAC demand calculation that uses the
//! building's total heat transfer conductance to compute zone-level
//! heating and cooling power. Originally part of the monolithic
//! `thermal_model_physics.rs` (Issue #898), extracted as part of the
//! Issue #902 modular split.
//!
//! The `impl` block below adds [`ThermalModel::compute_zone_hvac_load`]
//! to the unified `ThermalModel<T>` type.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Compute HVAC demand using total building heat transfer conductance.
    ///
    /// For ASHRAE 140 Case 600-series (low-mass buildings), the IdealLoadsSystem
    /// was giving ~21.7 W/K (ventilation only) instead of the building's actual
    /// ~1251 W/K total conductance (h_tr_is + h_ve + h_tr_w).
    ///
    /// This caused zones to never reach setpoint because HVAC demand was severely
    /// underestimated.
    ///
    /// Returns a VectorField of power values:
    /// - Positive = heating demand (W)
    /// - Negative = cooling demand (W)
    ///
    /// # Arguments
    /// * `zone_temps` - Current zone temperatures (°C)
    /// * `heating_setpoint` - Single heating setpoint (°C) applied to all zones
    /// * `cooling_setpoint` - Single cooling setpoint (°C) applied to all zones
    pub(crate) fn compute_zone_hvac_load(
        &self,
        zone_temps: &[f64],
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> T {
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let h_ve_vec = self.0.h_ve.as_ref();
        let h_tr_w_vec = self.0.h_tr_w.as_ref();
        let h_tr_ms_vec = self.0.h_tr_ms.as_ref();
        let h_tr_em_vec = self.0.h_tr_em.as_ref();
        let enabled_vec = self.0.hvac_enabled.as_ref();

        let heat_cap = self.0.hvac_heating_capacity;
        let cool_cap = self.0.hvac_cooling_capacity;

        let mut combined_demand = vec![0.0; self.0.num_zones];
        for zone_idx in 0..self.0.num_zones {
            // Check hvac_enabled flag before computing demand
            if enabled_vec[zone_idx] < 0.5 {
                combined_demand[zone_idx] = 0.0;
                continue;
            }

            // Issue #925: HVAC demand coefficient.
            //
            // The previous formula h_coeff = den / (2 * term_rest_1) over-weighted
            // the h_tr_ms contribution, producing excessive HVAC response for
            // high-mass buildings (Case 900 heating 6x over reference, cooling
            // 3x under reference). For Case 600 the same formula happened to
            // land near the reference range, which masked the underlying
            // physics error.
            //
            // Physics: at setpoint, the building's steady-state heat loss
            // coefficient (zone -> outdoor) is the parallel combination of
            // the direct paths (ventilation, window) and the series through-mass
            // path (h_tr_is -> h_tr_ms -> h_tr_em). This is H_total_simple in
            // test_case_600_htotal_verification and equals ~93 W/K for both
            // Case 600 and Case 900 (same envelope, same insulation).
            //
            // h_loss = h_ve + h_tr_w + (h_tr_is × h_tr_ms × h_tr_em) /
            //                            (h_tr_is × h_tr_ms + h_tr_ms × h_tr_em
            //                             + h_tr_em × h_tr_is)
            //
            // This is a true heat-loss coefficient, not the free-floating
            // denominator from t_i_free. The t_i_free formula already includes
            // the mass dynamics via the h_ms_is_prod term, so combining
            // h_loss with t_free correctly captures both the building loss
            // and the mass buffering effect.
            let h_ve = h_ve_vec[zone_idx];
            let h_tr_w = h_tr_w_vec[zone_idx];
            let h_tr_is = h_tr_is_vec[zone_idx];
            let h_tr_ms = h_tr_ms_vec[zone_idx];
            let h_tr_em = h_tr_em_vec[zone_idx];

            // Series conductance: air -> surface -> mass -> envelope exterior
            // 1/h_series = 1/h_tr_is + 1/h_tr_ms + 1/h_tr_em
            let h_loss_via_mass = if h_tr_is > 0.0 && h_tr_ms > 0.0 && h_tr_em > 0.0 {
                let denom = h_tr_is * h_tr_ms + h_tr_ms * h_tr_em + h_tr_em * h_tr_is;
                if denom > 0.0 {
                    h_tr_is * h_tr_ms * h_tr_em / denom
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let h_loss = h_ve + h_tr_w + h_loss_via_mass;

            // Fallback for the degenerate case where any of the series
            // conductances is zero: use the direct path only.
            let h_coeff = if h_loss > 0.0 { h_loss } else { h_ve + h_tr_w };

            let t_zone = zone_temps[zone_idx];

            let demand = if t_zone < heating_setpoint {
                // Heating needed: Q = h_loss × (T_setpoint - T_zone)
                h_coeff * (heating_setpoint - t_zone)
            } else if t_zone > cooling_setpoint {
                // Cooling needed: Q = -h_loss × (T_zone - T_cool_sp)
                -h_coeff * (t_zone - cooling_setpoint)
            } else {
                0.0
            };

            // Clamp to HVAC capacity limits to prevent numerical explosion
            combined_demand[zone_idx] = demand.clamp(-cool_cap, heat_cap);
        }

        T::from(VectorField::new(combined_demand))
    }
}

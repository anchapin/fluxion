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
use smallvec::SmallVec;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Compute the HVAC heat transfer coefficient for the zone air node.
    ///
    /// The HVAC coefficient `h_coeff` is the total effective thermal conductance
    /// from the zone air node to all boundaries, used in the ideal-system load
    /// `Q_HVAC = h_coeff * (T_setpoint - T_free)`.
    ///
    /// # Unified conductance (Issue #4240)
    ///
    /// Both the 5R1C and 9R4C arms use the identical expression — the total
    /// air-node conductance:
    ///
    /// ```text
    ///   h_hvac = (h_tr_w + h_ve) + (h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms))
    /// ```
    ///
    /// where:
    ///   - `h_tr_w + h_ve` is the direct exterior conductance (windows +
    ///     ventilation air exchange), both direct air-to-outdoor paths;
    ///   - `h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms)` is the interior mass path
    ///     (air → surface → mass series combination).
    ///
    /// This corrects the prior 5R1C formula which omitted `h_ve` (Issue #4156:
    /// for Case 600 the code used 134 W/K vs the true 212 W/K, a 0.63x
    /// sensitivity). The doc comment claiming `h_ve` was implicit in `T_free`
    /// was false — `h_ve` belongs in the conductance, not the driving
    /// temperature.
    ///
    /// Note: `derived_h_tr_1` (`= h_ve*h_tr_is/(h_ve+h_tr_is)`, ISO 13790 §C.6)
    /// is NOT used here. It is an intermediate in the Crank-Nicolson mass-update
    /// chain (`derived_h_tr_1 → derived_h_tr_2 → derived_h_tr_3`); `derived_h_tr_3`
    /// is consumed by the thermal time-constant calculation (Issue #894), so the
    /// chain is retained for that purpose. The HVAC coefficient uses the direct
    /// exterior conductance, not the §C.6 series form.
    pub(crate) fn compute_hvac_coefficient(&self, zone_idx: usize) -> f64 {
        let h_tr_is = self.0.conduction.h_tr_is.as_ref()[zone_idx];
        let h_tr_ms = self.0.conduction.h_tr_ms.as_ref()[zone_idx];
        let h_tr_w = self.0.conduction.h_tr_w.as_ref()[zone_idx];
        let h_ve = self.0.conduction.h_ve.as_ref()[zone_idx];
        // Note: h_tr_me (envelope-to-internal-mass coupling) is intentionally NOT used
        // in this function - it's only used for internal mass dynamics, not for
        // the building-to-outdoor HVAC coupling.

        // Interior mass path: air → surface → mass series combination.
        let h_is_m = if h_tr_is + h_tr_ms > 0.0 {
            h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms)
        } else {
            0.0
        };
        // Unified air-node conductance: direct exterior + interior mass path.
        // Identical for 5R1C and 9R4C arms (Issue #4240).
        (h_tr_w + h_ve) + h_is_m
    }

    /// Compute HVAC demand using the symmetric ASHRAE 140 ideal HVAC
    /// sensitivity formulation.
    ///
    /// For both heating and cooling, the demand is:
    ///
    /// ```text
    /// Q_HVAC = h_coeff × (T_setpoint − T_free)
    /// ```
    ///
    /// where `T_free` is the **free-floating zone air temperature** (`t_i_free`
    /// at the call sites) — the equilibrium temperature the zone would reach
    /// with HVAC disabled. `T_free` already includes every heat flow at the
    /// air node: solar gains, internal gains, envelope conduction, ventilation,
    /// AND the dynamic mass heat-release term `h_ms_is_prod × T_mass` that
    /// couples the thermal mass to the air node via the 5R1C heat balance
    /// (see `num_tm` in `step_physics_5r1c`). Using `T_free` therefore does
    /// NOT miss the mass heat release — it captures it exactly once, through
    /// the heat balance.
    ///
    /// # Why the symmetric formula (Issue #1163)
    ///
    /// The previous implementation used an asymmetric cooling formula
    /// `-h_coeff × (T_mass − T_cool_sp)` based on a derivation that claimed
    /// `h_tr_ms × (T_mass − T_zone) = h_coeff × (T_mass − T_zone)`. That
    /// identity holds only if `h_tr_ms = h_coeff`, but in practice they differ
    /// by more than an order of magnitude (`h_tr_ms ≈ 893 W/K` vs
    /// `h_coeff ≈ 70 W/K` for Case 600). The substitution was invalid, and the
    /// resulting cooling formula systematically under-predicted cooling load
    /// (sim/ref_mid ≈ 0.42 — only 42% of the reference). The 44 percentage-point
    /// gap between cooling MAE (69%) and heating MAE (25%) in the blind
    /// validation suite (#1148) was the direct signature of this bug.
    ///
    /// The corrected symmetric formula matches:
    ///   - The heating branch in this same function
    ///   - `MultiNodeSolver::compute_hvac_demand` (`physics/multi_node_solver.rs`),
    ///     which has always used the symmetric `T_air_free` formulation
    ///   - The ASHRAE 140 "ideal HVAC" assumption (infinite-capacity system
    ///     that holds the zone at the setpoint)
    ///
    /// Returns a VectorField of power values:
    /// - Positive = heating demand (W)
    /// - Negative = cooling demand (W)
    ///
    /// # Arguments
    /// * `zone_temps` - Free-floating zone air temperatures `t_i_free` (°C).
    ///   This is the driving temperature for BOTH heating and cooling.
    /// * `heating_setpoints` - Per-zone heating setpoints (°C). Falls back to
    ///   `default_heating_setpoint` (scalar) when `zone_idx` is out of range
    ///   for the slice — see Issue #2826.
    /// * `cooling_setpoints` - Per-zone cooling setpoints (°C). Falls back to
    ///   `default_cooling_setpoint` (scalar) when `zone_idx` is out of range
    ///   for the slice.
    /// * `default_heating_setpoint` - Per-zone-vector fallback scalar heating
    ///   setpoint (°C). Used when the per-zone slice is shorter than the
    ///   model's `num_zones`.
    /// * `default_cooling_setpoint` - Per-zone-vector fallback scalar cooling
    ///   setpoint (°C). Used when the per-zone slice is shorter than the
    ///   model's `num_zones`.
    ///
    /// Issue #2826: Historically the simulation step passed the model's
    /// single-scalar `heating_setpoint` / `cooling_setpoint` fields, so the
    /// `MultiZoneThermalModel.set_zone_setpoints` API (which writes the
    /// per-zone `heating_setpoints` / `cooling_setpoints` vectors) had no
    /// effect on simulated energy. This function now consumes the per-zone
    /// vectors and uses the scalar fields only as a fallback when the
    /// vectors are too short.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_zone_hvac_load(
        &self,
        zone_temps: &[f64],
        heating_setpoints: &[f64],
        cooling_setpoints: &[f64],
        default_heating_setpoint: f64,
        default_cooling_setpoint: f64,
        scratch: &mut SmallVec<[f64; 4]>,
    ) -> T {
        let enabled_vec = self.0.hvac.hvac_enabled.as_ref();

        let heat_cap = self.0.hvac.hvac_heating_capacity;
        let cool_cap = self.0.hvac.hvac_cooling_capacity;

        // Issue #3370: reuse the caller-provided scratch buffer instead of
        // heap-allocating `vec![0.0; n_zones]` on every call. The scratch is
        // cleared and resized to `num_zones` in place — zero allocation
        // after the pool is warm. This is the single biggest contributor to
        // the BatchOracle hot-loop regression (the dhat gate flagged four
        // `compute_zone_hvac_load` invocations per `step_physics_5r1c` call
        // at 140K/config vs the 88K/config #2687 baseline).
        scratch.clear();
        scratch.resize(self.0.hvac.num_zones, 0.0);
        for zone_idx in 0..self.0.hvac.num_zones {
            // Check hvac_enabled flag before computing demand
            if enabled_vec[zone_idx] < 0.5 {
                scratch[zone_idx] = 0.0;
                continue;
            }

            // Issue #907: Norton-equivalent heat-transfer coefficient at the air node
            // (see `compute_hvac_coefficient` doc-comment for derivation).
            let h_coeff = self.compute_hvac_coefficient(zone_idx);

            // Issue #1163: Both branches use the free-floating zone air temperature
            // (T_free), which is the correct driving temperature for the ASHRAE 140
            // ideal HVAC sensitivity formulation. T_free already embeds the mass
            // heat-release term via the 5R1C heat balance (`num_tm` in
            // `step_physics_5r1c`), so the mass contribution is captured exactly
            // once — not zero times, not twice.
            let t_free = zone_temps[zone_idx];

            // Issue #2826: per-zone setpoint read with scalar fallback. The
            // fallback is intentional — `apply_parameters` (BatchOracle) writes
            // only the scalar field, and any caller that has not yet populated
            // the per-zone vector (e.g. `ThermalModel::new` default path)
            // continues to get a sensible, non-NaN setpoint.
            let heating_setpoint = heating_setpoints
                .get(zone_idx)
                .copied()
                .unwrap_or(default_heating_setpoint);
            let cooling_setpoint = cooling_setpoints
                .get(zone_idx)
                .copied()
                .unwrap_or(default_cooling_setpoint);

            let demand = if t_free <= heating_setpoint {
                // Heating: Q = h_coeff × (T_heat_sp − T_free).
                // Use <= so the system actively maintains the setpoint (a zone
                // exactly at the heating setpoint still needs heat input to
                // offset envelope losses).
                h_coeff * (heating_setpoint - t_free)
            } else if t_free >= cooling_setpoint {
                // Cooling: Q = -h_coeff × (T_free − T_cool_sp).
                // Symmetric with heating. The mass heat-release contribution is
                // already in T_free via `num_tm = h_ms_is_prod × T_mass`.
                -h_coeff * (t_free - cooling_setpoint)
            } else {
                // Deadband: T_heat_sp < T_free < T_cool_sp — no HVAC demand.
                // This is the correct ASHRAE 140 behavior: the ideal HVAC system
                // is off when the zone is within the deadband, regardless of the
                // mass temperature. The mass may be warmer than the cooling
                // setpoint, but that heat reaches the zone through the 5R1C
                // coupling and will be removed NEXT timestep once T_free crosses
                // T_cool_sp. Cooling during deadband would violate ASHRAE 140.
                0.0
            };

            // Clamp to HVAC capacity limits to prevent numerical explosion
            scratch[zone_idx] = demand.clamp(-cool_cap, heat_cap);
        }

        T::from(VectorField::from_smallvec(std::mem::take(scratch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;

    /// Analytic test for the unified HVAC conductance (Issue #4240).
    ///
    /// Single-zone network with known conductances:
    /// - h_tr_is = 100 W/K (air -> surface)
    /// - h_tr_ms = 100 W/K (surface -> mass)
    /// - h_tr_w  =  30 W/K (windows, air -> outdoor)
    /// - h_ve    =  70 W/K (ventilation, air -> outdoor)
    ///
    /// Closed-form air-node conductance:
    ///   h_hvac = (h_tr_w + h_ve) + (h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms))
    ///          = (30 + 70) + (100*100/200)
    ///          = 100 + 50 = 150 W/K
    ///
    /// The pre-#4240 5R1C formula omitted h_ve: 50 + 30 = 80 W/K (46.7% low).
    /// Assert < 0.1% against the closed form.
    #[test]
    fn test_unified_hvac_conductance_matches_closed_form() {
        let mut model = ThermalModel::<VectorField>::new(1);

        // Set known conductance values directly.
        model.0.conduction.h_tr_is = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_ms = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_w = VectorField::from_scalar(30.0, 1);
        model.0.conduction.h_ve = VectorField::from_scalar(70.0, 1);

        let h_coeff = model.compute_hvac_coefficient(0);

        // Closed form: (30 + 70) + (100*100/(100+100)) = 150
        let expected = 150.0;
        let rel_err = ((h_coeff - expected) / expected).abs();
        assert!(
            rel_err < 0.001,
            "unified conductance {} W/K differs from closed-form {} W/K by {:.3}%",
            h_coeff,
            expected,
            rel_err * 100.0
        );
    }

    /// The unified conductance must include ventilation (Issue #4156a).
    ///
    /// With h_ve = 0, the conductance drops by exactly h_ve. This guards
    /// against regressions that drop h_ve from the coefficient.
    #[test]
    fn test_hvac_conductance_includes_ventilation() {
        let mut model = ThermalModel::<VectorField>::new(1);

        model.0.conduction.h_tr_is = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_ms = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_w = VectorField::from_scalar(30.0, 1);
        model.0.conduction.h_ve = VectorField::from_scalar(70.0, 1);
        let with_ve = model.compute_hvac_coefficient(0);

        model.0.conduction.h_ve = VectorField::from_scalar(0.0, 1);
        let without_ve = model.compute_hvac_coefficient(0);

        let delta = with_ve - without_ve;
        assert!(
            (delta - 70.0).abs() < 1e-9,
            "removing h_ve=70 should drop conductance by 70, got {}",
            delta
        );
    }
}

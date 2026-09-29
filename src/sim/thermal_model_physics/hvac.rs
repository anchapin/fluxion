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
use crate::sim::thermal_model_core::{ThermalModel, ThermalModelType};
use smallvec::SmallVec;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Compute the HVAC heat transfer coefficient for the 5R1C/6R2C thermal network.
    ///
    /// The HVAC coefficient `h_coeff` represents the total effective thermal conductance
    /// from the zone air node to the outdoor boundary when computing the heating/cooling
    /// load `Q_HVAC = h_coeff * (T_setpoint - T_free)`.
    ///
    /// # ISO 13790 Simple Hourly Method
    ///
    /// Per ISO 13790 §12.2.1 (simple hourly method for monthly/annual energy), the
    /// HVAC demand is:
    ///
    /// ```text
    ///   Φ_HC,heat = (H_tr,1 + H_tr,w) · (θ_int,set,H − θ_air)
    ///   Φ_HC,cool = (H_tr,1 + H_tr,w) · (θ_int,set,C − θ_air)
    /// ```
    ///
    /// where:
    ///   - `H_tr,1 = 1 / (1/H_tr,is + 1/H_tr,ms)` is the conductance from the air node
    ///     through the internal surface to the thermal mass node (series combination
    ///     of h_tr_is and h_tr_ms).
    ///   - `H_tr,w` is the direct window conductance (air to outdoor through glass).
    ///
    /// This coefficient includes ALL paths from air to outdoor for the 5R1C network:
    ///   - Air → Surface → Mass → Outdoor (via `h_tr_ms` × `h_tr_em` chain, captured by
    ///     H_tr,1 coupling)
    ///   - Air → Outdoor via windows (`h_tr_w`)
    ///   - Air → Outdoor via ventilation (`h_ve`, explicit in the coefficient since
    ///     Issue #4241; previously claimed implicit via `T_free`, which was incorrect
    ///     — embedding `h_ve` in `T_free` puts it in the driving temperature, not
    ///     in the coefficient).
    ///
    /// # History (Issue #1457)
    ///
    /// Earlier formulations undersized the load for the 600-series:
    ///   - `den/(2·term_rest_1)` (≈ 154 W/K for Case 600) → 18.62 MWh annual heating
    ///     (3.4x above ASHRAE 140 reference).
    ///   - Norton equivalent `h_is_to_boundary + h_ve` (≈ 76 W/K for Case 600) →
    ///     3.06 MWh annual heating (27-47% BELOW reference 4.36–5.79 MWh).
    ///   - ISO 13790 simple method `H_tr,1 + H_tr,w` (≈ 123 W/K for Case 600) →
    ///     within the published ASHRAE 140 ±15% band.
    ///
    /// The ISO 13790 simple method is a documented standard formula and replaces the
    /// ad-hoc Norton reduction. It does not introduce any free parameter — both
    /// `H_tr,1` and `H_tr,w` are computed directly from the wall assembly properties.
    pub(crate) fn compute_hvac_coefficient(&self, zone_idx: usize) -> f64 {
        let h_tr_is = self.0.conduction.h_tr_is.as_ref()[zone_idx];
        let h_tr_ms = self.0.conduction.h_tr_ms.as_ref()[zone_idx];
        let h_tr_w = self.0.conduction.h_tr_w.as_ref()[zone_idx];
        let h_ve = self.0.conduction.h_ve.as_ref()[zone_idx];
        // Note: h_tr_me (envelope-to-internal-mass coupling) is intentionally NOT used
        // in this function - it's only used for internal mass dynamics, not for
        // the building-to-outdoor HVAC coupling.

        // Issue #4156: Unified conductance (per Alex 2026-09-29, option #2).
        // The HVAC coefficient is the total effective conductance from the
        // zone air node to the outdoor boundary:
        //   h_coeff = (h_tr_w + h_ve) + h_interior_path
        // where:
        //   - (h_tr_w + h_ve) is the direct exterior path (windows + ventilation)
        //   - h_interior_path is the air→surface→mass series path (ISO 13790 §C.3)
        //
        // For 5R1C: h_interior_path = h_tr_is*h_tr_ms/(h_tr_is+h_tr_ms)
        // For 9R4C: h_interior_path = h_tr_is*h_ms_9r4c/(h_tr_is+h_ms_9r4c),
        //   where h_ms_9r4c = h_tr_ms_wall + h_tr_ms_roof + h_tr_ms_floor
        //   (parallel sum of per-surface mass couplings).
        //
        // The 9R4C-specific derived_h_tr_3 is NOT used here — it includes h_ve
        // and h_tr_w in its derivation, which would double-count those terms.
        //
        // Issue #2227: h_tr_me is the coupling between envelope mass and internal mass
        // (furniture/partitions), NOT the building-to-outdoor coupling. Using h_tr_me
        // would incorrectly include furniture thermal mass in the HVAC demand calculation.
        let h_interior_path = if self.0.hvac.thermal_model_type == ThermalModelType::NineRFourC {
            // 9R4C: Derive equivalent single h_tr_ms from per-surface values.
            // The three mass couplings (wall, roof, floor) are in parallel
            // from the interior surface node, so they sum.
            let h_ms_wall = self.0.conduction.h_tr_ms_wall.as_ref()
                .and_then(|v| v.as_ref().get(zone_idx).copied()).unwrap_or(0.0);
            let h_ms_roof = self.0.conduction.h_tr_ms_roof.as_ref()
                .and_then(|v| v.as_ref().get(zone_idx).copied()).unwrap_or(0.0);
            let h_ms_floor = self.0.conduction.h_tr_ms_floor.as_ref()
                .and_then(|v| v.as_ref().get(zone_idx).copied()).unwrap_or(0.0);
            let h_ms_9r4c = h_ms_wall + h_ms_roof + h_ms_floor;
            if h_tr_is + h_ms_9r4c > 0.0 {
                h_tr_is * h_ms_9r4c / (h_tr_is + h_ms_9r4c)
            } else {
                0.0
            }
        } else {
            // 5R1C/6R2C: ISO 13790 §C.3 series combination of air-to-surface
            // film (h_tr_is) and surface-to-mass coupling (h_tr_ms)
            if h_tr_is + h_tr_ms > 0.0 {
                h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms)
            } else {
                0.0
            }
        };
        // Unified: direct exterior (windows + ventilation) + interior mass path
        (h_tr_w + h_ve) + h_interior_path
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
        t_prev: &[f64],
        dt: f64,
        scratch: &mut SmallVec<[f64; 4]>,
    ) -> T {
        let enabled_vec = self.0.hvac.hvac_enabled.as_ref();

        let heat_cap = self.0.hvac.hvac_heating_capacity;
        let cool_cap = self.0.hvac.hvac_cooling_capacity;
        let c_air_vec = self.0.mass.air_thermal_capacitance.as_ref();

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

            // Issue #4241: Discrete zone energy-balance residual (replaces Norton product).
            //
            // The HVAC load is the residual of the discrete air-node energy balance,
            // solved for the Q_HVAC needed to bring the zone to the setpoint:
            //
            //   Q_HVAC = C_air * (T_sp - T_prev)/dt + h_coeff * (T_sp - T_free)
            //
            // where:
            //   - C_air * (T_sp - T_prev)/dt: energy to change air temp from T_prev to T_sp
            //   - h_coeff * (T_sp - T_free): steady-state load to maintain T_sp
            //     (h_coeff is the corrected conductance from #4240, including h_ve)
            //   - T_free: free-floating temperature (equilibrium without HVAC)
            //
            // This replaces the static Norton product Q = h_coeff * (T_sp - T_free),
            // which omitted the capacitance term and could not reproduce
            // capacitance-driven peak damping.
            let h_coeff = self.compute_hvac_coefficient(zone_idx);
            let t_free = zone_temps[zone_idx];
            let t_prev_zone = t_prev.get(zone_idx).copied().unwrap_or(t_free);
            let c_air = c_air_vec.get(zone_idx).copied().unwrap_or(0.0);

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

            // Compute residual loads for both setpoints
            let q_heat = if dt > 0.0 {
                c_air * (heating_setpoint - t_prev_zone) / dt
                    + h_coeff * (heating_setpoint - t_free)
            } else {
                h_coeff * (heating_setpoint - t_free)
            };
            let q_cool = if dt > 0.0 {
                c_air * (cooling_setpoint - t_prev_zone) / dt
                    + h_coeff * (cooling_setpoint - t_free)
            } else {
                h_coeff * (cooling_setpoint - t_free)
            };

            // Issue #4241: Load-based deadband (replaces T_free-based deadband).
            //
            // The ASHRAE 140 ideal system has unlimited capacity and tracks the
            // active setpoint continuously. Instead of using T_free to decide
            // heating/cooling/deadband, we use the sign of the residual load:
            //   - If Q_heat > 0: heating is needed (zone would drift below heating SP)
            //   - Else if Q_cool < 0: cooling is needed (zone would drift above cooling SP)
            //   - Else: true deadband (no load needed to maintain either setpoint)
            let demand = if q_heat > 0.0 {
                // Heating: positive load needed to reach/maintain heating setpoint
                q_heat
            } else if q_cool < 0.0 {
                // Cooling: negative load needed to reach/maintain cooling setpoint
                q_cool
            } else {
                // Deadband: no HVAC demand. The zone is within the control band
                // and the residual loads for both setpoints are non-driving.
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

    /// Tests the corrected 5R1C HVAC coefficient formula (Issue #4241).
    ///
    /// The 5R1C arm uses `(h_tr_w + h_ve) + h_tr_is*h_tr_ms/(h_tr_is+h_tr_ms)`.
    /// This includes `h_ve` (ventilation) in the direct exterior path, fixing
    /// the omission noted in Issue #4156a.
    #[test]
    fn test_5r1c_coefficient_corrected_formula() {
        let mut model = ThermalModel::<VectorField>::new(1);
        // Ensure 5R1C type (default).
        model.0.conduction.h_tr_is = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_ms = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_w = VectorField::from_scalar(30.0, 1);
        model.0.conduction.h_ve = VectorField::from_scalar(70.0, 1);

        let h_coeff = model.compute_hvac_coefficient(0);

        // Corrected: (30 + 70) + (100*100/200) = 100 + 50 = 150.
        let expected = 150.0;
        let rel_err = ((h_coeff - expected) / expected).abs();
        assert!(
            rel_err < 1e-9,
            "5R1C coefficient {} differs from corrected formula {}",
            h_coeff,
            expected
        );
    }

    /// Analytic test for the discrete residual HVAC formulation (Issue #4241).
    ///
    /// Verifies that compute_zone_hvac_load implements:
    ///   Q_HVAC = C_air * (T_sp - T_prev)/dt + h_coeff * (T_sp - T_free)
    ///
    /// to within 0.1% tolerance on a simple single-zone case with known
    /// closed-form solution.
    #[test]
    fn test_residual_formulation_analytic() {
        use smallvec::SmallVec;

        let mut model = ThermalModel::<VectorField>::new(1);
        // Set up simple network: h_tr_is=100, h_tr_ms=100, h_tr_w=30, h_ve=70
        // h_coeff = (30+70) + 100*100/200 = 150 W/K
        model.0.conduction.h_tr_is = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_ms = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_w = VectorField::from_scalar(30.0, 1);
        model.0.conduction.h_ve = VectorField::from_scalar(70.0, 1);
        // C_air = 156000 J/K (Case 600 value)
        model.0.mass.air_thermal_capacitance = VectorField::from_scalar(156000.0, 1);
        // Enable HVAC
        model.0.hvac.hvac_enabled = VectorField::from_scalar(1.0, 1);
        // Set capacities high (no clamping)
        model.0.hvac.hvac_heating_capacity = 1e9;
        model.0.hvac.hvac_cooling_capacity = 1e9;

        // Test case: T_free=15°C, T_prev=18°C, heating SP=20°C, dt=3600s
        // Expected Q_heat = C_air*(20-18)/3600 + 150*(20-15)
        //                = 156000*2/3600 + 150*5
        //                = 86.67 + 750 = 836.67 W
        let t_free = 15.0;
        let t_prev = 18.0;
        let heating_sp = 20.0;
        let cooling_sp = 27.0;
        let dt = 3600.0;

        let zone_temps = vec![t_free];
        let heating_sps = vec![heating_sp];
        let cooling_sps = vec![cooling_sp];
        let t_prev_vec = vec![t_prev];
        let mut scratch = SmallVec::<[f64; 4]>::new();

        let result = model.compute_zone_hvac_load(
            &zone_temps,
            &heating_sps,
            &cooling_sps,
            heating_sp,
            cooling_sp,
            &t_prev_vec,
            dt,
            &mut scratch,
        );

        let q_computed = result.as_ref()[0];
        let h_coeff = 150.0; // (30+70) + 50
        let q_expected = 156000.0 * (heating_sp - t_prev) / dt + h_coeff * (heating_sp - t_free);

        let rel_err = ((q_computed - q_expected) / q_expected).abs();
        assert!(
            rel_err < 1e-3, // 0.1% tolerance
            "Residual Q_heat {} differs from analytic {} (rel_err={})",
            q_computed,
            q_expected,
            rel_err
        );

        // Verify cooling case: T_free=30°C, T_prev=28°C, cooling SP=27°C
        // Expected Q_cool = C_air*(27-28)/3600 + 150*(27-30)
        //                = -43.33 + (-450) = -493.33 W (negative = cooling)
        let t_free_cool = 30.0;
        let t_prev_cool = 28.0;
        let zone_temps_cool = vec![t_free_cool];
        let t_prev_vec_cool = vec![t_prev_cool];
        let mut scratch2 = SmallVec::<[f64; 4]>::new();

        let result_cool = model.compute_zone_hvac_load(
            &zone_temps_cool,
            &heating_sps,
            &cooling_sps,
            heating_sp,
            cooling_sp,
            &t_prev_vec_cool,
            dt,
            &mut scratch2,
        );

        let q_cool_computed = result_cool.as_ref()[0];
        let q_cool_expected =
            156000.0 * (cooling_sp - t_prev_cool) / dt + h_coeff * (cooling_sp - t_free_cool);

        let rel_err_cool = ((q_cool_computed - q_cool_expected) / q_cool_expected.abs()).abs();
        assert!(
            rel_err_cool < 1e-3,
            "Residual Q_cool {} differs from analytic {} (rel_err={})",
            q_cool_computed,
            q_cool_expected,
            rel_err_cool
        );
        assert!(
            q_cool_computed < 0.0,
            "Cooling load must be negative, got {}",
            q_cool_computed
        );
    }
}

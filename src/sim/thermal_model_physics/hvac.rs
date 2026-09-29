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
// Issue #4241: only used by `compute_hvac_coefficient`, which is cfg-gated.
#[cfg(any(test, feature = "debug-physics", feature = "gauge-solver"))]
use crate::sim::thermal_model_core::ThermalModelType;
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
    ///   Φ_HC,heat = (H_tr,1 + H_tr,w + H_ve) · (θ_int,set,H − θ_air)
    ///   Φ_HC,cool = (H_tr,1 + H_tr,w + H_ve) · (θ_int,set,C − θ_air)
    /// ```
    ///
    /// where:
    ///   - `H_tr,1 = 1 / (1/H_tr,is + 1/H_tr,ms)` is the conductance from the air node
    ///     through the internal surface to the thermal mass node (series combination
    ///     of h_tr_is and h_tr_ms).
    ///   - `H_tr,w` is the direct window conductance (air to outdoor through glass).
    ///   - `H_ve` is the ventilation conductance (air to outdoor via infiltration/ventilation).
    ///
    /// This coefficient includes ALL paths from air to outdoor for the 5R1C network:
    ///   - Air → Surface → Mass → Outdoor (via `h_tr_ms` × `h_tr_em` chain, captured by
    ///     H_tr,1 coupling)
    ///   - Air → Outdoor via windows (`h_tr_w`)
    ///   - Air → Outdoor via ventilation (`h_ve`) — Issue #4156a / #4241
    ///
    /// # History (Issue #1457, #4156a, #4241)
    ///
    /// Earlier formulations undersized the load for the 600-series:
    ///   - `den/(2·term_rest_1)` (≈ 154 W/K for Case 600) → 18.62 MWh annual heating
    ///     (3.4x above ASHRAE 140 reference).
    ///   - Norton equivalent `h_is_to_boundary + h_ve` (≈ 76 W/K for Case 600) →
    ///     3.06 MWh annual heating (27-47% BELOW reference 4.36–5.79 MWh).
    ///   - ISO 13790 simple method `H_tr,1 + H_tr,w` (≈ 123 W/K for Case 600) →
    ///     within the published ASHRAE 140 ±15% band.
    ///   - Issue #4156a / #4241: added `H_ve` to capture ventilation losses, giving
    ///     `H_tr,1 + H_tr,w + H_ve` (≈ 150 W/K for Case 600). Since #4241 the
    ///     residual load law uses `den` (the step denominator) directly; this
    ///     coefficient remains available for diagnostics and the gauge path.
    ///
    /// The ISO 13790 simple method plus ventilation is a documented standard formula
    /// and replaces the ad-hoc Norton reduction. It does not introduce any free
    /// parameter — `H_tr,1`, `H_tr,w`, and `H_ve` are all computed directly from
    /// the wall assembly and ventilation properties.
    ///
    /// Issue #4241: no longer the load law (the residual law in
    /// `compute_zone_hvac_load` replaced it). Retained for the debug-physics
    /// breakdown, the gauge single-zone path, and unit tests.
    #[cfg(any(test, feature = "debug-physics", feature = "gauge-solver"))]
    pub(crate) fn compute_hvac_coefficient(&self, zone_idx: usize) -> f64 {
        let h_tr_is = self.0.conduction.h_tr_is.as_ref()[zone_idx];
        let h_tr_ms = self.0.conduction.h_tr_ms.as_ref()[zone_idx];
        let h_tr_w = self.0.conduction.h_tr_w.as_ref()[zone_idx];
        let h_ve = self.0.conduction.h_ve.as_ref()[zone_idx];
        // Note: h_tr_me (envelope-to-internal-mass coupling) is intentionally NOT used
        // in this function - it's only used for internal mass dynamics, not for
        // the building-to-outdoor HVAC coupling.

        // For 9R4C models (Case 900), the HVAC coupling to zone air uses
        // derived_h_tr_3 + h_tr_w instead of the 5R1C series formula
        // h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms).
        //
        // Issue #2227: h_tr_me is the coupling between envelope mass and internal mass
        // (furniture/partitions), NOT the building-to-outdoor coupling. Using h_tr_me
        // would incorrectly include furniture thermal mass in the HVAC demand calculation.
        //
        // Issue #4156a / #4241: added `h_ve` to capture ventilation losses. This is the
        // key change that makes the peak heating metrics move toward band.
        //
        // The correct formula uses derived_h_tr_3 (ISO 13790 combined air-to-mass conductance
        // ≈ 42.66 W/K for Case 900) which represents the effective thermal coupling from
        // zone air to the building's thermal mass (envelope), plus h_tr_w for windows,
        // plus h_ve for ventilation.
        let hvac_coeff = if self.0.hvac.thermal_model_type == ThermalModelType::NineRFourC {
            // 9R4C: derived_h_tr_3 + h_tr_w + h_ve is the total HVAC coupling
            let derived_h_tr_3 = self.0.conduction.derived_h_tr_3.as_ref()[zone_idx];
            derived_h_tr_3 + h_tr_w + h_ve
        } else {
            // 5R1C/6R2C: ISO 13790 §C.3 series combination of air-to-surface
            // film (h_tr_is) and surface-to-mass coupling (h_tr_ms), plus h_ve
            let h_tr_1 = if h_tr_is + h_tr_ms > 0.0 {
                h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms)
            } else {
                0.0
            };
            // ISO 13790 §12.2.1 — h_coeff = H_tr,1 + H_tr,w + H_ve
            h_tr_1 + h_tr_w + h_ve
        };
        hvac_coeff
    }

    /// Compute HVAC demand using the discrete energy balance residual formulation
    /// (Issue #4241).
    ///
    /// The ideal-system load is computed as the residual of the discrete zone
    /// air energy balance over the timestep:
    ///
    /// ```text
    /// Q_HVAC = den × T_sp − num_tm − num_rest
    ///          + C_air × (T_sp − T_prev) / dt
    /// ```
    ///
    /// where:
    ///   - `den` = total air-to-outdoor conductance denominator
    ///   - `T_sp` = active setpoint temperature (°C)
    ///   - `num_tm` = thermal mass heat release term (`h_ms_is_prod × T_mass`)
    ///   - `num_rest` = remaining gains numerator (solar Φ, internal gains φ_ia,
    ///     h_ext × T_outdoor, ground coupling)
    ///   - `C_air` = zone air thermal capacitance (J/K)
    ///   - `T_prev` = previous timestep air temperature (°C)
    ///   - `dt` = timestep duration (s)
    ///
    /// The load thereby gains time resolution from the zone air capacitance —
    /// the previous steady-state law `Q = h_coeff × (T_sp − T_free)` cannot
    /// reproduce capacitance-driven peak damping.
    ///
    /// # Why the discrete residual (Issue #4241)
    ///
    /// The previous steady-state law computed load as a sensitivity:
    /// `Q = h_coeff × (T_sp − T_free)`. This formulation has NO time constant
    /// — it responds to the free-floating temperature instantly with no damping.
    /// The symptom was Case 600/620 peak heating +15% OVER while annual heating
    /// sat in band, because the peak transient was not damped by the air
    /// node capacitance.
    ///
    /// The discrete residual formula includes `C_air × (T_sp − T_prev) / dt`,
    /// which captures the energy stored/released by the air node over the
    /// timestep. This gives the load law a time constant and enables
    /// capacitance-driven peak damping.
    ///
    /// # No deadband
    ///
    /// The ASHRAE 140 ideal system has unlimited capacity and tracks the active
    /// setpoint continuously. The deadband branch that let the zone free-float
    /// across the full 20→27°C band has been removed. The ideal system is ALWAYS
    /// active at whichever setpoint (heating or cooling) the zone is closer to.
    ///
    /// Returns a VectorField of power values:
    /// - Positive = heating demand (W)
    /// - Negative = cooling demand (W)
    ///
    /// # Arguments
    /// * `zone_temps` - Free-floating zone air temperatures `t_i_free` (°C).
    ///   Used only to determine the active setpoint (heating vs cooling).
    /// * `heating_setpoints` - Per-zone heating setpoints (°C). Falls back to
    ///   `default_heating_setpoint` (scalar) when `zone_idx` is out of range
    ///   for the slice — see Issue #2826.
    /// * `cooling_setpoints` - Per-zone cooling setpoints (°C). Falls back to
    ///   `default_cooling_setpoint` (scalar) when `zone_idx` is out of range
    ///   for the slice.
    /// * `default_heating_setpoint` - Per-zone-vector fallback scalar heating
    ///   setpoint (°C).
    /// * `default_cooling_setpoint` - Per-zone-vector fallback scalar cooling
    ///   setpoint (°C).
    /// * `num_tm` - Thermal mass heat release numerator (W): `h_ms_is_prod × T_mass`
    ///   for each zone.
    /// * `num_rest` - Remaining gains numerator (W): includes solar gains, internal
    ///   gains (φ_ia), outdoor-temperature conduction (h_ext × T_outdoor), ground
    ///   coupling (ground_coeff × T_g), and inter-zone exchange. Must include
    ///   ALL gains terms to avoid the #4241-previous-defect (dropping gains).
    /// * `den` - Total air-to-outdoor conductance denominator (W/K) for each zone,
    ///   in the SCALED basis (multiplied by `term_rest_1`).
    /// * `term_rest_1` - Per-zone scale factor (`h_tr_ms + h_tr_is`) that was
    ///   applied to `den`, `num_tm`, and `num_rest` to clear the
    ///   surface-temperature denominator. The residual is unscaled by this
    ///   factor to recover physical watts (Issue #2868 pattern:
    ///   `den_true = den / term_rest_1`). Values <= 0 fall back to 1.0.
    /// * `c_air` - Zone air thermal capacitance (J/K) for each zone (physical,
    ///   unscaled — the capacitive term is already in physical units).
    /// * `t_prev` - Previous timestep air temperature (°C) for each zone.
    /// * `dt_seconds` - Timestep duration in seconds.
    /// * `scratch` - Caller-provided scratch buffer (reused to avoid allocation).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_zone_hvac_load(
        &self,
        zone_temps: &[f64],
        heating_setpoints: &[f64],
        cooling_setpoints: &[f64],
        default_heating_setpoint: f64,
        default_cooling_setpoint: f64,
        num_tm: &[f64],
        num_rest: &[f64],
        den: &[f64],
        term_rest_1: &[f64],
        c_air: &[f64],
        t_prev: &[f64],
        dt_seconds: f64,
        scratch: &mut SmallVec<[f64; 4]>,
    ) -> T {
        let enabled_vec = self.0.hvac.hvac_enabled.as_ref();

        let heat_cap = self.0.hvac.hvac_heating_capacity;
        let cool_cap = self.0.hvac.hvac_cooling_capacity;

        // Issue #3370: reuse the caller-provided scratch buffer instead of
        // heap-allocating `vec![0.0; n_zones]` on every call.
        scratch.clear();
        scratch.resize(self.0.hvac.num_zones, 0.0);

        for zone_idx in 0..self.0.hvac.num_zones {
            // Check hvac_enabled flag before computing demand
            if enabled_vec[zone_idx] < 0.5 {
                scratch[zone_idx] = 0.0;
                continue;
            }

            // Get the thermal network quantities for this zone
            let num_tm_i = num_tm[zone_idx];
            let num_rest_i = num_rest[zone_idx];
            let den_i = den[zone_idx];
            let c_air_i = c_air[zone_idx];
            let t_prev_i = t_prev[zone_idx];
            let t_free = zone_temps[zone_idx];

            // Issue #2826: per-zone setpoint read with scalar fallback.
            let heating_setpoint = heating_setpoints
                .get(zone_idx)
                .copied()
                .unwrap_or(default_heating_setpoint);
            let cooling_setpoint = cooling_setpoints
                .get(zone_idx)
                .copied()
                .unwrap_or(default_cooling_setpoint);

            // Determine the active setpoint (the one closer to t_free)
            // The ASHRAE 140 ideal system tracks the active setpoint continuously.
            // No deadband: the system is ALWAYS active at whichever setpoint is relevant.
            let dist_to_heat = (t_free - heating_setpoint).abs();
            let dist_to_cool = (t_free - cooling_setpoint).abs();

            let t_sp: f64 = if dist_to_heat <= dist_to_cool {
                heating_setpoint
            } else {
                cooling_setpoint
            };

            // Discrete energy balance residual formula (physical watts):
            // Q_HVAC = (den × T_sp − num_tm − num_rest) / term_rest_1
            //          + C_air × (T_sp − T_prev) / dt
            //
            // Physics: the backward-Euler discrete air-node energy balance is
            //   C_air × (T_new − T_prev) / dt = num_true − den_true × T_new + Q_HVAC
            // where num_total = num_tm + num_rest. Steady state (Q_HVAC = 0,
            // dT/dt = 0) gives T_new = num_total / den = t_free, matching the
            // free-float calculation. Solving for the ideal-system load that
            // holds T_new = T_sp:
            //   Q_HVAC = den_true × T_sp − num_true + C_air × (T_sp − T_prev) / dt
            //
            // `den` / `num_tm` / `num_rest` arrive in the SCALED basis
            // (× term_rest_1 = h_tr_ms + h_tr_is, to clear the
            // surface-temperature denominator), so the residual is divided by
            // `term_rest_1` to recover physical watts — the Issue #2868
            // `den_true = den / term_rest_1` pattern. The capacitive term uses
            // the physical (unscaled) C_air; it must NOT be multiplied by
            // term_rest_1 (that would state the load in scaled units).
            //
            // The sign convention: positive = heating, negative = cooling.
            // When is_heating=true and T_sp > T_free, the demand is positive.
            // When is_heating=false and T_sp < T_free, the demand is negative.

            let c_air_dt = if dt_seconds > 0.0 && c_air_i > 0.0 {
                c_air_i / dt_seconds
            } else {
                0.0
            };

            // Unscale the residual to physical watts (see doc comment above).
            let scale_i = term_rest_1.get(zone_idx).copied().unwrap_or(1.0);
            let scale_i = if scale_i > 0.0 { scale_i } else { 1.0 };

            // Full numerator at setpoint: den × T_sp (all outflows at setpoint temperature)
            let num_outflow_at_sp = den_i * t_sp;

            // The energy balance residual: positive = heating needed, negative = cooling needed
            let demand = (num_outflow_at_sp - num_tm_i - num_rest_i) / scale_i
                + c_air_dt * (t_sp - t_prev_i);

            // Clamp to HVAC capacity limits to prevent numerical explosion
            scratch[zone_idx] = demand.clamp(-cool_cap, heat_cap);
        }

        T::from(VectorField::from_smallvec(std::mem::take(scratch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Documents the 5R1C HVAC coefficient formula (Issue #4241).
    ///
    /// The 5R1C arm uses `h_tr_is*h_tr_ms/(h_tr_is+h_tr_ms) + h_tr_w + h_ve`.
    /// Issue #4241 added `h_ve` to capture ventilation losses in the discrete
    /// energy balance residual load law.
    #[test]
    fn test_5r1c_coefficient_with_h_ve() {
        let mut model = ThermalModel::<VectorField>::new(1);
        // Ensure 5R1C type (default).
        model.0.conduction.h_tr_is = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_ms = VectorField::from_scalar(100.0, 1);
        model.0.conduction.h_tr_w = VectorField::from_scalar(30.0, 1);
        model.0.conduction.h_ve = VectorField::from_scalar(70.0, 1);

        let h_coeff = model.compute_hvac_coefficient(0);

        // After #4241: (100*100/200) + 30 + 70 = 150 (h_ve IS included).
        let expected = 150.0;
        let rel_err = ((h_coeff - expected) / expected).abs();
        assert!(
            rel_err < 1e-9,
            "5R1C coefficient {} differs from expected {} (with h_ve)",
            h_coeff,
            expected
        );
    }

    /// Tests the discrete energy balance residual HVAC load formula with gains
    /// (Issue #4241 previous-defect catch).
    ///
    /// The previous defect dropped `num_rest` from the numerator, so the test
    /// fixture MUST include non-zero gains (solar + internal) so the numerator
    /// matters. A gains-free fixture (h_ms_is_prod=0, phi_ia=0) would pass
    /// with the wrong formula.
    #[test]
    fn test_discrete_energy_balance_with_gains() {
        let mut model = ThermalModel::<VectorField>::new(1);
        // Enable HVAC
        model.0.hvac.hvac_enabled = VectorField::from_scalar(1.0, 1);

        // Set up a simple zone
        let _zone_idx = 0;

        // Thermal network parameters (free-float consistent: num_tm + num_rest = den * t_free)
        let den = 150.0; // W/K total conductance
        let num_tm = 1000.0; // Thermal mass heat release (h_ms_is_prod * T_mass)
        let num_rest = 1250.0; // Non-zero gains: solar + internal (the key test!)
        let c_air = 156000.0; // J/K for Case 600 (129.6 m³)
        let t_prev = 20.0; // °C from previous timestep (was held at heating setpoint)
        let dt = 3600.0; // 1 hour timestep

        // Setpoint
        let heating_sp = 20.0;
        let cooling_sp = 27.0;

        // Free-float temp (zone would reach without HVAC)
        let t_free = 15.0; // Below heating setpoint → heating needed
                           // Consistency: num_tm + num_rest = 2250 = den * t_free = 150 * 15. ✓

        // Compute load with active heating setpoint
        let mut scratch: SmallVec<[f64; 4]> = SmallVec::with_capacity(1);
        let load = model.compute_zone_hvac_load(
            &[t_free],     // zone_temps
            &[heating_sp], // heating_setpoints
            &[cooling_sp], // cooling_setpoints
            heating_sp,    // default_heating_setpoint
            cooling_sp,    // default_cooling_setpoint
            &[num_tm],     // num_tm
            &[num_rest],   // num_rest (non-zero to catch previous defect)
            &[den],        // den
            &[1.0],        // term_rest_1 (physical basis: scale = 1)
            &[c_air],      // c_air
            &[t_prev],     // t_prev
            dt,            // dt_seconds
            &mut scratch,
        );

        // Expected load: den * T_sp - num_tm - num_rest + C_air * (T_sp - T_prev) / dt
        // = 150 * 20 - 1000 - 1250 + 156000 * (20 - 20) / 3600
        // = 3000 - 2250 + 0
        // = 750 W (positive = heating)
        let expected = den * heating_sp - num_tm - num_rest + c_air * (heating_sp - t_prev) / dt;
        let actual = load.as_ref()[0];

        let rel_err = ((actual - expected) / expected.abs().max(1.0)).abs();
        assert!(
            rel_err < 1e-9,
            "Discrete load {} differs from expected {} by {:.2}%",
            actual,
            expected,
            rel_err * 100.0
        );

        // Now test with cooling setpoint active (t_free > cooling_sp).
        // Free-float consistent fixture: num_tm + num_rest = den * t_free.
        let num_tm_cool = 2000.0;
        let num_rest_cool = 2500.0; // 2000 + 2500 = 4500 = 150 * 30 ✓
        let t_prev_cool = 27.0; // zone was held at the cooling setpoint last step
        let t_free_cooling = 30.0; // Above cooling setpoint → cooling needed
        let load_cool = model.compute_zone_hvac_load(
            &[t_free_cooling], // zone_temps
            &[heating_sp],     // heating_setpoints
            &[cooling_sp],     // cooling_setpoints
            heating_sp,        // default_heating_setpoint
            cooling_sp,        // default_cooling_setpoint
            &[num_tm_cool],    // num_tm
            &[num_rest_cool],  // num_rest
            &[den],            // den
            &[1.0],            // term_rest_1 (physical basis: scale = 1)
            &[c_air],          // c_air
            &[t_prev_cool],    // t_prev (at cooling setpoint, not cold)
            dt,                // dt_seconds
            &mut scratch,
        );

        // Expected cooling load: den * T_cool_sp - num_tm - num_rest + C_air * (T_cool_sp - T_prev) / dt
        // = 150 * 27 - 2000 - 2500 + 156000 * (27 - 27) / 3600
        // = 4050 - 4500 + 0
        // = -450 W (negative = cooling)
        let expected_cool = den * cooling_sp - num_tm_cool - num_rest_cool
            + c_air * (cooling_sp - t_prev_cool) / dt;
        let actual_cool = load_cool.as_ref()[0];

        // The result should be negative (cooling)
        assert!(
            actual_cool < 0.0,
            "Cooling load should be negative, got {} W",
            actual_cool
        );

        let rel_err_cool = ((actual_cool - expected_cool) / expected_cool.abs().max(1.0)).abs();
        assert!(
            rel_err_cool < 1e-9,
            "Cooling load {} differs from expected {} by {:.2}%",
            actual_cool,
            expected_cool,
            rel_err_cool * 100.0
        );

        // Scaled-basis regression (Issue #4241 review): the 5R1C/9R4C step
        // functions pass `den` / `num_tm` / `num_rest` multiplied by
        // term_rest_1. The load must come back in physical watts, i.e. the
        // scaled residual divided by the scale factor. Without the unscaling
        // the Case 600 annual heating came out ~95x too high (487 MWh vs the
        // 4.36–5.79 MWh reference band).
        let scale = 95.0; // representative term_rest_1 magnitude
        let load_scaled = model.compute_zone_hvac_load(
            &[t_free],           // zone_temps
            &[heating_sp],       // heating_setpoints
            &[cooling_sp],       // cooling_setpoints
            heating_sp,          // default_heating_setpoint
            cooling_sp,          // default_cooling_setpoint
            &[num_tm * scale],   // num_tm (scaled basis)
            &[num_rest * scale], // num_rest (scaled basis)
            &[den * scale],      // den (scaled basis)
            &[scale],            // term_rest_1
            &[c_air],            // c_air (physical, unscaled)
            &[t_prev],           // t_prev
            dt,                  // dt_seconds
            &mut scratch,
        );
        let expected_heat =
            den * heating_sp - num_tm - num_rest + c_air * (heating_sp - t_prev) / dt;
        let actual_scaled = load_scaled.as_ref()[0];
        let rel_err_scaled = ((actual_scaled - expected_heat) / expected_heat.abs().max(1.0)).abs();
        assert!(
            rel_err_scaled < 1e-9,
            "Scaled-basis load {} differs from physical expected {} by {:.2}%",
            actual_scaled,
            expected_heat,
            rel_err_scaled * 100.0
        );
    }
}

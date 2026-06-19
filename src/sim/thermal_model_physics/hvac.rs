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
    /// Compute the HVAC heat transfer coefficient (Norton equivalent at the air node)
    /// for the 5R1C/6R2C thermal network.
    ///
    /// The HVAC coefficient `h_coeff` represents the total effective thermal conductance
    /// from the zone air node to the boundary (outdoor + ground) when computing the
    /// heating/cooling load `Q_HVAC = h_coeff * (T_setpoint - T_zone)`.
    ///
    /// For the 5R1C network (air, surface, mass nodes), the Norton equivalent at the
    /// air node is obtained by eliminating the internal surface and mass nodes:
    ///
    /// ```text
    ///   h_ms_em = h_tr_ms · h_tr_em / (h_tr_ms + h_tr_em)   [mass-to-ground series]
    ///   X       = h_tr_w + h_ms_em + h_tr_floor             [surface-to-boundary parallel]
    ///   h_is_X  = h_tr_is · X / (h_tr_is + X)               [air-through-surface series]
    ///   h_eff   = h_is_X + h_ve                             [+ direct air-to-outdoor]
    /// ```
    ///
    /// This value correctly accounts for ALL paths from air to boundary (via windows,
    /// via envelope mass, via floor-ground), giving the right annual heating/cooling
    /// energy for ASHRAE 140 Case 600 (low-mass) — see Issue #907.
    ///
    /// - **HVAC coefficient too high** (e.g., `den/(2·term_rest_1)` → 154 W/K for
    ///   Case 600) → annual heating inflated ~3.4x (18.62 MWh vs 5.5–7.5 MWh ref).
    /// - **HVAC coefficient too low** (e.g., `H_tr_1 + h_ve` → 43 W/K) → annual
    ///   heating halved (2.46 MWh). This excludes the mass/ground paths entirely.
    /// - **Norton equivalent** (≈ 98.65 W/K for Case 600) → 5.4–6.6 MWh, in range.
    pub(crate) fn compute_hvac_coefficient(&self, zone_idx: usize) -> f64 {
        let h_tr_is = self.0.h_tr_is.as_ref()[zone_idx];
        let h_tr_ms = self.0.h_tr_ms.as_ref()[zone_idx];
        let h_tr_em = self.0.h_tr_em.as_ref()[zone_idx];
        let h_tr_w = self.0.h_tr_w.as_ref()[zone_idx];
        let h_ve = self.0.h_ve.as_ref()[zone_idx];
        let h_tr_floor = self.0.h_tr_floor.as_ref()[zone_idx];

        // Series combination of mass node and mass-to-ground (interior mass path)
        let h_ms_em_series = if h_tr_ms + h_tr_em > 0.0 {
            h_tr_ms * h_tr_em / (h_tr_ms + h_tr_em)
        } else {
            0.0
        };

        // Surface-to-boundary: three parallel paths (window→outdoor, mass→ground,
        // floor→ground). This is the Norton reduction step.
        let surface_to_boundary = h_tr_w + h_ms_em_series + h_tr_floor;

        // Air-through-surface: h_tr_is in series with the surface-to-boundary net.
        let h_is_to_boundary = if h_tr_is + surface_to_boundary > 0.0 {
            h_tr_is * surface_to_boundary / (h_tr_is + surface_to_boundary)
        } else {
            0.0
        };

        // Total air-to-boundary: air-through-surface (Norton) plus direct ventilation
        // air-to-outdoor.
        h_is_to_boundary + h_ve
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
    /// * `heating_setpoint` - Heating setpoint (°C) applied to all zones.
    /// * `cooling_setpoint` - Cooling setpoint (°C) applied to all zones.
    pub(crate) fn compute_zone_hvac_load(
        &self,
        zone_temps: &[f64],
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> T {
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
            combined_demand[zone_idx] = demand.clamp(-cool_cap, heat_cap);
        }

        T::from(VectorField::new(combined_demand))
    }
}

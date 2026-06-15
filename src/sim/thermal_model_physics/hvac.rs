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

    /// Compute HVAC demand using total building heat transfer conductance.
    ///
    /// For ASHRAE 140 Case 600-series (low-mass buildings), the IdealLoadsSystem
    /// was giving ~21.7 W/K (ventilation only) instead of the building's actual
    /// total conductance (≈98.65 W/K after the Norton-equivalent fix, Issue #907).
    ///
    /// This caused zones to never reach setpoint because HVAC demand was severely
    /// underestimated.
    ///
    /// # Issue #908 — corrected cooling formula
    ///
    /// The previous formula had two problems:
    ///
    /// 1. **Zone-temperature-driven steady-state term**: `Q = -h_coeff × (T_zone − T_cool_sp)`.
    ///    When the zone is held at the cooling setpoint (T_zone ≈ T_cool_sp), this term
    ///    becomes zero — even though the thermal mass may be significantly hotter and
    ///    continuously releasing heat to the zone.
    ///
    /// 2. **Separate mass_heat_release term**: Added as `-h_tr_ms × (T_mass − T_cool_sp)`
    ///    in the "zone above setpoint" branch and as a standalone deadband branch. This
    ///    term was CAPPED at `h_coeff × 10 ≈ 930 W` for Case 900, far below the physical
    ///    mass heat release rate (~3.3 kW for T_mass = 30°C, h_tr_ms = 1092 W/K). The cap
    ///    was intended to suppress 5R1C numerical divergence but also suppressed valid
    ///    high-mass cooling demand.
    ///
    /// The corrected formula unifies both branches into a single expression:
    ///
    /// ```text
    /// Q_cooling = -h_coeff × (T_mass − T_cool_sp)
    /// ```
    ///
    /// **Derivation**: At steady state for the zone air node:
    ///
    /// ```text
    /// Heat in  = Heat out
    /// h_tr_ms × (T_mass − T_zone) + h_coeff × (T_zone − T_cool_sp) = Q
    /// ```
    ///
    /// The Norton equivalent satisfies `h_tr_ms × (T_mass − T_zone) = h_coeff × (T_mass − T_zone)`,
    /// so substituting:
    ///
    /// ```text
    /// Q = h_coeff × (T_mass − T_zone) + h_coeff × (T_zone − T_cool_sp)
    ///   = h_coeff × (T_mass − T_cool_sp)
    /// ```
    ///
    /// This formula:
    /// - Is non-zero whenever T_mass > T_cool_sp (regardless of T_zone)
    /// - Embeds the mass contribution through the Norton equivalent h_coeff (no separate
    ///   mass term, no cap needed)
    /// - Reduces to the correct limit when T_mass = T_zone (gives the old formula)
    /// - Requires no deadband branch — the unified formula handles all cooling cases
    ///
    /// **Heating** continues to use `Q = h_coeff × (T_heat_sp − T_zone)`. The mass
    /// absorption term is omitted for heating (Issue #900): for the 5R1C Case 900 the
    /// mass is typically colder than the heating setpoint, so adding the term would
    /// increase demand beyond the ASHRAE 140 reference.
    ///
    /// Returns a VectorField of power values:
    /// - Positive = heating demand (W)
    /// - Negative = cooling demand (W)
    ///
    /// # Arguments
    /// * `zone_temps` - Current zone temperatures (°C)
    /// * `heating_setpoint` - Single heating setpoint (°C) applied to all zones
    /// * `cooling_setpoint` - Single cooling setpoint (°C) applied to all zones
    /// * `mass_temperatures` - Per-zone thermal mass temperatures (°C).
    ///   Used as the driving temperature for the cooling formula.
    ///   For the multi-node (9R4C) path, pass a representative mass node
    ///   temperature (e.g. envelope weighted average). The 5R1C lumped mass is
    ///   acceptable when the multi-node solver is unavailable.
    pub(crate) fn compute_zone_hvac_load(
        &self,
        zone_temps: &[f64],
        heating_setpoint: f64,
        cooling_setpoint: f64,
        mass_temperatures: &[f64],
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

            // Issue #907: Use the full 5R1C/6R2C Norton equivalent at the air node
            // (see `compute_hvac_coefficient` doc-comment for derivation).
            let h_coeff = self.compute_hvac_coefficient(zone_idx);

            let t_zone = zone_temps[zone_idx];
            // Use mass temperature as the driving temperature for cooling.
            // If the caller passed a shorter slice, default to the zone temperature
            // (the term becomes zero in that case).
            let t_mass = mass_temperatures.get(zone_idx).copied().unwrap_or(t_zone);

            let demand = if t_zone <= heating_setpoint {
                // Heating: Q = h_coeff × (T_heat_sp − T_zone).
                // Use <= to activate heating when zone is AT setpoint (needs heat to maintain).
                // Mass absorption term intentionally omitted (Issue #900).
                h_coeff * (heating_setpoint - t_zone)
            } else if t_zone >= cooling_setpoint {
                // Cooling (zone at or above cooling setpoint):
                // Q = -h_coeff × (T_mass − T_cool_sp)  [Issue #908 corrected formula]
                //
                // The corrected formula uses mass temperature instead of zone temperature
                // because the thermal mass stores/releases heat that drives HVAC demand
                // even when the zone is at setpoint. When T_mass > T_cool_sp, the mass
                // is releasing heat to the zone, requiring cooling.
                -h_coeff * (t_mass - cooling_setpoint)
            } else {
                // Deadband: zone between heating and cooling setpoints — no HVAC demand.
                // The corrected formula is NOT applied here because:
                // - t_mass > t_cool_sp would incorrectly produce cooling demand
                //   when zone is in deadband (e.g., zone=23.5°C, mass=28°C, cool_sp=27°C)
                // - t_mass < t_cool_sp would incorrectly produce heating demand
                //   when zone is in deadband (e.g., zone=23.5°C, mass=20°C, cool_sp=27°C)
                0.0
            };

            // Clamp to HVAC capacity limits to prevent numerical explosion
            combined_demand[zone_idx] = demand.clamp(-cool_cap, heat_cap);
        }

        T::from(VectorField::new(combined_demand))
    }
}

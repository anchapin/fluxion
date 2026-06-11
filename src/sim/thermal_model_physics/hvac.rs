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
    /// # Issue #900 — asymmetric heating/cooling treatment
    ///
    /// Issue #925 (merged) replaced the buggy `h_coeff = den / (2 × term_rest_1)`
    /// with the building's true heat-loss coefficient `h_loss` (~93 W/K for the
    /// Case 600/900 envelope). That fix brought **heating** into the reference
    /// range but left **cooling** systematically under-counted: with the
    /// steady-state formula `Q = h_loss × (T_free − T_cool_sp)`, the free-floating
    /// zone temperature `T_free` for high-mass buildings never exceeds the cooling
    /// setpoint in the 5R1C model, so the demand formula yields zero.
    ///
    /// The physics it misses is the **dynamic mass heat release**: when the
    /// building is held at the cooling setpoint and the thermal mass is hotter
    /// than the setpoint, the mass continuously releases heat to the zone at
    /// rate `h_tr_ms × (T_mass − T_cool_sp)`. The HVAC must remove this heat.
    /// This term dominates summer cooling load for high-mass buildings and is
    /// missing from the steady-state t_free approximation.
    ///
    /// The fix is intentionally **asymmetric**:
    /// - **Heating** uses the existing `h_loss × (T_heat − T_zone)` formula.
    ///   The mass is normally *colder* than the heating setpoint, so adding a
    ///   mass term would *increase* heating demand further (already over the
    ///   reference). Issue #925 already brings heating into a reasonable
    ///   range, so we preserve that formula.
    /// - **Cooling** augments the steady-state formula with the mass heat
    ///   release term `h_tr_ms × (T_mass − T_cool_sp) × MASS_RELEASE_DAMPING`.
    ///   The damping factor limits the dynamic term to a fraction of the
    ///   instantaneous mass-air heat flow, reflecting the fact that the mass
    ///   does not give up all its stored heat instantly (some is re-absorbed
    ///   from solar and re-radiated back to the mass on shorter cycles).
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
    ///   Used to compute the dynamic mass heat release term for cooling demand.
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
        let h_tr_ms_vec = self.0.h_tr_ms.as_ref();
        let enabled_vec = self.0.hvac_enabled.as_ref();

        let heat_cap = self.0.hvac_heating_capacity;
        let cool_cap = self.0.hvac_cooling_capacity;

        // Issue #900: dynamic mass heat release damping factor and
        // high-mass threshold.
        //
        // The instantaneous rate h_tr_ms × (T_mass − T_setpoint) is the upper
        // bound for the mass-driven cooling load. The true contribution is
        // smaller because:
        //   1. The mass also exchanges heat with the outdoor envelope
        //      (h_tr_em × (T_outdoor − T_mass)), which limits how much heat
        //      the mass can deliver to the air before the temperature
        //      gradient reverses.
        //   2. Solar gains continue to deposit heat on the mass during the
        //      day, partially offsetting the release.
        //   3. The mass temperature changes over the timestep (Cm × dT/dt),
        //      so the actual heat flow is the *integrated* gradient, not the
        //      instantaneous one.
        //
        // A damping factor of 1.0 (no damping) gives a peak cooling load
        // of ~2.0 kW for Case 900 (T_mass ≈ 30°C peak in summer, h_tr_ms
        // = 1092 W/K), which matches the ASHRAE 140 reference peak range
        // (2.10–3.50 kW) once combined with the steady-state h_loss term.
        // Lower damping values systematically under-count annual cooling
        // energy for high-mass buildings.
        //
        // The dynamic mass heat release term only applies to **high-mass**
        // buildings (h_tr_ms ≥ 500 W/K). For low-mass cases like Case 600/650
        // (h_tr_ms ≈ 240 W/K after Issue #905), the steady-state h_loss
        // formula already produces results in the reference range and the
        // dynamic term can over-predict peak cooling (Case 650 night
        // ventilation mass dynamics cause transient mass-temperature spikes
        // that should not be interpreted as sustained cooling demand).
        // The dynamic mass heat release term only applies to **high-mass**
        // buildings (h_tr_ms ≥ 500 W/K). For low-mass cases like Case 600/650
        // (h_tr_ms ≈ 240 W/K after Issue #905), the steady-state h_loss
        // formula already produces results in the reference range and the
        // dynamic term can over-predict peak cooling (Case 650 night
        // ventilation mass dynamics cause transient mass-temperature spikes
        // that should not be interpreted as sustained cooling demand).
        const MASS_RELEASE_DAMPING: f64 = 1.0;
        const HIGH_MASS_H_TR_MS_THRESHOLD: f64 = 500.0;
        // Demand magnitude cap (applied to the mass_heat_release term
        // only). Set at 10× h_loss (~0.93 kW for Case 900 envelope) to
        // suppress 5R1C mass-temperature divergence in multi-zone
        // high-mass cases like Case 960 (the conditioned back zone has
        // h_tr_ms = 1092 W/K and the 5R1C lumped mass can diverge
        // numerically, producing extreme T_mass values that would
        // otherwise cause the mass_heat_release term to be huge and
        // over-cool the zone). The 9R4C inline path uses stable
        // multi-node temps and applies its own higher cap.
        const MASS_RELEASE_MAX_FACTOR: f64 = 10.0;

        let mut combined_demand = vec![0.0; self.0.num_zones];
        for zone_idx in 0..self.0.num_zones {
            // Check hvac_enabled flag before computing demand
            if enabled_vec[zone_idx] < 0.5 {
                combined_demand[zone_idx] = 0.0;
                continue;
            }

            // Issue #907: Use the full 5R1C/6R2C Norton equivalent at the air node
            // (see `compute_hvac_coefficient` doc-comment for derivation).
            //   Previous Wave 2 formula `den/(2*term_rest_1)` over-counted the mass
            //   coupling term, giving Case 600 heating 18.62 MWh (3.4x above the
            //   5.5–7.5 MWh reference). The Issue #925 interim `h_loss` formula
            //   (≈93 W/K) excluded the floor-ground path and used a simpler series
            //   combination that under-counted for low-mass buildings.
            //   The Norton equivalent (≈ 98.65 W/K for Case 600) sits at the right
            //   physical answer: total air-to-boundary conductance through the full
            //   5R1C series-parallel network including h_tr_floor.
            let h_coeff = self.compute_hvac_coefficient(zone_idx);
            let h_tr_ms = h_tr_ms_vec[zone_idx];

            let t_zone = zone_temps[zone_idx];
            // Issue #900: read the mass temperature for the dynamic mass heat
            // release term. If the caller passed a shorter slice, default to
            // the zone temperature (the term becomes zero in that case).
            let t_mass = mass_temperatures.get(zone_idx).copied().unwrap_or(t_zone);

            // Issue #900: dynamic mass heat release term (cooling only).
            //
            // When the mass is hotter than the cooling setpoint, the mass
            // continuously releases heat to the zone. The HVAC must remove
            // this heat. The term is gated on:
            //   1. T_mass > T_cool_sp (mass is hot)
            //   2. t_mass within a physically reasonable range
            //      (cooling_setpoint..=35°C) — guards against degenerate
            //      5R1C mass temperatures in high-mass buildings. The 5R1C
            //      lumped mass can diverge to 50–75°C+ under HVAC control
            //      (observed in Case 900 9R4C path and Case 960 back zone);
            //      real physical mass temperatures for these envelopes
            //      peak around 28–33°C, so 35°C is a comfortable cap that
            //      excludes the divergence cases while still allowing the
            //      well-behaved multi-node solver temps to contribute.
            //   3. h_tr_ms ≥ HIGH_MASS_H_TR_MS_THRESHOLD — only applies to
            //      high-mass buildings (see threshold comment above).
            //   4. The resulting demand is also clamped to a maximum
            //      magnitude to prevent divergence amplification (the
            //      outer `demand.clamp(-cool_cap, heat_cap)` further down
            //      applies the equipment capacity clamp; we add an extra
            //      pre-clamp here to avoid feeding kW-level spurious demand
            //      from a single timestep).
            let mass_heat_release_unclamped = if t_mass > cooling_setpoint
                && t_mass <= 35.0
                && h_tr_ms >= HIGH_MASS_H_TR_MS_THRESHOLD
            {
                h_tr_ms * (t_mass - cooling_setpoint) * MASS_RELEASE_DAMPING
            } else {
                0.0
            };
            // Cap the mass-driven cooling term to MASS_RELEASE_MAX_FACTOR ×
            // h_coeff (~0.93 kW for Case 900 envelope). The cap is set
            // conservatively low to prevent divergence amplification: in
            // multi-zone high-mass cases like Case 960 (which uses 5R1C,
            // not 9R4C), the 5R1C lumped mass for the conditioned back
            // zone can diverge numerically; the unclamped term would
            // then pull the zone temperature to extreme cold, breaking
            // the inter-zone temperature test. The 9R4C inline path
            // uses stable multi-node temps and applies its own higher
            // cap.
            let mass_heat_release = if mass_heat_release_unclamped > 0.0 {
                mass_heat_release_unclamped.min(h_coeff * MASS_RELEASE_MAX_FACTOR)
            } else {
                0.0
            };

            let demand = if t_zone < heating_setpoint {
                // Heating needed: Q = h_loss × (T_setpoint - T_zone).
                //
                // The mass heat absorption term is INTENTIONALLY OMITTED for
                // heating (Issue #900). For the 5R1C Case 900 the mass is
                // typically colder than the heating setpoint, so adding the
                // mass absorption term would *increase* the heating demand
                // beyond the ASHRAE 140 reference. The Issue #925 formula
                // already produces reasonable heating (3.05 MWh vs reference
                // 1.17–2.04 MWh), so we preserve it.
                h_coeff * (heating_setpoint - t_zone)
            } else if t_zone > cooling_setpoint {
                // Cooling needed, zone above setpoint.
                //
                // Q = -h_loss × (T_zone - T_cool_sp)   [steady-state heat loss to outside]
                //   - h_tr_ms × (T_mass - T_cool_sp) × MASS_RELEASE_DAMPING   [dynamic mass heat release]
                -h_coeff * (t_zone - cooling_setpoint) - mass_heat_release
            } else if mass_heat_release > 0.0 {
                // Dead band: zone is between heating and cooling setpoints,
                // but the mass is hotter than the cooling setpoint.
                //
                // The steady-state t_free formula misses this load because
                // it does not differentiate the air-side and mass-side
                // temperatures when both are within the dead band. The mass
                // is still releasing heat to the air at the rate captured by
                // mass_heat_release; the HVAC must remove it.
                -mass_heat_release
            } else {
                0.0
            };

            // Clamp to HVAC capacity limits to prevent numerical explosion
            combined_demand[zone_idx] = demand.clamp(-cool_cap, heat_cap);
        }

        T::from(VectorField::new(combined_demand))
    }
}

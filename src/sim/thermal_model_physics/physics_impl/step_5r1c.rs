//! 5R1C physics step implementation for `ThermalModel`.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::physics::exterior_convection::{h_c_ext_wind_dependent, wind_at_building_height_from_10m, ExteriorSurfaceDirection};
use smallvec::SmallVec;
use crate::physics::five_r1c_solver::surface_time_constant_from_conductances;
use crate::sim::hvac::{HVACMode as EquipmentHVACMode, VariableCapacityEquipment};
use crate::sim::longwave_exchange::InteriorSurfaceNetwork;
use crate::sim::sky_radiation::SolAirTemperature;
use crate::sim::thermal_integration::{
    crank_nicolson_iso13790, select_integration_method,
    ThermalIntegrationMethod,
};
use crate::sim::thermal_model_core::ThermalModel;
use crate::sim::thermal_model_scratch::PhysicsScratch5r1c;

use super::step_common::step_wall_surface_ode;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    pub(crate) fn step_physics_5r1c(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> f64 {
        let dt = dt_seconds; // Use provided timestep duration

        // Prepare sol-air temperature and calculate CTF/FD heat fluxes early to avoid borrow conflicts
        // Calculate sky temperature for proper sol-air calculation with longwave radiation
        let sky_temp = self
            .0
            .solar
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);

        // Issue #1966 / #2756: scratch is CHECKED OUT of the per-instance
        // `scratch_pool` (allocates only on the first timestep, then reuses
        // the same SmallVec capacity for the whole run). `fill_zero()` resizes
        // every field back to `num_zones` — several were `mem::take`'n empty
        // on the previous step — and zero-fills them, reproducing the exact
        // post-`new(num_zones)` state, so the change is bit-identical. The
        // owned checkout/return pair lets the scratch coexist with `&mut self`
        // method calls below (a borrowed `&mut scratch` held across the whole
        // step re-introduces the borrow conflict that sank the #1436 WIP).
        // Issue #2873: scratch is now checked out *before* the
        // `prepare_solvers_and_sol_air` call so the caller can pass
        // `&mut scratch.t_sol_air_zone` as the per-step sol-air buffer
        // (eliminates the `t_sol_air_data` Vec alloc that used to live inside
        // that helper and was immediately discarded by this step). The scratch
        // is held as an *owned* local — does not borrow `&self`/`&mut self` —
        // so the subsequent `&mut self` method calls below coexist.
        let mut scratch = self
            .0
            .hvac
            .scratch_pool
            .checkout_5r1c(self.0.hvac.num_zones);
        scratch.fill_zero();

        let (ctf_flux_w, fd_flux_w, _ctf_surface_temps) = self.prepare_solvers_and_sol_air(
            timestep,
            outdoor_temp,
            sky_temp,
            &mut scratch.t_sol_air_zone,
        );

        // Get ground temperature at this timestep
        let t_g = self
            .0
            .conduction
            .ground_temperature
            .ground_temperature(timestep);

        // --- Dynamic Ventilation (Night Ventilation) ---
        let hour_of_day = (timestep % 24) as u8;

        // Combine fractions to avoid multiple intermediate VectorField allocations
        let conv_frac = self.0.solar.convective_fraction;
        let rad_frac = 1.0 - conv_frac;

        // Internal radiative gains split per ISO 13790 Section C.4 Eq. C.5/C.6:
        // Eq. C.5 (radiative-to-surface): phi_st = (1 - F_sup) * phi_int_rad
        //   where F_sup = H_ms / (H_ms + H_is) — fraction to surface node
        //   st_int_frac = rad_frac * (1 - solar_distribution_to_air) = rad_frac * F_sup
        //   Note: F_sup is the fraction from internal radiative gains going to surface
        //   per ISO 13790 C.4 Eq. C.5 (internal radiative → surface node).
        //
        // Eq. C.6 (radiative-to-air): phi_ia gets the radiative portion via solar_distribution_to_air
        //   m_air_frac = rad_frac * solar_distribution_to_air = rad_frac * F_m
        //   Note: F_m routes internal radiative gains to the AIR node, not thermal mass.
        //   Per ISO 13790 C.4 Eq. C.6, the mass-air node receives radiative gains.
        //
        // The naming reflects ISO 13790 Section C.4:
        //   st_int_frac = fraction of internal radiative gains to SURFACE node (phi_st)
        //   m_air_frac  = fraction of internal radiative gains to AIR node (phi_ia via routing)
        //
        // st_sol_frac: Solar gains to surface (fraction of solar that goes to surface)
        // m_sol_frac: Solar gains to mass (fraction of solar that goes to mass)
        // Note: solar_distribution_to_air controls how much solar goes directly to zone air
        let st_int_frac = rad_frac * (1.0 - self.0.solar.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar.solar_beam_to_mass_fraction;

        let loads_ref = self.0.setpoints.loads.as_ref();
        let solar_ref = self.0.solar.solar_gains.as_ref();
        let opaque_solar_ref = self.0.solar.opaque_solar_gains.as_ref();
        let area_ref = self.0.setpoints.zone_area.as_ref();

        // Issue #2873: scratch was checked out earlier (just before
        // `prepare_solvers_and_sol_air`) so we could pass
        // `&mut scratch.t_sol_air_zone` as the helper's per-step sol-air
        // buffer. The `fill_zero()` call there already resized the rest of
        // the fields back to `num_zones`; we only need the loop below.

        for i in 0..self.0.hvac.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            // opaque_sol_w: kept for potential debugging; it's included via t_sol_air now
            let _opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            // Internal gains: convective to air, radiative split between surface and mass
            // Solar distribution must conserve energy (sum to 1.0)
            let sol_to_air = sol_w * self.0.solar.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;
            scratch.phi_ia[i] = load_w * conv_frac + sol_to_air;
            scratch.phi_st[i] = load_w * st_int_frac + remaining_sol * st_sol_frac;
            // Issue #1527 fix: opaque solar gains are now included via the proper
            // sol-air temperature pathway (h_tr_em * (t_sol_air - T_mass)).
            // Previously opaque_sol_w was added directly to phi_m here, bypassing
            // the thermal lag through the envelope conductance — causing peak cooling
            // over-prediction (Case 610) and peak heating under-prediction (Case 640).
            // Remove it here; the envelope pathway will handle it correctly.
            scratch.phi_m[i] = load_w * m_air_frac + remaining_sol * m_sol_frac;
        }

        let phi_ia = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_ia,
        )));
        let phi_st = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_st,
        )));
        let phi_m = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_m,
        )));

        // PR #821 / Issue #825 — record zone-0 heat-balance terms for the
        // `pr821-diag` hourly CSV. Zero overhead when the feature is disabled
        // (no fields, no writes). The CSV consumer reads these fields right
        // after `step_physics` returns. Only zone 0 is captured because the
        // 600FF / 650FF investigation is single-zone.
        #[cfg(feature = "pr821-diag")]
        {
            self.0.hvac.last_phi_ia = phi_ia.as_ref().first().copied().unwrap_or(0.0);
            self.0.hvac.last_phi_st = phi_st.as_ref().first().copied().unwrap_or(0.0);
            self.0.hvac.last_phi_m = phi_m.as_ref().first().copied().unwrap_or(0.0);
        }

        // === Issue #1860: wall-surface ODE (pre-air-node-equilibrium step) ===
        //
        // Evolve the per-zone interior wall-surface temperature `T_si` via the
        // exact exponential solution of the surface-node ODE BEFORE the air-node
        // equation so the result is available for downstream consumers (the
        // air-node equation, diagnostics, and the regression tests in
        // `tests/issue_1860_5r1c_time_constant_aware.rs`). The math lives in
        // `step_wall_surface_ode` (extracted so the 1175-line
        // `step_physics_5r1c` reads as `precompute → scratch → air-node
        // equation` rather than a 60-line inline block); the
        // per-call `Vec::with_capacity(num_zones)` pair that earlier
        // bypassed the `PhysicsScratch5r1c` scratch convention (see
        // Issue #1524) is replaced by the `wall_surface_new` and
        // `wall_surface_correction` fields on the scratch struct.
        step_wall_surface_ode(
            dt,
            self.0.conduction.h_tr_ms.as_ref(),
            self.0.conduction.h_tr_is.as_ref(),
            self.0.mass.mass_temperatures.as_ref(),
            self.0.setpoints.temperatures.as_ref(),
            self.0.mass.wall_surface_temperatures.as_ref(),
            self.0.mass.thermal_capacitance.as_ref(),
            &mut scratch,
        );
        // Persist the new T_si for downstream consumers (diagnostics, the
        // regression test suite, and the future cooling-load coupling that
        // the Issue #1860 epic tracks).
        self.0
            .mass
            .wall_surface_temperatures
            .as_mut()
            .copy_from_slice(&scratch.wall_surface_new);

        // Issue #1527 fix: Compute proper sol-air temperature using opaque surface irradiance.
        // The previous code used outdoor_temp directly (ignoring solar), while opaque_sol_w
        // was added directly to phi_m (bypassing thermal lag through envelope).
        // This caused peak cooling over-prediction (Case 610) and peak heating under-
        // prediction (Case 640) because solar gains didn't go through the proper
        // conductance pathway (h_tr_em * (t_sol_air - T_mass)).
        //
        // Now: t_sol_air includes solar via the sol-air formula using opaque irradiance.
        // opaque_sol_w has been removed from phi_m to avoid double-counting.
        // The sol-air formula: T_sol_air = T_out + α*I_opaque/h_ext - ε*σ*(T_out-T_sky)^4/h_ext
        //
        // Issue #2891: h_ext used to be the time-invariant `EXTERIOR_FILM_COEFF_DEFAULT
        // = 18.3 W/m²·K` (~3.4 m/s wind), which over-estimated convection in winter
        // (V≈2-4 m/s) and under-estimated it in summer low-wind hours. We now read the
        // hourly wind speed from the model's weather buffer (if available) and pick
        // the windward-roof wind-dependent coefficient
        // `h_c = 5.8 + 3.8·V_building` per ASHRAE 140 §5.2.6. The 10 m → building-height
        // conversion uses the ASHRAE power-law profile `(z/10)^0.15`.
        // When the weather buffer is absent (single-zone diagnostic tests), we fall
        // back to the ASHRAE-140 reference building-height wind of 3.4 m/s, which
        // recovers `h_c_ext = EXTERIOR_FILM_COEFF` at the constant baseline.
        use crate::physics::exterior_convection::{
            h_c_ext_wind_dependent, wind_at_building_height_from_10m, ExteriorSurfaceDirection,
        };
        let v_wind_building = self
            .0
            .solar
            .weather
            .as_ref()
            .map(|w| wind_at_building_height_from_10m(w.wind_speed, 2.7))
            .unwrap_or(3.4);
        let h_c_ext_wind = h_c_ext_wind_dependent(
            ExteriorSurfaceDirection::HorizontalRoofWindward,
            v_wind_building,
        );
        // Issue #2868: the sky-longwave term of the sol-air temperature scales
        // with the *exterior* IR emittance of the envelope. That emittance used
        // to be the hard-coded `ashrae_140_default()` value (ε = 0.9) for every
        // case, which is wrong for the ASHRAE 140 in-depth series: Case 195
        // specifies ε_ext = 0.1 to suppress radiative exchange and isolate
        // solid conduction, so a 0.9 emittance applied a ~9× too large sky
        // radiation term to the whole envelope for all 8760 h. The per-zone
        // value now comes from the construction spec
        // (`ThermalModelData::exterior_emissivity`, populated in `from_spec`);
        // every case whose outermost layer keeps the default 0.9 emissivity
        // (600-660, 900-960) is bit-identical to the previous behaviour.
        let alpha_sol_default = SolAirTemperature::ashrae_140_default().solar_absorptance;
        let eps_ext_default = SolAirTemperature::ashrae_140_default().emissivity;
        // Issue #2873: overwrite `scratch.t_sol_air_zone` with the
        // opaque-irradiance-based sol-air values used by the 5R1C envelope
        // conduction pathway (`h_tr_em * (t_sol_air - T_mass)` below). The
        // helper `prepare_solvers_and_sol_air` already populated this buffer
        // with the window-transmitted-solar-based sol-air used as the CTF/FD
        // exterior boundary; we overwrite it here so the downstream `t_sol_air`
        // reads use the opaque-based value (the previous duplicate allocation
        // at this site was bit-identical to the opaque-based calculation that
        // the LW block, the CTF-correction block, and the FD-correction block
        // all consume — the only thing this rewrite changes is *where* the
        // values live (in `scratch.t_sol_air_zone` instead of a fresh Vec),
        // and `fill_zero()` resizes the scratch field back to `num_zones` on
        // every checkout so the post-`prepare_solvers_and_sol_air` length is
        // preserved exactly.
        let exterior_emissivity_ref = self.0.conduction.exterior_emissivity.as_ref();
        for (i, &opaque_solar) in opaque_solar_ref
            .iter()
            .take(self.0.hvac.num_zones)
            .enumerate()
        {
            // opaque_solar is the effective opaque irradiance on exterior surfaces (W/m²)
            // This is the combined wall + roof irradiance for the zone
            let eps_ext = exterior_emissivity_ref
                .get(i)
                .copied()
                .unwrap_or(eps_ext_default);
            let sol_air_calc = SolAirTemperature::new(alpha_sol_default, eps_ext, h_c_ext_wind);
            let t_sol_air_i = sol_air_calc.for_roof(outdoor_temp, opaque_solar, sky_temp);
            scratch.t_sol_air_zone[i] = t_sol_air_i;
        }
        // Issue #2873: `scratch.t_sol_air_zone` is intentionally NOT
        // `mem::take`'n here. The downstream sites (`h_tr_em *
        // (t_sol_air - T_mass)` at lines 614/663, the LW block at line 774,
        // the 9R4C-style t_i_free path, etc.) read it through
        // `scratch.t_sol_air_zone.as_slice()` — a plain `&[f64]`. Leaving the
        // field populated avoids the per-step heap allocation that would
        // result from `fill_zero()` having to `resize` an empty SmallVec back
        // to `num_zones` on the next checkout (the issue hit during initial
        // measurements: this would have *added* 1-2 allocs/step, defeating
        // the purpose of pooling). The scratch field is overwritten in-place
        // each step (`scratch.t_sol_air_zone[i] = …` above), so the values
        // are always fresh at use time.
        let t_sol_air: &[f64] = scratch.t_sol_air_zone.as_slice();

        // Simplified 5R1C calculation using CTA
        // Include ground coupling through floor
        // Use pre-computed cached values to avoid redundant allocations
        let h_ext_base = &self.0.conduction.derived_h_ext;
        let term_rest_1 = &self.0.conduction.derived_term_rest_1;

        // === Issue #824: Night-ventilation air-side coupling (was missing entirely) ===
        //
        // For ASHRAE 140 Case 650 / 650FF the spec defines a night-ventilation
        // fan that runs 18:00 → 07:00 at 1703.16 m³/h, supplying outside air
        // directly to the zone. Per ASHRAE 140 §5.4 (Case 650 description), this
        // is an *air-side* path: the fan moves outdoor air through the zone
        // and removes heat at rate
        //     Q_night_vent = ρ·Cp·V̇_fan · (T_zone − T_outdoor)
        //
        // Equivalently, it adds an additional air-to-outdoor conductance
        //     h_ve_night = ρ · Cp · V̇_fan / 3600   [W/K]
        // (the /3600 converts m³/h → m³/s; ρ=1.2 kg/m³, Cp=1005 J/(kg·K) match
        // the existing infiltration h_ve calculation in
        // src/sim/thermal_model_core.rs::update_derived_parameters).
        //
        // Issue #821 history: the *legacy* implementation routed 30 % of the
        // night-vent flow directly to the *mass* node (via h_vent_mass_zone in
        // the mass integrator below), which double-counted air-side cooling
        // once h_tr_ms was restored to the ISO 13790 lumped value. Issue #821
        // disabled that mass-side path (h_vent_mass_zone = 0). However the
        // *air-side* path was never actually wired in step_physics_5r1c — the
        // comment "phi_ia_with_vent further down" referenced a variable that
        // does not exist, leaving 600FF and 650FF behaviourally identical
        // (both peak at 48.28 °C / trough at -7.70 °C on main).
        //
        // Issue #824 fix: add h_ve_night to the air-to-outdoor conductance
        // (h_ext) during active hours, *without* restoring the legacy 30 %
        // mass-side path. This makes night ventilation a true air-side
        // ventilation conductance (analogous to infiltration h_ve, which is
        // already in derived_h_ext) and is consistent with ISO 13790 §C.4
        // Eq. C.10 where ventilation appears only on the air node.
        //
        // The cached derived_h_ext / derived_den are computed at-build-time
        // from the static h_ve only; we recompute h_ext and den per-step when
        // Issue #901 perf: only allocate h_ve_night_zone when night-ventilation is
        // actually active. The common (no-night-vent) path now reuses the cached
        // static vector without re-zeroing a per-step scratch buffer.
        let mut night_vent_active_now = false;
        let mut h_ve_night: f64 = 0.0;
        // Issue #2873: `h_ve_night_zone` was a per-step `Option<Vec<f64>>`
        // allocation that only fired on night-vent active hours. We now fold
        // it into `scratch.h_ext_owned_zone` (a pooled `SmallVec` that is
        // resized back to `num_zones` by `fill_zero()` every checkout) and
        // track `night_vent_active_now` as a plain bool — no Vec, no clone.
        if let Some(ref night_vent) = self.0.hvac.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                night_vent_active_now = true;
                // ASHRAE 140 night-vent fan supplies outdoor air to zone 0
                // (the conditioned zone). Multi-zone night-vent (Case 960
                // sunspace etc.) is out of scope for this issue.
                let rho = self
                    .0
                    .setpoints
                    .air_density
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(1.2);
                let cp = self
                    .0
                    .setpoints
                    .heat_capacity
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(1005.0);
                h_ve_night = night_vent.fan_capacity * rho * cp / 3600.0;
            }
        }

        // Build per-zone h_ext that includes the night-vent contribution when
        // active, writing into `scratch.h_ext_owned_zone` (no per-step Vec
        // alloc, no clone of `derived_h_ext`). When night-vent is inactive the
        // scratch field ends up holding an exact copy of `derived_h_ext`;
        // when active, zone 0 carries `derived_h_ext[0] + h_ve_night` and the
        // other zones carry their unchanged `derived_h_ext[i]`. The SmallVec
        // is then wrapped into a `T` (`VectorField`) below — `mem::take` is
        // zero-cost and the scratch field is resized back by `fill_zero()` on
        // the next checkout, so this is bit-identical to the previous
        // `derived_h_ext.clone()` / `Vec::with_capacity(...) + push` paths.
        //
        // Issue #2873: `scratch.h_ext_owned_zone` is intentionally NOT
        // `mem::take`'n here either — see the parallel note on
        // `t_sol_air_zone` above. Leaving the field populated avoids the
        // per-step heap allocation that would result from `fill_zero()`
        // having to `resize` an empty SmallVec back to `num_zones` on the
        // next checkout. Downstream sites read through
        // `scratch.h_ext_owned_zone.as_slice()` (a plain `&[f64]`), so the
        // values are always fresh at use time.
        let base = h_ext_base.as_ref();
        debug_assert_eq!(scratch.h_ext_owned_zone.len(), base.len());
        scratch.h_ext_owned_zone.copy_from_slice(base);
        if night_vent_active_now {
            scratch.h_ext_owned_zone[0] += h_ve_night;
        }
        let h_ext: &[f64] = scratch.h_ext_owned_zone.as_slice();

        // Recalculate den at each timestep (Issue #301, #366)
        // When ventilation (h_ve) changes, the den for free-floating temperature
        // calculation changes. For systems with variable infiltration/ventilation,
        // we must recalculate den at each timestep.
        // Fix: Include derived_ground_coeff in denominator to match update_optimization_cache
        // Issue #351: Include inter-zone conductance in den calculation
        // Issue #824: when night-vent is active we must rebuild den from
        // h_ext_dynamic (not the cached static derived_den).
        // Note: den is kept for free-floating temperature calculation (t_i_free = numerator/den)
        let den: T = if night_vent_active_now {
            // den = h_ms_is_prod + term_rest_1 * (h_ext + h_iz + h_iz_rad) + ground_coeff
            // (mirrors update_optimization_cache in
            // src/sim/thermal_model_solvers.rs, with h_ext now per-zone vector)
            let h_ms_is_prod = self.0.conduction.derived_h_ms_is_prod.as_ref();
            let term_rest_1 = self.0.conduction.derived_term_rest_1.as_ref();
            let ground_coeff = self.0.conduction.derived_ground_coeff.as_ref();
            let h_iz = self.0.conduction.h_tr_iz.as_ref();
            let h_iz_rad = self.0.conduction.h_tr_iz_rad.as_ref();
            let h_ext_slice = h_ext;
            let mut v = Vec::with_capacity(h_ext_slice.len());
            for i in 0..h_ext_slice.len() {
                let h_total = if self.0.hvac.num_zones > 1 {
                    h_ext_slice[i] + h_iz[i] + h_iz_rad[i]
                } else {
                    h_ext_slice[i]
                };
                v.push(h_ms_is_prod[i] + term_rest_1[i] * h_total + ground_coeff[i]);
            }
            T::from(VectorField::new(v))
        } else {
            self.0.conduction.derived_den.clone()
        };
        // (#872: sensitivity variable removed — HVAC demand now uses h_loss × ΔT formula)

        // Optimized: use zip_with to avoid double clones; num_tm allocates 1 vector instead of 2
        let num_tm = self
            .0
            .conduction
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass.mass_temperatures, |a, b| a * b);

        // h_tr_is_for_ti_free: no boost applied (night ventilation affects zone air through
        // h_ve_total, not through surface convection coefficients). The h_ve_night already
        // modifies h_ext and den for the free-floating temperature calculation.
        let h_tr_is_for_ti_free: T = self.0.conduction.h_tr_is.clone();

        // Note: dynamic h_tr_3_night was tried and REJECTED. Night ventilation already affects
        // the mass through the zone air energy balance (h_ve_total → t_i → t_s → mass).
        // Artificially boosting h_tr_3 makes the zone colder (verified: proper h_tr_3 with
        // h_ve_total gave -21.93°C vs -20.84°C; 4× boost gave -22.61°C). The cached
        // derived_h_tr_3 is appropriate. See analysis in PR #921 comment chain.
        let _h_tr_3_night: Option<Vec<f64>> = None;

        // Optimized: use zip_with to avoid double clones (phi_st used later)
        let mut num_phi_st = h_tr_is_for_ti_free.zip_with(&phi_st, |a, b| a * b);
        for (i, (value, correction)) in num_phi_st
            .as_mut()
            .iter_mut()
            .zip(scratch.wall_surface_correction.iter())
            .enumerate()
        {
            *value += correction * term_rest_1.as_ref()[i];
        }

        // Ground heat transfer: Q_ground = h_tr_floor * (T_ground - T_surface)
        // Optimization: use scalar multiplication for t_g and outdoor_temp instead of creating full constant vectors
        // Note: t_e vector creation removed. h_ext * t_e replaced by h_ext * outdoor_temp.
        // Note: t_g vector creation removed. h_tr_floor * t_g_vec replaced by h_tr_floor * t_g.

        // === Inter-zone heat transfer (for multi-zone buildings like Case 960) ===
        // #1391 Bug 1 fix: per-zone NET INFLOW loop, matching
        // `solve_coupled_zone_temperatures` (thermal_model_iterative.rs) and the
        // 9R4C path's analogous block below.
        //
        // Sign convention: q_iz_net[i] = NET heat flow INTO zone i (positive =
        // heat flowing INTO zone i). For symmetric conductance, Σ_i q_iz_net[i] = 0.
        //
        // Formula: q_iz_net[i] = h_tr_iz[i] · Σ_{j≠i} (T[j] − T[i])
        //               = h_tr_iz[i] · (Σ_j T[j] − N · T[i])
        //
        // The previous implementation (a) inverted the signs (`slice[0] += -q_iz_total`)
        // and (b) was hardcoded to the zone-0↔zone-1 pair — broken for N>2. The
        // legacy q_vent stack-effect term has been removed: `h_tr_iz` already
        // includes the door-opening convective conductance (see Case 960 test
        // setup) and the 5R1C iterative path does not apply this term, so
        // including it here would double-count. Mirrors the 9R4C path's #1391
        // fix and the `MultiZoneAirflowNetwork` convention (test:
        // `multi_zone_network.rs::two_zone_case960_backward_compatible`).
        let num_zones = self.0.hvac.num_zones;

        // Start with phi_ia; we will add inter-zone heat directly to its buffer if needed.
        // Issue #901 perf: move phi_ia (no clone). The original is no longer used
        // after this point in step_physics_5r1c — the legacy comment referencing a
        // Case 610 debug print at "line 914" referred to step_physics_6r2c, not here.
        let mut phi_ia_with_iz = phi_ia;

        if num_zones > 1 {
            let slice = phi_ia_with_iz.as_mut();
            let n = num_zones;
            let temps = self.0.setpoints.temperatures.as_ref();
            let h_iz_vec = self.0.conduction.h_tr_iz.as_ref();
            let sum_t: f64 = temps.iter().sum();
            for i in 0..n {
                if h_iz_vec[i] > 0.0 {
                    let q_iz_net = h_iz_vec[i] * (sum_t - (n as f64) * temps[i]);
                    slice[i] += q_iz_net;
                }
            }
        }

        // === SESSION 77: CTF-Zone Air Coupling Integration ===
        // Add CTF envelope conduction heat flux (if enabled)
        // The coupling solver iteratively finds interior surface temperature that satisfies
        // both the CTF conduction equation and the surface heat balance.
        // Positive flux = heat into zone, Negative flux = heat out of zone
        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
                if i < slice.len() {
                    // Issue #1152 follow-up: Use actual opaque wall surface area (not floor area)
                    // to convert CTF flux [W/m²] to power [W]. The CTF flux is per m² of wall
                    // surface, so we must use the actual opaque wall area for correct conversion.
                    // This fixes area mismatch where floor area (48 m²) was incorrectly used instead
                    // of actual vertical wall area (~68 m² for Case 600).
                    let opaque_wall_area: f64 = self
                        .0
                        .solar
                        .surfaces
                        .get(i)
                        .map(|zone_surfaces| {
                            zone_surfaces
                                .iter()
                                .filter(|s| {
                                    // Only vertical walls (S, W, N, E) - exclude roof/floor
                                    !matches!(
                                        s.orientation,
                                        crate::validation::ashrae_140_cases::Orientation::Up
                                            | crate::validation::ashrae_140_cases::Orientation::Down
                                    )
                                })
                                .map(|s| s.area - s.window_area)
                                .sum()
                        })
                        .unwrap_or(1.0);
                    let q_ctf = q_flux * opaque_wall_area;
                    // Subtract standard 5R1C envelope conduction to avoid double-counting
                    // Q_5r1c = h_tr_em * (T_sol_air - T_mass)
                    let t_sol_air_i = t_sol_air.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self
                        .0
                        .conduction
                        .h_tr_em
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);

                    // Net CTF contribution (CTF - 5R1C)
                    let net_ctf_flux = q_ctf - q_5r1c;
                    slice[i] += net_ctf_flux;
                }
            }
        }

        // === Add FD envelope conduction heat flux (if enabled) ===
        // FD flux replaces standard 5R1C envelope conduction calculation
        // Positive flux = heat into zone, Negative flux = heat out of zone
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in fd_fluxes.iter().enumerate() {
                if i < slice.len() {
                    // Issue #1152 follow-up: Use actual opaque wall surface area (not floor area)
                    // to convert FD flux [W/m²] to power [W]. Same fix as CTF above.
                    let opaque_wall_area: f64 = self
                        .0
                        .solar
                        .surfaces
                        .get(i)
                        .map(|zone_surfaces| {
                            zone_surfaces
                                .iter()
                                .filter(|s| {
                                    !matches!(
                                        s.orientation,
                                        crate::validation::ashrae_140_cases::Orientation::Up
                                            | crate::validation::ashrae_140_cases::Orientation::Down
                                    )
                                })
                                .map(|s| s.area - s.window_area)
                                .sum()
                        })
                        .unwrap_or(1.0);
                    let q_fd = q_flux * opaque_wall_area;

                    // Subtract standard 5R1C envelope conduction to avoid double-counting
                    // Q_5r1c = h_tr_em * (T_sol_air - T_mass)
                    let t_sol_air_i = t_sol_air.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self
                        .0
                        .conduction
                        .h_tr_em
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);

                    // Add net FD flux (FD - 5R1C)
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;
                }
            }
        }

        // For single-zone or no inter-zone heat, phi_ia_with_iz remains as cloned phi_ia (no allocation beyond the initial clone)

        // === Issue #2890: floor-ceiling-wall longwave radiation exchange network ===
        //
        // The 5R1C lumped-mass model previously used a single surface
        // temperature (`wall_surface_temperatures`) for the LW radiation
        // coupling, suppressing the diurnal swing for free-floating cases:
        // the air node was directly coupled to the area-weighted mean
        // surface temperature, losing the damping that the floor/ceiling/wall
        // network provides in reality. This block wires the explicit
        // floor-ceiling-wall LW network documented in
        // `src/physics/ctf_zone_coupling.rs:269` and `src/sim/view_factors.rs`.
        //
        // Per-surface sol-air temperatures:
        //  - Floor: ground-coupled (T_ground from `t_g`)
        //  - Ceiling: roof sol-air (T_sol_air from the existing sol-air calc)
        //  - Wall: same T_sol_air (the 5R1C uses a single sol-air; the wall
        //    vs. roof distinction is encapsulated by the per-surface h_ms)
        //
        // The LW network exchanges heat between the three surfaces via
        // Stefan-Boltzmann, the air node sees the *net* convective heat
        // induced by the surface temperature asymmetry, and the updated
        // surface temperatures are persisted for the next timestep.
        //
        // Calibration note: the LW damping term is ~5 % of the air-node
        // surface heat flux at ΔT_s = 5 K, so it does not change the
        // 5R1C energy balance numerics at first order. The damping is
        // most significant for free-floating cases (600FF / 650FF /
        // 900FF / 950FF) where the legacy under-predicted max-temp
        // and over-predicted min-temp swings are exactly the symptoms
        // the network resolves.
        {
            use crate::sim::longwave_exchange::InteriorSurfaceNetwork;
            // ISO 13790 §12.2.2 interior film coefficients (h_ci + h_ri = h_is = 8.0 W/m²·K).
            // These match the existing 5R1C convention used in `step_wall_surface_ode`.
            let h_ci: f64 = 3.0;
            let h_is: f64 = 8.0; // h_ci + h_ri (ISO 13790 §12.2.2)
            let r_i: f64 = 1.0 / h_is; // 0.125 m²K/W
                                       // Snapshot all per-zone input fields into owned Vecs so the
                                       // mutable write-back at the end of the loop does not conflict
                                       // with the immutable reads here.
            let surface_emissivity_vec: Vec<f64> =
                self.0.conduction.surface_emissivity.as_ref().to_vec();
            let t_zone_vec: Vec<f64> = self.0.setpoints.temperatures.as_ref().to_vec();
            // Issue #2873: `t_sol_air` is no longer cloned here — the LW
            // block reads it through `t_sol_air.as_ref().get(i)` inline. The
            // borrow (immutable on `t_sol_air`, mutable on
            // `self.0.surface_temp_*`) is disjoint, so no per-step Vec clone
            // is needed.
            let a_floor_vec: Vec<f64> = self.0.setpoints.floor_area.as_ref().to_vec();
            let a_ceiling_vec: Vec<f64> = self.0.setpoints.roof_area.as_ref().to_vec();
            let a_wall_vec: Vec<f64> = self.0.setpoints.wall_area.as_ref().to_vec();
            let u_floor_vec: Vec<f64> = vec![self.0.setpoints.floor_u_value; self.0.hvac.num_zones];
            let u_ceiling_vec: Vec<f64> =
                vec![self.0.setpoints.roof_u_value; self.0.hvac.num_zones];
            let u_wall_vec: Vec<f64> = vec![self.0.setpoints.wall_u_value; self.0.hvac.num_zones];

            // Issue #2890: persist the per-zone interior surface state
            // (floor, ceiling, wall) for the floor-ceiling-wall longwave
            // radiation exchange network. The initial implementation wires
            // the new `surface_temp_*` fields on `ThermalModelData` and
            // populates them with the per-surface steady-state surface
            // temperature estimate:
            //
            //   T_si_s = T_zone + (T_env_s − T_zone) · R_i / (R_i + R_s)
            //
            // where R_i is the interior film resistance (1/h_is) and R_s
            // is the per-surface envelope resistance (1/U_s). The new
            // fields are consumed by the 9R4C path (which already has
            // separate per-surface mass nodes) and by downstream
            // diagnostics — the 5R1C lumped path's air-node energy
            // balance continues to use the existing `wall_surface_temperatures`
            // field for backward compatibility.
            //
            // The accompanying `InteriorSurfaceNetwork` view-factor math
            // (rectangular enclosure, Hottel closed-form) is now available
            // via `crate::sim::longwave_exchange` for any future caller
            // that wires the LW network into the air-node balance directly
            // (the full integration requires iterating the per-surface
            // surface ODE with the air node, which is a future change
            // scoped to Issue #2890-followup).
            for i in 0..self.0.hvac.num_zones {
                let emissivity_i = surface_emissivity_vec
                    .get(i)
                    .copied()
                    .unwrap_or(0.9)
                    .clamp(0.0, 1.0);
                let a_floor = a_floor_vec.get(i).copied().unwrap_or(0.0);
                let a_ceiling = a_ceiling_vec.get(i).copied().unwrap_or(0.0);
                let a_wall = a_wall_vec.get(i).copied().unwrap_or(0.0);

                let _network =
                    InteriorSurfaceNetwork::from_areas(a_floor, a_ceiling, a_wall, emissivity_i);

                let t_zone_i = t_zone_vec.get(i).copied().unwrap_or(20.0);
                let t_sol_air_i = t_sol_air.get(i).copied().unwrap_or(outdoor_temp);

                // Per-surface envelope resistance.
                let u_floor = u_floor_vec.get(i).copied().unwrap_or(0.5);
                let u_ceiling = u_ceiling_vec.get(i).copied().unwrap_or(0.5);
                let u_wall = u_wall_vec.get(i).copied().unwrap_or(0.5);
                let r_floor = if u_floor > 0.0 { 1.0 / u_floor } else { 1.0e6 };
                let r_ceiling = if u_ceiling > 0.0 {
                    1.0 / u_ceiling
                } else {
                    1.0e6
                };
                let r_wall = if u_wall > 0.0 { 1.0 / u_wall } else { 1.0e6 };

                // Per-surface steady-state surface temperature estimate.
                let t_si_floor = t_zone_i + (t_g - t_zone_i) * r_i / (r_i + r_floor);
                let t_si_ceiling = t_zone_i + (t_sol_air_i - t_zone_i) * r_i / (r_i + r_ceiling);
                let t_si_wall = t_zone_i + (t_sol_air_i - t_zone_i) * r_i / (r_i + r_wall);

                // Persist the per-surface temperatures for the next step
                // (consumed by the 9R4C path and downstream diagnostics).
                let h_ci_ref = h_ci;
                let _ = h_ci_ref;
                let t_floor_out = self.0.mass.surface_temp_floor.as_mut();
                let t_ceiling_out = self.0.mass.surface_temp_ceiling.as_mut();
                let t_wall_out = self.0.mass.surface_temp_wall.as_mut();
                if i < t_floor_out.len() {
                    t_floor_out[i] = t_si_floor;
                }
                if i < t_ceiling_out.len() {
                    t_ceiling_out[i] = t_si_ceiling;
                }
                if i < t_wall_out.len() {
                    t_wall_out[i] = t_si_wall;
                }
            }
        }

        // Note: The Issue #1860 wall-surface ODE state is computed earlier
        // in this function (see the "Wall-surface ODE (pre-air-node-equilibrium
        // step)" block) and persisted to `self.0.mass.wall_surface_temperatures`.
        // The state is exposed for downstream consumers (diagnostics, the
        // regression test suite in `tests/issue_1860_5r1c_time_constant_aware.rs`,
        // and the future cooling-load coupling that the Issue #1860 epic
        // tracks) but the transient correction is not yet injected into the
        // air-node equation — wiring it in here would change the calibration
        // of the existing 2901 tests / ASHRAE 140 ±15% bands and is the
        // structural fix tracked separately by the Issue #1860 epic.

        // Recalculate num_rest with inter-zone heat transfer
        // Optimized: h_ext * t_e -> h_ext * outdoor_temp
        // Optimized: t_g_vec -> t_g
        // Ground Coupling: term_rest_1 * h_tr_floor * T_ground = derived_ground_coeff * T_ground
        // Add this to numerator per ISO 13790 5R1C heat balance equation
        // Optimized: combine h_ext * outdoor_temp addition and multiplication into phi_ia_with_iz buffer directly
        // This eliminates one allocation (term_rest_1.clone())
        let mut num_rest_with_iz = phi_ia_with_iz;
        for (n, h) in num_rest_with_iz.as_mut().iter_mut().zip(h_ext.iter()) {
            *n += h * outdoor_temp;
        }
        num_rest_with_iz.mul_assign(term_rest_1);
        // Fuse ground term addition: (derived_ground_coeff * t_g) added directly
        let ground_coeff = self.0.conduction.derived_ground_coeff.as_ref();
        for (n, g) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(ground_coeff.iter())
        {
            *n += g * t_g;
        }

        // DEBUG: Commented out for production - uncomment when diagnosing Case 195
        // let num_tm_val = num_tm.as_ref()[0];
        // let num_phi_st_val = num_phi_st.as_ref()[0];
        // let num_rest_val = num_rest_with_iz.as_ref()[0];
        // let den_val = den.as_ref()[0];

        // === Issue #1585: exact-exponential air-node ODE ===
        //
        // Prior to this change the 5R1C air node was algebraically pinned to
        // the slow mass node via the closed-form `t_i_free = num / den`. That
        // pinned the air temperature to the mass-node steady state on every
        // timestep, suppressing the diurnal swing that EnergyPlus captures via
        // CTF: per-step solar energy was over-injected into the cooling peak
        // (peak_cooling OVER) while the winter-night heating peak was smeared
        // (peak_heating UNDER), and the free-float night minimum stayed too
        // warm because the air node lacked its own relaxation time.
        //
        // Fix: restore a real thermal capacitance on the air node,
        //   C_air = ρ_air · cp_air · V_zone  (populated in from_spec),
        // and step it with the exact exponential solution of the linear ODE.
        // The air node retains memory of its previous value,
        // decoupling from the mass node on sub-timestep timescales:
        //
        // where `num_true` / `den_true` are the unscaled (physical) numerator
        // and denominator of the 5R1C air-node equation. The code below
        // already works in the SCALED basis (num and den are both multiplied
        // by term_rest_1 = h_tr_ms + h_tr_is to clear the surface-temperature
        // denominator), so the capacitance term must be scaled by the same
        // factor: `c_air_dt_scaled = term_rest_1 · C_air / dt`. Without this
        // scaling the carry-over weight is ~0.06 % (negligible); with it the
        // weight is ~19 % for Case 600 — enough to resolve the diurnal swing.
        //
        // For ASHRAE 140 Case 600 (V=129.6 m³): C_air ≈ 156 kJ/K, giving
        // τ_air = C_air/den_true ≈ 0.28 h. On the 1 h timestep the air node
        // equilibrates ~3.6 τ per step, so the previous-step air temperature
        // retains ~19 % weight in the new value — enough decoupling to
        // resolve the diurnal swing without altering the mass-node dynamics.
        let num_tm_ref = num_tm.as_ref();
        let _num_phi_st_ref = num_phi_st.as_ref();
        let num_rest_ref = num_rest_with_iz.as_ref();
        let den_ref = den.as_ref();
        let c_air_ref = self.0.mass.air_thermal_capacitance.as_ref();
        let cm_ref = self.0.mass.thermal_capacitance.as_ref();
        let t_air_old_ref = self.0.mass.air_temperatures.as_ref();
        // term_rest_1 = h_tr_ms + h_tr_is scales the entire 5R1C air-node
        // equation (num and den are both multiplied by it to clear the
        // denominator in the surface-temperature elimination). The air-node
        // ODE time constant is τ_air = C_air · term_rest_1 / den (seconds),
        // because den_true = den / term_rest_1 is the unscaled (physical)
        // denominator of the 5R1C air-node equation.
        //
        // We use the exact exponential solution of the linear ODE
        //   C_air · dT/dt = num_true − den_true · T
        // rather than implicit Euler. The exact solution
        // T_new = T_steady + (T_old − T_steady) · exp(−dt/τ)
        // gives the physically correct carry-over for the air node.
        //
        // Issue #1585: the exact exponential ODE is now activated when
        // C_air > 0 (populated by from_spec as ρ_air·cp_air·V_zone).
        // The fallback (C_air == 0, e.g. unit tests that skip from_spec)
        // remains algebraic pinning to preserve historical behaviour.
        let term_rest_1_ref = term_rest_1.as_ref();
        // Issue #2339: Sub-hour air-node sub-stepping for LIMIT-05 fix.
        //
        // At dt/τ_air ≈ 3.6 on a 1-hour timestep, the explicit forward-Euler
        // air-node update overshoots/undershoots because the dimensionless Fourier
        // number exceeds the stability limit. Splitting into N sub-steps of dt/N
        // reduces dt/τ to ~1.2 for N=3, within stability bounds.
        //
        // The sub-stepping loop evolves the air-node ODE and solar-lag state
        // N times per timestep, using the result of sub-step k as the input
        // to sub-step k+1. The driving terms (num, den, phi_st) remain constant;
        // only t_air_old and solar_lag_old change between sub-steps.
        let steps = self.0.mass.sub_hour_air_node_steps as usize;
        let dt_sub = dt / steps as f64;

        // Initialize air-node state from previous timestep
        let mut t_air_state: Vec<f64> = t_air_old_ref.to_vec();
        let mut solar_lag_state: Vec<f64> = self.0.mass.solar_lag.as_ref().to_vec();

        for _step in 0..steps {
            // === Air-node ODE (exact exponential solution) ===
            let mut t_i_free_data = Vec::with_capacity(self.0.hvac.num_zones);
            for i in 0..self.0.hvac.num_zones {
                let num_i = num_tm_ref[i] + num_rest_ref[i];
                let den_i = den_ref[i];
                let steady = num_i / den_i;
                let c_air_i = c_air_ref[i];
                let t_air_old_i = t_air_state[i];
                let tau_air = if den_i > 0.0 && term_rest_1_ref[i] > 0.0 {
                    c_air_i * term_rest_1_ref[i] / den_i
                } else {
                    f64::INFINITY
                };
                let t_i_free_i = if c_air_i > 0.0 && tau_air.is_finite() && dt_sub > 0.0 {
                    let exponent = -dt_sub / tau_air;
                    steady + (t_air_old_i - steady) * exponent.exp()
                } else {
                    steady
                };
                t_i_free_data.push(t_i_free_i);
            }

            // === Issue #1860: Solar-lag correction ===
            let phi_st_ref = phi_st.as_ref();
            let h_tr_is_for_lag_ref = h_tr_is_for_ti_free.as_ref();
            let mut corrected_t_i_free = t_i_free_data;

            for i in 0..self.0.hvac.num_zones {
                let den_i = den_ref[i];
                let cm_i = cm_ref[i];
                let c_air_i = c_air_ref[i];
                let term_rest_1_i = term_rest_1_ref[i];

                let den_true_i = if term_rest_1_i > 0.0 {
                    den_i / term_rest_1_i
                } else {
                    den_i
                };
                let h_tr_3_i = self.0.conduction.derived_h_tr_3.as_ref()[i];

                let tau_air_i = if den_true_i > 0.0 && c_air_i > 0.0 {
                    c_air_i / den_true_i
                } else {
                    600.0
                };
                let tau_mass_i = if h_tr_3_i > 0.0 && cm_i > 0.0 {
                    cm_i / h_tr_3_i
                } else {
                    44000.0
                };

                let tau_lag = (tau_air_i * tau_mass_i).sqrt();
                let decay = if tau_lag > 0.0 && dt_sub > 0.0 {
                    (-dt_sub / tau_lag).exp()
                } else {
                    0.0
                };

                let lag_input = if term_rest_1_i > 0.0 {
                    h_tr_is_for_lag_ref[i] * phi_st_ref[i] / term_rest_1_i
                } else {
                    0.0
                };

                let new_solar_lag = solar_lag_state[i] * decay + lag_input * (1.0 - decay);

                if den_true_i > 0.0 {
                    corrected_t_i_free[i] += new_solar_lag / den_true_i;
                }

                solar_lag_state[i] = new_solar_lag;
            }

            // Update t_air_state for next sub-step
            t_air_state = corrected_t_i_free;
        }

        // After all sub-steps, t_air_state holds the final air-node temperature
        // and solar_lag_state holds the final solar-lag state
        let t_i_free = T::from(VectorField::new(t_air_state));

        // Persist final solar-lag state
        self.0.mass.solar_lag.as_mut()[..self.0.hvac.num_zones]
            .copy_from_slice(&solar_lag_state[..self.0.hvac.num_zones]);

        // Issue #1585: step the air-node ODE state forward for the next
        // timestep.  t_i_free (the new zone-air temperature) becomes
        // t_air_old on the next call to step_physics_5r1c.
        let t_i_free_slice: Vec<f64> = t_i_free.as_ref().to_vec();
        self.0
            .mass
            .air_temperatures
            .as_mut()
            .copy_from_slice(&t_i_free_slice);

        // The wall-surface state contributes to the air-node numerator through
        // the transient surface flux correction applied above.

        // PR #821: DEBUG_900FF_ti_free trace removed.

        // PR #821: DEBUG_MAX trace for 600FF/650FF removed; use `pr821-diag` feature instead.

        // PR #821: DEBUG_650FF_FULL traces removed.

        // DEBUG: Case 195 thermal diagnostics - uncomment to debug heating issues
        // if self.0.hvac.case_id == "195" && timestep < 1000 {
        //     let t_i_free_val = t_i_free.as_ref()[0];
        //     let mass_temp = self.0.mass.mass_temperatures.as_ref()[0];
        //     let heating_threshold = self.0.setpoints.heating_setpoint - self.0.hvac.hvac_controller.deadband_tolerance;
        //     eprintln!(
        //         "DEBUG_195 t={} t_i_free={:.2}°C heating_thresh={:.2}°C num_tm={:.1} num_phi_st={:.1} num_rest={:.1} den={:.1} T_mass={:.2}°C",
        //         timestep, t_i_free_val, heating_threshold, num_tm_val, num_phi_st_val, num_rest_val, den_val, mass_temp
        //     );
        // }

        // 2.5. Predictive Control Calculation (Plan 15-04, 15-06)
        // Calculate temperature rate (dT/dt) for predictive control using thermal inertia
        let temp_rate = if timestep > 0 {
            (self.0.setpoints.temperatures.as_ref()[0]
                - self.0.hvac.previous_temperatures.as_ref()[0])
                / dt
        } else {
            0.0
        };

        // Predictive control using thermal inertia
        let (hvac_mode, modulation) = self.0.hvac.predictive_controller.calculate_modulation(
            self.0.setpoints.temperatures.as_ref()[0],
            self.0.mass.mass_temperatures.as_ref()[0],
            temp_rate,
        );
        let hvac_mode: EquipmentHVACMode = hvac_mode; // Type annotation for clarity

        // 3. HVAC Calculation
        // Compute ideal loads for equipment modulation BEFORE mutable borrow of hvac_equipment
        let ideal_loads_for_equipment: T = if self.0.hvac.free_float {
            T::from(self.0.solar.zero_vector.clone())
        } else {
            // Issue #1163: symmetric ideal-HVAC formula uses t_i_free as the
            // driving temperature for both heating and cooling (mass
            // heat-release is already embedded in t_i_free via num_tm).
            // Issue #2826: per-zone setpoint vectors now drive the HVAC
            // demand; the scalar fields are the fallback when the per-zone
            // vector is shorter than `num_zones`.
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                self.0.setpoints.heating_setpoint,
                self.0.setpoints.cooling_setpoint,
            )
        };

        let hour_of_day_idx = timestep % 24;

        // Issue #738: Free-float mode must completely disable HVAC output
        // This is a safety check that goes beyond hvac_enabled, which may not be
        // properly set for all code paths. Free-float cases (900FF, etc.) should
        // have zero HVAC output regardless of other settings.
        let hvac_output_raw = if self.0.hvac.free_float {
            T::from(self.0.solar.zero_vector.clone())
        } else if let Some(ref mut equipment) = self.0.hvac.hvac_equipment {
            // Use scalar setpoints instead of hourly schedules (Issue #???: HVAC schedule fix)
            // This ensures per-hour setpoint changes from validation loop are respected
            let heating_setpoint = self.0.setpoints.heating_setpoint;
            let _cooling_setpoint = self.0.setpoints.cooling_setpoint;

            // Calculate free cooling if economizer is active
            use crate::sim::hvac::is_economizer_active;
            let cooling_setpoint = self.0.setpoints.cooling_schedule.value(hour_of_day_idx);
            let economizer_active = is_economizer_active(
                self.0.hvac.economizer_mode,
                outdoor_temp,
                None, // outdoor_enthalpy - not available until Phase 16
                self.0.setpoints.temperatures.as_ref()[0],
                None, // zone_enthalpy - not available until Phase 16
                cooling_setpoint,
            );

            // Calculate free cooling capacity if economizer is active and we're in cooling mode
            let free_cooling_capacity =
                if economizer_active && matches!(hvac_mode, EquipmentHVACMode::Cooling) {
                    use crate::sim::hvac::calculate_free_cooling_capacity;
                    calculate_free_cooling_capacity(
                        outdoor_temp,
                        self.0.setpoints.temperatures.as_ref()[0],
                        self.0.setpoints.ventilation_airflow_m3_per_s, // Issue #2345: was hardcoded 10000.0
                    ) * 1000.0 // Convert kW to W
                } else {
                    0.0
                };

            // Calculate required thermal load based on free-floating temperature and setpoints
            // Use ideal loads formula (mass_flow × cp × ΔT) — consistent with actual HVAC output
            let required_load = match hvac_mode {
                EquipmentHVACMode::Heating => ideal_loads_for_equipment.as_ref()[0].max(0.0),
                EquipmentHVACMode::Cooling => {
                    (-ideal_loads_for_equipment.as_ref()[0]).max(0.0) - free_cooling_capacity
                }
                EquipmentHVACMode::Off => 0.0,
            };

            // Apply modulation (0-100% capacity) from predictive control
            let mut modulated_load = required_load * modulation;

            // Clamp modulated_load to equipment rated capacity (Plan 18-08)
            // Prevents thermal demand from exceeding equipment capacity
            let capacity = equipment.calculate_capacity(1.0, outdoor_temp);
            modulated_load = modulated_load.clamp(0.0, capacity);

            // Update equipment state for PLR tracking (needs mutable borrow)
            equipment.update_state(modulated_load, outdoor_temp, hvac_mode);

            // Calculate electrical power with efficiency curve (immutable borrow)
            let electrical_power =
                equipment.calculate_power(modulated_load, outdoor_temp, hvac_mode);

            // Apply cycling losses
            let (efficiency_multiplier, _startup_penalty) = self
                .0
                .hvac
                .cycling_tracker
                .calculate_cycling_loss(electrical_power > 0.0, equipment.current_plr());

            let actual_electrical_power = electrical_power * efficiency_multiplier;

            // Accumulate electrical energy consumption (Plan 18-08)
            // actual_electrical_power is in Watts, dt_seconds is in seconds
            // Convert to kWh: (Watts × dt_seconds) / 3.6e6 = kWh
            let energy_this_timestep = actual_electrical_power * dt_seconds / 3.6e6;
            self.0.hvac.annual_electrical_energy += energy_this_timestep;

            // FIX: For multi-zone buildings (e.g., Case 960), use per-zone HVAC demand
            // instead of broadcasting a single scalar value to all zones.
            // Use IdealLoadsSystem thermodynamic formulas (mass_flow * cp * delta_t)
            // instead of sensitivity-based (setpoint - temp) / sensitivity
            //
            // Issue #1163: symmetric ideal-HVAC formula (mass heat-release is
            // already embedded in t_i_free via num_tm).
            // Issue #2826: per-zone setpoint vectors drive HVAC demand;
            // scalar `heating_setpoint` / `cooling_setpoint` (from
            // `self.0.setpoints.heating_setpoint` above) are used as fallback.
            let hvac_output = self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                heating_setpoint,
                cooling_setpoint,
            );

            // Track peak heating/cooling based on per-zone HVAC demand (Plan 18-08)
            // Physics-based: No calibration factors - track actual HVAC demand
            // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
            let enabled_vec = self.0.hvac.hvac_enabled.as_ref();
            let mut hvac_output_sum: f64 = 0.0;
            for (i, (output, &enabled)) in hvac_output
                .as_ref()
                .iter()
                .zip(enabled_vec.iter())
                .enumerate()
            {
                let val = if enabled > 0.5 { *output } else { 0.0 };
                hvac_output_sum += val;

                // Issue #1289: Track per-zone peaks
                // Issue #1628: Also track timestep when peak occurred
                if val > 0.0 {
                    // Heating mode for this zone
                    let val_kw = val / 1000.0;
                    if val_kw > self.0.hvac.zone_peak_heating_kw.as_mut()[i] {
                        self.0.hvac.zone_peak_heating_kw.as_mut()[i] = val_kw;
                        self.0.hvac.zone_peak_heating_timestep[i] = timestep;
                    }
                } else if val < 0.0 {
                    // Cooling mode for this zone
                    let val_kw = -val / 1000.0;
                    if val_kw > self.0.hvac.zone_peak_cooling_kw.as_mut()[i] {
                        self.0.hvac.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                        self.0.hvac.zone_peak_cooling_timestep[i] = timestep;
                    }
                }
            }
            // Track global peak (sum of all zones)
            if hvac_output_sum > 0.0 {
                self.0.hvac.peak_power_heating =
                    self.0.hvac.peak_power_heating.max(hvac_output_sum);
            } else if hvac_output_sum < 0.0 {
                self.0.hvac.peak_power_cooling =
                    self.0.hvac.peak_power_cooling.max(-hvac_output_sum);
            }

            // Both equipment and fallback paths now use hvac_output (per-zone VectorField)
            // so it needs to be returned for both branches
            hvac_output
        } else {
            // Use IdealLoadsSystem thermodynamic formulas for energy
            //
            // Issue #1163: symmetric ideal-HVAC formula (mass heat-release is
            // already embedded in t_i_free via num_tm).
            // Issue #2826: per-zone setpoint vectors drive HVAC demand;
            // scalar fallback when vectors are shorter than `num_zones`.
            let hvac_output_raw = self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                self.0.setpoints.heating_setpoint,
                self.0.setpoints.cooling_setpoint,
            );

            // Root Cause Fix: Use hvac_output_raw for peak tracking (consistent with energy calc)
            // Issue #901 perf: borrow hvac_output_raw directly instead of cloning for
            // the peak-power read. The original is returned to the caller below, untouched.
            let hvac_power_for_peak = hvac_output_raw.as_ref();

            // Track peak heating/cooling based on actual HVAC demand (only if not already tracked above)
            // Note: This is the fallback path when hvac_equipment is None
            // Note: hvac_output_raw is positive for heating, negative for cooling
            // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
            let enabled_vec = self.0.hvac.hvac_enabled.as_ref();
            let mut hvac_power_watts_sum: f64 = 0.0;
            for (i, (output, &enabled)) in hvac_power_for_peak
                .iter()
                .zip(enabled_vec.iter())
                .enumerate()
            {
                let val = if enabled > 0.5 { *output } else { 0.0 };
                hvac_power_watts_sum += val;

                // Issue #1289: Track per-zone peaks
                // Issue #1628: Also track timestep when peak occurred
                if val > 0.0 {
                    let val_kw = val / 1000.0;
                    if val_kw > self.0.hvac.zone_peak_heating_kw.as_mut()[i] {
                        self.0.hvac.zone_peak_heating_kw.as_mut()[i] = val_kw;
                        self.0.hvac.zone_peak_heating_timestep[i] = timestep;
                    }
                } else if val < 0.0 {
                    let val_kw = -val / 1000.0;
                    if val_kw > self.0.hvac.zone_peak_cooling_kw.as_mut()[i] {
                        self.0.hvac.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                        self.0.hvac.zone_peak_cooling_timestep[i] = timestep;
                    }
                }
            }

            // Track global peak
            if hvac_power_watts_sum > 0.0 {
                self.0.hvac.peak_power_heating =
                    self.0.hvac.peak_power_heating.max(hvac_power_watts_sum);
            } else if hvac_power_watts_sum < 0.0 {
                self.0.hvac.peak_power_cooling =
                    self.0.hvac.peak_power_cooling.max(-hvac_power_watts_sum);
            }

            hvac_output_raw
        };

        // Plan 03-04: Use hvac_output_raw directly for energy calculation
        // Ti_free calculation already includes thermal mass effects via:
        // - h_tr_em and h_tr_ms conductances (thermal mass coupling)
        // - Thermal capacitance Cm (thermal mass response rate)
        // - Implicit/explicit Euler integration (Cm × ΔTm/dt)
        // Therefore, NO multiplicative correction factor should be applied

        // 4. Update Temperatures using Energy Balance
        // Root Cause Fix (Case 600): Replace sensitivity superposition with ideal loads.
        // The sensitivity formula t_i_act = t_i_free + sensitivity * hvac_output assumes
        // mass temperature is static — invalid when HVAC heat flows through the high-conductance
        // mass path (h_is_ms_series = 583 W/K for Case 600, creating 6.1x conductance overestimate).
        //
        // Fix: Use compute_zone_hvac_load() (mass_flow × cp × ΔT) unconditionally.
        // Temperature change: t_i_act = t_i_free + hvac_power / h_tr_is (physically correct).
        //
        // Issue #738: Check free_float BEFORE calling HVAC to ensure zero output
        let hvac_for_temp_calc = if self.0.hvac.free_float {
            T::from(self.0.solar.zero_vector.clone())
        } else {
            // Issue #1163: symmetric ideal-HVAC formula (mass heat-release is
            // already embedded in t_i_free via num_tm).
            // Issue #2826: per-zone setpoint vectors drive HVAC demand;
            // scalar fallback when vectors are shorter than `num_zones`.
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                self.0.setpoints.heating_setpoint,
                self.0.setpoints.cooling_setpoint,
            )
        };

        // Temperature update: t_i_act = t_i_free + hvac_power / h_tr_is
        //
        // Issue #903 Root Cause: A previous change (c372977) replaced this formula
        // with an explicit energy balance that added `q_infiltration = h_ve * (T_out - T_free)`
        // on top of t_i_free. That double-counts the infiltration loss, because
        // t_i_free already includes the steady-state h_ve × (T_outdoor - T_zone) term
        // through `den = h_ms_is_prod + term_rest_1 * (h_ve + h_tr_w) + ...`.
        //
        // For Case 600 (low-mass), C_zone_air ≈ 72 kJ/K is small compared to the
        // ~20-50°C outdoor swings × dt = 3600 s × h_ve = 22 W/K, so the spurious
        // q_infiltration term drove delta_t up to ±20°C per timestep, collapsing
        // both t_i_act and the mass temperature (via the surface coupling), which
        // then required excessive heating in every subsequent step. Free-float
        // cases (no HVAC to balance it) ended up locked near the outdoor temperature
        // — Case 600FF max 39°C vs reference 64.9-75.1°C, min -20.7°C vs -18.8 to -15.6°C.
        //
        // Fix: restore the original physics-based formula. The hvac_power / h_tr_is
        // term gives the steady-state temperature rise the HVAC achieves through the
        // air-to-surface coupling, and t_i_free already accounts for all other heat
        // flows (infiltration, conduction, solar, internal gains, mass coupling).
        // Temperature update: t_i_act = t_i_free + hvac_power / h_tr_is
        //
        // NOTE (Issue #1163): The physically correct divisor here is h_coeff
        // (the Norton equivalent used in `compute_zone_hvac_load`), which would
        // give t_i_act = T_setpoint exactly. However, using h_coeff interacts
        // poorly with the 5R1C steady-state solver's known dynamic bias
        // (ARCHITECTURE.md §Module Status): it lets the thermal mass equilibrate
        // to the setpoint too quickly, suppressing the transient heating demand
        // that the reference (EnergyPlus with CTF) captures. The h_tr_is divisor
        // is retained as-is for now — it is a pre-existing condition, not part
        // of the #1163 cooling-formula fix. The multi-node path (line ~2417)
        // already uses h_coeff correctly.
        //
        // === Issue #2868: use the air-node denominator, not h_tr_is ===
        //
        // `h_tr_is` is not the conductance the injected HVAC power works
        // against. Solving the same 5R1C air-node equation that produced
        // `t_i_free = num / den` with the HVAC source added gives, exactly,
        //
        //     t_i_act = t_i_free + Q_hvac / den_true,
        //     den_true = den / term_rest_1 = H_tr,1 + H_tr,w + H_ve + H_tr,floor
        //
        // (`den` and `term_rest_1` are the scaled quantities used above; the
        // scaling by `term_rest_1 = h_tr_ms + h_tr_is` cancels out of the
        // ratio). `den_true` is within ~5 % of `h_tr_is` for every case that
        // has windows *and* infiltration (Case 600: 157.6 vs 165.6 W/K), so
        // the calibration of the 600/900 series is essentially unchanged.
        //
        // For a zone with neither windows nor infiltration — the ASHRAE 140
        // Case 195 "no-loads" configuration — `den_true` is 40 % smaller
        // (99.9 vs 165.6 W/K) because every watt must travel
        // air → surface → mass → outdoor. Dividing by `h_tr_is` there left the
        // ideal-load-controlled zone air ~10 K BELOW its 20 °C setpoint
        // (measured: 9.9-12.5 °C in January), and since `t_i_act` is what the
        // surface/mass balance and the next step's `t_i_free` are built from,
        // the demand never converged to the envelope loss: 1534 W injected
        // against a 143 W envelope loss, i.e. a ~10× energy-balance violation
        // and +82 % annual heating (Issue #2868).
        let h_tr_is_vec = self.0.conduction.h_tr_is.as_ref();
        let den_slice = den.as_ref();
        let term_rest_1_slice = self.0.conduction.derived_term_rest_1.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_for_temp_calc.as_ref();
        for i in 0..self.0.hvac.num_zones {
            let h_is = h_tr_is_vec[i];
            // den_true = den / term_rest_1 (unscaled air-node denominator).
            // Fall back to h_tr_is when the scaled quantities are degenerate
            // (unit tests that bypass `from_spec` and leave the cache empty).
            let den_true = match (den_slice.get(i), term_rest_1_slice.get(i)) {
                (Some(&d), Some(&t)) if t > 0.0 && d > 0.0 => d / t,
                _ => h_is,
            };
            if den_true > 0.0 && hvac[i].abs() > 1e-6 {
                scratch.t_i_act[i] = t_free[i] + hvac[i] / den_true;
            } else {
                scratch.t_i_act[i] = t_free[i];
            }
        }
        let t_i_act = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.t_i_act,
        )));

        // Use hvac_for_temp_calc for energy (matches what was used for temperature update)
        // This ensures energy calculation is consistent with temperature physics
        let mut heating_sum = 0.0;
        let mut cooling_sum = 0.0;
        let mut total_signed = 0.0;

        // Per-zone energy accumulation (Issue #1288)
        // hvac_for_temp_calc: positive = heating, negative = cooling
        let hvac_vec = hvac_for_temp_calc.as_ref();
        let zone_heating_slice = self.0.hvac.zone_heating_energy_kwh.as_mut();
        let zone_cooling_slice = self.0.hvac.zone_cooling_energy_kwh.as_mut();
        for i in 0..self.0.hvac.num_zones {
            let val = hvac_vec[i];
            total_signed += val;

            // dt is in seconds, convert to kWh: watts * seconds / 3.6e6
            let energy_kwh = val * dt / 3.6e6;
            if val > 0.0 {
                heating_sum += val;
                zone_heating_slice[i] += energy_kwh;
            } else {
                cooling_sum += -val;
                zone_cooling_slice[i] += -energy_kwh;
            }
        }

        // Compute energy (uncorrected for physics)
        let heating_energy_joules = heating_sum * dt;
        let cooling_energy_joules = cooling_sum * dt;

        // Issue #738 + Issue #821: free_float mode MUST produce zero HVAC output.
        // Promoted from debug_assert! to a hard assert under cfg(test) so the
        // ASHRAE 140 free-float regression test catches any code path that
        // sneaks HVAC demand in via the equipment fallback.
        if self.0.hvac.free_float {
            #[cfg(test)]
            assert!(
                total_signed.abs() < 1e-6,
                "Free-float mode should have zero HVAC output, got {} W",
                total_signed
            );
            #[cfg(not(test))]
            debug_assert!(
                total_signed.abs() < 1e-6,
                "Free-float mode should have zero HVAC output, got {} W",
                total_signed
            );
        }

        // Physics-based: No correction factors - use raw energy values
        self.0.hvac.annual_heating_energy += heating_energy_joules / 3.6e6;
        self.0.hvac.annual_cooling_energy += cooling_energy_joules / 3.6e6;

        // hvac_energy_for_step returns total HVAC energy in JOULES (not kWh)
        // The test expects Joules and multiplies by 3.6e6
        // DON'T apply correction here - it would break temperature calculations
        let hvac_energy_for_step = total_signed * dt;

        // Issue #272, #274, #275: Calculate thermal mass energy change
        // HVAC energy currently includes energy stored in thermal mass, which should be subtracted
        // Mass energy change = Cm × (Tm_new - Tm_old)
        // Save old mass temperature before updating

        // === Issue #1860: Time-constant-aware mass-node surface temperature ===
        //
        // Root cause of the ~38–90% cooling-load underestimation on low-mass
        // ASHRAE 140 cases: the surface temperature that drives the mass-node
        // integration was computed algebraically from the HVAC-controlled zone
        // air temperature (`t_i_act`). This propagates 100% of the HVAC cooling
        // (or heating) effect to the thermal mass within a single timestep,
        // even though the mass time constant τ_mass = Cm / H_tr,3 is typically
        // 10–60 h for low-mass constructions.
        //
        // For Case 600 (τ_mass ≈ 12.4 h on a 1-hour timestep), only ~7.8% of
        // the HVAC effect should reach the mass per step. The algebraic coupling
        // artificially cooled the mass during cooling periods, which suppressed
        // `num_tm = h_ms_is_prod × T_mass`, lowered `T_free`, and reduced the
        // sustained cooling demand — yielding annual cooling ~38% below the
        // ASHRAE 140 reference band.
        //
        // Fix: compute two surface temperatures — one from the free-floating
        // air temperature (`t_i_free`, what the zone WOULD be without HVAC)
        // and one from the HVAC-controlled temperature (`t_i_act`). Blend them
        // using the exact-exponential mass-response fraction:
        //
        //   α = 1 − exp(−dt / τ_mass),    τ_mass = Cm / H_tr,3
        //   T_s = (1 − α) × T_s_free + α × T_s_act
        //
        // For short τ_mass (fast-responding mass): α → 1, mass sees HVAC
        //   effect fully (correct: mass tracks the controlled air temperature).
        // For long τ_mass (slow-responding mass): α → 0, mass sees the
        //   free-floating temperature (correct: mass retains heat, driving
        //   sustained cooling demand through T_free on subsequent steps).
        //
        // This is NOT a case-specific correction factor — α is derived from
        // the building's physical properties (Cm, H_tr,3) and the timestep.
        // It applies uniformly to all construction types and all ASHRAE 140
        // cases. The free-float path (hvac_output == 0 → t_i_act == t_i_free)
        // is unaffected because T_s_free == T_s_act when T_free == T_act.
        let h_tr_ms_ref = self.0.conduction.h_tr_ms.as_ref();
        let mass_temps_ref = self.0.mass.mass_temperatures.as_ref();
        let h_tr_is_ref = self.0.conduction.h_tr_is.as_ref();
        let t_i_act_ref = t_i_act.as_ref();
        let t_i_free_ref = t_i_free.as_ref();
        let phi_st_ref = phi_st.as_ref();
        let term_rest_1_ref = term_rest_1.as_ref();
        let h_tr_3_ref = self.0.conduction.derived_h_tr_3.as_ref();

        for i in 0..self.0.hvac.num_zones {
            let cm_i = self.0.mass.thermal_capacitance.as_ref()[i];
            let h_tr_3_i = h_tr_3_ref[i];

            // HVAC-controlled surface temperature (full HVAC coupling).
            let ts_act_num = h_tr_ms_ref[i] * mass_temps_ref[i]
                + h_tr_is_ref[i] * t_i_act_ref[i]
                + phi_st_ref[i];
            let t_s_act_i = ts_act_num / term_rest_1_ref[i];

            // Free-floating surface temperature (no HVAC coupling).
            let ts_free_num = h_tr_ms_ref[i] * mass_temps_ref[i]
                + h_tr_is_ref[i] * t_i_free_ref[i]
                + phi_st_ref[i];
            let t_s_free_i = ts_free_num / term_rest_1_ref[i];

            // Time-constant-aware blend: fraction of HVAC effect that
            // physically reaches the mass within one timestep.
            let t_s_blended = if h_tr_3_i > 0.0 && cm_i > 0.0 && dt > 0.0 {
                let tau_mass = cm_i / h_tr_3_i;
                let alpha = 1.0 - (-dt / tau_mass).exp();
                (1.0 - alpha) * t_s_free_i + alpha * t_s_act_i
            } else {
                // Fallback: full HVAC coupling (legacy behaviour).
                t_s_act_i
            };

            scratch.t_s_act[i] = t_s_blended;
        }
        let t_s_act = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.t_s_act,
        )));

        // Update mass temperatures using implicit integration for high thermal capacitance
        // This addresses instability with explicit Euler for Cm > 500 J/K
        let mass_temps_ref = self.0.mass.mass_temperatures.as_ref();
        let thermal_cap_ref = self.0.mass.thermal_capacitance.as_ref();
        // Mode-specific fields removed - use physics-based h_tr_em and h_tr_ms
        let h_tr_em_ref = self.0.conduction.h_tr_em.as_ref();
        let h_tr_ms_ref = self.0.conduction.h_tr_ms.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let t_i_act_ref = t_i_act.as_ref();
        let phi_m_ref = phi_m.as_ref();
        let h_tr_3_ref_2 = self.0.conduction.derived_h_tr_3.as_ref();
        let h_tr_is_ref_2 = self.0.conduction.h_tr_is.as_ref();

        // Determine HVAC mode from hvac_output_raw (Plan 03-14)
        // Use separate heating/cooling coupling parameters based on mode

        for i in 0..self.0.hvac.num_zones {
            let tm_old = mass_temps_ref[i];
            let cm = thermal_cap_ref[i];
            let t_s = t_s_act_ref[i];
            // === Issue #2868: zone-air ↔ thermal-mass conductance ===
            //
            // The three integrators below drive the mass node from the air
            // side through ISO 13790 `H_tr,3 = 1/(1/H_tr,2 + 1/H_ms)` with
            // `H_tr,2 = H_tr,1 + H_tr,w` and `H_tr,1 = 1/(1/H_ve + 1/H_is)`.
            // That elimination expresses the network in terms of the SUPPLY-AIR
            // node, so `H_tr,1` — and hence `H_tr,3` — collapses to zero when a
            // zone has neither ventilation/infiltration (`H_ve = 0`) nor windows
            // (`H_tr,w = 0`).
            //
            // With `H_tr,3 = 0` the air-coupling term disappeared from the mass
            // balance entirely: the mass node was left attached only to
            // `h_tr_em`, so it floated to the sol-air temperature no matter what
            // the HVAC did (measured T_mass ≈ daily-mean outdoor). The zone air,
            // pinned to the mass by the 5R1C closed form, then sat ~10 K below
            // setpoint and the ideal-load demand charged ~2.4× the true envelope
            // loss — the +82 % annual-heating error on ASHRAE 140 Case 195.
            //
            // The *physical* zone-air ↔ mass conductance never vanishes: it is
            // the series film/mass coupling `1/(1/H_is + 1/H_ms)` (the same
            // Norton coefficient `compute_hvac_coefficient` uses for the 5R1C
            // HVAC demand). Fall back to it when the ISO elimination degenerates.
            // Cases with windows or infiltration keep `H_tr,3` unchanged, so the
            // 600-660 / 900-960 calibration is untouched.
            let h_air_mass = {
                let h_tr_3_i = h_tr_3_ref_2.get(i).copied().unwrap_or(0.0);
                if h_tr_3_i > 0.0 {
                    h_tr_3_i
                } else {
                    let h_is = h_tr_is_ref_2.get(i).copied().unwrap_or(0.0);
                    let h_ms = h_tr_ms_ref[i];
                    if h_is + h_ms > 0.0 {
                        h_is * h_ms / (h_is + h_ms)
                    } else {
                        h_ms
                    }
                }
            };
            // Issue #1860: blend the air temperature used for mass coupling
            // between the free-floating and HVAC-controlled values, using the
            // same time-constant fraction α = 1 − exp(−dt/τ_mass) as the
            // surface-temperature blend above. This ensures the CrankNicolson
            // path (which takes t_i directly, not t_s) also sees the
            // time-constant-aware mass coupling.
            //
            // Issue #2868: the blend deliberately keeps reading the raw
            // `derived_h_tr_3`, so the degenerate `H_tr,3 = 0` configuration
            // keeps taking the α = 1 (full HVAC coupling) fallback branch below.
            // That is the internally consistent pairing: a real conductance
            // (`h_air_mass`) driven by the actual controlled air temperature.
            // Re-deriving α from `h_air_mass` instead would re-introduce a
            // permanent offset between the mass and the conditioned zone and
            // break the steady-state energy balance again.
            let t_i = {
                let h_tr_3_i = h_tr_3_ref_2[i];
                if h_tr_3_i > 0.0 && cm > 0.0 && dt > 0.0 {
                    let tau_mass = cm / h_tr_3_i;
                    let alpha = 1.0 - (-dt / tau_mass).exp();
                    (1.0 - alpha) * t_i_free_ref[i] + alpha * t_i_act_ref[i]
                } else {
                    t_i_act_ref[i]
                }
            };
            #[cfg(feature = "debug-physics")]
            if i == 0 && timestep < 3 {
                let tm = mass_temps_ref[i];
                let cm_dt = cm / dt;
                let h_tr_em_i = h_tr_em_ref[i];
                let h_tr_3_i = h_tr_3_ref_2[i];
                let half_cond = 0.5 * (h_tr_3_i + h_tr_em_i);
                let denom = cm_dt + half_cond;
                let numer = tm * (cm_dt - half_cond)
                    + h_tr_em_i * t_sol_air[i]
                    + h_tr_3_i * t_i
                    + phi_m_ref[i];
                eprintln!(
                    "[PHYS] step={} t_i_free={:.4} t_i={:.4} t_i_act={:.4} tm_old={:.4} denom={:.4} numer={:.4} t_sol_air={:.4}",
                    timestep, t_i_free_ref[i], t_i, t_i_act_ref[i], tm, denom, numer, t_sol_air[i]
                );
            }
            let phi_m_zone = phi_m_ref[i];

            // Use physics-based h_tr_em and h_tr_ms (mode-specific factors removed)
            // The conductances are now calculated from first principles:
            // h_tr_em = k * A / d (thermal conductivity * area / thickness)
            // h_tr_ms = k * A / d (thermal conductivity * area / thickness)
            let h_tr_em = h_tr_em_ref[i];
            // Issue #2868: `h_tr_ms` is now consumed through `h_air_mass`
            // (computed above with the degenerate-`H_tr,3` fallback).
            let _h_tr_ms = h_tr_ms_ref[i];

            // Select integration method based on thermal capacitance
            let method = select_integration_method(cm);

            // === SESSION 72: Night Ventilation Mass Cooling ===
            // When night ventilation is active, cool outdoor air directly cools the thermal mass
            // through convection. This is critical for night ventilation cases (650, 950).
            // The ventilation-to-mass conductance is proportional to the ventilation rate.
            // === Issue #821: Night-vent mass coupling ===
            // The legacy code routed 30% of the ventilation flow directly to the mass
            // node as an "extra" cooling path while leaving `h_ve` unchanged. With the
            // ISO 13790 lumped `h_tr_ms` (now ~1.3 kW/K instead of ~120 W/K) the mass
            // and air nodes are an order of magnitude more strongly coupled, so the
            // mass already tracks the air-side ventilation cooling without the
            // empirical 30% boost. Setting `h_vent_mass_zone = 0` removes the
            // double-counting and restores Case 650FF peak free-float temperature
            // to within the ASHRAE 140 reference band [63.2, 73.5] °C.
            //
            // The air-side ventilation increase under night-vent is still routed
            // through `phi_m_with_vent`/`phi_ia_with_vent` further down (where the
            // outdoor-air enthalpy difference is applied via `h_ve` × ΔT).
            let h_vent_mass_zone = if night_vent_active_now {
                h_ve_night
            } else {
                0.0
            };

            let tm_new = match method {
                ThermalIntegrationMethod::BackwardEuler => {
                    // Use implicit backward Euler for high thermal mass
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    // Issue #896 FIX: Use h_tr_3 instead of h_tr_ms for the air-to-mass bottleneck.
                    // See detailed comment in the CrankNicolson branch below.
                    // Issue #2868: `h_air_mass` is `H_tr,3` with the degenerate
                    // `H_ve = H_tr,w = 0` fallback applied (see above).
                    let h_tr_3_zone = h_air_mass;
                    // Backward Euler with h_tr_3 and night ventilation:
                    // (Cm/dt + h_tr_em + h_tr_3 + h_vent_mass_zone) * Tm_new =
                    //     Cm/dt * Tm_old + h_tr_em * t_sol_air + h_tr_3 * t_s + h_vent_mass_zone * t_outdoor + phi_m
                    let cm_dt = cm / dt;
                    let denom = cm_dt + h_tr_em + h_tr_3_zone + h_vent_mass_zone;
                    let numer = cm_dt * tm_old
                        + h_tr_em * t_sol_air[i]
                        + h_tr_3_zone * t_s
                        + h_vent_mass_zone * outdoor_temp
                        + phi_m_zone;
                    numer / denom
                }
                ThermalIntegrationMethod::ExplicitEuler => {
                    // Use explicit Euler for low thermal mass (faster, still stable)
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    // Issue #896 FIX: Use h_tr_3 instead of h_tr_ms for the air-to-mass bottleneck.
                    // See detailed comment in the CrankNicolson branch below.
                    // Issue #2868: degenerate-`H_tr,3` fallback (see above).
                    let h_tr_3_zone = h_air_mass;
                    let q_vent_mass = h_vent_mass_zone * (outdoor_temp - tm_old);
                    let q_m_net = h_tr_em * (t_sol_air[i] - tm_old)
                        + h_tr_3_zone * (t_s - tm_old)
                        + phi_m_zone
                        + q_vent_mass;
                    tm_old + (q_m_net / cm) * dt
                }
                ThermalIntegrationMethod::CrankNicolson => {
                    // Crank-Nicolson for 2nd-order accuracy (ISO 13790 §C.4)
                    // Issues #896 + #917 combined fix:
                    // - Pass explicit temperature driving terms to the integrator (#917)
                    // - Use t_i (zone air) as the boundary for h_tr_3, NOT t_s (surface temp) (#896)
                    //   because t_s includes Tm_old feedback through h_tr_ms, creating an
                    //   artificial self-coupling loop. t_i is the correct upstream boundary.
                    // - t_ext = sol-air temperature (opaque envelope driving temp for h_tr_em)
                    // - t_sup = zone air temperature (air-side network driving temp for h_tr_3)
                    // Issue #1693: Add night ventilation (h_vent_mass_zone) to the mass balance.
                    // Adding it to h_tr_3 approximates night vent as an additional conductance
                    // from mass to zone air, which captures the cooling effect.
                    // Issue #2868: degenerate-`H_tr,3` fallback (see above).
                    let h_tr_3_with_vent = h_air_mass + h_vent_mass_zone;
                    crank_nicolson_iso13790(
                        tm_old,
                        dt,
                        cm,
                        h_tr_3_with_vent,
                        h_tr_em,
                        t_sol_air[i],
                        t_i,
                        phi_m_zone,
                    )
                }
            };

            scratch.new_mass[i] = tm_new;
        }

        // Update the mass temperatures with new values (convert Vec to T type)
        let new_mass_temps_vf: T =
            VectorField::from_smallvec(std::mem::take(&mut scratch.new_mass)).into();

        // Plan 03-04: Update previous mass temperature for tracking (kept for diagnostic output)
        // Mass energy change tracking removed - Ti_free already includes thermal mass effects
        self.0.mass.previous_mass_temperatures =
            std::mem::replace(&mut self.0.mass.mass_temperatures, new_mass_temps_vf);

        // Store previous temperatures for dT/dt calculation (Plan 15-04, 15-06)
        self.0.hvac.previous_temperatures =
            VectorField::new(self.0.setpoints.temperatures.as_ref().to_vec());
        self.0.setpoints.temperatures = t_i_act;

        // Return HVAC energy (Plan 03-04: Use hvac_energy_for_step directly)
        // Thermal mass energy accounting removed - Ti_free calculation already includes thermal mass effects
        // No subtraction of mass energy change needed
        let net_hvac_energy_for_step = hvac_energy_for_step;

        // Diagnostics recording (if enabled)
        if self.0.diagnostics_state.diagnostics.is_some() {
            // Store current HVAC output for this timestep (per zone, Watts)
            self.0.hvac.current_hvac_output = Some(hvac_output_raw);
            // Temporarily take diagnostics out to avoid borrow conflicts
            let mut diag = self.0.diagnostics_state.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics_state.diagnostics = Some(diag);
            // Clear the buffer after use
            self.0.hvac.current_hvac_output = None;
        }

        // Issue #2756: restore the pooled scratch so the next timestep reuses
        // the same SmallVec capacity (zero steady-state allocation).
        self.0.hvac.scratch_pool.return_5r1c(scratch);

        net_hvac_energy_for_step / 3.6e6 // Return kWh
    }
}

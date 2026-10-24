//! 5R1C physics step implementation for `ThermalModel`.

use crate::api::error::FluxionError;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::diagnostics_ext::SimulationDiagnosticsExt;
use crate::sim::hvac::{HVACMode as EquipmentHVACMode, VariableCapacityEquipment};
use crate::sim::moisture::step_zone_moisture;
use crate::sim::sky_radiation::SolAirTemperature;
use crate::sim::thermal_integration::{
    crank_nicolson_iso13790, select_integration_method, ThermalIntegrationMethod,
};
use crate::sim::thermal_model_core::ThermalModel;
use fluxion_core::weather::psychrometrics::{
    calculate_humidity_ratio, STANDARD_ATMOSPHERIC_PRESSURE_Pa,
};
use smallvec::SmallVec;

use super::step_common::step_wall_surface_ode;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    pub(crate) fn step_physics_5r1c(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> Result<f64, FluxionError> {
        let dt = dt_seconds; // Use provided timestep duration

        // [Loop round 86 — free-float energy-balance audit dump (read-only,
        // env-gated: FLX_FF_EB_AUDIT). Emits FFAUDIT_P/_Z/_C/_D lines for the
        // offline attribution harness (scripts/audit_free_float_balance.py);
        // default off, no effect on any computed value.]
        let ff_audit = crate::util::env_bool::env_bool("FLX_FF_EB_AUDIT", false);

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

        // [R91 FLIP — the ASHRAE 140 Table 5-138 window-loss solar sink is
        // the engine default since loop round 91 (bands-as-factor audit;
        // Alex directive 2026-10-22 21:20 ET: re-evaluate every decision
        // previously taken on band movement under the standing 2026-10-22
        // criteria — known, verifiable, tested physics; band movement is
        // recorded, not weighted). PREMISE (primary source, the licensed
        // 140-2017 text in docs/): §5.2.1.12.1 defines the interior solar
        // distribution as "accounting for solar lost (transmitted solar
        // radiation reflected by interior opaque surfaces and
        // retransmitted back out the windows)"; Table 5-138's own row
        // carries the value: "Solar Lost through Windows 0.035". The
        // engine transmitted 100% of window solar into the zone
        // distribution — the over-absorption rounds 81/83 disclosed. The
        // round-83 retransmission arm (FLX_R83_RETRANS) was left gated
        // solely on band direction; under the new criteria the spec sink
        // is reference-conformant and goes default on its own merits.
        // Measured on a0446c26 (arm battery, flag-off bit-identical to the
        // round-90 record 42/42, FLX_R91_RETRANS=1): 600/C 4.728 -> 4.490
        // MWh, 610-650/C -3.8..-5.3%, 600/H 5.959 -> 6.054 MWh (+1.6%),
        // 600FF Tmax 53.84 -> 53.05 C, 650FF Tmax 53.09 -> 52.25 C, both
        // FF minima stay in-band, no strict-gate row changes state
        // (check_strict 0), high-mass family byte-identical; 17/42
        // output-diff metrics moved, all low-mass family. A falsy token
        // (0/false/no/off) restores the x1.0 route for A/B archaeology
        // only; any other value behaves as unset, i.e. the retransmission
        // default.]
        let r91_retrans = crate::util::env_bool::env_bool("FLX_R91_RETRANS", true);

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
            // [R91 — FLX_R91_RETRANS sink: transmitted solar x 0.965
            // before the three-way split; the retransmitted fraction
            // leaves through the glazing and is not distributed to any
            // node (ASHRAE 140-2017 Table 5-138, "Solar Lost through
            // Windows 0.035"). Flag-off is the pre-round-91 x1.0 route.]
            let sol_w_raw = solar_ref[i] * area_ref[i];
            let sol_w = if r91_retrans {
                sol_w_raw * 0.965
            } else {
                sol_w_raw
            };
            // opaque_sol_w: kept for potential debugging; it's included via t_sol_air now
            let _opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            // Internal gains: convective to air, radiative split between surface and mass
            // Solar distribution must conserve energy (sum to 1.0)
            let sol_to_air = sol_w * self.0.solar.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;
            scratch.phi_ia[i] = load_w * conv_frac + sol_to_air;
            scratch.phi_st[i] = load_w * st_int_frac + remaining_sol * st_sol_frac;
            // [R94 — surface-pool decomposition (loop round 94). The
            // pool's two source components route by DIFFERENT references:
            // `phi_st_int` (internal-gain radiative, `rad_frac·φ_int`)
            // keeps the ISO 13790 §C.3 conductance partition everywhere
            // (per-class mass share `s_c`, air share `(1 − s_share)` via
            // the retained lag site), while `phi_st_sol` (transmitted
            // solar) follows ASHRAE 140-2017 Table 5-138 through the
            // per-class fractions — §5.2.1.12.2 scopes those constant
            // fractions to transmitted solar only. Both components use
            // the exact expressions and order of the `phi_st` sum above,
            // so `phi_st_int + phi_st_sol == phi_st` up to the same FP
            // rounding.]
            scratch.phi_st_int[i] = load_w * st_int_frac;
            scratch.phi_st_sol[i] = remaining_sol * st_sol_frac;
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
        let phi_st_sol = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_st_sol,
        )));
        let phi_st_int = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_st_int,
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
        if ff_audit {
            let fmt_slice = |v: &T| {
                v.as_ref()
                    .iter()
                    .map(|x| format!("{x:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            eprintln!(
                "FFAUDIT_P {} phi_ia={} phi_st={} phi_st_sol={} phi_st_int={} phi_m={}",
                timestep,
                fmt_slice(&phi_ia),
                fmt_slice(&phi_st),
                fmt_slice(&phi_st_sol),
                fmt_slice(&phi_st_int),
                fmt_slice(&phi_m)
            );
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
        use crate::sim::exterior_boundary::{
            aggregate_zone_boundary, is_windward, ExteriorBoundarySurface, ABSORPTANCE_ROOF,
            ABSORPTANCE_WALL,
        };
        use fluxion_core::ashrae_cases::Orientation;
        let v_wind_building = self
            .0
            .solar
            .weather
            .as_ref()
            .map(|w| wind_at_building_height_from_10m(w.wind_speed, 2.7))
            .unwrap_or(3.4);
        // Issue #4166: wind direction (meteorological, degrees) for
        // windward/leeward exterior convection selection. `None` when the
        // weather source does not provide it (synthetic weather, tests);
        // surfaces fall back to the windward coefficient.
        let wind_direction = self.0.solar.weather.as_ref().and_then(|w| w.wind_direction);
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
        let eps_ext_default = SolAirTemperature::ashrae_140_default().emissivity;
        // [D5 DEFAULT (loop round 71, #3983 §57, Alex-directed default flip
        // 2026-10-19): the per-surface-class mass split (wall / roof / floor)
        // with distributed sol-air driving is now the engine default. Measured
        // as the FLX_R66_PER_SURFACE harness arm in rounds 66-70 and verified
        // bit-identical flag-off through round 71 before the flip; see
        // docs/investigations/limit-35-case-900-heating-out-of-band.md §57.]
        let ps_enabled = true;
        // [D5 DEFAULT (loop round 71, #3983 §57): the interior film/surface
        // network is per-surface too: each class couples to zone air through
        // its own series (h_ms_c, h_is_c) with its real construction
        // capacitance. Measured as the FLX_R67_INTERIOR harness arm in rounds
        // 67-70; verified flag-off bit-identical through round 71.]
        let r67_interior = true;
        // Per-class harness tuple: (cls_g, cls_t, cls_area, cls_beam) —
        // per-class exterior conductance, sol-air, area and beam irradiance.
        type R66ClassTuples = [f64; 3];
        // [Round 78 (#4512): pooled in `ConductionState::r66_cls_workspace`
        // — the D5 default flip (2e51d3ff) rebuilt this with `vec![None;
        // num_zones]` every step (1 heap allocation per timestep). Values
        // identical: every element is overwritten or stays `None` before
        // first read (same semantics as the fresh `vec![None; n]`).]
        let mut r66_cls: Vec<
            Option<(
                R66ClassTuples,
                R66ClassTuples,
                R66ClassTuples,
                R66ClassTuples,
            )>,
        > = self
            .0
            .conduction
            .r66_cls_workspace
            .take()
            .unwrap_or_default();
        r66_cls.clear();
        r66_cls.resize_with(self.0.hvac.num_zones, || None);
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
        //
        // Issue #4166: the zone sol-air and h_tr_em are now aggregated from
        // per-surface exterior boundaries (orientation, tilt, sky view factor,
        // windward/leeward convection, absorbed irradiance) instead of a
        // single roof-style value with a direction-blind coefficient.
        let exterior_emissivity_ref = self.0.conduction.exterior_emissivity.as_ref();
        let num_zones = self.0.hvac.num_zones;
        // Reused per-zone surface buffer (avoids per-timestep allocation).
        // Round 16 (issue #2687/#2709 dhat ratchet): the buffer now lives in
        // the pooled scratch (`boundary_surfaces` field) instead of a fresh
        // `Vec::new()` per call — the fresh Vec re-grew from capacity 0 every
        // timestep, one heap allocation per step per config. `fill_zero`
        // clears it on checkout so the semantics (empty at loop start) are
        // unchanged; the capacity persists across steps.
        let boundary_surfaces = &mut scratch.boundary_surfaces;
        #[allow(clippy::needless_range_loop)]
        // i indexes several parallel zone vectors, not just r66_cls
        for i in 0..num_zones {
            // opaque_solar is the effective opaque irradiance on exterior surfaces (W/m²)
            // This is the combined wall + roof irradiance for the zone
            let eps_ext = exterior_emissivity_ref
                .get(i)
                .copied()
                .unwrap_or(eps_ext_default);
            boundary_surfaces.clear();
            if let Some(zone_surfaces) = self.0.solar.surfaces.get(i) {
                let irr_bd = self.0.solar.orientation_irradiance_beam_diffuse.get(i);
                let irr_gr = self.0.solar.orientation_irradiance_ground.get(i);
                for s in zone_surfaces {
                    let opaque_area = (s.area - s.window_area).max(0.0);
                    if opaque_area <= 0.0 {
                        continue;
                    }
                    // Floors (Down) are not part of the exterior solar boundary.
                    if s.orientation == Orientation::Down {
                        continue;
                    }
                    let is_roof =
                        matches!(s.orientation, Orientation::Up | Orientation::Horizontal);
                    let tilt_deg = match s.orientation {
                        Orientation::Up | Orientation::Horizontal => 0.0,
                        _ => 90.0,
                    };
                    let oi = s.orientation as usize;
                    let irr_beam_diffuse = irr_bd.map(|a| a[oi]).unwrap_or(0.0);
                    let irr_ground = irr_gr.map(|a| a[oi]).unwrap_or(0.0);
                    // Windward/leeward selection from actual wind direction.
                    // Horizontal surfaces (azimuth -1) use the windward
                    // coefficient (matches pre-#4166 behaviour).
                    let azimuth = s.orientation.azimuth_deg();
                    let windward = if azimuth < 0.0 {
                        true
                    } else {
                        wind_direction
                            .map(|wd| is_windward(wd, azimuth))
                            .unwrap_or(true)
                    };
                    let direction = match (is_roof, windward) {
                        (true, true) => ExteriorSurfaceDirection::HorizontalRoofWindward,
                        (true, false) => ExteriorSurfaceDirection::HorizontalRoofLeeward,
                        (false, true) => ExteriorSurfaceDirection::VerticalWallWindward,
                        (false, false) => ExteriorSurfaceDirection::VerticalWallLeeward,
                    };
                    let h_c_ext = h_c_ext_wind_dependent(direction, v_wind_building);
                    boundary_surfaces.push(ExteriorBoundarySurface {
                        area_opaque_m2: opaque_area,
                        azimuth_deg: azimuth,
                        tilt_deg,
                        f_sky: ExteriorBoundarySurface::sky_view_factor_from_tilt(tilt_deg),
                        h_c_ext,
                        windward: if azimuth < 0.0 { None } else { Some(windward) },
                        convection_ab: direction.ashrae_140_coefficients(),
                        irradiance_beam_diffuse_wm2: irr_beam_diffuse,
                        irradiance_ground_wm2: irr_ground,
                        absorptance: if is_roof {
                            ABSORPTANCE_ROOF
                        } else {
                            ABSORPTANCE_WALL
                        },
                        emissivity: eps_ext,
                        r_materials: (1.0 / s.u_value - 0.13 - 0.04).max(0.5),
                    });
                }
            }
            // [R66 HARNESS — per-class exterior aggregates (scratch).]
            if ps_enabled {
                let mut cls_g = [0.0f64; 3];
                let mut cls_gt = [0.0f64; 3];
                let mut cls_area = [0.0f64; 3];
                let mut cls_beam = [0.0f64; 3];
                for s in boundary_surfaces.iter() {
                    let c = if s.tilt_deg >= 45.0 { 1 } else { 0 };
                    let g = s.h_tr_em();
                    let t = s.sol_air_temperature(outdoor_temp, sky_temp);
                    cls_g[c] += g;
                    cls_gt[c] += g * t;
                    cls_area[c] += s.area_opaque_m2;
                    // [R68 HARNESS — per-class incident irradiance×area
                    // (scratch). The same weight convention as
                    // distribute_opaque_solar_gains (Issue #864: area ×
                    // irradiance, area fallback) used by the beam-weighted
                    // phi_st solar partition (FLX_R68_BEAM_PHIST).]
                    cls_beam[c] += (s.irradiance_beam_diffuse_wm2 + s.irradiance_ground_wm2)
                        * s.area_opaque_m2;
                }
                let floor_area: f64 = self
                    .0
                    .solar
                    .surfaces
                    .get(i)
                    .map(|zs| {
                        zs.iter()
                            .filter(|s| s.orientation == Orientation::Down)
                            .map(|s| (s.area - s.window_area).max(0.0))
                            .sum()
                    })
                    .unwrap_or(0.0);
                cls_area[2] = floor_area;
                let mut cls_t = [outdoor_temp; 3];
                for c in 0..2 {
                    if cls_g[c] > 1e-9 {
                        cls_t[c] = cls_gt[c] / cls_g[c];
                    }
                }
                r66_cls[i] = Some((cls_g, cls_t, cls_area, cls_beam));
            }
            let (h_tr_em, t_sol_air_i) =
                aggregate_zone_boundary(boundary_surfaces, outdoor_temp, sky_temp);
            scratch.t_sol_air_zone[i] = t_sol_air_i;
            scratch.h_tr_em_zone[i] = h_tr_em;
        }
        // Strict-energy premise refresh: persist the applied per-zone
        // boundary so `invariant_checker` reconstructs the mass-node
        // identity from the same wind-dependent aggregated values this
        // step used (the checker's legacy roof-style reconstruction
        // predates #4113/#4166 and no longer matches the integrator).
        // Issue #4373 follow-up (loop round 15): reuse the two buffers
        // instead of reallocating a fresh `Some((Vec, Vec))` every step. The
        // `to_vec()` pair put 2+ heap allocations per timestep back into the
        // BatchOracle hot loop and blew the dhat allocation-COUNT budget
        // (26,317 blocks/config vs the 60/config budget, CI run 38195944789).
        // `extend_from_slice` into the persisted buffers allocates only on
        // the first step; readers see the same applied values.
        let applied = self
            .0
            .conduction
            .applied_5r1c_boundary
            .get_or_insert_with(|| (Vec::new(), Vec::new()));
        applied.0.clear();
        applied.0.extend_from_slice(&scratch.h_tr_em_zone);
        applied.1.clear();
        applied.1.extend_from_slice(&scratch.t_sol_air_zone);
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
        // Issue #4166: per-zone h_tr_em aggregated from per-surface exterior
        // boundaries (populated above alongside t_sol_air_zone).
        let h_tr_em_zone: &[f64] = scratch.h_tr_em_zone.as_slice();

        // Simplified 5R1C calculation using CTA
        // Include ground coupling through floor
        // Use pre-computed cached values to avoid redundant allocations
        let h_ext_base = &self.0.conduction.derived_h_ext;
        let term_rest_1 = &self.0.conduction.derived_term_rest_1;

        // === Issue #824: Night-ventilation air-side coupling (was missing entirely) ===
        //
        // For ASHRAE 140 Case 650 / 650FF the spec defines a night-ventilation
        // fan that runs 18:00 → 07:00 at 1400 m³/h (ASHRAE 140 Table 5-160
        // sea-level route), supplying outside air
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

        // Round 26: consume the heating setpoint schedule (mirrors the 9R4C
        // path). The from-spec builder constructs `heating_schedule` (the
        // Case 640 thermostat setback, ASHRAE 140 §5.2.2.1.4.1: 20 °C
        // 07:00-23:00, step to 10 °C 23:00-07:00) but the HVAC demand below
        // read only the constant per-zone `heating_setpoints` vector, so the
        // setback never reached the demand and Case 640 simulated
        // byte-identical to 600 on the strict gate. Sync the per-zone vectors
        // from the schedule every step, but only when the schedule actually
        // varies: constant schedules are skipped so callers that drive the
        // demand by writing the vectors directly (the Issue #2826 contract)
        // keep working, and every non-setback case is unchanged.
        if !self.0.hvac.free_float && !self.0.setpoints.heating_schedule.is_constant() {
            let h_sp_now = self.0.setpoints.heating_schedule.value(timestep % 24);
            for v in self.0.setpoints.heating_setpoints.as_mut().iter_mut() {
                *v = h_sp_now;
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
            // Issue #3370: scratch pool buffer for the night-vent `den`
            // rebuild. The vector is only populated when night-ventilation
            // is configured on the model; the test fixture has
            // `night_vent == None` so this branch is never taken on the
            // dhat budget run, but routing through scratch keeps the path
            // zero-alloc for the cases that do exercise it (#650 / #950).
            let scratch_den = scratch.night_vent_den.as_mut();
            let n = h_ext_slice.len();
            debug_assert_eq!(scratch_den.len(), n);
            for i in 0..n {
                let h_total = if self.0.hvac.num_zones > 1 {
                    h_ext_slice[i] + h_iz[i] + h_iz_rad[i]
                } else {
                    h_ext_slice[i]
                };
                scratch_den[i] = h_ms_is_prod[i] + term_rest_1[i] * h_total + ground_coeff[i];
            }
            T::from(VectorField::from_smallvec(std::mem::take(
                &mut scratch.night_vent_den,
            )))
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
                                        fluxion_core::ashrae_cases::Orientation::Up
                                            | fluxion_core::ashrae_cases::Orientation::Down
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
                    // Issue #4166 — wind-dependent `h_tr_em`. Uses the
                    // per-surface-aggregated zone value from
                    // `scratch.h_tr_em_zone` (populated in the per-surface
                    // sol-air loop above) instead of the wall+roof
                    // `HorizontalRoofWindward` recomputation.
                    let h_tr_em_i = h_tr_em_zone.get(i).copied().unwrap_or(0.0);
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
                                        fluxion_core::ashrae_cases::Orientation::Up
                                            | fluxion_core::ashrae_cases::Orientation::Down
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
                    // Issue #4166 — wind-dependent `h_tr_em`. Uses the
                    // per-surface-aggregated zone value from
                    // `scratch.h_tr_em_zone` (populated in the per-surface
                    // sol-air loop above) instead of the wall+roof
                    // `HorizontalRoofWindward` recomputation.
                    let h_tr_em_i = h_tr_em_zone.get(i).copied().unwrap_or(0.0);
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
                                       // Issue #3370: snapshot all per-zone input fields into the
                                       // pooled `PhysicsScratch5r1c` `lw_*` scratch buffers instead of
                                       // allocating `vec![…]`/`to_vec()` per step. The scratch buffers
                                       // are `SmallVec<[f64; 4]>` sized back to `num_zones` by
                                       // `fill_zero()` on each checkout, so they reuse capacity
                                       // indefinitely after the first step. The per-zone loop then
                                       // reads via `.get(i)` as before — only the buffer ownership
                                       // changes (now `&mut SmallVec` via `.as_mut()` instead of an
                                       // owned `Vec`). Saves ~8 heap allocations per step.
            scratch
                .lw_surface_emissivity
                .as_mut()
                .copy_from_slice(self.0.conduction.surface_emissivity.as_ref());
            scratch
                .lw_t_zone
                .as_mut()
                .copy_from_slice(self.0.setpoints.temperatures.as_ref());
            scratch
                .lw_a_floor
                .as_mut()
                .copy_from_slice(self.0.setpoints.floor_area.as_ref());
            scratch
                .lw_a_ceiling
                .as_mut()
                .copy_from_slice(self.0.setpoints.roof_area.as_ref());
            scratch
                .lw_a_wall
                .as_mut()
                .copy_from_slice(self.0.setpoints.wall_area.as_ref());
            {
                // ROUND25 (§LIMIT-14 follow-up): per-zone U-values from the
                // `zone_constructions` split; the vectors are broadcast from
                // the legacy scalars when no per-zone construction exists, so
                // this reads identically for every pre-existing case.
                scratch
                    .lw_u_floor
                    .as_mut()
                    .copy_from_slice(self.0.setpoints.floor_u_values.as_ref());
                scratch
                    .lw_u_ceiling
                    .as_mut()
                    .copy_from_slice(self.0.setpoints.roof_u_values.as_ref());
                scratch
                    .lw_u_wall
                    .as_mut()
                    .copy_from_slice(self.0.setpoints.wall_u_values.as_ref());
            }
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
                let emissivity_i = scratch
                    .lw_surface_emissivity
                    .as_ref()
                    .get(i)
                    .copied()
                    .unwrap_or(0.9)
                    .clamp(0.0, 1.0);
                let a_floor = scratch.lw_a_floor.as_ref().get(i).copied().unwrap_or(0.0);
                let a_ceiling = scratch.lw_a_ceiling.as_ref().get(i).copied().unwrap_or(0.0);
                let a_wall = scratch.lw_a_wall.as_ref().get(i).copied().unwrap_or(0.0);

                let _network =
                    InteriorSurfaceNetwork::from_areas(a_floor, a_ceiling, a_wall, emissivity_i);

                let t_zone_i = scratch.lw_t_zone.as_ref().get(i).copied().unwrap_or(20.0);
                let t_sol_air_i = t_sol_air.get(i).copied().unwrap_or(outdoor_temp);

                // Per-surface envelope resistance.
                let u_floor = scratch.lw_u_floor.as_ref().get(i).copied().unwrap_or(0.5);
                let u_ceiling = scratch.lw_u_ceiling.as_ref().get(i).copied().unwrap_or(0.5);
                let u_wall = scratch.lw_u_wall.as_ref().get(i).copied().unwrap_or(0.5);
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
        // Issue #4241: Save previous air temperature for the discrete residual
        // HVAC formulation. The ODE below overwrites the working buffer, so we
        // need a copy of T_prev for the Q_HVAC residual calculation.
        // LIMIT-33 (issue #4314, Option A — state feedback): T_prev is the
        // PREVIOUS STEP'S CONTROLLED air state (`setpoints.temperatures` holds
        // the last step's `t_i_act` — the setpoint on unclamped conditioned
        // hours, the free temperature otherwise), not the free-float state in
        // `mass.air_temperatures`. The #4241 residual is derived by requiring
        // T^{n+1} = T_sp from the state the previous balance ended at, so the
        // storage term must charge off the controlled trajectory. First step
        // falls back to the from_spec initial temperature, which
        // `setpoints.temperatures` is initialized to.
        // Issue #3370: use scratch pool buffer to avoid allocation.
        scratch
            .air_node_t_prev
            .as_mut()
            .copy_from_slice(self.0.setpoints.temperatures.as_ref());
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
        // [R76 HARNESS — #3983 sub-hour mass-state resolution, round 76.
        // FLX_R76_MASS_SUBSTEPS=N: advance the mass node(s) at dt/N inside
        // this sub-step loop (Crank–Nicolson, the same integrator the
        // hourly advance uses), with the exterior driving term ramped
        // prev→current hour at the R71 midpoint fraction (persisted
        // `r76_sol_air_prev`/`r76_has_prev`), the sub-step's own air
        // temperature as the interior boundary, and the per-sub-step mass
        // temperature refreshed into the air-node numerator (`num_tm`),
        // which the default path freezes at its hour-start value. N also
        // sets the sub-step count (the air-node resolution composes with
        // the mass resolution). Structural choices disclosed for the arm:
        // (1) the lumped-node CN form, not the D5 per-class t_s-first
        // advance — the per-class decomposition is orthogonal and stays
        // hourly; (2) night-vent mass conductance held at its current-hour
        // value (R75 measured its transition effect at ≤0.0075 K);
        // (3) φ_m at its current-hour value (hourly gains). Harness arm:
        // default build (flag unset) executes the identical arithmetic
        // (bit-identical); the sub-step payoff measurement is the round-76
        // deliverable, no default change proposed.]
        let r76_n: usize = std::env::var("FLX_R76_MASS_SUBSTEPS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(0);
        // The air-node sub-step count is NOT coupled to the mass resolution:
        // the existing air sub-step loop is calibrated at its default
        // (N=3 for the 600-series, #2339) and is NOT convergent in N — the
        // per-sub-step solar-lag state injection grows with the sub-step
        // count (measured round 76: 650FF Tmax 33.1 °C at N=3 → 218 °C at
        // N=96 with the mass advance disabled; the R76 air-only diagnostic
        // `FLX_R76_NO_MASS` isolates it). The mass node therefore gets its
        // own independent sub-step count and the air machinery is untouched.
        // [R77 HARNESS — #3983 coupled single-N sub-stepping, round 77.
        // `FLX_R77_COUPLED_N=N`: air and mass sub-stepped TOGETHER at the
        // same dt/N, a single consistent N across the whole 5R1C network.
        // Two structural fixes vs the incumbent operator:
        // (1) the solar-lag term enters the air ODE as a per-sub-step DRIVER
        // (midpoint-frozen lag value added to the numerator as
        // `lag_mid * term_rest_1`, equivalent to adding `lag_mid/den_true` to
        // the steady state) and the lag STATE advances once per sub-step at
        // dt/N — the sub-step filters compose exactly to the hourly filter.
        // The incumbent per-sub-step additive post-correction
        // (`corrected += lag/den_true` fed back into `t_air_old`) is removed:
        // it injects the hour's solar-lag response N times against a state
        // that does not carry it, so the injected term accumulates ~linearly
        // in N — the measured round-76 divergence (650FF Tmax 33.1 °C at N=3
        // → 218 °C at N=96, `FLX_R76_NO_MASS` diagnostic).
        // (2) the mass node advances at the same dt/N with the sub-step air
        // temperature as interior boundary and the per-sub-step mass
        // temperature refreshed into the air-node numerator (R76 arm
        // mechanics, single N). Each driver is evaluated per sub-step exactly
        // once; the scheme's N→∞ limit is the continuous-time solution of the
        // 5R1C network with first-order solar lag. Harness arm: default build
        // (flag unset) executes the identical arithmetic (bit-identical).]
        // [DEFAULT FLIP — loop round 78 (#3983 §63/§64), Alex-directed
        // 2026-10-20 17:12 ET: "flip with N=12". The coupled single-N
        // scheme (round 77 harness `FLX_R77_COUPLED_N`) is promoted to the
        // engine default with the sub-step count FIXED at N=12 — the
        // smallest value past the monotone-convergence threshold of the
        // §63 N-sweep, chosen before the strict-gate re-record, NOT tuned
        // to bands (the §63 sweep showed no N holds the 650FF Tmin band at
        // the limit: the band is crossed, not held — the honesty proof).
        // The env var remains a positive-N override for measurement
        // (e.g. FLX_R77_COUPLED_N=96 reproduces §63's limit-proximity
        // record); the un-coupled hourly operator is no longer reachable.
        // Known cost, recorded honestly: N=12 executes 12× the default
        // sub-steps on the 5R1C path — measured annual-gate wall time
        // ~1.5× and hybrid throughput 195 → 64 configs/sec (3.0×; the
        // hybrid release-gate floor re-pinned 80 → 40 with dated
        // provenance in release_gates.yaml) — and the conditioned-family
        // movement of §63's N=12 record is adopted as-is (checker
        // statuses re-derived, rows in/out of band recorded, never
        // tuned).]
        let r77_n: usize = std::env::var("FLX_R77_COUPLED_N")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(12);
        let r77_coupled = r77_n > 0;
        let steps = if r77_coupled { r77_n } else { steps };
        let eff_mass_n: usize = if r77_coupled { r77_n } else { r76_n };
        let dt_sub = dt / steps as f64;
        // Local sub-step mass-state buffer (allocated only when the arm is
        // on); refreshed into the air-node numerator each sub-step and
        // carried to the hourly mass-advance block as the end-of-hour mass
        // state.
        let mut r76_mass: Vec<f64> = if eff_mass_n > 0 {
            self.0.mass.mass_temperatures.as_ref()[..self.0.hvac.num_zones].to_vec()
        } else {
            Vec::new()
        };
        // [Round 78 — checker persistence for the coupled scheme
        // (checker-lags-integrator, #4504 class). When the coupled
        // single-N scheme advances the mass node, record per zone the
        // linear coefficients the sub-step CN advances actually consumed,
        // `[a_0, b_0, a_1, b_1, …, a_{N−1}, b_{N−1}, denom_final]`, so
        // `invariant_checker` can mirror the composite sub-step identity
        // exactly (the hourly identity no longer matches a dt/N-mapped
        // mass advance). Pooled buffer; cleared each step, capacity kept.]
        let mut r77_coeffs: Vec<Vec<f64>> = self
            .0
            .mass
            .r77_mass_substep_coeffs
            .take()
            .unwrap_or_default();
        if r77_coupled {
            if r77_coeffs.len() != self.0.hvac.num_zones {
                r77_coeffs.clear();
                r77_coeffs.resize_with(self.0.hvac.num_zones, Vec::new);
            }
            for v in r77_coeffs.iter_mut() {
                v.clear();
            }
        }
        let r76_sol_air_prev_ref = self.0.mass.r76_sol_air_prev.as_ref();
        let r76_has_prev = self.0.mass.r76_has_prev;
        let mut r76_final: Option<Vec<f64>> = None;
        let phi_m_ref_early = phi_m.as_ref();

        // [R71 — sub-hour air-node driver resolution, default since loop
        // round 73 (#1522; env-gated harness `FLX_R71_AIR_CAP` in round 72,
        // Alex-directed default flip 2026-10-19). The exact-exponential
        // air-node ODE has had true `C_air` since #1585, but the ODE drivers
        // (num, den) were frozen at their top-of-hour values for the whole
        // step. With dt/τ_air ≈ 3.6 the node relaxes ~97% toward the frozen
        // steady state within the hour, so the air node saw a staircase, not
        // the within-hour ramp of the real drivers. Each sub-step k now
        // evaluates the drivers at the sub-step midpoint fraction
        // f = (k + 1/2)/N, linearly interpolated between the previous hour's
        // drivers (persisted in `mass.r71_air_driver_*_prev`) and this
        // hour's. Endpoint-consistent at f→1; the night-vent numerator term
        // is applied at its current-hour value after the ramp (binary
        // schedule, disclosed). φ_st / solar-lag stay hourly-frozen. Seeded
        // from zero previous drivers on the first step of each simulation
        // (no ramp on the seed hour).]
        let r71_num_prev_ref = self.0.mass.r71_air_driver_num_prev.as_ref();
        let r71_den_prev_ref = self.0.mass.r71_air_driver_den_prev.as_ref();
        let r75_vent_term_prev_ref = self.0.mass.r75_vent_term_prev.as_ref();
        let r75_vent_has_prev = self.0.mass.r75_vent_has_prev;

        // [R74 HARNESS — #1585 exact ramped-forcing increment, round 74.
        // `FLX_R74_RAMP_EXACT` integrates the exact-exponential air-node ODE
        // against the R71 linear driver ramp consistently: within each
        // sub-step the R71 default holds the ramped steady state FROZEN at
        // the sub-step midpoint fraction, then relaxes exponentially toward
        // that constant. The exact solution of dT/dt = (s(t) - T)/tau with
        // s(t) linear across the sub-step (s0 -> s1) is
        //   T(dt) = s1 - tau*g + (T0 - s0 + tau*g) * exp(-dt/tau),
        // g = (s1 - s0)/dt_sub; it reduces to the default form when g = 0
        // (constant drivers, seed hours), so seed steps stay bit-identical.
        // tau is evaluated at the same ramped-midpoint den as the default,
        // isolating only the forcing-shape change. Harness arm: default
        // build is untouched until measured evidence AND an explicit
        // Alex-level flip decision (#1585 progress thread).]
        let r74_ramp_exact = std::env::var("FLX_R74_RAMP_EXACT").is_ok();

        // [R75 HARNESS — #3983 free-float structural question, round 75.
        // `FLX_R75_VENT_RAMP` resolves the binary night-vent coupling's
        // intra-hour transitions: the R71 default ramps the num/den drivers
        // (den already carries the vent conductance transition through
        // den_prev) but re-applies the vent numerator term
        // `h_ve_night · T_outdoor` at its full current-hour value in every
        // sub-step — a step at the 07:00 / 18:00 schedule boundaries. The
        // arm interpolates that term across the existing sub-steps at the
        // same midpoint fraction, seeded from the persisted previous-hour
        // term (0 when the vent was inactive). Isolates the vent-transition
        // forcing shape; the R71 driver ramp, φ_st / solar-lag freezing and
        // the exact-exponential integrator are untouched. Harness arm:
        // default build (flag unset) never reads the r75 state fields and
        // executes the identical arithmetic (bit-identical).]
        let r75_vent_ramp = std::env::var("FLX_R75_VENT_RAMP").is_ok();

        // Initialize air-node state from previous timestep
        // Issue #3370: `t_air_state` / `solar_lag_state` are scratch pool
        // buffers (no per-step allocation); the pre-#3370 implementation
        // did `Vec::from(...)` here every call. The `t_air_old_ref`
        // initialization reads through `copy_from_slice` so the borrow
        // lifetime of `self.0.mass.air_temperatures` ends before the
        // mutable borrows of `scratch`.
        scratch
            .air_node_t_air
            .as_mut()
            .copy_from_slice(t_air_old_ref);
        scratch
            .air_node_solar_lag
            .as_mut()
            .copy_from_slice(self.0.mass.solar_lag.as_ref());

        // [R88 HARNESS — surface-pool mass share (FLX_R88_PHIST_MASS_SHARE,
        // loop round 88, Item 1). Premise verified on the live coupled-N
        // path: the per-class sub-step mass advance consumes
        // phi_c = phi_m_c only, so the ISO 13790 §C.3 mass share of the
        // surface pool, phi_st_ms = H_ms/(H_ms + H_is) · phi_st, never
        // reaches the mass node — the FULL phi_st enters the air node
        // through the R77 solar-lag filter (lag_input = h_tr_is·phi_st /
        // term_rest_1, tau_lag = sqrt(tau_air·tau_mass)). The ISO
        // algebraic surface node instead splits phi_st by conductance
        // share: the air share H_is/(H_ms + H_is) stays in the lag filter
        // and the mass share enters the mass ODE, returning on the mass's
        // own time constant. This arm restores the split. Per zone,
        // share_zone = sum_c w_c · s_c with s_c = h_ms_c/(h_ms_c +
        // h_is_i·w_c) when real per-class constructions exist (R85
        // routing, the same cls tensors the mass advance consumes), else
        // the zone-level s = h_ms/(h_ms + h_is) (tensor-less zones keep
        // the lumped sub-step advance — same pairing the R85 guard
        // uses). The lag-filter input scales by (1 − share_zone) and the
        // mass drives add their shares (per-class site: s_c·phi_st·w_c;
        // lumped sites: s·phi_st), so air + mass shares sum to phi_st up
        // to FP rounding and no energy is created or destroyed. The
        // persisted r77 affine record embeds the same share the mass CN
        // consumed, so the invariant-checker mirror stays exact by
        // construction.]
        // [DEFAULT FLIP — loop round 89, Alex-directed 2026-10-22 14:57 ET:
        // "shouldn't we enable by default correct ISO physics even if it
        // doesn't directly benefit the free-float fix?" — the share split is
        // correct spec-shaped physics and goes to the engine default on its
        // own merits, band outcomes being a downstream check, not the target.
        // Measured honestly (round-88 arm battery): every moved free-float
        // metric drifts slightly AWAY from the published band on the current
        // lag-filter + mass-coverage operator — recorded in KNOWN_ISSUES,
        // never tuned away. The falsy "0" token restores the pre-89 all-air
        // lag-filter route for A/B archaeology only; any other value behaves
        // as unset, i.e. the share-on default.]
        let r88_phist_share = crate::util::env_bool::env_bool("FLX_R88_PHIST_MASS_SHARE", true);
        // [R91 FLIP 3 - Table 5-138 per-class solar routing is the engine
        // default since loop round 91 (FLX_R91_T5138; bands-as-factor
        // audit, Alex directive 2026-10-22 21:20 ET). PREMISE (primary
        // sources, the licensed 140-2017 text in docs/): 5.2.1.12.2
        // sanctions the Table 5-138 constant interior-distribution
        // fractions for programs that do not compute the interior
        // distribution internally - Floor 0.642, Ceiling 0.1678, E 0.038,
        // W 0.038, N 0.053, S 0.0276, Solar Lost through Windows 0.035
        // (the lost row is the round-91 flip-1 retransmission sink,
        // FLX_R91_RETRANS). The round-83 discriminator measured this
        // routing class STRUCTURALLY INERT only because the hourly
        // per-class advance was then discarded by the coupled sub-step;
        // since the round-84/85 per-class coupled sub-step default
        // (#4531/#4533) the class drive at the phi_c site below BINDS the
        // committed mass state, so the routing is live. The class solar
        // partition changes from the ISO C.3 conductance share
        // (s_c*w*phi_st) to the Table 5-138 surface fractions mapped to
        // classes (floor/roof/wall), renormalized over the surface rows
        // (sum 0.9664) so the class partition conserves phi_st exactly
        // after the retransmission sink. Internal-gain mass share
        // (phi_m*w) is untouched - Table 5-138 routes transmitted solar,
        // not internal gains. A falsy token (0/false/no/off) restores the
        // C.3 conductance-share partition for A/B archaeology only; any
        // other value behaves as unset, i.e. the Table 5-138 default.]
        let r91_t5138 = crate::util::env_bool::env_bool("FLX_R91_T5138", true);
        // [R93 FIX — §C.3 lag-site air-share retirement under the T5138
        // class route. The air-share term (1 − s_share)·h_tr_is·phi_st is
        // the air side of the §C.3 partition whose mass side is
        // s_share·phi_st (the r88 arm: air + mass = phi_st exactly). Under
        // the Table 5-138 default the per-class drives already carry
        // 1.0·phi_st (fractions renormalized over the 0.9664 surface-row
        // sum), so keeping the air term over-subscribes the composed
        // network by (1 − s_share)·phi_st with no primary-source basis:
        // ASHRAE 140-2017 §5.2.1.12 / Table 5-138 distribute transmitted
        // solar among interior surfaces plus the 0.035 window-loss row
        // only (no air row), and ISO 13790 Annex C feeds Φst to the
        // surface node with Φia = 0.5·Φint at the air node (eq. C.1 /
        // C.10 / C.11). The term retires ONLY where the T5138 class route
        // binds the mass advance (same structural conditions as the class
        // drive site: per-class arm enabled, class tensors present at the
        // zone, positive class areas); lumped zones keep the intact
        // s_share partition. Falsy FLX_R93_T5138_AIRSHARE (or falsy
        // FLX_R91_T5138) restores the pre-r93 composed route
        // bit-identically.]
        let r93_airshare_legacy = crate::util::env_bool::env_bool("FLX_R93_T5138_AIRSHARE", false);
        // [R94 — surface-pool component partition (loop round 94). The
        // T5138 class route carries the whole `phi_st` pool at the Table
        // 5-138 fractions, but those fractions are scoped to TRANSMITTED
        // SOLAR (ASHRAE 140-2017 §5.2.1.12.2); the pool's internal-gain
        // radiative component (`rad_frac·φ_int`) has no row in the table
        // and keeps the ISO 13790 §C.3 conductance partition (per-class
        // mass share `s_c`, air share `(1 − s_share)` through the lag
        // site). PREMISE (verified against both primary sources, same
        // standard as the round-93 retirement): §5.2.1.12.2's constant
        // fractions distribute transmitted solar among interior surfaces
        // only; §C.3's algebraic surface node passes its source to air
        // and mass by conductance share — the r88 arm's partition, which
        // stays the engine default on tensor-less zones. Under the
        // pre-R94 composed route the class drive silently re-routed the
        // internal-gain component with the solar fractions and retired
        // its air share: a routing no primary source covers. NEW
        // DEFAULT: at the class drive, only `phi_st_sol` follows the
        // Table 5-138 fractions; `phi_st_int` follows `s_c·w` per class;
        // at the lag site, the internal component's air share stays
        // live under the T5138 route (the solar component's air share
        // remains retired — round 93). Total pool deposition stays
        // exactly `phi_st` (conserving by construction). Falsy
        // FLX_R94_T5138_GAINS_C3 (0/false/no/off) restores the
        // r93-composed route bit-identically for A/B archaeology.]
        let r94_gains_c3 = crate::util::env_bool::env_bool("FLX_R94_T5138_GAINS_C3", true);
        let r93_perclass_on = crate::util::env_bool::env_bool("FLX_R84_PERCLASS", true);
        let mut r93_zone_t5138_live: Vec<bool> = vec![false; self.0.hvac.num_zones];
        if r91_t5138 && r93_perclass_on && ps_enabled && !r93_airshare_legacy {
            for (i93, c93) in r66_cls.iter().enumerate() {
                let Some((_g, _t, cls_area, _beam)) = c93.as_ref() else {
                    continue;
                };
                if cls_area.iter().sum::<f64>() <= 1e-9 {
                    continue;
                }
                let cond93 = &self.0.conduction;
                let tensors93 = [
                    &cond93.cm_wall,
                    &cond93.cm_roof,
                    &cond93.cm_floor,
                    &cond93.h_tr_ms_wall,
                    &cond93.h_tr_ms_roof,
                    &cond93.h_tr_ms_floor,
                ]
                .iter()
                .all(|o| o.as_ref().is_some_and(|v| v.as_ref().len() > i93));
                if tensors93 {
                    r93_zone_t5138_live[i93] = true;
                }
            }
        }
        let mut r88_share_zone: Vec<f64> = Vec::new();
        if r88_phist_share {
            let n_z88 = self.0.hvac.num_zones;
            r88_share_zone.resize(n_z88, 0.0);
            let h_is_all88 = self.0.conduction.h_tr_is.as_ref();
            let h_ms_all88 = self.0.conduction.h_tr_ms.as_ref();
            let cond88 = &self.0.conduction;
            let rd88 =
                |o: &Option<T>, i: usize| o.as_ref().and_then(|v| v.as_ref().get(i).copied());
            for i in 0..n_z88 {
                let h_is_i = h_is_all88.get(i).copied().unwrap_or(0.0);
                let h_ms_i = h_ms_all88.get(i).copied().unwrap_or(0.0);
                let (cm_w, cm_r, cm_f) = (
                    rd88(&cond88.cm_wall, i),
                    rd88(&cond88.cm_roof, i),
                    rd88(&cond88.cm_floor, i),
                );
                let (ms_w, ms_r, ms_f) = (
                    rd88(&cond88.h_tr_ms_wall, i),
                    rd88(&cond88.h_tr_ms_roof, i),
                    rd88(&cond88.h_tr_ms_floor, i),
                );
                match (cm_w, cm_r, cm_f, ms_w, ms_r, ms_f, r66_cls[i].as_ref()) {
                    (
                        Some(_a),
                        Some(_b),
                        Some(_c),
                        Some(d),
                        Some(e),
                        Some(f),
                        Some((_g, _t, cls_area, _beam)),
                    ) if cls_area.iter().sum::<f64>() > 1e-9 => {
                        let cls_ms88 = [d, e, f];
                        let total88: f64 = cls_area.iter().sum();
                        let mut share88 = 0.0;
                        for cc88 in 0..3 {
                            let w88 = cls_area[cc88] / total88;
                            if w88 <= 0.0 {
                                continue;
                            }
                            let h_is_c88 = h_is_i * w88;
                            let s_c88 = if cls_ms88[cc88] + h_is_c88 > 0.0 {
                                cls_ms88[cc88] / (cls_ms88[cc88] + h_is_c88)
                            } else {
                                0.0
                            };
                            share88 += w88 * s_c88;
                        }
                        r88_share_zone[i] = share88;
                    }
                    _ => {
                        r88_share_zone[i] = if h_ms_i + h_is_i > 0.0 {
                            h_ms_i / (h_ms_i + h_is_i)
                        } else {
                            0.0
                        };
                    }
                }
            }
        }

        // [R90 DEFAULT FLIP — loop round 90, Item 2 (lag-filter form) /
        // Item 1 (§LIMIT-40 storage-release split discriminator),
        // Alex-directed standing rule 2026-10-22: "anything that can be
        // verified to improve physics accuracy should be turned on by
        // default" — correct physics goes to the default on its own
        // merits, band outcomes being a downstream check. PREMISE
        // (reference-verified): ISO 13790 §C.3's surface node is
        // ALGEBRAIC — t_s solves the steady H_is/H_ms balance each
        // instant and partitions phi_st with no smoothing element; the
        // reference formulation contains no lag on the air share. The
        // incumbent R77 first-order filter (tau_lag = sqrt(tau_air ·
        // tau_mass) ≈ 2.0-2.1 h on an hourly-frozen input) is a synthetic
        // stand-in for surface-node dynamics that the engine's operator
        // already carries: the air node integrates with true C_air
        // (#1585) and the mass node advances at full time coverage per
        // class (round 89b), so the two-node network's continuous limit
        // is the ISO §C.3 two-node system — the filter adds a delay the
        // reference does not contain. NEW DEFAULT: the air share of
        // phi_st enters the air ODE unfiltered (instant §C.3 partition);
        // the lag STATE still advances (unchanged arithmetic) so the
        // persisted solar-lag state stays consistent. Setting
        // FLX_R90_NO_LAG_FILTER to a falsy token (0/false/no/off)
        // restores the pre-round-90 R77 filtered route for A/B
        // archaeology; any other value behaves as unset, i.e. the
        // unfiltered default. VERIFIED (round-90 arm battery on
        // 6d4c5263, flag-off arm bit-identical to the main-checkout
        // battery): 600FF Tmax 53.40 -> 53.84 C (gap 18.35 -> 17.68 pp),
        // 650FF Tmax 52.79 -> 53.09 C (17.45 -> 16.98 pp), FF Tmin rows
        // unchanged in-band (-12.87 / -17.63), high-mass family
        // byte-identical, 13 of 42 output-diff metrics moved (all
        // low-mass family, <= 1.36%), one strict-tracked row worsened
        // (800 cooling 1130.07 -> 1137.48 pp — the §LIMIT-14 GaugeSolver
        // structural row), re-recorded honestly in the same PR with
        // KNOWN_ISSUES documentation. (b) `FLX_R90_PHIST_RAMP` remains an
        // env-gated measurement harness, default off: the sub-hour ramp
        // of the gain drives is an interpolation assumption, not a
        // reference requirement, and its measurement moved FF Tmax the
        // WRONG way (600FF 53.40 -> 52.87, 650FF 52.79 -> 52.14) while
        // improving four conditioned rows — honest verdict recorded, not
        // adopted. NOSTORE arm (FLX_R88_PHIST_MASS_SHARE=0): the §C.3
        // surface-pool mass share is load-bearing — removing it moved
        // every target row the wrong way (600FF Tmax 53.40 -> 52.53,
        // 650FF 52.79 -> 51.85, both FF Tmin colder) with broad
        // conditioned regressions; refuted as a fix direction.]
        let r90_no_lag_filter = crate::util::env_bool::env_bool("FLX_R90_NO_LAG_FILTER", true);
        let r90_phist_ramp = crate::util::env_bool::env_bool("FLX_R90_PHIST_RAMP", false);
        let r90_phi_st_prev_ref = self.0.mass.r90_phi_st_prev.as_ref();
        let r90_phi_m_prev_ref = self.0.mass.r90_phi_m_prev.as_ref();

        for _step in 0..steps {
            // === Air-node ODE (exact exponential solution) ===
            // Issue #3370: scratch pool buffer instead of per-sub-step
            // `Vec::with_capacity(num_zones)`.
            {
                let buf = scratch.air_node_t_i_free.as_mut();
                for i in 0..self.0.hvac.num_zones {
                    let f_r75 = (_step as f64 + 0.5) / steps as f64;
                    // [R76 HARNESS] with the arm on, the mass→air numerator
                    // term uses this sub-step's mass temperature instead of
                    // the hour-start value frozen into `num_tm_ref`.
                    let r76_no_refresh = std::env::var("FLX_R76_NO_NUM_REFRESH").is_ok();
                    let mut num_i = if eff_mass_n > 0 && !r76_no_refresh {
                        self.0.conduction.derived_h_ms_is_prod.as_ref()[i] * r76_mass[i]
                            + num_rest_ref[i]
                    } else {
                        num_tm_ref[i] + num_rest_ref[i]
                    };
                    {
                        // [R71 HARNESS -> default, round 73] sub-hour driver
                        // ramp: midpoint-fraction linear interpolation
                        // between the previous and current hour's drivers.
                        // Zero prev values mean "first step" — seed, no ramp.
                        let f = (_step as f64 + 0.5) / steps as f64;
                        let num_prev = r71_num_prev_ref[i];
                        if num_prev != 0.0 {
                            num_i = num_prev + f * (num_i - num_prev);
                        }
                    }
                    // LIMIT-17 fix: night ventilation removes heat from the zone proportional
                    // to (T_zone - T_outdoor). The h_ve_night term was added to h_ext (den)
                    // but the corresponding heat removal from the air-node energy balance was
                    // missing. The iterative solver correctly applies this to phi_m; we add
                    // the outdoor-temperature contribution here for the 5R1C steady-state.
                    let nv_cur = if night_vent_active_now && i == 0 {
                        h_ve_night * outdoor_temp
                    } else {
                        0.0
                    };
                    // [R75 HARNESS] FLX_R75_VENT_RAMP: interpolate the
                    // night-vent numerator term across the sub-steps at the
                    // same midpoint fraction as the R71 driver ramp. Default
                    // (flag unset): the full current-hour term — the identical
                    // add as before this arm existed.
                    // [R77 HARNESS] coupled mode ramps the vent term per
                    // sub-step exactly once (R75 interpolation): the
                    // incumbent full-current-hour value in every sub-step is
                    // the same per-sub-step additive accumulation as the
                    // solar-lag term and diverges with N.
                    if nv_cur != 0.0
                        || (r77_coupled && r75_vent_has_prev)
                        || (r75_vent_ramp && r75_vent_has_prev)
                    {
                        let nv_applied = if (r77_coupled || r75_vent_ramp) && r75_vent_has_prev {
                            let nv_prev_i = r75_vent_term_prev_ref[i];
                            nv_prev_i + f_r75 * (nv_cur - nv_prev_i)
                        } else {
                            nv_cur
                        };
                        num_i += nv_applied;
                    }
                    let den_i = {
                        let f = (_step as f64 + 0.5) / steps as f64;
                        let den_prev = r71_den_prev_ref[i];
                        if den_prev != 0.0 {
                            den_prev + f * (den_ref[i] - den_prev)
                        } else {
                            den_ref[i]
                        }
                    };
                    // [R77 HARNESS] solar-lag driver: midpoint-frozen lag
                    // value enters the numerator exactly once per sub-step;
                    // lag STATE advances once per sub-step at dt_sub
                    // (composes to the hourly filter). Replaces the
                    // incumbent additive post-correction in coupled mode.
                    if r77_coupled {
                        let term_rest_1_r77 = term_rest_1_ref[i];
                        let den_true_r77 = if term_rest_1_r77 > 0.0 {
                            den_i / term_rest_1_r77
                        } else {
                            den_i
                        };
                        let h_tr_3_r77 = self.0.conduction.derived_h_tr_3.as_ref()[i];
                        let tau_air_r77 = if den_true_r77 > 0.0 && c_air_ref[i] > 0.0 {
                            c_air_ref[i] / den_true_r77
                        } else {
                            600.0
                        };
                        let tau_mass_r77 = if h_tr_3_r77 > 0.0 && cm_ref[i] > 0.0 {
                            cm_ref[i] / h_tr_3_r77
                        } else {
                            44000.0
                        };
                        let tau_lag_r77 = (tau_air_r77 * tau_mass_r77).sqrt();
                        // [R93 FIX] the air-share lag input retires where
                        // the T5138 class route carries the full pool: the
                        // classes already deposit 1.0·phi_st on the mass
                        // side, and Table 5-138 / ISO 13790 §C.3 route no
                        // transmitted solar to zone air. Falsy
                        // FLX_R93_T5138_AIRSHARE restores this term.
                        let r93_retire_airshare =
                            r93_zone_t5138_live.get(i).copied().unwrap_or(false);
                        let mut lag_input_r77 = if !r93_retire_airshare && term_rest_1_r77 > 0.0 {
                            h_tr_is_for_ti_free.as_ref()[i] * phi_st.as_ref()[i] / term_rest_1_r77
                        } else if r94_gains_c3 && r93_retire_airshare && term_rest_1_r77 > 0.0 {
                            // [R94] under the T5138 route the §C.3 air
                            // share of the INTERNAL-GAIN surface
                            // component stays live: Table 5-138 routes
                            // transmitted solar only, so only the solar
                            // component's air share retires here.
                            h_tr_is_for_ti_free.as_ref()[i] * phi_st_int.as_ref()[i]
                                / term_rest_1_r77
                        } else {
                            0.0
                        };
                        // [R88 ARM — keep only the ISO §C.3 air share
                        // H_is/(H_ms + H_is) of the surface pool in the lag
                        // filter; the mass share enters the mass advance
                        // below (same share value, so air + mass = phi_st).
                        // Flag off: never executes, bit-identical. R94:
                        // the (1 − s_share) scaling applies to the
                        // internal-gain component's air share on
                        // T5138-live zones too — the per-class `s_c·w`
                        // mass share uses the same zone share, so
                        // air + mass = phi_st_int exactly.]
                        if r88_phist_share
                            && term_rest_1_r77 > 0.0
                            && (!r93_retire_airshare || r94_gains_c3)
                        {
                            lag_input_r77 *= 1.0 - r88_share_zone[i];
                        }
                        // [R90 HARNESS (b)] ramp the filter input across
                        // the hour at the R71 midpoint fraction; zero prev
                        // phi_st = first step, seed with the current value.
                        if r90_phist_ramp && !r93_retire_airshare && term_rest_1_r77 > 0.0 {
                            let phi_st_prev_r90 =
                                r90_phi_st_prev_ref.get(i).copied().unwrap_or(0.0);
                            if phi_st_prev_r90 != 0.0 {
                                let f_r90 = (_step as f64 + 0.5) / steps as f64;
                                let mut lag_input_prev_r90 = h_tr_is_for_ti_free.as_ref()[i]
                                    * phi_st_prev_r90
                                    / term_rest_1_r77;
                                if r88_phist_share {
                                    lag_input_prev_r90 *= 1.0 - r88_share_zone[i];
                                }
                                lag_input_r77 = lag_input_prev_r90
                                    + f_r90 * (lag_input_r77 - lag_input_prev_r90);
                            }
                        }
                        let decay_r77 = if tau_lag_r77 > 0.0 && dt_sub > 0.0 {
                            (-dt_sub / tau_lag_r77).exp()
                        } else {
                            0.0
                        };
                        let decay_half_r77 = if tau_lag_r77 > 0.0 && dt_sub > 0.0 {
                            (-(dt_sub / (2.0 * tau_lag_r77))).exp()
                        } else {
                            0.0
                        };
                        let lag_old_r77 = scratch.air_node_solar_lag.as_ref()[i];
                        let lag_mid_r77 =
                            lag_old_r77 * decay_half_r77 + lag_input_r77 * (1.0 - decay_half_r77);
                        scratch.air_node_solar_lag.as_mut()[i] =
                            lag_old_r77 * decay_r77 + lag_input_r77 * (1.0 - decay_r77);
                        // [R90 DEFAULT FLIP] unfiltered ISO §C.3 air-share
                        // drive by default; the falsy token restores the R77
                        // filtered route for A/B archaeology.
                        num_i += if r90_no_lag_filter {
                            lag_input_r77
                        } else {
                            lag_mid_r77
                        } * term_rest_1_r77;
                    }
                    let steady = num_i / den_i;
                    let c_air_i = c_air_ref[i];
                    let t_air_old_i = scratch.air_node_t_air.as_ref()[i];
                    let tau_air = if den_i > 0.0 && term_rest_1_ref[i] > 0.0 {
                        c_air_i * term_rest_1_ref[i] / den_i
                    } else {
                        f64::INFINITY
                    };
                    let t_i_free_i = if c_air_i > 0.0 && tau_air.is_finite() && dt_sub > 0.0 {
                        if r74_ramp_exact {
                            // [R74 HARNESS] exact ramped-forcing solution;
                            // see the FLX_R74_RAMP_EXACT note above.
                            let nv_at = |ff: f64| -> f64 {
                                let cur = if night_vent_active_now && i == 0 {
                                    h_ve_night * outdoor_temp
                                } else {
                                    0.0
                                };
                                if r75_vent_ramp && r75_vent_has_prev {
                                    let p = r75_vent_term_prev_ref[i];
                                    p + ff * (cur - p)
                                } else {
                                    cur
                                }
                            };
                            let ramp = |f: f64, cur: f64, prev: f64| -> f64 {
                                if prev != 0.0 {
                                    prev + f * (cur - prev)
                                } else {
                                    cur
                                }
                            };
                            let f0 = _step as f64 / steps as f64;
                            let f1 = (_step as f64 + 1.0) / steps as f64;
                            let num_base = num_tm_ref[i] + num_rest_ref[i];
                            let s0 = (ramp(f0, num_base, r71_num_prev_ref[i]) + nv_at(f0))
                                / ramp(f0, den_ref[i], r71_den_prev_ref[i]);
                            let s1 = (ramp(f1, num_base, r71_num_prev_ref[i]) + nv_at(f1))
                                / ramp(f1, den_ref[i], r71_den_prev_ref[i]);
                            let g = (s1 - s0) / dt_sub;
                            let exponent = -dt_sub / tau_air;
                            s1 - tau_air * g + (t_air_old_i - s0 + tau_air * g) * exponent.exp()
                        } else {
                            let exponent = -dt_sub / tau_air;
                            steady + (t_air_old_i - steady) * exponent.exp()
                        }
                    } else {
                        steady
                    };
                    buf[i] = t_i_free_i;
                }
            }

            // === Issue #1860: Solar-lag correction ===
            let phi_st_ref = phi_st.as_ref();
            let h_tr_is_for_lag_ref = h_tr_is_for_ti_free.as_ref();
            // Issue #3370: `corrected_t_i_free` is also a scratch buffer.
            // `copy_from_slice` from the pre-correction values rather than
            // `mem::take` to keep the bytes identical to the prior loop
            // semantics (every entry is overwritten below by `+= …`).
            scratch
                .air_node_corrected
                .as_mut()
                .copy_from_slice(scratch.air_node_t_i_free.as_ref());
            let corrected_t_i_free = scratch.air_node_corrected.as_mut();

            // [R77 HARNESS] coupled mode: the solar-lag state advanced
            // inside the ODE loop above (per-sub-step driver, no additive
            // post-correction) — this block is skipped.
            for i in 0..(if r77_coupled {
                0
            } else {
                self.0.hvac.num_zones
            }) {
                // [R71 HARNESS] same sub-hour den ramp as the ODE loop, so
                // den_true in the solar-lag correction sees the same driver
                // pair as the steady state.
                let den_i = {
                    let f = (_step as f64 + 0.5) / steps as f64;
                    let den_prev = r71_den_prev_ref[i];
                    if den_prev != 0.0 {
                        den_prev + f * (den_ref[i] - den_prev)
                    } else {
                        den_ref[i]
                    }
                };
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

                let new_solar_lag =
                    scratch.air_node_solar_lag.as_ref()[i] * decay + lag_input * (1.0 - decay);

                if den_true_i > 0.0 {
                    corrected_t_i_free[i] += new_solar_lag / den_true_i;
                }

                scratch.air_node_solar_lag.as_mut()[i] = new_solar_lag;
            }

            // [R76 HARNESS] sub-step mass advance: Crank–Nicolson at dt_sub
            // per zone, exterior term ramped at the midpoint fraction,
            // interior boundary = this sub-step's air temperature. When the
            // arm is off this block never runs (r76_n == 0).
            //
            // [R87 FLIP (see the h_vent_i site below): the mass-side
            // night-vent coupling on the live sub-step path is read here
            // once per step; FLX_R86_VENT_AIR_ONLY=0 restores the
            // pre-round-87 coupling.]
            let vent_air_only = crate::util::env_bool::env_bool("FLX_R86_VENT_AIR_ONLY", true);
            let r76_no_mass = std::env::var("FLX_R76_NO_MASS").is_ok();
            if eff_mass_n > 0 && !r76_no_mass {
                let mass_sub = eff_mass_n;
                // [R89 HARNESS — mass-node time-coverage alignment
                // (FLX_R89_MASS_DT_ALIGN, loop round 89, Item 2a). Premise
                // verified in code: `mass_sub` divides dt_sub but NO inner
                // loop iterates it — the sub-step mass advance executes
                // exactly ONE Crank–Nicolson step per sub-step at
                // dt_m = dt_sub/N = 25 s (N=12), so the mass node
                // integrates N·25 s = 300 s of physics per hour (1/12 of
                // real time) while the air node integrates the full
                // 3600 s. The R77 scheme's documented property — the
                // N→∞ limit is the continuous-time solution of the 5R1C
                // network — therefore does NOT hold for the mass state:
                // unaligned mass coverage is 3600/N s per hour, so its
                // N→∞ limit is a frozen mass, not the continuous system.
                // This arm aligns the mass CN step with the air sub-step
                // interval (dt_m = dt_sub), one CN advance per sub-step
                // over the full sub-step: mass coverage 3600 s/hour and
                // the coupled scheme converges, for both nodes, to the
                // continuous system. The r77 affine recorder consumes the
                // same dt_m local, so the checker mirror stays exact by
                // construction. Alex-directed 2026-10-22 (round 89):
                // anything verifiable to improve physics accuracy goes to
                // default — this arm lands env-gated first and flips on
                // verification. Flag off (default): bit-identical.]
                // [DEFAULT FLIP — loop round 89, Item 2a, Alex-directed
                // standing rule 2026-10-22: "anything that can be verified to
                // improve physics accuracy should be turned on by default" —
                // correct physics goes to the default on its own merits, band
                // outcomes being a downstream check. PREMISE (code-verified):
                // the coupled sub-step mass advance executed exactly ONE CN
                // application per sub-step at dt_m = dt_sub/N (N=12 ⇒ 25 s),
                // so the mass node integrated N·25 s = 300 s of physics per
                // hour — 1/12 of real time — while the air node integrated the
                // full 3600 s; the R77 scheme's N→∞ limit therefore did NOT
                // hold for the mass state (its limit depended on N). THE FIX
                // (new default): the mass advance runs mass_sub composed CN
                // applications at dt_m = dt_sub/mass_sub, i.e. full 3600 s/hour
                // coverage at the engine's established 25 s step accuracy; the
                // affine composition is recorded so the checker mirror stays
                // exact by construction. VERIFIED by the round-89 A/B/C
                // decomposition: coarse full coverage (B: dt_m = dt_sub, one
                // step) and inner-loop full coverage (C) agree to ~0.3-0.5%
                // everywhere — the movement is coverage physics, not step-size
                // numerics. Honest re-record: both FF Tmin rows ENTER their
                // published bands for the first time (600FF −12.85 in
                // [−13.80, −9.90]; 650FF −17.62 in [−17.80, −16.70]), FF Tmax
                // gaps roughly halve, conditioned cooling converges strongly
                // (600 C 31.10 → 3.92 pp); conditioned heating rows regress
                // and are recorded in KNOWN_ISSUES, never tuned away. The
                // falsy "0" token restores the legacy 1/12-coverage route for
                // A/B archaeology only; "1" (coarse single-step coverage) and
                // "2" (explicit inner-loop) remain harness values.]
                let r89_mode = std::env::var("FLX_R89_MASS_DT_ALIGN").unwrap_or_default();
                let r89_truthy = crate::util::env_bool::env_bool("FLX_R89_MASS_DT_ALIGN", true);
                let r89_inner_n: u32 = if !r89_truthy || r89_mode == "1" {
                    1
                } else {
                    mass_sub as u32
                };
                let dt_m = if r89_truthy && r89_inner_n == 1 {
                    dt_sub
                } else {
                    dt_sub / mass_sub as f64
                };
                let f_mid = (_step as f64 + 0.5) / steps as f64;
                let h_tr_3_all = self.0.conduction.derived_h_tr_3.as_ref();
                let h_is_all = self.0.conduction.h_tr_is.as_ref();
                let h_ms_all = self.0.conduction.h_tr_ms.as_ref();
                // [R85 FLIP — the per-class coupled sub-step mass advance
                // is the engine default since loop round 85 (Alex-directed
                // 2026-10-21 21:18 ET: "Flip the per class mass bind"; the
                // arm was verified env-gated in round 84, PR #4531).
                // FLX_R84_PERCLASS set to a falsy token (0/false/no/off)
                // opts back out to the pre-85 lumped sub-step advance for
                // A/B measurement; any other value behaves as unset, i.e.
                // the per-class default.]
                let r84_perclass = crate::util::env_bool::env_bool("FLX_R84_PERCLASS", true);
                for i in 0..self.0.hvac.num_zones {
                    let t_ext = if r76_has_prev {
                        let p = r76_sol_air_prev_ref[i];
                        p + f_mid * (t_sol_air[i] - p)
                    } else {
                        t_sol_air[i]
                    };
                    let h_tr_3_i = h_tr_3_all.get(i).copied().unwrap_or(0.0);
                    let h_air_mass_i = if h_tr_3_i > 0.0 {
                        h_tr_3_i
                    } else {
                        let h_is = h_is_all.get(i).copied().unwrap_or(0.0);
                        let h_ms = h_ms_all.get(i).copied().unwrap_or(0.0);
                        if h_is + h_ms > 0.0 {
                            h_is * h_ms / (h_is + h_ms)
                        } else {
                            h_ms
                        }
                    };
                    // [R87 FLIP — mass-side night-vent coupling off by
                    // default (Alex-approved 2026-10-22 06:41 ET: "I
                    // approve the vent arm"). Night ventilation is an
                    // air-side exchange: ASHRAE 140 §5.4 / Table 5-160
                    // define the Case 650/950 night-vent fan as outdoor
                    // air supplied to the ZONE, and ISO 13790 §C.4 Eq.
                    // C.10 routes ventilation only through the air node —
                    // the air-side h_ve_night conductance in h_ext
                    // (Issue #824/#839) implements exactly that. This
                    // h_vent_i term adds the same exchange a second time,
                    // mass-side, folded into the h3 air↔mass coupling:
                    // #821's own note records that setting the mass-side
                    // term to 0 removes the double-counting, the 9R4C
                    // path has run air-only since LIMIT-17
                    // (multi_node_solver h_vent_mass_zone = 0 pattern),
                    // and the round-86 audit measured it at 3.1x the
                    // air-side exchange on 650FF (-20,469 vs -6,509
                    // MJ/yr) while chilling the mass below freezing
                    // overnight. FLX_R86_VENT_AIR_ONLY=0 restores the
                    // pre-round-87 coupling for A/B verification only;
                    // any other value behaves as unset, i.e. the
                    // air-only default.]
                    let h_vent_i = if night_vent_active_now && i == 0 && !vent_air_only {
                        h_ve_night
                    } else {
                        0.0
                    };
                    let t_i_sub = scratch.air_node_t_air.as_ref()[i];
                    let h_tr_em_i = h_tr_em_zone.get(i).copied().unwrap_or(0.0);
                    let h_tr3_arg_i = h_air_mass_i + h_vent_i;
                    // [R84 HARNESS — per-class coupled sub-step mass
                    // advance (env-gated, FLX_R84_PERCLASS, loop round
                    // 84). Round 83 proved the hourly per-class mass
                    // advance is discarded on the coupled-N default: the
                    // R76 sub-stepped lumped mass state (r76_final)
                    // overrides it, so the Table 5-138 routing class is
                    // unreachable, not refuted. This arm moves the
                    // per-class decomposition INTO the sub-step: each
                    // zone's mass advances per class at dt_m with the
                    // same Crank-Nicolson integrator the lumped path
                    // uses — per-class capacitance cm_c (wall/roof/floor
                    // constructions, [cm; 3] fallback), per-class
                    // exterior driving (cls_g/cls_t R66 aggregates,
                    // floor class exterior-free), interior coupling
                    // h_am_c = series h_ms_c*h_is_c/(h_ms_c + h_is_c),
                    // or the degenerate h_air_mass*w fallback when
                    // H_tr,3 = 0, each partitioned by area share w
                    // (sum w = 1 over classes, so coupling, phi_m and
                    // night-vent partition energy-conservingly). The
                    // sub-step end state is the capacitance-weighted
                    // mean sum (cm_c/cm_sum)*tm_c. Disclosed
                    // simplifications vs. the hourly D5 route: the
                    // sub-step uses the algebraic-series interior
                    // coupling (no t_s-first surface-node sub-state),
                    // and the zone-level sol-air midpoint ramp source is
                    // not carried per class — class exterior temps are
                    // the current-hour cls_t values. The R77 folded
                    // air-side term stays the lumped aggregate
                    // recording (unchanged), so the arm isolates the
                    // mass state. Default on since loop round 85
                    // (Alex-directed 2026-10-21 21:18 ET); setting
                    // FLX_R84_PERCLASS to a falsy token measures the pre-85
                    // lumped sub-step path (bit-identical to the round-84
                    // flag-off state).]
                    let mut perclass_coeffs: Option<(f64, f64, f64)> = None;
                    // [R85 GUARD — the per-class advance requires real
                    // per-class construction state for this zone. The class
                    // tensors (cm_wall/cm_roof/cm_floor,
                    // h_tr_ms_wall/roof/floor) are assigned only by the
                    // spec-built constructor (from_spec stores them for every
                    // construction type since the D5 default, round 71).
                    // Hand-rolled models (ThermalModel::new plus scalar
                    // conductances) carry none: replicating the lumped
                    // scalar across three classes and re-partitioning the
                    // exterior driving through the class aggregates is NOT
                    // equivalent to the lumped advance there. Round-85 A/B
                    // evidence: CZ3, edge-case and metering regressions with
                    // the arm engaged on tensor-less zones (bounded state,
                    // negative energy accounting, 1 kW gains invisible).
                    // Zones without the tensors keep the lumped sub-step
                    // advance — the pre-85 state — so the arm commits class
                    // physics only where class constructions exist.]
                    let cls_tensors_ready = {
                        let cond = &self.0.conduction;
                        let has_i =
                            |o: &Option<T>| o.as_ref().is_some_and(|v| v.as_ref().len() > i);
                        has_i(&cond.cm_wall)
                            && has_i(&cond.cm_roof)
                            && has_i(&cond.cm_floor)
                            && has_i(&cond.h_tr_ms_wall)
                            && has_i(&cond.h_tr_ms_roof)
                            && has_i(&cond.h_tr_ms_floor)
                    };
                    // [R88 ARM — the lumped sub-step drive gains the ISO
                    // §C.3 mass share of the surface pool for zones on the
                    // lumped advance (tensor-less, or FLX_R84_PERCLASS
                    // opted out): s·phi_st with s = h_ms/(h_ms + h_is) —
                    // the same zone share the air-side lag filter consumed
                    // (r88_share_zone[i] holds exactly this value for
                    // tensor-less zones). The r77 coefficient recorder
                    // below consumes the same value so the persisted
                    // affine record mirrors the committed state exactly.
                    // Flag off: phi_m only, bit-identical.]
                    let phi_lumped_88 = if r88_phist_share {
                        let h_is_i88 = h_is_all.get(i).copied().unwrap_or(0.0);
                        let h_ms_i88 = h_ms_all.get(i).copied().unwrap_or(0.0);
                        let s88 = if h_ms_i88 + h_is_i88 > 0.0 {
                            h_ms_i88 / (h_ms_i88 + h_is_i88)
                        } else {
                            0.0
                        };
                        phi_m_ref_early[i] + s88 * phi_st.as_ref()[i]
                    } else {
                        phi_m_ref_early[i]
                    };
                    let tm_new = if r84_perclass && ps_enabled && cls_tensors_ready {
                        match r66_cls[i] {
                            Some((cls_g, cls_t_c, cls_area, _cls_beam))
                                if cls_area.iter().sum::<f64>() > 1e-9 =>
                            {
                                let total_area: f64 = cls_area.iter().sum();
                                // [R85] affine-map accumulators for the
                                // checker composite record (see the r77
                                // coefficient push below).
                                let mut a_sum = 0.0;
                                let mut b_sum = 0.0;
                                let mut d_sum = 0.0;
                                let cond = &self.0.conduction;
                                let grab = |o: &Option<T>| {
                                    o.as_ref().and_then(|v| v.as_ref().get(i).copied())
                                };
                                let (cm_w, cm_r, cm_f) = (
                                    grab(&cond.cm_wall),
                                    grab(&cond.cm_roof),
                                    grab(&cond.cm_floor),
                                );
                                let (ms_w, ms_r, ms_f) = (
                                    grab(&cond.h_tr_ms_wall),
                                    grab(&cond.h_tr_ms_roof),
                                    grab(&cond.h_tr_ms_floor),
                                );
                                let (cls_cm, cls_ms) = match (cm_w, cm_r, cm_f, ms_w, ms_r, ms_f) {
                                    (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => {
                                        ([a, b, c], [d, e, f])
                                    }
                                    _ => (
                                        [cm_ref[i]; 3],
                                        [h_ms_all.get(i).copied().unwrap_or(0.0); 3],
                                    ),
                                };
                                let h_is_i = h_is_all.get(i).copied().unwrap_or(0.0);
                                let degenerate = h_tr_3_i <= 0.0;
                                let cm_sum: f64 = cls_cm.iter().sum();
                                let phi_m_i = phi_m_ref_early[i];
                                let mut tm_w = 0.0;
                                for c in 0..3 {
                                    let w = cls_area[c] / total_area;
                                    if w <= 0.0 {
                                        continue;
                                    }
                                    let (h_em_c, t_ext_c) = if c == 2 {
                                        (0.0, outdoor_temp)
                                    } else {
                                        (cls_g[c], cls_t_c[c])
                                    };
                                    let h_ms_c = cls_ms[c];
                                    let h_is_c = h_is_i * w;
                                    let h_am_c = if degenerate {
                                        h_air_mass_i * w
                                    } else if h_ms_c + h_is_c > 0.0 {
                                        h_ms_c * h_is_c / (h_ms_c + h_is_c)
                                    } else {
                                        h_ms_c
                                    };
                                    // [R85] Bind the exact per-class CN
                                    // arguments and derive this class's
                                    // affine map from the same algebra the
                                    // call below consumes (including its
                                    // forward-Euler fallback), so the
                                    // persisted composite mirrors the
                                    // committed state exactly
                                    // (checker-lags-integrator, #4504
                                    // class).
                                    let h3_c = h_am_c + h_vent_i * w;
                                    // [R88 ARM — the class mass drive gains
                                    // the ISO §C.3 mass share of the surface
                                    // pool: s_c·phi_st·w with
                                    // s_c = h_ms_c/(h_ms_c + h_is_i·w) —
                                    // the same s_c the zone share sum above
                                    // consumed, so the air-side (1 − Σ w·s_c)
                                    // and the mass-side shares partition
                                    // phi_st. Flag off: phi_m·w unchanged,
                                    // bit-identical.]
                                    // [R91 FLIP 3 - the class solar term
                                    // carries the Table 5-138 surface
                                    // fractions instead of the C.3
                                    // conductance share: floor class
                                    // 0.642/0.9664, roof class
                                    // 0.1678/0.9664, wall class
                                    // (E+W+N+S) 0.1566/0.9664. The
                                    // renormalization absorbs the
                                    // published table's rounding so
                                    // sum_c phi_c = phi_m_i + phi_st
                                    // exactly (energy-conserving by
                                    // construction; the checker
                                    // composite mirrors this drive
                                    // through the affine maps below).
                                    // [R93] the §C.3 lag-site air-share
                                    // term retires under this route (see
                                    // FLX_R93_T5138_AIRSHARE) so the
                                    // composed network deposits exactly
                                    // phi_st.]
                                    // Falsy FLX_R91_T5138 restores the
                                    // s_c*w conductance-share partition
                                    // (the R88 arm) bit-identically.]
                                    // [R94 — the §C.3 per-class share of
                                    // the internal-gain surface
                                    // component: `s_c = h_ms_c/(h_ms_c +
                                    // h_is·w)`, the same share the r88
                                    // arm and the zone air-share sum
                                    // consume, so air + mass = phi_st_int
                                    // exactly across the decomposition.]
                                    let h_is_c94 = h_is_i * w;
                                    let s_c94 = if h_ms_c + h_is_c94 > 0.0 {
                                        h_ms_c / (h_ms_c + h_is_c94)
                                    } else {
                                        0.0
                                    };
                                    let phi_c = if r91_t5138 {
                                        let f_c_t5138 = match c {
                                            2 => 0.642_f64,
                                            1 => 0.1678,
                                            _ => 0.1566,
                                        };
                                        if r94_gains_c3 {
                                            // [R94 DEFAULT — the Table
                                            // 5-138 fractions carry the
                                            // TRANSMITTED-SOLAR component
                                            // only; the internal-gain
                                            // radiative component keeps
                                            // the §C.3 conductance
                                            // partition (`s_c·w` here,
                                            // `(1 − s_share)` at the lag
                                            // site).]
                                            phi_m_i * w
                                                + (f_c_t5138 / 0.9664) * phi_st_sol.as_ref()[i]
                                                + s_c94 * phi_st_int.as_ref()[i] * w
                                        } else {
                                            // Pre-R94 composed route
                                            // (whole pool at the solar
                                            // fractions) — archaeology.
                                            phi_m_i * w + (f_c_t5138 / 0.9664) * phi_st.as_ref()[i]
                                        }
                                    } else if r88_phist_share {
                                        let h_is_c88 = h_is_i * w;
                                        let s_c88 = if h_ms_c + h_is_c88 > 0.0 {
                                            h_ms_c / (h_ms_c + h_is_c88)
                                        } else {
                                            0.0
                                        };
                                        phi_m_i * w + s_c88 * phi_st.as_ref()[i] * w
                                    } else {
                                        phi_m_i * w
                                    };
                                    // [R90 HARNESS (b)] ramp the class mass
                                    // drive across the hour at the R71
                                    // midpoint fraction (same share algebra
                                    // on the previous hour's gains; zero prev
                                    // = seed, no ramp).
                                    let phi_c = if r90_phist_ramp {
                                        let phi_m_prev_r90 =
                                            r90_phi_m_prev_ref.get(i).copied().unwrap_or(0.0);
                                        let phi_st_prev_r90 =
                                            r90_phi_st_prev_ref.get(i).copied().unwrap_or(0.0);
                                        if phi_m_prev_r90 != 0.0 || phi_st_prev_r90 != 0.0 {
                                            let f_r90m = (_step as f64 + 0.5) / steps as f64;
                                            let h_is_c_r90 = h_is_i * w;
                                            let s_c_prev_r90 =
                                                if r88_phist_share && h_ms_c + h_is_c_r90 > 0.0 {
                                                    h_ms_c / (h_ms_c + h_is_c_r90)
                                                } else {
                                                    0.0
                                                };
                                            let phi_c_prev_r90 = phi_m_prev_r90 * w
                                                + s_c_prev_r90 * phi_st_prev_r90 * w;
                                            phi_c_prev_r90 + f_r90m * (phi_c - phi_c_prev_r90)
                                        } else {
                                            phi_c
                                        }
                                    } else {
                                        phi_c
                                    };
                                    let cm_dt_c = cls_cm[c] / dt_m;
                                    let half_c = 0.5 * (h3_c + h_em_c);
                                    let den_c = cm_dt_c + half_c;
                                    let numer_rest_c = h_em_c * t_ext_c + h3_c * t_i_sub + phi_c;
                                    let (a_c, b_c) = if den_c > 0.0 {
                                        ((cm_dt_c - half_c) / den_c, numer_rest_c / den_c)
                                    } else {
                                        let inv = dt_m / cls_cm[c];
                                        (1.0 - inv * (h3_c + h_em_c), inv * numer_rest_c)
                                    };
                                    if ff_audit && _step == steps - 1 {
                                        eprintln!(
                                            "FFAUDIT_C {} {} {} w={:.6} cm_c={:.6} a={:.6} b={:.6} h_am={:.6} h_em={:.6} h3={:.6} t_ext={:.6} phi={:.6} tm_prev={:.6}",
                                            timestep, i, c, w, cls_cm[c], a_c, b_c, h_am_c, h_em_c, h3_c, t_ext_c, phi_c, r76_mass[i]
                                        );
                                    }
                                    let w_cap = cls_cm[c] / cm_sum;
                                    a_sum += w_cap * a_c;
                                    b_sum += w_cap * b_c;
                                    d_sum += w_cap * den_c;
                                    let tm_c_new = crank_nicolson_iso13790(
                                        r76_mass[i],
                                        dt_m,
                                        cls_cm[c],
                                        h3_c,
                                        h_em_c,
                                        t_ext_c,
                                        t_i_sub,
                                        phi_c,
                                    )?;
                                    if ff_audit {
                                        eprintln!(
                                            "FFAUDIT_D {} {} {} {:.9}",
                                            timestep,
                                            i,
                                            c,
                                            cls_cm[c] * (tm_c_new - r76_mass[i])
                                        );
                                    }
                                    tm_w += w_cap * tm_c_new;
                                }
                                perclass_coeffs = Some((a_sum, b_sum, d_sum));
                                tm_w
                            }
                            _ => crank_nicolson_iso13790(
                                r76_mass[i],
                                dt_m,
                                cm_ref[i],
                                h_tr3_arg_i,
                                h_tr_em_i,
                                t_ext,
                                t_i_sub,
                                phi_lumped_88,
                            )?,
                        }
                    } else {
                        crank_nicolson_iso13790(
                            r76_mass[i],
                            dt_m,
                            cm_ref[i],
                            h_tr3_arg_i,
                            h_tr_em_i,
                            t_ext,
                            t_i_sub,
                            phi_lumped_88,
                        )?
                    };
                    // [Round 78; round 85 per-class composite] record this
                    // sub-step's linear coefficients — same branch the
                    // integrator took: CN when its denominator is positive,
                    // its forward-Euler fallback otherwise (both affine in
                    // tm_prev). When the per-class arm produced the state,
                    // the recorded triple is the capacitance-weighted
                    // composite of the per-class affine maps,
                    // (A, B, D) = Σ_c (cm_c/cm_sum)·(a_c, b_c, den_c), so
                    // the checker's composite mirror stays exact by
                    // construction (checker-lags-integrator, #4504 class).
                    let mut applied_inner = false;
                    if r77_coupled {
                        let (a, b, denom_m) = if let Some((a_s, b_s, d_s)) = perclass_coeffs {
                            (a_s, b_s, d_s)
                        } else {
                            let cm_dt_m = cm_ref[i] / dt_m;
                            let half_cond = 0.5 * (h_tr3_arg_i + h_tr_em_i);
                            let denom_m = cm_dt_m + half_cond;
                            let numer_rest =
                                h_tr_em_i * t_ext + h_tr3_arg_i * t_i_sub + phi_lumped_88;
                            let (a, b) = if denom_m > 0.0 {
                                ((cm_dt_m - half_cond) / denom_m, numer_rest / denom_m)
                            } else {
                                let inv = dt_m / cm_ref[i];
                                (1.0 - inv * (h_tr3_arg_i + h_tr_em_i), inv * numer_rest)
                            };
                            (a, b, denom_m)
                        };
                        let c = &mut r77_coeffs[i];
                        // [R89 mode "2": compose the affine map N times. The
                        // driving is held constant across the sub-step, so the
                        // per-sub-step map f(tm) = a*tm + b iterated N times is
                        // the affine map (a^N, b*(1-a^N)/(1-a)); recording the
                        // composed pair keeps the checker mirror exact by
                        // construction (same formula both sides).]
                        if r89_inner_n > 1 {
                            let n = r89_inner_n;
                            let an = a.powi(n as i32);
                            let one_minus_an = 1.0 - an;
                            let b_eff = if (1.0 - a).abs() < 1e-15 {
                                b * n as f64
                            } else {
                                b * one_minus_an / (1.0 - a)
                            };
                            c.push(an);
                            c.push(b_eff);
                            c.push(denom_m);
                            r76_mass[i] = an * r76_mass[i] + b_eff;
                            applied_inner = true;
                        } else {
                            c.push(a);
                            c.push(b);
                            c.push(denom_m);
                        }
                    }
                    if !applied_inner {
                        r76_mass[i] = tm_new;
                    }
                }
            }
            // Update t_air_state for next sub-step
            scratch
                .air_node_t_air
                .as_mut()
                .copy_from_slice(scratch.air_node_corrected.as_ref());
        }
        if r76_n > 0 || r77_coupled {
            // Persist the sol-air ramp source for the next hour and carry
            // the end-of-hour mass state to the mass-advance block.
            let sol_prev = self.0.mass.r76_sol_air_prev.as_mut();
            for (i, slot) in sol_prev.iter_mut().take(self.0.hvac.num_zones).enumerate() {
                *slot = t_sol_air[i];
            }
            self.0.mass.r76_has_prev = true;
            r76_final = Some(r76_mass.clone());
        }

        // Issue #4495: input-envelope guard. Extreme-but-finite inputs
        // (|T_outdoor| ≳ ~1e306 °C, dt ≳ ~1e29 s) overflow the f64
        // conductance × driving-temperature products in the network
        // numerator, so the air-node advance below can be ±inf even though
        // every input passed the finite/dt>0 checks. Without this guard the
        // non-finite value propagates into the committed zone state
        // (`mass.air_temperatures`, `setpoints.temperatures`) and corrupts
        // every subsequent step. Same graceful-degradation contract as the
        // Issue #3638 degenerate-dt/cm validation: typed `Validation` error,
        // state left untouched; the legacy `step_physics` wrapper logs and
        // returns 0.0 kWh for the step. Inputs far outside the solver's
        // finite envelope are *rejected*, never clamped — no silent tuning.
        for (i, &t) in scratch
            .air_node_t_air
            .as_ref()
            .iter()
            .enumerate()
            .take(self.0.hvac.num_zones)
        {
            if !t.is_finite() {
                // Issue #4495 rejection path must restore the pooled scratch
                // (Issue #2756 contract): a leaked checkout empties the pool
                // and the next step re-allocates (scratch_pool reuse gate).
                self.0.hvac.scratch_pool.return_5r1c(scratch);
                return Err(FluxionError::Validation(format!(
                    "Air-node temperature became non-finite ({t}) for zone {i}: \
                     the step's inputs are outside the solver's finite envelope \
                     (issue #4495); step rejected, state untouched"
                )));
            }
        }

        // [R71 HARNESS] persist this hour's drivers as the next hour's ramp
        // source. Stored without the night-vent term (the vent addition is
        // re-applied at its current-hour value inside the sub-step loop).
        // Write happens after the sub-step loop, where the immutable
        // `self.0.mass` borrows above are dead — same placement contract as
        // the `air_temperatures` commit below.
        {
            let num_prev = self.0.mass.r71_air_driver_num_prev.as_mut();
            let den_prev = self.0.mass.r71_air_driver_den_prev.as_mut();
            for i in 0..self.0.hvac.num_zones {
                num_prev[i] = num_tm_ref[i] + num_rest_ref[i];
                den_prev[i] = den_ref[i];
            }
            // [R75 HARNESS] persist this hour's night-vent term as the next
            // hour's ramp source; the seed flag flips after the first step
            // (mirrors the R71 zero-prev seed convention).
            {
                let vent_prev = self.0.mass.r75_vent_term_prev.as_mut();
                for (i, slot) in vent_prev.iter_mut().take(self.0.hvac.num_zones).enumerate() {
                    *slot = if night_vent_active_now && i == 0 {
                        h_ve_night * outdoor_temp
                    } else {
                        0.0
                    };
                }
                self.0.mass.r75_vent_has_prev = true;
            }
            // [R90 HARNESS] persist this hour's gain drives as the next
            // hour's ramp source. Unconditional write: the flag-off build
            // never reads these fields, so it stays bit-identical.
            {
                let st_prev_r90 = self.0.mass.r90_phi_st_prev.as_mut();
                let m_prev_r90 = self.0.mass.r90_phi_m_prev.as_mut();
                let n = self.0.hvac.num_zones;
                st_prev_r90[..n].copy_from_slice(&phi_st.as_ref()[..n]);
                m_prev_r90[..n].copy_from_slice(&phi_m.as_ref()[..n]);
            }
        }

        // After all sub-steps, air_node_t_air holds the final air-node
        // temperature and air_node_solar_lag holds the final solar-lag state.
        // Issue #3370: wrap the scratch SmallVec directly via `from_smallvec`
        // instead of `VectorField::new(Vec)` (which had been the largest
        // per-step allocation in the regressed hot loop).
        let t_i_free = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.air_node_t_air,
        )));

        // Persist final solar-lag state
        self.0.mass.solar_lag.as_mut()[..self.0.hvac.num_zones]
            .copy_from_slice(&scratch.air_node_solar_lag.as_ref()[..self.0.hvac.num_zones]);

        // Issue #1585: step the air-node ODE state forward for the next
        // timestep.  t_i_free (the new zone-air temperature) becomes
        // t_air_old on the next call to step_physics_5r1c.
        // Issue #3370: scratch pool buffer (`air_node_t_i_free_slice`)
        // replaces the per-step `t_i_free.as_ref().to_vec()` allocation.
        scratch
            .air_node_t_i_free_slice
            .as_mut()
            .copy_from_slice(t_i_free.as_ref());
        self.0
            .mass
            .air_temperatures
            .as_mut()
            .copy_from_slice(&scratch.air_node_t_i_free_slice.as_ref()[..self.0.hvac.num_zones]);

        // The wall-surface state contributes to the air-node numerator through
        // the transient surface flux correction applied above.

        // PR #821: DEBUG_900FF_ti_free trace removed.

        // PR #821: DEBUG_MAX trace for 600FF/650FF removed; use `pr821-diag` feature instead.

        // PR #821: DEBUG_650FF_FULL traces removed.

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
            // Issue #4241: discrete residual formulation needs T_prev and dt.
            let t_prev_slice: &[f64] = scratch.air_node_t_prev.as_ref();
            // Note: borrow checker requires we copy the slice reference before
            // the mutable borrow of hvac_combined_demand below. The slice
            // remains valid because air_node_t_prev is not modified by the call.
            let dt_copy = dt;
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                self.0.setpoints.heating_setpoint,
                self.0.setpoints.cooling_setpoint,
                t_prev_slice,
                dt_copy,
                &mut scratch.hvac_combined_demand,
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
            // Issue #4241: discrete residual formulation needs T_prev and dt.
            let t_prev_slice2: &[f64] = scratch.air_node_t_prev.as_ref();
            let dt_copy2 = dt;
            let hvac_output = self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                heating_setpoint,
                cooling_setpoint,
                t_prev_slice2,
                dt_copy2,
                &mut scratch.hvac_combined_demand,
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
            // Issue #4241: discrete residual formulation needs T_prev and dt.
            let t_prev_slice3: &[f64] = scratch.air_node_t_prev.as_ref();
            let dt_copy3 = dt;
            let hvac_output_raw = self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                self.0.setpoints.heating_setpoint,
                self.0.setpoints.cooling_setpoint,
                t_prev_slice3,
                dt_copy3,
                &mut scratch.hvac_combined_demand,
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
            // Issue #4241: discrete residual formulation needs T_prev and dt.
            let t_prev_slice4: &[f64] = scratch.air_node_t_prev.as_ref();
            let dt_copy4 = dt;
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.setpoints.heating_setpoints.as_ref(),
                self.0.setpoints.cooling_setpoints.as_ref(),
                self.0.setpoints.heating_setpoint,
                self.0.setpoints.cooling_setpoint,
                t_prev_slice4,
                dt_copy4,
                &mut scratch.hvac_combined_demand,
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
        let heating_setpoint_vec = self.0.setpoints.heating_setpoints.as_ref();
        let cooling_setpoint_vec = self.0.setpoints.cooling_setpoints.as_ref();
        let default_heating_sp = self.0.setpoints.heating_setpoint;
        let default_cooling_sp = self.0.setpoints.cooling_setpoint;
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
            // LIMIT-33 (issue #4314, Option A — state feedback): the #4241
            // residual load is derived by requiring T^{n+1} = T_sp in the
            // implicit-Euler air-node balance. Under that derivation the
            // computed load must actually drive the air-node state to the
            // setpoint. When the demand is active and unclamped (ideal
            // system, capacity sufficient), hold the zone air at the active
            // setpoint. When capacity-clamped the setpoint is unreachable;
            // keep the residual partial update t_free + Q/den_true, which
            // bounds the feedback loop.
            let heat_cap = self.0.hvac.hvac_heating_capacity;
            let cool_cap = self.0.hvac.hvac_cooling_capacity;
            let clamped = hvac[i] >= heat_cap || hvac[i] <= -cool_cap;
            if den_true > 0.0 && hvac[i].abs() > 1e-6 {
                if clamped {
                    scratch.t_i_act[i] = t_free[i] + hvac[i] / den_true;
                } else if hvac[i] > 0.0 {
                    scratch.t_i_act[i] = heating_setpoint_vec
                        .get(i)
                        .copied()
                        .unwrap_or(default_heating_sp);
                } else {
                    scratch.t_i_act[i] = cooling_setpoint_vec
                        .get(i)
                        .copied()
                        .unwrap_or(default_cooling_sp);
                }
            } else {
                scratch.t_i_act[i] = t_free[i];
            }
        }
        let t_i_act = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.t_i_act,
        )));

        // LIMIT-33 (issue #4314, Option A — state feedback): the controlled
        // state is persisted in `setpoints.temperatures` (assigned below),
        // where the next step's `air_node_t_prev` copy reads it and where the
        // strict-energy invariant gate already reads the controlled
        // temperature. `mass.air_temperatures` keeps holding the free-float
        // air state (the convention the invariant checker and the gauge
        // mass-state proxy rely on), so free-floating diagnostics are
        // unaffected and the mass-node energy balance stays auditable.

        // Use hvac_for_temp_calc for energy (matches what was used for temperature update)
        // This ensures energy calculation is consistent with temperature physics
        let mut heating_sum = 0.0;
        let mut cooling_sum = 0.0;
        let mut total_signed = 0.0;

        // Per-zone energy accumulation (Issue #1288)
        // hvac_for_temp_calc: positive = heating, negative = cooling
        let hvac_vec = hvac_for_temp_calc.as_ref();

        // Issue #4500: input-envelope guard on the HVAC demand itself. The
        // ideal loads can overflow to ±inf for extreme-but-finite `dt` even
        // though every input passed the finite checks and the #4495
        // temperature guards passed (the demand is a conductance ×
        // temperature-difference product whose magnitude scales with the
        // implicit step). Check the demand and its dt product BEFORE any
        // energy accumulation or state commit, mirroring the #4495
        // leave-state-untouched contract.
        {
            // The returned/accumulated energy scales with the SUM of the
            // zone demands, not the per-zone max — bound the product with
            // the zone count so a multi-zone sum cannot slip past the
            // per-zone check (issue #4500, crash-5447b7a2).
            let w_scale = hvac_vec.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            let w_bound = w_scale * self.0.hvac.num_zones as f64;
            if hvac_vec.iter().any(|&v| !v.is_finite()) || !(w_bound * dt).is_finite() {
                self.0.hvac.scratch_pool.return_5r1c(scratch);
                return Err(FluxionError::Validation(format!(
                    "HVAC demand became non-finite or overflows the step energy \
                     (max |W| = {w_scale}, dt = {dt} s): the step's inputs are \
                     outside the solver's finite envelope (issue #4500); step \
                     rejected, state untouched"
                )));
            }
        }

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

        // === Issue #4155: zone moisture balance + ideal-system latent load ===
        //
        // The ideal system that delivers the sensible load above also
        // dehumidifies: supply air leaves the cooling coil saturated at 13 °C,
        // so the supply mass flow implied by the sensible load removes
        // moisture at h_fg(T_zone). The latent load joins the total cooling
        // energy (the ideal-system load is a total energy); the latent part
        // is tracked separately in zone_latent_cooling_energy_kwh /
        // annual_latent_cooling_energy.
        //
        // The moisture state and latent path are required for correct
        // ideal-system physics in humid climates (e.g. Baltimore, Miami —
        // 98.7% of Miami cooling hours exceed the 13 °C coil ADP vs 3.0%
        // in Denver). In dry climates the annual latent is negligible,
        // which is the correct physical result, not a bug.
        //
        // In free-float mode the moisture state still advances (the
        // psychrometric invariant must hold every step) but no latent energy
        // is booked — Issues #738/#821 require free-float HVAC output to be
        // exactly zero.
        {
            let n = self.0.hvac.num_zones;
            // Outdoor humidity ratio from the current step's EPW weather
            // (the annual loop calls set_weather per step; humidity is the
            // parsed EPW relative humidity).
            let w_out = self
                .0
                .solar
                .weather
                .as_ref()
                .map(|w| {
                    calculate_humidity_ratio(
                        w.dry_bulb_temp,
                        w.humidity,
                        STANDARD_ATMOSPHERIC_PRESSURE_Pa,
                    )
                    .max(0.0)
                })
                .unwrap_or(0.0);
            // Per-zone ventilation conductance for moisture transport: the
            // pure infiltration h_ve (derived_h_ext also carries h_tr_w, which
            // moves heat but not moisture) plus the active night-ventilation
            // flow on zone 0. Stack-allocated for the usual zone counts —
            // no per-step heap allocation (Issue #2873 perf discipline).
            let mut h_ve_moist: SmallVec<[f64; 4]> =
                self.0.conduction.h_ve.as_ref().iter().copied().collect();
            if night_vent_active_now {
                if let Some(first) = h_ve_moist.first_mut() {
                    *first += h_ve_night;
                }
            }
            let mut latent_w: SmallVec<[f64; 4]> = SmallVec::new();
            step_zone_moisture(
                self.0.hvac.zone_humidity_ratio.as_mut(),
                t_i_act.as_ref(),
                hvac_vec,
                self.0.setpoints.cooling_setpoints.as_ref(),
                h_ve_moist.as_slice(),
                self.0.setpoints.zone_volume.as_ref(),
                w_out,
                dt,
                &mut latent_w,
            );
            if !self.0.hvac.free_float {
                let zone_latent_slice = self.0.hvac.zone_latent_cooling_energy_kwh.as_mut();
                let mut latent_sum = 0.0;
                for i in 0..n {
                    let q_lat = latent_w[i];
                    latent_sum += q_lat;
                    // dt is in seconds, convert to kWh: watts * seconds / 3.6e6
                    let e_lat_kwh = q_lat * dt / 3.6e6;
                    zone_latent_slice[i] += e_lat_kwh;
                    zone_cooling_slice[i] += e_lat_kwh;
                }
                self.0.hvac.annual_latent_cooling_energy += latent_sum * dt / 3.6e6;
                // Latent joins the total cooling energy below via cooling_sum.
                cooling_sum += latent_sum;
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
        // Issue #3063 — `h_tr_em` is recomputed per-timestep inside the loop
        // from material R-values and the wind-dependent exterior film. The
        // legacy construction-time vector is no longer needed.
        let h_tr_ms_ref = self.0.conduction.h_tr_ms.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let t_i_act_ref = t_i_act.as_ref();
        // [Round 73 (#1522/#4504) — checker persistence.] Reuse the applied
        // class-boundary buffer across steps (allocation only on the first
        // step; same dhat-budget pattern as the r71 driver persistence).
        // Filled per zone in the mass loop below; restored after it.
        let mut applied_class_buf: Vec<
            Option<crate::sim::thermal_model_data::Applied5R1CClassBoundary>,
        > = std::mem::take(&mut self.0.conduction.applied_5r1c_class_boundary).unwrap_or_default();
        applied_class_buf.clear();
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
                // Issue #4166 — wind-dependent `h_tr_em` (debug-physics
                // snapshot mirrors the production BE/CN integrators above).
                let h_tr_em_i = h_tr_em_zone.get(i).copied().unwrap_or(0.0);
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
            // h_tr_em = Σ A_s / (R_materials,s + 1/h_c_ext,s)  [Issue #4166]
            //   Aggregated per-surface from the exterior boundary (orientation,
            //   tilt, sky view factor, windward/leeward convection). Previously
            //   recomputed per-timestep as wall+roof with the
            //   `HorizontalRoofWindward` convention [Issue #3063].
            // h_tr_ms = k * A / d (thermal conductivity * area / thickness)
            // Issue #4166 — per-surface-aggregated `h_tr_em` from
            // `scratch.h_tr_em_zone` (populated in the per-surface sol-air
            // loop above).
            let h_tr_em = h_tr_em_zone.get(i).copied().unwrap_or(0.0);
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
            // double-counting. (The original note claimed this restores the
            // Case 650FF peak free-float temperature "to within the ASHRAE 140
            // reference band [63.2, 73.5] °C" — that band was the pre-round-33
            // value of unverified origin; the published Table B8-5 650FF Tmax
            // band is [61.1, 66.8] °C and the measured peak remains 54.30 °C,
            // below it. The mechanism claim stands; the band claim did not.)
            //
            // The air-side ventilation increase under night-vent is still routed
            // through `phi_m_with_vent`/`phi_ia_with_vent` further down (where the
            // outdoor-air enthalpy difference is applied via `h_ve` × ΔT).
            // [R87 FLIP — night ventilation is an air-side exchange on the
            // 5R1C path (Alex-approved 2026-10-22 06:41 ET: "I approve the
            // vent arm"): the mass-side coupling h_vent_mass_zone is now 0 by
            // default. ASHRAE 140 §5.4 / Table 5-160 define the Case 650/950
            // night-vent fan as outdoor air supplied to the ZONE, and ISO
            // 13790 §C.4 Eq. C.10 routes ventilation only through the air
            // node — the air-side h_ve_night conductance in h_ext (Issue
            // #824/#839) already implements exactly that. The mass-side term
            // double-counts the same exchange: #821's own note ("setting
            // h_vent_mass_zone = 0 removes the double-counting"), restored at
            // full h_ve_night by 2674f00c (2026-07-17, #1693 lineage), and
            // measured by the round-86 audit at 3.1x the air-side exchange on
            // 650FF (-20,469 vs -6,509 MJ/yr) while chilling the mass below
            // freezing overnight. The 9R4C path has run this same air-only
            // routing since LIMIT-17 (multi_node_solver h_vent_mass_zone = 0
            // pattern); this flip aligns 5R1C with it. FLX_R86_VENT_AIR_ONLY
            // set to a falsy token (0/false/no/off) restores the pre-round-87
            // mass coupling for A/B verification only; any other value
            // behaves as unset, i.e. the air-only default.]
            let vent_air_only = crate::util::env_bool::env_bool("FLX_R86_VENT_AIR_ONLY", true);
            let h_vent_mass_zone = if night_vent_active_now && !vent_air_only {
                h_ve_night
            } else {
                0.0
            };

            // [R66 HARNESS — env-gated per-surface mass-node advance (scratch).
            // Each class node: capacitance cm·w, exterior conductance cls_g[c]
            // driven by its own class sol-air (floor: no exterior term, mirrors
            // the lumped path), interior coupling h_air_mass·w, night-vent
            // coupling h_vent_mass_zone·w, gain share phi_m·w. The air side
            // reads the capacitance-weighted mean, and Σ w = 1 with interior
            // shares proportional to w, so the zone-level coupling is
            // energy-conserving vs. the lumped node.]
            let advance_node = |cm_n: f64,
                                h_em_n: f64,
                                t_ext_n: f64,
                                h_am_n: f64,
                                h_vm_n: f64,
                                phi_m_n: f64,
                                t_s_n: f64,
                                t_i_n: f64|
             -> Result<f64, FluxionError> {
                match select_integration_method(cm_n) {
                    ThermalIntegrationMethod::BackwardEuler => {
                        let cm_dt = cm_n / dt;
                        Ok((cm_dt * tm_old
                            + h_em_n * t_ext_n
                            + h_am_n * t_s_n
                            + h_vm_n * outdoor_temp
                            + phi_m_n)
                            / (cm_dt + h_em_n + h_am_n + h_vm_n))
                    }
                    ThermalIntegrationMethod::ExplicitEuler => {
                        let q_vent_mass = h_vm_n * (outdoor_temp - tm_old);
                        let q_m_net = h_em_n * (t_ext_n - tm_old)
                            + h_am_n * (t_s_n - tm_old)
                            + phi_m_n
                            + q_vent_mass;
                        Ok(tm_old + (q_m_net / cm_n) * dt)
                    }
                    ThermalIntegrationMethod::CrankNicolson => crank_nicolson_iso13790(
                        tm_old,
                        dt,
                        cm_n,
                        h_am_n + h_vm_n,
                        h_em_n,
                        t_ext_n,
                        t_i_n,
                        phi_m_n,
                    ),
                }
            };
            let tm_new = if ps_enabled {
                match r66_cls[i] {
                    Some((cls_g, cls_t, cls_area, cls_beam))
                        if cls_area.iter().sum::<f64>() > 1e-9 =>
                    {
                        let total_area: f64 = cls_area.iter().sum();
                        // [R67 HARNESS — per-surface interior film/surface
                        // network (scratch). When FLX_R67_INTERIOR is set and
                        // the real per-class constructions are stored, each
                        // class advances with its own series (h_ms_c, h_is_c)
                        // interior pair, its own algebraic surface node
                        // t_s_c = (h_ms_c·T_m + h_is_c·T_i + phi_st_c) /
                        // (h_ms_c + h_is_c), and its real construction
                        // capacitance cm_c; the zone-level read is the
                        // capacitance-weighted mean (energy-conserving). The
                        // area share w partitions h_is (same film coefficient
                        // per m²), phi_st, phi_m and the night-vent coupling.
                        // Without the flag the R66 form applies unchanged.]
                        let (cls_cm, cls_ms) = if r67_interior {
                            let cond = &self.0.conduction;
                            let grab =
                                |o: &Option<T>| o.as_ref().and_then(|v| v.as_ref().get(i).copied());
                            let cm_w = grab(&cond.cm_wall);
                            let cm_r = grab(&cond.cm_roof);
                            let cm_f = grab(&cond.cm_floor);
                            let ms_w = grab(&cond.h_tr_ms_wall);
                            let ms_r = grab(&cond.h_tr_ms_roof);
                            let ms_f = grab(&cond.h_tr_ms_floor);
                            match (cm_w, cm_r, cm_f, ms_w, ms_r, ms_f) {
                                (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => {
                                    ([a, b, c], [d, e, f])
                                }
                                _ => ([cm; 3], [h_tr_ms_ref[i]; 3]),
                            }
                        } else {
                            ([cm; 3], [h_tr_ms_ref[i]; 3])
                        };
                        // Structural equality on construction-derived f64s:
                        // the values are copied from the same source vectors,
                        // so bit-equality (not epsilon) is the intended test —
                        // it distinguishes "all three classes identical" from
                        // "distinct real constructions".
                        #[allow(clippy::float_cmp)]
                        let use_r67 = r67_interior
                            && cls_cm.iter().sum::<f64>() > 1e-9
                            && (cls_ms[0] != cls_ms[1]
                                || cls_ms[1] != cls_ms[2]
                                || cls_cm[0] != cls_cm[1]
                                || cls_cm[1] != cls_cm[2]);
                        let cm_sum: f64 = cls_cm.iter().sum();
                        let h_is_zone = h_tr_is_ref_2.get(i).copied().unwrap_or(0.0);
                        let phi_st_zone = phi_st_ref[i];
                        // [R68 HARNESS — beam-weighted phi_st solar partition
                        // (scratch, loop round 68, #3983). FLX_R68_BEAM_PHIST
                        // splits the zone-level phi_st into its solar component
                        // (remaining_sol × st_sol_frac, recomputed from the same
                        // zone solar terms used to build it) and its internal-
                        // radiative component; the solar component partitions by
                        // per-class incident irradiance×area (Issue #864
                        // convention, area fallback when the zone sees no sun),
                        // the internal component stays area-weighted. Σ_c
                        // phi_st_c = phi_st_zone (energy-conserving). Without the
                        // flag the R67 area-only partition applies unchanged.]
                        let r68_beam_phist = std::env::var("FLX_R68_BEAM_PHIST").is_ok();
                        let phi_st_sol = if r68_beam_phist {
                            self.0
                                .solar
                                .solar_gains
                                .as_ref()
                                .get(i)
                                .copied()
                                .unwrap_or(0.0)
                                * self
                                    .0
                                    .setpoints
                                    .zone_area
                                    .as_ref()
                                    .get(i)
                                    .copied()
                                    .unwrap_or(0.0)
                                * (1.0 - self.0.solar.solar_distribution_to_air)
                                * (1.0 - self.0.solar.solar_beam_to_mass_fraction)
                        } else {
                            0.0
                        };
                        let beam_sum: f64 = if r68_beam_phist {
                            cls_beam.iter().sum()
                        } else {
                            0.0
                        };
                        let mut tm_w = 0.0;
                        // [R70 HARNESS — t_s state read/write, see the
                        // FLX_R70_TS_ADVANCE note inside the class loop.
                        // Old values snapshot before the loop; new values
                        // written back after `tm_new` is bound.]
                        // [D5 DEFAULT (loop round 71, #3983 §57): the t_s-first
                        // advance (round-70 arm, FLX_R70_TS_ADVANCE) is on: the
                        // per-class surface node is advanced first by backward
                        // Euler consuming its own previous state, and the class
                        // mass advance consumes that t_s. Disclosed structural
                        // assumption carried from the round-70 record: the
                        // interior surface layer capacitance C_s_c = 0.1·cm_c.]
                        let r70_ts_on = true;
                        let n_z70 = self.0.hvac.num_zones;
                        // [Round 78 (#4512): the per-step `to_vec` + `clone`
                        // pair the D5 default flip made here (~2 heap
                        // allocations per timestep, breached #4512's budget
                        // while the dashboard dhat job was skipped) is pooled
                        // in `MassState` workspaces and reused. Values
                        // identical (clear + extend/resize reuses the
                        // allocation; contents equal the old copies).]
                        let mut r70_ts_old: Vec<f64> =
                            self.0.mass.r70_ts_old_workspace.take().unwrap_or_default();
                        r70_ts_old.clear();
                        match self.0.mass.r70_surface_node_temperatures.as_ref() {
                            Some(v) if v.as_ref().len() >= 3 * n_z70 => {
                                r70_ts_old.extend_from_slice(&v.as_ref()[..3 * n_z70])
                            }
                            _ => r70_ts_old.resize(3 * n_z70, 20.0),
                        }
                        let mut r70_ts_new: Vec<f64> =
                            self.0.mass.r70_ts_new_workspace.take().unwrap_or_default();
                        r70_ts_new.clear();
                        r70_ts_new.extend_from_slice(&r70_ts_old);
                        for c in 0..3 {
                            let w = cls_area[c] / total_area;
                            if w <= 0.0 {
                                continue;
                            }
                            let (h_em_c, t_ext_c) = if c == 2 {
                                (0.0, outdoor_temp)
                            } else {
                                (cls_g[c], cls_t[c])
                            };
                            if use_r67 {
                                let h_is_c = h_is_zone * w;
                                let h_ms_c = cls_ms[c];
                                // [R68 HARNESS — discriminating experiment
                                // (scratch, loop round 68, #3983). FLX_R68_DEGEN_LUMPED
                                // keeps the per-class interior network (t_s_c, cm_c,
                                // h_ms_c/h_is_c surface node) but replaces the per-class
                                // series coupling conductance with the zone-level
                                // `h_air_mass` — the exact degenerate H_tr,3 = 0
                                // fallback conductance the lumped path uses —
                                // partitioned by w (Σ w = 1, energy-conserving). If
                                // Case 195 cooling recovers toward 0.851 MWh, the
                                // regression came through the degenerate-fallback
                                // coupling rewrite; if it stays near 0.995, it comes
                                // from per-surface interior coupling generally.
                                // Non-degenerate zones are untouched.]
                                // [D5 DEFAULT (loop round 71, #3983 §57, Option
                                // A): a degenerate zone (H_tr,3 = 0 fallback)
                                // advances all three classes with the
                                // per-class-lumped h_air_mass·w interior
                                // coupling instead of the per-class series pair.
                                // Round-70 Option A record: recovers the Case
                                // 195 discriminator at zero cost elsewhere;
                                // round-71 degen+t_s battery: 195 C 1.219 →
                                // 0.931 MWh with every other case unchanged.]
                                let r68_degen_lumped =
                                    h_tr_3_ref_2.get(i).copied().unwrap_or(0.0) <= 0.0;
                                // Beam-weighted phi_st_c replaces the area-only
                                // phi_st_zone*w when the R68 arm is on (see the
                                // R68 HARNESS note above).
                                let phi_st_c = if r68_beam_phist {
                                    (phi_st_zone - phi_st_sol) * w
                                        + if beam_sum > 1e-9 {
                                            phi_st_sol * cls_beam[c] / beam_sum
                                        } else {
                                            phi_st_sol * w
                                        }
                                } else {
                                    phi_st_zone * w
                                };
                                let (h_am_c, t_s_c) = if h_ms_c + h_is_c > 0.0 {
                                    (
                                        h_ms_c * h_is_c / (h_ms_c + h_is_c),
                                        (h_ms_c * tm_old + h_is_c * t_i + phi_st_c)
                                            / (h_ms_c + h_is_c),
                                    )
                                } else {
                                    (h_ms_c, t_i)
                                };
                                let h_am_eff = if r68_degen_lumped {
                                    h_air_mass * w
                                } else {
                                    h_am_c
                                };
                                let cm_c = cls_cm[c];
                                // [D5 DEFAULT — t_s-first advance (loop round 71
                                // promotion of the round-70 arm, #3983 §57). The
                                // per-class surface node
                                // a step of real dynamics: it is advanced FIRST by
                                // backward Euler on its own heat balance,
                                //   C_s·dT_s/dt = h_is_c·(t_i − T_s) + h_ms_c·(T_m − T_s)
                                //                 + phi_st_c,
                                // consuming its own previous state (t_s_old), and the
                                // mass node then advances consuming that T_s. Round 68
                                // proved the phi_st beam weighting structurally inert on
                                // the production route because every real construction
                                // capacitance selects Crank–Nicolson, and
                                // `crank_nicolson_iso13790` consumes `t_i` only — the
                                // algebraic `t_s_c` it is handed is dropped. ISO 13790's
                                // own 5R1C keeps the surface node algebraic (§C.3), so
                                // this arm is a deliberate structural extension toward
                                // the #1522 dynamic-order requirement, not an ISO
                                // conformance change. Disclosed structural assumption,
                                // harness-only: C_s_c = 0.1·cm_c (interior surface
                                // layer ≈ 10 % of construction capacitance); the class
                                // mass integration switches to the BE form above so
                                // T_s actually binds. Flag off = R67 arm unchanged.]
                                let r70_ts_advance = r70_ts_on;
                                let (t_s_eff, be_mass) = if r70_ts_advance {
                                    let cs_c = 0.1 * cm_c;
                                    let t_s_old = r70_ts_old[3 * i + c];
                                    let cs_dt = cs_c / dt;
                                    let denom_s = cs_dt + h_is_c + h_ms_c;
                                    let t_s_new = if denom_s > 0.0 {
                                        (cs_dt * t_s_old
                                            + h_is_c * t_i
                                            + h_ms_c * tm_old
                                            + phi_st_c)
                                            / denom_s
                                    } else {
                                        t_s_c
                                    };
                                    r70_ts_new[3 * i + c] = t_s_new;
                                    (t_s_new, true)
                                } else {
                                    (t_s_c, false)
                                };
                                let tm_c_new = if be_mass {
                                    let cm_dt = cm_c / dt;
                                    (cm_dt * tm_old
                                        + h_em_c * t_ext_c
                                        + h_am_eff * t_s_eff
                                        + h_vent_mass_zone * w * outdoor_temp
                                        + phi_m_zone * w)
                                        / (cm_dt + h_em_c + h_am_eff + h_vent_mass_zone * w)
                                } else {
                                    advance_node(
                                        cm_c,
                                        h_em_c,
                                        t_ext_c,
                                        h_am_eff,
                                        h_vent_mass_zone * w,
                                        phi_m_zone * w,
                                        t_s_eff,
                                        t_i,
                                    )?
                                };
                                tm_w += (cm_c / cm_sum) * tm_c_new;
                            } else {
                                tm_w += w * advance_node(
                                    cm * w,
                                    h_em_c,
                                    t_ext_c,
                                    h_air_mass * w,
                                    h_vent_mass_zone * w,
                                    phi_m_zone * w,
                                    t_s,
                                    t_i,
                                )?;
                            }
                        }
                        // [D5 DEFAULT — write the advanced per-class
                        // surface-node state back (see the t_s-first note).]
                        if r70_ts_on {
                            let mut sv = smallvec::SmallVec::new();
                            sv.extend_from_slice(&r70_ts_new);
                            let v = T::from(VectorField::from_smallvec(sv));
                            self.0.mass.r70_surface_node_temperatures = Some(v);
                            // [Round 73 (#1522/#4504)] Persist the `t_s_old`
                            // snapshot this advance consumed so the checker
                            // can recompute the surface-node identity
                            // exactly. Write-only for the engine.
                            let mut svp = smallvec::SmallVec::new();
                            svp.extend_from_slice(&r70_ts_old);
                            self.0.mass.r70_surface_node_temperatures_prev =
                                Some(T::from(VectorField::from_smallvec(svp)));
                        }
                        // [Round 78 (#4512): return the t_s workspaces.]
                        self.0.mass.r70_ts_old_workspace = Some(r70_ts_old);
                        self.0.mass.r70_ts_new_workspace = Some(r70_ts_new);
                        tm_w
                    }
                    _ => advance_node(
                        cm,
                        h_tr_em,
                        t_sol_air[i],
                        h_air_mass,
                        h_vent_mass_zone,
                        phi_m_zone,
                        t_s,
                        t_i,
                    )?,
                }
            } else {
                match method {
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
                        // Issue #3638: propagate a typed validation error (degenerate
                        // dt/cm) instead of letting a panic abort the simulation.
                        crank_nicolson_iso13790(
                            tm_old,
                            dt,
                            cm,
                            h_tr_3_with_vent,
                            h_tr_em,
                            t_sol_air[i],
                            t_i,
                            phi_m_zone,
                        )?
                    }
                }
            };

            // [R76 HARNESS] with the arm on, the end-of-hour mass state comes
            // from the sub-step advances above and the hourly lumped/per-class
            // advance result is discarded (the r70 surface-node advance and the
            // applied-boundary recording above still run, recording the same
            // current-hour aggregates — under the arm the per-class
            // decomposition stays hourly by design). Flag-off: untouched.
            let tm_new = if let Some(mv) = r76_final.as_ref() {
                mv[i]
            } else {
                tm_new
            };

            scratch.new_mass[i] = tm_new;

            // [Round 73 (#1522/#4504) — checker persistence.] Record the
            // per-class exterior aggregates and the per-zone scalars this
            // step's mass advance actually consumed, so
            // `invariant_checker` mirrors the post-D5 per-class identity
            // from applied values (same pattern as
            // `applied_5r1c_boundary`, #4373). Write-only for the engine.
            match r66_cls[i] {
                Some((cls_g, cls_t, cls_area, cls_beam)) => {
                    applied_class_buf.push(Some(
                        crate::sim::thermal_model_data::Applied5R1CClassBoundary {
                            cls_g,
                            cls_t,
                            cls_area,
                            cls_beam,
                            h_air_mass,
                            t_i,
                            phi_m_zone,
                            phi_st_zone: phi_st_ref[i],
                            h_vent_mass_zone,
                        },
                    ));
                }
                None => applied_class_buf.push(None),
            }
            if ff_audit {
                eprintln!(
                    "FFAUDIT_Z {} {} t_out={:.6} t_sky={:.6} t_g={:.6} t_sol_air={:.6} h_tr_em={:.6} t_air={:.6} t_m_new={:.6} tm_start={:.6} cm={:.6} c_air={:.6} g_coeff={:.6} h_air_mass={:.6} h_ext={:.6} h_ve_night={:.6} nv={} h_vent_mass={:.6} phi_m={:.6} phi_st={:.6} lag={:.6} solar_gains={:.6} loads={:.6} term_rest_1={:.6} h_tr_is={:.6} s_share={:.6}",
                    timestep, i,
                    outdoor_temp, sky_temp, t_g,
                    scratch.t_sol_air_zone.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    scratch.h_tr_em_zone.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    // The pooled air-node buffer is drained back to the model
                    // before this site; the committed `air_temperatures` is
                    // the same final value.
                    self.0.mass.air_temperatures.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    scratch.new_mass.get(i).copied().unwrap_or(f64::NAN),
                    self.0.mass.mass_temperatures.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    cm_ref[i],
                    self.0.mass.air_thermal_capacitance.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    self.0.conduction.derived_ground_coeff.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    h_air_mass,
                    scratch.h_ext_owned_zone.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    h_ve_night,
                    night_vent_active_now as u8,
                    h_vent_mass_zone,
                    phi_m_zone,
                    phi_st_ref[i],
                    scratch.air_node_solar_lag.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    self.0.solar.solar_gains.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    self.0.setpoints.loads.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    self.0.conduction.derived_term_rest_1.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    // [R93 AUDIT] h_tr_is (the lag-site coupling actually
                    // consumed by the air-share term) + s_share (the r88
                    // mass share) let the harness compute the exact
                    // air-side phi_st excess (1 − s_share)·h_tr_is·phi_st.
                    h_tr_is_for_ti_free.as_ref().get(i).copied().unwrap_or(f64::NAN),
                    r88_share_zone.get(i).copied().unwrap_or(f64::NAN),
                );
            }
        }
        self.0.conduction.applied_5r1c_class_boundary = Some(applied_class_buf);
        // [Round 78] the checker record persists only for a step that
        // ran the coupled advance AND passed both input-envelope guards
        // above (a rejected step leaves state untouched, so it must also
        // leave no record for the checker to mirror).
        self.0.mass.r77_mass_substep_coeffs = if r77_coupled { Some(r77_coeffs) } else { None };
        // [Round 78 (#4512): return the per-class tuple scratch to the model.]
        self.0.conduction.r66_cls_workspace = Some(r66_cls);

        // Issue #4495: same input-envelope guard for the mass node — the
        // Crank–Nicolson / implicit-Euler numerator multiplies conductances
        // by the (possibly extreme) sol-air and outdoor driving temperatures,
        // so the mass advance can overflow to ±inf for the same inputs. Must
        // run before the mass-state commit below so a rejected step leaves
        // the state untouched.
        for (i, &t) in scratch.new_mass.iter().enumerate() {
            if !t.is_finite() {
                // Same pooled-scratch restoration as the air-node guard above.
                self.0.hvac.scratch_pool.return_5r1c(scratch);
                return Err(FluxionError::Validation(format!(
                    "Mass-node temperature became non-finite ({t}) for zone {i}: \
                     the step's inputs are outside the solver's finite envelope \
                     (issue #4495); step rejected, state untouched"
                )));
            }
        }

        // Update the mass temperatures with new values (convert Vec to T type)
        let new_mass_temps_vf: T =
            VectorField::from_smallvec(std::mem::take(&mut scratch.new_mass)).into();

        // Plan 03-04: Update previous mass temperature for tracking (kept for diagnostic output)
        // Mass energy change tracking removed - Ti_free already includes thermal mass effects
        self.0.mass.previous_mass_temperatures =
            std::mem::replace(&mut self.0.mass.mass_temperatures, new_mass_temps_vf);

        // Store previous temperatures for dT/dt calculation (Plan 15-04, 15-06)
        // Issue #3370: scratch pool buffer (`previous_temperatures`)
        // replaces the per-step `VectorField::new(...to_vec())` allocation.
        // `from_smallvec` consumes the scratch by ownership so the buffer
        // capacity is retained for the next step (no heap allocation once
        // the pool is warm).
        scratch
            .previous_temperatures
            .as_mut()
            .copy_from_slice(self.0.setpoints.temperatures.as_ref());
        self.0.hvac.previous_temperatures =
            VectorField::from_smallvec(std::mem::take(&mut scratch.previous_temperatures));
        self.0.setpoints.temperatures = t_i_act;

        // [Issue #4555 — thermostat sensor node (TRV sensor temperature).]
        // Observational wiring: the sensor nodes READ the committed zone air
        // temperature and the per-zone HVAC/radiator output and write ONLY
        // their own state. They never feed back into the zone energy
        // balance, so enabling them cannot change any simulation output.
        // The bank is `None` by default, so this block is inert unless a
        // caller attaches sensor nodes explicitly.
        {
            // Disjoint field borrows: `mass` (sensor state) and `setpoints`
            // (zone air temperature) are separate fields of the model data.
            let crate::sim::thermal_model_data::ThermalModelData {
                mass, setpoints, ..
            } = &mut self.0;
            if let Some(sensor_bank) = mass.sensor_nodes.as_mut() {
                sensor_bank.step_watts(
                    dt,
                    setpoints.temperatures.as_ref(),
                    hvac_output_raw.as_ref(),
                    outdoor_temp,
                );
            }
        }

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

        Ok(net_hvac_energy_for_step / 3.6e6) // Return kWh
    }
}

//! 9R4C physics step implementation for `ThermalModel`.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::physics::multi_node_solver::SurfaceExteriorTemperatures;
use crate::sim::boundary::distribute_opaque_solar_gains;
use crate::sim::hvac::{HVACMode as EquipmentHVACMode, VariableCapacityEquipment};
use crate::sim::sky_radiation::SolAirTemperature;
use crate::sim::solar::calculate_surface_irradiance;
use crate::sim::thermal_model_core::ThermalModel;
use crate::sim::ventilation::capped_h_tr_is_ach_multiplier;
use smallvec::SmallVec;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    pub(crate) fn step_physics_9r4c(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> f64 {
        let dt = dt_seconds;

        // Get ground temperature at this timestep
        let t_g = self
            .0
            .conduction
            .ground_temperature
            .ground_temperature(timestep);

        // Calculate sky temperature for sol-air calculation
        let sky_temp = self
            .0
            .solar
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);

        // === Issue #1279: Night ventilation forced convection (ACH-dependent) ===
        // When night ventilation fans are active, the natural convection
        // h_tr_is coefficient is insufficient to represent FORCED convection.
        //
        // We compute the ACH from fan_capacity and zone_volume, then apply the
        // ASHRAE/EnergyPlus empirical correlation for forced convection:
        //   h_c = h_c_still + 0.84 * ACH^0.8
        //   multiplier = h_c_forced / h_c_still
        //
        // This replaces the prior hardcoded 4× multiplier which was not ACH-accurate.
        //
        // References:
        // - ASHRAE Handbook — Fundamentals (ch. 4) for forced convection correlation
        // - EnergyPlus Engineering Reference for interior surface coefficients
        let hour_of_day = (timestep % 24) as u8;
        let mut night_vent_active_now = false;
        let mut ach_night_vent: f64 = 0.0; // ACH of night ventilation (for h_tr_is scaling)
        let h_ve_night: f64 = if let Some(ref night_vent) = self.0.hvac.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                night_vent_active_now = true;
                // ASHRAE 140 night-vent fan supplies outdoor air to zone 0
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
                // ACH = fan_capacity (m³/h) / zone_volume (m³)
                // Zone 0 is the conditioned zone per ASHRAE 140
                let zone_vol = self
                    .0
                    .setpoints
                    .zone_volume
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(129.6);
                ach_night_vent = night_vent.fan_capacity / zone_vol;
                night_vent.fan_capacity * rho * cp / 3600.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Prepare sol-air temperatures and fluxes
        // Issue #2873: the helper now writes its per-zone sol-air temperatures
        // into a caller-supplied buffer instead of returning a fresh Vec.
        // The 9R4C path discards the buffer (it has its own per-surface
        // sol-air computation further down), so we hand the helper a stack
        // `SmallVec` that is dropped at the end of the call — bit-identical
        // to the previous behaviour.
        let mut t_sol_air_buf: SmallVec<[f64; 4]> = SmallVec::with_capacity(self.0.hvac.num_zones);
        let (ctf_flux_w, fd_flux_w, _ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp, &mut t_sol_air_buf);

        // Combine fractions
        let conv_frac = self.0.solar.convective_fraction;
        let rad_frac = 1.0 - conv_frac;

        // Solar gain distribution fractions
        // Internal radiative gains split per ISO 13790 Section C.4 Eq. C.5/C.6:
        // Eq. C.5 (radiative-to-surface): phi_st = (1 - F_sup) * phi_int_rad
        //   where F_sup = H_ms / (H_ms + H_is) — fraction to surface node
        //   st_int_frac = rad_frac * (1 - solar_distribution_to_air) = rad_frac * F_sup
        //
        // Eq. C.6 (radiative-to-air): phi_ia gets the radiative portion via solar_distribution_to_air
        //   m_air_frac = rad_frac * solar_distribution_to_air = rad_frac * F_m
        //
        let st_int_frac = rad_frac * (1.0 - self.0.solar.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar.solar_beam_to_mass_fraction;

        let loads_ref = self.0.setpoints.loads.as_ref();
        let solar_ref = self.0.solar.solar_gains.as_ref();
        let opaque_solar_ref = self.0.solar.opaque_solar_gains.as_ref();
        let area_ref = self.0.setpoints.zone_area.as_ref();

        // Issue #1524: consolidated per-timestep scratch (replaces the fourteen
        // standalone `Vec::with_capacity(num_zones)` allocations in 9R4C; the
        // seven read-back intermediates share one flat buffer).
        // Issue #1966 / #2756: scratch is CHECKED OUT of `scratch_pool`
        // (allocates only on the first timestep) and `fill_zero()`'d back to
        // the post-`new(num_zones)` state (the `inter` buffer is resized to
        // `num_zones * 7`). Bit-identical vs. fresh construct.
        let mut scratch = self
            .0
            .hvac
            .scratch_pool
            .checkout_9r4c(self.0.hvac.num_zones);
        scratch.fill_zero();

        for i in 0..self.0.hvac.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            let sol_to_air = sol_w * self.0.solar.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;
            scratch.phi_ia[i] = load_w * conv_frac + sol_to_air;
            scratch.phi_st[i] = load_w * st_int_frac + remaining_sol * st_sol_frac;
            scratch.phi_m[i] = load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w;
        }

        // (#872) Save raw gain data for multi-node solver before moving into tensors.
        // Used for internal radiative gain injection via step_with_gains().
        let phi_ia = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_ia,
        )));
        let phi_st = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_st,
        )));
        let phi_m = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_m,
        )));

        // Issue #863: Compute per-surface sol-air temperature for walls.
        // The CTF/FD flux calculations use t_sol_air_data as the exterior boundary
        // temperature. Using outdoor_temp would ignore solar gain on west walls,
        // causing massive heating energy overcounting (9.45 MWh vs reference 1.17-2.04 MWh).
        let t_sol_air_wall = if let Some(weather) = &self.0.solar.weather {
            let hour_of_year = timestep % 8760;
            let month_days: [usize; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
            let day_of_year = hour_of_year / 24;
            let hour = (hour_of_year % 24) as f64 + 0.5;
            let month = month_days
                .iter()
                .position(|&d| d > day_of_year)
                .unwrap_or(12)
                .saturating_sub(1) as u32;
            let day =
                (day_of_year - month_days.get(month as usize).copied().unwrap_or(0)) as u32 + 1;

            // Issue #1212: Extract weather data before mutably borrowing self for cache
            let (dni, dhi, ghi) = (weather.dni, weather.dhi, weather.ghi);

            // Issue #1212: Use cached solar position to eliminate 5x redundant computation
            let sun_pos = self.cached_solar_position(hour_of_year, 2024, month, day.min(28), hour);

            let ground_reflectance = 0.2;
            let wall_irr = calculate_surface_irradiance(
                &sun_pos,
                dni,
                dhi,
                Some(ghi),
                crate::validation::ashrae_140_cases::Orientation::South,
                ground_reflectance,
                day_of_year + 1,
            );

            let sol_air = SolAirTemperature::ashrae_140_default();
            sol_air.for_wall(
                outdoor_temp,
                wall_irr.total_wm2,
                wall_irr.ground_reflected_wm2,
            )
        } else {
            outdoor_temp
        };

        for i in 0..self.0.hvac.num_zones {
            scratch.t_sol_air_mut()[i] = t_sol_air_wall;
        }

        // Use 5R1C network for free-floating temperature
        let term_rest_1 = &self.0.conduction.derived_term_rest_1;

        // Issue #1712 fix: Apply h_ve_night to h_ext and den when night ventilation
        // is active, matching the 5R1C path (lines 421-471). This ensures the
        // mass coupling pathway properly accounts for night vent cooling.
        //
        // The prior code cloned derived_h_ext without h_ve_night, and used the
        // cached derived_den without recalculating. This caused the 9R4C mass coupling
        // to not properly respond to night ventilation, making night vent less effective
        // than in the 5R1C path.
        let h_ext_for_free_float: T = if night_vent_active_now {
            let base = self.0.conduction.derived_h_ext.as_ref();
            let mut v = Vec::with_capacity(base.len());
            for (i, &b) in base.iter().enumerate() {
                let night_add = if i == 0 { h_ve_night } else { 0.0 };
                v.push(b + night_add);
            }
            T::from(VectorField::new(v))
        } else {
            self.0.conduction.derived_h_ext.clone()
        };
        let den: T = if night_vent_active_now {
            let h_ms_is_prod = self.0.conduction.derived_h_ms_is_prod.as_ref();
            let term_rest_1_slice = term_rest_1.as_ref();
            let ground_coeff = self.0.conduction.derived_ground_coeff.as_ref();
            let h_iz = self.0.conduction.h_tr_iz.as_ref();
            let h_iz_rad = self.0.conduction.h_tr_iz_rad.as_ref();
            let h_ext_slice = h_ext_for_free_float.as_ref();
            let mut v = Vec::with_capacity(h_ext_slice.len());
            for i in 0..h_ext_slice.len() {
                let h_total = if self.0.hvac.num_zones > 1 {
                    h_ext_slice[i] + h_iz[i] + h_iz_rad[i]
                } else {
                    h_ext_slice[i]
                };
                v.push(h_ms_is_prod[i] + term_rest_1_slice[i] * h_total + ground_coeff[i]);
            }
            T::from(VectorField::new(v))
        } else {
            self.0.conduction.derived_den.clone()
        };
        // (#872: sensitivity variable removed — HVAC demand now uses h_loss × ΔT formula)

        let num_tm = self
            .0
            .conduction
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass.mass_temperatures, |a, b| a * b);
        let num_phi_st = self.0.conduction.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        let mut phi_ia_with_iz = phi_ia;

        // Inter-zone heat transfer (if multi-zone) — #1391 Bug 1 fix.
        //
        // Sign convention: q_iz_net[i] is the NET heat flow INTO zone i
        // (positive = heat flowing into zone i). For a symmetric conductance
        // matrix, Σ_i q_iz_net[i] = 0 exactly (energy conservation).
        //
        // Formula: q_iz_net[i] = h_tr_iz[i] · Σ_{j≠i} (T[j] − T[i])
        //               = h_tr_iz[i] · (Σ_j T[j] − N · T[i])
        //
        // Replaces the previous hardcoded 2-zone slice[0]/slice[1] pair which
        // (a) had the sign inverted (`slice[0] += -q_iz_total`) and (b) only
        // handled the zone-0↔zone-1 pair. Now scales to N>2 zones and matches
        // the 5R1C iterative path's `solve_coupled_zone_temperatures` formulation
        // and the `MultiZoneAirflowNetwork` convention (test:
        // `multi_zone_network.rs::two_zone_case960_backward_compatible`).
        if self.0.hvac.num_zones > 1 {
            let slice = phi_ia_with_iz.as_mut();
            let n = self.0.hvac.num_zones;
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

        // Add CTF flux contributions (if enabled)
        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self
                        .0
                        .setpoints
                        .zone_area
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(1.0);
                    let q_ctf = q_flux * area;
                    let t_sol_air_i = scratch.t_sol_air().get(i).copied().unwrap_or(outdoor_temp);
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
                    let net_ctf_flux = q_ctf - q_5r1c;
                    slice[i] += net_ctf_flux;
                }
            }
        }

        // Add FD flux contributions (if enabled)
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in fd_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self
                        .0
                        .setpoints
                        .zone_area
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(1.0);
                    let q_fd = q_flux * area;
                    let t_sol_air_i = scratch.t_sol_air().get(i).copied().unwrap_or(outdoor_temp);
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
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;
                }
            }
        }

        // Build numerator with envelope and ground contributions
        // Issue #863: Use sol-air temperature for wall exterior BC instead of outdoor_temp.
        // This correctly accounts for solar radiation heating the wall surface, reducing
        // the net heat loss and fixing massive heating energy overcounting.
        // #1391: clone phi_ia_with_iz — we still need to read it at line ~2394
        // (compute_zone_air_temperature argument) for the multi-node solver.
        // Issue #1422: use the night-vent-aware `h_ext_for_free_float` (not the cached
        // `h_ext_base`) so the free-float numerator sees the same h_ve_total as the
        // denominator — otherwise the cooling effect of `h_ve_night` is cancelled.
        let mut num_rest_with_iz = phi_ia_with_iz.clone();
        for (i, (n, h)) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(h_ext_for_free_float.as_ref().iter())
            .enumerate()
        {
            let t_sol_air_i = scratch.t_sol_air().get(i).copied().unwrap_or(outdoor_temp);
            *n += h * t_sol_air_i;
        }
        num_rest_with_iz.mul_assign(term_rest_1);
        let ground_coeff = self.0.conduction.derived_ground_coeff.as_ref();
        for (n, g) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(ground_coeff.iter())
        {
            *n += g * t_g;
        }

        // Calculate free-floating temperature using 5R1C network
        let mut t_i_free = num_tm;
        t_i_free.add_assign(&num_phi_st);
        t_i_free.add_assign(&num_rest_with_iz);
        t_i_free.div_assign(&den);

        // === Update Multi-Node Thermal Mass (9R4C) ===
        //
        // (#872) Run the multi-node solver but do NOT write its mass temperatures
        // back to self.0.mass.mass_temperatures. The 5R1C model owns mass_temperatures
        // and uses it for the t_i_free formula. The multi-node solver maintains its
        // own internal mass node temperatures (solver.mass.wall.temperature, etc.)
        // which are independent.
        //
        // The multi-node solver is used ONLY for:
        // 1. Computing a better HVAC demand via compute_zone_air_temperature()
        // 2. Computing multi-node t_free for the HVAC demand formula
        //
        // The 5R1C t_i_free (using its own mass_temperatures) is used for:
        // 1. Free-floating zone temperature (validated at 42.87°C for 900FF)
        let t_i_free_5r1c = t_i_free.clone();

        // Issue #864: Store pre-gain mass temperatures and per-surface gains for
        // step_per_surface(). Using pre-gain temperatures avoids double-counting
        // gains that are added in SurfaceNode::update()'s backward Euler.
        // (Issue #1524: these six read-back intermediates share the flat
        // `scratch.inter` backing buffer — 6 allocations collapsed to 0 here.)

        #[allow(clippy::needless_range_loop)]
        for zone_idx in 0..self.0.hvac.num_zones {
            if zone_idx >= self.0.conduction.backend.multi_node_solvers.len() {
                continue;
            }

            // Issue #1212: Compute sun_pos BEFORE solver borrow to avoid borrow conflict.
            // sun_pos depends only on timestep/lat/lon, not on solver state.
            let hour_of_year = timestep % 8760;
            let month_days: [usize; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
            let day_of_year = hour_of_year / 24;
            let hour = (hour_of_year % 24) as f64 + 0.5;
            let month = month_days
                .iter()
                .position(|&d| d > day_of_year)
                .unwrap_or(12)
                .saturating_sub(1) as u32;
            let day =
                (day_of_year - month_days.get(month as usize).copied().unwrap_or(0)) as u32 + 1;

            // Issue #1212: Use cached solar position to eliminate 5x redundant computation
            // Called BEFORE solver borrow so there's no conflict
            let sun_pos = self.cached_solar_position(hour_of_year, 2024, month, day.min(28), hour);

            let solver = &mut self.0.conduction.backend.multi_node_solvers[zone_idx];
            // (#872) Use previous zone temperature as boundary, NOT 5R1C t_i_free.
            // This breaks the destructive feedback loop where 5R1C mass temps corrupt
            // the solver's boundary condition. The solver will compute its own
            // zone air temperature from the multi-node balance.
            let t_zone_prev = self.0.setpoints.temperatures.as_ref()[zone_idx];
            #[allow(unused_variables)]
            let t_ext = scratch
                .t_sol_air()
                .get(zone_idx)
                .copied()
                .unwrap_or(outdoor_temp);

            // Surface temperature: compute from CURRENT (pre-step) conductance-weighted
            // mass node temperatures instead of lagged t_zone_prev - 0.5 approximation.
            // This ensures the mass update sees the actual surface temperature.
            let h_ms_w = solver.mass.wall.h_tr_ms;
            let h_ms_r = solver.mass.roof.h_tr_ms;
            let h_ms_f = solver.mass.floor.h_tr_ms;
            let h_ms_total = h_ms_w + h_ms_r + h_ms_f;
            let t_surface = if h_ms_total > 0.0 {
                (h_ms_w * solver.mass.wall.temperature
                    + h_ms_r * solver.mass.roof.temperature
                    + h_ms_f * solver.mass.floor.temperature)
                    / h_ms_total
            } else {
                t_zone_prev - 0.5
            };

            // Issue #1615: Compute CURRENT multi-node air temperature BEFORE step_with_gains
            // so the mass node update uses the actual air temp that includes night
            // ventilation effect (h_ve_night), not the lagged t_zone_prev.
            // This fixes Case 950 night ventilation over-prediction where the mass was
            // not properly cooled by night vent until the next timestep.
            let h_ve_val = self.0.conduction.h_ve.as_ref()[zone_idx];
            let h_ve_night_zone = if night_vent_active_now && zone_idx == 0 {
                h_ve_night
            } else {
                0.0
            };
            let phi_ia_val = phi_ia_with_iz.as_ref()[zone_idx];

            // Issue #1279: Temporarily boost h_tr_is for compute_zone_air_temperature
            // during night ventilation (same boost used for step_with_gains).
            // This ensures t_air_mn_pre uses the ASHRAE/EnergyPlus forced convection
            // correlation, consistent with how step_with_gains and compute_zone_air_temperature
            // are called after the main loop.
            // Issue #2871: cap the multiplier at MAX_CONVECTIVE_TO_AIR_MULTIPLIER
            // to prevent mass-node pulsed-charging dump during the morning ramp.
            let h_tr_is_multiplier_pre = if night_vent_active_now {
                capped_h_tr_is_ach_multiplier(ach_night_vent)
            } else {
                1.0
            };
            // The `!= 1.0` guard is a sentinel check on a value that is
            // either the literal `1.0` (no night ventilation) or a value
            // returned by `capped_h_tr_is_ach_multiplier` — under
            // `fast-math` reassociation the latter could be a few ulp
            // away from `1.0` even at low `ach`, which is the *intended*
            // behaviour: when the multiplier is non-trivially different
            // from `1.0`, apply it. The exact-equality comparison is
            // therefore the correct semantic here, not a fast-math bug.
            // See Issue #3357.
            #[allow(clippy::float_cmp)]
            if h_tr_is_multiplier_pre != 1.0 {
                solver.h_tr_is *= h_tr_is_multiplier_pre;
            }
            let t_air_mn_pre = solver.compute_zone_air_temperature(
                outdoor_temp,
                h_ve_val,
                h_ve_night_zone,
                phi_ia_val,
            );
            // Restore h_tr_is - the main boost/restore block at lines ~2720 will handle it for step_with_gains.
            // Sentinel `!= 1.0` guard mirroring the boost check above;
            // see comment on the boost site (Issue #3357).
            #[allow(clippy::float_cmp)]
            if h_tr_is_multiplier_pre != 1.0 {
                solver.h_tr_is /= h_tr_is_multiplier_pre;
            }

            solver.set_zone_temperature(t_air_mn_pre);
            solver.set_surface_temperature(t_surface);

            let (surface_ext_temps, wall_irr_val, roof_irr_val) =
                if let Some(ref weather) = self.0.solar.weather {
                    // Issue #1212: Extract weather data for irradiance calculations
                    let (dni, dhi, ghi) = (weather.dni, weather.dhi, weather.ghi);

                    let ground_reflectance = 0.2;
                    let wall_irr = calculate_surface_irradiance(
                        &sun_pos,
                        dni,
                        dhi,
                        Some(ghi),
                        crate::validation::ashrae_140_cases::Orientation::South,
                        ground_reflectance,
                        day_of_year + 1,
                    );
                    let roof_irr = calculate_surface_irradiance(
                        &sun_pos,
                        dni,
                        dhi,
                        Some(ghi),
                        crate::validation::ashrae_140_cases::Orientation::Up,
                        ground_reflectance,
                        day_of_year + 1,
                    );

                    let sol_air = SolAirTemperature::ashrae_140_default();
                    // Issue #2872: per-surface sky view factor.
                    // Vertical wall (tilt = 90°) → F_sky ≈ 0.5; horizontal roof
                    // (tilt = 0°) → F_sky = 1.0 (already in for_roof). The
                    // wall sees half the sky dome, so the LW correction is
                    // halved. This brings the wall sol-air boundary closer
                    // to outdoor and reduces the over-cooling of the high-
                    // mass Case 950FF night minimum.
                    let f_sky_wall = 0.5;
                    let ext_temps = SurfaceExteriorTemperatures {
                        t_ext_wall: sol_air.for_wall_with_f_sky(
                            outdoor_temp,
                            wall_irr.total_wm2,
                            wall_irr.ground_reflected_wm2,
                            sky_temp,
                            f_sky_wall,
                        ),
                        t_ext_roof: sol_air.for_roof(outdoor_temp, roof_irr.total_wm2, sky_temp),
                        t_ext_floor: t_g,
                    };
                    (ext_temps, wall_irr.total_wm2, roof_irr.total_wm2)
                } else {
                    (
                        SurfaceExteriorTemperatures {
                            t_ext_wall: t_ext,
                            t_ext_roof: t_ext,
                            t_ext_floor: t_g,
                        },
                        0.0,
                        0.0,
                    )
                };

            solver.set_surface_exterior_temperatures(surface_ext_temps);

            // Issue #895/#873: Step solver with proper per-node gain injection.
            // - phi_st (radiative gains to surface): distributed to wall/roof/floor
            //   proportional to h_tr_ms (per Issue #873 requirement)
            // - phi_m (solar to mass): goes to internal mass node
            // - phi_ia (convective to air): handled via compute_zone_air_temperature
            let wall_area_val = self.0.setpoints.wall_area.as_ref()[zone_idx];
            let roof_area_val = self.0.setpoints.roof_area.as_ref()[zone_idx];
            let floor_area_val = self.0.setpoints.floor_area.as_ref()[zone_idx];
            let phi_st_zone = phi_st.as_ref()[zone_idx];
            // phi_m contains all solar gains (window + opaque) to mass
            let phi_m_zone = phi_m.as_ref()[zone_idx];
            // Distribute phi_st to envelope nodes proportional to h_tr_ms
            let h_ms_w = solver.mass.wall.h_tr_ms;
            let h_ms_r = solver.mass.roof.h_tr_ms;
            let h_ms_f = solver.mass.floor.h_tr_ms;
            let h_ms_total = h_ms_w + h_ms_r + h_ms_f;
            let wall_frac = if h_ms_total > 1e-6 {
                h_ms_w / h_ms_total
            } else {
                1.0 / 3.0
            };
            let roof_frac = if h_ms_total > 1e-6 {
                h_ms_r / h_ms_total
            } else {
                1.0 / 3.0
            };
            let floor_frac = if h_ms_total > 1e-6 {
                h_ms_f / h_ms_total
            } else {
                1.0 / 3.0
            };
            // phi_st goes to envelope nodes, phi_m goes to internal node
            let gains_wall = phi_st_zone * wall_frac;
            let gains_roof = phi_st_zone * roof_frac;
            let gains_floor = phi_st_zone * floor_frac;
            let gains_internal = phi_m_zone;
            // Floor gets no direct solar (horizontal down orientation)
            let floor_irr_val = 0.0;

            // Issue #864: Capture pre-gain mass temperatures BEFORE step_with_gains()
            // so step_per_surface can use them to avoid double-counting gains.
            // Also compute per-surface opaque solar gains for SurfaceNode::update().
            let mass_temp_wall_pre = solver.mass.wall.temperature;
            let mass_temp_roof_pre = solver.mass.roof.temperature;
            let mass_temp_floor_pre = solver.mass.floor.temperature;

            // Use opaque solar gain (phi_m_zone) distributed by irradiance × area.
            // Issue #2303 fix: Now uses actual surface areas from thermal model geometry
            // instead of per-unit areas (1.0). This correctly weights gains by the solar-
            // absorbing area of each surface.
            let solar_gains = distribute_opaque_solar_gains(
                phi_m_zone,
                wall_area_val,
                roof_area_val,
                floor_area_val,
                wall_irr_val,
                roof_irr_val,
                floor_irr_val,
            );

            scratch.pg_wall_mut()[zone_idx] = mass_temp_wall_pre;
            scratch.pg_roof_mut()[zone_idx] = mass_temp_roof_pre;
            scratch.pg_floor_mut()[zone_idx] = mass_temp_floor_pre;
            scratch.pm_wall_mut()[zone_idx] = solar_gains.phi_m_wall;
            scratch.pm_roof_mut()[zone_idx] = solar_gains.phi_m_roof;
            scratch.pm_floor_mut()[zone_idx] = solar_gains.phi_m_floor;

            // Issue #1279: Boost h_tr_is for forced convection during night ventilation.
            // Use ACH-dependent multiplier (ASHRAE/EnergyPlus correlation) instead of
            // the prior hardcoded 4× which was not ACH-accurate.
            // Issue #2871: cap the multiplier at MAX_CONVECTIVE_TO_AIR_MULTIPLIER
            // to prevent mass-node pulsed-charging dump during the morning ramp.
            // IMPORTANT: Restore h_tr_is after step to avoid persisting the boost to daytime.
            let h_tr_is_multiplier = if night_vent_active_now {
                capped_h_tr_is_ach_multiplier(ach_night_vent)
            } else {
                1.0
            };
            // Sentinel `!= 1.0` guard on a value that is either the literal `1.0`
            // (no night ventilation) or a value returned by
            // `capped_h_tr_is_ach_multiplier`. See comment on the boost
            // site above for the full justification (Issue #3357).
            #[allow(clippy::float_cmp)]
            let original_h_tr_is = if h_tr_is_multiplier != 1.0 {
                let original = solver.h_tr_is;
                solver.h_tr_is *= h_tr_is_multiplier;
                Some((original, h_tr_is_multiplier))
            } else {
                None
            };
            // Issue #1898: Pass night ventilation parameters to step_with_gains so the
            // 9R4C mass nodes are cooled by night ventilation (not just the air node).
            solver.step_with_gains(
                dt,
                gains_wall,
                gains_roof,
                gains_floor,
                gains_internal,
                h_ve_night_zone,
                outdoor_temp,
            );
            if let Some((original, _)) = original_h_tr_is {
                solver.h_tr_is = original;
            }
        }

        // Issue #1279: Apply ACH-dependent forced convection boost to h_tr_is when calling
        // compute_zone_air_temperature during night ventilation. This ensures
        // the zone air temperature calculation uses the enhanced surface-to-air
        // heat transfer from the ASHRAE/EnergyPlus forced convection correlation.
        // Issue #2871: cap the multiplier at MAX_CONVECTIVE_TO_AIR_MULTIPLIER.
        if night_vent_active_now {
            let multiplier = capped_h_tr_is_ach_multiplier(ach_night_vent);
            for solver in &mut self.0.conduction.backend.multi_node_solvers {
                solver.h_tr_is *= multiplier;
            }
        }

        // (#872) Compute zone air temperature from multi-node solver.
        // The solver's compute_zone_air_temperature() uses the multi-node energy
        // balance: T_air = (h_tr_is * T_surface + h_ve * T_outdoor + phi_ia) / (h_tr_is + h_ve)
        //
        // IMPORTANT: For free-floating mode, we use the MAX of the 5R1C and multi-node
        // temperatures. The 5R1C t_i_free correctly handles the phi_st gain path
        // (surface-to-mass coupling) which the multi-node solver doesn't yet model
        // (needs #873 for per-node solar injection). Using the maximum preserves the
        // better estimate from each model.
        //
        // For HVAC mode, the temperature update is self-consistent: t_act = T_setpoint
        // regardless of which t_free estimate we use, because the HVAC coefficient
        // cancels the free-floating temperature error.
        for zone_idx in 0..self.0.hvac.num_zones {
            if zone_idx < self.0.conduction.backend.multi_node_solvers.len() {
                let solver = &self.0.conduction.backend.multi_node_solvers[zone_idx];
                let h_ve_val = self.0.conduction.h_ve.as_ref()[zone_idx];
                // #1391 Bug 2 fix: use `phi_ia_with_iz` (convective gains + net
                // inter-zone air-flow energy) instead of the raw `phi_ia`. Without
                // the inter-zone term, downstream HVAC demand and the free-float
                // commit get zero inter-zone coupling and Case 900/960 violate energy
                // conservation. Mirrors the 5R1C pattern at
                // `thermal_model_iterative.rs:858-859` (`phi_ia + VectorField(q_iz)`).
                let phi_ia_val = phi_ia_with_iz.as_ref()[zone_idx];
                // h_ve_night: night ventilation fan conductance (only for zone 0, ASHRAE 140)
                let h_ve_night_zone = if night_vent_active_now && zone_idx == 0 {
                    h_ve_night
                } else {
                    0.0
                };
                let t_air_mn = solver.compute_zone_air_temperature(
                    outdoor_temp,
                    h_ve_val,
                    h_ve_night_zone,
                    phi_ia_val,
                );
                // Use multi-node temperature — it provides the correct air balance
                // from mass node temperatures stepped by the backward Euler.
                scratch.t_i_free[zone_idx] = t_air_mn;
            } else {
                scratch.t_i_free[zone_idx] = t_i_free.as_ref()[zone_idx];
            }
        }
        let t_i_free_mn = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.t_i_free,
        )));

        // Issue #1279: Restore h_tr_is to original value after computing zone air temperature.
        // Issue #2871: must divide by the SAME (capped) multiplier that was applied above.
        if night_vent_active_now {
            let multiplier = capped_h_tr_is_ach_multiplier(ach_night_vent);
            for solver in &mut self.0.conduction.backend.multi_node_solvers {
                solver.h_tr_is /= multiplier;
            }
        }

        // (#872) Do NOT write multi-node mass temperatures back to self.0.
        // The multi-node solver keeps its own internal state. Writing back would
        // corrupt the 5R1C mass_temperatures used by the t_i_free formula.

        // Update multi-node solver surface temperatures using conductance-weighted
        // envelope node temperatures (not the hardcoded t_zone - 0.5).
        // This ensures the solver's surface_temperature field is consistent with
        // the mass node temperatures it just computed.
        for zone_idx in 0..self.0.hvac.num_zones {
            if zone_idx >= self.0.conduction.backend.multi_node_solvers.len() {
                continue;
            }
            let solver = &mut self.0.conduction.backend.multi_node_solvers[zone_idx];
            let h_ms_w = solver.mass.wall.h_tr_ms;
            let h_ms_r = solver.mass.roof.h_tr_ms;
            let h_ms_f = solver.mass.floor.h_tr_ms;
            let h_ms_total = h_ms_w + h_ms_r + h_ms_f;
            let t_surface = if h_ms_total > 0.0 {
                (h_ms_w * solver.wall_temperature()
                    + h_ms_r * solver.roof_temperature()
                    + h_ms_f * solver.floor_temperature())
                    / h_ms_total
            } else {
                solver.surface_temperature
            };
            solver.set_surface_temperature(t_surface);
        }

        // === Issue #1005/#864: Per-surface conduction integration ===
        //
        // Refine the surface temperature for each zone using the per-surface
        // conduction solver. This tracks the air-side surface film independently
        // from the bulk mass node, using each surface's own conductances and
        // exterior boundary. The result is written back to the multi-node
        // solver's `surface_temperature` so subsequent air-node energy
        // balances use a more accurate boundary value.
        //
        // Issue #864: Pass pre-gain mass temperatures to avoid double-counting
        // gains (gains are added in SurfaceNode::update()'s backward Euler).
        // Also pass per-surface opaque solar gains for direct absorption.
        for zone_idx in 0..self.0.hvac.num_zones {
            if zone_idx >= self.0.conduction.backend.multi_node_solvers.len() {
                continue;
            }
            let solver = &mut self.0.conduction.backend.multi_node_solvers[zone_idx];
            let mass_temp_wall_pre = scratch.pg_wall().get(zone_idx).copied().unwrap_or(20.0);
            let mass_temp_roof_pre = scratch.pg_roof().get(zone_idx).copied().unwrap_or(20.0);
            let mass_temp_floor_pre = scratch.pg_floor().get(zone_idx).copied().unwrap_or(20.0);
            let phi_m_wall = scratch.pm_wall().get(zone_idx).copied().unwrap_or(0.0);
            let phi_m_roof = scratch.pm_roof().get(zone_idx).copied().unwrap_or(0.0);
            let phi_m_floor = scratch.pm_floor().get(zone_idx).copied().unwrap_or(0.0);
            solver.step_per_surface(
                dt,
                (mass_temp_wall_pre, mass_temp_roof_pre, mass_temp_floor_pre),
                (phi_m_wall, phi_m_roof, phi_m_floor),
            );
        }

        // Calculate HVAC demand
        // Issue #1345: Bind the predictive controller's `modulation` factor instead of
        // discarding it (it was previously `let (hvac_mode, _modulation) = ...`). The
        // modulation factor is the part-load ratio the predictive controller recommends
        // (0.0 = off, 1.0 = full capacity); it must be applied to the per-zone HVAC
        // demand below and forwarded to `VariableCapacityEquipment::update_state` so
        // equipment PLR tracking reflects predictive intent (and is ready to use
        // continuous modulation once the controller's curve is softened in Plan 15-04).
        let _hour_of_day_idx = timestep % 24;
        let temp_rate = if timestep > 0 {
            (self.0.setpoints.temperatures.as_ref()[0]
                - self.0.hvac.previous_temperatures.as_ref()[0])
                / dt
        } else {
            0.0
        };

        let (hvac_mode, modulation) = self.0.hvac.predictive_controller.calculate_modulation(
            self.0.setpoints.temperatures.as_ref()[0],
            self.0.mass.mass_temperatures.as_ref()[0],
            temp_rate,
        );
        let hvac_mode: EquipmentHVACMode = hvac_mode;

        // (#872) Compute HVAC demand BEFORE mass update, so the mass uses CURRENT t_i_act.
        // This matches the 5R1C ordering: compute Q → compute t_act → update mass using t_act.
        // The previous ordering (mass → HVAC) caused a one-timestep lag where mass used
        // the PREVIOUS step's t_act, slowing convergence and overestimating HVAC energy.
        //
        // Issue #860: Use multi-node t_air for HVAC demand when available.
        // The multi-node solver provides a physically accurate free-floating temperature
        // from the 9R4C thermal balance (wall/roof/floor nodes → surface → air).
        // This is more accurate than the 5R1C t_i_free which uses a lumped mass.
        let (hvac_for_temp_calc, t_i_act) = if self.0.hvac.free_float {
            // Free-float: no HVAC. `t_i_act` feeds ONLY the 5R1C lumped-mass update
            // below, so it keeps the 5R1C-consistent `t_i_free_5r1c` to preserve the
            // lumped-mass evolution (the lumped mass is the 5R1C mass node and must
            // stay self-consistent with the 5R1C air node).
            //
            // (ADR-002, #1175) The COMMITTED zone temperature for high-mass free-float
            // is the 9R4C multi-node air temperature (`t_i_free_mn`), applied in the
            // free-float commit block below — NOT `t_i_act`. That makes 9R4C the sole
            // driver of high-mass free-float, bypassing the coefficient-tuned
            // `h_ms_coeff` coupling. The lumped mass continues to evolve on its own
            // 5R1C dynamics (it no longer drives the high-mass air temperature).
            // Low-mass behaviour is unchanged (low-mass has no MultiNodeSolver, so
            // `t_i_free_mn` equals `t_i_free_5r1c` and the commit is a no-op change).
            (
                T::from(self.0.solar.zero_vector.clone()),
                t_i_free_5r1c.clone(),
            )
        } else {
            // HVAC mode: use multi-node t_air (from _t_i_free_mn) when available
            let heat_cap = self.0.hvac.hvac_heating_capacity;
            let cool_cap = self.0.hvac.hvac_cooling_capacity;
            // Issue #1524: hvac/t_i_act live in the local `scratch` struct, so
            // `scratch.hvac[i]` / `scratch.t_i_act[i]` (mutable borrows of a
            // local) coexist freely with `self.compute_hvac_coefficient(i)`
            // (an `&self` borrow) — the exact conflict that sank #1436.
            for i in 0..self.0.hvac.num_zones {
                // Issue #860: Prefer multi-node t_air over 5R1C t_free for HVAC demand
                let t_free_val =
                    if i < self.0.conduction.backend.multi_node_solvers.len() {
                        // Use multi-node computed free-float temperature (available at line 2534)
                        // The multi-node t_air uses conductance-weighted envelope node temperatures
                        // and the air energy balance: T_air = (h_tr_is*T_surface + h_ve*T_out + phi_ia)/(h_tr_is + h_ve)
                        t_i_free_mn.as_ref().get(i).copied().unwrap_or_else(|| {
                            t_i_free_5r1c.as_ref().get(i).copied().unwrap_or(20.0)
                        })
                    } else {
                        t_i_free_5r1c.as_ref()[i]
                    };
                // Issue #907: HVAC coefficient is the full 5R1C/6R2C Norton equivalent
                // at the air node (see `compute_hvac_coefficient`). Use it here so the
                // self-consistent t_act = t_free + Q/h_coeff check matches T_setpoint
                // (h_tr_1 + h_ve alone is too small — it ignores mass/ground paths).
                let h_coeff = self.compute_hvac_coefficient(i);
                let _h_tr_ms = self.0.conduction.h_tr_ms.as_ref()[i];

                // Issue #900: dynamic mass heat release term (cooling only).
                //
                // Use the multi-node solver's mass node temperatures when
                // available — they are stable and physically correct. The
                // 5R1C lumped mass can diverge numerically in the 9R4C path
                // (it is updated by the post-HVAC mass integrator at line
                // ~2240) and would produce degenerate heat release values
                // (T_mass > 70°C) if used directly.
                //
                // For the 5R1C path or when multi-node is unavailable, fall
                // back to the 5R1C lumped mass temperature. The hvac module
                // applies the same sanity guard (-20..=80°C) on its own.
                #[cfg(feature = "debug-physics")]
                let t_mass_mn = if i < self.0.conduction.backend.multi_node_solvers.len() {
                    let solver = &self.0.conduction.backend.multi_node_solvers[i];
                    // Conductance-weighted envelope temperature (wall/roof/floor).
                    let h_ms_w = solver.mass.wall.h_tr_ms;
                    let h_ms_r = solver.mass.roof.h_tr_ms;
                    let h_ms_f = solver.mass.floor.h_tr_ms;
                    let h_ms_total = h_ms_w + h_ms_r + h_ms_f;
                    if h_ms_total > 1e-6 {
                        (h_ms_w * solver.mass.wall.temperature
                            + h_ms_r * solver.mass.roof.temperature
                            + h_ms_f * solver.mass.floor.temperature)
                            / h_ms_total
                    } else {
                        solver.envelope_temperature()
                    }
                } else {
                    self.0.mass.mass_temperatures.as_ref()[i]
                };

                // DEBUG: Print h_coeff breakdown on first HVAC step after warmup
                #[cfg(feature = "debug-physics")]
                if timestep == 337 && i == 0 && !self.0.hvac.free_float {
                    let h_tr_is = self.0.conduction.h_tr_is.as_ref()[i];
                    let h_tr_ms = self.0.conduction.h_tr_ms.as_ref()[i];
                    let h_tr_em = self.0.conduction.h_tr_em.as_ref()[i];
                    let h_tr_w = self.0.conduction.h_tr_w.as_ref()[i];
                    let h_tr_floor = self.0.conduction.h_tr_floor.as_ref()[i];
                    let h_ve_scalar = self
                        .0
                        .conduction
                        .h_ve
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(0.0);
                    let h_ms_em = h_tr_ms * h_tr_em / (h_tr_ms + h_tr_em);
                    let stb = h_tr_w + h_ms_em + h_tr_floor;
                    let h_is_X = h_tr_is * stb / (h_tr_is + stb);
                    let h_coeff_check = h_is_X + h_ve_scalar;
                    eprintln!(
                        "HVAC_DBG[step=337]: h_tr_is={:.3}, h_tr_ms={:.1}, h_tr_em={:.3}, h_tr_w={:.3}, h_tr_floor={:.3}, h_ve={:.3}",
                        h_tr_is, h_tr_ms, h_tr_em, h_tr_w, h_tr_floor, h_ve_scalar
                    );
                    eprintln!(
                        "  h_ms_em={:.3}, stb={:.3}, h_is_X={:.3}, h_coeff={:.4} (check={:.4})",
                        h_ms_em, stb, h_is_X, h_coeff, h_coeff_check
                    );
                    eprintln!(
                        "  t_free={:.2}, t_mass_mn={:.2}, q_heating={:.2}",
                        t_free_val,
                        t_mass_mn,
                        h_coeff
                            * (self.0.setpoints.heating_setpoints.as_ref()[i] - t_free_val)
                                .max(0.0)
                    );
                }

                // CORRECTED cooling formula (symmetric with heating):
                //
                // For heating: Q = h_coeff × (T_heat_sp − T_free) > 0
                // For cooling: Q = h_coeff × (T_cool_sp − T_free) < 0  [same form]
                //
                // The driving temperature for BOTH is the FREE-FLOATING zone air temperature
                // (t_i_free), which already includes all heat flows (solar, internal, conduction,
                // ventilation, AND the dynamic mass heat exchange via the 5R1C network).
                //
                // Using t_free (zone air temperature) is correct because:
                // - t_free represents the equilibrium temperature the zone reaches WITHOUT HVAC
                // - If t_free > T_cool_sp, the zone needs cooling to bring it down
                // - If t_free < T_heat_sp, the zone needs heating to bring it up
                //
                // The OLD formula used t_mass_mn (conductance-weighted mass temperature) as
                // the driving temperature for cooling, which is WRONG because:
                // - During summer peak, T_mass ≈ 28-30°C but T_zone ≈ 33-36°C
                // - The HVAC needs to cool T_zone to 27°C, not T_mass to 27°C
                // - Using t_mass gives ~162 W demand instead of ~730 W (4.5× underestimate)
                //
                // Issue #2826: per-zone setpoints are read from the
                // `heating_setpoints` / `cooling_setpoints` vectors (with the
                // legacy scalar `heating_setpoint` / `cooling_setpoint` as
                // fallback when the slice is too short — slice read at index
                // `i` here is the same index used in `compute_zone_hvac_load`,
                // so the two paths produce identical demand figures).
                let heating_setpoint_i = self
                    .0
                    .setpoints
                    .heating_setpoints
                    .as_ref()
                    .get(i)
                    .copied()
                    .unwrap_or(self.0.setpoints.heating_setpoint);
                let cooling_setpoint_i = self
                    .0
                    .setpoints
                    .cooling_setpoints
                    .as_ref()
                    .get(i)
                    .copied()
                    .unwrap_or(self.0.setpoints.cooling_setpoint);
                let q = if t_free_val < heating_setpoint_i {
                    // Heating: Q = h_coeff × (T_heat_sp − T_free) > 0
                    h_coeff * (heating_setpoint_i - t_free_val)
                } else if t_free_val > cooling_setpoint_i {
                    // Cooling: Q = h_coeff × (T_cool_sp − T_free) = −h_coeff × (T_free − T_cool_sp) < 0
                    // Driving temperature is t_free (zone air), NOT t_mass_mn
                    -h_coeff * (t_free_val - cooling_setpoint_i)
                } else {
                    // Zone air is within deadband: no HVAC demand
                    0.0
                };

                let q_clamped = q.clamp(-cool_cap, heat_cap);
                scratch.hvac[i] = q_clamped;

                // Self-consistent: t_act = t_free + Q / h_coeff = T_setpoint (when not clamped)
                //
                // Issue #900 note: the "self-consistent" formula here uses
                // h_loss only. For a heating demand (q > 0) sized to the
                // Issue #925 formula, this gives t_i_act = T_setpoint, which
                // is the design intent. For a cooling demand that includes
                // the dynamic mass heat release term, this can give
                // t_i_act below T_cool_sp; that over-cooling is a known
                // limitation of the steady-state t_i_free approximation
                // (see #917, #924) and is accepted as a tractable
                // approximation here.
                if h_coeff > 0.0 && q_clamped.abs() > 1e-6 {
                    scratch.t_i_act[i] = t_free_val + q_clamped / h_coeff;
                } else {
                    scratch.t_i_act[i] = t_free_val;
                }
            }
            (
                T::from(VectorField::from_smallvec(std::mem::take(
                    &mut scratch.hvac,
                ))),
                T::from(VectorField::from_smallvec(std::mem::take(
                    &mut scratch.t_i_act,
                ))),
            )
        };

        // Issue #1345: forward the predictive controller's modulation factor to
        // `VariableCapacityEquipment::update_state` so equipment PLR tracking
        // reflects predictive intent (and so a future controller with continuous
        // modulation propagates naturally). Also wire `EconomizerMode` (free
        // cooling when outdoor air is cooler than zone and below the cooling
        // setpoint) — this mirrors the 5R1C path's wiring at lines ~542–599 and
        // is the only HVAC-mode branch in this function that touches the
        // equipment trait at all.
        //
        // Note: the per-zone `hvac_data` vector above deliberately uses the
        // un-modulated `q_clamped` so the 5R1C lumped-mass update is
        // self-consistent with `t_i_act` (the existing mass-temperature path).
        // The modulation is applied here, at the equipment dispatch boundary,
        // so the equipment's PLR tracks the predictive intent without changing
        // the established mass dynamics. This matches the issue's scope
        // ("Bind the previously-discarded `_modulation` to
        // `VariableCapacityEquipment::update_state`").
        if !self.0.hvac.free_float {
            if let Some(ref mut equipment) = self.0.hvac.hvac_equipment {
                // Economizer is only meaningful in cooling mode; the helper is
                // mode-agnostic so we still call it (it returns false for
                // `EconomizerMode::Disabled` and for non-cooling cases).
                use crate::sim::hvac::{calculate_free_cooling_capacity, is_economizer_active};
                let hour_of_day_idx = timestep % 24;
                let cooling_setpoint_for_econ =
                    self.0.setpoints.cooling_schedule.value(hour_of_day_idx);
                let economizer_active = is_economizer_active(
                    self.0.hvac.economizer_mode,
                    outdoor_temp,
                    None, // outdoor_enthalpy — only available in Enthalpy mode (not wired here)
                    self.0.setpoints.temperatures.as_ref()[0],
                    None, // zone_enthalpy
                    cooling_setpoint_for_econ,
                );
                // Free cooling capacity in W (the helper returns kW; convert).
                let free_cooling_capacity_w =
                    if economizer_active && matches!(hvac_mode, EquipmentHVACMode::Cooling) {
                        // Note: when economizer is active, the per-zone cooling demand
                        // already absorbed the free-cooling potential above (the outdoor
                        // air cools the zone), so we don't subtract from hvac_data again.
                        // We DO report it to the equipment as effective capacity for
                        // energy accounting and PLR tracking.
                        calculate_free_cooling_capacity(
                            outdoor_temp,
                            self.0.setpoints.temperatures.as_ref()[0],
                            self.0.setpoints.ventilation_airflow_m3_per_s, // Issue #2345: was hardcoded 10000.0
                        ) * 1000.0
                    } else {
                        0.0
                    };

                // Total HVAC demand across all zones (sign convention: positive =
                // heating, negative = cooling). The modulation factor (Issue #1345)
                // is applied here at the equipment dispatch boundary.
                let total_demand: f64 = hvac_for_temp_calc.as_ref().iter().sum();

                // Equipment `update_state` expects a positive load magnitude plus
                // the HVAC mode. In cooling mode, add the free-cooling capacity to
                // the magnitude (free cooling is part of the total delivered
                // capacity even though it doesn't draw electrical power from the
                // equipment). Apply the predictive controller's modulation factor
                // (previously discarded at the call site above) so the equipment
                // PLR tracks predictive intent.
                let load_magnitude: f64 = match hvac_mode {
                    EquipmentHVACMode::Heating => total_demand.max(0.0) * modulation,
                    EquipmentHVACMode::Cooling => {
                        (total_demand.abs() + free_cooling_capacity_w) * modulation
                    }
                    EquipmentHVACMode::Off => 0.0,
                };

                // Clamp to equipment rated capacity (matches the 5R1C path's
                // `capacity = equipment.calculate_capacity(1.0, outdoor_temp)`
                // followed by `modulated_load.clamp(0.0, capacity)`).
                let rated_capacity = equipment.calculate_capacity(1.0, outdoor_temp);
                let load_clamped = load_magnitude.clamp(0.0, rated_capacity);

                // Bind the previously-discarded modulation to PLR update via the
                // VariableCapacityEquipment::update_state Chiller/Boiler/HeatPump/
                // CAV/VAV path (src/sim/hvac/equipment.rs:117).
                equipment.update_state(load_clamped, outdoor_temp, hvac_mode);
            }
        }

        // (#872) Update 5R1C mass temperatures using CURRENT t_i_act.
        // For free-floating: t_i_act = t_i_free_5r1c.
        // For HVAC: t_i_act = T_setpoint (self-consistent with HVAC demand).
        //
        // CRITICAL: The surface temperature t_s uses a BLENDED t_i that accounts for
        // the air-to-surface bottleneck. The ISO 13790 shows that HVAC power reaches
        // the mass through H_tr_3 ≈ 40 W/K (the combined air-side bottleneck), not
        // through h_tr_ms = 1300 W/K. This means only ~3% of the HVAC signal reaches
        // the mass node per timestep.
        //
        // Without this blend, the mass converges in ~17 hours (wrong). With it, the
        // mass converges in ~500 hours (~21 days), matching the ISO 13790's dynamics.
        {
            let mass_temps_ref = self.0.mass.mass_temperatures.as_ref();
            let thermal_cap_ref = self.0.mass.thermal_capacitance.as_ref();
            let h_tr_em_ref = self.0.conduction.h_tr_em.as_ref();
            let h_tr_ms_ref = self.0.conduction.h_tr_ms.as_ref();

            for i in 0..self.0.hvac.num_zones {
                let tm_old = mass_temps_ref[i];
                let cm = thermal_cap_ref[i];
                let t_i = t_i_act.as_ref()[i];
                let h_tr_em = h_tr_em_ref[i];
                let h_tr_ms = h_tr_ms_ref[i];
                let h_tr_is_zone = self.0.conduction.h_tr_is.as_ref()[i];
                let h_tr_me_zone = self.0.mass.h_tr_me.as_ref()[i];
                let t_ext = scratch.t_sol_air().get(i).copied().unwrap_or(outdoor_temp);

                let t_i_blended = t_i; // Use full t_i for surface temperature

                // Surface temperature using blended t_i
                let phi_st_zone = phi_st.as_ref()[i];
                let ts_den = h_tr_ms + h_tr_is_zone + h_tr_me_zone;
                let t_s = if ts_den > 0.0 {
                    (h_tr_ms * tm_old + h_tr_is_zone * t_i_blended + phi_st_zone) / ts_den
                } else {
                    t_i_blended
                };

                // Issue #896 FIX: Use h_tr_3 (combined air-to-mass conductance ≈ 40 W/K)
                // instead of h_tr_ms (direct surface-to-mass ≈ 1300 W/K).
                //
                // The mass node in ISO 13790 receives heat from the air node through
                // H_tr_3, which is the SERIES combination of (air-to-surface + surface-to-mass).
                // This creates an air-side bottleneck that slows the mass response to ~6 days,
                // matching measured building dynamics. Using h_tr_ms directly gives ~4.5 hours,
                // which is far too fast and causes the mass to not cool sufficiently at night.
                //
                // The h_tr_3 conductance is computed once at initialization from:
                //   H_tr_3 = 1 / (1/H_tr_2 + 1/h_tr_ms)
                //   where H_tr_2 = H_tr_1 + h_tr_w, and H_tr_1 = h_ve * h_tr_is / (h_ve + h_tr_is)
                let h_tr_3_zone = *self
                    .0
                    .conduction
                    .derived_h_tr_3
                    .as_ref()
                    .get(i)
                    .unwrap_or(&h_tr_ms);

                // Backward Euler with h_tr_3 instead of h_tr_ms:
                // (Cm/dt + h_tr_em + h_tr_3) * Tm_new = Cm/dt * Tm_old + h_tr_em * t_ext + h_tr_3 * t_s + phi_m
                let cm_dt = cm / dt;
                let denom = cm_dt + h_tr_em + h_tr_3_zone;
                let numer =
                    cm_dt * tm_old + h_tr_em * t_ext + h_tr_3_zone * t_s + phi_m.as_ref()[i];
                // Issue #1219: Guard against division by zero when denom is near-zero
                let tm_new = if denom.abs() > 1e-10 {
                    numer / denom
                } else {
                    // Fallback: use previous mass temperature if system is degenerate
                    tm_old
                };

                scratch.new_mass[i] = tm_new;
            }
            let new_mass_temps_vf: T =
                VectorField::from_smallvec(std::mem::take(&mut scratch.new_mass)).into();
            self.0.mass.previous_mass_temperatures =
                std::mem::replace(&mut self.0.mass.mass_temperatures, new_mass_temps_vf);
        }

        // Issue #738 / ADR-002 (#1175): Free-float mode disables HVAC output.
        //
        // The COMMITTED zone temperature is `t_i_free_mn`, which holds, per zone:
        //   - high-mass (has a MultiNodeSolver): the 9R4C multi-node air temperature
        //     from `compute_zone_air_temperature` (mass/surface nodes stepped by
        //     backward Euler, sol-air driven, physics-based per-surface `h_tr_ms`);
        //   - low-mass (no MultiNodeSolver): the legacy 5R1C `t_i_free`.
        // This makes 9R4C the sole thermal solver for high-mass free-float: the
        // result is governed by the multi-node network's physics-based `h_tr_ms`
        // (k·A/d), not the legacy coefficient-tuned `h_ms_coeff`. The 5R1C lumped
        // mass (updated above from `t_i_act = t_i_free_5r1c`) continues to evolve on
        // its own dynamics but no longer drives the high-mass air temperature.
        // Low-mass free-float is unchanged.
        if self.0.hvac.free_float {
            let temps_slice = self.0.setpoints.temperatures.as_mut();
            for (i, t_val) in t_i_free_mn.as_ref().iter().enumerate() {
                if i < temps_slice.len() {
                    temps_slice[i] = *t_val;
                }
            }
            // Issue #2756: restore the pooled scratch on this early-return path
            // too so the next timestep reuses the same SmallVec capacity.
            self.0.hvac.scratch_pool.return_9r4c(scratch);
            return 0.0;
        }

        // Update zone temperatures with the HVAC-influenced t_i_act
        let temps_slice = self.0.setpoints.temperatures.as_mut();
        for (i, t_val) in t_i_act.as_ref().iter().enumerate() {
            if i < temps_slice.len() {
                temps_slice[i] = *t_val;
            }
        }

        // Calculate and return total HVAC output (energy)
        let hvac_output = hvac_for_temp_calc;

        // Accumulate annual energy and track peak power without cloning
        let mut hvac_power_watts = 0.0;
        {
            let mut heating_sum = 0.0_f64;
            let mut cooling_sum = 0.0_f64;
            for (i, (&output, &enabled)) in hvac_output
                .as_ref()
                .iter()
                .zip(self.0.hvac.hvac_enabled.as_ref().iter())
                .enumerate()
            {
                let val = if enabled > 0.5 { output } else { 0.0 };
                hvac_power_watts += val;

                if val > 0.0 {
                    heating_sum += val;
                    // Issue #1289: Track per-zone peaks
                    // Issue #1628: Also track timestep when peak occurred
                    let val_kw = val / 1000.0;
                    if val_kw > self.0.hvac.zone_peak_heating_kw.as_mut()[i] {
                        self.0.hvac.zone_peak_heating_kw.as_mut()[i] = val_kw;
                        self.0.hvac.zone_peak_heating_timestep[i] = timestep;
                    }
                } else if val < 0.0 {
                    cooling_sum += -val;
                    // Issue #1289: Track per-zone peaks
                    // Issue #1628: Also track timestep when peak occurred
                    let val_kw = -val / 1000.0;
                    if val_kw > self.0.hvac.zone_peak_cooling_kw.as_mut()[i] {
                        self.0.hvac.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                        self.0.hvac.zone_peak_cooling_timestep[i] = timestep;
                    }
                }
            }

            let heating_energy_joules = heating_sum * dt;
            let cooling_energy_joules = cooling_sum * dt;

            self.0.hvac.annual_heating_energy += heating_energy_joules / 3.6e6;
            self.0.hvac.annual_cooling_energy += cooling_energy_joules / 3.6e6;

            // Per-zone energy accumulation (Issue #1288)
            // Use enabled-masked values for per-zone accumulation
            let enabled_vec = self.0.hvac.hvac_enabled.as_ref();
            let zone_heating_slice = self.0.hvac.zone_heating_energy_kwh.as_mut();
            let zone_cooling_slice = self.0.hvac.zone_cooling_energy_kwh.as_mut();
            for i in 0..self.0.hvac.num_zones {
                let val = if enabled_vec[i] > 0.5 {
                    hvac_output.as_ref()[i]
                } else {
                    0.0
                };
                // dt is in seconds, convert to kWh: watts * seconds / 3.6e6
                let energy_kwh = val * dt / 3.6e6;
                if val > 0.0 {
                    zone_heating_slice[i] += energy_kwh;
                } else {
                    zone_cooling_slice[i] += -energy_kwh;
                }
            }

            if hvac_power_watts > 0.0 {
                self.0.hvac.peak_power_heating =
                    self.0.hvac.peak_power_heating.max(hvac_power_watts);
            } else if hvac_power_watts < 0.0 {
                self.0.hvac.peak_power_cooling =
                    self.0.hvac.peak_power_cooling.max(-hvac_power_watts);
            }
        }

        // Diagnostics recording (if enabled)
        if self.0.diagnostics_state.diagnostics.is_some() {
            self.0.hvac.current_hvac_output = Some(hvac_output);
            let mut diag = self.0.diagnostics_state.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics_state.diagnostics = Some(diag);
            self.0.hvac.current_hvac_output = None;
        }

        // Issue #2756: restore the pooled scratch so the next timestep reuses
        // the same SmallVec capacity (zero steady-state allocation).
        self.0.hvac.scratch_pool.return_9r4c(scratch);

        // Return kWh
        hvac_power_watts * dt / 3.6e6
    }
}

#[cfg(test)]
mod scratch_pool_tests {
    //! Issue #2756 — prove `PhysicsScratchPool` is live (no longer dead code)
    //! and that its buffers are reused across timesteps rather than
    //! re-allocated per step.
    //!
    //! These tests live in-crate (not under `tests/`) because they must read
    //! `model.0.hvac.scratch_pool` — a `pub(crate)` field — to assert reuse. The
    //! dhat gate under `tests/dhat_step_physics_zero_alloc.rs` provides the
    //! independent heap-growth measurement.

    use crate::sim::construction::WallSurface;
    use crate::sim::solar::WindowProperties;
    use crate::sim::thermal_model_core::ThermalModel;
    use crate::weather::HourlyWeatherData;
    use fluxion_core::ashrae_cases::Orientation;
    // `VectorField` is already imported by the outer module (line 16). Reuse it
    // via `super::` rather than a fresh `use crate::physics::cta::VectorField` —
    // that would add a new sim→physics edge and trip the Physics-Sim-Cycle-Check
    // gate (#2463 / scripts/check_physics_sim_cycle.py).
    use super::VectorField;

    /// More than 4 zones so every `SmallVec<[f64; 4]>` field SPILLS to heap —
    /// that is the regime where per-step `new(num_zones)` allocation is
    /// observable and where the pool pays off. At ≤4 zones the fields are inline
    /// and the pool is a no-op (still correct, just not measurable by pointer
    /// stability).
    const NUM_ZONES: usize = 10;

    /// Mirrors the `dhat_zone_solar_gain_zero_alloc` fixture: a 10-zone model
    /// each with 5 windowed surfaces so `calc_analytical_loads` →
    /// `calculate_zone_solar_gain` is fully populated and `step_physics` runs the
    /// production 5R1C analytical path without panic.
    fn multizone_model() -> ThermalModel<VectorField> {
        let mut model = ThermalModel::<VectorField>::new(NUM_ZONES);
        model.solar.window_u_value = 1.5;
        model.setpoints.heating_setpoint = 20.0;
        model.setpoints.cooling_setpoint = 26.0;
        model.setpoints.temperatures = VectorField::from_scalar(20.0, NUM_ZONES);
        model.mass.mass_temperatures = VectorField::from_scalar(20.0, NUM_ZONES);
        model.setpoints.zone_area = VectorField::from_scalar(50.0, NUM_ZONES);

        let wp = WindowProperties::double_clear(8.0);
        model.solar.window_properties = vec![wp; NUM_ZONES];

        let surfaces_per_zone: Vec<Vec<WallSurface>> = (0..NUM_ZONES)
            .map(|_| {
                vec![
                    WallSurface::new(10.0, 0.5, Orientation::North).with_window(2.0),
                    WallSurface::new(10.0, 0.5, Orientation::East).with_window(2.0),
                    WallSurface::new(10.0, 0.5, Orientation::South).with_window(2.0),
                    WallSurface::new(10.0, 0.5, Orientation::West).with_window(2.0),
                    WallSurface::new(25.0, 0.3, Orientation::Up),
                ]
            })
            .collect();
        model.solar.surfaces = surfaces_per_zone;

        model
    }

    fn weather(hour: usize) -> HourlyWeatherData {
        HourlyWeatherData::new(20.0, 400.0, 100.0, 500.0, 2.0, 50.0, hour)
    }

    /// The pool must be populated by every 5R1C step, and the NON-`mem::take`'n
    /// fields (e.g. `wall_surface_new`) must keep the SAME heap buffer across
    /// steps — the direct proof that `checkout`/`fill_zero`/`return` reuses
    /// capacity instead of re-allocating.
    ///
    /// (`phi_ia`/`phi_st`/`phi_m`/`t_i_act`/`new_mass` are emptied each step by
    /// `mem::take` into a `VectorField`, so their pointer is expected to move;
    /// they are the residual per-step allocation tracked separately by the dhat
    /// gate. `wall_surface_new` and `wall_surface_correction` are written by
    /// index and never taken, so the pool reuses their buffer verbatim.)
    #[test]
    fn scratch_pool_5r1c_is_reused_across_timesteps() {
        let mut model = multizone_model();

        // Before any step the pool is empty (lazy).
        assert!(
            model.0.hvac.scratch_pool.r5r1c.is_none(),
            "pool must start empty"
        );

        // First step: checkout allocates, return restores into the pool.
        model.solar.weather = Some(weather(0));
        model.step_physics(0, 20.0, 3600.0);
        let pool = model
            .0
            .hvac
            .scratch_pool
            .r5r1c
            .as_ref()
            .expect("5R1C pool must be populated after step 1 (Issue #2756)");
        let ptr_warm = pool.wall_surface_new.as_ptr();
        let len_warm = pool.wall_surface_new.len();
        assert_eq!(len_warm, NUM_ZONES);

        // Drive 99 more steps — the pool must remain populated and the
        // wall_surface_new buffer must be the SAME allocation (pointer-stable).
        for step in 1..100 {
            model.solar.weather = Some(weather(step % 24));
            model.step_physics(step, 20.0, 3600.0);
        }
        let pool = model
            .0
            .hvac
            .scratch_pool
            .r5r1c
            .as_ref()
            .expect("5R1C pool must still be populated after 100 steps");
        let ptr_steady = pool.wall_surface_new.as_ptr();

        assert_eq!(
            ptr_warm, ptr_steady,
            "wall_surface_new buffer must be reused (same heap allocation) \
             across 100 timesteps — if this fails, the pool stopped reusing \
             the non-taken scratch fields and per-timestep allocation regressed."
        );
        assert_eq!(pool.wall_surface_new.len(), NUM_ZONES);
    }

    /// Same reuse property for the 9R4C pool. Exercises the dispatcher's 9R4C
    /// branch and the early-return path in `step_physics_9r4c` (free_float is
    /// false here, so the early return is not taken; the final return restores
    /// the pool). The `inter` flat buffer (num_zones × 7) is never `mem::take`'n
    /// — only sliced — so its pointer must be stable across steps.
    #[test]
    fn scratch_pool_9r4c_is_reused_across_timesteps() {
        let mut model = multizone_model();
        model.0.hvac.thermal_model_type =
            crate::sim::thermal_model_core::ThermalModelType::NineRFourC;

        assert!(model.0.hvac.scratch_pool.r9r4c.is_none());

        model.solar.weather = Some(weather(0));
        model.step_physics(0, 20.0, 3600.0);
        let inter_ptr_warm = model
            .0
            .hvac
            .scratch_pool
            .r9r4c
            .as_ref()
            .expect("9R4C pool populated after step 1")
            .inter
            .as_ptr();

        for step in 1..50 {
            model.solar.weather = Some(weather(step % 24));
            model.step_physics(step, 20.0, 3600.0);
        }
        let inter_ptr_steady = model
            .0
            .hvac
            .scratch_pool
            .r9r4c
            .as_ref()
            .expect("9R4C pool still populated")
            .inter
            .as_ptr();

        assert_eq!(
            inter_ptr_warm, inter_ptr_steady,
            "9R4C `inter` buffer must be reused across timesteps"
        );
    }
}

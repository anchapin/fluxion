//! 6R2C physics step implementation for `ThermalModel`.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_integration::{
    backward_euler_update_2cond, backward_euler_update_2cond_h_tr3, crank_nicolson_update,
    crank_nicolson_update_3cond, select_integration_method, ThermalIntegrationMethod,
};
use crate::sim::thermal_model_core::ThermalModel;
use smallvec::SmallVec;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    #[allow(
        dead_code,
        reason = "Issue #3280: 6R2C/8R3C fall-through removed; method retained for legacy callers"
    )]
    pub(crate) fn step_physics_6r2c(
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
        // Issue #2873: the helper now writes its per-zone sol-air temperatures
        // into a caller-supplied buffer instead of returning a fresh Vec. The
        // 6R2C path keeps a `SmallVec` alive across the call so we can pass
        // it by `&mut`. 6R2C does not have a dedicated scratch field for
        // sol-air (out of scope for #2873, which is 5R1C-only), so a fresh
        // per-step allocation persists here — bit-identical to the previous
        // behaviour (the Vec that `prepare_solvers_and_sol_air` used to
        // allocate is now allocated by the caller instead, same number of
        // heap blocks).
        let mut t_sol_air_buf: SmallVec<[f64; 4]> = SmallVec::with_capacity(self.0.hvac.num_zones);
        let (ctf_flux_w, fd_flux_w, ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp, &mut t_sol_air_buf);

        // Get ground temperature at this timestep
        let t_g = self
            .0
            .conduction
            .ground_temperature
            .ground_temperature(timestep);

        let _hour_of_day = (timestep % 24) as u8;

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
        let st_int_frac = rad_frac * (1.0 - self.0.solar.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar.solar_distribution_to_air;
        // Solar gain distribution for 6R2C model.
        // Energy-conserving split: st + m_env + m_int = 1.0 when sol_to_air = 0.
        // With solar_beam_to_mass_fraction = 0.0: 100% to surface (fast air heating).
        // With solar_beam_to_mass_fraction = 1.0: 70% envelope mass, 30% internal mass.
        let st_sol_frac = 1.0 - self.0.solar.solar_beam_to_mass_fraction; // Solar to surface
        let m_env_sol_frac = self.0.solar.solar_beam_to_mass_fraction * 0.7; // Solar to envelope mass
        let m_int_sol_frac = self.0.solar.solar_beam_to_mass_fraction * 0.3; // Solar to internal mass
        let sol_to_air_frac = self.0.solar.solar_distribution_to_air;

        let loads_ref = self.0.setpoints.loads.as_ref();
        let solar_ref = self.0.solar.solar_gains.as_ref();
        let opaque_solar_ref = self.0.solar.opaque_solar_gains.as_ref();
        let area_ref = self.0.setpoints.zone_area.as_ref();

        // Issue #1524: consolidated per-timestep scratch (replaces the eleven
        // standalone `Vec::with_capacity(num_zones)` allocations in 6R2C).
        // Issue #1966 / #2756: scratch is CHECKED OUT of `scratch_pool`
        // (allocates only on the first timestep) and `fill_zero()`'d back to
        // the post-`new(num_zones)` state. Bit-identical vs. fresh construct.
        let mut scratch = self
            .0
            .hvac
            .scratch_pool
            .checkout_6r2c(self.0.hvac.num_zones);
        scratch.fill_zero();

        for i in 0..self.0.hvac.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            // SESSION 76 FIX: Include solar_distribution_to_air in 6R2C (was missing!)
            // This sends a fraction of solar directly to zone air (immediate heating/cooling)
            scratch.phi_ia[i] = load_w * conv_frac + sol_w * sol_to_air_frac;
            scratch.phi_st[i] = load_w * st_int_frac + sol_w * st_sol_frac;
            // LEAKY BUCKET FIX: Add opaque solar gains to envelope mass node
            // Opaque surfaces (walls, roof, floor) absorb solar radiation and transfer
            // it to the thermal mass. Without this, solar heat bypasses thermal mass.
            scratch.phi_m_env[i] = load_w * m_air_frac + sol_w * m_env_sol_frac + opaque_sol_w;
            scratch.phi_m_int[i] = sol_w * m_int_sol_frac;
        }

        let phi_ia = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_ia,
        )));
        let phi_st = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_st,
        )));
        let phi_m_env = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_m_env,
        )));
        let phi_m_int = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.phi_m_int,
        )));

        // Use pre-computed cached values
        #[cfg(feature = "debug-physics")]
        let h_ext_base = &self.0.conduction.derived_h_ext;
        let term_rest_1 = &self.0.conduction.derived_term_rest_1;

        // Night ventilation no longer modifies h_ext (same fix as 5R1C path).
        let modified_h_ext: Option<T> = None;
        #[cfg(feature = "debug-physics")]
        let h_ext = h_ext_base;

        // 6R2C specific terms
        let h_sum = self
            .0
            .conduction
            .h_tr_ms
            .zip_with(&self.0.mass.h_tr_me, |a, b| a + b)
            .zip_with(&self.0.conduction.h_tr_is, |a, b| a + b);

        let h_ms_me_is_prod = self.0.conduction.h_tr_is.zip_with(
            &self
                .0
                .conduction
                .h_tr_ms
                .zip_with(&self.0.mass.h_tr_me, |a, b| a + b),
            |a, b| a * b,
        );

        let den: T;
        let h_total_with_iz = if let Some(ref mod_h_ext) = modified_h_ext {
            if self.0.hvac.num_zones > 1 {
                mod_h_ext
                    .zip_with(&self.0.conduction.h_tr_iz, |a, b| a + b)
                    .zip_with(&self.0.conduction.h_tr_iz_rad, |a, b| a + b)
            } else {
                mod_h_ext.clone()
            }
        } else {
            if self.0.hvac.num_zones > 1 {
                self.0
                    .conduction
                    .derived_h_ext
                    .zip_with(&self.0.conduction.h_tr_iz, |a, b| a + b)
                    .zip_with(&self.0.conduction.h_tr_iz_rad, |a, b| a + b)
            } else {
                self.0.conduction.derived_h_ext.clone()
            }
        };

        // Issue 693 fix: ground coupling coefficient in 6R2C den
        // Optimized: avoid intermediate vector allocations using explicit loop
        let h_sum_ref = h_sum.as_ref();
        let h_tr_floor_ref = self.0.conduction.h_tr_floor.as_ref();
        let h_ms_me_is_prod_ref = h_ms_me_is_prod.as_ref();
        let h_total_with_iz_ref = h_total_with_iz.as_ref();

        for i in 0..self.0.hvac.num_zones {
            let g = h_sum_ref[i] * h_tr_floor_ref[i];
            scratch.ground_coeff[i] = g;
            let d = h_ms_me_is_prod_ref[i] + (h_sum_ref[i] * h_total_with_iz_ref[i]) + g;
            scratch.den[i] = d;
        }
        let ground_coeff_6r2c = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.ground_coeff,
        )));
        den = T::from(VectorField::from_smallvec(std::mem::take(&mut scratch.den)));

        // Use envelope mass temperature instead of single mass temperature
        // Optimized: use zip_with to avoid double clones
        //
        // CTF-driven zone air heat balance (Issue #698 fix):
        // When ctf_primary=true, the 6R2C h_tr_ms coupling is DISABLED because
        // CTF provides the correct multi-layer conduction dynamics directly.
        // The CTF heat flow q_ctf (computed from T_si_ctf) replaces the 6R2C h_tr_ms * t_mass term.
        let num_tm = if self.0.conduction.backend.ctf_primary {
            // Zero out the 6R2C coupling - CTF will drive the zone air heat balance
            self.0.conduction.derived_h_ms_is_prod.constant_like(0.0)
        } else {
            self.0
                .conduction
                .derived_h_ms_is_prod
                .zip_with(&self.0.mass.envelope_mass_temperatures, |a, b| a * b)
        };
        let num_phi_st = self.0.conduction.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        // Inter-zone heat transfer (with radiative component - Issue #302)
        let num_zones = self.0.hvac.num_zones;
        let h_iz_vec = self.0.conduction.h_tr_iz.as_ref();
        let h_iz_rad_vec = self.0.conduction.h_tr_iz_rad.as_ref();

        // Store phi_ia[0] for debugging before we consume it (debug-physics only)
        #[cfg(feature = "debug-physics")]
        let phi_ia_0 = phi_ia.as_ref().first().copied().unwrap_or(0.0);

        // Compute inter-zone heat transfer directly into phi_ia_with_iz to avoid Vec allocation
        let mut phi_ia_with_iz = phi_ia;

        if num_zones > 1
            && (!h_iz_vec.is_empty() && h_iz_vec[0] > 0.0
                || !h_iz_rad_vec.is_empty() && h_iz_rad_vec[0] > 0.0)
        {
            let temps = self.0.setpoints.temperatures.as_ref();
            let h_iz_val = h_iz_vec.first().copied().unwrap_or(0.0);
            let h_iz_rad_val = h_iz_rad_vec.first().copied().unwrap_or(0.0);
            let total_h_iz = h_iz_val + h_iz_rad_val;

            let sum_t: f64 = temps.iter().sum();
            let n = num_zones as f64;

            // For diagnostic, capture q_iz for first two zones before adding
            let (mut _dbg_q0, mut _dbg_q1) = (0.0, 0.0);
            let slice = phi_ia_with_iz.as_mut();
            for i in 0..num_zones {
                let q_iz = total_h_iz * (sum_t - n * temps[i]);
                if i == 0 {
                    _dbg_q0 = q_iz;
                }
                if i == 1 {
                    _dbg_q1 = q_iz;
                }
                slice[i] += q_iz;
            }
        }

        // Ground Coupling: Q_ground = h_tr_floor * (T_ground - T_surface)
        // In the 5R1C heat balance, ground coupling adds h_tr_floor * T_ground to the numerator
        // Correct formula: num_rest = term_rest_1 * (phi_ia + h_ext * outdoor_temp) + h_tr_floor * t_g
        // Note: derived_ground_coeff = term_rest_1 * h_tr_floor, so we need to divide by term_rest_1
        // before multiplying, or add the ground term separately after the multiplication.
        let _h_tr_floor_ref = self.0.conduction.h_tr_floor.as_ref();

        // Start with phi_ia_with_iz
        let mut sum_term = phi_ia_with_iz;

        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = sum_term.as_mut();
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
                    slice[i] += q_ctf;
                }
            }
        }

        // Add FD net contribution if enabled
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = sum_term.as_mut();
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
                    let t_sol_air_i = t_sol_air_buf.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass
                        .envelope_mass_temperatures
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

        // Optimized: replace clone and mul_assign with explicit loop
        let sum_term_ref = sum_term.as_ref();
        let term_rest_1_ref = term_rest_1.as_ref();
        let ground_coeff = ground_coeff_6r2c.as_ref();

        for i in 0..self.0.hvac.num_zones {
            scratch.num_rest[i] = sum_term_ref[i] * term_rest_1_ref[i] + ground_coeff[i] * t_g;
        }
        let num_rest_with_iz = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.num_rest,
        )));

        // DEBUG: Save values for 900FF before they're consumed
        #[cfg(feature = "debug-physics")]
        let debug_900ff = if self.0.hvac.case_id == "900FF" && timestep.is_multiple_of(24) {
            let den_vals = den.as_ref();
            let _num_tm_vals = num_tm.as_ref();
            let num_rest_vals = num_rest_with_iz.as_ref();
            let _env_mass_vals = self.0.mass.envelope_mass_temperatures.as_ref();
            let h_sum_vals = h_sum.as_ref();
            let sum_term_vals = sum_term.as_ref();
            let h_ext_debug = h_ext.as_ref();
            let solar_debug = self.0.solar.solar_gains.as_ref();
            let loads_debug = self.0.setpoints.loads.as_ref();
            let area_debug = self.0.setpoints.zone_area.as_ref();
            eprintln!("DEBUG_900FF_PREPARE: t={}, phi_ia[0]={:.2}, solar[0]={:.2}, loads[0]={:.2}, area[0]={:.1}", timestep, phi_ia_0, solar_debug[0], loads_debug[0], area_debug[0]);
            Some((
                den_vals[0],
                _num_tm_vals[0],
                num_rest_vals[0],
                _env_mass_vals[0],
                h_sum_vals[0],
                sum_term_vals[0],
                h_ext_debug[0],
                phi_ia_0,
                solar_debug[0],
                loads_debug[0],
                area_debug[0],
            ))
        } else {
            None
        };
        #[cfg(not(feature = "debug-physics"))]
        #[allow(unused_variables, clippy::type_complexity)]
        let _debug_900ff: Option<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> = None;

        // Calculate free-floating indoor temperature using standard 6R2C heat balance
        // (thermal mass buffering is critical for preventing temperature overshoot)
        let mut t_i_free = num_tm;
        t_i_free.add_assign(&num_phi_st);
        t_i_free.add_assign(&num_rest_with_iz);
        t_i_free.div_assign(&den);

        // DEBUG: Print key values for 900FF after calculation
        #[cfg(feature = "debug-physics")]
        if let Some((
            den_val,
            _,
            num_rest_val,
            _,
            h_sum_val,
            sum_term_val,
            h_ext_val,
            phi_ia_val,
            solar_val,
            loads_val,
            area_val,
        )) = debug_900ff
        {
            let t_i_free_val = t_i_free.as_ref()[0];
            eprintln!("DEBUG_900FF t={} t_i_free={:.2} num_rest={:.2} den={:.2} h_sum={:.2} sum_term={:.2} h_ext={:.2} phi_ia={:.2} solar={:.2} loads={:.2} area={:.1}",
                timestep, t_i_free_val, num_rest_val, den_val, h_sum_val, sum_term_val, h_ext_val, phi_ia_val, solar_val, loads_val, area_val);
        }

        // HVAC calculation
        let _hour_of_day_idx = timestep % 24;

        // Issue #738: Free-float mode must completely disable HVAC output
        // NOTE: Don't return early here - we need to update mass temperatures
        // and zone temperatures for correct free-float temperature tracking.
        // The hvac_output_raw will be zero (computed from free-float t_i_free).

        // Root Cause Fix (Case 600): Use thermodynamic ideal loads unconditionally.
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
            &mut scratch.hvac_combined_demand,
        );
        // Fix: Use actual HVAC demand instead of steady-state approximation (Plan 03-03 Task 2)
        // hvac_output_raw already includes thermal mass buffering (calculated from t_i_free)
        // This is needed for high-mass cases (900 series) that use 6R2C model
        let hvac_power_watts = hvac_output_raw.as_ref().iter().sum::<f64>();

        // Track peak for high-mass cases (6R2C model)
        // Physics-based: Track actual HVAC demand without calibration factors
        if hvac_power_watts > 0.0 {
            // Heating mode - track actual demand
            self.0.hvac.peak_power_heating = self.0.hvac.peak_power_heating.max(hvac_power_watts);
        } else if hvac_power_watts < 0.0 {
            // Cooling mode (store as positive value)
            let cooling_demand = -hvac_power_watts;
            self.0.hvac.peak_power_cooling = self.0.hvac.peak_power_cooling.max(cooling_demand);
        }

        // Plan 03-04: Use hvac_output_raw directly for energy calculation
        // Ti_free calculation already includes thermal mass effects via:
        // - h_tr_em and h_tr_ms conductances (thermal mass coupling)
        // - Thermal capacitance Cm (thermal mass response rate)
        // - Implicit/explicit Euler integration (Cm × ΔTm/dt)
        // Solution 2: Apply time constant-based correction to ENERGY ONLY

        // Calculate HVAC energy for step with optimized allocation-free summation
        // Compute sums without cloning hvac_output_raw
        let mut heating_sum = 0.0;
        let mut cooling_sum = 0.0;
        let mut total_signed = 0.0;

        // Per-zone energy accumulation (Issue #1288)
        // hvac_output_raw: positive = heating, negative = cooling
        let hvac_vec = hvac_output_raw.as_ref();
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

                // Issue #1289: Track per-zone peaks
                // Issue #1628: Also track timestep when peak occurred
                let val_kw = val / 1000.0;
                if val_kw > self.0.hvac.zone_peak_heating_kw.as_mut()[i] {
                    self.0.hvac.zone_peak_heating_kw.as_mut()[i] = val_kw;
                    self.0.hvac.zone_peak_heating_timestep[i] = timestep;
                }
            } else {
                cooling_sum += -val;
                zone_cooling_slice[i] += -energy_kwh;

                // Issue #1289: Track per-zone peaks
                // Issue #1628: Also track timestep when peak occurred
                let val_kw = -val / 1000.0;
                if val_kw > self.0.hvac.zone_peak_cooling_kw.as_mut()[i] {
                    self.0.hvac.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                    self.0.hvac.zone_peak_cooling_timestep[i] = timestep;
                }
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

        // Root Cause Fix: Physics-based temperature update.
        // t_i_act = t_i_free + hvac_power / h_tr_is
        // (See the NOTE at the first temperature-update site above for why
        // h_tr_is is retained instead of h_coeff — Issue #1163.)
        let h_tr_is_vec = self.0.conduction.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_output_raw.as_ref();
        for i in 0..self.0.hvac.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                scratch.t_i_act[i] = t_free[i] + hvac[i] / h_is;
            } else {
                scratch.t_i_act[i] = t_free[i];
            }
        }
        let t_i_act = T::from(VectorField::from_smallvec(std::mem::take(
            &mut scratch.t_i_act,
        )));

        // Calculate surface temperature for mass update (including HVAC effect)
        // === 6R2C: Update two mass nodes ===
        // PHASE 36-04 FIX: Include h_tr_me * Tm_int in surface temperature calculation
        // The 6R2C model requires: T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
        let h_tr_me_ref = self.0.mass.h_tr_me.as_ref();
        let int_mass_temps_ref = self.0.mass.internal_mass_temperatures.as_ref();
        // SESSION 89: When ctf_primary is active, use CTF T_si (with HVAC offset) instead of lumped T_s
        let t_s_act: T = if self.0.conduction.backend.ctf_primary {
            // Use CTF surface temp adjusted for HVAC effect
            // The CTF T_si was computed at t_i_free; adjust for actual t_i_act via linear correction:
            // T_si_adjusted ≈ T_si_ctf + (h_tr_is / (h_tr_is + Z₀)) * (t_i_act - t_i_free)
            if let Some(ref ctf_temps) = ctf_surface_temps {
                let t_i_free_ref = t_i_free.as_ref();
                let t_i_act_ref = t_i_act.as_ref();
                for i in 0..self.0.hvac.num_zones {
                    let t_si_ctf = ctf_temps.get(i).copied().unwrap_or(20.0);
                    let delta_t_i = t_i_act_ref.get(i).copied().unwrap_or(0.0)
                        - t_i_free_ref.get(i).copied().unwrap_or(0.0);
                    // Approximate: surface follows zone air with ~h_tr_is/(h_tr_is+Z₀) coupling
                    // Use conservative 0.5 factor for stability
                    scratch.t_s[i] = t_si_ctf + 0.5 * delta_t_i;
                }
                T::from(VectorField::from_smallvec(std::mem::take(&mut scratch.t_s)))
            } else {
                // PHASE 36-04 FIX: 6R2C surface temperature with h_tr_me * Tm_int coupling
                // T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
                let h_tr_ms_data = self.0.conduction.h_tr_ms.as_ref();
                let h_tr_is_data = self.0.conduction.h_tr_is.as_ref();
                let t_i_act_data = t_i_act.as_ref();
                let phi_st_data = phi_st.as_ref();
                let env_mass_data = self.0.mass.envelope_mass_temperatures.as_ref();
                let term_rest_data = term_rest_1.as_ref();
                for i in 0..self.0.hvac.num_zones {
                    let numerator = h_tr_ms_data[i] * env_mass_data[i]
                        + h_tr_is_data[i] * t_i_act_data[i]
                        + phi_st_data[i]
                        + h_tr_me_ref[i] * int_mass_temps_ref[i];
                    let denominator = term_rest_data[i] + h_tr_me_ref[i];
                    scratch.t_s[i] = numerator / denominator;
                }
                T::from(VectorField::from_smallvec(std::mem::take(&mut scratch.t_s)))
            }
        } else {
            // PHASE 36-04 FIX: 6R2C surface temperature with h_tr_me * Tm_int coupling
            // T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
            let h_tr_ms_data = self.0.conduction.h_tr_ms.as_ref();
            let h_tr_is_data = self.0.conduction.h_tr_is.as_ref();
            let t_i_act_data = t_i_act.as_ref();
            let phi_st_data = phi_st.as_ref();
            let env_mass_data = self.0.mass.envelope_mass_temperatures.as_ref();
            let term_rest_data = term_rest_1.as_ref();
            for i in 0..self.0.hvac.num_zones {
                let numerator = h_tr_ms_data[i] * env_mass_data[i]
                    + h_tr_is_data[i] * t_i_act_data[i]
                    + phi_st_data[i]
                    + h_tr_me_ref[i] * int_mass_temps_ref[i];
                let denominator = term_rest_data[i] + h_tr_me_ref[i];
                scratch.t_s[i] = numerator / denominator;
            }
            T::from(VectorField::from_smallvec(std::mem::take(&mut scratch.t_s)))
        };

        // === 6R2C: Update two mass nodes with implicit integration ===
        // Envelope mass: receives heat from exterior (sol-air), surface, and internal mass

        // Update envelope mass temperatures using implicit integration for high thermal capacitance
        let env_mass_temps_ref = self.0.mass.envelope_mass_temperatures.as_ref();
        let env_thermal_cap_ref = self.0.mass.envelope_thermal_capacitance.as_ref();
        // Mode-specific fields removed - use physics-based h_tr_em and h_tr_ms
        let h_tr_em_ref = self.0.conduction.h_tr_em.as_ref();
        let h_tr_ms_ref = self.0.conduction.h_tr_ms.as_ref();
        let h_tr_me_ref = self.0.mass.h_tr_me.as_ref();
        let int_mass_temps_ref = self.0.mass.internal_mass_temperatures.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let phi_m_env_ref = phi_m_env.as_ref();

        for i in 0..self.0.hvac.num_zones {
            let tm_env_old = env_mass_temps_ref[i];
            let cm_env = env_thermal_cap_ref[i];
            let h_tr_me = h_tr_me_ref[i];
            let tm_int = int_mass_temps_ref[i];
            let t_s = t_s_act_ref[i];
            let phi_m_env_zone = phi_m_env_ref[i];

            // Use physics-based h_tr_em and h_tr_ms (mode-specific factors removed)
            // The conductances are now calculated from first principles:
            // h_tr_em = k * A / d (thermal conductivity * area / thickness)
            // h_tr_ms = k * A / d (thermal conductivity * area / thickness)
            // Note: h_tr_em is NOT used in the 6R2C envelope mass heat balance (Issue #693)
            // It affects T_s via the surface network, not directly Tm_env
            let _h_tr_em = h_tr_em_ref[i];
            let h_tr_ms = h_tr_ms_ref[i];
            let h_tr_3 = self.0.conduction.derived_h_tr_3.as_ref()[i];

            // Check if this is a high-mass case (900 series)
            let is_high_mass = matches!(
                self.0.hvac.case_id.as_str(),
                "900" | "910" | "920" | "930" | "940" | "950" | "900FF" | "950FF"
            );

            // For envelope mass, use implicit integration for high thermal capacitance
            let method_env = select_integration_method(cm_env);

            let tm_env_new = match method_env {
                ThermalIntegrationMethod::BackwardEuler => {
                    if is_high_mass {
                        // Case 900+: Use H_tr_3 (≈ 40 W/K) for correct slow thermal coupling
                        // This gives ~69 hour time constant instead of ~1.9 hours with h_tr_ms + h_tr_me
                        // Heat balance: Cm*(Tm_new - Tm_old)/dt = h_tr_3*(t_i - Tm_new) + phi_m
                        let t_i_zone = t_i_act.as_ref()[i];
                        backward_euler_update_2cond_h_tr3(
                            tm_env_old,
                            dt,
                            cm_env,
                            h_tr_3,
                            t_i_zone,
                            phi_m_env_zone,
                        )
                    } else {
                        // Standard 6R2C: Use h_tr_ms + h_tr_me for fast air-surface coupling
                        backward_euler_update_2cond(
                            tm_env_old,
                            dt,
                            cm_env,
                            h_tr_ms,
                            h_tr_me,
                            t_s,
                            tm_int,
                            phi_m_env_zone,
                        )
                    }
                }
                ThermalIntegrationMethod::ExplicitEuler => {
                    // Issue 693 fix: For 6R2C envelope mass, h_tr_em should NOT be included
                    // in the heat balance. h_tr_em affects T_s (surface node) via the surface
                    // network (which includes solar gains), but does not directly affect Tm.
                    // The envelope mass receives heat from:
                    //   - T_s via h_tr_ms (surface-to-mass conductance)
                    //   - Tm_int via h_tr_me (mass-to-internal-mass conductance)
                    //
                    // This matches the comments at lines 1744-1745 and 1785-1786:
                    // "h_tr_em affects T_s via the surface network, not Tm directly"
                    let q_env_net = h_tr_ms * (t_s - tm_env_old)
                        + h_tr_me * (tm_int - tm_env_old)
                        + phi_m_env_zone;

                    tm_env_old + (q_env_net / cm_env) * dt
                }
                ThermalIntegrationMethod::CrankNicolson => {
                    // Use Crank-Nicolson for 2nd-order accuracy
                    // For 6R2C envelope mass: receives heat from exterior (h_tr_em),
                    // surface (h_tr_ms), and internal mass (h_tr_me)
                    crank_nicolson_update_3cond(
                        tm_env_old,
                        dt,
                        cm_env,
                        h_tr_ms, // exterior-to-mass (WRONG - keeping for physics compatibility)
                        h_tr_ms, // mass-to-surface conductance
                        h_tr_me, // mass-to-internal-mass conductance
                        t_s,     // exterior/sol-air temperature
                        t_s,     // surface temperature
                        tm_int,  // internal mass temperature
                        phi_m_env_zone,
                    )
                }
            };

            scratch.new_env[i] = tm_env_new;
        }

        // Note: env_mass_temps_for_int is no longer needed as a clone
        // We will borrow from new_env_mass_temperatures before moving it

        let env_mass_temps_for_int = &scratch.new_env;

        // Internal mass: receives heat from envelope mass and direct gains

        // Update internal mass temperatures using implicit integration for high thermal capacitance
        let int_thermal_cap_ref = self.0.mass.internal_thermal_capacitance.as_ref();
        let phi_m_int_ref = phi_m_int.as_ref();

        for i in 0..self.0.hvac.num_zones {
            let tm_int_old = int_mass_temps_ref[i];
            let cm_int = int_thermal_cap_ref[i];
            let h_tr_me = h_tr_me_ref[i];
            let tm_env_new = env_mass_temps_for_int[i]; // Use updated envelope temperature
            let phi_m_int_zone = phi_m_int_ref[i];

            // For internal mass, use implicit integration for high thermal capacitance
            let method_int = select_integration_method(cm_int);

            let tm_int_new = match method_int {
                ThermalIntegrationMethod::BackwardEuler => {
                    // Internal mass: receives heat from envelope mass (h_tr_me) and direct gains
                    // Physics: Cm * (Tm_int_new - Tm_int_old) / dt = h_tr_me * (Tm_env - Tm_int_new) + phi_m_int
                    // Rearranged: (Cm/dt + h_tr_me) * Tm_int_new = Cm/dt * Tm_int_old + h_tr_me * Tm_env + phi_m_int
                    let denom_int = cm_int / dt + h_tr_me;
                    let numer_int =
                        cm_int / dt * tm_int_old + h_tr_me * tm_env_new + phi_m_int_zone;
                    numer_int / denom_int
                }
                ThermalIntegrationMethod::ExplicitEuler => {
                    // Use explicit Euler for low thermal mass
                    let q_int_net = h_tr_me * (tm_env_new - tm_int_old) + phi_m_int_zone;
                    tm_int_old + (q_int_net / cm_int) * dt
                }
                ThermalIntegrationMethod::CrankNicolson => {
                    // Use Crank-Nicolson for 2nd-order accuracy
                    crank_nicolson_update(
                        tm_int_old,
                        dt,
                        cm_int,
                        h_tr_me,
                        0.0,
                        tm_env_new,
                        0.0,
                        phi_m_int_zone,
                    )
                }
            };

            scratch.new_int[i] = tm_int_new;
        }

        let new_env_temps_vf: T =
            VectorField::from_smallvec(std::mem::take(&mut scratch.new_env)).into();
        let old_env_mass_temperatures = std::mem::replace(
            &mut self.0.mass.envelope_mass_temperatures,
            new_env_temps_vf,
        );

        let new_int_temps_vf: T =
            VectorField::from_smallvec(std::mem::take(&mut scratch.new_int)).into();
        let old_int_mass_temperatures = std::mem::replace(
            &mut self.0.mass.internal_mass_temperatures,
            new_int_temps_vf,
        );

        // Issue #272, #274, #275: Calculate thermal mass energy change for 6R2C
        // For 6R2C, we track energy changes in both envelope and internal masses
        // Envelope mass energy change (Cm × (Tm_new - Tm_old))
        let env_mass_temp_change = self
            .0
            .mass
            .envelope_mass_temperatures
            .zip_with(&old_env_mass_temperatures, |a, b| a - b);
        let env_mass_energy_change = self
            .0
            .mass
            .envelope_thermal_capacitance
            .zip_with(&env_mass_temp_change, |a, b| a * b);

        // Internal mass energy change (Cm × (Tm_new - Tm_old))
        let int_mass_temp_change = self
            .0
            .mass
            .internal_mass_temperatures
            .zip_with(&old_int_mass_temperatures, |a, b| a - b);
        let int_mass_energy_change = self
            .0
            .mass
            .internal_thermal_capacitance
            .zip_with(&int_mass_temp_change, |a, b| a * b);

        // Total mass energy change for this timestep
        let mass_energy_change_for_step_6r2c =
            env_mass_energy_change.zip_with(&int_mass_energy_change, |a, b| a + b);

        // Track cumulative mass energy change
        let mass_energy_change_for_step_total =
            mass_energy_change_for_step_6r2c.reduce(0.0, |acc, val| acc + val);
        self.0.mass.mass_energy_change_cumulative += mass_energy_change_for_step_total;

        // Plan 03-04: Update single mass temperature for backward compatibility (average of two masses)
        let total_cap = self
            .0
            .mass
            .envelope_thermal_capacitance
            .zip_with(&self.0.mass.internal_thermal_capacitance, |a, b| a + b);

        self.0.mass.mass_temperatures = self
            .0
            .mass
            .envelope_mass_temperatures
            .zip_with(&self.0.mass.envelope_thermal_capacitance, |a, b| a * b)
            .zip_with(
                &self
                    .0
                    .mass
                    .internal_mass_temperatures
                    .zip_with(&self.0.mass.internal_thermal_capacitance, |a, b| a * b),
                |a, b| a + b,
            )
            .zip_with(&total_cap, |a, b| a / b);

        // DEBUG: Print t_i_act before storing
        self.0.setpoints.temperatures = t_i_act;

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

        // Return HVAC energy (Plan 03-04: Use hvac_energy_for_step directly)
        // Thermal mass energy accounting removed - Ti_free calculation already includes thermal mass effects
        // Issue #2756: restore the pooled scratch so the next timestep reuses
        // the same SmallVec capacity (zero steady-state allocation).
        self.0.hvac.scratch_pool.return_6r2c(scratch);

        hvac_energy_for_step / 3.6e6 // Return kWh
    }
}

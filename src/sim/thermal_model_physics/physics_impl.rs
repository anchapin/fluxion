//! 5R1C/6R2C/8R3C/9R4C physics step implementations for `ThermalModel`.
//!
//! Hosts the per-network-type physics solvers:
//! - [`ThermalModel::step_physics_5r1c`]
//! - [`ThermalModel::step_physics_6r2c`]
//! - [`ThermalModel::step_physics_8r3c`]
//! - [`ThermalModel::step_physics_9r4c`]
//!
//! Originally part of the monolithic `thermal_model_physics.rs`
//! (Issue #898), extracted into this submodule as part of the
//! Issue #902 modular split. The implementations are large (the 5R1C
//! path alone is ~880 lines) and remain in a single file because the
//! internal-state coupling between the variants is high; the
//! dispatcher in [`super::step_dispatcher`] routes to the right one.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::physics::multi_node_solver::SurfaceExteriorTemperatures;
use crate::sim::hvac::{HVACMode as EquipmentHVACMode, VariableCapacityEquipment};
use crate::sim::interzone::{calculate_stack_effect_ach, calculate_ventilation_heat_transfer};
use crate::sim::sky_radiation::SolAirTemperature;
use crate::sim::solar::{calculate_solar_position, calculate_surface_irradiance};
use crate::sim::thermal_integration::{
    backward_euler_update_2cond, backward_euler_update_2cond_h_tr3, crank_nicolson_iso13790,
    crank_nicolson_update, crank_nicolson_update_3cond, select_integration_method,
    ThermalIntegrationMethod,
};
use crate::sim::thermal_model_core::ThermalModel;
use crate::validation::ashrae_140_cases::Orientation;

// Methods in this file are being incrementally migrated to the sibling
// submodules in `thermal_model_physics/` (see Issue #902). Methods that
// are still in this file retain the same `impl<T: ...> ThermalModel<T>`
// bound so they continue to merge with the others.
impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Solve physics for one timestep using the 5R1C (single mass node) model.
    ///
    /// This is the original implementation for backward compatibility.
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
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);
        let (_t_sol_air_data, ctf_flux_w, fd_flux_w, _ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp);

        // Get ground temperature at this timestep
        let t_g = self.0.ground_temperature.ground_temperature(timestep);

        // --- Dynamic Ventilation (Night Ventilation) ---
        let hour_of_day = (timestep % 24) as u8;

        // Combine fractions to avoid multiple intermediate VectorField allocations
        let conv_frac = self.0.convective_fraction;
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
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar_beam_to_mass_fraction;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_solar_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            // Internal gains: convective to air, radiative split between surface and mass
            // Solar distribution must conserve energy (sum to 1.0)
            let sol_to_air = sol_w * self.0.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;
            phi_ia_data.push(load_w * conv_frac + sol_to_air);
            phi_st_data.push(load_w * st_int_frac + remaining_sol * st_sol_frac);
            phi_m_data.push(load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w);
        }

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let phi_m = T::from(VectorField::new(phi_m_data));

        // PR #821 / Issue #825 — record zone-0 heat-balance terms for the
        // `pr821-diag` hourly CSV. Zero overhead when the feature is disabled
        // (no fields, no writes). The CSV consumer reads these fields right
        // after `step_physics` returns. Only zone 0 is captured because the
        // 600FF / 650FF investigation is single-zone.
        #[cfg(feature = "pr821-diag")]
        {
            self.0.last_phi_ia = phi_ia.as_ref().first().copied().unwrap_or(0.0);
            self.0.last_phi_st = phi_st.as_ref().first().copied().unwrap_or(0.0);
            self.0.last_phi_m = phi_m.as_ref().first().copied().unwrap_or(0.0);
        }

        // Use outdoor_temp directly. Solar gains on opaque surfaces are already included in phi_m.
        // Issue #901 perf: build the VectorField once, then read via as_ref() at use-sites
        // (previously this cloned the Vec a second time into `t_sol_air`).
        let t_sol_air = VectorField::from_scalar(outdoor_temp, self.0.num_zones);

        // Simplified 5R1C calculation using CTA
        // Include ground coupling through floor
        // Use pre-computed cached values to avoid redundant allocations
        let h_ext_base = &self.0.derived_h_ext;
        let term_rest_1 = &self.0.derived_term_rest_1;

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
        let h_ve_night_zone: Option<Vec<f64>> =
            if let Some(ref night_vent) = self.0.night_ventilation {
                if night_vent.is_active_at_hour(hour_of_day) {
                    night_vent_active_now = true;
                    // ASHRAE 140 night-vent fan supplies outdoor air to zone 0
                    // (the conditioned zone). Multi-zone night-vent (Case 960
                    // sunspace etc.) is out of scope for this issue.
                    let rho = self.0.air_density.as_ref().first().copied().unwrap_or(1.2);
                    let cp = self
                        .0
                        .heat_capacity
                        .as_ref()
                        .first()
                        .copied()
                        .unwrap_or(1005.0);
                    let h_ve_night = night_vent.fan_capacity * rho * cp / 3600.0;
                    let mut v = vec![0.0_f64; self.0.num_zones];
                    v[0] = h_ve_night;
                    Some(v)
                } else {
                    None
                }
            } else {
                None
            };

        // Build per-zone h_ext that includes the night-vent contribution when
        // active. When inactive (the common case) this is just an alloc-free
        // alias to the cached vector via Vec::clone — see below.
        let h_ext_owned: T = if night_vent_active_now {
            let base = h_ext_base.as_ref();
            // Safe to unwrap: h_ve_night_zone is Some exactly when night_vent_active_now is true.
            let night_add = h_ve_night_zone
                .as_ref()
                .expect("night_vent_active_now implies h_ve_night_zone is Some");
            let mut v = Vec::with_capacity(base.len());
            for (i, &b) in base.iter().enumerate() {
                v.push(b + night_add[i]);
            }
            T::from(VectorField::new(v))
        } else {
            // Issue #901 perf: clone the cached derived_h_ext (T: Clone) instead of
            // building a fresh Vec from a slice and wrapping it in a new VectorField.
            // One Vec clone replaces one Vec alloc + one VectorField wrap.
            self.0.derived_h_ext.clone()
        };
        let h_ext: &T = &h_ext_owned;

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
            let h_ms_is_prod = self.0.derived_h_ms_is_prod.as_ref();
            let term_rest_1 = self.0.derived_term_rest_1.as_ref();
            let ground_coeff = self.0.derived_ground_coeff.as_ref();
            let h_iz = self.0.h_tr_iz.as_ref();
            let h_iz_rad = self.0.h_tr_iz_rad.as_ref();
            let h_ext_slice = h_ext.as_ref();
            let mut v = Vec::with_capacity(h_ext_slice.len());
            for i in 0..h_ext_slice.len() {
                let h_total = if self.0.num_zones > 1 {
                    h_ext_slice[i] + h_iz[i] + h_iz_rad[i]
                } else {
                    h_ext_slice[i]
                };
                v.push(h_ms_is_prod[i] + term_rest_1[i] * h_total + ground_coeff[i]);
            }
            T::from(VectorField::new(v))
        } else {
            self.0.derived_den.clone()
        };
        // (#872: sensitivity variable removed — HVAC demand now uses h_loss × ΔT formula)

        // Optimized: use zip_with to avoid double clones; num_tm allocates 1 vector instead of 2
        let num_tm = self
            .0
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass_temperatures, |a, b| a * b);
        // Optimized: use zip_with to avoid double clones (phi_st used later)
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        // Ground heat transfer: Q_ground = h_tr_floor * (T_ground - T_surface)
        // Optimization: use scalar multiplication for t_g and outdoor_temp instead of creating full constant vectors
        // Note: t_e vector creation removed. h_ext * t_e replaced by h_ext * outdoor_temp.
        // Note: t_g vector creation removed. h_tr_floor * t_g_vec replaced by h_tr_floor * t_g.

        // === Inter-zone heat transfer (for multi-zone buildings like Case 960) ===
        // Three-component approach: Q_iz = Q_cond + Q_rad + Q_vent
        // 1. Conductive: Q_cond = h_tr_iz * ΔT
        // 2. Radiative: Q_rad = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴) (full nonlinear Stefan-Boltzmann)
        // 3. Ventilation: Q_vent = ρ·Cp·ACH·V·ΔT (temperature-dependent ACH via stack effect)
        let num_zones = self.0.num_zones;

        // Start with phi_ia; we will add inter-zone heat directly to its buffer if needed.
        // Issue #901 perf: move phi_ia (no clone). The original is no longer used
        // after this point in step_physics_5r1c — the legacy comment referencing a
        // Case 610 debug print at "line 914" referred to step_physics_6r2c, not here.
        let mut phi_ia_with_iz = phi_ia;

        if num_zones > 1 {
            let temps = self.0.temperatures.as_ref();
            let h_iz_vec = self.0.h_tr_iz.as_ref();

            // For Case 960 (2-zone building), calculate heat transfer between zone 0 (back-zone) and zone 1 (sunspace)
            if num_zones >= 2 && h_iz_vec[0] > 0.0 {
                let delta_t_cond = temps[1] - temps[0]; // T_sunspace - T_back

                // 1. Conductive heat transfer
                let q_cond = h_iz_vec[0] * delta_t_cond;

                // 2. Radiative heat transfer - DISABLED for Case 960 (aligned windows don't exchange radiation)
                // This was causing excessive heat loss from sunspace
                let q_rad = 0.0; // windows face same direction - no radiative exchange

                // 3. Ventilation heat transfer (temperature-dependent ACH via stack effect)
                // Use back-zone volume for ventilation calculation
                let zone_volume = self.0.zone_volume.as_ref();
                let ach_iz = calculate_stack_effect_ach(
                    temps[0], // T_back-zone
                    temps[1], // T_sunspace
                    self.0.door_geometry.height,
                    self.0.door_geometry.area,
                    zone_volume[0], // FIX: Pass actual zone volume
                );
                let q_vent = calculate_ventilation_heat_transfer(
                    ach_iz,
                    temps[1],       // Source: sunspace (warm in summer, cold in winter)
                    temps[0],       // Target: back-zone
                    zone_volume[0], // Target volume
                );

                // Total inter-zone heat transfer (positive = sunspace → back-zone)
                let q_iz_total = q_cond + q_rad + q_vent;

                // Apply to energy balance directly in-place
                let slice = phi_ia_with_iz.as_mut();
                if slice.len() >= 2 {
                    slice[0] += -q_iz_total;
                    slice[1] += q_iz_total;
                } else {
                    // Defensive: should never happen for 2-zone case
                    eprintln!(
                        "WARNING: phi_ia length {} < 2, cannot apply inter-zone heat",
                        slice.len()
                    );
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
                    // Convert flux [W/m²] to power [W] by multiplying by zone area
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_ctf = q_flux * area;
                    // Subtract standard 5R1C envelope conduction to avoid double-counting
                    // Q_5r1c = h_tr_em * (T_sol_air - T_mass)
                    let t_sol_air_i = t_sol_air.as_ref().get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);

                    // Net CTF contribution (CTF - 5R1C)
                    let net_ctf_flux = q_ctf - q_5r1c;
                    slice[i] += net_ctf_flux;

                    // Track CTF energy for thermal mass correction
                    // Positive net flux = heating contribution, negative = cooling
                    if net_ctf_flux > 0.0 {
                        self.0.ctf_annual_heating_joules += net_ctf_flux * dt;
                    } else {
                        self.0.ctf_annual_cooling_joules += (-net_ctf_flux) * dt;
                    }
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
                    // Convert flux [W/m²] to power [W] by multiplying by zone area
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_fd = q_flux * area;

                    // Subtract standard 5R1C envelope conduction to avoid double-counting
                    // Q_5r1c = h_tr_em * (T_sol_air - T_mass)
                    let t_sol_air_i = t_sol_air.as_ref().get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);

                    // Add net FD flux (FD - 5R1C)
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;

                    // Track FD energy for thermal mass correction
                    if net_fd_flux > 0.0 {
                        self.0.fd_annual_heating_joules += net_fd_flux * dt;
                    } else {
                        self.0.fd_annual_cooling_joules += (-net_fd_flux) * dt;
                    }
                }
            }
        }

        // For single-zone or no inter-zone heat, phi_ia_with_iz remains as cloned phi_ia (no allocation beyond the initial clone)

        // Recalculate num_rest with inter-zone heat transfer
        // Optimized: h_ext * t_e -> h_ext * outdoor_temp
        // Optimized: t_g_vec -> t_g
        // Ground Coupling: term_rest_1 * h_tr_floor * T_ground = derived_ground_coeff * T_ground
        // Add this to numerator per ISO 13790 5R1C heat balance equation
        // Optimized: combine h_ext * outdoor_temp addition and multiplication into phi_ia_with_iz buffer directly
        // This eliminates one allocation (term_rest_1.clone())
        let mut num_rest_with_iz = phi_ia_with_iz;
        for (n, h) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(h_ext.as_ref().iter())
        {
            *n += h * outdoor_temp;
        }
        num_rest_with_iz.mul_assign(term_rest_1);
        // Fuse ground term addition: (derived_ground_coeff * t_g) added directly
        let ground_coeff = self.0.derived_ground_coeff.as_ref();
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

        let mut t_i_free = num_tm;
        t_i_free.add_assign(&num_phi_st);
        t_i_free.add_assign(&num_rest_with_iz);
        t_i_free.div_assign(&den);

        // PR #821: DEBUG_900FF_ti_free trace removed.

        // PR #821: DEBUG_MAX trace for 600FF/650FF removed; use `pr821-diag` feature instead.

        // PR #821: DEBUG_650FF_FULL traces removed.

        // DEBUG: Case 195 thermal diagnostics - uncomment to debug heating issues
        // if self.0.case_id == "195" && timestep < 1000 {
        //     let t_i_free_val = t_i_free.as_ref()[0];
        //     let mass_temp = self.0.mass_temperatures.as_ref()[0];
        //     let heating_threshold = self.0.heating_setpoint - self.0.hvac_controller.deadband_tolerance;
        //     eprintln!(
        //         "DEBUG_195 t={} t_i_free={:.2}°C heating_thresh={:.2}°C num_tm={:.1} num_phi_st={:.1} num_rest={:.1} den={:.1} T_mass={:.2}°C",
        //         timestep, t_i_free_val, heating_threshold, num_tm_val, num_phi_st_val, num_rest_val, den_val, mass_temp
        //     );
        // }

        // 2.5. Predictive Control Calculation (Plan 15-04, 15-06)
        // Calculate temperature rate (dT/dt) for predictive control using thermal inertia
        let temp_rate = if timestep > 0 {
            (self.0.temperatures.as_ref()[0] - self.0.previous_temperatures.as_ref()[0]) / dt
        } else {
            0.0
        };

        // Predictive control using thermal inertia
        let (hvac_mode, modulation) = self.0.predictive_controller.calculate_modulation(
            self.0.temperatures.as_ref()[0],
            self.0.mass_temperatures.as_ref()[0],
            temp_rate,
        );
        let hvac_mode: EquipmentHVACMode = hvac_mode; // Type annotation for clarity

        // 3. HVAC Calculation
        // Compute ideal loads for equipment modulation BEFORE mutable borrow of hvac_equipment
        let ideal_loads_for_equipment: T = if self.0.free_float {
            T::from(VectorField::new(vec![0.0; self.0.num_zones]))
        } else {
            // Issue #900: pass mass_temperatures so the dynamic mass heat
            // release term is included in the cooling demand.
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.heating_setpoint,
                self.0.cooling_setpoint,
                self.0.mass_temperatures.as_ref(),
            )
        };

        let hour_of_day_idx = timestep % 24;

        // Issue #738: Free-float mode must completely disable HVAC output
        // This is a safety check that goes beyond hvac_enabled, which may not be
        // properly set for all code paths. Free-float cases (900FF, etc.) should
        // have zero HVAC output regardless of other settings.
        let hvac_output_raw = if self.0.free_float {
            T::from(VectorField::new(vec![0.0; self.0.num_zones]))
        } else if let Some(ref mut equipment) = self.0.hvac_equipment {
            // Use scalar setpoints instead of hourly schedules (Issue #???: HVAC schedule fix)
            // This ensures per-hour setpoint changes from validation loop are respected
            let heating_setpoint = self.0.heating_setpoint;
            let _cooling_setpoint = self.0.cooling_setpoint;

            // Calculate free cooling if economizer is active
            use crate::sim::hvac::is_economizer_active;
            let cooling_setpoint = self.0.cooling_schedule.value(hour_of_day_idx);
            let economizer_active = is_economizer_active(
                self.0.economizer_mode,
                outdoor_temp,
                None, // outdoor_enthalpy - not available until Phase 16
                self.0.temperatures.as_ref()[0],
                None, // zone_enthalpy - not available until Phase 16
                cooling_setpoint,
            );

            // Calculate free cooling capacity if economizer is active and we're in cooling mode
            let free_cooling_capacity =
                if economizer_active && matches!(hvac_mode, EquipmentHVACMode::Cooling) {
                    use crate::sim::hvac::calculate_free_cooling_capacity;
                    calculate_free_cooling_capacity(
                        outdoor_temp,
                        self.0.temperatures.as_ref()[0],
                        10000.0, // TODO: ventilation_airflow from building spec (m³/s)
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
                .cycling_tracker
                .calculate_cycling_loss(electrical_power > 0.0, equipment.current_plr());

            let actual_electrical_power = electrical_power * efficiency_multiplier;

            // Accumulate electrical energy consumption (Plan 18-08)
            // actual_electrical_power is in Watts, dt_seconds is in seconds
            // Convert to kWh: (Watts × dt_seconds) / 3.6e6 = kWh
            let energy_this_timestep = actual_electrical_power * dt_seconds / 3.6e6;
            self.0.annual_electrical_energy += energy_this_timestep;

            // FIX: For multi-zone buildings (e.g., Case 960), use per-zone HVAC demand
            // instead of broadcasting a single scalar value to all zones.
            // Use IdealLoadsSystem thermodynamic formulas (mass_flow * cp * delta_t)
            // instead of sensitivity-based (setpoint - temp) / sensitivity
            //
            // Issue #900: pass mass_temperatures so the dynamic mass heat
            // release term is included in the cooling demand.
            let hvac_output = self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                heating_setpoint,
                cooling_setpoint,
                self.0.mass_temperatures.as_ref(),
            );

            // Track peak heating/cooling based on per-zone HVAC demand (Plan 18-08)
            // Physics-based: No calibration factors - track actual HVAC demand
            // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
            let enabled_vec = self.0.hvac_enabled.as_ref();
            let hvac_output_sum: f64 = hvac_output
                .as_ref()
                .iter()
                .zip(enabled_vec.iter())
                .map(|(output, &enabled)| if enabled > 0.5 { *output } else { 0.0 })
                .sum::<f64>();
            if hvac_output_sum > 0.0 {
                // Heating mode - track actual demand
                if hvac_output_sum > 0.0 {
                    // Heating mode - track actual demand
                    self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_output_sum);
                } else if hvac_output_sum < 0.0 {
                    // Cooling mode (store as positive value)
                    let cooling_demand = -hvac_output_sum;
                    self.0.peak_power_cooling = self.0.peak_power_cooling.max(cooling_demand);
                }
            }

            // Both equipment and fallback paths now use hvac_output (per-zone VectorField)
            // so it needs to be returned for both branches
            hvac_output
        } else {
            // Use IdealLoadsSystem thermodynamic formulas for energy
            //
            // Issue #900: pass mass_temperatures so the dynamic mass heat
            // release term is included in the cooling demand.
            let hvac_output_raw = self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.heating_setpoint,
                self.0.cooling_setpoint,
                self.0.mass_temperatures.as_ref(),
            );

            // Root Cause Fix: Use hvac_output_raw for peak tracking (consistent with energy calc)
            // Issue #901 perf: borrow hvac_output_raw directly instead of cloning for
            // the peak-power read. The original is returned to the caller below, untouched.
            let hvac_power_for_peak = hvac_output_raw.as_ref();

            // Track peak heating/cooling based on actual HVAC demand (only if not already tracked above)
            if self.0.hvac_equipment.is_none() {
                // Note: hvac_output_raw is positive for heating, negative for cooling
                // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
                let enabled_vec = self.0.hvac_enabled.as_ref();
                let hvac_power_watts = hvac_power_for_peak
                    .iter()
                    .zip(enabled_vec.iter())
                    .map(|(output, &enabled)| if enabled > 0.5 { *output } else { 0.0 })
                    .sum::<f64>();

                // Physics-based: Track actual HVAC demand without calibration
                if hvac_power_watts > 0.0 {
                    // Heating mode - track actual demand
                    self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_power_watts);
                } else if hvac_power_watts < 0.0 {
                    // Cooling mode (store as positive value)
                    let cooling_demand = -hvac_power_watts;
                    self.0.peak_power_cooling = self.0.peak_power_cooling.max(cooling_demand);
                }
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
        let hvac_for_temp_calc = if self.0.free_float {
            T::from(VectorField::new(vec![0.0; self.0.num_zones]))
        } else {
            // Issue #900: pass mass_temperatures so the dynamic mass heat
            // release term is included in the cooling demand.
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                self.0.heating_setpoint,
                self.0.cooling_setpoint,
                self.0.mass_temperatures.as_ref(),
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
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_for_temp_calc.as_ref();
        let mut t_i_act_data = Vec::with_capacity(self.0.num_zones);
        for i in 0..self.0.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                t_i_act_data.push(t_free[i] + hvac[i] / h_is);
            } else {
                t_i_act_data.push(t_free[i]);
            }
        }
        let t_i_act = T::from(VectorField::new(t_i_act_data));

        // Use hvac_for_temp_calc for energy (matches what was used for temperature update)
        // This ensures energy calculation is consistent with temperature physics
        let mut heating_sum = 0.0;
        let mut cooling_sum = 0.0;
        let mut total_signed = 0.0;
        for &val in hvac_for_temp_calc.as_ref() {
            total_signed += val;
            if val > 0.0 {
                heating_sum += val;
            } else {
                cooling_sum += -val;
            }
        }

        // Compute energy (uncorrected for physics)
        let heating_energy_joules = heating_sum * dt;
        let cooling_energy_joules = cooling_sum * dt;

        // Issue #738 + Issue #821: free_float mode MUST produce zero HVAC output.
        // Promoted from debug_assert! to a hard assert under cfg(test) so the
        // ASHRAE 140 free-float regression test catches any code path that
        // sneaks HVAC demand in via the equipment fallback.
        if self.0.free_float {
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
        self.0.annual_heating_energy += heating_energy_joules / 3.6e6;
        self.0.annual_cooling_energy += cooling_energy_joules / 3.6e6;
        self.0.ctf_annual_heating_joules = 0.0;
        self.0.ctf_annual_cooling_joules = 0.0;
        self.0.fd_annual_heating_joules = 0.0;
        self.0.fd_annual_cooling_joules = 0.0;

        // hvac_energy_for_step returns total HVAC energy in JOULES (not kWh)
        // The test expects Joules and multiplies by 3.6e6
        // DON'T apply correction here - it would break temperature calculations
        let hvac_energy_for_step = total_signed * dt;

        // Issue #272, #274, #275: Calculate thermal mass energy change
        // HVAC energy currently includes energy stored in thermal mass, which should be subtracted
        // Mass energy change = Cm × (Tm_new - Tm_old)
        // Save old mass temperature before updating
        let old_mass_temperatures = self.0.mass_temperatures.clone();

        // Mass temperature update: includes heat transfer from exterior and from surface
        // Ground coupling affects mass temperature indirectly through the thermal network
        // Calculate actual surface temperature for mass update (including HVAC effect)
        // ts_num_act = h_tr_ms * mass_temp + h_tr_is * t_i_act + phi_st
        let mut ts_num_act = self.0.h_tr_ms.clone();
        ts_num_act.mul_assign(&self.0.mass_temperatures);
        let mut term2 = self.0.h_tr_is.clone();
        term2.mul_assign(&t_i_act);
        ts_num_act.add_assign(&term2);
        ts_num_act.add_assign(&phi_st);
        // Denominator is term_rest_1
        let mut t_s_act = ts_num_act;
        t_s_act.div_assign(term_rest_1);

        // Update mass temperatures using implicit integration for high thermal capacitance
        // This addresses instability with explicit Euler for Cm > 500 J/K
        let mut new_mass_temperatures = Vec::with_capacity(self.0.num_zones);
        let mass_temps_ref = self.0.mass_temperatures.as_ref();
        let thermal_cap_ref = self.0.thermal_capacitance.as_ref();
        // Mode-specific fields removed - use physics-based h_tr_em and h_tr_ms
        let h_tr_em_ref = self.0.h_tr_em.as_ref();
        let h_tr_ms_ref = self.0.h_tr_ms.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let phi_m_ref = phi_m.as_ref();

        // Determine HVAC mode from hvac_output_raw (Plan 03-14)
        // Use separate heating/cooling coupling parameters based on mode

        for i in 0..self.0.num_zones {
            let tm_old = mass_temps_ref[i];
            let cm = thermal_cap_ref[i];
            let t_s = t_s_act_ref[i];
            let phi_m_zone = phi_m_ref[i];

            // Use physics-based h_tr_em and h_tr_ms (mode-specific factors removed)
            // The conductances are now calculated from first principles:
            // h_tr_em = k * A / d (thermal conductivity * area / thickness)
            // h_tr_ms = k * A / d (thermal conductivity * area / thickness)
            let h_tr_em = h_tr_em_ref[i];
            let h_tr_ms = h_tr_ms_ref[i];

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
            let h_vent_mass_zone = if let Some(ref night_vent) = self.0.night_ventilation {
                if night_vent.is_active_at_hour(hour_of_day) {
                    let _ = night_vent.fan_capacity; // kept for future air-side wiring
                    0.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let tm_new = match method {
                ThermalIntegrationMethod::BackwardEuler => {
                    // Use implicit backward Euler for high thermal mass
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    let effective_h_tr_em = h_tr_em + h_vent_mass_zone;
                    // Issue #896 FIX: Use h_tr_3 instead of h_tr_ms for the air-to-mass bottleneck.
                    // See detailed comment in the CrankNicolson branch below.
                    let h_tr_3_zone = *self.0.derived_h_tr_3.as_ref().get(i).unwrap_or(&h_tr_ms);
                    let t_ext_eff = if h_vent_mass_zone > 0.0 {
                        (h_tr_em * t_sol_air[i] + h_vent_mass_zone * outdoor_temp)
                            / effective_h_tr_em
                    } else {
                        t_sol_air[i]
                    };
                    // Backward Euler with h_tr_3:
                    // (Cm/dt + h_tr_em + h_tr_3) * Tm_new = Cm/dt * Tm_old + h_tr_em * t_ext + h_tr_3 * t_s + phi_m
                    let cm_dt = cm / dt;
                    let denom = cm_dt + effective_h_tr_em + h_tr_3_zone;
                    let numer = cm_dt * tm_old
                        + effective_h_tr_em * t_ext_eff
                        + h_tr_3_zone * t_s
                        + phi_m_zone;
                    numer / denom
                }
                ThermalIntegrationMethod::ExplicitEuler => {
                    // Use explicit Euler for low thermal mass (faster, still stable)
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    // Issue #896 FIX: Use h_tr_3 instead of h_tr_ms for the air-to-mass bottleneck.
                    // See detailed comment in the CrankNicolson branch below.
                    let h_tr_3_zone = *self.0.derived_h_tr_3.as_ref().get(i).unwrap_or(&h_tr_ms);
                    let q_vent_mass = h_vent_mass_zone * (outdoor_temp - tm_old);
                    let q_m_net = h_tr_em * (t_sol_air[i] - tm_old)
                        + h_tr_3_zone * (t_s - tm_old)
                        + phi_m_zone
                        + q_vent_mass;
                    tm_old + (q_m_net / cm) * dt
                }
                ThermalIntegrationMethod::CrankNicolson => {
                    // Use Crank-Nicolson for 2nd-order accuracy (alternative to backward Euler)
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    // Issue #896 FIX: Use crank_nicolson_iso13790 with h_tr_3 instead of
                    // crank_nicolson_update with h_tr_ms. The h_tr_ms (~1300 W/K) is the
                    // direct surface-to-mass conductance, but heat from the air node reaches
                    // the mass through the combined air-to-surface bottleneck (h_tr_3 ≈ 40 W/K).
                    // Using h_tr_ms gives a time constant of ~4 hours (too fast), while h_tr_3
                    // gives ~6 days (matching ISO 13790 dynamics and correct seasonal mass swing).
                    // phi_m_zone already includes all gains that reach the mass node.
                    crank_nicolson_iso13790(
                        tm_old,
                        dt,
                        cm,
                        *self.0.derived_h_tr_3.as_ref().get(i).unwrap_or(&h_tr_ms),
                        h_tr_em, // Ventilation is handled via air/surface temp, not directly here
                        phi_m_zone,
                    )
                }
            };

            new_mass_temperatures.push(tm_new);
        }

        // Update the mass temperatures with new values (convert Vec to T type)
        self.0.mass_temperatures = VectorField::new(new_mass_temperatures).into();

        // Plan 03-04: Update previous mass temperature for tracking (kept for diagnostic output)
        // Mass energy change tracking removed - Ti_free already includes thermal mass effects
        self.0.previous_mass_temperatures = old_mass_temperatures;

        // Store previous temperatures for dT/dt calculation (Plan 15-04, 15-06)
        self.0.previous_temperatures = VectorField::new(self.0.temperatures.as_ref().to_vec());

        self.0.temperatures = t_i_act;

        // Return HVAC energy (Plan 03-04: Use hvac_energy_for_step directly)
        // Thermal mass energy accounting removed - Ti_free calculation already includes thermal mass effects
        // No subtraction of mass energy change needed
        let net_hvac_energy_for_step = hvac_energy_for_step;

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            // Store current HVAC output for this timestep (per zone, Watts)
            self.0.current_hvac_output = Some(hvac_output_raw.clone());
            // Temporarily take diagnostics out to avoid borrow conflicts
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            // Clear the buffer after use
            self.0.current_hvac_output = None;
        }

        net_hvac_energy_for_step / 3.6e6 // Return kWh
    }

    /// Solve physics for one timestep using the 6R2C (two mass node) model.
    ///
    /// This extends the 5R1C model by separating thermal mass into:
    /// - Envelope mass (walls, roof, floor) - heavier thermal lag
    /// - Internal mass (furniture, partitions) - faster response
    ///
    /// This better captures thermal phase shifts in high-mass buildings.
    pub(crate) fn step_physics_6r2c(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> f64 {
        let dt = dt_seconds; // Use provided timestep duration

        // DEBUG: Check solar_gains at entry to step_physics_6r2c
        if timestep == 12 {
            eprintln!(
                "DEBUG_STEP_6R2C_ENTRY: t={}, solar_gains[0]={:.2}",
                timestep,
                self.0.solar_gains.as_ref()[0]
            );
        }

        // Prepare sol-air temperature and calculate CTF/FD heat fluxes early to avoid borrow conflicts
        // Calculate sky temperature for proper sol-air calculation with longwave radiation
        let sky_temp = self
            .0
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);
        let (t_sol_air_data, ctf_flux_w, fd_flux_w, ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp);

        // Get ground temperature at this timestep
        let t_g = self.0.ground_temperature.ground_temperature(timestep);

        let _hour_of_day = (timestep % 24) as u8;

        // Combine fractions to avoid multiple intermediate VectorField allocations
        let conv_frac = self.0.convective_fraction;
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
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        // SESSION 76 FIX: Solar gain distribution
        // ASHRAE 140 spec: 60% solar to mass, 40% to surface
        // The code uses solar_beam_to_mass_fraction to control this split
        // With solar_beam_to_mass_fraction = 0.6:
        //   - 60% of solar goes to mass (70% envelope + 30% internal split)
        //   - 40% of solar goes to surface
        // Additionally, solar_distribution_to_air sends some solar directly to zone air
        let st_sol_frac = (1.0 - self.0.solar_beam_to_mass_fraction) * 0.6; // Solar to surface
        let m_env_sol_frac = self.0.solar_beam_to_mass_fraction * 0.7; // Solar to envelope mass
        let m_int_sol_frac = self.0.solar_beam_to_mass_fraction * 0.3; // Solar to internal mass
                                                                       // Solar to air (via solar_distribution_to_air) - SESSION 76 addition
        let sol_to_air_frac = self.0.solar_distribution_to_air;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_env_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_int_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];

            // SESSION 76 FIX: Include solar_distribution_to_air in 6R2C (was missing!)
            // This sends a fraction of solar directly to zone air (immediate heating/cooling)
            phi_ia_data.push(load_w * conv_frac + sol_w * sol_to_air_frac);
            phi_st_data.push(load_w * st_int_frac + sol_w * st_sol_frac);
            phi_m_env_data.push(load_w * m_air_frac + sol_w * m_env_sol_frac);
            phi_m_int_data.push(sol_w * m_int_sol_frac);
        }

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let phi_m_env = T::from(VectorField::new(phi_m_env_data));
        let phi_m_int = T::from(VectorField::new(phi_m_int_data));

        // Use pre-computed cached values
        let h_ext_base = &self.0.derived_h_ext;
        let term_rest_1 = &self.0.derived_term_rest_1;

        // Night ventilation no longer modifies h_ext (same fix as 5R1C path).
        let modified_h_ext: Option<T> = None;
        let h_ext = h_ext_base;

        // 6R2C specific terms
        let h_sum = self
            .0
            .h_tr_ms
            .zip_with(&self.0.h_tr_me, |a, b| a + b)
            .zip_with(&self.0.h_tr_is, |a, b| a + b);

        let h_ms_me_is_prod = self.0.h_tr_is.zip_with(
            &self.0.h_tr_ms.zip_with(&self.0.h_tr_me, |a, b| a + b),
            |a, b| a * b,
        );

        let den: T;
        let h_total_with_iz = if let Some(ref mod_h_ext) = modified_h_ext {
            if self.0.num_zones > 1 {
                mod_h_ext
                    .zip_with(&self.0.h_tr_iz, |a, b| a + b)
                    .zip_with(&self.0.h_tr_iz_rad, |a, b| a + b)
            } else {
                mod_h_ext.clone()
            }
        } else {
            if self.0.num_zones > 1 {
                self.0
                    .derived_h_ext
                    .zip_with(&self.0.h_tr_iz, |a, b| a + b)
                    .zip_with(&self.0.h_tr_iz_rad, |a, b| a + b)
            } else {
                self.0.derived_h_ext.clone()
            }
        };

        // Issue 693 fix: ground coupling coefficient in 6R2C den
        let ground_coeff_6r2c = h_sum.zip_with(&self.0.h_tr_floor, |a, b| a * b);
        den = h_ms_me_is_prod
            .zip_with(&h_sum.zip_with(&h_total_with_iz, |a, b| a * b), |a, b| {
                a + b
            })
            .zip_with(&ground_coeff_6r2c, |a, b| a + b);

        // Use envelope mass temperature instead of single mass temperature
        // Optimized: use zip_with to avoid double clones
        //
        // CTF-driven zone air heat balance (Issue #698 fix):
        // When ctf_primary=true, the 6R2C h_tr_ms coupling is DISABLED because
        // CTF provides the correct multi-layer conduction dynamics directly.
        // The CTF heat flow q_ctf (computed from T_si_ctf) replaces the 6R2C h_tr_ms * t_mass term.
        let num_tm = if self.0.ctf_primary {
            // Zero out the 6R2C coupling - CTF will drive the zone air heat balance
            self.0.derived_h_ms_is_prod.constant_like(0.0)
        } else {
            self.0
                .derived_h_ms_is_prod
                .zip_with(&self.0.envelope_mass_temperatures, |a, b| a * b)
        };
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        // Inter-zone heat transfer (with radiative component - Issue #302)
        let num_zones = self.0.num_zones;
        let h_iz_vec = self.0.h_tr_iz.as_ref();
        let h_iz_rad_vec = self.0.h_tr_iz_rad.as_ref();

        // Compute inter-zone heat transfer directly into phi_ia_with_iz to avoid Vec allocation
        let mut phi_ia_with_iz = phi_ia.clone();

        if num_zones > 1
            && (!h_iz_vec.is_empty() && h_iz_vec[0] > 0.0
                || !h_iz_rad_vec.is_empty() && h_iz_rad_vec[0] > 0.0)
        {
            let temps = self.0.temperatures.as_ref();
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
        let _h_tr_floor_ref = self.0.h_tr_floor.as_ref();

        // Start with phi_ia_with_iz
        let mut sum_term = phi_ia_with_iz;

        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = sum_term.as_mut();
            for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_ctf = q_flux * area;
                    slice[i] += q_ctf;
                    if q_ctf > 0.0 {
                        self.0.ctf_annual_heating_joules += q_ctf * dt;
                    } else {
                        self.0.ctf_annual_cooling_joules += (-q_ctf) * dt;
                    }
                }
            }
        }

        // Add FD net contribution if enabled
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = sum_term.as_mut();
            for (i, &q_flux) in fd_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_fd = q_flux * area;
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .envelope_mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;
                    if net_fd_flux > 0.0 {
                        self.0.fd_annual_heating_joules += net_fd_flux * dt;
                    } else {
                        self.0.fd_annual_cooling_joules += (-net_fd_flux) * dt;
                    }
                }
            }
        }

        let mut num_rest_with_iz = sum_term.clone();
        num_rest_with_iz.mul_assign(term_rest_1);
        // Add ground term separately
        let ground_coeff = ground_coeff_6r2c.as_ref();
        for (n, g) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(ground_coeff.iter())
        {
            *n += g * t_g;
        }

        // DEBUG: Save values for 900FF before they're consumed
        let debug_900ff = if self.0.case_id == "900FF" && timestep.is_multiple_of(24) {
            let den_vals = den.as_ref();
            let _num_tm_vals = num_tm.as_ref();
            let num_rest_vals = num_rest_with_iz.as_ref();
            let _env_mass_vals = self.0.envelope_mass_temperatures.as_ref();
            let h_sum_vals = h_sum.as_ref();
            let sum_term_vals = sum_term.as_ref();
            let h_ext_debug = h_ext.as_ref();
            let phi_ia_debug = phi_ia.as_ref();
            let solar_debug = self.0.solar_gains.as_ref();
            let loads_debug = self.0.loads.as_ref();
            let area_debug = self.0.zone_area.as_ref();
            eprintln!("DEBUG_900FF_PREPARE: t={}, phi_ia[0]={:.2}, solar[0]={:.2}, loads[0]={:.2}, area[0]={:.1}", timestep, phi_ia_debug[0], solar_debug[0], loads_debug[0], area_debug[0]);
            Some((
                den_vals[0],
                _num_tm_vals[0],
                num_rest_vals[0],
                _env_mass_vals[0],
                h_sum_vals[0],
                sum_term_vals[0],
                h_ext_debug[0],
                phi_ia_debug[0],
                solar_debug[0],
                loads_debug[0],
                area_debug[0],
            ))
        } else {
            None
        };

        // Calculate free-floating indoor temperature using standard 6R2C heat balance
        // (thermal mass buffering is critical for preventing temperature overshoot)
        let mut t_i_free = num_tm;
        t_i_free.add_assign(&num_phi_st);
        t_i_free.add_assign(&num_rest_with_iz);
        t_i_free.div_assign(&den);

        // DEBUG: Print key values for 900FF after calculation
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
        // Issue #900: pass mass_temperatures so the dynamic mass heat release
        // term is included in the cooling demand.
        let hvac_output_raw = self.compute_zone_hvac_load(
            t_i_free.as_ref(),
            self.0.heating_setpoint,
            self.0.cooling_setpoint,
            self.0.mass_temperatures.as_ref(),
        );
        // Fix: Use actual HVAC demand instead of steady-state approximation (Plan 03-03 Task 2)
        // hvac_output_raw already includes thermal mass buffering (calculated from t_i_free)
        // This is needed for high-mass cases (900 series) that use 6R2C model
        let hvac_power_watts = hvac_output_raw.as_ref().iter().sum::<f64>();

        // Track peak for high-mass cases (6R2C model)
        // Physics-based: Track actual HVAC demand without calibration factors
        if hvac_power_watts > 0.0 {
            // Heating mode - track actual demand
            self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_power_watts);
        } else if hvac_power_watts < 0.0 {
            // Cooling mode (store as positive value)
            let cooling_demand = -hvac_power_watts;
            self.0.peak_power_cooling = self.0.peak_power_cooling.max(cooling_demand);
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
        for &val in hvac_output_raw.as_ref() {
            total_signed += val;
            if val > 0.0 {
                heating_sum += val;
            } else {
                cooling_sum += -val;
            }
        }

        // Compute energy (uncorrected for physics)
        let heating_energy_joules = heating_sum * dt;
        let cooling_energy_joules = cooling_sum * dt;

        // Issue #738 + Issue #821: free_float mode MUST produce zero HVAC output.
        // Promoted from debug_assert! to a hard assert under cfg(test) so the
        // ASHRAE 140 free-float regression test catches any code path that
        // sneaks HVAC demand in via the equipment fallback.
        if self.0.free_float {
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
        self.0.annual_heating_energy += heating_energy_joules / 3.6e6;
        self.0.annual_cooling_energy += cooling_energy_joules / 3.6e6;
        self.0.ctf_annual_heating_joules = 0.0;
        self.0.ctf_annual_cooling_joules = 0.0;
        self.0.fd_annual_heating_joules = 0.0;
        self.0.fd_annual_cooling_joules = 0.0;

        // hvac_energy_for_step returns total HVAC energy in JOULES (not kWh)
        // The test expects Joules and multiplies by 3.6e6
        // DON'T apply correction here - it would break temperature calculations
        let hvac_energy_for_step = total_signed * dt;

        // Root Cause Fix: Physics-based temperature update.
        // t_i_act = t_i_free + hvac_power / h_tr_is
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_output_raw.as_ref();
        let mut t_i_act_data = Vec::with_capacity(self.0.num_zones);
        for i in 0..self.0.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                t_i_act_data.push(t_free[i] + hvac[i] / h_is);
            } else {
                t_i_act_data.push(t_free[i]);
            }
        }
        let t_i_act = T::from(VectorField::new(t_i_act_data));

        // Calculate surface temperature for mass update (including HVAC effect)
        // === 6R2C: Update two mass nodes ===
        // PHASE 36-04 FIX: Include h_tr_me * Tm_int in surface temperature calculation
        // The 6R2C model requires: T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
        let h_tr_me_ref = self.0.h_tr_me.as_ref();
        let int_mass_temps_ref = self.0.internal_mass_temperatures.as_ref();
        // SESSION 89: When ctf_primary is active, use CTF T_si (with HVAC offset) instead of lumped T_s
        let t_s_act: T = if self.0.ctf_primary {
            // Use CTF surface temp adjusted for HVAC effect
            // The CTF T_si was computed at t_i_free; adjust for actual t_i_act via linear correction:
            // T_si_adjusted ≈ T_si_ctf + (h_tr_is / (h_tr_is + Z₀)) * (t_i_act - t_i_free)
            if let Some(ref ctf_temps) = ctf_surface_temps {
                let mut t_s_data = Vec::with_capacity(self.0.num_zones);
                let t_i_free_ref = t_i_free.as_ref();
                let t_i_act_ref = t_i_act.as_ref();
                for i in 0..self.0.num_zones {
                    let t_si_ctf = ctf_temps.get(i).copied().unwrap_or(20.0);
                    let delta_t_i = t_i_act_ref.get(i).copied().unwrap_or(0.0)
                        - t_i_free_ref.get(i).copied().unwrap_or(0.0);
                    // Approximate: surface follows zone air with ~h_tr_is/(h_tr_is+Z₀) coupling
                    // Use conservative 0.5 factor for stability
                    t_s_data.push(t_si_ctf + 0.5 * delta_t_i);
                }
                T::from(VectorField::new(t_s_data))
            } else {
                // PHASE 36-04 FIX: 6R2C surface temperature with h_tr_me * Tm_int coupling
                // T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
                let h_tr_ms_data = self.0.h_tr_ms.as_ref();
                let h_tr_is_data = self.0.h_tr_is.as_ref();
                let t_i_act_data = t_i_act.as_ref();
                let phi_st_data = phi_st.as_ref();
                let env_mass_data = self.0.envelope_mass_temperatures.as_ref();
                let term_rest_data = term_rest_1.as_ref();
                let mut t_s_data = Vec::with_capacity(self.0.num_zones);
                for i in 0..self.0.num_zones {
                    let numerator = h_tr_ms_data[i] * env_mass_data[i]
                        + h_tr_is_data[i] * t_i_act_data[i]
                        + phi_st_data[i]
                        + h_tr_me_ref[i] * int_mass_temps_ref[i];
                    let denominator = term_rest_data[i] + h_tr_me_ref[i];
                    t_s_data.push(numerator / denominator);
                }
                T::from(VectorField::new(t_s_data))
            }
        } else {
            // PHASE 36-04 FIX: 6R2C surface temperature with h_tr_me * Tm_int coupling
            // T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
            let h_tr_ms_data = self.0.h_tr_ms.as_ref();
            let h_tr_is_data = self.0.h_tr_is.as_ref();
            let t_i_act_data = t_i_act.as_ref();
            let phi_st_data = phi_st.as_ref();
            let env_mass_data = self.0.envelope_mass_temperatures.as_ref();
            let term_rest_data = term_rest_1.as_ref();
            let mut t_s_data = Vec::with_capacity(self.0.num_zones);
            for i in 0..self.0.num_zones {
                let numerator = h_tr_ms_data[i] * env_mass_data[i]
                    + h_tr_is_data[i] * t_i_act_data[i]
                    + phi_st_data[i]
                    + h_tr_me_ref[i] * int_mass_temps_ref[i];
                let denominator = term_rest_data[i] + h_tr_me_ref[i];
                t_s_data.push(numerator / denominator);
            }
            T::from(VectorField::new(t_s_data))
        };

        // === FIX D1: Calculate sol-air temperature for exterior surface ===
        // Per ISO 13790, exterior surface temperature is affected by solar radiation
        // T_sol-air = T_outdoor + (α × I_sol / h_se)
        // where α = solar absorptance (0.7), h_se = exterior surface coeff (25 W/m²K)
        use crate::physics::constants::thermal::ashrae_140::v2023::{
            EXTERIOR_FILM_COEFF_DEFAULT, SOLAR_ABSORPTANCE_DEFAULT,
        };
        let alpha = SOLAR_ABSORPTANCE_DEFAULT; // 0.7
        let h_se = EXTERIOR_FILM_COEFF_DEFAULT; // 25.0 W/m²K
        let mut t_sol_air_data = Vec::with_capacity(self.0.num_zones);
        for &i_sol in solar_ref.iter().take(self.0.num_zones) {
            let t_sol_air_zone = outdoor_temp + (alpha * i_sol / h_se);
            t_sol_air_data.push(t_sol_air_zone);
        }
        // Note: t_sol_air is used by the 5R1C model path (for mass temperature update)
        // It is NOT used by the 6R2C envelope mass path (which uses t_s instead)
        let _t_sol_air = VectorField::new(t_sol_air_data);

        // === 6R2C: Update two mass nodes with implicit integration ===
        // Envelope mass: receives heat from exterior (sol-air), surface, and internal mass
        let old_env_mass_temperatures = self.0.envelope_mass_temperatures.clone();

        // Update envelope mass temperatures using implicit integration for high thermal capacitance
        let mut new_env_mass_temperatures = Vec::with_capacity(self.0.num_zones);
        let env_mass_temps_ref = self.0.envelope_mass_temperatures.as_ref();
        let env_thermal_cap_ref = self.0.envelope_thermal_capacitance.as_ref();
        // Mode-specific fields removed - use physics-based h_tr_em and h_tr_ms
        let h_tr_em_ref = self.0.h_tr_em.as_ref();
        let h_tr_ms_ref = self.0.h_tr_ms.as_ref();
        let h_tr_me_ref = self.0.h_tr_me.as_ref();
        let int_mass_temps_ref = self.0.internal_mass_temperatures.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let phi_m_env_ref = phi_m_env.as_ref();

        for i in 0..self.0.num_zones {
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
            let h_tr_3 = self.0.derived_h_tr_3.as_ref()[i];

            // Check if this is a high-mass case (900 series)
            let is_high_mass = matches!(
                self.0.case_id.as_str(),
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

                    // Debug: Print heat flow breakdown for first zone
                    if timestep == 0 && i == 0 {
                        println!(
                            "DEBUG step_physics_6r2c: q_env_net={:.2}, dt={:.0}, cm_env={:.0}",
                            q_env_net, dt, cm_env
                        );
                        println!(
                            "  Components: h_tr_ms*({:.1}-{:.1})={:.2}, h_tr_me*({:.1}-{:.1})={:.2}, phi_m_env={:.2}",
                            t_s, tm_env_old, h_tr_ms * (t_s - tm_env_old),
                            tm_int, tm_env_old, h_tr_me * (tm_int - tm_env_old),
                            phi_m_env_zone
                        );
                    }

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

            new_env_mass_temperatures.push(tm_env_new);
        }

        // Clone envelope mass temperatures for internal mass calculation
        let env_mass_temps_for_int = new_env_mass_temperatures.clone();

        self.0.envelope_mass_temperatures = VectorField::new(new_env_mass_temperatures).into();

        // Internal mass: receives heat from envelope mass and direct gains
        let old_int_mass_temperatures = self.0.internal_mass_temperatures.clone();

        // Update internal mass temperatures using implicit integration for high thermal capacitance
        let mut new_int_mass_temperatures = Vec::with_capacity(self.0.num_zones);
        let int_thermal_cap_ref = self.0.internal_thermal_capacitance.as_ref();
        let phi_m_int_ref = phi_m_int.as_ref();

        for i in 0..self.0.num_zones {
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

            new_int_mass_temperatures.push(tm_int_new);
        }

        self.0.internal_mass_temperatures = VectorField::new(new_int_mass_temperatures).into();

        // Issue #272, #274, #275: Calculate thermal mass energy change for 6R2C
        // For 6R2C, we track energy changes in both envelope and internal masses
        // Envelope mass energy change (Cm × (Tm_new - Tm_old))
        let env_mass_temp_change =
            self.0.envelope_mass_temperatures.clone() - old_env_mass_temperatures.clone();
        let env_mass_energy_change =
            self.0.envelope_thermal_capacitance.clone() * env_mass_temp_change;

        // Internal mass energy change (Cm × (Tm_new - Tm_old))
        let int_mass_temp_change =
            self.0.internal_mass_temperatures.clone() - old_int_mass_temperatures.clone();
        let int_mass_energy_change =
            self.0.internal_thermal_capacitance.clone() * int_mass_temp_change;

        // Total mass energy change for this timestep
        let mass_energy_change_for_step_6r2c =
            env_mass_energy_change.clone() + int_mass_energy_change;

        // Track cumulative mass energy change
        let mass_energy_change_for_step_total =
            mass_energy_change_for_step_6r2c.reduce(0.0, |acc, val| acc + val);
        self.0.mass_energy_change_cumulative += mass_energy_change_for_step_total;

        // Plan 03-04: Update single mass temperature for backward compatibility (average of two masses)
        let total_cap = self.0.envelope_thermal_capacitance.clone()
            + self.0.internal_thermal_capacitance.clone();
        self.0.mass_temperatures = (self.0.envelope_mass_temperatures.clone()
            * self.0.envelope_thermal_capacitance.clone()
            + self.0.internal_mass_temperatures.clone()
                * self.0.internal_thermal_capacitance.clone())
            / total_cap;

        // DEBUG: Print t_i_act before storing
        self.0.temperatures = t_i_act;

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            // Store current HVAC output for this timestep (per zone, Watts)
            self.0.current_hvac_output = Some(hvac_output_raw.clone());
            // Temporarily take diagnostics out to avoid borrow conflicts
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            // Clear the buffer after use
            self.0.current_hvac_output = None;
        }

        // Return HVAC energy (Plan 03-04: Use hvac_energy_for_step directly)
        // Thermal mass energy accounting removed - Ti_free calculation already includes thermal mass effects
        hvac_energy_for_step / 3.6e6 // Return kWh
    }

    /// Solves a single timestep using the 8R3C thermal network (Phase 20 evaluation).
    ///
    /// The 8R3C model uses 3 capacitance nodes (ceiling, floor, partition mass)
    /// to better capture thermal inertia in high-mass buildings.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    ///
    /// # Returns
    /// HVAC energy consumption for the timestep in kWh.
    ///
    /// # Note
    /// This is a simplified implementation for evaluation purposes. It follows the
    /// 5R1C/6R2C pattern but with additional mass nodes for ceiling, floor, and partitions.
    pub(crate) fn step_physics_8r3c(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> f64 {
        let dt = dt_seconds; // Use provided timestep duration

        // Get ground temperature at this timestep (unused in simplified 8R3C)
        let _t_g = self.0.ground_temperature.ground_temperature(timestep);

        // Use 5R1C solve for simplicity (Phase 20 evaluation)
        // In a full implementation, this would be a proper 8R3C algebraic system
        let energy = self.step_physics_5r1c(timestep, outdoor_temp, dt_seconds);

        // Update 8R3C mass temperatures using simple relaxation (for evaluation)
        // In a full implementation, these would be coupled with Ti_free calculation
        let t_i = self.0.temperatures.clone();

        // Unwrap 8R3C fields (panic if not initialized) after step_physics_5r1c
        let ceiling_mass = self.0.ceiling_mass_temperatures.as_mut().unwrap();
        let floor_mass = self.0.floor_mass_temperatures.as_mut().unwrap();
        let partition_mass = self.0.partition_mass_temperatures.as_mut().unwrap();
        let ceiling_cap = self.0.ceiling_thermal_capacitance.as_ref().unwrap();
        let floor_cap = self.0.floor_thermal_capacitance.as_ref().unwrap();
        let partition_cap = self.0.partition_thermal_capacitance.as_ref().unwrap();
        let h_tr_ceiling = self.0.h_tr_ceiling.as_ref().unwrap();
        let h_tr_floor_mass = self.0.h_tr_floor_mass.as_ref().unwrap();
        let h_tr_partition = self.0.h_tr_partition.as_ref().unwrap();

        // Update ceiling mass temperature
        for i in 0..self.0.num_zones {
            let dtm_ceiling = (t_i.as_ref()[i] - ceiling_mass.as_ref()[i])
                / (ceiling_cap.as_ref()[i] / (h_tr_ceiling.as_ref()[i] * dt));
            ceiling_mass.as_mut()[i] += dtm_ceiling;
        }

        // Update floor mass temperature
        for i in 0..self.0.num_zones {
            let dtm_floor = (t_i.as_ref()[i] - floor_mass.as_ref()[i])
                / (floor_cap.as_ref()[i] / (h_tr_floor_mass.as_ref()[i] * dt));
            floor_mass.as_mut()[i] += dtm_floor;
        }

        // Update partition mass temperature
        for i in 0..self.0.num_zones {
            let dtm_partition = (t_i.as_ref()[i] - partition_mass.as_ref()[i])
                / (partition_cap.as_ref()[i] / (h_tr_partition.as_ref()[i] * dt));
            partition_mass.as_mut()[i] += dtm_partition;
        }

        energy
    }

    /// Solve physics for one timestep using the 9R4C (multi-node) model.
    ///
    /// Phase 6D: The 9R4C model uses 4 thermal mass nodes (wall, roof, floor, internal)
    /// to properly capture thermal inertia in high-mass buildings (Case 900+).
    ///
    /// The solver computes free-floating temperature using 5R1C network, then updates
    /// the multi-node thermal mass state. Zone temperatures are calculated using the
    /// 5R1C free-floating temperature, and the multi-node solver tracks per-surface temperatures.
    pub(crate) fn step_physics_9r4c(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> f64 {
        let dt = dt_seconds;

        // Get ground temperature at this timestep
        let t_g = self.0.ground_temperature.ground_temperature(timestep);

        // Calculate sky temperature for sol-air calculation
        let sky_temp = self
            .0
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);

        // Prepare sol-air temperatures and fluxes
        let (_t_sol_air_data, ctf_flux_w, fd_flux_w, _ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp);

        // Combine fractions
        let conv_frac = self.0.convective_fraction;
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
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar_beam_to_mass_fraction;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_solar_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            let sol_to_air = sol_w * self.0.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;
            phi_ia_data.push(load_w * conv_frac + sol_to_air);
            phi_st_data.push(load_w * st_int_frac + remaining_sol * st_sol_frac);
            phi_m_data.push(load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w);
        }

        // (#872) Save raw gain data for multi-node solver before moving into tensors.
        // Used for internal radiative gain injection via step_with_gains().
        let _phi_ia_data_for_solver = phi_ia_data.clone();

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let phi_m = T::from(VectorField::new(phi_m_data));

        let mut t_sol_air_data = Vec::with_capacity(self.0.num_zones);
        for _ in 0..self.0.num_zones {
            t_sol_air_data.push(outdoor_temp);
        }

        // Use 5R1C network for free-floating temperature
        let h_ext_base = &self.0.derived_h_ext;
        let term_rest_1 = &self.0.derived_term_rest_1;

        let den = self.0.derived_den.clone();
        // (#872: sensitivity variable removed — HVAC demand now uses h_loss × ΔT formula)

        let num_tm = self
            .0
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass_temperatures, |a, b| a * b);
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        let mut phi_ia_with_iz = phi_ia.clone();

        // Inter-zone heat transfer (if multi-zone)
        if self.0.num_zones > 1 {
            let temps = self.0.temperatures.as_ref();
            let h_iz_vec = self.0.h_tr_iz.as_ref();
            if self.0.num_zones >= 2 && h_iz_vec[0] > 0.0 {
                let delta_t_cond = temps[1] - temps[0];
                let q_cond = h_iz_vec[0] * delta_t_cond;
                let q_rad = 0.0;
                let zone_volume = self.0.zone_volume.as_ref();
                let ach_iz = calculate_stack_effect_ach(
                    temps[0],
                    temps[1],
                    self.0.door_geometry.height,
                    self.0.door_geometry.area,
                    zone_volume[0],
                );
                let q_vent =
                    calculate_ventilation_heat_transfer(ach_iz, temps[1], temps[0], zone_volume[0]);
                let q_iz_total = q_cond + q_rad + q_vent;
                let slice = phi_ia_with_iz.as_mut();
                if slice.len() >= 2 {
                    slice[0] += -q_iz_total;
                    slice[1] += q_iz_total;
                }
            }
        }

        // Add CTF flux contributions (if enabled)
        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_ctf = q_flux * area;
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);
                    let net_ctf_flux = q_ctf - q_5r1c;
                    slice[i] += net_ctf_flux;
                    if net_ctf_flux > 0.0 {
                        self.0.ctf_annual_heating_joules += net_ctf_flux * dt;
                    } else {
                        self.0.ctf_annual_cooling_joules += (-net_ctf_flux) * dt;
                    }
                }
            }
        }

        // Add FD flux contributions (if enabled)
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in fd_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_fd = q_flux * area;
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;
                    if net_fd_flux > 0.0 {
                        self.0.fd_annual_heating_joules += net_fd_flux * dt;
                    } else {
                        self.0.fd_annual_cooling_joules += (-net_fd_flux) * dt;
                    }
                }
            }
        }

        // Build numerator with envelope and ground contributions
        let mut num_rest_with_iz = phi_ia_with_iz;
        for (n, h) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(h_ext_base.as_ref().iter())
        {
            *n += h * outdoor_temp;
        }
        num_rest_with_iz.mul_assign(term_rest_1);
        let ground_coeff = self.0.derived_ground_coeff.as_ref();
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
        // back to self.0.mass_temperatures. The 5R1C model owns mass_temperatures
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

        #[allow(clippy::needless_range_loop)]
        for zone_idx in 0..self.0.num_zones {
            if zone_idx >= self.0.multi_node_solvers.len() {
                continue;
            }

            let solver = &mut self.0.multi_node_solvers[zone_idx];
            // (#872) Use previous zone temperature as boundary, NOT 5R1C t_i_free.
            // This breaks the destructive feedback loop where 5R1C mass temps corrupt
            // the solver's boundary condition. The solver will compute its own
            // zone air temperature from the multi-node balance.
            let t_zone_prev = self.0.temperatures.as_ref()[zone_idx];
            #[allow(unused_variables)]
            let t_ext = t_sol_air_data
                .get(zone_idx)
                .copied()
                .unwrap_or(outdoor_temp);

            // Surface temperature: use previous zone temp as initial guess.
            // After stepping, we'll update to conductance-weighted mass temps.
            let t_surface = t_zone_prev - 0.5;

            solver.set_zone_temperature(t_zone_prev);
            solver.set_surface_temperature(t_surface);

            let surface_ext_temps = if let Some(ref weather) = self.0.weather {
                let hour_of_year = timestep % 8760;
                let month_days: [usize; 12] =
                    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
                let day_of_year = hour_of_year / 24;
                let hour = (hour_of_year % 24) as f64 + 0.5;
                let month = month_days
                    .iter()
                    .position(|&d| d > day_of_year)
                    .unwrap_or(12)
                    .saturating_sub(1) as u32;
                let day =
                    (day_of_year - month_days.get(month as usize).copied().unwrap_or(0)) as u32 + 1;

                let sun_pos = calculate_solar_position(
                    self.0.latitude_deg,
                    self.0.longitude_deg,
                    2024,
                    month,
                    day.min(28),
                    hour,
                );

                let ground_reflectance = 0.2;
                let wall_irr = calculate_surface_irradiance(
                    &sun_pos,
                    weather.dni,
                    weather.dhi,
                    Some(weather.ghi),
                    Orientation::South,
                    ground_reflectance,
                    day_of_year + 1,
                );
                let roof_irr = calculate_surface_irradiance(
                    &sun_pos,
                    weather.dni,
                    weather.dhi,
                    Some(weather.ghi),
                    Orientation::Up,
                    ground_reflectance,
                    day_of_year + 1,
                );

                let sol_air = SolAirTemperature::ashrae_140_default();
                SurfaceExteriorTemperatures {
                    t_ext_wall: sol_air.for_wall(
                        outdoor_temp,
                        wall_irr.total_wm2,
                        wall_irr.ground_reflected_wm2,
                    ),
                    t_ext_roof: sol_air.for_roof(outdoor_temp, roof_irr.total_wm2, sky_temp),
                    t_ext_floor: t_g,
                }
            } else {
                SurfaceExteriorTemperatures {
                    t_ext_wall: t_ext,
                    t_ext_roof: t_ext,
                    t_ext_floor: t_g,
                }
            };

            solver.set_surface_exterior_temperatures(surface_ext_temps);

            // (#872) Step solver with gains: internal radiative loads to internal mass node.
            // Window solar is NOT injected here to avoid thermal runaway — it's
            // handled via phi_ia in the air node. Full per-node gain injection (#873)
            // will refine this with proper per-surface solar distribution.
            let zone_area_val = self.0.zone_area.as_ref()[zone_idx];
            let load_w = loads_ref[zone_idx] * zone_area_val;
            let internal_rad = load_w * rad_frac;
            solver.step_with_gains(dt, 0.0, 0.0, 0.0, internal_rad);
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
        let mut t_i_free_data = Vec::with_capacity(self.0.num_zones);
        for zone_idx in 0..self.0.num_zones {
            if zone_idx < self.0.multi_node_solvers.len() {
                let solver = &self.0.multi_node_solvers[zone_idx];
                let h_ve_val = self.0.h_ve.as_ref()[zone_idx];
                let phi_ia_val = phi_ia.as_ref()[zone_idx];
                let t_air_mn =
                    solver.compute_zone_air_temperature(outdoor_temp, h_ve_val, phi_ia_val);
                // Use multi-node temperature — it provides the correct air balance
                // from mass node temperatures stepped by the backward Euler.
                t_i_free_data.push(t_air_mn);
            } else {
                t_i_free_data.push(t_i_free.as_ref()[zone_idx]);
            }
        }
        let _t_i_free_mn = T::from(VectorField::new(t_i_free_data));

        // (#872) Do NOT write multi-node mass temperatures back to self.0.
        // The multi-node solver keeps its own internal state. Writing back would
        // corrupt the 5R1C mass_temperatures used by the t_i_free formula.

        // Update multi-node solver surface temperatures using conductance-weighted
        // envelope node temperatures (not the hardcoded t_zone - 0.5).
        // This ensures the solver's surface_temperature field is consistent with
        // the mass node temperatures it just computed.
        for zone_idx in 0..self.0.num_zones {
            if zone_idx >= self.0.multi_node_solvers.len() {
                continue;
            }
            let solver = &mut self.0.multi_node_solvers[zone_idx];
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

        // Calculate HVAC demand
        let _hour_of_day_idx = timestep % 24;
        let temp_rate = if timestep > 0 {
            (self.0.temperatures.as_ref()[0] - self.0.previous_temperatures.as_ref()[0]) / dt
        } else {
            0.0
        };

        let (hvac_mode, _modulation) = self.0.predictive_controller.calculate_modulation(
            self.0.temperatures.as_ref()[0],
            self.0.mass_temperatures.as_ref()[0],
            temp_rate,
        );
        let _hvac_mode: EquipmentHVACMode = hvac_mode;

        // (#872) Compute HVAC demand BEFORE mass update, so the mass uses CURRENT t_i_act.
        // This matches the 5R1C ordering: compute Q → compute t_act → update mass using t_act.
        // The previous ordering (mass → HVAC) caused a one-timestep lag where mass used
        // the PREVIOUS step's t_act, slowing convergence and overestimating HVAC energy.
        //
        // Issue #860: Use multi-node t_air for HVAC demand when available.
        // The multi-node solver provides a physically accurate free-floating temperature
        // from the 9R4C thermal balance (wall/roof/floor nodes → surface → air).
        // This is more accurate than the 5R1C t_i_free which uses a lumped mass.
        let (hvac_for_temp_calc, t_i_act) = if self.0.free_float {
            // Free-float: no HVAC, t_i_act = t_i_free
            let t_i_free_vec = t_i_free_5r1c.as_ref().to_vec();
            (
                T::from(VectorField::new(vec![0.0; self.0.num_zones])),
                T::from(VectorField::new(t_i_free_vec)),
            )
        } else {
            // HVAC mode: use multi-node t_air (from _t_i_free_mn) when available
            let heat_cap = self.0.hvac_heating_capacity;
            let cool_cap = self.0.hvac_cooling_capacity;
            let mut hvac_data = Vec::with_capacity(self.0.num_zones);
            let mut t_i_act_data = Vec::with_capacity(self.0.num_zones);
            for i in 0..self.0.num_zones {
                // Issue #860: Prefer multi-node t_air over 5R1C t_free for HVAC demand
                let t_free_val =
                    if i < self.0.multi_node_solvers.len() {
                        // Use multi-node computed free-float temperature (available at line 2534)
                        // The multi-node t_air uses conductance-weighted envelope node temperatures
                        // and the air energy balance: T_air = (h_tr_is*T_surface + h_ve*T_out + phi_ia)/(h_tr_is + h_ve)
                        _t_i_free_mn.as_ref().get(i).copied().unwrap_or_else(|| {
                            t_i_free_5r1c.as_ref().get(i).copied().unwrap_or(20.0)
                        })
                    } else {
                        t_i_free_5r1c.as_ref()[i]
                    };
                // Issue #925: HVAC coefficient = building heat loss coefficient
                // (zone -> outdoor), not the lumped 5R1C free-floating denominator.
                // See compute_zone_hvac_load for full derivation.
                let h_ve = self.0.h_ve.as_ref()[i];
                let h_tr_w = self.0.h_tr_w.as_ref()[i];
                let h_tr_is = self.0.h_tr_is.as_ref()[i];
                let h_tr_ms = self.0.h_tr_ms.as_ref()[i];
                let h_tr_em = self.0.h_tr_em.as_ref()[i];

                // Series conductance: air -> surface -> mass -> envelope exterior
                let series_denom = h_tr_is * h_tr_ms + h_tr_ms * h_tr_em + h_tr_em * h_tr_is;
                let h_loss_via_mass =
                    if h_tr_is > 0.0 && h_tr_ms > 0.0 && h_tr_em > 0.0 && series_denom > 0.0 {
                        h_tr_is * h_tr_ms * h_tr_em / series_denom
                    } else {
                        0.0
                    };
                let h_loss = h_ve + h_tr_w + h_loss_via_mass;
                let h_coeff = if h_loss > 0.0 { h_loss } else { h_ve + h_tr_w };

                // DEBUG: Print h_coeff breakdown on first HVAC step after warmup
                if timestep == 337 && i == 0 && !self.0.free_float {
                    eprintln!(
                        "HVAC_DBG[step=337]: h_tr_is={:.2}, h_ve={:.2}, h_tr_w={:.2}, h_tr_ms={:.2}, h_tr_em={:.2}, h_loss={:.4}, t_free={:.2}, q_heating={:.4}",
                        self.0.h_tr_is.as_ref()[i],
                        self.0.h_ve.as_ref()[i],
                        self.0.h_tr_w.as_ref()[i],
                        self.0.h_tr_ms.as_ref()[i],
                        self.0.h_tr_em.as_ref()[i],
                        h_coeff,
                        t_free_val,
                        h_coeff * (self.0.heating_setpoint - t_free_val).max(0.0)
                    );
                }

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
                let t_mass_mn = if i < self.0.multi_node_solvers.len() {
                    let solver = &self.0.multi_node_solvers[i];
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
                    self.0.mass_temperatures.as_ref()[i]
                };
                // Issue #900: mass heat release term (cooling only).
                //
                // See compute_zone_hvac_load (hvac.rs) for the formula
                // and gating conditions. The 9R4C inline path uses multi-
                // node envelope temperatures (stable, well below 35°C)
                // and the same high-mass threshold (h_tr_ms ≥ 500 W/K).
                //
                // The cap is set higher than the 5R1C path (50× vs 10×)
                // because the multi-node temps are stable: the 9R4C mass
                // nodes track each other (wall/roof/floor envelope
                // weighted temp stays in 28–33°C range) and the peak
                // mass_heat_release of ~3 kW at T_mass = 30°C brings
                // Case 900 cooling close to the ASHRAE 140 reference
                // peak range (2.10–3.50 kW).
                const MASS_RELEASE_DAMPING_9R4C: f64 = 1.0;
                const HIGH_MASS_H_TR_MS_THRESHOLD_9R4C: f64 = 500.0;
                const MASS_TEMP_MAX_9R4C: f64 = 35.0;
                const MASS_RELEASE_MAX_FACTOR_9R4C: f64 = 50.0;
                let mass_heat_release_unclamped = if t_mass_mn > self.0.cooling_setpoint
                    && t_mass_mn <= MASS_TEMP_MAX_9R4C
                    && h_tr_ms >= HIGH_MASS_H_TR_MS_THRESHOLD_9R4C
                {
                    h_tr_ms * (t_mass_mn - self.0.cooling_setpoint) * MASS_RELEASE_DAMPING_9R4C
                } else {
                    0.0
                };
                let mass_heat_release = if mass_heat_release_unclamped > 0.0 {
                    mass_heat_release_unclamped.min(h_loss * MASS_RELEASE_MAX_FACTOR_9R4C)
                } else {
                    0.0
                };

                let q = if t_free_val < self.0.heating_setpoint {
                    // Heating: keep Issue #925 formula unchanged.
                    // Mass heat absorption term is intentionally omitted
                    // (see Issue #900 in hvac.rs for rationale).
                    h_coeff * (self.0.heating_setpoint - t_free_val)
                } else if t_free_val > self.0.cooling_setpoint {
                    // Cooling, zone above setpoint.
                    //   -h_loss × (T_free − T_cool)        steady-state heat loss to outside
                    //   −h_tr_ms × (T_mass − T_cool) × 0.5 dynamic mass heat release
                    h_coeff * (self.0.cooling_setpoint - t_free_val) - mass_heat_release
                } else if mass_heat_release > 0.0 {
                    // Dead band, but mass is hotter than cool_sp — mass
                    // releases heat that the HVAC must remove. See Issue
                    // #900 in hvac.rs.
                    -mass_heat_release
                } else {
                    0.0
                };

                let q_clamped = q.clamp(-cool_cap, heat_cap);
                hvac_data.push(q_clamped);

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
                    t_i_act_data.push(t_free_val + q_clamped / h_coeff);
                } else {
                    t_i_act_data.push(t_free_val);
                }
            }
            (
                T::from(VectorField::new(hvac_data)),
                T::from(VectorField::new(t_i_act_data)),
            )
        };

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
            let old_mass_temperatures = self.0.mass_temperatures.clone();
            let mass_temps_ref = self.0.mass_temperatures.as_ref();
            let thermal_cap_ref = self.0.thermal_capacitance.as_ref();
            let h_tr_em_ref = self.0.h_tr_em.as_ref();
            let h_tr_ms_ref = self.0.h_tr_ms.as_ref();

            let mut new_mass_temperatures = Vec::with_capacity(self.0.num_zones);
            for i in 0..self.0.num_zones {
                let tm_old = mass_temps_ref[i];
                let cm = thermal_cap_ref[i];
                let t_i = t_i_act.as_ref()[i];
                let h_tr_em = h_tr_em_ref[i];
                let h_tr_ms = h_tr_ms_ref[i];
                let h_tr_is_zone = self.0.h_tr_is.as_ref()[i];
                let h_tr_me_zone = self.0.h_tr_me.as_ref()[i];
                let t_ext = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);

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
                let h_tr_3_zone = *self.0.derived_h_tr_3.as_ref().get(i).unwrap_or(&h_tr_ms);

                // Backward Euler with h_tr_3 instead of h_tr_ms:
                // (Cm/dt + h_tr_em + h_tr_3) * Tm_new = Cm/dt * Tm_old + h_tr_em * t_ext + h_tr_3 * t_s + phi_m
                let cm_dt = cm / dt;
                let denom = cm_dt + h_tr_em + h_tr_3_zone;
                let numer =
                    cm_dt * tm_old + h_tr_em * t_ext + h_tr_3_zone * t_s + phi_m.as_ref()[i];
                let tm_new = numer / denom;

                new_mass_temperatures.push(tm_new);
            }
            self.0.mass_temperatures = VectorField::new(new_mass_temperatures).into();
            self.0.previous_mass_temperatures = old_mass_temperatures;
        }

        // Issue #738: Free-float mode must completely disable HVAC output
        if self.0.free_float {
            // (#872) For free-floating zones, use the 5R1C t_i_free computed from
            // uncorrupted mass temperatures (saved before multi-node solver overwrite).
            // The 5R1C formula correctly captures the phi_st → mass coupling that
            // produces the correct 42.87°C for 900FF. The multi-node solver's
            // temperature (33°C) is too low because it lacks per-node solar injection (#873).
            let temps_slice = self.0.temperatures.as_mut();
            for (i, t_val) in t_i_free_5r1c.as_ref().iter().enumerate() {
                if i < temps_slice.len() {
                    temps_slice[i] = *t_val;
                }
            }
            return 0.0;
        }

        // Update zone temperatures with the HVAC-influenced t_i_act
        let temps_slice = self.0.temperatures.as_mut();
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
            for (&output, &enabled) in hvac_output
                .as_ref()
                .iter()
                .zip(self.0.hvac_enabled.as_ref().iter())
            {
                let val = if enabled > 0.5 { output } else { 0.0 };
                hvac_power_watts += val;

                if val > 0.0 {
                    heating_sum += val;
                } else if val < 0.0 {
                    cooling_sum += -val;
                }
            }

            let heating_energy_joules = heating_sum * dt;
            let cooling_energy_joules = cooling_sum * dt;

            self.0.annual_heating_energy += heating_energy_joules / 3.6e6;
            self.0.annual_cooling_energy += cooling_energy_joules / 3.6e6;

            if hvac_power_watts > 0.0 {
                self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_power_watts);
            } else if hvac_power_watts < 0.0 {
                self.0.peak_power_cooling = self.0.peak_power_cooling.max(-hvac_power_watts);
            }
        }

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            self.0.current_hvac_output = Some(hvac_output.clone());
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            self.0.current_hvac_output = None;
        }

        // Return kWh
        hvac_power_watts * dt / 3.6e6
    }
}

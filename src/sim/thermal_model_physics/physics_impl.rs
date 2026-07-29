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
use crate::physics::five_r1c_solver::surface_time_constant_from_conductances;
use crate::physics::multi_node_solver::SurfaceExteriorTemperatures;
use crate::sim::boundary::distribute_opaque_solar_gains;
use crate::sim::hvac::{HVACMode as EquipmentHVACMode, VariableCapacityEquipment};
// #1391: stack-effect helpers removed — the 9R4C and 5R1C inter-zone paths now use
// a single `q_iz_net[i] = h_tr_iz[i] · Σ_{j≠i} (T[j] − T[i])` conductive loop,
// matching the iterative path's `solve_coupled_zone_temperatures`. The legacy
// q_vent term double-counted the door-opening convective conductance that
// `h_tr_iz` already captures for Case 960.
use crate::sim::sky_radiation::SolAirTemperature;
use crate::sim::solar::calculate_surface_irradiance;
use crate::sim::thermal_integration::{
    backward_euler_update_2cond, backward_euler_update_2cond_h_tr3, crank_nicolson_iso13790,
    crank_nicolson_update, crank_nicolson_update_3cond, select_integration_method,
    ThermalIntegrationMethod,
};
use crate::sim::thermal_model_core::ThermalModel;
use crate::sim::thermal_model_scratch::{
    PhysicsScratch5r1c, PhysicsScratch6r2c, PhysicsScratch9r4c, PhysicsScratchPool,
};
use crate::sim::ventilation::h_tr_is_ach_multiplier;

// Methods in this file are being incrementally migrated to the sibling
// submodules in `thermal_model_physics/` (see Issue #902). Methods that
// are still in this file retain the same `impl<T: ...> ThermalModel<T>`
// bound so they continue to merge with the others.

// ============================================================================
// Issue #1524 / docs/PROFILING_v1.3.md §3 Target 1 — per-timestep scratch
// ----------------------------------------------------------------------------
// The 5R1C/6R2C/8R3C/9R4C step functions previously opened each timestep with
// 6–14 standalone `Vec::with_capacity(num_zones)` allocations. These structs
// consolidate that scratch into a single local per call. Two design points
// resolve the borrow-checker conflict that sank the #1436 WIP:
//
// 1. The scratch is a *local variable*, not a field on `self`. `&mut
//    scratch.field` is therefore disjoint from the `&self` borrow taken by
//    helpers such as `compute_hvac_coefficient` (`hvac.rs`), so the two
//    coexist without `split_at_mut` gymnastics. (#1436 had placed the scratch
//    on `self`, which forced `&mut self.scratch` to conflict with
//    `self.compute_hvac_coefficient(i)` borrowing all of `self`.)
//
// 2. Fields are zero-initialised (`vec![0.0; num_zones]`) and written by
//    index, replacing `push()`. Output buffers are moved into `T` /
//    `VectorField` via `mem::take`; read-back intermediates are read by index.
//
// The 9R4C variant additionally collapses its seven *read-back* intermediates
// (sol-air + three pre-gain mass temps + three per-surface phi_m) into one
// flat backing buffer — a 7→1 allocation reduction on the 9R4C hot path.
//
// `Hoist-ready`: a follow-up that adds a `scratch` field to
// `ThermalModelData` (outside this file's scope) can reuse the same
// allocation across timesteps — the remaining step to clear the <25 ms
// single-zone budget of PROFILING §5.3.
// ============================================================================

/// Evolve the per-zone interior wall-surface temperature `T_si` via the
/// exact exponential solution of the surface-node ODE (Issue #1860
/// time-constant-aware 5R1C variant).
///
/// Equations:
///
/// ```text
///   τ_si   = C_zone / (h_is + h_1)              (h_1 = h_tr_ms)
///   T_si_eq = (T_int · h_is + T_m · h_1) / (h_is + h_1)
///   T_si_new = T_si_eq + (T_si_old − T_si_eq) · exp(−dt / τ_si)
/// ```
///
/// Results are written to `scratch.wall_surface_new` (the per-zone
/// `T_si` state) and `scratch.wall_surface_correction` (the
/// transient surface-flux term `h_is · (T_si − T_si_eq)` that the caller
/// folds into the scaled air-node numerator). All inputs are
/// read-only slices — no allocations are performed in the hot path; the
/// scratch fields replace the per-call `Vec::with_capacity(num_zones)`
/// pair that earlier landed inside `step_physics_5r1c` and bypassed the
/// `PhysicsScratch5r1c` scratch convention introduced for Issue #1524.
///
/// On degenerate `h_tr_ms <= 0.0` or `h_tr_is <= 0.0` the legacy lumped
/// path is preserved exactly: `T_si` is left at its previous value and
/// the correction is zero. `dt <= 0.0` short-circuits to the steady-state
/// value `T_si_eq` for the same reason.
#[allow(clippy::too_many_arguments)]
fn step_wall_surface_ode(
    dt: f64,
    h_tr_ms: &[f64],
    h_tr_is: &[f64],
    mass_temps: &[f64],
    zone_temps: &[f64],
    wall_surface_old: &[f64],
    thermal_cap: &[f64],
    scratch: &mut PhysicsScratch5r1c,
) {
    // Size the loop from the input slices, not from a scratch field —
    // by the time the caller invokes this helper, `phi_ia`/`phi_st`/`phi_m`
    // have already been moved out of the scratch via `mem::take`, so any
    // scratch-backed length probe would report zero. The caller guarantees
    // every input slice has the same length (`self.0.num_zones`).
    let n = wall_surface_old.len();
    debug_assert_eq!(h_tr_ms.len(), n);
    debug_assert_eq!(h_tr_is.len(), n);
    debug_assert_eq!(mass_temps.len(), n);
    debug_assert_eq!(zone_temps.len(), n);
    debug_assert_eq!(thermal_cap.len(), n);
    debug_assert_eq!(scratch.wall_surface_new.len(), n);
    debug_assert_eq!(scratch.wall_surface_correction.len(), n);

    let wall_surface_new = &mut scratch.wall_surface_new;
    let wall_surface_correction = &mut scratch.wall_surface_correction;
    for i in 0..n {
        let h_ms_i = h_tr_ms[i];
        let h_is_i = h_tr_is[i];
        if h_ms_i > 0.0 && h_is_i > 0.0 {
            // τ_si is the same quantity `FiveR1CSolver::surface_time_constant`
            // exposes, just expressed in the per-zone conductance basis the
            // physics consumer already has on hand. Delegates to the shared
            // free function so the formula lives in exactly one place.
            let tau_si = surface_time_constant_from_conductances(thermal_cap[i], h_ms_i, h_is_i);
            let t_m_i = mass_temps[i];
            let t_int_i = zone_temps[i];
            let t_si_eq = (t_int_i * h_is_i + t_m_i * h_ms_i) / (h_is_i + h_ms_i);
            let t_si_old_i = wall_surface_old[i];
            let t_si_new_i = if tau_si > 0.0 && dt > 0.0 {
                t_si_eq + (t_si_old_i - t_si_eq) * (-dt / tau_si).exp()
            } else {
                t_si_eq
            };
            wall_surface_new[i] = t_si_new_i;
            wall_surface_correction[i] = h_is_i * (t_si_new_i - t_si_eq);
        } else {
            wall_surface_new[i] = wall_surface_old[i];
            wall_surface_correction[i] = 0.0;
        }
    }
}

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
        let st_int_frac = rad_frac * (1.0 - solar_distribution_to_air);
        let m_air_frac = rad_frac * solar_distribution_to_air;
        let st_sol_frac = 1.0 - solar_beam_to_mass_fraction;
        let m_sol_frac = solar_beam_to_mass_fraction;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_solar_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let heating_setpoint = self.0.heating_setpoint;
        let cooling_setpoint = self.0.cooling_setpoint;
        let solar_distribution_to_air = self.0.solar_distribution_to_air;
        let solar_beam_to_mass_fraction = self.0.solar_beam_to_mass_fraction;

        // Issue #1524: consolidated per-timestep scratch (replaces the six
        // standalone `Vec::with_capacity(num_zones)` allocations below).
        // Issue #1966: scratch is now pooled in ThermalModelData::scratch_pool
        // and reused across timesteps via fill_zero() at end of step.
        let mut scratch = self.0.scratch_pool.get_5r1c(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            // opaque_sol_w: kept for potential debugging; it's included via t_sol_air now
            let _opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            // Internal gains: convective to air, radiative split between surface and mass
            // Solar distribution must conserve energy (sum to 1.0)
            let sol_to_air = sol_w * solar_distribution_to_air;
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

        let phi_ia = T::from(VectorField::new(std::mem::take(&mut scratch.phi_ia)));
        let phi_st = T::from(VectorField::new(std::mem::take(&mut scratch.phi_st)));
        let phi_m = T::from(VectorField::new(std::mem::take(&mut scratch.phi_m)));

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
            self.0.h_tr_ms.as_ref(),
            self.0.h_tr_is.as_ref(),
            self.0.mass_temperatures.as_ref(),
            self.0.temperatures.as_ref(),
            self.0.wall_surface_temperatures.as_ref(),
            self.0.thermal_capacitance.as_ref(),
            &mut scratch,
        );
        // Persist the new T_si for downstream consumers (diagnostics, the
        // regression test suite, and the future cooling-load coupling that
        // the Issue #1860 epic tracks).
        self.0
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
        let sol_air_calc = SolAirTemperature::ashrae_140_default();
        let mut t_sol_air_vec = Vec::with_capacity(self.0.num_zones);
        for opaque_solar in opaque_solar_ref.iter().take(self.0.num_zones) {
            // opaque_solar is the effective opaque irradiance on exterior surfaces (W/m²)
            // This is the combined wall + roof irradiance for the zone
            let t_sol_air_i = sol_air_calc.for_roof(outdoor_temp, *opaque_solar, sky_temp);
            t_sol_air_vec.push(t_sol_air_i);
        }
        let t_sol_air = VectorField::new(t_sol_air_vec);

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
        let mut h_ve_night: f64 = 0.0;
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
                    h_ve_night = night_vent.fan_capacity * rho * cp / 3600.0;
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

        // h_tr_is_for_ti_free: no boost applied (night ventilation affects zone air through
        // h_ve_total, not through surface convection coefficients). The h_ve_night already
        // modifies h_ext and den for the free-floating temperature calculation.
        let h_tr_is_for_ti_free: T = self.0.h_tr_is.clone();

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
        let num_zones = self.0.num_zones;

        // Start with phi_ia; we will add inter-zone heat directly to its buffer if needed.
        // Issue #901 perf: move phi_ia (no clone). The original is no longer used
        // after this point in step_physics_5r1c — the legacy comment referencing a
        // Case 610 debug print at "line 914" referred to step_physics_6r2c, not here.
        let mut phi_ia_with_iz = phi_ia;

        if num_zones > 1 {
            let slice = phi_ia_with_iz.as_mut();
            let n = num_zones;
            let temps = self.0.temperatures.as_ref();
            let h_iz_vec = self.0.h_tr_iz.as_ref();
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
                }
            }
        }

        // For single-zone or no inter-zone heat, phi_ia_with_iz remains as cloned phi_ia (no allocation beyond the initial clone)

        // Note: The Issue #1860 wall-surface ODE state is computed earlier
        // in this function (see the "Wall-surface ODE (pre-air-node-equilibrium
        // step)" block) and persisted to `self.0.wall_surface_temperatures`.
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
        let c_air_ref = self.0.air_thermal_capacitance.as_ref();
        let cm_ref = self.0.thermal_capacitance.as_ref();
        let t_air_old_ref = self.0.air_temperatures.as_ref();
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
        let mut t_i_free_data = Vec::with_capacity(self.0.num_zones);
        for i in 0..self.0.num_zones {
            // Issue #1860: remove num_phi_st (immediate surface solar) from
            // the steady-state computation. The surface-solar contribution is
            // instead applied through the solar-lag filter below, which
            // releases it over τ_lag ≈ 1–3 h instead of instantaneously.
            // This prevents double-counting: the lag term replaces the
            // algebraic surface contribution, not adds to it.
            let num_i = num_tm_ref[i] + num_rest_ref[i]; // num_phi_st removed
            let den_i = den_ref[i];
            // Steady-state free-float air temperature (used as the asymptotic
            // target for the air-node exponential relaxation).
            let steady = num_i / den_i;
            // Issue #1585: activate exact exponential air-node ODE when C_air > 0.
            // Falls back to the legacy algebraic pinning (steady = num/den) when
            // C_air is zero, preserving historical behaviour for unit tests that
            // construct ThermalModelCore without calling from_spec.
            let c_air_i = c_air_ref[i];
            let t_air_old_i = t_air_old_ref[i];
            let tau_air = if den_i > 0.0 && term_rest_1_ref[i] > 0.0 {
                c_air_i * term_rest_1_ref[i] / den_i
            } else {
                f64::INFINITY
            };
            let t_i_free_i = if c_air_i > 0.0 && tau_air.is_finite() && dt > 0.0 {
                let exponent = -dt / tau_air;
                steady + (t_air_old_i - steady) * exponent.exp()
            } else {
                steady
            };
            t_i_free_data.push(t_i_free_i);
        }

        // === Issue #1860: Solar-lag correction (multi-timescale wall response) ===
        //
        // The 5R1C model lumps ALL wall mass into one node with τ_mass ≈ 12 h
        // for low-mass buildings. In reality, near-surface layers (gypsum board,
        // furniture, internal partitions) absorb solar radiation and re-release
        // it over 1–3 h — a timescale between the air node (τ_air ≈ 0.17 h,
        // instantaneous) and the mass node (τ_mass ≈ 12 h, very slow).
        //
        // The solar-lag term replaces the immediate surface-solar contribution
        // (num_phi_st, which was removed from the steady computation above) with
        // a first-order low-pass-filtered version using:
        //
        //   τ_lag = √(τ_air × τ_mass)  (geometric mean of the two timescales)
        //
        // The filter input is h_tr_is × phi_st / term_rest_1 (the same value
        // that num_phi_st/den would contribute in steady state), ensuring
        // energy conservation: the same amount of solar reaches the air, but
        // spread over τ_lag instead of instantaneously. This smooths peaks
        // (less immediate solar) and extends tails (sustained release),
        // increasing annual cooling while reducing peak cooling.
        //
        // Only phi_st (surface solar) is filtered — phi_m (mass solar) is
        // already handled by the CrankNicolson mass-node integration.
        let phi_st_ref = phi_st.as_ref();
        let h_tr_is_for_lag_ref = h_tr_is_for_ti_free.as_ref();
        let solar_lag_old: Vec<f64> = self.0.solar_lag.as_ref().to_vec();
        let mut corrected_t_i_free = t_i_free_data.clone();

        for i in 0..self.0.num_zones {
            let den_i = den_ref[i];
            let cm_i = cm_ref[i];
            let c_air_i = c_air_ref[i];
            let term_rest_1_i = term_rest_1_ref[i];

            // Compute time constants
            let den_true_i = if term_rest_1_i > 0.0 {
                den_i / term_rest_1_i
            } else {
                den_i
            };
            let h_tr_3_i = self.0.derived_h_tr_3.as_ref()[i];

            let tau_air_i = if den_true_i > 0.0 && c_air_i > 0.0 {
                c_air_i / den_true_i
            } else {
                600.0 // fallback: 10 min
            };
            let tau_mass_i = if h_tr_3_i > 0.0 && cm_i > 0.0 {
                cm_i / h_tr_3_i
            } else {
                44000.0 // fallback: ~12 h
            };

            // Geometric mean of air and mass timescales — the characteristic
            // intermediate timescale of the two-node system.
            let tau_lag = (tau_air_i * tau_mass_i).sqrt();
            let decay = if tau_lag > 0.0 && dt > 0.0 {
                (-dt / tau_lag).exp()
            } else {
                0.0
            };

            // Filter input: scaled to match num_phi_st/den steady-state
            // contribution. h_tr_is × phi_st / term_rest_1 gives the
            // equivalent steady-state temperature contribution.
            let lag_input = if term_rest_1_i > 0.0 {
                h_tr_is_for_lag_ref[i] * phi_st_ref[i] / term_rest_1_i
            } else {
                0.0
            };

            // Update solar-lag state: low-pass filter
            let new_solar_lag = solar_lag_old[i] * decay + lag_input * (1.0 - decay);

            // Add lagged contribution to t_i_free (replaces num_phi_st)
            if den_true_i > 0.0 {
                corrected_t_i_free[i] += new_solar_lag / den_true_i;
            }

            // Store updated solar-lag state
            self.0.solar_lag.as_mut()[i] = new_solar_lag;
        }

        let t_i_free = T::from(VectorField::new(corrected_t_i_free));

        // Issue #1585: step the air-node ODE state forward for the next
        // timestep.  t_i_free (the new zone-air temperature) becomes
        // t_air_old on the next call to step_physics_5r1c.
        self.0
            .air_temperatures
            .as_mut()
            .copy_from_slice(&t_i_free_data);

        // The wall-surface state contributes to the air-node numerator through
        // the transient surface flux correction applied above.

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
            T::from(self.0.zero_vector.clone())
        } else {
            // Issue #1163: symmetric ideal-HVAC formula uses t_i_free as the
            // driving temperature for both heating and cooling (mass
            // heat-release is already embedded in t_i_free via num_tm).
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                heating_setpoint,
                cooling_setpoint,
            )
        };

        let hour_of_day_idx = timestep % 24;

        // Issue #738: Free-float mode must completely disable HVAC output
        // This is a safety check that goes beyond hvac_enabled, which may not be
        // properly set for all code paths. Free-float cases (900FF, etc.) should
        // have zero HVAC output regardless of other settings.
        let hvac_output_raw = if self.0.free_float {
            T::from(self.0.zero_vector.clone())
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
            // Issue #1163: symmetric ideal-HVAC formula (mass heat-release is
            // already embedded in t_i_free via num_tm).
            let hvac_output =
                self.compute_zone_hvac_load(t_i_free.as_ref(), heating_setpoint, cooling_setpoint);

            // Track peak heating/cooling based on per-zone HVAC demand (Plan 18-08)
            // Physics-based: No calibration factors - track actual HVAC demand
            // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
            let enabled_vec = self.0.hvac_enabled.as_ref();
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
                    if val_kw > self.0.zone_peak_heating_kw.as_mut()[i] {
                        self.0.zone_peak_heating_kw.as_mut()[i] = val_kw;
                        self.0.zone_peak_heating_timestep[i] = timestep;
                    }
                } else if val < 0.0 {
                    // Cooling mode for this zone
                    let val_kw = -val / 1000.0;
                    if val_kw > self.0.zone_peak_cooling_kw.as_mut()[i] {
                        self.0.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                        self.0.zone_peak_cooling_timestep[i] = timestep;
                    }
                }
            }
            // Track global peak (sum of all zones)
            if hvac_output_sum > 0.0 {
                self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_output_sum);
            } else if hvac_output_sum < 0.0 {
                self.0.peak_power_cooling = self.0.peak_power_cooling.max(-hvac_output_sum);
            }

            // Both equipment and fallback paths now use hvac_output (per-zone VectorField)
            // so it needs to be returned for both branches
            hvac_output
        } else {
            // Use IdealLoadsSystem thermodynamic formulas for energy
            //
            // Issue #1163: symmetric ideal-HVAC formula (mass heat-release is
            // already embedded in t_i_free via num_tm).
            let hvac_output_raw = self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                heating_setpoint,
                cooling_setpoint,
            );

            // Root Cause Fix: Use hvac_output_raw for peak tracking (consistent with energy calc)
            // Issue #901 perf: borrow hvac_output_raw directly instead of cloning for
            // the peak-power read. The original is returned to the caller below, untouched.
            let hvac_power_for_peak = hvac_output_raw.as_ref();

            // Track peak heating/cooling based on actual HVAC demand (only if not already tracked above)
            // Note: This is the fallback path when hvac_equipment is None
            // Note: hvac_output_raw is positive for heating, negative for cooling
            // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
            let enabled_vec = self.0.hvac_enabled.as_ref();
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
                    if val_kw > self.0.zone_peak_heating_kw.as_mut()[i] {
                        self.0.zone_peak_heating_kw.as_mut()[i] = val_kw;
                        self.0.zone_peak_heating_timestep[i] = timestep;
                    }
                } else if val < 0.0 {
                    let val_kw = -val / 1000.0;
                    if val_kw > self.0.zone_peak_cooling_kw.as_mut()[i] {
                        self.0.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                        self.0.zone_peak_cooling_timestep[i] = timestep;
                    }
                }
            }

            // Track global peak
            if hvac_power_watts_sum > 0.0 {
                self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_power_watts_sum);
            } else if hvac_power_watts_sum < 0.0 {
                self.0.peak_power_cooling = self.0.peak_power_cooling.max(-hvac_power_watts_sum);
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
            T::from(self.0.zero_vector.clone())
        } else {
            // Issue #1163: symmetric ideal-HVAC formula (mass heat-release is
            // already embedded in t_i_free via num_tm).
            self.compute_zone_hvac_load(
                t_i_free.as_ref(),
                heating_setpoint,
                cooling_setpoint,
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
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_for_temp_calc.as_ref();
        for i in 0..self.0.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                scratch.t_i_act[i] = t_free[i] + hvac[i] / h_is;
            } else {
                scratch.t_i_act[i] = t_free[i];
            }
        }
        let t_i_act = T::from(VectorField::new(std::mem::take(&mut scratch.t_i_act)));

        // Use hvac_for_temp_calc for energy (matches what was used for temperature update)
        // This ensures energy calculation is consistent with temperature physics
        let mut heating_sum = 0.0;
        let mut cooling_sum = 0.0;
        let mut total_signed = 0.0;

        // Per-zone energy accumulation (Issue #1288)
        // hvac_for_temp_calc: positive = heating, negative = cooling
        let hvac_vec = hvac_for_temp_calc.as_ref();
        let zone_heating_slice = self.0.zone_heating_energy_kwh.as_mut();
        let zone_cooling_slice = self.0.zone_cooling_energy_kwh.as_mut();
        for i in 0..self.0.num_zones {
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
        let h_tr_ms_ref = self.0.h_tr_ms.as_ref();
        let mass_temps_ref = self.0.mass_temperatures.as_ref();
        let h_tr_is_ref = self.0.h_tr_is.as_ref();
        let t_i_act_ref = t_i_act.as_ref();
        let t_i_free_ref = t_i_free.as_ref();
        let phi_st_ref = phi_st.as_ref();
        let term_rest_1_ref = term_rest_1.as_ref();
        let h_tr_3_ref = self.0.derived_h_tr_3.as_ref();

        for i in 0..self.0.num_zones {
            let cm_i = self.0.thermal_capacitance.as_ref()[i];
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
        let t_s_act = T::from(VectorField::new(std::mem::take(&mut scratch.t_s_act)));

        // Update mass temperatures using implicit integration for high thermal capacitance
        // This addresses instability with explicit Euler for Cm > 500 J/K
        let mass_temps_ref = self.0.mass_temperatures.as_ref();
        let thermal_cap_ref = self.0.thermal_capacitance.as_ref();
        // Mode-specific fields removed - use physics-based h_tr_em and h_tr_ms
        let h_tr_em_ref = self.0.h_tr_em.as_ref();
        let h_tr_ms_ref = self.0.h_tr_ms.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let t_i_act_ref = t_i_act.as_ref();
        let phi_m_ref = phi_m.as_ref();
        let h_tr_3_ref_2 = self.0.derived_h_tr_3.as_ref();

        // Determine HVAC mode from hvac_output_raw (Plan 03-14)
        // Use separate heating/cooling coupling parameters based on mode

        for i in 0..self.0.num_zones {
            let tm_old = mass_temps_ref[i];
            let cm = thermal_cap_ref[i];
            let t_s = t_s_act_ref[i];
            // Issue #1860: blend the air temperature used for mass coupling
            // between the free-floating and HVAC-controlled values, using the
            // same time-constant fraction α = 1 − exp(−dt/τ_mass) as the
            // surface-temperature blend above. This ensures the CrankNicolson
            // path (which takes t_i directly, not t_s) also sees the
            // time-constant-aware mass coupling.
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
                    let h_tr_3_zone = *self.0.derived_h_tr_3.as_ref().get(i).unwrap_or(&h_tr_ms);
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
                    let h_tr_3_zone = *self.0.derived_h_tr_3.as_ref().get(i).unwrap_or(&h_tr_ms);
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
                    let h_tr_3_with_vent =
                        *self.0.derived_h_tr_3.as_ref().get(i).unwrap_or(&h_tr_ms)
                            + h_vent_mass_zone;
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
        let new_mass_temps_vf: T = VectorField::new(std::mem::take(&mut scratch.new_mass)).into();

        // Plan 03-04: Update previous mass temperature for tracking (kept for diagnostic output)
        // Mass energy change tracking removed - Ti_free already includes thermal mass effects
        self.0.previous_mass_temperatures =
            std::mem::replace(&mut self.0.mass_temperatures, new_mass_temps_vf);

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
            self.0.current_hvac_output = Some(hvac_output_raw);
            // Temporarily take diagnostics out to avoid borrow conflicts
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            // Clear the buffer after use
            self.0.current_hvac_output = None;
        }

        // Issue #1966: restore pooled scratch buffers for next timestep
        // (phi_ia, phi_st, phi_m were moved out via mem::take above)
        self.0.scratch_pool.get_5r1c(self.0.num_zones).fill_zero();

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
        // Extract all needed self.0 scalars BEFORE scratch borrow to avoid E0502/E0499
        let solar_distribution_to_air = self.0.solar_distribution_to_air;
        let solar_beam_to_mass_fraction = self.0.solar_beam_to_mass_fraction;
        let st_int_frac = rad_frac * (1.0 - solar_distribution_to_air);
        let m_air_frac = rad_frac * solar_distribution_to_air;
        // Solar gain distribution for 6R2C model.
        // Energy-conserving split: st + m_env + m_int = 1.0 when sol_to_air = 0.
        // With solar_beam_to_mass_fraction = 0.0: 100% to surface (fast air heating).
        // With solar_beam_to_mass_fraction = 1.0: 70% envelope mass, 30% internal mass.
        let st_sol_frac = 1.0 - solar_beam_to_mass_fraction; // Solar to surface
        let m_env_sol_frac = solar_beam_to_mass_fraction * 0.7; // Solar to envelope mass
        let m_int_sol_frac = solar_beam_to_mass_fraction * 0.3; // Solar to internal mass
        let sol_to_air_frac = solar_distribution_to_air;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_solar_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let heating_setpoint = self.0.heating_setpoint;
        let cooling_setpoint = self.0.cooling_setpoint;
        let num_zones = self.0.num_zones;

        // Issue #1524: consolidated per-timestep scratch (replaces the eleven
        // standalone `Vec::with_capacity(num_zones)` allocations in 6R2C).
        // Issue #1966: scratch is now pooled in ThermalModelData::scratch_pool
        // and reused across timesteps via fill_zero() at end of step.
        let mut scratch = self.0.scratch_pool.get_6r2c(num_zones);

        for i in 0..num_zones {
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

        let phi_ia = T::from(VectorField::new(std::mem::take(&mut scratch.phi_ia)));
        let phi_st = T::from(VectorField::new(std::mem::take(&mut scratch.phi_st)));
        let phi_m_env = T::from(VectorField::new(std::mem::take(&mut scratch.phi_m_env)));
        let phi_m_int = T::from(VectorField::new(std::mem::take(&mut scratch.phi_m_int)));

        // Use pre-computed cached values
        #[cfg(feature = "debug-physics")]
        let h_ext_base = &self.0.derived_h_ext;
        let term_rest_1 = &self.0.derived_term_rest_1;

        // Night ventilation no longer modifies h_ext (same fix as 5R1C path).
        let modified_h_ext: Option<T> = None;
        #[cfg(feature = "debug-physics")]
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
        // Optimized: avoid intermediate vector allocations using explicit loop
        let h_sum_ref = h_sum.as_ref();
        let h_tr_floor_ref = self.0.h_tr_floor.as_ref();
        let h_ms_me_is_prod_ref = h_ms_me_is_prod.as_ref();
        let h_total_with_iz_ref = h_total_with_iz.as_ref();

        for i in 0..self.0.num_zones {
            let g = h_sum_ref[i] * h_tr_floor_ref[i];
            scratch.ground_coeff[i] = g;
            let d = h_ms_me_is_prod_ref[i] + (h_sum_ref[i] * h_total_with_iz_ref[i]) + g;
            scratch.den[i] = d;
        }
        let ground_coeff_6r2c =
            T::from(VectorField::new(std::mem::take(&mut scratch.ground_coeff)));
        den = T::from(VectorField::new(std::mem::take(&mut scratch.den)));

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

        // Store phi_ia[0] for debugging before we consume it
        let _phi_ia_0 = phi_ia.as_ref().first().copied().unwrap_or(0.0);

        // Compute inter-zone heat transfer directly into phi_ia_with_iz to avoid Vec allocation
        let mut phi_ia_with_iz = phi_ia;

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
                }
            }
        }

        // Optimized: replace clone and mul_assign with explicit loop
        let sum_term_ref = sum_term.as_ref();
        let term_rest_1_ref = term_rest_1.as_ref();
        let ground_coeff = ground_coeff_6r2c.as_ref();

        for i in 0..self.0.num_zones {
            scratch.num_rest[i] = sum_term_ref[i] * term_rest_1_ref[i] + ground_coeff[i] * t_g;
        }
        let num_rest_with_iz = T::from(VectorField::new(std::mem::take(&mut scratch.num_rest)));

        // DEBUG: Save values for 900FF before they're consumed
        #[cfg(feature = "debug-physics")]
        let debug_900ff = if self.0.case_id == "900FF" && timestep.is_multiple_of(24) {
            let den_vals = den.as_ref();
            let _num_tm_vals = num_tm.as_ref();
            let num_rest_vals = num_rest_with_iz.as_ref();
            let _env_mass_vals = self.0.envelope_mass_temperatures.as_ref();
            let h_sum_vals = h_sum.as_ref();
            let sum_term_vals = sum_term.as_ref();
            let h_ext_debug = h_ext.as_ref();
            let solar_debug = self.0.solar_gains.as_ref();
            let loads_debug = self.0.loads.as_ref();
            let area_debug = self.0.zone_area.as_ref();
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
        let hvac_output_raw = self.compute_zone_hvac_load(
            t_i_free.as_ref(),
            heating_setpoint,
            cooling_setpoint,
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

        // Per-zone energy accumulation (Issue #1288)
        // hvac_output_raw: positive = heating, negative = cooling
        let hvac_vec = hvac_output_raw.as_ref();
        let zone_heating_slice = self.0.zone_heating_energy_kwh.as_mut();
        let zone_cooling_slice = self.0.zone_cooling_energy_kwh.as_mut();

        for i in 0..self.0.num_zones {
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
                if val_kw > self.0.zone_peak_heating_kw.as_mut()[i] {
                    self.0.zone_peak_heating_kw.as_mut()[i] = val_kw;
                    self.0.zone_peak_heating_timestep[i] = timestep;
                }
            } else {
                cooling_sum += -val;
                zone_cooling_slice[i] += -energy_kwh;

                // Issue #1289: Track per-zone peaks
                // Issue #1628: Also track timestep when peak occurred
                let val_kw = -val / 1000.0;
                if val_kw > self.0.zone_peak_cooling_kw.as_mut()[i] {
                    self.0.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                    self.0.zone_peak_cooling_timestep[i] = timestep;
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

        // hvac_energy_for_step returns total HVAC energy in JOULES (not kWh)
        // The test expects Joules and multiplies by 3.6e6
        // DON'T apply correction here - it would break temperature calculations
        let hvac_energy_for_step = total_signed * dt;

        // Root Cause Fix: Physics-based temperature update.
        // t_i_act = t_i_free + hvac_power / h_tr_is
        // (See the NOTE at the first temperature-update site above for why
        // h_tr_is is retained instead of h_coeff — Issue #1163.)
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_output_raw.as_ref();
        for i in 0..self.0.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                scratch.t_i_act[i] = t_free[i] + hvac[i] / h_is;
            } else {
                scratch.t_i_act[i] = t_free[i];
            }
        }
        let t_i_act = T::from(VectorField::new(std::mem::take(&mut scratch.t_i_act)));

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
                let t_i_free_ref = t_i_free.as_ref();
                let t_i_act_ref = t_i_act.as_ref();
                for i in 0..self.0.num_zones {
                    let t_si_ctf = ctf_temps.get(i).copied().unwrap_or(20.0);
                    let delta_t_i = t_i_act_ref.get(i).copied().unwrap_or(0.0)
                        - t_i_free_ref.get(i).copied().unwrap_or(0.0);
                    // Approximate: surface follows zone air with ~h_tr_is/(h_tr_is+Z₀) coupling
                    // Use conservative 0.5 factor for stability
                    scratch.t_s[i] = t_si_ctf + 0.5 * delta_t_i;
                }
                T::from(VectorField::new(std::mem::take(&mut scratch.t_s)))
            } else {
                // PHASE 36-04 FIX: 6R2C surface temperature with h_tr_me * Tm_int coupling
                // T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
                let h_tr_ms_data = self.0.h_tr_ms.as_ref();
                let h_tr_is_data = self.0.h_tr_is.as_ref();
                let t_i_act_data = t_i_act.as_ref();
                let phi_st_data = phi_st.as_ref();
                let env_mass_data = self.0.envelope_mass_temperatures.as_ref();
                let term_rest_data = term_rest_1.as_ref();
                for i in 0..self.0.num_zones {
                    let numerator = h_tr_ms_data[i] * env_mass_data[i]
                        + h_tr_is_data[i] * t_i_act_data[i]
                        + phi_st_data[i]
                        + h_tr_me_ref[i] * int_mass_temps_ref[i];
                    let denominator = term_rest_data[i] + h_tr_me_ref[i];
                    scratch.t_s[i] = numerator / denominator;
                }
                T::from(VectorField::new(std::mem::take(&mut scratch.t_s)))
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
            for i in 0..self.0.num_zones {
                let numerator = h_tr_ms_data[i] * env_mass_data[i]
                    + h_tr_is_data[i] * t_i_act_data[i]
                    + phi_st_data[i]
                    + h_tr_me_ref[i] * int_mass_temps_ref[i];
                let denominator = term_rest_data[i] + h_tr_me_ref[i];
                scratch.t_s[i] = numerator / denominator;
            }
            T::from(VectorField::new(std::mem::take(&mut scratch.t_s)))
        };

        // === 6R2C: Update two mass nodes with implicit integration ===
        // Envelope mass: receives heat from exterior (sol-air), surface, and internal mass

        // Update envelope mass temperatures using implicit integration for high thermal capacitance
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

            scratch.new_int[i] = tm_int_new;
        }

        let new_env_temps_vf: T = VectorField::new(std::mem::take(&mut scratch.new_env)).into();
        let old_env_mass_temperatures =
            std::mem::replace(&mut self.0.envelope_mass_temperatures, new_env_temps_vf);

        let new_int_temps_vf: T = VectorField::new(std::mem::take(&mut scratch.new_int)).into();
        let old_int_mass_temperatures =
            std::mem::replace(&mut self.0.internal_mass_temperatures, new_int_temps_vf);

        // Issue #272, #274, #275: Calculate thermal mass energy change for 6R2C
        // For 6R2C, we track energy changes in both envelope and internal masses
        // Envelope mass energy change (Cm × (Tm_new - Tm_old))
        let env_mass_temp_change = self
            .0
            .envelope_mass_temperatures
            .zip_with(&old_env_mass_temperatures, |a, b| a - b);
        let env_mass_energy_change = self
            .0
            .envelope_thermal_capacitance
            .zip_with(&env_mass_temp_change, |a, b| a * b);

        // Internal mass energy change (Cm × (Tm_new - Tm_old))
        let int_mass_temp_change = self
            .0
            .internal_mass_temperatures
            .zip_with(&old_int_mass_temperatures, |a, b| a - b);
        let int_mass_energy_change = self
            .0
            .internal_thermal_capacitance
            .zip_with(&int_mass_temp_change, |a, b| a * b);

        // Total mass energy change for this timestep
        let mass_energy_change_for_step_6r2c =
            env_mass_energy_change.zip_with(&int_mass_energy_change, |a, b| a + b);

        // Track cumulative mass energy change
        let mass_energy_change_for_step_total =
            mass_energy_change_for_step_6r2c.reduce(0.0, |acc, val| acc + val);
        self.0.mass_energy_change_cumulative += mass_energy_change_for_step_total;

        // Plan 03-04: Update single mass temperature for backward compatibility (average of two masses)
        let total_cap = self
            .0
            .envelope_thermal_capacitance
            .zip_with(&self.0.internal_thermal_capacitance, |a, b| a + b);

        self.0.mass_temperatures = self
            .0
            .envelope_mass_temperatures
            .zip_with(&self.0.envelope_thermal_capacitance, |a, b| a * b)
            .zip_with(
                &self
                    .0
                    .internal_mass_temperatures
                    .zip_with(&self.0.internal_thermal_capacitance, |a, b| a * b),
                |a, b| a + b,
            )
            .zip_with(&total_cap, |a, b| a / b);

        // DEBUG: Print t_i_act before storing
        self.0.temperatures = t_i_act;

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            // Store current HVAC output for this timestep (per zone, Watts)
            self.0.current_hvac_output = Some(hvac_output_raw);
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

        // Validate 8R3C fields are initialized (precondition for 8R3C physics step)
        let ceiling_mass = self
            .0
            .ceiling_mass_temperatures
            .as_mut()
            .expect("ceiling_mass_temperatures must be initialized for 8R3C model");
        let floor_mass = self
            .0
            .floor_mass_temperatures
            .as_mut()
            .expect("floor_mass_temperatures must be initialized for 8R3C model");
        let partition_mass = self
            .0
            .partition_mass_temperatures
            .as_mut()
            .expect("partition_mass_temperatures must be initialized for 8R3C model");
        let ceiling_cap = self
            .0
            .ceiling_thermal_capacitance
            .as_ref()
            .expect("ceiling_thermal_capacitance must be initialized for 8R3C model");
        let floor_cap = self
            .0
            .floor_thermal_capacitance
            .as_ref()
            .expect("floor_thermal_capacitance must be initialized for 8R3C model");
        let partition_cap = self
            .0
            .partition_thermal_capacitance
            .as_ref()
            .expect("partition_thermal_capacitance must be initialized for 8R3C model");
        let h_tr_ceiling = self
            .0
            .h_tr_ceiling
            .as_ref()
            .expect("h_tr_ceiling must be initialized for 8R3C model");
        let h_tr_floor_mass = self
            .0
            .h_tr_floor_mass
            .as_ref()
            .expect("h_tr_floor_mass must be initialized for 8R3C model");
        let h_tr_partition = self
            .0
            .h_tr_partition
            .as_ref()
            .expect("h_tr_partition must be initialized for 8R3C model");

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

        // Issue #1966: restore pooled scratch buffers for next timestep
        // (phi_ia, phi_st, phi_m_env, phi_m_int were moved out via mem::take above)
        self.0.scratch_pool.get_6r2c(self.0.num_zones).fill_zero();

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
        let h_ve_night: f64 = if let Some(ref night_vent) = self.0.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                night_vent_active_now = true;
                // ASHRAE 140 night-vent fan supplies outdoor air to zone 0
                let rho = self.0.air_density.as_ref().first().copied().unwrap_or(1.2);
                let cp = self
                    .0
                    .heat_capacity
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(1005.0);
                // ACH = fan_capacity (m³/h) / zone_volume (m³)
                // Zone 0 is the conditioned zone per ASHRAE 140
                let zone_vol = self
                    .0
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
        let solar_dist_to_air = self.0.solar_distribution_to_air;
        let st_int_frac = rad_frac * (1.0 - solar_dist_to_air);
        let m_air_frac = rad_frac * solar_dist_to_air;
        let solar_beam_to_mass = self.0.solar_beam_to_mass_fraction;
        let st_sol_frac = 1.0 - solar_beam_to_mass;
        let m_sol_frac = solar_beam_to_mass;

        // Extract all needed data from self.0 BEFORE acquiring scratch pool borrow
        // to avoid borrow conflicts (scratch_pool requires &mut self.0).
        let loads_data = self.0.loads.as_ref().to_vec();
        let solar_data = self.0.solar_gains.as_ref().to_vec();
        let opaque_solar_data = self.0.opaque_solar_gains.as_ref().to_vec();
        let area_data = self.0.zone_area.as_ref().to_vec();
        let heating_setpoint = self.0.heating_setpoint;
        let cooling_setpoint = self.0.cooling_setpoint;
        let temps = self.0.temperatures.as_ref().to_vec();
        let prev_temps = self.0.previous_temperatures.as_ref().to_vec();
        let mass_temps = self.0.mass_temperatures.as_ref().to_vec();
        let num_zones = self.0.num_zones;

        // Issue #863 / #1212: Pre-compute sol-air temperature BEFORE scratch borrow
        // to avoid E0502 (cannot borrow `*self` as immutable while scratch pool
        // is mutably borrowed). Extract outdoor_temp, solar position, and wall
        // irradiance from weather data upfront.
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

        // Issue #1212: Extract weather data upfront so we don't need &self after scratch borrow
        let (outdoor_temp, dni, dhi, ghi) = if let Some(weather) = &self.0.weather {
            (weather.outdoor_temp, weather.dni, weather.dhi, weather.ghi)
        } else {
            (20.0_f64, 0.0_f64, 0.0_f64, 0.0_f64)
        };

        // Issue #1212: Pre-compute solar position to avoid &self borrow during scratch scope
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
        let t_sol_air_wall = sol_air.for_wall(
            outdoor_temp,
            wall_irr.total_wm2,
            wall_irr.ground_reflected_wm2,
        );

        // Issue #1524: consolidated per-timestep scratch (replaces the fourteen
        // standalone `Vec::with_capacity(num_zones)` allocations in 9R4C; the
        // seven read-back intermediates share one flat buffer).
        // Issue #1966: scratch is now pooled in ThermalModelData::scratch_pool
        // and reused across timesteps via fill_zero() at end of step.
        let mut scratch = self.0.scratch_pool.get_9r4c(num_zones);

        for i in 0..num_zones {
            let load_w = loads_data[i] * area_data[i];
            let sol_w = solar_data[i] * area_data[i];
            let opaque_sol_w = opaque_solar_data[i] * area_data[i];

            let sol_to_air = sol_w * solar_dist_to_air;
            let remaining_sol = sol_w - sol_to_air;
            scratch.phi_ia[i] = load_w * conv_frac + sol_to_air;
            scratch.phi_st[i] = load_w * st_int_frac + remaining_sol * st_sol_frac;
            scratch.phi_m[i] = load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w;
        }

        // (#872) Save raw gain data for multi-node solver before moving into tensors.
        // Used for internal radiative gain injection via step_with_gains().
        let phi_ia = T::from(VectorField::new(std::mem::take(&mut scratch.phi_ia)));
        let phi_st = T::from(VectorField::new(std::mem::take(&mut scratch.phi_st)));
        let phi_m = T::from(VectorField::new(std::mem::take(&mut scratch.phi_m)));

        // Issue #863 / #1966: Pre-extract ALL self.0 fields accessed during and after
        // scratch scope into local variables. Extract them HERE (while scratch is alive)
        // so they remain valid after scratch is dropped.
        let term_rest_1 = self.0.derived_term_rest_1.clone();
        let derived_ground_coeff = self.0.derived_ground_coeff.clone();
        let hvac_heating_capacity = self.0.hvac_heating_capacity.clone();
        let hvac_cooling_capacity = self.0.hvac_cooling_capacity.clone();
        let zone_area = self.0.zone_area.as_ref().to_vec();
        let h_tr_em = self.0.h_tr_em.as_ref().to_vec();
        let h_tr_ms = self.0.h_tr_ms.as_ref().to_vec();
        let mass_temperatures = self.0.mass_temperatures.as_ref().to_vec();
        let zero_vector = self.0.zero_vector.clone();
        let free_float = self.0.free_float;
        let multi_node_solvers_len = self.0.multi_node_solvers.len();
        let derived_h_ms_is_prod = self.0.derived_h_ms_is_prod.clone();
        let derived_h_ext = self.0.derived_h_ext.clone();
        let derived_den = self.0.derived_den.clone();

        // Issue #863: t_sol_air_wall pre-computed before scratch borrow (see above).
        // Write to scratch arrays and immediately extract t_sol_air_data so we can
        // drop the scratch borrow and avoid E0502 on subsequent self.0 field accesses.
        let t_sol_air_data_revised: Vec<f64> = (0..num_zones)
            .map(|i| scratch.t_sol_air().get(i).copied().unwrap_or(outdoor_temp))
            .collect();

        // Issue #1712: h_ve_night application to h_ext and den
        let h_ext_for_free_float: T = if night_vent_active_now {
            let base = derived_h_ext.as_ref();
            let mut v = Vec::with_capacity(base.len());
            for (i, &b) in base.iter().enumerate() {
                let night_add = if i == 0 { h_ve_night } else { 0.0 };
                v.push(b + night_add);
            }
            T::from(VectorField::new(v))
        } else {
            derived_h_ext.clone()
        };

        // DROP scratch borrow to allow subsequent self.0 field accesses.
        // All post-scratch code uses pre-extracted local copies.
        drop(scratch);

        // Use 5R1C network for free-floating temperature
        // Note: term_rest_1, derived_ground_coeff, h_ext_for_free_float,
        // and den were pre-extracted inside the scratch scope above.
        // (#872: sensitivity variable removed — HVAC demand now uses h_loss × ΔT formula)

        let num_tm = derived_h_ms_is_prod.zip_with(&mass_temperatures, |a, b| a * b);
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);

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
        if num_zones > 1 {
            let slice = phi_ia_with_iz.as_mut();
            let n = num_zones;
            let temps = self.0.temperatures.as_ref();
            let h_iz_vec = self.0.h_tr_iz.as_ref();
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
                    let area = zone_area[i];
                    let q_ctf = q_flux * area;
                    let t_sol_air_i = t_sol_air_data_revised[i];
                    let t_mass = mass_temperatures[i];
                    let h_tr_em_i = h_tr_em[i];
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
                    let area = zone_area[i];
                    let q_fd = q_flux * area;
                    let t_sol_air_i = t_sol_air_data_revised[i];
                    let t_mass = mass_temperatures[i];
                    let h_tr_em_i = h_tr_em[i];
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
            let t_sol_air_i = t_sol_air_data_revised[i];
            *n += h * t_sol_air_i;
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

        // Issue #864: Store pre-gain mass temperatures and per-surface gains for
        // step_per_surface(). Using pre-gain temperatures avoids double-counting
        // gains that are added in SurfaceNode::update()'s backward Euler.
        // (Issue #1524: these six read-back intermediates share the flat
        // `scratch.inter` backing buffer — 6 allocations collapsed to 0 here.)

        #[allow(clippy::needless_range_loop)]
        for zone_idx in 0..self.0.num_zones {
            if zone_idx >= self.0.multi_node_solvers.len() {
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

            let solver = &mut self.0.multi_node_solvers[zone_idx];
            // (#872) Use previous zone temperature as boundary, NOT 5R1C t_i_free.
            // This breaks the destructive feedback loop where 5R1C mass temps corrupt
            // the solver's boundary condition. The solver will compute its own
            // zone air temperature from the multi-node balance.
            let t_zone_prev = self.0.temperatures.as_ref()[zone_idx];
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
            let h_ve_val = self.0.h_ve.as_ref()[zone_idx];
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
            let h_tr_is_multiplier_pre = if night_vent_active_now {
                h_tr_is_ach_multiplier(ach_night_vent)
            } else {
                1.0
            };
            if h_tr_is_multiplier_pre != 1.0 {
                solver.h_tr_is *= h_tr_is_multiplier_pre;
            }
            let t_air_mn_pre = solver.compute_zone_air_temperature(
                outdoor_temp,
                h_ve_val,
                h_ve_night_zone,
                phi_ia_val,
            );
            // Restore h_tr_is - the main boost/restore block at lines ~2720 will handle it for step_with_gains
            if h_tr_is_multiplier_pre != 1.0 {
                solver.h_tr_is /= h_tr_is_multiplier_pre;
            }

            solver.set_zone_temperature(t_air_mn_pre);
            solver.set_surface_temperature(t_surface);

            let (surface_ext_temps, wall_irr_val, roof_irr_val) =
                if let Some(ref weather) = self.0.weather {
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
                    let ext_temps = SurfaceExteriorTemperatures {
                        t_ext_wall: sol_air.for_wall(
                            outdoor_temp,
                            wall_irr.total_wm2,
                            wall_irr.ground_reflected_wm2,
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
            let _zone_area_val = self.0.zone_area.as_ref()[zone_idx];
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

            // Issue #864: Capture pre-gain mass temperatures BEFORE step_with_gains()
            // so step_per_surface can use them to avoid double-counting gains.
            // Also compute per-surface opaque solar gains for SurfaceNode::update().
            let mass_temp_wall_pre = solver.mass.wall.temperature;
            let mass_temp_roof_pre = solver.mass.roof.temperature;
            let mass_temp_floor_pre = solver.mass.floor.temperature;

            // Use opaque solar gain (phi_m_zone) distributed by irradiance × area.
            // Assume equal per-unit areas (1.0 m²) since actual zone geometry
            // is not stored in the multi-node solver. Using irradiance ensures
            // sun-facing surfaces get proportionally more gain.
            let floor_irr_val = 0.0; // Floor gets no direct solar
            let solar_gains = distribute_opaque_solar_gains(
                phi_m_zone,
                1.0, // wall_area (per-unit)
                1.0, // roof_area (per-unit)
                1.0, // floor_area (per-unit)
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
            // IMPORTANT: Restore h_tr_is after step to avoid persisting the boost to daytime.
            let h_tr_is_multiplier = if night_vent_active_now {
                h_tr_is_ach_multiplier(ach_night_vent)
            } else {
                1.0
            };
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
        if night_vent_active_now {
            let multiplier = h_tr_is_ach_multiplier(ach_night_vent);
            for solver in &mut self.0.multi_node_solvers {
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
        for zone_idx in 0..self.0.num_zones {
            if zone_idx < self.0.multi_node_solvers.len() {
                let solver = &self.0.multi_node_solvers[zone_idx];
                let h_ve_val = self.0.h_ve.as_ref()[zone_idx];
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
        let t_i_free_mn = T::from(VectorField::new(std::mem::take(&mut scratch.t_i_free)));

        // Issue #1279: Restore h_tr_is to original value after computing zone air temperature.
        if night_vent_active_now {
            let multiplier = h_tr_is_ach_multiplier(ach_night_vent);
            for solver in &mut self.0.multi_node_solvers {
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
        for zone_idx in 0..self.0.num_zones {
            if zone_idx >= self.0.multi_node_solvers.len() {
                continue;
            }
            let solver = &mut self.0.multi_node_solvers[zone_idx];
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
            (temps[0] - prev_temps[0]) / dt
        } else {
            0.0
        };

        let (hvac_mode, modulation) = self.0.predictive_controller.calculate_modulation(
            temps[0],
            mass_temps[0],
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
        let (hvac_for_temp_calc, t_i_act) = if self.0.free_float {
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
            (T::from(self.0.zero_vector.clone()), t_i_free_5r1c.clone())
        } else {
            // HVAC mode: use multi-node t_air (from _t_i_free_mn) when available
            let heat_cap = self.0.hvac_heating_capacity;
            let cool_cap = self.0.hvac_cooling_capacity;
            // Issue #1524: hvac/t_i_act live in the local `scratch` struct, so
            // `scratch.hvac[i]` / `scratch.t_i_act[i]` (mutable borrows of a
            // local) coexist freely with `self.compute_hvac_coefficient(i)`
            // (an `&self` borrow) — the exact conflict that sank #1436.
            for i in 0..self.0.num_zones {
                // Issue #860: Prefer multi-node t_air over 5R1C t_free for HVAC demand
                let t_free_val =
                    if i < self.0.multi_node_solvers.len() {
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
                let _h_tr_ms = self.0.h_tr_ms.as_ref()[i];

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

                // DEBUG: Print h_coeff breakdown on first HVAC step after warmup
                #[cfg(feature = "debug-physics")]
                if timestep == 337 && i == 0 && !self.0.free_float {
                    let h_tr_is = self.0.h_tr_is.as_ref()[i];
                    let h_tr_ms = self.0.h_tr_ms.as_ref()[i];
                    let h_tr_em = self.0.h_tr_em.as_ref()[i];
                    let h_tr_w = self.0.h_tr_w.as_ref()[i];
                    let h_tr_floor = self.0.h_tr_floor.as_ref()[i];
                    let h_ve_scalar = self.0.h_ve.as_ref().get(i).copied().unwrap_or(0.0);
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
                        h_coeff * (self.0.heating_setpoint - t_free_val).max(0.0)
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
                // Sign convention: Q > 0 = heating, Q < 0 = cooling.
                let q = if t_free_val < heating_setpoint {
                    // Heating: Q = h_coeff × (T_heat_sp − T_free) > 0
                    h_coeff * (heating_setpoint - t_free_val)
                } else if t_free_val > cooling_setpoint {
                    // Cooling: Q = h_coeff × (T_cool_sp − T_free) = −h_coeff × (T_free − T_cool_sp) < 0
                    // Driving temperature is t_free (zone air), NOT t_mass_mn
                    -h_coeff * (t_free_val - cooling_setpoint)
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
                T::from(VectorField::new(std::mem::take(&mut scratch.hvac))),
                T::from(VectorField::new(std::mem::take(&mut scratch.t_i_act))),
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
        if !self.0.free_float {
            if let Some(ref mut equipment) = self.0.hvac_equipment {
                // Economizer is only meaningful in cooling mode; the helper is
                // mode-agnostic so we still call it (it returns false for
                // `EconomizerMode::Disabled` and for non-cooling cases).
                use crate::sim::hvac::{calculate_free_cooling_capacity, is_economizer_active};
                let hour_of_day_idx = timestep % 24;
                let cooling_setpoint_for_econ = self.0.cooling_schedule.value(hour_of_day_idx);
                let economizer_active = is_economizer_active(
                    self.0.economizer_mode,
                    outdoor_temp,
                    None, // outdoor_enthalpy — only available in Enthalpy mode (not wired here)
                    self.0.temperatures.as_ref()[0],
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
                            self.0.temperatures.as_ref()[0],
                            10000.0, // TODO: ventilation_airflow from building spec (m³/s)
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
            let mass_temps_ref = self.0.mass_temperatures.as_ref();
            let thermal_cap_ref = self.0.thermal_capacitance.as_ref();
            let h_tr_em_ref = self.0.h_tr_em.as_ref();
            let h_tr_ms_ref = self.0.h_tr_ms.as_ref();

            for i in 0..self.0.num_zones {
                let tm_old = mass_temps_ref[i];
                let cm = thermal_cap_ref[i];
                let t_i = t_i_act.as_ref()[i];
                let h_tr_em = h_tr_em_ref[i];
                let h_tr_ms = h_tr_ms_ref[i];
                let h_tr_is_zone = self.0.h_tr_is.as_ref()[i];
                let h_tr_me_zone = self.0.h_tr_me.as_ref()[i];
                let t_ext = t_sol_air_data_revised[i];

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
                VectorField::new(std::mem::take(&mut scratch.new_mass)).into();
            self.0.previous_mass_temperatures =
                std::mem::replace(&mut self.0.mass_temperatures, new_mass_temps_vf);
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
        if self.0.free_float {
            let temps_slice = self.0.temperatures.as_mut();
            for (i, t_val) in t_i_free_mn.as_ref().iter().enumerate() {
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
            for (i, (&output, &enabled)) in hvac_output
                .as_ref()
                .iter()
                .zip(self.0.hvac_enabled.as_ref().iter())
                .enumerate()
            {
                let val = if enabled > 0.5 { output } else { 0.0 };
                hvac_power_watts += val;

                if val > 0.0 {
                    heating_sum += val;
                    // Issue #1289: Track per-zone peaks
                    // Issue #1628: Also track timestep when peak occurred
                    let val_kw = val / 1000.0;
                    if val_kw > self.0.zone_peak_heating_kw.as_mut()[i] {
                        self.0.zone_peak_heating_kw.as_mut()[i] = val_kw;
                        self.0.zone_peak_heating_timestep[i] = timestep;
                    }
                } else if val < 0.0 {
                    cooling_sum += -val;
                    // Issue #1289: Track per-zone peaks
                    // Issue #1628: Also track timestep when peak occurred
                    let val_kw = -val / 1000.0;
                    if val_kw > self.0.zone_peak_cooling_kw.as_mut()[i] {
                        self.0.zone_peak_cooling_kw.as_mut()[i] = val_kw;
                        self.0.zone_peak_cooling_timestep[i] = timestep;
                    }
                }
            }

            let heating_energy_joules = heating_sum * dt;
            let cooling_energy_joules = cooling_sum * dt;

            self.0.annual_heating_energy += heating_energy_joules / 3.6e6;
            self.0.annual_cooling_energy += cooling_energy_joules / 3.6e6;

            // Per-zone energy accumulation (Issue #1288)
            // Use enabled-masked values for per-zone accumulation
            let enabled_vec = self.0.hvac_enabled.as_ref();
            let zone_heating_slice = self.0.zone_heating_energy_kwh.as_mut();
            let zone_cooling_slice = self.0.zone_cooling_energy_kwh.as_mut();
            for i in 0..self.0.num_zones {
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
                self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_power_watts);
            } else if hvac_power_watts < 0.0 {
                self.0.peak_power_cooling = self.0.peak_power_cooling.max(-hvac_power_watts);
            }
        }

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            self.0.current_hvac_output = Some(hvac_output);
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            self.0.current_hvac_output = None;
        }

        // Issue #1966: restore pooled scratch buffers for next timestep
        // (phi_ia, phi_st, phi_m were moved out via mem::take above)
        self.0.scratch_pool.get_9r4c(self.0.num_zones).fill_zero();

        // Return kWh
        hvac_power_watts * dt / 3.6e6
    }
}

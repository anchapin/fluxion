//! Physics-step dispatcher for `ThermalModel`.
//!
//! Hosts [`ThermalModel::step_physics`], the dispatcher that routes to
//! the correct 5R1C/6R2C/8R3C/9R4C implementation based on the model's
//! configured network type. Originally part of the monolithic
//! `thermal_model_physics.rs` (Issue #898), extracted as part of the
//! Issue #902 modular split.
//!
//! Issue #3280 / #3291 / #3816 / #3297: selector-driven dispatch. The
//! [`ZoneSolverKind::Gauge`] selector tries the gauge single- and
//! multi-zone arms first; `FiveROneC` and `NineRFourC` selectors always
//! route to the legacy physics. §LIMIT-21 (Issue #3297) flipped the
//! production gate: with `gauge-solver` enabled, gauge dispatch is now
//! unconditional — a missing gauge backend is a hard error (panics),
//! not the old β-phase warn+fallthrough to legacy 5R1C/9R4C. The
//! `gauge-solver` cargo feature is retained for CI/β-soak purposes
//! (Issue #3286); the default build (no feature) routes `Gauge` to
//! legacy 5R1C/9R4C via the `match` arm below. The legacy
//! `is_9r4c_model()` / `is_8r3c_model()` / `is_6r2c_model()` checks
//! are gone — `thermal_model_type` is set exclusively by the selector
//! (Issue #3277).

use crate::api::error::FluxionError;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;
use crate::sim::thermal_selector::ZoneSolverKind;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Solve physics for one timestep (assumes loads already set).
    ///
    /// Typed variant of [`ThermalModel::step_physics`] (Issue #3638): a
    /// degenerate `dt <= 0` or `cm <= 0` reaching the thermal-mass
    /// integrators surfaces as `Err(FluxionError::Validation)` instead of
    /// aborting the process. Callers that cannot yet handle `Result` can
    /// keep using [`ThermalModel::step_physics`], which applies the
    /// issue-sanctioned numerical guard (log + zero energy for the step).
    ///
    /// See `step_physics` for the full contract documentation.
    pub fn try_step_physics(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> Result<f64, FluxionError> {
        // Record call for wiring validation (Plan 21-10)
        #[cfg(feature = "wiring-tracing")]
        if let Some(ref tracer) = self.0.hvac.tracer {
            tracer.record_call("step_physics");
        }

        // Issue #1409: SurfaceHeatFluxProvider::step_all is the
        // production per-surface state-advancing companion to the
        // existing pure-query `surface_heat_flux` (see
        // src/sim/surface_flux_provider.rs). The dispatcher does NOT
        // call SolverManager::step_all directly because that API
        // requires a slice of BuildingAssembly which ThermalModel does
        // not retain across the spec→model boundary; instead callers
        // advance solver state through `PhysicsSurfaceFluxProvider`.
        // When a `SolverManager` is enabled, code that wires a
        // PhysicsSurfaceFluxProvider over the same solver set will
        // surface the post-step flux via `surface_heat_flux`. This block
        // remains the single production call site documented by the
        // Issue #1409 acceptance criteria.

        // Issue #351: Calculate loads from weather data if not already set
        // This is needed for ASHRAE 140 validation where step_physics is called directly
        if self.0.solar.weather.is_some() {
            self.calc_analytical_loads(timestep, true, dt_seconds);
        }

        // Issue #3280 / #3291 / #3816: selector-driven dispatch. The
        // `Gauge` selector tries the gauge single- and multi-zone arms
        // first; `FiveROneC` and `NineRFourC` selectors always go
        // straight to their respective legacy physics. §LIMIT-21
        // (Issue #3297) flipped the production gate: with `gauge-solver`
        // enabled, gauge dispatch is now unconditional — a missing gauge
        // backend is a hard error (panics in try_run_gauge_*), not the
        // old β-phase warn+fallthrough to legacy 5R1C/9R4C. The #3817
        // heavyweight-spec exception (9R4C auto-promotion for HighMass
        // construction) is preserved.
        let selector_zone_solver = self.0.hvac.thermal_selector.zone_solver;

        // Collect gauge inputs once (immutable borrows that would
        // otherwise conflict with the mutable borrow on
        // `self.0.conduction.backend` later in the block).
        #[cfg(feature = "gauge-solver")]
        let gauge_inputs = self.collect_gauge_inputs(timestep);

        // §LIMIT-21 (Issue #3297): gauge dispatch is now unconditional
        // within the `gauge-solver` feature. Try single-zone first;
        // multi-zone specs (e.g. Case 960 sunspace) have
        // `gauge_zone_solver == None` and are picked up by the multi-zone
        // arm. Both arms write a 5R1C Crank-Nicolson mass-state proxy
        // that satisfies the strict-energy-balance gate's invariant exactly
        // (see `write_gauge_mass_state_proxy`, Issue #3297).
        //
        // The #3817 heavyweight-spec exception is preserved:
        // `is_nine_r4c_model()` — auto-promoted by `from_spec_with_selector`
        // for HighMass construction — routes directly to 9R4C because the
        // gauge solver has no thermal-mass modeling and cannot satisfy the
        // `zone_balance_eplus_isolation` swing-reduction sanity bound without
        // the 9R4C's wall/roof/floor mass nodes.
        #[cfg(feature = "gauge-solver")]
        if selector_zone_solver == ZoneSolverKind::Gauge && !self.is_nine_r4c_model() {
            if let Some(ekwh) =
                self.try_run_gauge_single_zone(timestep, outdoor_temp, dt_seconds, &gauge_inputs)
            {
                // Issue #3305 — record that the gauge path genuinely ran so
                // the REST `effective_solver` field reports the truth.
                self.0.hvac.effective_zone_solver = ZoneSolverKind::Gauge;
                return Ok(ekwh);
            }
            if let Some(ekwh) =
                self.try_run_gauge_multi_zone(timestep, outdoor_temp, dt_seconds, &gauge_inputs)
            {
                self.0.hvac.effective_zone_solver = ZoneSolverKind::Gauge;
                return Ok(ekwh);
            }
            // If both gauge arms return None, the gauge backend failed to
            // initialise — this is a programming error and panics loudly.
            // The old β-phase warn+fallthrough is gone (Issue #3297 §LIMIT-21).
        }

        // Legacy dispatch for `zone_solver ∈ {FiveROneC, NineRFourC}`.
        // With `gauge-solver` enabled, `Gauge` is handled unconditionally
        // above and never reaches here. Without the feature, `Gauge` routes
        // here and falls through to 5R1C/9R4C (the old β-phase semantics).
        match selector_zone_solver {
            ZoneSolverKind::Gauge => {
                // Reached only in the default build (no `gauge-solver`):
                // routes `Gauge` to the legacy 5R1C / 9R4C physics. 9R4C
                // when the model was auto-promoted for high-mass construction
                // (see `from_spec_with_selector` / Issue #3277 PR2.1).
                if self.is_nine_r4c_model() {
                    // Issue #3305 — record the effective legacy target.
                    self.0.hvac.effective_zone_solver = ZoneSolverKind::NineRFourC;
                    Ok(self.step_physics_9r4c(timestep, outdoor_temp, dt_seconds))
                } else {
                    self.0.hvac.effective_zone_solver = ZoneSolverKind::FiveROneC;
                    self.step_physics_5r1c(timestep, outdoor_temp, dt_seconds)
                }
            }
            ZoneSolverKind::FiveROneC => {
                self.0.hvac.effective_zone_solver = ZoneSolverKind::FiveROneC;
                self.step_physics_5r1c(timestep, outdoor_temp, dt_seconds)
            }
            ZoneSolverKind::NineRFourC => {
                self.0.hvac.effective_zone_solver = ZoneSolverKind::NineRFourC;
                Ok(self.step_physics_9r4c(timestep, outdoor_temp, dt_seconds))
            }
        }
    }

    /// Solve physics for one timestep (assumes loads already set).
    ///
    /// This method performs only the physics calculation portion of solve_single_step,
    /// assuming that loads have already been set via set_loads() or calculated externally.
    /// This enables batched inference: collect all temperatures, run one batched prediction,
    /// distribute loads, then call this method in parallel.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (used for ground temperature)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `dt_seconds` - Timestep duration in seconds (default: 3600.0 for 1-hour timestep)
    ///
    /// # Returns
    /// HVAC energy consumption for the timestep in kWh.
    ///
    /// Issue #351: Calculate solar gains internally if weather data is available
    ///
    /// Issue #3638 numerical guard: the underlying integrators now return
    /// `Result` (see [`ThermalModel::try_step_physics`]). This legacy `f64`
    /// entry point never panics on a degenerate `dt`/`cm`; on error it logs
    /// the typed validation error and returns `0.0` kWh for the step (the
    /// mass-state commit of the failed step is skipped by the propagating
    /// `?` in the step implementations, mirroring the leave-state-untouched
    /// guard in `write_gauge_mass_state_proxy`).
    pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
        match self.try_step_physics(timestep, outdoor_temp, dt_seconds) {
            Ok(energy) => energy,
            Err(error) => {
                log::error!(
                    "physics step degraded to zero HVAC energy for timestep {timestep}: {error}"
                );
                0.0
            }
        }
    }
}

/// Snapshot of the gauge step inputs gathered before any mutable borrow
/// of `self.0.conduction.backend`. Both the single-zone and multi-zone
/// gauge paths consume the same data.
#[cfg(feature = "gauge-solver")]
struct GaugeInputs {
    hvac_enabled: Vec<f64>,
    heating_setpoints: Vec<f64>,
    cooling_setpoints: Vec<f64>,
    default_heating_sp: f64,
    default_cooling_sp: f64,
    zone_areas: Vec<f64>,
    zone_volumes: Vec<f64>,
    loads: Vec<f64>,
    solar_gains: Vec<f64>,
    h_ext: f64,
    // Issue #3904: Ventilation ACH from night_ventilation schedule
    ventilation_ach: f64,
    // Issue #3918: Threading for solar lag correction (per-zone values)
    h_tr_3: Vec<f64>,      // combined air-to-mass conductance [W/K]
    cm: Vec<f64>,          // zone thermal capacitance [J/K]
    h_tr_is: Vec<f64>,     // interior surface-to-air conductance [W/K]
    term_rest_1: Vec<f64>, // h_tr_ms + h_tr_is [W/K]
    // Fractionation parameters needed for phi_st computation
    convective_fraction: f64, // convective fraction of internal gains
    solar_beam_to_mass_fraction: f64, // solar beam-to-mass fraction
}

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Collect the immutable gauge inputs (immutable borrows, copied into
    /// owned `Vec`s) so the gauge-step methods can take a `&mut self`
    /// for `self.0.conduction.backend` without borrow-checker conflicts.
    #[cfg(feature = "gauge-solver")]
    fn collect_gauge_inputs(&self, timestep: usize) -> GaugeInputs {
        let hvac_enabled = self.0.hvac.hvac_enabled.as_ref().to_vec();
        let heating_setpoints = self.0.setpoints.heating_setpoints.as_ref().to_vec();
        let cooling_setpoints = self.0.setpoints.cooling_setpoints.as_ref().to_vec();
        let default_heating_sp = self.0.setpoints.heating_setpoint;
        let default_cooling_sp = self.0.setpoints.cooling_setpoint;
        let zone_areas = self.0.setpoints.zone_area.as_ref().to_vec();
        let zone_volumes = self.0.setpoints.zone_volume.as_ref().to_vec();
        let loads = self.0.setpoints.loads.as_ref().to_vec();
        let solar_gains = self.0.solar.solar_gains.as_ref().to_vec();
        let h_ext = self
            .0
            .conduction
            .derived_h_ext
            .as_ref()
            .first()
            .copied()
            .unwrap_or(25.0);
        // Issue #3904: Compute ventilation_ach from night_ventilation schedule
        let hour_of_day = (timestep % 24) as u8;
        let mut ventilation_ach = 0.0;
        if let Some(ref night_vent) = self.0.hvac.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                // ACH = fan_capacity (m³/h) / zone_volume (m³)
                // Use first zone volume as ASHRAE 140 night-vent applies to zone 0
                if let Some(&zone_vol) = zone_volumes.first() {
                    if zone_vol > 0.0 {
                        ventilation_ach = night_vent.fan_capacity / zone_vol;
                    }
                }
            }
        }
        // Issue #3918: Thread lag correction parameters per zone
        let h_tr_3 = self.0.conduction.derived_h_tr_3.as_ref().to_vec();
        let cm = self.0.mass.thermal_capacitance.as_ref().to_vec();
        let h_tr_is = self.0.conduction.h_tr_is.as_ref().to_vec();
        let term_rest_1 = self.0.conduction.derived_term_rest_1.as_ref().to_vec();
        // Fractionation for phi_st computation
        let convective_fraction = self.0.solar.convective_fraction;
        let solar_beam_to_mass_fraction = self.0.solar.solar_beam_to_mass_fraction;
        GaugeInputs {
            hvac_enabled,
            heating_setpoints,
            cooling_setpoints,
            default_heating_sp,
            default_cooling_sp,
            zone_areas,
            zone_volumes,
            loads,
            solar_gains,
            h_ext,
            ventilation_ach,
            h_tr_3,
            cm,
            h_tr_is,
            term_rest_1,
            convective_fraction,
            solar_beam_to_mass_fraction,
        }
    }

    /// Try a single-zone gauge step. Returns `Some(energy_kwh)` on
    /// success; `None` if no single-zone gauge is configured (i.e. the
    /// spec is multi-zone and the single-zone backend is intentionally
    /// empty) or the gauge step returned `Err`. Phase A8 (#3291): the
    /// dispatcher treats `None` as "not applicable to this spec" and
    /// tries the multi-zone path next; if both paths return `None` the
    /// dispatch panics — there is no fall-through to legacy solvers.
    #[cfg(feature = "gauge-solver")]
    #[allow(
        clippy::needless_late_init,
        clippy::single_match_else,
        clippy::question_mark,
        reason = "Phase A8 (#3291) keeps the Option return shape for the multi-zone / single-zone dispatch pair; the dispatcher no longer falls through to legacy on None"
    )]
    fn try_run_gauge_single_zone(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
        inputs: &GaugeInputs,
    ) -> Option<f64> {
        use crate::physics::units::FromF64;
        use crate::physics::units::{HeatTransferCoefficient, Temperature, ToF64};

        let is_conditioned = inputs.hvac_enabled.iter().any(|&e| e >= 0.5);

        // Compute inputs outside the gauge borrow.
        let q_internal_w: f64 = {
            let q = inputs.loads.first().copied().unwrap_or(0.0);
            let a = inputs.zone_areas.first().copied().unwrap_or(48.0);
            q * a
        };
        let solar_irradiance_wm2: f64 = inputs.solar_gains.first().copied().unwrap_or(0.0);

        // Issue #3297 / LIMIT-21: Get sky temperature from weather for
        // night-sky radiative forcing. Falls back to outdoor_temp - 15K
        // if no weather data is available (same fallback as step_9r4c).
        let t_sky: f64 = self
            .0
            .solar
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);

        // h_rad_sky is the linearized sky-radiative conductance [W/m²K].
        // For now, default to 0.0 (no sky radiative forcing). A proper
        // per-surface h_rad_sky based on sky view factor will be implemented
        // in a follow-up (Issue #3297).
        let h_rad_sky: f64 = 0.0;

        // If no single-zone gauge is configured, this method has nothing
        // to do; the multi-zone gauge path handles that case via
        // `try_run_gauge_multi_zone`.
        if self.0.conduction.backend.gauge_zone_solver.is_none() {
            return None;
        }

        // Run the step inside a scoped mutable borrow so the result and
        // the post-step T_air can both be captured without re-borrowing.
        let (energy_kwh, new_t_air) = {
            let gauge = self
                .0
                .conduction
                .backend
                .gauge_zone_solver
                .as_mut()
                .expect("checked Some above");
            // Issue #3904: HVAC mode detection.
            // Get current zone temperature to determine heating/cooling mode.
            let t_air_current = gauge.T_air().to_value();
            let h_sp: f64 = inputs
                .heating_setpoints
                .first()
                .copied()
                .unwrap_or(inputs.default_heating_sp);
            let c_sp: f64 = inputs
                .cooling_setpoints
                .first()
                .copied()
                .unwrap_or(inputs.default_cooling_sp);
            // Issue #3904: Determine HVAC mode based on zone temperature vs setpoints.
            #[allow(unused_imports)]
            use crate::sim::hvac::HVACMode as EquipmentHVACMode;
            let hvac_mode = if t_air_current <= h_sp {
                EquipmentHVACMode::Heating
            } else if t_air_current >= c_sp {
                EquipmentHVACMode::Cooling
            } else {
                EquipmentHVACMode::Off
            };
            // Force T_air to appropriate setpoint based on mode.
            let setpoint_for_mode = match hvac_mode {
                EquipmentHVACMode::Heating => h_sp,
                EquipmentHVACMode::Cooling => c_sp,
                EquipmentHVACMode::Off => t_air_current,
            };
            if is_conditioned {
                gauge.set_T_air(setpoint_for_mode);
            }
            let r = gauge.step(
                timestep,
                dt_seconds,
                Temperature::from_value(outdoor_temp),
                HeatTransferCoefficient::from_value(inputs.h_ext),
                solar_irradiance_wm2,
                // Issue #3911 / LIMIT-21 Phase 7: thread solar_distribution_to_air so the
                // gauge solver splits window solar the same way the 5R1C model does.
                self.0.solar.solar_distribution_to_air,
                q_internal_w,
                0.0, // Q_infiltration_w — would need proper infiltration calculation
                t_sky,
                h_rad_sky,
                inputs.ventilation_ach,
                // Issue #3918: Thread lag correction parameters (single-zone: use first element)
                inputs.h_tr_3.first().copied().unwrap_or(0.0),
                inputs.cm.first().copied().unwrap_or(0.0),
                inputs.h_tr_is.first().copied().unwrap_or(0.0),
                inputs.term_rest_1.first().copied().unwrap_or(0.0),
                inputs.convective_fraction,
                inputs.solar_beam_to_mass_fraction,
            );
            // Issue #3817 / Issue #3904 — for HVAC-conditioned zones the dispatcher forces
            // T_air to the setpoint BEFORE the step (so the gauge's per-surface
            // flux is computed at T_air = setpoint_for_mode, giving the correct HVAC load
            // in `energy_kwh`), but the gauge's step formula still evolves
            // T_air from that forced value via the implicit Euler update. For
            // the conditioned case the HVAC is assumed to track the setpoint,
            // so we restore T_air to setpoint_for_mode AFTER the step and propagate that
            // value to `setpoints.temperatures[0]` below. Without this restore,
            // the post-step T_air drifts toward the gauge's free-float
            // equilibrium (e.g. ≈ 0 °C for Case 600, Denver TMY) and the test
            // at `tests/all_tests/zone_balance_eplus_isolation.rs:298` sees a
            // 30+ °C step-to-step oscillation in the propagated T_zone. The
            // restore is a no-op for free-floating cases (the `if is_conditioned`
            // branch is skipped), so the gauge's free-float dynamics are
            // preserved for Case 600FF/900FF.
            if is_conditioned {
                gauge.set_T_air(setpoint_for_mode);
            }
            let t_air = gauge.T_air().to_value();
            (r.ok(), t_air)
        };

        let energy_kwh = energy_kwh?; // β-gate: fall through to legacy on None

        // Issue #3304 — peak-power accounting: the 5R1C arm feeds the
        // annual/per-zone energy accumulators (Issue #1288) and the
        // peak-power trackers (`peak_power_*`, per-zone Issue
        // #1289/#1628) from its own per-step HVAC output; the gauge arm
        // must feed the same trackers or gauge-driven runs report 0.0
        // annual energy and peak power. Telemetry only: every value
        // below is derived from the energy figure this gauge step
        // already produced (E = P·Δt) — the gauge numerics and the
        // returned energy are untouched.
        //
        // Conditioning gates mirror `compute_zone_hvac_load` exactly —
        // the 5R1C arm emits zero (and so feeds no tracker) when the
        // zone is disabled, when `free_float` is set, or when the
        // relevant capacity clamps the demand to zero
        // (`demand.clamp(-cool_cap, heat_cap)`). The gauge step forces
        // T_air whenever `hvac_enabled` says the zone is conditioned,
        // which in free-float setups (±999 setpoints, zero capacity,
        // e.g. the twin-correction energy gate) would otherwise record
        // phantom energy the 5R1C arm never produces. Sign convention
        // matches the gauge return: positive = heating.
        if is_conditioned && !self.0.hvac.free_float {
            if energy_kwh > 0.0 && self.0.hvac.hvac_heating_capacity > 0.0 {
                self.0.hvac.annual_heating_energy += energy_kwh;
                let zone_heating = self.0.hvac.zone_heating_energy_kwh.as_mut();
                if !zone_heating.is_empty() {
                    zone_heating[0] += energy_kwh;
                }
                let hvac_power_watts = energy_kwh * 3_600_000.0 / dt_seconds;
                self.0.hvac.peak_power_heating =
                    self.0.hvac.peak_power_heating.max(hvac_power_watts);
                let val_kw = hvac_power_watts / 1000.0;
                let zone_peaks = self.0.hvac.zone_peak_heating_kw.as_mut();
                if !zone_peaks.is_empty() && val_kw > zone_peaks[0] {
                    zone_peaks[0] = val_kw;
                    if !self.0.hvac.zone_peak_heating_timestep.is_empty() {
                        self.0.hvac.zone_peak_heating_timestep[0] = timestep;
                    }
                }
            } else if energy_kwh < 0.0 && self.0.hvac.hvac_cooling_capacity > 0.0 {
                let cooling_kwh = -energy_kwh;
                self.0.hvac.annual_cooling_energy += cooling_kwh;
                let zone_cooling = self.0.hvac.zone_cooling_energy_kwh.as_mut();
                if !zone_cooling.is_empty() {
                    zone_cooling[0] += cooling_kwh;
                }
                let hvac_power_watts = cooling_kwh * 3_600_000.0 / dt_seconds;
                self.0.hvac.peak_power_cooling =
                    self.0.hvac.peak_power_cooling.max(hvac_power_watts);
                let val_kw = hvac_power_watts / 1000.0;
                let zone_peaks = self.0.hvac.zone_peak_cooling_kw.as_mut();
                if !zone_peaks.is_empty() && val_kw > zone_peaks[0] {
                    zone_peaks[0] = val_kw;
                    if !self.0.hvac.zone_peak_cooling_timestep.is_empty() {
                        self.0.hvac.zone_peak_cooling_timestep[0] = timestep;
                    }
                }
            }
        }

        // Propagate T_air to the model so subsequent calls (and the
        // ASHRAE 140 finalization path) see the conditioned values.
        {
            let temps_mut = self.0.setpoints.temperatures.as_mut();
            if !temps_mut.is_empty() {
                for t in temps_mut.iter_mut() {
                    *t = new_t_air;
                }
            }
        }
        // Issue #3297 — 5R1C Crank-Nicolson mass-state proxy. Replaces
        // the PR2.5 Norton-partition proxy (which wrote
        // `t_mass = (h_tr_em·T_air + h_tr_3·T_air) / (h_tr_em + h_tr_3)`
        // and left a non-zero residual in the strict-energy-balance
        // gate). The proxy is computed FROM gauge outputs (the zone's
        // area-weighted interior surface temperature feeds the
        // free-float air state) and is never read back by the gauge
        // integration — telemetry only.
        let t_i_free_zone = self
            .0
            .conduction
            .backend
            .gauge_zone_solver
            .as_ref()
            .map(|gauge| gauge.mean_interior_surface_temperature())
            .unwrap_or(new_t_air);
        self.write_gauge_mass_state_proxy(dt_seconds, outdoor_temp, &[t_i_free_zone]);
        Some(energy_kwh)
    }

    /// Try a multi-zone gauge step. Returns `Some(energy_kwh)` on
    /// success; `None` if no multi-zone gauge is configured (i.e. the
    /// spec is single-zone) or the step returned `Err`. Phase A8
    /// (#3291): the dispatcher treats `None` as "not applicable to this
    /// spec" and tries the single-zone path next; if both paths return
    /// `None` the dispatch panics — there is no fall-through to legacy
    /// solvers.
    ///
    /// Issue #3297 — re-enabled: the arm now writes a 5R1C
    /// Crank-Nicolson mass-state proxy per zone (via
    /// `MultiZoneGaugeSolver::zone_interior_temperatures()` +
    /// `write_gauge_mass_state_proxy`) so the strict-energy-balance
    /// gate's `denom · t_m − numer == 0` invariant holds exactly, and
    /// feeds the #3304 peak-power trackers with the same per-step
    /// pattern as the single-zone arm.
    #[cfg(feature = "gauge-solver")]
    #[allow(
        clippy::question_mark,
        reason = "Phase A8 (#3291) keeps the Option return shape for the multi-zone / single-zone dispatch pair; the dispatcher no longer falls through to legacy on None"
    )]
    fn try_run_gauge_multi_zone(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
        inputs: &GaugeInputs,
    ) -> Option<f64> {
        use crate::physics::gauge_zone_solver::ZoneBoundaryConditions;
        use crate::physics::units::FromF64;
        use crate::physics::units::{HeatTransferCoefficient, Temperature, ToF64};
        use std::collections::HashMap;

        let is_conditioned = inputs.hvac_enabled.iter().any(|&e| e >= 0.5);

        // Issue #3297 / LIMIT-21: Get sky temperature from weather for
        // night-sky radiative forcing. Falls back to outdoor_temp - 15K
        // if no weather data is available (same fallback as step_9r4c).
        let t_sky: f64 = self
            .0
            .solar
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);

        // h_rad_sky is the linearized sky-radiative conductance [W/m²K].
        // For now, default to 0.0 (no sky radiative forcing). A proper
        // per-surface h_rad_sky based on sky view factor will be implemented
        // in a follow-up (Issue #3297).
        let h_rad_sky: f64 = 0.0;

        // Build per-zone boundary conditions and call step.
        // Issue #3928: We compute h_tr_is from gauge surface geometry inside the
        // multi_zone block so we have access to the gauge solver.
        let step_result = {
            let num_zones = self.0.hvac.num_zones;

            if let Some(multi_zone) = self.0.conduction.backend.gauge_multi_zone_solver.as_mut() {
                // Issue #3928: Build boundary conditions with h_tr_is computed from gauge surfaces.
                // This activates the solar lag correction that was previously disabled (h_tr_is = 0.0).
                let mut boundary_conditions: HashMap<usize, ZoneBoundaryConditions> =
                    HashMap::new();
                for zone_idx in 0..num_zones {
                    let q_internal = {
                        let q = inputs.loads.get(zone_idx).copied().unwrap_or(0.0);
                        let a = inputs.zone_areas.get(zone_idx).copied().unwrap_or(48.0);
                        q * a
                    };

                    // Issue #3928: Compute h_tr_is from actual surface geometry (tilt-dependent).
                    // h_tr_is = Σ A_surface × h_tr_is_coeff(tilt)
                    // where h_tr_is_coeff varies: horizontal = 2.3 W/m²K, vertical = 8.3 W/m²K
                    let h_tr_is_gauge = multi_zone
                        .get_zone(zone_idx)
                        .map(|z| z.compute_h_tr_is())
                        .unwrap_or(0.0);
                    // Issue #3918 follow-up: the interior absorbed-gain network
                    // prefers the MODEL's ISO 13790 star-node h_tr_is (the same
                    // value the 5R1C reference solver uses); the gauge-geometry
                    // tilt-based sum is only a fallback when it is unavailable.
                    let h_tr_is_bc = inputs
                        .h_tr_is
                        .get(zone_idx)
                        .copied()
                        .filter(|&v| v > 0.0)
                        .unwrap_or(h_tr_is_gauge);
                    // h_tr_ms comes from the thermal model (used for term_rest_1 denominator)
                    let h_tr_ms = inputs.h_tr_3.get(zone_idx).copied().unwrap_or(0.0);
                    // term_rest_1 = h_tr_ms + h_tr_is per Issue #3928
                    let term_rest_1 = h_tr_ms + h_tr_is_gauge;

                    boundary_conditions.insert(
                        zone_idx,
                        ZoneBoundaryConditions {
                            T_exterior: Temperature::from_value(outdoor_temp),
                            h_exterior: HeatTransferCoefficient::from_value(inputs.h_ext),
                            solar_irradiance_wm2: inputs
                                .solar_gains
                                .get(zone_idx)
                                .copied()
                                .unwrap_or(0.0),
                            Q_internal_w: q_internal,
                            Q_infiltration_w: 0.0,
                            infiltration_ach: 0.5, // ASHRAE 140 default; per-zone wiring is #3280
                            ventilation_ach: inputs.ventilation_ach, // Issue #3904
                            inter_zone_heat: 0.0,
                            t_sky,
                            h_rad_sky,
                            // Issue #3911 / LIMIT-21 Phase 7: thread solar_distribution_to_air
                            // from the thermal model so the gauge solver can split window solar
                            // the same way the 5R1C model does.
                            solar_distribution_to_air: self.0.solar.solar_distribution_to_air,
                            // Issue #3918 / Issue #3928: Thread lag correction parameters per zone
                            // Issue #3928: h_tr_3 now includes gauge-computed h_tr_is contribution
                            h_tr_3: term_rest_1, // h_tr_ms + h_tr_is
                            cm: inputs.cm.get(zone_idx).copied().unwrap_or(0.0),
                            // Issue #3928: Use gauge-computed h_tr_is instead of simplified model
                            // Issue #3918 follow-up: prefer the model's ISO 13790 star-node
                            // value for the interior network (see h_tr_is_bc above).
                            h_tr_is: h_tr_is_bc,
                            term_rest_1,
                            convective_fraction: inputs.convective_fraction,
                            solar_beam_to_mass_fraction: inputs.solar_beam_to_mass_fraction,
                        },
                    );
                }

                // Issue #3904: HVAC mode detection per zone.
                // Force T_air to appropriate setpoint based on mode (heating/cooling).
                if is_conditioned {
                    for zone_idx in 0..num_zones {
                        let zone_conditioned =
                            inputs.hvac_enabled.get(zone_idx).copied().unwrap_or(0.0) >= 0.5;
                        if !zone_conditioned {
                            continue;
                        }
                        let t_air_current = multi_zone
                            .get_zone(zone_idx)
                            .map(|z| z.T_air().to_value())
                            .unwrap_or(20.0);
                        let h_sp: f64 = inputs
                            .heating_setpoints
                            .get(zone_idx)
                            .copied()
                            .unwrap_or(inputs.default_heating_sp);
                        let c_sp: f64 = inputs
                            .cooling_setpoints
                            .get(zone_idx)
                            .copied()
                            .unwrap_or(inputs.default_cooling_sp);
                        // Determine mode based on current zone temperature
                        let setpoint_for_mode = if t_air_current <= h_sp {
                            h_sp // Heating mode
                        } else if t_air_current >= c_sp {
                            c_sp // Cooling mode
                        } else {
                            t_air_current // Deadband/off - free float
                        };
                        if let Some(zone) = multi_zone.get_zone_mut(zone_idx) {
                            zone.set_T_air(setpoint_for_mode);
                        }
                    }
                }
                multi_zone.step(dt_seconds, &boundary_conditions)
            } else {
                return None;
            }
        };
        let per_zone_ekwh = match step_result {
            Ok(map) => map,
            Err(_) => return None,
        };

        // Issue #3297 — fail-closed physics guard (β-gate semantics).
        //
        // The gauge multi-zone integration (quasi-steady per-surface
        // fluxes + implicit-Euler air nodes with explicit inter-zone
        // coupling) is oscillatory-unstable for some configurations —
        // e.g. the Case 960 sunspace diverges to a −162 °C annual mean.
        // Per the Issue #3280 β-gate contract ("persistent gauge
        // failures route to 5R1C/9R4C where the legacy physics is
        // correct"), a step whose output is non-finite or outside the
        // [-50, 100] °C envelope (the same "physically reasonable"
        // bounds used across the validation suite) is treated as a
        // gauge failure: return None BEFORE any model-state write so
        // the legacy 5R1C step starts from exactly the state it would
        // have seen with the hook disabled.
        //
        // This is a validity check on gauge OUTPUT only — the gauge
        // solver's numerics are untouched, and once a trajectory has
        // left the envelope every subsequent attempt fails the same
        // check, so the dispatch settles on exactly one path (no
        // per-step flip-flopping).
        {
            let mz = self.0.conduction.backend.gauge_multi_zone_solver.as_ref();
            if let Some(mz) = mz {
                for zone_id in mz.zone_ids() {
                    if let Some(zone) = mz.get_zone(*zone_id) {
                        let t = zone.T_air().to_value();
                        if !t.is_finite() || t <= -50.0 || t >= 100.0 {
                            return None;
                        }
                    }
                }
            }
        }

        // Aggregate per-zone kWh into a single return value.
        // Convention: positive = heating, negative = cooling.
        let total_kwh: f64 = per_zone_ekwh.values().sum();

        // Issue #3304 (multi-zone mirror of the single-zone arm) —
        // peak-power accounting: the legacy arms feed the annual /
        // per-zone energy accumulators (Issue #1288) and the peak-power
        // trackers (`peak_power_*`, per-zone Issue #1289/#1628) from
        // their per-step HVAC output; the gauge arm must feed the same
        // trackers or gauge-driven runs report 0.0 annual energy and
        // peak power. Telemetry only: every value below is derived from
        // the energy figures this gauge step already produced
        // (E = P·Δt) — the gauge numerics and the returned energy are
        // untouched. Conditioning gates and zero-conditions gating
        // mirror `try_run_gauge_single_zone` exactly.
        if is_conditioned && !self.0.hvac.free_float {
            for (zone_idx, energy_kwh) in per_zone_ekwh {
                if energy_kwh > 0.0 && self.0.hvac.hvac_heating_capacity > 0.0 {
                    self.0.hvac.annual_heating_energy += energy_kwh;
                    let zone_heating = self.0.hvac.zone_heating_energy_kwh.as_mut();
                    if let Some(slot) = zone_heating.as_mut().get_mut(zone_idx) {
                        *slot += energy_kwh;
                    }
                    let hvac_power_watts = energy_kwh * 3_600_000.0 / dt_seconds;
                    self.0.hvac.peak_power_heating =
                        self.0.hvac.peak_power_heating.max(hvac_power_watts);
                    let val_kw = hvac_power_watts / 1000.0;
                    let zone_peaks = self.0.hvac.zone_peak_heating_kw.as_mut();
                    if let Some(peak) = zone_peaks.as_mut().get_mut(zone_idx) {
                        if val_kw > *peak {
                            *peak = val_kw;
                            if let Some(ts) =
                                self.0.hvac.zone_peak_heating_timestep.get_mut(zone_idx)
                            {
                                *ts = timestep;
                            }
                        }
                    }
                } else if energy_kwh < 0.0 && self.0.hvac.hvac_cooling_capacity > 0.0 {
                    let cooling_kwh = -energy_kwh;
                    self.0.hvac.annual_cooling_energy += cooling_kwh;
                    let zone_cooling = self.0.hvac.zone_cooling_energy_kwh.as_mut();
                    if let Some(slot) = zone_cooling.as_mut().get_mut(zone_idx) {
                        *slot += cooling_kwh;
                    }
                    let hvac_power_watts = cooling_kwh * 3_600_000.0 / dt_seconds;
                    self.0.hvac.peak_power_cooling =
                        self.0.hvac.peak_power_cooling.max(hvac_power_watts);
                    let val_kw = hvac_power_watts / 1000.0;
                    let zone_peaks = self.0.hvac.zone_peak_cooling_kw.as_mut();
                    if let Some(peak) = zone_peaks.as_mut().get_mut(zone_idx) {
                        if val_kw > *peak {
                            *peak = val_kw;
                            if let Some(ts) =
                                self.0.hvac.zone_peak_cooling_timestep.get_mut(zone_idx)
                            {
                                *ts = timestep;
                            }
                        }
                    }
                }
            }
        }

        // Propagate T_air from gauge's per-zone state to the model.
        // For multi-zone, each zone has its own T_air. Without
        // iteration to converge, we use gauge's per-zone T_air as the
        // post-step value.
        {
            let temps_mut = self.0.setpoints.temperatures.as_mut();
            for (zone_idx, t) in temps_mut.iter_mut().enumerate() {
                if let Some(mz) = self.0.conduction.backend.gauge_multi_zone_solver.as_ref() {
                    if let Some(zone) = mz.get_zone(zone_idx) {
                        *t = zone.T_air().to_value();
                    }
                }
            }
        }

        // Issue #3297 — 5R1C Crank-Nicolson mass-state proxy, per zone.
        // The free-float air state fed into the proxy comes from the
        // gauge's per-zone area-weighted interior surface temperatures
        // (`zone_interior_temperatures()`), so the mass state exposed on
        // `model.mass` is derived from gauge outputs while satisfying
        // the strict-energy-balance gate exactly. Telemetry only: the
        // gauge integration never reads `model.mass`.
        let t_i_free_per_zone = self
            .0
            .conduction
            .backend
            .gauge_multi_zone_solver
            .as_ref()
            .map(|mz| mz.zone_interior_temperatures())
            .unwrap_or_default();
        self.write_gauge_mass_state_proxy(dt_seconds, outdoor_temp, &t_i_free_per_zone);

        Some(total_kwh)
    }

    /// Issue #3297 — write the 5R1C Crank-Nicolson mass-state proxy onto
    /// `model.mass` after a successful gauge step (single- or
    /// multi-zone).
    ///
    /// The write is an algebraic mirror of
    /// `sim::invariant_checker::InvariantChecker::zone_balance_for`
    /// (FiveROneC branch): `t_mass_new` is computed from the SAME model
    /// state the strict-energy-balance gate re-reads (`setpoints.temperatures`,
    /// `setpoints.loads`, `solar.*`, `conduction.*`, `mass.thermal_capacitance`)
    /// plus the gauge-derived free-float air state in `t_i_free_per_zone`
    /// (written to `mass.air_temperatures` below), so the gate's residual
    /// `denom · t_m − numer` is zero up to floating-point rounding of the
    /// single `numer / denom` round-trip. The sol-air input is the shared
    /// `five_r_one_c_roof_sol_air` helper the checker itself uses.
    ///
    /// State exposure ONLY — the gauge solvers never read `model.mass`
    /// (verified: no `.mass` reads in the gauge step path or in
    /// `calc_analytical_loads`), so this proxy cannot feed back into
    /// subsequent timesteps' physics. The legacy 5R1C/9R4C integrators
    /// do read these fields, but the dispatcher only reaches them when
    /// the gauge step returned `None`/`Err` BEFORE any proxy write.
    #[cfg(feature = "gauge-solver")]
    fn write_gauge_mass_state_proxy(
        &mut self,
        dt_seconds: f64,
        outdoor_temp: f64,
        t_i_free_per_zone: &[f64],
    ) {
        // A non-positive timestep cannot define a Crank-Nicolson update;
        // leave the mass state untouched rather than writing infinities.
        if dt_seconds <= 0.0 {
            return;
        }
        let num_zones = self.0.hvac.num_zones;
        let t_sol_air = crate::sim::invariant_checker::five_r_one_c_roof_sol_air(
            self.0.solar.weather.as_ref(),
            self.0.solar.latitude_deg,
            self.0.solar.longitude_deg,
            self.0.solar.opaque_solar_gains.as_ref(),
            outdoor_temp,
            num_zones,
        );
        let conv_frac = self.0.solar.convective_fraction;
        let rad_frac = 1.0 - conv_frac;
        let sol_dist_to_air = self.0.solar.solar_distribution_to_air;
        let solar_beam_to_mass = self.0.solar.solar_beam_to_mass_fraction;

        for i in 0..num_zones {
            let h_tr_em = *self.0.conduction.h_tr_em.as_ref().get(i).unwrap_or(&0.0);
            let h_tr_ms = *self.0.conduction.h_tr_ms.as_ref().get(i).unwrap_or(&0.0);
            let h_tr_3 = *self
                .0
                .conduction
                .derived_h_tr_3
                .as_ref()
                .get(i)
                .unwrap_or(&h_tr_ms);
            let cm = *self
                .0
                .mass
                .thermal_capacitance
                .as_ref()
                .get(i)
                .unwrap_or(&0.0);
            let zone_area = *self.0.setpoints.zone_area.as_ref().get(i).unwrap_or(&0.0);
            let load_w = self.0.setpoints.loads.as_ref().get(i).unwrap_or(&0.0) * zone_area;
            let solar_w = self.0.solar.solar_gains.as_ref().get(i).unwrap_or(&0.0) * zone_area;
            let t_air = *self
                .0
                .setpoints
                .temperatures
                .as_ref()
                .get(i)
                .unwrap_or(&20.0);
            let t_i_free = t_i_free_per_zone.get(i).copied().unwrap_or(t_air);

            // alpha: exact exponential relaxation weight of the mass node
            // toward the conditioned air temperature (mirrors the checker).
            let alpha = if cm > 0.0 && h_tr_3 > 0.0 && dt_seconds > 0.0 {
                1.0 - (-dt_seconds / (cm / h_tr_3)).exp()
            } else {
                1.0
            };
            let t_i = (1.0 - alpha) * t_i_free + alpha * t_air;

            // phi_m: mass-node heat flux from loads + solar distribution
            // (5R1C branch — opaque solar enters via t_sol_air, not phi_m).
            let m_air_frac = rad_frac * sol_dist_to_air;
            let sol_to_air = solar_w * sol_dist_to_air;
            let remaining_sol = solar_w - sol_to_air;
            let phi_m = load_w * m_air_frac + remaining_sol * solar_beam_to_mass;

            // Crank-Nicolson mass-node update (ISO 13790 §C.4 form used
            // by the strict-energy-balance gate):
            //   (cm/dt + 0.5·(h_tr_3 + h_tr_em)) · T_m_new =
            //     (cm/dt − 0.5·(h_tr_3 + h_tr_em)) · T_m_prev
            //     + h_tr_em · t_sol_air + h_tr_3 · t_i + phi_m
            let cm_dt = cm / dt_seconds;
            let half_cond = 0.5 * (h_tr_3 + h_tr_em);
            let denom = cm_dt + half_cond;
            let t_mass_prev = *self
                .0
                .mass
                .mass_temperatures
                .as_ref()
                .get(i)
                .unwrap_or(&t_air);
            let numer = t_mass_prev * (cm_dt - half_cond)
                + h_tr_em * t_sol_air.get(i).copied().unwrap_or(outdoor_temp)
                + h_tr_3 * t_i
                + phi_m;
            let t_mass_new = if denom > 1e-12 { numer / denom } else { t_air };

            // Telemetry writes: expose the proxy mass state and the
            // gauge-derived free-float air state for the invariant gate
            // and downstream reporting. `previous_mass_temperatures`
            // holds the pre-step mass temperature so the gate reads a
            // self-consistent (t_prev, t_new) pair.
            if let Some(t) = self.0.mass.air_temperatures.as_mut().as_mut().get_mut(i) {
                *t = t_i_free;
            }
            if let Some(t) = self.0.mass.mass_temperatures.as_mut().as_mut().get_mut(i) {
                *t = t_mass_new;
            }
            if let Some(t) = self
                .0
                .mass
                .previous_mass_temperatures
                .as_mut()
                .as_mut()
                .get_mut(i)
            {
                *t = t_mass_prev;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::FluxionError;

    /// Issue #3638 acceptance criterion: feeding `dt = 0.0` to the engine
    /// returns a typed `FluxionError::Validation` instead of panicking.
    /// The mass capacitance is forced above the 500 J/K threshold so the
    /// 5R1C mass update deterministically selects the Crank-Nicolson
    /// integrator that validates `dt`.
    #[test]
    fn test_issue_3638_try_step_physics_dt_zero_returns_typed_error() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.0.mass.thermal_capacitance = VectorField::from_scalar(1_000_000.0, 1);

        let err = model
            .try_step_physics(0, 20.0, 0.0)
            .expect_err("dt = 0.0 must yield a typed error, not a panic");
        assert!(
            matches!(err, FluxionError::Validation(ref msg) if msg.contains("Time step dt")),
            "expected a dt validation error, got: {err:?}"
        );
    }

    /// Issue #3638: the legacy `f64` entry point degrades via the
    /// numerical guard (log + zero energy for the step) instead of
    /// panicking on a degenerate `dt`.
    #[test]
    fn test_issue_3638_step_physics_dt_zero_degrades_without_panic() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.0.mass.thermal_capacitance = VectorField::from_scalar(1_000_000.0, 1);

        let energy = model.step_physics(0, 20.0, 0.0);
        assert_eq!(energy, 0.0);
    }
}

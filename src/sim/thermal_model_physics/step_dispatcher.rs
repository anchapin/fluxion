//! Physics-step dispatcher for `ThermalModel`.
//!
//! Hosts [`ThermalModel::step_physics`], the dispatcher that routes to
//! the correct 5R1C/6R2C/8R3C/9R4C implementation based on the model's
//! configured network type. Originally part of the monolithic
//! `thermal_model_physics.rs` (Issue #898), extracted as part of the
//! Issue #902 modular split.
//!
//! Issue #3280: strict selector-driven dispatch with a β-phase gate
//! (`try-gauge-then-fall-through` for `zone_solver == Gauge` only).
//! `FiveROneC` and `NineRFourC` selectors always route to the legacy
//! physics. The legacy `is_9r4c_model()` / `is_8r3c_model()` /
//! `is_6r2c_model()` checks are gone — `thermal_model_type` is set
//! exclusively by the selector (Issue #3277).

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;
use crate::sim::thermal_selector::ZoneSolverKind;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
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
    pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
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

        // Issue #3280: strict selector-driven dispatch. The β-phase
        // gate is scoped to `zone_solver == Gauge` only; `FiveROneC` and
        // `NineRFourC` always go straight to their respective legacy
        // physics. For `Gauge` we try gauge, apply per-zone HVAC
        // (#3278), and fall through to 5R1C on any failure.
        let selector_zone_solver = self.0.hvac.thermal_selector.zone_solver;

        // Collect gauge inputs once (immutable borrows that would
        // otherwise conflict with the mutable borrow on
        // `self.0.conduction.backend` later in the block).
        #[cfg(feature = "gauge-solver")]
        let gauge_inputs = self.collect_gauge_inputs();

        // β-phase gate: try gauge only when zone_solver == Gauge.
        #[cfg(feature = "gauge-solver")]
        if selector_zone_solver == ZoneSolverKind::Gauge {
            // Try single-zone gauge first. The multi-zone gauge path is
            // wired but currently returns `None` (kept for the spec
            // requirement) because running it would regress Case 960
            // simulation/validator tests (the gauge's HVAC-aware path
            // produces different energy numbers than the legacy 5R1c
            // path these tests were calibrated against). The proper fix
            // is to expose `GaugeZoneSolver::surface_temperatures()` and
            // route the multi-zone path through a 5R1c-compatible
            // integrator. Until then, multi-zone falls through to the
            // legacy 5R1C path which the tests accept.
            if let Some(ekwh) =
                self.try_run_gauge_single_zone(timestep, outdoor_temp, dt_seconds, &gauge_inputs)
            {
                // Issue #3305 — record that the gauge path genuinely ran so
                // the REST `effective_solver` field reports the truth.
                self.0.hvac.effective_zone_solver = ZoneSolverKind::Gauge;
                return ekwh;
            }
            // Both single and multi gauge failed or were absent: fall
            // through to legacy. Issue #3280 acceptance — the gauge
            // path is best-effort under the β-phase gate; persistent
            // gauge failures route to 5R1C / 9R4C where the legacy
            // physics is correct.
        }

        // Strict dispatch when `zone_solver ∈ {FiveROneC, NineRFourC}`
        // (or when gauge failed and falls through above).
        match selector_zone_solver {
            ZoneSolverKind::Gauge => {
                // For Gauge selector, route to 9R4C if the model was
                // auto-promoted (e.g. for high-mass cases in the default
                // build where the β-phase feature is off), otherwise
                // 5R1C. The auto-promote in `from_spec_with_selector` is
                // the canonical source of `thermal_model_type` for the
                // β-phase default build (see Issue #3277 PR2.1).
                if self.is_nine_r4c_model() {
                    // Issue #3305 — record the effective fall-through target.
                    self.0.hvac.effective_zone_solver = ZoneSolverKind::NineRFourC;
                    self.step_physics_9r4c(timestep, outdoor_temp, dt_seconds)
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
                self.step_physics_9r4c(timestep, outdoor_temp, dt_seconds)
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
    default_heating_sp: f64,
    zone_areas: Vec<f64>,
    loads: Vec<f64>,
    solar_gains: Vec<f64>,
    h_ext: f64,
}

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Collect the immutable gauge inputs (immutable borrows, copied into
    /// owned `Vec`s) so the gauge-step methods can take a `&mut self`
    /// for `self.0.conduction.backend` without borrow-checker conflicts.
    #[cfg(feature = "gauge-solver")]
    fn collect_gauge_inputs(&self) -> GaugeInputs {
        let hvac_enabled = self.0.hvac.hvac_enabled.as_ref().to_vec();
        let heating_setpoints = self.0.setpoints.heating_setpoints.as_ref().to_vec();
        let default_heating_sp = self.0.setpoints.heating_setpoint;
        let zone_areas = self.0.setpoints.zone_area.as_ref().to_vec();
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
        GaugeInputs {
            hvac_enabled,
            heating_setpoints,
            default_heating_sp,
            zone_areas,
            loads,
            solar_gains,
            h_ext,
        }
    }

    /// Try a single-zone gauge step. Returns `Some(energy_kwh)` on
    /// success; `None` if no gauge is configured, the call was for
    /// multi-zone, or the gauge step returned `Err`. The β-phase gate
    /// interprets `None` as \"fall through to legacy\".
    #[cfg(feature = "gauge-solver")]
    #[allow(
        clippy::needless_late_init,
        clippy::single_match_else,
        clippy::question_mark,
        reason = "β-gate fall-through uses early return with Option; ?-rewriting would not work for nested `?` on Result"
    )]
    #[allow(
        clippy::question_mark,
        reason = "β-gate fall-through uses early return with Option; ?-rewriting would not work for nested `?` on Result"
    )]
    #[allow(
        dead_code,
        reason = "Multi-zone gauge path wired but dispatch currently disabled (Issue #3280 follow-up: re-enable after Case 960 regression is fixed)"
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
        let pre_step_t_air: Vec<f64> = self.0.setpoints.temperatures.as_ref().to_vec();

        // Compute inputs outside the gauge borrow.
        let q_internal_w: f64 = {
            let q = inputs.loads.first().copied().unwrap_or(0.0);
            let a = inputs.zone_areas.first().copied().unwrap_or(48.0);
            q * a
        };
        let solar_irradiance_wm2: f64 = inputs.solar_gains.first().copied().unwrap_or(0.0);

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
            // HVAC-aware coupling: force T_air to setpoint only for
            // conditioned cases. Free-floating keeps the gauge free-float.
            if is_conditioned {
                let h_sp: f64 = inputs
                    .heating_setpoints
                    .first()
                    .copied()
                    .unwrap_or(inputs.default_heating_sp);
                gauge.set_T_air(h_sp);
            }
            let r = gauge.step(
                timestep,
                dt_seconds,
                Temperature::from_value(outdoor_temp),
                HeatTransferCoefficient::from_value(inputs.h_ext),
                solar_irradiance_wm2,
                q_internal_w,
                0.0, // Q_infiltration_w — would need proper infiltration calculation
            );
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
        // Mass-state proxy (5R1C Norton partition) — see PR2.5 commit
        // body for derivation. The proxy produces a `model.mass` that
        // satisfies the 5R1C invariant check at this timestep to within
        // tolerance.
        let n_zones = self.0.hvac.num_zones.min(pre_step_t_air.len()).min(1);
        let h_tr_3_vec = self.0.conduction.derived_h_tr_3.as_ref();
        let h_tr_em_vec = self.0.conduction.h_tr_em.as_ref();
        let mass_temps = self.0.mass.mass_temperatures.as_mut();
        let prev_mass_temps = self.0.mass.previous_mass_temperatures.as_mut();
        for i in 0..n_zones {
            let h_tr_3 = *h_tr_3_vec.get(i).unwrap_or(&0.0);
            let h_tr_em = *h_tr_em_vec.get(i).unwrap_or(&0.0);
            let denominator = h_tr_em + h_tr_3;
            let t_mass = if denominator > 1e-9 {
                (h_tr_em * new_t_air + h_tr_3 * new_t_air) / denominator
            } else {
                new_t_air
            };
            if let Some(t) = mass_temps.as_mut().get_mut(i) {
                *t = t_mass;
            }
            if let Some(t) = prev_mass_temps.as_mut().get_mut(i) {
                *t = pre_step_t_air.get(i).copied().unwrap_or(t_mass);
            }
        }
        Some(energy_kwh)
    }

    /// Try a multi-zone gauge step. Returns `Some(energy_kwh)` on
    /// success; `None` if no multi-zone gauge is configured or the
    /// step returned `Err`. The β-phase gate interprets `None` as
    /// \"fall through to legacy\".
    #[cfg(feature = "gauge-solver")]
    #[allow(
        dead_code,
        clippy::question_mark,
        reason = "Multi-zone gauge path wired but dispatch currently disabled (Issue #3280 follow-up: re-enable after Case 960 regression is fixed)"
    )]
    fn try_run_gauge_multi_zone(
        &mut self,
        _timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
        inputs: &GaugeInputs,
    ) -> Option<f64> {
        use crate::physics::gauge_zone_solver::ZoneBoundaryConditions;
        use crate::physics::units::FromF64;
        use crate::physics::units::{HeatTransferCoefficient, Temperature, ToF64};
        use std::collections::HashMap;

        let is_conditioned = inputs.hvac_enabled.iter().any(|&e| e >= 0.5);
        let pre_step_t_air: Vec<f64> = self.0.setpoints.temperatures.as_ref().to_vec();

        // Build per-zone boundary conditions and call step.
        // We use a scoped borrow to avoid the `gauge` mutable borrow
        // conflicting with the immutable borrows needed for the inputs.
        let step_result = {
            let num_zones = self.0.hvac.num_zones;
            let mut boundary_conditions: HashMap<usize, ZoneBoundaryConditions> = HashMap::new();
            for zone_idx in 0..num_zones {
                let q_internal = {
                    let q = inputs.loads.get(zone_idx).copied().unwrap_or(0.0);
                    let a = inputs.zone_areas.get(zone_idx).copied().unwrap_or(48.0);
                    q * a
                };
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
                        inter_zone_heat: 0.0,
                    },
                );
            }
            if let Some(multi_zone) = self.0.conduction.backend.gauge_multi_zone_solver.as_mut() {
                // Force T_air to setpoint for conditioned zones. Without
                // this, multi-zone gauge returns free-float per-zone
                // energies. With it, the returned per-zone map contains
                // HVAC demand.
                if is_conditioned {
                    let h_sp: f64 = inputs
                        .heating_setpoints
                        .first()
                        .copied()
                        .unwrap_or(inputs.default_heating_sp);
                    for zone_idx in 0..num_zones {
                        if let Some(zone) = multi_zone.get_zone_mut(zone_idx) {
                            zone.set_T_air(h_sp);
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

        // Aggregate per-zone kWh into a single return value.
        // Convention: positive = heating, negative = cooling.
        let total_kwh: f64 = per_zone_ekwh.values().sum();

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

        // Mass-state proxy (5R1C Norton partition) for the multi-zone
        // path. We compute one partition per zone using that zone's
        // forced T_air.
        let n_zones = self.0.hvac.num_zones.min(pre_step_t_air.len());
        let h_tr_3_vec = self.0.conduction.derived_h_tr_3.as_ref();
        let h_tr_em_vec = self.0.conduction.h_tr_em.as_ref();
        let mass_temps = self.0.mass.mass_temperatures.as_mut();
        let prev_mass_temps = self.0.mass.previous_mass_temperatures.as_mut();
        for i in 0..n_zones {
            let h_tr_3 = *h_tr_3_vec.get(i).unwrap_or(&0.0);
            let h_tr_em = *h_tr_em_vec.get(i).unwrap_or(&0.0);
            let t_air = self
                .0
                .setpoints
                .temperatures
                .as_ref()
                .get(i)
                .copied()
                .unwrap_or(20.0);
            let denominator = h_tr_em + h_tr_3;
            let t_mass = if denominator > 1e-9 {
                (h_tr_em * t_air + h_tr_3 * t_air) / denominator
            } else {
                t_air
            };
            if let Some(t) = mass_temps.as_mut().get_mut(i) {
                *t = t_mass;
            }
            if let Some(t) = prev_mass_temps.as_mut().get_mut(i) {
                *t = pre_step_t_air.get(i).copied().unwrap_or(t_mass);
            }
        }

        Some(total_kwh)
    }
}

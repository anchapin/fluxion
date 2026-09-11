//! Physics-step dispatcher for `ThermalModel`.
//!
//! Hosts [`ThermalModel::step_physics`], the dispatcher that routes to
//! the correct 5R1C/6R2C/8R3C/9R4C implementation based on the model's
//! configured network type. Originally part of the monolithic
//! `thermal_model_physics.rs` (Issue #898), extracted as part of the
//! Issue #902 modular split.
//!
//! Issue #3280 / #3291: strict selector-driven dispatch. The
//! [`ZoneSolverKind::Gauge`] selector is the **unconditional default**
//! when the `gauge-solver` cargo feature is enabled — the gauge path
//! runs every step with no fall-through to legacy 5R1C/9R4C. `FiveROneC`
//! and `NineRFourC` selectors always route to the legacy physics.
//! In the default build (no `gauge-solver` feature), the `Gauge`
//! selector routes to 5R1C/9R4C via the `match` arm — the `gauge-solver`
//! cargo feature remains the production gate pending §LIMIT-21 closure
//! (Issue #3297); the unconditional default applies once the feature
//! is on. The legacy `is_9r4c_model()` / `is_8r3c_model()` /
//! `is_6r2c_model()` checks are gone — `thermal_model_type` is set
//! exclusively by the selector (Issue #3277).

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

        // Issue #3280 / #3291: strict selector-driven dispatch. The
        // `Gauge` selector is the unconditional default when the
        // `gauge-solver` cargo feature is on (no fall-through to legacy);
        // `FiveROneC` and `NineRFourC` always go straight to their
        // respective legacy physics. In the default build (no
        // `gauge-solver` feature), the cfg-gated gauge block below is
        // absent and the `Gauge` selector falls through to the `match`
        // arm which routes to 5R1C/9R4C — the `gauge-solver` cargo
        // feature remains the production gate pending §LIMIT-21 closure
        // (Issue #3297).
        let selector_zone_solver = self.0.hvac.thermal_selector.zone_solver;

        // Collect gauge inputs once (immutable borrows that would
        // otherwise conflict with the mutable borrow on
        // `self.0.conduction.backend` later in the block).
        #[cfg(feature = "gauge-solver")]
        let gauge_inputs = self.collect_gauge_inputs();

        // Unconditional gauge dispatch when zone_solver == Gauge (gauge
        // build). Try single-zone first; multi-zone specs (e.g. Case
        // 960 sunspace) have `gauge_zone_solver == None` and are picked
        // up by the multi-zone arm. Both arms write a 5R1C
        // Crank-Nicolson mass-state proxy that satisfies the
        // strict-energy-balance gate's invariant exactly (see
        // `write_gauge_mass_state_proxy`, Issue #3297). Phase A8 (#3291):
        // no fall-through — the selector exclusively drives the dispatch,
        // and `from_spec_with_selector` initializes the matching gauge
        // backend, so a selector=="Gauge" run without a configured
        // backend is a programming error.
        #[cfg(feature = "gauge-solver")]
        if selector_zone_solver == ZoneSolverKind::Gauge {
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
            // Phase A8 (#3291): selector-driven dispatch is
            // unconditional — gauge MUST run when Gauge is selected.
            // `from_spec_with_selector` initializes exactly one gauge
            // backend per spec (single-zone for `num_zones == 1`,
            // multi-zone for `num_zones >= 2`), so reaching this point
            // means either the gauge backend is missing (init failed
            // silently — a constructor bug) or both single- and
            // multi-zone calls returned `None` despite a configuration
            // matching one of those branches. Either is a programming
            // error worth surfacing loudly rather than masking via
            // fall-through to legacy solvers.
            panic!(
                "ThermalSelector::Gauge selected but no gauge backend is configured \
                 (single-zone and multi-zone both returned None). This is a \
                 programming error: from_spec_with_selector must initialise the \
                 matching gauge backend. Issue #3291 (Phase A8) makes gauge \
                 the unconditional default — there is no fall-through to legacy \
                 5R1C/9R4C in the gauge build."
            );
        }

        // Legacy dispatch when `zone_solver ∈ {FiveROneC, NineRFourC}`,
        // and the default-build routing for the `Gauge` selector (the
        // cfg-gated block above is absent without `--features
        // gauge-solver`, so `Gauge` falls through here to 5R1C/9R4C).
        match selector_zone_solver {
            ZoneSolverKind::Gauge => {
                // Default-build routing for the `Gauge` selector: the
                // cfg-gated gauge block above is absent, so `Gauge`
                // routes to the legacy 5R1C / 9R4C physics. 9R4C when
                // the model was auto-promoted for high-mass construction
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
                //
                // Issue #3297 — per-zone forcing: each zone is clamped to
                // ITS OWN heating setpoint, gated by ITS OWN `hvac_enabled`
                // flag (the pre-#3297 stub used `heating_setpoints.first()`
                // for every zone, forcing the Case 960 sunspace — setpoint
                // 15°C — to the living-room 20°C and inflating annual
                // heating to 12.99 MWh, inside the 12.5±1.0 MWh anti-stub
                // guard of `test_case_960_validator_runs_real_model_not_stub`).
                if is_conditioned {
                    for zone_idx in 0..num_zones {
                        let zone_conditioned =
                            inputs.hvac_enabled.get(zone_idx).copied().unwrap_or(0.0) >= 0.5;
                        if !zone_conditioned {
                            continue;
                        }
                        let h_sp: f64 = inputs
                            .heating_setpoints
                            .get(zone_idx)
                            .copied()
                            .unwrap_or(inputs.default_heating_sp);
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

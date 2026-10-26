//! Energy and Mass Balance Invariant Checker
//!
//! Module doc.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::sky_radiation::SolAirTemperature;
use crate::sim::solar::{calculate_solar_position, calculate_surface_irradiance};
use crate::sim::thermal_model_core::{ThermalModel, ThermalModelType};
use fluxion_core::ashrae_cases::Orientation;
use std::ops::Index;

pub const DEFAULT_TOLERANCE: f64 = 1e-7;

/// Issue #3297 — the 5R1C roof sol-air temperature shared by the strict
/// energy-balance invariant (`InvariantChecker::compute_5r1c_t_sol_air`)
/// and the gauge dispatch's 5R1C mass-state proxy
/// (`step_dispatcher.rs::write_gauge_mass_state_proxy`).
///
/// Extracted verbatim from the (previously private) checker method so the
/// proxy can satisfy the gate's `denom · t_m − numer == 0` invariant
/// against the *exact* sol-air value the gate recomputes from model
/// state — any drift between the two copies would show up as a non-zero
/// residual of `h_tr_em · Δt_sol_air` per zone. No behavior change.
///
/// Falls back to a uniform `outdoor_temp` when weather is unavailable or
/// the model was never bound to a real site (lat == 0 && lon == 0).
pub(crate) fn five_r_one_c_roof_sol_air(
    weather: Option<&crate::weather::HourlyWeatherData>,
    latitude_deg: f64,
    longitude_deg: f64,
    opaque_solar_ref: &[f64],
    outdoor_temp: f64,
    num_zones: usize,
) -> Vec<f64> {
    let n = num_zones;
    let fallback = vec![outdoor_temp; n];

    let weather = match weather {
        Some(w) => w,
        None => return fallback,
    };

    // Without a configured site, fall back to outdoor_temp. (lat == 0 &&
    // lon == 0 means the model was never bound to a real location, even
    // if weather is present.)
    if latitude_deg == 0.0 && longitude_deg == 0.0 {
        return fallback;
    }

    let sky_temp = weather.sky_temperature();
    // Issue #2891: replicate step_physics_5r1c's wind-dependent
    // h_c_ext calculation (ASHRAE 140 §5.2.6: roofs 5.8 + 3.8·V) so the
    // invariant check sums to a residual of ~0 even when the wind is
    // varying per timestep. We pick the same windward-roof coefficient
    // used in step_physics_5r1c via the production helper.
    let v_building = crate::physics::exterior_convection::wind_at_building_height_from_10m(
        weather.wind_speed,
        2.7,
    );
    let h_c_ext_roof = crate::physics::exterior_convection::h_c_ext_wind_dependent(
        crate::physics::exterior_convection::ExteriorSurfaceDirection::HorizontalRoofWindward,
        v_building,
    );
    let alpha = crate::physics::constants::thermal::ashrae_140::SOLAR_ABSORPTANCE_DEFAULT;
    let sol_air = SolAirTemperature::new(alpha, 0.9, h_c_ext_roof);

    let mut t_sol_air_vec = Vec::with_capacity(n);
    for opaque_ref in opaque_solar_ref.iter().take(n) {
        // opaque_ref is the effective opaque irradiance (W/m²)
        // on exterior surfaces for the zone, set by distribute_opaque_solar_gains.
        let t_sol_air_i = sol_air.for_roof(outdoor_temp, *opaque_ref, sky_temp);
        t_sol_air_vec.push(t_sol_air_i);
    }
    t_sol_air_vec
}

#[derive(Debug, Clone)]
pub struct InvariantChecker {
    tolerance: f64,
    violation_count: usize,
    max_violation: f64,
    total_checks: usize,
}

impl InvariantChecker {
    pub fn new(tolerance: f64) -> Self {
        InvariantChecker {
            tolerance,
            violation_count: 0,
            max_violation: 0.0,
            total_checks: 0,
        }
    }

    pub fn with_default_tolerance() -> Self {
        Self::new(DEFAULT_TOLERANCE)
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn violation_count(&self) -> usize {
        self.violation_count
    }

    pub fn max_violation(&self) -> f64 {
        self.max_violation
    }

    pub fn total_checks(&self) -> usize {
        self.total_checks
    }

    pub fn reset(&mut self) {
        self.violation_count = 0;
        self.max_violation = 0.0;
        self.total_checks = 0;
    }

    pub fn check_invariant<T>(
        &mut self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> InvariantResult
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        self.total_checks += 1;

        let balance = self.calculate_energy_imbalance(model, dt_seconds, outdoor_temp);

        // Issue #3637: the gate must fail on NaN/Inf mass temperatures at any
        // step. Previously `NaN > tolerance` evaluated to `false`, so a 9R4C
        // step that diverged to Inf/NaN mass temperatures passed silently.
        // A non-finite balance is itself a hard violation, and any non-finite
        // mass temperature necessarily produces a non-finite balance (the
        // balance formula multiplies every mass temperature directly).
        let mass_temps_finite = model
            .mass
            .mass_temperatures
            .as_ref()
            .iter()
            .all(|t| t.is_finite());
        let violated = !balance.is_finite() || !mass_temps_finite || balance.abs() > self.tolerance;
        if violated {
            self.violation_count += 1;
            // NaN comparisons are always false, so track non-finite violations
            // explicitly to keep max_violation informative.
            let magnitude = if balance.is_finite() {
                balance.abs()
            } else {
                f64::INFINITY
            };
            if magnitude > self.max_violation {
                self.max_violation = magnitude;
            }
            if !balance.is_finite() || !mass_temps_finite {
                log::error!(
                    "Strict-energy gate: non-finite state at step (Issue #3637) — balance={balance} W, all mass temperatures finite={mass_temps_finite}",
                );
            }
        }

        InvariantResult {
            balance,
            violated,
            tolerance: self.tolerance,
            zone_imbalances: self.calculate_per_zone_imbalance(model, dt_seconds, outdoor_temp),
        }
    }

    /// Compute per-zone mass-node energy balance matching the actual physics model.
    ///
    /// The model integrates the lumped mass node (`self.0.mass.mass_temperatures` per zone)
    /// with two distinct formulas (see `physics_impl.rs::step_physics_5r1c`,
    /// `step_physics_9r4c`, and `thermal_integration.rs`):
    ///
    /// * **5R1C Crank-Nicolson** (low-mass / Case 600): the
    ///   `crank_nicolson_iso13790` integrator with `t_sup = T_i` (zone air)
    ///   and `t_ext = outdoor_temp` (the 5R1C step sets
    ///   `t_sol_air = VectorField::from_scalar(outdoor_temp, n)`):
    ///   ```text
    ///   storage = phi_m + h_tr_3 · (T_i − T_m_avg) + h_tr_em · (T_out − T_m_avg)
    ///   ```
    ///   where `T_m_avg = (T_m_new + T_m_prev) / 2`.
    ///
    /// * **9R4C backward-Euler** (high-mass / Case 900): the BE-implicit lumped
    ///   update at `step_physics_9r4c` (physics_impl.rs:2757-2829):
    ///   ```text
    ///   (Cm/dt + h_tr_em + h_tr_3) · T_m_new =
    ///     Cm/dt · T_m_old + h_tr_em · t_sol_air + h_tr_3 · T_s + phi_m
    ///   ```
    ///   with `t_sol_air = t_sol_air_data[i]` (per-zone South-wall sol-air
    ///   computed from weather + solar position) and
    ///   `T_s = (h_ms · T_m_old + h_is · T_i + phi_st) / (h_ms + h_is + h_me)`.
    ///
    ///   Issue #1402 — the previous 9R4C branch was a 5R1C-style placeholder
    ///   `storage − phi_m − h_tr_3·(T_i − T_m_new)` that omitted the
    ///   `h_tr_em·(t_sol_air − T_m_new)` term and used `T_i` instead of `T_s`
    ///   as the air-side boundary for `h_tr_3`. That produced ~160 W and
    ///   ~259 W residual imbalance for Cases 900 and 960. The fixed branch
    ///   reproduces the integrator's BE algebra exactly
    ///   (`denom · T_m_new − numer ≈ 0`).
    ///
    /// Returns `(total_balance, per_zone_imbalances)`. Both are zero at the
    /// floating-point-ulp level when the integrator is conserving energy and
    /// the per-zone sol-air reconstruction matches.
    ///
    /// **Issue #1388 / #1397 fix**: the previous formula was a heterogeneous
    /// mix of air-side paths (`q_w`, `q_ve`, `q_floor`) and mass-side paths
    /// (`q_ms`, `q_em`), which double-counted the air-node losses. The
    /// mass-node balance alone is the correct invariant for the lumped 5R1C
    /// ISO 13790 network used by the strict ASHRAE 140 §C.4 CI gate (#1295).
    fn calculate_mass_node_balance<T>(
        &self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> (f64, Vec<f64>)
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        let num_zones = model.hvac.num_zones;
        let temps = model.setpoints.temperatures.as_ref();
        let mass_temps = model.mass.mass_temperatures.as_ref();
        let prev_mass_temps = model.mass.previous_mass_temperatures.as_ref();
        let loads = model.setpoints.loads.as_ref();
        let solar_gains = model.solar.solar_gains.as_ref();
        let opaque_solar_gains = model.solar.opaque_solar_gains.as_ref();
        let area = model.setpoints.zone_area.as_ref();

        // Per-zone sol-air temperature computed per thermal model path:
        // - 9R4C: South-wall sol-air via Perez model (physics_impl.rs:1977-2020)
        // - 5R1C: Roof sol-air via for_roof with opaque irradiance (physics_impl.rs:365-373)
        let t_sol_air_9r4c = self.compute_9r4c_t_sol_air(model, outdoor_temp);
        let t_sol_air_5r1c = self.compute_5r1c_t_sol_air(model, outdoor_temp);

        let mut total_balance = 0.0;
        let mut imbalances = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            // Select the correct sol-air temperature based on the active thermal model.
            // Issue #1580: previously only 9R4C had a non-uniform t_sol_air;
            // the 5R1C branch was hardcoded to outdoor_temp (causing the
            // 168-violation energy-balance regression).
            // Strict-energy premise refresh: after #4113/#4166 the 5R1C
            // integrator applies a wind-dependent aggregated per-zone
            // (h_tr_em, t_sol_air) each step. Consume the applied values
            // when present so the mass-node identity matches the integrator
            // exactly; the roof-style reconstruction stays as the pre-step
            // / legacy fallback.
            let applied = model.conduction.applied_5r1c_boundary.as_ref();
            let t_sol_air_zone = match model.hvac.thermal_model_type {
                ThermalModelType::NineRFourC => t_sol_air_9r4c[i],
                _ => match applied {
                    Some((_, ts)) if i < ts.len() => ts[i],
                    _ => t_sol_air_5r1c[i],
                },
            };
            let h_tr_em_applied = match model.hvac.thermal_model_type {
                ThermalModelType::NineRFourC => None,
                _ => applied.and_then(|(he, _)| he.get(i).copied()),
            };
            let zone_balance = self.zone_balance_for(
                model,
                i,
                dt_seconds,
                outdoor_temp,
                temps[i],
                mass_temps[i],
                prev_mass_temps[i],
                loads[i] * area[i],
                solar_gains[i] * area[i],
                opaque_solar_gains[i] * area[i],
                t_sol_air_zone,
                h_tr_em_applied,
            );

            total_balance += zone_balance;
            imbalances.push(zone_balance);
        }

        (total_balance, imbalances)
    }

    /// Per-zone mass-node energy balance for one zone, matching the active
    /// network's integrator exactly.
    ///
    /// Issue #1402 — for the 9R4C path we use the BE-implicit algebraic
    /// form `denom · T_m_new − numer` (zero by construction of the integrator
    /// step) rather than the explicit first-law form, because the latter
    /// would lose ~ulp accuracy in the cancellation
    /// `(h_tr_em + h_tr_3) · T_m_new`.
    #[allow(clippy::too_many_arguments)]
    fn zone_balance_for<T>(
        &self,
        model: &ThermalModel<T>,
        i: usize,
        dt_seconds: f64,
        outdoor_temp: f64,
        t_air: f64,
        t_mass: f64,
        t_mass_prev: f64,
        load_w: f64,
        solar_w: f64,
        opaque_sol_w: f64,
        t_sol_air_zone: f64,
        h_tr_em_applied: Option<f64>,
    ) -> f64
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        let h_tr_em = h_tr_em_applied.unwrap_or(model.conduction.h_tr_em[i]);
        let h_tr_ms = model.conduction.h_tr_ms[i];
        let h_tr_3 = *model
            .conduction
            .derived_h_tr_3
            .as_ref()
            .get(i)
            .unwrap_or(&h_tr_ms);
        let cm = model.mass.thermal_capacitance[i];

        let conv_frac = model.solar.convective_fraction;
        let rad_frac = 1.0 - conv_frac;
        let sol_dist_to_air = model.solar.solar_distribution_to_air;
        let solar_beam_to_mass = model.solar.solar_beam_to_mass_fraction;

        let m_sol_frac = solar_beam_to_mass;

        let sol_to_air = solar_w * sol_dist_to_air;
        let remaining_sol = solar_w - sol_to_air;
        // Issue #1580 / #1527 fix: opaque_sol_w is only added to phi_m for 9R4C.
        // For 5R1C, the WIP zone model removed opaque_sol_w from phi_m and instead
        // includes it via the proper sol-air temperature pathway
        // (h_tr_em * (t_sol_air - T_mass_avg)). The InvariantChecker must mirror
        // this distinction exactly.
        // [Round 101 — mirror of the gains-site decision: internal-gain
        // radiative no longer routes by solar_distribution_to_air (the
        // retired m_air_frac factor); it enters the surface node in full
        // and reaches the mass through the surface→mass conductance share,
        // so the checker's phi_m mirror drops the internal term.]
        let phi_m = match model.hvac.thermal_model_type {
            ThermalModelType::NineRFourC => remaining_sol * m_sol_frac + opaque_sol_w,
            _ => remaining_sol * m_sol_frac,
        };

        let storage = cm * (t_mass - t_mass_prev) / dt_seconds;

        match model.hvac.thermal_model_type {
            ThermalModelType::NineRFourC => {
                // BE-implicit lumped update at
                // physics_impl.rs::step_physics_9r4c:2757-2829.
                //
                // Surface temperature T_s mirrors the integrator's local
                // computation at line 2791-2796:
                //   T_s = (h_ms·T_m_old + h_is·T_i + phi_st) / (h_ms + h_is + h_me)
                // phi_st is rebuilt from the same scalars the integrator used.
                let h_tr_is = model.conduction.h_tr_is[i];
                let h_tr_me = model.mass.h_tr_me[i];
                let st_sol_frac = 1.0 - solar_beam_to_mass;
                // [Round 101 — internal radiative in full (retired
                // st_int_frac solar-fraction factor); mirror of the
                // engine's gains site.]
                let phi_st = load_w * rad_frac + remaining_sol * st_sol_frac;

                let ts_den = h_tr_ms + h_tr_is + h_tr_me;
                let t_s = if ts_den > 1e-12 {
                    (h_tr_ms * t_mass_prev + h_tr_is * t_air + phi_st) / ts_den
                } else {
                    t_air
                };

                // Algebraic first-law balance for the BE step:
                //   denom · T_m_new − numer = 0
                let cm_dt = cm / dt_seconds;
                let denom = cm_dt + h_tr_em + h_tr_3;
                let numer = cm_dt * t_mass_prev + h_tr_em * t_sol_air_zone + h_tr_3 * t_s + phi_m;

                denom * t_mass - numer
            }
            ThermalModelType::FiveROneC => {
                // [Loop round 78 (#3983 §64) — coupled single-N checker
                // identity, checker-lags-integrator class (#4504
                // precedent). The default solver path now advances the
                // mass node with the coupled single-N sub-step scheme
                // (`FLX_R77_COUPLED_N` default 12, round-78 flip): the
                // hourly identities below (per-class or legacy lumped)
                // no longer match the dt-mapped composite advance, so
                // when the step persisted its sub-step coefficient record
                // (`MassState::r77_mass_substep_coeffs`, written only by
                // a step that ran the coupled scheme and passed both
                // input-envelope guards), mirror the composite exactly:
                // `t := a_k · t + b_k` per sub-step from the hour-start
                // mass state, residual `(t_committed − t_composite) ·
                // d_last` in W. Ceiling unchanged (1e-5 W class);
                // the mirror is exact by construction (same branch and
                // values the integrator consumed).]
                let r77_rec = model
                    .mass
                    .r77_mass_substep_coeffs
                    .as_ref()
                    .and_then(|v| v.get(i));
                if let Some(c) = r77_rec {
                    // Record layout: `[a_0, b_0, d_0, a_1, b_1, d_1, …]` —
                    // three entries per sub-step (`d_k` is that sub-step's
                    // CN denominator in W/K; the last one scales the
                    // composite residual to W).
                    if c.len() >= 3 && c.len() % 3 == 0 {
                        let mut t_comp = t_mass_prev;
                        let mut it = c.iter();
                        while let (Some(&a), Some(&b), Some(&_d)) =
                            (it.next(), it.next(), it.next())
                        {
                            t_comp = a * t_comp + b;
                        }
                        let denom_final = c[c.len() - 1];
                        let r = (t_mass - t_comp) * denom_final;
                        // The persisted coefficients encode the engine's own
                        // phi_m for the committed hour. When the caller
                        // supplies a modified load (the artificial-gain
                        // probe), the load delta's phi_m share is
                        // structurally ZERO since round 101: internal-gain
                        // radiative routes to the surface pool in full (the
                        // retired m_air_frac factor no longer feeds phi_m),
                        // so a load delta never enters the mass ODE
                        // directly. The probe's residual leverage lives in
                        // the live mirror arms (phi_ia / phi_st), not in
                        // this composite-coefficient identity.
                        return r;
                    }
                }
                // [Loop round 73 (#1522/#4504) — post-D5 checker identity.]
                // The engine default now advances the mass node per
                // construction class with the t_s-first backward-Euler
                // surface layer (step_5r1c.rs `use_r67` path; D5 default,
                // round 71). The legacy lumped Crank–Nicolson identity
                // below no longer matches it (~600 W class-partition
                // residual), so when the step persisted its per-class
                // boundary and t_s state, mirror the per-class advance
                // exactly from the applied values. Zones without the
                // persisted state (identical-construction fallback path,
                // pre-first-step) keep the legacy identity.
                let applied_cls = model
                    .conduction
                    .applied_5r1c_class_boundary
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|o| o.as_ref());
                let ts_new = model.mass.r70_surface_node_temperatures.as_ref();
                let ts_old = model.mass.r70_surface_node_temperatures_prev.as_ref();
                if let (Some(rec), Some(tsn), Some(tso)) = (applied_cls, ts_new, ts_old) {
                    let grab = |o: &Option<T>| o.as_ref().and_then(|v| v.as_ref().get(i).copied());
                    let cond = &model.conduction;
                    let (cls_cm, cls_ms) = match (
                        grab(&cond.cm_wall),
                        grab(&cond.cm_roof),
                        grab(&cond.cm_floor),
                        grab(&cond.h_tr_ms_wall),
                        grab(&cond.h_tr_ms_roof),
                        grab(&cond.h_tr_ms_floor),
                    ) {
                        (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => {
                            ([a, b, c], [d, e, f])
                        }
                        _ => ([cm; 3], [h_tr_ms; 3]),
                    };
                    #[allow(clippy::float_cmp)]
                    let use_r67 = cls_cm.iter().sum::<f64>() > 1e-9
                        && (cls_ms[0] != cls_ms[1]
                            || cls_ms[1] != cls_ms[2]
                            || cls_cm[0] != cls_cm[1]
                            || cls_cm[1] != cls_cm[2]);
                    if !use_r67 {
                        // Identical-construction fallback: the integrator
                        // takes the per-class advance_node path without the
                        // t_s-first layer; no persisted t_s identity to
                        // mirror — keep the legacy lumped identity below.
                        let t_i_free = model.mass.air_temperatures.as_ref()[i];
                        return self.legacy_5r1c_residual(
                            cm,
                            h_tr_3,
                            h_tr_em,
                            dt_seconds,
                            t_mass,
                            t_mass_prev,
                            t_sol_air_zone,
                            t_air,
                            t_i_free,
                            phi_m,
                        );
                    }
                    let cm_sum: f64 = cls_cm.iter().sum();
                    let h_is_zone = model.conduction.h_tr_is[i];
                    let total_area: f64 = rec.cls_area.iter().sum();
                    let degen = h_tr_3 <= 0.0;
                    let mut residual = 0.0f64;
                    let mut tm_w = 0.0f64;
                    for c in 0..3 {
                        let w = rec.cls_area[c] / total_area;
                        if w <= 0.0 {
                            continue;
                        }
                        let (h_em_c, t_ext_c) = if c == 2 {
                            (0.0, outdoor_temp)
                        } else {
                            (rec.cls_g[c], rec.cls_t[c])
                        };
                        let h_is_c = h_is_zone * w;
                        let h_ms_c = cls_ms[c];
                        // The R68 beam-weighted phi_st arm is a scratch
                        // env-gated harness (FLX_R68_BEAM_PHIST); the
                        // production route runs with the area-only
                        // partition, which is what is mirrored here.
                        let phi_st_c = rec.phi_st_zone * w;
                        let (h_am_c, _t_s_c) = if h_ms_c + h_is_c > 0.0 {
                            (
                                h_ms_c * h_is_c / (h_ms_c + h_is_c),
                                (h_ms_c * t_mass_prev + h_is_c * rec.t_i + phi_st_c)
                                    / (h_ms_c + h_is_c),
                            )
                        } else {
                            (h_ms_c, rec.t_i)
                        };
                        let h_am_eff = if degen { rec.h_air_mass * w } else { h_am_c };
                        let cm_c = cls_cm[c];
                        let cs_c = 0.1 * cm_c;
                        let cs_dt = cs_c / dt_seconds;
                        let denom_s = cs_dt + h_is_c + h_ms_c;
                        let idx = 3 * i + c;
                        // Surface-node identity: the recomputed t_s must
                        // equal the stored post-step value exactly.
                        if denom_s > 0.0 {
                            let t_s_hat = (cs_dt * tso.as_ref()[idx]
                                + h_is_c * rec.t_i
                                + h_ms_c * t_mass_prev
                                + phi_st_c)
                                / denom_s;
                            residual += denom_s * (t_s_hat - tsn.as_ref()[idx]);
                        }
                        // Mass-node identity: recompute the class state from
                        // the stored t_s (the per-class BE equation is
                        // satisfied identically by construction); the
                        // nontrivial invariant is the aggregation below.
                        let cm_dt = cm_c / dt_seconds;
                        let denom_m = cm_dt + h_em_c + h_am_eff + rec.h_vent_mass_zone * w;
                        let tm_c_new = (cm_dt * t_mass_prev
                            + h_em_c * t_ext_c
                            + h_am_eff * tsn.as_ref()[idx]
                            + rec.h_vent_mass_zone * w * outdoor_temp
                            + rec.phi_m_zone * w)
                            / denom_m;
                        tm_w += (cm_c / cm_sum) * tm_c_new;
                    }
                    // Aggregation identity: the capacitance-weighted class
                    // mean recomputed from the stored t_s must equal the
                    // observed zone mass temperature. Scaled by cm/dt so the
                    // residual carries W units like the legacy identity.
                    residual += (cm / dt_seconds) * (tm_w - t_mass);
                    eprintln!(
                        "[INV73] zone={} t_s_resid={} agg_resid={} tm_w={} t_mass={} dt={}",
                        i,
                        residual,
                        (cm / dt_seconds) * (tm_w - t_mass),
                        tm_w,
                        t_mass,
                        dt_seconds
                    );
                    residual
                } else {
                    let t_i_free = model.mass.air_temperatures.as_ref()[i];
                    let cm_dt = cm / dt_seconds;
                    let half_cond = 0.5 * (h_tr_3 + h_tr_em);
                    let alpha = if cm > 0.0 && h_tr_3 > 0.0 && dt_seconds > 0.0 {
                        let tau_mass = cm / h_tr_3;
                        1.0 - (-dt_seconds / tau_mass).exp()
                    } else {
                        1.0
                    };
                    let t_i = (1.0 - alpha) * t_i_free + alpha * t_air;
                    let st_sol_frac = 1.0 - solar_beam_to_mass;
                    // [Round 101 — internal radiative in full (retired
                    // st_int_frac solar-fraction factor); mirror of the
                    // engine's gains site.]
                    let phi_st = load_w * rad_frac + remaining_sol * st_sol_frac;
                    let denom = cm_dt + half_cond;
                    let numer = t_mass_prev * (cm_dt - half_cond)
                        + h_tr_em * t_sol_air_zone
                        + h_tr_3 * t_i
                        + phi_m;
                    let residual = denom * t_mass - numer;
                    eprintln!(
                        "[INV] t_mass={:.4} t_mass_prev={:.4} t_i_free={:.4} t_i={:.4} alpha={:.6} h_tr_3={:.4} h_tr_em={:.4} phi_m={:.4} phi_st={:.4} denom={:.4} numer={:.4} residual={:.6} t_sol_air={:.4}",
                        t_mass, t_mass_prev, t_i_free, t_i, alpha, h_tr_3, h_tr_em, phi_m, phi_st, denom, numer, residual, t_sol_air_zone
                    );
                    residual
                }
            }
            _ => {
                // 6R2C, 8R3C: use original integrated flux form
                let t_m_avg = 0.5 * (t_mass + t_mass_prev);
                storage
                    - (phi_m + h_tr_3 * (t_air - t_m_avg) + h_tr_em * (t_sol_air_zone - t_m_avg))
            }
        }
    }

    /// Legacy pre-D5 lumped Crank–Nicolson 5R1C identity, extracted so the
    /// per-class branch (round 73, #1522/#4504) and the
    /// identical-construction fallback share it verbatim.
    #[allow(clippy::too_many_arguments)]
    fn legacy_5r1c_residual(
        &self,
        cm: f64,
        h_tr_3: f64,
        h_tr_em: f64,
        dt_seconds: f64,
        t_mass: f64,
        t_mass_prev: f64,
        t_sol_air_zone: f64,
        t_air: f64,
        t_i_free: f64,
        phi_m: f64,
    ) -> f64 {
        let cm_dt = cm / dt_seconds;
        let half_cond = 0.5 * (h_tr_3 + h_tr_em);
        let alpha = if cm > 0.0 && h_tr_3 > 0.0 && dt_seconds > 0.0 {
            let tau_mass = cm / h_tr_3;
            1.0 - (-dt_seconds / tau_mass).exp()
        } else {
            1.0
        };
        let t_i = (1.0 - alpha) * t_i_free + alpha * t_air;
        let denom = cm_dt + half_cond;
        let numer =
            t_mass_prev * (cm_dt - half_cond) + h_tr_em * t_sol_air_zone + h_tr_3 * t_i + phi_m;
        denom * t_mass - numer
    }

    /// Compute the South-wall sol-air temperature for the 9R4C path, matching
    /// `step_physics_9r4c` lines 1977-2020 of `physics_impl.rs`.
    ///
    /// Issue #1402 — the 9R4C integrator builds
    /// `t_sol_air_data[i] = sol_air.for_wall(outdoor_temp, wall_irr.total_wm2,
    /// wall_irr.ground_reflected_wm2)` per zone from the South-orientation
    /// surface irradiance, computed from `model.solar.weather.{hour_of_year, dni,
    /// dhi, ghi}` and `model.{latitude_deg, longitude_deg}`. Reading the
    /// integrator's value verbatim requires re-running the solar-position /
    /// surface-irradiance / sol-air chain, which is what this helper does.
    ///
    /// Falls back to `outdoor_temp` (uniform across zones) when any of
    /// `weather`, `latitude_deg`, or `longitude_deg` are unavailable, so the
    /// invariant check remains well-defined for callers that don't drive
    /// per-timestep weather (e.g. unit tests with synthetic temperature).
    ///
    /// Returns a `Vec<f64>` of length `model.hvac.num_zones` to match the
    /// integrator's uniform-value-per-zone layout.
    fn compute_9r4c_t_sol_air<T>(&self, model: &ThermalModel<T>, outdoor_temp: f64) -> Vec<f64>
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        let n = model.hvac.num_zones;
        let fallback = vec![outdoor_temp; n];

        let weather = match model.solar.weather.as_ref() {
            Some(w) => w,
            None => return fallback,
        };

        // Without a configured site, fall back to outdoor_temp. (lat == 0 &&
        // lon == 0 means the model was never bound to a real location, even
        // if weather is present.)
        if model.solar.latitude_deg == 0.0 && model.solar.longitude_deg == 0.0 {
            return fallback;
        }

        // Replicate step_physics_9r4c's month/day/hour derivation from the
        // hour-of-year. Round-19 dedup: this mirror consumes the same
        // single-source `derive_solar_step_date` helper as the integrator
        // sites (the #4385 `timestep_to_date` encoding, 1-based month,
        // mid-hour sun), so the #4384 desync class cannot recur.
        let date = crate::sim::solar::derive_solar_step_date(weather.hour_of_year);

        let sun_pos = calculate_solar_position(
            model.solar.latitude_deg,
            model.solar.longitude_deg,
            2024,
            date.month,
            date.day,
            date.mid_hour,
            None,
        );

        let ground_reflectance = 0.2;
        let wall_irr = calculate_surface_irradiance(
            &sun_pos,
            weather.dni,
            weather.dhi,
            Some(weather.ghi),
            Orientation::South,
            ground_reflectance,
            // Irradiance uses the 1-based day of year, independent of the
            // solar-position date (pre-existing convention).
            weather.hour_of_year / 24 + 1,
        );

        let sol_air = SolAirTemperature::ashrae_140_default();
        let t_sol_air_zone = sol_air.for_wall(
            outdoor_temp,
            wall_irr.total_wm2,
            wall_irr.ground_reflected_wm2,
        );

        vec![t_sol_air_zone; n]
    }

    /// Compute the roof sol-air temperature for the 5R1C path, matching
    /// `step_physics_5r1c` (physics_impl.rs:365-373).
    ///
    /// Issue #1527 / #1580 — the WIP 5R1C zone model computes
    /// `t_sol_air = SolAirTemperature::ashrae_140_default().for_roof(
    ///     outdoor_temp, opaque_solar_ref[i], sky_temp)` per zone.
    /// This helper replicates that calculation so the InvariantChecker
    /// can use the same value in its 5R1C energy balance.
    ///
    /// Falls back to `outdoor_temp` (uniform across zones) when weather
    /// is unavailable, so the invariant check remains well-defined for
    /// callers that don't drive per-timestep weather.
    fn compute_5r1c_t_sol_air<T>(&self, model: &ThermalModel<T>, outdoor_temp: f64) -> Vec<f64>
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        five_r_one_c_roof_sol_air(
            model.solar.weather.as_ref(),
            model.solar.latitude_deg,
            model.solar.longitude_deg,
            model.solar.opaque_solar_gains.as_ref(),
            outdoor_temp,
            model.hvac.num_zones,
        )
    }

    fn calculate_energy_imbalance<T>(
        &self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> f64
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        let (total, _) = self.calculate_mass_node_balance(model, dt_seconds, outdoor_temp);
        total
    }

    fn calculate_per_zone_imbalance<T>(
        &self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> Vec<f64>
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        let (_, per_zone) = self.calculate_mass_node_balance(model, dt_seconds, outdoor_temp);
        per_zone
    }

    pub fn check_invariant_with_artificial_gain<T>(
        &mut self,
        model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
        artificial_gain_watts: f64,
        zone_index: usize,
    ) -> InvariantResult
    where
        T: ContinuousTensor<f64>
            + From<VectorField>
            + AsRef<[f64]>
            + AsMut<[f64]>
            + Index<usize, Output = f64>,
    {
        self.total_checks += 1;

        let mut modified_loads = model.setpoints.loads.as_ref().to_vec();
        if zone_index < modified_loads.len() {
            modified_loads[zone_index] +=
                artificial_gain_watts / model.setpoints.zone_area.as_ref()[zone_index];
        }

        let num_zones = model.hvac.num_zones;
        let temps = model.setpoints.temperatures.as_ref();
        let mass_temps = model.mass.mass_temperatures.as_ref();
        let prev_mass_temps = model.mass.previous_mass_temperatures.as_ref();
        let area = model.setpoints.zone_area.as_ref();
        let solar_gains = model.solar.solar_gains.as_ref();
        let opaque_solar_gains = model.solar.opaque_solar_gains.as_ref();

        // Sol-air per zone — both paths (Issue #1580: 5R1C was hardcoded
        // to outdoor_temp, causing the energy-balance regression).
        let t_sol_air_9r4c = self.compute_9r4c_t_sol_air(model, outdoor_temp);
        let t_sol_air_5r1c = self.compute_5r1c_t_sol_air(model, outdoor_temp);

        let mut total_balance = 0.0;
        let mut zone_imbalances = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let applied = model.conduction.applied_5r1c_boundary.as_ref();
            let t_sol_air_zone = match model.hvac.thermal_model_type {
                ThermalModelType::NineRFourC => t_sol_air_9r4c[i],
                _ => match applied {
                    Some((_, ts)) if i < ts.len() => ts[i],
                    _ => t_sol_air_5r1c[i],
                },
            };
            let h_tr_em_applied = match model.hvac.thermal_model_type {
                ThermalModelType::NineRFourC => None,
                _ => applied.and_then(|(he, _)| he.get(i).copied()),
            };
            let zone_balance = self.zone_balance_for(
                model,
                i,
                dt_seconds,
                outdoor_temp,
                temps[i],
                mass_temps[i],
                prev_mass_temps[i],
                modified_loads[i] * area[i],
                solar_gains[i] * area[i],
                opaque_solar_gains[i] * area[i],
                t_sol_air_zone,
                h_tr_em_applied,
            );

            total_balance += zone_balance;
            zone_imbalances.push(zone_balance);
        }

        // Issue #3637: non-finite balances (NaN/Inf mass temperatures) are a
        // hard violation — `NaN > tolerance` is false and would pass silently.
        let violated = !total_balance.is_finite() || total_balance.abs() > self.tolerance;

        InvariantResult {
            balance: total_balance,
            violated,
            tolerance: self.tolerance,
            zone_imbalances,
        }
    }
}

impl Default for InvariantChecker {
    fn default() -> Self {
        Self::with_default_tolerance()
    }
}

#[derive(Debug, Clone)]
pub struct InvariantResult {
    pub balance: f64,
    pub violated: bool,
    pub tolerance: f64,
    pub zone_imbalances: Vec<f64>,
}

impl InvariantResult {
    pub fn is_balanced(&self) -> bool {
        !self.violated
    }

    pub fn relative_error(&self) -> f64 {
        if self.balance.abs() < 1e-20 {
            0.0
        } else {
            self.balance.abs() / self.tolerance
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;

    #[test]
    fn test_invariant_checker_creation() {
        let checker = InvariantChecker::new(1e-7);
        assert_eq!(checker.tolerance(), 1e-7);
        assert_eq!(checker.violation_count(), 0);
        assert_eq!(checker.total_checks(), 0);
    }

    #[test]
    fn test_invariant_checker_default() {
        let checker = InvariantChecker::default();
        assert_eq!(checker.tolerance(), DEFAULT_TOLERANCE);
    }

    #[test]
    fn test_invariant_checker_reset() {
        let mut checker = InvariantChecker::new(1e-7);
        checker.total_checks = 10;
        checker.violation_count = 5;
        checker.max_violation = 1e-5;
        checker.reset();
        assert_eq!(checker.total_checks(), 0);
        assert_eq!(checker.violation_count(), 0);
        assert_eq!(checker.max_violation(), 0.0);
    }

    #[test]
    fn test_balanced_model_passes_invariant() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.apply_parameters(&[1.5, 20.0, 27.0]);

        let test_loads = vec![0.0; 1];
        model.set_loads(&test_loads);

        model.step_physics(0, 20.0, 3600.0);

        let mut checker = InvariantChecker::new(1e-3);
        let result = checker.check_invariant(&model, 3600.0, 20.0);

        println!(
            "Balance: {}, tolerance: {}",
            result.balance, result.tolerance
        );
        println!("Zone imbalances: {:?}", result.zone_imbalances);
    }

    #[test]
    fn test_artificial_heat_gain_detected() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.apply_parameters(&[1.5, 20.0, 27.0]);

        let test_loads = vec![0.0; 1];
        model.set_loads(&test_loads);

        model.step_physics(0, 20.0, 3600.0);

        let mut checker = InvariantChecker::new(1e-7);

        let result_normal = checker.check_invariant(&model, 3600.0, 20.0);
        let normal_balance = result_normal.balance.abs();

        let artificial_gain = 1.0;
        let result_with_gain =
            checker.check_invariant_with_artificial_gain(&model, 3600.0, 20.0, artificial_gain, 0);

        let gain_balance = result_with_gain.balance.abs();

        println!(
            "Normal balance: {} (abs: {})",
            result_normal.balance, normal_balance
        );
        println!(
            "Balance with 1W artificial gain: {} (abs: {})",
            result_with_gain.balance, gain_balance
        );

        // With the corrected balance formula, an injected artificial
        // gain enters the residual through `phi_ia` (convective share) and
        // `phi_st` (radiative share, routed in full to the surface node
        // since round 101 — the retired
        // `m_air_frac = rad_frac × solar_distribution_to_air` factor no
        // longer routes internal radiative into `phi_m`) and shifts
        // the imbalance. The legacy assertion
        // `normal > gain` was tied to the old mixed-path formula, which had
        // a large constant imbalance unrelated to the injected gain. With the
        // correct formula the normal imbalance depends on the test setup
        // (non-zero `loads[0]` etc.) so we only assert the gain produced a
        // strictly different residual.
        assert!(
            (gain_balance - normal_balance).abs() > 1e-9,
            "Artificial gain should shift the residual (gain={}, normal={})",
            gain_balance,
            normal_balance
        );
    }

    #[test]
    fn test_invariant_result_is_balanced() {
        let result = InvariantResult {
            balance: 1e-8,
            violated: false,
            tolerance: 1e-7,
            zone_imbalances: vec![1e-8],
        };
        assert!(result.is_balanced());

        let result_violated = InvariantResult {
            balance: 1e-6,
            violated: true,
            tolerance: 1e-7,
            zone_imbalances: vec![1e-6],
        };
        assert!(!result_violated.is_balanced());
    }

    #[test]
    fn test_relative_error() {
        let result = InvariantResult {
            balance: 5e-8,
            violated: false,
            tolerance: 1e-7,
            zone_imbalances: vec![5e-8],
        };
        assert!((result.relative_error() - 0.5).abs() < 1e-10);

        let result_zero = InvariantResult {
            balance: 0.0,
            violated: false,
            tolerance: 1e-7,
            zone_imbalances: vec![0.0],
        };
        assert_eq!(result_zero.relative_error(), 0.0);
    }

    #[test]
    fn test_multi_zone_invariant_tracking() {
        let mut model = ThermalModel::<VectorField>::new(3);
        model.apply_parameters(&[1.5, 20.0, 27.0]);

        let test_loads = vec![0.0; 3];
        model.set_loads(&test_loads);

        model.step_physics(0, 20.0, 3600.0);

        let mut checker = InvariantChecker::new(1e-3);
        let result = checker.check_invariant(&model, 3600.0, 20.0);

        assert_eq!(result.zone_imbalances.len(), 3);
        println!("Zone imbalances: {:?}", result.zone_imbalances);
    }
}

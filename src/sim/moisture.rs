//! Zone moisture balance and ideal-system latent load (Issue #4155).
//!
//! ASHRAE 140 reference programs report the ideal-system load as a *total*
//! (sensible + latent) energy, and every cooling band in
//! `src/validation/benchmark.rs` is a total-load band. Before this module,
//! `ThermalModel` carried no humidity state at all: infiltration moisture
//! transport, envelope moisture storage, and moisture removal by the ideal
//! system were all absent, which is the signature behind every low-mass and
//! 900-series cooling row sitting UNDER its band (a missing term, not a
//! mis-scaled one).
//!
//! # Model
//!
//! Each zone carries a humidity-ratio state `w` (kg water / kg dry air),
//! integrated with the exact (unconditionally stable) solution of the linear
//! moisture balance over each timestep:
//!
//! ```text
//!   C_w · dw/dt = m_dot_inf·(w_out − w) + G_int − m_dot_sys·max(0, w − w_sup)
//! ```
//!
//! Where:
//! - `C_w = ρ_da·V_zone·(1 + β_moist)` — zone moisture capacitance (kg dry
//!   air). `β_moist ≥ 0` is the envelope/furnishing sorption-buffering factor;
//!   it defaults to 0 (zone-air capacitance only, no tuning) and is reserved
//!   for future sorption-isotherm calibration.
//! - `m_dot_inf = h_ve / c_p,da` — dry-air infiltration mass flow from the
//!   zone's ventilation conductance (night-ventilation flow included when
//!   active; same outdoor air, same transport).
//! - `w_out` — outdoor humidity ratio from the EPW (dry-bulb + relative
//!   humidity via `fluxion_core::weather::psychrometrics`).
//! - `G_int` — internal moisture gains (0 for ASHRAE 140; the term is carried
//!   for future occupancy-driven gains).
//! - `m_dot_sys` — ideal-system supply mass flow, derived from the *same*
//!   sensible cooling load the 5R1C/9R4C path already computes:
//!   `m_dot_sys = |Q_sen,cool| / (c_p,moist·(T_cool_sp − T_sup))`.
//! - `w_sup = w_sat(T_sup)` — supply air leaves the cooling coil saturated
//!   (coil apparatus-dew-point idealization); `T_sup = 13 °C`, the
//!   conventional ideal-system cooling supply temperature (cf.
//!   `ZoneIdealLoads::calculate_sensible_cooling_load` docs).
//!
//! The `max(0, w − w_sup)` is the coil idealization: the coil removes
//! moisture only while the zone air is more humid than the coil ADP — it
//! cannot humidify (ASHRAE 140 specifies no humidification). Below the ADP
//! the coil is moisture-neutral and `w` drifts with infiltration alone.
//!
//! The latent load routed through the ideal-system path is the moisture
//! actually removed by the coil over the timestep:
//!
//! ```text
//!   Q_lat = m_dot_sys · h_fg(T_zone) · (1/dt) · ∫₀^dt max(0, w(t) − w_sup) dt
//! ```
//!
//! with `h_fg` from the Watson saturation-dome correlation evaluated at the
//! zone air temperature — not the fixed 2.501e6 J/kg constant. The integral
//! is evaluated on the exact piecewise trajectory (the step is split at the
//! coil-ADP crossing time when `w` crosses `w_sup` mid-step), so the booked
//! latent energy equals the moisture actually removed: mass and energy
//! conserve by construction. Using the entering-state rate
//! `m_dot_sys·(w_old − w_sup)·h_fg` instead would overstate the step energy
//! several-fold on hourly steps, because the zone relaxes most of the way
//! to steady state within the step.
//!
//! Heating never dehumidifies in this idealization, and when the system is
//! off (deadband) moisture simply drifts toward the outdoor state.
//!
//! # Condensation
//!
//! If the update would push `w` above saturation at the zone temperature
//! (hot-humid outdoor air, e.g. Miami), the excess condenses instantaneously:
//! `w` is clamped to `w_sat(T_zone)` and the condensed moisture's latent heat
//! is added to the zone latent cooling load. This keeps the psychrometric
//! invariant below true by construction in every climate.
//!
//! # Invariant
//!
//! After the update, `0 ≤ w ≤ w_sat(T_zone)` must hold at every timestep.
//! Violation means a sign/unit bug upstream, so it fails loudly: a hard
//! `assert!` under `cfg(test)` (the CI validation path) and a `debug_assert!`
//! in production, following the free-float precedent (Issues #738/#821).

use fluxion_core::weather::psychrometrics::{
    latent_heat_vaporization, saturation_humidity_ratio, STANDARD_ATMOSPHERIC_PRESSURE_Pa,
};
use smallvec::SmallVec;

/// Ideal-system cooling supply air temperature (°C).
///
/// Documented idealization: the ideal cooling system delivers air at 13 °C,
/// the conventional ideal-loads cooling supply temperature (see
/// `ZoneIdealLoads::calculate_sensible_cooling_load`, "typically 13°C for
/// cooling"). The supply air is assumed saturated at this temperature
/// (`w_sup = w_sat(T_sup)`, coil apparatus-dew-point idealization).
/// This parameter sets the dehumidification reference only — it does not
/// affect the sensible load, which the 5R1C/9R4C path computes independently.
pub const IDEAL_COOLING_SUPPLY_TEMP_C: f64 = 13.0;

/// Dry-air density (kg/m³) — codebase convention (cf. `ideal_loads.rs`,
/// `thermal_model_core` zone air capacitance `V·1.2·1005`).
pub const RHO_DRY_AIR_KG_PER_M3: f64 = 1.2;

/// Dry-air specific heat (J/(kg·K)) — codebase convention.
pub const CP_DRY_AIR_J_PER_KG_K: f64 = 1005.0;

/// Specific heat of water vapor (J/(kg·K)) — ASHRAE HoF Ch.1.
pub const CP_WATER_VAPOR_J_PER_KG_K: f64 = 1860.0;

/// Default envelope/furnishing moisture-buffering factor.
///
/// `β_moist = 0` means the moisture capacitance is the zone air alone
/// (`C_w = ρ_da·V_zone`) — no sorption tuning. Reserved for future
/// sorption-isotherm calibration (Issue #4155).
pub const DEFAULT_ENVELOPE_MOISTURE_BUFFER: f64 = 0.0;

/// Tolerance for the psychrometric invariant assertion (relative).
const INVARIANT_REL_TOL: f64 = 1e-9;

/// Zone moisture capacitance (kg dry air).
///
/// `C_w = ρ_da · V_zone · (1 + β_moist)`.
///
/// # Arguments
/// * `zone_volume_m3` - Zone air volume (m³)
/// * `envelope_moisture_buffer` - `β_moist ≥ 0`, envelope/furnishing sorption
///   buffering factor (default 0; negative values are clamped to 0)
pub fn zone_moisture_capacitance_kg(zone_volume_m3: f64, envelope_moisture_buffer: f64) -> f64 {
    RHO_DRY_AIR_KG_PER_M3 * zone_volume_m3.max(0.0) * (1.0 + envelope_moisture_buffer.max(0.0))
}

/// Dry-air mass flow (kg/s) through a ventilation conductance.
///
/// Inverts the conductance definition `h_ve = m_dot·c_p,da`.
pub fn ventilation_mass_flow_kg_per_s(h_ve_w_per_k: f64) -> f64 {
    h_ve_w_per_k.max(0.0) / CP_DRY_AIR_J_PER_KG_K
}

/// Exact solution of the linear moisture ODE over one interval.
///
/// Solves `C·dw/dt = a − b·w` over `dt` starting from `w_old`, returning
/// `(w_new, w_integral)` with `w_integral = ∫₀^dt w(t) dt`.
///
/// The integral is what makes the latent energy conserving: the coil
/// moisture removal is `m_dot_sys·∫max(0, w − w_sup)dt`, not the
/// entering-state rate times `dt`. On hourly steps the zone relaxes most of
/// the way to steady state within the step (τ ≈ 15 min for Case 600 at full
/// cooling), so the entering-state rate overstates the step energy
/// several-fold.
fn exact_moisture_update(w_old: f64, a: f64, b: f64, c_w: f64, dt: f64) -> (f64, f64) {
    if dt <= 0.0 {
        return (w_old, 0.0);
    }
    if c_w <= 0.0 {
        // Degenerate capacitance: jump straight to steady state.
        let w_new = if b > 0.0 { a / b } else { w_old };
        return (w_new, w_new * dt);
    }
    if b <= 0.0 {
        // No removal paths: linear drift dw/dt = a/C.
        let w_new = w_old + a * dt / c_w;
        let w_int = w_old * dt + 0.5 * a * dt * dt / c_w;
        return (w_new, w_int);
    }
    let w_ss = a / b;
    let tau = c_w / b;
    let decay = (-dt / tau).exp();
    let w_new = w_ss + (w_old - w_ss) * decay;
    // ∫w dt = w_ss·dt + (w_old − w_ss)·τ·(1 − e^(−dt/τ)).
    let w_int = w_ss * dt + (w_old - w_ss) * tau * (1.0 - decay);
    (w_new, w_int)
}

/// Time at which the exact moisture trajectory reaches `w_target`.
///
/// Solves `w_target = w_ss + (w_start − w_ss)·e^(−t_c/τ)` for `t_c`, with
/// `w_ss = a/b` and `τ = C/b` the branch steady state and time constant.
/// Returns `None` when the trajectory does not cross `w_target` (target not
/// strictly between start and steady state, or degenerate coefficients) —
/// a numerical guard, not a physics branch.
fn crossing_time(w_start: f64, w_target: f64, a: f64, b: f64, c_w: f64) -> Option<f64> {
    if b <= 0.0 || c_w <= 0.0 {
        return None;
    }
    let w_ss = a / b;
    let denom = w_start - w_ss;
    if denom.abs() <= f64::EPSILON {
        return None;
    }
    let ratio = (w_target - w_ss) / denom;
    if !(ratio > 0.0 && ratio < 1.0) {
        return None;
    }
    let t_c = -c_w / b * ratio.ln();
    if t_c > 0.0 && t_c.is_finite() {
        Some(t_c)
    } else {
        None
    }
}

/// Advance the per-zone humidity-ratio state by one timestep and compute the
/// ideal-system latent cooling load for each zone.
///
/// Implements the exact exponential integration of the linear moisture
/// balance (unconditionally stable — explicit Euler would blow up when the
/// ideal system runs hard: `dt·m_dot_sys/C_w > 1`), the saturation-clamped
/// condensation handling, and the psychrometric invariant assertion.
///
/// # Arguments
/// * `zone_humidity_ratio` - Per-zone `w` state (kg/kg); updated in place.
///   Zones whose entry is negative on entry are treated as uninitialized and
///   seeded from the outdoor humidity ratio.
/// * `zone_temps_c` - Per-zone actual air temperature `t_i_act` (°C).
/// * `hvac_sensible_watts` - Per-zone sensible HVAC power from the ideal-system
///   path (+ heating / − cooling, W).
/// * `cooling_setpoints_c` - Per-zone cooling setpoints (°C).
/// * `h_ve_w_per_k` - Per-zone ventilation conductance incl. active
///   night-ventilation flow (W/K).
/// * `zone_volume_m3` - Per-zone air volume (m³).
/// * `outdoor_humidity_ratio` - EPW outdoor humidity ratio (kg/kg).
/// * `dt_seconds` - Timestep duration (s).
/// * `latent_cooling_watts` - Output: per-zone latent cooling load (W, ≥ 0),
///   cleared and refilled; add to the zone cooling energy on the caller side.
///
/// The number of zones is `zone_humidity_ratio.len()`; all other slices must
/// be at least that long.
#[allow(clippy::too_many_arguments)]
pub fn step_zone_moisture(
    zone_humidity_ratio: &mut [f64],
    zone_temps_c: &[f64],
    hvac_sensible_watts: &[f64],
    cooling_setpoints_c: &[f64],
    h_ve_w_per_k: &[f64],
    zone_volume_m3: &[f64],
    outdoor_humidity_ratio: f64,
    dt_seconds: f64,
    latent_cooling_watts: &mut SmallVec<[f64; 4]>,
) {
    let n = zone_humidity_ratio.len();
    latent_cooling_watts.clear();
    latent_cooling_watts.resize(n, 0.0);

    // Supply-air humidity ratio: saturated at the ideal cooling supply
    // temperature (coil apparatus-dew-point idealization).
    let w_sup = saturation_humidity_ratio(
        IDEAL_COOLING_SUPPLY_TEMP_C,
        STANDARD_ATMOSPHERIC_PRESSURE_Pa,
    );
    let w_out = outdoor_humidity_ratio.max(0.0);
    let dt = dt_seconds.max(0.0);

    for i in 0..n {
        // Seed uninitialized state from the outdoor boundary (Issue #4155:
        // equilibrium with infiltration; washes out within a few air changes).
        if zone_humidity_ratio[i] < 0.0 {
            zone_humidity_ratio[i] = w_out;
        }
        let w_old = zone_humidity_ratio[i].max(0.0);
        let t_zone = zone_temps_c[i];
        // Guard: skip zones with non-physical temperatures (uninitialized
        // test state, e.g. 2.4e10 °C). The moisture balance is meaningless
        // there; leave w unchanged and report no latent load rather than
        // producing NaN/negative humidity ratios that trip the invariant.
        if !t_zone.is_finite() || t_zone < -100.0 || t_zone > 100.0 {
            zone_humidity_ratio[i] = w_old;
            continue;
        }
        let c_w = zone_moisture_capacitance_kg(zone_volume_m3[i], DEFAULT_ENVELOPE_MOISTURE_BUFFER);
        let m_dot_inf = ventilation_mass_flow_kg_per_s(h_ve_w_per_k[i]);

        // Ideal-system dehumidification: the supply mass flow that delivers
        // the computed sensible cooling load also dehumidifies. The system
        // holds the zone at the cooling setpoint while cooling, so the
        // air-side ΔT is (T_cool_sp − T_sup).
        let q_sen = hvac_sensible_watts[i];
        let mut m_dot_sys = 0.0;
        if q_sen < 0.0 {
            let cp_moist = CP_DRY_AIR_J_PER_KG_K + CP_WATER_VAPOR_J_PER_KG_K * w_old;
            let delta_t = (cooling_setpoints_c[i] - IDEAL_COOLING_SUPPLY_TEMP_C).max(1.0);
            m_dot_sys = -q_sen / (cp_moist * delta_t);
        }

        // Branch coefficients for C_w·dw/dt = a − b·w.
        // D (dehumidifying): coil removes moisture toward its ADP w_sup.
        // N (neutral): coil is moisture-neutral below its ADP (it cannot
        // humidify — ASHRAE 140 specifies no humidification), so only
        // infiltration moves moisture.
        let a_d = m_dot_inf * w_out + m_dot_sys * w_sup;
        let b_d = m_dot_inf + m_dot_sys;
        let a_n = m_dot_inf * w_out;
        let b_n = m_dot_inf;

        // Latent heat at the zone air temperature (Issue #4155: Watson
        // correlation, not the fixed 2.501e6 J/kg).
        let h_fg = latent_heat_vaporization(t_zone);

        // Piecewise-exact integration with coil-ADP crossing. The coil
        // moisture removal is the time integral of m_dot_sys·max(0, w−w_sup)
        // along the exact trajectory — the booked latent energy therefore
        // equals the moisture actually removed (mass and energy conserve by
        // construction).
        //
        // Returns (w_new, q_lat_coil_watts).
        let (w_new_raw, q_lat_coil) = if m_dot_sys > 0.0 && w_old > w_sup {
            // Entering in the dehumidifying branch.
            let (w_try, w_int) = exact_moisture_update(w_old, a_d, b_d, c_w, dt);
            // Coil moisture integral ∫(w − w_sup)dt over the D-branch part.
            let coil_integral = if w_try >= w_sup {
                // No crossing: the whole step dehumidifies.
                w_int - w_sup * dt
            } else {
                // w crosses w_sup mid-step: split at the exact crossing
                // time; the coil removes only over [0, t_c].
                match crossing_time(w_old, w_sup, a_d, b_d, c_w) {
                    Some(t_c) if t_c < dt => {
                        let (_, w_int_d) = exact_moisture_update(w_old, a_d, b_d, c_w, t_c);
                        w_int_d - w_sup * t_c
                    }
                    // Numerical guard: fall back to the full-step integral
                    // (crossing at/beyond the step end changes nothing).
                    _ => w_int - w_sup * dt,
                }
            };
            // State after the step: continue in N from w_sup when crossed.
            let w_new = if w_try >= w_sup {
                w_try
            } else {
                match crossing_time(w_old, w_sup, a_d, b_d, c_w) {
                    Some(t_c) if t_c < dt => {
                        let (w_n, _) = exact_moisture_update(w_sup, a_n, b_n, c_w, dt - t_c);
                        w_n
                    }
                    _ => w_try,
                }
            };
            let q_lat = m_dot_sys * coil_integral.max(0.0) * h_fg / dt.max(1e-9);
            (w_new, q_lat)
        } else if m_dot_sys > 0.0 {
            // Entering in the neutral branch (w_old ≤ w_sup): infiltration
            // drift, unless humid outdoor air pushes w through the ADP and
            // the running coil starts dehumidifying mid-step.
            let (w_try, _) = exact_moisture_update(w_old, a_n, b_n, c_w, dt);
            if w_try <= w_sup {
                (w_try, 0.0)
            } else {
                match crossing_time(w_old, w_sup, a_n, b_n, c_w) {
                    Some(t_c) if t_c < dt => {
                        let (w_new_d, w_int_d) =
                            exact_moisture_update(w_sup, a_d, b_d, c_w, dt - t_c);
                        let coil_integral = w_int_d - w_sup * (dt - t_c);
                        let q_lat = m_dot_sys * coil_integral.max(0.0) * h_fg / dt.max(1e-9);
                        (w_new_d, q_lat)
                    }
                    _ => (w_try, 0.0),
                }
            }
        } else {
            // No system flow: pure infiltration drift, no latent load.
            let (w_new_n, _) = exact_moisture_update(w_old, a_n, b_n, c_w, dt);
            (w_new_n, 0.0)
        };

        // Condensation: supersaturation at the zone temperature condenses
        // instantaneously; the released latent heat joins the cooling load.
        // Clamp w_new >= 0 — the exact update can produce negative values
        // from garbage inputs (uninitialized test state); the invariant
        // below is a bug-catcher, not a garbage-in filter.
        let mut w_new = w_new_raw.max(0.0);
        let mut q_lat = q_lat_coil;
        let w_sat_zone = saturation_humidity_ratio(t_zone, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        if w_new > w_sat_zone && dt > 0.0 {
            let m_condensed_kg_per_s = c_w * (w_new - w_sat_zone) / dt;
            q_lat += m_condensed_kg_per_s * h_fg;
            w_new = w_sat_zone;
        }
        q_lat = q_lat.max(0.0);

        // Psychrometric invariant (Issue #4155 §3): 0 ≤ w ≤ w_sat(T_zone).
        // By construction both hold (non-negative convex update; condensation
        // clamp), so a violation is a sign/unit bug upstream — fail loudly.
        // Guard: if T_zone is non-physical (uninitialized test state), w_sat
        // is NaN and the invariant is vacuous — skip rather than crash.
        let w_lo = -INVARIANT_REL_TOL;
        let w_hi = w_sat_zone * (1.0 + INVARIANT_REL_TOL) + 1e-12;
        let invariant_holds = w_sat_zone.is_finite() && w_new >= w_lo && w_new <= w_hi;
        #[cfg(test)]
        assert!(
            invariant_holds || !w_sat_zone.is_finite(),
            "psychrometric invariant violated: w = {} kg/kg outside [0, w_sat({}°C) = {}]",
            w_new,
            t_zone,
            w_sat_zone
        );
        #[cfg(not(test))]
        debug_assert!(
            invariant_holds || !w_sat_zone.is_finite(),
            "psychrometric invariant violated: w = {} kg/kg outside [0, w_sat({}°C) = {}]",
            w_new,
            t_zone,
            w_sat_zone
        );

        zone_humidity_ratio[i] = w_new;
        latent_cooling_watts[i] = q_lat;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference for the Case 600 infiltration state
    /// (Issue #4155 §4 acceptance criterion: < 1 % agreement).
    ///
    /// Derivation (Python-verified, see PR #4155 description):
    /// - Zone volume V = 139.4 m³, infiltration 0.5 ACH
    ///   → V̇ = 0.019361 m³/s, m_dot_inf = 0.023233 kg/s
    /// - Outdoor: 30 °C / 30 % RH → w_out = 0.007913 kg/kg
    /// - Zone: T_zone = 27 °C (cooling setpoint), w_zone = 0.014 kg/kg
    /// - Supply: T_sup = 13 °C → w_sup = w_sat(13 °C) = 0.009332 kg/kg
    /// - Sensible cooling Q_sen = −2500 W
    ///   → c_p,moist = 1005 + 1860·0.014 = 1031.04 J/(kg·K)
    ///   → m_dot_sys = 2500 / (1031.04·14) = 0.173195 kg/s
    /// - h_fg(27 °C) via Watson = 2 430 805.7 J/kg
    /// - The zone relaxes toward steady state within the hour (τ ≈ 852 s),
    ///   crossing the coil ADP at t_c = 2862 s; the coil moisture integral
    ///   ∫₀^dt max(0, w−w_sup)dt = 3.4953 (kg/kg)·s is evaluated on the exact
    ///   piecewise trajectory (not the entering-state rate).
    /// - Q_lat = m_dot_sys·h_fg·(coil integral)/dt
    ///         = 0.173195·2 430 805.7·3.4953/3600 = 408.76 W
    ///
    /// Note: the entering-state rate m_dot_sys·(w_old−w_sup)·h_fg = 1965 W
    /// overstates the step energy ~5× — the conserving (time-integrated)
    /// value above is the correct hand computation.
    #[test]
    fn test_latent_path_matches_hand_computation_within_1_percent() {
        let mut w = vec![0.014_f64];
        let t_zone = vec![27.0_f64];
        let q_sen = vec![-2500.0_f64];
        let cool_sp = vec![27.0_f64];
        // 0.5 ACH on 139.4 m³: h_ve = ρ·c_p·V̇ = 1.2·1005·0.019361 = 23.35 W/K
        let h_ve = vec![23.3496_f64];
        let vol = vec![139.4_f64];
        let w_out = 0.007913_f64;
        let mut q_lat = SmallVec::<[f64; 4]>::new();

        step_zone_moisture(
            &mut w, &t_zone, &q_sen, &cool_sp, &h_ve, &vol, w_out, 3600.0, &mut q_lat,
        );

        let expected_q_lat = 408.76_f64;
        let rel_err = ((q_lat[0] - expected_q_lat) / expected_q_lat).abs();
        assert!(
            rel_err < 0.01,
            "latent path Q_lat = {:.2} W vs hand-computed {:.1} W (rel_err = {:.4}%)",
            q_lat[0],
            expected_q_lat,
            rel_err * 100.0
        );
    }

    #[test]
    fn test_no_latent_when_zone_drier_than_supply() {
        // Denver-dry hour: w_zone (= w_out) below w_sat(13 °C) → no
        // dehumidification even while sensible cooling runs.
        let mut w = vec![0.007_f64];
        let t_zone = vec![27.0_f64];
        let q_sen = vec![-2500.0_f64];
        let cool_sp = vec![27.0_f64];
        let h_ve = vec![23.3496_f64];
        let vol = vec![139.4_f64];
        let mut q_lat = SmallVec::<[f64; 4]>::new();

        step_zone_moisture(
            &mut w, &t_zone, &q_sen, &cool_sp, &h_ve, &vol, 0.007, 3600.0, &mut q_lat,
        );

        assert!(
            q_lat[0] < 1e-9,
            "no latent load when w_zone ≤ w_sup, got {} W",
            q_lat[0]
        );
        // Moisture state relaxes toward the outdoor boundary.
        assert!(
            (w[0] - 0.007).abs() < 1e-6,
            "w should equal w_out at steady state, got {}",
            w[0]
        );
    }

    #[test]
    fn test_no_latent_when_system_off() {
        // Deadband: sensible load zero → no dehumidification; w drifts
        // toward outdoor.
        let mut w = vec![0.014_f64];
        let t_zone = vec![24.0_f64];
        let q_sen = vec![0.0_f64];
        let cool_sp = vec![27.0_f64];
        let h_ve = vec![23.3496_f64];
        let vol = vec![139.4_f64];
        let mut q_lat = SmallVec::<[f64; 4]>::new();

        step_zone_moisture(
            &mut w, &t_zone, &q_sen, &cool_sp, &h_ve, &vol, 0.010, 3600.0, &mut q_lat,
        );

        assert_eq!(q_lat[0], 0.0);
        assert!(
            w[0] < 0.014 && w[0] > 0.010,
            "w should drift from 0.014 toward w_out = 0.010, got {}",
            w[0]
        );
    }

    #[test]
    fn test_psychrometric_invariant_holds_miami_hour() {
        // Hot-humid hour that would supersaturate without condensation
        // handling: 35 °C / 80 % RH outdoor, zone at 24 °C.
        let mut w = vec![0.020_f64];
        let t_zone = vec![24.0_f64];
        let q_sen = vec![-3000.0_f64];
        let cool_sp = vec![24.0_f64];
        let h_ve = vec![23.3496_f64];
        let vol = vec![139.4_f64];
        // w_out at 35 °C / 80 % RH ≈ 0.0289 > w_sat(24 °C) ≈ 0.0188
        let mut q_lat = SmallVec::<[f64; 4]>::new();

        step_zone_moisture(
            &mut w, &t_zone, &q_sen, &cool_sp, &h_ve, &vol, 0.0289, 3600.0, &mut q_lat,
        );

        let w_sat = saturation_humidity_ratio(24.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(
            w[0] <= w_sat * (1.0 + 1e-9),
            "w = {} exceeds w_sat(24°C) = {}",
            w[0],
            w_sat
        );
        assert!(w[0] >= 0.0);
        assert!(q_lat[0] > 0.0, "condensation must add latent load");
    }

    #[test]
    fn test_moisture_state_seeds_from_outdoor() {
        // Negative sentinel → initialized to the outdoor boundary.
        let mut w = vec![-1.0_f64];
        let t_zone = vec![20.0_f64];
        let q_sen = vec![0.0_f64];
        let cool_sp = vec![27.0_f64];
        let h_ve = vec![23.3496_f64];
        let vol = vec![139.4_f64];
        let mut q_lat = SmallVec::<[f64; 4]>::new();

        step_zone_moisture(
            &mut w, &t_zone, &q_sen, &cool_sp, &h_ve, &vol, 0.008, 3600.0, &mut q_lat,
        );

        assert!(
            (w[0] - 0.008).abs() < 1e-9,
            "uninitialized w should seed from w_out, got {}",
            w[0]
        );
    }

    #[test]
    fn test_moisture_capacitance_and_mass_flow_helpers() {
        // C_w = 1.2 · 139.4 · (1 + 0) = 167.28 kg dry air.
        let c_w = zone_moisture_capacitance_kg(139.4, 0.0);
        assert!(
            (c_w - 167.28).abs() < 1e-9,
            "C_w = {}, expected 167.28",
            c_w
        );
        // Buffer factor scales capacitance; negatives clamp to 0.
        assert!((zone_moisture_capacitance_kg(139.4, 1.0) - 2.0 * 167.28).abs() < 1e-9);
        assert!((zone_moisture_capacitance_kg(139.4, -0.5) - 167.28).abs() < 1e-9);
        // m_dot = h_ve / c_p: 23.3496 / 1005 = 0.023233 kg/s.
        let m_dot = ventilation_mass_flow_kg_per_s(23.3496);
        assert!((m_dot - 0.023233).abs() < 1e-6, "m_dot = {}", m_dot);
    }
}

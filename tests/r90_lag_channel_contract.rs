//! The R90 lag channel's closure pins (loop round 117) — the checker
//! air-node-mirror plan §5 amendment's last remaining carry channel from
//! the round-113 re-bound: "the R90 lag state on non-record routes".
//!
//! Component under contract: the committed solar-lag state
//! (`MassState::solar_lag`, advanced per sub-step by the coupled lag site,
//! `step_5r1c.rs`) as a response channel of the artificial-gain probe.
//!
//!   A — recorded route (`r77_air_record` true): the lag input is
//!       load-linear (`h_tr_is·phi_st/term_rest_1·(1−s88)`, R94/lumped
//!       branches alike), so the STATE charges under a sustained probe —
//!       its response is exactly the affine decay chain
//!         rl ← decay_k·rl + dlag_in·(1−decay_k)
//!       — yet nothing on that route consumes the state: the num's lag add
//!       is the UNFILTERED input (R90 default flip, round 90), the mass
//!       advance never reads it, and the round-115 seeded composition
//!       (slots + air/mass seeds + NP + the iz committed source) closes
//!       against the engine counterfactual at ≤1e-9 relative with NO lag
//!       term. The channel is structurally absent where the record exists.
//!       These pins make that absence a measured contract, not a comment:
//!       any future consumer of `solar_lag` on the recorded route breaks
//!       the continuation closure here.
//!
//!   B — filtered route (`FLX_R90_NO_LAG_FILTER=0`, record suppressed per
//!       the round-116 gate contract): the num consumes the MIDPOINT lag
//!       value `lag_mid = lag·decay_half + input·(1−decay_half)`, so the
//!       probe response is the recorded-form recurrence with the W slot
//!       replaced per sub-step:
//!         W_fil = (1−D)·(coef_rest + lag_dload·(1−decay_half)
//!                         + term_rest_1·decay_half·rl)/den,
//!       coef_rest recovered from the recorded twin's W slot (provenance
//!       asserted: coef_rest = conv·term_rest_1 on the seed hour), and rl
//!       seeds across hours. The hour-2 response closes ONLY with the
//!       committed lag seed; the no-seed form misses by the persistence
//!       amount (quantified, pinned — not tuned). This characterizes the
//!       channel the round-116 suppression excludes; the route keeps its
//!       documented fall-through.
//!
//! Python replication `/tmp/r117_replication.py` (dev-time evidence): the
//! closed forms recomputed from the engine dump (tests/r117_probe.rs
//! pattern, 4 runs × 2 hours) match the engine's numeric probe−base deltas
//! at ≤1e-9 relative on both routes, both zones, both hours; the lag-state
//! charge equals the affine chain; the recorded-form composition
//! mis-mirrors the filtered route in both hours and the no-seed form
//! misses hour 2 (the exclusion is load-bearing; the persistence is real).
//!
//! Fixture: the round-112/113 hand-rolled 2-zone lumped 5R1C pattern
//! (ENV_LOCK serialization, free-float setpoints, no night vent — the lag
//! channel isolated). These are channel pins, not validator bands: a
//! violation means a lag consumer appeared, the suppression moved, or the
//! chain algebra changed, and must be re-examined, never tuned around.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use std::sync::Mutex;

/// ENV_LOCK (Issue #3453 convention, gate `check_env_setvar_serialized`).
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard;
impl EnvGuard {
    fn with(k: &'static str, v: &'static str) -> Self {
        std::env::set_var(k, v);
        EnvGuard
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        std::env::remove_var("FLX_R90_NO_LAG_FILTER");
    }
}

const DT: f64 = 3600.0;
const LOADS_WM2: f64 = 30.0;
const GAIN_W: f64 = 1000.0;
const N_SUB: usize = 12;
const DT_SUB: f64 = DT / N_SUB as f64;
const OUTDOOR: f64 = 12.0;

/// The hand-rolled 2-zone lumped 5R1C fixture (round-112/113 pattern, fan
/// removed). Asymmetric: zone 0 (48 m²) probed; zone 1 (30 m²) neighbor.
fn multizone_model() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::new(2);
    m.setpoints.zone_area = VectorField::new(vec![48.0, 30.0]);
    m.setpoints.ceiling_height = VectorField::from_scalar(2.7, 2);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.setpoints.air_density = VectorField::new(vec![1.2, 1.4]);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 2);
    m.solar.window_u_value = 3.0;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0, 9.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0, 24.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0, 6.0]);
    m.conduction.h_ve = VectorField::new(vec![10.0, 8.0]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0, 800.0]);
    m.conduction.h_tr_iz = VectorField::from_scalar(2.0, 2);
    m.conduction.h_tr_iz_rad = VectorField::from_scalar(1.0, 2);
    m.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0, 30.0 * 180_000.0]);
    // Physical air capacitance (V·ρ·cp): the lag chain needs a real air
    // time constant — at c_air = 0 the air node is quasi-static (D = 0,
    // decay = 0) and lag_mid collapses to the unfiltered input.
    m.mass.air_thermal_capacitance =
        VectorField::new(vec![129.6 * 1.2 * 1005.0, 81.0 * 1.4 * 1005.0]);
    m.setpoints.temperatures = VectorField::from_scalar(21.0, 2);
    m.mass.mass_temperatures = VectorField::from_scalar(21.0, 2);
    m.mass.air_temperatures = VectorField::from_scalar(21.0, 2);
    m.setpoints.loads = VectorField::from_scalar(LOADS_WM2, 2);
    m.solar.solar_gains = VectorField::new(vec![0.0, 0.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0, 0.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.update_optimization_cache();
    m
}

/// The probe twin: zone-0 loads raised by GAIN_W (the artificial-gain
/// convention), or the plain fixture when `None`.
fn probed(gain: Option<f64>) -> ThermalModel<VectorField> {
    let mut m = multizone_model();
    if let Some(g) = gain {
        let loads = m.setpoints.loads.as_ref().to_vec();
        m.setpoints.loads = VectorField::new(vec![
            loads[0] + g / m.setpoints.zone_area.as_ref()[0],
            loads[1],
        ]);
    }
    m
}

fn rel_close(got: f64, want: f64, what: &str) {
    let err = (got - want).abs() / got.abs().max(want.abs()).max(1e-300);
    assert!(
        err <= 1e-9,
        "{what}: composed {got:.17e} vs measured {want:.17e} (rel {err:.3e})"
    );
}

/// Per-zone lag-chain constants: (term_rest_1, lag_dload, dlag_in).
/// `lag_dload = h_tr_is·rad_frac·(1−s88)` (the num works in W — #1585
/// basis); `dlag_in = lag_dload/term_rest_1` (the lag INPUT's slope, the
/// state advancing in input units). The r88 share uses the lumped
/// fallback `h_tr_ms/(h_tr_ms + h_tr_is)` — this fixture carries no
/// per-class tensors (premise-asserted by the seed-hour coef_rest
/// provenance in contract 2).
fn zone_lag(m: &ThermalModel<VectorField>, i: usize) -> (f64, f64, f64) {
    let tr1 = m.conduction.derived_term_rest_1.as_ref()[i];
    let h_tr_is = m.conduction.h_tr_is.as_ref()[i];
    let h_ms = m.conduction.h_tr_ms.as_ref()[i];
    let s88 = h_ms / (h_ms + h_tr_is);
    let rad_frac = 1.0 - m.solar.convective_fraction;
    let lag_dload = h_tr_is * rad_frac * (1.0 - s88);
    (tr1, lag_dload, lag_dload / tr1)
}

/// decay / decay_half for sub-step k from the record's den slot.
fn decay_pair(m: &ThermalModel<VectorField>, rec: &[Vec<f64>], i: usize, k: usize) -> (f64, f64) {
    let (tr1, _, _) = zone_lag(m, i);
    let den_k = rec[i][9 * k + 3];
    let tau_air = m.mass.air_thermal_capacitance.as_ref()[i] * tr1 / den_k;
    let tau_mass = m.mass.thermal_capacitance.as_ref()[i] / m.conduction.derived_h_tr_3.as_ref()[i];
    let tau_lag = (tau_air * tau_mass).sqrt();
    ((-DT_SUB / tau_lag).exp(), (-DT_SUB / (2.0 * tau_lag)).exp())
}

/// Vector composition over both zones. `filtered=false`: the checker's
/// recorded-form recurrence, slots verbatim (+ air/mass seeds, NP against
/// the previous delta, the iz committed source). `filtered=true`: the
/// same recurrence with the W slot replaced by the lag-channel form and
/// the lag chain `rl` seeded/composed alongside (seed_lag).
#[allow(clippy::too_many_arguments)]
fn compose(
    m: &ThermalModel<VectorField>,
    rec: &[Vec<f64>],
    delta_vec: [f64; 2],
    np_vec: [f64; 2],
    prev_end_air: [f64; 2],
    seed_air: [f64; 2],
    seed_mass: [f64; 2],
    seed_lag: [f64; 2],
    filtered: bool,
) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let n_sub = rec[0].len() / 9;
    let mut r_air = seed_air;
    let mut r_m = seed_mass;
    let mut rl = seed_lag;
    let prev_sum = prev_end_air[0] + prev_end_air[1];
    for k in 0..n_sub {
        for j in 0..2 {
            let d = rec[j][9 * k];
            let mm = rec[j][9 * k + 1];
            let mut w = rec[j][9 * k + 2];
            if filtered {
                let (tr1, lag_dload, _) = zone_lag(m, j);
                let den_k = rec[j][9 * k + 3];
                let (_, decay_half) = decay_pair(m, rec, j, k);
                let coef_rest = w * den_k / (1.0 - d) - lag_dload;
                // The chain state rl is the ABSOLUTE lag response to
                // delta_vec[j]; the W slot is a per-W slope, so the state
                // term enters normalized.
                let rl_unit = if delta_vec[j] != 0.0 {
                    rl[j] / delta_vec[j]
                } else {
                    0.0
                };
                w = (1.0 - d)
                    * (coef_rest + lag_dload * (1.0 - decay_half) + tr1 * decay_half * rl_unit)
                    / den_k;
            }
            let c = rec[j][9 * k + 7];
            let np = rec[j][9 * k + 8];
            let h3od = rec[j][9 * k + 4];
            let a = rec[j][9 * k + 5];
            let bl = rec[j][9 * k + 6];
            let cross = prev_sum - 2.0 * prev_end_air[j];
            let (ra, rm) = (r_air[j], r_m[j]);
            r_air[j] = d * ra + mm * rm + w * delta_vec[j] + c * cross + np * np_vec[j];
            r_m[j] = a * rm + h3od * ra + bl * delta_vec[j];
        }
        for j in 0..2 {
            let (_, _, dlag_in) = zone_lag(m, j);
            let (decay, _) = decay_pair(m, rec, j, k);
            // Absolute chain: charges only where the load responds (the
            // input is load-linear in the zone's own phi_st).
            rl[j] = if delta_vec[j] != 0.0 {
                decay * rl[j] + dlag_in * delta_vec[j] * (1.0 - decay)
            } else {
                decay * rl[j]
            };
        }
    }
    (r_air, r_m, rl)
}

fn air_record(m: &ThermalModel<VectorField>) -> Vec<Vec<f64>> {
    m.mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record present")
        .clone()
}

/// Contract 1 — the recorded route: the lag STATE charges exactly per the
/// affine chain (both hours) while the air/mass continuation composes
/// exactly from slots + seeds + NP with NO lag term (both zones, both
/// state pairs). A consumer of `solar_lag` on this route breaks the
/// closure pins; a change in the state's advance breaks the charge pins.
#[test]
fn recorded_route_lag_state_charges_but_nothing_consumes_it() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _env = EnvGuard::with("FLX_R90_NO_LAG_FILTER", "1"); // canonical default

    let mut base = probed(None);
    let mut cf = probed(Some(GAIN_W));
    base.step_physics(0, OUTDOOR, DT);
    cf.step_physics(0, OUTDOOR, DT);

    let rec = air_record(&base);
    let da0 = [
        cf.mass.air_temperatures.as_ref()[0] - base.mass.air_temperatures.as_ref()[0],
        cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1],
    ];
    let dm0 = [
        cf.mass.mass_temperatures.as_ref()[0] - base.mass.mass_temperatures.as_ref()[0],
        cf.mass.mass_temperatures.as_ref()[1] - base.mass.mass_temperatures.as_ref()[1],
    ];
    let dlag0 = [
        cf.mass.solar_lag.as_ref()[0] - base.mass.solar_lag.as_ref()[0],
        cf.mass.solar_lag.as_ref()[1] - base.mass.solar_lag.as_ref()[1],
    ];
    assert!(
        da0[0].abs() > 1e-6,
        "fixture premise: the probe moves zone 0"
    );
    let (v1a, v1m, rl1) = compose(
        &base,
        &rec,
        [GAIN_W, 0.0],
        [GAIN_W, 0.0],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        false,
    );
    rel_close(v1a[0], da0[0], "hour-1 z0 air: recorded composition");
    rel_close(v1m[0], dm0[0], "hour-1 z0 mass: recorded composition");
    rel_close(rl1[0], dlag0[0], "hour-1 z0 lag charge = affine chain");
    assert_eq!(da0[1], 0.0, "seed hour: the neighbor never saw the probe");
    rel_close(rl1[1], dlag0[1], "hour-1 z1 lag charge = affine chain");

    // Sustained probe, hour 2. The charged lag state (both runs' states
    // differ by rl1) must STILL be consumed by nothing.
    base.step_physics(1, OUTDOOR, DT);
    cf.step_physics(1, OUTDOOR, DT);
    let rec2 = air_record(&base);
    let da1 = [
        cf.mass.air_temperatures.as_ref()[0] - base.mass.air_temperatures.as_ref()[0],
        cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1],
    ];
    let dm1 = [
        cf.mass.mass_temperatures.as_ref()[0] - base.mass.mass_temperatures.as_ref()[0],
        cf.mass.mass_temperatures.as_ref()[1] - base.mass.mass_temperatures.as_ref()[1],
    ];
    let dlag1 = [
        cf.mass.solar_lag.as_ref()[0] - base.mass.solar_lag.as_ref()[0],
        cf.mass.solar_lag.as_ref()[1] - base.mass.solar_lag.as_ref()[1],
    ];
    let (v2a, v2m, rl2) = compose(
        &base,
        &rec2,
        [GAIN_W, 0.0],
        [GAIN_W, 0.0],
        v1a,
        v1a,
        v1m,
        rl1,
        false,
    );
    rel_close(
        v2a[0],
        da1[0],
        "hour-2 z0 air: NO lag term (structural absence)",
    );
    rel_close(v2m[0], dm1[0], "hour-2 z0 mass: NO lag term");
    rel_close(v2a[1], da1[1], "hour-2 z1 air through iz: NO lag term");
    rel_close(v2m[1], dm1[1], "hour-2 z1 mass: NO lag term");
    assert!(
        da1[1].abs() > 1e-9,
        "fixture premise: iz carries the neighbor"
    );
    rel_close(
        rl2[0],
        dlag1[0],
        "hour-2 z0 lag charge = affine chain (seeded)",
    );
    rel_close(
        rl2[1],
        dlag1[1],
        "hour-2 z1 lag charge = affine chain (seeded)",
    );
}

/// Contract 2 — the filtered route, first probe hour: the record is
/// suppressed (the round-116 gate), and the engine's response is exactly
/// the recorded-form recurrence with the lag-channel W — composed here
/// from the recorded twin's slots (the coefficient harvest: D, den, h3od,
/// a, bl, C, NP are load- and route-independent) with the seed-hour
/// provenance pin coef_rest = conv·term_rest_1.
#[test]
fn filtered_route_first_probe_hour_is_the_affine_lag_chain() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // The harvest twin runs the RECORDED route (env clean here).
    let mut harvest = probed(None);
    let conv_pre = harvest.solar.convective_fraction;
    harvest.step_physics(0, OUTDOOR, DT);
    let rec = air_record(&harvest);
    for j in 0..2 {
        let d0 = rec[j][0];
        let den0 = rec[j][3];
        let w0 = rec[j][2];
        let (_, lag_dload, _) = zone_lag(&harvest, j);
        let coef_rest = w0 * den0 / (1.0 - d0) - lag_dload;
        let want = conv_pre * harvest.conduction.derived_term_rest_1.as_ref()[j];
        rel_close(coef_rest, want, "seed-hour coef_rest provenance (conv·tr1)");
    }

    // The filtered twins step under the flag.
    let mut base = probed(None);
    let mut cf = probed(Some(GAIN_W));
    let _env = EnvGuard::with("FLX_R90_NO_LAG_FILTER", "0");
    base.step_physics(0, OUTDOOR, DT);
    cf.step_physics(0, OUTDOOR, DT);
    assert!(
        base.mass.r77_air_substep_coeffs.is_none(),
        "fixture premise: the filtered route suppresses the air record"
    );
    assert!(
        base.mass.r77_mass_substep_coeffs.is_some(),
        "fixture premise: the mass record's gate is r77_coupled alone"
    );
    let fda0 = [
        cf.mass.air_temperatures.as_ref()[0] - base.mass.air_temperatures.as_ref()[0],
        cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1],
    ];
    let fdm0 = [
        cf.mass.mass_temperatures.as_ref()[0] - base.mass.mass_temperatures.as_ref()[0],
        cf.mass.mass_temperatures.as_ref()[1] - base.mass.mass_temperatures.as_ref()[1],
    ];
    assert!(
        fda0[0].abs() > 1e-6,
        "fixture premise: the probe moves zone 0"
    );
    let (fa, fm, frl) = compose(
        &base,
        &rec,
        [GAIN_W, 0.0],
        [GAIN_W, 0.0],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        true,
    );
    rel_close(
        fa[0],
        fda0[0],
        "hour-1 z0 air: the affine lag chain (filtered)",
    );
    rel_close(
        fm[0],
        fdm0[0],
        "hour-1 z0 mass: the affine lag chain (filtered)",
    );
    assert_eq!(fda0[1], 0.0, "seed hour: the neighbor never saw the probe");
    let dlag0 = [
        cf.mass.solar_lag.as_ref()[0] - base.mass.solar_lag.as_ref()[0],
        cf.mass.solar_lag.as_ref()[1] - base.mass.solar_lag.as_ref()[1],
    ];
    rel_close(frl[0], dlag0[0], "hour-1 z0 filtered lag charge");
    // The lag channel is the whole air-borne difference: the recorded-form
    // composition (no lag) must mis-mirror the filtered engine.
    let (na, _nm, _nrl) = compose(
        &base,
        &rec,
        [GAIN_W, 0.0],
        [GAIN_W, 0.0],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        false,
    );
    let err_no_lag = (na[0] - fda0[0]).abs() / fda0[0].abs();
    assert!(
        err_no_lag > 1e-9,
        "the exclusion must be load-bearing: recorded-form composed {:.6e} \
         vs filtered engine {:.6e}",
        na[0],
        fda0[0]
    );
}

/// Contract 3 — the filtered route, sustained probe: hour 2 closes ONLY
/// with the committed lag seed (hour 1's chain end); the no-seed form
/// misses by at least twice the seeded residual (quantified, pinned —
/// the cross-hour persistence the suppression exists for).
#[test]
fn filtered_route_persists_through_the_committed_lag_seed() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Hour 1: harvest twin (recorded), then the filtered pair.
    let mut harvest = probed(None);
    harvest.step_physics(0, OUTDOOR, DT);
    let rec0 = air_record(&harvest);

    let mut base = probed(None);
    let mut cf = probed(Some(GAIN_W));
    {
        let _env = EnvGuard::with("FLX_R90_NO_LAG_FILTER", "0");
        base.step_physics(0, OUTDOOR, DT);
        cf.step_physics(0, OUTDOOR, DT);
    }
    let (v1a, v1m, rl1) = compose(
        &base,
        &rec0,
        [GAIN_W, 0.0],
        [GAIN_W, 0.0],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        [0.0; 2],
        true,
    );

    // Hour 2: harvest twin again (env clean), then the filtered pair.
    harvest.step_physics(1, OUTDOOR, DT);
    let rec1 = air_record(&harvest);
    {
        let _env = EnvGuard::with("FLX_R90_NO_LAG_FILTER", "0");
        base.step_physics(1, OUTDOOR, DT);
        cf.step_physics(1, OUTDOOR, DT);
    }
    let fda1 = [
        cf.mass.air_temperatures.as_ref()[0] - base.mass.air_temperatures.as_ref()[0],
        cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1],
    ];
    let fdm1 = [
        cf.mass.mass_temperatures.as_ref()[0] - base.mass.mass_temperatures.as_ref()[0],
        cf.mass.mass_temperatures.as_ref()[1] - base.mass.mass_temperatures.as_ref()[1],
    ];
    assert!(
        fda1[0].abs() > 1e-6 && fda1[1].abs() > 1e-9,
        "fixture premise: the sustained probe moves both zones"
    );

    // Seeded (air/mass = hour-1 pair, lag = hour-1 chain end): exact.
    let (fa, fm, _) = compose(
        &base,
        &rec1,
        [GAIN_W, 0.0],
        [GAIN_W, 0.0],
        v1a,
        v1a,
        v1m,
        rl1,
        true,
    );
    rel_close(fa[0], fda1[0], "hour-2 z0 air: closes WITH the lag seed");
    rel_close(fm[0], fdm1[0], "hour-2 z0 mass: closes WITH the lag seed");
    rel_close(
        fa[1],
        fda1[1],
        "hour-2 z1 air (iz): closes WITH the lag seed",
    );
    rel_close(fm[1], fdm1[1], "hour-2 z1 mass: closes WITH the lag seed");

    // No lag seed: the persistence miss, quantified (≥ 2× the seeded
    // residual — the seeded form is exact to float noise, the no-seed
    // form misses by the hour-1 chain the hour-2 num consumes).
    let (fa_ns, _, _) = compose(
        &base,
        &rec1,
        [GAIN_W, 0.0],
        [GAIN_W, 0.0],
        v1a,
        v1a,
        v1m,
        [0.0; 2],
        true,
    );
    let miss = (fa_ns[0] - fda1[0]).abs();
    let seeded_err = (fa[0] - fda1[0]).abs();
    assert!(
        miss >= 2.0 * seeded_err,
        "the no-seed form must miss the persistence amount: miss {:.6e} K, \
         seeded residual {:.6e} K",
        miss,
        seeded_err
    );
}

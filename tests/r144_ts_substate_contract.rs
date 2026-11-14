//! Loop round 144 — the rank-5 arm: the t_s-first surface-node sub-state
//! INSIDE the per-class coupled sub-step (`FLX_R144_TS_SUBSTATE`, env-gated
//! OFF by default).
//!
//! Premise (verified live at tip `c73dc7f8`): the r85 per-class sub-step —
//! which commits the mass state on the coupled-N default — discloses the
//! algebraic-series interior coupling with NO t_s sub-state
//! (`zone_balance.rs`, the R85 HARNESS note), while the hourly D5 route
//! (default since round 71, #3983 §57) advances the same per-class surface
//! node FIRST by backward Euler with the disclosed interior surface-layer
//! capacitance `C_s_c = 0.1·cm_c`, and its committed t_s state feeds the
//! inter-surface LW network and the checker's surface identities. The two
//! interior couplings are structurally different orders of the same
//! physical layer: the arm mirrors the hourly D5 structure into the
//! sub-step at dt_m (BE t_s first, BE mass consuming t_s_eff via
//! h_am_c — the D5 disclosed hybrid form), with the sub-step's own
//! T5138/R94 surface-pool partition re-destined to the surface node
//! (phi_m_i·w stays at the mass) and the R90 prev-ramp applied
//! componentwise. The r89 inner-step composition generalizes to the 4-d
//! affine system x = [tm, ts_0..2]: x_N = M^N·x + S·c, committed from the
//! composed slots (the r89 composed-form precedent) and recorded on the
//! mass record as a 22-slot-per-sub-step layout (dated plan amendment,
//! checker-air-node-mirror-plan §4).
//!
//! Python replication (`/tmp/r144_ts_substate_replication.py`): composed
//! form vs sequential per-iteration evaluation 4.0e-15 worst relative over
//! 400 randomized 3-class configs; load channel (S·blv) vs finite
//! difference 3.5e-15; air channel (S·airv) 3.7e-15; structured
//! single-class case vs numpy matrix power exact to 1e-9.
//!
//! Battery expectation (recorded, not weighted, per the standing
//! bands-ignored direction): flag-off byte-identity — the strict gate
//! partition stays 14 PASS / 28 KNOWN-FAIL / 0 REGRESSION, output-diff
//! 42/42 +0.0000%, hotloop/grid goldens hold — so no re-record is owed;
//! the flag-on A/B free-float movement is recorded for the record only.

use std::sync::Mutex;

use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::sim::thermal_model_core::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// ENV_LOCK (Issue #3453 convention): process-global env mutation
/// serialized through a file-local mutex; every variable removed on guard
/// drop. The env-race rule puts every env leg of this contract in the
/// single #[test] fn below.
static ENV_LOCK: Mutex<()> = Mutex::new(());
const ENV_VAR: &str = "FLX_R144_TS_SUBSTATE";

struct EnvGuard;

impl EnvGuard {
    fn set(value: &str) -> Self {
        std::env::set_var(ENV_VAR, value);
        EnvGuard
    }
    fn clear() -> Self {
        std::env::remove_var(ENV_VAR);
        EnvGuard
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        std::env::remove_var(ENV_VAR);
    }
}

const DT: f64 = 3600.0;
const OUTDOOR_C: f64 = 5.0;
/// The artificial gain, W (the counterfactual probe).
const GAIN_W: f64 = 1000.0;

fn case900_model() -> ThermalModel<fluxion::physics::cta::VectorField> {
    // Case 600FF under the explicit legacy selector: the 5R1C per-class
    // coupled route with real class tensors and free-float driving. (The
    // default selector promotes the high-mass cases to 9R4C, and the
    // 900FF/950FF free-float routes are 9R4C outright — neither persists
    // the 5R1C sub-step records this arm's contract reads. The low-mass
    // FF pair is exactly the brainstorm's cohort: 600FF/650FF run the
    // per-class 5R1C sub-step.)
    let spec = ASHRAE140Case::Case600FF.spec();
    ThermalModel::<fluxion::physics::cta::VectorField>::from_spec_with_selector(
        &spec,
        &ThermalSelector::legacy(),
    )
    .expect("legacy 5R1C selector must initialize")
}

fn mass_record(m: &ThermalModel<fluxion::physics::cta::VectorField>) -> &Vec<f64> {
    m.mass
        .r77_mass_substep_coeffs
        .as_ref()
        .and_then(|v| v.first())
        .expect("a live coupled 5R1C step must persist the mass record")
}

fn air_record(m: &ThermalModel<fluxion::physics::cta::VectorField>) -> &Vec<f64> {
    m.mass
        .r77_air_substep_coeffs
        .as_ref()
        .and_then(|v| v.first())
        .expect("a live coupled 5R1C step must persist the air record")
}

#[test]
fn r144_ts_substate_contracts() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // ---- Phase A: env legs. Unset and falsy "0" must be byte-identical
    // on every committed surface; truthy engages the arm (22-slot record,
    // moved t_s pair).
    let mut base = case900_model();
    base.step_physics(0, OUTDOOR_C, DT);
    base.step_physics(1, OUTDOOR_C, DT);
    let _g0 = EnvGuard::set("0");
    let mut falsy = case900_model();
    falsy.step_physics(0, OUTDOOR_C, DT);
    falsy.step_physics(1, OUTDOOR_C, DT);
    drop(_g0);
    {
        let bm = base.mass.mass_temperatures.as_ref();
        let fm = falsy.mass.mass_temperatures.as_ref();
        for (i, (b, f)) in bm.iter().zip(fm.iter()).enumerate() {
            assert_eq!(
                b, f,
                "falsy leg must be byte-identical to unset (mass zone {i})"
            );
        }
        let bb = base.mass.air_temperatures.as_ref();
        let fb = falsy.mass.air_temperatures.as_ref();
        for (i, (b, f)) in bb.iter().zip(fb.iter()).enumerate() {
            assert_eq!(b, f, "falsy leg byte-identity (air zone {i})");
        }
        assert_eq!(
            mass_record(&base),
            mass_record(&falsy),
            "falsy leg byte-identity (mass record)"
        );
        assert_eq!(
            base.mass
                .r70_surface_node_temperatures
                .as_ref()
                .map(|v| v.as_ref().to_vec()),
            falsy
                .mass
                .r70_surface_node_temperatures
                .as_ref()
                .map(|v| v.as_ref().to_vec()),
            "falsy leg byte-identity (t_s pair)"
        );
    }

    let _g1 = EnvGuard::set("1");
    let mut armed = case900_model();
    armed.step_physics(0, OUTDOOR_C, DT);
    let (mrec, arec) = (mass_record(&armed).clone(), air_record(&armed).clone());
    assert_eq!(arec.len() % 9, 0, "air record stays 9-stride under the arm");
    let n = arec.len() / 9;
    assert_eq!(
        mrec.len(),
        n * 22,
        "arm mass record must carry 22 slots per sub-step (got {}, n={n})",
        mrec.len()
    );
    drop(_g1);

    // Contrast: the arm's committed state differs from the default's.
    {
        let am = armed.mass.mass_temperatures.as_ref();
        let bm = base.mass.mass_temperatures.as_ref();
        let d = (am[0] - bm[0]).abs();
        assert!(
            d > 1e-9,
            "the surface-node sub-state must move the committed mass \
             trajectory (|Δ| = {d} K after one step)"
        );
        let ats = armed
            .mass
            .r70_surface_node_temperatures
            .as_ref()
            .expect("the arm commits the t_s pair");
        let bts = base
            .mass
            .r70_surface_node_temperatures
            .as_ref()
            .expect("the default commits the t_s pair (hourly D5)");
        let dts = (ats.as_ref()[0] - bts.as_ref()[0]).abs();
        assert!(
            dts > 1e-9,
            "the committed t_s state must differ under the arm (|Δ| = {dts} K)"
        );
    }

    // ---- Phase B: residual-zero at Δ = 0 (the #3297 property through the
    // arm's 2-d replay). The checker's mass residual on the armed model
    // must sit at the mirror's noise floor.
    let _g2 = EnvGuard::set("1");
    let mut armed2 = case900_model();
    armed2.step_physics(0, OUTDOOR_C, DT);
    let mut checker = InvariantChecker::default();
    let res0 = checker.check_invariant_with_artificial_gain(&armed2, DT, OUTDOOR_C, 0.0, 0);
    let r0 = res0.zone_imbalances[0];
    assert!(
        r0.abs() <= 1e-4,
        "arm replay must be exact at Δ=0: residual {r0} W exceeds 1e-4 W"
    );

    // ---- Phase C: the flagship — the composed probe response equals the
    // engine counterfactual to ≤1e-9 relative. The load delta reaches the
    // mass only through the surface node under the arm; a record that
    // dropped the surface rows or the tau/bl channels cannot compose this.
    let mut base_c = case900_model();
    let mut cf_c = case900_model();
    {
        let loads = cf_c.setpoints.loads.as_ref().to_vec();
        let mut perturbed = loads.clone();
        perturbed[0] += GAIN_W / cf_c.setpoints.zone_area.as_ref()[0];
        cf_c.setpoints.loads = fluxion::physics::cta::VectorField::new(perturbed);
    }
    base_c.step_physics(0, OUTDOOR_C, DT);
    cf_c.step_physics(0, OUTDOOR_C, DT);
    let measured_dt =
        cf_c.mass.mass_temperatures.as_ref()[0] - base_c.mass.mass_temperatures.as_ref()[0];
    let res_g = checker.check_invariant_with_artificial_gain(&base_c, DT, OUTDOOR_C, GAIN_W, 0);
    let r_g = res_g.zone_imbalances[0];
    let n_c = air_record(&base_c).len() / 9;
    let denom_final = mass_record(&base_c)[22 * (n_c - 1) + 6];
    // [Sign convention: the checker's residual is (committed −
    // counterfactual)·den, so the composed response carries the minus; the
    // magnitudes agreed to all printed digits on the first run.]
    let composed_dt = -r_g / denom_final;
    assert!(
        measured_dt > 1e-6,
        "the probe must move the armed engine (measured {measured_dt} K)"
    );
    assert!(
        (composed_dt - measured_dt).abs() <= 1e-9 * measured_dt.abs(),
        "composed response {composed_dt} K must equal the engine \
         counterfactual {measured_dt} K at <=1e-9 relative"
    );

    // ---- Phase D: recovery-twin / hour-boundary continuity. The sub-step
    // arm owns the t_s state: after step 2, the committed pair must equal
    // what step 2's advance consumed (step 1's end state), bit-exact —
    // no double-advance from the gated hourly D5 write.
    let mut armed3 = case900_model();
    armed3.step_physics(0, OUTDOOR_C, DT);
    let ts_after_1: Vec<f64> = armed3
        .mass
        .r70_surface_node_temperatures
        .as_ref()
        .expect("arm commits the t_s pair")
        .as_ref()
        .to_vec();
    let prev_after_1: Vec<f64> = armed3
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("arm persists the t_s prev snapshot")
        .as_ref()
        .to_vec();
    armed3.step_physics(1, OUTDOOR_C, DT);
    let prev_after_2: Vec<f64> = armed3
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("arm persists the t_s prev snapshot")
        .as_ref()
        .to_vec();
    for (i, (a, b)) in ts_after_1.iter().zip(prev_after_2.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step 2 must consume exactly step 1's committed t_s state \
             (lane {i})"
        );
    }
    // And the advance actually moved the state (not a stale echo).
    let moved = ts_after_1
        .iter()
        .zip(prev_after_1.iter())
        .any(|(a, b)| (a - b).abs() > 0.0);
    assert!(
        moved,
        "the armed t_s advance must move the state within the hour"
    );

    // ---- Phase E: the air record's h3od slot carries the arm's composed
    // mass-row air gain — distinct from the default route's direct-series
    // gain on the same fixture.
    drop(armed2);
    let _g3 = EnvGuard::clear();
    let mut disarmed = case900_model();
    disarmed.step_physics(0, OUTDOOR_C, DT);
    let h3od_off = air_record(&disarmed)[4];
    drop(_g3);
    let _g4 = EnvGuard::set("1");
    let mut armed4 = case900_model();
    armed4.step_physics(0, OUTDOOR_C, DT);
    let h3od_on = air_record(&armed4)[4];
    drop(_g4);
    assert!(
        h3od_on.abs() > 0.0,
        "the arm's air-borne mass gain must be nonzero"
    );
    assert_ne!(
        h3od_on, h3od_off,
        "the arm routes the air-borne mass gain through the surface node; \
         the h3od slot must differ from the default route's (on {h3od_on}, \
         off {h3od_off})"
    );

    // ---- Phase F: bounded-state sanity over a day (RULES.md §1's
    // envelope: no NaN/Inf, states stay bounded under the arm).
    let _g5 = EnvGuard::set("1");
    let mut armed5 = case900_model();
    for step in 0..24 {
        armed5.step_physics(step, OUTDOOR_C, DT);
        for (zi, t) in armed5.mass.mass_temperatures.as_ref().iter().enumerate() {
            assert!(
                t.is_finite() && (-100.0..=200.0).contains(t),
                "armed mass state left the bounded envelope (zone {zi}, \
                 step {step}: {t} K)"
            );
        }
        if let Some(ts) = armed5.mass.r70_surface_node_temperatures.as_ref() {
            for (ci, t) in ts.as_ref().iter().enumerate() {
                assert!(
                    t.is_finite() && (-150.0..=250.0).contains(t),
                    "armed t_s state left the bounded envelope (lane {ci}, \
                     step {step}: {t} K)"
                );
            }
        }
    }
    drop(_g5);
}

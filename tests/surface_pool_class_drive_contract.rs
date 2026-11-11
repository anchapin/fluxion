//! Surface-pool class-drive contract suite (loop round 125; plan §7
//! rank-1 remainder — the surface-pools component contracts at the class
//! drive seam).
//!
//! Component under contract: the per-class coupled mass drive's
//! LOAD-LINEAR coefficient — the 4th entry of each composed sub-step
//! 4-tuple in `MassState::r77_mass_substep_coeffs` (round-103/105 record,
//! r89-composed: one 4-tuple per air sub-step, `a^N` composition,
//! N = 12). The load coefficient is temperature-state-independent, so the
//! closed form below replicates the engine bit-exactly from the engine's
//! own persisted record and conduction tensors alone.
//!
//! What this seam pins (the named plan-§7 remainder that round-92's slice
//! 5 explicitly deferred — `tests/mass_advance_contract.rs` slice-5
//! header: "The per-class fraction constants live behind the per-step
//! class-boundary aggregation, so their numeric content is pinned through
//! the strict-gate baseline, not at this seam"; round-94's landing
//! re-recorded baselines but added no component contract):
//!
//! - Contract 1 (r94 DEFAULT, `FLX_R94_T5138_GAINS_C3` unset): the
//!   internal-gain radiative component keeps the ISO 13790 §C.3
//!   conductance partition at the class drive — db_c/dload =
//!   s_c94·rad_frac·w / den_c with s_c94 = h_ms_c/(h_ms_c + h_is·w)
//!   and w the class area share. The solar component's air share stays
//!   retired (round 93); the radiative component's air share stays live
//!   at the lag site (round 94), so this coefficient is the mass-side
//!   half of the §C.3 gains split.
//! - Contract 2 (pre-r94 falsy restoration, `FLX_R94_T5138_GAINS_C3=0`):
//!   db_c/dload = (f_c/0.9664)·rad_frac / den_c, where f_c carries the
//!   Table 5-138 interior-distribution class fractions (floor 0.642,
//!   ceiling/roof 0.1678, walls E+W+N+S = 0.038+0.038+0.053+0.0276 =
//!   0.1566) renormalized over the 0.9664 surface-row sum so Σ_c phi_c
//!   conserves phi_st exactly (ASHRAE 140-2017 §5.2.1.12.2). This pins
//!   the fraction constants' numeric content AT the seam. Note the
//!   archaeology route carries NO area weight on the fraction — the
//!   pre-r94 defect shape, pinned verbatim as the documented
//!   bit-identical restoration.
//! - Contract 3 (route discrimination): with the radiative component
//!   live (rad_frac > 0) the two routes' recorded composites differ —
//!   the round-94 partition is live on the coupled composite route, not
//!   structurally inert (the round-83-era inertness came from the
//!   discarded hourly advance; the coupled composite binds the committed
//!   mass state since rounds 84/85).
//!
//! Verification chain: Python replication
//! `/tmp/r125_class_drive_replication.py` (500 synthetic configs x 2
//! routes: closed-form `bl·(1-a^N)/(1-a)` composition vs explicit
//! N-fold iteration of the affine load coefficient, worst relative error
//! 1.089e-12; Table 5-138 row sum = 0.9664 asserted; CN-branch
//! reachability premise — with positive capacitances and conductances
//! den_c > 0 always, so the forward-Euler fallback is a defensive arm
//! the fixtures never produce). The Rust assertions are BIT-EXACT:
//! every operand is consumed from the engine's persisted state
//! (`Applied5R1CClassBoundary`, the conduction tensors) and the
//! arithmetic replicates the engine's operand order term by term.
//!
//! Fixture premises: default coupled route (r77_n = 12 air sub-steps,
//! r89 aligned dt_m = dt_sub/12 = 25 s, r89_inner_n = 12), fresh-fill
//! hour (no r90 ramp — previous-hour gains are zero on the first step,
//! the seed branch), mass-side night vent 0 by default, non-degenerate
//! spec tensors (the series h_air_mass arm), gains live
//! (phi_st_zone_load_coef > 0).

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_model_data::Applied5R1CClassBoundary;
use std::sync::Mutex;

/// Hourly step used across the suite (matches the other contract suites).
const DT: f64 = 3600.0;

/// r77 coupled sub-step count (the `FLX_R77_COUPLED_N` default).
const R77_N: usize = 12;

/// every test holds the file-local ENV_LOCK for its whole body (round-86
/// in-process env-leak lesson): the falsy-token tests flip
/// `FLX_R94_T5138_GAINS_C3` for the run and remove it before releasing.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Spec-built Case 600 baseline with internal gains live (the spec
/// default load set — no neutralization) and the per-class construction
/// tensors present (the default coupled arm engages).
fn spec_model_gains_live() -> ThermalModel<VectorField> {
    BatchOracle::default_model().expect("spec model builds")
}

/// Grab one zone's optional vector scalar, with a loud premise failure.
fn grab(o: &Option<VectorField>, label: &'static str) -> f64 {
    o.as_ref()
        .and_then(|v| v.as_ref().first().copied())
        .expect(label)
}

/// Replicate the composed per-sub-step load-linear coefficient of the
/// class drive, verbatim in the engine's operand order.
///
/// Route selection mirrors the engine's own branch: `pre_r94 = false` is
/// the r94 default (§C.3 conductance share on the radiative component);
/// `pre_r94 = true` is the falsy-`FLX_R94_T5138_GAINS_C3` archaeology
/// route (the whole pool at the Table 5-138 class fractions).
fn replicate_composed_bload(
    rec: &Applied5R1CClassBoundary,
    cls_cm: &[f64; 3],
    cls_ms: &[f64; 3],
    h_is_i: f64,
    rad_frac: f64,
    pre_r94: bool,
) -> f64 {
    // r89 aligned inner sub-step: dt_sub = DT/r77_n, dt_m = dt_sub/mass_sub,
    // mass_sub = r77_n on the default route.
    let dt_m: f64 = DT / R77_N as f64 / R77_N as f64;
    let total_area: f64 = rec.cls_area.iter().sum();
    let cm_sum: f64 = cls_cm.iter().sum();
    let mut bl_inner = 0.0_f64;
    let mut a_inner = 0.0_f64;
    for c in 0..3 {
        let w = rec.cls_area[c] / total_area;
        if w <= 0.0 {
            continue;
        }
        let h_em_c = if c == 2 { 0.0 } else { rec.cls_g[c] };
        let h_is_c = h_is_i * w;
        let h_ms_c = cls_ms[c];
        // non-degenerate premise: the series arm must be the live branch
        assert!(
            h_ms_c + h_is_c > 0.0,
            "premise: class {c} carries positive ms+is conductance (non-degenerate fixture)"
        );
        let h_am_c = h_ms_c * h_is_c / (h_ms_c + h_is_c);
        let h3_c = h_am_c + rec.h_vent_mass_zone * w;
        let cm_dt_c = cls_cm[c] / dt_m;
        let half_c = 0.5 * (h3_c + h_em_c);
        let den_c = cm_dt_c + half_c;
        let dphi = if pre_r94 {
            let f_c = match c {
                2 => 0.642_f64,
                1 => 0.1678,
                _ => 0.1566,
            };
            (f_c / 0.9664) * rad_frac
        } else {
            let s_c94 = h_ms_c / (h_ms_c + h_is_c);
            s_c94 * rad_frac * w
        };
        // CN branch is the only reachable one under the fixture premises
        // (den_c > 0 whenever capacitances and conductances are positive).
        assert!(den_c > 0.0, "premise: class {c} CN denominator positive");
        let db_c = dphi / den_c;
        let a_c = (cm_dt_c - half_c) / den_c;
        let w_cap = cls_cm[c] / cm_sum;
        bl_inner += w_cap * db_c;
        a_inner += w_cap * a_c;
    }
    // r89 composition over the inner sub-steps (driving held constant per
    // air sub-step): the composed load coefficient scales by
    // (1 - a^N)/(1 - a), the engine's own formula including its
    // near-unity guard.
    let n = R77_N;
    let an = a_inner.powi(n as i32);
    if (1.0 - a_inner).abs() < 1e-15 {
        bl_inner * n as f64
    } else {
        bl_inner * (1.0 - an) / (1.0 - a_inner)
    }
}

/// Run one default-route step on a fresh gains-live spec model, pull the
/// persisted record, replicate the composed load coefficient under
/// `pre_r94`, and assert every composed sub-step tuple's 4th entry
/// bit-exactly. Returns the expected (and asserted) composite.
#[allow(clippy::float_cmp)] // bit-exact provenance: the closed form replicates the engine's own operand order
fn assert_bload_matches_route(pre_r94: bool) -> f64 {
    let mut m = spec_model_gains_live();
    let h_is_i = m.conduction.h_tr_is.as_ref()[0];
    m.step_physics(0, 12.0, DT);
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("premise: the default coupled route persists the class boundary")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");
    assert_eq!(
        rec.h_vent_mass_zone, 0.0,
        "premise: mass-side night vent is 0 on the default route"
    );
    let rad_frac = rec.phi_st_zone_load_coef;
    assert!(
        rad_frac > 0.0,
        "premise: gains-live spec model carries a nonzero radiative fraction"
    );
    let cls_cm: [f64; 3] = [
        grab(&m.conduction.cm_wall, "cm_wall"),
        grab(&m.conduction.cm_roof, "cm_roof"),
        grab(&m.conduction.cm_floor, "cm_floor"),
    ];
    let cls_ms: [f64; 3] = [
        grab(&m.conduction.h_tr_ms_wall, "h_tr_ms_wall"),
        grab(&m.conduction.h_tr_ms_roof, "h_tr_ms_roof"),
        grab(&m.conduction.h_tr_ms_floor, "h_tr_ms_floor"),
    ];
    let coeffs = m
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("premise: the default coupled route persists the r77 record");
    let c0 = coeffs.first().expect("zone 0 record");
    assert!(
        c0.len() >= 4 && c0.len().is_multiple_of(4),
        "record shape: four entries per composed sub-step (got len={})",
        c0.len()
    );
    let bl_expected = replicate_composed_bload(rec, &cls_cm, &cls_ms, h_is_i, rad_frac, pre_r94);
    let n_sub = c0.len() / 4;
    assert_eq!(
        n_sub, R77_N,
        "premise: r89-composed record carries one 4-tuple per air sub-step"
    );
    for k in 0..n_sub {
        assert_eq!(
            c0[4 * k + 3],
            bl_expected,
            "sub-step {k}: the recorded class-drive load coefficient must equal the closed form (bit-exact)"
        );
    }
    bl_expected
}

#[test]
/// Contract 1 — the r94 default route: the internal-gain radiative
/// component keeps the §C.3 conductance partition at the class drive
/// (`s_c94·rad_frac·w`), so the recorded composed load coefficient is the
/// §C.3 closed form, bit-exact on every sub-step.
fn class_drive_bload_default_route_is_c3_gains_share() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_bload_matches_route(false);
}

#[test]
/// Contract 2 — the pre-r94 falsy restoration: with
/// `FLX_R94_T5138_GAINS_C3=0` the recorded load coefficient follows the
/// Table 5-138 class fractions renormalized over the 0.9664 surface-row
/// sum, bit-exact — the fraction constants' numeric content (floor
/// 0.642, roof 0.1678, walls 0.1566) pinned at the component seam where
/// they enter the persisted record, plus the documented bit-identical
/// restoration of the composed route.
fn class_drive_bload_pre_r94_route_carries_t5138_fractions() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("FLX_R94_T5138_GAINS_C3", "0");
    let bl = assert_bload_matches_route(true);
    std::env::remove_var("FLX_R94_T5138_GAINS_C3");
    let _ = bl;
}

#[test]
#[allow(clippy::float_cmp)] // route discrimination on bit-exact provenance values
/// Contract 3 — route discrimination: with the radiative component live
/// the two routes' recorded composites differ (the partition is live on
/// the coupled route). The differentiation is algebraic — the §C.3 share
/// vs the renormalized Table 5-138 fraction — so it holds on the same
/// fixture, and it is the discriminator the round-84/85 coupled composite
/// made observable.
fn r94_partition_routes_differ_on_the_recorded_composite() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bl_default = assert_bload_matches_route(false);
    std::env::set_var("FLX_R94_T5138_GAINS_C3", "0");
    let bl_falsy = assert_bload_matches_route(true);
    std::env::remove_var("FLX_R94_T5138_GAINS_C3");
    assert_ne!(
        bl_default, bl_falsy,
        "the r94 partition must change the recorded composite with gains live"
    );
}

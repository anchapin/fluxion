//! HVAC-masked probe contract (loop round 118) — the checker air-node-mirror
//! plan §5 out-of-scope list's last un-disposed item: "HVAC-masked zones
//! (probe fixtures use free-float setpoints)". Every probe fixture the plan
//! landed so far (rounds 103–117) pins `hvac.free_float = true` plus ±999
//! setpoints — "the same HVAC-masking lesson the round-103/104 probe fixtures
//! encoded" (the round-105 fixture's own comment). This file disposes that
//! line: fixtures with ACTIVE setpoint control, bounding exactly what the
//! committed states do under it.
//!
//! Component under contract: the LIMIT-33 Option-A state-feedback commit
//! (`step_5r1c.rs`, the `t_i_act` site — issue #4314) and its interaction
//! with the R77 coupled air/mass record. Under active control the engine
//! commits the controlled state in `setpoints.temperatures` while
//! `mass.air_temperatures` keeps the free-float air state (the documented
//! convention the invariant checker and the gauge mass-state proxy rely on),
//! and the ideal-load demand (`thermal_model_physics/hvac.rs::
//! compute_zone_hvac_load`, issues #4241/#2826) is capacity-clamped to
//! ±(heating, cooling) capacity. Three routes follow, each exactly:
//!
//!   1. UNCLAMPED IDEAL — demand positive and below capacity: the controlled
//!      state is the setpoint, verbatim, so the committed controlled response
//!      to the artificial-gain probe is EXACTLY 0 (the mask), while the free
//!      advance is untouched: `mass.air_temperatures` and the mass node
//!      respond with the record-composed r_air/r_m (the round-105 §2
//!      recurrence) and are BIT-IDENTICAL to a free-float twin carrying the
//!      same load. The HVAC machinery is response-invisible off the
//!      controlled state.
//!   2. CAPACITY-CLAMPED — demand at capacity: the clamp pins Q to ±cap,
//!      load-INDEPENDENT, so the residual commit
//!      `t_i_act = t_free + Q/den_true` (den_true = den/term_rest_1, the
//!      #2868 form) passes the free response through UNSCALED — the
//!      controlled response equals r_air·Δ and the level identity closes
//!      from the committed record's final den slot.
//!   3. DEADBAND — demand 0 by the load-sign rule: `t_i_act = t_free`, so
//!      the controlled state IS the free state (parity with free-float).
//!
//! Under every route the coupled advance itself — and therefore the
//! committed R77 air/mass record — is response- and HVAC-blind: the records
//! are BIT-IDENTICAL across the three routes and the free-float control.
//!
//! Provenance: every claim was replicated in Python against an engine dump
//! of these fixtures (compose from the committed 9-slot record, level
//! identity, parity) — ALL CLAIMS OK (19/19) before the pins were written.
//! The pins recompute everything IN TEST from committed values (the
//! round-103+ convention): no printed decimal is trusted here.
//!
//! Zero-delta / no-engine-change note: this file is contracts only — no
//! engine behavior changed in this round, so the default build's strict
//! battery and output-diff must be bit-identical (42/42 from-zero rows at
//! +0.0000%) and no re-record is owed. The battery proves that bit-identity,
//! not these tests.
//!
//! Tooth provenance (round 118, temporary engine mutations, each reverted):
//!   T1 — t_i_act unclamped branch commits t_free instead of the setpoint →
//!        exactly `hvac_unclamped_ideal_route_masks_committed_air...` fails
//!        (the mask claim + its parity twins); the clamped and deadband pins
//!        stay green (different branch sites).
//!   T2 — the clamped commit divides by h_tr_is (the pre-#2868 form) →
//!        exactly the clamped route's level pin fails (the pass-through
//!        response is divisor-independent and stays green).
//!   T3 — compute_zone_hvac_load's capacity clamp removed → exactly the
//!        clamped route's pass-through and level pins fail: the t_i_act
//!        site's own demand >= capacity check keeps the route selected, but
//!        Q is now load-dependent (q_heat, not cap), so the controlled
//!        response is r_air·Δ·(1 − h_coeff/den_true), not r_air·Δ; the
//!        unclamped/deadband pins stay green (the demand stays below the
//!        100 kW cap either way / the deadband branch is unchanged).
//!
//! Red-first note: these are contracts on EXISTING behavior — the red-first
//! obligation for this round is the Python replication against the engine
//! dump (done, ALL CLAIMS OK); the tests themselves must pass at this tree,
//! and the teeth above prove they would fail exactly where they claim to.

use fluxion::physics::cta::VectorField;
use fluxion::sim::thermal_model_core::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

const DELTA_W: f64 = 1000.0;
const DT: f64 = 3600.0;
const OUTDOOR: f64 = 10.0;

/// Fixture build: Case 600 under the legacy 5R1C selector, HVAC ACTIVE
/// (free_float off, zone 0 enabled), authored setpoints, authored heating
/// capacity. `h_sp` 30 °C puts both probe arms deep in heating demand;
/// `h_sp` 5 °C with cooling at 999 °C gives the deadband.
fn build(h_sp: f64, heat_cap: f64, free_float: bool, gain: bool) -> ThermalModel<VectorField> {
    let spec = ASHRAE140Case::Case600.spec();
    let mut m = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::legacy())
        .expect("legacy 5R1C selector must initialize");
    m.hvac.free_float = free_float;
    if !free_float {
        // The demand site skips zones with hvac_enabled < 0.5; Case 600's
        // spec ships the mechanical system disabled (free-float case), so
        // the fixture enables zone 0 explicitly.
        m.hvac.hvac_enabled.as_mut()[0] = 1.0;
        m.setpoints.heating_setpoints.as_mut()[0] = h_sp;
        m.setpoints.cooling_setpoints.as_mut()[0] = 999.0;
        m.setpoints.heating_setpoint = h_sp;
        m.setpoints.cooling_setpoint = 999.0;
        m.hvac.hvac_heating_capacity = heat_cap;
    } else {
        m.setpoints.heating_setpoint = -999.0;
        m.setpoints.cooling_setpoint = 999.0;
    }
    if gain {
        let area = m.setpoints.zone_area.as_ref()[0];
        m.setpoints.loads.as_mut()[0] += DELTA_W / area;
    }
    m
}

/// The committed-state snapshot the contracts assert on, plus the branch
/// quantities recomputed from committed fields (per-model, in-test — the
/// ideal-load residual form, issues #4241/#2826, and the #4156 unified
/// conductance; the pre-step controlled temperature is snapshotted by the
/// caller before `step_physics`).
struct Committed {
    t_prev: f64,
    ctrl: f64,
    free_air: f64,
    mass: f64,
    q_heat: f64,
    q_cool: f64,
    h_coeff: f64,
    c_air: f64,
}

fn step_and_observe(m: &mut ThermalModel<VectorField>) -> Committed {
    let t_prev = m.setpoints.temperatures.as_ref()[0];
    let h_tr_is = m.conduction.h_tr_is.as_ref()[0];
    let h_tr_ms = m.conduction.h_tr_ms.as_ref()[0];
    let h_tr_w = m.conduction.h_tr_w.as_ref()[0];
    let h_ve = m.conduction.h_ve.as_ref()[0];
    let c_air = m.mass.air_thermal_capacitance.as_ref()[0];
    let h_sp_h = m.setpoints.heating_setpoints.as_ref()[0];
    let h_sp_c = m.setpoints.cooling_setpoints.as_ref()[0];
    m.step_physics(0, OUTDOOR, DT);
    let free_air = m.mass.air_temperatures.as_ref()[0];
    // Ideal-load residual (heating + cooling legs), recomputed from
    // committed values — the exact form `compute_zone_hvac_load` evaluates
    // (the capacitance term uses the pre-step controlled temperature).
    let q_heat = c_air * (h_sp_h - t_prev) / DT
        + h_coeff_of(h_tr_is, h_tr_ms, h_tr_w, h_ve) * (h_sp_h - free_air);
    let q_cool = c_air * (h_sp_c - t_prev) / DT
        + h_coeff_of(h_tr_is, h_tr_ms, h_tr_w, h_ve) * (h_sp_c - free_air);
    Committed {
        ctrl: m.setpoints.temperatures.as_ref()[0],
        free_air,
        mass: m.mass.mass_temperatures.as_ref()[0],
        q_heat,
        q_cool,
        h_coeff: h_coeff_of(h_tr_is, h_tr_ms, h_tr_w, h_ve),
        c_air,
        t_prev,
    }
}

fn h_coeff_of(h_tr_is: f64, h_tr_ms: f64, h_tr_w: f64, h_ve: f64) -> f64 {
    // Issue #4156 (option #2) unified conductance, 5R1C arm:
    // (h_tr_w + h_ve) + h_tr_is·h_tr_ms/(h_tr_is + h_tr_ms).
    (h_tr_w + h_ve) + h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms)
}

/// The §2 staggered response recurrence composed from the committed air
/// record alone (the round-107 self-contained layout, nine slots per
/// sub-step since r113/r115). Same stride the round-105 fixture consumes;
/// slots 7/8 (iz cross-response, ramp-source persistence) carry no
/// single-zone probe response here and stay unconsumed by this composition.
fn compose_responses(rec: &[f64]) -> (f64, f64) {
    assert!(
        !rec.is_empty() && rec.len() % 9 == 0,
        "air record must carry 9 entries per sub-step (r113/r115 layout), got len {}",
        rec.len()
    );
    let n = rec.len() / 9;
    let (mut r_air, mut r_m) = (0.0f64, 0.0f64);
    for k in 0..n {
        let (ra, rm) = (r_air, r_m);
        r_air = rec[9 * k] * ra + rec[9 * k + 1] * rm + rec[9 * k + 2];
        r_m = rec[9 * k + 5] * rm + rec[9 * k + 4] * ra + rec[9 * k + 6];
    }
    (r_air, r_m)
}

fn rel(pred: f64, act: f64) -> f64 {
    (pred - act).abs() / act.abs().max(1e-300)
}

/// Route 1 — UNCLAMPED IDEAL: the mask. The committed controlled state is
/// the setpoint in BOTH arms, so the controlled response is exactly 0 while
/// the free advance moves with the record-composed responses and stays
/// bit-identical to a free-float twin carrying the same load.
#[test]
fn hvac_unclamped_ideal_route_masks_committed_air_and_preserves_composition() {
    let mut a = build(30.0, 100_000.0, false, false);
    let mut b = build(30.0, 100_000.0, false, true);
    let mut ff = build(-999.0, f64::NAN, true, true);
    let ca = step_and_observe(&mut a);
    let cb = step_and_observe(&mut b);
    let ff_c = step_and_observe(&mut ff);

    // Branch precondition: both arms in heating demand, both below capacity
    // (the route the pins below claim).
    let cap = 100_000.0f64;
    assert!(
        ca.q_heat > 0.0 && ca.q_heat < cap && cb.q_heat > 0.0 && cb.q_heat < cap,
        "unclamped route requires 0 < q_heat < capacity in both arms: A {:.3} B {:.3}",
        ca.q_heat,
        cb.q_heat
    );

    // The mask: the controlled state is the setpoint, verbatim, in both arms.
    assert_eq!(
        ca.ctrl, 30.0,
        "unclamped ideal commit must hold the heating setpoint exactly (A)"
    );
    assert_eq!(
        cb.ctrl, 30.0,
        "unclamped ideal commit must hold the heating setpoint exactly (B)"
    );
    assert_eq!(
        cb.ctrl - ca.ctrl,
        0.0,
        "the committed controlled response under the probe must be EXACTLY zero \
         (the HVAC mask): got {}",
        cb.ctrl - ca.ctrl
    );

    // The free advance is untouched: composed responses from A's committed
    // record predict both committed deltas (plan §2 recurrence, ≤1e-9).
    let rec = a
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    let (r_air, r_m) = compose_responses(&rec[0]);
    let d_free = cb.free_air - ca.free_air;
    let d_mass = cb.mass - ca.mass;
    assert!(
        rel(r_air * DELTA_W, d_free) <= 1e-9,
        "free-air response must equal the composed record prediction: predicted \
         {} K, got {} K",
        r_air * DELTA_W,
        d_free
    );
    assert!(
        rel(r_m * DELTA_W, d_mass) <= 1e-9,
        "mass response must equal the composed record prediction: predicted \
         {} K, got {} K",
        r_m * DELTA_W,
        d_mass
    );

    // Free-float parity: the HVAC-active free advance is BIT-IDENTICAL to a
    // free-float twin carrying the same load — the demand computation never
    // touches the advance.
    assert_eq!(
        cb.free_air, ff_c.free_air,
        "HVAC-active free-air state must be bit-identical to the free-float twin"
    );
    assert_eq!(
        cb.mass, ff_c.mass,
        "HVAC-active mass state must be bit-identical to the free-float twin"
    );
}

/// Route 2 — CAPACITY-CLAMPED: pass-through. The clamp pins Q at ±cap
/// (load-independent), so the residual commit passes the free response
/// through unscaled — the controlled response equals r_air·Δ — and the
/// level closes as t_free + cap/den_true with den_true recomputed from the
/// committed record's final den slot and derived_term_rest_1 (the #2868
/// form). The free advance stays bit-identical to the unclamped route's.
#[test]
fn hvac_capacity_clamped_route_passes_free_response_through() {
    let mut a = build(30.0, 100.0, false, false);
    let mut b = build(30.0, 100.0, false, true);
    // Unclamped twins for the advance-parity claim (same loads, huge cap).
    let mut au = build(30.0, 100_000.0, false, false);
    let mut bu = build(30.0, 100_000.0, false, true);
    let ca = step_and_observe(&mut a);
    let cb = step_and_observe(&mut b);
    let cau = step_and_observe(&mut au);
    let cbu = step_and_observe(&mut bu);

    // Branch precondition: the clamp is live in BOTH arms (demand ≥ cap).
    let cap = 100.0f64;
    assert!(
        ca.q_heat >= cap && cb.q_heat >= cap,
        "clamped route requires demand >= capacity in both arms: A {:.3} B {:.3}",
        ca.q_heat,
        cb.q_heat
    );

    // The pass-through: the committed controlled response equals the
    // composed free response exactly (the clamp is load-independent).
    let rec = a
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    let (r_air, _r_m) = compose_responses(&rec[0]);
    let d_ctrl = cb.ctrl - ca.ctrl;
    assert!(
        rel(r_air * DELTA_W, d_ctrl) <= 1e-9,
        "clamped controlled response must pass the composed free response \
         through unscaled: predicted {} K, got {} K",
        r_air * DELTA_W,
        d_ctrl
    );

    // The level identity (#2868): ctrl = t_free + cap/den_true, with
    // den_true = den/term_rest_1 recomputed from committed values — the
    // record's FINAL sub-step den slot and derived_term_rest_1.
    let term_rest_1 = a
        .conduction
        .derived_term_rest_1
        .as_ref()
        .get(0)
        .copied()
        .expect("derived_term_rest_1 must be populated for a from-spec model");
    let n = rec[0].len() / 9;
    let den_last = rec[0][9 * (n - 1) + 3];
    let den_true = den_last / term_rest_1;
    let lvl_pred = ca.free_air + cap / den_true;
    assert!(
        rel(lvl_pred, ca.ctrl) <= 1e-9,
        "clamped commit level must equal t_free + cap/den_true: predicted {} °C, \
         got {} °C (den_true {} W/K)",
        lvl_pred,
        ca.ctrl,
        den_true
    );

    // Advance parity: the clamped route's free advance is bit-identical to
    // the unclamped route's (the demand never touches the advance).
    assert_eq!(
        ca.free_air, cau.free_air,
        "clamped-route free-air state must be bit-identical to the unclamped twin"
    );
    assert_eq!(
        cb.free_air, cbu.free_air,
        "clamped-route free-air state must be bit-identical to the unclamped twin"
    );
    assert_eq!(
        cb.mass, cbu.mass,
        "clamped-route mass state must be bit-identical to the unclamped twin"
    );
}

/// Route 3 — DEADBAND: parity. With both residual legs non-driving the
/// demand is 0 and `t_i_act = t_free`: the committed controlled state IS
/// the free state (exactly), and the free advance stays bit-identical to
/// the free-float twin.
#[test]
fn hvac_deadband_route_commits_free_state_parity() {
    let mut a = build(5.0, 100_000.0, false, false);
    let mut b = build(5.0, 100_000.0, false, true);
    let mut ff_a = build(-999.0, f64::NAN, true, false);
    let mut ff_b = build(-999.0, f64::NAN, true, true);
    let ca = step_and_observe(&mut a);
    let cb = step_and_observe(&mut b);
    let ff_a_c = step_and_observe(&mut ff_a);
    let ff_b_c = step_and_observe(&mut ff_b);

    // Branch precondition: the load-sign deadband is live in both arms
    // (heating leg non-driving, cooling leg non-driving).
    assert!(
        ca.q_heat <= 0.0 && ca.q_cool >= 0.0 && cb.q_heat <= 0.0 && cb.q_cool >= 0.0,
        "deadband route requires q_heat <= 0 and q_cool >= 0 in both arms: \
         A ({:.3}, {:.3}) B ({:.3}, {:.3})",
        ca.q_heat,
        ca.q_cool,
        cb.q_heat,
        cb.q_cool
    );

    // The commit IS the free state, exactly, in both arms.
    assert_eq!(
        ca.ctrl, ca.free_air,
        "deadband commit must carry the free state verbatim (A)"
    );
    assert_eq!(
        cb.ctrl, cb.free_air,
        "deadband commit must carry the free state verbatim (B)"
    );

    // Parity with free-float, bit-identical, both arms (per-load twins).
    assert_eq!(
        ca.free_air, ff_a_c.free_air,
        "deadband free-air state must be bit-identical to the free-float twin (A-load)"
    );
    assert_eq!(
        cb.free_air, ff_b_c.free_air,
        "deadband free-air state must be bit-identical to the free-float twin (B-load)"
    );
    assert_eq!(
        cb.mass, ff_b_c.mass,
        "deadband mass state must be bit-identical to the free-float twin"
    );
}

/// The record is HVAC-blind: the committed R77 air and mass records are
/// BIT-IDENTICAL across the three routes and the free-float control (same
/// loads). One record serves every route — the probe contracts landed in
/// rounds 105–117 compose the same values under active control.
#[test]
fn hvac_record_is_bit_identical_across_hvac_routes() {
    let mut u = build(30.0, 100_000.0, false, false);
    let mut c = build(30.0, 100.0, false, false);
    let mut d = build(5.0, 100_000.0, false, false);
    let mut f = build(-999.0, f64::NAN, true, false);
    let (_, _, _) = (
        step_and_observe(&mut u),
        step_and_observe(&mut c),
        step_and_observe(&mut d),
    );
    step_and_observe(&mut f);

    let ru = u
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("record must exist (unclamped)");
    let rc = c
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("record must exist (clamped)");
    let rd = d
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("record must exist (deadband)");
    let rf = f
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("record must exist (free-float control)");
    for (name, rx) in [("clamped", rc), ("deadband", rd), ("free-float", rf)] {
        assert_eq!(
            ru, rx,
            "air record must be bit-identical across the {:?} and unclamped routes",
            name
        );
    }
    let mu = u
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record must exist (unclamped)");
    let mc = c
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record must exist (clamped)");
    let md = d
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record must exist (deadband)");
    let mf = f
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record must exist (free-float)");
    for (name, mx) in [("clamped", mc), ("deadband", md), ("free-float", mf)] {
        assert_eq!(
            mu, mx,
            "mass record must be bit-identical across the {:?} and unclamped routes",
            name
        );
    }
}

/// Premise guard: under ACTIVE control every route still persists the
/// coupled record (the write is HVAC-independent — `r77_coupled &&
/// !r74_ramp_exact && r90_no_lag_filter`), with the nine-slot r113/r115
/// layout and the full twelve sub-steps. If a future change suppresses the
/// record under HVAC, every composition contract above loses its anchor —
/// this precondition names that failure instead of letting it pass
/// silently.
#[test]
fn hvac_record_shape_premise_under_active_hvac() {
    for (tag, h_sp, cap) in [
        ("unclamped", 30.0, 100_000.0),
        ("clamped", 30.0, 100.0),
        ("deadband", 5.0, 100_000.0),
    ] {
        let mut m = build(h_sp, cap, false, false);
        step_and_observe(&mut m);
        let rec = m
            .mass
            .r77_air_substep_coeffs
            .as_ref()
            .unwrap_or_else(|| panic!("{} route must persist the air record", tag));
        assert_eq!(
            rec[0].len() % 9,
            0,
            "{} route: air record must carry the 9-slot r113/r115 stride",
            tag
        );
        assert_eq!(
            rec[0].len() / 9,
            12,
            "{} route: air record must carry the full twelve coupled sub-steps",
            tag
        );
        let mrec = m
            .mass
            .r77_mass_substep_coeffs
            .as_ref()
            .unwrap_or_else(|| panic!("{} route must persist the mass record", tag));
        assert_eq!(
            mrec[0].len() / 4,
            12,
            "{} route: mass record must carry the same sub-step count",
            tag
        );
    }
}

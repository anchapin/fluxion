//! 5R1C ramp-source persistence through rejected steps — the r136 routed
//! finding, fixed this round (r137). The verdict is the r136 semantics
//! extended to the 5R1C harness seam: **a rejected step advances nothing.**
//!
//! The seams (zone_balance.rs, at the r137 tip): the R76 sol-air persist
//! (`r76_sol_air_prev`/`r76_has_prev`) ran BEFORE the #4495 air-node
//! guard, and the R71/R75/R90 harness block (`r71_air_driver_num_prev`/
//! `r71_air_driver_den_prev`/`r71_num_prev_load_slope`,
//! `r75_vent_term_prev`/
//! `r75_vent_has_prev`, `r90_phi_st_prev`/`r90_phi_m_prev`) ran between
//! the air guard and the #4500 HVAC-demand / #4495 mass-node guards — so
//! a rejected step left the ramp sources advanced with the rejected
//! step's drivers while the guards' own messages claimed "state
//! untouched". Default builds are fully exposed: `FLX_R77_COUPLED_N`
//! defaults to 12 (the r78 Alex-directed flip), so the R76 persist gate
//! (`r76_n > 0 || r77_coupled`) is always true.
//!
//! The fix relocates both persist blocks into the commit zone BELOW every
//! rejection path (the air guard, #4500, the mass guard), preserving the
//! write order (R76 first, then R71/R75/R90) and the values (nothing
//! reads these fields between the old and new sites; the source locals
//! are stable below their old positions — verified at the tip before the
//! move). Accepted-path bytes are unchanged; the battery proves it.
//!
//! Contracts here (the mass-guard arm lives in its own file because it
//! arms `FLX_R76_NO_NUM_REFRESH`, a process-global env read):
//! - C1: the accepted-step advancement premise (exact engine literals,
//!   Python-replicated slope closed form — `/tmp/r137_replication.py`);
//! - C2: air-guard rejection via the #4495 crash recipe — the ramp source
//!   restored to the step-0 values (pre-fix RED: `r76_sol_air_prev`
//!   = [NaN], the exact corrupted shape);
//! - C3: air-guard rejection via the degenerate-cm recipe (finite
//!   sol-air shape) as the FIRST step — every family restored to init;
//! - C4: the recovery twins A=[0,1,reject@2,3] vs B=[0,1,3] — post-fix
//!   A composes bit-exactly onto B's step-3 state (pre-fix RED: A's step 3
//!   is itself rejected by the NaN ramp source and A freezes at its step-1
//!   commit — air 32.80770541160536 vs B 36.78922134936172, a 3.98 K
//!   divergence from one rejected degenerate step).
//!
//! Bound recorded honestly: the #4500 demand-guard seam (guard at the
//! R71/R75/R90 block and the mass guard) is covered by the same
//! relocation, but no live isolation fixture exists — every overflow
//! recipe tried either trips the air guard first (huge drivers overflow
//! the air numerator; sub-step mass overflow poisons the per-sub-step air
//! numerator through the default R77 num-refresh) or stays accepted
//! (demand too small). Proven by code order (writes 2904–2940 sit above
//! the 3428 guard), not by a live fixture.
//!
//! Teeth (run at the fixed tree, each a targeted un-fix):
//! T1 revert only the R76 relocation → {C2, C3, C4, C5} red, C1 green;
//! T2 revert only the R71/R75/R90 relocation → this file's contracts stay
//!    green (their rejections fire at the air guard, above that block);
//!    the mass-guard file's contract reds instead;
//! T3 revert both → {C2, C3, C4} red here, {C5} red in the mass-guard file.

use fluxion::physics::cta::VectorField;
use fluxion::sim::boundary::ConstantGroundTemperature;
use fluxion::sim::engine::ThermalModel;

const DT: f64 = 3600.0;
const T_OUT: f64 = -10.0;
const T_G: f64 = 13.0;

/// The #4495 crash recipe (finite extreme inputs that pass the input
/// checks and reject at the air-node guard; the tests/all_tests
/// issue_4495 constants).
const CRASH_OUTDOOR: f64 = -2.8088955232222888e306;
const CRASH_DT: f64 = 1.5845632502852868e29;

/// The step-0 accepted-step literals (engine probe at the tip,
/// `/tmp/r137_replication.py` provenance for the slope closed form).
const STEP0_AIR: f64 = 29.081879497340154;
const STEP0_MASS: f64 = 28.325025107023894;
const STEP0_LAG: f64 = 10.065746182553235;
const STEP0_R76_SOL_AIR: f64 = -11.454605297768234;
const STEP0_R71_NUM: f64 = 3460701.6;
const STEP0_R71_DEN: f64 = 138271.19999999998;
const STEP0_R71_SLOPE: f64 = 427.6;
const STEP0_R90_ST: f64 = 7534.080000000001;
const STEP0_R90_M: f64 = 10005.12;

/// The hand-rolled single-zone 5R1C free-float fixture (the r136 9R4C
/// fixture's model-side construction, default 5R1C solver).
fn fixture_5r1c() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::<VectorField>::new(1);
    m.setpoints.zone_area = VectorField::from_scalar(48.0, 1);
    m.setpoints.zone_volume = VectorField::from_scalar(129.6, 1);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.air_density = VectorField::from_scalar(1.2, 1);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 1);
    m.hvac.free_float = true;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0]);
    m.conduction.h_ve = VectorField::new(vec![1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0)]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0]);
    m.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0]);
    m.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.loads = VectorField::from_scalar(30.0, 1);
    m.solar.solar_gains = VectorField::new(vec![400.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.conduction.ground_temperature = Box::new(ConstantGroundTemperature::new(T_G));
    m.update_optimization_cache();
    m
}

fn assert_step0_ramp(m: &ThermalModel<VectorField>, what: &str) {
    assert_eq!(
        m.mass.r76_sol_air_prev.as_ref(),
        &[STEP0_R76_SOL_AIR],
        "{what}: r76_sol_air_prev must hold the step-0 sol-air"
    );
    assert!(
        m.mass.r76_has_prev,
        "{what}: r76_has_prev must be set by the accepted step-0 persist"
    );
    assert_eq!(
        m.mass.r71_air_driver_num_prev.as_ref(),
        &[STEP0_R71_NUM],
        "{what}: r71_air_driver_num_prev must hold the step-0 drivers"
    );
    assert_eq!(
        m.mass.r71_air_driver_den_prev.as_ref(),
        &[STEP0_R71_DEN],
        "{what}: r71_air_driver_den_prev must hold the step-0 denominator"
    );
    assert_eq!(
        m.mass.r71_num_prev_load_slope.as_ref(),
        &[STEP0_R71_SLOPE],
        "{what}: the R115 slope slot must hold conv_frac * term_rest_1"
    );
    assert_eq!(
        m.mass.r75_vent_term_prev.as_ref(),
        &[0.0],
        "{what}: r75_vent_term_prev must hold the vent-off step-0 term"
    );
    assert!(
        m.mass.r75_vent_has_prev,
        "{what}: r75_vent_has_prev must be set by the accepted step-0 persist"
    );
    assert_eq!(
        m.mass.r90_phi_st_prev.as_ref(),
        &[STEP0_R90_ST],
        "{what}: r90_phi_st_prev must hold the step-0 surface gain"
    );
    assert_eq!(
        m.mass.r90_phi_m_prev.as_ref(),
        &[STEP0_R90_M],
        "{what}: r90_phi_m_prev must hold the step-0 mass gain"
    );
}

#[test]
/// C1 — the accepted-step advancement premise: one accepted step persists
/// the full ramp-source set, and the slope slot carries the documented
/// closed form `conv_frac * term_rest_1` (0.4 * 1069.0, bit-exact with
/// the Python replication). The den value is the r136 suite's pinned
/// DEN_OFF for this identical fixture (shared derivation, not re-derived
/// here); r76/r90 pin engine literals (the sol-air aggregate and the
/// solar-split pipeline have their own contract suites — sol-air
/// composition and window_solar_split — this file consumes their
/// machinery as provenance).
fn accepted_step_persists_the_ramp_sources_with_exact_literals() {
    let mut m = fixture_5r1c();
    let k = m.step_physics(0, T_OUT, DT);
    assert_eq!(k, 0.0, "free-float step returns 0.0 kWh");
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        STEP0_AIR,
        "the step-0 committed air (bit-exact probe literal)"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        STEP0_MASS,
        "the step-0 committed mass (bit-exact probe literal)"
    );
    assert_eq!(
        m.mass.solar_lag.as_ref()[0],
        STEP0_LAG,
        "the step-0 solar-lag state (bit-exact probe literal)"
    );
    assert_step0_ramp(&m, "C1");
}

#[test]
/// C2 — the air-guard rejection seam: the #4495 crash recipe rejects at
/// the air-node guard, and (the fix) the R76 sol-air persist must NOT
/// advance with the rejected step's drivers. Pre-fix this exact shape
/// persisted `r76_sol_air_prev = [NaN]` with `r76_has_prev = true` — the
/// NaN ramp source that then bricks every later step (see C4). The
/// R71/R75/R90 block sat BELOW this guard, so its fields legitimately
/// hold the step-0 values both pre- and post-fix; the physical commits
/// are untouched by the pre-existing guard.
fn air_guard_rejection_leaves_the_ramp_source_at_the_step0_values() {
    let mut m = fixture_5r1c();
    m.step_physics(0, T_OUT, DT);
    let k = m.step_physics(1, CRASH_OUTDOOR, CRASH_DT);
    assert_eq!(k, 0.0, "the rejected step degrades to 0.0 kWh");
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        STEP0_AIR,
        "the rejected step leaves the zone-air commit untouched"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        STEP0_MASS,
        "the rejected step leaves the mass commit untouched"
    );
    assert_eq!(
        m.mass.solar_lag.as_ref()[0],
        STEP0_LAG,
        "the rejected step leaves the solar-lag state untouched"
    );
    assert_step0_ramp(&m, "C2");
}

#[test]
/// C3 — the degenerate-cm air-guard shape, as the FIRST step: cm near the
/// f64 ceiling with a normal 100 s step sends the sub-step mass advance
/// to inf (tm_prev * cm/dt_sub overflows), the default R77 num-refresh
/// feeds that inf into the air numerator, and the air-node guard rejects.
/// The sol-air driver here is FINITE (-11.454605297768234), so pre-fix
/// the advanced shape was a plausible-looking driver, not NaN — the
/// second corrupted shape. Post-fix every ramp family stays at init.
fn degenerate_cm_rejection_leaves_every_ramp_family_at_init() {
    let mut m = fixture_5r1c();
    m.mass.thermal_capacitance = VectorField::new(vec![1.7e308]);
    m.update_optimization_cache();
    let k = m.step_physics(0, T_OUT, 100.0);
    assert_eq!(k, 0.0, "the rejected step degrades to 0.0 kWh");
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        20.0,
        "the rejected first step leaves the zone-air commit at init"
    );
    assert_eq!(
        m.mass.r76_sol_air_prev.as_ref(),
        &[0.0],
        "the rejected first step must not persist the sol-air driver"
    );
    assert!(
        !m.mass.r76_has_prev,
        "the rejected first step must not set the sol-air seed flag"
    );
    assert_eq!(
        m.mass.r71_air_driver_num_prev.as_ref(),
        &[0.0],
        "the rejected first step must not persist the num driver"
    );
    assert_eq!(
        m.mass.r71_air_driver_den_prev.as_ref(),
        &[0.0],
        "the rejected first step must not persist the den driver"
    );
    assert_eq!(
        m.mass.r71_num_prev_load_slope.as_ref(),
        &[0.0],
        "the rejected first step must not persist the load slope"
    );
    assert!(
        !m.mass.r75_vent_has_prev,
        "the rejected first step must not set the vent seed flag"
    );
    assert_eq!(
        m.mass.r90_phi_st_prev.as_ref(),
        &[0.0],
        "the rejected first step must not persist the surface gain"
    );
    assert_eq!(
        m.mass.r90_phi_m_prev.as_ref(),
        &[0.0],
        "the rejected first step must not persist the mass gain"
    );
}

#[test]
/// C4 — the recovery twins: A runs [0, 1, reject@2, 3] and B runs
/// [0, 1, 3]; a rejected step advances nothing, so A's step 3 must
/// compose bit-exactly onto B's step-3 state and both must carry the
/// same ramp sources. Pre-fix A's step 3 was itself rejected by the NaN
/// ramp source left by the rejected step 2 — A froze at its step-1
/// commit (air 32.80770541160536, mass 36.43271186146832) while B
/// advanced (air 36.78922134936172, mass 44.383328855093694): a 3.98 K
/// divergence in committed state from one rejected degenerate step, the
/// 5R1C twin of the r136 bricking finding.
fn recovery_twin_rejected_step_advances_nothing() {
    let mut a = fixture_5r1c();
    a.step_physics(0, T_OUT, DT);
    a.step_physics(1, T_OUT, DT);
    let k_reject = a.step_physics(2, CRASH_OUTDOOR, CRASH_DT);
    let k_a3 = a.step_physics(3, T_OUT, DT);
    let mut b = fixture_5r1c();
    b.step_physics(0, T_OUT, DT);
    b.step_physics(1, T_OUT, DT);
    let k_b3 = b.step_physics(3, T_OUT, DT);
    assert_eq!(k_reject, 0.0, "the rejected step degrades to 0.0 kWh");
    assert_eq!(k_a3, k_b3, "the twins' step-3 energies agree");
    assert_eq!(
        a.setpoints.temperatures.as_ref(),
        b.setpoints.temperatures.as_ref(),
        "the twins' committed air must compose bit-exactly"
    );
    assert_eq!(
        a.mass.mass_temperatures.as_ref(),
        b.mass.mass_temperatures.as_ref(),
        "the twins' committed mass must compose bit-exactly"
    );
    assert_eq!(
        a.mass.solar_lag.as_ref(),
        b.mass.solar_lag.as_ref(),
        "the twins' solar-lag state must compose bit-exactly"
    );
    assert_eq!(
        b.setpoints.temperatures.as_ref()[0],
        36.78922134936172,
        "B's step-3 committed air (bit-exact probe literal)"
    );
    assert_eq!(
        a.mass.r76_sol_air_prev.as_ref(),
        b.mass.r76_sol_air_prev.as_ref(),
        "the twins' sol-air ramp sources agree"
    );
    assert_eq!(
        a.mass.r71_air_driver_num_prev.as_ref(),
        b.mass.r71_air_driver_num_prev.as_ref(),
        "the twins' num ramp sources agree"
    );
    assert_eq!(
        a.mass.r90_phi_st_prev.as_ref(),
        b.mass.r90_phi_st_prev.as_ref(),
        "the twins' surface-gain ramp sources agree"
    );
}

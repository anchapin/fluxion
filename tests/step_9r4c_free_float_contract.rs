//! 9R4C free-float committed-temperature chain contracts (loop round 131) —
//! the blitz's remaining mass-node component.
//!
//! Component under contract: the high-mass free-float step
//! `step_physics_9r4c` (`src/sim/thermal_model_physics/physics_impl/
//! step_9r4c.rs`) — the solver 900FF/950FF dispatch through
//! (`ZoneSolverKind::NineRFourC`, `thermal_selector.rs`; the from-spec
//! builder promotes HighMass constructions selector-independently,
//! #3277). The round-85 blitz suite (`mass_advance_contract.rs`) pins the
//! **5R1C** sub-step mass advance; the `MultiNodeSolver`'s own unit tests
//! (`src/physics/multi_node_solver/tests.rs`) pin the solver's methods in
//! isolation (directional/property pins). Nothing pinned the orchestration
//! chain this suite pins — the input→output map from the per-step driving
//! set to the COMMITTED free-float state (the FREE_FLOAT_BRAINSTORM_R85.md
//! rank-1 enumeration's per-class mass nodes for the high-mass family):
//!
//! - (1) the gains decomposition site: `phi_ia = load·conv + sol·sda`,
//!   `phi_st = load·rad + (sol − sol_to_air)·(1 − sbm)`,
//!   `phi_m = (sol − sol_to_air)·sbm + opaque`, with the bare-constructor
//!   defaults `conv 0.4 / sda 0.1 / sbm 0.6` (thermal_model_core mod.rs
//!   `new`) premise-pinned and the exact conservation identity
//!   `phi_ia + phi_st + phi_m == load_w + sol_w + opaque_w`;
//! - (2) the committed free-float air closed form — the parallel-resistance
//!   air balance (#1281, the #1859 production default mode) on the
//!   post-step mass nodes, `h_path_k = h_ms_k·h_is/(h_ms_k + h_is)`, with
//!   the solver's own `h_tr_is` (120.0 in the fixture — distinct from the
//!   model's 69.0 so the seam is observable);
//! - (3) the multi-node mass advance: per-node backward Euler with the
//!   per-surface `t_s_k = (h_ms_k·t_m + h_is·t_air)/(h_ms_k + h_is)`
//!   blend against `zone_temperature = t_air_mn_pre` (the free-float
//!   set), the `h_tr_ms`-proportional gains distribution (LIMIT-35 §12:
//!   phi_st + phi_m per envelope node), and the internal node composed
//!   through `internal_node_envelope_temperature` with the NODE's own
//!   `h_tr_me` (100.0) while the MODEL's `h_tr_me` (0.0) enters
//!   `term_rest_1`/the lumped `ts_den` — both seams pinned by the same
//!   bit-exact node literals;
//! - (4) the lumped mass backward Euler (the #896 `h_tr_3` air-side
//!   bottleneck, 49.83639567077775 W/K in the fixture) whose surface
//!   temperature blends `phi_st` with `t_i = t_i_act =` the **5R1C
//!   estimate** (29.428271397080522 at step 0), NOT the committed air —
//!   the lumped mass stays self-consistent with the 5R1C network while
//!   no longer driving the high-mass air (ADR-002/#1175);
//! - (5) the commit identity: free-float commits `t_i_free_mn` and early-
//!   returns 0.0 kWh exactly (Issue #738) — the committed air equals the
//!   multi-node value and is provably distinct from the 5R1C estimate;
//! - (6) the night-vent estimate arm: `h_ve_night = fan·ρ·cp/3600`
//!   (469.0, the r112-pinned zone-0 property read) rebuilds the 5R1C
//!   denominator through the vent-active-hour closed form
//!   (`den = h_ms_is_prod + tr1·(h_ext + h_ve_night) + gc` = 639632.2),
//!   reaching the lumped mass only through the estimate (ACH =
//!   fan/zone_volume = 10.80246913580247, the §5.2.2.1.5.2 fan);
//! - (7) the night-vent committed-air arm: the fan couples air-side ONLY
//!   (§5.2.1.10.1 / Issue #1898 / LIMIT-17) — `h_ve_night` enters the
//!   parallel air balance num+den and never the mass Euler, proven by the
//!   vent-on node literals composing bit-exactly from the vent-shifted
//!   `zone_temperature` alone (the r110/r112 air-side-only record at the
//!   9R4C committed-air site; the in-crate
//!   `night_vent_routes_air_side_only_h_tr_is_stays_built` test pins the
//!   solver-API-level premise this chain builds on);
//! - (8) the two-step composition: every step-N state composes
//!   bit-exactly from the step-(N−1) committed state — no hidden state
//!   (the committed air, the lumped mass, all four solver nodes, and the
//!   5R1C estimate at step 1) — plus the #4495 input-envelope guard: a
//!   NaN outdoor temperature rejects the step (0.0 kWh) with the zone air
//!   and lumped mass commits untouched, while the multi-node solver's own
//!   nodes HAVE advanced into NaN (committed inside `step_with_gains`
//!   before the guard) and poison subsequent steps — pinned as-is, the
//!   guard's "state untouched" scope being the commit, not the solver.
//!
//! Fixture: hand-rolled single-zone model (`ThermalModel::new(1)`, the
//! r112/mass_advance fixture recipe) with the selector set to
//! `NineRFourC`, one `MultiNodeSolver` in ParallelResistance mode
//! (asymmetric nodes 18/22/20 °C, solver `h_tr_is` 120.0), NO weather
//! (sol-air falls back to the outdoor dry-bulb; the exterior temps are
//! t_out/t_out/t_g with `ConstantGroundTemperature(13.0)`), zero
//! inter-zone coupling. Envelope time constants (min τ = 2e7/55 ≈ 3.6e5 s)
//! are far above `4·dt`, so `step_with_gains` runs exactly one Euler
//! sub-step — the closed forms are the single-step recurrences.
//!
//! Provenance: Python replication `/tmp/r131_replication.py` (dev-time
//! evidence), transcribed operation-for-operation from tip `25cc0ea4`;
//! every literal below was produced by the replication and matched the
//! engine bit-for-bit on the scratch probe before the suite was written
//! (both vent arms, two steps, all four nodes + internal). The first
//! fixture draft dispatched 5R1C (the selector premise was missing) —
//! corrected before any pin was written; disclosed in the duck review.
//!
//! Round-109/112 suite shape: characterization pins with bit-exact teeth,
//! no engine change, no re-record owed (battery bit-identity holds at
//! Δ = 0 — the suite writes no engine state). A violation means the 9R4C
//! chain moved and must be re-examined, never tuned around.

use fluxion::physics::cta::VectorField;
use fluxion::physics::multi_node_solver::MultiNodeSolver;
use fluxion::sim::boundary::{ConstantGroundTemperature, GroundTemperature};
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::multi_node_thermal::{MassAirCouplingMode, ThermalMassNode};
use fluxion::sim::thermal_selector::ZoneSolverKind;
use fluxion::validation::NightVentilation;

const DT: f64 = 3600.0;
const T_OUT: f64 = -10.0;
const T_G: f64 = 13.0;

/// The replication's h_path set (parallel-resistance series pairs, W/K):
/// h_path_w = 50·120/170, h_path_r = 40·120/160 (exactly 30.0),
/// h_path_f = 60·120/180 (exactly 40.0).
const H_PATH_W: f64 = 35.294117647058826;
const H_PATH_R: f64 = 30.0;
const H_PATH_F: f64 = 40.0;
const H_PATH_TOTAL: f64 = 105.29411764705883;

/// The replication's derived-cache anchors (single zone, model
/// `h_tr_is` 69.0, `h_tr_me` 0.0 — premise-pinned in contract 1).
const DEN_OFF: f64 = 138271.19999999998;
const GROUND_COEFF: f64 = 7696.799999999999;
const H_TR_3: f64 = 49.83639567077775;
const DEN_ON: f64 = 639632.2000000001;
const H_VE_NIGHT: f64 = 469.0;
const ACH_NIGHT: f64 = 10.80246913580247;

/// The night-vent fan, m³/h (Table 5-160, the r110/r112 fixture value).
const FAN_M3_H: f64 = 1400.0;

/// The hand-rolled single-zone 9R4C free-float fixture.
fn fixture_9r4c(vent: Option<NightVentilation>) -> ThermalModel<VectorField> {
    let mut m = ThermalModel::<VectorField>::new(1);
    m.setpoints.zone_area = VectorField::from_scalar(48.0, 1);
    m.setpoints.zone_volume = VectorField::from_scalar(129.6, 1);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.air_density = VectorField::from_scalar(1.2, 1);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 1);
    m.hvac.free_float = true;
    m.hvac.night_ventilation = vent;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0]);
    // The r112 fixture's own expression (op order mirrored by the replication).
    m.conduction.h_ve = VectorField::new(vec![1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0)]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0]);
    m.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0]);
    m.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.loads = VectorField::from_scalar(30.0, 1);
    m.solar.solar_gains = VectorField::new(vec![400.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.hvac.thermal_selector.zone_solver = ZoneSolverKind::NineRFourC;
    m.conduction.ground_temperature = Box::new(ConstantGroundTemperature::new(T_G));
    m.update_optimization_cache();
    // One 9R4C solver; ParallelResistance is the 9R4C production default
    // (#1859). Asymmetric node temperatures and a solver `h_tr_is` distinct
    // from the model's 69.0 make every coupling seam observable.
    let wall = ThermalMassNode::new(18.0, 4.0e7, 50.0, 20.0);
    let roof = ThermalMassNode::new(22.0, 2.0e7, 40.0, 15.0);
    let floor = ThermalMassNode::new(20.0, 3.0e7, 60.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 1.0e7, 10.0, 5.0).with_h_tr_me(100.0);
    let solver = MultiNodeSolver::new_with_mode(
        120.0,
        wall,
        roof,
        floor,
        internal,
        MassAirCouplingMode::ParallelResistance,
    );
    m.conduction.backend.multi_node_solvers = vec![solver];
    m
}

/// The engine's own night-vent reads (zone 0's air properties via
/// `.first()`, the r112-pinned convention): 1400·1.2·1005/3600 = 469.0.
fn h_ve_night_engine_read(m: &ThermalModel<VectorField>) -> f64 {
    let rho = m
        .setpoints
        .air_density
        .as_ref()
        .first()
        .copied()
        .unwrap_or(1.2);
    let cp = m
        .setpoints
        .heat_capacity
        .as_ref()
        .first()
        .copied()
        .unwrap_or(1005.0);
    FAN_M3_H * rho * cp / 3600.0
}

#[test]
/// Contract 1 — the premise set and the gains-decomposition conservation:
/// the bare-constructor solar defaults (conv 0.4 / sda 0.1 / sbm 0.6), the
/// model's `h_tr_is` 69.0 and `h_tr_me` 0.0 (so `term_rest_1` = 1069.0 —
/// the 9R4C lumped algebra is the 5R1C one), the cached denominator
/// 138271.19999999998 and ground coefficient 7696.799999999999, the #896
/// bottleneck 49.83639567077775, the engine's vent reads (469.0 W/K over
/// zone 0's 1.2/1005 properties; ACH 10.80246913580247 over 129.6 m³), and
/// the exact decomposition identity `phi_ia + phi_st + phi_m ==
/// load_w + sol_w + opaque_w` (= 23040.0 W at the fixture's 30/400/50
/// W/m² over 48 m²): the sda split redistributes transmitted solar
/// between the three sinks, nothing is created or lost.
fn fixture_premises_and_gains_conservation() {
    let m = fixture_9r4c(None);
    assert_eq!(m.solar.convective_fraction, 0.4, "premise: conv default");
    assert_eq!(
        m.solar.solar_distribution_to_air, 0.1,
        "premise: solar_distribution_to_air default (ThermalModel::new, not the r91 spec 0.0)"
    );
    assert_eq!(
        m.solar.solar_beam_to_mass_fraction, 0.6,
        "premise: solar_beam_to_mass_fraction default"
    );
    assert_eq!(
        m.conduction.h_tr_is.as_ref()[0],
        69.0,
        "premise: model h_tr_is"
    );
    assert_eq!(m.mass.h_tr_me.as_ref()[0], 0.0, "premise: model h_tr_me");
    assert_eq!(
        m.conduction.derived_term_rest_1.as_ref()[0],
        1069.0,
        "premise: term_rest_1 = h_tr_ms + h_tr_is + h_tr_me"
    );
    assert_eq!(
        m.conduction.derived_h_ms_is_prod.as_ref()[0],
        69000.0,
        "premise: h_ms_is_prod = h_tr_ms · h_tr_is"
    );
    assert_eq!(
        m.conduction.derived_den.as_ref()[0],
        DEN_OFF,
        "premise: cached den (Python-replicated bits)"
    );
    assert_eq!(
        m.conduction.derived_ground_coeff.as_ref()[0],
        GROUND_COEFF,
        "premise: ground_coeff = term_rest_1 · h_tr_floor"
    );
    assert_eq!(
        m.conduction.derived_h_tr_3.as_ref()[0],
        H_TR_3,
        "premise: derived_h_tr_3 (#896 bottleneck)"
    );
    let hv = h_ve_night_engine_read(&m);
    assert_eq!(
        hv, H_VE_NIGHT,
        "premise: the vent conductance reads zone 0's properties"
    );
    assert_eq!(
        FAN_M3_H / 129.6,
        ACH_NIGHT,
        "premise: ACH = fan_capacity / zone_volume"
    );
    // The decomposition identity at the fixture's driving set.
    let load_w = 30.0 * 48.0;
    let sol_w = 400.0 * 48.0;
    let opaque_w = 50.0 * 48.0;
    let sol_to_air = sol_w * 0.1;
    let phi_ia = load_w * 0.4 + sol_to_air;
    let phi_st = load_w * 0.6 + (sol_w - sol_to_air) * 0.4;
    let phi_m = (sol_w - sol_to_air) * 0.6 + opaque_w;
    assert_eq!(phi_ia, 2496.0);
    assert_eq!(phi_st, 7776.0);
    assert_eq!(phi_m, 12768.0);
    assert_eq!(
        phi_ia + phi_st + phi_m,
        load_w + sol_w + opaque_w,
        "the sda split must conserve the driving set exactly"
    );
}

#[test]
/// Contract 2 — the committed free-float air closed form: after one step
/// the committed zone temperature equals the parallel-resistance air
/// balance on the post-step mass nodes, bit-exact against the
/// Python-replicated literal 35.204524659159546 (0x40419a2ddd31286c). The
/// pre-step balance is pinned by the solver's `zone_temperature`
/// (34.47988132764695 — the free-float `set_zone_temperature(t_air_mn_pre)`
/// premise): same closed form on the pre-step nodes. The h_path anchors
/// (35.294117647058826 / 30.0 / 40.0, total 105.29411764705883) are exact.
fn committed_air_is_the_parallel_balance_closed_form() {
    let mut m = fixture_9r4c(None);
    let hs_w = 50.0 * 120.0 / (50.0 + 120.0);
    let hs_r = 40.0 * 120.0 / (40.0 + 120.0);
    let hs_f = 60.0 * 120.0 / (60.0 + 120.0);
    assert_eq!(hs_w, H_PATH_W);
    assert_eq!(hs_r, H_PATH_R);
    assert_eq!(hs_f, H_PATH_F);
    assert_eq!(hs_w + hs_r + hs_f, H_PATH_TOTAL);
    let kwh = m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        s.zone_temperature, 34.47988132764695,
        "pre-step air (set_zone_temperature premise)"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        35.204524659159546,
        "committed air = the parallel balance on the POST-step nodes (bit-exact)"
    );
    assert_eq!(kwh, 0.0, "free-float returns exactly 0.0 kWh");
}

#[test]
/// Contract 3 — the multi-node mass advance: one Euler sub-step (the
/// envelope time constants are far above 4·dt) with the per-surface
/// `t_s_k` blend against `zone_temperature`, the `h_tr_ms`-proportional
/// gains distribution (phi_st + phi_m per envelope node), and the internal
/// node composed through its own `h_tr_me` (100.0) while the model's
/// `h_tr_me` (0.0) stays in `term_rest_1` — all four node literals
/// bit-exact against the replication (wall 18.614397156250497,
/// roof 22.957622892533212, floor 21.038492096759924,
/// internal 20.605551422826654).
fn mn_nodes_compose_the_euler_closed_forms() {
    let mut m = fixture_9r4c(None);
    m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(s.mass.wall.temperature, 18.614397156250497, "wall node");
    assert_eq!(s.mass.roof.temperature, 22.957622892533212, "roof node");
    assert_eq!(s.mass.floor.temperature, 21.038492096759924, "floor node");
    assert_eq!(
        s.mass.internal.temperature, 20.605551422826654,
        "internal node (own h_tr_me seam)"
    );
}

#[test]
/// Contract 4 — the lumped mass backward Euler and its self-consistency
/// premise: the committed `mass_temperatures` equal the #896 closed form
/// (26.149556909916907 at step 0) whose surface temperature blends
/// `phi_st` with `t_i = t_i_act =` the 5R1C ESTIMATE (29.428271397080522),
/// NOT the committed air (35.204524659159546) — the discriminator is the
/// value itself: composing from the estimate matches bit-exactly, and the
/// estimate is provably distinct from the committed air (the ADR-002
/// split: the lumped mass rides the 5R1C network, the committed air is
/// the multi-node value).
fn lumped_mass_advances_on_the_5r1c_estimate() {
    let mut m = fixture_9r4c(None);
    m.step_physics(0, T_OUT, DT);
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        26.149556909916907,
        "lumped mass step 0 (bit-exact)"
    );
    // The estimate, recomposed in-test from the pinned closed form
    // (num/den, Python-replicated operand order).
    let est = {
        let msip = 69000.0_f64;
        let tr1 = 1069.0_f64;
        let gc = GROUND_COEFF;
        let phi_ia = 2496.0_f64;
        let phi_st = 7776.0_f64;
        let t_sol_air = T_OUT; // no weather: sol-air falls back to the dry-bulb
        let mut num_rest = phi_ia + 57.6 * t_sol_air;
        num_rest = num_rest * tr1;
        num_rest = num_rest + gc * T_G;
        (msip * 20.0 + 69.0 * phi_st + num_rest) / DEN_OFF
    };
    assert_eq!(
        est, 29.428271397080522,
        "the 5R1C estimate, in-test recomposition"
    );
    assert_ne!(
        est,
        m.setpoints.temperatures.as_ref()[0],
        "the estimate and the committed air must stay distinct in this fixture"
    );
}

#[test]
/// Contract 5 — the commit identity: free-float commits `t_i_free_mn` and
/// early-returns 0.0 kWh exactly (Issue #738 / ADR-002). The committed air
/// equals the multi-node value recomposed in-test from the post-step
/// nodes through the same parallel balance, and is distinct from the 5R1C
/// estimate the lumped mass rode (contract 4's discriminator, at the
/// commit site).
fn free_float_commits_the_multi_node_air() {
    let mut m = fixture_9r4c(None);
    let kwh = m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    let recomposed = {
        let (tw, tr, tf) = (
            s.mass.wall.temperature,
            s.mass.roof.temperature,
            s.mass.floor.temperature,
        );
        let phi_ia = 2496.0_f64;
        let num = H_PATH_W * tw + H_PATH_R * tr + H_PATH_F * tf + 21.6 * T_OUT + 0.0 * 0.0 + phi_ia;
        num / (H_PATH_TOTAL + 21.6 + 0.0)
    };
    assert_eq!(kwh, 0.0);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        recomposed,
        "committed air == the parallel balance recomposed from the committed nodes"
    );
    assert_eq!(m.setpoints.temperatures.as_ref()[0], 35.204524659159546);
    assert_ne!(
        m.setpoints.temperatures.as_ref()[0],
        29.428271397080522,
        "the commit is NOT the 5R1C estimate (ADR-002 high-mass semantics)"
    );
}

#[test]
/// Contract 6 — the night-vent estimate arm: with the fan attached the
/// 5R1C denominator rebuilds through the vent-active-hour closed form
/// (`h_ms_is_prod + tr1·(h_ext + h_ve_night) + gc` = 639632.2000000001,
/// the same operand order as the cached composition) and the estimate
/// moves from +29.43 to −1.48 °C; the lumped mass composes from the
/// vent-shifted estimate bit-exactly (26.10141928940402). The fan reaches
/// the lumped mass ONLY through the estimate — the mass Euler itself is
/// vent-free (the r110/R86 air-side-only record, at this path's site).
fn vent_rebuilds_the_estimate_denominator() {
    let mut m = fixture_9r4c(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let hv = h_ve_night_engine_read(&m);
    assert_eq!(hv, H_VE_NIGHT);
    // The vent-hour den closed form, recomposed in-test (the rebuild's
    // operand order mirrors the cached composition).
    let den_on = 69000.0 + 1069.0 * (57.6 + hv) + GROUND_COEFF;
    assert_eq!(den_on, DEN_ON, "vent-hour den rebuild closed form");
    let est_on = {
        let phi_ia = 2496.0_f64;
        let phi_st = 7776.0_f64;
        let mut num_rest = phi_ia + (57.6 + hv) * T_OUT;
        num_rest = num_rest * 1069.0;
        num_rest = num_rest + GROUND_COEFF * T_G;
        (69000.0 * 20.0 + 69.0 * phi_st + num_rest) / DEN_ON
    };
    assert_eq!(est_on, -1.4766730005149835, "vent-on estimate");
    let kwh = m.step_physics(0, T_OUT, DT);
    assert_eq!(kwh, 0.0);
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        26.10141928940402,
        "lumped mass composes from the vent-shifted estimate (bit-exact)"
    );
}

#[test]
/// Contract 7 — the night-vent committed-air arm: the fan couples
/// air-side ONLY (§5.2.1.10.1; Issue #1898 / LIMIT-17). The committed air
/// carries the vent term (−0.400967803731356 vs +35.20 fan-less), and the
/// vent-on NODE literals compose bit-exactly from the vent-shifted
/// `zone_temperature` alone — if the mass Euler consumed `h_ve_night`
/// directly (double-counted), the replication would miss. The h_ve_night
/// add is zone-0-gated by construction here (single zone = the conditioned
/// zone the §5.2.2.1.5.2 fan serves).
fn vent_couples_air_side_only_in_the_committed_chain() {
    let mut m = fixture_9r4c(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        -0.400967803731356,
        "vent-on committed air (bit-exact)"
    );
    assert_eq!(
        s.mass.wall.temperature, 18.503891443511804,
        "vent-on wall node"
    );
    assert_eq!(
        s.mass.roof.temperature, 22.770432846153163,
        "vent-on roof node"
    );
    assert_eq!(
        s.mass.floor.temperature, 20.871853436757974,
        "vent-on floor node"
    );
    assert_eq!(
        s.mass.internal.temperature, 19.19906528765946,
        "vent-on internal node"
    );
}

#[test]
/// Contract 8 — the two-step composition and the input-envelope guard:
/// every step-1 state composes bit-exactly from the step-0 committed
/// state (no hidden state: the committed air 35.927198831761395, the
/// lumped mass 32.24857751084929, the nodes, and the estimate
/// 32.49701909569214; `zone_temperature` equals the prior committed air —
/// the set_zone_temperature premise). Then a NaN outdoor temperature
/// rejects the step: 0.0 kWh returned, the zone air and lumped mass
/// commits untouched (both still the step-1 values) — while the
/// multi-node solver's own nodes HAVE advanced into NaN (committed inside
/// `step_with_gains` before the #4495 mass guard) and poison subsequent
/// steps; pinned as-is — the guard's "state untouched" scope is the
/// commit, not the solver.
fn two_step_composition_and_the_nan_guard() {
    let mut m = fixture_9r4c(None);
    m.step_physics(0, T_OUT, DT);
    let kwh1 = m.step_physics(1, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(kwh1, 0.0);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        35.927198831761395,
        "step-1 committed air (bit-exact)"
    );
    assert_eq!(
        s.zone_temperature, 35.204524659159546,
        "zone_temperature == prior committed"
    );
    assert_eq!(s.mass.wall.temperature, 19.228043322589375);
    assert_eq!(s.mass.roof.temperature, 23.91143979294904);
    assert_eq!(s.mass.floor.temperature, 22.074254458703006);
    assert_eq!(s.mass.internal.temperature, 21.22471359759717);
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        32.24857751084929,
        "step-1 lumped mass (bit-exact)"
    );
    // The vent-on step-1 committed air composes too.
    let mut v = fixture_9r4c(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    v.step_physics(0, T_OUT, DT);
    v.step_physics(1, T_OUT, DT);
    assert_eq!(
        v.setpoints.temperatures.as_ref()[0],
        -0.27452005481529207,
        "vent-on step-1 committed air (bit-exact)"
    );
    assert_eq!(v.mass.mass_temperatures.as_ref()[0], 32.148980580323496);
    // The guard: a NaN driving temperature degrades the step.
    let kwh_nan = v.step_physics(2, f64::NAN, DT);
    assert_eq!(kwh_nan, 0.0, "the rejected step returns 0.0 kWh");
    assert_eq!(
        v.setpoints.temperatures.as_ref()[0],
        -0.27452005481529207,
        "the zone-air commit is untouched by the rejected step"
    );
    assert_eq!(
        v.mass.mass_temperatures.as_ref()[0],
        32.148980580323496,
        "the lumped-mass commit is untouched by the rejected step"
    );
    let s = &v.conduction.backend.multi_node_solvers[0];
    assert!(
        s.mass.wall.temperature.is_nan()
            && s.mass.roof.temperature.is_nan()
            && s.mass.floor.temperature.is_nan(),
        "the solver nodes advanced into NaN before the guard — pinned as-is"
    );
}

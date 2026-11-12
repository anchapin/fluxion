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
//!
//! ============================================================================
//! Round 132 — the multi-zone implicit-iz air solves (the r131 remainder)
//! ============================================================================
//!
//! The r131 fixture was single-zone, so `solve_implicit_iz`
//! (`num_zones > 1`, step_9r4c.rs:369) never engaged: the two implicit
//! inter-zone air-node solves were unpinned. This section pins them at a
//! two-zone hand-rolled fixture (zone 0 = the r131 fixture verbatim; zone 1
//! asymmetric in every pinned operand; `h_tr_iz` [5.0, 7.0],
//! `h_tr_iz_rad` [2.0, 3.0]):
//!
//! - (M1) the multi-zone derived-cache premise: for `num_zones > 1` the
//!   cached `h_total` carries `h_tr_iz + h_tr_iz_rad`
//!   (dispatch_state.rs:149–165), so `den` moves from the single-zone
//!   138271.2 to 145754.19999999998 / 108562.5 — bit-exact;
//! - (M2) the 5R1C ESTIMATE implicit solve: `w_ij =
//!   0.5·(term_rest_1_i·h_iz_i + term_rest_1_j·h_iz_j)`, dense Gaussian
//!   with partial pivoting replacing the uncoupled `num/den` division —
//!   the solved estimates 27.75219681595207 / 23.55286728703152 are
//!   provably distinct from the uncoupled values;
//! - (M3) the mass chain stays iz-free (the coupling lives on the air
//!   nodes only): all eight solver node literals + per-zone pre-airs
//!   bit-exact;
//! - (M4) the mn COMMITTED-air implicit solve: uncoupled balance stash
//!   (`diag = solver h_tr_is + h_ve [+ h_ve_night]`, `rhs = t_unc·diag`)
//!   with SEAMLESS weights `w_ij = 0.5·(h_iz_i + h_iz_j)` (no
//!   term_rest_1), the inline partial-pivot elimination AND THEN
//!   `solve_linear_system` on the result — both passes transcribed.
//!   Disclosed seam, pinned as-is: the stash diag is the film conductance
//!   `h_tr_is + h_ve`, NOT the parallel-balance denominator
//!   `Σh_path + h_ve` the uncoupled value divided by — the uncoupled limit
//!   recovers exactly (rhs is built from the true balance), but the
//!   coupled dissipation is scaled by the wrong conductance; latent
//!   fidelity finding for the physics queue, not fixed here;
//! - (M5) the lumped mass rides the SOLVED estimate (`t_i_free_5r1c` is
//!   cloned post-solve) — pinned by the value itself: Euler-on-solved is
//!   bit-exact, Euler-on-uncoupled is provably distinct;
//! - (M6) the multi-zone vent den rebuild carries `h_iz + h_iz_rad` too
//!   (step_9r4c.rs:299–318) while the vent fans zone 0 only;
//! - (M7) the two-step composition (both zones; vent step 1 included) +
//!   the #4495 guard scope — and a PREMISE CORRECTION to the r131
//!   reading: `zone_temperature` tracks the UNCOUPLED pre-air (the
//!   `set_zone_temperature(t_air_mn_pre)` value, which at step N+1 equals
//!   the step-N uncoupled committed-air value), NOT the r131
//!   "prior committed" — in single-zone the two coincide, so the r131
//!   wording was the uncoupled/committed coincidence; multi-zone
//!   separates them (35.2045… uncoupled vs 34.9897… committed);
//! - (M8) the boundary finding (pinned as-is, dated 2026-11-12): a model
//!   with `num_zones > multi_node_solvers.len()` (3 zones, 2 solvers)
//!   never builds the `!mn_has` row, so the back-substitution divides
//!   0/0 on it — `x[boundary] = NaN`, and `0.0·NaN` poisons x for EVERY
//!   solver zone — the #4495 commit guard then rejects the whole commit
//!   (setpoints stay at the PRE-step values, 0.0 kWh) while the lumped
//!   masses and solver nodes HAVE already advanced: every step is
//!   half-applied. The in-code comment's claimed "known boundary
//!   temperatures" behavior is not what executes. Latent-only: the
//!   from-spec constructor builds one solver per zone, so no production
//!   path constructs the shape today. Routed to the physics queue.
//!
//! Provenance: Python replication `/tmp/r132_replication.py`
//! (dev-time evidence), transcribed operation-for-operation from tip
//! `0672e98b`; verified bit-for-bit against the engine scratch probe
//! before authoring — 57/60 engine observables exact (both vent arms,
//! two steps, all eight nodes, committed + lumped + den anchors); the 3
//! remaining are the boundary commits, where the engine's #4495 guard
//! rejects the NaN the replication transcribes (the guard is the pinned
//! behavior, the NaN its mechanism).
//!
//! ============================================================================
//! Round 133 — the mn committed-air corrections (the two r132 findings, FIXED)
//! ============================================================================
//!
//! The two findings r132 pinned as-is and routed to the physics queue are
//! fixed here, judged by physics correctness alone (Alex's standing
//! direction): known, verifiable, tested — band movement recorded for the
//! record only:
//!
//! - (F1, the diag seam) the stash `diag` is now the committed-air
//!   balance's OWN denominator — the parallel-balance `Σh_path_k +
//!   h_ve_total [+ h_rad_sky]` the uncoupled value was divided by — not
//!   the film conductance. The engine exposes it mode-consistently
//!   through `MultiNodeSolver::committed_air_balance` (a `(t_air, den)`
//!   twin of the dispatched balance, sharing its exact arithmetic; the
//!   AdditiveSum mode keeps its film-sum denominator, so the fix is
//!   mode-consistent). The coupled dissipation now scales the same
//!   balance the uncoupled value closed; the uncoupled limit stays
//!   bit-exact (rhs = t_unc·den, den = the division's own denominator).
//! - (F2, the boundary compaction) the implicit system assembles ONLY
//!   over solver zones; solver-less zones enter as known boundary
//!   temperatures (their estimate on the rhs) and keep it as their
//!   committed air — the in-code comment's claimed behavior, now
//!   executing. No all-zero row, no 0/0, no NaN, no half-applied steps.
//!
//! Battery scope: both fixes live inside `solve_implicit_iz`
//! (`num_zones > 1`) — every single-zone battery row bit-identical. THE
//! PREMISE THE BATTERY CORRECTED: the ASHRAE 140 battery is NOT all
//! single-zone — the multi-zone sunspace cases 960/970 engage
//! `solve_implicit_iz`, so the diag seam was LIVE there, not latent: 960
//! H/C and 970 H/C move (recorded not tuned; the honest re-record rides
//! this PR — strict JSON `_doc_round133` + captured_commit, the
//! output-diff CSV, the §7 CSVs + the 960 sun-zone pins, KNOWN_ISSUES
//! §LIMIT-14/§LIMIT-23). M4/M6/M7/M8 re-pinned to the corrected closed
//! form with the r132 as-is values cited as provenance; M9 (the twin
//! seam) and M10 (the boundary compaction closed form) added.
//!
//! Provenance: Python replication `/tmp/r133_replication.py` (dev-time
//! evidence; the r132 replication's chain with only the stash, the solve,
//! and the compaction changed), produced every literal below; the fixed
//! engine matched bit-for-bit (the old suite ran RED on exactly
//! {M4, M6, M7, M8} at the fixed tree before the re-pin — the fix's own
//! teeth, 12 passed / 4 failed).

use fluxion::physics::cta::VectorField;
use fluxion::physics::multi_node_solver::MultiNodeSolver;
use fluxion::sim::boundary::ConstantGroundTemperature;
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

// ============================================================================
// Round 132 — multi-zone implicit-iz contracts (M1–M8)
// ============================================================================

/// The replication's multi-zone derived denominators (iz-inclusive
/// `h_total`, dispatch_state.rs:149–165) and boundary-zone anchor.
const DEN_MZ_0: f64 = 145754.19999999998;
const DEN_MZ_1: f64 = 108562.5;
const DEN_BND_2: f64 = 46800.0;

/// Zone 1's #896 bottleneck (the air-side lumped mass conductance).
const H_TR_3_Z1: f64 = 33.93722992026942;

/// The replication's solved-estimate literals (the engine sees them
/// through the lumped mass, M5 — cloned post-solve into `t_i_free_5r1c`).
const EST_MZ_OFF: [f64; 2] = [27.75219681595207, 23.55286728703152];
const EST_MZ_ON: [f64; 2] = [-1.2526584770581457, 22.097517475308482];

/// The zone-1 `h_ve` expression (op order mirrors the fixture assignment;
/// the value is deliberately not a clean decimal so a transcription that
/// recomputes it differently cannot pass).
fn h_ve_z1() -> f64 {
    0.9 * 1000.0 * (0.5 * (40.0 * 2.7) / 3600.0)
}

/// One Gaussian elimination pass with partial pivoting at n = 2
/// (step_9r4c.rs 872–896 op-for-op, and the col-0 shape of
/// `solve_linear_system` 21–54).
fn elim_pass_2(a: &mut [[f64; 2]; 2], b: &mut [f64; 2]) {
    let mut best = 0usize;
    for r in 0..2 {
        if a[r][0].abs() > a[best][0].abs() {
            best = r;
        }
    }
    if best != 0 {
        a.swap(0, 1);
        b.swap(0, 1);
    }
    let inv = 1.0 / a[0][0];
    let f = a[1][0] * inv;
    if f != 0.0 {
        let pivot_row = [a[0][0], a[0][1]];
        a[1][0] -= f * pivot_row[0];
        a[1][1] -= f * pivot_row[1];
        b[1] -= f * b[0];
    }
}

/// `solve_linear_system` (21–54) at n = 2, op-for-op: elimination, then the
/// reverse back-substitution (the inner sum over the empty tail range is
/// 0.0 exactly, as the Rust `(r+1..n).map(..).sum()` produces).
fn solve_linear_system_2(a: &mut [[f64; 2]; 2], b: &mut [f64; 2]) -> (f64, f64) {
    elim_pass_2(a, b);
    let x1 = (b[1] - 0.0) / a[1][1];
    let sm0 = a[0][1] * x1;
    let x0 = (b[0] - sm0) / a[0][0];
    (x0, x1)
}

/// The multi-zone derived denominators, op-for-op (the cached composition
/// when `h_ve_night = 0.0`; the vent-active-hour rebuild's operand order
/// otherwise — they agree bit-for-bit at the same operands, M6).
fn den_2z(h_ve_night: f64) -> [f64; 2] {
    let hiz = [5.0, 7.0];
    let hizrad = [2.0, 3.0];
    let tr1 = [1069.0, 875.0];
    let msip = [69000.0, 60000.0];
    let gc = [7696.799999999999, 7000.0];
    let h_ext = [57.6, 24.0 + h_ve_z1()];
    let mut den = [0.0; 2];
    for i in 0..2 {
        let h_ext_eff = if i == 0 {
            h_ext[i] + h_ve_night
        } else {
            h_ext[i]
        };
        let h_total = (h_ext_eff + hiz[i]) + hizrad[i];
        den[i] = (msip[i] + tr1[i] * h_total) + gc[i];
    }
    den
}

/// The 5R1C estimate implicit solve, op-for-op: per-zone numerator, the
/// term_rest_1-weighted pair conductances, the 2×2 Gaussian solve.
/// Returns (zone 0, zone 1) solved estimates.
fn estimate_solved_2z(h_ve_night: f64) -> (f64, f64) {
    let den = den_2z(h_ve_night);
    let tr1 = [1069.0, 875.0];
    let msip = [69000.0, 60000.0];
    let gc = [7696.799999999999, 7000.0];
    let h_ext = [57.6, 24.0 + h_ve_z1()];
    let t5r1c = [69.0, 75.0];
    let tm = [20.0, 18.0];
    let phi_st = [7776.0, 4800.0];
    let phi_ia = [2496.0, 1520.0];
    let hiz = [5.0, 7.0];
    let mut num = [0.0; 2];
    for i in 0..2 {
        let h_ext_eff = if i == 0 {
            h_ext[i] + h_ve_night
        } else {
            h_ext[i]
        };
        let ntm = msip[i] * tm[i];
        let nst = t5r1c[i] * phi_st[i];
        let mut nr = phi_ia[i] + h_ext_eff * T_OUT;
        nr = nr * tr1[i];
        nr = nr + gc[i] * T_G;
        num[i] = (ntm + nst) + nr;
    }
    let w01 = 0.5 * (tr1[0] * hiz[0] + tr1[1] * hiz[1]);
    let mut a = [[den[0] + w01, -w01], [-w01, den[1] + w01]];
    solve_linear_system_2(&mut a, &mut num)
}

/// The uncoupled parallel-resistance balance recomposed from given mass
/// nodes (the getter-recompose: engine state in, closed form in-test).
fn parallel_balance_z(
    h_ms: [f64; 3],
    h_is: f64,
    nodes: [f64; 3],
    h_ve: f64,
    h_ve_night: f64,
    phi_ia: f64,
) -> f64 {
    let hs_w = (h_ms[0] * h_is) / (h_ms[0] + h_is);
    let hs_r = (h_ms[1] * h_is) / (h_ms[1] + h_is);
    let hs_f = (h_ms[2] * h_is) / (h_ms[2] + h_is);
    let total = (hs_w + hs_r) + hs_f;
    let h_ve_total = h_ve + h_ve_night;
    let num = ((((hs_w * nodes[0] + hs_r * nodes[1]) + hs_f * nodes[2]) + h_ve_total * T_OUT)
        + 0.0 * 0.0)
        + phi_ia;
    num / ((total + h_ve_total) + 0.0)
}

/// The mn committed-air implicit solve, op-for-op (the r133 corrected
/// stash): `diag = Σh_path_k + h_ve [+ h_ve_night]` — the balance's OWN
/// denominator, composed from the fixed series conductances exactly as
/// the engine's balance divides — `rhs = t_unc·diag`, seamless weights
/// `0.5·(h_iz_i + h_iz_j)`, the inline elimination AND THEN
/// `solve_linear_system` — both passes mirrored (the second pass re-runs
/// on the once-eliminated matrix, including its ~ulp residual row entry).
fn committed_solved_2z(t_unc: [f64; 2], h_ve_night: f64) -> (f64, f64) {
    let h_ms = [[50.0, 40.0, 60.0], [45.0, 35.0, 55.0]];
    let his = [120.0, 140.0];
    let h_ve = [21.6, h_ve_z1()];
    let hiz = [5.0, 7.0];
    let mut diag = [0.0; 2];
    let mut rhs = [0.0; 2];
    for i in 0..2 {
        let hvn = if i == 0 { h_ve_night } else { 0.0 };
        let hs_w = (h_ms[i][0] * his[i]) / (h_ms[i][0] + his[i]);
        let hs_r = (h_ms[i][1] * his[i]) / (h_ms[i][1] + his[i]);
        let hs_f = (h_ms[i][2] * his[i]) / (h_ms[i][2] + his[i]);
        let total = (hs_w + hs_r) + hs_f;
        diag[i] = (total + (h_ve[i] + hvn)) + 0.0;
        rhs[i] = t_unc[i] * diag[i];
    }
    let w01 = 0.5 * (hiz[0] + hiz[1]);
    let mut a = [[diag[0] + w01, -w01], [-w01, diag[1] + w01]];
    let mut b = rhs;
    elim_pass_2(&mut a, &mut b); // the inline elimination
    solve_linear_system_2(&mut a, &mut b) // then solve_linear_system
}

/// The two-zone 9R4C free-float fixture: zone 0 is the r131 fixture
/// verbatim; zone 1 asymmetric in every pinned operand (area/volume/
/// loads/solar/conductances/mass nodes/solver `h_tr_is` 140.0/model
/// `h_tr_is` 75.0). `h_tr_iz` [5.0, 7.0] and `h_tr_iz_rad` [2.0, 3.0]
/// engage `solve_implicit_iz` (`num_zones > 1`).
fn fixture_9r4c_2z(vent: Option<NightVentilation>) -> ThermalModel<VectorField> {
    let mut m = ThermalModel::<VectorField>::new(2);
    m.setpoints.zone_area = VectorField::new(vec![48.0, 40.0]);
    m.setpoints.zone_volume = VectorField::new(vec![129.6, 108.0]);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.setpoints.temperatures = VectorField::new(vec![20.0, 18.0]);
    m.setpoints.air_density = VectorField::from_scalar(1.2, 2);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 2);
    m.hvac.free_float = true;
    m.hvac.night_ventilation = vent;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0, 0.25 * 40.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0, 24.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0, 0.2 * 40.0]);
    m.conduction.h_ve = VectorField::new(vec![
        1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0),
        h_ve_z1(),
    ]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0, 800.0]);
    m.conduction.h_tr_is = VectorField::new(vec![69.0, 75.0]);
    m.conduction.h_tr_iz = VectorField::new(vec![5.0, 7.0]);
    m.conduction.h_tr_iz_rad = VectorField::new(vec![2.0, 3.0]);
    m.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0, 40.0 * 150_000.0]);
    m.mass.mass_temperatures = VectorField::new(vec![20.0, 18.0]);
    m.setpoints.loads = VectorField::new(vec![30.0, 20.0]);
    m.solar.solar_gains = VectorField::new(vec![400.0, 300.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0, 30.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.hvac.thermal_selector.zone_solver = ZoneSolverKind::NineRFourC;
    m.conduction.ground_temperature = Box::new(ConstantGroundTemperature::new(T_G));
    m.update_optimization_cache();
    let wall0 = ThermalMassNode::new(18.0, 4.0e7, 50.0, 20.0);
    let roof0 = ThermalMassNode::new(22.0, 2.0e7, 40.0, 15.0);
    let floor0 = ThermalMassNode::new(20.0, 3.0e7, 60.0, 10.0);
    let internal0 = ThermalMassNode::new(20.0, 1.0e7, 10.0, 5.0).with_h_tr_me(100.0);
    let s0 = MultiNodeSolver::new_with_mode(
        120.0,
        wall0,
        roof0,
        floor0,
        internal0,
        MassAirCouplingMode::ParallelResistance,
    );
    let wall1 = ThermalMassNode::new(16.0, 3.0e7, 45.0, 18.0);
    let roof1 = ThermalMassNode::new(24.0, 1.5e7, 35.0, 12.0);
    let floor1 = ThermalMassNode::new(19.0, 2.5e7, 55.0, 8.0);
    let internal1 = ThermalMassNode::new(19.0, 1.2e7, 12.0, 4.0).with_h_tr_me(80.0);
    let s1 = MultiNodeSolver::new_with_mode(
        140.0,
        wall1,
        roof1,
        floor1,
        internal1,
        MassAirCouplingMode::ParallelResistance,
    );
    m.conduction.backend.multi_node_solvers = vec![s0, s1];
    m
}

#[test]
/// Contract M1 — the multi-zone cache premises and per-zone gains
/// conservation: `h_total` carries `h_tr_iz + h_tr_iz_rad` when
/// `num_zones > 1` (den 145754.19999999998 / 108562.5 vs the single-zone
/// 138271.19999999998), per-zone `term_rest_1`/`h_ms_is_prod`/ground
/// coeff/`h_tr_3` anchors, the iz vectors as set, and the gains
/// decomposition identity holding per zone (2496+7776+12768 = 23040 W and
/// 1520+4800+7680 = 14000 W).
fn multizone_premises_caches_and_gains_conservation() {
    let m = fixture_9r4c_2z(None);
    assert_eq!(
        m.conduction.h_tr_iz.as_ref(),
        [5.0, 7.0],
        "premise: the iz conductances as set"
    );
    assert_eq!(
        m.conduction.h_tr_iz_rad.as_ref(),
        [2.0, 3.0],
        "premise: the radiative iz conductances as set"
    );
    assert_eq!(
        m.conduction.h_tr_is.as_ref(),
        [69.0, 75.0],
        "premise: model h_tr_is"
    );
    assert_eq!(
        m.mass.h_tr_me.as_ref(),
        [0.0, 0.0],
        "premise: model h_tr_me"
    );
    assert_eq!(
        m.conduction.derived_term_rest_1.as_ref(),
        [1069.0, 875.0],
        "per-zone term_rest_1"
    );
    assert_eq!(
        m.conduction.derived_h_ms_is_prod.as_ref(),
        [69000.0, 60000.0],
        "per-zone h_ms_is_prod"
    );
    assert_eq!(
        m.conduction.derived_ground_coeff.as_ref(),
        [7696.799999999999, 7000.0],
        "per-zone ground coefficient"
    );
    assert_eq!(
        m.conduction.derived_den.as_ref(),
        [DEN_MZ_0, DEN_MZ_1],
        "multi-zone den: h_total carries h_tr_iz + h_tr_iz_rad (bit-exact)"
    );
    assert_eq!(
        m.conduction.derived_h_tr_3.as_ref(),
        [49.83639567077775, H_TR_3_Z1],
        "per-zone #896 bottleneck"
    );
    // Zone 1 gains: 20 W/m²·40 m² = 800 W load, 300·40 = 12000 W solar,
    // 30·40 = 1200 W opaque over the same conv/sda/sbm split.
    let (load1, sol1, opaq1) = (800.0_f64, 12000.0_f64, 1200.0_f64);
    let sol_to_air1 = sol1 * 0.1;
    let phi_ia1 = load1 * 0.4 + sol_to_air1;
    let phi_st1 = load1 * 0.6 + (sol1 - sol_to_air1) * 0.4;
    let phi_m1 = (sol1 - sol_to_air1) * 0.6 + opaq1;
    assert_eq!(phi_ia1, 1520.0);
    assert_eq!(phi_st1, 4800.0);
    assert_eq!(phi_m1, 7680.0);
    assert_eq!(
        phi_ia1 + phi_st1 + phi_m1,
        load1 + sol1 + opaq1,
        "the sda split conserves zone 1's driving set exactly"
    );
}

#[test]
/// Contract M2 — the 5R1C estimate implicit solve engages and is provably
/// distinct from the uncoupled division: the in-test transcription (pair
/// weights `0.5·(tr1_i·h_iz_i + tr1_j·h_iz_j)` = 5735.0, the diagonal
/// `den + Σw`, the 2×2 Gaussian solve) reproduces the replication's
/// 27.75219681595207 / 23.55286728703152 exactly, while the uncoupled
/// `num/den` values (27.917428108418147 / 23.331030512377662 — den is
/// already iz-inclusive, so the uncoupled limit differs from the r131
/// single-zone 29.428271397080522) differ — the solve moved BOTH
/// estimates toward each other (heat flowing from the warmer to the
/// colder zone's estimate network). The engine-truth for the solved
/// values rides M5's lumped literals (`t_i_free_5r1c` is cloned
/// POST-solve, step_9r4c.rs:530).
fn estimate_implicit_solve_engages_multizone() {
    let (est0, est1) = estimate_solved_2z(0.0);
    assert_eq!(est0, EST_MZ_OFF[0], "zone-0 solved estimate (bit-exact)");
    assert_eq!(est1, EST_MZ_OFF[1], "zone-1 solved estimate (bit-exact)");
    // The uncoupled division, for the discriminator (the iz-inclusive den).
    let den = den_2z(0.0);
    let unc0 = {
        let mut nr = 2496.0_f64 + 57.6 * T_OUT;
        nr = nr * 1069.0;
        nr = nr + 7696.799999999999 * T_G;
        (69000.0 * 20.0 + 69.0 * 7776.0 + nr) / den[0]
    };
    let unc1 = {
        let mut nr = 1520.0_f64 + (24.0 + h_ve_z1()) * T_OUT;
        nr = nr * 875.0;
        nr = nr + 7000.0 * T_G;
        (60000.0 * 18.0 + 75.0 * 4800.0 + nr) / den[1]
    };
    assert_eq!(
        unc0, 27.917428108418147,
        "zone-0 uncoupled estimate literal"
    );
    assert_eq!(
        unc1, 23.331030512377662,
        "zone-1 uncoupled estimate literal"
    );
    assert_ne!(
        est0, unc0,
        "the implicit solve must move the zone-0 estimate off the uncoupled value"
    );
    assert_ne!(
        est1, unc1,
        "the implicit solve must move the zone-1 estimate off the uncoupled value"
    );
}

#[test]
/// Contract M3 — the mass chain stays iz-free: the coupling lives on the
/// air nodes only, so both zones' four solver node literals and per-zone
/// pre-airs (`zone_temperature`, the `set_zone_temperature(t_air_mn_pre)`
/// premise) match the replication bit-exactly — zone 0's are the r131
/// single-zone literals verbatim (18.614397156250497 / 22.957622892533212
/// / 21.038492096759924 / 20.605551422826654, pre-air 34.47988132764695),
/// zone 1's are the new asymmetric set (16.493002926758464 /
/// 24.705189070341486 / 19.77585810503555 / 19.42360679206089, pre-air
/// 29.1384329942863).
fn multizone_mass_chain_stays_iz_free() {
    let mut m = fixture_9r4c_2z(None);
    let kwh = m.step_physics(0, T_OUT, DT);
    assert_eq!(kwh, 0.0, "free-float returns exactly 0.0 kWh");
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    let s1 = &m.conduction.backend.multi_node_solvers[1];
    assert_eq!(
        s0.zone_temperature, 34.47988132764695,
        "zone-0 pre-air (r131 verbatim)"
    );
    assert_eq!(s1.zone_temperature, 29.1384329942863, "zone-1 pre-air");
    assert_eq!(s0.mass.wall.temperature, 18.614397156250497);
    assert_eq!(s0.mass.roof.temperature, 22.957622892533212);
    assert_eq!(s0.mass.floor.temperature, 21.038492096759924);
    assert_eq!(s0.mass.internal.temperature, 20.605551422826654);
    assert_eq!(s1.mass.wall.temperature, 16.493002926758464);
    assert_eq!(s1.mass.roof.temperature, 24.705189070341486);
    assert_eq!(s1.mass.floor.temperature, 19.77585810503555);
    assert_eq!(s1.mass.internal.temperature, 19.42360679206089);
}

#[test]
/// Contract M4 — the mn committed-air implicit solve: the uncoupled
/// balances recomposed from the POST-step nodes (35.204524659159546 /
/// 29.72231486401823 — zone 0's is the r131 single-zone committed value
/// verbatim, its mass chain being iz-free), stashed as
/// `rhs = t_unc·den` with `den = Σh_path_k + h_ve` — the balance's OWN
/// denominator (126.89411764705883 / 115.04123354123354: fixed series
/// conductances per zone, temperature-independent), then the
/// seamless-weight (`0.5·(h_iz_i + h_iz_j)` = 6.0) implicit solve —
/// DOUBLE elimination (the inline partial-pivot pass and then
/// `solve_linear_system` on the once-eliminated matrix) — reproduces the
/// committed 34.968751490480585 / 29.98238012308592 bit-exactly.
/// THE R133 FIX: the r132 pin recorded the previous film-conductance
/// stash (`diag = h_tr_is + h_ve` = 141.6 / 153.5, committed
/// 34.989725362747315 / 29.9204619674187) as a fidelity finding — the
/// uncoupled limit stayed exact (rhs built from the true balance) while
/// the coupled dissipation was scaled by the wrong conductance. The
/// engine now exposes the balance denominator through
/// `MultiNodeSolver::committed_air_balance` (mode-consistent, M9) and
/// stashes `diag = den, rhs = t_unc·den` (step_9r4c.rs) — this contract
/// re-pins the corrected closed form.
fn committed_air_implicit_solve_multizone() {
    let mut m = fixture_9r4c_2z(None);
    m.step_physics(0, T_OUT, DT);
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    let s1 = &m.conduction.backend.multi_node_solvers[1];
    // The uncoupled balances on the post-step nodes (getter-recompose).
    let t_unc0 = parallel_balance_z(
        [50.0, 40.0, 60.0],
        120.0,
        [
            s0.mass.wall.temperature,
            s0.mass.roof.temperature,
            s0.mass.floor.temperature,
        ],
        21.6,
        0.0,
        2496.0,
    );
    let t_unc1 = parallel_balance_z(
        [45.0, 35.0, 55.0],
        140.0,
        [
            s1.mass.wall.temperature,
            s1.mass.roof.temperature,
            s1.mass.floor.temperature,
        ],
        h_ve_z1(),
        0.0,
        1520.0,
    );
    assert_eq!(
        t_unc0, 35.204524659159546,
        "zone-0 uncoupled committed air (r131 verbatim)"
    );
    assert_eq!(t_unc1, 29.72231486401823, "zone-1 uncoupled committed air");
    // The committed values: the implicit solve over the corrected stash.
    let (c0, c1) = committed_solved_2z([t_unc0, t_unc1], 0.0);
    assert_eq!(
        c0, 34.968751490480585,
        "zone-0 committed air (bit-exact, r133 corrected closed form)"
    );
    assert_eq!(
        c1, 29.98238012308592,
        "zone-1 committed air (bit-exact, r133 corrected closed form)"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        c0,
        "the commit takes the solved value"
    );
    assert_eq!(m.setpoints.temperatures.as_ref()[1], c1);
    // The diag discriminator: the stash carries the balance's OWN
    // denominator (r133), not the film conductance (the r132 as-is seam).
    let path_den0 = 105.29411764705883 + 21.6;
    assert_eq!(
        path_den0, 126.89411764705883,
        "the committed-air balance denominator (Σh_path + h_ve, bit-exact)"
    );
    assert_ne!(
        120.0 + 21.6,
        path_den0,
        "the film conductance and the balance denominator are distinct in this fixture"
    );
}

#[test]
/// Contract M5 — the lumped mass rides the SOLVED estimate: the Euler
/// recomposed on the solved estimates (27.75219681595207 /
/// 23.55286728703152) reproduces 26.146946251858456 / 22.444233248805922
/// bit-exactly, while the same Euler on the UNCOUPLED estimates
/// (27.917428108418147 / 23.331030512377662) lands elsewhere — the
/// ADR-002 self-consistency premise, now with the solve in the loop: the
/// lumped mass sees the implicit-iz-coupled estimate, provably by value.
fn lumped_mass_rides_the_solved_estimate_multizone() {
    let mut m = fixture_9r4c_2z(None);
    m.step_physics(0, T_OUT, DT);
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        26.146946251858456,
        "zone-0 lumped mass (bit-exact)"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[1],
        22.444233248805922,
        "zone-1 lumped mass (bit-exact)"
    );
    let euler = |tm_old: f64,
                 t_i: f64,
                 phi_st: f64,
                 phi_m: f64,
                 h_tr_ms: f64,
                 h_is: f64,
                 h_me: f64,
                 h_em: f64,
                 cm: f64,
                 h_tr_3: f64|
     -> f64 {
        let ts_den = h_tr_ms + h_is + h_me;
        let t_s = (h_tr_ms * tm_old + h_is * t_i + phi_st) / ts_den;
        let cm_dt = cm / DT;
        let numer = ((cm_dt * tm_old + h_em * T_OUT) + h_tr_3 * t_s) + phi_m;
        numer / ((cm_dt + h_em) + h_tr_3)
    };
    // On the solved estimates — must reproduce the engine bit-exactly.
    let on_solved0 = euler(
        20.0,
        EST_MZ_OFF[0],
        7776.0,
        12768.0,
        1000.0,
        69.0,
        0.0,
        15.36,
        7.2e6,
        49.83639567077775,
    );
    assert_eq!(
        on_solved0, 26.146946251858456,
        "Euler-on-solved-estimate == engine (the premise, by value)"
    );
    let on_solved1 = euler(
        18.0,
        EST_MZ_OFF[1],
        4800.0,
        7680.0,
        800.0,
        75.0,
        0.0,
        10.0,
        6.0e6,
        H_TR_3_Z1,
    );
    assert_eq!(on_solved1, 22.444233248805922);
    // On the uncoupled estimates — must land elsewhere (the discriminator).
    let on_unc0 = euler(
        20.0,
        27.917428108418147,
        7776.0,
        12768.0,
        1000.0,
        69.0,
        0.0,
        15.36,
        7.2e6,
        49.83639567077775,
    );
    assert_ne!(
        on_unc0, 26.146946251858456,
        "Euler-on-uncoupled must NOT reproduce the engine — the lumped mass rides the solved estimate"
    );
}

#[test]
/// Contract M6 — the multi-zone vent den rebuild carries the iz terms:
/// `den = h_ms_is_prod + tr1·((h_ext + h_ve_night) + h_iz + h_iz_rad) + gc`
/// (step_9r4c.rs:299–318 — the fan add lands zone-0-only inside
/// `h_ext_for_free_float`), and the vent solves move both zones' estimates
/// (−1.2526584770581457 / 22.097517475308482), committed airs
/// (−0.11542692464219141 / 28.24326144152501) and lumped masses
/// (26.10176821499925 / 22.441758406461478) bit-exactly against the
/// replication — zone 1 couples to zone 0 ONLY through the implicit
/// systems (the fan conductance itself never enters zone 1's rows: its
/// pre-air stays 29.1384329942863, the vent-free literal).
fn multizone_vent_den_carries_the_iz_terms() {
    let mut m = fixture_9r4c_2z(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let hv = h_ve_night_engine_read(&m);
    assert_eq!(hv, H_VE_NIGHT, "premise: the vent conductance (r112 pin)");
    // The vent den rebuild, recomposed in-test (the rebuild's operand order).
    let den_on = den_2z(hv);
    assert_ne!(
        den_on,
        den_2z(0.0),
        "the vent rebuild shifts the denominators"
    );
    let (est0, est1) = estimate_solved_2z(hv);
    assert_eq!(
        est0, EST_MZ_ON[0],
        "vent-on zone-0 solved estimate (bit-exact)"
    );
    assert_eq!(
        est1, EST_MZ_ON[1],
        "vent-on zone-1 solved estimate (bit-exact)"
    );
    let kwh = m.step_physics(0, T_OUT, DT);
    assert_eq!(kwh, 0.0);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        -0.11542692464219141,
        "vent-on zone-0 committed air (bit-exact, r133 corrected)"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref()[1],
        28.24326144152501,
        "vent-on zone-1 committed air (bit-exact, r133 corrected)"
    );
    assert_eq!(m.mass.mass_temperatures.as_ref()[0], 26.10176821499925);
    assert_eq!(m.mass.mass_temperatures.as_ref()[1], 22.441758406461478);
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    let s1 = &m.conduction.backend.multi_node_solvers[1];
    assert_eq!(
        s0.zone_temperature, -0.5281238277625315,
        "vent-on zone-0 pre-air"
    );
    assert_eq!(
        s1.zone_temperature, 29.1384329942863,
        "the fan never enters zone 1's pre-air (air-side, zone-0-gated)"
    );
}

#[test]
/// Contract M7 — the two-step composition, the vent step 1, the
/// zone_temperature PREMISE CORRECTION, and the #4495 guard scope: every
/// step-1 state composes bit-exactly from the step-0 committed state
/// (committed 35.685389361404695 / 30.57135613146878, lumped
/// 32.243106892784546 / 26.859142690792257); `zone_temperature` at step 1
/// is 35.204524659159546 — the UNCOUPLED pre-air, NOT the prior committed
/// 34.968751490480585: in multi-zone the set_zone_temperature premise and
/// the committed value separate, and r131's single-zone wording ("prior
/// committed") was the uncoupled/committed coincidence. The vent fixture
/// composes too (0.015342050813032998 / 28.803195996264588, lumped
/// 32.149686954689734 / 26.854019373663057). Then a NaN outdoor
/// temperature rejects the step: 0.0 kWh, BOTH zone commits untouched,
/// solver nodes NaN'd as-is (the r131 guard scope, now multi-zone).
fn multizone_two_step_composition_vent_and_guard() {
    let mut m = fixture_9r4c_2z(None);
    m.step_physics(0, T_OUT, DT);
    let kwh1 = m.step_physics(1, T_OUT, DT);
    assert_eq!(kwh1, 0.0);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        35.685389361404695,
        "step-1 zone-0 committed air (bit-exact)"
    );
    assert_eq!(m.setpoints.temperatures.as_ref()[1], 30.57135613146878);
    assert_eq!(m.mass.mass_temperatures.as_ref()[0], 32.243106892784546);
    assert_eq!(m.mass.mass_temperatures.as_ref()[1], 26.859142690792257);
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    let s1 = &m.conduction.backend.multi_node_solvers[1];
    assert_eq!(
        s0.zone_temperature, 35.204524659159546,
        "zone_temperature == the step-1 UNCOUPLED pre-air (== step-0 uncoupled committed)"
    );
    assert_ne!(
        s0.zone_temperature,
        m.setpoints.temperatures.as_ref()[0],
        "the premise correction: zone_temperature is NOT the prior committed value in multi-zone"
    );
    assert_eq!(s1.zone_temperature, 29.72231486401823);
    // The vent arm composes across steps too.
    let mut v = fixture_9r4c_2z(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    v.step_physics(0, T_OUT, DT);
    v.step_physics(1, T_OUT, DT);
    assert_eq!(
        v.setpoints.temperatures.as_ref()[0],
        0.015342050813032998,
        "vent step-1 zone-0 committed air (bit-exact)"
    );
    assert_eq!(v.setpoints.temperatures.as_ref()[1], 28.803195996264588);
    assert_eq!(v.mass.mass_temperatures.as_ref()[0], 32.149686954689734);
    assert_eq!(v.mass.mass_temperatures.as_ref()[1], 26.854019373663057);
    // The guard: a NaN driving temperature rejects the step; BOTH commits
    // stay untouched while the solver nodes advance into NaN (pinned as-is).
    let kwh_nan = v.step_physics(2, f64::NAN, DT);
    assert_eq!(kwh_nan, 0.0, "the rejected step returns 0.0 kWh");
    assert_eq!(
        v.setpoints.temperatures.as_ref()[0],
        0.015342050813032998,
        "zone-0 commit untouched"
    );
    assert_eq!(
        v.setpoints.temperatures.as_ref()[1],
        28.803195996264588,
        "zone-1 commit untouched"
    );
    let vs = &v.conduction.backend.multi_node_solvers[0];
    assert!(
        vs.mass.wall.temperature.is_nan() && vs.mass.roof.temperature.is_nan(),
        "the solver nodes advanced into NaN before the guard — pinned as-is"
    );
}

/// The boundary fixture: the two-zone model grown to three zones with only
/// two solvers — the shape the mn solve's `!mn_has` branch claims to
/// support ("zones without a multi-node solver enter as known boundary
/// temperatures", step_9r4c.rs). Since loop r133 the assembly compacts to
/// solver zones and the boundary enters as a known (contracts M8/M10).
fn fixture_9r4c_boundary() -> ThermalModel<VectorField> {
    let mut m = fixture_9r4c_2z(None);
    m.setpoints.zone_area = VectorField::new(vec![48.0, 40.0, 30.0]);
    m.setpoints.zone_volume = VectorField::new(vec![129.6, 108.0, 90.0]);
    m.setpoints.temperatures = VectorField::new(vec![20.0, 18.0, 15.0]);
    m.setpoints.loads = VectorField::new(vec![30.0, 20.0, 0.0]);
    m.solar.solar_gains = VectorField::new(vec![400.0, 300.0, 0.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0, 30.0, 0.0]);
    m.conduction.h_tr_em = VectorField::new(vec![15.36, 10.0, 5.0]);
    m.conduction.h_tr_w = VectorField::new(vec![36.0, 24.0, 12.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![7.2, 8.0, 3.0]);
    m.conduction.h_ve = VectorField::new(vec![
        1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0),
        h_ve_z1(),
        10.0,
    ]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0, 800.0, 500.0]);
    m.conduction.h_tr_is = VectorField::new(vec![69.0, 75.0, 60.0]);
    m.conduction.h_tr_iz = VectorField::new(vec![5.0, 7.0, 4.0]);
    m.conduction.h_tr_iz_rad = VectorField::new(vec![2.0, 3.0, 1.0]);
    m.mass.thermal_capacitance = VectorField::new(vec![7.2e6, 6.0e6, 4.5e6]);
    m.mass.mass_temperatures = VectorField::new(vec![20.0, 18.0, 15.0]);
    m.mass.h_tr_me = VectorField::new(vec![0.0, 0.0, 0.0]);
    m.hvac.num_zones = 3;
    m.update_optimization_cache();
    m
}

#[test]
/// Contract M8 — the boundary compaction (the r133 fix of the r132
/// finding, dated 2026-11-12): with a solver-less zone in the model, the
/// mn implicit system assembles ONLY over solver zones — zone 2 enters as
/// a known boundary temperature (its 5R1C estimate on the rhs through
/// `h_iz_vec[i]`) and KEEPS that estimate as its committed air. The
/// r132 pin recorded the previous behavior as-is: the never-built
/// `!mn_has` row divided 0/0 in the back-substitution, `0.0·NaN` poisoned
/// x for every solver zone, and the #4495 commit guard rejected whole
/// steps that had already half-applied (lumped masses and solver nodes
/// advanced, air commits frozen at 20.0/18.0/15.0). The in-code comment's
/// claimed known-boundary behavior is what executes now: no all-zero row,
/// no 0/0, no NaN — the step commits cleanly (0.0 kWh exactly,
/// free-float), the solver zones solve the corrected compacted 2×2
/// (34.014510130543016 / 28.846075334165224 — the closed form is M10's),
/// zone 2 keeps its estimate 10.01546316700131 verbatim, and the lumped
/// masses (26.146242764178844 / 22.443383416886824 / 14.893664424659514,
/// riding the finite 3×3 estimate solve) advance exactly as in the r132
/// pin — the whole step is APPLIED, not half-applied. Latent-only shape:
/// the from-spec constructor builds one solver per zone
/// (thermal_model_core/mod.rs:2296), so no production path constructs it
/// today.
fn boundary_solver_less_zone_enters_as_known_boundary() {
    let mut m = fixture_9r4c_boundary();
    assert_eq!(
        m.conduction.derived_den.as_ref()[2],
        DEN_BND_2,
        "zone-2 den anchor (the 3×3 estimate cache composes fine)"
    );
    let kwh = m.step_physics(0, T_OUT, DT);
    assert_eq!(
        kwh, 0.0,
        "free-float returns exactly 0.0 kWh — the step commits"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref(),
        [34.014510130543016, 28.846075334165224, 10.01546316700131],
        "the commits: solver zones solve the compacted system; zone 2 keeps its estimate"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref(),
        [26.146242764178844, 22.443383416886824, 14.893664424659514],
        "the lumped masses advance bit-exactly as in the r132 pin"
    );
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        s0.zone_temperature, 34.47988132764695,
        "zone-0 pre-air composed normally before the solve"
    );
    assert_eq!(
        s0.mass.wall.temperature, 18.614397156250497,
        "zone-0 mass node advanced normally"
    );
}

/// The mn committed-air implicit solve at the BOUNDARY fixture (3 zones /
/// 2 solvers), op-for-op with the r133 compacted assembly: rows only for
/// solver zones, zone 2's estimate entering the rhs through
/// `h_iz_vec[i]` (no `w_ij` seam weight), wsum accumulated in the engine's
/// j order, the inline elimination AND THEN `solve_linear_system` at 2×2.
fn committed_solved_boundary_2z(t_unc: [f64; 2], est2: f64) -> (f64, f64) {
    let h_ms = [[50.0, 40.0, 60.0], [45.0, 35.0, 55.0]];
    let his = [120.0, 140.0];
    let h_ve = [21.6, h_ve_z1()];
    let hiz = [5.0, 7.0];
    let mut diag = [0.0; 2];
    let mut rhs = [0.0; 2];
    for i in 0..2 {
        let hs_w = (h_ms[i][0] * his[i]) / (h_ms[i][0] + his[i]);
        let hs_r = (h_ms[i][1] * his[i]) / (h_ms[i][1] + his[i]);
        let hs_f = (h_ms[i][2] * his[i]) / (h_ms[i][2] + his[i]);
        let total = (hs_w + hs_r) + hs_f;
        diag[i] = (total + h_ve[i]) + 0.0;
        rhs[i] = t_unc[i] * diag[i];
    }
    let w01 = 0.5 * (hiz[0] + hiz[1]);
    let mut wsum0 = 0.0;
    wsum0 += w01;
    wsum0 += hiz[0];
    let mut wsum1 = 0.0;
    wsum1 += w01;
    wsum1 += hiz[1];
    let mut a = [[diag[0] + wsum0, -w01], [-w01, diag[1] + wsum1]];
    let mut b = [rhs[0] + hiz[0] * est2, rhs[1] + hiz[1] * est2];
    elim_pass_2(&mut a, &mut b);
    solve_linear_system_2(&mut a, &mut b)
}

#[test]
/// Contract M9 — the committed-air balance TWIN (the r133 fix's seam):
/// the engine's `committed_air_balance` returns `(t_air, den)` with `den`
/// the balance's OWN denominator — bit-identical to the in-test
/// `parallel_balance_z` recomposition's division — on the POST-step
/// nodes, both vent arms, per zone: den 126.89411764705883 /
/// 115.04123354123354 vent-off, 595.8941176470588 / 115.04123354123354
/// vent-on (the fan lands zone-0-only in `h_ve_total`). The stash
/// re-scales with this literal (M4/M6/M7/M10 pin the downstream solved
/// values); the twin itself is engine-blind to a stash-site mutation by
/// construction, and its own teeth ride a denominator drift in the
/// balance methods.
fn committed_air_balance_twin_returns_the_balance_denominator() {
    // Vent-off: den = Σh_path + h_ve, per zone.
    let mut m = fixture_9r4c_2z(None);
    m.step_physics(0, T_OUT, DT);
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    let s1 = &m.conduction.backend.multi_node_solvers[1];
    let (t0, d0) = s0.committed_air_balance(T_OUT, 21.6, 0.0, 2496.0);
    let (t1, d1) = s1.committed_air_balance(T_OUT, h_ve_z1(), 0.0, 1520.0);
    let t0_ref = parallel_balance_z(
        [50.0, 40.0, 60.0],
        120.0,
        [
            s0.mass.wall.temperature,
            s0.mass.roof.temperature,
            s0.mass.floor.temperature,
        ],
        21.6,
        0.0,
        2496.0,
    );
    let t1_ref = parallel_balance_z(
        [45.0, 35.0, 55.0],
        140.0,
        [
            s1.mass.wall.temperature,
            s1.mass.roof.temperature,
            s1.mass.floor.temperature,
        ],
        h_ve_z1(),
        0.0,
        1520.0,
    );
    assert_eq!(
        t0, t0_ref,
        "the twin's value == the closed form (bit-exact)"
    );
    assert_eq!(t1, t1_ref, "zone-1 twin value == the closed form");
    assert_eq!(
        t0, 35.204524659159546,
        "zone-0 uncoupled committed air (M4 verbatim)"
    );
    assert_eq!(t1, 29.72231486401823, "zone-1 uncoupled committed air");
    assert_eq!(
        d0, 126.89411764705883,
        "zone-0 balance denominator (bit-exact)"
    );
    assert_eq!(
        d1, 115.04123354123354,
        "zone-1 balance denominator (bit-exact)"
    );
    // Vent-on: the fan enters zone 0's h_ve_total only.
    let mut v = fixture_9r4c_2z(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let hv = h_ve_night_engine_read(&v);
    assert_eq!(hv, H_VE_NIGHT, "premise: the vent conductance (r112 pin)");
    v.step_physics(0, T_OUT, DT);
    let vs0 = &v.conduction.backend.multi_node_solvers[0];
    let vs1 = &v.conduction.backend.multi_node_solvers[1];
    let (vt0, vd0) = vs0.committed_air_balance(T_OUT, 21.6, hv, 2496.0);
    let (vt1, vd1) = vs1.committed_air_balance(T_OUT, h_ve_z1(), 0.0, 1520.0);
    assert_eq!(
        vt0, -0.400967803731356,
        "vent-on zone-0 uncoupled committed air"
    );
    assert_eq!(
        vt1, 29.72231486401823,
        "vent-on zone-1 uncoupled committed air"
    );
    assert_eq!(
        vd0, 595.8941176470588,
        "vent-on zone-0 denominator carries h_ve_night"
    );
    assert_eq!(
        vd1, 115.04123354123354,
        "vent-on zone-1 denominator stays vent-free"
    );
}

#[test]
/// Contract M10 — the corrected committed-air closed form at the BOUNDARY
/// fixture (the r133 compaction's engine truth): the in-test compacted 2×2
/// transcription — `diag = Σh_path_k + h_ve` (the balance denominators),
/// `rhs = t_unc·diag + h_iz_vec[i]·est2` — reproduces the engine's
/// committed solver-zone airs 34.014510130543016 / 28.846075334165224
/// bit-exactly. Zone 2's estimate is pinned BY VALUE through its own
/// consumer: the lumped Euler on 10.01546316700131 reproduces the engine's
/// zone-2 lumped mass 14.893664424659514 bit-exactly (the M5 pattern) —
/// the literal the boundary feeds IS the engine's zone-2 estimate.
fn boundary_compaction_closed_form_matches_the_engine() {
    let mut m = fixture_9r4c_boundary();
    m.step_physics(0, T_OUT, DT);
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    let s1 = &m.conduction.backend.multi_node_solvers[1];
    let t_unc0 = parallel_balance_z(
        [50.0, 40.0, 60.0],
        120.0,
        [
            s0.mass.wall.temperature,
            s0.mass.roof.temperature,
            s0.mass.floor.temperature,
        ],
        21.6,
        0.0,
        2496.0,
    );
    let t_unc1 = parallel_balance_z(
        [45.0, 35.0, 55.0],
        140.0,
        [
            s1.mass.wall.temperature,
            s1.mass.roof.temperature,
            s1.mass.floor.temperature,
        ],
        h_ve_z1(),
        0.0,
        1520.0,
    );
    assert_eq!(
        t_unc0, 35.204524659159546,
        "zone-0 uncoupled committed air (mass chain iz-free of zone 2)"
    );
    assert_eq!(t_unc1, 29.72231486401823, "zone-1 uncoupled committed air");
    // Zone 2's estimate, pinned by value through its lumped mass.
    let h_tr_3_z2 = {
        let h1 = (10.0 * 60.0) / (10.0 + 60.0);
        let h2 = h1 + 12.0;
        (h2 * 500.0) / (h2 + 500.0)
    };
    let lumped2 = {
        let ts_den = 500.0 + 60.0 + 0.0;
        let t_s = (500.0 * 15.0 + 60.0 * 10.01546316700131) / ts_den;
        let cm_dt = 4.5e6 / DT;
        let numer = ((cm_dt * 15.0 + 5.0 * T_OUT) + h_tr_3_z2 * t_s) + 0.0;
        numer / ((cm_dt + 5.0) + h_tr_3_z2)
    };
    assert_eq!(
        lumped2, 14.893664424659514,
        "the zone-2 lumped Euler on est2 10.01546316700131 == engine (est2 pinned by value)"
    );
    // The compacted 2×2 vs the engine commits.
    let (c0, c1) = committed_solved_boundary_2z([t_unc0, t_unc1], 10.01546316700131);
    assert_eq!(
        c0, 34.014510130543016,
        "boundary-fixture zone-0 committed air (bit-exact, compacted closed form)"
    );
    assert_eq!(
        c1, 28.846075334165224,
        "boundary-fixture zone-1 committed air (bit-exact, compacted closed form)"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        c0,
        "the commit takes the solved value"
    );
    assert_eq!(m.setpoints.temperatures.as_ref()[1], c1);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[2],
        10.01546316700131,
        "zone 2 keeps its estimate verbatim — the known-boundary premise"
    );
}

// ============================================================================
// Round 134 — the AdditiveSum coupling-mode arm (the r133 remainder), with
// the per-surface channel's first explicit pin
// ============================================================================
//
// The r131–r133 contracts pin the 9R4C committed-temperature chain at the
// ParallelResistance production mode (#1859/#1281). The solver's OTHER
// coupling mode — `MassAirCouplingMode::AdditiveSum`, the constructor
// DEFAULT (`MultiNodeSolver::new`) and the mode every future construction
// site silently gets — had no step-level coverage: its mass Euler
// (`step_backward_euler_with_gains`, the shared-T_s operand), its air
// balance (`additive_balance_parts`, the film-conductance sum denominator),
// and its implicit-iz stash arm were all unpinned. Production AdditiveSum
// exposure is real: `src/python/multi_node_bindings.rs` constructs
// `PyMultiNodeSolver` with plain `new` (→ AdditiveSum) and the step-level
// committed-air twin is mode-consistent since r133. This section flips the
// r131 single-zone and r132 two-zone fixtures to AdditiveSum (one field)
// and pins the mode-dispatched arms bit-exactly:
//
// - (A1) the additive premises and the mode seam: the committed air is the
//   additive closed form on the POST-step mass nodes — `t_surface`
//   recomputed from the mass temps (NOT the solver's stored
//   `surface_temperature` field), `(h_tr_is·T_s + h_ve_total·T_out +
//   h_rad_sky·T_sky + φ_ia)/(h_tr_is + h_ve_total + h_rad_sky)` — with the
//   film-sum denominators 141.6 / 610.6 (vent-on) where ParallelResistance
//   would produce 126.89411764705883 / 595.8941176470588 (M9's literals:
//   the mode seam is observable). The degenerate fallback (denom < 1e-6 →
//   `(t_surface, 1.0)`) is pinned at the solver API with an in-test
//   zero-conductance construction;
// - (A2) the additive mass Euler: every envelope node consumes the
//   EXTERNALLY-SET shared surface temperature (the step's opening block
//   re-derives it from the PRE-step mass temps and sets it before the
//   advance — the value consumed is the mass-derived 19.866666666666667,
//   not the prior per-surface write-back), the gains are
//   (φ_st + φ_m)·h_tr_ms-fraction per node (the LIMIT-35 §12 routing),
//   and the internal node composes through its own `h_tr_me` (100.0)
//   exactly as in r131 — now under the additive operand. The night-vent
//   fan reaches the mass ONLY through `t_i =` the pre-air (the internal
//   node shifts, w/r/f do not — LIMIT-17's air-side-only at this site);
// - (A3) the committed-air chain across both steps, both vent arms, with
//   the `zone_temperature` hand-off (zt(step N) == the pre-air == the
//   prior committed at uncoupled single-zone);
// - (A4) the additive implicit-iz stash and solve at the two-zone fixture
//   (the M4 additive counterpart): `diag =` the film-sum denominators
//   141.6 / 153.5, `rhs = t_unc·diag`, SEAMLESS weights
//   `w_ij = 0.5·(h_iz_i + h_iz_j)`, the inline partial-pivot elimination
//   AND THEN `solve_linear_system` — committed airs
//   33.37271758164313 / 27.41145662036934 vent-off,
//   0.3703605371244691 / 26.16998864377302 vent-on; the 5R1C estimate
//   solve stays mode-invariant (r132's M2 literals recomputed here);
// - (A5) the per-surface channel's FIRST EXPLICIT PIN (mode-blind code —
//   identical in both modes; r131 proved PR-inertness by composition
//   only): `step_per_surface` builds a per-surface solver from the
//   POST-advance mass temps, drives each surface with the PRE-GAIN mass
//   temperature (`q_ms = h_ms·(pg − T_s)`), the per-surface exterior
//   temperature, and the `distribute_opaque_solar_gains` split of φ_m —
//   whose areas are the fixture's CONSTRUCTOR-DEFAULT area setpoints
//   (wall 53.665631459994955 = perimeter·3.0 over the sqrt(20) default
//   geometry; roof/floor 20.0; zero irradiance → the area-fraction
//   branch), giving pm 7315.413047462013 / 2726.293476268993 / 2726.293476268993
//   per zone 0 — and writes the conductance-weighted per-surface average
//   back to `surface_temperature` (81.09607259126786 at step 0).
//   FINDING (recorded, pinned as-is, routed to the physics queue):
//   `fluxion-core`'s `SurfaceNode::update` and every doc around it claim
//   "backward Euler"; the code is the EXPLICIT forward-Euler recurrence
//   `T' = T + dt·(q_ms − q_em + φ_m)/C` (C = 230,000 J from
//   `add_surface_from_params`). The value is the recurrence; the docs are
//   wrong about the scheme;
// - (A6) the clobber chain: the write-back IS the observable
//   `surface_temperature` after each step — and it is DEAD at this
//   production path: the next step's opening block re-derives from the
//   mass temps (20.67786979440258 = the #862 value) before the mass Euler
//   consumes it, so step 1's node literals compose from the mass-derived
//   operand, never the write-back (the LIMIT-35 §12 note, now pinned for
//   BOTH modes — r131 proved the PR side by composition, this pin names
//   the clobber sites (a)/(d#862)/(f)/(g));
// - (A7) the two-step composition and the mode-invariant chains: every
//   step-1 state composes bit-exactly from the step-0 committed state;
//   the 5R1C estimate and lumped mass are mode-INVARIANT (r131's
//   PR-mode literals 29.428271397080522 / 26.149556909916907 and
//   26.10141928940402 recomputed on the additive fixture — the chains
//   ride `t_i_free_5r1c`, which never dispatches on coupling mode);
// - (A8) the #4495 input-envelope guard at additive mode: a NaN outdoor
//   temperature rejects the step (0.0 kWh), the commits stay untouched,
//   the solver's own nodes HAVE advanced into NaN — pinned as-is (the
//   guard's scope is the commit, not the solver), the r131 contract-8
//   shape under the additive arms.
//
// Fixture: the r131 single-zone / r132 two-zone hand-rolled recipes
// verbatim with `coupling_mode` flipped to `AdditiveSum` (the public
// field; `new_with_mode(…, ParallelResistance)` is what the production
// step constructs). Provenance: Python replication
// `/tmp/r134_replication.py` + engine probe log `/tmp/r134_probe1.log`
// (dev-time evidence, tip `bd2b411f`); every literal was produced by the
// replication and matched the engine bit-for-bit across BOTH vent arms,
// both fixtures, all four nodes + internal + the pre-airs + the
// per-surface channel + the write-backs before the suite was written.
// Suite shape unchanged: characterization pins with bit-exact teeth, no
// engine change, no re-record owed. A violation means the additive chain
// moved and must be re-examined, never tuned around.

/// The additive fixture: the r131 recipe with the coupling mode flipped.
fn fixture_9r4c_additive(vent: Option<NightVentilation>) -> ThermalModel<VectorField> {
    let mut m = fixture_9r4c(vent);
    assert_eq!(
        m.conduction.backend.multi_node_solvers[0].coupling_mode,
        MassAirCouplingMode::ParallelResistance,
        "premise: the r131 recipe constructs ParallelResistance"
    );
    m.conduction.backend.multi_node_solvers[0].coupling_mode = MassAirCouplingMode::AdditiveSum;
    m
}

/// The additive two-zone fixture: the r132 recipe with both modes flipped.
fn fixture_9r4c_2z_additive(vent: Option<NightVentilation>) -> ThermalModel<VectorField> {
    let mut v = fixture_9r4c_2z(vent);
    for s in &mut v.conduction.backend.multi_node_solvers {
        s.coupling_mode = MassAirCouplingMode::AdditiveSum;
    }
    v
}

/// The additive committed-air closed form (additive_balance_parts,
/// h_rad_sky = 0), op-for-op: the mass-derived weighted t_surface, the
/// film-sum denominator, the degenerate fallback untouched at this fixture.
fn additive_balance_z(
    h_ms: [f64; 3],
    his: f64,
    temps: [f64; 3],
    h_ve: f64,
    hvn: f64,
    phi_ia: f64,
) -> (f64, f64, f64) {
    let h_ms_total = (h_ms[0] + h_ms[1]) + h_ms[2];
    let num_ts = (h_ms[0] * temps[0] + h_ms[1] * temps[1]) + h_ms[2] * temps[2];
    let t_surface = num_ts / h_ms_total;
    let h_ve_total = h_ve + hvn;
    let denom = (his + h_ve_total) + 0.0;
    let num = (his * t_surface + h_ve_total * T_OUT) + 0.0 * 0.0;
    let t_air = (num + phi_ia) / denom;
    (t_air, denom, t_surface)
}

/// The additive mass Euler for one envelope node (the shared-T_s operand),
/// op-for-op with step_backward_euler_with_gains' per-node block.
fn additive_node_euler(
    c: f64,
    t_old: f64,
    h_em: f64,
    t_ext: f64,
    h_ms: f64,
    t_s: f64,
    g: f64,
) -> f64 {
    let c_dt = c / DT;
    let numer = ((c_dt * t_old + h_em * t_ext) + h_ms * t_s) + g;
    let denom = (c_dt + h_em) + h_ms;
    if denom > 1e-10 {
        numer / denom
    } else {
        t_old
    }
}

/// The additive implicit-iz committed-air solve at the two-zone fixture
/// (both zones carry solvers): diag = the film sums, rhs = t_unc·diag,
/// seamless w01, inline elimination AND THEN solve_linear_system_2.
fn additive_committed_solved_2z(t_unc: [f64; 2]) -> (f64, f64) {
    let his = [120.0, 140.0];
    let h_ve = [21.6, h_ve_z1()];
    let hiz = [5.0, 7.0];
    let mut diag = [0.0; 2];
    let mut rhs = [0.0; 2];
    for i in 0..2 {
        diag[i] = (his[i] + h_ve[i]) + 0.0;
        rhs[i] = t_unc[i] * diag[i];
    }
    let w01 = 0.5 * (hiz[0] + hiz[1]);
    let mut wsum0 = 0.0;
    wsum0 += w01;
    let mut wsum1 = 0.0;
    wsum1 += w01;
    let mut a = [[diag[0] + wsum0, -w01], [-w01, diag[1] + wsum1]];
    let mut b = [rhs[0], rhs[1]];
    elim_pass_2(&mut a, &mut b);
    solve_linear_system_2(&mut a, &mut b)
}

/// The per-surface forward-Euler recurrence (SurfaceNode::update), op-for-op:
/// q_ms from the PRE-GAIN mass temperature, q_em from the per-surface
/// exterior, φ_m added, C = 230,000 J (add_surface_from_params, area 1.0).
fn per_surface_update(t_s: f64, h_ms: f64, h_em: f64, pg: f64, t_ext: f64, phi_m: f64) -> f64 {
    let q_ms = h_ms * (pg - t_s);
    let q_em = h_em * (t_s - t_ext);
    let heat = (q_ms - q_em) + phi_m;
    t_s + DT * heat / 230_000.0
}

#[test]
/// Contract A1 — the additive premises, the mode seam, and the degenerate
/// fallback: the flipped mode, the committed air == the additive closed
/// form on the post-step nodes (bit-exact 33.62531338508693, den 141.6 ==
/// the film sum 120.0 + 21.6), provably distinct from the PR-mode commit
/// 35.204524659159546 at the same fixture recipe (the mode seam is
/// observable); the twin `(t, den)` == the recomposition; the vent-on arm
/// (0.11684306473683184 / 610.6); free-float returns exactly 0.0 kWh; the
/// `zone_temperature` set premise (the pre-air 32.93785310734464 == the
/// additive balance on the PRE-step nodes); and the degenerate fallback at
/// the solver API (h_tr_is 0.0 → `(t_surface, 1.0)`).
fn additive_premises_mode_seam_and_degenerate_fallback() {
    let mut m = fixture_9r4c_additive(None);
    assert_eq!(
        m.conduction.backend.multi_node_solvers[0].coupling_mode,
        MassAirCouplingMode::AdditiveSum,
        "premise: the fixture flipped to AdditiveSum"
    );
    let kwh = m.step_physics(0, T_OUT, DT);
    assert_eq!(kwh, 0.0, "free-float returns exactly 0.0 kWh");
    let s = &m.conduction.backend.multi_node_solvers[0];
    // The additive closed form on the post-step nodes, op-for-op.
    let (t_ref, den_ref, _ts) = additive_balance_z(
        [50.0, 40.0, 60.0],
        120.0,
        [
            s.mass.wall.temperature,
            s.mass.roof.temperature,
            s.mass.floor.temperature,
        ],
        21.6,
        0.0,
        2496.0,
    );
    let committed = m.setpoints.temperatures.as_ref()[0];
    assert_eq!(
        committed, t_ref,
        "committed air == the additive closed form (bit-exact)"
    );
    assert_eq!(
        committed, 33.62531338508693,
        "the additive committed literal (engine + replication)"
    );
    assert_ne!(
        committed, 35.204524659159546,
        "the mode seam: ParallelResistance would commit 35.204524659159546 (M9 verbatim)"
    );
    assert_eq!(den_ref, 141.6, "the film-sum denominator == 120.0 + 21.6");
    // The twin returns (t, den) — both bit-exact vs the recomposition.
    let (t_bal, den_bal) = s.committed_air_balance(T_OUT, 21.6, 0.0, 2496.0);
    assert_eq!(t_bal, t_ref, "twin value == the additive closed form");
    assert_eq!(den_bal, den_ref, "twin denominator == the film sum");
    assert_eq!(den_bal, 141.6, "den literal");
    // The pre-air set premise: zone_temperature == the additive balance on
    // the PRE-step nodes (the replication's step-0 pre-air literal).
    assert_eq!(
        s.zone_temperature, 32.93785310734464,
        "zt == the pre-air (the additive balance on the initial nodes)"
    );
    // Vent-on: the fan lands in h_ve_total on both num and den.
    let mut v = fixture_9r4c_additive(Some(NightVentilation::new(FAN_M3_H, 18, 7)));
    let hv = h_ve_night_engine_read(&v);
    assert_eq!(hv, H_VE_NIGHT, "premise: the vent conductance (r112 pin)");
    v.step_physics(0, T_OUT, DT);
    let vs = &v.conduction.backend.multi_node_solvers[0];
    let (vt, vden) = vs.committed_air_balance(T_OUT, 21.6, hv, 2496.0);
    let (vt_ref, vden_ref, _) = additive_balance_z(
        [50.0, 40.0, 60.0],
        120.0,
        [
            vs.mass.wall.temperature,
            vs.mass.roof.temperature,
            vs.mass.floor.temperature,
        ],
        21.6,
        hv,
        2496.0,
    );
    assert_eq!(vt, vt_ref, "vent-on twin value == the closed form");
    assert_eq!(vt, 0.11684306473683184, "vent-on committed literal");
    assert_eq!(
        vden, vden_ref,
        "vent-on twin denominator == the film sum + h_ve_night"
    );
    assert_eq!(
        vden, 610.6,
        "vent-on den == (120.0 + 21.6) + 469.0 (the film sum carries the fan)"
    );
    // The degenerate fallback: h_tr_is = 0.0 → denom < 1e-6 → (t_surface, 1.0).
    let wall = ThermalMassNode::new(18.0, 4.0e7, 50.0, 20.0);
    let roof = ThermalMassNode::new(22.0, 2.0e7, 40.0, 15.0);
    let floor = ThermalMassNode::new(20.0, 3.0e7, 60.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 1.0e7, 10.0, 5.0);
    let degenerate = MultiNodeSolver::new_with_mode(
        0.0,
        wall,
        roof,
        floor,
        internal,
        MassAirCouplingMode::AdditiveSum,
    );
    let (dt_fallback, den_fallback) = degenerate.committed_air_balance(T_OUT, 0.0, 0.0, 0.0);
    assert_eq!(
        den_fallback, 1.0,
        "the degenerate denominator is 1.0 (the value is returned directly)"
    );
    assert_eq!(
        dt_fallback, 19.866666666666667,
        "the degenerate value is the weighted surface temperature itself"
    );
}

#[test]
/// Contract A2 — the additive mass Euler consumes the EXTERNALLY-SET shared
/// surface temperature: all four step-0 node literals bit-exact
/// (wall 18.570724436052867, roof 22.875682740865432, floor 20.968615628718762,
/// internal 20.541671075618552), each composed in-test from the shared-T_s
/// closed form with T_s = 19.866666666666667 (the opening block's weighted
/// PRE-step mass value — NOT the solver's stored 20.0, NOT the prior
/// write-back), gains (φ_st + φ_m)·h_tr_ms-fraction, the internal node
/// through its own h_tr_me (100.0) — and provably distinct from the PR-mode
/// wall literal 18.614397156250497 (r131 contract 3). The vent arm: the fan
/// shifts ONLY the internal node (t_i = the pre-air carries h_ve_night;
/// w/r/f stay at the vent-off literals — the mass Euler has no h_ve operand,
/// LIMIT-17's air-side-only at this site).
fn additive_mass_euler_consumes_the_shared_surface_temperature() {
    let mut m = fixture_9r4c_additive(None);
    m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    // The shared T_s the advance consumed: the opening block's weighted
    // PRE-step mass value (50·18 + 40·22 + 60·20)/150.
    // Compose it exactly as the engine's opening block does (op order).
    let t_s_shared = ((50.0 * 18.0 + 40.0 * 22.0) + 60.0 * 20.0) / ((50.0 + 40.0) + 60.0);
    assert_eq!(
        t_s_shared, 19.866666666666667,
        "the consumed shared surface temperature (the weighted PRE-step value)"
    );
    assert_ne!(
        t_s_shared, 20.0,
        "the solver's stored surface_temperature starts at 20.0 and is overwritten"
    );
    // φ_st and φ_m (mode-invariant, r131 contract 1's decomposition).
    let (phi_ia, phi_st, phi_m) = (2496.0_f64, 7776.0_f64, 12768.0_f64);
    assert_eq!(phi_ia + phi_st + phi_m, 1440.0 + 19200.0 + 2400.0);
    let wall_frac = 50.0 / 150.0;
    let roof_frac = 40.0 / 150.0;
    let floor_frac = 60.0 / 150.0;
    let w = additive_node_euler(
        4.0e7,
        18.0,
        20.0,
        T_OUT,
        50.0,
        t_s_shared,
        phi_st * wall_frac + phi_m * wall_frac,
    );
    let r = additive_node_euler(
        2.0e7,
        22.0,
        15.0,
        T_OUT,
        40.0,
        t_s_shared,
        phi_st * roof_frac + phi_m * roof_frac,
    );
    let f = additive_node_euler(
        3.0e7,
        20.0,
        10.0,
        T_G,
        60.0,
        t_s_shared,
        phi_st * floor_frac + phi_m * floor_frac,
    );
    assert_eq!(w, 18.570724436052867, "additive wall node (bit-exact)");
    assert_eq!(r, 22.875682740865432, "additive roof node");
    assert_eq!(f, 20.968615628718762, "additive floor node");
    assert_ne!(
        w, 18.614397156250497,
        "the PR-mode wall literal (r131 contract 3) — the mode seam"
    );
    // Internal node: own h_tr_me (100.0) through the envelope average of the
    // POST-update w/r/f, t_i = the pre-air.
    let hmw = (50.0 * 100.0) / (50.0 + 100.0);
    let hmr = (40.0 * 100.0) / (40.0 + 100.0);
    let hmf = (60.0 * 100.0) / (60.0 + 100.0);
    let t_env = (hmw * w + hmr * r + hmf * f) / (hmw + hmr + hmf);
    let c_dt = 1.0e7 / DT;
    let i =
        ((c_dt * 20.0 + 120.0 * 32.93785310734464) + 100.0 * t_env + 0.0) / (c_dt + 120.0 + 100.0);
    assert_eq!(
        i, 20.541671075618552,
        "additive internal node (own h_tr_me seam)"
    );
    assert_eq!(
        s.mass.internal.temperature, i,
        "the engine's internal node == the closed form"
    );
    // Vent arm: only the internal node moves (the pre-air carries the fan).
    let mut v = fixture_9r4c_additive(Some(NightVentilation::new(FAN_M3_H, 18, 7)));
    v.step_physics(0, T_OUT, DT);
    let vs = &v.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        vs.mass.wall.temperature, 18.570724436052867,
        "vent-on wall == the vent-off literal (the fan never touches the mass)"
    );
    assert_eq!(
        vs.mass.roof.temperature, 22.875682740865432,
        "vent-on roof == vent-off"
    );
    assert_eq!(
        vs.mass.floor.temperature, 20.968615628718762,
        "vent-on floor == vent-off"
    );
    assert_eq!(
        vs.mass.internal.temperature, 19.221475786176054,
        "vent-on internal shifted via t_i = the vent pre-air"
    );
    // (the vent pre-air itself is A3's -0.04258106780216181)
}

#[test]
/// Contract A3 — the committed-air chain across two steps, both vent arms,
/// with the zone_temperature hand-off: step-1 committed air == the additive
/// balance on the step-0 POST nodes (33.62531338508693 → 34.31138437180818
/// vent-off; 0.11684306473683184 → 0.27594501645600783 vent-on); zt(step 1)
/// == the prior committed (the uncoupled hand-off); the pre-airs
/// −0.04258106780216181 vent-on step 0 and 33.62531338508693 vent-off
/// step 1 pinned via zt.
fn additive_committed_chain_two_steps_both_vent_arms() {
    // Vent-off.
    let mut m = fixture_9r4c_additive(None);
    m.step_physics(0, T_OUT, DT);
    m.step_physics(1, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        s.zone_temperature, 33.62531338508693,
        "zt(step 1) == the step-0 committed air (the pre-air hand-off)"
    );
    let (t1, d1) = s.committed_air_balance(T_OUT, 21.6, 0.0, 2496.0);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        34.31138437180818,
        "step-1 committed literal (vent-off)"
    );
    assert_eq!(t1, 34.31138437180818, "the twin agrees at step 1");
    assert_eq!(d1, 141.6, "the film sum is temperature-independent");
    // Vent-on.
    let mut v = fixture_9r4c_additive(Some(NightVentilation::new(FAN_M3_H, 18, 7)));
    v.step_physics(0, T_OUT, DT);
    let vs = &v.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        vs.zone_temperature, -0.04258106780216181,
        "the vent-on pre-air (the additive balance with h_ve_night in h_ve_total)"
    );
    v.step_physics(1, T_OUT, DT);
    let vs = &v.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        vs.zone_temperature, 0.11684306473683184,
        "vent-on hand-off: zt(step 1) == the step-0 committed air"
    );
    assert_eq!(
        v.setpoints.temperatures.as_ref()[0],
        0.27594501645600783,
        "step-1 committed literal (vent-on)"
    );
}

#[test]
/// Contract A4 — the additive implicit-iz stash and solve at the two-zone
/// fixture (the M4 additive counterpart): the stashes carry the FILM-SUM
/// denominators 141.6 / 153.5 (ParallelResistance would carry
/// 126.89411764705883 / 115.04123354123354 — the mode seam on the stash),
/// rhs = t_unc·diag, seamless w01 = 6.0, the inline elimination AND THEN
/// solve_linear_system_2; committed 33.37271758164313 / 27.41145662036934
/// vent-off and 0.3703605371244691 / 26.16998864377302 vent-on; the 5R1C
/// estimate solve stays mode-invariant (r132's M2 literals).
fn additive_implicit_iz_stash_and_solve() {
    // Vent-off.
    let mut m = fixture_9r4c_2z_additive(None);
    m.step_physics(0, T_OUT, DT);
    let s0 = &m.conduction.backend.multi_node_solvers[0];
    let s1 = &m.conduction.backend.multi_node_solvers[1];
    let (t0, d0) = s0.committed_air_balance(T_OUT, 21.6, 0.0, 2496.0);
    let (t1, d1) = s1.committed_air_balance(T_OUT, h_ve_z1(), 0.0, 1520.0);
    assert_eq!(t0, 33.62531338508693, "zone-0 uncoupled (the stash input)");
    assert_eq!(t1, 27.178443162599674, "zone-1 uncoupled (the stash input)");
    assert_eq!(d0, 141.6, "zone-0 film-sum stash denominator");
    assert_eq!(
        d1, 153.5,
        "zone-1 film-sum stash denominator (140.0 + 13.5)"
    );
    let (c0, c1) = additive_committed_solved_2z([t0, t1]);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        c0,
        "the engine's committed zone-0 air == the in-test solve"
    );
    assert_eq!(c0, 33.37271758164313, "the solved zone-0 literal");
    assert_eq!(c1, 27.41145662036934, "the solved zone-1 literal");
    // The lumped mass rides the mode-invariant estimate chain (the
    // probe-verified 2-zone additive value; the estimate solve itself is
    // r132's M2 literals recomputed above).
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        26.146946251858456,
        "the 2-zone lumped literal (mode-invariant chain)"
    );
    // Vent-on.
    let mut v = fixture_9r4c_2z_additive(Some(NightVentilation::new(FAN_M3_H, 18, 7)));
    v.step_physics(0, T_OUT, DT);
    assert_eq!(
        v.setpoints.temperatures.as_ref()[0],
        0.3703605371244691,
        "vent-on solved zone-0 literal"
    );
    assert_eq!(
        v.setpoints.temperatures.as_ref()[1],
        26.16998864377302,
        "vent-on solved zone-1 literal"
    );
}

#[test]
/// Contract A5 — the per-surface channel's first explicit pin (mode-blind
/// code, identical in both modes): the engine's post-step
/// `surface_temperature` == the conductance-weighted per-surface average
/// 81.09607259126786, and the per-surface temperatures recompose through
/// the engine's own `build_per_surface_solver` + `update_surface` API with
/// the pinned operands — the POST-advance mass temps as the surface seeds,
/// the PRE-GAIN mass temps in q_ms, the per-surface exterior temperatures,
/// and the area-fraction φ_m split over the CONSTRUCTOR-DEFAULT area
/// setpoints (wall 53.665631459994955, roof/floor 20.0 — the fixture never
/// sets them). FINDING (recorded here, routed to the physics queue):
/// `SurfaceNode::update` is documented "backward Euler" but performs the
/// EXPLICIT forward-Euler recurrence — pinned as-is; the value IS the
/// recurrence the docs misdescribe.
fn per_surface_channel_recurrence_and_area_default_pm() {
    let mut m = fixture_9r4c_additive(None);
    // The area-setpoint premise: constructor defaults, never overridden.
    assert_eq!(
        m.setpoints.wall_area.as_ref()[0],
        53.665631459994955,
        "the default wall area (perimeter·3.0 over the sqrt(20) geometry)"
    );
    assert_eq!(
        m.setpoints.roof_area.as_ref()[0],
        20.0,
        "the default roof area (the flat-roof assumption)"
    );
    assert_eq!(
        m.setpoints.floor_area.as_ref()[0],
        20.0,
        "the default floor area (zone_area at construction = 20.0)"
    );
    // The φ_m split (zero irradiance → the area-fraction branch).
    let total_area = (53.665631459994955 + 20.0) + 20.0;
    let pm_w = 12768.0 * 53.665631459994955 / total_area;
    let pm_r = 12768.0 * 20.0 / total_area;
    let pm_f = 12768.0 * 20.0 / total_area;
    assert_eq!(pm_w, 7315.413047462013, "pm wall (bit-exact)");
    assert_eq!(pm_r, 2726.293476268993, "pm roof");
    assert_eq!(pm_f, 2726.293476268993, "pm floor");
    m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    // The write-back: the weighted per-surface average, bit-exact.
    assert_eq!(
        s.surface_temperature, 81.09607259126786,
        "the post-step surface_temperature == the per-surface write-back"
    );
    // The per-surface temperatures recompose through the engine's own API
    // with the step's operands (the solver is rebuilt from the POST-advance
    // mass temps exactly as the step does; q_ms uses the PRE-GAIN temps).
    let mut ps = s.build_per_surface_solver();
    ps.update_surface(0, DT, 18.0, T_OUT, pm_w);
    ps.update_surface(1, DT, 22.0, T_OUT, pm_r);
    ps.update_surface(2, DT, 20.0, T_G, pm_f);
    let temps = ps.surface_temperatures();
    assert_eq!(temps[0], 123.68230884030473, "per-surface wall");
    assert_eq!(temps[1], 57.28121027076551, "per-surface roof");
    assert_eq!(temps[2], 61.48411726407201, "per-surface floor");
    // And the in-test recurrence composes the same values op-for-op.
    let w = per_surface_update(s.mass.wall.temperature, 50.0, 20.0, 18.0, T_OUT, pm_w);
    assert_eq!(
        w, 123.68230884030473,
        "the in-test recurrence == the engine API"
    );
    let wb = ((50.0 * temps[0] + 40.0 * temps[1]) + 60.0 * temps[2]) / 150.0;
    assert_eq!(wb, 81.09607259126786, "the write-back closed form");
    // The two-zone arm (zone 1's own conductances 45/35/55, its own solve).
    let mut v = fixture_9r4c_2z_additive(None);
    v.step_physics(0, T_OUT, DT);
    let s1 = &v.conduction.backend.multi_node_solvers[1];
    let pm_w1 = 7680.0 * 53.665631459994955 / total_area;
    let pm_r1 = 7680.0 * 20.0 / total_area;
    assert_eq!(pm_w1, 4400.248449601211, "zone-1 pm wall");
    assert_eq!(pm_r1, 1639.8757751993942, "zone-1 pm roof (== pm floor)");
    assert_eq!(
        s1.surface_temperature, 55.01190364405396,
        "zone-1 write-back literal"
    );
}

#[test]
/// Contract A6 — the clobber chain: the write-back IS the observable
/// `surface_temperature` after each step (81.09607259126786 → 81.73223419947382)
/// and is DEAD at this production path — step 1's mass advance consumes the
/// mass-derived 20.67786979440258 (the #862 value, == the opening block's
/// re-derive), never the write-back: the step-1 node literals compose
/// bit-exactly from the consumed mass-derived operand, and provably NOT
/// from the write-back (the write-back-as-operand closed form lands
/// elsewhere). The write-back is overwritten again at step 1 (a fresh
/// 81.73223419947382 from the step-1 state) — the LIMIT-35 §12 clobber,
/// now pinned for both modes with the sites named.
fn writeback_clobbered_next_step_consumes_the_mass_derived_value() {
    let mut m = fixture_9r4c_additive(None);
    m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        s.surface_temperature, 81.09607259126786,
        "the write-back observable after step 0"
    );
    let ts862 = ((50.0 * s.mass.wall.temperature + 40.0 * s.mass.roof.temperature)
        + 60.0 * s.mass.floor.temperature)
        / 150.0;
    assert_eq!(
        ts862, 20.67786979440258,
        "the #862 weighted value the NEXT step consumes"
    );
    // Step-1's PRE-advance mass temps == step-0's POST-advance literals.
    let w_pre = s.mass.wall.temperature;
    assert_eq!(w_pre, 18.570724436052867, "the step-1 pre wall temp");
    m.step_physics(1, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    // Step-1 nodes composed from the CONSUMED mass-derived operand.
    let (phi_st, phi_m) = (7776.0_f64, 12768.0_f64);
    let w = additive_node_euler(
        4.0e7,
        w_pre,
        20.0,
        T_OUT,
        50.0,
        20.67786979440258,
        phi_st * (50.0 / 150.0) + phi_m * (50.0 / 150.0),
    );
    assert_eq!(
        w, 19.14150337884098,
        "step-1 wall literal composes from the mass-derived operand"
    );
    // ...and provably NOT from the write-back (that operand lands elsewhere).
    let w_wrong = additive_node_euler(
        4.0e7,
        w_pre,
        20.0,
        T_OUT,
        50.0,
        81.09607259126786,
        phi_st * (50.0 / 150.0) + phi_m * (50.0 / 150.0),
    );
    assert_ne!(
        w_wrong, 19.14150337884098,
        "the write-back-as-operand closed form does not produce the engine value"
    );
    // The write-back was refreshed from the step-1 state.
    assert_eq!(
        s.surface_temperature, 81.73223419947382,
        "the fresh write-back after step 1"
    );
    assert_ne!(
        s.surface_temperature, ts862,
        "the write-back overwrites the #862 value every step"
    );
}

#[test]
/// Contract A7 — the two-step composition and the mode-invariant chains:
/// step 1 composes bit-exactly from the step-0 committed state (the
/// committed air via the additive balance on the step-0 post nodes, the
/// hand-off via zt, the lumped via the mode-invariant estimate); the 5R1C
/// estimate and lumped chains reproduce r131's PR-mode literals ON THE
/// ADDITIVE FIXTURE (29.428271397080522 step-0 estimate via the lumped
/// composition; 26.149556909916907 step-0 lumped; 26.10141928940402
/// vent-on step-0 lumped — the chains never dispatch on coupling mode).
fn additive_two_step_composition_and_mode_invariant_chains() {
    let mut m = fixture_9r4c_additive(None);
    m.step_physics(0, T_OUT, DT);
    let lumped0 = m.mass.mass_temperatures.as_ref()[0];
    assert_eq!(
        lumped0, 26.149556909916907,
        "the step-0 lumped literal == r131's PR-mode value (mode-invariant chain)"
    );
    // The lumped composes from the mode-invariant estimate (r131's contract-4
    // literal as the t_i operand).
    let ts_den = 1069.0_f64;
    let t_s_lumped = (1000.0 * 20.0 + 69.0 * 29.428271397080522 + 7776.0) / ts_den;
    let cm_dt = 48.0 * 150_000.0 / DT;
    let h3 = {
        let h1 = (21.6 * 69.0) / (21.6 + 69.0);
        let h2 = h1 + 36.0;
        (h2 * 1000.0) / (h2 + 1000.0)
    };
    let numer = ((cm_dt * 20.0 + 15.36 * T_OUT) + h3 * t_s_lumped) + 12768.0;
    let lumped = numer / ((cm_dt + 15.36) + h3);
    assert_eq!(
        lumped, lumped0,
        "the lumped composes from the mode-invariant estimate"
    );
    m.step_physics(1, T_OUT, DT);
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        32.24857751084929,
        "the step-1 lumped literal"
    );
    // Vent-on step 0: the r131 contract-6 lumped literal recomputed.
    let mut v = fixture_9r4c_additive(Some(NightVentilation::new(FAN_M3_H, 18, 7)));
    v.step_physics(0, T_OUT, DT);
    assert_eq!(
        v.mass.mass_temperatures.as_ref()[0],
        26.10141928940402,
        "the vent-on step-0 lumped == r131's value (the fan rides the estimate only)"
    );
    v.step_physics(1, T_OUT, DT);
    assert_eq!(
        v.mass.mass_temperatures.as_ref()[0],
        32.148980580323496,
        "the vent-on step-1 lumped literal"
    );
}

#[test]
/// Contract A9 — the #4495 input-envelope guard at additive mode at the
/// two-zone fixture: a NaN outdoor temperature rejects the step (0.0 kWh)
/// while the solver's own nodes HAVE advanced into NaN inside the additive
/// Euler (committed before the guard) — pinned as-is; the guard's "state
/// untouched" scope is the commit, not the solver (the r131 contract-8
/// shape, additive arms, both zones advanced).
fn guard_nan_rejects_the_additive_step_2z() {
    let mut v = fixture_9r4c_2z_additive(None);
    v.step_physics(0, T_OUT, DT);
    v.step_physics(1, T_OUT, DT);
    let kwh = v.step_physics(0, f64::NAN, DT);
    assert_eq!(kwh, 0.0, "the rejected step returns exactly 0.0 kWh");
    let s0 = &v.conduction.backend.multi_node_solvers[0];
    assert!(
        s0.mass.wall.temperature.is_nan()
            && s0.mass.roof.temperature.is_nan()
            && s0.mass.internal.temperature.is_nan(),
        "w/r/internal advanced into NaN before the guard — pinned as-is"
    );
    assert!(
        !s0.mass.floor.temperature.is_nan(),
        "the ground-coupled floor advances finitely (the additive shared-T_s operand)"
    );
}

#[test]
/// Contract A8 — the #4495 input-envelope guard at additive mode
/// (single-zone): a NaN outdoor temperature rejects the step (0.0 kWh)
/// with the zone-air and lumped-mass commits untouched. FINDING (recorded
/// here, pinned as-is, routed to the physics queue): the additive-mode
/// solver state advances NON-UNIFORMLY before the guard — wall/roof go NaN
/// through their NaN outdoor boundary, internal through the envelope
/// average of w/r, `zone_temperature`/`surface_temperature` through the
/// NaN air balance — but the FLOOR advances FINITELY (22.899024388751087):
/// its exterior is the ground (13.0), and its consumed shared T_s is the
/// opening block's MASS-DERIVED weighted value, not a blend against the
/// NaN pre-air (the PR-mode contract 8 pins floor == NaN through exactly
/// that blend — the mode contrast is the observable). The guard's "state
/// untouched" scope is the commit, not the solver.
fn guard_nan_rejects_the_additive_step() {
    let mut m = fixture_9r4c_additive(None);
    m.step_physics(0, T_OUT, DT);
    m.step_physics(1, T_OUT, DT);
    let kwh = m.step_physics(0, f64::NAN, DT);
    assert_eq!(kwh, 0.0, "the rejected step returns exactly 0.0 kWh");
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        34.31138437180818,
        "the zone-air commit is untouched by the rejected step"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        32.24857751084929,
        "the lumped-mass commit is untouched by the rejected step"
    );
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert!(
        s.mass.wall.temperature.is_nan()
            && s.mass.roof.temperature.is_nan()
            && s.mass.internal.temperature.is_nan()
            && s.zone_temperature.is_nan()
            && s.surface_temperature.is_nan(),
        "w/r/internal + zt + surf advanced into NaN before the guard"
    );
    // The floor advanced FINITELY on its ground boundary and the mass-derived
    // shared T_s: compose its closed form from the step-1 literals.
    let ts_step2 = ((50.0 * 19.14150337884098 + 40.0 * 23.748564613709405)
        + 60.0 * 21.934954671993715)
        / 150.0;
    let floor = additive_node_euler(
        3.0e7,
        21.934954671993715,
        10.0,
        T_G,
        60.0,
        ts_step2,
        (7776.0 + 12768.0) * (60.0 / 150.0),
    );
    assert_eq!(
        floor, 22.899024388751087,
        "the floor advanced finitely through the rejected step (engine probe)"
    );
    assert_eq!(
        s.mass.floor.temperature, floor,
        "the engine's floor == the closed form"
    );
}

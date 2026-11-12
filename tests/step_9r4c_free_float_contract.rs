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

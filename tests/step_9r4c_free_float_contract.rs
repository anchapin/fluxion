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
//!   and lumped mass commits untouched, and since r136 the solver advances
//!   nothing either (the step-level snapshot + rejection-path rollback;
//!   the r131/r134 as-is NaN-advance pins are superseded by contract A8's
//!   dated verdict).
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
/// commits untouched (both still the step-1 values) — and since loop r136
/// the SOLVER is untouched too: every node (floor included — the pre-air
/// blend no longer fires) restores to its step-1 value. [Supersedes the
/// r131 as-is pin, which recorded the solver advancing into NaN through
/// `step_with_gains` and poisoning subsequent steps — the r134 finding,
/// fixed in r136 by the step-level pre-step snapshot + rejection-path
/// rollback; see contract A8 for the verdict.]
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
    // [R136 rollback] the rejected step advances NOTHING at the solver:
    // every node restores to its step-1 value (engine-probe literals).
    assert_eq!(s.mass.wall.temperature, 19.005692363742437);
    assert_eq!(s.mass.roof.temperature, 23.535366273033063);
    assert_eq!(s.mass.floor.temperature, 21.739124592967357);
    assert_eq!(s.mass.internal.temperature, 18.485860474480138);
    assert_eq!(s.zone_temperature, -0.400967803731356);
    assert_eq!(s.surface_temperature, 81.66479128963712);
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
    // stay untouched and — since r136 — both zones' solvers restore to
    // their step-1 state (the rejection advances nothing).
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
    // [R136 rollback] both zones restore to their step-1 values
    // (engine-probe literals; was: pinned advancing into NaN as-is).
    let vs0 = &v.conduction.backend.multi_node_solvers[0];
    assert_eq!(vs0.mass.wall.temperature, 19.005692363742437);
    assert_eq!(vs0.mass.roof.temperature, 23.535366273033063);
    assert_eq!(vs0.mass.floor.temperature, 21.739124592967357);
    assert_eq!(vs0.mass.internal.temperature, 18.485860474480138);
    assert_eq!(vs0.zone_temperature, -0.400967803731356);
    assert_eq!(vs0.surface_temperature, 81.66479128963712);
    let vs1 = &v.conduction.backend.multi_node_solvers[1];
    assert_eq!(vs1.mass.wall.temperature, 16.985317546369533);
    assert_eq!(vs1.mass.roof.temperature, 25.40756375800886);
    assert_eq!(vs1.mass.floor.temperature, 20.54974866469699);
    assert_eq!(vs1.mass.internal.temperature, 19.858828053437428);
    assert_eq!(vs1.zone_temperature, 29.72231486401823);
    assert_eq!(vs1.surface_temperature, 55.560222449881444);
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
//   RESOLVED (r136): the r134 finding recorded here — `fluxion-core`'s
//   `SurfaceNode::update` and every doc around it claiming "backward
//   Euler" while the code runs the EXPLICIT forward-Euler recurrence —
//   is fixed by the dated doc corrections in the same round's PR; the
//   VALUE was always the recurrence, the docs now say so;
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
//   temperature rejects the step (0.0 kWh), the commits stay untouched —
//   and since r136 the SOLVER advances nothing (the step-level snapshot +
//   rejection-path rollback). [Supersedes the r134 as-is pin ("the
//   solver's own nodes HAVE advanced into NaN — the guard's scope is the
//   commit, not the solver"): that behavior was the r134 finding, ruled a
//   defect in r136 — it defeated #4495's own intent, permanently bricking
//   the zone after one bad step; see contract A8's verdict.]
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
/// sets them). RESOLVED (r136): the r134 finding recorded here —
/// `SurfaceNode::update` documented "backward Euler" while performing the
/// EXPLICIT forward-Euler recurrence — is fixed by the round's dated doc
/// corrections; the value IS the recurrence, the docs now say so.
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
/// two-zone fixture: a NaN outdoor temperature rejects the step (0.0 kWh),
/// both zone-air and lumped-mass commits stay untouched, and since loop
/// r136 BOTH zones' solvers restore to their step-1 state — the rejection
/// advances nothing. [Supersedes the r134 as-is pin, which recorded both
/// zones advancing into NaN inside the additive Euler (committed before
/// the guard) with the guard's "state untouched" scope limited to the
/// commit — the r134 finding, fixed in r136.]
fn guard_nan_rejects_the_additive_step_2z() {
    let mut v = fixture_9r4c_2z_additive(None);
    v.step_physics(0, T_OUT, DT);
    v.step_physics(1, T_OUT, DT);
    let kwh = v.step_physics(0, f64::NAN, DT);
    assert_eq!(kwh, 0.0, "the rejected step returns exactly 0.0 kWh");
    // [R136 rollback] both zones restore to their step-1 values
    // (engine-probe literals; the r134 pins showed these going NaN).
    let s0 = &v.conduction.backend.multi_node_solvers[0];
    assert_eq!(s0.mass.wall.temperature, 19.14150337884098);
    assert_eq!(s0.mass.roof.temperature, 23.748564613709405);
    assert_eq!(s0.mass.floor.temperature, 21.934954671993715);
    assert_eq!(s0.mass.internal.temperature, 21.09802336491205);
    assert_eq!(s0.zone_temperature, 33.62531338508693);
    assert_eq!(s0.surface_temperature, 81.73223419947382);
    let s1 = &v.conduction.backend.multi_node_solvers[1];
    assert_eq!(s1.mass.wall.temperature, 16.914602166385745);
    assert_eq!(s1.mass.roof.temperature, 25.261965385769624);
    assert_eq!(s1.mass.floor.temperature, 20.44041225796746);
    assert_eq!(s1.mass.internal.temperature, 19.662060370598887);
    assert_eq!(s1.zone_temperature, 27.178443162599674);
    assert_eq!(s1.surface_temperature, 55.51131295675001);
}

#[test]
/// Contract A8 — the #4495 input-envelope guard at additive mode
/// (single-zone): a NaN outdoor temperature rejects the step (0.0 kWh),
/// the zone-air and lumped-mass commits stay untouched, and — the r136
/// verdict on the r134 finding — the SOLVER advances nothing: every node
/// restores to its step-1 value.
///
/// THE VERDICT (r136, superseding the r134 as-is pin): the pre-fix engine
/// advanced the solver non-uniformly through a rejected step — wall/roof
/// NaN through their NaN outdoor boundary, internal through the envelope
/// average of w/r, `zone_temperature`/`surface_temperature` through the
/// NaN air balance, and the FLOOR finitely (22.899024388751087, closed
/// form below in the r134 suite history) on its ground boundary and the
/// mass-derived shared T_s, the mode contrast with PR's pre-air blend.
/// That behavior is a defect, not semantics: it defeats #4495's own
/// stated intent ("without this guard the non-finite value propagates
/// into the committed zone state and corrupts every subsequent step") —
/// the poisoned solver re-entered every later step's air balance, so one
/// bad step bricked the zone permanently (every subsequent step rejected,
/// 0.0 kWh, silently) — exactly the corruption the guard was built to
/// prevent, one level down; and the r133 precedent (solver-less zones'
/// half-applied steps) had already ruled "rejected steps that
/// half-applied" a defect. Correct semantics: a rejected step advances
/// NOTHING — enforced since r136 by the step-level pre-step snapshot
/// (solver nodes, lumped-mass pair, controller previous zone temperature)
/// restored verbatim on every #4495/#4500 rejection path of
/// `step_physics_9r4c`. Accepted-step arithmetic is untouched (the
/// battery is byte-identical — proven by the strict+output-diff run
/// riding this PR), so only rejection-path observables move.
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
    // [R136 rollback] the solver restores to its step-1 state
    // (engine-probe literals; the floor's r134 finite-advance value
    // 22.899024388751087 no longer occurs — nothing integrates).
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(s.mass.wall.temperature, 19.14150337884098);
    assert_eq!(s.mass.roof.temperature, 23.748564613709405);
    assert_eq!(s.mass.floor.temperature, 21.934954671993715);
    assert_eq!(s.mass.internal.temperature, 21.09802336491205);
    assert_eq!(s.zone_temperature, 33.62531338508693);
    assert_eq!(s.surface_temperature, 81.73223419947382);
}

#[test]
/// Contract A8b — the rollback verdict's RECOVERY arm (additive): a
/// #4495-rejected step leaves no trace, so the model resumes exactly as
/// if the bad step never happened. Model A runs timesteps 0, 1, a NaN
/// step at 2, then 3; model C skips the bad step entirely (0, 1, 3).
/// Both compose timestep 3 from the identical post-step-1 state, so every
/// durable field must agree bit-exactly. [Pre-fix, A's solver stayed
/// NaN-poisoned and timestep 3 rejected with 0.0 kWh forever — the
/// silent brick the #4495 guard was built to prevent.]
fn rejected_step_leaves_no_trace_additive() {
    let mut a = fixture_9r4c_additive(None);
    a.step_physics(0, T_OUT, DT);
    a.step_physics(1, T_OUT, DT);
    let kwh_bad = a.step_physics(2, f64::NAN, DT);
    assert_eq!(kwh_bad, 0.0, "the rejected step returns exactly 0.0 kWh");
    let mut c = fixture_9r4c_additive(None);
    c.step_physics(0, T_OUT, DT);
    c.step_physics(1, T_OUT, DT);
    c.step_physics(3, T_OUT, DT);
    let kwh_a = a.step_physics(3, T_OUT, DT);
    assert_eq!(
        kwh_a, 0.0,
        "timestep 3 integrates normally (the free-float success return)"
    );
    assert_eq!(
        a.setpoints.temperatures.as_ref()[0],
        c.setpoints.temperatures.as_ref()[0],
        "the recovered committed air == the never-poisoned twin"
    );
    assert_ne!(
        a.setpoints.temperatures.as_ref()[0],
        34.31138437180818,
        "timestep 3 actually integrated (a bricked model freezes at the step-1 commit)"
    );
    assert_eq!(
        a.mass.mass_temperatures.as_ref()[0],
        c.mass.mass_temperatures.as_ref()[0],
        "the recovered lumped mass == the twin's"
    );
    let (sa, sc) = (
        &a.conduction.backend.multi_node_solvers[0],
        &c.conduction.backend.multi_node_solvers[0],
    );
    assert_eq!(sa.mass.wall.temperature, sc.mass.wall.temperature);
    assert_eq!(sa.mass.roof.temperature, sc.mass.roof.temperature);
    assert_eq!(sa.mass.floor.temperature, sc.mass.floor.temperature);
    assert_eq!(sa.mass.internal.temperature, sc.mass.internal.temperature);
    assert_eq!(sa.zone_temperature, sc.zone_temperature);
    assert_eq!(sa.surface_temperature, sc.surface_temperature);
    assert!(
        sa.mass.wall.temperature.is_finite() && sa.zone_temperature.is_finite(),
        "the recovered solver state is finite (the r134 NaN shape is gone)"
    );
}

#[test]
/// Contract A8c — the recovery arm at the #1859 default (parallel
/// resistance) coupling: same shape as A8b, PR mode. The PR pre-air
/// blend NaNs the floor too (the r131 contract-8 shape), so pre-fix the
/// brick was uniform; the rollback erases either shape identically.
fn rejected_step_leaves_no_trace_parallel_resistance() {
    let mut a = fixture_9r4c(None);
    a.step_physics(0, T_OUT, DT);
    a.step_physics(1, T_OUT, DT);
    let kwh_bad = a.step_physics(2, f64::NAN, DT);
    assert_eq!(kwh_bad, 0.0, "the rejected step returns exactly 0.0 kWh");
    let mut c = fixture_9r4c(None);
    c.step_physics(0, T_OUT, DT);
    c.step_physics(1, T_OUT, DT);
    c.step_physics(3, T_OUT, DT);
    let kwh_a = a.step_physics(3, T_OUT, DT);
    assert_eq!(
        kwh_a, 0.0,
        "timestep 3 integrates normally (the free-float success return)"
    );
    assert_eq!(
        a.setpoints.temperatures.as_ref()[0],
        c.setpoints.temperatures.as_ref()[0],
        "the recovered committed air == the never-poisoned twin"
    );
    assert_ne!(
        a.setpoints.temperatures.as_ref()[0],
        35.927198831761395,
        "timestep 3 actually integrated (a bricked model freezes at the step-1 commit)"
    );
    assert_eq!(
        a.mass.mass_temperatures.as_ref()[0],
        c.mass.mass_temperatures.as_ref()[0],
        "the recovered lumped mass == the twin's"
    );
    let (sa, sc) = (
        &a.conduction.backend.multi_node_solvers[0],
        &c.conduction.backend.multi_node_solvers[0],
    );
    assert_eq!(sa.mass.wall.temperature, sc.mass.wall.temperature);
    assert_eq!(sa.mass.roof.temperature, sc.mass.roof.temperature);
    assert_eq!(sa.mass.floor.temperature, sc.mass.floor.temperature);
    assert_eq!(sa.mass.internal.temperature, sc.mass.internal.temperature);
    assert_eq!(sa.zone_temperature, sc.zone_temperature);
    assert_eq!(sa.surface_temperature, sc.surface_temperature);
}

#[test]
/// Contract A8d — the mass-commit rollback: a poison that reaches ONLY
/// the multi-node air balance (NaN solver-side `h_tr_is`; the model's own
/// 69.0 `h_tr_is` and the whole 5R1C chain stay finite) leaves the lumped
/// mass advance finite, so the mass COMMITTED through the rejected step
/// under the pre-fix behavior (the free-float air guard at step_9r4c.rs
/// fires AFTER the mass commit) while the solver half-advanced (internal
/// + zone_temperature NaN through the poisoned air). Post-fix the whole
/// step rolls back: mass, commits and solver all at their step-1 values.
fn air_rejected_step_rolls_back_the_mass_commit() {
    let mut m = fixture_9r4c_additive(None);
    m.step_physics(0, T_OUT, DT);
    m.step_physics(1, T_OUT, DT);
    m.conduction.backend.multi_node_solvers[0].h_tr_is = f64::NAN;
    let kwh = m.step_physics(2, T_OUT, DT);
    assert_eq!(kwh, 0.0, "the air-rejected step returns exactly 0.0 kWh");
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        32.24857751084929,
        "the lumped mass rolls back to the step-1 commit (was: committed through the rejection)"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        34.31138437180818,
        "the zone-air commit untouched"
    );
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(s.mass.wall.temperature, 19.14150337884098);
    assert_eq!(s.mass.roof.temperature, 23.748564613709405);
    assert_eq!(s.mass.floor.temperature, 21.934954671993715);
    assert_eq!(s.mass.internal.temperature, 21.09802336491205);
    assert_eq!(s.zone_temperature, 33.62531338508693);
    assert_eq!(s.surface_temperature, 81.73223419947382);
}

// ============================================================================
// Round 135 — the Case-900FF spec-path engine-truth arm (the r133
// remainder's last item). Every r131-r134 fixture was HAND-ROLLED
// (`ThermalModel::new` + injected solver); this arm drives the PRODUCTION
// construction route: `ThermalModel::from_spec(&case_900ff())` — HighMass
// promotes the selector to 9R4C selector-independently (#3277/#4175), the
// solver is built `new_with_mode(.., ParallelResistance)` (#1859) at the
// spec-derived per-surface parameters, and every conductance the chain
// consumes is the SPEC's own. The spec path differs from the analytic
// fixture in every seam that matters, and the differences are pinned:
//
// - (S1) the spec solar partition: `sda = 0.0`, `sbm = 0.0` (the r91 spec
//   values, not `ThermalModel::new`'s 0.1/0.6 defaults) with zero internal
//   loads — so phi_ia commits to exactly 0.0 and the whole transmitted
//   solar load rides phi_st (surfaces) while opaque rides phi_m (mass);
//   the air node is heated only THROUGH the mass nodes;
// - (S2) the collapsed h_tr_is seam: from-spec builds feed the MODEL's
//   `H_SI·A_floor` conductance (3.45·48 = 165.6, Issue #714/#4164) to BOTH
//   the model and the solver — the analytic fixture's artificial
//   120-vs-69 asymmetry does not exist on the production route;
// - (S3) the #2889 frame bridge is LIVE at 900FF (no WINDOW-01 U override:
//   the override is a 900/950FF-only builder call) — h_tr_w composes as
//   (5.8 + 0.1) + 0.05·16/12 over the geometric 2·(h+w) perimeter;
// - (S4) the zero-`h_tr_me` internal-node path: every node carries
//   h_tr_me 0.0 at the spec path, and the internal advance consumes the
//   envelope environment ONLY through the `0.0·t_env` operand — the env
//   is structurally inert for the VALUE (any finite perturbation of it
//   vanishes; the mean-vs-blend choice is not value-observable at zero),
//   but it must still be FINITE: a NaN env would poison through
//   0·NaN = NaN. `internal_node_envelope_temperature` supplies that
//   finite env through TWO redundant guard layers at zero (the
//   `h_tr_me <= 0.0` first branch AND the `h_me_sum > 1e-6` fallback —
//   flipping either alone is absorbed by the other; the teeth disclosed
//   in the duck review), and the finite literal IS the pin;
// - (S5) the construction-derived node parameters (cm, h_ms, h_em per
//   surface, the lumped h_tr_em/h_tr_ms/h_tr_me, the floor U-value) are
//   pinned as ENGINE-READ literals: their derivation from the ASHRAE 140
//   assemblies (wall_cap_for's κ blend, iso_13790 effective capacitance,
//   the #2229 h_ms_of_kappa calibration) is the recorded remainder seam —
//   the literals pin the pipeline I/O, the derivation interior is the next
//   round's candidate.
//
// Provenance: Python replication /tmp/r135_replication.py (dev-time
// evidence) — 24 checked observables bit-exact against the engine probe
// /tmp/r135_probe1.log at tip `92b28131` (both steps, all four nodes, both
// committed airs, both lumped masses, both pre-airs, 10 conductance/cache
// closed forms, the gains partition, both 0.0 kWh commits). No engine
// change, no re-record owed (the suite writes no engine state; the battery
// is untouched by construction).
// ============================================================================

fn fixture_spec_900ff() -> ThermalModel<VectorField> {
    use fluxion::sim::boundary::GroundTemperature;
    use fluxion::validation::ashrae140::cases::series_900::case_900ff;
    let mut m = ThermalModel::<VectorField>::from_spec(&case_900ff());
    // The driving set (spec-honest: 900FF carries NO internal loads; the
    // solar terms are the contract's drivers, as in every fixture above).
    m.solar.solar_gains = VectorField::new(vec![400.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0]);
    m
}

#[test]
/// Contract B1 — the spec-path dispatch and partition premises: HighMass
/// promotes to `NineRFourC` in BOTH the selector and the model type
/// (selector-independent, #3277/#4175), one solver in ParallelResistance
/// (#1859), free-float with inert HVAC (-999/999 setpoints, 100 kW
/// capacities), the spec geometry (8·6·2.7: floor 48.0, volume
/// 129.60000000000002, wall 75.60000000000001, roof 48.0), ground 9.4 °C
/// (Annex B §B3.3), the r91 spec solar partition (sda 0.0 / sbm 0.0 with
/// bare conv 0.4), no internal loads, and the single-clear spec window
/// (u 5.8 / SHGC 0.86 / frame fraction 0.15 — NOT the analytic
/// constructor's double-clear 0.787 path, and NOT the 900-series 3.0
/// W/m²K override, which only 900/950FF carry).
fn spec_path_dispatch_and_partition_premises() {
    let m = fixture_spec_900ff();
    assert_eq!(
        m.hvac.thermal_selector.zone_solver,
        ZoneSolverKind::NineRFourC,
        "premise: HighMass promotes the selector selector-independently"
    );
    assert_eq!(
        m.hvac.thermal_model_type,
        fluxion::sim::thermal_model_core::ThermalModelType::NineRFourC,
        "premise: the model type flag rides in lockstep (#4175)"
    );
    assert_eq!(m.conduction.backend.multi_node_solvers.len(), 1);
    assert_eq!(
        m.conduction.backend.multi_node_solvers[0].coupling_mode,
        MassAirCouplingMode::ParallelResistance,
        "premise: #1859 production coupling mode"
    );
    assert!(m.hvac.free_float, "premise: 900FF is free-floating");
    assert!(
        m.hvac.night_ventilation.is_none(),
        "premise: no night vent at 900FF"
    );
    assert_eq!(m.setpoints.heating_setpoint, -999.0);
    assert_eq!(m.setpoints.cooling_setpoint, 999.0);
    assert_eq!(m.hvac.hvac_heating_capacity, 100000.0);
    assert_eq!(m.hvac.hvac_cooling_capacity, 100000.0);
    assert_eq!(m.setpoints.zone_area.as_ref()[0], 48.0);
    assert_eq!(m.setpoints.zone_volume.as_ref()[0], 129.60000000000002);
    assert_eq!(
        m.setpoints.loads.as_ref()[0],
        0.0,
        "premise: no internal loads at the free-floating spec"
    );
    assert_eq!(m.setpoints.temperatures.as_ref()[0], 20.0);
    assert_eq!(
        m.conduction.ground_temperature.ground_temperature(0),
        9.4,
        "premise: Annex B §B3.3 ground temperature"
    );
    assert_eq!(
        m.solar.convective_fraction, 0.4,
        "premise: bare conv default"
    );
    assert_eq!(
        m.solar.solar_distribution_to_air, 0.0,
        "premise: the r91 spec partition (S1) — NOT the constructor's 0.1"
    );
    assert_eq!(
        m.solar.solar_beam_to_mass_fraction, 0.0,
        "premise: the r91 spec partition (S1) — NOT the constructor's 0.6"
    );
    let spec_window = &m.solar;
    let _ = spec_window;
    assert_eq!(m.solar.solar_gains.as_ref()[0], 400.0, "driving set");
    assert_eq!(m.solar.opaque_solar_gains.as_ref()[0], 50.0, "driving set");
}

#[test]
/// Contract B2 — the spec-path conductances and caches: the replicated
/// closed forms (h_tr_is = H_SI·A; h_ve = ACH·V·ρ·cp/3600; the #2889 frame
/// bridge (S3); h_tr_floor = floor_u·A) hit the engine bit-exactly, the
/// collapsed h_tr_is seam (S2) holds (model == solver), and the
/// construction-derived parameters ride as engine-read literals (S5, the
/// recorded derivation seam). The cache closed forms recompose from the
/// engine-read inputs bit-exactly (tr1, msip, den, gc, the #896 h_tr_3).
fn spec_path_conductances_and_caches() {
    let m = fixture_spec_900ff();
    // (S2) the collapsed seam:
    assert_eq!(m.conduction.h_tr_is.as_ref()[0], 165.60000000000002);
    assert_eq!(
        m.conduction.backend.multi_node_solvers[0].h_tr_is,
        m.conduction.h_tr_is.as_ref()[0],
        "premise: model and solver carry the SAME h_tr_is at the spec path"
    );
    // replicated closed forms:
    assert_eq!(
        3.45 * 48.0,
        m.conduction.h_tr_is.as_ref()[0],
        "H_SI·A_floor recomposed"
    );
    assert_eq!(
        (0.414 * (129.60000000000002 * 1.2 * 1005.0)) / 3600.0,
        m.conduction.h_ve.as_ref()[0],
        "ACH·V·ρ·cp/3600 recomposed"
    );
    let u_eff = (5.8_f64 + 0.1) + (0.05 * (2.0 * (2.0 + 6.0))) / 12.0;
    assert_eq!(
        12.0 * u_eff,
        m.conduction.h_tr_w.as_ref()[0],
        "#2889 frame bridge recomposed (S3): u + frame_u + psi·perimeter/area"
    );
    let floor_u = 0.18109607694010674_f64;
    assert_eq!(
        m.setpoints.floor_u_value, floor_u,
        "engine-read floor U (S5)"
    );
    assert_eq!(floor_u * 48.0, m.conduction.h_tr_floor.as_ref()[0]);
    // engine-read construction-derived parameters (S5):
    assert_eq!(m.conduction.h_tr_ms.as_ref()[0], 1608.0);
    assert_eq!(m.conduction.h_tr_em.as_ref()[0], 48.880343578250944);
    assert_eq!(m.mass.h_tr_me.as_ref()[0], 131.04);
    assert_eq!(m.mass.thermal_capacitance.as_ref()[0], 18014427.840000004);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(s.mass.wall.capacitance, 7829058.24);
    assert_eq!(s.mass.wall.h_tr_ms, 76.34726688102896);
    assert_eq!(s.mass.wall.h_tr_em, 71.64739830382713);
    assert_eq!(s.mass.roof.capacitance, 5480428.800000001);
    assert_eq!(s.mass.roof.h_tr_ms, 32.91511795494197);
    assert_eq!(s.mass.roof.h_tr_em, 31.726281109549806);
    assert_eq!(s.mass.floor.capacitance, 4704940.800000001);
    assert_eq!(s.mass.floor.h_tr_ms, 17.981819851253327);
    assert_eq!(s.mass.floor.h_tr_em, 8.96001344976471);
    assert_eq!(s.mass.internal.capacitance, 792000.0);
    assert_eq!(
        s.mass.internal.h_tr_ms, 10.0,
        "not used for the internal node"
    );
    assert_eq!(
        s.mass.internal.h_tr_em, 5.0,
        "not used for the internal node"
    );
    assert_eq!(
        s.mass.internal.h_tr_me, 0.0,
        "premise (S4): every node h_tr_me is 0.0 at the spec path"
    );
    // cache closed forms recompose from the engine-read inputs:
    let tr1 = (1608.0_f64 + 165.60000000000002) + 131.04;
    assert_eq!(tr1, m.conduction.derived_term_rest_1.as_ref()[0]);
    let msip = 1608.0_f64 * 165.60000000000002;
    assert_eq!(msip, m.conduction.derived_h_ms_is_prod.as_ref()[0]);
    let h_ext = 71.6_f64 + 17.974224;
    let gc = tr1 * (floor_u * 48.0);
    assert_eq!(gc, m.conduction.derived_ground_coeff.as_ref()[0]);
    assert_eq!(
        (msip + tr1 * h_ext) + gc,
        m.conduction.derived_den.as_ref()[0],
        "cached den (Python-replicated operand order)"
    );
    let h1 = (17.974224_f64 * 165.60000000000002) / (17.974224 + 165.60000000000002);
    let h2 = h1 + 71.6_f64;
    let h3 = (h2 * 1608.0) / (h2 + 1608.0);
    assert_eq!(
        h3,
        m.conduction.derived_h_tr_3.as_ref()[0],
        "#896 bottleneck recomposed"
    );
    assert_eq!(
        m.mass.air_thermal_capacitance.as_ref()[0],
        129.60000000000002 * 1.2 * 1005.0,
        "air capacitance = ρ·cp·V"
    );
}

#[test]
/// Contract B3 — the committed chain through the spec fixture, two steps:
/// the pre-airs (the solver's `zone_temperature` set), all four node
/// literals per step, both committed airs (the parallel balance on the
/// post-step nodes, recomposed in-test from the engine node reads), both
/// 0.0 kWh free-float commits, and the hand-off (step 2's pre-air IS
/// step 1's committed air). The gains partition premise is pinned at the
/// engine's own fractions: phi_ia = 0.0 exactly (S1 — the air is heated
/// only through the mass nodes), phi_st carries the whole transmitted
/// solar load, phi_m the opaque share; the decomposition conserves.
fn spec_path_chain_two_steps() {
    let mut m = fixture_spec_900ff();
    // gains partition (S1) at the engine's own fractions:
    let load_w = 0.0_f64 * 48.0;
    let sol_w = 400.0_f64 * 48.0;
    let opaque_w = 50.0_f64 * 48.0;
    let sol_to_air = sol_w * 0.0;
    let phi_ia = load_w * 0.4 + sol_to_air;
    let phi_st = load_w * (1.0 - 0.4) + (sol_w - sol_to_air) * (1.0 - 0.0);
    let phi_m = (sol_w - sol_to_air) * 0.0 + opaque_w;
    assert_eq!(
        phi_ia, 0.0,
        "S1: the air gets NO direct gains at the spec path"
    );
    assert_eq!(phi_st, 19200.0);
    assert_eq!(phi_m, 2400.0);
    assert_eq!(
        phi_ia + phi_st + phi_m,
        load_w + sol_w + opaque_w,
        "the sda/sbm-0 partition conserves the driving set exactly"
    );
    let kwh0 = m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        s.zone_temperature, 15.266117931266905,
        "step-1 pre-air: the parallel balance on the PRE-step nodes"
    );
    assert_eq!(s.mass.wall.temperature, 24.547802980996828, "wall node");
    assert_eq!(s.mass.roof.temperature, 22.8391289109059, "roof node");
    assert_eq!(s.mass.floor.temperature, 22.159651283235362, "floor node");
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        18.344345521157422,
        "committed air step 1 (bit-exact)"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        20.31631526994135,
        "lumped mass step 1 (bit-exact)"
    );
    assert_eq!(kwh0, 0.0, "free-float commits exactly 0.0 kWh");
    // the committed air recomposed from the engine node reads:
    let hp_w =
        (76.34726688102896_f64 * 165.60000000000002) / (76.34726688102896 + 165.60000000000002);
    let hp_r =
        (32.91511795494197_f64 * 165.60000000000002) / (32.91511795494197 + 165.60000000000002);
    let hp_f =
        (17.981819851253327_f64 * 165.60000000000002) / (17.981819851253327 + 165.60000000000002);
    let recomposed = {
        let (tw, tr, tf) = (
            s.mass.wall.temperature,
            s.mass.roof.temperature,
            s.mass.floor.temperature,
        );
        let num = (hp_w * tw + hp_r * tr) + hp_f * tf;
        let num = (num + 17.974224 * T_OUT) + 0.0 * T_OUT;
        let num = num + phi_ia;
        num / ((hp_w + hp_r) + hp_f + 17.974224 + 0.0)
    };
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        recomposed,
        "committed air == the parallel balance recomposed from the committed nodes"
    );
    let kwh1 = m.step_physics(0, T_OUT, DT);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        s.zone_temperature, 18.344345521157422,
        "the hand-off: step 2's pre-air IS step 1's committed air"
    );
    assert_eq!(s.mass.wall.temperature, 28.92226196100471);
    assert_eq!(s.mass.roof.temperature, 25.625635993882);
    assert_eq!(s.mass.floor.temperature, 24.31596585762751);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        21.32989132022129,
        "committed air step 2 (bit-exact)"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        20.629083347822874,
        "lumped mass step 2 (bit-exact)"
    );
    assert_eq!(kwh1, 0.0);
}

#[test]
/// Contract B4 — the estimate/lumped chain and the zero-`h_tr_me` guard at
/// the spec fixture: the 5R1C estimate recomposes in-test from the
/// engine-read cache inputs (mode-independent, exactly the analytic
/// fixture's composition with the spec's values — gc·T_ground carries the
/// spec's 9.4, not the analytic 13.0), the lumped-mass engine literals
/// recompose THROUGH that estimate (the composition arbitrates the
/// estimate against engine truth: t_s is strictly monotone in t_i_act),
/// the estimate stays provably distinct from the committed air (ADR-002 at
/// the spec path), and the internal node composes with the zero-`h_tr_me`
/// env operand (S4 — the value-level inertness of the env at zero and the
/// finiteness pin; the NaN-poison tooth is what bites, see the header).
fn spec_path_estimate_lumped_and_internal_guard() {
    let mut m = fixture_spec_900ff();
    let kwh0 = m.step_physics(0, T_OUT, DT);
    let _ = kwh0;
    let committed1 = m.setpoints.temperatures.as_ref()[0];
    let lumped1 = m.mass.mass_temperatures.as_ref()[0];
    // the estimate, recomposed from the engine-read cache inputs:
    let est1 = {
        let msip = 1608.0_f64 * 165.60000000000002;
        let tr1 = (1608.0_f64 + 165.60000000000002) + 131.04;
        let gc = tr1 * 8.692611693125123;
        let h_ext = 71.6_f64 + 17.974224;
        let phi_ia = 0.0_f64;
        let phi_st = 19200.0_f64;
        let mut num_rest = phi_ia + h_ext * T_OUT;
        num_rest = num_rest * tr1;
        num_rest = num_rest + gc * 9.4;
        (msip * 20.0 + 165.60000000000002 * phi_st + num_rest) / ((msip + tr1 * h_ext) + gc)
    };
    assert_eq!(
        est1, 15.337552659928775,
        "the 5R1C estimate at the spec fixture"
    );
    assert_ne!(
        est1, committed1,
        "ADR-002: the estimate and the committed air stay distinct"
    );
    let s1 = &m.conduction.backend.multi_node_solvers[0];
    // S4: the internal node's environment is the ARITHMETIC MEAN of the
    // post-advance envelope nodes (the h_series blend at h_me = 0 would be
    // 0/0 = NaN). Recompose the step-1 internal literal from the engine's
    // own step-1 envelope reads:
    let env_mean =
        (s1.mass.wall.temperature + s1.mass.roof.temperature + s1.mass.floor.temperature) / 3.0;
    let c_int = 792000.0_f64 / DT;
    let internal_rec = ((c_int * 20.0 + 165.60000000000002 * 15.266117931266905) + 0.0 * env_mean)
        / (c_int + 165.60000000000002 + 0.0);
    assert_eq!(
        internal_rec, s1.mass.internal.temperature,
        "S4: the internal advance composes through the 0.0·env operand \
         (the env must be FINITE, or 0·NaN poisons the node)"
    );
    assert_eq!(
        s1.mass.internal.temperature, 17.966984256788898,
        "internal node literal, step 1 (finite by value IS the S4 pin)"
    );
    assert!(
        internal_rec.is_finite(),
        "the finite literal IS the zero-h_tr_me guard pin"
    );
    // the lumped mass recomposed THROUGH the estimate (arbitrates est1):
    let ts_den = (1608.0_f64 + 165.60000000000002) + 131.04;
    let t_s = (1608.0_f64 * 20.0 + 165.60000000000002 * est1 + 19200.0) / ts_den;
    let h3 = {
        let h1 = (17.974224_f64 * 165.60000000000002) / (17.974224 + 165.60000000000002);
        let h2 = h1 + 71.6_f64;
        (h2 * 1608.0) / (h2 + 1608.0)
    };
    let cm_dt = 18014427.840000004_f64 / DT;
    let lumped_rec = {
        let denom = (cm_dt + 48.880343578250944) + h3;
        let numer = ((cm_dt * 20.0 + 48.880343578250944 * T_OUT) + h3 * t_s) + 2400.0;
        numer / denom
    };
    assert_eq!(
        lumped_rec, lumped1,
        "the engine's lumped mass composes through the recomposed estimate"
    );
    // step 2 rides the step-1 lumped state:
    m.step_physics(0, T_OUT, DT);
    let est2 = {
        let msip = 1608.0_f64 * 165.60000000000002;
        let tr1 = (1608.0_f64 + 165.60000000000002) + 131.04;
        let gc = tr1 * 8.692611693125123;
        let h_ext = 71.6_f64 + 17.974224;
        let mut num_rest = 0.0_f64 + h_ext * T_OUT;
        num_rest = num_rest * tr1;
        num_rest = num_rest + gc * 9.4;
        (msip * lumped1 + 165.60000000000002 * 19200.0 + num_rest) / ((msip + tr1 * h_ext) + gc)
    };
    assert_eq!(est2, 15.523307135826942);
    let t_s2 = (1608.0_f64 * lumped1 + 165.60000000000002 * est2 + 19200.0) / ts_den;
    let lumped2_rec = {
        let denom = (cm_dt + 48.880343578250944) + h3;
        let numer = ((cm_dt * lumped1 + 48.880343578250944 * T_OUT) + h3 * t_s2) + 2400.0;
        numer / denom
    };
    assert_eq!(lumped2_rec, m.mass.mass_temperatures.as_ref()[0]);
    {
        let s2 = &m.conduction.backend.multi_node_solvers[0];
        assert_eq!(
            s2.mass.internal.temperature, 18.129046044598617,
            "internal node literal, step 2"
        );
    }
}

// ============================================================================
// Round 141 — the S5 remainder: the assembly→conductance derivation INTERIOR.
//
// Contract B2 above pins S5 as ENGINE-READ literals (cm, h_ms, h_em per
// surface, the lumped h_tr_em/h_tr_ms/h_tr_me, the floor U-value). This
// section closes the recorded remainder seam: it replicates the derivation
// interior operation-for-operation — the ISO 13790 effective capacitance
// (`iso_13790_effective_capacitance_per_area`: layers interior to AND
// including the dominant-insulation layer, 100 mm active-thickness cap
// checked before each layer), the `wall_cap_for` κ blend
// (`massiveness_weight` smoothstep on the wall's FULL κ, 80k→100k edges),
// the #2229 `h_ms_of_kappa` calibration (2.0 + 11.4·w), the A_m blend
// (`a_m_blend_weight` smoothstep 165k→260k on the wall's EFFECTIVE κ), the
// half-insulation h_ms walks (interior layers full, insulation half, 0.001
// floor guard), the exterior-side h_tr_em walks, and the floor U-value
// forms (film 5.88 + materials + ground 0.17 for the setpoints U; plain
// 8.29/18.3 films for the builder's h_tr_em input) — all from the CASE
// SPEC's own assembly data (`spec.construction.{wall,roof,floor}.layers`),
// asserted bit-exact against the engine values.
//
// Exterior-walk transcription note (pin-what-IS): the builder's h_tr_em walk
// iterates `enumerate()` computing `reverse_idx = len-1-idx`, so the FIRST
// layer visited is `layers[0]` — which the `Construction` doc calls the
// interior-most — and the loop breaks at the insulation. For this
// middle-insulation stack the walk therefore rides the interior-side layer
// at full R plus the insulation half over the exterior film, and the true
// exterior layer never contributes; the interior-side h_ms walk adds the
// same layer to the OTHER side. The asymmetry is observable through the
// pinned h_tr_ms/h_tr_em pair and is pinned AS-IS (engine truth), not as
// the walk's own comment describes it. Any re-derivation of that walk is an
// engine-semantics change and routes as a finding, never as a pin edit.
//
// Teeth hygiene: every intermediate consumed by an asserted closed form is
// recomposed IN-TEST from the spec layers — no engine-read intermediates —
// so a mutation anywhere in the derivation interior flips the matching pin.
//
// Provenance: Python replication /tmp/r141_s5_replication.py — 16/16 targets
// bit-exact against the engine-read literals at tip `e1a54a11` (loop round
// 141 base), including the two 1-ULP-sensitive floor U-values (the
// `r_value_total` association `r_film_int + r_materials + r_exterior`, with
// `r_materials` as its own left-to-right sum, is load-bearing there; folding
// the film into the layer sum flips the last digit). No engine change, no
// re-record owed (tests-only; the battery is untouched by construction).
// ============================================================================

/// Layer R-value: δ / k (`ConstructionLayer::r_value`).
fn s5_r_value(layer: &fluxion_core::construction::ConstructionLayer) -> f64 {
    layer.thickness / layer.conductivity
}

/// Layer capacitance per area: ρ·δ·cp
/// (`ConstructionLayer::thermal_capacitance_per_area`).
fn s5_cap_per_area(layer: &fluxion_core::construction::ConstructionLayer) -> f64 {
    layer.density * layer.thickness * layer.specific_heat
}

/// Dominant-insulation index: highest R, LAST max on ties (`Iterator::max_by`).
fn s5_dominant_ins_idx(layers: &[fluxion_core::construction::ConstructionLayer]) -> usize {
    let mut best = 0usize;
    let mut best_r = s5_r_value(&layers[0]);
    for (j, layer) in layers.iter().enumerate().skip(1) {
        let r = s5_r_value(layer);
        if r >= best_r {
            best_r = r;
            best = j;
        }
    }
    best
}

/// ISO 13790 Annex C effective κ: interior-to-insulation (inclusive) layers,
/// full ρ·c·δ each, the 100 mm active-thickness cap checked BEFORE each layer.
fn s5_kappa_eff(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    let ins = s5_dominant_ins_idx(layers);
    let mut total = 0.0;
    let mut active = 0.0;
    for (j, layer) in layers.iter().enumerate() {
        if active >= 0.10 {
            break;
        }
        if j <= ins {
            total += s5_cap_per_area(layer);
            active += layer.thickness;
        }
    }
    total
}

/// Total κ: every layer's full ρ·c·δ, left-to-right sum.
fn s5_kappa_full(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    let mut total = 0.0;
    for layer in layers {
        total += s5_cap_per_area(layer);
    }
    total
}

/// smoothstep01 as the engine writes it: clamp the linear term, then t²(3−2t).
fn s5_smoothstep01(x: f64, edge0: f64, edge1: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `massiveness_weight`: smoothstep on the FULL κ, 80k→100k edges.
fn s5_massiveness_weight(kappa_wall_full: f64) -> f64 {
    s5_smoothstep01(kappa_wall_full, 80_000.0, 100_000.0)
}

/// `wall_cap_for`: (1−w)·κ_eff·A + w·κ_full·A.
fn s5_wall_cap_for(
    layers: &[fluxion_core::construction::ConstructionLayer],
    opaque_area: f64,
) -> f64 {
    let kappa_wall = s5_kappa_eff(layers);
    let kappa_wall_full = s5_kappa_full(layers);
    let w_mass = s5_massiveness_weight(kappa_wall_full);
    let cap_low = kappa_wall * opaque_area;
    let cap_high = kappa_wall_full * opaque_area;
    (1.0 - w_mass) * cap_low + w_mass * cap_high
}

/// Interior→mass R walk: interior layers full, insulation half, no film.
fn s5_r_interior_to_mass(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    let ins = s5_dominant_ins_idx(layers);
    let mut r = 0.0;
    for (idx, layer) in layers.iter().enumerate() {
        let layer_r = s5_r_value(layer);
        if idx < ins {
            r += layer_r;
        } else if idx == ins {
            r += layer_r / 2.0;
            break;
        }
    }
    r
}

/// Exterior→mass R walk AS WRITTEN: `enumerate()` order with
/// `reverse_idx = len−1−idx`; full R while `reverse_idx > ins`, half at it,
/// break at the first layer inside. Visits `layers[0]` first — see the
/// section comment.
fn s5_r_exterior_to_mass(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    let ins = s5_dominant_ins_idx(layers);
    let n = layers.len();
    let mut r = 1.0 / 18.3; // EXTERIOR_FILM_COEFF_DEFAULT (the v2023 re-export)
    for (idx, layer) in layers.iter().enumerate() {
        let reverse_idx = n - 1 - idx;
        let layer_r = s5_r_value(layer);
        if reverse_idx > ins {
            r += layer_r;
        } else if reverse_idx == ins {
            r += layer_r / 2.0;
            break;
        } else {
            break;
        }
    }
    r
}

/// `r_value_total` association: `r_film_int + r_materials + r_exterior`, with
/// `r_materials` its own left-to-right sum (the 1-ULP-bearing shape).
fn s5_r_total(
    layers: &[fluxion_core::construction::ConstructionLayer],
    film_int: f64,
    exterior: f64,
) -> f64 {
    let r_film_int = 1.0 / film_int;
    let mut r_materials = 0.0;
    for layer in layers {
        r_materials += s5_r_value(layer);
    }
    (r_film_int + r_materials) + exterior
}

/// `u_value(None, None)`: plain interior film 8.29 + materials + exterior
/// film 18.3.
fn s5_u_value_plain(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    1.0 / s5_r_total(layers, 8.29, 1.0 / 18.3)
}

/// `u_value(Some(Floor), None)`: floor interior film 5.88 + materials +
/// ground coupling R 0.17.
fn s5_u_value_floor(layers: &[fluxion_core::construction::ConstructionLayer]) -> f64 {
    1.0 / s5_r_total(layers, 5.88, 0.17)
}

/// The 900FF spec's geometry-derived areas, composed exactly as the builder
/// reads them: exterior faces [S, W, N, E] summed left-to-right, the south
/// window from the spec, `opaque = (ext − window).max(0.0)`.
fn s5_900ff_areas(spec: &fluxion_core::ashrae_cases::CaseSpec) -> (f64, f64, f64) {
    use fluxion_core::ashrae_cases::Orientation;
    let g = &spec.geometry[0];
    let south = g.width * g.height;
    let west = g.depth * g.height;
    let north = g.width * g.height;
    let east = g.depth * g.height;
    let ext_area = ((south + west) + north) + east;
    let window = spec.window_area_by_zone_and_orientation(0, Orientation::South);
    let opaque_area = (ext_area - window).max(0.0);
    (ext_area, window, opaque_area)
}

#[test]
/// Contract S5a — the κ blend and the capacitances: the per-surface
/// capacitances the 9R4C nodes carry recompose bit-exactly from the spec's
/// own layers — wall via the `wall_cap_for` blend (weight 1.0 at
/// κ_full 123,098.4 → the full side), roof and floor via the EFFECTIVE κ the
/// builder multiplies by the floor area — and the lumped
/// `thermal_capacitance` is their three-term sum in the builder's order
/// (air cap stored separately since #1522).
fn s5_interior_kappa_blend_and_capacitances() {
    use fluxion::validation::ashrae140::cases::series_900::case_900ff;
    let spec = case_900ff();
    let m = fixture_spec_900ff();
    let s = &m.conduction.backend.multi_node_solvers[0];
    let (_ext_area, _window, opaque_area) = s5_900ff_areas(&spec);
    let floor_area = spec.geometry[0].floor_area();

    // Area seam: the recomposed opaque area IS the builder's input.
    assert_eq!(
        opaque_area,
        m.conduction.opaque_wall_area.as_ref()[0],
        "S5a premise: the recomposed opaque area (the spec geometry minus the south window)"
    );
    // Dominant-insulation seam: the middle layer (foam) on all three assemblies.
    assert_eq!(
        s5_dominant_ins_idx(&spec.construction.wall.layers),
        1,
        "S5a premise: wall insulation is the middle layer"
    );
    assert_eq!(
        s5_dominant_ins_idx(&spec.construction.roof.layers),
        1,
        "S5a premise: roof insulation is the middle layer"
    );
    assert_eq!(
        s5_dominant_ins_idx(&spec.construction.floor.layers),
        1,
        "S5a premise: floor insulation is the middle layer"
    );
    // The κ regime: the wall's full κ sits above the 100k edge.
    let kappa_wall_full = s5_kappa_full(&spec.construction.wall.layers);
    assert_eq!(
        kappa_wall_full, 123098.4,
        "S5a: the wall's full κ recomposed from the spec layers (all three ρ·δ·cp)"
    );
    assert_eq!(
        s5_massiveness_weight(kappa_wall_full),
        1.0,
        "S5a premise: the #4072 blend weight is pinned at the heavyweight asymptote"
    );

    // The blend output: wall cap = κ_full·A (w=1), roof/floor = κ_eff·floor.
    assert_eq!(
        s.mass.wall.capacitance,
        s5_wall_cap_for(&spec.construction.wall.layers, opaque_area),
        "S5a: wall cm = the wall_cap_for blend of the spec's own layers"
    );
    assert_eq!(
        s.mass.roof.capacitance,
        s5_kappa_eff(&spec.construction.roof.layers) * floor_area,
        "S5a: roof cm = the builder's EFFECTIVE-κ × floor-area form"
    );
    assert_eq!(
        s.mass.floor.capacitance,
        s5_kappa_eff(&spec.construction.floor.layers) * floor_area,
        "S5a: floor cm = the builder's EFFECTIVE-κ × floor-area form"
    );
    assert_eq!(
        m.mass.thermal_capacitance.as_ref()[0],
        s5_wall_cap_for(&spec.construction.wall.layers, opaque_area)
            + s5_kappa_eff(&spec.construction.roof.layers) * floor_area
            + s5_kappa_eff(&spec.construction.floor.layers) * floor_area,
        "S5a: the lumped C_m is the builder's wall+roof+floor sum in that order"
    );
}

#[test]
/// Contract S5b — the #2229 h_ms calibration and the A_m blend: the per-class
/// h_tr_ms values are the half-insulation walks (interior layers full,
/// insulation half, 0.001 floor guard) over the spec's own layers, and the
/// lumped h_tr_ms is `h_ms_of_kappa(κ_wall_full)·A_m` with A_m on the
/// simplified-Table-C.2 side of the blend (w_am 0 at κ_eff 5,498.4,
/// A_m = 2.5·A_floor = 120).
fn s5_interior_h_ms_calibration_and_am_blend() {
    use fluxion::validation::ashrae140::cases::series_900::case_900ff;
    let spec = case_900ff();
    let m = fixture_spec_900ff();
    let s = &m.conduction.backend.multi_node_solvers[0];
    let (_ext_area, _window, opaque_area) = s5_900ff_areas(&spec);
    let floor_area = spec.geometry[0].floor_area();

    // Per-class walks:
    assert_eq!(
        s.mass.wall.h_tr_ms,
        opaque_area / s5_r_interior_to_mass(&spec.construction.wall.layers).max(0.001),
        "S5b: wall h_tr_ms = the half-insulation interior walk over the spec layers"
    );
    assert_eq!(
        s.mass.roof.h_tr_ms,
        floor_area / s5_r_interior_to_mass(&spec.construction.roof.layers).max(0.001),
        "S5b: roof h_tr_ms = the half-insulation interior walk (the Phase 34-02 roof term)"
    );
    assert_eq!(
        s.mass.floor.h_tr_ms,
        floor_area / s5_r_interior_to_mass(&spec.construction.floor.layers).max(0.001),
        "S5b: floor h_tr_ms = the half-insulation interior walk (the Phase 34-02 floor term)"
    );

    // The calibration + A_m blend, operation-for-operation:
    let kappa_wall_eff = s5_kappa_eff(&spec.construction.wall.layers);
    let kappa_roof_eff = s5_kappa_eff(&spec.construction.roof.layers);
    let kappa_floor_eff = s5_kappa_eff(&spec.construction.floor.layers);
    let kappa_wall_full = s5_kappa_full(&spec.construction.wall.layers);
    let a_kappa_sum =
        (opaque_area * kappa_wall_eff) + (floor_area * (kappa_roof_eff + kappa_floor_eff));
    let a_kappa_sq_sum = (opaque_area * kappa_wall_eff * kappa_wall_eff)
        + (floor_area * (kappa_roof_eff * kappa_roof_eff + kappa_floor_eff * kappa_floor_eff));
    let a_m_table = 2.5 * floor_area;
    let a_m_weighted = if a_kappa_sq_sum > 0.0 {
        (a_kappa_sum * a_kappa_sum) / a_kappa_sq_sum
    } else {
        a_m_table
    };
    let w_am = s5_smoothstep01(kappa_wall_eff, 165_000.0, 260_000.0);
    let a_m = (1.0 - w_am) * a_m_table + w_am * a_m_weighted;
    let h_ms_coeff = 2.0 + (13.4 - 2.0) * s5_massiveness_weight(kappa_wall_full);
    assert_eq!(
        h_ms_coeff, 13.4,
        "S5b premise: the #2229 calibration sits at its heavyweight endpoint"
    );
    assert_eq!(
        w_am, 0.0,
        "S5b premise: the A_m blend sits on the simplified side (κ_eff < 165k)"
    );
    assert_eq!(
        m.conduction.h_tr_ms.as_ref()[0],
        h_ms_coeff * a_m,
        "S5b: the lumped h_tr_ms = h_ms_of_kappa(κ_full)·A_m recomposed (Issue #821 Probe A)"
    );
}

#[test]
/// Contract S5c — the h_tr_em walks and the floor U forms: the per-class
/// exterior walks ride layers[0] full + insulation half over the 18.3 film
/// (the as-written reversed-index convention — see the section comment), the
/// floor's exterior coupling goes through the builder's own
/// `u_value(None, None)` reciprocal, the setpoints floor U carries the
/// floor-film/ground-resistance variant (5.88 + 0.17), and the lumped
/// h_tr_em is the Issue #831 Eq. 64 series remainder 1/(1/h_op − 1/h_ms).
fn s5_interior_h_em_walks_and_floor_u_forms() {
    use fluxion::validation::ashrae140::cases::series_900::case_900ff;
    let spec = case_900ff();
    let m = fixture_spec_900ff();
    let s = &m.conduction.backend.multi_node_solvers[0];
    let (_ext_area, _window, opaque_area) = s5_900ff_areas(&spec);
    let opaque_area_ext = opaque_area; // single zone, no partitions: same arithmetic
    let floor_area = spec.geometry[0].floor_area();

    assert_eq!(
        s.mass.wall.h_tr_em,
        opaque_area_ext / s5_r_exterior_to_mass(&spec.construction.wall.layers),
        "S5c: wall h_tr_em = the as-written exterior walk over the spec layers"
    );
    assert_eq!(
        s.mass.roof.h_tr_em,
        floor_area / s5_r_exterior_to_mass(&spec.construction.roof.layers),
        "S5c: roof h_tr_em = the as-written exterior walk"
    );
    // Floor: the builder divides by the reciprocal of the plain-film U:
    // h = A / (1 / u_plain).
    assert_eq!(
        s.mass.floor.h_tr_em,
        floor_area / (1.0 / s5_u_value_plain(&spec.construction.floor.layers)),
        "S5c: floor h_tr_em = floor_area / (1 / u_value(None, None)) — the reciprocal form"
    );
    // The setpoints floor U: film 5.88 + materials + ground 0.17, and the
    // h_tr_floor that consumes it:
    assert_eq!(
        m.setpoints.floor_u_value,
        s5_u_value_floor(&spec.construction.floor.layers),
        "S5c: setpoints floor U = u_value(Some(Floor), None) recomposed (Issue #588 ground coupling)"
    );
    assert_eq!(
        m.conduction.h_tr_floor.as_ref()[0],
        s5_u_value_floor(&spec.construction.floor.layers) * floor_area,
        "S5c: h_tr_floor = the setpoints floor U × floor area"
    );
    // The lumped h_tr_em: Eq. 64 series remainder off the plain-film U pair,
    // with h_ms_iso recomposed IN-TEST (not engine-read — teeth hygiene):
    let h_op = (s5_u_value_plain(&spec.construction.wall.layers) * opaque_area_ext)
        + (s5_u_value_plain(&spec.construction.roof.layers) * floor_area);
    let h_ms_iso = {
        let kw = s5_kappa_eff(&spec.construction.wall.layers);
        let kr = s5_kappa_eff(&spec.construction.roof.layers);
        let kf = s5_kappa_eff(&spec.construction.floor.layers);
        let a_m = {
            let a_kappa_sum = (opaque_area * kw) + (floor_area * (kr + kf));
            let a_kappa_sq_sum = (opaque_area * kw * kw) + (floor_area * (kr * kr + kf * kf));
            let table = 2.5 * floor_area;
            let weighted = if a_kappa_sq_sum > 0.0 {
                (a_kappa_sum * a_kappa_sum) / a_kappa_sq_sum
            } else {
                table
            };
            let w = s5_smoothstep01(kw, 165_000.0, 260_000.0);
            (1.0 - w) * table + w * weighted
        };
        (2.0 + (13.4 - 2.0) * s5_massiveness_weight(s5_kappa_full(&spec.construction.wall.layers)))
            * a_m
    };
    let h_em = if h_op > 0.0 && h_op < h_ms_iso {
        1.0 / (1.0 / h_op - 1.0 / h_ms_iso)
    } else {
        0.0 // unreachable at this spec; the branch shape is transcribed verbatim
    };
    assert_eq!(
        m.conduction.h_tr_em.as_ref()[0],
        h_em.max(0.1),
        "S5c: the lumped h_tr_em = the Issue #831 Eq. 64 series remainder (walls+roof h_op)"
    );
    // The internal node: furniture-factor derivation (the CaseBuilder default
    // building type is Residential → 0.3), with the SOLVER node's own h_tr_me
    // left at its default 0.0 — the 131.04 lands on the MODEL's mass vector
    // only after the solvers are built (the B2/S4 stale-early-read seam).
    assert_eq!(
        spec.building_type,
        fluxion_core::ashrae_cases::BuildingType::Residential,
        "S5c premise: the CaseBuilder default building type"
    );
    let a_int = 0.3 * floor_area;
    assert_eq!(
        m.mass.h_tr_me.as_ref()[0],
        9.1 * a_int,
        "S5c: the model-level h_tr_me = 9.1·(f_furniture·floor_area) (Issue #1213 calibration)"
    );
    assert_eq!(
        s.mass.internal.capacitance,
        floor_area * 55_000.0 * 0.3,
        "S5c: the internal node cm = A_floor·55,000·f_furniture (Issue #1 formula)"
    );
    assert_eq!(
        s.mass.internal.h_tr_me, 0.0,
        "S5c premise: the SOLVER's internal h_tr_me is the default 0.0 (read before the model vector is filled)"
    );
}
// ============================================================================
// Round 141 — the Case-950FF night-vent spec-path arm (the r135 Case-900FF
// template applied to the vent case, the pin backlog's second item).
//
// The r100-era brief said the night-vent route multiplied h_tr_is; round
// 100/#4612 removed that multiplier. This arm pins WHAT IS at tip
// `e1a54a11`, verified by reading the 9R4C route before authoring
// (pin-what-IS, not the round-100-era brief). The 9R4C step carries its OWN
// vent route (`physics_impl/step_9r4c.rs`), documented there as the
// Round-100 vent-slice contract:
//
// - ASHRAE 140-2017 §5.2.1.10.1 (normative): operating the ideal mechanical
//   system shall not affect the interior-surface convective flow regime —
//   the fan couples to the zone AIR node only, and **h_tr_is keeps its
//   built value during vent hours**; the former #1279/#2871 forced-
//   convection multiplier was removed (contract V2 pins the built 165.6
//   surviving a vent-hour step verbatim);
// - the vent conductance is `h_ve_night = fan_capacity·ρ·cp/3600` with ρ/cp
//   read from the builder-set per-zone `air_density`/`heat_capacity` (1.2 /
//   1005.0 for the 900-series specs) — 1400·1.2·1005/3600 = 469.0 W/K
//   exactly at the Case 650 fan (1400 m³/h, ASHRAE 140 Table 5-160
//   sea-level route), gated by `is_active_at_hour(timestep % 24)` on the
//   (18:00 → 07:00 wrap) window, zone 0 only;
// - the LIVE default-route effect is Issue #1712's zone-0 h_ext add
//   (`h_ext_for_free_float[0] = derived_h_ext[0] + h_ve_night`, the den
//   rebuilt from it while active) — and it vanishes exactly when the fan
//   is zero: the vent-active advance with `fan_capacity = 0.0` is
//   BIT-IDENTICAL to the inactive-hour advance (contract V3 below collapses
//   the twins on every committed observable; the `+0.0` add and the den
//   rebuild are value-exact no-ops);
// - the 5R1C-side retired route (`FLX_R86_VENT_AIR_ONLY`, default true at
//   both of its sites) and the 5R1C `h_tr_is_for_ti_free` plain clone sit on
//   the 5R1C path, structurally OFF this route — a default flip there moved
//   nothing on the 9R4C arm (tooth measured all-green, prediction missed,
//   disclosed in the duck review); the r112 multi-zone vent suite owns the
//   5R1C sites.
// - the WINDOW-01 spec U override (3.0 W/m²K, `ASHRAE140_WINDOW_U_VALUE`)
//   REPLACES the #2889 frame bridge on the override route: h_tr_w =
//   12·3.0 = 36.0 exactly — contrast with the 900FF arm's composed bridge
//   (12·(5.8+0.1+0.05·16/12)).
//
// The derivation-interior forms are the 900FF arm's (same assemblies — the
// S5 remainder closes those separately); the carryover literals re-pinned
// here are witnesses that the 950FF spec derives identically.
//
// Provenance: temporary-probe capture (removed before commit, the r135
// practice) at tip `e1a54a11` — three twins (fan 1400 @ hour 20/21, fan
// 1400 @ hour 12, fan 0 @ hour 20), all printed observables transcribed
// verbatim with `{:?}` shortest-round-trip literals. The inactive-hour twin
// reproduces the 900FF arm's step-1 committed states bit-for-bit (identical
// caches + absent vent term). Tests-only: no src delta, no re-record owed.
// ============================================================================

/// The 950FF spec-path fixture with the vent fan capacity under test: the
/// production route (`from_spec(&case_950ff())`), the contract's solar
/// driving set as in every fixture above, and the fan capacity the caller
/// pins (1400.0 production, 0.0 the collapse twin).
fn fixture_spec_950ff_with_fan(fan_capacity: f64) -> ThermalModel<VectorField> {
    use fluxion::validation::ashrae140::cases::series_900::case_950ff;
    let mut spec = case_950ff();
    if let Some(night_vent) = spec.night_ventilation.as_mut() {
        night_vent.fan_capacity = fan_capacity;
    }
    let mut m = ThermalModel::<VectorField>::from_spec(&spec);
    m.solar.solar_gains = VectorField::new(vec![400.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0]);
    m
}

fn fixture_spec_950ff() -> ThermalModel<VectorField> {
    fixture_spec_950ff_with_fan(1400.0)
}

#[test]
/// Contract V1 — the 950FF spec-path premises: the vent spec (fan 1400,
/// hours 18→07 wrap, no added heat, the hour-gate truth table on the engine's
/// own `is_active_at_hour`), the 9R4C dispatch / coupling / free-float /
/// inert-HVAC frame, the r91 spec partition with zero loads, the builder-set
/// ρ/cp the vent conductance consumes, the WINDOW-01 override REPLACING the
/// frame bridge (h_tr_w = 12·3.0 = 36.0 exactly), and the derivation
/// carryover witnesses (the 900FF-arm conductance set, bit-identical — same
/// assemblies).
fn spec_path_950ff_premises_and_window_override() {
    let m = fixture_spec_950ff();
    let spec_window_override = 3.0_f64; // ASHRAE140_WINDOW_U_VALUE
    assert_eq!(
        m.hvac.thermal_selector.zone_solver,
        ZoneSolverKind::NineRFourC
    );
    assert_eq!(
        m.conduction.backend.multi_node_solvers[0].coupling_mode,
        MassAirCouplingMode::ParallelResistance
    );
    assert!(m.hvac.free_float, "premise: 950FF is free-floating");
    // The vent spec, read from the model the builder built:
    let nv = m
        .hvac
        .night_ventilation
        .as_ref()
        .expect("premise: 950FF carries night ventilation (the 900FF arm asserts None)");
    assert_eq!(
        nv.fan_capacity, 1400.0,
        "ASHRAE 140 Table 5-160 sea-level fan"
    );
    assert_eq!(
        nv.operating_hours,
        (18, 7),
        "18:00 → 07:00, wrapping midnight"
    );
    assert!(!nv.adds_heat, "the fan moves outdoor air, no heat added");
    // The hour gate on the engine's own method (timestep % 24):
    for (hour, active) in [
        (0u8, true),
        (6, true),
        (7, false),
        (12, false),
        (17, false),
        (18, true),
        (23, true),
    ] {
        assert_eq!(
            nv.is_active_at_hour(hour),
            active,
            "hour-gate truth at hour {hour}"
        );
    }
    // Inert HVAC + zero loads + ground, as at 900FF:
    assert_eq!(m.setpoints.heating_setpoint, -999.0);
    assert_eq!(m.setpoints.cooling_setpoint, 999.0);
    assert_eq!(m.setpoints.loads.as_ref()[0], 0.0);
    assert_eq!(
        m.conduction.ground_temperature.ground_temperature(0),
        9.4,
        "Annex B §B3.3 ground temperature"
    );
    assert_eq!(m.solar.convective_fraction, 0.4);
    assert_eq!(m.solar.solar_distribution_to_air, 0.0, "r91 spec partition");
    assert_eq!(
        m.solar.solar_beam_to_mass_fraction, 0.0,
        "r91 spec partition"
    );
    // The ρ/cp the vent route consumes (builder-set per-zone vectors):
    assert_eq!(m.setpoints.air_density.as_ref()[0], 1.2);
    assert_eq!(m.setpoints.heat_capacity.as_ref()[0], 1005.0);
    // The vent conductance, recomposed in-test (engine truth: 469.0 exactly):
    assert_eq!(
        1400.0 * 1.2 * 1005.0 / 3600.0,
        469.0,
        "h_ve_night = fan_capacity·ρ·cp/3600 at the Case-650 fan"
    );
    // WINDOW-01 override replaces the #2889 bridge on the override route:
    assert_eq!(
        m.conduction.h_tr_w.as_ref()[0],
        12.0 * spec_window_override,
        "h_tr_w = area·override — NO bridge composition on the override route"
    );
    // Derivation carryover witnesses (same assemblies as 900FF):
    assert_eq!(m.conduction.h_ve.as_ref()[0], 17.974224);
    assert_eq!(m.conduction.h_tr_is.as_ref()[0], 165.60000000000002);
    assert_eq!(
        m.conduction.backend.multi_node_solvers[0].h_tr_is,
        m.conduction.h_tr_is.as_ref()[0],
        "the collapsed h_tr_is seam carries to 950FF"
    );
    assert_eq!(m.setpoints.floor_u_value, 0.18109607694010674);
    assert_eq!(m.conduction.h_tr_floor.as_ref()[0], 8.692611693125123);
    assert_eq!(m.conduction.h_tr_ms.as_ref()[0], 1608.0);
    assert_eq!(m.conduction.h_tr_em.as_ref()[0], 48.880343578250944);
    assert_eq!(m.mass.thermal_capacitance.as_ref()[0], 18014427.840000004);
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(s.mass.wall.capacitance, 7829058.24);
    assert_eq!(s.mass.wall.h_tr_ms, 76.34726688102896);
    assert_eq!(s.mass.wall.h_tr_em, 71.64739830382713);
    assert_eq!(s.mass.roof.h_tr_ms, 32.91511795494197);
    assert_eq!(s.mass.floor.h_tr_ms, 17.981819851253327);
    assert_eq!(s.mass.internal.capacitance, 792000.0);
}

#[test]
/// Contract V2 — the vent-ACTIVE committed route: two consecutive vent-hour
/// steps (hours 20 → 21, T_OUT −10 °C, the contract's solar driving set).
/// Every committed observable is an engine-read literal from the probe; the
/// free-float commits stay exactly 0.0 kWh; and the active-hour advance is
/// NON-INERT — the step-1 zone temperature sits far from the inactive-hour
/// value (the fan pulls the air toward outdoor), the discriminator that the
/// route is live on the default build.
fn spec_path_950ff_night_vent_active_hour_route() {
    let mut m = fixture_spec_950ff();
    let kwh1 = m.step_physics(20, T_OUT, DT);
    assert_eq!(
        kwh1, 0.0,
        "free-float commits exactly 0.0 kWh at the vent hour"
    );
    // The round-100 removal, pinned on the route: h_tr_is keeps its BUILT
    // value through a vent-hour step — the former #1279/#2871 multiplier is
    // gone (§5.2.1.10.1: the fan must not affect the surface flow regime).
    assert_eq!(
        m.conduction.h_tr_is.as_ref()[0],
        165.60000000000002,
        "h_tr_is keeps its built value during vent hours (r100/#4612)"
    );
    {
        let s = &m.conduction.backend.multi_node_solvers[0];
        assert_eq!(s.zone_temperature, -5.062665818997672);
        assert_eq!(s.surface_temperature, 6.800941271931854);
        assert_eq!(s.mass.wall.temperature, 24.090456649161467);
        assert_eq!(s.mass.roof.temperature, 22.487405434691762);
        assert_eq!(s.mass.floor.temperature, 21.912443437794053);
        assert_eq!(s.mass.internal.temperature, 9.236572978148303);
    }
    assert_eq!(m.mass.mass_temperatures.as_ref()[0], 20.251083026984688);
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        -4.525585616481499,
        "the committed zone air at the vent-active hour"
    );
    assert_ne!(
        m.setpoints.temperatures.as_ref()[0],
        18.344345521157422,
        "the vent-active air must NOT equal the inactive-hour value — the route is live"
    );
    // Step 2 rides the step-1 committed state, still inside the vent window:
    let kwh2 = m.step_physics(21, T_OUT, DT);
    assert_eq!(kwh2, 0.0);
    {
        let s = &m.conduction.backend.multi_node_solvers[0];
        assert_eq!(s.zone_temperature, -4.525585616481499);
        assert_eq!(s.surface_temperature, 7.219813856698825);
        assert_eq!(s.mass.wall.temperature, 27.974796410976378);
        assert_eq!(s.mass.roof.temperature, 24.89133968809225);
        assert_eq!(s.mass.floor.temperature, 23.795315323152938);
        assert_eq!(s.mass.internal.temperature, 3.326268353483637);
    }
    assert_eq!(m.mass.mass_temperatures.as_ref()[0], 20.499418743655955);
    assert_eq!(m.setpoints.temperatures.as_ref()[0], -4.01173781324285);
}

#[test]
/// Contract V3 — the collapse twins: the vent-INACTIVE hour (12) pins the
/// no-vent advance — bit-identical to the 900FF arm's step-1 committed
/// states (identical caches, absent vent term: the same arithmetic) — and
/// the fan-0 twin at a VENT-ACTIVE hour (night_vent_active_now = true, the
/// den rebuilt with +0.0 and the numerator carrying 0.0·T_out) collapses
/// onto those same literals on every committed observable. The vent route's
/// entire default-build effect is the h_ve_night term; zeroing the fan
/// erases it exactly.
fn spec_path_950ff_vent_collapse_twins() {
    // Twin B — the inactive hour:
    let mut b = fixture_spec_950ff();
    let kwh_b = b.step_physics(12, T_OUT, DT);
    assert_eq!(kwh_b, 0.0);
    {
        let s = &b.conduction.backend.multi_node_solvers[0];
        assert_eq!(
            s.zone_temperature, 15.266117931266905,
            "the 900FF arm's step-1 zone pin — the vent-inactive arithmetic is identical"
        );
        assert_eq!(s.surface_temperature, 6.45882803741936);
        assert_eq!(s.mass.wall.temperature, 24.547802980996828);
        assert_eq!(s.mass.roof.temperature, 22.8391289109059);
        assert_eq!(s.mass.floor.temperature, 22.159651283235362);
        assert_eq!(s.mass.internal.temperature, 17.966984256788898);
    }
    assert_eq!(b.mass.mass_temperatures.as_ref()[0], 20.269011527775067);
    assert_eq!(b.setpoints.temperatures.as_ref()[0], 18.344345521157422);
    // Twin C — fan_capacity 0.0 at the VENT-ACTIVE hour 20: night_vent is
    // `active_now` (the gate fires), but the conductance is exactly 0.0, so
    // `h_ext_for_free_float[0] = derived_h_ext[0] + 0.0` and the den rebuilt
    // from it are value-exact no-ops. Every committed observable collapses
    // onto twin B:
    let mut c = fixture_spec_950ff_with_fan(0.0);
    let kwh_c = c.step_physics(20, T_OUT, DT);
    assert_eq!(kwh_c, 0.0);
    {
        let s = &c.conduction.backend.multi_node_solvers[0];
        assert_eq!(
            s.zone_temperature, 15.266117931266905,
            "fan-0 collapse: the active-hour gate fires but 0.0·T_out and +0.0 are exact no-ops"
        );
        assert_eq!(s.surface_temperature, 6.45882803741936);
        assert_eq!(s.mass.wall.temperature, 24.547802980996828);
        assert_eq!(s.mass.roof.temperature, 22.8391289109059);
        assert_eq!(s.mass.floor.temperature, 22.159651283235362);
        assert_eq!(s.mass.internal.temperature, 17.966984256788898);
    }
    assert_eq!(c.mass.mass_temperatures.as_ref()[0], 20.269011527775067);
    assert_eq!(c.setpoints.temperatures.as_ref()[0], 18.344345521157422);
}

// ===== R142: the HVAC-mode rejection-path rollback family =====
//
// The r136 step-level transaction (#4795) snapshot/restore covers the
// solver nodes, the lumped-mass pair and the predictive controller's
// previous zone temperature, and its contracts (A7/A8/A8b/A8c above)
// close the r134 additive-vs-PR contrast as history. Two seams remained
// outside that rollback, both pinned here red-first at the r142 tip:
//
//   1. the #4500 demand-overflow rejection path restored NOTHING — the
//      solver loop, the lumped-mass commit and the controller advance
//      all persisted through the rejection while the guard's own log
//      line claimed "state untouched" (its R136 hoist comment only
//      justified not re-restoring BETWEEN the #4495 t_i_act check and
//      this guard; everything before that check was never restored on
//      this path);
//   2. the equipment part-load ratio mutated inside the equipment
//      dispatch block (`update_state`) BEFORE every guard, outside the
//      r136 snapshot scope — the r136 round documented exactly this
//      state as its stated bound ("HVAC equipment PLR state isn't
//      rolled back (unreachable in the fixtures and under physical
//      inputs)"). The bound closed in r142: under the guard family's
//      own invalid-input domain the PLR advance is fixture-reachable,
//      and the verdict semantics ("a rejected step advances nothing,
//      period") put it inside the snapshot.

use fluxion::sim::hvac::{AnyEquipment, Boiler, VariableCapacityEquipment};

/// The r134-suite free-float fixture flipped to HVAC mode (`free_float =
/// false`, per-zone setpoint vectors raised to 30/26 so the ideal-loads
/// demand fires and the predictive controller leaves Off exactly once —
/// the PLR-observable window; the demand branch reads the per-zone
/// `heating_setpoints` VECTOR, not the scalar, so the scalar fallback at
/// 22.0 is inert here). Same solver construction as `fixture_9r4c`:
/// selector NineRFourC, ParallelResistance production default (#1859).
fn fixture_9r4c_hvac(equipment: bool) -> ThermalModel<VectorField> {
    let mut m = ThermalModel::<VectorField>::new(1);
    m.setpoints.zone_area = VectorField::from_scalar(48.0, 1);
    m.setpoints.zone_volume = VectorField::from_scalar(129.6, 1);
    m.setpoints.heating_setpoint = 22.0;
    m.setpoints.cooling_setpoint = 26.0;
    m.setpoints.heating_setpoints = VectorField::from_scalar(30.0, 1);
    m.setpoints.cooling_setpoints = VectorField::from_scalar(26.0, 1);
    m.hvac.predictive_controller.heating_setpoint = 30.0;
    m.hvac.predictive_controller.cooling_setpoint = 26.0;
    m.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.air_density = VectorField::from_scalar(1.2, 1);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 1);
    m.hvac.free_float = false;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0]);
    m.conduction.h_ve = VectorField::new(vec![1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0)]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0]);
    m.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0]);
    m.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.loads = VectorField::from_scalar(0.0, 1);
    m.solar.solar_gains = VectorField::new(vec![0.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0]);
    m.hvac.hvac_heating_capacity = 5000.0;
    m.hvac.hvac_cooling_capacity = 5000.0;
    m.hvac.thermal_selector.zone_solver = ZoneSolverKind::NineRFourC;
    m.conduction.ground_temperature = Box::new(ConstantGroundTemperature::new(T_G));
    if equipment {
        m.hvac.hvac_equipment = Some(AnyEquipment::Boiler(Boiler::new(
            "r142-rollback-boiler".to_string(),
            5000.0,
            0.9,
            0.0,
        )));
    }
    m.update_optimization_cache();
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

#[test]
/// Contract H1 — the #4500 rejection advances NOTHING (HVAC mode). An
/// extreme-but-finite `dt` — the guard's own stated envelope ("the ideal
/// loads can overflow to ±inf for extreme-but-finite `dt` even though
/// every input passed the finite checks and the #4495 temperature guards
/// passed") — makes `w_bound * dt` overflow to inf, so the step is
/// rejected with 0.0 kWh AFTER the solver loop ran and the lumped mass
/// committed. [R142 rollback] every snapshot-scope durable field
/// restores to its step-1 value (engine-probe literals). At the pre-fix
/// tip this step left the mass committed at 13.130030240901489, the
/// solver nodes at 15.531914911447066 / 16.66666667344683 /
/// 26.59999805605751 / 25.469752620844055, and the solver's derived
/// surface temperature overflowed to -inf — all persisting through the
/// rejection under a log line that claimed "state untouched".
fn rejected_dt_overflow_advances_nothing_hvac() {
    let mut m = fixture_9r4c_hvac(false);
    m.step_physics(0, T_OUT, DT);
    m.step_physics(1, T_OUT, DT);
    let kwh = m.step_physics(2, T_OUT, 1.0e306);
    assert_eq!(kwh, 0.0, "the #4500-rejected step returns exactly 0.0 kWh");
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        19.586766041249877,
        "the lumped-mass commit rolls back (pre-fix it stayed at the committed advance)"
    );
    assert_eq!(
        m.mass.previous_mass_temperatures.as_ref()[0],
        19.79244954670227,
        "the mass pair rolls back verbatim (pre-fix the commit had shifted it)"
    );
    let s = &m.conduction.backend.multi_node_solvers[0];
    assert_eq!(
        s.mass.wall.temperature, 17.94423970468105,
        "the solver wall node rolls back (pre-fix it stayed at the dt-poisoned advance)"
    );
    assert_eq!(s.mass.roof.temperature, 21.86174829383185);
    assert_eq!(s.mass.floor.temperature, 20.030989666964107);
    assert_eq!(s.mass.internal.temperature, 20.391173887693007);
    assert_eq!(s.zone_temperature, 30.0);
    assert_eq!(
        s.surface_temperature, 14.468034357149172,
        "the solver's derived surface temperature rolls back (pre-fix it overflowed to -inf)"
    );
    assert!(
        s.surface_temperature.is_finite(),
        "the restored solver state is finite"
    );
    assert_eq!(
        m.hvac.predictive_controller.previous_zone_temp, 30.0,
        "the controller's previous zone temperature rolls back"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        30.0,
        "the committed zone air is untouched (it was written by step 1, not step 2)"
    );
}

#[test]
/// Contract H2 — the #4500 rollback's RECOVERY arm (HVAC mode): a
/// demand-overflow-rejected step leaves no trace, so the model resumes
/// exactly as if the bad step never happened. Model A runs steps 0, 1,
/// the dt=1e306 rejection at 2, then 3; model C skips the bad step
/// (0, 1, 3). Both compose step 3 from the identical post-step-1 state,
/// so every durable field must agree bit-exactly. [Pre-fix, A carried
/// step-2's committed mass and dt-poisoned solver nodes through: mass
/// 12.984276208837787 vs the twin's 19.382932690163447, wall
/// 15.531914911358891 vs 17.932309998359347, step-3 energy
/// 1.8391507964573286 vs 1.860206... kWh — the zone kept integrating
/// from poisoned state.]
fn rejected_dt_overflow_leaves_no_trace_hvac() {
    let mut a = fixture_9r4c_hvac(false);
    a.step_physics(0, T_OUT, DT);
    a.step_physics(1, T_OUT, DT);
    assert_eq!(
        a.step_physics(2, T_OUT, 1.0e306),
        0.0,
        "the rejected step returns exactly 0.0 kWh"
    );
    let kwh_a = a.step_physics(3, T_OUT, DT);
    let mut c = fixture_9r4c_hvac(false);
    c.step_physics(0, T_OUT, DT);
    c.step_physics(1, T_OUT, DT);
    let kwh_c = c.step_physics(3, T_OUT, DT);
    // Twin literals (fix-independent: C never rejects).
    assert_eq!(c.mass.mass_temperatures.as_ref()[0], 19.382932690163447);
    assert_eq!(
        c.mass.previous_mass_temperatures.as_ref()[0],
        19.586766041249877
    );
    // Every durable field bit-exact.
    assert_eq!(kwh_a, kwh_c, "step-3 energy composes identically");
    assert_eq!(
        a.mass.mass_temperatures.as_ref()[0],
        c.mass.mass_temperatures.as_ref()[0],
        "the recovered lumped mass == the never-poisoned twin's"
    );
    assert_eq!(
        a.mass.previous_mass_temperatures.as_ref()[0],
        c.mass.previous_mass_temperatures.as_ref()[0]
    );
    let (sa, sc) = (
        &a.conduction.backend.multi_node_solvers[0],
        &c.conduction.backend.multi_node_solvers[0],
    );
    assert_eq!(sa.mass.wall.temperature, sc.mass.wall.temperature);
    assert_eq!(sa.mass.roof.temperature, sc.mass.roof.temperature);
    assert_eq!(sa.mass.floor.temperature, sc.mass.floor.temperature);
    assert_eq!(sa.mass.internal.temperature, sc.mass.internal.temperature);
    assert_eq!(sa.zone_temperature, sc.zone_temperature);
    assert_eq!(sa.surface_temperature, sc.surface_temperature);
    assert_eq!(
        a.hvac.predictive_controller.previous_zone_temp,
        c.hvac.predictive_controller.previous_zone_temp
    );
    assert_eq!(
        a.setpoints.temperatures.as_ref()[0],
        c.setpoints.temperatures.as_ref()[0]
    );
}

#[test]
/// Contract H3 — the equipment PLR restores through a #4495 NaN-input
/// rejection (HVAC mode with a Boiler). The equipment dispatch runs
/// `update_state` BEFORE every guard; a NaN outdoor temperature drives
/// the multi-node air balance non-finite (deadband comparison q = 0.0,
/// t_act = the NaN free temperature), the #4495 mass-node guard rejects
/// with the r136 rollback restoring the thermal fields — and the
/// Off-mode `update_state` PLR zeroing (0.37575697542907543 -> 0.0)
/// persisted through the rejection at the pre-fix tip. That is exactly
/// the state r136 documented as its out-of-scope bound ("HVAC equipment
/// PLR state isn't rolled back (unreachable in the fixtures and under
/// physical inputs)"); this fixture reaches it under the guard family's
/// own invalid-input domain, and the verdict semantics ("a rejected
/// step advances nothing, period") put the PLR inside the snapshot.
fn rejected_nan_input_restores_equipment_plr() {
    let mut m = fixture_9r4c_hvac(true);
    m.step_physics(0, T_OUT, DT);
    // Step 0 fired the Boiler (Heating; the committed air sits at the
    // 30 °C setpoint — the engine-probe literal below).
    assert_eq!(
        m.hvac.hvac_equipment.as_ref().map(|e| e.current_plr()),
        Some(0.37575697542907543),
        "step 0's Heating dispatch fired the Boiler (premise)"
    );
    let kwh = m.step_physics(1, f64::NAN, DT);
    assert_eq!(kwh, 0.0, "the #4495-rejected step returns exactly 0.0 kWh");
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        19.79244954670227,
        "the #4495 thermal rollback (green since r136)"
    );
    assert_eq!(
        m.hvac.hvac_equipment.as_ref().map(|e| e.current_plr()),
        Some(0.37575697542907543),
        "the fired PLR restores — pre-fix it read 0.0 through the rejection"
    );
}

#[test]
/// Contract H4 — the equipment recovery twin: a NaN-rejected step
/// leaves no trace in the thermal state AND the equipment state, so the
/// model resumes exactly as if it never happened. [Green at the tip by
/// construction: the #4495 rollback already restores the thermal fields,
/// and the next accepted step's own Off-mode dispatch overwrites the
/// PLR — this twin is the standing union guarantee, not a red-first
/// witness; H3 carries the red.]
fn rejected_nan_step_leaves_no_trace_with_equipment() {
    let mut a = fixture_9r4c_hvac(true);
    a.step_physics(0, T_OUT, DT);
    assert_eq!(
        a.step_physics(1, f64::NAN, DT),
        0.0,
        "the rejected step returns exactly 0.0 kWh"
    );
    a.step_physics(2, T_OUT, DT);
    let mut c = fixture_9r4c_hvac(true);
    c.step_physics(0, T_OUT, DT);
    c.step_physics(2, T_OUT, DT);
    // Twin literals (fix-independent: C never rejects).
    assert_eq!(c.mass.mass_temperatures.as_ref()[0], 19.586766041249877);
    assert_eq!(
        c.mass.previous_mass_temperatures.as_ref()[0],
        19.79244954670227
    );
    // Every durable field bit-exact — thermal and equipment.
    assert_eq!(
        a.mass.mass_temperatures.as_ref()[0],
        c.mass.mass_temperatures.as_ref()[0]
    );
    assert_eq!(
        a.mass.previous_mass_temperatures.as_ref()[0],
        c.mass.previous_mass_temperatures.as_ref()[0]
    );
    let (sa, sc) = (
        &a.conduction.backend.multi_node_solvers[0],
        &c.conduction.backend.multi_node_solvers[0],
    );
    assert_eq!(sa.mass.wall.temperature, sc.mass.wall.temperature);
    assert_eq!(sa.mass.roof.temperature, sc.mass.roof.temperature);
    assert_eq!(sa.mass.floor.temperature, sc.mass.floor.temperature);
    assert_eq!(sa.mass.internal.temperature, sc.mass.internal.temperature);
    assert_eq!(sa.zone_temperature, sc.zone_temperature);
    assert_eq!(sa.surface_temperature, sc.surface_temperature);
    assert_eq!(
        a.hvac.predictive_controller.previous_zone_temp,
        c.hvac.predictive_controller.previous_zone_temp
    );
    assert_eq!(
        a.setpoints.temperatures.as_ref()[0],
        c.setpoints.temperatures.as_ref()[0]
    );
    assert_eq!(
        a.hvac.hvac_equipment.as_ref().map(|e| e.current_plr()),
        c.hvac.hvac_equipment.as_ref().map(|e| e.current_plr()),
        "the recovered equipment PLR == the never-poisoned twin's"
    );
}

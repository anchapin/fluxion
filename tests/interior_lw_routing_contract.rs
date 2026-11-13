//! Loop round 138 — the #2890-followup InteriorSurfaceNetwork routing: the
//! inter-pool longwave exchange enters the production per-class `t_s` advance.
//!
//! Component under contract: the D5 t_s-first per-class surface advance
//! (`thermal_model_physics/zone_balance.rs`, the `use_r67` class loop — the
//! only per-class t_s advance in the engine) now adds the pairwise
//! Stefan–Boltzmann net terms of `InteriorSurfaceNetwork`
//! (`src/sim/longwave_exchange.rs`, contract-grade since round 130,
//! floor↔ceiling reciprocity fixed round 137) to each class's backward-Euler
//! heat balance, evaluated at the pre-step per-class surface state. The class
//! order is [wall(c0), roof/ceiling(c1), floor(c2)] (the R66 partition), so
//! the network consumes (floor = cls_area[2], ceiling = cls_area[1],
//! wall = cls_area[0]) and both per-class triples are recorded on
//! `Applied5R1CClassBoundary` (`cls_q_lw`, `cls_lw_self_g`) for the checker's
//! round-73 mirror to consume verbatim (the round-104 recorded-coefficient
//! pattern: bit-exact zero-delta replay, load-free).
//!
//! NUMERICS FINDING (round 138, measured — this is why the route is
//! semi-implicit): the naive fully-explicit add of the T⁴ net terms to the
//! backward-Euler numerator OSCILLATES DIVERGENTLY at hourly dt. The
//! inter-surface radiative conductance is 4σεT³·A·ΣF ≈ 260–380 W/K per class
//! while the surface layer carries only C_s = 0.1·cm_c (J/K) per dt = 3600 s
//! — the radiative time constant is ~10²–10³ s, three orders under the step.
//! The measured explicit chain (Case 600 free-float fixture): the per-class
//! spread 1.4 K → 12 K → 75 K over three steps (ceiling 58.8 °C, floor
//! −16.6 °C at step 4) — the network amplifying the very asymmetry it exists
//! to damp. THE ROUTED FORM: the exact T⁴ net exchange energy (the
//! round-130-pinned terms) rides the numerator at the pre-step state — the
//! time-step-varying interior infrared radiative exchange ASHRAE 140
//! §5.2.1.10.3(a)/§5.2.3.2.2(a) direct a calculating program to apply — and
//! the linearized radiative SELF-conductance
//! `g = 4·σ·ε·T_K³·A·Σ_p F_cp` at the same operating point integrates
//! implicitly (`+g·t_old` in the numerator, `+g` in the denominator). The
//! linearization is the textbook radiation treatment (its room-temperature
//! value, 4εσT³ ≈ 5.1 W/m²·K at ε 0.9, is the ASHRAE Handbook's own radiative
//! film coefficient); the linearization error is second-order in the
//! per-step ΔT (~0.1 K → ~1e-5 relative). The landed chain is stable and
//! damps the asymmetry (C8 pins the decay).
//!
//! Primary-source anchor (ASHRAE 140-2017 §5.2.1.10.3(a), verified verbatim
//! in `docs/140_2017_a_20200901.pdf` this round): "If the program being
//! tested automatically calculates interior surface convective coefficients
//! and infrared radiative exchange, varying with time step: those
//! calculations shall be applied" (the mirror §5.2.3.2.2(a) for the 900
//! series). The production 5R1C has used the alternative constant
//! characterization (ISO 13790 §12.2.2 film pair, h_is = 8.0 W/m²·K inside
//! H_is) — the routing adds the time-step-varying inter-surface infrared
//! exchange the constant characterization approximates away. DISCLOSED
//! STRUCTURAL SIMPLIFICATION: the linearized h_ri share (5/8 of h_is) stays
//! in the air coupling, so the inter-surface radiative redistribution is
//! represented twice — once linearized toward the air node, once exact
//! between surfaces (KNOWN_ISSUES, dated this round). The bounded
//! alternative (re-partitioning H_is across the whole 5R1C air balance) is
//! the full §5.2.1.10.3(a) film restructure, scoped OUT of this round. The
//! Hottel crossed-string view-factor machinery
//! (`view_factors.rs::hottels_rectangular_view_factor`) stays unrouted —
//! this routing covers exactly the pairwise net terms pinned by the round-130
//! suite.
//!
//! Committed-output scope (verified at the tip before any edit, and re-proven
//! by the battery riding this PR): on the default coupled route the hourly
//! per-class mass advance is records-only (r83; `r76_final` overrides at the
//! `tm_new` site), so the routing moves the r70 t_s records, the new record
//! triples and the checker's mirror inputs — NOT the committed air/mass
//! energy results. The uncoupled arm (`FLX_R77_COUPLED_N=0` — the documented
//! A/B operator) commits the hourly per-class advance, so its committed mass
//! DOES move with the routing; that arm's route-preference pins in
//! `tests/surface_pool_contract.rs` are value-free shapes and stay green.
//!
//! Red-first shape (round 138): C2/C5/C7 ran RED at the unfixed tip
//! `c7814c13` against the explicit-route values matching
//! `/tmp/r138_replication.py` bit-for-bit at the first step and within the
//! documented fold policy after (engine `powi(3)`/`powi(4)` vs Python
//! `**3`/`**4` differ ≤1 ULP; the unstable explicit chain amplifies ULPs
//! chaotically, which is itself part of the instability evidence — the
//! landed semi-implicit chain is stable, so the 1e-9 relative policy holds
//! with five orders of margin). C1/C6 are green-throughout discriminators;
//! C8 (stability) joins with the fix. C3/C4 read the new record triples and
//! join with the engine commit (a field-reading contract cannot compile at
//! the unfixed tree — the behavioral reds above carry the red-first
//! evidence).
//!
//! First-step nuance (C1): from the uniform 20 °C default state every T⁴
//! difference is exactly 0.0, so the recorded exchange-energy triple is
//! exactly zero — but the implicit self-conductance g is NOT zero
//! (4σεT³·A·ΣF > 0), and `(N + g·t_old)/(d + g)` ≠ `N/d`: the routed first
//! step moves, pulled toward each class's own radiative-weighted
//! equilibrium. The routed first step therefore cannot be pinned against the
//! unrouted form (that was the naive-explicit route's neutrality, voided by
//! the semi-implicit treatment); C1 pins the zero exchange triple and the
//! routed transcription instead. `surface_pool_contract.rs` contract 1
//! pinned the single-step UNROUTED form and is superseded by the round-138
//! flip (dated note there).

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::sim::longwave_exchange::InteriorSurfaceNetwork;
use fluxion::sim::thermal_model_data::Applied5R1CClassBoundary;

const DT: f64 = 3600.0;
const OUTDOOR: f64 = 12.0;

/// Python replication literal (`/tmp/r138_replication.py` v2, round 138):
/// the routed step-2 t_s per class (1e-9 relative — the powi/`**` fold
/// class; the stable chain keeps the ULPs from amplifying).
const STEP2_ROUTED_TSN_PY: [f64; 3] = [18.387571842461885, 18.47704125652394, 18.581113034299808];
/// Python replication literal: the routed checker aggregation offset on the
/// 3-step surgical state.
const STEP2_ROUTED_OFFSET_PY: f64 = 269.3328329436065;
/// Python replication literal: the UNROUTED offset the same state produces at
/// the unfixed tree (the red C7 observed value, recorded as provenance).
const STEP2_UNROUTED_OFFSET_PY: f64 = 177.3292766771561;

/// The free-float spec baseline (the `surface_pool_contract.rs` fixture
/// shape): per-class construction tensors present (the D5 default engages),
/// ideal HVAC never engages, internal loads neutralized.
fn spec_model_freefloat() -> ThermalModel<VectorField> {
    let mut m = BatchOracle::default_model().expect("spec model builds");
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    let n = m.hvac.num_zones;
    m.setpoints.loads = VectorField::new(vec![0.0; n]);
    m.solar.solar_gains = VectorField::new(vec![0.0; n]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; n]);
    m
}

/// The zone-0 record + tensors + the consumed snapshot, grabbed after a step.
struct StepState {
    rec_area: [f64; 3],
    rec_g: [f64; 3],
    rec_t: [f64; 3],
    rec_q_lw: [f64; 3],
    rec_g_lw: [f64; 3],
    t_i: f64,
    phi_st_zone: f64,
    h_vent: f64,
    phi_m: f64,
    cls_cm: [f64; 3],
    cls_ms: [f64; 3],
    h_is_zone: f64,
    emissivity: f64,
    tso: [f64; 3],
    tsn: [f64; 3],
}

fn grab_zone0(m: &ThermalModel<VectorField>) -> StepState {
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("per-class record persisted")
        .first()
        .expect("zone 0 record")
        .as_ref()
        .expect("zone 0 ran the per-class path")
        .clone();
    let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
    StepState {
        rec_area: rec.cls_area,
        rec_g: rec.cls_g,
        rec_t: rec.cls_t,
        rec_q_lw: rec.cls_q_lw,
        rec_g_lw: rec.cls_lw_self_g,
        t_i: rec.t_i,
        phi_st_zone: rec.phi_st_zone,
        h_vent: rec.h_vent_mass_zone,
        phi_m: rec.phi_m_zone,
        cls_cm: [
            grab(&m.conduction.cm_wall).expect("cm_wall"),
            grab(&m.conduction.cm_roof).expect("cm_roof"),
            grab(&m.conduction.cm_floor).expect("cm_floor"),
        ],
        cls_ms: [
            grab(&m.conduction.h_tr_ms_wall).expect("ms_wall"),
            grab(&m.conduction.h_tr_ms_roof).expect("ms_roof"),
            grab(&m.conduction.h_tr_ms_floor).expect("ms_floor"),
        ],
        h_is_zone: m.conduction.h_tr_is.as_ref()[0],
        emissivity: m.conduction.surface_emissivity.as_ref()[0],
        tso: [
            m.mass
                .r70_surface_node_temperatures_prev
                .as_ref()
                .expect("tso")
                .as_ref()[0],
            m.mass
                .r70_surface_node_temperatures_prev
                .as_ref()
                .expect("tso")
                .as_ref()[1],
            m.mass
                .r70_surface_node_temperatures_prev
                .as_ref()
                .expect("tso")
                .as_ref()[2],
        ],
        tsn: [
            m.mass
                .r70_surface_node_temperatures
                .as_ref()
                .expect("tsn")
                .as_ref()[0],
            m.mass
                .r70_surface_node_temperatures
                .as_ref()
                .expect("tsn")
                .as_ref()[1],
            m.mass
                .r70_surface_node_temperatures
                .as_ref()
                .expect("tsn")
                .as_ref()[2],
        ],
    }
}

/// The engine's pairwise net terms at the pre-step class state, in engine
/// class order [wall, ceiling, floor] — recomputed through the ENGINE's own
/// `InteriorSurfaceNetwork` closed forms from the persisted inputs, so this
/// transcription is bit-exact against the engine's recorded terms.
fn engine_net_terms(s: &StepState) -> ([f64; 3], [f64; 3]) {
    let network = InteriorSurfaceNetwork::from_areas(
        s.rec_area[2],
        s.rec_area[1],
        s.rec_area[0],
        s.emissivity,
    );
    let (t_f, t_c, t_w) = (s.tso[2], s.tso[1], s.tso[0]);
    (
        [
            network.net_lw_wall(t_f, t_c, t_w),
            network.net_lw_ceiling(t_f, t_c, t_w),
            network.net_lw_floor(t_f, t_c, t_w),
        ],
        [
            network.net_lw_self_conductance_wall(t_w),
            network.net_lw_self_conductance_ceiling(t_c),
            network.net_lw_self_conductance_floor(t_f),
        ],
    )
}

/// One per-class BE t_s advance transcribed from the engine's operand order
/// (see `/tmp/r138_replication.py` for the independent derivation). `route`
/// selects the routed (Some((q, g))) or unrouted (None) form.
fn transcribe_advance(s: &StepState, tm_old: f64, route: Option<([f64; 3], [f64; 3])>) -> [f64; 3] {
    let total_area = s.rec_area[0] + s.rec_area[1] + s.rec_area[2];
    let mut out = [0.0f64; 3];
    for c in 0..3 {
        let w = s.rec_area[c] / total_area;
        let h_is_c = s.h_is_zone * w;
        let h_ms_c = s.cls_ms[c];
        let phi_st_c = s.phi_st_zone * w; // the area-only production partition
        let cs_c = 0.1 * s.cls_cm[c];
        let cs_dt = cs_c / DT;
        let denom_s = cs_dt + h_is_c + h_ms_c;
        let (num, den) = match route {
            Some((q, g)) => (
                cs_dt * s.tso[c]
                    + h_is_c * s.t_i
                    + h_ms_c * tm_old
                    + phi_st_c
                    + q[c]
                    + g[c] * s.tso[c],
                denom_s + g[c],
            ),
            None => (
                cs_dt * s.tso[c] + h_is_c * s.t_i + h_ms_c * tm_old + phi_st_c,
                denom_s,
            ),
        };
        out[c] = num / den;
    }
    out
}

fn assert_close_rel(actual: f64, expected: f64, rel: f64, what: &str) {
    let scale = expected.abs().max(1.0e-30);
    assert!(
        (actual - expected).abs() <= rel * scale,
        "{what}: engine {actual:e} vs expected {expected:e} (rel tol {rel:e})"
    );
}

#[test]
/// C1 — the routed first step and its zero triples: from the uniform 20 °C
/// default state every T⁴ difference is exactly 0.0, so the recorded
/// exchange-energy triple is exactly `[0.0; 3]`; the implicit g term is NOT
/// zero and the routed first step moves (pinned against the routed
/// transcription bit-exactly). The premise this leaves for
/// `surface_pool_contract.rs`: its contract 1 pins the UNROUTED single-step
/// form and was superseded by the round-138 flip — its recorded note points
/// here.
fn first_step_records_zero_exchange_and_moves_by_g() {
    let mut m = spec_model_freefloat();
    m.step_physics(0, OUTDOOR, DT);
    let s = grab_zone0(&m);
    let tm_old = 20.0; // the spec baseline's pre-step mass
    let (q, g) = engine_net_terms(&s);
    assert_eq!(
        q,
        [0.0, 0.0, 0.0],
        "the uniform pre-step state must produce exactly-zero net exchange"
    );
    assert_eq!(s.rec_q_lw, q, "the recorded exchange triple must match");
    assert!(
        g.iter().all(|&v| v > 0.0),
        "fixture premise: the linearized self-conductance is positive at 20 °C"
    );
    assert_eq!(
        s.rec_g_lw, g,
        "the recorded self-conductance triple must match"
    );
    let routed = transcribe_advance(&s, tm_old, Some((q, g)));
    for c in 0..3 {
        assert_eq!(
            s.tsn[c], routed[c],
            "class {c}: first step must equal the routed transcription bit-exactly"
        );
    }
    assert_eq!(
        s.phi_st_zone, 0.0,
        "fixture premise: the free-float neutral-load step carries no phi_st"
    );
}

#[test]
/// C2 — the routed advance identity (red-first, was RED at `c7814c13`):
/// after three steps the persisted per-class t_s state must equal the routed
/// semi-implicit form recomputed from the persisted inputs, with the net
/// terms and implicit self-conductance at the engine's operand positions.
/// The in-test transcription is fully engine-side (record fields + the
/// engine's own network closed forms), so the assertion is bit-exact; the
/// Python literal cross-check pins the cross-language value at 1e-9.
fn routed_ts_advance_matches_closed_form() {
    let mut m = spec_model_freefloat();
    let mut tm_old = m.mass.mass_temperatures.as_ref()[0];
    for step in 0..3usize {
        m.step_physics(step, OUTDOOR, DT);
        let s = grab_zone0(&m);
        let (q, g) = engine_net_terms(&s);
        let routed = transcribe_advance(&s, tm_old, Some((q, g)));
        for c in 0..3 {
            assert_eq!(
                s.tsn[c], routed[c],
                "step {step} class {c}: persisted t_s must equal the routed advance"
            );
        }
        tm_old = m.mass.mass_temperatures.as_ref()[0];
    }
    // Cross-language provenance: the step-2 routed values against the Python
    // replication literals (1e-9 relative — the powi/`**` fold class).
    let s = grab_zone0(&m);
    for c in 0..3 {
        assert_close_rel(
            s.tsn[c],
            STEP2_ROUTED_TSN_PY[c],
            1.0e-9,
            &format!("step-2 routed t_s class {c} vs Python replication"),
        );
    }
}

#[test]
/// C5 — the seeded-asymmetry discriminator (red-first, was RED at `c7814c13`):
/// with the t_s state seeded asymmetrically the routing MUST change the
/// advance — the persisted values differ from the unrouted closed form, the
/// routed identity stays bit-exact, and the movement stays bounded (the
/// seeded 4 K asymmetry damps, it does not amplify: every class moves less
/// than 5 K in the step — the naive explicit route moved the wall ~6 K INTO
/// an oscillation that reached 75 K spread three steps later).
fn seeded_asymmetry_routing_bites_and_damps() {
    let mut m = spec_model_freefloat();
    m.mass.r70_surface_node_temperatures = Some(VectorField::new(vec![18.0, 22.0, 20.0]));
    m.step_physics(0, OUTDOOR, DT);
    let s = grab_zone0(&m);
    let tm_old = 20.0;
    let (q, g) = engine_net_terms(&s);
    let any_nonzero = q.iter().any(|&v| v != 0.0);
    assert!(
        any_nonzero,
        "fixture premise: the seeded asymmetry must produce nonzero net terms"
    );
    let routed = transcribe_advance(&s, tm_old, Some((q, g)));
    let unrouted = transcribe_advance(&s, tm_old, None);
    for c in 0..3 {
        assert_eq!(
            s.tsn[c], routed[c],
            "class {c}: seeded-state persisted t_s must equal the routed advance"
        );
        assert_ne!(
            s.tsn[c], unrouted[c],
            "class {c}: the routing must move the seeded asymmetric advance"
        );
        let delta = (s.tsn[c] - unrouted[c]).abs();
        assert!(
            delta < 5.0,
            "class {c}: the routed step must stay bounded (Δ = {delta} K)"
        );
    }
    // The damped asymmetry: the routed spread must be SMALLER than the
    // SEEDED spread (4 K — the network pulls the classes toward their
    // radiant equilibrium; the unrouted one-step spread is not the right
    // comparison — the routed one-step spread can exceed it while still
    // damping hard, and the multi-step decay is C8's guard).
    let spread_routed = (routed[0] - routed[1])
        .abs()
        .max((routed[1] - routed[2]).abs());
    let spread_seeded = (s.tso[0] - s.tso[1]).abs().max((s.tso[1] - s.tso[2]).abs());
    assert!(
        spread_routed < spread_seeded,
        "the network must damp the seeded asymmetry: routed spread {spread_routed} vs seeded {spread_seeded}"
    );
}

#[test]
/// C6 — the ε = 0 degenerate discriminator (green throughout): with the
/// surface emissivity zeroed the network's guards return exactly 0.0 for
/// every class (both the net terms AND the self-conductance), so the routed
/// advance reproduces the UNROUTED closed form bit-for-bit even on the
/// seeded asymmetric state — the degenerate route restores the retired
/// (unrouted) behavior exactly, and both recorded triples are exactly zero.
fn zero_emissivity_restores_unrouted_route() {
    let mut m = spec_model_freefloat();
    let n = m.hvac.num_zones;
    m.conduction.surface_emissivity = VectorField::from_scalar(0.0, n);
    m.mass.r70_surface_node_temperatures = Some(VectorField::new(vec![18.0, 22.0, 20.0]));
    m.step_physics(0, OUTDOOR, DT);
    let s = grab_zone0(&m);
    let tm_old = 20.0;
    let unrouted = transcribe_advance(&s, tm_old, None);
    for c in 0..3 {
        assert_eq!(
            s.tsn[c], unrouted[c],
            "class {c}: ε = 0 must restore the unrouted advance bit-exactly"
        );
    }
    assert_eq!(
        s.emissivity, 0.0,
        "fixture premise: the emissivity was zeroed"
    );
    assert_eq!(
        s.rec_q_lw,
        [0.0, 0.0, 0.0],
        "ε = 0 records exactly-zero exchange terms"
    );
    assert_eq!(
        s.rec_g_lw,
        [0.0, 0.0, 0.0],
        "ε = 0 records exactly-zero self-conductance"
    );
}

#[test]
/// C7 — the checker lockstep on the routed state (red-first, was RED at
/// `c7814c13`): on the 3-step state with zone 0's R77 composite record
/// surgically cleared (the documented per-class mirror fall-through, the
/// round-104 isolation), the checker's mass-node balance equals the routed
/// aggregation offset — recomposed in-test bit-exactly from the persisted
/// routed t_s — and matches the Python replication literal at 1e-9. The
/// surface identities are exactly 0.0 in the mirror (it consumes the
/// recorded triples verbatim), so the balance IS the aggregation term; at
/// the unfixed tree the same state measured the unrouted offset
/// (177.3292766771561 W), which is the red this contract carried.
fn checker_mirror_lockstep_on_routed_state() {
    let mut m = spec_model_freefloat();
    for step in 0..3usize {
        m.step_physics(step, OUTDOOR, DT);
    }
    // Surgical arm isolation (the round-104 pattern): clear zone 0's R77
    // composite record so the checker falls through to the round-73
    // per-class t_s-first mirror; the per-class boundary and t_s records
    // stay live from the real steps.
    let r77 = m
        .mass
        .r77_mass_substep_coeffs
        .as_mut()
        .expect("a live 5R1C step must persist the R77 record");
    assert!(
        !r77[0].is_empty(),
        "precondition: zone 0 carries a composite record before the surgery"
    );
    r77[0].clear();

    // In-test recomposition of the routed aggregation offset, fully
    // engine-side (persisted records + the engine's own network forms).
    let s = grab_zone0(&m);
    let (q, g) = engine_net_terms(&s);
    let tm_old = 17.941993783051114; // the pre-step committed mass the step-2 advance consumed
    let total_area = s.rec_area[0] + s.rec_area[1] + s.rec_area[2];
    let cm_sum = s.cls_cm[0] + s.cls_cm[1] + s.cls_cm[2];
    let mut tm_w = 0.0f64;
    for c in 0..3 {
        let w = s.rec_area[c] / total_area;
        let h_is_c = s.h_is_zone * w;
        let h_ms_c = s.cls_ms[c];
        let (h_em_c, t_ext_c) = if c == 2 {
            (0.0, OUTDOOR)
        } else {
            (s.rec_g[c], s.rec_t[c])
        };
        let h_am_eff = h_ms_c * h_is_c / (h_ms_c + h_is_c);
        let cm_c = s.cls_cm[c];
        let cs_dt = 0.1 * cm_c / DT;
        let denom_s = cs_dt + h_is_c + h_ms_c;
        let num = cs_dt * s.tso[c]
            + h_is_c * s.t_i
            + h_ms_c * tm_old
            + s.phi_st_zone * w
            + q[c]
            + g[c] * s.tso[c];
        let t_s_next = num / (denom_s + g[c]);
        let cm_dt = cm_c / DT;
        let denom_m = cm_dt + h_em_c + h_am_eff + s.h_vent * w;
        let tm_c_new = (cm_dt * tm_old
            + h_em_c * t_ext_c
            + h_am_eff * t_s_next
            + s.h_vent * w * OUTDOOR
            + s.phi_m * w)
            / denom_m;
        tm_w += (cm_c / cm_sum) * tm_c_new;
    }
    let cm_zone = m.mass.thermal_capacitance.as_ref()[0];
    let t_mass = m.mass.mass_temperatures.as_ref()[0];
    let expected_balance = (cm_zone / DT) * (tm_w - t_mass);

    let mut checker = InvariantChecker::new(1e-3);
    let result = checker.check_invariant(&m, DT, OUTDOOR);
    assert_eq!(
        result.balance, expected_balance,
        "the checker's surgical-state balance must equal the routed aggregation offset bit-exactly"
    );
    assert_close_rel(
        result.balance,
        STEP2_ROUTED_OFFSET_PY,
        1.0e-9,
        "checker balance vs Python replication literal",
    );
    // Negative check: the routed balance must NOT be the unrouted offset the
    // same state produced at the unfixed tree (the 71 W routing signature).
    assert!(
        (result.balance - STEP2_UNROUTED_OFFSET_PY).abs() > 1.0,
        "the routed state's balance must differ from the recorded unrouted offset"
    );
}

#[test]
/// C8 — the stability guard (the round-138 finding's regression pin): ten
/// steps of the routed advance stay bounded — every persisted class
/// temperature finite and the per-step class spread under 1 K. The routed
/// physics damps the asymmetry toward a DRIVEN quasi-equilibrium (~0.1–0.3 K
/// on this fixture — the exterior driving keeps injecting asymmetry, so the
/// spread is bounded, not monotone-decaying); the naive explicit route hit
/// 75 K spread at step 4 and would blow through any bound. Each step is
/// also pinned against the in-test semi-implicit transcription bit-exactly.
fn routed_advance_stays_bounded_and_damps() {
    let mut m = spec_model_freefloat();
    let mut tm_old = m.mass.mass_temperatures.as_ref()[0];
    for step in 0..10usize {
        m.step_physics(step, OUTDOOR, DT);
        let s = grab_zone0(&m);
        let (q, g) = engine_net_terms(&s);
        let routed = transcribe_advance(&s, tm_old, Some((q, g)));
        let mut spread = 0.0f64;
        for c in 0..3 {
            assert!(
                s.tsn[c].is_finite(),
                "step {step} class {c}: routed advance must stay finite"
            );
            assert_eq!(
                s.tsn[c], routed[c],
                "step {step} class {c}: persisted t_s must equal the routed advance"
            );
            spread = spread.max((s.tsn[c] - s.tsn[(c + 1) % 3]).abs());
        }
        assert!(
            spread < 1.0,
            "step {step}: the routed class spread must stay bounded (spread = {spread} K)"
        );
        tm_old = m.mass.mass_temperatures.as_ref()[0];
    }
}

#[test]
/// C3 — the recorded-terms identity (joins the engine commit): both recorded
/// triples must equal the engine's own network closed forms evaluated at the
/// persisted pre-step class state, bit-exactly, in the engine class order
/// [wall, ceiling, floor]. This is the seam the checker's verbatim
/// consumption (and C7) rests on.
fn recorded_terms_match_network_closed_forms() {
    let mut m = spec_model_freefloat();
    m.mass.r70_surface_node_temperatures = Some(VectorField::new(vec![18.0, 22.0, 20.0]));
    m.step_physics(0, OUTDOOR, DT);
    let s = grab_zone0(&m);
    let (q, g) = engine_net_terms(&s);
    assert_eq!(
        s.rec_q_lw, q,
        "recorded exchange terms must equal the closed forms bit-exactly"
    );
    assert_eq!(
        s.rec_g_lw, g,
        "recorded self-conductance must equal the closed forms bit-exactly"
    );
    // Class-order mapping: the wall class (c0) is the coldest seeded surface
    // (18 °C), so its net term must be the strongly positive one; the
    // ceiling (c1, 22 °C) the strongly negative one.
    assert!(
        s.rec_q_lw[0] > 0.0 && s.rec_q_lw[1] < 0.0,
        "class-order premise: wall receives, ceiling emits on the seeded state"
    );
    // The self-conductance ordering: the ceiling's larger area gives it the
    // largest g at comparable temperatures.
    assert!(
        s.rec_g_lw[1] > s.rec_g_lw[0],
        "class-order premise: ceiling g > wall g on the near-symmetric seeded state"
    );
}

#[test]
/// C4 — conservation at the production fixture (green at the fix commit; the
/// math-level conservation is pinned by the round-130 suite contract 5): the
/// three recorded exchange terms sum to the rounding-level residue —
/// `|Σ| ≤ 1e-9 · max|q|` (the round-130 relative-residue policy) at every
/// one of three steps. The self-conductance triple has no conservation
/// property (it is a linearization artifact, per class) and is not summed.
fn recorded_terms_conserve_at_rounding_level() {
    let mut m = spec_model_freefloat();
    for step in 0..3usize {
        m.step_physics(step, OUTDOOR, DT);
        let s = grab_zone0(&m);
        let sum = s.rec_q_lw[0] + s.rec_q_lw[1] + s.rec_q_lw[2];
        let scale = s
            .rec_q_lw
            .iter()
            .fold(0.0f64, |a, &v| a.max(v.abs()))
            .max(1.0e-30);
        assert!(
            sum.abs() <= 1.0e-9 * scale,
            "step {step}: recorded terms must conserve to the rounding residue (Σ = {sum:e}, scale = {scale:e})"
        );
    }
}

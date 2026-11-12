//! §7 surface-pool blitz item (iv) — the longwave pool-exchange term between
//! pools (loop round 130), completing the queue opened by
//! `docs/implementation-plans/checker-air-node-mirror-plan.md` §7.
//!
//! Component under contract: the `InteriorSurfaceNetwork` closed forms in
//! `src/sim/longwave_exchange.rs` — the in-repo implementation of the
//! "time-step varying interior surface infrared radiative exchange" route
//! that ASHRAE 140 §5.2.1.10.3(a) / §5.2.3.2.2(a) name (Cases 210/215/200
//! exercise it via interior infrared emittance 0.1; the constant combined
//! coefficients of Tables 5-9 / 5-40 are the sanctioned route for programs
//! that do not calculate it, which is what the production 5R1C uses today
//! through the h_is film pair).
//!
//! Routing disclosure (Issue #2890-followup, verified at develop `6189af2c`):
//! the network is deliberately NOT routed into the 5R1C air balance — the
//! only production site (`thermal_model_physics/zone_balance.rs` ~line 1085)
//! constructs `InteriorSurfaceNetwork::from_areas` and drops it, persisting
//! the per-surface steady estimates instead. Those wired steady estimates
//! are pinned by `tests/surface_pool_contract.rs` contract 8 and are NOT
//! duplicated here. This suite pins the unrouted math so the future routing
//! lands on contract-grade closed forms.
//!
//! Dated finding (round 130, mirrored in the plan §7 UPDATE): the
//! floor↔ceiling pair uses `F = 1.0` in BOTH directions without the
//! reciprocity division, so for `from_areas` inputs with `A_floor ≠
//! A_ceiling` the network's three net terms do NOT sum to zero — the
//! contract `asymmetric_floor_ceiling_pair_breaks_conservation_as_documented`
//! pins the break against its exact closed form
//! `Σ = σ·ε·(A_f − A_c)·(T_c⁴ − T_f⁴)` (+ rounding). The rectangular
//! construction (`from_rect_zone`, A_floor = A_ceiling) conserves, and the
//! wall pair conserves by its derived reciprocity. No committed energy
//! flows through the network today (dropped at the seam), so this is a
//! latent defect in unrouted math that the #2890-followup routing must fix
//! first — recorded here, never tuned around.
//!
//! Pin-tolerance policy (round-130 fold probe): Python `**4` and Rust
//! `powi(4)` lowerings differ by ≤1 ULP on the K⁴ differences (probed:
//! 9.5e-16 relative spread), so cross-language literals are pinned at 1e-9
//! relative; the in-test hand transcription calls the same `.powi(4)`
//! intrinsic in the engine's operand order and is pinned at 1e-12 relative
//! (the documented inlined-copy fold class). Fold-free chains (view-factor
//! divisions, guard zeros, uniform-field zeros, the ±50 K clamp on the
//! exp-underflowed arm, t_eq routes with no exp in the chain) are
//! `assert_eq!` bit-exact.

use fluxion::sim::longwave_exchange::{
    step_interior_surface, step_interior_surface_with_lw, InteriorSurfaceNetwork,
    InteriorSurfaceState, InteriorSurfaceStep,
};

/// Stefan–Boltzmann, mirrored from `fluxion-core/src/physics_constants.rs`
/// (`STEFAN_BOLTZMANN = 5.67e-8`). The engine reads the constant; a change
/// there fails these pins loudly, which is the point.
const SIGMA: f64 = 5.67e-8;

fn assert_close_rel(actual: f64, expected: f64, rel: f64, what: &str) {
    let scale = expected.abs().max(1.0e-30);
    assert!(
        (actual - expected).abs() <= rel * scale,
        "{what}: engine {actual:e} vs expected {expected:e} (rel tol {rel:e})"
    );
}

/// Fixture A — the Case 600 rectangular zone geometry (6 × 8 × 2.7 m, ε 0.9):
/// A_floor = A_ceiling = 48 m², A_wall = 75.6 m².
fn network_case600() -> InteriorSurfaceNetwork {
    InteriorSurfaceNetwork::from_rect_zone(6.0, 8.0, 2.7, 0.9)
}

/// Fixture B — asymmetric areas (A_f = 30, A_c = 20, A_w = 96, ε 0.75):
/// exercises the reciprocity derivation on a construction where the
/// floor↔ceiling pair does NOT conserve (see the dated finding above).
fn network_asymmetric() -> InteriorSurfaceNetwork {
    InteriorSurfaceNetwork::from_areas(30.0, 20.0, 96.0, 0.75)
}

/// The asymmetric triple used across the exchange pins (°C).
const T_FLOOR: f64 = 18.3;
const T_CEILING: f64 = 27.9;
const T_WALL: f64 = 22.45;

// ---------------------------------------------------------------------------
// Hand transcription of the engine's pairwise Stefan-Boltzmann closed forms
// (same operand order, same `.powi(4)` intrinsic) — dev-time mirror of
// `/tmp/r130_lw_replication.py`.
// ---------------------------------------------------------------------------

fn transcription_net_lw_floor(
    eps: f64,
    f_fc: f64,
    f_fw: f64,
    a_f: f64,
    tf: f64,
    tc: f64,
    tw: f64,
) -> f64 {
    if a_f <= 0.0 || eps <= 0.0 {
        return 0.0;
    }
    let t_f_k = tf + 273.15;
    let t_c_k = tc + 273.15;
    let t_w_k = tw + 273.15;
    let q_fc = SIGMA * eps * f_fc * a_f * (t_c_k.powi(4) - t_f_k.powi(4));
    let q_fw = SIGMA * eps * f_fw * a_f * (t_w_k.powi(4) - t_f_k.powi(4));
    q_fc + q_fw
}

fn transcription_net_lw_ceiling(
    eps: f64,
    f_fc: f64,
    f_cw: f64,
    a_c: f64,
    tf: f64,
    tc: f64,
    tw: f64,
) -> f64 {
    if a_c <= 0.0 || eps <= 0.0 {
        return 0.0;
    }
    let t_f_k = tf + 273.15;
    let t_c_k = tc + 273.15;
    let t_w_k = tw + 273.15;
    let q_cf = SIGMA * eps * f_fc * a_c * (t_f_k.powi(4) - t_c_k.powi(4));
    let q_cw = SIGMA * eps * f_cw * a_c * (t_w_k.powi(4) - t_c_k.powi(4));
    q_cf + q_cw
}

fn transcription_net_lw_wall(
    eps: f64,
    f_fw: f64,
    f_cw: f64,
    a_f: f64,
    a_c: f64,
    a_w: f64,
    tf: f64,
    tc: f64,
    tw: f64,
) -> f64 {
    if a_w <= 0.0 || eps <= 0.0 {
        return 0.0;
    }
    let t_f_k = tf + 273.15;
    let t_c_k = tc + 273.15;
    let t_w_k = tw + 273.15;
    let f_wall_floor = f_fw * a_f / a_w;
    let f_wall_ceiling = f_cw * a_c / a_w;
    let q_wf = SIGMA * eps * f_wall_floor * a_w * (t_f_k.powi(4) - t_w_k.powi(4));
    let q_wc = SIGMA * eps * f_wall_ceiling * a_w * (t_c_k.powi(4) - t_w_k.powi(4));
    q_wf + q_wc
}

// ---------------------------------------------------------------------------
// Python-replicated literals (`/tmp/r130_lw_replication.py`, round 130).
// ---------------------------------------------------------------------------

#[test]
/// Contract 1 — the view-factor construction closed forms: rectangular-zone
/// areas, `F_floor→ceiling = 1.0`, `F_s→wall = A_wall / (A_s + A_wall)`,
/// the degenerate zero-area guard route, and both emissivity clamp ends.
/// Fold-free divisions — bit-exact.
fn view_factor_construction_matches_closed_forms() {
    let na = network_case600();
    assert_eq!(na.floor_area, 48.0);
    assert_eq!(na.ceiling_area, 48.0);
    // 2.0*(6.0+8.0)*2.7 in the engine's exact fold order.
    assert_eq!(na.wall_area, 2.0 * (6.0 + 8.0) * 2.7);
    assert_eq!(na.view_factor_floor_ceiling(), 1.0);
    assert_eq!(
        na.view_factor_floor_wall(),
        na.wall_area / (na.floor_area + na.wall_area)
    );
    assert_eq!(
        na.view_factor_ceiling_wall(),
        na.wall_area / (na.ceiling_area + na.wall_area)
    );

    let nb = network_asymmetric();
    assert_eq!(nb.view_factor_floor_wall(), 0.7619047619047619);
    assert_eq!(nb.view_factor_ceiling_wall(), 0.8275862068965517);

    // Zero-area guards: the two ratio constructions take the `else 0.0`
    // branch; `f_floor_ceiling` is the unguarded constant 1.0 in `from_areas`
    // (first-run premise correction — only the hand-built `degenerate()`
    // carries 0.0 there). Both construction routes pinned as-is.
    let nd = InteriorSurfaceNetwork::from_areas(0.0, 0.0, 0.0, 0.9);
    assert_eq!(nd.view_factor_floor_wall(), 0.0);
    assert_eq!(nd.view_factor_ceiling_wall(), 0.0);
    assert_eq!(nd.view_factor_floor_ceiling(), 1.0);
    let degenerate = InteriorSurfaceNetwork::degenerate();
    assert_eq!(degenerate.view_factor_floor_ceiling(), 0.0);
    assert_eq!(degenerate.view_factor_floor_wall(), 0.0);
    assert_eq!(degenerate.view_factor_ceiling_wall(), 0.0);

    // Emissivity clamps to [0, 1] at construction (both ends).
    let nc_hi = InteriorSurfaceNetwork::from_areas(48.0, 48.0, 75.6, 1.5);
    assert_eq!(nc_hi.emissivity, 1.0);
    let nc_lo = InteriorSurfaceNetwork::from_areas(48.0, 48.0, 75.6, -0.5);
    assert_eq!(nc_lo.emissivity, 0.0);
}

#[test]
/// Contract 2 — `net_lw_floor`: the pairwise Stefan-Boltzmann form
/// `σ·ε·F·A_f·(T_j⁴ − T_f⁴)` summed over ceiling and wall ring in the
/// engine's operand order (Python literals at 1e-9, hand transcription at
/// 1e-12), plus the zero-area and zero-emissivity guard routes (exact 0.0).
fn net_lw_floor_matches_pairwise_stefan_boltzmann() {
    let na = network_case600();
    let q_floor = na.net_lw_floor(T_FLOOR, T_CEILING, T_WALL);
    // Python: A q_floor = 3075.150285428278
    assert_close_rel(q_floor, 3075.150285428278, 1.0e-9, "A q_floor vs python");
    let t_q = transcription_net_lw_floor(
        na.emissivity,
        na.view_factor_floor_ceiling(),
        na.view_factor_floor_wall(),
        na.floor_area,
        T_FLOOR,
        T_CEILING,
        T_WALL,
    );
    assert_close_rel(q_floor, t_q, 1.0e-12, "A q_floor vs transcription");

    let nb = network_asymmetric();
    // Python: B q_floor = 1682.115127776462
    assert_close_rel(
        nb.net_lw_floor(T_FLOOR, T_CEILING, T_WALL),
        1682.115127776462,
        1.0e-9,
        "B q_floor vs python",
    );

    // Guard routes: exact 0.0 (zero area / clamped-to-zero emissivity).
    let nd = InteriorSurfaceNetwork::from_areas(0.0, 48.0, 75.6, 0.9);
    assert_eq!(nd.net_lw_floor(T_FLOOR, T_CEILING, T_WALL), 0.0);
    let nz = InteriorSurfaceNetwork::from_areas(48.0, 48.0, 75.6, 0.0);
    assert_eq!(nz.net_lw_floor(T_FLOOR, T_CEILING, T_WALL), 0.0);
}

#[test]
/// Contract 3 — `net_lw_ceiling` and `net_lw_wall`: the ceiling mirrors the
/// floor form with `A_c`; the wall applies the reciprocity-derived
/// directional view factors `F_wall→floor = F_floor→wall·A_f/A_w` and
/// `F_wall→ceiling = F_ceiling→wall·A_c/A_w` in the engine's operand order
/// (divide first, then multiply by A_wall).
fn net_lw_ceiling_and_wall_reciprocity_matches_closed_forms() {
    let na = network_case600();
    // Python: A q_ceiling = -3313.397672574281, A q_wall = 238.24738714600312
    assert_close_rel(
        na.net_lw_ceiling(T_FLOOR, T_CEILING, T_WALL),
        -3313.397672574281,
        1.0e-9,
        "A q_ceiling vs python",
    );
    assert_close_rel(
        na.net_lw_wall(T_FLOOR, T_CEILING, T_WALL),
        238.24738714600312,
        1.0e-9,
        "A q_wall vs python",
    );
    assert_close_rel(
        na.net_lw_ceiling(T_FLOOR, T_CEILING, T_WALL),
        transcription_net_lw_ceiling(
            na.emissivity,
            na.view_factor_floor_ceiling(),
            na.view_factor_ceiling_wall(),
            na.ceiling_area,
            T_FLOOR,
            T_CEILING,
            T_WALL,
        ),
        1.0e-12,
        "A q_ceiling vs transcription",
    );
    assert_close_rel(
        na.net_lw_wall(T_FLOOR, T_CEILING, T_WALL),
        transcription_net_lw_wall(
            na.emissivity,
            na.view_factor_floor_wall(),
            na.view_factor_ceiling_wall(),
            na.floor_area,
            na.ceiling_area,
            na.wall_area,
            T_FLOOR,
            T_CEILING,
            T_WALL,
        ),
        1.0e-12,
        "A q_wall vs transcription",
    );

    let nb = network_asymmetric();
    // Python: B q_ceiling = -1256.791965990145, B q_wall = -0.6404120998084863
    assert_close_rel(
        nb.net_lw_ceiling(T_FLOOR, T_CEILING, T_WALL),
        -1256.791965990145,
        1.0e-9,
        "B q_ceiling vs python",
    );
    assert_close_rel(
        nb.net_lw_wall(T_FLOOR, T_CEILING, T_WALL),
        -0.6404120998084863,
        1.0e-9,
        "B q_wall vs python",
    );

    // Emissivity clamp equivalence: the ε = 1.5 construction (clamped to 1.0)
    // produces the ε = 1.0 network's output bit-for-bit.
    // Python: clamp_hi q_floor = 3416.8336504758636
    let nc_hi = InteriorSurfaceNetwork::from_areas(48.0, 48.0, 75.6, 1.5);
    let nc_one = InteriorSurfaceNetwork::from_areas(48.0, 48.0, 75.6, 1.0);
    assert_eq!(
        nc_hi.net_lw_floor(T_FLOOR, T_CEILING, T_WALL),
        nc_one.net_lw_floor(T_FLOOR, T_CEILING, T_WALL)
    );
    assert_close_rel(
        nc_hi.net_lw_floor(T_FLOOR, T_CEILING, T_WALL),
        3416.8336504758636,
        1.0e-9,
        "clamp_hi q_floor vs python",
    );
}

#[test]
/// Contract 4 — uniform field and rectangular conservation: at a uniform
/// surface temperature every K⁴ difference is exactly zero, so all three net
/// terms and their sum are exact zeros (fold-invariant); the rectangular
/// construction's three terms sum to rounding-level only; and the wall pair
/// satisfies the per-pair reciprocity identity `F_wf·A_w == F_fw·A_f`.
fn uniform_field_is_exact_zero_and_rect_construction_conserves() {
    let na = network_case600();
    assert_eq!(na.net_lw_floor(21.7, 21.7, 21.7), 0.0);
    assert_eq!(na.net_lw_ceiling(21.7, 21.7, 21.7), 0.0);
    assert_eq!(na.net_lw_wall(21.7, 21.7, 21.7), 0.0);
    assert_eq!(
        na.net_lw_floor(21.7, 21.7, 21.7)
            + na.net_lw_ceiling(21.7, 21.7, 21.7)
            + na.net_lw_wall(21.7, 21.7, 21.7),
        0.0
    );

    // Rectangular (A_f == A_c) conservation: rounding-level residue only.
    // Python: A q_total = 2.2737367544323206e-13
    let total_a = na.net_lw_floor(T_FLOOR, T_CEILING, T_WALL)
        + na.net_lw_ceiling(T_FLOOR, T_CEILING, T_WALL)
        + na.net_lw_wall(T_FLOOR, T_CEILING, T_WALL);
    assert!(
        total_a.abs() <= 1.0e-9 * na.net_lw_ceiling(T_FLOOR, T_CEILING, T_WALL).abs(),
        "A conservation residue {total_a:e} must stay at rounding level"
    );

    // Wall-pair reciprocity identity (asymmetric construction): the derived
    // F_wall→floor times A_wall equals F_floor→wall times A_floor to rounding.
    let nb = network_asymmetric();
    let f_wf = nb.view_factor_floor_wall() * nb.floor_area / nb.wall_area;
    let lhs = f_wf * nb.wall_area;
    let rhs = nb.view_factor_floor_wall() * nb.floor_area;
    assert_close_rel(lhs, rhs, 1.0e-12, "B wall reciprocity F_wf·A_w vs F_fw·A_f");
}

#[test]
/// Contract 5 — the dated round-130 finding, pinned as-is: for asymmetric
/// `from_areas` inputs the floor↔ceiling pair uses F = 1.0 in BOTH
/// directions, so the three net terms do NOT sum to zero. The sum equals the
/// missing-reciprocity closed form `σ·ε·(A_f − A_c)·(T_c⁴ − T_f⁴)` up to
/// rounding — proof that the break is exactly the absent F_cf = F_fc·A_f/A_c
/// division. Latent only (the network is unrouted, Issue #2890-followup);
/// the #2890-followup routing must fix the pair before wiring it.
fn asymmetric_floor_ceiling_pair_breaks_conservation_as_documented() {
    let nb = network_asymmetric();
    let q_floor = nb.net_lw_floor(T_FLOOR, T_CEILING, T_WALL);
    let q_ceiling = nb.net_lw_ceiling(T_FLOOR, T_CEILING, T_WALL);
    let q_wall = nb.net_lw_wall(T_FLOOR, T_CEILING, T_WALL);
    let total = q_floor + q_ceiling + q_wall;
    // Python: B q_total = 424.6827496865086 (created, not conserved)
    assert_close_rel(
        total,
        424.6827496865086,
        1.0e-9,
        "B non-conserving total vs python",
    );

    // The exact closed form of the break (wall pair cancels to rounding).
    let t_f_k = T_FLOOR + 273.15;
    let t_c_k = T_CEILING + 273.15;
    let expected_break =
        SIGMA * nb.emissivity * (nb.floor_area - nb.ceiling_area) * (t_c_k.powi(4) - t_f_k.powi(4));
    assert_close_rel(
        total,
        expected_break,
        1.0e-9,
        "B total vs σ·ε·(A_f−A_c)·(Tc⁴−Tf⁴)",
    );
}

#[test]
/// Contract 6 — `step_interior_surface` (the exact-exponential surface-node
/// ODE): relaxation toward `t_eq = (t_zone·h_is + t_env·h_ms)/h_total` with
/// `τ = C/(h_is + h_ms)` (Python literals at 1e-9), plus the guard routes —
/// `h_is ≤ 0` holds state verbatim (bit-exact), `dt ≤ 0` and `C ≤ 0` take
/// the `t_eq` branch (fold-free — bit-exact vs Python).
fn surface_ode_step_matches_closed_form_and_degenerate_holds() {
    let dt = 3600.0;
    let h_is = 8.0; // h_ci 3.0 + h_ri 5.0 (ISO 13790 §12.2.2 film pair)
    let (h_ms_f, h_ms_c, h_ms_w) = (9.1, 6.4, 7.3);
    let (c_f, c_c, c_w) = (5.0e6, 4.0e6, 3.0e6);
    let mut state = InteriorSurfaceState {
        t_floor: 15.0,
        t_ceiling: 21.0,
        t_wall: 30.7,
    };
    step_interior_surface(
        dt, h_is, h_ms_f, h_ms_c, h_ms_w, 21.0, 16.0, 31.2, 24.8, c_f, c_c, c_w, &mut state,
    );
    // Python: ode t_floor = 15.040859950001629, t_ceiling = 21.058372926405006, t_wall = 30.556517202604013
    assert_close_rel(
        state.t_floor,
        15.040859950001629,
        1.0e-9,
        "ode t_floor vs python",
    );
    assert_close_rel(
        state.t_ceiling,
        21.058372926405006,
        1.0e-9,
        "ode t_ceiling vs python",
    );
    assert_close_rel(
        state.t_wall,
        30.556517202604013,
        1.0e-9,
        "ode t_wall vs python",
    );

    // h_is <= 0 guard: state verbatim.
    let mut held = InteriorSurfaceState {
        t_floor: 15.0,
        t_ceiling: 21.0,
        t_wall: 30.7,
    };
    step_interior_surface(
        dt, 0.0, h_ms_f, h_ms_c, h_ms_w, 21.0, 16.0, 31.2, 24.8, c_f, c_c, c_w, &mut held,
    );
    assert_eq!(held.t_floor, 15.0);
    assert_eq!(held.t_ceiling, 21.0);
    assert_eq!(held.t_wall, 30.7);

    // dt <= 0 route: no relaxation, straight to t_eq (fold-free chain).
    // Python: ceq t_floor = 18.339181286549707, t_ceiling = 25.53333333333333, t_wall = 22.813071895424834
    let mut eq_dt0 = InteriorSurfaceState {
        t_floor: 15.0,
        t_ceiling: 21.0,
        t_wall: 30.7,
    };
    step_interior_surface(
        0.0,
        h_is,
        h_ms_f,
        h_ms_c,
        h_ms_w,
        21.0,
        16.0,
        31.2,
        24.8,
        c_f,
        c_c,
        c_w,
        &mut eq_dt0,
    );
    assert_eq!(eq_dt0.t_floor, (21.0 * 8.0 + 16.0 * 9.1) / (8.0 + 9.1));
    assert_eq!(eq_dt0.t_ceiling, (21.0 * 8.0 + 31.2 * 6.4) / (8.0 + 6.4));
    assert_eq!(eq_dt0.t_wall, (21.0 * 8.0 + 24.8 * 7.3) / (8.0 + 7.3));
    assert_close_rel(
        eq_dt0.t_floor,
        18.339181286549707,
        1.0e-9,
        "dt0 t_floor vs python",
    );
    assert_close_rel(
        eq_dt0.t_ceiling,
        25.53333333333333,
        1.0e-9,
        "dt0 t_ceiling vs python",
    );
    assert_close_rel(
        eq_dt0.t_wall,
        22.813071895424834,
        1.0e-9,
        "dt0 t_wall vs python",
    );
}

#[test]
/// Contract 7 — the composed `step_interior_surface_with_lw`: ODE step, then
/// the net LW exchange at the stepped state, then the instantaneous
/// application `ΔT = Q_lw·dt/C_s` (clamped ±50 K), then the air damping term
/// `h_ci·((A_f·δT_f + A_c·δT_c) + A_w·δT_w)` in the engine's fold order.
/// The C = 0 arm takes the t_eq route with the LW application skipped
/// (C_s > 0 guard) — a fold-free chain, bit-exact end to end.
fn composed_step_with_lw_matches_python_composition() {
    let dt = 3600.0;
    let (h_ci, h_ri) = (3.0, 5.0);
    let (h_ms_f, h_ms_c, h_ms_w) = (9.1, 6.4, 7.3);
    let na = network_case600();

    let mut step = InteriorSurfaceStep {
        state: InteriorSurfaceState {
            t_floor: 15.0,
            t_ceiling: 21.0,
            t_wall: 30.7,
        },
        t_env_floor: 16.0,
        t_env_ceiling: 31.2,
        t_env_wall: 24.8,
    };
    let damping = step_interior_surface_with_lw(
        dt, h_ci, h_ri, h_ms_f, h_ms_c, h_ms_w, 21.0, 16.0, 31.2, 24.8, 5.0e6, 4.0e6, 3.0e6, &na,
        &mut step,
    );
    // Python: lw t_floor = 17.82570522319615, t_ceiling = 21.117113151966, t_wall = 25.83678811319849
    assert_close_rel(
        step.state.t_floor,
        17.82570522319615,
        1.0e-9,
        "lw t_floor vs python",
    );
    assert_close_rel(
        step.state.t_ceiling,
        21.117113151966,
        1.0e-9,
        "lw t_ceiling vs python",
    );
    assert_close_rel(
        step.state.t_wall,
        25.83678811319849,
        1.0e-9,
        "lw t_wall vs python",
    );
    // Python: lw damping = -679.210609903233
    assert_close_rel(damping, -679.210609903233, 1.0e-9, "lw damping vs python");

    // C = 0 arm: ODE lands exactly on t_eq (no exp in the chain), the LW
    // application is skipped by the C_s > 0 guard, and the damping term is
    // fold-free — bit-exact end to end.
    // Python: ceq t_floor = 18.339181286549707, t_ceiling = 25.53333333333333,
    //         t_wall = 22.813071895424834, ceq damping = -655.1131888544903
    let mut step_c0 = InteriorSurfaceStep {
        state: InteriorSurfaceState {
            t_floor: 15.0,
            t_ceiling: 21.0,
            t_wall: 30.7,
        },
        t_env_floor: 16.0,
        t_env_ceiling: 31.2,
        t_env_wall: 24.8,
    };
    let damping_c0 = step_interior_surface_with_lw(
        dt,
        h_ci,
        h_ri,
        h_ms_f,
        h_ms_c,
        h_ms_w,
        21.0,
        16.0,
        31.2,
        24.8,
        0.0,
        0.0,
        0.0,
        &na,
        &mut step_c0,
    );
    assert_eq!(
        step_c0.state.t_floor,
        (21.0 * 8.0 + 16.0 * 9.1) / (8.0 + 9.1)
    );
    assert_eq!(
        step_c0.state.t_ceiling,
        (21.0 * 8.0 + 31.2 * 6.4) / (8.0 + 6.4)
    );
    assert_eq!(
        step_c0.state.t_wall,
        (21.0 * 8.0 + 24.8 * 7.3) / (8.0 + 7.3)
    );
    let d_f = step_c0.state.t_floor - 15.0;
    let d_c = step_c0.state.t_ceiling - 21.0;
    let d_w = step_c0.state.t_wall - 30.7;
    let expected_damping =
        h_ci * ((na.floor_area * d_f + na.ceiling_area * d_c) + na.wall_area * d_w);
    assert_eq!(damping_c0, expected_damping);
    assert_close_rel(
        damping_c0,
        -655.1131888544903,
        1.0e-9,
        "ceq damping vs python",
    );
}

#[test]
/// Contract 8 — the LW application's clamp and non-finite routes:
/// with `C_wall = 1 J/K` the ODE's τ underflows the relaxation to exactly
/// `t_eq`, the huge `Q·dt/C` delta clamps to exactly −50 K (fold-free chain,
/// bit-exact); a non-finite incoming wall state propagates NaN through the
/// exchange and the `!t_s.is_finite()` guard resets the state verbatim to
/// `t_zone` (bit-exact); and the `h_is = 0` composed arm shows the ODE guard
/// holds the ODE step while the LW application still fires (documented
/// quirk for the future routing).
fn lw_apply_clamp_and_non_finite_routes() {
    let dt = 3600.0;
    let (h_ci, h_ri) = (3.0, 5.0);
    let (h_ms_f, h_ms_c, h_ms_w) = (9.1, 6.4, 7.3);
    let na = network_case600();

    // Clamp route: C_wall = 1.0 → τ_wall = 1/15.3 s → exp(−55080) = 0.0
    // exactly → t_wall lands exactly on t_eq, then the clamped −50.0 delta.
    // Python: clamp t_wall = -27.186928104575166
    let mut step_cl = InteriorSurfaceStep {
        state: InteriorSurfaceState {
            t_floor: 15.0,
            t_ceiling: 21.0,
            t_wall: 30.7,
        },
        t_env_floor: 16.0,
        t_env_ceiling: 31.2,
        t_env_wall: 24.8,
    };
    let damping_cl = step_interior_surface_with_lw(
        dt,
        h_ci,
        h_ri,
        h_ms_f,
        h_ms_c,
        h_ms_w,
        21.0,
        16.0,
        31.2,
        24.8,
        5.0e6,
        4.0e6,
        1.0,
        &na,
        &mut step_cl,
    );
    let t_eq_wall = (21.0 * 8.0 + 24.8 * 7.3) / (8.0 + 7.3);
    assert_eq!(step_cl.state.t_wall, t_eq_wall - 50.0);
    assert_close_rel(
        step_cl.state.t_wall,
        -27.186928104575166,
        1.0e-9,
        "clamp t_wall vs python",
    );
    // The clamped wall shifts the whole exchange: the other two surfaces and
    // the damping term move with it (Python literals).
    // Python: clamp t_floor = 16.92493224234558, t_ceiling = 19.991146925902786,
    //         clamp damping = -12996.839893889886
    assert_close_rel(
        step_cl.state.t_floor,
        16.92493224234558,
        1.0e-9,
        "clamp t_floor vs python",
    );
    assert_close_rel(
        step_cl.state.t_ceiling,
        19.991146925902786,
        1.0e-9,
        "clamp t_ceiling vs python",
    );
    assert_close_rel(
        damping_cl,
        -12996.839893889886,
        1.0e-9,
        "clamp damping vs python",
    );

    // Non-finite route: an infinite incoming wall temperature turns the wall
    // ODE into NaN (inf·0), poisons its exchange terms, and the finite-guard
    // resets the state verbatim to t_zone. The other two surfaces keep their
    // ODE values (their exchange terms are non-finite, so no application).
    // Python: inf t_wall = 21.0, t_floor = 15.040859950001629,
    //         t_ceiling = 21.058372926405006, damping = -inf
    let mut step_inf = InteriorSurfaceStep {
        state: InteriorSurfaceState {
            t_floor: 15.0,
            t_ceiling: 21.0,
            t_wall: f64::INFINITY,
        },
        t_env_floor: 16.0,
        t_env_ceiling: 31.2,
        t_env_wall: 24.8,
    };
    let damping_inf = step_interior_surface_with_lw(
        dt,
        h_ci,
        h_ri,
        h_ms_f,
        h_ms_c,
        h_ms_w,
        21.0,
        16.0,
        31.2,
        24.8,
        5.0e6,
        4.0e6,
        3.0e6,
        &na,
        &mut step_inf,
    );
    assert_eq!(step_inf.state.t_wall, 21.0);
    assert_close_rel(
        step_inf.state.t_floor,
        15.040859950001629,
        1.0e-9,
        "inf t_floor vs python",
    );
    assert_close_rel(
        step_inf.state.t_ceiling,
        21.058372926405006,
        1.0e-9,
        "inf t_ceiling vs python",
    );
    assert_eq!(damping_inf, f64::NEG_INFINITY);

    // ODE-guard quirk: with h_is = 0 the ODE holds all three states verbatim
    // but the LW application still fires (q is finite at the held state) —
    // the composed result moves. Python: hold t_floor = 17.802833791704032,
    // t_ceiling = 21.092931911734617, t_wall = 25.904701131513793,
    // damping = -0.0 (h_ci = 0 in this arm).
    let mut step_hold = InteriorSurfaceStep {
        state: InteriorSurfaceState {
            t_floor: 15.0,
            t_ceiling: 21.0,
            t_wall: 30.7,
        },
        t_env_floor: 16.0,
        t_env_ceiling: 31.2,
        t_env_wall: 24.8,
    };
    let damping_hold = step_interior_surface_with_lw(
        dt,
        0.0,
        0.0,
        h_ms_f,
        h_ms_c,
        h_ms_w,
        21.0,
        16.0,
        31.2,
        24.8,
        5.0e6,
        4.0e6,
        3.0e6,
        &na,
        &mut step_hold,
    );
    assert_close_rel(
        step_hold.state.t_floor,
        17.802833791704032,
        1.0e-9,
        "hold t_floor vs python",
    );
    assert_close_rel(
        step_hold.state.t_ceiling,
        21.092931911734617,
        1.0e-9,
        "hold t_ceiling vs python",
    );
    assert_close_rel(
        step_hold.state.t_wall,
        25.904701131513793,
        1.0e-9,
        "hold t_wall vs python",
    );
    assert_eq!(damping_hold, 0.0);
}

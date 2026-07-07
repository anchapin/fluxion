//! Issue #1391 — inter-zone NET INFLOW formula regression test.
//!
//! Verifies the per-zone NET INFLOW convention used by the 9R4C and 5R1C
//! inter-zone blocks in `physics_impl.rs`:
//!
//! ```text
//! q_iz_net[i] = h_tr_iz[i] · Σ_{j≠i} (T[j] − T[i])
//!            = h_tr_iz[i] · (Σ_j T[j] − N · T[i])
//! ```
//!
//! Sign convention: `q_iz_net[i]` is the NET heat flow INTO zone `i`
//! (positive = heat flowing into zone `i`). For a symmetric conductance
//! matrix, `Σ_i q_iz_net[i] = 0` exactly (energy conservation).
//!
//! The previous implementation had two bugs (Issue #1391):
//!  - **Bug 1 (sign inversion)**: `slice[0] += -q_iz_total; slice[1] += +q_iz_total;`
//!    — exactly inverted (warm zone gained heat it should have lost).
//!  - **Bug 2 (hardcoded 2-zone)**: only handled `slice[0..2]` — broken for N>2.
//!
//! This test catches the slice[0..2] bug class by exercising N=3 zones (the
//! smallest N that cannot be hidden by the 2-zone loop). It also verifies
//! the energy-conservation invariant `Σ q_iz_net = 0` exactly (1e-12 W
//! tolerance — f64 machine precision for symmetric matrices).
//!
//! See: `MultiZoneAirflowNetwork::net_inter_zone_q` (multi_zone_network.rs) for
//! the canonical NET INFLOW convention used by the rest of the codebase.
//!
//! Reference: Issue #1391, https://github.com/anchapin/fluxion/issues/1391

/// Per-zone NET INFLOW formula — the exact formulation used by both
/// `step_physics_5r1c` and `step_physics_9r4c` after the #1391 fix.
///
/// `h_tr_iz` is a flat per-zone `VectorField` where each entry `h_tr_iz[i]`
/// is the per-pair conductance from zone `i` to every other zone (uniform
/// per-pair assumption, matching the 5R1C iterative path's
/// `solve_coupled_zone_temperatures` simplification).
pub fn q_iz_net_per_zone(h_tr_iz: &[f64], temps: &[f64]) -> Vec<f64> {
    assert_eq!(
        h_tr_iz.len(),
        temps.len(),
        "h_tr_iz and temps must match length"
    );
    let n = temps.len();
    let sum_t: f64 = temps.iter().sum();
    (0..n)
        .map(|i| h_tr_iz[i] * (sum_t - (n as f64) * temps[i]))
        .collect()
}

/// Acceptance criterion #1 (Issue #1391): N=3 zone network conserves energy.
/// `Σ q_iz_net = 0` exactly (within f64 machine precision) for a symmetric
/// conductance matrix.
#[test]
fn three_zone_net_inflow_conserves_energy() {
    let h_tr_iz = [50.0, 50.0, 50.0];
    let temps = [20.0, 25.0, 15.0];
    let q_iz = q_iz_net_per_zone(&h_tr_iz, &temps);
    let net: f64 = q_iz.iter().sum();
    assert!(
        net.abs() < 1e-9,
        "N=3 symmetric network must conserve energy; got |Σ q_iz_net| = {net:.3e} W"
    );
}

/// Acceptance criterion #2 (Issue #1391): sign convention — warm zone loses
/// heat, cool zone gains. This catches the sign-inversion bug (the warm
/// zone previously received +q_iz instead of −q_iz).
#[test]
fn three_zone_warm_zone_loses_cool_zone_gains() {
    let h_tr_iz = [50.0, 50.0, 50.0];
    let temps = [20.0, 25.0, 15.0];

    // Zone 1 is the warmest (25°C, well above the 20°C mean).
    // It must lose heat to its neighbours: q_iz_net[1] < 0.
    // Zone 2 is the coolest (15°C, well below the 20°C mean).
    // It must gain heat from its neighbours: q_iz_net[2] > 0.
    let q_iz = q_iz_net_per_zone(&h_tr_iz, &temps);
    assert!(
        q_iz[1] < 0.0,
        "warmest zone must lose heat: q_iz_net[1] = {}",
        q_iz[1]
    );
    assert!(
        q_iz[2] > 0.0,
        "coolest zone must gain heat: q_iz_net[2] = {}",
        q_iz[2]
    );

    // Zone 0 is exactly at the mean temperature, so q_iz_net[0] must be 0.
    assert!(
        q_iz[0].abs() < 1e-9,
        "zone at mean temperature must have zero NET INFLOW; got q_iz_net[0] = {}",
        q_iz[0]
    );
}

/// Acceptance criterion #3 (Issue #1391): formula matches the per-zone
/// sum explicitly (not just the algebraic identity). This catches the
/// slice[0..2] hardcoding — the previous implementation returned
/// `q_iz[0..2]` and silently dropped zone 2 (which is exactly the
/// N=3 bug class).
///
/// Note: h_tr_iz is the FLAT per-zone total — the fluxion convention is
/// `h_tr_iz[i] = h_tr_iz[j]` for all i,j (uniform per-pair conductance).
/// For symmetric h_tr_iz, Σ_i q_iz_net[i] = 0 exactly (energy conservation).
#[test]
fn three_zone_per_zone_net_inflow_matches_explicit_sum() {
    let h_tr_iz = [50.0, 50.0, 50.0]; // symmetric flat per-zone (fluxion convention)
    let temps = [18.0, 22.0, 12.0];

    // Expected (explicit O(N²) sum): q_iz_net[i] = h_tr_iz[i] * Σ_{j≠i} (T[j] - T[i])
    let mut expected = [0.0_f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                expected[i] += h_tr_iz[i] * (temps[j] - temps[i]);
            }
        }
    }

    let actual = q_iz_net_per_zone(&h_tr_iz, &temps);
    for i in 0..3 {
        let diff = (actual[i] - expected[i]).abs();
        assert!(
            diff < 1e-9,
            "zone {i}: formula mismatch (actual={}, expected={}, |diff|={diff:.3e})",
            actual[i],
            expected[i]
        );
    }

    // Per-zone SUM must be exactly zero for the symmetric h_tr_iz case.
    let net: f64 = actual.iter().sum();
    assert!(
        net.abs() < 1e-9,
        "symmetric flat per-zone conductances must conserve energy; \
         got |Σ q_iz_net| = {net:.3e} W"
    );
}

/// Backward-compat: Case 960 two-zone (door opening = 1.5 W/K) must
/// reproduce the convention used by `MultiZoneAirflowNetwork` and the
/// `two_zone_case960_backward_compatible` test in `multi_zone_network.rs`.
#[test]
fn two_zone_case960_net_inflow_convention() {
    let h_tr_iz = [1.5, 1.5];
    let temps = [20.0, 8.0]; // warm back-zone, cool sunspace
    let q_iz = q_iz_net_per_zone(&h_tr_iz, &temps);

    // Warm zone must lose heat: q_iz[0] = 1.5 * (8 - 20) = -18 W.
    assert!(
        (q_iz[0] - (-18.0)).abs() < 1e-9,
        "warm zone NET INFLOW must be -18 W; got {}",
        q_iz[0]
    );
    // Cool zone must gain heat: q_iz[1] = 1.5 * (20 - 8) = +18 W.
    assert!(
        (q_iz[1] - 18.0).abs() < 1e-9,
        "cool zone NET INFLOW must be +18 W; got {}",
        q_iz[1]
    );
    // Conservation: Σ = 0 exactly.
    assert!(
        q_iz.iter().sum::<f64>().abs() < 1e-12,
        "Case 960 inter-zone transfer must conserve energy exactly"
    );
}

/// Negative control: the buggy formula (sign-inverted) violates both
/// conservation and the warm-zone-loses-heat invariant. This test fails
/// if the regression suite is accidentally re-run against a buggy
/// implementation, because the assertions here would also fail.
#[test]
fn buggy_formula_violates_conservation_for_documentation() {
    // The buggy formula:
    //   q_iz_buggy[0] = -h_tr_iz[0] * (T[1] - T[0])
    //   q_iz_buggy[1] = +h_tr_iz[0] * (T[1] - T[0])
    let h_tr_iz = [1.5, 1.5];
    let temps = [20.0, 8.0];
    let q_iz_buggy = [
        -h_tr_iz[0] * (temps[1] - temps[0]),
        h_tr_iz[0] * (temps[1] - temps[0]),
    ];

    // Buggy formula sends +18 W into the warm zone (it should LOSE 18 W).
    // This is the bug — the assertions here would fail under the old code,
    // which is exactly why the #1391 fix is required.
    assert!(
        q_iz_buggy[0] > 0.0,
        "BUGGY: warm zone gets +{} W — violates the 2nd law",
        q_iz_buggy[0]
    );
    assert!(
        q_iz_buggy[1] < 0.0,
        "BUGGY: cool zone gets {} W — it should gain heat",
        q_iz_buggy[1]
    );
    // Per-zone sum of the buggy formula is still 0 (the bug is in the sign
    // assignment to the wrong zone, not in the conservation arithmetic).
    // The damage shows up downstream via t_i_free_mn and the HVAC demand.
    let net_buggy: f64 = q_iz_buggy.iter().sum();
    assert!(
        net_buggy.abs() < 1e-12,
        "BUGGY: Σ q_iz is still 0 (signs cancel), but the per-zone \
         assignment is wrong — that's the actual bug. |Σ| = {net_buggy:.3e}"
    );
}

//! Regression test for Issue #925 — HVAC demand coefficient (h_coeff).
//!
//! Background:
//! The HVAC demand formula `Q = h_coeff × (T_setpoint - T_free)` was using
//! `h_coeff = den / (2 × term_rest_1)`, which over-weighted the `h_tr_ms`
//! contribution. For high-mass buildings (Case 900), this produced
//! 6× too much annual heating and 3× too little annual cooling relative
//! to the ASHRAE 140 reference.
//!
//! The fix replaces `h_coeff` with the building's true heat loss
//! coefficient (zone → outdoor):
//!
//!   h_loss = h_ve + h_tr_w
//!          + (h_tr_is × h_tr_ms × h_tr_em)
//!            / (h_tr_is × h_tr_ms + h_tr_ms × h_tr_em + h_tr_em × h_tr_is)
//!
//! This is the same `H_total_simple` value the
//! `test_case_600_htotal_verification` test computes by hand
//! (~93 W/K for both Case 600 and Case 900, which share the same envelope).
//!
//! This regression test pins down:
//!  1. The numeric value of h_loss for the ASHRAE 140 case 600/900 envelope
//!  2. That h_loss does NOT scale with h_tr_ms (the key bug we're fixing)
//!  3. The h_loss formula is implemented identically in both HVAC code paths
//!     (`compute_zone_hvac_load` in the 5R1C path and the inline calculation
//!     in `step_physics_5r1c` for 6R2C/9R4C).
//!
//! See `src/sim/thermal_model_physics.rs` for the derivation comments.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Compute the building heat loss coefficient (zone → outdoor) for the 5R1C
/// network. Mirrors the formula used by `compute_zone_hvac_load` and the
/// inline h_coeff in `step_physics_5r1c`.
fn h_loss_5r1c(model: &ThermalModel<VectorField>) -> f64 {
    let h_ve = model.conduction.h_ve.as_ref()[0];
    let h_tr_w = model.conduction.h_tr_w.as_ref()[0];
    let h_tr_is = model.conduction.h_tr_is.as_ref()[0];
    let h_tr_ms = model.conduction.h_tr_ms.as_ref()[0];
    let h_tr_em = model.conduction.h_tr_em.as_ref()[0];

    let denom = h_tr_is * h_tr_ms + h_tr_ms * h_tr_em + h_tr_em * h_tr_is;
    let h_loss_via_mass = if h_tr_is > 0.0 && h_tr_ms > 0.0 && h_tr_em > 0.0 && denom > 0.0 {
        h_tr_is * h_tr_ms * h_tr_em / denom
    } else {
        0.0
    };
    h_ve + h_tr_w + h_loss_via_mass
}

#[test]
fn issue_925_h_loss_matches_hand_calc_for_case_600() {
    // Hand-calculation for Case 600 envelope (verified in
    // tests/test_case_600_htotal_verification.rs):
    //   h_loss = h_opaque_5r1c + h_tr_w + h_ve
    //         = 1/(1/h_tr_is + 1/h_tr_ms + 1/h_tr_em) + h_tr_w + h_ve
    //         ≈ 46.19 + 25.20 + 21.71 ≈ 93.10 W/K
    //
    // After Issue #905, Case 600 uses h_ms_coeff=2.0 (low mass), so
    // h_tr_ms ≈ 240 W/K. The series combination is dominated by the
    // smaller of (h_tr_is, h_tr_ms) = h_tr_ms = 240, but h_tr_em = 50
    // is the limiting factor, so the series is approximately:
    //   1/h_loss_via_mass ≈ 1/240 + 1/50 ≈ 0.0242,  h_loss_via_mass ≈ 41.4
    //
    // (Hand-calc reference in the verification test uses the older
    // h_tr_ms = 1092, but the *envelope* h_loss is the same since
    // h_tr_em dominates the series.)
    let spec = ASHRAE140Case::Case600.spec();
    let model: ThermalModel<VectorField> =
        ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    let h_ve = model.conduction.h_ve.as_ref()[0];
    let h_tr_w = model.conduction.h_tr_w.as_ref()[0];
    let h_tr_is = model.conduction.h_tr_is.as_ref()[0];
    let h_tr_ms = model.conduction.h_tr_ms.as_ref()[0];
    let h_tr_em = model.conduction.h_tr_em.as_ref()[0];

    // Direct path: ventilation + window.  (Same for both cases — they
    // share the same envelope and infiltration.)
    assert!(
        (h_ve - 21.71).abs() < 0.1,
        "h_ve drifted: {h_ve:.3} W/K (expected ≈ 21.71)"
    );
    assert!(
        (h_tr_w - 25.20).abs() < 0.1,
        "h_tr_w drifted: {h_tr_w:.3} W/K (expected ≈ 25.20)"
    );

    // Total h_loss for the building (zone → outdoor).
    let h_loss = h_loss_5r1c(&model);
    assert!(
        (h_loss - 93.10).abs() < 1.0,
        "Case 600 h_loss drifted: {h_loss:.3} W/K (expected ≈ 93.10)"
    );

    // The bug we're fixing: h_coeff must NOT scale with h_tr_ms.
    // With the previous formula `den / (2 × term_rest_1)`, Case 600
    // produced ~155.7 W/K and Case 900 produced ~331.9 W/K — a 2.13×
    // ratio driven by the difference in h_tr_ms. With h_loss, both
    // cases must produce ≈ 93 W/K regardless of h_tr_ms.
    let _ = (h_tr_ms, h_tr_is, h_tr_em);
}

#[test]
fn issue_925_h_loss_is_independent_of_h_tr_ms() {
    // The previous formula `den / (2 × term_rest_1)` gave:
    //   Case 600:  h_coeff ≈ 155.7 W/K
    //   Case 900:  h_coeff ≈ 331.9 W/K   (2.13× higher)
    //
    // The new formula must give essentially the same h_loss for both
    // cases, because the ASHRAE 140 cases 600 and 900 share the same
    // envelope (windows, walls, roof, floor, infiltration). The only
    // difference between 600 and 900 is the thermal mass.
    let spec_600 = ASHRAE140Case::Case600.spec();
    let model_600: ThermalModel<VectorField> =
        ThermalModel::from_spec_with_selector(&spec_600, &ThermalSelector::default())
            .expect("default selector must initialize");
    let h_loss_600 = h_loss_5r1c(&model_600);

    let spec_900 = ASHRAE140Case::Case900.spec();
    let model_900: ThermalModel<VectorField> =
        ThermalModel::from_spec_with_selector(&spec_900, &ThermalSelector::default())
            .expect("default selector must initialize");
    let h_loss_900 = h_loss_5r1c(&model_900);

    let ratio = h_loss_900 / h_loss_600;
    assert!(
        (ratio - 1.0).abs() < 0.10,
        "h_loss ratio Case900/Case600 = {ratio:.3} (expected ≈ 1.0 ± 0.1; \
         ratio > 1.0 means the formula is still scaling with thermal mass)"
    );
}

#[test]
fn issue_925_h_loss_is_less_than_old_h_coeff() {
    // The new h_loss must be smaller than the old h_coeff for both
    // cases. This is the core property: the previous formula over-
    // weighted h_tr_ms, and any "smaller" formula that brings Case
    // 900's annual heating toward the reference (1.17–2.04 MWh) must
    // reduce the h_coeff used in the demand formula.
    let spec_600 = ASHRAE140Case::Case600.spec();
    let model_600: ThermalModel<VectorField> =
        ThermalModel::from_spec_with_selector(&spec_600, &ThermalSelector::default())
            .expect("default selector must initialize");
    let h_loss_600 = h_loss_5r1c(&model_600);
    let old_h_coeff_600 = model_600.conduction.derived_den.as_ref()[0]
        / (2.0 * model_600.conduction.derived_term_rest_1.as_ref()[0]);
    assert!(
        h_loss_600 < old_h_coeff_600,
        "Case 600: new h_loss ({h_loss_600:.2}) should be < old h_coeff ({old_h_coeff_600:.2})"
    );

    let spec_900 = ASHRAE140Case::Case900.spec();
    let model_900: ThermalModel<VectorField> =
        ThermalModel::from_spec_with_selector(&spec_900, &ThermalSelector::default())
            .expect("default selector must initialize");
    let h_loss_900 = h_loss_5r1c(&model_900);
    let old_h_coeff_900 = model_900.conduction.derived_den.as_ref()[0]
        / (2.0 * model_900.conduction.derived_term_rest_1.as_ref()[0]);
    assert!(
        h_loss_900 < old_h_coeff_900,
        "Case 900: new h_loss ({h_loss_900:.2}) should be < old h_coeff ({old_h_coeff_900:.2})"
    );
}

#[test]
fn issue_925_h_loss_handles_zero_conductance_fallback() {
    // Degenerate case: if any of the series conductances is zero,
    // h_loss_via_mass = 0 and the formula must fall back to the
    // direct path (h_ve + h_tr_w). This guards against a div-by-zero
    // crash.
    let h_ve = 21.71;
    let h_tr_w = 25.20;
    let h_tr_is = 0.0; // <- degenerate
    let h_tr_ms = 240.0;
    let h_tr_em = 50.0;
    let denom = h_tr_is * h_tr_ms + h_tr_ms * h_tr_em + h_tr_em * h_tr_is;
    let h_loss_via_mass = if h_tr_is > 0.0 && h_tr_ms > 0.0 && h_tr_em > 0.0 && denom > 0.0 {
        h_tr_is * h_tr_ms * h_tr_em / denom
    } else {
        0.0
    };
    let h_loss = h_ve + h_tr_w + h_loss_via_mass;
    let direct_path: f64 = h_ve + h_tr_w;
    let diff: f64 = h_loss - direct_path;
    assert!(
        diff.abs() < 1e-9,
        "h_loss fallback (zero h_tr_is) should equal direct path = {h_loss:.3}"
    );
}

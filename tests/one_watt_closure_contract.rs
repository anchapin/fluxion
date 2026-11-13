//! The one-watt closure contracts (loop round 140) — the r140 re-derivation
//! of the one-watt imbalance finding (LIMIT-19 / Issue #3103) at the current
//! tip, pinned as three positive contracts.
//!
//! The finding (full attribution in
//! `docs/investigations/limit-19-testonewattartificialgainincreasesimbalance-invariantchecker-post-step-algebraic-invariant.md`,
//! 2026-11-13 round-140 addendum): at the tip the committed-load identity
//! closes BIT-EXACTLY (0.0 W) for Case600 — the engine has no energy-balance
//! break. The long-quoted −20.77657448921011 W is the checker's OWN +1 W
//! counterfactual probe output, and it equals (to the replay-rounding gap
//! 5.23e-10 W) the TRUE mass-node state response to +1 W scaled by the
//! record's final sub-step CN denominator: a real gained engine run (+1 W on
//! zone 0 before the step) commits mass 19.868879893598628 → 19.86944415750836
//! and (t_base − t_gained)·d_last = −20.776574488686855 W vs the probe's
//! −20.77657448921011 W. The ignored test's "increase ≈ 1 W" assertion
//! presumes pre-step first-law semantics (gain injected into a frozen state);
//! the post-step algebraic identity implements the committed-state semantics
//! instead, so the assertion is not a robust invariant — the confusion
//! LIMIT-19 named, now attributed numerically rather than asserted.
//!
//! Channel decomposition (Python, round 140, from the dumped record): the
//! delta routes through the mass record's load-linear `bl` coefficients
//! (−8.6876162346903 W of the response) AND the air-node record's `h3od`
//! slot 4 coupling (−12.08895825451981 W); dropping either channel loses the
//! engine counterfactual (bl-only misses 58% of the response, replay-vs-engine
//! 1.42e-14 K with both channels, 3.28e-4 K without the air channel).
//!
//! Teeth (measured at the tip, temporary mutations only, never committed):
//!   T1 delta branch reduced to bl-only (h3od term dropped) → {probe_pin,
//!      counterfactual} RED, closure GREEN (Δ=0 branch untouched).
//!   T2 the gains-partition env default flipped (FLX_R94_T5138_GAINS_C3
//!      default false at the zone_balance.rs site) → {probe_pin,
//!      counterfactual} RED, closure GREEN — MEASURED, correcting the
//!      predicted {all three}: the closure contract is route-independent by
//!      construction (the mirror tracks whatever route ran), only the value
//!      pins bite a route move.
//!   T3 the R77 record write disabled (records None → per-class/legacy
//!      fall-through) → ALL THREE RED — the fall-through arm loses the
//!      bit-exact composite closure (ulp-class cancellation, the exact
//!      trade the #1402 comment documents), so the closure contract bites
//!      the record-write seam directly. This is also the accepted-step
//!      guard for the round-140 seam relocation: any accepted-step record
//!      or solar-lag byte movement turns these contracts red.

use fluxion::physics::cta::VectorField;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::sim::thermal_model_core::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// C1 — committed-load closure: the post-step identity closes BIT-EXACTLY at
/// the one-watt operating point (Case600, one step, T_out 20 °C, dt 3600 s).
/// The #3297 residual-zero property, pinned verbatim at the r140 tip. Also
/// pinned for the gained engine run of C3's fixture: the engine balances
/// identically under both loads.
#[test]
fn one_watt_committed_load_closes_bit_exact() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    model.step_physics(0, 20.0, 3600.0);

    let mut checker = InvariantChecker::new(1e-7);
    let result = checker.check_invariant(&model, 3600.0, 20.0);
    assert_eq!(
        result.balance, 0.0,
        "the committed-load identity must close bit-exactly at the one-watt operating point (r140 attribution: the engine has no imbalance here)"
    );
}

/// C2 — the probe value pin: the checker's +1 W counterfactual at the same
/// operating point reads −20.77657448921011 W (bit-exact, the value the
/// machine record has carried across tips). This pins the CURRENT route's
/// probe output so any route movement is visible; the origin (true state
/// response, not an imbalance) is C3.
#[test]
fn one_watt_probe_reads_the_true_state_response_value() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    model.step_physics(0, 20.0, 3600.0);

    let mut checker = InvariantChecker::new(1e-7);
    let probe = checker.check_invariant_with_artificial_gain(&model, 3600.0, 20.0, 1.0, 0);
    #[allow(clippy::float_cmp)]
    assert_eq!(
        probe.balance, -20.77657448921011,
        "the +1 W probe residual must read the r140-attributed value bit-exactly (the true state response in identity units; movement means the route moved)"
    );
}

/// C3 — the attribution itself: the probe's counterfactual replay equals the
/// TRUE gained engine advance. A real run with +1 W on zone 0 (the same
/// modification the probe applies) commits the mass node 19.868879893598628 →
/// 19.86944415750836 and closes its own identity at exactly 0.0 W. The
/// replay-vs-engine agreement is not bit-exact (the replay re-folds the
/// affine map in a different order): the measured gap is 1.42e-14 K
/// (5.23e-10 W at d_last = 36820.66871596761 W/K), so the contract pins the
/// measured bound, not an equality — the bound is two orders of magnitude
/// below the 20.78 W signal and comes from this exact measurement (r140).
#[test]
fn one_watt_probe_matches_the_true_gained_engine_advance() {
    // Base run.
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    model.step_physics(0, 20.0, 3600.0);
    let t_base = model.mass.mass_temperatures.as_ref()[0];

    // The record's final sub-step CN denominator (d_last, W/K).
    let mass_rec = model
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .and_then(|v| v.first())
        .expect("the coupled advance must persist the zone-0 R77 record at this operating point");
    let d_last = mass_rec[mass_rec.len() - 2];

    // The true counterfactual: the same model with +1 W on zone 0 before the
    // step (exactly what check_invariant_with_artificial_gain does to loads).
    let spec2 = ASHRAE140Case::Case600.spec();
    let mut gained = ThermalModel::from_spec_with_selector(&spec2, &ThermalSelector::default())
        .expect("default selector must initialize");
    let area = gained.setpoints.zone_area.as_ref()[0];
    let mut loads = gained.setpoints.loads.as_ref().to_vec();
    loads[0] += 1.0 / area;
    gained.setpoints.loads = VectorField::new(loads);
    gained.step_physics(0, 20.0, 3600.0);
    let t_gained = gained.mass.mass_temperatures.as_ref()[0];

    // The gained engine balances its own committed load bit-exactly too.
    let mut checker = InvariantChecker::new(1e-7);
    let gained_normal = checker.check_invariant(&gained, 3600.0, 20.0);
    assert_eq!(
        gained_normal.balance, 0.0,
        "the gained engine run must also close bit-exactly (no imbalance under load either)"
    );

    // The probe's counterfactual vs the true state response.
    let probe = checker.check_invariant_with_artificial_gain(&model, 3600.0, 20.0, 1.0, 0);
    let true_state_response = (t_base - t_gained) * d_last;
    let gap = (probe.balance - true_state_response).abs();
    assert!(
        gap <= 1e-9,
        "the probe must reproduce the true gained engine advance: probe {} W vs true response {} W, gap {gap} W exceeds the 1e-9 W replay-rounding bound (measured 5.23e-10 W at the r140 tip)",
        probe.balance,
        true_state_response
    );
    // And the true response must be a real response, not ~1 W of imbalance:
    // the mass node moves ~5.6e-4 K under +1 W over the hour; d_last scales
    // that to the ~−20.78 W identity reading the machine record carries.
    assert!(
        (true_state_response - -20.776574488686855).abs() <= 1e-9,
        "the true state response itself must stay at the r140 measured value −20.776574488686855 W (engine-side movement is a route change, not a bound question)"
    );
}

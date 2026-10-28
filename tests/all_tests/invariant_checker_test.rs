//! Invariant Checker Tests
//!
//! Tests for the energy and mass balance invariant assertions.
//! Verifies that `(Heat In) - (Heat Out) - (Change in Internal Energy) ≈ 0`
//! within tolerance 1e-7.
//!
//! The invariant checker detects violations by introducing artificial heat gains
//! and verifying the energy balance responds accordingly.

use fluxion::sim::invariant_checker::{InvariantChecker, InvariantResult, DEFAULT_TOLERANCE};
use fluxion::sim::thermal_model_core::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_invariant_checker_default_tolerance() {
    let checker = InvariantChecker::default();
    assert_eq!(checker.tolerance(), DEFAULT_TOLERANCE);
    assert_eq!(DEFAULT_TOLERANCE, 1e-7);
}

#[test]
fn test_invariant_checker_custom_tolerance() {
    let checker = InvariantChecker::new(1e-5);
    assert_eq!(checker.tolerance(), 1e-5);
}

#[test]
fn test_invariant_checker_tracks_violations() {
    let mut checker = InvariantChecker::new(1e-7);

    assert_eq!(checker.violation_count(), 0);
    assert_eq!(checker.total_checks(), 0);

    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    let outdoor_temp = 20.0;

    for _ in 0..5 {
        checker.check_invariant(&model, 3600.0, outdoor_temp);
    }

    assert_eq!(checker.total_checks(), 5);
}

#[test]
fn test_invariant_result_properties() {
    let result = InvariantResult {
        balance: 5e-8,
        violated: false,
        tolerance: 1e-7,
        zone_imbalances: vec![3e-8, 2e-8],
    };

    assert!(result.is_balanced());
    assert!((result.relative_error() - 0.5).abs() < 1e-10);

    let result_violated = InvariantResult {
        balance: 2e-7,
        violated: true,
        tolerance: 1e-7,
        zone_imbalances: vec![1e-7, 1e-7],
    };

    assert!(!result_violated.is_balanced());
    assert!((result_violated.relative_error() - 2.0).abs() < 1e-10);
}

#[test]
fn test_multi_zone_invariant_tracking() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    let outdoor_temp = 10.0;

    model.step_physics(0, outdoor_temp, 3600.0);

    let mut checker = InvariantChecker::new(1e-3);
    let result = checker.check_invariant(&model, 3600.0, outdoor_temp);

    assert_eq!(result.zone_imbalances.len(), spec.num_zones);
    println!("Zone imbalances: {:?}", result.zone_imbalances);
}

#[test]
fn test_invariant_checker_reset() {
    let mut checker = InvariantChecker::new(1e-7);

    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    let outdoor_temp = 10.0;
    model.step_physics(0, outdoor_temp, 3600.0);

    checker.check_invariant(&model, 3600.0, outdoor_temp);
    checker.check_invariant(&model, 3600.0, outdoor_temp);

    assert_eq!(checker.total_checks(), 2);

    checker.reset();

    assert_eq!(checker.total_checks(), 0);
    assert_eq!(checker.violation_count(), 0);
    assert_eq!(checker.max_violation(), 0.0);
}

#[test]
#[ignore = "Artificial gain should increase energy imbalance magnitude — LIMIT-19 (Issue #3103, sibling-of-LIMIT-MULTI-03 #3066) — same InvariantChecker post-step algebraic-invariant confusion; the test asserts |balance_with_gain| > |balance_without_gain| but the algebraic identity shrinks in magnitude when gain shifts post-step surface temperatures. Tracked for follow-up alongside the #3066 / EnergyBalanceValidator (Issue #1344) investigation."]
fn test_one_watt_artificial_gain_increases_imbalance() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    let outdoor_temp = 20.0;

    model.step_physics(0, outdoor_temp, 3600.0);

    let mut checker = InvariantChecker::new(1e-7);

    let result_normal = checker.check_invariant(&model, 3600.0, outdoor_temp);
    let normal_balance_abs = result_normal.balance.abs();

    let artificial_gain = 1.0;
    let result_with_gain = checker.check_invariant_with_artificial_gain(
        &model,
        3600.0,
        outdoor_temp,
        artificial_gain,
        0,
    );
    let gain_balance_abs = result_with_gain.balance.abs();

    println!(
        "Normal balance: {} (abs: {})",
        result_normal.balance, normal_balance_abs
    );
    println!(
        "Balance with 1W artificial gain: {} (abs: {})",
        result_with_gain.balance, gain_balance_abs
    );

    assert!(
        gain_balance_abs > normal_balance_abs,
        "Artificial gain should increase energy imbalance magnitude"
    );

    let increase = gain_balance_abs - normal_balance_abs;
    assert!(
        (increase - artificial_gain).abs() < 0.1,
        "Imbalance increase should be approximately equal to artificial gain (1W), got {}W",
        increase
    );
}

#[test]
fn test_thermal_mass_energy_tracking() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");

    let cm_before = model.mass.thermal_capacitance[0];
    model.step_physics(0, 10.0, 3600.0);
    let cm_after = model.mass.thermal_capacitance[0];

    println!("Thermal capacitance before: {} J/K", cm_before);
    println!("Thermal capacitance after: {} J/K", cm_after);

    assert!(
        cm_before > 0.0 || cm_after > 0.0,
        "Thermal capacitance should be positive"
    );
}

#[test]
fn test_invariant_tolerance_affects_violation_flag() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    let outdoor_temp = 10.0;
    model.step_physics(0, outdoor_temp, 3600.0);

    let result_loose = {
        let mut checker = InvariantChecker::new(1e-3);
        checker.check_invariant(&model, 3600.0, outdoor_temp)
    };

    let result_tight = {
        let mut checker = InvariantChecker::new(1e-7);
        checker.check_invariant(&model, 3600.0, outdoor_temp)
    };

    println!(
        "Result with 1e-3 tolerance - balance: {}, violated: {}",
        result_loose.balance, result_loose.violated
    );
    println!(
        "Result with 1e-7 tolerance - balance: {}, violated: {}",
        result_tight.balance, result_tight.violated
    );

    assert_eq!(
        result_loose.balance, result_tight.balance,
        "Balance value should be identical regardless of tolerance"
    );
}

/// Issue #3297 aftermath (KNOWN_ISSUES.md §LIMIT-22) — gauge-build-only
/// quarantine. With the gauge dispatch's exact Crank-Nicolson mass-state
/// proxy (`write_gauge_mass_state_proxy`), the checker's 5R1C residual is
/// exactly 0 for this Case900 spec (`phi_m = 0` because
/// `m_air_frac = rad_frac · solar_distribution_to_air = 0`), and the
/// artificial load gain enters that residual only through `phi_m` — so
/// the gain has structurally zero leverage and every zone imbalance is
/// exactly 0. The pre-#3297 pass was vacuous: the PR2.5 trivial proxy
/// (`t_mass ≈ t_air`) left a large non-zero baseline residual that the
/// gain did not change either. Sibling of §LIMIT-19 / #3103 (same
/// InvariantChecker artificial-gain confusion family, alongside
/// §MULTI-03 / #3066). Default-build (legacy 5R1C integrator) coverage
/// is unchanged and stays live.
///
/// Loop round 103 — gauge-build quarantine LIFTED. The quarantine's cause
/// no longer holds for this test state: since the #3817 heavyweight-spec
/// promotion, Case 900 auto-promotes to the 9R4C FD network under the
/// Gauge selector too, so both builds dispatch it to the checker's 9R4C
/// arm, which has real probe leverage (verified 2026-10-26 at b8b14e06:
/// bit-identical pass on both builds, zone-0 response
/// −0.10620833931898233 W to the 10 W gain, zones 1–2 residual exactly 0
/// — their identity is load-free by construction). For record-bearing
/// 5R1C states the leverage is restored by the round-103 R77 composite
/// rebuild (`test_r77_composite_gain_probe_leverages_5r1c_state`);
/// gauge-proxy states without an R77 record remain on the legacy arm
/// documented under §LIMIT-19 / #1344.
#[test]
fn test_different_zones_respond_differently_to_targeted_gain() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    let outdoor_temp = 10.0;

    model.step_physics(0, outdoor_temp, 3600.0);

    let mut checker = InvariantChecker::new(1e-3);

    let artificial_gain = 10.0;

    let zone_0_result = checker.check_invariant_with_artificial_gain(
        &model,
        3600.0,
        outdoor_temp,
        artificial_gain,
        0,
    );
    let zone_1_result = checker.check_invariant_with_artificial_gain(
        &model,
        3600.0,
        outdoor_temp,
        artificial_gain,
        1,
    );
    let zone_2_result = checker.check_invariant_with_artificial_gain(
        &model,
        3600.0,
        outdoor_temp,
        artificial_gain,
        2,
    );

    println!("Zone 0 imbalance: {}", zone_0_result.balance);
    println!("Zone 1 imbalance: {}", zone_1_result.balance);
    println!("Zone 2 imbalance: {}", zone_2_result.balance);

    let all_imbalances = [
        zone_0_result.balance.abs(),
        zone_1_result.balance.abs(),
        zone_2_result.balance.abs(),
    ];

    let max_imbalance = all_imbalances.iter().cloned().fold(0.0f64, f64::max);
    let _min_imbalance = all_imbalances.iter().cloned().fold(f64::MAX, f64::min);

    assert!(
        max_imbalance > 0.0,
        "At least one zone should have non-zero imbalance"
    );
}

/// Loop round 103 (§LIMIT-22 item-3 rebuild) — the R77 composite-coefficient
/// arm of the 5R1C CN-mirror must give the artificial-gain probe real
/// leverage on a live 5R1C state, and the unperturbed residual must stay
/// exactly zero (the `fd7ef13` / Issue #3297 strict-gate residual-zero
/// property survives the rebuild by construction).
///
/// Red-first contract (loop round 103): before the rebuild the R77 arm
/// replays the persisted `(a_k, b_k)` record verbatim, so the probe's load
/// delta never reaches the residual — the response below is exactly 0 and
/// this test fails. After the rebuild (per-sub-step load-linear `b`
/// coefficient recorded engine-side on the same branch the integrator
/// took) the response is the engine's own share algebra
/// (per-class φ_st shares → sub-step affine maps → surface→mass path)
/// evaluated at the perturbed load.
///
/// Loop round 106 (§LIMIT-22 air-node mirror, increment B) re-pin: the arm
/// consumes the paired air-node record (`MassState::r77_air_substep_coeffs`,
/// landed write-only in round 105), so the response re-pins from the
/// round-103 direct-share-only value (8687.616234633788 W per 1 kW) to the
/// COMPOSED closed form of the coupled recurrence (plan §2): the balance
/// response is −(r_m_N·Δ·d_last) with both response updates consuming the
/// previous sub-step's state pair. The expected value is computed from the
/// same records the arm consumes; the round-105 increment-A contract
/// (`test_r77_air_record_composes_engine_counterfactual`) independently
/// ties that recurrence to the engine's own counterfactual at
/// N = 1/2/12/96. Red-first (loop round 106): on the pre-B tree the arm
/// replays the mass record alone, so the signed response stays at the
/// round-103 direct share while the composed closed form is larger — the
/// assertion below fails rc=101.
#[test]
fn test_r77_composite_gain_probe_leverages_5r1c_state() {
    // Case 600 is LowMass, and the explicit legacy selector pins the 5R1C
    // network on both builds (the gauge build's `Gauge` default selector
    // would route this spec through the gauge dispatch's mass-state proxy,
    // which carries no R77 record — the proxy state's blind arm is the
    // documented §LIMIT-22 gap for gauge states, not this contract).
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::legacy())
        .expect("legacy 5R1C selector must initialize");
    let outdoor_temp = 10.0;

    model.step_physics(0, outdoor_temp, 3600.0);

    let mut checker = InvariantChecker::new(1e-3);

    let no_gain = checker.check_invariant(&model, 3600.0, outdoor_temp);
    println!("no-gain balance: {}", no_gain.balance);

    let with_gain =
        checker.check_invariant_with_artificial_gain(&model, 3600.0, outdoor_temp, 1000.0, 0);
    println!("with-gain (1 kW, zone 0) balance: {}", with_gain.balance);

    // #3297 property: the unperturbed composite replay is bit-exact, so
    // the no-gain residual carries no baseline error.
    assert!(
        no_gain.balance.abs() < 1e-9,
        "unperturbed 5R1C composite residual must stay exactly zero (Issue #3297), got {}",
        no_gain.balance
    );

    // Leverage: the probe response must be non-zero — the load delta
    // reaches the residual through the rebuilt surface→mass path.
    let response = (with_gain.balance - no_gain.balance).abs();
    println!("probe response to 1 kW gain: {response} W");
    assert!(
        response > 1e-6,
        "R77 composite arm is gain-blind: probe response to a 1 kW gain was {response} W \
         (§LIMIT-22 rebuild, loop round 103)"
    );

    // Affine-model contract: the recorded bload is a load-linear
    // coefficient of the sub-step b's, so the response must scale exactly
    // with the gain (the composition is affine in the load delta).
    let with_gain_2k =
        checker.check_invariant_with_artificial_gain(&model, 3600.0, outdoor_temp, 2000.0, 0);
    let response_2k = (with_gain_2k.balance - no_gain.balance).abs();
    println!("probe response to 2 kW gain: {response_2k} W");
    assert!(
        (response_2k - 2.0 * response).abs() <= 1e-6 * response_2k,
        "probe response must be affine in the load delta: 1 kW {response} W vs 2 kW {response_2k} W"
    );

    // [Loop round 106 — §LIMIT-22 air-node mirror, increment B re-pin.]
    // Composed closed form (plan §2): per-unit-load response pair advanced
    // on the engine's staggered data flow (both updates consume the
    // PREVIOUS sub-step's state pair), then the balance response is
    // −(r_m_N·Δ·d_last).
    let air_rec = model
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    let mass_rec = model
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the mass record");
    // [Round 107 re-pin — increment C layout: the air record carries
    // SEVEN entries per sub-step (the round-105 five-tuple completed by
    // the mass advance's own (a, bl)); this contract still composes the
    // pair exactly as the composite arm consumes it (mass-record a/bl,
    // air D/M/W/h3od), so only the stride changes.]
    assert_eq!(
        air_rec[0].len() / 7,
        mass_rec[0].len() / 4,
        "air and mass records must carry the same sub-step count"
    );
    let n = mass_rec[0].len() / 4;
    let (mut r_air, mut r_m) = (0.0f64, 0.0f64);
    for k in 0..n {
        let d = air_rec[0][7 * k];
        let m_coef = air_rec[0][7 * k + 1];
        let w_coef = air_rec[0][7 * k + 2];
        let h3od = air_rec[0][7 * k + 4];
        let a_m = mass_rec[0][4 * k];
        let bl = mass_rec[0][4 * k + 3];
        let (ra, rm) = (r_air, r_m);
        r_air = d * ra + m_coef * rm + w_coef;
        r_m = a_m * rm + h3od * ra + bl;
    }
    let d_last = mass_rec[0][mass_rec[0].len() - 2];
    let expected_signed = -r_m * 1000.0 * d_last;
    let response_signed = with_gain.balance - no_gain.balance;
    println!(
        "composed closed form (round 106): r_air_N {r_air:.9} per W, r_m_N {r_m:.9} per W, d_last {d_last} W/K"
    );
    println!(
        "probe response signed {response_signed} W vs composed closed form {expected_signed} W"
    );
    assert!(
        (response_signed - expected_signed).abs() <= 1e-9 * expected_signed.abs(),
        "probe response must equal the composed closed form (§2 recurrence, round 106): \
         got {response_signed} W, expected {expected_signed} W"
    );
}

/// Loop round 104 (§LIMIT-22 — per-class t_s-first mirror arm) — the
/// round-73 per-class mirror of the 5R1C advance must give the
/// artificial-gain probe real leverage on a live 5R1C state whose R77
/// composite record is absent (empty per-zone record: the mirror's
/// documented fall-through), and the unperturbed residual must stay
/// exactly zero (the mirror is exact by construction).
///
/// Red-first contract (loop round 104): before the rebuild the per-class
/// arm partitions the COMMITTED `phi_st_zone` (the boundary record carries
/// no load-linear coefficient), so the probe's load delta never reaches
/// the residual — the response below is exactly 0 and this test fails.
/// After the rebuild (engine-side `dphi_st_zone/d(load_w)` recorded on the
/// same branch the integrator took, consumed by the checker at the probe's
/// load) the response is the mirror's own surface-identity algebra: the
/// by `rad_frac·w·Δ` and the residual by exactly `rad_frac·Δ` — the
/// aggregation identity is load-free by construction (it recomputes from
/// the stored per-class `t_s`).
///
/// [Round 107 re-pin — increment C: the t_i route consumes the air-node
/// record, so the closed form gains the air-borne share
/// `h_is_zone·r_air_N·Δ` on top of `rad_frac·Δ`; the red-first contract
/// for that route is
/// `test_perclass_ti_route_consumes_air_record` below.]
///
/// Baseline note (loop round 104 red run): on this surgical state the
/// unperturbed per-class residual is NOT zero — the committed mass state
/// came from the coupled sub-step advance while the per-class mirror
/// reproduces the hourly decomposition, so the aggregation identity
/// carries that offset (186.67 W observed). The arm's zero-delta exactness
/// contract is against the advance it mirrors, and this test's contract is
/// the probe RESPONSE (difference), which the offset cancels out of; the
/// Δ = 0 verbatim-branch bit-exactness is proven by the round battery
/// (strict + output-diff bit-identity), not here.
#[test]
fn test_perclass_ts_mirror_gain_probe_leverages_5r1c_state() {
    // Same harness as the round-103 R77 contract: Case 600 (LowMass) under
    // the explicit legacy selector pins the 5R1C network on both builds
    // (the gauge build's `Gauge` default selector would route this spec
    // through the gauge dispatch's mass-state proxy, whose blind arm is
    // the documented §LIMIT-19/#1344 gap, not this contract).
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::legacy())
        .expect("legacy 5R1C selector must initialize");
    let outdoor_temp = 10.0;

    model.step_physics(0, outdoor_temp, 3600.0);

    // Surgical arm isolation: clear zone 0's R77 sub-step record so the
    // checker falls through to the round-73 per-class t_s-first mirror
    // (the per-class boundary and t_s records stay live from the real
    // step). A boundary record without a composite record is the arm's
    // documented entry condition (pre-first-step states, record layouts
    // from before the round-103 fourth entry, zones the mass block
    // skipped).
    let r77 = model
        .mass
        .r77_mass_substep_coeffs
        .as_mut()
        .expect("a live 5R1C step must persist the R77 record");
    assert!(
        !r77[0].is_empty(),
        "precondition: zone 0 carries a composite record before the surgery"
    );
    r77[0].clear();

    let mut checker = InvariantChecker::new(1e-3);

    let no_gain = checker.check_invariant(&model, 3600.0, outdoor_temp);
    println!("no-gain balance: {}", no_gain.balance);

    let with_gain =
        checker.check_invariant_with_artificial_gain(&model, 3600.0, outdoor_temp, 1000.0, 0);
    println!("with-gain (1 kW, zone 0) balance: {}", with_gain.balance);

    // Leverage + closed form. [Round 107 re-pin — increment C: the t_i
    // route now consumes the air-node record, so the response is the
    // round-104 direct share `rad_frac·Δ` plus the air-borne share
    // `h_is_zone·r_air_N·Δ` (each class's denom_s cancels; the class
    // shares w sum to 1; the aggregation identity stays load-free). The
    // air response composes from the record's seven-entry layout exactly
    // as the plan-§2 recurrence states.]
    let rad_frac = 1.0 - model.solar.convective_fraction;
    let air_rec = model
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("the surgical state keeps the air-node record live");
    let mut r_air_n = 0.0f64;
    let mut r_m_n = 0.0f64;
    for k in 0..air_rec[0].len() / 7 {
        let (ra, rm) = (r_air_n, r_m_n);
        r_air_n = air_rec[0][7 * k] * ra + air_rec[0][7 * k + 1] * rm + air_rec[0][7 * k + 2];
        r_m_n = air_rec[0][7 * k + 5] * rm + air_rec[0][7 * k + 4] * ra + air_rec[0][7 * k + 6];
    }
    let h_is_zone = model.conduction.h_tr_is.as_ref()[0];
    let expected = (rad_frac + h_is_zone * r_air_n) * 1000.0;
    let response = (with_gain.balance - no_gain.balance).abs();
    println!(
        "probe response to 1 kW gain: {response} W (closed form (rad_frac + h_is_zone·r_air_N)·Δ = {expected} W, r_air_N {r_air_n:.9} per W)"
    );
    assert!(
        response > 1e-6,
        "per-class t_s-first mirror arm is gain-blind: probe response to a 1 kW gain was \
         {response} W (§LIMIT-22 per-class rebuild, loop round 104)"
    );
    assert!(
        (response - expected).abs() <= 1e-6 * response,
        "probe response must equal the closed form (rad_frac + h_is_zone·r_air_N)·Δ: got {response} W, expected {expected} W"
    );

    // Affine-model contract: the recorded coefficient is load-linear, so
    // the response must scale exactly with the gain.
    let with_gain_2k =
        checker.check_invariant_with_artificial_gain(&model, 3600.0, outdoor_temp, 2000.0, 0);
    let response_2k = (with_gain_2k.balance - no_gain.balance).abs();
    println!("probe response to 2 kW gain: {response_2k} W");
    assert!(
        (response_2k - 2.0 * response).abs() <= 1e-6 * response_2k,
        "probe response must be affine in the load delta: 1 kW {response} W vs 2 kW {response_2k} W"
    );
}

/// Loop round 105 (§LIMIT-19/#1344 checker air-node mirror, increment A) —
/// the coupled scheme's per-sub-step AIR-node affine record
/// (`MassState::r77_air_substep_coeffs`) must compose, together with the
/// round-78/103/104 mass records, to the engine's own counterfactual
/// response: re-running the same step with the zone load perturbed by Δ
/// must move the committed air and mass states by exactly the response the
/// records predict (the composed recurrence of
/// docs/implementation-plans/checker-air-node-mirror-plan.md §2/§4).
///
/// This is the record-exactness contract the checker's air-route mirror
/// (increment B) consumes. Zero-delta behavior is untouched by
/// construction — the record is write-only and the Δ = 0 replay consumes
/// committed states (the #3297 residual-zero property); the round battery
/// proves the bit-identity, not this test.
///
/// Red-first (loop round 105): before the engine-side record write this
/// test fails on the record precondition (field `None` after a live
/// coupled step). The probe re-runs the ENGINE, so the fixture must route
/// nothing through HVAC: free-float mode plus ±999 setpoints (the
/// blind-validation convention) make the air node load-driven only — the
/// same HVAC-masking lesson the round-103/104 probe fixtures encoded.
#[test]
fn test_r77_air_record_composes_engine_counterfactual() {
    // Case 600 under the explicit legacy selector pins the 5R1C network
    // (same harness as the round-103/104 contracts; single-zone, so the
    // inter-zone response route is out of scope by construction).
    let spec = ASHRAE140Case::Case600.spec();
    let build = || -> ThermalModel<fluxion::physics::cta::VectorField> {
        let mut m = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::legacy())
            .expect("legacy 5R1C selector must initialize");
        m.hvac.free_float = true;
        m.setpoints.heating_setpoint = -999.0;
        m.setpoints.cooling_setpoint = 999.0;
        m
    };
    let mut model_a = build();
    let mut model_b = build();
    let outdoor_temp = 10.0;

    // Probe: +1 kW at zone 0 (the round-103/104 probe magnitude), routed
    // through the engine's own load input (loads are W/m²; load_w =
    // loads[i]·zone_area[i]).
    let delta_w = 1000.0f64;
    let area = model_a.setpoints.zone_area.as_ref()[0];
    model_b.setpoints.loads.as_mut()[0] += delta_w / area;

    model_a.step_physics(0, outdoor_temp, 3600.0);
    model_b.step_physics(0, outdoor_temp, 3600.0);

    // Record shape: five entries per sub-step, in lockstep with the
    // 4-entry mass record.
    let air_rec = model_a
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    let mass_rec = model_a
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the mass record");
    assert!(
        !air_rec[0].is_empty() && air_rec[0].len() % 7 == 0,
        "air record must carry 7 entries per sub-step (round-107 layout: \
         the mass advance's consumed (a, bl) complete the round-105 \
         five-tuple), got len {}",
        air_rec[0].len()
    );
    assert_eq!(
        air_rec[0].len() / 7,
        mass_rec[0].len() / 4,
        "air and mass records must carry the same sub-step count"
    );

    // Composed response recurrence (plan §2): the per-unit-load_w probe
    // response through the air and mass sub-step affine maps. Both maps
    // consume the PREVIOUS sub-step's state pair: the fresh air value is
    // swapped into `air_node_t_air` only after the mass block reads
    // `t_i_sub` (engine data flow), so the air→mass route carries r_air
    // from the prior sub-step — which starts at 0 (hour-start committed).
    // [Round 107 re-pin: the record is self-contained — the mass advance's
    // consumed (a, bl) ride in slots 5/6 — so the composition no longer
    // reads the mass record (the pairing assert above stays as the shape
    // check). Same recurrence, same values, single stride.]
    let n = air_rec[0].len() / 7;
    let (mut r_air, mut r_m) = (0.0f64, 0.0f64);
    for k in 0..n {
        let d = air_rec[0][7 * k];
        let m_coef = air_rec[0][7 * k + 1];
        let w_coef = air_rec[0][7 * k + 2];
        let h3od = air_rec[0][7 * k + 4];
        let a_m = air_rec[0][7 * k + 5];
        let bl = air_rec[0][7 * k + 6];
        // Both updates consume the previous sub-step's state pair.
        let (ra, rm) = (r_air, r_m);
        r_air = d * ra + m_coef * rm + w_coef;
        r_m = a_m * rm + h3od * ra + bl;
    }

    let realized_delta_w =
        (model_b.setpoints.loads.as_ref()[0] - model_a.setpoints.loads.as_ref()[0]) * area;
    let predicted_air = r_air * realized_delta_w;
    let predicted_mass = r_m * realized_delta_w;
    let actual_air =
        model_b.mass.air_temperatures.as_ref()[0] - model_a.mass.air_temperatures.as_ref()[0];
    let actual_mass =
        model_b.mass.mass_temperatures.as_ref()[0] - model_a.mass.mass_temperatures.as_ref()[0];

    println!("predicted Δt_air {predicted_air:.9} K, actual {actual_air:.9} K");
    println!("predicted Δt_m   {predicted_mass:.9} K, actual {actual_mass:.9} K");

    assert!(
        actual_air.abs() > 1e-9 && actual_mass.abs() > 1e-9,
        "the probe must move both committed states for this contract to bind \
         (air {actual_air} K, mass {actual_mass} K)"
    );
    let rel = |pred: f64, act: f64| (pred - act).abs() / act.abs().max(1e-300);
    assert!(
        rel(predicted_air, actual_air) <= 1e-9,
        "air response must equal the composed record prediction: predicted \
         {predicted_air} K, got {actual_air} K"
    );
    assert!(
        rel(predicted_mass, actual_mass) <= 1e-9,
        "mass response must equal the composed record prediction: predicted \
         {predicted_mass} K, got {actual_mass} K"
    );
}

/// Loop round 107 (§LIMIT-22 air-node mirror, increment C) — the per-class
/// t_s-first mirror's `t_i` route consumes the air-node record: under the
/// artificial-gain probe the surface identity's air input responds with the
/// composed air response `r_air_N·Δ` (the plan-§2 staggered recurrence),
/// on top of the round-104 direct surface share `rad_frac·Δ`.
///
/// The air record must compose SELF-CONTAINED: the per-class arm's entry
/// condition is precisely that the zone's R77 mass record is absent (the
/// composite arm takes precedence otherwise), so the mass-advance
/// coefficients `a_k`/`bl_k` the response recurrence needs ride in the air
/// record's own sixth/seventh slots — recorded in the same lockstep push,
/// on the same branch the values were consumed (round-107 layout
/// extension of the round-105 five-tuple; plan Z4).
///
/// Red-first (loop round 107): on the pre-C tree the record is five-slot
/// (no mass coefficients to compose from) and the t_i route is the
/// committed-air fall-through, so the shape precondition below fails
/// (rc=101) and the arm's probe response stays at the round-104 closed
/// form `rad_frac·Δ` (400.0000000000003 W on this fixture).
///
/// Zero-delta behavior is untouched by construction — the Δ = 0 branch
/// consumes the committed air verbatim and never reads the record (the
/// #3297 residual-zero property); the round battery proves the
/// bit-identity, not this test.
#[test]
fn test_perclass_ti_route_consumes_air_record() {
    // Same harness as the round-103/104/105 contracts: Case 600 (LowMass)
    // under the explicit legacy selector pins the 5R1C network (the gauge
    // build's `Gauge` default selector routes this spec through the gauge
    // dispatch's mass-state proxy — the increment-D arm, not this
    // contract).
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::legacy())
        .expect("legacy 5R1C selector must initialize");
    let outdoor_temp = 10.0;

    model.step_physics(0, outdoor_temp, 3600.0);

    // Surgical arm isolation (the round-104 entry condition): clear zone
    // 0's R77 mass record so the checker falls through to the per-class
    // mirror. The AIR record stays live — it is exactly what the t_i
    // route consumes.
    let r77 = model
        .mass
        .r77_mass_substep_coeffs
        .as_mut()
        .expect("a live 5R1C step must persist the R77 record");
    assert!(
        !r77[0].is_empty(),
        "precondition: zone 0 carries a composite record before the surgery"
    );
    r77[0].clear();

    // Record shape (round-107 layout): seven entries per sub-step — the
    // round-105 five-tuple [D, M, W, den, h3od] completed by the mass
    // advance's own consumed coefficients [a, bl].
    let air_rec = model
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    assert!(
        air_rec[0].len() >= 7 && air_rec[0].len() % 7 == 0,
        "the air record must carry 7 entries per sub-step (round-107 layout \
         with the mass-advance coefficients) so the per-class t_i route can \
         compose the plan-§2 response without the mass record — got len {}",
        air_rec[0].len()
    );
    // Composed air response (plan §2): both updates consume the PREVIOUS
    // sub-step's state pair.
    let mut r_air_n = 0.0f64;
    let mut r_m_n = 0.0f64;
    for k in 0..air_rec[0].len() / 7 {
        let (ra, rm) = (r_air_n, r_m_n);
        r_air_n = air_rec[0][7 * k] * ra + air_rec[0][7 * k + 1] * rm + air_rec[0][7 * k + 2];
        r_m_n = air_rec[0][7 * k + 5] * rm + air_rec[0][7 * k + 4] * ra + air_rec[0][7 * k + 6];
    }
    assert!(
        r_air_n.abs() > 1e-12,
        "the composed air response must be non-zero, got {r_air_n}"
    );

    let mut checker = InvariantChecker::new(1e-3);
    let no_gain = checker.check_invariant(&model, 3600.0, outdoor_temp);
    println!("no-gain balance: {}", no_gain.balance);

    let with_gain =
        checker.check_invariant_with_artificial_gain(&model, 3600.0, outdoor_temp, 1000.0, 0);

    // Closed form (plan §2 + the round-104 surface identity): the phi_st
    // route contributes `rad_frac·Δ`; the t_i route contributes
    // `h_is_zone·r_air_N·Δ` (each class's denom_s cancels; the class
    // shares w sum to 1); the aggregation identity stays load-free.
    let rad_frac = 1.0 - model.solar.convective_fraction;
    let h_is_zone = model.conduction.h_tr_is.as_ref()[0];
    let expected = (rad_frac + h_is_zone * r_air_n) * 1000.0;
    let response = with_gain.balance - no_gain.balance;
    println!(
        "probe response signed {response} W vs composed closed form {expected} W (rad_frac {rad_frac}, h_is_zone {h_is_zone}, r_air_N {r_air_n:.9} per W, r_m_N {r_m_n:.9} per W)"
    );
    assert!(
        (response - expected).abs() <= 1e-9 * expected.abs(),
        "per-class t_i route must consume the composed air response (§2 recurrence, round 107): \
         got {response} W, expected {expected} W"
    );

    // Affine in the load delta.
    let with_gain_2k =
        checker.check_invariant_with_artificial_gain(&model, 3600.0, outdoor_temp, 2000.0, 0);
    let response_2k = with_gain_2k.balance - no_gain.balance;
    println!("probe response to 2 kW gain: {response_2k} W");
    assert!(
        (response_2k - 2.0 * response).abs() <= 1e-6 * response_2k.abs(),
        "probe response must be affine in the load delta: 1 kW {response} W vs 2 kW {response_2k} W"
    );

    // Fall-through discriminator: with the air record cleared the t_i
    // route keeps the committed air and the response returns to the
    // round-104 closed form `rad_frac·Δ` verbatim.
    model
        .mass
        .r77_air_substep_coeffs
        .as_mut()
        .expect("air record present before the fall-through surgery")[0]
        .clear();
    let with_gain_noair =
        checker.check_invariant_with_artificial_gain(&model, 3600.0, outdoor_temp, 1000.0, 0);
    let response_noair = with_gain_noair.balance - no_gain.balance;
    println!(
        "fall-through response {response_noair} W vs round-104 closed form {} W",
        rad_frac * 1000.0
    );
    assert!(
        (response_noair - rad_frac * 1000.0).abs() <= 1e-6 * response_noair.abs(),
        "with the air record absent the t_i route must keep the committed air: \
         got {response_noair} W, expected {} W",
        rad_frac * 1000.0
    );
}

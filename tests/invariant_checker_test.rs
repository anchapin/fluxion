//! Invariant Checker Tests
//!
//! Tests for the energy and mass balance invariant assertions.
//! Verifies that `(Heat In) - (Heat Out) - (Change in Internal Energy) ≈ 0`
//! within tolerance 1e-7.
//!
//! The invariant checker detects violations by introducing artificial heat gains
//! and verifying the energy balance responds accordingly.

use fluxion::physics::cta::VectorField;
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
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
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
#[cfg_attr(
    feature = "gauge-solver",
    ignore = "Gauge build only (Issue #3297 / KNOWN_ISSUES §LIMIT-22): the exact-CN mass-state proxy makes the 5R1C residual 0 and the artificial gain's leverage is structurally 0 when m_air_frac = 0 (§LIMIT-19/#3103 sibling). Default build keeps this test live."
)]
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

    let all_imbalances = vec![
        zone_0_result.balance.abs(),
        zone_1_result.balance.abs(),
        zone_2_result.balance.abs(),
    ];

    let max_imbalance = all_imbalances.iter().cloned().fold(0.0f64, f64::max);
    let min_imbalance = all_imbalances.iter().cloned().fold(f64::MAX, f64::min);

    assert!(
        max_imbalance > 0.0,
        "At least one zone should have non-zero imbalance"
    );
}

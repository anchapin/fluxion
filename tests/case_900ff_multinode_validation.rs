//! Thin replacement for `tests/case_900ff_multinode_validation.rs` (Issue #2877).
//!
//! The previous test exercised a stand-alone `MultiNodeHvacRunner` (now
//! `DeprecatedMultiNodeHvacRunner`) that duplicated the thermal-solver state and
//! was never on the production code path. It has been replaced with a thin
//! regression check that runs Case 900FF through the production 9R4C physics
//! path: `ThermalModel::step_physics(...)`, which dispatches to
//! `thermal_model_physics::physics_impl::step_physics_9r4c` for high-mass
//! constructions (see ADR-002).
//!
//! Two assertions only:
//!   1. The dispatcher actually routes Case 900FF to the 9R4C solver (no
//!      silent regression to 5R1C).
//!   2. The zone temperatures stay within a physically reasonable range
//!      across a full year of hourly timesteps in Denver TMY weather.
//!
//! Deeper ASHRAE 140 Case 900FF regression coverage lives in
//! `tests/case_900_multinode_validation.rs` (production 9R4C path) and
//! `tests/zone_balance_eplus_isolation.rs`. This file exists solely so the
//! issue #2877 acceptance criterion (a thin test against `step_physics_9r4c`)
//! is met.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Run Case 900FF for one year on the production 9R4C path and return
/// (min_zone_temp, max_zone_temp). Mirrors the helper of the same name in
/// `tests/case_900_multinode_validation.rs` so this file stays a true "thin
/// replacement".
fn simulate_case_900ff_step_physics_9r4c() -> (f64, f64) {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        let _ = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if let Some(&t) = model.temperatures.as_slice().first() {
            if t < min_temp {
                min_temp = t;
            }
            if t > max_temp {
                max_temp = t;
            }
        }
    }

    (min_temp, max_temp)
}

/// Test: Case 900FF is dispatched to the 9R4C physics path, not 5R1C.
///
/// `step_physics_9r4c` is reached only when the model reports
/// `is_nine_r4c_model() == true`. Case 900FF is a high-mass construction,
/// so the dispatcher must select the 9R4C branch on the very first
/// `step_physics` call.
#[test]
fn test_case_900ff_dispatches_to_step_physics_9r4c() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    assert!(
        model.is_nine_r4c_model(),
        "Case 900FF (HighMass) must route through step_physics_9r4c; \
         is_nine_r4c_model() returned false — the dispatcher is regressed \
         and would silently fall back to 5R1C. See Issue #2877 / ADR-002."
    );
}

/// Test: Case 900FF stays in a physically reasonable temperature range across
/// a full year of hourly timesteps on the production 9R4C path.
///
/// The exact ASHRAE 140 reference range is exercised in depth by
/// `tests/case_900_multinode_validation.rs`; here we only pin that the
/// dispatched 9R4C solver produces finite, physically reasonable zone
/// temperatures. This is the "thin" acceptance test called out by the
/// issue description.
#[test]
fn test_case_900ff_step_physics_9r4c_stays_finite_and_physical() {
    let (min_temp, max_temp) = simulate_case_900ff_step_physics_9r4c();

    println!("\n=== Case 900FF via step_physics_9r4c (Issue #2877 thin test) ===");
    println!("Min zone temperature: {min_temp:.2} C");
    println!("Max zone temperature: {max_temp:.2} C");

    assert!(
        min_temp.is_finite() && max_temp.is_finite(),
        "step_physics_9r4c produced a non-finite zone temperature \
         (min={min_temp}, max={max_temp}); the 9R4C path is regressed."
    );

    // High-mass building, free-floating, full-year hourly simulation in Denver
    // TMY weather: physically reasonable zone temperatures must fall well
    // inside [-30, 80] C. The wider ±15% ASHRAE 140 acceptance check is in
    // tests/case_900_multinode_validation.rs.
    assert!(
        min_temp > -30.0 && min_temp < 30.0,
        "Case 900FF min zone temperature {min_temp:.2} C is outside the \
         physically reasonable range [-30, 30] C"
    );
    assert!(
        max_temp > 0.0 && max_temp < 80.0,
        "Case 900FF max zone temperature {max_temp:.2} C is outside the \
         physically reasonable range [0, 80] C"
    );

    // Min must be strictly less than max (zone moves with weather).
    assert!(
        min_temp < max_temp,
        "Min zone temperature {min_temp:.2} C must be below max \
         {max_temp:.2} C"
    );
}

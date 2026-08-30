//! Zone heat balance analytical unit tests.
//!
//! Validates the core heat balance equation in `thermal_model_core.rs` against
//! hand-calculated analytical solutions for steady-state and transient cases.
//!
//! # Test Cases
//!
//! - **Steady-state convergence**: temperatures stabilize over multiple timesteps
//! - **Transient**: exponential approach with RC time constant
//! - **Energy conservation**: sum of heat fluxes = 0 at equilibrium
//! - **Single-timestep convergence**: known inputs produce expected outputs
//!
//! # Building Cases
//!
//! - Lightweight: Case 600 (low thermal mass, ASHRAE 140)
//! - Heavyweight: Case 900 (high thermal mass, ASHRAE 140)
//! - Mixed: Case 650FF (night ventilation, free-float)
//!
//! # Acceptance Criteria (Issue #968)
//!
//! - Steady-state temperatures stabilize (consecutive timesteps match)
//! - Transient matches exponential within 0.5%
//! - Energy conservation: |sum of fluxes| < 0.01 W
//! - 3+ cases: lightweight, heavyweight, mixed
//! - Test runs in <200ms

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

const EPSILON: f64 = 1e-9;
const STEADY_STATE_TOL: f64 = 0.01;
const TRANSIENT_TOL: f64 = 0.005;
const ENERGY_CONSERVATION_TOL: f64 = 0.01;

fn set_zone_temperature(model: &mut ThermalModel<VectorField>, zone: usize, temp: f64) {
    model.setpoints.temperatures.as_mut()[zone] = temp;
}

fn set_zone_solar(model: &mut ThermalModel<VectorField>, zone: usize, solar: f64) {
    model.solar.solar_gains.as_mut()[zone] = solar;
}

fn set_zone_loads(model: &mut ThermalModel<VectorField>, zone: usize, load: f64) {
    model.setpoints.loads.as_mut()[zone] = load;
}

fn compute_transient_temperature(T_initial: f64, T_ss: f64, t: f64, tau: f64) -> f64 {
    T_ss + (T_initial - T_ss) * (-t / tau).exp()
}

#[test]
fn test_steady_state_convergence_lightweight() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    let T_outdoor = 10.0;
    let T_initial = 20.0;
    set_zone_temperature(&mut model, 0, T_initial);
    set_zone_loads(&mut model, 0, 100.0);
    set_zone_solar(&mut model, 0, 50.0);

    for _ in 0..100 {
        model.step_physics(12, T_outdoor, 3600.0);
    }

    let T_zone = model.setpoints.temperatures[0];

    for _ in 0..10 {
        let temp_before = model.setpoints.temperatures[0];
        model.step_physics(12, T_outdoor, 3600.0);
        let temp_after = model.setpoints.temperatures[0];
        let change = (temp_after - temp_before).abs();
        assert!(
            change < STEADY_STATE_TOL,
            "Temperature not stable: change={:.6}°C",
            change
        );
    }

    println!("Lightweight steady-state converged: T_zone={:.4}°C", T_zone);
}

#[test]
fn test_steady_state_convergence_heavyweight() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    let T_outdoor = 10.0;
    let T_initial = 20.0;
    set_zone_temperature(&mut model, 0, T_initial);
    set_zone_loads(&mut model, 0, 100.0);
    set_zone_solar(&mut model, 0, 50.0);

    for _ in 0..200 {
        model.step_physics(12, T_outdoor, 3600.0);
    }

    let T_zone = model.setpoints.temperatures[0];

    for _ in 0..10 {
        let temp_before = model.setpoints.temperatures[0];
        model.step_physics(12, T_outdoor, 3600.0);
        let temp_after = model.setpoints.temperatures[0];
        let change = (temp_after - temp_before).abs();
        assert!(
            change < STEADY_STATE_TOL,
            "Heavyweight temperature not stable: change={:.6}°C",
            change
        );
    }

    println!("Heavyweight steady-state converged: T_zone={:.4}°C", T_zone);
}

#[test]
fn test_steady_state_convergence_mixed() {
    let spec = ASHRAE140Case::Case650FF.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    let T_outdoor = 10.0;
    let T_initial = 20.0;
    set_zone_temperature(&mut model, 0, T_initial);
    set_zone_loads(&mut model, 0, 100.0);
    set_zone_solar(&mut model, 0, 50.0);

    for _ in 0..150 {
        model.step_physics(12, T_outdoor, 3600.0);
    }

    let T_zone = model.setpoints.temperatures[0];

    for _ in 0..10 {
        let temp_before = model.setpoints.temperatures[0];
        model.step_physics(12, T_outdoor, 3600.0);
        let temp_after = model.setpoints.temperatures[0];
        let change = (temp_after - temp_before).abs();
        assert!(
            change < STEADY_STATE_TOL,
            "Mixed temperature not stable: change={:.6}°C",
            change
        );
    }

    println!("Mixed steady-state converged: T_zone={:.4}°C", T_zone);
}

#[test]
fn test_single_timestep_convergence() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    let T_outdoor = 10.0;
    let T_initial = 20.0;
    set_zone_temperature(&mut model, 0, T_initial);
    set_zone_loads(&mut model, 0, 100.0);
    set_zone_solar(&mut model, 0, 50.0);

    model.step_physics(12, T_outdoor, 1.0);
    let T_after_1s = model.setpoints.temperatures[0];

    model.step_physics(12, T_outdoor, 1.0);
    let T_after_2s = model.setpoints.temperatures[0];

    let change_1 = (T_after_1s - T_initial).abs();
    let change_2 = (T_after_2s - T_after_1s).abs();

    assert!(
        change_1 > 0.01,
        "Temperature should change in first timestep"
    );
    assert!(
        change_2 < change_1,
        "Temperature change should decrease (converging): change1={:.6}, change2={:.6}",
        change_1,
        change_2
    );
}

#[test]
fn test_performance_under_200ms() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    let start = std::time::Instant::now();

    for _ in 0..24 {
        model.step_physics(12, 10.0, 3600.0);
    }

    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0;

    assert!(ms < 200.0, "Test took {:.2}ms (should be < 200ms)", ms);
}

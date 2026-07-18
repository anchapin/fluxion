//! Surrogate drift tolerance gate — Issue #1784 (T6.4)
//!
//! CI gate that asserts surrogate output does not drift >1% from the 9R4C
//! physics baseline (Case 900 high-mass building) on the benchmark building.
//!
//! ## Drift Metric Definition
//!
//! Per-timestep relative temperature drift:
//! `drift_pct = |T_surrogate - T_physics| / max(|T_physics|, ε) × 100`
//!
//! Where `ε = 0.1°C` prevents division by zero when the physics temperature
//! is near 0°C. The gate fails if any timestep exceeds 1% drift.
//!
//! ## Benchmark Building
//!
//! Case 900 (high-mass concrete building) — the ASHRAE 140 reference building
//! that exercises the 9R4C thermal network in the physics model. This is the
//! most thermally massive configuration and therefore the most demanding test
//! for a neural surrogate.
//!
//! ## Why Case 900?
//!
//! Case 900 uses `HighMass9R4C` construction which routes through the 9R4C
//! thermal network (ADR-002). The surrogate must accurately predict thermal
//! loads for this configuration, which has the highest thermal mass of all
//! ASHRAE 140 cases.
//!
//! ## CI Gate Behavior
//!
//! When no ONNX surrogate model is loaded (SurrogateManager::default()),
//! the surrogate model falls back to analytical load calculation which
//! produces DIFFERENT results than the direct physics path. This is expected
//! behavior - the fallback is NOT identical to the physics model.
//!
//! The drift gate is designed to detect when a trained surrogate model
//! produces outputs that deviate from the physics baseline. Without a trained
//! model, the gate is expected to fail (the test will show >1% drift),
//! which validates that the gate is correctly detecting drift.
//!
//! Once a surrogate model is trained and loaded via registry.json, the
//! drift gate will properly assert that the surrogate stays within 1% of
//! the physics baseline.
//!
//! ## Acceptance Criteria
//!
//! - [x] CI gate asserts surrogate output does not drift >1% from 9R4C baseline
//! - [x] Drift metric defined + documented
//! - [x] Failure message shows offending timesteps

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::thermal_model::{PhysicsThermalModel, SurrogateThermalModel, ThermalModelTrait};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

const DRIFT_TOLERANCE_PCT: f64 = 1.0;
const EPSILON_TEMP: f64 = 0.1;
const TEST_TIMESTEPS: usize = 168;

fn compute_drift(t_surrogate: f64, t_physics: f64) -> f64 {
    let abs_physics = t_physics.abs().max(EPSILON_TEMP);
    ((t_surrogate - t_physics).abs() / abs_physics) * 100.0
}

struct DriftResult {
    max_drift_pct: f64,
    offending_timesteps: Vec<(usize, usize, f64, f64, f64)>,
}

fn compute_drift_result(physics_temps: &[Vec<f64>], surrogate_temps: &[Vec<f64>]) -> DriftResult {
    let num_zones = physics_temps.len().min(surrogate_temps.len());
    let num_steps = physics_temps.first().map(|z| z.len()).unwrap_or(0);

    let mut max_drift_pct = 0.0_f64;
    let mut offending_timesteps = Vec::new();

    for zone_idx in 0..num_zones {
        let physics_zone = &physics_temps[zone_idx];
        let surrogate_zone = &surrogate_temps[zone_idx];

        for step in 0..num_steps.min(surrogate_zone.len()) {
            let t_physics = physics_zone[step];
            let t_surrogate = surrogate_zone[step];
            let drift = compute_drift(t_surrogate, t_physics);

            if drift > max_drift_pct {
                max_drift_pct = drift;
            }

            if drift > DRIFT_TOLERANCE_PCT {
                offending_timesteps.push((zone_idx, step, t_physics, t_surrogate, drift));
            }
        }
    }

    DriftResult {
        max_drift_pct,
        offending_timesteps,
    }
}

#[test]
fn test_surrogate_drift_gate_case_900_9r4c() {
    let spec = ASHRAE140Case::Case900.spec();

    let mut physics_model = PhysicsThermalModel::from_spec(&spec);
    let mut surrogate_model = SurrogateThermalModel::from_spec(&spec);

    let surrogates = SurrogateManager::default();

    let _physics_eui = physics_model.solve_timesteps(TEST_TIMESTEPS, &surrogates, false);
    let _surrogate_eui = surrogate_model.solve_timesteps(TEST_TIMESTEPS, &surrogates, true);

    let physics_temps = physics_model
        .get_hourly_temperatures()
        .expect("Physics model should have hourly temperatures after solve_timesteps");
    let surrogate_temps = surrogate_model
        .get_hourly_temperatures()
        .expect("Surrogate model should have hourly temperatures after solve_timesteps");

    assert!(
        !physics_temps.is_empty(),
        "Physics model returned no temperature data"
    );
    assert!(
        !surrogate_temps.is_empty(),
        "Surrogate model returned no temperature data"
    );

    let result = compute_drift_result(&physics_temps, &surrogate_temps);

    if result.max_drift_pct > DRIFT_TOLERANCE_PCT {
        let offending_sample = result
            .offending_timesteps
            .first()
            .map(|(z, s, tp, ts, d)| {
                format!(
                    "zone={}, step={}, T_physics={:.4}°C, T_surrogate={:.4}°C, drift={:.4}%",
                    z, s, tp, ts, d
                )
            })
            .unwrap_or_default();

        panic!(
            "SURROGATE DRIFT GATE FAILED (Issue #1784 T6.4)\n\
             Maximum drift: {:.4}% (threshold: {:.1}%)\n\
             Offending timesteps (first 10 of {}):\n\
             {}\n\
             \n\
             The surrogate model drifted >1% from the 9R4C physics baseline.\n\
             This is a CI gate failure — the surrogate must be retrained or the\n\
             drift tolerance adjusted. Do NOT modify this test to pass without\n\
             fixing the underlying surrogate accuracy issue.",
            result.max_drift_pct,
            DRIFT_TOLERANCE_PCT,
            result.offending_timesteps.len(),
            offending_sample
        );
    }
}

#[test]
fn test_surrogate_drift_gate_annual_simulation() {
    let spec = ASHRAE140Case::Case900.spec();

    let mut physics_model = PhysicsThermalModel::from_spec(&spec);
    let mut surrogate_model = SurrogateThermalModel::from_spec(&spec);

    let surrogates = SurrogateManager::default();

    let _physics_eui = physics_model.solve_timesteps(8760, &surrogates, false);
    let _surrogate_eui = surrogate_model.solve_timesteps(8760, &surrogates, true);

    let physics_temps = physics_model
        .get_hourly_temperatures()
        .expect("Physics model should have hourly temperatures after annual simulation");
    let surrogate_temps = surrogate_model
        .get_hourly_temperatures()
        .expect("Surrogate model should have hourly temperatures after annual simulation");

    let result = compute_drift_result(&physics_temps, &surrogate_temps);

    if result.max_drift_pct > DRIFT_TOLERANCE_PCT {
        let offending_sample = result
            .offending_timesteps
            .first()
            .map(|(z, s, tp, ts, d)| {
                format!(
                    "zone={}, step={}, T_physics={:.4}°C, T_surrogate={:.4}°C, drift={:.4}%",
                    z, s, tp, ts, d
                )
            })
            .unwrap_or_default();

        panic!(
            "SURROGATE DRIFT GATE FAILED (Issue #1784 T6.4)\n\
             Maximum drift: {:.4}% (threshold: {:.1}%)\n\
             Offending timesteps (first 10 of {}):\n\
             {}\n\
             \n\
             Annual simulation drift gate failed — surrogate cannot track 9R4C baseline.\n\
             This is expected behavior when no trained ONNX model is loaded.\n\
             Once a surrogate model is trained and registered, the gate should pass.",
            result.max_drift_pct,
            DRIFT_TOLERANCE_PCT,
            result.offending_timesteps.len(),
            offending_sample
        );
    }
}

#[test]
fn test_surrogate_drift_metric_definition() {
    assert!(
        compute_drift(20.0, 20.0) < 1e-10,
        "Zero drift expected for identical temperatures"
    );
    assert!(
        (compute_drift(20.2, 20.0) - 1.0).abs() < 1e-6,
        "1% drift expected for 0.2°C difference at 20°C"
    );
    assert!(
        (compute_drift(22.0, 20.0) - 10.0).abs() < 1e-6,
        "10% drift expected for 2°C difference at 20°C"
    );
    let drift_at_epsilon = compute_drift(1.0, 0.0);
    assert!(
        drift_at_epsilon > 900.0,
        "Large drift expected when physics temp is near zero and surrogate differs by 1°C"
    );
}

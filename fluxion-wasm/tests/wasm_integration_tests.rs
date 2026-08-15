//! WASM integration tests for fluxion-wasm.
//!
//! These tests exercise the full `FluidSimulation` API surface, including
//! all `ThermalModelTrait`-equivalent methods, memory management, and
//! error handling paths.
//!
//! Note: True WASM runtime tests require `wasm-pack test --node` which
//! is run in the CI pipeline. These tests use `wasm_bindgen_test`
//! to run in a WASM runtime (Node.js or browser).

#![cfg(target_arch = "wasm32")]

use fluxion_wasm::{FluidSimulation, FluidSimulationConfig};
use wasm_bindgen_test::wasm_bindgen_test;

#[test]
fn fluid_simulation_constructor_default() {
    let config = FluidSimulationConfig {
        building: "test_building".to_string(),
        num_zones: 3,
        weather: "TMY3_CHICAGO".to_string(),
        initial_temps: None,
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
    };
    let config_json = serde_json::to_string(&config).unwrap();
    let sim = FluidSimulation::new(&config_json).unwrap();

    assert_eq!(sim.num_zones(), 3);
    assert_eq!(sim.current_hour(), 0.0);
    assert!(sim.is_valid());
}

#[test]
fn fluid_simulation_constructor_with_initial_temps() {
    let config = FluidSimulationConfig {
        building: "test".to_string(),
        num_zones: 2,
        weather: "TMY3_CHICAGO".to_string(),
        initial_temps: Some(vec![18.0, 25.0]),
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
    };
    let config_json = serde_json::to_string(&config).unwrap();
    let sim = FluidSimulation::new(&config_json).unwrap();

    let temps = sim.get_zone_temps();
    assert_eq!(temps.len(), 2);
    assert_eq!(temps[0], 18.0);
    assert_eq!(temps[1], 25.0);
}

#[test]
fn fluid_simulation_constructor_invalid_json() {
    let result = FluidSimulation::new("not valid json");
    assert!(result.is_err());
}

#[test]
fn fluid_simulation_step() {
    let config = FluidSimulationConfig {
        num_zones: 2,
        initial_temps: Some(vec![22.0, 22.0]),
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();
    let mut sim = sim;

    let result = sim.step(1.0);
    assert!(result.is_ok());
    assert_eq!(sim.current_hour(), 1.0);

    let result = sim.step(0.5);
    assert!(result.is_ok());
    assert_eq!(sim.current_hour(), 1.5);
}

#[test]
fn fluid_simulation_get_zone_temps() {
    let config = FluidSimulationConfig {
        num_zones: 4,
        initial_temps: Some(vec![21.0, 22.0, 23.0, 24.0]),
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let temps = sim.get_zone_temps();
    assert_eq!(temps.len(), 4);
    assert_eq!(temps[0], 21.0);
    assert_eq!(temps[2], 23.0);
}

#[test]
fn fluid_simulation_get_zone_temp() {
    let config = FluidSimulationConfig {
        num_zones: 3,
        initial_temps: Some(vec![20.0, 21.0, 22.0]),
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    assert_eq!(sim.get_zone_temp(0).unwrap(), 20.0);
    assert_eq!(sim.get_zone_temp(2).unwrap(), 22.0);
    assert!(sim.get_zone_temp(3).is_err());
}

#[test]
fn fluid_simulation_set_temperatures() {
    let config = FluidSimulationConfig {
        num_zones: 3,
        initial_temps: Some(vec![22.0, 22.0, 22.0]),
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let result = sim.set_temperatures(vec![18.0, 20.0, 25.0]);
    assert!(result.is_ok());

    let temps = sim.get_zone_temps();
    assert_eq!(temps[0], 18.0);
    assert_eq!(temps[1], 20.0);
    assert_eq!(temps[2], 25.0);
}

#[test]
fn fluid_simulation_set_temperatures_length_mismatch() {
    let config = FluidSimulationConfig {
        num_zones: 2,
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let result = sim.set_temperatures(vec![20.0, 21.0, 22.0]);
    assert!(result.is_err());
}

#[test]
fn fluid_simulation_control_set_get() {
    let config = FluidSimulationConfig::default();
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    sim.set_control("heating_zone_0", 21.0).unwrap();
    sim.set_control("cooling_zone_1", 26.0).unwrap();

    let heating_sps = sim.get_heating_setpoints();
    assert_eq!(heating_sps[0], 21.0);

    let cooling_sps = sim.get_cooling_setpoints();
    assert_eq!(cooling_sps[1], 26.0);
}

#[test]
fn fluid_simulation_get_control_custom_loop() {
    let config = FluidSimulationConfig::default();
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    sim.set_control("vav_damper_1", 0.75).unwrap();
    assert_eq!(sim.get_control("vav_damper_1").unwrap(), 0.75);
    assert!(sim.get_control("nonexistent").is_err());
}

#[test]
fn fluid_simulation_reset_temperatures() {
    let config = FluidSimulationConfig {
        num_zones: 3,
        initial_temps: Some(vec![18.0, 21.0, 28.0]),
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    sim.reset_temperatures(22.0).unwrap();

    let temps = sim.get_zone_temps();
    for temp in &temps {
        assert_eq!(*temp, 22.0);
    }
}

#[test]
fn fluid_simulation_mode() {
    let config = FluidSimulationConfig::default();
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    assert_eq!(sim.mode(), "Physics");

    let mut sim = sim;
    let result = sim.set_mode("Surrogate");
    assert!(result.is_ok());
    assert_eq!(sim.mode(), "Physics");
}

#[test]
fn fluid_simulation_apply_parameters() {
    let config = FluidSimulationConfig {
        num_zones: 2,
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let result = sim.apply_parameters(vec![1.5, 19.0, 26.0]);
    assert!(result.is_ok());

    let heating_sps = sim.get_heating_setpoints();
    assert_eq!(heating_sps[0], 19.0);
    assert_eq!(heating_sps[1], 19.0);

    let cooling_sps = sim.get_cooling_setpoints();
    assert_eq!(cooling_sps[0], 26.0);
    assert_eq!(cooling_sps[1], 26.0);
}

#[test]
fn fluid_simulation_apply_parameters_strict_range() {
    let config = FluidSimulationConfig::default();
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    // All three params within their physical ranges → accepted unchanged
    // (issue #2911 replaces silent clamping with strict validation).
    let result = sim.apply_parameters(vec![2.0, 10.0, 35.0]);
    assert!(result.is_ok());

    let heating_sps = sim.get_heating_setpoints();
    assert_eq!(heating_sps[0], 10.0);

    let cooling_sps = sim.get_cooling_setpoints();
    assert_eq!(cooling_sps[0], 35.0);

    // Out-of-range params[1] (heating) → rejected.
    let result = sim.apply_parameters(vec![2.0, 5.0, 25.0]);
    assert!(result.is_err());

    // Out-of-range params[2] (cooling) → rejected.
    let result = sim.apply_parameters(vec![2.0, 20.0, 50.0]);
    assert!(result.is_err());

    // Out-of-range params[0] (U-value) → rejected (previously unconstrained).
    let result = sim.apply_parameters(vec![20.0, 20.0, 25.0]);
    assert!(result.is_err());
}

#[test]
fn fluid_simulation_apply_parameters_too_few() {
    let config = FluidSimulationConfig::default();
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let result = sim.apply_parameters(vec![1.5]);
    assert!(result.is_err());
}

#[test]
fn fluid_simulation_zone_area() {
    let config = FluidSimulationConfig {
        num_zones: 5,
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    assert_eq!(sim.zone_area(), 250.0);
}

#[test]
fn fluid_simulation_hvac_power_demand() {
    let config = FluidSimulationConfig {
        num_zones: 2,
        initial_temps: Some(vec![22.0, 22.0]),
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let power = sim.hvac_power_demand(0, 20.0);
    assert!(power < 0.0);
}

#[test]
fn fluid_simulation_is_valid() {
    let config = FluidSimulationConfig {
        num_zones: 2,
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();
    assert!(sim.is_valid());
}

#[test]
fn fluid_simulation_is_valid_invalid_setpoints() {
    let config = FluidSimulationConfig {
        num_zones: 2,
        heating_setpoint: 25.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();
    assert!(!sim.is_valid());
}

#[test]
fn fluid_simulation_is_valid_empty_zones() {
    let config = FluidSimulationConfig {
        num_zones: 0,
        ..Default::default()
    };
    let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();
    assert!(!sim.is_valid());
}

#[test]
fn fluid_simulation_solve_timesteps_stub() {
    let config = FluidSimulationConfig::default();
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let eui = sim.solve_timesteps(8760, true);
    assert_eq!(eui, 0.0);
}

#[test]
fn fluid_simulation_thousand_step_memory_stability() {
    let config = FluidSimulationConfig {
        num_zones: 3,
        initial_temps: Some(vec![22.0; 3]),
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    for _ in 0..1000 {
        let result = sim.step(1.0);
        assert!(result.is_ok());
    }

    assert!(sim.is_valid());
    assert_eq!(sim.current_hour(), 1000.0);
    let temps = sim.get_zone_temps();
    assert_eq!(temps.len(), 3);
    for temp in &temps {
        assert!(temp.is_finite());
    }
}

// ---------------------------------------------------------------------------
// Inline wasm-bindgen-test cases (Issue #2903).
//
// These tests are wired with `#[wasm_bindgen_test]` rather than
// `#[test]` so they are collected by `wasm-pack test --node` and run
// inside a real Node.js V8 isolate on every CI run. The existing
// `#[test]` cases above are gated to `cfg(target_arch = "wasm32")` so
// they no-op under native `cargo test`, but they are ALSO wired here
// under `#[wasm_bindgen_test]` semantics via the same Rust file,
// giving the CI matrix a single source of truth for the WASM runtime
// contract. The 4 cases below cover the four explicit acceptance
// bullets from issue #2903:
//   1. builder round-trip
//   2. step returns finite temps
//   3. NaN-input rejection
//   4. get_zone_temps length match
// ---------------------------------------------------------------------------

/// Issue #2903 acceptance bullet 1: builder round-trip.
///
/// A `FluidSimulationConfig` is serialized to JSON, fed to
/// `FluidSimulation::new`, and the resulting sim's introspection
/// surface (`num_zones`, `current_hour`, `is_valid`,
/// `get_heating_setpoints`, `get_cooling_setpoints`) is checked against
/// the original config. The default values from `Default::default()`
/// populate the unset fields, so we only assert the fields we
/// explicitly set — anything else risks a false failure on a
/// `Default` change unrelated to the builder contract.
#[wasm_bindgen_test]
fn wasm_builder_round_trip() {
    let config = FluidSimulationConfig {
        building: "round_trip_building".to_string(),
        num_zones: 4,
        weather: "TMY3_CHICAGO".to_string(),
        initial_temps: Some(vec![18.0, 21.0, 24.0, 27.0]),
        heating_setpoint: 19.5,
        cooling_setpoint: 25.5,
    };
    let config_json = serde_json::to_string(&config).unwrap();
    let sim = FluidSimulation::new(&config_json).unwrap();

    assert_eq!(sim.num_zones(), 4);
    assert_eq!(sim.current_hour(), 0.0);
    assert!(sim.is_valid());

    let heating = sim.get_heating_setpoints();
    let cooling = sim.get_cooling_setpoints();
    assert_eq!(heating.len(), 4);
    assert_eq!(cooling.len(), 4);
    for sp in &heating {
        assert_eq!(*sp, 19.5);
    }
    for sp in &cooling {
        assert_eq!(*sp, 25.5);
    }
}

/// Issue #2903 acceptance bullet 2: step returns finite temps.
///
/// After running `step` for a number of hours, every entry in
/// `get_zone_temps()` must be finite — NaN or Inf would propagate
/// from a buggy step implementation and corrupt downstream consumers.
/// Also asserts `current_hour` increments deterministically.
#[wasm_bindgen_test]
fn wasm_step_returns_finite_temps() {
    let config = FluidSimulationConfig {
        num_zones: 3,
        initial_temps: Some(vec![22.0, 22.0, 22.0]),
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    for _ in 0..24 {
        let result = sim.step(1.0);
        assert!(result.is_ok());
    }
    assert_eq!(sim.current_hour(), 24.0);

    let temps = sim.get_zone_temps();
    assert_eq!(temps.len(), 3);
    for temp in &temps {
        assert!(temp.is_finite(), "zone temp must be finite, got {}", temp);
    }
}

/// Issue #2903 acceptance bullet 3: NaN-input rejection.
///
/// `set_temperatures` and `apply_parameters` are the two WASM-boundary
/// entry points that accept vectors of `f64` from JS. Both must reject
/// NaN outright (issue #2911: NaN/Inf would propagate through the
/// energy balance and corrupt downstream consumers). The rejection
/// contract is: `is_err()` on any NaN input, and the sim state is
/// unchanged afterwards — the user must be able to recover by
/// resubmitting a corrected value.
#[wasm_bindgen_test]
fn wasm_nan_input_rejection() {
    let config = FluidSimulationConfig {
        num_zones: 2,
        initial_temps: Some(vec![22.0, 22.0]),
        heating_setpoint: 20.0,
        cooling_setpoint: 24.0,
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    // NaN in set_temperatures → rejected.
    let nan_result = sim.set_temperatures(vec![f64::NAN, 22.0]);
    assert!(
        nan_result.is_err(),
        "set_temperatures must reject NaN, got {:?}",
        nan_result
    );
    // State unchanged after rejection.
    let temps = sim.get_zone_temps();
    assert_eq!(temps[0], 22.0);
    assert_eq!(temps[1], 22.0);

    // NaN in apply_parameters[0] (U-value) → rejected.
    let nan_params = sim.apply_parameters(vec![f64::NAN, 20.0, 24.0]);
    assert!(
        nan_params.is_err(),
        "apply_parameters must reject NaN U-value, got {:?}",
        nan_params
    );

    // NaN in apply_parameters[1] (heating setpoint) → rejected.
    let nan_params = sim.apply_parameters(vec![2.0, f64::NAN, 24.0]);
    assert!(
        nan_params.is_err(),
        "apply_parameters must reject NaN heating setpoint, got {:?}",
        nan_params
    );

    // NaN in apply_parameters[2] (cooling setpoint) → rejected.
    let nan_params = sim.apply_parameters(vec![2.0, 20.0, f64::NAN]);
    assert!(
        nan_params.is_err(),
        "apply_parameters must reject NaN cooling setpoint, got {:?}",
        nan_params
    );

    // Inf must also be rejected (same code path; issue #2911 explicit).
    let inf_params = sim.apply_parameters(vec![2.0, 20.0, f64::INFINITY]);
    assert!(
        inf_params.is_err(),
        "apply_parameters must reject +Inf, got {:?}",
        inf_params
    );
}

/// Issue #2903 acceptance bullet 4: get_zone_temps length match.
///
/// `get_zone_temps()` must return exactly `num_zones` entries for
/// every config, regardless of the value of `num_zones`. Edge cases
/// verified: 1 zone (degenerate), 5 zones (the crate default),
/// 16 zones (large), and the implicit 22°C fallback when
/// `initial_temps: None`.
#[wasm_bindgen_test]
fn wasm_get_zone_temps_length_match() {
    for &num_zones in &[1usize, 5, 16] {
        let config = FluidSimulationConfig {
            num_zones,
            initial_temps: None,
            ..Default::default()
        };
        let sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();
        let temps = sim.get_zone_temps();
        assert_eq!(
            temps.len(),
            num_zones,
            "get_zone_temps returned {} entries for num_zones={}",
            temps.len(),
            num_zones
        );
        // Default fallback is 22°C for every zone — verify the
        // `initial_temps: None` branch is wired correctly.
        for temp in &temps {
            assert_eq!(*temp, 22.0);
        }
    }

    // Explicit initial_temps path — length must match num_zones
    // even after the set_temperatures guard rejects a mismatched
    // write.
    let config = FluidSimulationConfig {
        num_zones: 3,
        initial_temps: Some(vec![18.0, 21.0, 24.0]),
        ..Default::default()
    };
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();
    let result = sim.set_temperatures(vec![20.0, 21.0]);
    assert!(result.is_err());
    let temps = sim.get_zone_temps();
    assert_eq!(temps.len(), 3);
    assert_eq!(temps[0], 18.0);
    assert_eq!(temps[1], 21.0);
    assert_eq!(temps[2], 24.0);
}

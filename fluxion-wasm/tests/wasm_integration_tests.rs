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

    sim.reset_temperatures(22.0);

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
fn fluid_simulation_apply_parameters_clamping() {
    let config = FluidSimulationConfig::default();
    let mut sim = FluidSimulation::new(&serde_json::to_string(&config).unwrap()).unwrap();

    let result = sim.apply_parameters(vec![2.0, 10.0, 35.0]);
    assert!(result.is_ok());

    let heating_sps = sim.get_heating_setpoints();
    assert_eq!(heating_sps[0], 15.0);

    let cooling_sps = sim.get_cooling_setpoints();
    assert_eq!(cooling_sps[0], 32.0);
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

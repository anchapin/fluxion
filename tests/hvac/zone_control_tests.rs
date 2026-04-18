//! Integration tests for zone-level HVAC control.
//!
//! These tests validate the complete HVAC control system including
//! setpoint management, control logic, and energy calculations.

use fluxion::hvac::zone_control::{HVACStatus, ZoneControl};
use fluxion::hvac::zone_setpoints::ZoneSetpoints;
use fluxion::physics::cta::VectorField;
use fluxion::thermal::thermal_model::ThermalModel;
use std::sync::Arc;

#[test]
fn test_setpoint_validation() {
    let mut setpoints = ZoneSetpoints::new(3);

    // Test valid temperature ranges
    assert!(setpoints.set_heating_setpoint(0, 15.0).is_ok());
    assert!(setpoints.set_cooling_setpoint(0, 35.0).is_ok());

    // Test invalid temperature ranges
    assert!(setpoints.set_heating_setpoint(1, 5.0).is_err());
    assert!(setpoints.set_cooling_setpoint(1, 45.0).is_err());

    // Test valid deadband
    assert!(setpoints.set_deadband(0, 1.0).is_ok());
    assert!(setpoints.set_deadband(0, 4.0).is_ok());

    // Test invalid deadband
    assert!(setpoints.set_deadband(2, 0.0).is_err());
    assert!(setpoints.set_deadband(2, 6.0).is_err());

    // Test setpoint order validation
    setpoints.set_heating_setpoint(0, 25.0).unwrap();
    setpoints.set_cooling_setpoint(0, 23.0).unwrap();
    assert!(setpoints.validate_setpoints().is_err());
}

#[test]
fn test_heating_control() {
    let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
    let mut setpoints = ZoneSetpoints::new(1);
    setpoints.set_heating_setpoint(0, 22.0).unwrap();
    setpoints.set_cooling_setpoint(0, 26.0).unwrap();

    let mut zone_control = ZoneControl::new(thermal_model, setpoints);
    let current_temps = VectorField::from_scalar(19.0, 1); // Below heating setpoint

    let energy_input = zone_control.update_zone_controls(&current_temps);

    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);
    assert!(energy_input.as_slice()[0] > 0.0);
}

#[test]
fn test_cooling_control() {
    let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
    let mut setpoints = ZoneSetpoints::new(1);
    setpoints.set_heating_setpoint(0, 22.0).unwrap();
    setpoints.set_cooling_setpoint(0, 26.0).unwrap();

    let mut zone_control = ZoneControl::new(thermal_model, setpoints);
    let current_temps = VectorField::from_scalar(27.0, 1); // Above cooling setpoint

    let energy_input = zone_control.update_zone_controls(&current_temps);

    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Cooling);
    assert!(energy_input.as_slice()[0] > 0.0);
}

#[test]
fn test_deadband_control() {
    let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
    let mut setpoints = ZoneSetpoints::new(1);
    setpoints.set_heating_setpoint(0, 22.0).unwrap();
    setpoints.set_cooling_setpoint(0, 26.0).unwrap();
    setpoints.set_deadband(0, 2.0).unwrap();

    let mut zone_control = ZoneControl::new(thermal_model, setpoints);

    // Test within deadband range (23°C to 25°C for 2°C deadband)
    let current_temps = VectorField::from_scalar(24.0, 1);
    let energy_input = zone_control.update_zone_controls(&current_temps);

    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Off);
    assert_eq!(energy_input.as_slice()[0], 0.0);
}

#[test]
fn test_independent_zone_control() {
    let thermal_model = Arc::new(ThermalModel::new(3, 20.0));
    let mut setpoints = ZoneSetpoints::new(3);

    // Configure different setpoints for each zone
    setpoints.set_heating_setpoint(0, 22.0).unwrap();
    setpoints.set_cooling_setpoint(0, 26.0).unwrap();

    setpoints.set_heating_setpoint(1, 20.0).unwrap();
    setpoints.set_cooling_setpoint(1, 24.0).unwrap();

    setpoints.set_heating_setpoint(2, 18.0).unwrap();
    setpoints.set_cooling_setpoint(2, 22.0).unwrap();

    let mut zone_control = ZoneControl::new(thermal_model, setpoints);

    // Test temperatures that trigger different HVAC states
    let current_temps = VectorField::new(vec![
        19.0, // Zone 0: below heating setpoint -> heating
        23.0, // Zone 1: within deadband -> off
        25.0, // Zone 2: above cooling setpoint -> cooling
    ]);

    let energy_input = zone_control.update_zone_controls(&current_temps);

    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);
    assert_eq!(zone_control.get_zone_hvac_status(1), HVACStatus::Off);
    assert_eq!(zone_control.get_zone_hvac_status(2), HVACStatus::Cooling);

    assert!(energy_input.as_slice()[0] > 0.0); // Heating energy
    assert_eq!(energy_input.as_slice()[1], 0.0); // No energy in deadband
    assert!(energy_input.as_slice()[2] > 0.0); // Cooling energy
}

#[test]
fn test_energy_calculation() {
    let thermal_model = Arc::new(ThermalModel::new(2, 20.0));
    let mut setpoints = ZoneSetpoints::new(2);

    setpoints.set_heating_setpoint(0, 22.0).unwrap();
    setpoints.set_heating_setpoint(1, 25.0).unwrap();

    let mut zone_control = ZoneControl::new(thermal_model, setpoints);

    // Test with different temperature differences
    let current_temps = VectorField::new(vec![
        20.0, // 2°C below setpoint -> 2000W
        23.0, // 2°C below setpoint -> 2000W
    ]);

    let energy_input = zone_control.update_zone_controls(&current_temps);

    // Ideal loads: zone_volume=129.6, ACH=0.5, supply=40°C, efficiency=0.9
    // airflow = 129.6 * 0.5 / 3600 = 0.018 m³/s
    // mass_flow = 0.018 * 1.2 = 0.0216 kg/s
    // delta_t = 40 - 20 = 20°C
    // Q = 0.0216 * 1005 * 20 = 434.16 W (thermal)
    // Electrical = 434.16 / 0.9 = 482.4 W
    assert!((energy_input.as_slice()[0] - 482.4).abs() < 1.0);
    assert!((energy_input.as_slice()[1] - 482.4).abs() < 1.0);
}

#[test]
fn test_hvac_status_transitions() {
    let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
    let mut setpoints = ZoneSetpoints::new(1);
    setpoints.set_heating_setpoint(0, 22.0).unwrap();
    setpoints.set_cooling_setpoint(0, 26.0).unwrap();
    setpoints.set_deadband(0, 2.0).unwrap();

    let mut zone_control = ZoneControl::new(thermal_model, setpoints);

    // Start with heating (below heating threshold: 22 - 1 = 21°C)
    let mut current_temps = VectorField::from_scalar(20.0, 1);
    zone_control.update_zone_controls(&current_temps);
    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);

    // Transition to deadband (21°C to 25°C)
    current_temps = VectorField::from_scalar(23.0, 1);
    zone_control.update_zone_controls(&current_temps);
    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Off);

    // Transition to cooling (above cooling threshold: 26 + 1 = 27°C)
    current_temps = VectorField::from_scalar(28.0, 1);
    zone_control.update_zone_controls(&current_temps);
    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Cooling);

    // Transition back to deadband
    current_temps = VectorField::from_scalar(25.0, 1);
    zone_control.update_zone_controls(&current_temps);
    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Off);
}

#[test]
fn test_boundary_temperatures() {
    let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
    let mut setpoints = ZoneSetpoints::new(1);
    setpoints.set_heating_setpoint(0, 22.0).unwrap();
    setpoints.set_cooling_setpoint(0, 26.0).unwrap();
    setpoints.set_deadband(0, 2.0).unwrap();

    let mut zone_control = ZoneControl::new(thermal_model, setpoints);

    // Test exact boundary temperatures
    // Heating threshold: 22 - 1 = 21°C
    let current_temps = VectorField::from_scalar(21.0, 1);
    zone_control.update_zone_controls(&current_temps);
    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);

    // Cooling threshold: 26 + 1 = 27°C
    let current_temps = VectorField::from_scalar(27.0, 1);
    zone_control.update_zone_controls(&current_temps);
    assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Cooling);
}

#[test]
fn test_invalid_inputs() {
    let mut setpoints = ZoneSetpoints::new(2);

    // Test invalid zone ID
    assert!(setpoints.set_heating_setpoint(5, 22.0).is_err());
    assert!(setpoints.set_cooling_setpoint(5, 26.0).is_err());
    assert!(setpoints.set_deadband(5, 2.0).is_err());

    // Test temperature validation
    assert!(setpoints.set_heating_setpoint(0, 9.0).is_err());
    assert!(setpoints.set_heating_setpoint(0, 41.0).is_err());
    assert!(setpoints.set_cooling_setpoint(1, 9.0).is_err());
    assert!(setpoints.set_cooling_setpoint(1, 41.0).is_err());
}

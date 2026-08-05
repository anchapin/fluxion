//! Fluid network VAV system example tests.
//!
//! These tests demonstrate the integration of fluxion-fluid components
//! (Chiller, Boiler, VAV boxes) into a complete VAV HVAC system
//! using the FluidNetworkAdapter.

#[cfg(feature = "fluid")]
use crate::sim::hvac::fluid_adapter::{
    FluidNetworkAdapter, FluidSystemMode, ThermalBoundaryConditions, VavSystemConfig,
};

#[cfg(feature = "fluid")]
#[test]
fn test_simple_vav_system_cooling() {
    let config = VavSystemConfig {
        chiller_capacity: 100_000.0,
        chiller_cop: 5.0,
        boiler_capacity: 80_000.0,
        boiler_efficiency: 0.9,
        num_zones: 2,
        rated_airflow_per_zone: 0.1,
        static_pressure_setpoint: 250.0,
        chilled_water_supply_temp: 7.0,
        hot_water_supply_temp: 60.0,
    };

    let mut adapter = FluidNetworkAdapter::vav_system(config);

    adapter.set_cooling_load(0, 5000.0);
    adapter.set_cooling_load(1, 3000.0);

    adapter.solve().expect("VAV system solve should succeed");

    assert_eq!(adapter.mode(), FluidSystemMode::Cooling);

    let bc0 = adapter.thermal_boundary_conditions(0).unwrap();
    let bc1 = adapter.thermal_boundary_conditions(1).unwrap();

    assert_eq!(bc0.mode, FluidSystemMode::Cooling);
    assert_eq!(bc1.mode, FluidSystemMode::Cooling);

    assert!(bc0.cooling_load_w > 0.0);
    assert!(bc1.cooling_load_w > 0.0);
}

#[cfg(feature = "fluid")]
#[test]
fn test_simple_vav_system_heating() {
    let config = VavSystemConfig {
        chiller_capacity: 100_000.0,
        chiller_cop: 5.0,
        boiler_capacity: 80_000.0,
        boiler_efficiency: 0.9,
        num_zones: 2,
        rated_airflow_per_zone: 0.1,
        static_pressure_setpoint: 250.0,
        chilled_water_supply_temp: 7.0,
        hot_water_supply_temp: 60.0,
    };

    let mut adapter = FluidNetworkAdapter::vav_system(config);

    adapter.set_heating_load(0, 3000.0);
    adapter.set_heating_load(1, 2000.0);

    adapter.solve().expect("VAV system solve should succeed");

    assert_eq!(adapter.mode(), FluidSystemMode::Heating);

    let bc0 = adapter.thermal_boundary_conditions(0).unwrap();
    assert_eq!(bc0.mode, FluidSystemMode::Heating);
    assert!(bc0.heating_load_w > 0.0);
    assert!(bc0.supply_air_temp_c > 22.0);
}

#[cfg(feature = "fluid")]
#[test]
fn test_simple_vav_system_off_mode() {
    let config = VavSystemConfig {
        num_zones: 3,
        ..Default::default()
    };

    let mut adapter = FluidNetworkAdapter::vav_system(config);

    adapter.solve().expect("VAV system solve should succeed");

    assert_eq!(adapter.mode(), FluidSystemMode::Off);

    for zone_id in 0..3 {
        let bc = adapter.thermal_boundary_conditions(zone_id).unwrap();
        assert_eq!(bc.mode, FluidSystemMode::Off);
        assert_eq!(bc.heating_load_w, 0.0);
        assert_eq!(bc.cooling_load_w, 0.0);
    }
}

#[cfg(feature = "fluid")]
#[test]
fn test_vav_system_damper_control() {
    let config = VavSystemConfig {
        num_zones: 1,
        rated_airflow_per_zone: 0.2,
        ..Default::default()
    };

    let mut adapter = FluidNetworkAdapter::vav_system(config);

    adapter.set_cooling_load(0, 5000.0);
    adapter.set_zone_damper(0, 0.5);

    adapter.solve().expect("VAV system solve should succeed");

    let bc = adapter.thermal_boundary_conditions(0).unwrap();
    assert!(bc.supply_mass_flow_kg_s > 0.0);
}

#[cfg(feature = "fluid")]
#[test]
fn test_vav_system_all_zones_independent() {
    let config = VavSystemConfig {
        num_zones: 4,
        rated_airflow_per_zone: 0.1,
        ..Default::default()
    };

    let mut adapter = FluidNetworkAdapter::vav_system(config);

    adapter.set_cooling_load(0, 5000.0);
    adapter.set_cooling_load(1, 0.0);
    adapter.set_heating_load(2, 3000.0);
    adapter.set_heating_load(3, 0.0);

    adapter.solve().expect("VAV system solve should succeed");

    let bc0 = adapter.thermal_boundary_conditions(0).unwrap();
    let bc1 = adapter.thermal_boundary_conditions(1).unwrap();
    let bc2 = adapter.thermal_boundary_conditions(2).unwrap();
    let bc3 = adapter.thermal_boundary_conditions(3).unwrap();

    assert_eq!(bc0.mode, FluidSystemMode::Cooling);
    assert_eq!(bc1.mode, FluidSystemMode::Off);
    assert_eq!(bc2.mode, FluidSystemMode::Heating);
    assert_eq!(bc3.mode, FluidSystemMode::Off);
}

#[cfg(feature = "fluid")]
#[test]
fn test_vav_system_thermal_boundary_conditions_structure() {
    let config = VavSystemConfig {
        num_zones: 1,
        ..Default::default()
    };

    let adapter = FluidNetworkAdapter::vav_system(config);

    let bc = adapter.thermal_boundary_conditions(0).unwrap();

    assert_eq!(bc.zone_id, 0);
    assert!(bc.supply_air_temp_c > 0.0 && bc.supply_air_temp_c < 60.0);
    assert!(bc.supply_mass_flow_kg_s >= 0.0);
    assert!(bc.supply_humidity_ratio > 0.0);
    assert!(bc.return_air_temp_c > 0.0);
}

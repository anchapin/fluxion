//! Pre-built test scenarios for common integration test cases
//!
//! Provides ready-to-use scenarios for low-mass, high-mass, and multi-zone buildings.

use super::fixtures::{BuildingScenario, HvacType};

/// Create a low-mass building scenario (ASHRAE 140 Case 600-like)
pub fn low_mass_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(1)
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("low_mass_scenario validation failed")
}

/// Create a high-mass building scenario (ASHRAE 140 Case 900-like)
pub fn high_mass_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(1)
        .with_window_u_value(2.0)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("high_mass_scenario validation failed")
}

/// Create a multi-zone building scenario (ASHRAE 140 Case 960-like)
pub fn multi_zone_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(3)
        .with_window_u_value(2.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("multi_zone_scenario validation failed")
}

/// Create a scenario with VAV HVAC
pub fn vav_scenario() -> BuildingScenario {
    low_mass_scenario()
        .with_hvac(HvacType::VAV)
        .build()
        .expect("vav_scenario validation failed")
}

/// Create a scenario with Heat Pump HVAC
pub fn heat_pump_scenario() -> BuildingScenario {
    low_mass_scenario()
        .with_hvac(HvacType::HeatPump)
        .build()
        .expect("heat_pump_scenario validation failed")
}

/// Create a scenario with CAV HVAC
pub fn cav_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(1)
        .with_hvac(HvacType::CAV)
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("cav_scenario validation failed")
}

/// Create a scenario with Chiller HVAC
pub fn chiller_scenario() -> BuildingScenario {
    BuildingScenario::new()
        .with_zone_count(1)
        .with_hvac(HvacType::Chiller)
        .with_window_u_value(1.5)
        .with_heating_setpoint(20.0)
        .with_cooling_setpoint(26.0)
        .build()
        .expect("chiller_scenario validation failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_mass_scenario() {
        let scenario = low_mass_scenario();
        let result = scenario.build();
        assert!(result.is_ok());

        let scenario = result.unwrap();
        assert_eq!(scenario.window_u_value, Some(1.5));
        assert_eq!(scenario.heating_setpoint, Some(20.0));
        assert_eq!(scenario.cooling_setpoint, Some(26.0));
    }

    #[test]
    fn test_high_mass_scenario() {
        let scenario = high_mass_scenario();
        let result = scenario.build();
        assert!(result.is_ok());

        let scenario = result.unwrap();
        assert_eq!(scenario.window_u_value, Some(2.0));
        assert_eq!(scenario.heating_setpoint, Some(20.0));
        assert_eq!(scenario.cooling_setpoint, Some(26.0));
    }

    #[test]
    fn test_multi_zone_scenario() {
        let scenario = multi_zone_scenario();
        let result = scenario.build();
        assert!(result.is_ok());

        let scenario = result.unwrap();
        assert_eq!(scenario.window_u_value, Some(2.5));
        assert_eq!(scenario.heating_setpoint, Some(20.0));
        assert_eq!(scenario.cooling_setpoint, Some(26.0));
    }

    #[test]
    fn test_vav_scenario() {
        let scenario = vav_scenario();
        assert!(scenario.build().is_ok());
        assert_eq!(scenario.hvac_type, Some(HvacType::VAV));
    }

    #[test]
    fn test_heat_pump_scenario() {
        let scenario = heat_pump_scenario();
        assert!(scenario.build().is_ok());
        assert_eq!(scenario.hvac_type, Some(HvacType::HeatPump));
    }

    #[test]
    fn test_cav_scenario() {
        let scenario = cav_scenario();
        assert!(scenario.build().is_ok());
        assert_eq!(scenario.hvac_type, Some(HvacType::CAV));
    }

    #[test]
    fn test_chiller_scenario() {
        let scenario = chiller_scenario();
        assert!(scenario.build().is_ok());
        assert_eq!(scenario.hvac_type, Some(HvacType::Chiller));
    }

    #[test]
    fn test_all_scenarios_create_valid_models() {
        let scenarios = [
            ("low_mass", low_mass_scenario()),
            ("high_mass", high_mass_scenario()),
            ("multi_zone", multi_zone_scenario()),
            ("vav", vav_scenario()),
            ("heat_pump", heat_pump_scenario()),
            ("cav", cav_scenario()),
            ("chiller", chiller_scenario()),
        ];

        for (name, scenario) in scenarios {
            let built = scenario
                .build()
                .unwrap_or_else(|_| panic!("{} scenario failed to build", name));
            let model = built.create_model();
            assert!(
                model.window_u_value > 0.0,
                "{}: window_u_value should be positive",
                name
            );
            assert!(
                model.heating_setpoint > 0.0,
                "{}: heating_setpoint should be positive",
                name
            );
            assert!(
                model.cooling_setpoint > model.heating_setpoint,
                "{}: cooling_setpoint should be > heating_setpoint",
                name
            );
        }
    }
}

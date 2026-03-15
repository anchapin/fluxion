//! Unit tests for HVAC equipment models

use fluxion::sim::hvac::{
    Boiler, CAVSystem, Chiller, HVACMode, HeatPump, HeatPumpMode, VAVTerminal,
    VariableCapacityEquipment,
};

#[test]
fn test_variable_capacity_trait() {
    // Create instances of all equipment types
    let chiller = Chiller::new("Chiller-1".to_string(), 10000.0, 4.0, 35.0);
    let boiler = Boiler::new("Boiler-1".to_string(), 10000.0, 0.85, -5.0);
    let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);
    let cav = CAVSystem::new("CAV-1".to_string(), 1.0);
    let heatpump = HeatPump::new("HP-1".to_string(), 12000.0, 10000.0, 3.5, 3.0);

    // Verify they implement VariableCapacityEquipment by calling trait methods
    // This test just ensures the trait is implemented and methods compile
    let _capacity = chiller.calculate_capacity(0.5, 20.0);
    let _efficiency = boiler.calculate_efficiency(0.5, 20.0, HVACMode::Heating);
    let _power = vav.calculate_power(5000.0, 20.0, HVACMode::Heating);
    let _rated = cav.rated_capacity();
    let _current = heatpump.current_plr();

    // Basic assertions to verify implementations exist
    assert!(chiller.rated_capacity() > 0.0);
    assert!(boiler.rated_efficiency(HVACMode::Heating) > 0.0);
    assert!(vav.rated_capacity() > 0.0);
    assert!(cav.rated_capacity() > 0.0);
    assert!(heatpump.rated_capacity() > 0.0);
}

#[test]
fn test_plr_tracking() {
    // Test Chiller PLR tracking
    let mut chiller = Chiller::new("Chiller-1".to_string(), 10000.0, 4.0, 35.0);

    // Update state with various loads
    chiller.update_state(0.0, 20.0, HVACMode::Cooling); // No load
    assert_eq!(chiller.current_plr(), 0.0);

    chiller.update_state(5000.0, 20.0, HVACMode::Cooling); // 50% load
    assert!((chiller.current_plr() - 0.5).abs() < 0.01);

    chiller.update_state(10000.0, 20.0, HVACMode::Cooling); // Full load
    assert!((chiller.current_plr() - 1.0).abs() < 0.01);

    chiller.update_state(15000.0, 20.0, HVACMode::Cooling); // Overload
    assert_eq!(chiller.current_plr(), 1.0); // Should clamp to 1.0

    // Test CAVSystem PLR tracking
    let mut cav = CAVSystem::new("CAV-1".to_string(), 1.0);
    cav.update_state(0.0, 20.0, HVACMode::Heating);
    assert_eq!(cav.current_plr(), 0.0);

    cav.update_state(5000.0, 20.0, HVACMode::Cooling);
    assert!(cav.current_plr() > 0.0);
}

#[test]
fn test_vav_implementation() {
    let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);

    // Test calculate_capacity at PLR=0.5
    let capacity = vav.calculate_capacity(0.5, 20.0);
    assert!(capacity > 0.0);
    assert!((capacity - 2500.0).abs() < 0.1); // 5000W reheat * 0.5

    // Test calculate_efficiency for heating
    let eff_heating = vav.calculate_efficiency(0.5, 20.0, HVACMode::Heating);
    assert!(eff_heating > 0.0);
    assert!((eff_heating - 0.8).abs() < 0.1); // Fan + reheat COP ~0.8

    // Test calculate_efficiency for cooling
    let eff_cooling = vav.calculate_efficiency(0.5, 20.0, HVACMode::Cooling);
    assert!(eff_cooling > 0.0);
    assert!((eff_cooling - 3.0).abs() < 0.1); // Fan + cooling coil COP ~3.0

    // Test calculate_power
    let load = 2500.0;
    let power = vav.calculate_power(load, 20.0, HVACMode::Heating);
    assert!(power > 0.0);
    // Power = load / efficiency = 2500 / 0.8 = 3125
    assert!((power - 3125.0).abs() < 10.0);
}

#[test]
fn test_cav_implementation() {
    let cav = CAVSystem::new("CAV-1".to_string(), 1.0);

    // Test calculate_capacity at PLR=0.5
    let capacity = cav.calculate_capacity(0.5, 20.0);
    assert!(capacity > 0.0);
    assert!((capacity - 5000.0).abs() < 0.1); // Max(heating,cooling) * 0.5 = 10000 * 0.5

    // Test calculate_efficiency for heating
    let eff_heating = cav.calculate_efficiency(0.5, 20.0, HVACMode::Heating);
    assert!(eff_heating > 0.0);
    assert!((eff_heating - 0.85).abs() < 0.1); // Fan + heating coil COP ~0.85

    // Test calculate_efficiency for cooling
    let eff_cooling = cav.calculate_efficiency(0.5, 20.0, HVACMode::Cooling);
    assert!(eff_cooling > 0.0);
    assert!((eff_cooling - 3.2).abs() < 0.1); // Fan + cooling coil COP ~3.2

    // Test calculate_power
    let load = 5000.0;
    let power = cav.calculate_power(load, 20.0, HVACMode::Heating);
    assert!(power > 0.0);
    // Power = load / efficiency = 5000 / 0.85 ≈ 5882
    assert!((power - 5882.0).abs() < 10.0);
}

#[test]
fn test_heatpump_implementation() {
    let hp = HeatPump::new(
        "HP-1".to_string(),
        12000.0, // 12kW heating
        10000.0, // 10kW cooling
        3.5,     // COP 3.5
        3.0,     // EER 3.0
    );

    // Test calculate_capacity at PLR=0.5, outdoor_temp=20°C
    let capacity = hp.calculate_capacity(0.5, 20.0);
    assert!(capacity > 0.0);
    // At 20°C (moderate temp), capacity should be close to rated * PLR
    assert!((capacity - 6000.0).abs() < 100.0); // 12000 * 0.5

    // Test calculate_efficiency for heating mode
    let eff_heating = hp.calculate_efficiency(0.5, -5.0, HVACMode::Heating);
    assert!(eff_heating > 0.0);
    assert!(eff_heating < 3.5); // Part load degradation
    assert!(eff_heating > 2.0); // But not too low

    // Test calculate_efficiency for cooling mode
    let eff_cooling = hp.calculate_efficiency(0.5, 35.0, HVACMode::Cooling);
    assert!(eff_cooling > 0.0);
    assert!(eff_cooling < 3.0); // Part load degradation
    assert!(eff_cooling > 2.0); // But not too low

    // Test calculate_power for heating
    let load = 6000.0;
    let power_heating = hp.calculate_power(load, -5.0, HVACMode::Heating);
    assert!(power_heating > 0.0);
    // Power = load / COP (with PLR degradation)
    assert!(power_heating < 3000.0); // Should be less than load/2

    // Test calculate_power for cooling
    let power_cooling = hp.calculate_power(load, 35.0, HVACMode::Cooling);
    assert!(power_cooling > 0.0);
    // Power = load / EER (with PLR degradation)
    assert!(power_cooling < 3000.0); // Should be less than load/2
}

#[test]
fn test_chiller_implementation() {
    let chiller = Chiller::new("Chiller-1".to_string(), 10000.0, 4.0, 35.0);

    // Test calculate_capacity at PLR=0.5, outdoor_temp=20°C
    let capacity = chiller.calculate_capacity(0.5, 20.0);
    assert!(capacity > 0.0);
    assert!((capacity - 5000.0).abs() < 100.0); // 10000 * 0.5 at moderate temp

    // Test calculate_efficiency for cooling mode
    let eff_cooling = chiller.calculate_efficiency(0.5, 35.0, HVACMode::Cooling);
    assert!(eff_cooling > 0.0);
    assert!(eff_cooling < 4.0); // Part load degradation
    assert!(eff_cooling > 2.0); // But not too low

    // Test calculate_efficiency for heating mode (should be 0)
    let eff_heating = chiller.calculate_efficiency(0.5, 35.0, HVACMode::Heating);
    assert_eq!(eff_heating, 0.0); // Chillers don't heat

    // Test calculate_power for cooling
    let load = 5000.0;
    let power = chiller.calculate_power(load, 35.0, HVACMode::Cooling);
    assert!(power > 0.0);
    // Power = load / COP (with PLR degradation)
    assert!(power < 2500.0); // Should be less than load/2

    // Test calculate_power for heating (should be 0)
    let power_heating = chiller.calculate_power(load, 35.0, HVACMode::Heating);
    assert_eq!(power_heating, 0.0); // Chillers don't heat
}

#[test]
fn test_boiler_implementation() {
    let boiler = Boiler::new("Boiler-1".to_string(), 10000.0, 0.85, -5.0);

    // Test calculate_capacity at PLR=0.5, outdoor_temp=20°C
    let capacity = boiler.calculate_capacity(0.5, 20.0);
    assert!(capacity > 0.0);
    assert!((capacity - 5000.0).abs() < 100.0); // 10000 * 0.5

    // Test calculate_efficiency for heating mode
    let eff_heating = boiler.calculate_efficiency(0.5, -5.0, HVACMode::Heating);
    assert!(eff_heating > 0.0);
    assert!(eff_heating < 0.85); // Part load degradation
    assert!(eff_heating > 0.5); // But not too low

    // Test calculate_efficiency for cooling mode (should be 0)
    let eff_cooling = boiler.calculate_efficiency(0.5, -5.0, HVACMode::Cooling);
    assert_eq!(eff_cooling, 0.0); // Boilers don't cool

    // Test calculate_power for heating
    let load = 5000.0;
    let power = boiler.calculate_power(load, -5.0, HVACMode::Heating);
    assert!(power > 0.0);
    // Power = load / efficiency (with PLR degradation)
    assert!(power > 5882.0); // Should be more than load/0.85
    assert!(power < 10000.0); // But less than load

    // Test calculate_power for cooling (should be 0)
    let power_cooling = boiler.calculate_power(load, -5.0, HVACMode::Cooling);
    assert_eq!(power_cooling, 0.0); // Boilers don't cool
}

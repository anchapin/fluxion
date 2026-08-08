//! Integration tests for the DHW (Domestic Hot Water) module.
//!
//! These tests verify the DHWTank model implementation including:
//! - Tank creation and configuration
//! - Standby loss calculation
//! - Water heating energy calculation
//! - Draw profile scheduling
//! - Energy accumulation

use fluxion::sim::hvac::dhw::{DHWTank, DHWResult, HeatingSource};
use fluxion::sim::schedule::DailySchedule;

#[test]
fn test_dhw_tank_creation() {
    let tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        50.0,
        0,
        HeatingSource::Electric,
    );

    assert_eq!(tank.id, "DHW-1");
    assert_eq!(tank.volume_L, 200.0);
    assert_eq!(tank.setpoint_C, 60.0);
    assert_eq!(tank.standby_loss_W, 50.0);
    assert_eq!(tank.tank_location_zone_id, 0);
    assert_eq!(tank.heating_source, HeatingSource::Electric);
}

#[test]
fn test_dhw_tank_standby_loss_200l_50w() {
    let mut tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        50.0,
        0,
        HeatingSource::Electric,
    );

    let result = tank.step(12, 3600.0);

    assert_eq!(
        result.standby_loss_w, 50.0,
        "200L tank with 50W standby loss should have 50W continuous zone gain"
    );
}

#[test]
fn test_dhw_tank_heating_energy_100l_draw() {
    let mut tank = DHWTank::new(
        "DHW-1".to_string(),
        100.0,
        60.0,
        0.0,
        0,
        HeatingSource::Electric,
    );
    tank.supply_temp_C = 10.0;

    let mut draw_profile = DailySchedule::new();
    draw_profile.fill_range(0, 24, 100.0);
    tank.draw_profile = draw_profile;

    let result = tank.step(12, 3600.0);

    // Energy = draw_L * cp * delta_T / 3600 (to kWh)
    // = 100L * 4.186 kJ/(kg·K) * 50K / 3600
    // = 100 * 4.186 * 50 / 3600 = 5.81 kWh
    let expected_kwh = 100.0 * 4.186 * (60.0 - 10.0) / 3600.0;

    assert!(
        (result.heating_energy_kwh - expected_kwh).abs() < 0.01,
        "100L draw at 60°C with 10°C supply water should be ~{:.1} kWh, got {:.2}",
        expected_kwh,
        result.heating_energy_kwh
    );
}

#[test]
fn test_dhw_tank_100l_draw_approx_5_8_kwh() {
    let mut tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        0.0,
        0,
        HeatingSource::Electric,
    );
    tank.supply_temp_C = 10.0;

    let mut draw_profile = DailySchedule::new();
    draw_profile.fill_range(0, 24, 100.0);
    tank.draw_profile = draw_profile;

    let result = tank.step(12, 3600.0);

    let expected = 5.8;
    assert!(
        (result.heating_energy_kwh - expected).abs() < 0.2,
        "100L draw at 60°C with 10°C supply water should be ~5.8 kWh, got {:.2}",
        result.heating_energy_kwh
    );
}

#[test]
fn test_dhw_tank_total_energy_accumulation() {
    let mut tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        0.0,
        0,
        HeatingSource::Electric,
    );
    tank.supply_temp_C = 10.0;

    let mut draw_profile = DailySchedule::new();
    draw_profile.fill_range(0, 24, 100.0);
    tank.draw_profile = draw_profile;

    let first_result = tank.step(12, 3600.0);
    let second_result = tank.step(13, 3600.0);

    assert!(
        second_result.total_dhw_energy_kwh > first_result.heating_energy_kwh,
        "Total DHW energy should accumulate across timesteps"
    );
}

#[test]
fn test_dhw_tank_gas_heating_source() {
    let tank = DHWTank::new(
        "DHW-Gas-1".to_string(),
        200.0,
        60.0,
        50.0,
        0,
        HeatingSource::Gas,
    );

    assert_eq!(tank.heating_source, HeatingSource::Gas);
}

#[test]
fn test_dhw_tank_reset() {
    let mut tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        0.0,
        0,
        HeatingSource::Electric,
    );
    tank.supply_temp_C = 10.0;

    let mut draw_profile = DailySchedule::new();
    draw_profile.fill_range(0, 24, 100.0);
    tank.draw_profile = draw_profile;

    tank.step(12, 3600.0);
    assert!(tank.total_dhw_energy() > 0.0);

    tank.reset();
    assert_eq!(tank.total_dhw_energy(), 0.0);
}

#[test]
fn test_dhw_tank_with_supply_temp() {
    let tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        50.0,
        0,
        HeatingSource::Electric,
    )
    .with_supply_temp(15.0);

    assert_eq!(tank.supply_temp_C, 15.0);
}

#[test]
fn test_dhw_tank_draw_profile_schedule() {
    let mut tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        0.0,
        0,
        HeatingSource::Electric,
    );

    let mut draw_profile = DailySchedule::new();
    draw_profile.set_hour(7, 50.0);
    draw_profile.set_hour(8, 100.0);
    draw_profile.set_hour(9, 80.0);
    tank.draw_profile = draw_profile;

    assert_eq!(tank.draw_profile.value(7), 50.0);
    assert_eq!(tank.draw_profile.value(8), 100.0);
    assert_eq!(tank.draw_profile.value(9), 80.0);
}

#[test]
fn test_dhw_result_default() {
    let result = DHWResult::default();
    assert_eq!(result.heating_energy_kwh, 0.0);
    assert_eq!(result.standby_loss_w, 0.0);
    assert_eq!(result.draw_liters, 0.0);
}

#[test]
fn test_heating_source_default() {
    assert_eq!(HeatingSource::default(), HeatingSource::Electric);
}

#[test]
fn test_dhw_tank_with_draw_profile() {
    let mut draw_profile = DailySchedule::new();
    draw_profile.fill_range(7, 22, 80.0);

    let tank = DHWTank::with_draw_profile(
        "DHW-WithProfile".to_string(),
        150.0,
        55.0,
        30.0,
        1,
        HeatingSource::Gas,
        draw_profile.clone(),
    );

    assert_eq!(tank.draw_profile.value(8), 80.0);
    assert_eq!(tank.draw_profile.value(22), 0.0);
}

#[test]
fn test_dhw_tank_zero_draw_no_heating_energy() {
    let mut tank = DHWTank::new(
        "DHW-1".to_string(),
        200.0,
        60.0,
        0.0,
        0,
        HeatingSource::Electric,
    );
    tank.supply_temp_C = 10.0;

    let mut draw_profile = DailySchedule::new();
    draw_profile.fill_range(0, 24, 0.0);
    tank.draw_profile = draw_profile;

    let result = tank.step(12, 3600.0);

    assert_eq!(result.heating_energy_kwh, 0.0);
    assert_eq!(result.draw_liters, 0.0);
}

#[test]
fn test_dhw_tank_standby_loss_injected_to_zone() {
    let mut tank = DHWTank::new(
        "DHW-MechanicalRoom".to_string(),
        300.0,
        65.0,
        75.0,
        2,
        HeatingSource::Electric,
    );

    let result = tank.step(12, 3600.0);

    assert_eq!(
        result.standby_loss_w, 75.0,
        "75W standby loss should be reported as 75W for zone 2"
    );
}

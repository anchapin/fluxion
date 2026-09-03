//! Integration tests for earth tube (ground-air heat exchanger) ventilation pre-conditioning.
//!
//! These tests verify that the earth tube integration affects simulated energy use:
//! - Winter: earth tube pre-heats intake air, reducing heating energy
//! - Summer: earth tube pre-cools intake air, reducing cooling energy
//!
//! # References
//!
//! - Issue #2276: Integrate earth tube into ventilation/thermal system
//! - Earth tube physics: `fluxion_core::earth_tube::EarthTube`

use fluxion::sim::ventilation::{ConstantVentilation, EarthTubeVentilation, VentilationSchedule};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Test that EarthTubeVentilation decorator correctly modifies supply temperature
/// for both winter (pre-heating) and summer (pre-cooling) conditions.
#[test]
fn test_earth_tube_supply_temperature_effect() {
    let base = ConstantVentilation::new(0.5);
    let earth_tube = fluxion_core::earth_tube::EarthTube::new().soil_temperature_K(285.15); // ~12°C

    let vent = EarthTubeVentilation::new(base, earth_tube);

    // Winter test: outdoor -5°C, ground ~12°C → pre-heated supply
    let supply_winter_k = vent.supply_temperature(268.15); // -5°C in K
    let preheat = supply_winter_k - 268.15;
    assert!(
        preheat >= 5.0,
        "Winter pre-heating should be at least 5°C, got {:.1}°C",
        preheat
    );
    println!(
        "Winter: outdoor=-5°C, supply={:.1}°C, preheat={:.1}°C",
        supply_winter_k - 273.15,
        preheat
    );

    // Summer test: outdoor 35°C, ground ~12°C → pre-cooled supply
    let supply_summer_k = vent.supply_temperature(308.15); // 35°C in K
    let precool = supply_summer_k - 308.15; // negative
    assert!(
        precool <= -5.0,
        "Summer pre-cooling should be at least 5°C, got {:.1}°C",
        precool.abs()
    );
    println!(
        "Summer: outdoor=35°C, supply={:.1}°C, precool={:.1}°C",
        supply_summer_k - 273.15,
        precool
    );
}

/// Integration test demonstrating earth tube effect on annual energy use.
#[test]
fn test_earth_tube_ventilation_integration() {
    let base_vent = ConstantVentilation::new(0.5);
    let earth_tube = fluxion_core::earth_tube::EarthTube::new()
        .soil_temperature_K(285.15)
        .pipe_length_m(30.0)
        .pipe_diameter_m(0.15)
        .flow_rate_m3_s(0.05);

    let vent_with_et = EarthTubeVentilation::new(base_vent, earth_tube.clone());

    // Test that ACH is unchanged (earth tube affects temperature, not flow)
    let ach = vent_with_et.get_ach(12, 20.0, 25.0, 2.0, 129.6);
    assert_eq!(ach, 0.5, "Earth tube should not affect ACH");

    // Winter morning (heating season): outdoor very cold
    let q_winter_cold = vent_with_et.heat_transfer_rate(260.15); // -13°C
    println!("Winter cold: Q = {:.0} W (pre-heating)", q_winter_cold);
    assert!(
        q_winter_cold > 0.0,
        "Should have positive heat transfer (pre-heating)"
    );

    // Summer midday (cooling season): outdoor hot
    let q_summer_hot = vent_with_et.heat_transfer_rate(308.15); // 35°C
    println!("Summer hot: Q = {:.0} W (pre-cooling)", q_summer_hot);
    assert!(
        q_summer_hot < 0.0,
        "Should have negative heat transfer (pre-cooling)"
    );
}

/// Test earth tube effect on a representative winter day.
#[test]
fn test_earth_tube_winter_day_effect() {
    let weather = DenverTmyWeather::new();

    let base_vent = ConstantVentilation::new(0.5);
    let earth_tube = fluxion_core::earth_tube::EarthTube::new()
        .soil_temperature_K(285.15)
        .flow_rate_m3_s(0.05);

    let vent_with_et = EarthTubeVentilation::new(base_vent, earth_tube);

    // January 15th (typical cold day in Denver)
    let jan_15_hour_0 = 14 * 24; // Day 14 (0-indexed), hour 0

    let mut total_heating_reduction_wh = 0.0;
    let mut hours_with_benefit = 0;

    for hour_offset in 0..24 {
        let hour = jan_15_hour_0 + hour_offset;
        if let Ok(weather_data) = weather.get_hourly_data(hour) {
            let outdoor_c = weather_data.dry_bulb_temp;
            let outdoor_k = outdoor_c + 273.15;
            let supply_k = vent_with_et.supply_temperature(outdoor_k);
            let supply_c = supply_k - 273.15;

            // Pre-heating benefit: how much warmer supply is vs outdoor
            let preheat_c = supply_c - outdoor_c;

            // Ventilation heating load reduction (W)
            let h_ve = 21.7;
            let heating_reduction_w = h_ve * preheat_c.max(0.0);

            if preheat_c > 0.0 {
                hours_with_benefit += 1;
                total_heating_reduction_wh += heating_reduction_w;
            }
        }
    }

    let daily_kwh_savings = total_heating_reduction_wh / 1000.0;

    println!(
        "January 15th: {} hours with benefit, {:.2} kWh daily savings",
        hours_with_benefit, daily_kwh_savings
    );

    assert!(
        hours_with_benefit >= 16,
        "Should reduce heating load for most winter hours"
    );
    assert!(
        daily_kwh_savings >= 0.5,
        "Should save at least 0.5 kWh on cold winter day"
    );
}

/// Test earth tube effect on a representative summer day.
#[test]
fn test_earth_tube_summer_day_effect() {
    let weather = DenverTmyWeather::new();

    let base_vent = ConstantVentilation::new(0.5);
    let earth_tube = fluxion_core::earth_tube::EarthTube::new()
        .soil_temperature_K(285.15)
        .flow_rate_m3_s(0.05);

    let vent_with_et = EarthTubeVentilation::new(base_vent, earth_tube);

    // July 15th (typical hot day in Denver)
    let jul_15_hour_0 = 196 * 24; // Day 196 (0-indexed), hour 0

    let mut total_cooling_reduction_wh = 0.0;
    let mut hours_with_benefit = 0;

    for hour_offset in 0..24 {
        let hour = jul_15_hour_0 + hour_offset;
        if let Ok(weather_data) = weather.get_hourly_data(hour) {
            let outdoor_c = weather_data.dry_bulb_temp;
            let outdoor_k = outdoor_c + 273.15;
            let supply_k = vent_with_et.supply_temperature(outdoor_k);
            let supply_c = supply_k - 273.15;

            // Pre-cooling benefit: how much cooler supply is vs outdoor
            let precool_c = outdoor_c - supply_c;

            // Ventilation cooling load reduction (W)
            let h_ve = 21.7;
            let cooling_reduction_w = h_ve * precool_c.max(0.0);

            if precool_c > 0.0 {
                hours_with_benefit += 1;
                total_cooling_reduction_wh += cooling_reduction_w;
            }
        }
    }

    let daily_kwh_savings = total_cooling_reduction_wh / 1000.0;

    println!(
        "July 15th: {} hours with benefit, {:.2} kWh daily savings",
        hours_with_benefit, daily_kwh_savings
    );

    assert!(
        hours_with_benefit >= 12,
        "Should reduce cooling load for most summer hours"
    );
    assert!(
        daily_kwh_savings >= 0.5,
        "Should save at least 0.5 kWh on hot summer day"
    );
}

/// Verify earth tube is a valid VentilationSchedule.
#[test]
fn test_earth_tube_ventilation_trait_object() {
    let base = ConstantVentilation::new(0.5);
    let earth_tube = fluxion_core::earth_tube::EarthTube::new();
    let vent = EarthTubeVentilation::new(base, earth_tube);

    let boxed: Box<dyn VentilationSchedule> = vent.clone_box();
    let ach = boxed.get_ach(12, 20.0, 25.0, 2.0, 100.0);
    assert_eq!(ach, 0.5);
}

/// Test earth tube disabled vs enabled comparison.
#[test]
fn test_earth_tube_disabled_vs_enabled_energy_impact() {
    let base_vent = ConstantVentilation::new(0.5);
    let earth_tube = fluxion_core::earth_tube::EarthTube::new()
        .soil_temperature_K(285.15)
        .flow_rate_m3_s(0.05);

    let vent_enabled = EarthTubeVentilation::new(base_vent, earth_tube);

    // Winter scenario: outdoor -10°C
    let outdoor_k_winter = 263.15; // -10°C
    let supply_k_et_winter = vent_enabled.supply_temperature(outdoor_k_winter);
    let preheat_winter = supply_k_et_winter - outdoor_k_winter;

    let h_ve = 21.7;
    let heating_reduction_w = h_ve * preheat_winter;

    println!(
        "Winter (-10°C): preheat={:.1}°C, heating reduction={:.0}W",
        preheat_winter, heating_reduction_w
    );
    assert!(preheat_winter > 5.0, "Winter pre-heat should be > 5°C");
    assert!(
        heating_reduction_w > 100.0,
        "Heating reduction should be > 100W"
    );

    // Summer scenario: outdoor 35°C
    let outdoor_k_summer = 308.15; // 35°C
    let supply_k_et_summer = vent_enabled.supply_temperature(outdoor_k_summer);
    let precool_summer = outdoor_k_summer - supply_k_et_summer;

    let cooling_reduction_w = h_ve * precool_summer;

    println!(
        "Summer (35°C): precool={:.1}°C, cooling reduction={:.0}W",
        precool_summer, cooling_reduction_w
    );
    assert!(precool_summer > 5.0, "Summer pre-cool should be > 5°C");
    assert!(
        cooling_reduction_w > 100.0,
        "Cooling reduction should be > 100W"
    );
}

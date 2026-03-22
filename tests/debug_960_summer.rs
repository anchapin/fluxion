//! Diagnostic test for Case 960 summer cooling behavior
//! Run with: cargo test test_960_summer_debug --release -- --nocapture

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn test_960_summer_debug() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    println!("\n=== Case 960 Configuration ===");
    println!("Zones: {}", model.num_zones);
    println!("Zone areas: {:?}", model.zone_area.as_ref());
    println!("Window areas by zone: ");
    for (i, wins) in spec.windows.iter().enumerate() {
        let total: f64 = wins.iter().map(|w| w.area).sum();
        println!("  Zone {}: {:.2} m²", i, total);
    }
    println!("Cooling setpoints: {:?}", model.cooling_setpoints.as_ref());
    println!("Heating setpoints: {:?}", model.heating_setpoints.as_ref());
    println!("HVAC enabled: {:?}", model.hvac_enabled.as_ref());
    println!(
        "Solar beam to mass fraction: {}",
        model.solar_beam_to_mass_fraction
    );
    println!();

    // Simulate a peak summer week (July 1-7, day 182-188)
    let start_day = 182;
    let start_hour = start_day * 24;
    println!(
        "=== Summer Week Simulation (Days {}-{}) ===",
        start_day,
        start_day + 7
    );
    println!("Hour | Outdoor | T0 | T1 | HVAC (kWh) | solar0 | solar1 | loads0 | loads1");
    println!("     | (C)     |    |    |            | (W/m2) | (W/m2) | (W/m2) | (W/m2)");
    println!("-----|---------|----|----|------------|--------|--------|--------|--------");

    let mut total_cooling = 0.0;
    let mut total_heating = 0.0;
    for hour in start_hour..start_hour + 7 * 24 {
        let weather_data = weather.get_hourly_data(hour).unwrap();
        model.set_weather(weather_data.clone());
        let hvac_kwh = model.step_physics(hour, weather_data.dry_bulb_temp);
        let temps = model.temperatures.as_ref();
        let loads = model.loads.as_ref();
        let solar = model.solar_gains.as_ref();

        if hour % 6 == 0 {
            // print every 6 hours
            println!(
                "{:4} | {:7.1} | {:3.1} | {:3.1} | {:10.3} | {:6.1} | {:6.1} | {:6.1} | {:6.1}",
                hour,
                weather_data.dry_bulb_temp,
                temps[0],
                temps[1],
                hvac_kwh,
                solar[0],
                solar[1],
                loads[0],
                loads[1]
            );
        }

        if hvac_kwh > 0.0 {
            total_heating += hvac_kwh;
        } else {
            total_cooling += -hvac_kwh;
        }
    }

    println!(
        "\nWeek totals: Heating = {:.3} MWh, Cooling = {:.3} MWh",
        total_heating / 1000.0,
        total_cooling / 1000.0
    );

    // Annual totals would be higher. Estimate: if a week is typical, annual cooling = week * 52
    let estimated_annual_cooling = total_cooling / 1000.0 * 52.0;
    let estimated_annual_heating = total_heating / 1000.0 * 52.0;
    println!(
        "Estimated annual (extrapolated): Heating = {:.2} MWh, Cooling = {:.2} MWh",
        estimated_annual_heating, estimated_annual_cooling
    );

    // Compare to reference
    println!("\n=== Reference Ranges ===");
    println!("Annual Heating: 5.0 - 15.0 MWh");
    println!("Annual Cooling: 1.0 - 3.5 MWh");
    if estimated_annual_cooling > 3.5 {
        println!("⚠️  Estimated cooling EXCEEDS reference max!");
    } else {
        println!("✓ Cooling within range (if extrapolation valid)");
    }
}

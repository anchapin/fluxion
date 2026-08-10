//! Diagnostic: Track mass temperature trajectory throughout the year
//! to understand why T_mass stays cold (~15.7°C) instead of heating up.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
#[ignore = "diagnostic-only test with no assertion; quarantined per #2536. Run manually with --ignored if needed."]
fn diag_mass_trajectory() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;
    model.hvac_heating_capacity = 0.0;
    model.hvac_cooling_capacity = 0.0;

    // Track at specific hours
    let check_points: &[usize] = &[
        0, 1, 6, 12, 24, 168, 720, 1416, 2160, 2880, 3624, 4344, 4368, 4392, 5088, 5832, 6552,
        7296, 8004, 8760,
    ];

    let mut max_mass = f64::NEG_INFINITY;
    let mut min_mass = f64::INFINITY;
    let mut max_zone = f64::NEG_INFINITY;
    let mut min_zone = f64::INFINITY;
    let mut max_mass_step = 0usize;
    let mut min_mass_step = 0usize;
    let mut max_zone_step = 0usize;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        let mass_temp = model.mass_temperatures.as_slice()[0];
        let zone_temp = model.temperatures.as_slice()[0];
        let outdoor_temp = weather_data.dry_bulb_temp;

        if mass_temp > max_mass {
            max_mass = mass_temp;
            max_mass_step = step;
        }
        if mass_temp < min_mass {
            min_mass = mass_temp;
            min_mass_step = step;
        }
        if zone_temp > max_zone {
            max_zone = zone_temp;
            max_zone_step = step;
        }
        if zone_temp < min_zone {
            min_zone = zone_temp;
        }

        if check_points.contains(&step) {
            let day = step / 24;
            let hour = step % 24;
            println!(
                "step={:5} (day {:3} hr {:2}) T_out={:6.1} T_zone={:.2} T_mass={:.2}",
                step, day, hour, outdoor_temp, zone_temp, mass_temp
            );
        }
    }

    println!("\n=== Summary ===");
    println!(
        "T_mass: min={:.2}°C (step {}, day {}) max={:.2}°C (step {}, day {})",
        min_mass,
        min_mass_step,
        min_mass_step / 24,
        max_mass,
        max_mass_step,
        max_mass_step / 24
    );
    println!(
        "T_zone: min={:.2}°C max={:.2}°C (step {}, day {}, hour {})",
        min_zone,
        max_zone,
        max_zone_step,
        max_zone_step / 24,
        max_zone_step % 24
    );

    assert!(max_zone > 0.0, "sanity");
}

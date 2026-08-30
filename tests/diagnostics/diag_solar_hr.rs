//! Diagnostic: solar gains by hour for Case 600FF
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;
use fluxion::sim::thermal_selector::ThermalSelector;

#[test]
#[ignore = "diagnostic-only test with no assertion; quarantined per #2536. Run manually with --ignored if needed."]
fn solar_diagnostic() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default()).expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    model.setpoints.heating_setpoint = -999.0;
    model.setpoints.cooling_setpoint = 999.0;

    let check_hours: Vec<usize> = vec![
        6184, 6183, 6182, 6185, 6186, 4344, 4345, 4346, 4347, 12, 13, 14, 15,
    ];

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if check_hours.contains(&step) {
            let zone = model.setpoints.temperatures.as_slice()[0];
            let mass = model.mass.mass_temperatures.as_slice()[0];
            let day = step / 24 + 1;
            let hour = step % 24;
            let t_out = weather_data.dry_bulb_temp;
            let dni = weather_data.dni;
            let dhi = weather_data.dhi;
            println!(
                "H{} (D{} {:02}): zone={:.1} mass={:.1} T_out={:.1} DNI={} DHI={}",
                step, day, hour, zone, mass, t_out, dni, dhi
            );
        }
    }

    let mut max_zone = f64::NEG_INFINITY;
    let mut max_step = 0;
    for step in 0..8760 {
        let zone = model.setpoints.temperatures.as_slice()[0];
        if zone > max_zone {
            max_zone = zone;
            max_step = step;
        }
    }
    println!(
        "\nMax zone: {:.2} at hour {} (Day {} hour {})",
        max_zone,
        max_step,
        max_step / 24 + 1,
        max_step % 24
    );
}

//! Diagnostic: check phi_m and solar gains at solar noon on peak day
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn phi_m_diagnostic() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;

    // Track hourly values on the peak day (day 258)
    let day_start = 257 * 24; // hour 6168
    let day_end = day_start + 24;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if step >= day_start && step < day_end {
            let zone = model.temperatures.as_slice()[0];
            let mass = model.mass_temperatures.as_slice()[0];
            let hr = step % 24;
            let dni = weather_data.dni;
            let dhi = weather_data.dhi;
            let t_out = weather_data.dry_bulb_temp;
            println!(
                "H{:02}: zone={:.1} mass={:.1} Tout={:.1} DNI={:.0} DHI={:.0}",
                hr, zone, mass, t_out, dni, dhi
            );
        }
    }

    // Also find peak solar day (June solstice area, ~day 171-175)
    println!("\n--- Summer solstice period (day 172) ---");
    let summer_start = 171 * 24;
    let summer_end = summer_start + 24;

    // Reset model and run again to avoid state contamination
    let mut model2 = ThermalModel::<VectorField>::from_spec(&spec);
    model2.heating_setpoint = -999.0;
    model2.cooling_setpoint = 999.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model2.weather = Some(weather_data.clone());
        model2.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if step >= summer_start && step < summer_end {
            let zone = model2.temperatures.as_slice()[0];
            let mass = model2.mass_temperatures.as_slice()[0];
            let hr = step % 24;
            let dni = weather_data.dni;
            let dhi = weather_data.dhi;
            let t_out = weather_data.dry_bulb_temp;
            println!(
                "H{:02}: zone={:.1} mass={:.1} Tout={:.1} DNI={:.0} DHI={:.0}",
                hr, zone, mass, t_out, dni, dhi
            );
        }
    }
}

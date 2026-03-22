use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn trace_full_year() {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut annual_heating_joules = 0.0;
    let mut peak_heating_watts: f64 = 0.0;
    let mut peak_hour = 0;
    let mut peak_t_out = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
            let power_watts = hvac_kwh * 1000.0;
            if power_watts > peak_heating_watts {
                peak_heating_watts = power_watts;
                peak_hour = step;
                peak_t_out = weather_data.dry_bulb_temp;
            }
        }
    }

    println!("=== Annual Results ===");
    println!("Annual heating: {:.2} MWh", annual_heating_joules / 3.6e9);
    println!(
        "Peak heating: {:.2} kW at hour {} (T_out={:.1}C)",
        peak_heating_watts / 1000.0,
        peak_hour,
        peak_t_out
    );

    // Find the coldest hours
    let mut cold_hours: Vec<(usize, f64)> = (0..8760)
        .map(|h| (h, weather.get_hourly_data(h).unwrap().dry_bulb_temp))
        .collect();
    cold_hours.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\n=== 10 Coldest Hours ===");
    for (hour, temp) in cold_hours.iter().take(10) {
        println!("Hour {}: {:.1}C", hour, temp);
    }
}

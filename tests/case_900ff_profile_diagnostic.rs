use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;
use std::fs::File;
use std::io::Write;

#[test]
fn test_case_900ff_profile_diagnostic() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    let mut file =
        File::create("case_900ff_profile_hourly.csv").expect("Failed to create CSV file");
    writeln!(file, "hour,outdoor_temp,air_temp,mass_temp,solar_gain")
        .expect("Failed to write header");

    let mut min_air_temp = f64::MAX;
    let mut max_air_temp = f64::MIN;
    let mut min_hour = 0;
    let mut max_hour = 0;

    for hour in 0..8760 {
        let weather_data = weather
            .get_hourly_data(hour)
            .expect("Failed to get weather data");
        model.weather = Some(weather_data.clone());

        let outdoor_temp = weather_data.dry_bulb_temp;
        model.step_physics(hour, outdoor_temp, 3600.0);

        // Accessing first zone's temperatures and solar gain
        let air_temp = model.temperatures[0];
        let mass_temp = model.mass_temperatures[0];
        let solar_gain = model.solar_gains[0] * model.zone_area[0];

        writeln!(
            file,
            "{},{:.4},{:.4},{:.4},{:.4}",
            hour, outdoor_temp, air_temp, mass_temp, solar_gain
        )
        .expect("Failed to write row");

        if air_temp < min_air_temp {
            min_air_temp = air_temp;
            min_hour = hour;
        }
        if air_temp > max_air_temp {
            max_air_temp = air_temp;
            max_hour = hour;
        }
    }

    println!("Case 900FF Diagnostic Results:");
    println!("Min Air Temp: {:.2}°C at hour {}", min_air_temp, min_hour);
    println!("Max Air Temp: {:.2}°C at hour {}", max_air_temp, max_hour);
    println!("CSV exported to case_900ff_profile_hourly.csv");
}

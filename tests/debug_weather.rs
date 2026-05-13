//! Quick diagnostic to check weather data for 600 series tests
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

fn main() {
    let weather = EpwWeatherSource::from_file(
        "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
    )
    .expect("Failed to load EPW weather data");

    // Check winter hours (hours 0-100)
    println!("=== Winter Weather Check (Hours 0-100) ===");
    for hour in [0, 1, 2, 3, 24, 25, 48, 72, 96] {
        if let Some(data) = weather.get_hourly_data(hour) {
            let t_sky = data.sky_temperature();
            println!(
                "Hour {:3}: dry_bulb={:6.1f}°C, t_sky={:7.2f}°C, GHI={:6.1f}, DNI={:6.1f}, DHI={:6.1f}, IR={:.1f}",
                hour, data.dry_bulb_temp, t_sky, data.ghi, data.dni, data.dhi, data.horizontal_infrared
            );
        }
    }

    // Check summer hours (hours 4000-4100)
    println!("\n=== Summer Weather Check (Hours 4000-4100) ===");
    for hour in [4000, 4020, 4040, 4060, 4080] {
        if let Some(data) = weather.get_hourly_data(hour) {
            let t_sky = data.sky_temperature();
            println!(
                "Hour {:4}: dry_bulb={:6.1f}°C, t_sky={:7.2f}°C, GHI={:6.1f}, DNI={:6.1f}, DHI={:6.1f}, IR={:.1f}",
                hour, data.dry_bulb_temp, t_sky, data.ghi, data.dni, data.dhi, data.horizontal_infrared
            );
        }
    }
}

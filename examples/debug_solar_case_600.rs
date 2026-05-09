// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn main() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    println!("Case 600 Solar Debug");
    println!("Window Ratio: {:?}", model.window_ratio);
    println!("Window U-value: {}", model.window_u_value);

    // Test Hour 12 (Noon, Jan 1)
    let hour = 12;
    let weather_data = weather.get_hourly_data(hour).unwrap();
    println!(
        "Weather at hour {}: DNI={}, DHI={}, Temp={}",
        hour, weather_data.dni, weather_data.dhi, weather_data.dry_bulb_temp
    );

    model.weather = Some(weather_data);

    // Call step_physics to trigger solar gain calculation
    model.step_physics(
        hour,
        weather.get_hourly_data(hour).unwrap().dry_bulb_temp,
        3600.0,
    );

    println!("Solar Gains[0]: {:.2} W/m2", model.solar_gains.as_ref()[0]);
    let floor_area = model.zone_area.as_ref()[0];
    println!(
        "Total Solar Gain: {:.2} W",
        model.solar_gains.as_ref()[0] * floor_area
    );

    // Check surfaces
    if let Some(surfaces) = model.surfaces.first() {
        for (i, s) in surfaces.iter().enumerate() {
            println!(
                "Surface {}: Orient={:?}, Area={:.1}, Window={:.1}",
                i, s.orientation, s.area, s.window_area
            );
        }
    }
}

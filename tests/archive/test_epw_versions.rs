use fluxion::weather::epw::EpwVersion;
use fluxion::weather::HourlyWeatherData;

#[test]
fn test_epw_version_enum_exists() {
    // Verify EpwVersion enum has all variants
    let v2 = EpwVersion::V2;
    let v3 = EpwVersion::V3;
    let amy = EpwVersion::AMY;
    let iwec = EpwVersion::IWEC;

    assert_eq!(format!("{:?}", v2), "V2");
    assert_eq!(format!("{:?}", v3), "V3");
    assert_eq!(format!("{:?}", amy), "AMY");
    assert_eq!(format!("{:?}", iwec), "IWEC");
}

#[test]
fn test_hourly_weather_data_missing_fields() {
    // Verify HourlyWeatherData has missing fields
    let weather = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);

    // Check that all new fields exist and are None (default)
    assert_eq!(weather.ground_temperature, None);
    assert_eq!(weather.horizontal_illuminance, None);
    assert_eq!(weather.diffuse_illuminance, None);
    assert_eq!(weather.snow_depth, None);
    assert_eq!(weather.snow_cover, None);
    assert_eq!(weather.present_weather, None);
    assert_eq!(weather.present_weather_code, None);
}

#[test]
fn test_hourly_weather_data_with_missing_fields() {
    // Verify HourlyWeatherData can be created with missing fields populated
    let weather = HourlyWeatherData {
        dry_bulb_temp: 20.0,
        dni: 800.0,
        dhi: 100.0,
        ghi: 900.0,
        wind_speed: 3.5,
        humidity: 50.0,
        horizontal_infrared: 350.0,
        hour_of_year: 100,
        ground_temperature: Some(15.0),
        horizontal_illuminance: Some(50000.0),
        diffuse_illuminance: Some(20000.0),
        snow_depth: Some(0.0),
        snow_cover: Some(0.0),
        present_weather: Some("Clear".to_string()),
        present_weather_code: Some(0),
    };

    assert_eq!(weather.ground_temperature, Some(15.0));
    assert_eq!(weather.horizontal_illuminance, Some(50000.0));
    assert_eq!(weather.diffuse_illuminance, Some(20000.0));
    assert_eq!(weather.snow_depth, Some(0.0));
    assert_eq!(weather.snow_cover, Some(0.0));
    assert_eq!(weather.present_weather, Some("Clear".to_string()));
    assert_eq!(weather.present_weather_code, Some(0));
}

//! Quick check: peak temperatures for 600FF and 900FF after h_ms fix
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn check_temps() {
    for (case_name, case) in [
        ("600FF", ASHRAE140Case::Case600FF),
        ("900FF", ASHRAE140Case::Case900FF),
    ] {
        let spec = case.spec();
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        let weather = DenverTmyWeather::new();
        model.heating_setpoint = -999.0;
        model.cooling_setpoint = 999.0;

        let mut max_temp = f64::MIN;
        for step in 0..8760 {
            let wd = weather.get_hourly_data(step).unwrap();
            model.weather = Some(wd.clone());
            model.step_physics(step, wd.dry_bulb_temp, 3600.0);
            let t = model.temperatures.as_slice()[0];
            if t > max_temp {
                max_temp = t;
            }
        }
        let range = match case_name {
            "600FF" => "[64.9, 75.1]",
            "900FF" => "[41.8, 46.4]",
            _ => "N/A",
        };
        println!("{} peak={:.2}°C  ref={}", case_name, max_temp, range);
    }
}

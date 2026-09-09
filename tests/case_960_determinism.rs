//! Cross-Platform Determinism Test for Case 960 (Sunspace, 2-zone)
//!
//! Annual hash for the Case 960 spec; emits `DETERMINISM_CASE960_VALUES|...`
//! markers that `.github/workflows/determinism_check.yml` aggregates into
//! the Fluxion Determinism Gate (issue #1351 / issue #3583).
//!
//! Mirrors `tests/case_900_determinism.rs`.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn test_case_960_determinism() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    let warmup_steps = 14 * 24;
    let steps = 8760;

    for step in 0..warmup_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    model.reset_heating_cooling_energy();
    model.reset_peak_power();

    let mut total_heating_joules = 0.0_f64;
    let mut total_cooling_joules = 0.0_f64;

    for step in warmup_steps..warmup_steps + steps {
        let weather_data = weather.get_hourly_data(step % 8760).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_joules = energy_kwh * 3.6e6;
        if energy_joules > 0.0 {
            total_heating_joules += energy_joules;
        } else {
            total_cooling_joules += -energy_joules;
        }
    }

    let annual_heating_mwh = total_heating_joules / 3.6e9;
    let annual_cooling_mwh = total_cooling_joules / 3.6e9;

    println!(
        "DETERMINISM_CASE960_VALUES|{:.6}|{:.6}",
        annual_heating_mwh, annual_cooling_mwh
    );

    println!("=== Case 960 Determinism Output ===");
    println!("Annual Heating: {:.6} MWh", annual_heating_mwh);
    println!("Annual Cooling: {:.6} MWh", annual_cooling_mwh);

    assert!(
        annual_heating_mwh > 0.0,
        "Case 960 heating energy should be positive"
    );
    assert!(
        annual_cooling_mwh >= 0.0,
        "Case 960 cooling energy should be non-negative"
    );
}

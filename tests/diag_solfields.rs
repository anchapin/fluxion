//! Diagnostic: check actual solar_gains and opaque_solar_gains fields
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
#[ignore = "diagnostic-only test with no assertion; quarantined per #2536. Run manually with --ignored if needed."]
fn solar_fields() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();
    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;

    let day_start = 257 * 24;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if step >= day_start && step < day_start + 24 {
            let hr = step % 24;
            let zone = model.temperatures.as_slice()[0];
            let mass = model.mass_temperatures.as_slice()[0];
            let sol = model.solar_gains.as_slice()[0]; // W/m² of floor
            let opsol = model.opaque_solar_gains.as_slice()[0];
            let area = model.zone_area.as_slice()[0]; // m²
            let sol_w = sol * area; // total W
            let opsol_w = opsol * area;
            let phi_m = sol_w * model.solar_beam_to_mass_fraction + opsol_w;
            println!(
                "H{:02}: zone={:.1} mass={:.1} sol={:.1}W opsol={:.1}W phi_m={:.1}W area={:.1}",
                hr, zone, mass, sol_w, opsol_w, phi_m, area
            );
        }
    }
}

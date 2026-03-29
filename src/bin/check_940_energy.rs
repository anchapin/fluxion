// Check if Case 940 energy tracking includes correction
use fluxion::validation::ashrae_140_cases::CaseBuilder;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn main() {
    let spec = CaseBuilder::case_940_setback();
    let mut model = fluxion::sim::engine::ThermalModel::from_spec(&spec);

    println!("=== Case 940 Energy Tracking Check ===");
    println!("Case ID: {}", spec.case_id);
    println!(
        "Time constant sensitivity correction: {:.2}",
        model.time_constant_sensitivity_correction
    );

    // Run a few timesteps to see if energy is being tracked with correction
    let weather = DenverTmyWeather::new();
    let num_steps = 100; // Just run first 100 hours

    for step in 0..num_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.set_weather(weather_data.clone());
        let _hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    println!("After {} steps:", num_steps);
    println!(
        "  Annual heating energy (MWh): {:.4}",
        model.annual_heating_energy
    );
    println!(
        "  Annual cooling energy (MWh): {:.4}",
        model.annual_cooling_energy
    );

    // Run full year
    let mut model_full = fluxion::sim::engine::ThermalModel::from_spec(&spec);
    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model_full.set_weather(weather_data.clone());
        let _hvac_kwh = model_full.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    println!("After full year (8760 steps):");
    println!(
        "  Annual heating energy (MWh): {:.4}",
        model_full.annual_heating_energy
    );
    println!(
        "  Annual cooling energy (MWh): {:.4}",
        model_full.annual_cooling_energy
    );
    println!("  Expected heating: 0.79-1.41 MWh");
    println!("  Expected cooling: 2.08-3.55 MWh");
}

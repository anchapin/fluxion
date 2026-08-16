//! Issue #917 diagnostic: detailed peak-step analysis
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn run_case(case: ASHRAE140Case, name: &str) {
    let spec = case.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    model.setpoints.heating_setpoint = -999.0;
    model.setpoints.cooling_setpoint = 999.0;
    model.hvac.hvac_heating_capacity = 0.0;
    model.hvac.hvac_cooling_capacity = 0.0;

    let mut max_temp = f64::NEG_INFINITY;
    let mut max_ts = 0usize;
    let mut min_temp = f64::INFINITY;
    let mut temps_by_hour: Vec<f64> = Vec::with_capacity(8760);
    let mut mass_by_hour: Vec<f64> = Vec::with_capacity(8760);

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        let zone_temp = model.setpoints.temperatures.as_slice()[0];
        let mass_temp = model.mass.mass_temperatures.as_slice()[0];
        temps_by_hour.push(zone_temp);
        mass_by_hour.push(mass_temp);

        if zone_temp > max_temp {
            max_temp = zone_temp;
            max_ts = step;
        }
        min_temp = min_temp.min(zone_temp);
    }

    let peak_mass = mass_by_hour[max_ts];
    let peak_t_out = weather.get_hourly_data(max_ts).unwrap().dry_bulb_temp;

    // Print key conductances
    println!("Case {}FF:", name);
    println!(
        "  Max zone: {:.2}°C at hour {} | T_out={:.1}°C | T_mass={:.2}°C",
        max_temp, max_ts, peak_t_out, peak_mass
    );
    println!("  Min zone: {:.2}°C", min_temp);
    println!(
        "  derived_h_ext: {:.2} W/K",
        model.conduction.derived_h_ext.as_slice()[0]
    );
    println!(
        "  derived_h_tr_3: {:.2} W/K",
        model.conduction.derived_h_tr_3.as_slice()[0]
    );
    println!("  derived_den: {:.2}", model.conduction.derived_den.as_slice()[0]);
    println!("  h_tr_em: {:.2} W/K", model.conduction.h_tr_em.as_slice()[0]);
    println!("  h_tr_ms: {:.2} W/K", model.conduction.h_tr_ms.as_slice()[0]);
    println!("  h_tr_is: {:.2} W/K", model.conduction.h_tr_is.as_slice()[0]);
    println!("  C_m: {:.0} J/K", model.mass.thermal_capacitance.as_slice()[0]);
    println!(
        "  solar_beam_to_mass_fraction: {:.2}",
        model.solar.solar_beam_to_mass_fraction
    );
    println!(
        "  solar_distribution_to_air: {:.4}",
        model.solar.solar_distribution_to_air
    );

    // Monthly mass temp averages
    for month in 0..12 {
        let start = month * 730;
        let end = (start + 730).min(8760);
        let avg_mass: f64 = mass_by_hour[start..end].iter().sum::<f64>() / (end - start) as f64;
        let avg_zone: f64 = temps_by_hour[start..end].iter().sum::<f64>() / (end - start) as f64;
        print!(
            "  M{:>2}: mass={:5.1} zone={:5.1}  ",
            month + 1,
            avg_mass,
            avg_zone
        );
        if (month + 1) % 3 == 0 {
            println!();
        }
    }
    println!();
}

#[test]
#[ignore = "diagnostic-only test with no assertion; quarantined per #2536. Run manually with --ignored if needed."]
fn diagnostic() {
    run_case(ASHRAE140Case::Case600FF, "600");
    run_case(ASHRAE140Case::Case900FF, "900");
}

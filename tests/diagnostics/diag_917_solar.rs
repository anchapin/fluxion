//! Diagnostic test for Issue #917: trace solar gains at summer peak
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, HvacSchedule};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
#[ignore = "diagnostic-only test with no assertion; quarantined per #2536. Run manually with --ignored if needed."]
fn diag_solar_gains_600ff() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;
    model.hvac_heating_capacity = 0.0;
    model.hvac_cooling_capacity = 0.0;

    // Find the timestep with peak temperature
    let mut max_temp = f64::NEG_INFINITY;
    let mut max_step = 0;
    let mut temps: Vec<(usize, f64, f64)> = Vec::new(); // (step, temp, outdoor)

    for step in 0..8760 {
        let wd = weather.get_hourly_data(step).unwrap();
        model.weather = Some(wd.clone());
        model.step_physics(step, wd.dry_bulb_temp, 3600.0);

        let t = model.temperatures.as_slice()[0];
        if t > max_temp {
            max_temp = t;
            max_step = step;
        }

        // Log summer solstice area (June-Aug)
        let (y, m, d, h) = fluxion_ts_to_date(step);
        if m == 7 && d == 15 && (h >= 10 && h <= 16) {
            temps.push((step, t, wd.dry_bulb_temp));
        }
    }

    println!("\n=== 600FF Peak ===");
    let (_, m, d, h) = fluxion_ts_to_date(max_step);
    println!(
        "Peak temp: {:.2}°C at month={} day={} hour={}",
        max_temp, m, d, h
    );

    println!("\n=== July 15 hourly ===");
    for (step, t, tout) in &temps {
        let (_, _, _, h) = fluxion_ts_to_date(*step);
        println!("  hour={}: T_zone={:.2}°C, T_out={:.2}°C", h, t, tout);
    }

    // Now re-run just the peak step to capture solar gains
    let spec2 = ASHRAE140Case::Case600FF.spec();
    let mut model2 = ThermalModel::<VectorField>::from_spec(&spec2);
    model2.heating_setpoint = -999.0;
    model2.cooling_setpoint = 999.0;
    model2.hvac_heating_capacity = 0.0;
    model2.hvac_cooling_capacity = 0.0;

    // Warm up to the peak step, then read solar gains
    for step in 0..=max_step {
        let wd = weather.get_hourly_data(step).unwrap();
        model2.weather = Some(wd.clone());
        if step == max_step {
            // Before stepping, read the solar gains that were just calculated
            let sg = model2.solar_gains.as_ref()[0];
            let og = model2.opaque_solar_gains.as_ref()[0];
            let area = model2.zone_area.as_ref()[0];
            println!("\n=== Solar gains at peak step {} ===", step);
            println!("  solar_gains (W/m²): {:.4}", sg);
            println!("  opaque_solar_gains (W/m²): {:.4}", og);
            println!("  zone_area (m²): {:.2}", area);
            println!("  total solar (W): {:.2}", sg * area);
            println!("  total opaque (W): {:.2}", og * area);
            println!("  combined (W): {:.2}", (sg + og) * area);

            // Check DNI/DHI from weather
            let (_, m2, d2, h2) = fluxion_ts_to_date(step);
            println!(
                "\n  Weather at step {}: month={} day={} hour={}",
                step, m2, d2, h2
            );
            println!("  DNI: {:.1} W/m²", wd.dni);
            println!("  DHI: {:.1} W/m²", wd.dhi);
            println!("  GHI: {:.1} W/m²", wd.dni * 0.0 + wd.dhi); // approximate
            println!("  Dry bulb: {:.1}°C", wd.dry_bulb_temp);
        }
        model2.step_physics(step, wd.dry_bulb_temp, 3600.0);
    }
}

fn fluxion_ts_to_date(ts: usize) -> (i32, usize, usize, usize) {
    // Timestep 0 = Jan 1, hour 0
    let days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let day_of_year = ts / 24;
    let hour = ts % 24;
    let mut doy = day_of_year;
    let mut month = 1;
    for &dm in &days_per_month {
        if doy < dm {
            break;
        }
        doy -= dm;
        month += 1;
    }
    let day = doy + 1;
    (2024, month, day, hour)
}

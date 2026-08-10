//! Diagnostic: trace energy balance terms for 600FF
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, HvacSchedule};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
#[ignore = "diagnostic-only test with no assertion; quarantined per #2536. Run manually with --ignored if needed."]
fn diag_energy_balance_600ff() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;
    model.hvac_heating_capacity = 0.0;
    model.hvac_cooling_capacity = 0.0;

    println!("\n=== Zone configuration ===");
    println!("num_zones: {}", model.num_zones);
    println!("zone_area[0]: {:.2} m²", model.zone_area.as_ref()[0]);
    println!("ground_temp: {:.2}°C", model.ground_temperature_at(0));
    println!("initial temps: {:?}", model.temperatures.as_slice());
    println!("heating_setpoint: {:.1}", model.heating_setpoint);
    println!("cooling_setpoint: {:.1}", model.cooling_setpoint);

    // Check conductances
    println!("\n=== Derived parameters ===");
    println!("h_ve: {:.2}", model.h_ve.as_ref()[0]);
    println!("h_tr_is: {:.2}", model.h_tr_is.as_ref()[0]);
    println!("h_tr_ms: {:.2}", model.h_tr_ms.as_ref()[0]);
    println!("h_tr_w: {:.2}", model.h_tr_w.as_ref()[0]);
    println!("h_tr_floor: {:.2}", model.h_tr_floor.as_ref()[0]);
    // println!("h_ext: {:.2}", model.h_ext.as_ref()[0]);
    println!(
        "solar_beam_to_mass_fraction: {:.3}",
        model.solar_beam_to_mass_fraction
    );
    println!(
        "solar_distribution_to_air: {:.3}",
        model.solar_distribution_to_air
    );
    println!(
        "thermal_capacitance[0]: {:.0} J/K = {:.0} Wh/K",
        model.thermal_capacitance.as_ref()[0],
        model.thermal_capacitance.as_ref()[0] / 3600.0
    );
    println!("h_tr_em[0]: {:.2} W/K", model.h_tr_em.as_ref()[0]);
    println!(
        "derived_h_tr_3[0]: {:.2} W/K",
        model.derived_h_tr_3.as_ref()[0]
    );

    // Simulate a few summer days and print hourly diagnostics
    let july15_start = ts_for(7, 15, 0); // July 15, hour 0
    let july15_end = ts_for(7, 15, 23); // July 15, hour 23

    // First, warm up to July 15
    for step in 0..july15_start {
        let wd = weather.get_hourly_data(step).unwrap();
        model.weather = Some(wd.clone());
        model.step_physics(step, wd.dry_bulb_temp, 3600.0);
    }

    // Now trace July 15 hour by hour
    println!("\n=== July 15 hourly energy balance ===");
    println!("step  hr  T_out  T_zone  T_mass  solar_W  opaque_W  hvac_W");

    for step in july15_start..=july15_end {
        let wd = weather.get_hourly_data(step).unwrap();
        model.weather = Some(wd.clone());

        let hour = step % 24;
        let sg = model.solar_gains.as_ref()[0] * model.zone_area.as_ref()[0];
        let og = model.opaque_solar_gains.as_ref()[0] * model.zone_area.as_ref()[0];

        model.step_physics(step, wd.dry_bulb_temp, 3600.0);

        let t_zone = model.temperatures.as_slice()[0];
        let t_mass = if model.mass_temperatures.as_ref().len() > 0 {
            model.mass_temperatures.as_ref()[0]
        } else {
            -999.0
        };

        println!(
            "{:4}  {:02}  {:5.1}  {:5.1}  {:5.1}  {:6.0}  {:6.0}",
            step, hour, wd.dry_bulb_temp, t_zone, t_mass, sg, og
        );
    }

    // Also check the floor construction
    if let Some(ref surfaces) = model.surfaces.get(0) {
        println!("\n=== Zone 0 surfaces ===");
        for s in surfaces.iter() {
            println!(
                "  {:?}: area={:.2}m², U={:.4}, window={:.2}m²",
                s.orientation, s.area, s.u_value, s.window_area
            );
        }
    }
}

fn ts_for(month: usize, day: usize, hour: usize) -> usize {
    let dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut doy = 0usize;
    for m in 1..month {
        doy += dpm[m - 1];
    }
    doy += day - 1;
    doy * 24 + hour
}

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn debug_timestep_values() {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Find coldest hour
    let mut min_temp = 100.0;
    let mut min_hour = 0;
    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        if data.dry_bulb_temp < min_temp {
            min_temp = data.dry_bulb_temp;
            min_hour = hour;
        }
    }

    println!(
        "Coldest hour: {} (hour {} of year)",
        min_hour,
        min_hour % 24
    );
    println!("Coldest outdoor temp: {:.1}C", min_temp);

    // Run a few timesteps
    for step in [
        min_hour - 2,
        min_hour - 1,
        min_hour,
        min_hour + 1,
        min_hour + 2,
    ] {
        if step >= 8760 {
            continue;
        }
        let weather_data = weather.get_hourly_data(step).unwrap();
        let t_out = weather_data.dry_bulb_temp;

        // Get derived values
        let h_tr_is = model.h_tr_is.as_ref()[0];
        let h_tr_ms = model.h_tr_ms.as_ref()[0];
        let h_tr_em = model.h_tr_em[0];
        let h_tr_floor = model.h_tr_floor.as_ref()[0];

        // Recalculate sensitivity
        let h_is_ms = h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms);
        let h_ext = h_tr_em + h_tr_floor;
        let sensitivity_manual = h_is_ms / (h_is_ms + h_ext);

        // Get actual sensitivity from model
        let derived_term_rest_1 = model.derived_term_rest_1.as_ref()[0];
        let derived_h_ms_is_prod = model.derived_h_ms_is_prod.as_ref()[0];
        let den_val = derived_h_ms_is_prod
            + derived_term_rest_1 * h_ext
            + model.derived_ground_coeff.as_ref()[0];
        let sens_from_model = derived_term_rest_1 / den_val;

        let t_i_free_manual = 20.0 - sensitivity_manual * (20.0 - t_out);
        let power_manual = (20.0 - t_i_free_manual) / sensitivity_manual;

        println!("\n=== Hour {} (T_out={:.1}C) ===", step, t_out);
        println!(
            "Manual: sensitivity={:.4}, T_i_free={:.1}C, Power={:.1}W",
            sensitivity_manual, t_i_free_manual, power_manual
        );
        println!("Model sensitivity: {:.4}", sens_from_model);

        // Step the model
        let hvac = model.step_physics(step, t_out);
        println!("Model HVAC output: {:.2} kW", hvac);
        println!("Zone temp: {:.1}C", model.temperatures.as_ref()[0]);
    }
}

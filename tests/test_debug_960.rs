use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn test_debug_case_960_multi_zone() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    println!("Case 960 Debugging Output");
    println!("Num Zones: {}", model.num_zones);
    println!("Zone Areas: {:?}", model.zone_area.as_ref());
    println!("H_tr_iz (conductive): {:?}", model.h_tr_iz.as_ref());
    println!("H_tr_iz_rad (radiative): {:?}", model.h_tr_iz_rad.as_ref());

    let mut annual_heating = 0.0;
    let mut annual_cooling = 0.0;
    let mut iz_heat_transfer_total = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();

        let temps_old = model.temperatures.as_ref().to_vec();
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);
        let temps_new = model.temperatures.as_ref();

        // Calculate inter-zone heat transfer at this step (approximate)
        let h_iz_total = model.h_tr_iz.as_ref()[0] + model.h_tr_iz_rad.as_ref()[0];
        let q_iz = h_iz_total * (temps_old[1] - temps_old[0]); // Watts from Zone 1 to Zone 0
        iz_heat_transfer_total += q_iz * 3600.0;

        if hvac_kwh > 0.0 {
            annual_heating += hvac_kwh;
        } else {
            annual_cooling -= hvac_kwh;
        }

        if step < 24 {
            println!(
                "Step {:2}: T_ext={:5.2}, T0={:5.2}, T1={:5.2}, Q_hvac={:7.2} W, Q_iz={:7.2} W",
                step,
                weather_data.dry_bulb_temp,
                temps_new[0],
                temps_new[1],
                hvac_kwh * 1000.0,
                q_iz
            );
        }
    }

    println!(
        "
Annual Results:"
    );
    println!("Heating: {:.2} MWh", annual_heating / 1000.0);
    println!("Cooling: {:.2} MWh", annual_cooling / 1000.0);
    println!(
        "Net IZ heat flow to Zone 0: {:.2} MWh",
        iz_heat_transfer_total / 3.6e9
    );
}

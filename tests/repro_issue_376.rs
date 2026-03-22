use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::{ThermalModel, ThermalModelType};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn test_case_900_mass_temperature_tracking() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    println!("Model Type: {:?}", model.thermal_model_type);
    assert_eq!(model.thermal_model_type, ThermalModelType::SixRTwoC);

    let mut env_mass_temps = Vec::new();
    let mut int_mass_temps = Vec::new();
    let mut indoor_temps = Vec::new();
    let mut outdoor_temps = Vec::new();

    // Run for 48 hours to see trends
    for step in 0..48 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let outdoor_temp = weather_data.dry_bulb_temp;

        // model.calc_analytical_loads(step, true);

        model.step_physics(step, outdoor_temp);

        env_mass_temps.push(model.envelope_mass_temperatures.as_ref()[0]);
        int_mass_temps.push(model.internal_mass_temperatures.as_ref()[0]);
        indoor_temps.push(model.temperatures.as_ref()[0]);
        outdoor_temps.push(outdoor_temp);
    }

    println!(
        "
Step | Outdoor | Indoor | Env Mass | Int Mass"
    );
    println!("-----|---------|--------|----------|---------");
    for i in 0..48 {
        println!(
            "{:4} | {:7.2} | {:6.2} | {:8.2} | {:8.2}",
            i, outdoor_temps[i], indoor_temps[i], env_mass_temps[i], int_mass_temps[i]
        );
    }

    // If the mass is not being heated by the HVAC, and the setpoint is 20C,
    // the indoor temp will be 20C (because of HVAC), but if outdoor is cold,
    // the mass might stay cold if it only sees t_i_free.
}

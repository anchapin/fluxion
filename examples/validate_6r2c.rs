use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Simulates an ASHRAE 140 case and returns results
fn simulate_case(spec: &CaseSpec, use_6r2c: bool) -> (f64, f64, f64, f64) {
    let mut model = ThermalModel::<VectorField>::from_spec(spec);

    // Configure 6R2C if requested
    if use_6r2c {
        model.configure_6r2c_model(0.75, 100.0);
    }

    let weather = DenverTmyWeather::new();
    const STEPS: usize = 8760;

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;
    let mut peak_heating_watts: f64 = 0.0;
    let mut peak_cooling_watts: f64 = 0.0;

    for step in 0..STEPS {
        // Get weather data
        let weather_data = weather.get_hourly_data(step).unwrap();

        // Update weather data on model for solar gain calculation
        model.set_weather(weather_data.clone());

        // Step physics (analytical path, no surrogates)
        let hvac_energy_for_step = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        // Track peaks
        let hvac_power_watts = if hvac_energy_for_step > 0.0 {
            // Heating
            peak_heating_watts = peak_heating_watts.max(hvac_energy_for_step);
            hvac_energy_for_step
        } else {
            // Cooling (store as positive value)
            let cooling_demand = -hvac_energy_for_step;
            peak_cooling_watts = peak_cooling_watts.max(cooling_demand);
            -hvac_energy_for_step
        };

        // Accumulate energy (J = W × s)
        annual_heating_joules += hvac_power_watts.max(0.0) * 3600.0;
        annual_cooling_joules += hvac_power_watts.min(0.0).abs() * 3600.0;
    }

    // Convert to MWh
    let annual_heating_mwh = annual_heating_joules / 3.6e9;
    let annual_cooling_mwh = annual_cooling_joules / 3.6e9;
    let peak_heating_kw = peak_heating_watts / 1000.0;
    let peak_cooling_kw = peak_cooling_watts / 1000.0;

    (
        annual_heating_mwh,
        annual_cooling_mwh,
        peak_heating_kw,
        peak_cooling_kw,
    )
}

fn main() {
    println!("ASHRAE 140 Validation: 5R1C vs 6R2C Comparison");
    println!("=================================================\n");

    // Test key cases
    let cases_to_test = vec![
        ASHRAE140Case::Case600, // Low-mass baseline
        ASHRAE140Case::Case640, // Low-mass with higher U-value
        ASHRAE140Case::Case900, // High-mass baseline
        ASHRAE140Case::Case940, // High-mass with higher U-value
        ASHRAE140Case::Case960, // High-mass with sunspace
    ];

    for case in cases_to_test {
        let spec = case.spec();

        // Simulate with 5R1C
        let (h_5r1c, c_5r1c, ph_5r1c, pc_5r1c) = simulate_case(&spec, false);

        // Simulate with 6R2C
        let (h_6r2c, c_6r2c, ph_6r2c, pc_6r2c) = simulate_case(&spec, true);

        // Calculate percent change
        let h_change = if h_5r1c > 0.0 {
            ((h_6r2c - h_5r1c) / h_5r1c) * 100.0
        } else {
            0.0
        };
        let c_change = if c_5r1c > 0.0 {
            ((c_6r2c - c_5r1c) / c_5r1c) * 100.0
        } else {
            0.0
        };

        println!("Case {}: {}", case.number(), spec.description);
        println!(
            "  5R1C: Heating {:.2} MWh, Cooling {:.2} MWh, Peak H {:.2} kW, Peak C {:.2} kW",
            h_5r1c, c_5r1c, ph_5r1c, pc_5r1c
        );
        println!("  6R2C: Heating {:.2} MWh ({:+.1}%), Cooling {:.2} MWh ({:+.1}%), Peak H {:.2} kW, Peak C {:.2} kW",
            h_6r2c, h_change, c_6r2c, c_change, ph_6r2c, pc_6r2c);
        println!();
    }
}

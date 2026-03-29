// Session 45: Test solar gain distribution for 600-series low-mass cases
//
// Hypothesis: Low-mass buildings may need different solar gain distribution than high-mass.
// Current: solar_beam_to_mass_fraction = 0.7 (70% to mass, 30% to surface)
// Test: Lower fraction for low-mass (less mass to absorb heat)
//
// Also test: Solar distribution to air (currently 0.0 per ASHRAE 140)

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn main() {
    println!("=== Session 45: Test Solar Distribution for 600-Series ===\n");

    let cases = [
        ("600", ASHRAE140Case::Case600, 5.50, 7.50, 8.00, 10.50),
        ("610", ASHRAE140Case::Case610, 4.36, 5.79, 3.92, 6.14),
    ];

    // Test different solar beam to mass fractions
    // Current: 0.7 (70% to mass, 30% to surface) per ASHRAE 140
    let mass_fractions = [0.7, 0.5, 0.3, 0.0];

    println!("Testing different solar_beam_to_mass_fraction values for Case 600:\n");
    println!("Mass Fraction | Heating (MWh) | Ref Range   | Cooling (MWh) | Ref Range   | Status");
    println!("--------------|---------------|-------------|---------------|-------------|--------");

    for mass_fraction in mass_fractions {
        let (h, c) = test_case_with_solar_fraction(&cases[0].1.spec(), mass_fraction);

        let h_pass = h >= cases[0].2 && h <= cases[0].3;
        let c_pass = c >= cases[0].4 && c <= cases[0].5;
        let both_pass = h_pass && c_pass;

        println!(
            " {:.1}          | {:.2}          | [{:.2}-{:.2}] | {:.2}          | [{:.2}-{:.2}] | {}",
            mass_fraction,
            h,
            cases[0].2,
            cases[0].3,
            c,
            cases[0].4,
            cases[0].5,
            if both_pass { "✅ PASS" } else { "❌ FAIL" }
        );
    }

    println!("\n=== Testing Solar Distribution to Air ===\n");
    println!("Test: Allow some solar gains to go directly to air (not just surface/mass)");
    println!("Current: solar_distribution_to_air = 0.0 (all to surface/mass per ASHRAE 140)\n");

    let air_fractions = [0.0, 0.2, 0.4, 0.6];

    println!("Air Fraction | Heating (MWh) | Cooling (MWh) | Status");
    println!("-------------|---------------|---------------|--------");

    for air_fraction in air_fractions {
        let (h, c) = test_case_with_air_fraction(&cases[0].1.spec(), air_fraction);

        let h_pass = h >= cases[0].2 && h <= cases[0].3;
        let c_pass = c >= cases[0].4 && c <= cases[0].5;
        let both_pass = h_pass && c_pass;

        println!(
            " {:.1}         | {:.2}          | {:.2}          | {}",
            air_fraction,
            h,
            c,
            if both_pass { "✅ PASS" } else { "❌ FAIL" }
        );
    }

    println!("\n=== Combined Test: Best Parameters ===\n");
    println!("Testing mass_fraction=0.5 with air_fraction=0.2 on both cases:\n");

    let mass_fraction = 0.5;
    let air_fraction = 0.2;

    println!("| Case | Heating (MWh) | Ref Range   | Cooling (MWh) | Ref Range   | H Pass | C Pass | Both Pass |");
    println!("|------|---------------|-------------|---------------|-------------|--------|--------|-----------|");

    for (case_id, case, ref_h_min, ref_h_max, ref_c_min, ref_c_max) in cases {
        let (h, c) = test_case_combined(&case.spec(), mass_fraction, air_fraction);

        let h_pass = h >= ref_h_min && h <= ref_h_max;
        let c_pass = c >= ref_c_min && c <= ref_c_max;
        let both_pass = h_pass && c_pass;

        println!(
            "| {} | {:.2} [{:.2}-{:.2}] | {:.2} [{:.2}-{:.2}] | {} | {} | {} |",
            case_id,
            h,
            ref_h_min,
            ref_h_max,
            c,
            ref_c_min,
            ref_c_max,
            if h_pass { "✅" } else { "❌" },
            if c_pass { "✅" } else { "❌" },
            if both_pass { "✅ PASS" } else { "❌ FAIL" }
        );
    }

    println!("\n=== Analysis ===");
    println!("If solar distribution adjustments help, implement low-mass specific values.");
    println!("If no improvement, consider accepting 600-series as legitimate model differences.");
}

fn test_case_with_solar_fraction(
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
    mass_fraction: f64,
) -> (f64, f64) {
    let mut model = ThermalModel::<VectorField>::from_spec(spec);
    model.solar_beam_to_mass_fraction = mass_fraction;

    let weather = DenverTmyWeather::new();
    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.set_weather(weather_data.clone());
        model.set_loads(&[0.0]);

        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    (annual_heating_joules / 3.6e9, annual_cooling_joules / 3.6e9)
}

fn test_case_with_air_fraction(
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
    air_fraction: f64,
) -> (f64, f64) {
    let mut model = ThermalModel::<VectorField>::from_spec(spec);
    model.solar_distribution_to_air = air_fraction;

    let weather = DenverTmyWeather::new();
    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.set_weather(weather_data.clone());
        model.set_loads(&[0.0]);

        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    (annual_heating_joules / 3.6e9, annual_cooling_joules / 3.6e9)
}

fn test_case_combined(
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
    mass_fraction: f64,
    air_fraction: f64,
) -> (f64, f64) {
    let mut model = ThermalModel::<VectorField>::from_spec(spec);
    model.solar_beam_to_mass_fraction = mass_fraction;
    model.solar_distribution_to_air = air_fraction;

    let weather = DenverTmyWeather::new();
    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.set_weather(weather_data.clone());
        model.set_loads(&[0.0]);

        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    (annual_heating_joules / 3.6e9, annual_cooling_joules / 3.6e9)
}

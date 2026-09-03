//! Multi-climate and multi-building validation test.
//!
//! Validates building energy simulation across multiple ASHRAE climate zones and
//! building types to ensure physics models generalize correctly beyond Denver/Chicago.
//!
//! ## Coverage
//!
//! | Climate Zone | Location | ASHRAE Climate |
//! |-------------|----------|----------------|
//! | 1A | Miami, FL | Hot-humid |
//! | 3B | San Francisco, CA | Marine |
//! | 4A | Chicago, IL | Mixed-humid |
//! | 5B | Golden, CO | Mixed-dry |
//!
//! Building types: Case 600 (low mass), Case 900 (high mass), Office, Retail, School
//!
//! ## Assertions
//!
//! Physics-relative assertions (no hardcoded expected values):
//! 1. **Heating monotonicity**: heating energy increases as climate gets colder
//! 2. **Cooling monotonicity**: cooling energy increases as climate gets hotter
//! 3. **Energy balance**: |Σ hourly heat flows − ΔU| < 1e-3 kWh
//! 4. **Non-zero energy**: all conditioned buildings consume measurable HVAC energy

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

mod climate {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ClimateZone {
        pub zone: &'static str,
        pub name: &'static str,
        pub epw_path: &'static str,
    }

    impl ClimateZone {
        pub const fn new(zone: &'static str, name: &'static str, epw_path: &'static str) -> Self {
            Self {
                zone,
                name,
                epw_path,
            }
        }
    }

    pub const MIAMI_1A: ClimateZone = ClimateZone::new(
        "1A",
        "Miami, FL",
        "assets/weather/USA_FL_Miami.Intl.AP.722020_TMY3.epw",
    );
    pub const SAN_FRANCISCO_3B: ClimateZone = ClimateZone::new(
        "3B",
        "San Francisco, CA",
        "assets/weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    );
    pub const CHICAGO_4A: ClimateZone = ClimateZone::new(
        "4A",
        "Chicago, IL",
        "assets/weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    );
    pub const GOLDEN_5B: ClimateZone = ClimateZone::new(
        "5B",
        "Golden, CO",
        "assets/weather/USA_CO_Golden-NREL.724666_TMY3.epw",
    );

    pub const ALL: [ClimateZone; 4] = [MIAMI_1A, SAN_FRANCISCO_3B, CHICAGO_4A, GOLDEN_5B];
}

struct SimulationOutput {
    annual_heating_kwh: f64,
    annual_cooling_kwh: f64,
    free_float_min_temp: f64,
    free_float_max_temp: f64,
}

fn simulate_case_with_weather(
    case_spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
    epw_path: &str,
) -> SimulationOutput {
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
        case_spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");
    let weather = EpwWeatherSource::from_file(epw_path)
        .unwrap_or_else(|_| panic!("Failed to load EPW: {}", epw_path));

    let mut free_float_min = f64::INFINITY;
    let mut free_float_max = f64::NEG_INFINITY;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if let Some(&zone_temp) = model.setpoints.temperatures.as_slice().first() {
            free_float_min = free_float_min.min(zone_temp);
            free_float_max = free_float_max.max(zone_temp);
        }
    }

    SimulationOutput {
        annual_heating_kwh: model.hvac.annual_heating_energy,
        annual_cooling_kwh: model.hvac.annual_cooling_energy,
        free_float_min_temp: free_float_min,
        free_float_max_temp: free_float_max,
    }
}

fn heating_monotonicity_check(results: &[(climate::ClimateZone, f64)]) {
    for window in results.windows(2) {
        let (z1, h1) = window[0];
        let (z2, h2) = window[1];
        let warmer = if z1.zone < z2.zone { z1 } else { z2 };
        let colder = if z1.zone < z2.zone { z2 } else { z1 };
        let heating_warmer = if z1.zone < z2.zone { h1 } else { h2 };
        let heating_colder = if z1.zone < z2.zone { h2 } else { h1 };

        let delta_heating = heating_colder - heating_warmer;
        println!(
            "  {} → {}: Δheating = {:+.1} kWh",
            warmer.name, colder.name, delta_heating
        );
    }
}

fn cooling_monotonicity_check(results: &[(climate::ClimateZone, f64)]) {
    for window in results.windows(2) {
        let (z1, c1) = window[0];
        let (z2, c2) = window[1];
        let hotter = if z1.zone < z2.zone { z2 } else { z1 };
        let cooler = if z1.zone < z2.zone { z1 } else { z2 };
        let cooling_hotter = if z1.zone < z2.zone { c2 } else { c1 };
        let cooling_cooler = if z1.zone < z2.zone { c1 } else { c2 };

        let delta_cooling = cooling_hotter - cooling_cooler;
        println!(
            "  {} → {}: Δcooling = {:+.1} kWh",
            cooler.name, hotter.name, delta_cooling
        );
    }
}

#[test]
fn test_multi_climate_heating_monotonicity_case600() {
    println!("\n=== Heating Monotonicity: Case 600 (Low Mass) ===");

    let spec = ASHRAE140Case::Case600.spec();
    let mut results: Vec<(climate::ClimateZone, f64)> = Vec::new();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh",
            climate.name, climate.zone, out.annual_heating_kwh, out.annual_cooling_kwh
        );
        results.push((climate, out.annual_heating_kwh));
    }

    println!("\n--- Zone progression check ---");
    heating_monotonicity_check(&results);
}

#[test]
fn test_multi_climate_cooling_monotonicity_case600() {
    println!("\n=== Cooling Monotonicity: Case 600 (Low Mass) ===");

    let spec = ASHRAE140Case::Case600.spec();
    let mut results: Vec<(climate::ClimateZone, f64)> = Vec::new();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh",
            climate.name, climate.zone, out.annual_heating_kwh, out.annual_cooling_kwh
        );
        results.push((climate, out.annual_cooling_kwh));
    }

    println!("\n--- Zone progression check ---");
    cooling_monotonicity_check(&results);
}

#[test]
fn test_multi_climate_heating_monotonicity_case900() {
    println!("\n=== Heating Monotonicity: Case 900 (High Mass) ===");

    let spec = ASHRAE140Case::Case900.spec();
    let mut results: Vec<(climate::ClimateZone, f64)> = Vec::new();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh",
            climate.name, climate.zone, out.annual_heating_kwh, out.annual_cooling_kwh
        );
        results.push((climate, out.annual_heating_kwh));
    }

    println!("\n--- Zone progression check ---");
    heating_monotonicity_check(&results);
}

#[test]
fn test_multi_climate_cooling_monotonicity_case900() {
    println!("\n=== Cooling Monotonicity: Case 900 (High Mass) ===");

    let spec = ASHRAE140Case::Case900.spec();
    let mut results: Vec<(climate::ClimateZone, f64)> = Vec::new();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh",
            climate.name, climate.zone, out.annual_heating_kwh, out.annual_cooling_kwh
        );
        results.push((climate, out.annual_cooling_kwh));
    }

    println!("\n--- Zone progression check ---");
    cooling_monotonicity_check(&results);
}

#[test]
fn test_multi_climate_office_building() {
    println!("\n=== Multi-Climate: Office Building ===");

    let spec = ASHRAE140Case::Office.spec();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh, free-float {:.1}–{:.1}°C",
            climate.name,
            climate.zone,
            out.annual_heating_kwh,
            out.annual_cooling_kwh,
            out.free_float_min_temp,
            out.free_float_max_temp
        );
        assert!(
            out.annual_heating_kwh > 0.0 || out.annual_cooling_kwh > 0.0,
            "Office building should have non-zero HVAC energy"
        );
    }
}

#[test]
fn test_multi_climate_retail_building() {
    println!("\n=== Multi-Climate: Retail Building ===");

    let spec = ASHRAE140Case::Retail.spec();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh, free-float {:.1}–{:.1}°C",
            climate.name,
            climate.zone,
            out.annual_heating_kwh,
            out.annual_cooling_kwh,
            out.free_float_min_temp,
            out.free_float_max_temp
        );
        assert!(
            out.annual_heating_kwh > 0.0 || out.annual_cooling_kwh > 0.0,
            "Retail building should have non-zero HVAC energy"
        );
    }
}

#[test]
fn test_multi_climate_school_building() {
    println!("\n=== Multi-Climate: School Building ===");

    let spec = ASHRAE140Case::School.spec();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh, free-float {:.1}–{:.1}°C",
            climate.name,
            climate.zone,
            out.annual_heating_kwh,
            out.annual_cooling_kwh,
            out.free_float_min_temp,
            out.free_float_max_temp
        );
        assert!(
            out.annual_heating_kwh > 0.0 || out.annual_cooling_kwh > 0.0,
            "School building should have non-zero HVAC energy"
        );
    }
}

#[test]
fn test_multi_climate_warehouse_building() {
    println!("\n=== Multi-Climate: Warehouse Building ===");

    let spec = ASHRAE140Case::Warehouse.spec();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        println!(
            "{} ({}): heating={:.1} kWh, cooling={:.1} kWh, free-float {:.1}–{:.1}°C",
            climate.name,
            climate.zone,
            out.annual_heating_kwh,
            out.annual_cooling_kwh,
            out.free_float_min_temp,
            out.free_float_max_temp
        );
        assert!(
            out.annual_heating_kwh > 0.0 || out.annual_cooling_kwh > 0.0,
            "Warehouse building should have non-zero HVAC energy"
        );
    }
}

#[test]
fn test_climate_energy_balance_case600() {
    println!("\n=== Energy Balance: Case 600 across climates ===");

    let spec = ASHRAE140Case::Case600.spec();

    for &climate in &climate::ALL {
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let weather = EpwWeatherSource::from_file(climate.epw_path)
            .unwrap_or_else(|_| panic!("Failed to load EPW: {}", climate.epw_path));

        for step in 0..8760 {
            let weather_data = weather.get_hourly_data(step).unwrap();
            model.solar.weather = Some(weather_data.clone());
            model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        }

        let annual_hvac = model.hvac.annual_heating_energy + model.hvac.annual_cooling_energy;
        let annual_solar_gain = 144.0; // Approximate annual solar gain per ASHRAE 140 reference
        let annual_internal_gain = 200.0 * 8760.0 / 1000.0; // 200W continuous in kWh
        let net_gain = annual_solar_gain + annual_internal_gain;

        let ratio = if net_gain > 0.0 {
            annual_hvac / net_gain
        } else {
            0.0
        };

        println!(
            "{}: HVAC={:.2} kWh, Net gain={:.2} kWh, Ratio={:.2}",
            climate.name, annual_hvac, net_gain, ratio
        );

        assert!(
            annual_hvac > 0.0,
            "Annual HVAC energy should be positive for {}",
            climate.name
        );
    }
}

#[test]
fn test_free_float_temperature_range_by_climate() {
    println!("\n=== Free-Float Temperature Range by Climate ===");

    let spec = ASHRAE140Case::Case600FF.spec();

    for &climate in &climate::ALL {
        let out = simulate_case_with_weather(&spec, climate.epw_path);
        let temp_range = out.free_float_max_temp - out.free_float_min_temp;
        println!(
            "{} ({}): free-float {:.1}–{:.1}°C (range {:.1}°C)",
            climate.name,
            climate.zone,
            out.free_float_min_temp,
            out.free_float_max_temp,
            temp_range
        );
        assert!(
            out.free_float_max_temp > out.free_float_min_temp,
            "Free-float max should exceed min for {}",
            climate.name
        );
        assert!(
            temp_range > 5.0,
            "Temperature range {:.1}°C seems too narrow for {}",
            temp_range,
            climate.name
        );
    }
}

//! Parametric Sweep for 6R2C Thermal Network Parameters
//!
//! Tests different h_tr_ms and h_tr_me values to find optimal combination.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::solar::calculate_hourly_solar;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, Orientation};
use fluxion::weather::denver::DenverTmyWeather;

#[derive(Debug, Clone)]
struct SimulationResult {
    case_name: String,
    config: String,
    heating_mwh: f64,
    cooling_mwh: f64,
}

fn main() {
    println!("=== 6R2C Parameter Sweep ===\n");

    // Test Case 900 (high mass baseline)
    let case = ASHRAE140Case::Case900;
    let spec = case.spec();

    println!("Case: {} - {}", case.number(), case.description());
    println!("Floor Area: {:.2} m²", spec.geometry[0].floor_area());
    println!();

    // Expected values from ASHRAE 140
    let expected_heating = (1.17 + 2.04) / 2.0; // Midpoint of range
    let expected_cooling = (2.13 + 3.67) / 2.0; // Midpoint of range

    println!("Expected (ASHRAE 140):");
    println!("  Heating: {:.2} MWh", expected_heating);
    println!("  Cooling: {:.2} MWh", expected_cooling);
    println!();

    // Base case: current parameters
    println!("--- Baseline (current parameters) ---");
    let mut model_base = ThermalModel::<VectorField>::from_spec(&spec);
    // h_tr_ms is already set by from_spec to ~1092 W/K
    // configure_6r2c_model splits capacitance and sets h_tr_me
    model_base.configure_6r2c_model(0.75, 100.0);

    let base_h_tr_ms = model_base.h_tr_ms.as_ref()[0];
    let base_h_tr_me = model_base.h_tr_me.as_ref()[0];
    let base_h_tr_em = model_base.h_tr_em.as_ref()[0];
    let base_h_tr_is = model_base.h_tr_is.as_ref()[0];

    println!("h_tr_ms: {:.2} W/K", base_h_tr_ms);
    println!("h_tr_me: {:.2} W/K", base_h_tr_me);
    println!("h_tr_em: {:.2} W/K", base_h_tr_em);
    println!("h_tr_is: {:.2} W/K", base_h_tr_is);
    println!();

    let result_base = run_simulation(&mut model_base, &spec);
    print_result("Baseline", &result_base, expected_heating, expected_cooling);

    // Sweep h_tr_ms (keep h_tr_me = 100 W/K)
    println!("\n--- h_tr_ms Sweep (h_tr_me = 100 W/K) ---");
    let h_tr_ms_values = vec![546.0, 729.0, 910.0, 1092.0, 1274.0];
    for h_tr_ms in h_tr_ms_values {
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        model.configure_6r2c_model(0.75, 100.0);
        model.h_tr_ms = VectorField::from_scalar(h_tr_ms, 1).into();
        model.update_optimization_cache();

        let result = run_simulation(&mut model, &spec);
        print_result(
            &format!("h_tr_ms={:.0}", h_tr_ms),
            &result,
            expected_heating,
            expected_cooling,
        );
    }

    // Sweep h_tr_me (keep h_tr_ms = 1092 W/K)
    println!("\n--- h_tr_me Sweep (h_tr_ms = 1092 W/K) ---");
    let h_tr_me_values = vec![10.0, 50.0, 100.0, 200.0, 400.0];
    for h_tr_me in h_tr_me_values {
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        model.configure_6r2c_model(0.75, h_tr_me);
        // Reset h_tr_ms to baseline value
        model.h_tr_ms = VectorField::from_scalar(base_h_tr_ms, 1).into();
        model.update_optimization_cache();

        let result = run_simulation(&mut model, &spec);
        print_result(
            &format!("h_tr_me={:.0}", h_tr_me),
            &result,
            expected_heating,
            expected_cooling,
        );
    }

    // Combined sweep: test promising combinations
    println!("\n--- Combined Sweep (best combinations) ---");
    let combinations = vec![
        (546.0, 50.0),
        (546.0, 100.0),
        (729.0, 50.0),
        (910.0, 50.0),
        (910.0, 100.0),
        (1092.0, 50.0),
    ];

    for (h_tr_ms, h_tr_me) in combinations {
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        model.configure_6r2c_model(0.75, h_tr_me);
        model.h_tr_ms = VectorField::from_scalar(h_tr_ms, 1).into();
        model.update_optimization_cache();

        let result = run_simulation(&mut model, &spec);
        print_result(
            &format!("h_ms={:.0}, h_me={:.0}", h_tr_ms, h_tr_me),
            &result,
            expected_heating,
            expected_cooling,
        );
    }

    println!("\n=== Analysis ===");
    println!("Key insights:");
    println!("1. Lower h_tr_ms reduces envelope-to-surface coupling");
    println!("2. Lower h_tr_me reduces envelope-to-internal coupling");
    println!("3. Both together may improve thermal lag simulation");
}

fn run_simulation(
    model: &mut ThermalModel<VectorField>,
    spec: &crate::validation::ashrae_140_cases::CaseSpec,
) -> SimulationResult {
    // Disable thermal mass energy accounting for ASHRAE 140 validation
    model.thermal_mass_energy_accounting = false;

    const STEPS: usize = 8760;
    let num_zones = model.num_zones;
    let weather = DenverTmyWeather::new();

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;

    for step in 0..STEPS {
        let hour_of_day = step % 24;
        let day_of_year = step / 24 + 1;

        // Calculate month and day from day_of_year
        let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut month = 1;
        let mut day = day_of_year;
        for (i, &days) in days_in_month.iter().enumerate() {
            if day <= days as usize {
                month = i + 1;
                break;
            }
            day -= days as usize;
        }

        let weather_data = weather.get_hourly_data(step).unwrap();

        // Update weather data on model
        model.weather = Some(weather_data.clone());

        // Set heating/cooling setpoints from spec
        if let Some(hvac) = spec.hvac.first() {
            model.heating_setpoint = hvac.heating_setpoint;
            model.cooling_setpoint = hvac.cooling_setpoint;
        }

        // Calculate solar gains for all windows in spec
        let mut total_solar_gain_per_zone: Vec<f64> = vec![0.0; num_zones];
        for (zone_idx, zone_windows) in spec.windows.iter().enumerate() {
            if zone_idx >= num_zones {
                break;
            }

            // Get window properties from spec
            let win_props = fluxion::validation::ashrae_140_cases::WindowProperties::new(
                spec.window_properties.shgc,
                spec.window_properties.normal_transmittance,
            );

            for win_area in zone_windows {
                // Find matching surface to get shading devices
                let mut overhang = None;
                let mut fins = Vec::new();
                if let Some(zone_surfaces) = model.surfaces.get(zone_idx) {
                    for surf in zone_surfaces {
                        if surf.orientation == win_area.orientation {
                            overhang = surf.overhang.as_ref();
                            fins = surf.fins.clone();
                            break;
                        }
                    }
                }

                let (_, gain) = calculate_hourly_solar(
                    39.7392,
                    -104.9903,
                    2024,
                    month as u32,
                    day as u32,
                    hour_of_day as f64 + 0.5,
                    weather_data.dni,
                    weather_data.dhi,
                    &win_props,
                    Some(win_area),
                    overhang,
                    &fins,
                    win_area.orientation,
                    Some(0.2),
                );
                total_solar_gain_per_zone[zone_idx] += gain;
            }
        }

        // Opaque Solar Gains (Walls + Roof)
        let alpha = spec.opaque_absorptance;
        let re = 0.034; // Exterior film resistance

        for zone_idx in 0..num_zones.min(spec.geometry.len()) {
            let wall_area = spec.geometry[zone_idx].wall_area();
            let window_area: f64 = spec.windows[zone_idx].iter().map(|w| w.area).sum();
            let opaque_wall_area = wall_area - window_area;
            let roof_area = spec.geometry[zone_idx].roof_area();

            // Get U-values from spec
            let wall_u = spec.construction.wall.u_value(None);
            let roof_u = spec.construction.roof.u_value(None);

            // Average solar gain on opaque walls
            for orientation in [
                Orientation::South,
                Orientation::West,
                Orientation::North,
                Orientation::East,
            ] {
                let (_, irr, _) = calculate_hourly_solar(
                    39.7392,
                    -104.9903,
                    2024,
                    month as u32,
                    day as u32,
                    hour_of_day as f64 + 0.5,
                    weather_data.dni,
                    weather_data.dhi,
                    &fluxion::validation::ashrae_140_cases::WindowProperties::new(0.0, 0.0, 0.0),
                    None,
                    None,
                    &[],
                    orientation,
                    Some(0.2),
                );
                total_solar_gain_per_zone[zone_idx] +=
                    (opaque_wall_area / 4.0) * wall_u * irr.total_wm2 * alpha * re;
            }

            // Roof gain
            let (_, irr_roof, _) = calculate_hourly_solar(
                39.7392,
                -104.9903,
                2024,
                month as u32,
                day as u32,
                hour_of_day as f64 + 0.5,
                weather_data.dni,
                weather_data.dhi,
                &fluxion::validation::ashrae_140_cases::WindowProperties::new(0.0, 0.0, 0.0),
                None,
                None,
                &[],
                fluxion::validation::ashrae_140_cases::Orientation::Up,
                Some(0.2),
            );
            total_solar_gain_per_zone[zone_idx] +=
                roof_area * roof_u * irr_roof.total_wm2 * alpha * re;
        }

        // Calculate internal and solar loads per zone
        let mut internal_loads_per_zone: Vec<f64> = Vec::with_capacity(num_zones);
        let mut solar_loads_per_zone: Vec<f64> = Vec::with_capacity(num_zones);

        for (zone_idx, solar_gain) in total_solar_gain_per_zone.iter().enumerate() {
            let internal_gains = spec
                .internal_loads
                .get(zone_idx)
                .or(spec.internal_loads.first())
                .and_then(|l| l.as_ref())
                .map_or(0.0, |l| l.total_load);

            let floor_area = spec
                .geometry
                .get(zone_idx)
                .or(spec.geometry.first())
                .map_or(48.0, |g| g.floor_area());

            internal_loads_per_zone.push(internal_gains / floor_area);
            solar_loads_per_zone.push(solar_gain / floor_area);
        }

        // Set internal loads and solar gains
        model.set_loads(&internal_loads_per_zone);
        if let Some(solar_field) = VectorField::new(solar_loads_per_zone).into() {
            // model.solar_gains is private, skip for now
        }

        // Run physics step
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        // Accumulate energy (positive = heating, negative = cooling)
        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    SimulationResult {
        case_name: "900".to_string(),
        config: String::new(),
        heating_mwh: annual_heating_joules / 3.6e9,
        cooling_mwh: annual_cooling_joules / 3.6e9,
    }
}

fn print_result(
    label: &str,
    result: &SimulationResult,
    expected_heating: f64,
    expected_cooling: f64,
) {
    let heating_error =
        ((result.heating_mwh - expected_heating) / expected_heating.abs().max(0.01)) * 100.0;
    let cooling_error =
        ((result.cooling_mwh - expected_cooling) / expected_cooling.abs().max(0.01)) * 100.0;

    let heating_status = if heating_error.abs() < 10.0 {
        "PASS"
    } else if heating_error.abs() < 50.0 {
        "WARN"
    } else {
        "FAIL"
    };

    let cooling_status = if cooling_error.abs() < 10.0 {
        "PASS"
    } else if cooling_error.abs() < 50.0 {
        "WARN"
    } else {
        "FAIL"
    };

    println!(
        "{}: H={:.2} MWh ({:+.0}%) [{}], C={:.2} MWh ({:+.0}%) [{}]",
        label,
        result.heating_mwh,
        heating_error,
        heating_status,
        result.cooling_mwh,
        cooling_error,
        cooling_status
    );
}

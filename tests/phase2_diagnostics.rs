//! Phase 2 Diagnostic Test for ASHRAE 140 High-Mass Cases
//!
//! This test runs detailed diagnostics to isolate the root cause of:
//! 1. Case 900 over-prediction (heating/cooling 2-3x too high)
//! 2. Case 960 under-heating (heating ~98% too low)
//!
//! Diagnostics include:
//! - Hourly solar gains (phi_st, phi_m)
//! - Zone temperatures vs setpoints
//! - HVAC runtime and energy
//! - CTF flux and surface temperatures
//! - Energy balance verification

use fluxion::physics::cta::VectorField;
use fluxion::physics::ctf_coefficients::CTFMaterial;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;
use std::fs::File;
use std::io::Write;

/// Diagnostic data collector for a single timestep
#[derive(Debug, Clone)]
struct DiagnosticRecord {
    hour: usize,
    outdoor_temp: f64,
    zone_temp: f64,
    mass_temp: f64,
    surface_temp: f64,
    sol_air_temp: f64,
    heating_setpoint: f64,
    cooling_setpoint: f64,
    hvac_power_w: f64,
    solar_gain_air_w: f64,
    solar_gain_mass_w: f64,
    ctf_flux_w_m2: f64,
    envelope_conductance: f64,
}

/// Run diagnostic simulation for a case
fn run_diagnostic_simulation(case_id: &str, use_ctf: bool) -> (Vec<DiagnosticRecord>, f64, f64) {
    println!("\n{:═^80}", "");
    println!("DIAGNOSTIC SIMULATION: Case {} (CTF: {})", case_id, use_ctf);
    println!("{:═^80}\n", "");

    // Load case specification
    let case = match case_id {
        "900" => ASHRAE140Case::Case900,
        "960" => ASHRAE140Case::Case960,
        _ => panic!("Unsupported case: {}", case_id),
    };

    let spec = case.spec();
    let weather = DenverTmyWeather::new();

    // Create thermal model
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Enable CTF if requested
    if use_ctf {
        let ctf_layers = vec![
            CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            CTFMaterial::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
            CTFMaterial::new("Insulation", 0.050, 0.04, 50.0, 840.0),
            CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
        ];
        model.enable_ctf(&ctf_layers, 3600.0, 50);
        println!("✅ CTF solver enabled");
    } else {
        println!("ℹ️  Using 5R1C solver (CTF disabled)");
    }

    // Print model parameters
    println!("\n📊 Model Parameters:");
    println!("  Heating setpoint: {:.1}°C", model.heating_setpoint);
    println!("  Cooling setpoint: {:.1}°C", model.cooling_setpoint);
    println!(
        "  Solar beam to mass fraction: {:.2}",
        model.solar_beam_to_mass_fraction
    );
    println!(
        "  Solar distribution to air: {:.2}",
        model.solar_distribution_to_air
    );
    println!(
        "  Thermal capacitance: {:.0} kJ/K",
        model.thermal_capacitance.as_ref()[0] / 1000.0
    );
    println!("  Wall U-value: {:.3} W/m²K", model.wall_u_value);
    println!("  Envelope area: {:.1} m²", model.zone_area.as_ref()[0]);

    // Run simulation and collect diagnostics
    let mut records = Vec::with_capacity(8760);
    let mut annual_heating_kwh = 0.0;
    let mut annual_cooling_kwh = 0.0;

    const STEPS: usize = 8760;

    for step in 0..STEPS {
        let hour_of_day = step % 24;
        let weather_data = weather.get_hourly_data(step).unwrap();

        // Update weather on model
        model.weather = Some(weather_data.clone());

        // Calculate solar gains manually for diagnostics
        let solar_beam = weather_data.dni;
        let solar_diffuse = weather_data.dhi;
        let floor_area = model.zone_area.as_ref()[0];

        // Get internal loads
        let loads = model.loads.as_ref();
        let phi_st = loads[0] * model.convective_fraction; // To air
        let phi_m = loads[0] * (1.0 - model.convective_fraction); // To mass

        // Get temperatures before step
        let t_zone_before = model.temperatures.as_ref()[0];
        let t_mass_before = model.mass_temperatures.as_ref()[0];

        // Estimate surface temperature (same as in step_physics)
        let h_tr_is = model.h_tr_is.as_ref()[0];
        let h_tr_ms = model.h_tr_ms.as_ref()[0];
        let t_surface = if (h_tr_is + h_tr_ms) > 0.001 {
            (h_tr_is * t_zone_before + h_tr_ms * t_mass_before + phi_st) / (h_tr_is + h_tr_ms)
        } else {
            t_zone_before
        };

        // Calculate sol-air temperature
        let alpha_solar = 0.7; // Typical exterior absorptance
        let h_ext = 25.0; // Exterior film coefficient
        let t_sol_air = weather_data.dry_bulb_temp + (alpha_solar * solar_beam) / h_ext + 3.0; // +3K for LW radiation

        // Run physics step
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        // Get temperatures after step
        let t_zone_after = model.temperatures.as_ref()[0];

        // Accumulate energy
        if hvac_kwh > 0.0 {
            annual_heating_kwh += hvac_kwh;
        } else {
            annual_cooling_kwh += -hvac_kwh;
        }

        // Estimate CTF flux (if enabled)
        let ctf_flux = if use_ctf && !model.ctf_solvers.is_empty() {
            model.ctf_solvers[0].interior_flux()
        } else {
            0.0
        };

        // Estimate HVAC power (from energy)
        let hvac_power_w = hvac_kwh * 1000.0; // kWh → Wh (approximate for timestep)

        // Get envelope conductance
        let h_tr_em = model.h_tr_em.as_ref()[0];

        // Record diagnostic data
        records.push(DiagnosticRecord {
            hour: step,
            outdoor_temp: weather_data.dry_bulb_temp,
            zone_temp: t_zone_after,
            mass_temp: model.mass_temperatures.as_ref()[0],
            surface_temp: t_surface,
            sol_air_temp: t_sol_air,
            heating_setpoint: model.heating_setpoint,
            cooling_setpoint: model.cooling_setpoint,
            hvac_power_w,
            solar_gain_air_w: phi_st,
            solar_gain_mass_w: phi_m,
            ctf_flux_w_m2: ctf_flux,
            envelope_conductance: h_tr_em,
        });

        // Print progress for first few days
        if step < 72 {
            println!(
                "Hour {:3}: T_out={:5.1}°C, T_zone={:5.1}°C, T_mass={:5.1}°C, HVAC={:7.1}W, Solar={:6.0}W",
                step,
                weather_data.dry_bulb_temp,
                t_zone_after,
                model.mass_temperatures.as_ref()[0],
                hvac_power_w,
                phi_st + phi_m
            );
        }
    }

    println!("\n📊 Annual Energy Results:");
    println!("  Heating: {:.2} MWh", annual_heating_kwh / 1000.0);
    println!("  Cooling: {:.2} MWh", annual_cooling_kwh / 1000.0);
    println!(
        "  Total:   {:.2} MWh",
        (annual_heating_kwh + annual_cooling_kwh) / 1000.0
    );

    (
        records,
        annual_heating_kwh / 1000.0,
        annual_cooling_kwh / 1000.0,
    )
}

/// Write diagnostic data to CSV
fn write_diagnostics_csv(records: &[DiagnosticRecord], filename: &str) {
    let mut file = File::create(filename).expect("Failed to create CSV file");

    // Write header
    writeln!(
        file,
        "Hour,Outdoor_Temp,Zone_Temp,Mass_Temp,Surface_Temp,SolAir_Temp,Heating_SP,Cooling_SP,\
         HVAC_Power_W,Solar_Gain_Air_W,Solar_Gain_Mass_W,CTF_Flux_Wm2,Envelope_Conductance"
    )
    .unwrap();

    // Write data
    for record in records {
        writeln!(
            file,
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            record.hour,
            record.outdoor_temp,
            record.zone_temp,
            record.mass_temp,
            record.surface_temp,
            record.sol_air_temp,
            record.heating_setpoint,
            record.cooling_setpoint,
            record.hvac_power_w,
            record.solar_gain_air_w,
            record.solar_gain_mass_w,
            record.ctf_flux_w_m2,
            record.envelope_conductance
        )
        .unwrap();
    }

    println!("✅ Diagnostics written to {}", filename);
}

/// Analyze diagnostic results
fn analyze_diagnostics(records: &[DiagnosticRecord], case_id: &str, use_ctf: bool) {
    println!("\n{:═^80}", "");
    println!("DIAGNOSTIC ANALYSIS: Case {} (CTF: {})", case_id, use_ctf);
    println!("{:═^80}\n", "");

    // Count HVAC runtime
    let mut heating_hours = 0;
    let mut cooling_hours = 0;
    let mut deadband_hours = 0;

    for record in records {
        if record.hvac_power_w > 10.0 {
            heating_hours += 1;
        } else if record.hvac_power_w < -10.0 {
            cooling_hours += 1;
        } else {
            deadband_hours += 1;
        }
    }

    println!("📊 HVAC Runtime Analysis:");
    println!(
        "  Heating hours: {} ({:.1}%)",
        heating_hours,
        heating_hours as f64 / 87.6
    );
    println!(
        "  Cooling hours: {} ({:.1}%)",
        cooling_hours,
        cooling_hours as f64 / 87.6
    );
    println!(
        "  Deadband hours: {} ({:.1}%)",
        deadband_hours,
        deadband_hours as f64 / 87.6
    );

    // Check for HVAC in deadband
    let mut hvac_in_deadband = 0;
    let mut hvac_with_large_error = 0;
    for record in records {
        let in_deadband = record.zone_temp >= record.heating_setpoint
            && record.zone_temp <= record.cooling_setpoint;
        let hvac_running = record.hvac_power_w.abs() > 100.0;

        if in_deadband && hvac_running {
            hvac_in_deadband += 1;
        }

        // Check for large temperature error (should trigger HVAC)
        let temp_error = if record.hvac_power_w > 0.0 {
            record.heating_setpoint - record.zone_temp
        } else if record.hvac_power_w < 0.0 {
            record.zone_temp - record.cooling_setpoint
        } else {
            0.0
        };

        if temp_error.abs() > 2.0 && !hvac_running {
            hvac_with_large_error += 1;
        }
    }

    if hvac_in_deadband > 0 {
        println!(
            "  ⚠️  HVAC running in deadband: {} hours (should be 0!)",
            hvac_in_deadband
        );
        println!("     This indicates a control logic bug - HVAC should be OFF when T_zone is within setpoints");
    } else {
        println!("  ✅ No HVAC operation in deadband");
    }

    if hvac_with_large_error > 0 {
        println!(
            "  ⚠️  HVAC OFF with large temp error (>2°C): {} hours",
            hvac_with_large_error
        );
    }

    // Temperature statistics
    let avg_zone_temp: f64 =
        records.iter().map(|r| r.zone_temp).sum::<f64>() / records.len() as f64;
    let min_zone_temp = records
        .iter()
        .map(|r| r.zone_temp)
        .fold(f64::INFINITY, f64::min);
    let max_zone_temp = records
        .iter()
        .map(|r| r.zone_temp)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n📊 Zone Temperature Statistics:");
    println!("  Average: {:.2}°C", avg_zone_temp);
    println!("  Minimum: {:.2}°C", min_zone_temp);
    println!("  Maximum: {:.2}°C", max_zone_temp);
    println!(
        "  Expected range: {:.1}°C - {:.1}°C (setpoints)",
        records[0].heating_setpoint, records[0].cooling_setpoint
    );

    // Solar gain analysis
    let total_solar_gain_kwh: f64 = records
        .iter()
        .map(|r| r.solar_gain_air_w + r.solar_gain_mass_w)
        .sum::<f64>()
        / 1000.0;
    let avg_solar_gain_w: f64 = records
        .iter()
        .map(|r| r.solar_gain_air_w + r.solar_gain_mass_w)
        .sum::<f64>()
        / records.len() as f64;

    println!("\n📊 Solar Gain Analysis:");
    println!("  Total annual solar gain: {:.1} kWh", total_solar_gain_kwh);
    println!("  Average solar gain: {:.1} W", avg_solar_gain_w);

    // Check solar distribution
    let solar_to_air: f64 = records.iter().map(|r| r.solar_gain_air_w).sum::<f64>();
    let solar_to_mass: f64 = records.iter().map(|r| r.solar_gain_mass_w).sum::<f64>();
    let air_fraction = solar_to_air / (solar_to_air + solar_to_mass);

    println!(
        "  Solar distribution: {:.1}% to air, {:.1}% to mass",
        air_fraction * 100.0,
        (1.0 - air_fraction) * 100.0
    );
    println!("  Expected: ~30% to air, ~70% to mass (for Case 900)");

    // Surface temperature analysis
    let avg_surface_temp: f64 =
        records.iter().map(|r| r.surface_temp).sum::<f64>() / records.len() as f64;
    let avg_sol_air_temp: f64 =
        records.iter().map(|r| r.sol_air_temp).sum::<f64>() / records.len() as f64;
    let avg_outdoor_temp: f64 =
        records.iter().map(|r| r.outdoor_temp).sum::<f64>() / records.len() as f64;

    println!("\n📊 Surface Temperature Analysis:");
    println!("  Average surface temp: {:.2}°C", avg_surface_temp);
    println!("  Average sol-air temp: {:.2}°C", avg_sol_air_temp);
    println!("  Average outdoor temp: {:.2}°C", avg_outdoor_temp);
    println!("  Average zone temp: {:.2}°C", avg_zone_temp);

    // Check if surface temp is physically reasonable
    if avg_surface_temp < avg_outdoor_temp - 5.0 || avg_surface_temp > avg_sol_air_temp + 5.0 {
        println!(
            "  ⚠️  Surface temperature may be incorrect (should be between T_out and T_sol_air)"
        );
    } else {
        println!("  ✅ Surface temperature appears physically reasonable");
    }
}

#[test]
fn test_case_900_diagnostics() {
    println!("\n{:█^80}", "");
    println!("PHASE 2 DIAGNOSTIC: ASHRAE 140 Case 900");
    println!("{:█^80}\n", "");

    // Run with CTF enabled
    let (records_ctf, heating_ctf, cooling_ctf) = run_diagnostic_simulation("900", true);
    write_diagnostics_csv(&records_ctf, "diagnostics_case_900_ctf.csv");
    analyze_diagnostics(&records_ctf, "900", true);

    // Run with CTF disabled (5R1C)
    let (records_5r1c, heating_5r1c, cooling_5r1c) = run_diagnostic_simulation("900", false);
    write_diagnostics_csv(&records_5r1c, "diagnostics_case_900_5r1c.csv");
    analyze_diagnostics(&records_5r1c, "900", false);

    // Compare results
    println!("\n{:═^80}", "");
    println!("COMPARISON: CTF vs 5R1C");
    println!("{:═^80}\n", "");

    println!("Annual Heating Energy:");
    println!("  CTF:   {:.2} MWh", heating_ctf);
    println!("  5R1C:  {:.2} MWh", heating_5r1c);
    println!(
        "  Diff:  {:.2} MWh ({:+.1}%)",
        heating_ctf - heating_5r1c,
        (heating_ctf - heating_5r1c) / heating_5r1c * 100.0
    );

    println!("\nAnnual Cooling Energy:");
    println!("  CTF:   {:.2} MWh", cooling_ctf);
    println!("  5R1C:  {:.2} MWh", cooling_5r1c);
    println!(
        "  Diff:  {:.2} MWh ({:+.1}%)",
        cooling_ctf - cooling_5r1c,
        (cooling_ctf - cooling_5r1c) / cooling_5r1c * 100.0
    );

    // Reference values
    println!("\n📊 ASHRAE 140 Reference Values:");
    println!("  Heating: 5.50 - 7.50 MWh");
    println!("  Cooling: 8.00 - 10.50 MWh");

    println!("\n📊 Error vs Reference:");
    let ref_heating_mid = 6.5;
    let ref_cooling_mid = 9.25;
    println!(
        "  CTF Heating Error:  {:+.1}%",
        (heating_ctf - ref_heating_mid) / ref_heating_mid * 100.0
    );
    println!(
        "  CTF Cooling Error:  {:+.1}%",
        (cooling_ctf - ref_cooling_mid) / ref_cooling_mid * 100.0
    );
    println!(
        "  5R1C Heating Error: {:+.1}%",
        (heating_5r1c - ref_heating_mid) / ref_heating_mid * 100.0
    );
    println!(
        "  5R1C Cooling Error: {:+.1}%",
        (cooling_5r1c - ref_cooling_mid) / ref_cooling_mid * 100.0
    );

    // Determine if CTF is making things better or worse
    let ctf_heating_error = (heating_ctf - ref_heating_mid).abs();
    let r1c_heating_error = (heating_5r1c - ref_heating_mid).abs();

    if ctf_heating_error < r1c_heating_error {
        println!("\n✅ CTF improves heating prediction accuracy");
    } else {
        println!("\n❌ CTF makes heating prediction worse");
    }
}

#[test]
fn test_case_960_diagnostics() {
    println!("\n{:█^80}", "");
    println!("PHASE 2 DIAGNOSTIC: ASHRAE 140 Case 960 (Sunspace)");
    println!("{:█^80}\n", "");

    // Run with CTF enabled
    let (records_ctf, heating_ctf, cooling_ctf) = run_diagnostic_simulation("960", true);
    write_diagnostics_csv(&records_ctf, "diagnostics_case_960_ctf.csv");
    analyze_diagnostics(&records_ctf, "960", true);

    // Reference values for Case 960
    println!("\n📊 ASHRAE 140 Reference Values:");
    println!("  Heating: 5.00 - 15.00 MWh");
    println!("  Cooling: 1.55 - 2.78 MWh");

    println!("\n📊 Error vs Reference:");
    let ref_heating_mid = 10.0;
    let ref_cooling_mid = 2.16;
    println!(
        "  CTF Heating Error:  {:+.1}%",
        (heating_ctf - ref_heating_mid) / ref_heating_mid * 100.0
    );
    println!(
        "  CTF Cooling Error:  {:+.1}%",
        (cooling_ctf - ref_cooling_mid) / ref_cooling_mid * 100.0
    );
}

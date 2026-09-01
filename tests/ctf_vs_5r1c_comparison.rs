//! CTF vs 5R1C comparison test for ASHRAE 140 cases
//!
//! This test compares cooling energy results between CTF and 5R1C solvers
//! to isolate the effect of CTF on E/W window cases (920, 930).

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Run a single case with specified CTF setting using the validator framework
/// Returns: (annual_heating_MWh, annual_cooling_MWh, peak_heating_kW, peak_cooling_kW)
fn run_case_with_ctf_setting(case: ASHRAE140Case, enable_ctf: bool) -> (f64, f64, f64, f64) {
    let spec = case.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Enable/disable CTF based on parameter
    if enable_ctf {
        // Try to enable CTF (may fall back to FD) - use the same approach as validator
        use fluxion::physics::fd_discretization::MaterialLayer as FDMaterialLayer;

        // Get wall construction layers from spec (same as validator does)
        let fd_layers: Vec<FDMaterialLayer> = spec
            .construction
            .wall
            .layers
            .iter()
            .map(|layer| {
                FDMaterialLayer::new(
                    &layer.name,
                    layer.thickness,
                    layer.conductivity,
                    layer.density,
                    layer.specific_heat,
                )
            })
            .collect();

        model.enable_ctf_with_fd_fallback(&fd_layers, 3600.0, 50, 5);
        println!("  Solver: CTF (with FD fallback)");
    } else {
        // Ensure CTF is disabled (5R1C only)
        model.disable_ctf();
        println!("  Solver: 5R1C");
    }

    // Configure HVAC setpoints (same as validator does)
    if let Some(hvac) = spec.hvac.first() {
        model.setpoints.heating_setpoint = hvac.heating_setpoint;
        model.setpoints.cooling_setpoint = hvac.cooling_setpoint;
    } else {
        // Default setpoints for ASHRAE 140
        model.setpoints.heating_setpoint = 20.0;
        model.setpoints.cooling_setpoint = 27.0;
    }

    // Set HVAC enabled
    let num_zones = model.hvac.num_zones;
    let hvac_enabled_vals = vec![1.0; num_zones];
    model.hvac.hvac_enabled = VectorField::new(hvac_enabled_vals);

    const STEPS: usize = 8760;
    let mut annual_heating_joules: f64 = 0.0;
    let mut annual_cooling_joules: f64 = 0.0;
    let mut peak_heating_w: f64 = 0.0;
    let mut peak_cooling_w: f64 = 0.0;

    for step in 0..STEPS {
        let weather_data = weather.get_hourly_data(step).unwrap();

        // Update weather on model for solar gain calculation
        model.solar.weather = Some(weather_data.clone());

        // Apply setpoints from HVAC schedule
        if let Some(hvac) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            if let Some(heating_sp) = hvac.heating_setpoint_at_hour(hour) {
                model.setpoints.heating_setpoint = heating_sp;
            }
            if let Some(cooling_sp) = hvac.cooling_setpoint_at_hour(hour) {
                model.setpoints.cooling_setpoint = cooling_sp;
            }
        }

        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_joules = energy_kwh * 3.6e6;

        if energy_joules > 0.0 {
            annual_heating_joules += energy_joules;
            peak_heating_w = peak_heating_w.max(energy_joules / 3600.0);
        } else if energy_joules < 0.0 {
            annual_cooling_joules += -energy_joules;
            peak_cooling_w = peak_cooling_w.max(-energy_joules / 3600.0);
        }
    }

    let annual_heating_mwh = annual_heating_joules / 3.6e9;
    let annual_cooling_mwh = annual_cooling_joules / 3.6e9;
    let peak_heating_kw = peak_heating_w / 1000.0;
    let peak_cooling_kw = peak_cooling_w / 1000.0;

    (
        annual_heating_mwh,
        annual_cooling_mwh,
        peak_heating_kw,
        peak_cooling_kw,
    )
}

#[test]
fn test_ctf_vs_5r1c_case_920() {
    println!("\n{}", "=".repeat(70));
    println!("CTF vs 5R1C Comparison Test - Case 920 (East + West Windows)");
    println!("{}\n", "=".repeat(70));

    let case = ASHRAE140Case::Case920;
    let spec = case.spec();

    println!("Case 920 Spec:");
    println!("  Case ID: {}", spec.case_id);
    println!("  Num zones: {}", spec.num_zones);
    println!("  Reference cooling: 6.50-8.50 MWh\n");

    // Run with 5R1C
    println!("Running Case 920 with 5R1C solver:");
    let (heat_5r1c, cool_5r1c, peak_h_5r1c, peak_c_5r1c) = run_case_with_ctf_setting(case, false);
    println!(
        "  Heating: {:.2} MWh, Cooling: {:.2} MWh",
        heat_5r1c, cool_5r1c
    );
    println!(
        "  Peak H: {:.2} kW, Peak C: {:.2} kW\n",
        peak_h_5r1c, peak_c_5r1c
    );

    // Run with CTF
    println!("Running Case 920 with CTF solver:");
    let (heat_ctf, cool_ctf, peak_h_ctf, peak_c_ctf) = run_case_with_ctf_setting(case, true);
    println!(
        "  Heating: {:.2} MWh, Cooling: {:.2} MWh",
        heat_ctf, cool_ctf
    );
    println!(
        "  Peak H: {:.2} kW, Peak C: {:.2} kW\n",
        peak_h_ctf, peak_c_ctf
    );

    // Calculate differences
    let cool_diff = cool_ctf - cool_5r1c;
    let cool_diff_pct = if cool_5r1c > 0.0 {
        (cool_diff / cool_5r1c) * 100.0
    } else {
        0.0
    };

    let heat_diff = heat_ctf - heat_5r1c;
    let heat_diff_pct = if heat_5r1c > 0.0 {
        (heat_diff / heat_5r1c) * 100.0
    } else {
        0.0
    };

    println!("{}", "=".repeat(70));
    println!("Comparison Results (CTF - 5R1C):");
    println!(
        "  Cooling energy difference: {:.3} MWh ({:+.1}%)",
        cool_diff, cool_diff_pct
    );
    println!(
        "  Heating energy difference: {:.3} MWh ({:+.1}%)",
        heat_diff, heat_diff_pct
    );
    println!("{}", "=".repeat(70));

    // Analysis
    println!("\nAnalysis:");
    if cool_diff.abs() > 0.5 {
        println!("  ⚠️  LARGE CTF EFFECT: CTF changes cooling by >0.5 MWh");
        if cool_diff < 0.0 {
            println!("  ⚠️  CTF REDUCES COOLING: This suggests CTF over-damping!");
            println!("  Hypothesis: CTF smooths sharp E/W solar peaks, reducing cooling load");
        } else {
            println!("  ✓ CTF INCREASES COOLING: CTF captures more solar gain");
        }
    } else {
        println!("  ✓ Small CTF effect (<0.5 MWh difference)");
        println!("  CTF is NOT the dominant cause of E/W cooling error");
    }

    // Reference check
    let ref_cool_min = 6.50;
    let ref_cool_max = 8.50;
    println!("\nReference Comparison:");
    println!(
        "  5R1C cooling: {:.2} MWh (Ref: {:.2}-{:.2}, Error: {:.1}%)",
        cool_5r1c,
        ref_cool_min,
        ref_cool_max,
        (cool_5r1c - ref_cool_min) / ref_cool_min * 100.0
    );
    println!(
        "  CTF cooling:  {:.2} MWh (Ref: {:.2}-{:.2}, Error: {:.1}%)",
        cool_ctf,
        ref_cool_min,
        ref_cool_max,
        (cool_ctf - ref_cool_min) / ref_cool_min * 100.0
    );

    // Assert that we got results
    assert!(cool_5r1c > 0.0, "5R1C cooling should be positive");
    assert!(cool_ctf > 0.0, "CTF cooling should be positive");
}

#[test]
fn test_ctf_vs_5r1c_case_930() {
    println!("\n{}", "=".repeat(70));
    println!("CTF vs 5R1C Comparison Test - Case 930 (East + West + South Windows)");
    println!("{}\n", "=".repeat(70));

    let case = ASHRAE140Case::Case930;

    println!("Case 930 Spec:");
    println!("  Reference cooling: 4.50-6.50 MWh\n");

    // Run with 5R1C
    println!("Running Case 930 with 5R1C solver:");
    let (heat_5r1c, cool_5r1c, peak_h_5r1c, peak_c_5r1c) = run_case_with_ctf_setting(case, false);
    println!(
        "  Heating: {:.2} MWh, Cooling: {:.2} MWh",
        heat_5r1c, cool_5r1c
    );
    println!(
        "  Peak H: {:.2} kW, Peak C: {:.2} kW\n",
        peak_h_5r1c, peak_c_5r1c
    );

    // Run with CTF
    println!("Running Case 930 with CTF solver:");
    let (heat_ctf, cool_ctf, peak_h_ctf, peak_c_ctf) = run_case_with_ctf_setting(case, true);
    println!(
        "  Heating: {:.2} MWh, Cooling: {:.2} MWh",
        heat_ctf, cool_ctf
    );
    println!(
        "  Peak H: {:.2} kW, Peak C: {:.2} kW\n",
        peak_h_ctf, peak_c_ctf
    );

    // Calculate differences
    let cool_diff = cool_ctf - cool_5r1c;
    let cool_diff_pct = if cool_5r1c > 0.0 {
        (cool_diff / cool_5r1c) * 100.0
    } else {
        0.0
    };

    println!("{}", "=".repeat(70));
    println!("Comparison Results (CTF - 5R1C):");
    println!(
        "  Cooling energy difference: {:.3} MWh ({:+.1}%)",
        cool_diff, cool_diff_pct
    );
    println!("{}", "=".repeat(70));

    // Analysis
    println!("\nAnalysis:");
    if cool_diff.abs() > 0.5 {
        println!("  ⚠️  LARGE CTF EFFECT: CTF changes cooling by >0.5 MWh");
        if cool_diff < 0.0 {
            println!("  ⚠️  CTF REDUCES COOLING: This suggests CTF over-damping!");
        }
    } else {
        println!("  ✓ Small CTF effect (<0.5 MWh difference)");
        println!("  CTF is NOT the dominant cause of E/W cooling error");
    }
}

#[test]
fn test_ctf_vs_5r1c_case_900() {
    println!("\n{}", "=".repeat(70));
    println!("CTF vs 5R1C Comparison Test - Case 900 (South Windows - Control Case)");
    println!("{}\n", "=".repeat(70));

    let case = ASHRAE140Case::Case900;

    println!("Case 900 Spec (South-facing control case):");
    println!("  Reference cooling: 8.00-10.50 MWh\n");

    // Run with 5R1C
    println!("Running Case 900 with 5R1C solver:");
    let (heat_5r1c, cool_5r1c, peak_h_5r1c, peak_c_5r1c) = run_case_with_ctf_setting(case, false);
    println!(
        "  Heating: {:.2} MWh, Cooling: {:.2} MWh",
        heat_5r1c, cool_5r1c
    );
    println!(
        "  Peak H: {:.2} kW, Peak C: {:.2} kW\n",
        peak_h_5r1c, peak_c_5r1c
    );

    // Run with CTF
    println!("Running Case 900 with CTF solver:");
    let (heat_ctf, cool_ctf, peak_h_ctf, peak_c_ctf) = run_case_with_ctf_setting(case, true);
    println!(
        "  Heating: {:.2} MWh, Cooling: {:.2} MWh",
        heat_ctf, cool_ctf
    );
    println!(
        "  Peak H: {:.2} kW, Peak C: {:.2} kW\n",
        peak_h_ctf, peak_c_ctf
    );

    // Calculate differences
    let cool_diff = cool_ctf - cool_5r1c;
    let cool_diff_pct = if cool_5r1c > 0.0 {
        (cool_diff / cool_5r1c) * 100.0
    } else {
        0.0
    };

    println!("{}", "=".repeat(70));
    println!("Comparison Results (CTF - 5R1C):");
    println!(
        "  Cooling energy difference: {:.3} MWh ({:+.1}%)",
        cool_diff, cool_diff_pct
    );
    println!("{}", "=".repeat(70));

    println!("\nNote: Case 900 (South-facing) should show smaller CTF effect than E/W cases");
    println!("if CTF over-damping is the root cause of E/W errors.");
}

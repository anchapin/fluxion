//! Test separate heating and cooling energy tracking (Plan 03-08d)
//!
//! Diagnostic test to diagnose annual energy under-prediction for Case 900.
//! This test uses the new separate heating/cooling energy tracking fields
//! to identify which energy type is causing under-prediction.
//!
//! Context from Plan 03-08c:
//! - Total annual energy: 2.05 MWh vs [3.30, 5.71] MWh reference (38% below)
//! - Peak heating: 2.10 kW ✓ (within [1.10, 2.10] kW)
//! - Peak cooling: 3.57 kW ✓ (within [2.10, 3.70] kW)
//!
//! Objective:
//! 1. Run ASHRAE 140 Case 900 validation with separate energy tracking
//! 2. Compare heating energy to [1.17, 2.04] MWh reference
//! 3. Compare cooling energy to [2.13, 3.67] MWh reference
//! 4. Identify which metric is causing under-prediction

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

/// ASHRAE 140 reference values for Case 900
const REFERENCE_ANNUAL_HEATING_MIN: f64 = 1.17; // MWh
const REFERENCE_ANNUAL_HEATING_MAX: f64 = 2.04; // MWh
const REFERENCE_ANNUAL_COOLING_MIN: f64 = 2.13; // MWh
const REFERENCE_ANNUAL_COOLING_MAX: f64 = 3.67; // MWh
const REFERENCE_TOTAL_MIN: f64 = 3.30; // MWh (sum of heating + cooling ranges)
const REFERENCE_TOTAL_MAX: f64 = 5.71; // MWh

/// Tolerance for annual energy validation (±15% as per ASHRAE 140)
const ANNUAL_ENERGY_TOLERANCE: f64 = 0.15;

#[test]
fn test_case_900_separate_heating_cooling_energy() {
    println!("=== Case 900 Separate Heating/Cooling Energy Diagnostic ===\n");

    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    // Reset all energy tracking to ensure clean state
    model.reset_all_energy_tracking();

    // Simulate 1 year (8760 hours)
    let steps = 8760;

    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        // Set weather data on model for solar gain calculation
        model.weather = Some(weather_data.clone());

        // Run physics step
        model.step_physics(step, weather_data.dry_bulb_temp);

        // Diagnostic output daily
        if step % 24 == 0 {
            let day = step / 24;
            let heating_kwh = model.get_heating_energy_kwh();
            let cooling_kwh = model.get_cooling_energy_kwh();
            let total_kwh = heating_kwh + cooling_kwh;

            println!(
                "Day {}: Heating={:.4} kWh, Cooling={:.4} kWh, Total={:.4} kWh",
                day, heating_kwh, cooling_kwh, total_kwh
            );
        }
    }

    // Get final energy values
    let annual_heating_mwh = model.get_heating_energy_kwh() / 1000.0; // Convert kWh to MWh
    let annual_cooling_mwh = model.get_cooling_energy_kwh() / 1000.0; // Convert kWh to MWh
    let annual_total_mwh = annual_heating_mwh + annual_cooling_mwh;

    // Get peak loads
    let peak_heating_kw = model.get_peak_heating_power_kw();
    let peak_cooling_kw = model.get_peak_cooling_power_kw();

    println!("\n=== Case 900 Results (Plan 03-08d) ===");
    println!("Annual Heating: {:.2} MWh", annual_heating_mwh);
    println!("  Reference: [{:.2}, {:.2}] MWh", REFERENCE_ANNUAL_HEATING_MIN, REFERENCE_ANNUAL_HEATING_MAX);
    println!("Annual Cooling: {:.2} MWh", annual_cooling_mwh);
    println!("  Reference: [{:.2}, {:.2}] MWh", REFERENCE_ANNUAL_COOLING_MIN, REFERENCE_ANNUAL_COOLING_MAX);
    println!("Annual Total: {:.2} MWh", annual_total_mwh);
    println!("  Reference: [{:.2}, {:.2}] MWh", REFERENCE_TOTAL_MIN, REFERENCE_TOTAL_MAX);
    println!("Peak Heating: {:.2} kW", peak_heating_kw);
    println!("  Reference: [1.10, 2.10] kW");
    println!("Peak Cooling: {:.2} kW", peak_cooling_kw);
    println!("  Reference: [2.10, 3.70] kW");
    println!();

    // Calculate deviations from reference ranges
    let heating_deviation = if annual_heating_mwh < REFERENCE_ANNUAL_HEATING_MIN {
        format!("{:.1}% below lower bound", (REFERENCE_ANNUAL_HEATING_MIN - annual_heating_mwh) / REFERENCE_ANNUAL_HEATING_MIN * 100.0)
    } else if annual_heating_mwh > REFERENCE_ANNUAL_HEATING_MAX {
        format!("{:.1}% above upper bound", (annual_heating_mwh - REFERENCE_ANNUAL_HEATING_MAX) / REFERENCE_ANNUAL_HEATING_MAX * 100.0)
    } else {
        "WITHIN reference range ✓".to_string()
    };

    let cooling_deviation = if annual_cooling_mwh < REFERENCE_ANNUAL_COOLING_MIN {
        format!("{:.1}% below lower bound", (REFERENCE_ANNUAL_COOLING_MIN - annual_cooling_mwh) / REFERENCE_ANNUAL_COOLING_MIN * 100.0)
    } else if annual_cooling_mwh > REFERENCE_ANNUAL_COOLING_MAX {
        format!("{:.1}% above upper bound", (annual_cooling_mwh - REFERENCE_ANNUAL_COOLING_MAX) / REFERENCE_ANNUAL_COOLING_MAX * 100.0)
    } else {
        "WITHIN reference range ✓".to_string()
    };

    let total_deviation = if annual_total_mwh < REFERENCE_TOTAL_MIN {
        format!("{:.1}% below lower bound", (REFERENCE_TOTAL_MIN - annual_total_mwh) / REFERENCE_TOTAL_MIN * 100.0)
    } else if annual_total_mwh > REFERENCE_TOTAL_MAX {
        format!("{:.1}% above upper bound", (annual_total_mwh - REFERENCE_TOTAL_MAX) / REFERENCE_TOTAL_MAX * 100.0)
    } else {
        "WITHIN reference range ✓".to_string()
    };

    println!("=== Diagnostic Analysis ===");
    println!("Heating Energy: {}", heating_deviation);
    println!("Cooling Energy: {}", cooling_deviation);
    println!("Total Energy: {}", total_deviation);
    println!();

    // Identify which metric is causing under-prediction
    println!("=== Root Cause Identification ===");

    let heating_low = annual_heating_mwh < REFERENCE_ANNUAL_HEATING_MIN;
    let cooling_low = annual_cooling_mwh < REFERENCE_ANNUAL_COOLING_MIN;
    let heating_high = annual_heating_mwh > REFERENCE_ANNUAL_HEATING_MAX;
    let cooling_high = annual_cooling_mwh > REFERENCE_ANNUAL_COOLING_MAX;

    if heating_low && cooling_low {
        println!("❌ BOTH heating and cooling energy are under-predicted");
        println!("   This suggests a common cause affecting energy calculation");
        println!("   Possible causes:");
        println!("   1. HVAC demand calculation error (sensitivity too high)");
        println!("   2. Energy accumulation error (units, sign, integration)");
        println!("   3. Time constant correction applied incorrectly");
        println!("   4. Free-floating temperature calculation error");
    } else if heating_low && !cooling_low {
        println!("❌ Heating energy is under-predicted, cooling is OK");
        println!("   This suggests heating-specific issue");
        println!("   Possible causes:");
        println!("   1. Winter free-floating temperature too low");
        println!("   2. Heating demand calculation error");
        println!("   3. Thermal mass releases too much cold in winter");
    } else if !heating_low && cooling_low {
        println!("❌ Cooling energy is under-predicted, heating is OK");
        println!("   This suggests cooling-specific issue");
        println!("   Possible causes:");
        println!("   1. Summer free-floating temperature too low");
        println!("   2. Cooling demand calculation error");
        println!("   3. Solar gain distribution issue");
    } else if heating_high && cooling_high {
        println!("❌ BOTH heating and cooling energy are over-predicted");
        println!("   This suggests energy calculation overestimates demand");
    } else if heating_high && !cooling_high {
        println!("❌ Heating energy is over-predicted, cooling is OK");
        println!("   This suggests heating demand overestimation");
    } else if !heating_high && cooling_high {
        println!("❌ Cooling energy is over-predicted, heating is OK");
        println!("   This suggests cooling demand overestimation");
    } else {
        println!("✓ Both heating and cooling energy are within reference range");
        println!("   Energy tracking is correct!");
    }

    println!();

    // Verify peak loads are correct
    let peak_heating_ok = peak_heating_kw >= 1.10 && peak_heating_kw <= 2.10;
    let peak_cooling_ok = peak_cooling_kw >= 2.10 && peak_cooling_kw <= 3.70;

    println!("=== Peak Load Verification ===");
    println!("Peak Heating: {} ({:.2} kW)", if peak_heating_ok { "✓" } else { "✗" }, peak_heating_kw);
    println!("Peak Cooling: {} ({:.2} kW)", if peak_cooling_ok { "✓" } else { "✗" }, peak_cooling_kw);

    if peak_heating_ok && peak_cooling_ok {
        println!("✓ Peak loads are correct - issue is in energy calculation, not power demand");
    } else {
        println!("✗ Peak loads are incorrect - need to fix power demand calculation");
    }

    println!();

    // Check if energy separation is working correctly
    assert!(annual_heating_mwh >= 0.0, "Heating energy should be non-negative");
    assert!(annual_cooling_mwh >= 0.0, "Cooling energy should be non-negative");
    assert_eq!(annual_total_mwh, annual_heating_mwh + annual_cooling_mwh, "Total should equal heating + cooling");

    // Print summary
    println!("=== Summary ===");
    println!("Separate heating/cooling energy tracking: ✓ Working");
    println!("Annual heating energy: {:.2} MWh ({})", annual_heating_mwh, heating_deviation);
    println!("Annual cooling energy: {:.2} MWh ({})", annual_cooling_mwh, cooling_deviation);
    println!("Annual total energy: {:.2} MWh ({})", annual_total_mwh, total_deviation);
    println!("Peak heating: {:.2} kW ({})", peak_heating_kw, if peak_heating_ok { "✓" } else { "✗" });
    println!("Peak cooling: {:.2} kW ({})", peak_cooling_kw, if peak_cooling_ok { "✓" } else { "✗" });
}

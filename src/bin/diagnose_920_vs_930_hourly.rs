//! Hourly comparison of Case 920 vs 930 to identify cooling discrepancy
//!
//! This diagnostic compares free-floating temperatures and cooling demand
//! throughout the day to identify when the 3.4x discrepancy occurs.

use fluxion::sim::engine::ThermalModel;
use fluxion::physics::cta::VectorField;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Hourly Comparison: Case 920 vs 930 ===");
    println!("Investigating 3.4x discrepancy between shading effects");
    println!();

    // Get case specifications
    let spec_920 = ASHRAE140Case::Case920.spec();
    let spec_930 = ASHRAE140Case::Case930.spec();

    println!("Case 920: {} ({})", spec_920.case_id, spec_920.description);
    println!("Case 930: {} ({})", spec_930.case_id, spec_930.description);
    println!();

    // Create models
    let mut model_920 = ThermalModel::<VectorField>::from_spec(&spec_920);
    let mut model_930 = ThermalModel::<VectorField>::from_spec(&spec_930);

    // Test key hours on a hot summer day
    let outdoor_temp = 35.0; // Hot day
    let hours = [8, 10, 12, 14, 16, 18];

    println!("Outdoor temp: {}°C (constant for testing)", outdoor_temp);
    println!();
    println!("Hour | Ti_920 (°C) | Ti_930 (°C) | Diff (°C) | Cooling_920 (W) | Cooling_930 (W) | Ratio");
    println!("-----|-------------|-------------|-----------|----------------|----------------|-------");

    let cooling_setpoint = 27.0;
    let mut total_cooling_920 = 0.0;
    let mut total_cooling_930 = 0.0;

    for hour in hours {
        let ti_920 = model_920.calculate_free_float_temperature(hour, outdoor_temp);
        let ti_930 = model_930.calculate_free_float_temperature(hour, outdoor_temp);

        let diff = ti_930 - ti_920;

        // Calculate cooling demand
        let cooling_920 = if ti_920 > cooling_setpoint {
            (ti_920 - cooling_setpoint) * 100.0 // Simplified sensitivity
        } else {
            0.0
        };

        let cooling_930 = if ti_930 > cooling_setpoint {
            (ti_930 - cooling_setpoint) * 100.0
        } else {
            0.0
        };

        total_cooling_920 += cooling_920;
        total_cooling_930 += cooling_930;

        let ratio = if cooling_920 > 0.0 {
            cooling_930 / cooling_920
        } else {
            0.0
        };

        println!("{:5} | {:11.2} | {:11.2} | {:9.2} | {:14.0} | {:14.0} | {:.2}",
            hour, ti_920, ti_930, diff, cooling_920, cooling_930, ratio);
    }

    println!();
    println!("=== Summary ===");
    println!("Total cooling 920: {:.0} Wh", total_cooling_920);
    println!("Total cooling 930: {:.0} Wh", total_cooling_930);
    println!("Ratio (930/920): {:.2}", total_cooling_930 / total_cooling_920);
    println!();

    println!("=== Validation Results ===");
    println!("Case 920 cooling: 1.29 MWh (Ref: 1.84-3.31)");
    println!("Case 930 cooling: 0.49 MWh (Ref: 1.04-2.24)");
    println!("Actual ratio: {:.2}", 0.49 / 1.29);
    println!();

    println!("=== Solar Gain Analysis (from earlier diagnostic) ===");
    println!("Solar gain reduction from shading: 17.6%");
    println!("Cooling load reduction from shading: 62%");
    println!("Discrepancy: {:.1}x", 62.0 / 17.6);
    println!();

    println!("=== Key Question ===");
    println!("Why does shading reduce cooling by 62% when solar gains are only reduced by 17.6%?");
    println!();
    println!("Possible causes:");
    println!("1. Free-floating temp calculation is incorrect for shaded windows");
    println!("2. Shading affects view factors (solar distribution to air vs mass)");
    println!("3. Thermal mass coupling is different for shaded windows");
    println!("4. Mode-specific coupling factors are incorrect for E/W shaded windows");
}

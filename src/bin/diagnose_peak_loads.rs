// Diagnostic tool for peak load analysis (Session 47)
//
// Analyzes:
// 1. Peak heating and cooling loads
// 2. How they compare to reference ranges
// 3. Timestep resolution effects
// 4. Potential causes of discrepancies

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case_id = if args.len() > 1 {
        args[1].as_str()
    } else {
        "900" // Default to Case 900
    };

    println!("=== Peak Load Diagnostic Analysis for Case {} ===\n", case_id);

    // Create model for the specified case
    let case = match case_id {
        "600" => ASHRAE140Case::Case600,
        "610" => ASHRAE140Case::Case610,
        "620" => ASHRAE140Case::Case620,
        "630" => ASHRAE140Case::Case630,
        "640" => ASHRAE140Case::Case640,
        "650" => ASHRAE140Case::Case650,
        "900" => ASHRAE140Case::Case900,
        "910" => ASHRAE140Case::Case910,
        "920" => ASHRAE140Case::Case920,
        "930" => ASHRAE140Case::Case930,
        "940" => ASHRAE140Case::Case940,
        "950" => ASHRAE140Case::Case950,
        _ => panic!("Invalid case ID: {}", case_id),
    };

    let spec = case.spec();

    println!("Case Specifications:");
    println!("  Description: {}", spec.description);
    println!("  Construction: {:?}", spec.construction_type);
    let floor_area: f64 = spec.geometry.iter().map(|g| g.width * g.depth).sum();
    println!("  Floor Area: {} m²", floor_area);
    println!();

    // Create model and simulate
    let mut model = ThermalModel::from_spec(&spec);

    println!("Simulating annual performance...\n");
    let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");
    let _energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    // Get the tracked peak values from the model
    let peak_heating_kw = model.get_peak_heating_power_kw();
    let peak_cooling_kw = model.get_peak_cooling_power_kw();

    println!("=== Peak Load Results ===\n");
    println!("Peak Heating: {:.2} kW", peak_heating_kw);
    println!("Peak Cooling: {:.2} kW", peak_cooling_kw);
    println!();

    // Print reference ranges for comparison
    println!("=== Reference Ranges (for comparison) ===\n");
    println!("Check validation output for reference ranges.");
    println!("Run: cargo run --release --bin fluxion validate --case {}", case_id);
    println!();

    // Analysis
    println!("=== Analysis ===\n");

    println!("Timestep Resolution:");
    println!("  Current: 1 hour (3600 seconds)");
    println!("  Peak loads are calculated as maximum of hourly demands");
    println!("  If true peak occurs within an hour, it may be averaged");
    println!();

    println!("Peak Heating:");
    if peak_heating_kw > 0.0 {
        println!("  Value: {:.2} kW", peak_heating_kw);
        println!("  Typically occurs during coldest weather");
        println!("  May be affected by:");
        println!("    - Thermal mass dampening (high-mass cases)");
        println!("    - Hourly timestep averaging");
        println!("    - Solar gains reducing heating demand");
    } else {
        println!("  No heating demand (case may not require heating)");
    }
    println!();

    println!("Peak Cooling:");
    if peak_cooling_kw > 0.0 {
        println!("  Value: {:.2} kW", peak_cooling_kw);
        println!("  Typically occurs during hottest weather with high solar gains");
        println!("  May be affected by:");
        println!("    - Solar gain timing and distribution");
        println!("    - Internal heat gains");
        println!("    - Thermal mass buffering");
    } else {
        println!("  No cooling demand (case may not require cooling)");
    }
    println!();

    // Potential issues
    println!("=== Potential Issues ===\n");
    println!("1. Timestep Averaging Effect:");
    println!("   - Current: Hourly timesteps (3600 seconds)");
    println!("   - Effect: Sub-hourly peaks are averaged");
    println!("   - Reference tools (EnergyPlus, ESP-r) may use sub-hourly timesteps");
    println!("   - Impact: Peak loads may be underestimated by 10-30%");
    println!();

    println!("2. Thermal Mass Effects:");
    if matches!(spec.construction_type, fluxion::validation::ashrae_140_cases::ConstructionType::HighMass) {
        println!("   - High-mass construction (900 series)");
        println!("   - Thermal mass buffers temperature extremes");
        println!("   - Peak loads may be dampened compared to low-mass");
        println!("   - This is a legitimate 5R1C model characteristic");
    } else {
        println!("   - Low-mass construction (600 series)");
        println!("   - Less thermal buffering");
        println!("   - Peak loads should be more responsive");
    }
    println!();

    println!("3. HVAC Control:");
    println!("   - Ideal loads assumed (infinite capacity)");
    println!("   - Instantaneous response to setpoint deviations");
    println!("   - Real equipment has capacity limits and cycling losses");
    println!();

    // Recommendations
    println!("=== Recommendations ===\n");
    println!("If peak loads are significantly different from reference:");

    println!("\nOption 1: Accept as 5R1C Model Limitation");
    println!("  - 5R1C model uses lumped thermal mass");
    println!("  - Hourly timesteps average sub-hourly peaks");
    println!("  - This is a known limitation of the ISO 13790 standard");
    println!("  - Annual energies are the primary validation metric");

    println!("\nOption 2: Implement Peak Load Correction");
    println!("  - Apply correction factor based on timestep analysis");
    println!("  - Use empirical factor derived from reference comparison");
    println!("  - Only for reporting, not for physics calculation");

    println!("\nOption 3: Sub-Hourly Simulation");
    println!("  - Reduce timestep to 15 minutes or less");
    println!("  - Increases computational cost 4x");
    println!("  - May not be worth it for annual energy accuracy");

    println!("\n=== Decision Framework ===\n");
    println!("Accept as model limitation if:");
    println!("  ✓ Annual energies pass validation (primary metric)");
    println!("  ✓ Peak loads are consistently different (systematic, not random)");
    println!("  ✓ Root cause is fundamental 5R1C characteristic");
    println!("  ✓ Difference is within 20-30% of reference");

    println!("\nFix if:");
    println!("  ✓ Peak load calculation error identified");
    println!("  ✓ Simple physics-based correction available");
    println!("  ✓ Fix improves both peaks and annual energies");
    println!("  ✓ Difference is >50% (indicates error, not limitation)");

    println!("\nFor Case {}:", case_id);
    if peak_heating_kw > 0.0 {
        println!("  - Check if peak heating is within 20% of reference minimum");
        println!("  - If yes: Accept as 5R1C limitation");
        println!("  - If no: Investigate further");
    }
    if peak_cooling_kw > 0.0 {
        println!("  - Check if peak cooling is within 20% of reference range");
        println!("  - If yes: Accept as 5R1C limitation");
        println!("  - If no: Investigate solar gain distribution");
    }

    println!();
}

//! Diagnostic tests for thermal mass calibration (Plan 03-07c)
//!
//! This module provides diagnostic analysis for thermal mass conductances
//! and their impact on annual energy over-prediction for Case 900.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

/// Convert Joules to MWh
const J_TO_MWH: f64 = 1.0 / 3_600_000_000.0; // 1 MWh = 3.6e9 J

#[test]
fn test_case_900_thermal_mass_conductance_analysis() {
    // Diagnostic test to analyze thermal mass conductances (h_tr_em, h_tr_ms) and their impact on annual energy
    // This helps identify if conductance values need calibration to reduce annual energy over-prediction
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Thermal Mass Conductance Analysis (Plan 03-07c Task 2) ===");
    println!();

    // Check thermal mass conductances
    println!("Thermal Mass Conductances:");
    let h_tr_em_avg = model.h_tr_em.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;
    let h_tr_ms_avg = model.h_tr_ms.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;
    println!("  h_tr_em (Exterior -> Mass): {:.2} W/K", h_tr_em_avg);
    println!("  h_tr_ms (Mass -> Surface): {:.2} W/K", h_tr_ms_avg);
    println!();

    // Calculate coupling ratios
    let em_ms_ratio = h_tr_em_avg / h_tr_ms_avg;
    let em_total_ratio = h_tr_em_avg / (h_tr_em_avg + h_tr_ms_avg);
    println!("Coupling Ratios:");
    println!("  h_tr_em / h_tr_ms: {:.3}", em_ms_ratio);
    println!("  h_tr_em / (h_tr_em + h_tr_ms): {:.3}", em_total_ratio);
    println!();

    // Get derived sensitivity
    let sensitivity_avg = model
        .derived_sensitivity
        .as_ref()
        .to_vec()
        .iter()
        .sum::<f64>()
        / model.num_zones as f64;
    println!("Sensitivity Calculation:");
    println!("  Average sensitivity: {:.6} K/W", sensitivity_avg);
    println!(
        "  Inverse (1/sensitivity): {:.2} W/K",
        1.0 / sensitivity_avg
    );
    println!();

    // Analysis and recommendations
    println!("Analysis:");
    println!("Current State:");
    println!("  h_tr_em / h_tr_ms ratio = {:.3} (very low)", em_ms_ratio);
    println!(
        "  → Thermal mass is strongly coupled to interior (h_tr_ms = {:.2} W/K)",
        h_tr_ms_avg
    );
    println!(
        "  → Thermal mass is weakly coupled to exterior (h_tr_em = {:.2} W/K)",
        h_tr_em_avg
    );
    println!(
        "  → Sensitivity = {:.6} K/W (very low, causing high HVAC demand)",
        sensitivity_avg
    );
    println!();

    // Diagnostic hypothesis
    println!("Hypothesis:");
    println!(
        "The low h_tr_em / h_tr_ms ratio ({:.3}) means thermal mass releases stored energy",
        em_ms_ratio
    );
    println!(
        "primarily to interior (h_tr_ms = {:.2} W/K) rather than to exterior",
        h_tr_ms_avg
    );
    println!(
        "(h_tr_em = {:.2} W/K). This causes HVAC to work against thermal mass energy",
        h_tr_em_avg
    );
    println!("release, increasing annual energy demand.");
    println!();
    println!(
        "Solution: Increase h_tr_em to allow thermal mass to release more energy to exterior,"
    );
    println!("reducing HVAC burden. This can be achieved by increasing the");
    println!("thermal_mass_coupling_enhancement factor beyond the current 1.15x.");
    println!();

    // Expected improvement with higher h_tr_em
    println!("Expected Improvement with Higher h_tr_em:");
    println!("  Current: coupling_enhancement = 1.15x");
    println!("  Try: coupling_enhancement = 1.5x to 2.0x");
    println!("  Expected: Annual heating and cooling reduced toward reference ranges");
    println!("  Risk: Temperature swing reduction may decrease (trade-off)");
    println!();

    println!("✅ Thermal mass conductance analysis complete");
}

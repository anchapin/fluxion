//! Diagnostic tests for HVAC sensitivity calculation (Plan 03-08)
//!
//! This module provides diagnostic analysis for HVAC sensitivity calculation
//! and its impact on annual energy over-prediction for Case 900.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Convert Joules to MWh
const J_TO_MWH: f64 = 1.0 / 3_600_000_000.0; // 1 MWh = 3.6e9 J

#[test]
fn test_case_900_sensitivity_analysis() {
    // Diagnostic test to analyze HVAC sensitivity calculation for Case 900
    // This helps identify why sensitivity is 0.002065 K/W (very low, causing high HVAC demand)
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== HVAC Sensitivity Analysis (Plan 03-08 Task 1) ===");
    println!();

    // Check thermal capacitance
    let thermal_cap_avg =
        model.thermal_capacitance.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    println!("Thermal Capacitance:");
    println!("  Cm: {:.2} MJ/K", thermal_cap_avg / 1_000_000.0);
    println!();

    // Check conductances
    let h_tr_em_avg = model.h_tr_em.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    let h_tr_ms_avg = model.h_tr_ms.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    let h_tr_is_avg = model.h_tr_is.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    let h_tr_w_avg = model.h_tr_w.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    let h_ve_avg = model.h_ve.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    println!("5R1C Conductances:");
    println!("  h_tr_em (Exterior -> Mass): {:.2} W/K", h_tr_em_avg);
    println!("  h_tr_ms (Mass -> Surface): {:.2} W/K", h_tr_ms_avg);
    println!("  h_tr_is (Surface -> Interior): {:.2} W/K", h_tr_is_avg);
    println!(
        "  h_tr_w (Exterior -> Interior via windows): {:.2} W/K",
        h_tr_w_avg
    );
    println!("  h_ve (Ventilation): {:.2} W/K", h_ve_avg);
    println!();

    // Check derived parameters for sensitivity calculation
    let term_rest_1_avg =
        model.derived_term_rest_1.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    let h_ms_is_prod_avg =
        model.derived_h_ms_is_prod.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    let den_avg = model.derived_den.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    let sensitivity_avg =
        model.derived_sensitivity.as_ref().iter().sum::<f64>() / model.num_zones as f64;
    println!("Sensitivity Calculation Parameters:");
    println!(
        "  term_rest_1 = h_tr_ms * (h_tr_w + h_ve): {:.2} (W/K)²",
        term_rest_1_avg
    );
    println!(
        "  h_ms_is_prod = h_tr_ms * h_tr_is: {:.2} (W/K)²",
        h_ms_is_prod_avg
    );
    println!(
        "  den = h_ms_is_prod + term_rest_1 * (h_tr_w + h_ve + h_tr_floor): {:.2} (W/K)²",
        den_avg
    );
    println!(
        "  sensitivity = term_rest_1 / den: {:.6} K/W",
        sensitivity_avg
    );
    println!(
        "  Inverse (1/sensitivity): {:.2} W/K",
        1.0 / sensitivity_avg
    );
    println!();

    // Calculate thermal mass time constant
    // τ = C / (h_tr_em + h_tr_ms)
    let time_constant_hours = (thermal_cap_avg / (h_tr_em_avg + h_tr_ms_avg)) / 3600.0;
    println!("Thermal Mass Time Constant:");
    println!(
        "  τ = C / (h_tr_em + h_tr_ms): {:.2} hours",
        time_constant_hours
    );
    println!("  τ / timestep (1 hour): {:.2}", time_constant_hours / 1.0);
    println!();

    // Analysis of low sensitivity
    println!("Analysis:");
    println!("Why is sensitivity so low? ({:.6} K/W)", sensitivity_avg);
    println!();
    println!(
        "1. High thermal mass (C = {:.2} MJ/K):",
        thermal_cap_avg / 1_000_000.0
    );
    println!("   - Large thermal capacitance stores more energy");
    println!("   - Mass temperature changes slowly");
    println!("   - HVAC must work harder to change zone temperature");
    println!();
    println!(
        "2. Low exterior coupling (h_tr_em = {:.2} W/K):",
        h_tr_em_avg
    );
    println!("   - Thermal mass weakly coupled to exterior");
    println!("   - Mass releases stored energy primarily to interior");
    println!("   - HVAC must work against mass energy release");
    println!();
    println!(
        "3. High interior coupling (h_tr_ms = {:.2} W/K):",
        h_tr_ms_avg
    );
    println!("   - Thermal mass strongly coupled to interior");
    println!("   - Mass temperature dominates interior temperature");
    println!("   - Sensitivity to HVAC power is reduced");
    println!();
    println!("4. Time constant analysis:");
    println!("   - τ = {:.2} hours", time_constant_hours);
    println!("   - This is much larger than timestep (1 hour)");
    println!("   - Thermal mass provides significant damping");
    println!("   - HVAC effectiveness reduced by time constant");
    println!();

    // Calculate expected HVAC demand
    let temp_diff = 10.0; // 10°C temperature difference
    let expected_demand = temp_diff / sensitivity_avg;
    println!("HVAC Demand Calculation:");
    println!("  ΔT = 10°C (temperature difference)");
    println!("  Power = ΔT / sensitivity = {:.0} W", expected_demand);
    println!(
        "  This is {:.2}x the maximum heating capacity (2100 W)",
        expected_demand / 2100.0
    );
    println!();

    // Hypothesis for fix
    println!("Hypothesis for Fix:");
    println!("The sensitivity calculation doesn't account for thermal mass time constant effects.");
    println!("For high-mass buildings, HVAC effectiveness is reduced by thermal mass damping.");
    println!();
    println!("Proposed solution: Add thermal mass time constant factor to sensitivity calculation");
    println!("  sensitivity_corrected = sensitivity * (τ / timestep)^α");
    println!("  where α is a tuning parameter (try α = 0.5 to 1.0)");
    println!("  This reduces HVAC demand for high-mass buildings");
    println!();

    println!("✅ Sensitivity analysis complete");
}

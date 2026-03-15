//! Thermal mass time constant analysis for Case 900
//!
//! Diagnostic test to understand thermal mass dynamics and sensitivity calculation.
//! This helps identify the root cause of annual energy over-prediction.
//!
//! Key questions:
//! 1. What is the thermal mass time constant (τ = C / (h_tr_em + h_tr_ms))?
//! 2. What is the sensitivity calculation (sensitivity = term_rest_1 / den)?
//! 3. Why is free-floating temperature so low during winter (7-10°C)?
//! 4. Why is HVAC demand always at maximum (2100 W)?

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

#[test]
fn test_case_900_thermal_mass_time_constant_analysis() {
    println!("=== Thermal Mass Time Constant Analysis ===");

    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Extract thermal mass parameters
    let thermal_capacitance_j_k = model.thermal_capacitance.as_ref()[0]; // J/K
    let h_tr_em_w_k = model.h_tr_em.as_ref()[0]; // W/K (exterior -> mass)
    let h_tr_ms_w_k = model.h_tr_ms.as_ref()[0]; // W/K (mass -> surface)
    let h_tr_is_w_k = model.h_tr_is.as_ref()[0]; // W/K (surface -> interior)
    let h_tr_w_w_k = model.h_tr_w.as_ref()[0]; // W/K (windows)
    let h_ve_w_k = model.h_ve.as_ref()[0]; // W/K (ventilation)

    // Calculate thermal mass time constant
    let total_mass_conductance = h_tr_em_w_k + h_tr_ms_w_k;
    let time_constant_hours = thermal_capacitance_j_k / (total_mass_conductance * 3600.0); // Convert W to J/h

    println!("Thermal Mass Parameters:");
    println!(
        "  Thermal capacitance (C): {:.2} J/K",
        thermal_capacitance_j_k
    );
    println!("  h_tr_em (exterior -> mass): {:.2} W/K", h_tr_em_w_k);
    println!("  h_tr_ms (mass -> surface): {:.2} W/K", h_tr_ms_w_k);
    println!("  h_tr_is (surface -> interior): {:.2} W/K", h_tr_is_w_k);
    println!("  h_tr_w (windows): {:.2} W/K", h_tr_w_w_k);
    println!("  h_ve (ventilation): {:.2} W/K", h_ve_w_k);
    println!();
    println!("Thermal Mass Time Constant:");
    println!(
        "  Total mass conductance: {:.2} W/K",
        total_mass_conductance
    );
    println!("  Time constant (τ): {:.2} hours", time_constant_hours);
    println!(
        "  Time constant / timestep: {:.2}x",
        time_constant_hours / 1.0
    );
    println!();

    // Calculate sensitivity components
    let term_rest_1 = h_tr_ms_w_k + h_tr_is_w_k;
    let h_ext = (model.h_tr_w.as_ref()[0] + h_ve_w_k) + model.derived_h_ext.as_ref()[0];
    let h_ms_is_prod = model.derived_h_ms_is_prod.as_ref()[0];
    let h_tr_floor_w_k = model.h_tr_floor.as_ref()[0];
    let derived_ground_coeff = model.derived_ground_coeff.as_ref()[0];

    let den = h_ms_is_prod + term_rest_1 * h_ext + derived_ground_coeff;
    let sensitivity = term_rest_1 / den;

    println!("Sensitivity Calculation:");
    println!("  term_rest_1 (h_tr_ms + h_tr_is): {:.2} W/K", term_rest_1);
    println!("  h_ext (opaque + windows + vent): {:.2} W/K", h_ext);
    println!("  h_ms_is_prod: {:.2} W/K", h_ms_is_prod);
    println!("  h_tr_floor: {:.2} W/K", h_tr_floor_w_k);
    println!("  derived_ground_coeff: {:.2} (W/K)²", derived_ground_coeff);
    println!(
        "  den (h_ms_is_prod + term_rest_1 * h_ext + ground): {:.2} (W/K)²",
        den
    );
    println!("  sensitivity (term_rest_1 / den): {:.6} K/W", sensitivity);
    println!();

    // Calculate HVAC demand with current sensitivity
    let t_free = 7.06; // Free-floating temp during winter (from test output)
    let setpoint = 20.0;
    let delta_t = setpoint - t_free;
    let hvac_demand_w = delta_t / sensitivity;

    println!("HVAC Demand Calculation (Winter):");
    println!("  Free-floating temp: {:.2}°C", t_free);
    println!("  Setpoint: {:.2}°C", setpoint);
    println!("  Delta T: {:.2}°C", delta_t);
    println!("  Sensitivity: {:.6} K/W", sensitivity);
    println!("  HVAC demand: {:.2} W", hvac_demand_w);
    println!("  Heating capacity: 2100.0 W");
    println!(
        "  HVAC running at capacity: {:.1}%",
        (hvac_demand_w / 2100.0) * 100.0
    );
    println!();

    // Analyze the problem
    println!("=== Problem Analysis ===");

    if time_constant_hours > 2.0 {
        println!(
            "❌ Time constant too large: {:.2} hours > 2 hours",
            time_constant_hours
        );
        println!("   This causes thermal mass to dampen HVAC effectiveness");
    } else {
        println!(
            "✓ Time constant reasonable: {:.2} hours",
            time_constant_hours
        );
    }

    if sensitivity < 0.002 {
        println!("❌ Sensitivity too low: {:.6} K/W < 0.002 K/W", sensitivity);
        println!("   This causes HVAC demand = ΔT / sensitivity to be too high");
    } else {
        println!("✓ Sensitivity reasonable: {:.6} K/W", sensitivity);
    }

    if hvac_demand_w > 2000.0 {
        println!("❌ HVAC demand too high: {:.2} W > 2000 W", hvac_demand_w);
        println!("   HVAC runs at maximum capacity constantly");
    } else {
        println!("✓ HVAC demand reasonable: {:.2} W", hvac_demand_w);
    }

    if t_free < 15.0 {
        println!("❌ Free-floating temp too low: {:.2}°C < 15°C", t_free);
        println!("   HVAC must run constantly to maintain 20°C setpoint");
    } else {
        println!("✓ Free-floating temp reasonable: {:.2}°C", t_free);
    }

    println!();
    println!("=== Recommendations ===");

    if time_constant_hours > 2.0 && sensitivity < 0.002 {
        println!("1. Investigate thermal mass coupling (h_tr_em, h_tr_ms)");
        println!("   - High time constant + low sensitivity suggests mass releases too much heat to interior");
        println!("   - Consider adjusting h_tr_em/h_tr_ms ratio");
    }

    if t_free < 15.0 {
        println!("2. Investigate free-floating temperature calculation");
        println!("   - Why is Ti_free so low during winter?");
        println!("   - Check if thermal mass is releasing heat to interior (high h_tr_ms)");
    }

    if hvac_demand_w > 2000.0 {
        println!("3. Investigate HVAC sensitivity calculation");
        println!("   - Low sensitivity causes high HVAC demand");
        println!("   - Consider time constant-based correction");
    }

    println!();
    println!("=== Expected ASHRAE 140 Reference Values ===");
    println!("Annual heating: 1.17 - 2.04 MWh (current: 6.86 MWh)");
    println!("Annual cooling: 2.13 - 3.67 MWh (current: ~0.70 MWh)");
    println!("Peak heating: 1.10 - 2.10 kW (current: ~2.10 kW)");
    println!("Peak cooling: 2.10 - 3.50 kW (current: ~3.54 kW)");
}

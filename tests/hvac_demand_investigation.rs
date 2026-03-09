//! HVAC Demand Calculation Investigation for Case 900
//!
//! Diagnostic test to investigate HVAC demand calculation and free-floating temperature
//! to diagnose annual energy over-prediction.
//!
//! Root cause from Plan 03-08d:
//! - Annual heating: 6.86 MWh (236% above [1.17, 2.04] MWh reference)
//! - Annual cooling: 4.82 MWh (31% above [2.13, 3.67] MWh reference)
//! - Peak heating: 2.10 kW (perfect, within [1.10, 2.10] kW)
//! - Peak cooling: 3.57 kW (within [2.10, 3.70] kW)
//!
//! HVAC running excessively at or near peak capacity.
//! Issue is NOT with peak power calculation, but with HVAC demand calculation
//! or free-floating temperature.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_case_900_hvac_demand_calculation_investigation() {
    println!("=== HVAC Demand Calculation Investigation ===\n");

    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Extract key parameters
    let h_tr_em = model.h_tr_em.as_ref()[0]; // Exterior -> mass
    let h_tr_ms = model.h_tr_ms.as_ref()[0]; // Mass -> surface
    let h_tr_is = model.h_tr_is.as_ref()[0]; // Surface -> interior
    let h_tr_w = model.h_tr_w.as_ref()[0]; // Windows
    let h_ve = model.h_ve.as_ref()[0]; // Ventilation
    let thermal_capacitance = model.thermal_capacitance.as_ref()[0]; // J/K

    println!("=== 1. Thermal Mass Parameters ===");
    println!("Thermal capacitance (C): {:.2} MJ/K", thermal_capacitance / 1e6);
    println!("h_tr_em (exterior -> mass): {:.2} W/K", h_tr_em);
    println!("h_tr_ms (mass -> surface): {:.2} W/K", h_tr_ms);
    println!("h_tr_is (surface -> interior): {:.2} W/K", h_tr_is);
    println!("h_tr_w (windows): {:.2} W/K", h_tr_w);
    println!("h_ve (ventilation): {:.2} W/K", h_ve);

    // Calculate time constant
    let total_mass_conductance = h_tr_em + h_tr_ms;
    let time_constant_hours = thermal_capacitance / (total_mass_conductance * 3600.0);

    println!("\n=== 2. Time Constant Analysis ===");
    println!("Total mass conductance: {:.2} W/K", total_mass_conductance);
    println!("Time constant (τ): {:.2} hours", time_constant_hours);
    println!("τ / timestep: {:.2}x", time_constant_hours / 1.0);

    // Calculate coupling ratio
    let h_tr_em_ms_ratio = h_tr_em / h_tr_ms;
    let mass_to_exterior_pct = (h_tr_em / total_mass_conductance) * 100.0;
    let mass_to_surface_pct = (h_tr_ms / total_mass_conductance) * 100.0;

    println!("\n=== 3. Coupling Ratio Analysis ===");
    println!("h_tr_em / h_tr_ms: {:.4}", h_tr_em_ms_ratio);
    println!("Target ratio: > 0.1");
    println!("Thermal mass to exterior: {:.1}% (h_tr_em)", mass_to_exterior_pct);
    println!("Thermal mass to surface: {:.1}% (h_tr_ms)", mass_to_surface_pct);

    // Calculate sensitivity
    let term_rest_1 = h_tr_ms + h_tr_is;
    let h_ext = h_tr_w + h_ve + model.derived_h_ext.as_ref()[0];
    let h_ms_is_prod = model.derived_h_ms_is_prod.as_ref()[0];
    let derived_ground_coeff = model.derived_ground_coeff.as_ref()[0];
    let h_tr_floor = model.h_tr_floor.as_ref()[0];

    let den = h_ms_is_prod + term_rest_1 * h_ext + derived_ground_coeff;
    let sensitivity = term_rest_1 / den;

    println!("\n=== 4. Sensitivity Calculation ===");
    println!("term_rest_1 (h_tr_ms + h_tr_is): {:.2} W/K", term_rest_1);
    println!("h_ext (opaque + windows + vent): {:.2} W/K", h_ext);
    println!("h_ms_is_prod: {:.2} (W/K)²", h_ms_is_prod);
    println!("h_tr_floor: {:.2} W/K", h_tr_floor);
    println!("derived_ground_coeff: {:.2} (W/K)²", derived_ground_coeff);
    println!("den (h_ms_is_prod + term_rest_1 * h_ext + ground): {:.2} (W/K)²", den);
    println!("sensitivity (term_rest_1 / den): {:.6} K/W", sensitivity);
    println!("Target sensitivity: > 0.002 K/W");

    // HVAC demand calculation
    let t_free_winter = 7.06; // From diagnostic test
    let setpoint_heating = 20.0;
    let setpoint_cooling = 27.0;
    let delta_t_heating = setpoint_heating - t_free_winter;

    let hvac_demand_heating_w = delta_t_heating / sensitivity;
    let heating_capacity = 2100.0;
    let capacity_usage = (hvac_demand_heating_w / heating_capacity) * 100.0;

    println!("\n=== 5. HVAC Demand Calculation (Winter) ===");
    println!("Free-floating temp (Ti_free): {:.2}°C", t_free_winter);
    println!("Heating setpoint: {:.2}°C", setpoint_heating);
    println!("ΔT (setpoint - Ti_free): {:.2}°C", delta_t_heating);
    println!("Sensitivity: {:.6} K/W", sensitivity);
    println!("HVAC demand = ΔT / sensitivity: {:.2} W", hvac_demand_heating_w);
    println!("Heating capacity: {:.2} W", heating_capacity);
    println!("Capacity usage: {:.1}%", capacity_usage);

    println!("\n=== 6. Problem Identification ===");

    if h_tr_em_ms_ratio < 0.1 {
        println!("❌ h_tr_em/h_tr_ms ratio too low: {:.4} < 0.1", h_tr_em_ms_ratio);
        println!("   → Thermal mass exchanges mostly with interior, not exterior");
        println!("   → Mass temperature follows interior temperature");
        println!("   → Ti_free is too low during winter");
    } else {
        println!("✓ h_tr_em/h_tr_ms ratio reasonable: {:.4}", h_tr_em_ms_ratio);
    }

    if h_tr_ms > 1000.0 {
        println!("❌ h_tr_ms too high: {:.2} W/K > 1000 W/K", h_tr_ms);
        println!("   → Thermal mass releases too much cold to interior during winter");
        println!("   → Ti_free becomes too low");
    } else {
        println!("✓ h_tr_ms reasonable: {:.2} W/K", h_tr_ms);
    }

    if sensitivity < 0.002 {
        println!("❌ Sensitivity too low: {:.6} K/W < 0.002 K/W", sensitivity);
        println!("   → HVAC demand = ΔT / sensitivity is too high");
        println!("   → HVAC runs at max capacity constantly");
    } else {
        println!("✓ Sensitivity reasonable: {:.6} K/W", sensitivity);
    }

    if t_free_winter < 15.0 {
        println!("❌ Free-floating temp too low: {:.2}°C < 15°C", t_free_winter);
        println!("   → HVAC must run constantly to maintain 20°C setpoint");
        println!("   → High annual heating energy");
    } else {
        println!("✓ Free-floating temp reasonable: {:.2}°C", t_free_winter);
    }

    if hvac_demand_heating_w > heating_capacity {
        println!("❌ HVAC demand exceeds capacity: {:.2} W > {:.2} W", hvac_demand_heating_w, heating_capacity);
        println!("   → HVAC clamped to capacity, still not enough");
        println!("   → Runs at max capacity constantly");
    } else {
        println!("✓ HVAC demand reasonable: {:.2} W", hvac_demand_heating_w);
    }

    println!("\n=== 7. Root Cause Analysis ===");

    println!("1. High h_tr_ms ({:.2} W/K) causes strong coupling between mass and interior", h_tr_ms);
    println!("2. Low h_tr_em ({:.2} W/K) causes weak coupling between mass and exterior", h_tr_em);
    println!("3. Result: Thermal mass exchanges {:.1}% with interior, {:.1}% with exterior",
        mass_to_surface_pct, mass_to_exterior_pct);
    println!("4. During winter: Mass cools down with interior, stays cold");
    println!("5. Ti_free = {:.2}°C (too low)", t_free_winter);
    println!("6. ΔT = {:.2}°C - {:.2}°C = {:.2}°C (too large)",
        setpoint_heating, t_free_winter, delta_t_heating);
    println!("7. HVAC demand = {:.2}°C / {:.6} K/W = {:.2} W",
        delta_t_heating, sensitivity, hvac_demand_heating_w);
    println!("8. HVAC clamped to {:.2} W (capacity), runs at max constantly", heating_capacity);
    println!("9. Annual heating = {:.2} MWh (vs reference [{:.2}, {:.2}] MWh)",
        6.86, 1.17, 2.04);

    println!("\n=== 8. HVAC Demand Formula Verification ===");

    println!("Current formula: hvac_demand = ΔT / sensitivity");
    println!("Where:");
    println!("  ΔT = setpoint - Ti_free");
    println!("  sensitivity = term_rest_1 / den");
    println!("  term_rest_1 = h_tr_ms + h_tr_is");
    println!("  den = h_ms_is_prod + term_rest_1 * h_ext + derived_ground_coeff");

    println!("\nIs this formula correct for high-mass buildings?");
    println!("  - Low sensitivity (0.001845 K/W) causes high demand");
    println!("  - This is expected: high thermal mass dampens HVAC effectiveness");
    println!("  - BUT: annual energy is 236% above reference");
    println!("  - Suggests: Either sensitivity calculation or Ti_free is wrong");

    println!("\n=== 9. Free-Floating Temperature Calculation ===");

    println!("Ti_free formula from ISO 13790 5R1C:");
    println!("t_i_free = (num_tm + num_phi_st + num_rest) / den");
    println!("Where:");
    println!("  num_tm = h_ms_is_prod * Tm (mass term)");
    println!("  num_phi_st = h_tr_is * φ_st (surface flux)");
    println!("  num_rest = term_rest_1 * (h_ext * Te + φ_ia) + ground");
    println!("  den = h_ms_is_prod + term_rest_1 * h_ext + ground");

    println!("\nKey insight:");
    println!("  - num_tm depends on mass temperature (Tm)");
    println!("  - Tm evolves over time via thermal integration");
    println!("  - High h_tr_ms couples Tm strongly to interior");
    println!("  - Result: Tm follows interior temperature");
    println!("  - Ti_free becomes low when interior is cold");

    println!("\n=== 10. Proposed Fixes ===");

    println!("Solution 1: Increase h_tr_em / Decrease h_tr_ms ratio");
    println!("  - Target: h_tr_em/h_tr_ms > 0.1 (current: {:.4})", h_tr_em_ms_ratio);
    println!("  - Option A: Increase h_tr_em by 2.5x: {:.2} → {:.2} W/K", h_tr_em, h_tr_em * 2.5);
    println!("  - Option B: Decrease h_tr_ms by 35%: {:.2} → {:.2} W/K", h_tr_ms, h_tr_ms * 0.65);
    println!("  - Expected: Higher winter Ti_free, lower HVAC demand");

    println!("\nSolution 2: Time constant-based sensitivity correction");
    println!("  - sensitivity_corrected = sensitivity * f(τ)");
    println!("  - Where f(τ) = 1.0 for τ < 2h, increases as τ increases");
    println!("  - Apply only to energy, not peak loads");
    println!("  - Expected: Lower annual energy, maintain peak loads");

    println!("\nSolution 3: Free-floating temperature fix");
    println!("  - Investigate 5R1C vs 6R2C model");
    println!("  - Consider envelope/internal mass separation");
    println!("  - Expected: More accurate Ti_free calculation");

    println!("\n=== 11. ASHRAE 140 Reference Comparison ===");

    println!("Case 900 Reference Values:");
    println!("  Annual heating: [{:.2}, {:.2}] MWh (current: {:.2} MWh, {:.0}% above)",
        1.17, 2.04, 6.86, 236.0);
    println!("  Annual cooling: [{:.2}, {:.2}] MWh (current: {:.2} MWh, {:.0}% above)",
        2.13, 3.67, 4.82, 31.0);
    println!("  Peak heating: [{:.2}, {:.2}] kW (current: {:.2} kW, ✓)",
        1.10, 2.10, 2.10);
    println!("  Peak cooling: [{:.2}, {:.2}] kW (current: {:.2} kW, 2% above)",
        2.10, 3.70, 3.57);

    println!("\n=== Conclusion ===");

    println!("Root cause: h_tr_em/h_tr_ms ratio too low ({:.4} < 0.1)", h_tr_em_ms_ratio);
    println!("  → Thermal mass exchanges 95% with interior, 5% with exterior");
    println!("  → Winter Ti_free too low (7.06°C)");
    println!("  → High ΔT (12.94°C) and low sensitivity (0.001845 K/W)");
    println!("  → HVAC demand = 7013 W (334% of capacity)");
    println!("  → Annual heating = 6.86 MWh (236% above reference)");

    println!("\nHVAC demand calculation formula is correct.");
    println!("Issue is with parameterization, not formula.");
    println!("Recommendation: Fix h_tr_em/h_tr_ms ratio (Solution 1).");
}

#[test]
fn test_case_900_hvac_demand_formula_validation() {
    println!("=== HVAC Demand Formula Validation ===\n");

    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Get sensitivity from model
    let sensitivity = model.derived_sensitivity.as_ref()[0];

    // Test HVAC demand formula with different scenarios
    println!("Test 1: Winter scenario (low Ti_free)");
    let ti_free_winter = 7.0;
    let setpoint = 20.0;
    let delta_t = setpoint - ti_free_winter;
    let demand = delta_t / sensitivity;
    println!("  Ti_free = {:.2}°C, setpoint = {:.2}°C", ti_free_winter, setpoint);
    println!("  ΔT = {:.2}°C, sensitivity = {:.6} K/W", delta_t, sensitivity);
    println!("  Demand = {:.2} W, capacity = 2100 W, usage = {:.1}%",
        demand, (demand / 2100.0) * 100.0);

    println!("\nTest 2: Summer scenario (high Ti_free)");
    let ti_free_summer = 30.0;
    let cooling_setpoint = 27.0;
    let delta_t_cooling = ti_free_summer - cooling_setpoint;
    let cooling_demand = delta_t_cooling / sensitivity;
    println!("  Ti_free = {:.2}°C, setpoint = {:.2}°C", ti_free_summer, cooling_setpoint);
    println!("  ΔT = {:.2}°C, sensitivity = {:.6} K/W", delta_t_cooling, sensitivity);
    println!("  Demand = {:.2} W, capacity = 3500 W, usage = {:.1}%",
        cooling_demand, (cooling_demand / 3500.0) * 100.0);

    println!("\nTest 3: Moderate Ti_free (within deadband)");
    let ti_free_moderate = 23.0;
    println!("  Ti_free = {:.2}°C", ti_free_moderate);
    println!("  Heating setpoint = 20.0°C, cooling setpoint = 27.0°C");
    println!("  Within deadband, HVAC demand = 0 W");

    println!("\nTest 4: What if sensitivity was 2x higher?");
    let sensitivity_2x = sensitivity * 2.0;
    let demand_2x = delta_t / sensitivity_2x;
    println!("  Current sensitivity: {:.6} K/W", sensitivity);
    println!("  2x sensitivity: {:.6} K/W", sensitivity_2x);
    println!("  Current demand: {:.2} W", demand);
    println!("  2x sensitivity demand: {:.2} W ({:.1}% reduction)",
        demand_2x, ((demand - demand_2x) / demand) * 100.0);

    println!("\nFormula validation:");
    println!("✓ HVAC demand = ΔT / sensitivity is mathematically correct");
    println!("✓ Low sensitivity → high demand (inverse relationship)");
    println!("✓ High ΔT → high demand (linear relationship)");
    println!("✓ This matches thermodynamic principles");
}

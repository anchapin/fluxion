//! Thermal mass coupling investigation for Case 900
//!
//! Investigate how h_tr_em and h_tr_ms coupling affects free-floating temperature
//! and HVAC demand. Test alternative coupling strategies to fix annual energy.
//!
//! Key hypothesis:
//! - High h_tr_ms (1092 W/K) causes thermal mass to release too much cold to interior
//! - This makes Ti_free very low during winter (7-10°C)
//! - HVAC must run constantly to maintain 20°C setpoint
//! - High annual heating energy (6.86 MWh vs [1.17, 2.04] MWh reference)

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_thermal_mass_coupling_analysis() {
    println!("=== Thermal Mass Coupling Analysis ===");

    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Extract thermal mass coupling parameters
    let h_tr_em = model.h_tr_em.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_is = model.h_tr_is.as_ref()[0];
    let thermal_capacitance = model.thermal_capacitance.as_ref()[0];

    println!("Thermal Mass Coupling Parameters:");
    println!("  h_tr_em (exterior -> mass): {:.2} W/K", h_tr_em);
    println!("  h_tr_ms (mass -> surface): {:.2} W/K", h_tr_ms);
    println!("  h_tr_is (surface -> interior): {:.2} W/K", h_tr_is);
    println!("  Thermal capacitance: {:.2} J/K", thermal_capacitance);
    println!();

    // Calculate time constant
    let total_mass_conductance = h_tr_em + h_tr_ms;
    let time_constant_hours = thermal_capacitance / (total_mass_conductance * 3600.0);

    println!("Time Constant Analysis:");
    println!("  Total mass conductance: {:.2} W/K", total_mass_conductance);
    println!("  Time constant (τ): {:.2} hours", time_constant_hours);
    println!("  τ / timestep: {:.2}x", time_constant_hours / 1.0);
    println!();

    // Analyze coupling ratio
    let h_tr_em_ms_ratio = h_tr_em / h_tr_ms;
    let h_tr_ms_is_ratio = h_tr_ms / h_tr_is;

    println!("Coupling Ratio Analysis:");
    println!("  h_tr_em / h_tr_ms: {:.4}", h_tr_em_ms_ratio);
    println!("  h_tr_ms / h_tr_is: {:.2}", h_tr_ms_is_ratio);
    println!();

    // Analyze heat flow pathways
    println!("Heat Flow Pathways (from mass):");
    println!("  To exterior: h_tr_em = {:.2} W/K ({:.1}%)", h_tr_em, (h_tr_em / total_mass_conductance) * 100.0);
    println!("  To surface: h_tr_ms = {:.2} W/K ({:.1}%)", h_tr_ms, (h_tr_ms / total_mass_conductance) * 100.0);
    println!();

    // Analyze surface heat flow pathways
    let total_surface_conductance = h_tr_ms + h_tr_is;
    println!("Surface Heat Flow Pathways (to interior):");
    println!("  From mass: h_tr_ms = {:.2} W/K ({:.1}%)", h_tr_ms, (h_tr_ms / total_surface_conductance) * 100.0);
    println!("  To interior: h_tr_is = {:.2} W/K ({:.1}%)", h_tr_is, (h_tr_is / total_surface_conductance) * 100.0);
    println!();

    // Identify problems
    println!("=== Problem Identification ===");

    if h_tr_em_ms_ratio < 0.1 {
        println!("❌ h_tr_em/h_tr_ms ratio too low: {:.4} < 0.1", h_tr_em_ms_ratio);
        println!("   Thermal mass exchanges mostly with interior, not exterior");
        println!("   This causes mass temperature to follow interior temperature");
        println!("   Result: Ti_free is too low during winter");
    } else {
        println!("✓ h_tr_em/h_tr_ms ratio reasonable: {:.4}", h_tr_em_ms_ratio);
    }

    if h_tr_ms > 1000.0 {
        println!("❌ h_tr_ms too high: {:.2} W/K > 1000 W/K", h_tr_ms);
        println!("   Thermal mass releases too much heat/cold to interior");
        println!("   Result: Ti_free is too low/high during winter/summer");
    } else {
        println!("✓ h_tr_ms reasonable: {:.2} W/K", h_tr_ms);
    }

    if time_constant_hours > 4.0 {
        println!("❌ Time constant too large: {:.2} hours > 4 hours", time_constant_hours);
        println!("   Thermal mass responds too slowly to exterior conditions");
        println!("   Result: Thermal inertia causes temperature lag");
    } else {
        println!("✓ Time constant reasonable: {:.2} hours", time_constant_hours);
    }

    println!();
    println!("=== Root Cause Analysis ===");

    println!("1. High h_tr_ms ({:.2} W/K) causes strong coupling between mass and interior", h_tr_ms);
    println!("2. Low h_tr_em ({:.2} W/K) causes weak coupling between mass and exterior", h_tr_em);
    println!("3. Result: Thermal mass temperature follows interior temperature");
    println!("4. During winter: Mass cools down with interior, stays cold");
    println!("5. Ti_free is low because mass releases cold to interior via h_tr_ms");
    println!("6. HVAC must run constantly to maintain 20°C setpoint");
    println!("7. Low Ti_free → high ΔT → high HVAC demand → high annual heating");
    println!();

    println!("=== Proposed Solutions ===");

    println!("Solution 1: Increase h_tr_em / Decrease h_tr_ms ratio");
    println!("  - Target h_tr_em/h_tr_ms ratio: >0.1 (current: {:.4})", h_tr_em_ms_ratio);
    println!("  - Options:");
    println!("    a) Increase h_tr_em by 2-3x: {:.2} → {:.2} W/K", h_tr_em, h_tr_em * 2.5);
    println!("    b) Decrease h_tr_ms by 30-40%: {:.2} → {:.2} W/K", h_tr_ms, h_tr_ms * 0.6);
    println!("    c) Both: Increase h_tr_em 2x, decrease h_tr_ms 30%");
    println!("  - Expected impact:");
    println!("    * Better thermal mass exchange with exterior");
    println!("    * Higher winter Ti_free (less cold released to interior)");
    println!("    * Lower HVAC demand, lower annual heating");
    println!();

    println!("Solution 2: Time constant-based sensitivity correction");
    println!("  - sensitivity_corrected = sensitivity * f(τ)");
    println!("  - Where f(τ) = 1.0 for τ < 2h, increases as τ increases");
    println!("  - Expected impact:");
    println!("    * Increase sensitivity for high-τ buildings");
    println!("    * Lower HVAC demand = ΔT / sensitivity_corrected");
    println!("    * Lower annual heating energy");
    println!("  - Risk: May affect peak loads (thermal_mass_correction_factor issue)");
    println!();

    println!("Solution 3: Free-floating temperature calculation fix");
    println!("  - Investigate why Ti_free is so low (7-10°C)");
    println!("  - Check if 5R1C network correctly models thermal mass buffering");
    println!("  - Consider 6R2C model with envelope/internal mass separation");
    println!("  - Expected impact:");
    println!("    * More accurate Ti_free calculation");
    println!("    * Better HVAC demand prediction");
    println!("  - Risk: Requires model complexity increase");
    println!();

    println!("=== Recommended Next Steps ===");

    println!("1. Test Solution 1 (increase h_tr_em / decrease h_tr_ms):");
    println!("   - Modify case_builder.rs to adjust coupling");
    println!("   - Run full simulation and measure impact");
    println!("   - Verify peak loads remain in range");
    println!();

    println!("2. If Solution 1 insufficient, test Solution 2:");
    println!("   - Implement time constant-based correction");
    println!("   - Apply only to annual energy, not peak loads");
    println!("   - Verify no peak load regressions");
    println!();

    println!("3. As last resort, consider Solution 3:");
    println!("   - Investigate 6R2C model parameterization");
    println!("   - Compare with ASHRAE 140 reference implementation");
    println!("   - More complex but may be more accurate");
}

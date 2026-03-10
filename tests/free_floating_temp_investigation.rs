//! Free-floating temperature investigation for Case 900
//!
//! Investigate why Ti_free is so low during winter (7-10°C) and compare
//! 5R1C vs 6R2C thermal network behavior.
//!
//! Key hypothesis:
//! - 5R1C single mass node cannot properly separate envelope/internal mass effects
//! - h_tr_em/h_tr_ms = 0.0525 is too low (95% to interior, 5% to exterior)
//! - 6R2C with separate envelope/internal mass nodes might naturally fix coupling issue
//! - Envelope mass should couple more strongly to exterior (higher h_tr_em)
//! - Internal mass should couple more strongly to interior (higher h_tr_ms)

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn test_case_900_free_floating_temp_analysis() {
    println!("=== Free-Floating Temperature Investigation: 5R1C vs 6R2C ===");

    let spec = ASHRAE140Case::Case900.spec();

    // Create 5R1C model
    let mut model_5r1c = ThermalModel::<VectorField>::from_spec(&spec);

    // Create 6R2C model
    let mut model_6r2c = ThermalModel::<VectorField>::from_spec(&spec);
    model_6r2c.configure_6r2c_model(0.75, 100.0); // 75% envelope mass, h_tr_me = 100 W/K

    // Analyze thermal mass coupling in 5R1C
    let h_tr_em_5r1c = model_5r1c.h_tr_em.as_ref()[0];
    let h_tr_ms_5r1c = model_5r1c.h_tr_ms.as_ref()[0];
    let h_tr_is_5r1c = model_5r1c.h_tr_is.as_ref()[0];
    let thermal_capacitance_5r1c = model_5r1c.thermal_capacitance.as_ref()[0];

    println!("=== 5R1C Model Analysis ===");
    println!("Thermal Mass Parameters:");
    println!(
        "  Thermal capacitance (C): {:.2} J/K",
        thermal_capacitance_5r1c
    );
    println!("  h_tr_em (exterior -> mass): {:.2} W/K", h_tr_em_5r1c);
    println!("  h_tr_ms (mass -> surface): {:.2} W/K", h_tr_ms_5r1c);
    println!("  h_tr_is (surface -> interior): {:.2} W/K", h_tr_is_5r1c);
    println!();

    // Calculate coupling ratio
    let h_tr_em_ms_ratio_5r1c = h_tr_em_5r1c / h_tr_ms_5r1c;
    let total_mass_conductance_5r1c = h_tr_em_5r1c + h_tr_ms_5r1c;
    let time_constant_5r1c = thermal_capacitance_5r1c / (total_mass_conductance_5r1c * 3600.0);

    println!("Coupling Analysis:");
    println!("  h_tr_em / h_tr_ms: {:.4}", h_tr_em_ms_ratio_5r1c);
    println!(
        "  Total mass conductance: {:.2} W/K",
        total_mass_conductance_5r1c
    );
    println!("  Time constant (τ): {:.2} hours", time_constant_5r1c);
    println!();

    println!("Heat Flow Pathways (from mass):");
    println!(
        "  To exterior: h_tr_em = {:.2} W/K ({:.1}%)",
        h_tr_em_5r1c,
        (h_tr_em_5r1c / total_mass_conductance_5r1c) * 100.0
    );
    println!(
        "  To surface: h_tr_ms = {:.2} W/K ({:.1}%)",
        h_tr_ms_5r1c,
        (h_tr_ms_5r1c / total_mass_conductance_5r1c) * 100.0
    );
    println!();

    // Analyze thermal mass coupling in 6R2C
    let h_tr_em_envelope = model_6r2c.h_tr_em.as_ref()[0];
    let h_tr_ms_envelope = model_6r2c.h_tr_ms.as_ref()[0];
    let h_tr_me = model_6r2c.h_tr_me.as_ref()[0]; // Envelope <-> internal mass coupling
    let cm_envelope = model_6r2c.envelope_thermal_capacitance.as_ref()[0];
    let cm_internal = model_6r2c.internal_thermal_capacitance.as_ref()[0];
    let total_cm_6r2c = cm_envelope + cm_internal;

    println!("=== 6R2C Model Analysis ===");
    println!("Thermal Mass Parameters:");
    println!(
        "  Envelope capacitance (C_envelope): {:.2} J/K ({:.1}%)",
        cm_envelope,
        (cm_envelope / total_cm_6r2c) * 100.0
    );
    println!(
        "  Internal capacitance (C_internal): {:.2} J/K ({:.1}%)",
        cm_internal,
        (cm_internal / total_cm_6r2c) * 100.0
    );
    println!("  Total capacitance: {:.2} J/K", total_cm_6r2c);
    println!();

    println!("Coupling Parameters:");
    println!(
        "  h_tr_em (exterior -> envelope mass): {:.2} W/K",
        h_tr_em_envelope
    );
    println!(
        "  h_tr_ms (envelope mass -> surface): {:.2} W/K",
        h_tr_ms_envelope
    );
    println!(
        "  h_tr_me (envelope mass <-> internal mass): {:.2} W/K",
        h_tr_me
    );
    println!(
        "  h_tr_is (surface -> interior): {:.2} W/K",
        model_6r2c.h_tr_is.as_ref()[0]
    );
    println!();

    // Calculate time constants for both masses
    let time_constant_envelope =
        cm_envelope / ((h_tr_em_envelope + h_tr_ms_envelope + h_tr_me) * 3600.0);
    let time_constant_internal = cm_internal / (h_tr_me * 3600.0);

    println!("Time Constants:");
    println!(
        "  Envelope time constant (τ_envelope): {:.2} hours",
        time_constant_envelope
    );
    println!(
        "  Internal time constant (τ_internal): {:.2} hours",
        time_constant_internal
    );
    println!();

    println!("Heat Flow Pathways:");
    println!(
        "  Envelope mass -> exterior: h_tr_em = {:.2} W/K ({:.1}%)",
        h_tr_em_envelope,
        (h_tr_em_envelope / (h_tr_em_envelope + h_tr_ms_envelope + h_tr_me)) * 100.0
    );
    println!(
        "  Envelope mass -> surface: h_tr_ms = {:.2} W/K ({:.1}%)",
        h_tr_ms_envelope,
        (h_tr_ms_envelope / (h_tr_em_envelope + h_tr_ms_envelope + h_tr_me)) * 100.0
    );
    println!(
        "  Envelope <-> internal mass: h_tr_me = {:.2} W/K ({:.1}%)",
        h_tr_me,
        (h_tr_me / (h_tr_em_envelope + h_tr_ms_envelope + h_tr_me)) * 100.0
    );
    println!();

    // Simulate one winter day and compare temperatures
    println!("=== Winter Day Simulation (Hour 5000-5024) ===");
    println!();

    let winter_day_start = 5000; // January (hour 0-743)
    let winter_day_end = winter_day_start + 24;

    println!("5R1C Model:");
    let mut min_temp_5r1c = f64::MAX;
    let mut max_temp_5r1c = f64::MIN;
    let mut sum_temp_5r1c = 0.0;
    let mut count = 0;

    for hour in winter_day_start..winter_day_end {
        // Use synthetic weather data (10°C base + 10°C daily cycle)
        let hour_of_day = hour % 24;
        let daily_cycle = ((hour_of_day as f64) - 12.0) * std::f64::consts::PI / 12.0;
        let outdoor_temp = 10.0 + 10.0 * daily_cycle.cos();
        model_5r1c.step_physics(hour, outdoor_temp);
        let temp = model_5r1c.temperatures.as_ref()[0];
        min_temp_5r1c = min_temp_5r1c.min(temp);
        max_temp_5r1c = max_temp_5r1c.max(temp);
        sum_temp_5r1c += temp;
        count += 1;
    }

    let avg_temp_5r1c = sum_temp_5r1c / count as f64;
    println!("  Min temperature: {:.2}°C", min_temp_5r1c);
    println!("  Max temperature: {:.2}°C", max_temp_5r1c);
    println!("  Avg temperature: {:.2}°C", avg_temp_5r1c);
    println!();

    println!("6R2C Model:");
    let mut min_temp_6r2c = f64::MAX;
    let mut max_temp_6r2c = f64::MIN;
    let mut sum_temp_6r2c = 0.0;
    let mut count = 0;

    for hour in winter_day_start..winter_day_end {
        // Use synthetic weather data (10°C base + 10°C daily cycle)
        let hour_of_day = hour % 24;
        let daily_cycle = ((hour_of_day as f64) - 12.0) * std::f64::consts::PI / 12.0;
        let outdoor_temp = 10.0 + 10.0 * daily_cycle.cos();
        model_6r2c.step_physics(hour, outdoor_temp);
        let temp = model_6r2c.temperatures.as_ref()[0];
        min_temp_6r2c = min_temp_6r2c.min(temp);
        max_temp_6r2c = max_temp_6r2c.max(temp);
        sum_temp_6r2c += temp;
        count += 1;
    }

    let avg_temp_6r2c = sum_temp_6r2c / count as f64;
    println!("  Min temperature: {:.2}°C", min_temp_6r2c);
    println!("  Max temperature: {:.2}°C", max_temp_6r2c);
    println!("  Avg temperature: {:.2}°C", avg_temp_6r2c);
    println!();

    // Compare results
    println!("=== Comparison ===");
    println!("Temperature Improvement:");
    let temp_improvement = avg_temp_6r2c - avg_temp_5r1c;
    println!("  5R1C avg temperature: {:.2}°C", avg_temp_5r1c);
    println!("  6R2C avg temperature: {:.2}°C", avg_temp_6r2c);
    println!(
        "  Improvement: {:.2}°C ({:.1}%)",
        temp_improvement,
        (temp_improvement / avg_temp_5r1c.abs()) * 100.0
    );
    println!();

    // Identify problem
    println!("=== Problem Analysis ===");

    if h_tr_em_ms_ratio_5r1c < 0.1 {
        println!(
            "❌ 5R1C h_tr_em/h_tr_ms ratio too low: {:.4} < 0.1",
            h_tr_em_ms_ratio_5r1c
        );
        println!("   Thermal mass exchanges 95% with interior, 5% with exterior");
        println!("   This causes mass temperature to follow interior temperature");
        println!("   Result: Ti_free is too low during winter");
    }

    if avg_temp_5r1c < 15.0 {
        println!(
            "❌ 5R1C temperature too low during winter: {:.2}°C < 15°C",
            avg_temp_5r1c
        );
        println!("   HVAC must run constantly to maintain 20°C setpoint");
        println!(
            "   High ΔT = 20 - {:.2} = {:.2}°C causes high HVAC demand",
            avg_temp_5r1c,
            20.0 - avg_temp_5r1c
        );
    }

    if avg_temp_6r2c > avg_temp_5r1c {
        println!(
            "✓ 6R2C temperature improved: {:.2}°C > 5R1C {:.2}°C",
            avg_temp_6r2c, avg_temp_5r1c
        );
        println!("   Envelope mass couples better to exterior");
        println!("   Internal mass couples better to interior");
        println!("   Better separation of thermal effects");
    } else {
        println!(
            "⚠ 6R2C temperature not improved: {:.2}°C (need further investigation)",
            avg_temp_6r2c
        );
    }

    println!();
    println!("=== Proposed Solution ===");

    if avg_temp_6r2c > avg_temp_5r1c {
        println!("1. Enable 6R2C model for Case 900:");
        println!("   - Envelope mass (75%): couples to exterior via h_tr_em");
        println!("   - Internal mass (25%): couples to interior via h_tr_me");
        println!("   - Better separation of thermal effects");
        println!("   - Higher winter temperature = lower HVAC demand = lower annual energy");
        println!();
        println!("2. Benefits of 6R2C model:");
        println!("   - Natural coupling ratio: envelope mass couples to exterior");
        println!("   - No manual parameter tuning needed");
        println!("   - More accurate representation of building physics");
        println!("   - Envelope mass absorbs exterior energy during day");
        println!("   - Internal mass buffers interior temperature swings");
    } else {
        println!("6R2C model does not improve temperature significantly");
        println!("Need to investigate further:");
        println!("   - Check 6R2C parameterization (h_tr_em, h_tr_ms, h_tr_me)");
        println!("   - Compare with ASHRAE 140 reference implementation");
        println!("   - Consider simplified 6R2C with different mass split");
    }

    println!();
    println!("=== Next Steps ===");

    println!("If 6R2C improves Ti_free:");
    println!("1. Enable 6R2C for Case 900 in case_builder.rs");
    println!("2. Run full ASHRAE 140 validation");
    println!("3. Verify annual heating within [1.17, 2.04] MWh");
    println!("4. Verify annual cooling within [2.13, 3.67] MWh");
    println!("5. Verify peak loads remain in range");

    println!();
    println!("If 6R2C does not improve Ti_free:");
    println!("1. Investigate 6R2C parameterization");
    println!("2. Compare h_tr_em/h_tr_ms ratio with reference");
    println!("3. Consider Solution 1 Revisited (more aggressive coupling adjustment)");
    println!("4. Consider separate heating/cooling coupling parameters");
}

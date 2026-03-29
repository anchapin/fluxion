//! Phase 1 Task 1.1: Analyze thermal time constants for all 600-series cases
//!
//! This tool outputs τ (tau) values for all 600-series cases and compares them
//! to ISO 13790 recommended ranges.
//!
//! Purpose: Identify if h_tr_ms is too low (causing τ too high) or too high (causing τ too fast)

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Phase 1 Task 1.1: Thermal Time Constant Analysis ===");
    println!("Analyzing 600-series low-mass cases\n");

    // All 600-series cases to analyze
    let cases = vec![
        ("600", ASHRAE140Case::Case600, "Low-Mass Baseline"),
        ("610", ASHRAE140Case::Case610, "South Shading"),
        ("620", ASHRAE140Case::Case620, "East/West Windows"),
        ("630", ASHRAE140Case::Case630, "East/West Shading"),
        ("640", ASHRAE140Case::Case640, "Thermostat Setback"),
        ("650", ASHRAE140Case::Case650, "Night Ventilation"),
        ("600FF", ASHRAE140Case::Case600FF, "Free-Floating"),
        (
            "650FF",
            ASHRAE140Case::Case650FF,
            "Free-Floating + Night Vent",
        ),
    ];

    // ISO 13790 Recommended Ranges
    println!("=== ISO 13790 Recommended Ranges ===");
    println!("Thermal Time Constant (τ = C_m / h_tr_ms):");
    println!("  Very Light Mass:  0.5 - 1.5 hours");
    println!("  Light Mass:       1.0 - 2.5 hours");
    println!("  Medium Mass:      1.5 - 4.0 hours");
    println!("  Heavy Mass:       2.5 - 6.0 hours");
    println!("  Very Heavy Mass:  4.0 - 8.0 hours");
    println!();

    // Table header
    println!("┌────────┬────────────────────────────┬─────────────┬─────────────┬─────────────┬───────────────┐");
    println!("│ Case   │ Description              │ C_m (kJ/K)  │ h_tr_ms (W/K)│ τ (hours)   │ Status        │");
    println!("├────────┼────────────────────────────┼─────────────┼─────────────┼─────────────┼───────────────┤");

    let mut all_cases = Vec::new();

    for (case_id, case_variant, description) in cases {
        let model = ThermalModel::from_spec(&case_variant.spec());

        // Extract thermal parameters
        let h_tr_ms = model.h_tr_ms.as_ref()[0];
        let h_tr_em = model.h_tr_em.as_ref()[0];
        let thermal_capacitance = model.thermal_capacitance.as_ref()[0]; // J/K

        // Convert to kJ/K for readability
        let thermal_capacitance_kj = thermal_capacitance / 1000.0;

        // Calculate thermal time constant τ = C_m / h_tr_ms
        let tau_seconds = thermal_capacitance / h_tr_ms;
        let tau_hours = tau_seconds / 3600.0;

        // Determine status
        let status = if tau_hours < 0.5 {
            "❌ TOO FAST (<0.5h)"
        } else if tau_hours > 8.0 {
            "❌ TOO SLOW (>8h)"
        } else {
            "✓ OK"
        };

        // Store for later analysis
        all_cases.push((
            case_id.to_string(),
            description.to_string(),
            thermal_capacitance_kj,
            h_tr_ms,
            h_tr_em,
            tau_hours,
            status.to_string(),
        ));

        // Print table row
        println!(
            "│ {:<6} │ {:<24} │ {:>10.2} │ {:>11.2} │ {:>11.2} │ {:<13} │",
            case_id, description, thermal_capacitance_kj, h_tr_ms, tau_hours, status
        );
    }

    println!("└────────┴────────────────────────────┴─────────────┴─────────────┴─────────────┴───────────────┘");
    println!();

    // Analysis
    println!("=== Analysis ===");

    let tau_values: Vec<f64> = all_cases.iter().map(|c| c.5).collect();
    let avg_tau = tau_values.iter().sum::<f64>() / tau_values.len() as f64;

    println!("Average τ across all cases: {:.2} hours", avg_tau);
    println!();

    // Count issues
    let too_fast = all_cases.iter().filter(|c| c.5 < 0.5).count();
    let too_slow = all_cases.iter().filter(|c| c.5 > 8.0).count();
    let ok = all_cases
        .iter()
        .filter(|c| c.5 >= 0.5 && c.5 <= 8.0)
        .count();

    println!("τ Distribution:");
    println!("  Too fast (<0.5h): {} cases", too_fast);
    println!("  Too slow (>8h):   {} cases", too_slow);
    println!("  OK (0.5-8h):      {} cases", ok);
    println!();

    // Check if h_tr_ms is too low (causing τ too high) or too high (causing τ too fast)
    println!("=== Root Cause Analysis ===");

    let avg_h_tr_ms: f64 = all_cases.iter().map(|c| c.3).sum::<f64>() / all_cases.len() as f64;
    println!("Average h_tr_ms across all cases: {:.2} W/K", avg_h_tr_ms);
    println!("Expected h_tr_ms range: 10-100 W/K for realistic thermal lag");
    println!();

    if avg_h_tr_ms > 500.0 {
        println!("⚠️  h_tr_ms is VERY HIGH!");
        println!("   Current: {:.2} W/K", avg_h_tr_ms);
        println!("   Expected: 10-100 W/K");
        println!("   Impact: Thermal mass responds too fast (τ too small)");
        println!();
        println!("   Formula: h_tr_ms = 9.1 W/m²K × A_m");
        println!(
            "   Current: 9.1 × {:.2} m² = {:.2} W/K",
            all_cases[0].3 / 9.1,
            avg_h_tr_ms
        );
        println!();
        println!("   Possible causes:");
        println!("   1. A_m factor may be too high for low-mass buildings");
        println!("   2. ISO 13790 9.1 W/m²K coefficient may not apply to ASHRAE 140");
        println!("   3. h_tr_ms should be derived from thermal resistance, not mass area");
    } else if avg_h_tr_ms < 10.0 {
        println!("⚠️  h_tr_ms is VERY LOW!");
        println!("   Current: {:.2} W/K", avg_h_tr_ms);
        println!("   Expected: 10-100 W/K");
        println!("   Impact: Thermal mass decoupled from zone (τ too large)");
    } else {
        println!("✓ h_tr_ms is within expected range");
    }

    println!();

    // Detailed breakdown for problematic cases
    println!("=== Detailed Analysis of Problematic Cases ===");

    for (case_id, description, cm, h_tr_ms, h_tr_em, tau, status) in &all_cases {
        if status.contains("TOO") {
            println!("Case {} ({}):", case_id, description);
            println!("  C_m = {:.2} kJ/K", cm);
            println!("  h_tr_ms = {:.2} W/K", h_tr_ms);
            println!("  h_tr_em = {:.2} W/K", h_tr_em);
            println!(
                "  τ = C_m / h_tr_ms = {:.2} / {:.2} = {:.2} hours ({:.1} minutes)",
                cm,
                h_tr_ms,
                tau,
                tau * 60.0
            );

            if *tau < 0.5 {
                println!(
                    "  Issue: τ too fast - thermal mass responds in {:.0} minutes",
                    tau * 60.0
                );
                println!("  Expected: 1-4 hours for proper thermal lag");
            } else if *tau > 8.0 {
                println!(
                    "  Issue: τ too slow - thermal mass takes {:.1} hours to respond",
                    tau
                );
                println!("  Expected: 1-4 hours for proper thermal lag");
            }
            println!();
        }
    }

    // Recommendations
    println!("=== Recommendations ===");
    println!("Based on τ analysis:");
    println!();
    println!("1. Check ISO 13790 A_m factor calculation");
    println!("   - Verify mass class mapping is correct for ASHRAE 140 construction");
    println!("   - Low-mass buildings should have smaller A_m (2.0-2.5)");
    println!();
    println!("2. Consider physics-based h_tr_ms calculation");
    println!("   - h_tr_ms = 1 / (R_si + R_mass + R_em)");
    println!("   - Not: h_tr_ms = 9.1 × A_m");
    println!();
    println!("3. Test with adjusted h_tr_ms values");
    println!("   - For τ = 2 hours: h_tr_ms = C_m / (2 × 3600)");
    println!("   - For τ = 4 hours: h_tr_ms = C_m / (4 × 3600)");
    println!();
    println!("4. Verify h_tr_em calculation");
    println!("   - Current h_tr_em formula: 1 / ((1/h_tr_op) - (1/h_tr_ms))");
    println!("   - This subtractive formula may be physics-incorrect");
    println!("   - Alternative: Set h_tr_em = 0.0 (mass only coupled to surface)");

    println!();
    println!("=== Task 1.1 Complete ===");
}

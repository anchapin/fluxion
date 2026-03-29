//! 6R2C Conductance Balance Parametric Study for Case 900
//!
//! Tests different combinations of:
//! - Envelope mass fraction (0.5 - 0.9)
//! - h_tr_me (envelope-to-internal conductance, 10 - 1000 W/K)
//!
//! Goal: Find optimal 6R2C configuration for Case 900 validation.

use std::env;

/// Result from a single configuration test.
#[derive(Debug, Clone)]
struct ConfigResult {
    /// Envelope mass fraction (0.5 = 50%, 0.75 = 75%)
    envelope_fraction: f64,
    /// h_tr_me value (W/K)
    h_tr_me: f64,
    /// Heating energy (MWh)
    heating_mwh: f64,
    /// Cooling energy (MWh)
    cooling_mwh: f64,
    /// Heating error (%)
    heating_error_pct: f64,
    /// Cooling error (%)
    cooling_error_pct: f64,
    /// Combined score (lower is better)
    combined_score: f64,
    /// Number of metrics within 15% error
    pass_count: u32,
}

impl ConfigResult {
    fn new(
        envelope_fraction: f64,
        h_tr_me: f64,
        heating_mwh: f64,
        cooling_mwh: f64,
        heating_ref: f64,
        cooling_ref: f64,
    ) -> Self {
        let heating_error_pct = ((heating_mwh - heating_ref) / heating_ref) * 100.0;
        let cooling_error_pct = ((cooling_mwh - cooling_ref) / cooling_ref) * 100.0;

        // Pass criteria: within 15% of reference
        let heating_pass = heating_error_pct.abs() < 15.0;
        let cooling_pass = cooling_error_pct.abs() < 15.0;
        let pass_count = if heating_pass { 1 } else { 0 } + if cooling_pass { 1 } else { 0 };

        // Combined score: minimize absolute error
        // Weighted: 2x heating error (more severe overprediction)
        let combined_score = heating_error_pct.abs() * 2.0 + cooling_error_pct.abs();

        Self {
            envelope_fraction,
            h_tr_me,
            heating_mwh,
            cooling_mwh,
            heating_error_pct,
            cooling_error_pct,
            combined_score,
            pass_count,
        }
    }
}

fn main() {
    println!("=== 6R2C Conductance Balance Parametric Study ===");
    println!("Case: ASHRAE 140 Case 900 (High Mass Baseline)");
    println!();

    // Reference values (midpoint of ASHRAE 140 ranges)
    let heating_ref = (1.17 + 2.04) / 2.0; // 1.61 MWh
    let cooling_ref = (2.13 + 3.67) / 2.0; // 2.90 MWh

    println!("Reference values:");
    println!("  Heating: {:.2} MWh (range: 1.17-2.04)", heating_ref);
    println!("  Cooling: {:.2} MWh (range: 2.13-3.67)", cooling_ref);
    println!();

    // Test parameters
    let envelope_fractions = vec![0.5, 0.6, 0.7, 0.75, 0.8, 0.9];
    let h_tr_me_values = vec![10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0];

    println!(
        "Testing {} configurations...",
        envelope_fractions.len() * h_tr_me_values.len()
    );
    println!();
    println!("┌────────────┬──────────┬─────────────┬────────┬────────┬──────────┬──────────┐");
    println!("│ Env Frac  │ h_tr_me   │ Heating MWh │ H %Err │ Cool MWh │ C %Err   │ Score    │");
    println!("├────────────┼──────────┼─────────────┼────────┼────────┼──────────┼──────────┤");

    let mut results = Vec::new();

    // Test all combinations
    for &envelope_fraction in &envelope_fractions {
        for &h_tr_me in &h_tr_me_values {
            // Parse validation output to extract results
            if let Some(result) = run_fluxion_with_config(envelope_fraction, h_tr_me) {
                let config_result = ConfigResult::new(
                    envelope_fraction,
                    h_tr_me,
                    result.heating_mwh,
                    result.cooling_mwh,
                    heating_ref,
                    cooling_ref,
                );
                println!(
                    "│ {:>8.1} │ {:>8.1} │ {:>11.3} │ {:>+6.1} │ {:>7.3} │ {:>+7.1} │ {:>8.1} │",
                    result.envelope_fraction * 100.0,
                    result.h_tr_me,
                    result.heating_mwh,
                    config_result.heating_error_pct,
                    result.cooling_mwh,
                    config_result.cooling_error_pct,
                    config_result.combined_score
                );
                results.push(config_result);
            }
        }
    }

    println!("└────────────┴──────────┴─────────────┴────────┴────────┴──────────┴──────────┘");
    println!();

    // Analysis
    analyze_results(&results, heating_ref, cooling_ref);
}

fn analyze_results(results: &[ConfigResult], heating_ref: f64, cooling_ref: f64) {
    println!("=== Analysis ===");

    // Find best result by combined score
    if let Some(best) = results
        .iter()
        .min_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap())
    {
        println!("Best configuration (by combined score):");
        println!(
            "  Envelope fraction: {:.0}%",
            best.envelope_fraction * 100.0
        );
        println!("  h_tr_me: {:.1} W/K", best.h_tr_me);
        println!(
            "  Heating: {:.3} MWh ({:.1}% error)",
            best.heating_mwh, best.heating_error_pct
        );
        println!(
            "  Cooling: {:.3} MWh ({:.1}% error)",
            best.cooling_mwh, best.cooling_error_pct
        );
        println!("  Combined score: {:.1}", best.combined_score);
        println!();
    }

    // Find best for heating
    if let Some(best_heating) = results.iter().min_by(|a, b| {
        a.heating_error_pct
            .abs()
            .partial_cmp(&b.heating_error_pct.abs())
            .unwrap()
    }) {
        println!("Best for heating (minimum absolute error):");
        println!(
            "  Envelope fraction: {:.0}%",
            best_heating.envelope_fraction * 100.0
        );
        println!("  h_tr_me: {:.1} W/K", best_heating.h_tr_me);
        println!(
            "  Heating: {:.3} MWh ({:.1}% error)",
            best_heating.heating_mwh, best_heating.heating_error_pct
        );
        println!();
    }

    // Find best for cooling
    if let Some(best_cooling) = results.iter().min_by(|a, b| {
        a.cooling_error_pct
            .abs()
            .partial_cmp(&b.cooling_error_pct.abs())
            .unwrap()
    }) {
        println!("Best for cooling (minimum absolute error):");
        println!(
            "  Envelope fraction: {:.0}%",
            best_cooling.envelope_fraction * 100.0
        );
        println!("  h_tr_me: {:.1} W/K", best_cooling.h_tr_me);
        println!(
            "  Cooling: {:.3} MWh ({:.1}% error)",
            best_cooling.cooling_mwh, best_cooling.cooling_error_pct
        );
        println!();
    }

    // Check for any passing configurations
    let passing: Vec<_> = results
        .iter()
        .filter(|r| r.pass_count >= 2)
        .cloned()
        .collect();

    if passing.is_empty() {
        println!("No configuration achieves 15% accuracy on both metrics");
    } else {
        println!("Configurations passing 15% accuracy on both metrics:");
        for r in passing {
            println!(
                "  Env Frac: {:.0}%, h_tr_me: {:.1} W/K",
                r.envelope_fraction * 100.0,
                r.h_tr_me
            );
        }
    }
    println!();

    // Sensitivity analysis by h_tr_me
    println!("=== Sensitivity Analysis by h_tr_me ===");
    for h_tr_me in [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0] {
        let results_at_h_tr_me: Vec<_> = results
            .iter()
            .filter(|r| (r.h_tr_me - h_tr_me).abs() < 0.1)
            .cloned()
            .collect();

        if let Some(best_at_h_tr) = results_at_h_tr_me
            .iter()
            .min_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap())
        {
            println!(
                "h_tr_me = {:.1} W/K: best_env_frac = {:.0}%, score = {:.1}, H = {:.3}, C = {:.3}",
                h_tr_me,
                best_at_h_tr.envelope_fraction * 100.0,
                best_at_h_tr.combined_score,
                best_at_h_tr.heating_mwh,
                best_at_h_tr.cooling_mwh
            );
        }
    }
    println!();

    // Sensitivity analysis by envelope fraction
    println!("=== Sensitivity Analysis by Envelope Fraction ===");
    for env_frac in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9] {
        let results_at_frac: Vec<_> = results
            .iter()
            .filter(|r| (r.envelope_fraction - env_frac).abs() < 0.01)
            .cloned()
            .collect();

        if let Some(best_at_frac) = results_at_frac
            .iter()
            .min_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap())
        {
            println!(
                "Env Frac = {:.0}%: best_h_tr_me = {:.1} W/K, score = {:.1}, H = {:.3}, C = {:.3}",
                env_frac * 100.0,
                best_at_frac.h_tr_me,
                best_at_frac.combined_score,
                best_at_frac.heating_mwh,
                best_at_frac.cooling_mwh
            );
        }
    }
    println!();

    // Physics interpretation
    println!("=== Physics Interpretation ===");
    println!("h_tr_me (envelope-to-internal conductance):");
    println!("  - Low h_tr_me (< 25 W/K): Envelope and internal masses thermally decoupled");
    println!("    → Each mass responds independently to its heat sources");
    println!("    → May cause temperature gradients between envelope and internal mass");
    println!("  - High h_tr_me (> 200 W/K): Envelope and internal masses tightly coupled");
    println!("    → Heat flows quickly between envelope and internal mass");
    println!("    → Behaves more like single thermal mass (similar to 5R1C)");
    println!();
    println!("Envelope fraction:");
    println!("  - Low fraction (< 0.6): Less mass in envelope (walls, roof, floor)");
    println!("    → Faster envelope response to outdoor changes");
    println!("  - High fraction (> 0.8): More mass in envelope");
    println!("    → Slower envelope response, more thermal lag");
    println!();
    println!("Current default: env_frac = 75%, h_tr_me = 100 W/K");
}

/// Run fluxion validation with custom 6R2C configuration via environment variables.
fn run_fluxion_with_config(envelope_fraction: f64, h_tr_me: f64) -> Option<ValidationResult> {
    // Set environment variables for this test
    env::set_var(
        "FLUXION_6R2C_ENVELOPE_FRACTION",
        envelope_fraction.to_string(),
    );
    env::set_var("FLUXION_6R2C_H_TR_ME", h_tr_me.to_string());

    // Run validation for Case 900 only
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "fluxion",
            "validate",
            "--case",
            "900",
        ])
        .env(
            "FLUXION_6R2C_ENVELOPE_FRACTION",
            envelope_fraction.to_string(),
        )
        .env("FLUXION_6R2C_H_TR_ME", h_tr_me.to_string())
        .output();

    let output_str: String = match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout),
        Err(_) => return None,
    };

    // Parse Case 900 results from output
    for line in output_str.lines() {
        if line.contains("Case 900:") {
            if let Some(rest) = line.strip_prefix("Case 900:") {
                return parse_case_900_line(rest, envelope_fraction, h_tr_me);
            }
        }
    }

    None
}

/// Parse Case 900 validation line to extract heating and cooling values.
fn parse_case_900_line(
    line: &str,
    envelope_fraction: f64,
    h_tr_me: f64,
) -> Option<ValidationResult> {
    // Expected format: "Case 900: Heating=X.XX (Ref: X.XX-X.XX), Cooling=X.XX (Ref: X.XX-X.XX), ..."
    // Simple string parsing instead of regex to avoid dependency
    if let Some(heating_part) = line.split("Heating=").nth(1) {
        if let Some(ref_part) = heating_part.split(')').next() {
            if let Some(heating_mwh) = ref_part.trim().parse().ok() {
                if let Some(cooling_part) = line.split("Cooling=").nth(1) {
                    if let Some(cooling_mwh) = cooling_part.split(')').next() {
                        if let Ok(c_mwh) = cooling_mwh.trim().parse() {
                            return Some(ValidationResult {
                                envelope_fraction,
                                h_tr_me,
                                heating_mwh,
                                cooling_mwh: c_mwh,
                            });
                        }
                    }
                }
            }
        }
    }

    None
}

/// Validation result extracted from fluxion output.
#[derive(Debug, Clone)]
struct ValidationResult {
    envelope_fraction: f64,
    h_tr_me: f64,
    heating_mwh: f64,
    cooling_mwh: f64,
}

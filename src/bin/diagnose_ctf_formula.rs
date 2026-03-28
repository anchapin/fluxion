#!/usr/bin/env rust-script
//! Diagnostic tool to understand CTF solver behavior
//!
//! Tests CTF flux calculation with simple temperature differences
//! to verify sign convention and magnitude

use fluxion::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};
use fluxion::physics::ctf_solver::{CTFSolver, CTFSolverConfig};

fn main() {
    println!("🔬 CTF Solver Diagnostic Tool");
    println!("{}", "=".repeat(60));

    // Define Case 900 wall construction
    let layers = vec![
        // Interior to exterior
        CTFMaterial::new("Gypsum Board", 0.013, 0.16, 800.0, 1090.0),
        CTFMaterial::new("Concrete", 0.100, 0.51, 2240.0, 920.0),
        CTFMaterial::new("Foam Insulation", 0.0615, 0.04, 30.0, 840.0),
        CTFMaterial::new("Wood Siding", 0.009, 0.16, 500.0, 2090.0),
    ];

    // Calculate expected U-value
    let r_total: f64 = layers.iter().map(|l| l.thickness / l.conductivity).sum();
    let r_si = 0.125; // Interior film
    let r_se = 0.044; // Exterior film
    let u_value = 1.0 / (r_si + r_total + r_se);

    println!("\n📊 Wall Properties:");
    println!("  Total resistance: {:.3} m²K/W", r_total);
    println!("  U-value: {:.3} W/m²K", u_value);
    println!("  Time constant: {:.1} hours", r_total * 2240.0 * 920.0 / 3600.0);

    // Compute CTF coefficients
    let coeffs = CTFCalculator::with_defaults(&layers, 3600.0).compute_coefficients();

    println!("\n🔢 CTF Coefficients:");
    println!("  X[0] = {:.6} (should ≈ U-value)", coeffs.x[0]);
    println!("  Y[0] = {:.6} (should ≈ U-value)", coeffs.y[0]);
    println!("  Z[0] = {:.6} (should ≈ U-value)", coeffs.z[0]);
    println!("  Sum X = {:.6} (should = {:.6})", coeffs.x.iter().sum::<f64>(), u_value);
    println!("  Sum Y = {:.6} (should = {:.6})", coeffs.y.iter().sum::<f64>(), u_value);

    // Create solver
    let config = CTFSolverConfig::new(3600.0, 50);
    let mut solver = CTFSolver::new(coeffs.clone(), config);

    println!("\n🧪 Test 1: Steady-state heat loss (winter)");
    println!("  T_interior = 20°C");
    println!("  T_exterior = 0°C");
    println!("  ΔT = 20°C");

    // Run several timesteps to reach steady state
    let mut flux_sum = 0.0;
    for i in 0..10 {
        let q = solver.step(20.0, 0.0);
        flux_sum += q;
        println!("  Timestep {}: Q = {:.2} W/m²", i, q);
    }
    let avg_flux = flux_sum / 10.0;
    let expected_flux = u_value * (0.0 - 20.0); // Heat loss, should be negative

    println!("\n  Average flux: {:.2} W/m²", avg_flux);
    println!("  Expected flux (U·ΔT): {:.2} W/m²", expected_flux);
    println!("  Difference: {:.2} W/m² ({:.1}%)",
        avg_flux - expected_flux,
        100.0 * (avg_flux - expected_flux).abs() / expected_flux.abs());

    // Check sign
    if avg_flux < 0.0 {
        println!("  ✅ Sign correct: Negative flux (heat loss)");
    } else {
        println!("  ❌ Sign wrong: Positive flux (heat gain during winter!)");
    }

    println!("\n🧪 Test 2: Steady-state heat gain (summer)");
    println!("  T_interior = 20°C");
    println!("  T_exterior = 35°C");
    println!("  ΔT = 15°C");

    // Reset solver
    solver.reset(20.0);

    flux_sum = 0.0;
    for i in 0..10 {
        let q = solver.step(20.0, 35.0);
        flux_sum += q;
        println!("  Timestep {}: Q = {:.2} W/m²", i, q);
    }
    let avg_flux = flux_sum / 10.0;
    let expected_flux = u_value * (35.0 - 20.0); // Heat gain, should be positive

    println!("\n  Average flux: {:.2} W/m²", avg_flux);
    println!("  Expected flux (U·ΔT): {:.2} W/m²", expected_flux);
    println!("  Difference: {:.2} W/m² ({:.1}%)",
        avg_flux - expected_flux,
        100.0 * (avg_flux - expected_flux).abs() / expected_flux.abs());

    // Check sign
    if avg_flux > 0.0 {
        println!("  ✅ Sign correct: Positive flux (heat gain)");
    } else {
        println!("  ❌ Sign wrong: Negative flux (heat loss during summer!)");
    }

    println!("\n🧪 Test 3: Zero temperature difference");
    println!("  T_interior = 20°C");
    println!("  T_exterior = 20°C");
    println!("  ΔT = 0°C");

    solver.reset(20.0);

    for i in 0..5 {
        let q = solver.step(20.0, 20.0);
        println!("  Timestep {}: Q = {:.4} W/m²", i, q);
    }

    println!("\n📋 Summary:");
    println!("  If Test 1 shows positive flux or Test 2 shows negative flux,");
    println!("  the sign convention is WRONG and needs to be flipped.");
    println!("  If Test 3 shows non-zero flux, there's a DC offset issue.");
    println!("  If magnitudes are far from expected, coefficients are wrong.");
}

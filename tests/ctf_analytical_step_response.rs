//! FD vs Analytical Solution: 1D Transient Heat Conduction Through Concrete Wall
//!
//! Issue #1098: Component Test for 1D Conduction Step Response
//!
//! This test validates that the FD solver produces correct 1D transient
//! heat conduction through a high-mass concrete wall by comparing against the
//! analytical solution.
//!
//! # Problem Setup
//!
//! **Wall**: 200mm concrete slab, 3m × 3m surface
//! - Conductivity k = 1.75 W/(m·K)
//! - Density ρ = 2300 kg/m³
//! - Specific Heat c_p = 880 J/(kg·K)
//! - Thermal diffusivity α = k/(ρ·c_p) ≈ 8.68×10⁻⁷ m²/s
//!
//! **Boundary Conditions**:
//! - Initial temperature: 20°C throughout wall
//! - Exterior surface (x=L): step to 35°C at t=0
//! - Interior surface (x=0): insulated (zero flux)
//!
//! **Duration**: 24 hours
//!
//! # Analytical Solution
//!
//! For a finite slab with insulated at x=0 and step temperature at x=L:
//! - Heat enters at x=L but cannot exit at x=0 (insulated)
//! - Heat accumulates in the wall, raising its temperature
//! - At steady state: wall reaches uniform 35°C, flux = 0
//!
//! Temperature at interior surface x=0:
//! T(0,t) = T_s - 2·dT·Σ[(-1)^n/((n+½)π)·exp(-α·(n+½)²π²t/L²)]
//!
//! Heat flux at exterior surface x=L (into wall, negative):
//! q(L,t) = -2·k·dT/L·Σ[exp(-α·(n+½)²π²t/L²)]
//!
//! Reference: Carslaw & Jaeger (1959), Chapter 2

use fluxion::physics::fd_discretization::{MaterialLayer, WallDiscretization};
use fluxion::physics::fd_solver::{ImplicitFDSolver, SurfaceBC};
use std::f64::consts::PI;

/// Material properties for concrete (from issue spec)
const K: f64 = 1.75; // W/(m·K)
const RHO: f64 = 2300.0; // kg/m³
const CP: f64 = 880.0; // J/(kg·K)
const L: f64 = 0.2; // m (200mm wall thickness)

/// Thermal diffusivity α = k/(ρ·c_p) [m²/s]
const ALPHA: f64 = K / (RHO * CP);

/// Initial wall temperature [°C]
const T_INITIAL: f64 = 20.0;

/// Exterior step temperature [°C]
const T_EXTERIOR: f64 = 35.0;

/// Temperature difference [K]
const DT: f64 = T_EXTERIOR - T_INITIAL;

/// Simulation timestep [s]
const TIMESTEP: f64 = 3600.0;

/// Number of timesteps in 24 hours
const NUM_STEPS: usize = 24;

/// Number of FD nodes per layer
const FD_NODES: usize = 40;

/// Maximum relative error tolerance (5% for FD discretization)
const REL_TOL: f64 = 0.05;

/// Maximum absolute error tolerance [W/m²]
const ABS_TOL: f64 = 10.0;

/// Correct analytical solution for heat flux at x=L (exterior surface).
///
/// For a slab with:
/// - Insulated at x=0 (∂T/∂x = 0)
/// - Step temperature at x=L (T(L,t) = T_s)
///
/// Heat flux at x=L (into wall, negative):
/// q(L,t) = -2·k·dT/L·Σ exp(-α·(n+½)²π²t/L²)
///
/// This is NEGATIVE because heat flows INTO the wall.
fn analytical_flux(t_hours: f64, num_terms: usize) -> f64 {
    let t_seconds = t_hours * 3600.0;
    let mut series_sum = 0.0_f64;

    for n in 0..num_terms {
        let lam = (n as f64 + 0.5) * PI / L; // λ_n = (n+½)π/L
        let exp_term = (-ALPHA * lam.powi(2) * t_seconds).exp();
        series_sum += exp_term;
    }

    -2.0 * K * DT / L * series_sum
}

/// Correct analytical solution for temperature at interior surface x=0.
///
/// T(0,t) = T_s - 2·dT·Σ[(-1)^n/((n+½)π)·exp(-α·(n+½)²π²t/L²)]
fn analytical_interior_temp(t_hours: f64, num_terms: usize) -> f64 {
    let t_seconds = t_hours * 3600.0;
    let mut series_sum = 0.0_f64;

    for n in 0..num_terms {
        let n_f = n as f64;
        let lam = (n_f + 0.5) * PI / L; // λ_n = (n+½)π/L
        let exp_term = (-ALPHA * lam.powi(2) * t_seconds).exp();
        series_sum += (-1.0_f64).powi(n as i32) / ((n_f + 0.5) * PI) * exp_term;
    }

    T_EXTERIOR - 2.0 * DT * series_sum
}

/// Run FD solver for the step response problem.
fn run_fd_solver() -> Vec<(f64, f64, f64)> {
    // Create material layer
    let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
    let disc = WallDiscretization::from_layers(&layers, FD_NODES);

    // Create FD solver initialized at T_INITIAL
    let mut solver = ImplicitFDSolver::new(disc, T_INITIAL);

    // Boundary conditions
    // Exterior: high h to approximate Dirichlet (T_surface = T_exterior)
    let h_bc = 1e9;
    // Interior: insulated (zero flux) - low h gives q ≈ 0
    let interior_bc = SurfaceBC::new_interior(1e-10, T_INITIAL);
    let exterior_bc = SurfaceBC::new_exterior(h_bc, T_EXTERIOR, 0.0);

    let k_fd = solver.discretization.conductivity[0];

    let mut results = Vec::new();

    // Use sub-stepping for FD to get accurate results
    let fd_timestep = 300.0; // 5-minute timesteps
    let steps_per_hour = (TIMESTEP / fd_timestep) as usize;

    for hour in 0..NUM_STEPS {
        for _ in 0..steps_per_hour {
            solver.step(fd_timestep, &interior_bc, &exterior_bc);
        }

        let elapsed_hours = (hour + 1) as f64;

        // Heat flux at exterior surface (node at x=L, last node)
        // q = -k * ∂T/∂x ≈ -k * (T[last] - T[second_last]) / dx
        // This gives negative value (heat INTO wall)
        let n = solver.temperatures.len();
        let dx = solver.discretization.node_volumes[n - 1];
        let t_exterior_surface = solver.temperatures[n - 1];
        let t_next_to_surface = solver.temperatures[n - 2];
        let q_flux = -k_fd * (t_exterior_surface - t_next_to_surface) / dx;

        // Interior temperature (at x=0, first node)
        let t_interior = solver.temperatures[0];

        results.push((elapsed_hours, q_flux, t_interior));
    }

    results
}

#[test]
fn test_fd_analytical_step_response() {
    println!("\n{}", "=".repeat(70));
    println!("FD Solver vs Analytical Step Response Test");
    println!("Issue #1098: 1D Conduction Step Response CTF vs Analytical");
    println!("{}", "=".repeat(70));

    println!("\n--- Problem Parameters ---");
    println!("Wall thickness: {} m", L);
    println!(
        "k = {} W/(m·K), ρ = {} kg/m³, c_p = {} J/(kg·K)",
        K, RHO, CP
    );
    println!("α = {:.6e} m²/s", ALPHA);
    println!("Initial temperature: {} °C", T_INITIAL);
    println!("Exterior step temperature: {} °C", T_EXTERIOR);
    println!("Temperature step: {} K", DT);
    println!("Interior surface: INSULATED (zero flux)");
    println!("");
    println!("Physical interpretation:");
    println!("- Heat enters at x=L (exterior) but cannot exit at x=0 (interior)");
    println!("- Heat accumulates in wall, raising its temperature");
    println!(
        "- At steady state: wall reaches uniform {} °C, flux = 0",
        T_EXTERIOR
    );

    // Run FD solver
    let fd_results = run_fd_solver();

    println!("\n--- FD vs Analytical Checkpoint Comparison ---");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "Time", "Analytical", "FD Solver", "Abs Error", "% Error"
    );
    println!("{}", "-".repeat(60));

    let mut all_passed = true;

    for &(hour, q_fd, _t_int_fd) in &fd_results {
        let q_analytical = analytical_flux(hour, 100);
        let diff = (q_fd - q_analytical).abs();
        let pct_error = if q_analytical.abs() > 1e-6 {
            diff / q_analytical.abs() * 100.0
        } else {
            0.0
        };

        let status = if diff <= ABS_TOL || pct_error <= REL_TOL * 100.0 {
            "PASS"
        } else {
            "FAIL"
        };

        println!(
            "{:>7.1}h {:>12.4} {:>12.4} {:>12.4} {:>9.2}% {}",
            hour, q_analytical, q_fd, diff, pct_error, status
        );

        if diff > ABS_TOL && pct_error > REL_TOL * 100.0 {
            all_passed = false;
        }
    }

    println!("\n{}", "=".repeat(70));
    if all_passed {
        println!("RESULT: PASS - FD solver matches analytical to within 5% tolerance");
    } else {
        println!("RESULT: FAIL - See above for details");
    }
    println!("{}", "=".repeat(70));

    assert!(
        all_passed,
        "FD solver step response did not match analytical solution"
    );
}

#[test]
fn test_fd_interior_temperature() {
    //! Verify FD solver interior temperature matches analytical solution.

    println!("\n{}", "=".repeat(70));
    println!("FD Solver Interior Temperature Check");
    println!("{}", "=".repeat(70));

    let fd_results = run_fd_solver();

    println!("\n--- Interior Temperature at x=0 (Insulated Surface) ---");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "Time", "Analytical", "FD Solver", "Abs Error", "% Error"
    );
    println!("{}", "-".repeat(60));

    let mut all_passed = true;

    for &(hour, _q_fd, t_int_fd) in &fd_results {
        let t_analytical = analytical_interior_temp(hour, 100);
        let diff = (t_int_fd - t_analytical).abs();
        let pct_error = if t_analytical.abs() > 1e-10 {
            diff / t_analytical.abs() * 100.0
        } else {
            0.0
        };

        let status = if diff < 0.5 || pct_error <= 5.0 {
            "PASS"
        } else {
            "FAIL"
        };

        println!(
            "{:>7.1}h {:>12.4} {:>12.4} {:>12.4} {:>9.2}% {}",
            hour, t_analytical, t_int_fd, diff, pct_error, status
        );

        if diff >= 0.5 && pct_error > 5.0 {
            all_passed = false;
        }
    }

    println!("\n{}", "=".repeat(70));
    if all_passed {
        println!("RESULT: PASS - Interior temperature matches analytical");
    } else {
        println!("RESULT: FAIL - Interior temperature deviation exceeds tolerance");
    }
    println!("{}", "=".repeat(70));

    assert!(
        all_passed,
        "FD solver interior temperature does not match analytical"
    );
}

#[test]
fn test_fd_steady_state_convergence() {
    //! Verify FD solver converges to correct steady state (uniform T, zero flux).

    println!("\n{}", "=".repeat(70));
    println!("FD Solver Steady-State Convergence Check");
    println!("{}", "=".repeat(70));

    // Create FD discretization
    let layers = vec![MaterialLayer::new("Concrete", L, K, RHO, CP)];
    let disc = WallDiscretization::from_layers(&layers, FD_NODES);
    let mut solver = ImplicitFDSolver::new(disc, T_INITIAL);

    let h_bc = 1e9;
    let interior_bc = SurfaceBC::new_interior(1e-10, T_INITIAL);
    let exterior_bc = SurfaceBC::new_exterior(h_bc, T_EXTERIOR, 0.0);

    let k_fd = solver.discretization.conductivity[0];

    // Run for extended period
    let fd_timestep = 300.0;
    let total_steps = (200.0 * 3600.0 / fd_timestep) as usize;

    println!("\nRunning FD solver for 200 hours...");

    for _ in 0..total_steps {
        solver.step(fd_timestep, &interior_bc, &exterior_bc);
    }

    // Check final state
    let n = solver.temperatures.len();
    let t_surface = solver.temperatures[0]; // Interior
    let t_exterior = solver.temperatures[n - 1]; // Exterior

    // Compute flux
    let dx = solver.discretization.node_volumes[n - 1];
    let t_next = solver.temperatures[n - 2];
    let q_flux = -k_fd * (t_exterior - t_next) / dx;

    println!("\n--- Steady State Results (t=200h) ---");
    println!("Interior temperature T(0): {:.6} °C", t_surface);
    println!("Exterior temperature T(L): {:.6} °C", t_exterior);
    println!("Heat flux q: {:.6} W/m²", q_flux);
    println!("Expected: uniform {:.1} °C, q ≈ 0", T_EXTERIOR);

    println!("\n--- Temperature Profile ---");
    for i in (0..n).step_by(n / 5) {
        let x = solver.discretization.node_positions[i];
        let t = solver.temperatures[i];
        println!("  x={:.3}m: T={:.4}°C", x, t);
    }

    println!("\n{}", "=".repeat(70));

    let temp_uniform =
        (t_surface - T_EXTERIOR).abs() < 0.5 && (t_exterior - T_EXTERIOR).abs() < 0.5;
    let flux_near_zero = q_flux.abs() < 1.0;

    if temp_uniform && flux_near_zero {
        println!(
            "RESULT: PASS - Wall converged to uniform {:.1}°C, q ≈ 0",
            T_EXTERIOR
        );
    } else {
        println!("RESULT: FAIL - Wall did not converge as expected");
        println!(
            "  Temperature uniform: {}",
            if temp_uniform { "YES" } else { "NO" }
        );
        println!(
            "  Flux near zero: {}",
            if flux_near_zero { "YES" } else { "NO" }
        );
    }
    println!("{}", "=".repeat(70));

    assert!(temp_uniform, "Wall should reach uniform temperature");
    assert!(flux_near_zero, "Flux should approach zero at steady state");
}

#[test]
fn test_fd_monotonic_temperature_rise() {
    //! Verify interior temperature rises monotonically (heat accumulates).

    println!("\n{}", "=".repeat(70));
    println!("FD Solver Monotonic Temperature Rise Check");
    println!("{}", "=".repeat(70));

    let fd_results = run_fd_solver();

    println!("\n--- Interior Temperature Rise (Heat Accumulation) ---");
    println!("{:>8} {:>12}", "Time", "T_int (°C)");
    println!("{}", "-".repeat(25));

    let mut prev_temp = T_INITIAL;
    let mut is_monotonic = true;

    for &(hour, _q_fd, t_int) in &fd_results {
        println!("{:>7.1}h {:>12.4}", hour, t_int);

        if t_int < prev_temp {
            is_monotonic = false;
        }
        prev_temp = t_int;
    }

    println!("\n--- Heat Flux Decay (Entering Wall) ---");
    println!("{:>8} {:>12}", "Time", "Flux (W/m²)");
    println!("{}", "-".repeat(25));

    let mut prev_flux = 0.0_f64;
    let mut flux_is_decaying = true;

    for &(hour, q_fd, _t_int) in &fd_results {
        println!("{:>7.1}h {:>12.4}", hour, q_fd);

        if prev_flux != 0.0 && q_fd < prev_flux {
            flux_is_decaying = false;
        }
        prev_flux = q_fd;
    }

    println!("\n{}", "=".repeat(70));
    if is_monotonic && flux_is_decaying {
        println!("RESULT: PASS - Temperature rises and flux decays monotonically");
    } else {
        println!("RESULT: FAIL");
        println!(
            "  Monotonic temperature rise: {}",
            if is_monotonic { "YES" } else { "NO" }
        );
        println!(
            "  Flux decaying: {}",
            if flux_is_decaying { "YES" } else { "NO" }
        );
    }
    println!("{}", "=".repeat(70));

    assert!(
        is_monotonic,
        "Interior temperature should rise monotonically"
    );
    assert!(
        flux_is_decaying,
        "Flux magnitude should decay monotonically"
    );
}

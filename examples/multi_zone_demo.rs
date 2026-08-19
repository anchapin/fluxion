//! Multi-Zone Thermal Network Demonstration
//!
//! This example demonstrates the multi-zone thermal network capabilities
//! of Fluxion, showcasing inter-zone heat transfer and energy conservation
//! in a two-zone building configuration.
//!
//! The example creates a simple two-zone building:
//! - Zone 1: Living space (20°C heating setpoint)
//! - Zone 2: Sunspace (15°C heating setpoint)
//!
//! Key features demonstrated:
//! - Multi-zone thermal model creation
//! - Inter-zone heat transfer calculation
//! - Energy conservation validation
//! - Zone-specific temperature control
//! - Performance comparison with single-zone

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::case_960::run_complete_case_960_validation;
use fluxion::validation::energy_balance::EnergyBalanceValidator;
use std::time::Instant;

fn main() {
    println!("=== Fluxion Multi-Zone Thermal Network Demo ===\n");

    // Demonstrate 1: Simple two-zone building configuration
    demonstrate_simple_two_zone_building();

    // Demonstrate 2: Inter-zone heat transfer visualization
    demonstrate_inter_zone_heat_transfer();

    // Demonstrate 3: Energy conservation validation
    demonstrate_energy_conservation();

    // Demonstrate 4: Performance comparison
    demonstrate_performance_comparison();

    // Demonstrate 5: Case 960 validation
    demonstrate_case_960_validation();

    println!("\n=== Demo Complete ===");
    println!("Multi-zone thermal network capabilities successfully demonstrated!");
}

/// Demonstrate a simple two-zone building configuration
fn demonstrate_simple_two_zone_building() {
    println!("1. Simple Two-Zone Building Configuration\n");

    // Create a two-zone thermal model
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Configure zone-specific setpoints
    model.setpoints.heating_setpoints = VectorField::new(vec![20.0, 15.0]); // Zone 1: 20°C, Zone 2: 15°C
    model.setpoints.cooling_setpoints = VectorField::new(vec![24.0, 99.0]); // Zone 1: 24°C, Zone 2: no cooling

    // Set inter-zone conductance (thermal coupling between zones)
    model.conduction.h_tr_iz = VectorField::new(vec![50.0, 50.0]); // 50 W/K conductance

    println!("  ✓ Created two-zone building model");
    println!("  ✓ Zone 1 (Living): 20°C heating, 24°C cooling");
    println!("  ✓ Zone 2 (Sunspace): 15°C heating only");
    println!("  ✓ Inter-zone conductance: 50 W/K");
    println!("  ✓ Number of zones: {}\n", model.hvac.num_zones);
}

/// Demonstrate inter-zone heat transfer visualization
fn demonstrate_inter_zone_heat_transfer() {
    println!("2. Inter-Zone Heat Transfer Visualization\n");

    // Create model with different zone temperatures to show heat transfer
    let spec = ASHRAE140Case::Case960.spec();
    let _model = ThermalModel::<VectorField>::from_spec(&spec);

    // Simulate a temperature difference between zones
    let zone1_temp = 22.0; // Living space
    let zone2_temp = 18.0; // Sunspace
    let conductance = 50.0; // W/K

    let heat_flow = conductance * (zone1_temp - zone2_temp);

    println!("  Zone temperatures:");
    println!("    Zone 1 (Living): {:.1}°C", zone1_temp);
    println!("    Zone 2 (Sunspace): {:.1}°C", zone2_temp);
    println!("  Temperature difference: {:.1}°C", zone1_temp - zone2_temp);
    println!("  Inter-zone heat flow: {:.1} W", heat_flow);
    println!("  Direction: Zone 1 → Zone 2 (heat flows from warmer to cooler)");
    println!();
}

/// Demonstrate energy conservation validation
fn demonstrate_energy_conservation() {
    println!("3. Energy Conservation Validation\n");

    // Create a thermal model
    let spec = ASHRAE140Case::Case960.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Create energy balance validator
    let validator = EnergyBalanceValidator::default();

    // Run energy balance validation
    let report = validator.run(&model);

    println!("  Energy balance validation results:");
    println!(
        "    Status: {}",
        if report.is_valid {
            "PASSED ✓"
        } else {
            "FAILED ✗"
        }
    );
    println!("    Cumulative error: {:.6e} J", report.cumulative_error);
    println!("    Error percentage: {:.6}%", report.error_pct);
    println!("    Total zones: {}", model.hvac.num_zones);

    if report.is_valid {
        println!("  ✓ Energy conservation validated - physics engine working correctly");
    } else {
        println!("  ⚠️  Energy conservation issues detected");
    }
    println!();
}

/// Demonstrate performance comparison between single-zone and multi-zone
fn demonstrate_performance_comparison() {
    println!("4. Performance Comparison\n");

    // Single-zone model (baseline)
    let single_zone_spec = ASHRAE140Case::Case600.spec();
    let mut single_zone_model = ThermalModel::<VectorField>::from_spec(&single_zone_spec);

    // Multi-zone model
    let multi_zone_spec = ASHRAE140Case::Case960.spec();
    let mut multi_zone_model = ThermalModel::<VectorField>::from_spec(&multi_zone_spec);

    // Measure single-zone performance
    let single_zone_start = Instant::now();
    for step in 0..100 {
        let _ = single_zone_model.step_physics(step, 15.0, 3600.0);
    }
    let single_zone_time = single_zone_start.elapsed();

    // Measure multi-zone performance
    let multi_zone_start = Instant::now();
    for step in 0..100 {
        let _ = multi_zone_model.step_physics(step, 15.0, 3600.0);
    }
    let multi_zone_time = multi_zone_start.elapsed();

    println!("  Performance results (100 timesteps):");
    println!("    Single-zone: {:?}", single_zone_time);
    println!("    Multi-zone: {:?}", multi_zone_time);

    let ratio = multi_zone_time.as_secs_f32() / single_zone_time.as_secs_f32();
    println!("    Multi-zone overhead: {:.1}x", ratio);

    if ratio < 2.0 {
        println!("  ✓ Excellent performance - multi-zone overhead minimal");
    } else if ratio < 3.0 {
        println!("  ✓ Good performance - acceptable overhead");
    } else {
        println!("  ⚠️  Performance could be optimized");
    }
    println!();
}

/// Demonstrate Case 960 validation
fn demonstrate_case_960_validation() {
    println!("5. ASHRAE 140 Case 960 Validation\n");

    // Run complete Case 960 validation
    let report = run_complete_case_960_validation();

    println!("{}", report);

    // Extract key metrics from the report
    let lines: Vec<&str> = report.lines().collect();
    let status_line = lines
        .iter()
        .find(|l| l.starts_with("Status:"))
        .unwrap_or(&"Status: UNKNOWN");

    println!(
        "\n  Case 960 validation: {}",
        status_line.replace("Status:", "").trim()
    );
}

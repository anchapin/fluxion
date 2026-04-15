//! Multi-zone thermal model tests.
//!
//! This module contains comprehensive tests for multi-zone energy conservation,
//! inter-zone heat transfer, and performance validation.

use crate::thermal::{coupled_solver, inter_zone, thermal_model::ThermalModel};

#[test]
fn test_symmetric_zones_equalize() {
    // Test that two identical zones with different initial temperatures
    // reach the same temperature over time due to inter-zone heat transfer

    let mut model = ThermalModel::new(2, 0.0); // Start with 0°C

    // Set different initial temperatures
    model.set_temperatures(vec![30.0, 10.0]); // Zone 0: 30°C, Zone 1: 10°C
    model.set_inter_zone_conductance(vec![50.0, 50.0]); // Symmetric conductance

    // Set equal thermal capacitances
    model.set_thermal_capacitances(vec![1000.0, 1000.0]);

    // Simulate for several time steps
    let c = model.get_thermal_capacitances();
    let h_tr_iz = model.get_inter_zone_conductance();
    let q = vec![0.0, 0.0]; // No external heat
    let dt = 3600.0; // 1 hour

    let mut temps = model.get_temperatures();

    // Run simulation
    for _ in 0..10 {
        let new_temps = coupled_solver::solve_coupled_system(&c, &h_tr_iz, &q, dt, &temps);
        temps = new_temps;
    }

    // Temperatures should be moving toward each other
    let temp_diff = (temps[0] - temps[1]).abs();
    assert!(
        temp_diff < 15.0,
        "Temperatures should converge, diff: {}",
        temp_diff
    );
}

#[test]
fn test_energy_conservation() {
    // Test that total energy is conserved in an isolated multi-zone system

    let mut model = ThermalModel::new(3, 20.0);

    // Set different temperatures but same capacitances
    model.set_temperatures(vec![25.0, 20.0, 15.0]);
    model.set_thermal_capacitances(vec![1000.0, 1000.0, 1000.0]);
    model.set_inter_zone_conductance(vec![50.0, 50.0, 50.0]);

    let c = model.get_thermal_capacitances();
    let h_tr_iz = model.get_inter_zone_conductance();
    let q = vec![0.0, 0.0, 0.0]; // No external heat
    let dt = 3600.0;

    let initial_temps = model.get_temperatures();

    // Calculate initial total energy
    let initial_energy: f64 = c
        .iter()
        .zip(initial_temps.iter())
        .map(|(cap, temp)| cap * temp)
        .sum();

    // Run simulation
    let mut temps = initial_temps;
    for _ in 0..5 {
        let new_temps = coupled_solver::solve_coupled_system(&c, &h_tr_iz, &q, dt, &temps);
        temps = new_temps;
    }

    // Calculate final total energy
    let final_energy: f64 = c
        .iter()
        .zip(temps.iter())
        .map(|(cap, temp)| cap * temp)
        .sum();

    // Energy should be conserved (within numerical tolerance)
    let energy_diff = (initial_energy - final_energy).abs();
    assert!(
        energy_diff < 1.0,
        "Energy conservation violated: diff = {}",
        energy_diff
    );
}

#[test]
fn test_inter_zone_sign_convention() {
    // Test that inter-zone heat flow follows Q_ij = -Q_ji convention

    let h_tr_ij = 50.0; // W/K
    let ti = 25.0; // °C
    let tj = 20.0; // °C

    let q_ij = inter_zone::inter_zone_heat_flow(h_tr_ij, ti, tj);
    let q_ji = inter_zone::inter_zone_heat_flow(h_tr_ij, tj, ti);

    // Verify sign convention
    assert_eq!(
        q_ij, -q_ji,
        "Sign convention violated: Q_ij should equal -Q_ji"
    );
    assert!(q_ij > 0.0, "Heat should flow from warmer to cooler zone");
    assert!(q_ji < 0.0, "Reverse direction should be negative");
}

#[test]
fn test_performance_regression() {
    // Test that multi-zone performance is within acceptable bounds
    // compared to single-zone simulation

    let num_zones = 10;
    let mut model = ThermalModel::new(num_zones, 20.0);

    // Set up a realistic scenario
    model.set_thermal_capacitances(vec![1000000.0; num_zones]);
    model.set_inter_zone_conductance(vec![100.0; num_zones]);

    let c = model.get_thermal_capacitances();
    let h_tr_iz = model.get_inter_zone_conductance();
    let q = vec![1000.0; num_zones]; // Some heat gain
    let dt = 3600.0;

    let initial_temps = model.get_temperatures();

    // Measure execution time
    let start_time = std::time::Instant::now();

    // Run multiple time steps
    let mut temps = initial_temps;
    for _ in 0..100 {
        let new_temps = coupled_solver::solve_coupled_system(&c, &h_tr_iz, &q, dt, &temps);
        temps = new_temps;
    }

    let duration = start_time.elapsed();

    // Performance should be reasonable for N=10 zones
    // This is a placeholder - in practice, we'd compare against baseline
    assert!(
        duration.as_secs() < 10,
        "Multi-zone simulation too slow: {:?}",
        duration
    );
}

#[test]
fn test_single_zone_compatibility() {
    // Test that multi-zone model works correctly with single zone

    let mut model = ThermalModel::new(1, 20.0);
    model.set_thermal_capacitances(vec![1000000.0]);
    model.set_inter_zone_conductance(vec![0.0]); // No inter-zone coupling

    let c = model.get_thermal_capacitances();
    let h_tr_iz = model.get_inter_zone_conductance();
    let q = vec![1000.0]; // 1 kW heat gain
    let dt = 3600.0;

    let initial_temp = model.get_temperatures()[0];
    let new_temps =
        coupled_solver::solve_coupled_system(&c, &h_tr_iz, &q, dt, &model.get_temperatures());
    let final_temp = new_temps[0];

    // Temperature should increase due to heat gain
    assert!(
        final_temp > initial_temp,
        "Temperature should increase with heat gain"
    );

    // Check that the change is reasonable
    let expected_change = (q[0] / c[0]) * dt; // ΔT = Q*dt/C
    let actual_change = final_temp - initial_temp;
    assert!(
        (actual_change - expected_change).abs() < 1.0,
        "Temperature change should match expected: expected {}, got {}",
        expected_change,
        actual_change
    );
}

#[test]
fn test_thermal_model_initialization() {
    // Test that ThermalModel initializes correctly

    let model = ThermalModel::new(5, 20.0);
    assert_eq!(model.num_zones, 5);

    let temps = model.get_temperatures();
    assert_eq!(temps.len(), 5);
    assert!(temps.iter().all(|&t| t == 20.0));

    let capacitances = model.get_thermal_capacitances();
    assert_eq!(capacitances.len(), 5);
    assert!(capacitances.iter().all(|&c| c == 1000000.0));
}

#[test]
fn test_inter_zone_conductance_calculation() {
    // Test inter-zone conductance calculation

    let area = 20.0; // m²
    let u_value = 0.5; // W/m²·K

    let conductance = inter_zone::calculate_inter_zone_conductance(area, u_value);
    assert_eq!(conductance, 10.0); // 20 * 0.5 = 10 W/K

    // Test with zero area
    let zero_area_conductance = inter_zone::calculate_inter_zone_conductance(0.0, u_value);
    assert_eq!(zero_area_conductance, 0.0);

    // Test with zero U-value
    let zero_u_conductance = inter_zone::calculate_inter_zone_conductance(area, 0.0);
    assert_eq!(zero_u_conductance, 0.0);
}

#[test]
fn test_coupled_solver_stability() {
    // Test that coupled solver produces stable results

    let num_zones = 3;
    let mut model = ThermalModel::new(num_zones, 20.0);

    // Set up a stable configuration
    model.set_thermal_capacitances(vec![1000000.0; num_zones]);
    model.set_inter_zone_conductance(vec![50.0; num_zones]);

    let c = model.get_thermal_capacitances();
    let h_tr_iz = model.get_inter_zone_conductance();
    let q = vec![0.0; num_zones]; // No external heat
    let dt = 3600.0;

    let mut temps = model.get_temperatures();

    // Run many time steps and check for stability
    for _ in 0..100 {
        let new_temps = coupled_solver::solve_coupled_system(&c, &h_tr_iz, &q, dt, &temps);

        // Check that temperatures are finite and reasonable
        for &temp in &new_temps {
            assert!(temp.is_finite(), "Temperature became non-finite");
            assert!(
                temp > -100.0 && temp < 100.0,
                "Temperature out of reasonable range: {}",
                temp
            );
        }

        temps = new_temps;
    }
}

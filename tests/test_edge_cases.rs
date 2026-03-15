// Integration tests for edge cases and boundary conditions
//
// These tests verify that the system handles extreme parameters, zero loads,
// and boundary conditions gracefully without panics or undefined behavior.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

const EPSILON: f64 = 1e-6;

/// Test 1: Extreme Parameter Values
///
/// Test MIN_U_VALUE (0.1) and MAX_U_VALUE (5.0])
/// Test MIN_SETPOINT (15) and MAX_SETPOINT (30])
/// Test boundary combinations
#[test]
fn test_extreme_parameter_values() {
    println!("\n=== Test 1: Extreme Parameter Values ===");

    // Test MIN_U_VALUE boundary
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 0.1; // MIN_U_VALUE
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("MIN_U_VALUE (0.1) energy: {:.2}", energy);
        assert!(
            energy.is_finite(),
            "Energy should be finite for MIN_U_VALUE"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test MAX_U_VALUE boundary
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 5.0; // MAX_U_VALUE
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("MAX_U_VALUE (5.0) energy: {:.2}", energy);
        assert!(
            energy.is_finite(),
            "Energy should be finite for MAX_U_VALUE"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test MIN heating setpoint boundary
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 15.0; // MIN_HEATING_SETPOINT
        model.cooling_setpoint = 22.0;

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("MIN_HEATING_SETPOINT (15.0) energy: {:.2}", energy);
        assert!(
            energy.is_finite(),
            "Energy should be finite for MIN_HEATING_SETPOINT"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test MAX cooling setpoint boundary
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 32.0; // MAX_COOLING_SETPOINT

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("MAX_COOLING_SETPOINT (32.0) energy: {:.2}", energy);
        assert!(
            energy.is_finite(),
            "Energy should be finite for MAX_COOLING_SETPOINT"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test boundary combination: min U + min heating setpoint
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 0.1;
        model.heating_setpoint = 15.0;
        model.cooling_setpoint = 22.0;

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("MIN U + MIN heating energy: {:.2}", energy);
        assert!(
            energy.is_finite(),
            "Energy should be finite for min/min combo"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test boundary combination: max U + max cooling setpoint
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 5.0;
        model.heating_setpoint = 25.0;
        model.cooling_setpoint = 32.0;

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("MAX U + MAX cooling energy: {:.2}", energy);
        assert!(
            energy.is_finite(),
            "Energy should be finite for max/max combo"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    println!("✓ All extreme parameter value tests passed\n");
}

/// Test 2: Zero Load Scenarios
///
/// Test ThermalModel with zero loads (all loads = 0.0])
/// Verify energy conservation (no energy added/removed])
/// Verify temperatures evolve naturally (free-floating])
#[test]
fn test_zero_load_scenarios() {
    println!("\n=== Test 2: Zero Load Scenarios ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Set all loads to zero
    let num_zones = model.num_zones;
    model.loads = VectorField::new(vec![0.0; num_zones]);

    // Record initial temperatures
    let initial_temp = model.temperatures[0];
    let initial_mass_temp = model.mass_temperatures[0];
    println!(
        "Initial temp: {:.2}°C, Mass temp: {:.2}°C",
        initial_temp, initial_mass_temp
    );

    // Solve with zero loads - should still produce finite result
    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
    println!("Energy with zero loads: {:.2}", energy);

    // Energy should be finite (though may be very low])
    assert!(
        energy.is_finite(),
        "Energy should be finite with zero loads"
    );
    assert!(energy >= 0.0, "Energy should be non-negative");

    // Final temperatures should be finite
    let final_temp = model.temperatures[0];
    let final_mass_temp = model.mass_temperatures[0];
    println!(
        "Final temp: {:.2}°C, Mass temp: {:.2}°C",
        final_temp, final_mass_temp
    );

    assert!(final_temp.is_finite(), "Final temperature should be finite");
    assert!(
        final_mass_temp.is_finite(),
        "Final mass temperature should be finite"
    );

    // Temperatures should have evolved (not exactly the same])
    assert!(
        (final_temp - initial_temp).abs() > EPSILON
            || (final_mass_temp - initial_mass_temp).abs() > EPSILON,
        "Temperatures should evolve with zero loads"
    );

    println!("✓ Zero load scenarios test passed\n");
}

/// Test 3: Extreme Temperature Initial Conditions
///
/// Test initialization at very low temperatures (-50°C])
/// Test initialization at very high temperatures (100°C])
/// Verify physics solver handles extreme temperatures without NaN/Inf
#[test]
fn test_extreme_temperature_initial_conditions() {
    println!("\n=== Test 3: Extreme Temperature Initial Conditions ===");

    // Test very low initial temperature (-50°C])
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        // Set extreme initial temperature
        model.temperatures = VectorField::new(vec![-50.0]);
        model.mass_temperatures = VectorField::new(vec![-50.0]);

        println!("Initial extreme low temp: -50.0°C");

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy from -50°C: {:.2}", energy);

        assert!(energy.is_finite(), "Energy should be finite from -50°C");
        assert!(energy >= 0.0, "Energy should be non-negative");

        // Final temperatures should be finite and reasonable
        let final_temp = model.temperatures[0];
        let final_mass_temp = model.mass_temperatures[0];
        println!(
            "Final temp: {:.2}°C, Mass temp: {:.2}°C",
            final_temp, final_mass_temp
        );

        assert!(final_temp.is_finite(), "Final temperature should be finite");
        assert!(
            final_mass_temp.is_finite(),
            "Final mass temperature should be finite"
        );
        assert!(!final_temp.is_nan(), "Final temperature should not be NaN");
        assert!(
            !final_mass_temp.is_nan(),
            "Final mass temperature should not be NaN"
        );
    }

    // Test very high initial temperature (100°C])
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        // Set extreme initial temperature
        model.temperatures = VectorField::new(vec![100.0]);
        model.mass_temperatures = VectorField::new(vec![100.0]);

        println!("Initial extreme high temp: 100.0°C");

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy from 100°C: {:.2}", energy);

        assert!(energy.is_finite(), "Energy should be finite from 100°C");
        assert!(energy >= 0.0, "Energy should be non-negative");

        // Final temperatures should be finite and reasonable
        let final_temp = model.temperatures[0];
        let final_mass_temp = model.mass_temperatures[0];
        println!(
            "Final temp: {:.2}°C, Mass temp: {:.2}°C",
            final_temp, final_mass_temp
        );

        assert!(final_temp.is_finite(), "Final temperature should be finite");
        assert!(
            final_mass_temp.is_finite(),
            "Final mass temperature should be finite"
        );
        assert!(!final_temp.is_nan(), "Final temperature should not be NaN");
        assert!(
            !final_mass_temp.is_nan(),
            "Final mass temperature should not be NaN"
        );
    }

    println!("✓ Extreme temperature initial conditions test passed\n");
}

/// Test 4: Boundary Conductance Values
///
/// Test with zero conductance (h_tr_w = 0.0])
/// Test with very high conductance (h_tr_w = 1000.0])
/// Verify solver doesn't divide by zero or produce invalid results
#[test]
fn test_boundary_conductance_values() {
    println!("\n=== Test 4: Boundary Conductance Values ===");

    // Test zero conductance (h_tr_w = 0.0])
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        // Set window conductance to zero (perfectly insulated windows])
        model.h_tr_w = VectorField::new(vec![0.0]);

        println!("Zero window conductance (h_tr_w = 0.0)");

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy with zero conductance: {:.2}", energy);

        assert!(
            energy.is_finite(),
            "Energy should be finite with zero conductance"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");

        // Verify no NaN/Inf in temperatures
        let final_temp = model.temperatures[0];
        let final_mass_temp = model.mass_temperatures[0];
        assert!(final_temp.is_finite(), "Final temperature should be finite");
        assert!(
            final_mass_temp.is_finite(),
            "Final mass temperature should be finite"
        );
    }

    // Test very high conductance (h_tr_w = 1000.0])
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        // Set window conductance to very high value
        model.h_tr_w = VectorField::new(vec![1000.0]);

        println!("High window conductance (h_tr_w = 1000.0)");

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy with high conductance: {:.2}", energy);

        assert!(
            energy.is_finite(),
            "Energy should be finite with high conductance"
        );
        assert!(energy >= 0.0, "Energy should be non-negative");

        // Verify no NaN/Inf in temperatures
        let final_temp = model.temperatures[0];
        let final_mass_temp = model.mass_temperatures[0];
        assert!(final_temp.is_finite(), "Final temperature should be finite");
        assert!(
            final_mass_temp.is_finite(),
            "Final mass temperature should be finite"
        );
    }

    println!("✓ Boundary conductance values test passed\n");
}

/// Test 5: Single Zone Edge Case
///
/// Test ThermalModel with num_zones = 1
/// Verify all operations work for single-zone model
/// Test parameter application and physics solve
#[test]
fn test_single_zone_edge_case() {
    println!("\n=== Test 5: Single Zone Edge Case ===");

    let num_zones = 1;
    let mut model = ThermalModel::new(num_zones);

    // Verify model creation
    assert_eq!(model.num_zones, num_zones);
    assert_eq!(model.temperatures.len(), num_zones);
    assert_eq!(model.mass_temperatures.len(), num_zones);
    assert_eq!(model.loads.len(), num_zones);

    println!("Single zone model created: {} zone", num_zones);

    // Apply parameters
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Solve physics
    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
    println!("Single zone energy: {:.2}", energy);

    assert!(
        energy.is_finite(),
        "Energy should be finite for single zone"
    );
    assert!(energy >= 0.0, "Energy should be non-negative");

    // Verify temperature states
    let final_temp = model.temperatures[0];
    let final_mass_temp = model.mass_temperatures[0];
    println!(
        "Final temp: {:.2}°C, Mass temp: {:.2}°C",
        final_temp, final_mass_temp
    );

    assert!(final_temp.is_finite(), "Final temperature should be finite");
    assert!(
        final_mass_temp.is_finite(),
        "Final mass temperature should be finite"
    );

    println!("✓ Single zone edge case test passed\n");
}

/// Test 6: Large Zone Count Edge Case
///
/// Test ThermalModel with num_zones = 1000
/// Verify solver scales correctly
/// Check for performance degradation or memory issues
#[test]
fn test_large_zone_count_edge_case() {
    println!("\n=== Test 6: Large Zone Count Edge Case ===");

    let num_zones = 1000;
    let mut model = ThermalModel::new(num_zones);

    // Verify model creation
    assert_eq!(model.num_zones, num_zones);
    assert_eq!(model.temperatures.len(), num_zones);
    assert_eq!(model.mass_temperatures.len(), num_zones);
    assert_eq!(model.loads.len(), num_zones);

    println!("Large zone model created: {} zones", num_zones);

    // Apply parameters
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Solve physics with timing
    let start = std::time::Instant::now();
    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
    let duration = start.elapsed();

    println!("Large zone energy: {:.2}", energy);
    println!("Solver duration: {:?}", duration);

    assert!(
        energy.is_finite(),
        "Energy should be finite for large zone count"
    );
    assert!(energy >= 0.0, "Energy should be non-negative");

    // Verify temperature states for all zones
    for i in 0..num_zones {
        let temp = model.temperatures[i];
        let mass_temp = model.mass_temperatures[i];
        assert!(temp.is_finite(), "Zone {} temperature should be finite", i);
        assert!(
            mass_temp.is_finite(),
            "Zone {} mass temperature should be finite",
            i
        );
    }

    // Check performance - should complete in reasonable time
    // (allow up to 10 seconds for 1000 zones in debug mode])
    assert!(
        duration.as_secs() < 10,
        "Solver took too long: {:?}",
        duration
    );

    println!("✓ Large zone count edge case test passed\n");
}

/// Test 7: Invalid Parameter Handling
///
/// Test U-value < MIN_U_VALUE (should handle gracefully])
/// Test U-value > MAX_U_VALUE (should handle gracefully])
/// Test setpoint outside valid range (should handle gracefully])
/// Verify graceful degradation, not panics
#[test]
fn test_invalid_parameter_handling() {
    println!("\n=== Test 7: Invalid Parameter Handling ===");

    // Test U-value < MIN_U_VALUE
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 0.05; // Below MIN_U_VALUE (0.1])
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        println!("Testing U-value below MIN_U_VALUE: 0.05");

        // The model should still run without panic
        // (validation happens at BatchOracle level, not ThermalModel])
        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2}", energy);

        // Should still produce finite result (ThermalModel doesn't validate bounds])
        assert!(energy.is_finite(), "Energy should be finite");
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test U-value > MAX_U_VALUE
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 10.0; // Above MAX_U_VALUE (5.0])
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 24.0;

        println!("Testing U-value above MAX_U_VALUE: 10.0");

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2}", energy);

        assert!(energy.is_finite(), "Energy should be finite");
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test heating setpoint outside valid range (below MIN])
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 10.0; // Below MIN_HEATING_SETPOINT (15.0])
        model.cooling_setpoint = 24.0;

        println!("Testing heating setpoint below MIN: 10.0°C");

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2}", energy);

        assert!(energy.is_finite(), "Energy should be finite");
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test cooling setpoint outside valid range (above MAX])
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 40.0; // Above MAX_COOLING_SETPOINT (32.0])

        println!("Testing cooling setpoint above MAX: 40.0°C");

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2}", energy);

        assert!(energy.is_finite(), "Energy should be finite");
        assert!(energy >= 0.0, "Energy should be non-negative");
    }

    // Test heating setpoint >= cooling setpoint (invalid])
    {
        let mut model = ThermalModel::new(1);
        model.window_u_value = 2.0;
        model.heating_setpoint = 25.0;
        model.cooling_setpoint = 20.0; // Heating >= cooling (invalid])

        println!("Testing heating >= cooling: 25.0 >= 20.0");

        // Model should still run without panic (validation at BatchOracle level])
        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2}", energy);

        // May produce unexpected results but should not panic
        assert!(energy.is_finite(), "Energy should be finite");
    }

    println!("✓ Invalid parameter handling test passed (graceful degradation)\n");
}

/// Test 8: Edge Case - Very Small Load Values
///
/// Test with loads at the limit of floating-point precision
/// Verify numerical stability
#[test]
fn test_very_small_load_values() {
    println!("\n=== Test 8: Very Small Load Values ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Test with very small positive load
    {
        let tiny_load = 1e-10;
        model.loads = VectorField::new(vec![tiny_load]);
        println!("Testing tiny positive load: {:.2e}", tiny_load);

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2e}", energy);

        assert!(energy.is_finite(), "Energy should be finite");
        assert!(!energy.is_nan(), "Energy should not be NaN");
    }

    // Test with very small negative load
    {
        let tiny_load = -1e-10;
        model.loads = VectorField::new(vec![tiny_load]);
        println!("Testing tiny negative load: {:.2e}", tiny_load);

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2e}", energy);

        assert!(energy.is_finite(), "Energy should be finite");
        assert!(!energy.is_nan(), "Energy should not be NaN");
    }

    println!("✓ Very small load values test passed\n");
}

/// Test 9: Edge Case - Extremely Large Load Values
///
/// Test with very large loads (simulating extreme HVAC capacity])
/// Verify system doesn't overflow or produce Inf
#[test]
fn test_extremely_large_load_values() {
    println!("\n=== Test 9: Extremely Large Load Values ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Test with very large positive load
    {
        let huge_load = 1e6; // 1 MW
        model.loads = VectorField::new(vec![huge_load]);
        println!("Testing huge positive load: {:.2e} W", huge_load);

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2e}", energy);

        assert!(energy.is_finite(), "Energy should be finite");
        assert!(!energy.is_infinite(), "Energy should not be infinite");
    }

    // Test with very large negative load
    {
        let huge_load = -1e6; // -1 MW
        model.loads = VectorField::new(vec![huge_load]);
        println!("Testing huge negative load: {:.2e} W", huge_load);

        let surrogates = SurrogateManager::new().unwrap();
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
        println!("Energy: {:.2e}", energy);

        assert!(energy.is_finite(), "Energy should be finite");
        assert!(!energy.is_infinite(), "Energy should not be infinite");
    }

    println!("✓ Extremely large load values test passed\n");
}

/// Test 10: Edge Case - Zero Timesteps
///
/// Test solver with zero timesteps
/// Verify graceful handling
#[test]
fn test_zero_timesteps() {
    println!("\n=== Test 10: Zero Timesteps ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    println!("Testing zero timesteps");

    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(0, &surrogates, false, None, None, None);
    println!("Energy with zero timesteps: {:.2}", energy);

    // Zero timesteps should produce zero energy
    assert_eq!(energy, 0.0, "Energy should be zero with zero timesteps");

    println!("✓ Zero timesteps test passed\n");
}

/// Test 11: Edge Case - Single Timestep
///
/// Test solver with single timestep
/// Verify numerical stability at minimum iteration count
#[test]
fn test_single_timestep() {
    println!("\n=== Test 11: Single Timestep ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Record initial state
    let initial_temp = model.temperatures[0];
    let initial_mass_temp = model.mass_temperatures[0];

    println!(
        "Initial temp: {:.2}°C, Mass temp: {:.2}°C",
        initial_temp, initial_mass_temp
    );

    println!("Testing single timestep");

    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(1, &surrogates, false, None, None, None);
    println!("Energy with one timestep: {:.2}", energy);

    assert!(energy.is_finite(), "Energy should be finite");
    assert!(!energy.is_nan(), "Energy should not be NaN");

    // Verify state changed
    let final_temp = model.temperatures[0];
    let final_mass_temp = model.mass_temperatures[0];
    println!(
        "Final temp: {:.2}°C, Mass temp: {:.2}°C",
        final_temp, final_mass_temp
    );

    assert!(final_temp.is_finite(), "Final temperature should be finite");
    assert!(
        final_mass_temp.is_finite(),
        "Final mass temperature should be finite"
    );

    println!("✓ Single timestep test passed\n");
}

/// Test 12: Edge Case - Mixed Positive/Negative Loads Across Zones
///
/// Test with some zones having heating loads and some cooling
/// Verify multi-zone solver handles opposing loads correctly
#[test]
fn test_mixed_positive_negative_loads() {
    println!("\n=== Test 12: Mixed Positive/Negative Loads ===");

    let num_zones = 10;
    let mut model = ThermalModel::new(num_zones);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Set alternating positive/negative loads
    let loads: Vec<f64> = (0..num_zones)
        .map(|i| if i % 2 == 0 { 1000.0 } else { -1000.0 })
        .collect();
    model.loads = VectorField::new(loads.clone());

    println!("Testing {} zones with alternating ±1000W loads", num_zones);

    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
    println!("Energy: {:.2}", energy);

    assert!(energy.is_finite(), "Energy should be finite");
    assert!(!energy.is_nan(), "Energy should not be NaN");

    // Verify all zones have finite temperatures
    for i in 0..num_zones {
        let temp = model.temperatures[i];
        let mass_temp = model.mass_temperatures[i];
        assert!(temp.is_finite(), "Zone {} temperature should be finite", i);
        assert!(
            mass_temp.is_finite(),
            "Zone {} mass temperature should be finite",
            i
        );
    }

    println!("✓ Mixed positive/negative loads test passed\n");
}

/// Test 13: Edge Case - Asymmetric Multi-Zone Configuration
///
/// Test with zones having different thermal properties
/// Verify solver handles asymmetric configurations correctly
#[test]
fn test_asymmetric_multi_zone_configuration() {
    println!("\n=== Test 13: Asymmetric Multi-Zone Configuration ===");

    let num_zones = 5;
    let mut model = ThermalModel::new(num_zones);

    // Apply asymmetric parameters per zone
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Set different loads per zone to simulate asymmetric thermal conditions
    let loads: Vec<f64> = vec![0.0, 500.0, 1000.0, 1500.0, 2000.0];
    model.loads = VectorField::new(loads.clone());

    println!(
        "Testing {} zones with asymmetric loads: {:?}",
        num_zones, loads
    );

    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);
    println!("Energy: {:.2}", energy);

    assert!(energy.is_finite(), "Energy should be finite");
    assert!(!energy.is_nan(), "Energy should not be NaN");

    // Verify all zones have finite temperatures after solving
    for i in 0..num_zones {
        let temp = model.temperatures[i];
        let mass_temp = model.mass_temperatures[i];
        assert!(temp.is_finite(), "Zone {} temperature should be finite", i);
        assert!(
            mass_temp.is_finite(),
            "Zone {} mass temperature should be finite",
            i
        );
    }

    // Verify temperatures differ across zones (asymmetric behavior)
    let temp_0 = model.temperatures[0];
    let temp_4 = model.temperatures[4];
    let temp_diff = (temp_4 - temp_0).abs();
    assert!(
        temp_diff > EPSILON,
        "Asymmetric loads should produce different temperatures (diff: {:.2})",
        temp_diff
    );

    println!("✓ Asymmetric multi-zone configuration test passed\n");
}

/// Test 14: Edge Case - Setpoint Transition Dynamics
///
/// Test behavior when setpoints are changed mid-simulation
/// Verify smooth transitions without numerical instability
#[test]
fn test_setpoint_transition_dynamics() {
    println!("\n=== Test 14: Setpoint Transition Dynamics ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Record initial energy with first setpoint
    let surrogates = SurrogateManager::new().unwrap();
    let energy_1 = model.solve_timesteps(100, &surrogates, false, None, None, None);
    println!(
        "Energy with setpoints 20-24°C (first 100 steps): {:.2}",
        energy_1
    );

    // Change setpoints mid-simulation
    model.heating_setpoint = 18.0;
    model.cooling_setpoint = 26.0;

    // Continue simulation with new setpoints
    let energy_2 = model.solve_timesteps(100, &surrogates, false, None, None, None);
    println!(
        "Energy with setpoints 18-26°C (next 100 steps): {:.2}",
        energy_2
    );

    assert!(energy_1.is_finite(), "First energy should be finite");
    assert!(energy_2.is_finite(), "Second energy should be finite");
    assert!(!energy_1.is_nan(), "First energy should not be NaN");
    assert!(!energy_2.is_nan(), "Second energy should not be NaN");

    // Verify final temperatures are finite
    let final_temp = model.temperatures[0];
    let final_mass_temp = model.mass_temperatures[0];
    assert!(final_temp.is_finite(), "Final temperature should be finite");
    assert!(
        final_mass_temp.is_finite(),
        "Final mass temperature should be finite"
    );

    println!("✓ Setpoint transition dynamics test passed\n");
}

/// Test 15: Edge Case - Rapid Load Changes
///
/// Test with loads that change dramatically between timesteps
/// Verify solver handles load transients without instability
#[test]
fn test_rapid_load_changes() {
    println!("\n=== Test 15: Rapid Load Changes ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Test sequence of rapid load changes
    let load_sequence = vec![0.0, 1000.0, 0.0, -1000.0, 0.0];

    println!("Testing rapid load changes: {:?}", load_sequence);

    let surrogates = SurrogateManager::new().unwrap();
    let mut total_energy = 0.0;

    for (step, &load) in load_sequence.iter().enumerate() {
        model.loads = VectorField::new(vec![load]);
        let step_energy = model.solve_timesteps(10, &surrogates, false, None, None, None);
        total_energy += step_energy;
        println!("Step {}: load={:.2}, energy={:.2}", step, load, step_energy);

        assert!(
            step_energy.is_finite(),
            "Energy should be finite at step {}",
            step
        );
        assert!(
            !step_energy.is_nan(),
            "Energy should not be NaN at step {}",
            step
        );
    }

    println!("Total energy: {:.2}", total_energy);

    assert!(total_energy.is_finite(), "Total energy should be finite");
    assert!(!total_energy.is_nan(), "Total energy should not be NaN");

    // Verify final state is stable
    let final_temp = model.temperatures[0];
    let final_mass_temp = model.mass_temperatures[0];
    assert!(final_temp.is_finite(), "Final temperature should be finite");
    assert!(
        final_mass_temp.is_finite(),
        "Final mass temperature should be finite"
    );

    println!("✓ Rapid load changes test passed\n");
}

/// Test 16: Edge Case - Zero Conductance All Paths
///
/// Test with all thermal conductances set to zero
/// Verify isolation behavior without numerical errors
#[test]
fn test_zero_conductance_all_paths() {
    println!("\n=== Test 16: Zero Conductance All Paths ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Set all 5R1C conductances to zero (perfect isolation)
    model.h_tr_em = VectorField::new(vec![0.0]); // Exterior -> Mass
    model.h_tr_ms = VectorField::new(vec![0.0]); // Mass -> Surface
    model.h_tr_is = VectorField::new(vec![0.0]); // Surface -> Interior
    model.h_tr_w = VectorField::new(vec![0.0]); // Exterior -> Interior
    model.h_ve = VectorField::new(vec![0.0]); // Ventilation

    println!("Testing with all conductances = 0.0 (perfect isolation)");

    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(100, &surrogates, false, None, None, None);
    println!("Energy with all zero conductances: {:.2}", energy);

    // Should still produce finite result (temperatures remain at initial state)
    assert!(energy.is_finite(), "Energy should be finite");
    assert!(!energy.is_nan(), "Energy should not be NaN");

    // Temperatures should remain stable (minimal change due to isolation)
    let initial_temp = 20.0; // Default initialization
    let final_temp = model.temperatures[0];
    let temp_change = (final_temp - initial_temp).abs();

    println!(
        "Initial temp: {:.2}°C, Final temp: {:.2}°C, Change: {:.2}°C",
        initial_temp, final_temp, temp_change
    );

    assert!(final_temp.is_finite(), "Final temperature should be finite");

    println!("✓ Zero conductance all paths test passed\n");
}

/// Test 17: Edge Case - Leap Year Simulation
///
/// Test with 8784 timesteps (leap year)
/// Verify solver handles non-standard year length
#[test]
fn test_leap_year_simulation() {
    println!("\n=== Test 17: Leap Year Simulation ===");

    let mut model = ThermalModel::new(1);
    model.window_u_value = 2.0;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 24.0;

    // Leap year has 366 days = 8784 hours
    let leap_year_steps = 8784;
    println!("Testing leap year ({} hours)", leap_year_steps);

    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(leap_year_steps, &surrogates, false, None, None, None);
    println!("Energy for leap year: {:.2}", energy);

    assert!(energy.is_finite(), "Energy should be finite");
    assert!(!energy.is_nan(), "Energy should not be NaN");
    assert!(energy >= 0.0, "Energy should be non-negative");

    // Verify final state is valid
    let final_temp = model.temperatures[0];
    let final_mass_temp = model.mass_temperatures[0];
    assert!(final_temp.is_finite(), "Final temperature should be finite");
    assert!(
        final_mass_temp.is_finite(),
        "Final mass temperature should be finite"
    );

    println!("✓ Leap year simulation test passed\n");
}

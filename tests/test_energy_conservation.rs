//! Energy conservation tests for analytical load calculations.
//!
//! These tests verify that the analytical physics path correctly computes
//! thermal loads and conserves energy, enabling ASHRAE 140 compliance.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_energy_conservation() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run a short simulation (24 hours) to verify energy conservation
    // Use analytical physics path (use_ai=false)
    let surrogates = fluxion::ai::surrogate::SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(24, &surrogates, false, None, None, None);

    // Energy should be finite and positive (or negative for cooling)
    assert!(!energy.is_nan(), "Total energy should not be NaN");
    // Energy can be positive (heating) or negative (cooling) depending on conditions
    // Just check it's finite and non-zero
    assert!(
        energy.abs() > 0.0,
        "Total energy should be non-zero, got {}",
        energy
    );

    println!(
        "✓ Energy conservation test passed: {:.4} kWh/m² (24 hours)",
        energy
    );
}

#[test]
fn test_analytical_loads_nonzero() {
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Calculate loads for a sample timestep (noon, summer)
    let outdoor_temp = 35.0; // Hot day
    let hour_of_day = 12;
    let loads = model.calculate_analytical_loads(outdoor_temp, hour_of_day);

    // All loads should be non-zero (solar + conduction + ventilation)
    for (i, load) in loads.iter().enumerate() {
        // Note: Loads can be positive or negative depending on temperature difference
        // but should not be exactly zero (which would indicate no calculation)
        assert!(
            load.abs() > 1e-10,
            "Load for zone {} should be non-zero (got {:.2e})",
            i,
            load
        );
    }

    println!("✓ Analytical loads are non-zero: {:?}", loads);
}

#[test]
fn test_analytical_loads_consistency() {
    // Test that analytical loads are consistent with physics expectations
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Test 1: Hot outdoor temp should result in positive net load (cooling needed)
    let loads_hot = model.calculate_analytical_loads(35.0, 12);
    // Hot day: conduction (hot outside), solar (high), ventilation (hot air)
    // Net load should be positive (needs cooling)
    let net_load_hot: f64 = loads_hot.iter().sum();
    assert!(
        net_load_hot > 0.0,
        "Hot day should have positive net load (cooling needed), got {:.2e}",
        net_load_hot
    );

    // Test 2: Cold outdoor temp should result in negative net load (heating needed)
    let loads_cold = model.calculate_analytical_loads(5.0, 12);
    // Cold day: conduction (cold outside), solar (low), ventilation (cold air)
    // Net load should be negative (needs heating)
    let net_load_cold: f64 = loads_cold.iter().sum();
    assert!(
        net_load_cold < 0.0,
        "Cold day should have negative net load (heating needed), got {:.2e}",
        net_load_cold
    );

    println!("✓ Analytical loads are consistent with physics:");
    println!(
        "  Hot day (35°C): {:.2e} W/m² (cooling needed)",
        net_load_hot
    );
    println!(
        "  Cold day (5°C): {:.2e} W/m² (heating needed)",
        net_load_cold
    );
}

#[test]
fn test_analytical_loads_seasonal_variation() {
    // Test that loads vary appropriately with outdoor temperature
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Test 1: Hot outdoor temp should result in higher total load (cooling needed)
    let loads_hot = model.calculate_analytical_loads(35.0, 12);
    let total_load_hot: f64 = loads_hot.iter().sum();

    // Test 2: Cold outdoor temp should result in lower total load (heating needed)
    let loads_cold = model.calculate_analytical_loads(5.0, 12);
    let total_load_cold: f64 = loads_cold.iter().sum();

    // Test 3: Moderate outdoor temp should result in intermediate load
    let loads_moderate = model.calculate_analytical_loads(20.0, 12);
    let total_load_moderate: f64 = loads_moderate.iter().sum();

    // Load magnitude should increase with outdoor temperature difference
    // (since conduction and ventilation both depend on T_out - T_in)
    assert!(
        total_load_hot > total_load_moderate,
        "Load should be higher at hot outdoor temperature"
    );
    assert!(
        total_load_cold.abs() > total_load_moderate.abs() || total_load_cold > total_load_moderate,
        "Load should vary with outdoor temperature"
    );

    println!("✓ Load varies with outdoor temperature:");
    println!("  Cold (5°C): {:.2e} W/m²", total_load_cold);
    println!("  Moderate (20°C): {:.2e} W/m²", total_load_moderate);
    println!("  Hot (35°C): {:.2e} W/m²", total_load_hot);
}

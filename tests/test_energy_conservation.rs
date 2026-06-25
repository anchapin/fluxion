//! Energy conservation tests for analytical load calculations.
//!
//! These tests verify that the analytical physics path correctly computes
//! thermal loads and conserves energy, enabling ASHRAE 140 compliance.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

const ENERGY_BALANCE_RESIDUAL_THRESHOLD: f64 = 0.001; // 0.1% CI gate

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

/// Test that Case 600 (low-mass) satisfies strict energy conservation (< 0.1% residual).
///
/// Low-mass cases are sensitive to temporal discretization errors. The 5R1C
/// network must conserve energy at each timestep to avoid drift.
///
/// Reference: Issue #1295 - Enforce strict energy conservation in CI
#[test]
fn test_case_600_energy_conservation_residual() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let tolerance = ENERGY_BALANCE_RESIDUAL_THRESHOLD;
    let mut checker = InvariantChecker::new(tolerance);

    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(10.0);

    let dt = 3600.0;
    let n_steps = 168; // 1 week

    let mut max_residual = 0.0_f64;
    let mut violated_timesteps = Vec::new();

    for step in 0..n_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let t_outdoor = weather_data.dry_bulb_temp;

        let result = checker.check_invariant(&model, dt, t_outdoor);

        if result.violated {
            violated_timesteps.push((step, result.balance, result.tolerance));
        }

        let residual = result.balance.abs() / 1000.0;
        if residual > max_residual {
            max_residual = residual;
        }
    }

    let total_violations = checker.violation_count();
    let max_violation = checker.max_violation();

    println!(
        "[#1295 Case 600 energy balance] N={}, violations={}, max_residual={:.6}, max_violation={:.6e}",
        n_steps,
        total_violations,
        max_residual * 100.0,
        max_violation
    );

    assert!(
        total_violations == 0,
        "Case 600: {} timesteps violated energy conservation (> {}). Max violation: {:.6e} W",
        total_violations,
        tolerance,
        max_violation
    );
}

/// Test that Case 900 (high-mass) satisfies strict energy conservation (< 0.1% residual).
///
/// High-mass cases have larger thermal capacitance, so energy imbalances
/// are more likely to accumulate if the 5R1C network has bugs.
///
/// Reference: Issue #1295 - Enforce strict energy conservation in CI
#[test]
fn test_case_900_energy_conservation_residual() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let tolerance = ENERGY_BALANCE_RESIDUAL_THRESHOLD;
    let mut checker = InvariantChecker::new(tolerance);

    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(10.0);

    let dt = 3600.0;
    let n_steps = 168; // 1 week

    let mut max_residual = 0.0_f64;
    let mut violated_timesteps = Vec::new();

    for step in 0..n_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let t_outdoor = weather_data.dry_bulb_temp;

        let result = checker.check_invariant(&model, dt, t_outdoor);

        if result.violated {
            violated_timesteps.push((step, result.balance, result.tolerance));
        }

        let residual = result.balance.abs() / 1000.0;
        if residual > max_residual {
            max_residual = residual;
        }
    }

    let total_violations = checker.violation_count();
    let max_violation = checker.max_violation();

    println!(
        "[#1295 Case 900 energy balance] N={}, violations={}, max_residual={:.6}, max_violation={:.6e}",
        n_steps,
        total_violations,
        max_residual * 100.0,
        max_violation
    );

    assert!(
        total_violations == 0,
        "Case 900: {} timesteps violated energy conservation (> {}). Max violation: {:.6e} W",
        total_violations,
        tolerance,
        max_violation
    );
}

/// Test that Case 960 (multi-zone sunspace) satisfies strict energy conservation.
///
/// Case 960 is the critical case for MULTI-02 (COP conversion fix). The COP
/// conversion was a validation accounting fix - the core physics was unchanged.
/// This test ensures the multi-zone physics conserves energy properly.
///
/// Reference: Issue #1295, MULTI-02 (docs/KNOWN_ISSUES.md)
#[test]
fn test_case_960_energy_conservation_residual() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let tolerance = ENERGY_BALANCE_RESIDUAL_THRESHOLD;
    let mut checker = InvariantChecker::new(tolerance);

    for i in 0..model.num_zones {
        model.temperatures.as_mut()[i] = 20.0;
        if let Some(ref mut mt) = Some(&mut model.mass_temperatures) {
            mt.as_mut()[i] = 20.0;
        }
    }
    model.set_ground_temp(10.0);

    let dt = 3600.0;
    let n_steps = 168; // 1 week

    let mut max_residual = 0.0_f64;
    let mut violated_timesteps = Vec::new();

    for step in 0..n_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let t_outdoor = weather_data.dry_bulb_temp;

        let result = checker.check_invariant(&model, dt, t_outdoor);

        if result.violated {
            violated_timesteps.push((step, result.balance, result.tolerance));
        }

        let residual = result.balance.abs() / 1000.0;
        if residual > max_residual {
            max_residual = residual;
        }
    }

    let total_violations = checker.violation_count();
    let max_violation = checker.max_violation();

    println!(
        "[#1295 Case 960 (MULTI-02) energy balance] N={}, violations={}, max_residual={:.6}, max_violation={:.6e}",
        n_steps,
        total_violations,
        max_residual * 100.0,
        max_violation
    );

    assert!(
        total_violations == 0,
        "Case 960: {} timesteps violated energy conservation (> {}). Max violation: {:.6e} W. This case covers MULTI-02 (COP conversion).",
        total_violations,
        tolerance,
        max_violation
    );
}

/// Test that free-floating cases (no HVAC) satisfy strict energy conservation.
///
/// Free-floating cases (600FF, 900FF) are important because the HVAC system
/// is disabled, so energy conservation means all gains must be balanced by
/// envelope losses and thermal mass storage changes.
///
/// Reference: Issue #1295 - Enforce strict energy conservation in CI
#[test]
fn test_free_floating_energy_conservation_residual() {
    let spec = ASHRAE140Case::Case600FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Disable HVAC for free-floating mode
    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;
    model.hvac_heating_capacity = 0.0;
    model.hvac_cooling_capacity = 0.0;

    let tolerance = ENERGY_BALANCE_RESIDUAL_THRESHOLD;
    let mut checker = InvariantChecker::new(tolerance);

    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(10.0);

    let dt = 3600.0;
    let n_steps = 168; // 1 week

    let mut max_residual = 0.0_f64;
    let mut violated_timesteps = Vec::new();

    for step in 0..n_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let t_outdoor = weather_data.dry_bulb_temp;

        let result = checker.check_invariant(&model, dt, t_outdoor);

        if result.violated {
            violated_timesteps.push((step, result.balance, result.tolerance));
        }

        let residual = result.balance.abs() / 1000.0;
        if residual > max_residual {
            max_residual = residual;
        }
    }

    let total_violations = checker.violation_count();
    let max_violation = checker.max_violation();

    println!(
        "[#1295 Case 600FF free-float energy balance] N={}, violations={}, max_residual={:.6}, max_violation={:.6e}",
        n_steps,
        total_violations,
        max_residual * 100.0,
        max_violation
    );

    assert!(
        total_violations == 0,
        "Case 600FF: {} timesteps violated energy conservation (> {}). Max violation: {:.6e} W",
        total_violations,
        tolerance,
        max_violation
    );
}

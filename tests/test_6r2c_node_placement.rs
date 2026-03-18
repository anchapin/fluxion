//! Plan 24-04: Node Placement and Boundary Condition Tests
//!
//! This test suite validates:
//! - Thermal mass node physical interpretation (surface vs core temperature)
//! - Boundary condition application (interior/exterior)
//! - Initial conditions for capacitance nodes
//! - Solar gain distribution to nodes
//! - Internal gain distribution (convective/radiative split)
//!
//! Reference: docs/ISO_13790_6R2C_SPECIFICATION.md §4

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

// ============================================================================
// Section 1: Mass Node Position Tests
// ============================================================================

#[test]
fn test_envelope_mass_node_exists() {
    // 6R2C model should have separate envelope mass node
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Envelope mass temperature should be initialized
    let t_env = model.envelope_mass_temperatures.as_ref()[0];
    assert!(
        t_env.is_finite(),
        "Envelope mass temperature should be finite"
    );

    // Should be in reasonable range (typically 15-25°C initial)
    assert!(
        t_env > -50.0 && t_env < 100.0,
        "Envelope mass temp should be reasonable, got {:.1}°C",
        t_env
    );
}

#[test]
fn test_internal_mass_node_exists() {
    // 6R2C model should have separate internal mass node
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Internal mass temperature should be initialized
    let t_int = model.internal_mass_temperatures.as_ref()[0];
    assert!(
        t_int.is_finite(),
        "Internal mass temperature should be finite"
    );

    // Should be in reasonable range
    assert!(
        t_int > -50.0 && t_int < 100.0,
        "Internal mass temp should be reasonable, got {:.1}°C",
        t_int
    );
}

#[test]
fn test_mass_nodes_initialized_from_single_mass() {
    // When configuring 6R2C, both mass nodes should initialize from single mass temp
    let mut model = ThermalModel::new(1);
    let initial_temp = model.mass_temperatures.as_ref()[0];

    model.configure_6r2c_model(0.75, 100.0);

    let t_env = model.envelope_mass_temperatures.as_ref()[0];
    let t_int = model.internal_mass_temperatures.as_ref()[0];

    // Both should equal the initial single mass temperature
    assert_eq!(
        t_env, initial_temp,
        "Envelope mass should initialize from single mass temp"
    );
    assert_eq!(
        t_int, initial_temp,
        "Internal mass should initialize from single mass temp"
    );
}

#[test]
fn test_mass_nodes_diverge_during_simulation() {
    // During simulation, envelope and internal mass temps should diverge
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Set up temperature gradient to drive heat transfer
    // Cold outdoor temp will cool envelope mass faster than internal mass
    let initial_t_env = model.envelope_mass_temperatures.as_ref()[0];
    let initial_t_int = model.internal_mass_temperatures.as_ref()[0];

    // Run several timesteps with cold outdoor temperature
    for timestep in 0..24 {
        model.step_physics(timestep, 0.0, 3600.0); // 0°C outdoor
    }

    let final_t_env = model.envelope_mass_temperatures.as_ref()[0];
    let final_t_int = model.internal_mass_temperatures.as_ref()[0];

    // Envelope mass should cool more than internal mass (coupled through h_tr_me)
    let delta_t_env = initial_t_env - final_t_env;
    let delta_t_int = initial_t_int - final_t_int;

    assert!(delta_t_env > 0.0, "Envelope mass should cool");
    assert!(
        delta_t_int > 0.0,
        "Internal mass should cool (through coupling)"
    );
    assert!(
        delta_t_env > delta_t_int,
        "Envelope mass should cool more than internal mass: ΔT_env={:.2}, ΔT_int={:.2}",
        delta_t_env,
        delta_t_int
    );
}

// ============================================================================
// Section 2: Boundary Condition Tests
// ============================================================================

#[test]
fn test_exterior_boundary_drives_envelope_mass() {
    // Envelope mass should respond to exterior temperature changes
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    let initial_t_env = model.envelope_mass_temperatures.as_ref()[0];

    // Run with hot outdoor temperature
    for timestep in 0..12 {
        model.step_physics(timestep, 40.0, 3600.0); // 40°C outdoor (hot day)
    }

    let final_t_env = model.envelope_mass_temperatures.as_ref()[0];

    // Envelope mass should warm up
    assert!(
        final_t_env > initial_t_env,
        "Envelope mass should warm with hot outdoor temp: {:.1} → {:.1}°C",
        initial_t_env,
        final_t_env
    );
}

#[test]
fn test_interior_boundary_drives_zone_air() {
    // Zone air temperature should respond to HVAC setpoints
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    let initial_t_zone = model.temperatures.as_ref()[0];

    // Run simulation (HVAC will maintain setpoint)
    for timestep in 0..24 {
        model.step_physics(timestep, 10.0, 3600.0); // 10°C outdoor
    }

    let final_t_zone = model.temperatures.as_ref()[0];

    // Zone temperature should be maintained near setpoint (not drop to outdoor temp)
    assert!(
        final_t_zone > 15.0,
        "Zone temp should be maintained by HVAC: {:.1}°C",
        final_t_zone
    );
}

#[test]
fn test_ground_boundary_applied() {
    // Ground coupling should affect zone temperature
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Ground temperature is typically ~10-15°C constant
    // This test just verifies the model runs without error with ground coupling
    for timestep in 0..24 {
        model.step_physics(timestep, 30.0, 3600.0); // Hot outdoor
    }

    // Model should complete without NaN or Inf
    let t_zone = model.temperatures.as_ref()[0];
    assert!(
        t_zone.is_finite(),
        "Zone temp should be finite with ground coupling"
    );
}

// ============================================================================
// Section 3: Initial Condition Tests
// ============================================================================

#[test]
fn test_initial_conditions_finite() {
    // All initial temperatures should be finite
    let model = ThermalModel::new(1);

    assert!(
        model.temperatures.as_ref()[0].is_finite(),
        "Zone temp should be finite"
    );
    assert!(
        model.mass_temperatures.as_ref()[0].is_finite(),
        "Mass temp should be finite"
    );
}

#[test]
fn test_6r2c_initial_conditions_finite() {
    // All 6R2C initial temperatures should be finite
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    assert!(
        model.envelope_mass_temperatures.as_ref()[0].is_finite(),
        "Envelope mass temp should be finite"
    );
    assert!(
        model.internal_mass_temperatures.as_ref()[0].is_finite(),
        "Internal mass temp should be finite"
    );
}

#[test]
fn test_warm_start_continuity() {
    // Reloading final state as initial state should have no temperature jumps
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Run for 24 hours
    for timestep in 0..24 {
        model.step_physics(timestep, 15.0, 3600.0);
    }

    // Save final state
    let t_env_final = model.envelope_mass_temperatures.as_ref()[0];
    let t_int_final = model.internal_mass_temperatures.as_ref()[0];
    let t_zone_final = model.temperatures.as_ref()[0];

    // Continue simulation (no reset)
    model.step_physics(24, 15.0, 3600.0);

    // Temperatures should be continuous (no jumps)
    let t_env_next = model.envelope_mass_temperatures.as_ref()[0];
    let t_int_next = model.internal_mass_temperatures.as_ref()[0];
    let t_zone_next = model.temperatures.as_ref()[0];

    // Changes should be reasonable (not jumps)
    let delta_t_env = (t_env_next - t_env_final).abs();
    let delta_t_int = (t_int_next - t_int_final).abs();
    let delta_t_zone = (t_zone_next - t_zone_final).abs();

    assert!(
        delta_t_env < 10.0,
        "Envelope temp change should be < 10°C, got {:.2}",
        delta_t_env
    );
    assert!(
        delta_t_int < 10.0,
        "Internal temp change should be < 10°C, got {:.2}",
        delta_t_int
    );
    assert!(
        delta_t_zone < 10.0,
        "Zone temp change should be < 10°C, got {:.2}",
        delta_t_zone
    );
}

// ============================================================================
// Section 4: Solar Gain Distribution Tests
// ============================================================================

#[test]
fn test_solar_gain_distributed_to_nodes() {
    // Solar gains should be distributed to mass nodes
    // Note: In actual ASHRAE 140 simulation, solar gains come from weather data
    // This test verifies the model structure supports solar distribution

    let model = ThermalModel::new(1);

    // Verify solar distribution parameters are configured
    let beam_to_mass = model.solar_beam_to_mass_fraction;
    let dist_to_air = model.solar_distribution_to_air;

    // Both should be in valid ranges
    assert!(
        beam_to_mass >= 0.0 && beam_to_mass <= 1.0,
        "Solar beam to mass fraction should be [0,1], got {}",
        beam_to_mass
    );
    assert!(
        dist_to_air >= 0.0 && dist_to_air <= 1.0,
        "Solar distribution to air should be [0,1], got {}",
        dist_to_air
    );

    // Test passes if parameters are valid (actual distribution tested in integration tests)
}

#[test]
fn test_solar_beam_to_mass_fraction_configured() {
    // solar_beam_to_mass_fraction should be in valid range [0, 1]
    let model = ThermalModel::new(1);
    let fraction = model.solar_beam_to_mass_fraction;

    assert!(
        fraction >= 0.0 && fraction <= 1.0,
        "Solar beam to mass fraction should be [0,1], got {}",
        fraction
    );
}

#[test]
fn test_solar_distribution_to_air_configured() {
    // solar_distribution_to_air should be in valid range [0, 1]
    let model = ThermalModel::new(1);
    let fraction = model.solar_distribution_to_air;

    assert!(
        fraction >= 0.0 && fraction <= 1.0,
        "Solar distribution to air should be [0,1], got {}",
        fraction
    );
}

// ============================================================================
// Section 5: Internal Gain Distribution Tests
// ============================================================================

#[test]
fn test_convective_fraction_configured() {
    // convective_fraction should be in valid range [0, 1]
    let model = ThermalModel::new(1);
    let fraction = model.convective_fraction;

    assert!(
        fraction >= 0.0 && fraction <= 1.0,
        "Convective fraction should be [0,1], got {}",
        fraction
    );
}

#[test]
fn test_internal_gain_split() {
    // Internal gains should split into convective and radiative fractions
    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    let conv_frac = model.convective_fraction;
    let rad_frac = 1.0 - conv_frac;

    // Typical split is ~60% convective, ~40% radiative
    assert!(
        conv_frac > 0.3 && conv_frac < 0.9,
        "Convective fraction should be reasonable, got {}",
        conv_frac
    );
    assert!(
        rad_frac > 0.1 && rad_frac < 0.7,
        "Radiative fraction should be reasonable, got {}",
        rad_frac
    );
}

// ============================================================================
// Section 6: Multi-Zone Node Coupling Tests
// ============================================================================

#[test]
fn test_multi_zone_mass_nodes() {
    // Multi-zone buildings should have mass nodes for each zone
    let mut model = ThermalModel::new(2);
    model.configure_6r2c_model(0.75, 100.0);

    // Both zones should have mass nodes
    assert_eq!(
        model.envelope_mass_temperatures.as_ref().len(),
        2,
        "Should have 2 envelope mass temps"
    );
    assert_eq!(
        model.internal_mass_temperatures.as_ref().len(),
        2,
        "Should have 2 internal mass temps"
    );

    // All should be initialized
    for i in 0..2 {
        assert!(
            model.envelope_mass_temperatures.as_ref()[i].is_finite(),
            "Zone {} envelope mass temp should be finite",
            i
        );
        assert!(
            model.internal_mass_temperatures.as_ref()[i].is_finite(),
            "Zone {} internal mass temp should be finite",
            i
        );
    }
}

#[test]
fn test_case_960_interzone_coupling() {
    // Case 960 (sunspace) should have inter-zone coupling
    // This test verifies the model runs without error
    let mut model = ThermalModel::new(2);
    model.configure_6r2c_model(0.75, 100.0);

    // Run simulation
    for timestep in 0..24 {
        model.step_physics(timestep, 15.0, 3600.0);
    }

    // Both zones should have finite temperatures
    for i in 0..2 {
        let t_zone = model.temperatures.as_ref()[i];
        let t_env = model.envelope_mass_temperatures.as_ref()[i];
        let t_int = model.internal_mass_temperatures.as_ref()[i];

        assert!(t_zone.is_finite(), "Zone {} temp should be finite", i);
        assert!(
            t_env.is_finite(),
            "Zone {} envelope mass should be finite",
            i
        );
        assert!(
            t_int.is_finite(),
            "Zone {} internal mass should be finite",
            i
        );
    }
}

// ============================================================================
// Section 7: Diagnostic - What Does Mass Node Represent?
// ============================================================================

#[test]
fn test_diagnostic_mass_node_response_time() {
    // DIAGNOSTIC: Measure response time of mass nodes to step change
    // This helps identify what physical temperature the node represents

    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Apply step change in outdoor temperature
    let initial_t_env = model.envelope_mass_temperatures.as_ref()[0];
    let outdoor_temp_step = 30.0; // Step from 20°C to 50°C

    // Track temperature response over time
    let mut response_curve = Vec::new();
    for timestep in 0..48 {
        model.step_physics(timestep, outdoor_temp_step, 3600.0);
        response_curve.push(model.envelope_mass_temperatures.as_ref()[0]);
    }

    // Calculate time to reach 63.2% of final value (1 time constant)
    let final_temp = response_curve[response_curve.len() - 1];
    let delta_total = final_temp - initial_t_env;
    let target = initial_t_env + 0.632 * delta_total;

    let mut tau_steps = 0;
    for (i, &t) in response_curve.iter().enumerate() {
        if t >= target {
            tau_steps = i;
            break;
        }
    }

    println!("📊 DIAGNOSTIC: Envelope mass response time");
    println!("   Initial temp: {:.1}°C", initial_t_env);
    println!("   Final temp: {:.1}°C", final_temp);
    println!(
        "   Time to 63.2%: {} hours ({} steps)",
        tau_steps, tau_steps
    );
    println!("   This approximates τ_env = C_env / Σh");

    // Response time should be positive and reasonable
    assert!(tau_steps > 0, "Response time should be > 0 hours");
    assert!(tau_steps < 48, "Response time should be < 48 hours");
}

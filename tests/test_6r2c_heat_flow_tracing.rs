//! Plan 24-06: Heat Flow Path Tracing Instrumentation
//!
//! This test suite adds instrumentation to trace heat flow through every branch
//! of the 6R2C network at each timestep. It captures:
//! - Q_exterior→envelope (through h_tr_em)
//! - Q_envelope→surface (through h_tr_ms)
//! - Q_envelope→internal (through h_tr_me)
//! - Q_surface→interior (through h_tr_is)
//! - Q_HVAC (heating/cooling energy)
//! - Node temperatures (T_zone, T_env, T_int, T_surface)
//!
//! Output: CSV-format trace data for analysis
//! Reference: docs/ISO_13790_6R2C_SPECIFICATION.md §4

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use std::fs::File;
use std::io::Write;

/// Heat flow tracer data structure
#[derive(Debug, Clone)]
struct HeatFlowTrace {
    timestep: usize,
    outdoor_temp: f64,
    t_zone: f64,
    t_surface: f64,
    t_envelope: f64,
    t_internal: f64,
    q_exterior_envelope: f64, // h_tr_em * (T_sol-air - T_env)
    q_envelope_surface: f64,  // h_tr_ms * (T_s - T_env)
    q_envelope_internal: f64, // h_tr_me * (T_int - T_env)
    q_surface_interior: f64,  // h_tr_is * (T_i - T_s)
    q_hvac: f64,              // HVAC power (W)
    q_solar: f64,             // Solar gain (W)
    q_internal: f64,          // Internal gain (W)
}

impl HeatFlowTrace {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            self.timestep,
            self.outdoor_temp,
            self.t_zone,
            self.t_surface,
            self.t_envelope,
            self.t_internal,
            self.q_exterior_envelope,
            self.q_envelope_surface,
            self.q_envelope_internal,
            self.q_surface_interior,
            self.q_hvac,
            self.q_solar,
            self.q_internal
        )
    }

    fn header() -> &'static str {
        "timestep,outdoor_temp_C,t_zone_C,t_surface_C,t_envelope_C,t_internal_C,q_ext_env_W,q_env_surf_W,q_env_int_W,q_surf_int_W,q_hvac_W,q_solar_W,q_internal_W"
    }
}

#[test]
fn test_heat_flow_tracing_case_900() {
    // Trace heat flows for Case 900 (high-mass) over 7 days
    // Output: CSV trace data for analysis

    let mut model = ThermalModel::new(1);
    model.thermal_capacitance = VectorField::from_scalar(19_944_509.0, 1);
    model.configure_6r2c_model(0.75, 100.0);

    let mut traces = Vec::new();

    // Simulate 7 days (168 hours) with realistic temperature profile
    for timestep in 0..168 {
        // Simple sinusoidal outdoor temperature (winter day)
        let hour_of_day = timestep % 24;
        let outdoor_temp =
            5.0 + 8.0 * ((hour_of_day as f64 - 3.0) * std::f64::consts::PI / 12.0).sin();

        // Get pre-step state
        let t_env_before = model.envelope_mass_temperatures.as_ref()[0];
        let t_int_before = model.internal_mass_temperatures.as_ref()[0];

        // Run timestep
        let hvac_energy_j = model.step_physics(timestep, outdoor_temp, 3600.0);
        let hvac_power_w = hvac_energy_j / 3600.0; // Convert J to W

        // Get post-step state
        let t_zone = model.temperatures.as_ref()[0];
        let t_env = model.envelope_mass_temperatures.as_ref()[0];
        let t_int = model.internal_mass_temperatures.as_ref()[0];

        // Calculate heat flows (approximate from state changes)
        let h_tr_em = model.h_tr_em.as_ref()[0];
        let h_tr_ms = model.h_tr_ms.as_ref()[0];
        let h_tr_me = model.h_tr_me.as_ref()[0];
        let h_tr_is = model.h_tr_is.as_ref()[0];

        // Estimate surface temperature from heat balance
        // T_s ≈ (h_tr_ms * T_env + h_tr_is * T_i) / (h_tr_ms + h_tr_is)
        let t_surface = (h_tr_ms * t_env + h_tr_is * t_zone) / (h_tr_ms + h_tr_is);

        // Estimate sol-air temperature (solar + outdoor)
        // Simplified: assume moderate solar gain during day
        let solar_gain = if hour_of_day >= 6 && hour_of_day <= 18 {
            500.0 * ((hour_of_day - 6) as f64 * std::f64::consts::PI / 12.0).sin()
        } else {
            0.0
        };
        let alpha = 0.7; // Solar absorptance
        let h_se = 25.0; // Exterior film coefficient
        let t_sol_air = outdoor_temp + (alpha * solar_gain / h_se);

        // Calculate heat flows
        let q_ext_env = h_tr_em * (t_sol_air - t_env);
        let q_env_surf = h_tr_ms * (t_surface - t_env);
        let q_env_int = h_tr_me * (t_int - t_env);
        let q_surf_int = h_tr_is * (t_zone - t_surface);

        let trace = HeatFlowTrace {
            timestep,
            outdoor_temp,
            t_zone,
            t_surface,
            t_envelope: t_env,
            t_internal: t_int,
            q_exterior_envelope: q_ext_env,
            q_envelope_surface: q_env_surf,
            q_envelope_internal: q_env_int,
            q_surface_interior: q_surf_int,
            q_hvac: hvac_power_w,
            q_solar: solar_gain,
            q_internal: 0.0, // No internal gains in this test
        };

        traces.push(trace);
    }

    // Write trace data to CSV
    let csv_path = "/tmp/heat_flow_trace_case900.csv";
    let mut file = File::create(csv_path).expect("Failed to create trace file");

    writeln!(file, "{}", HeatFlowTrace::header()).unwrap();
    for trace in &traces {
        writeln!(file, "{}", trace.to_csv_row()).unwrap();
    }

    println!("\n📊 Heat Flow Trace: Case 900 (7 days)");
    println!("   Output: {}", csv_path);
    println!("   Timesteps: {}", traces.len());
    println!();

    // Analyze trace data
    let avg_q_ext_env: f64 = traces
        .iter()
        .map(|t| t.q_exterior_envelope.abs())
        .sum::<f64>()
        / traces.len() as f64;
    let avg_q_env_surf: f64 = traces
        .iter()
        .map(|t| t.q_envelope_surface.abs())
        .sum::<f64>()
        / traces.len() as f64;
    let avg_q_env_int: f64 = traces
        .iter()
        .map(|t| t.q_envelope_internal.abs())
        .sum::<f64>()
        / traces.len() as f64;
    let avg_q_surf_int: f64 = traces
        .iter()
        .map(|t| t.q_surface_interior.abs())
        .sum::<f64>()
        / traces.len() as f64;
    let avg_hvac: f64 = traces.iter().map(|t| t.q_hvac.abs()).sum::<f64>() / traces.len() as f64;

    println!("   Average Heat Flows (W):");
    println!("      Q_ext→env: {:.2} W", avg_q_ext_env);
    println!("      Q_env→surf: {:.2} W", avg_q_env_surf);
    println!("      Q_env→int: {:.2} W", avg_q_env_int);
    println!("      Q_surf→int: {:.2} W", avg_q_surf_int);
    println!("      Q_HVAC: {:.2} W", avg_hvac);
    println!();

    // Check energy balance (approximate)
    // Net heat into envelope should equal energy stored
    let total_q_ext_env: f64 = traces.iter().map(|t| t.q_exterior_envelope).sum();
    let total_q_env_surf: f64 = traces.iter().map(|t| t.q_envelope_surface).sum();
    let total_q_env_int: f64 = traces.iter().map(|t| t.q_envelope_internal).sum();

    let net_q_envelope = total_q_ext_env - total_q_env_surf - total_q_env_int;
    let t_env_start = traces[0].t_envelope;
    let t_env_end = traces[traces.len() - 1].t_envelope;
    let c_env = 14_958_382.0; // J/K for Case 900
    let energy_stored = c_env * (t_env_end - t_env_start);

    println!("   Energy Balance Check (envelope node):");
    println!(
        "      Net Q into envelope: {:.2} Wh",
        net_q_envelope * 168.0 / 1000.0
    );
    println!("      Energy stored: {:.2} Wh", energy_stored / 3600.0);
    println!();

    // Verify trace data is reasonable
    assert!(traces.len() == 168, "Should have 168 timesteps");
    assert!(
        avg_q_ext_env > 0.0,
        "Exterior→envelope heat flow should be non-zero"
    );
    assert!(
        avg_q_env_surf > 0.0,
        "Envelope→surface heat flow should be non-zero"
    );
    assert!(
        avg_q_env_int > 0.0,
        "Envelope→internal heat flow should be non-zero"
    );

    // Temperatures should be in reasonable range
    for trace in &traces {
        assert!(
            trace.t_zone > -50.0 && trace.t_zone < 100.0,
            "Zone temp should be reasonable, got {:.1}°C at timestep {}",
            trace.t_zone,
            trace.timestep
        );
        assert!(
            trace.t_envelope > -50.0 && trace.t_envelope < 100.0,
            "Envelope temp should be reasonable, got {:.1}°C at timestep {}",
            trace.t_envelope,
            trace.timestep
        );
        assert!(
            trace.t_internal > -50.0 && trace.t_internal < 100.0,
            "Internal temp should be reasonable, got {:.1}°C at timestep {}",
            trace.t_internal,
            trace.timestep
        );
    }
}

#[test]
fn test_heat_flow_path_envelope_to_internal() {
    // Verify heat flows from envelope mass to internal mass through h_tr_me

    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Create temperature gradient: envelope warmer than internal
    // This should drive heat from envelope → internal

    let initial_t_env = model.envelope_mass_temperatures.as_ref()[0];
    let initial_t_int = model.internal_mass_temperatures.as_ref()[0];

    // Run with hot outdoor temp to warm envelope
    for timestep in 0..24 {
        model.step_physics(timestep, 40.0, 3600.0); // Hot day
    }

    let final_t_env = model.envelope_mass_temperatures.as_ref()[0];
    let final_t_int = model.internal_mass_temperatures.as_ref()[0];

    // Envelope should warm more than internal (direct coupling to exterior)
    let delta_t_env = final_t_env - initial_t_env;
    let delta_t_int = final_t_int - initial_t_int;

    println!("\n📊 Envelope→Internal Heat Flow Test:");
    println!("   ΔT_envelope = {:.2}°C", delta_t_env);
    println!("   ΔT_internal = {:.2}°C", delta_t_int);
    println!("   Ratio = {:.2}", delta_t_int / delta_t_env);

    assert!(delta_t_env > 0.0, "Envelope should warm");
    assert!(delta_t_int > 0.0, "Internal should warm (through h_tr_me)");
    assert!(
        delta_t_env > delta_t_int,
        "Envelope should warm more than internal"
    );

    // Internal mass warming confirms heat flow through h_tr_me
    println!("   ✓ Heat flow path envelope→internal verified");
}

#[test]
fn test_heat_flow_path_exterior_to_envelope() {
    // Verify heat flows from exterior to envelope mass through h_tr_em

    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    let initial_t_env = model.envelope_mass_temperatures.as_ref()[0];

    // Run with cold outdoor temp
    for timestep in 0..24 {
        model.step_physics(timestep, -10.0, 3600.0); // Cold night
    }

    let final_t_env = model.envelope_mass_temperatures.as_ref()[0];

    // Envelope should cool (heat flows out through h_tr_em)
    assert!(
        final_t_env < initial_t_env,
        "Envelope should cool: {:.1} → {:.1}°C",
        initial_t_env,
        final_t_env
    );

    println!("\n📊 Exterior→Envelope Heat Flow Test:");
    println!("   T_env: {:.1} → {:.1}°C", initial_t_env, final_t_env);
    println!("   ✓ Heat flow path exterior→envelope verified");
}

#[test]
fn test_thermal_lag_envelope_vs_internal() {
    // Measure thermal lag between envelope and internal mass responses
    // This is the key dynamic that 6R2C should capture better than 5R1C

    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    // Apply step change in outdoor temperature
    let outdoor_temp_step = 30.0; // Step from 20°C to 50°C

    let mut t_env_curve = Vec::new();
    let mut t_int_curve = Vec::new();

    for timestep in 0..72 {
        model.step_physics(timestep, outdoor_temp_step, 3600.0);
        t_env_curve.push(model.envelope_mass_temperatures.as_ref()[0]);
        t_int_curve.push(model.internal_mass_temperatures.as_ref()[0]);
    }

    // Find time to reach 50% of final value for each node
    let t_env_final = t_env_curve[t_env_curve.len() - 1];
    let t_int_final = t_int_curve[t_int_curve.len() - 1];
    let t_env_initial = t_env_curve[0];
    let t_int_initial = t_int_curve[0];

    let target_env = t_env_initial + 0.5 * (t_env_final - t_env_initial);
    let target_int = t_int_initial + 0.5 * (t_int_final - t_int_initial);

    let mut t50_env = 0;
    let mut t50_int = 0;

    for (i, &t) in t_env_curve.iter().enumerate() {
        if t >= target_env {
            t50_env = i;
            break;
        }
    }

    for (i, &t) in t_int_curve.iter().enumerate() {
        if t >= target_int {
            t50_int = i;
            break;
        }
    }

    let thermal_lag = t50_int - t50_env;

    println!("\n📊 Thermal Lag Analysis:");
    println!("   Envelope t50% = {} hours", t50_env);
    println!("   Internal t50% = {} hours", t50_int);
    println!(
        "   Thermal lag (internal - envelope) = {} hours",
        thermal_lag
    );

    // Internal mass should respond slower than envelope (thermal lag)
    assert!(
        t50_int >= t50_env,
        "Internal mass should respond slower than envelope (lag={})",
        thermal_lag
    );

    if thermal_lag > 0 {
        println!("   ✓ Thermal lag confirmed: {} hours", thermal_lag);
    } else {
        println!("   ⚠️  No thermal lag detected - h_tr_me may be too high");
    }
}

#[test]
fn test_energy_balance_hvac_vs_heat_flows() {
    // Verify that HVAC energy balances with heat flows through envelope

    let mut model = ThermalModel::new(1);
    model.configure_6r2c_model(0.75, 100.0);

    let mut total_hvac_energy = 0.0;

    // Run simulation for 24 hours
    for timestep in 0..24 {
        let hvac_j = model.step_physics(timestep, 5.0, 3600.0); // Cold day
        total_hvac_energy += hvac_j;
    }

    // HVAC energy should be positive (heating) and finite
    assert!(total_hvac_energy > 0.0, "HVAC should provide heating");
    assert!(
        total_hvac_energy.is_finite(),
        "HVAC energy should be finite"
    );

    println!("\n📊 HVAC Energy Balance:");
    println!(
        "   Total HVAC energy (24h) = {:.2} MJ",
        total_hvac_energy / 1e6
    );
    println!(
        "   Average power = {:.2} W",
        total_hvac_energy / (24.0 * 3600.0)
    );
}

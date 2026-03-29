//! Phase 3: Diagnostic Instrumentation for ASHRAE 140 Thermal Network Analysis
//!
//! This tool runs ASHRAE 140 validation cases with comprehensive diagnostic tracking:
//! - Conductance values (h_tr_em, h_tr_ms, h_tr_is, h_tr_w, h_ve)
//! - Energy flow breakdown by thermal path
//! - Thermal mass energy tracking
//! - Energy balance verification
//!
//! Purpose: Verify the physics errors identified in Phases 1-2:
//! - h_tr_em uses incorrect resistance subtraction formula
//! - h_tr_ms uses arbitrary 9.1 W/m²K coefficient
//! - Thermal time constant τ is ~1.1 minutes (should be 1-4 hours)
//! - Double-counting of heat flows to mass

use fluxion::sim::diagnostics::{
    ConductanceDiagnostics, EnergyFlowDiagnostics, ThermalNetworkDiagnostics,
};
use fluxion::validation::ashrae_140_validator::{Ashrae140Validator, CaseSpec, CaseType};

use std::collections::HashMap;

fn main() {
    println!("=== Phase 3: Thermal Network Physics Diagnostics ===");
    println!("Analyzing conductances, energy flows, and thermal mass behavior\n");

    // Cases to analyze (focus on problematic cases from Phases 1-2)
    let cases_to_analyze = vec![
        "600", // Low mass, heavy overprediction (+294% error)
        "620", // Low mass with night ventilation
        "650", // Low mass with high solar
        "900", // High mass, extreme overprediction (+1304% error)
        "920", // High mass, south-facing
        "940", // High mass with internal mass
    ];

    for case_id in cases_to_analyze {
        println!(
            "{} Processing Case {} {}",
            "=".repeat(60),
            case_id,
            "=".repeat(60)
        );
        analyze_case(case_id);
    }

    println!("\n=== Phase 3 Complete ===");
    println!("Diagnostic data collected. Review findings in output above.");
}

/// Analyze a single ASHRAE 140 case with full diagnostics.
fn analyze_case(case_id: &str) {
    // Create validator and get specification
    let validator = Ashrae140Validator::new();
    let case_spec = validator.get_case_spec(case_id);

    if case_spec.is_none() {
        println!("⚠️  Case {} not found", case_id);
        return;
    }

    let spec = case_spec.unwrap();
    println!("Case Type: {:?}", spec.case_type);

    // Create thermal model from spec
    let mut model = fluxion::ThermalModel::from_spec(&spec);

    // Extract conductances for diagnostics
    let num_zones = spec.num_zones;
    let conductances = if num_zones > 0 {
        ConductanceDiagnostics {
            h_tr_em: model.h_tr_em.as_ref().get(0),
            h_tr_ms: model.h_tr_ms.as_ref().get(0),
            h_tr_is: model.h_tr_is.as_ref().get(0),
            h_tr_w: model.h_tr_w.as_ref().get(0),
            h_ve: model.h_ve.as_ref().get(0),
            h_tr_floor: model.h_tr_floor.as_ref().get(0),
            thermal_capacitance: model.thermal_capacitance.as_ref().get(0),
            a_m: calculate_a_m(&spec),
            thermal_time_constant: 0.0, // Will be calculated
        }
    } else {
        ConductanceDiagnostics::default()
    };

    // Calculate thermal time constant
    let mut conductances = conductances;
    conductances.calculate_time_constant();

    // Initialize diagnostics
    let mut diagnostics = ThermalNetworkDiagnostics::new(&format!("Case {}", case_id));
    diagnostics.conductances = conductances;

    // Run simulation for one year (8760 hours)
    let num_hours = 8760;
    let dt = 3600.0; // 1 hour timestep

    println!(
        "Running {}-hour simulation with diagnostic tracking...\n",
        num_hours
    );

    for hour in 0..num_hours {
        // Get outdoor temperature
        let outdoor_temp = get_hourly_weather_temp(hour);

        // Collect diagnostic data before timestep
        let old_mass_temp = model.mass_temperatures.as_ref().get(0);
        let old_zone_temp = model.temperatures.as_ref().get(0);

        // Solve physics for one timestep
        // We'll manually track energy flows to populate diagnostics
        let hvac_output_j =
            track_energy_flows(&model, &spec, hour, outdoor_temp, dt, &mut diagnostics);

        // Update temperature tracking
        let new_mass_temp = model.mass_temperatures.as_ref().get(0);
        let new_zone_temp = model.temperatures.as_ref().get(0);

        diagnostics.update_temperatures(new_mass_temp, new_zone_temp);
    }

    // Print comprehensive diagnostic report
    diagnostics.print_report();

    // Export CSV for detailed analysis
    let csv_data = diagnostics.to_csv();
    let csv_filename = format!("diagnostics_case_{}.csv", case_id);
    if let Err(e) = std::fs::write(&csv_filename, csv_data) {
        println!("⚠️  Failed to write CSV: {}", e);
    } else {
        println!("CSV data written to: {}", csv_filename);
    }
}

/// Track energy flows for a single timestep.
///
/// This manually calculates heat flows through each thermal path to populate
/// the EnergyFlowDiagnostics struct.
fn track_energy_flows(
    model: &fluxion::ThermalModel,
    spec: &CaseSpec,
    timestep: usize,
    outdoor_temp: f64,
    dt: f64,
    diagnostics: &mut ThermalNetworkDiagnostics,
) -> f64 {
    let num_zones = spec.num_zones;
    if num_zones == 0 {
        return 0.0;
    }

    // Get model parameters
    let h_tr_em = model.h_tr_em.as_ref().get(0);
    let h_tr_ms = model.h_tr_ms.as_ref().get(0);
    let h_tr_is = model.h_tr_is.as_ref().get(0);
    let h_tr_w = model.h_tr_w.as_ref().get(0);
    let h_ve = model.h_ve.as_ref().get(0);
    let h_tr_floor = model.h_tr_floor.as_ref().get(0);

    let mass_temp = model.mass_temperatures.as_ref().get(0);
    let zone_temp = model.temperatures.as_ref().get(0);
    let thermal_cap = model.thermal_capacitance.as_ref().get(0);

    // Calculate surface temperature (Ts_free) using 5R1C formula
    // From engine.rs: ts_num_free = h_tr_ms * mass_temp + h_tr_is * t_i_free + phi_st
    // Then t_s_free = ts_num_free / (h_tr_ms + h_tr_is)

    let term_rest_1 = h_tr_ms + h_tr_is;
    let h_ext = h_tr_w + h_ve;

    // Free-floating zone temperature
    let num_tm = h_tr_ms * h_tr_is * mass_temp;

    // Get internal and solar gains
    let (internal_load_w, conv_frac, solar_gain_w) = get_gains_for_timestep(spec, timestep);
    let phi_ia = internal_load_w * conv_frac;
    let phi_rad_total = internal_load_w * (1.0 - conv_frac) + solar_gain_w;

    // Split radiative gains (using model's distribution fraction)
    let solar_to_air = 0.1; // From engine.rs model.solar_distribution_to_air
    let phi_st = phi_rad_total * solar_to_air;
    let phi_m = phi_rad_total * (1.0 - solar_to_air);

    let num_phi_st = h_tr_is * phi_st;
    let num_rest = term_rest_1 * (h_ext * outdoor_temp + phi_ia);
    let den = h_tr_ms * h_tr_is + term_rest_1 * h_ext;

    let t_i_free = (num_tm + num_phi_st + num_rest) / den;

    // Calculate surface temperature
    let ts_num_free = h_tr_ms * mass_temp + h_tr_is * t_i_free + phi_st;
    let t_s_free = ts_num_free / term_rest_1;

    // Calculate heat flows through each path
    // Q_em: exterior to mass
    let q_em = h_tr_em * (outdoor_temp - mass_temp);

    // Q_ms: mass to surface
    let q_ms = h_tr_ms * (t_s_free - mass_temp);

    // Q_is: surface to interior
    let q_is = h_tr_is * (t_s_free - t_i_free);

    // Q_w: windows
    let q_w = h_tr_w * (outdoor_temp - t_i_free);

    // Q_ve: ventilation
    let q_ve = h_ve * (outdoor_temp - t_i_free);

    // Q_floor: ground coupling
    let t_g = get_ground_temp(timestep);
    let q_floor = h_tr_floor * (t_g - t_s_free);

    // HVAC output (simplified calculation for diagnostic)
    let hvac_demand = calculate_hvac_demand(t_i_free, 20.0, 27.0);
    let hvac_output = hvac_demand * thermal_cap; // Approximate

    // Mass energy change
    let q_m_net = q_em + q_ms + phi_m;
    let dt_m = (q_m_net / thermal_cap) * dt;
    let new_mass_temp = mass_temp + dt_m;
    let mass_energy_change = thermal_cap * (new_mass_temp - mass_temp);

    // Create energy flow diagnostics
    let flow = EnergyFlowDiagnostics {
        q_em,
        q_ms,
        q_is,
        q_w,
        q_ve,
        q_floor,
        phi_st,
        phi_m,
        phi_ia,
        q_iz: 0.0, // Single zone
        hvac_output,
        mass_energy_change,
        net_energy: hvac_output * dt - mass_energy_change,
    };

    // Add to diagnostics
    diagnostics.add_hourly_flow(flow);
    diagnostics.cumulative.add_timestep(&flow, dt);

    hvac_output
}

/// Calculate HVAC demand based on temperature.
fn calculate_hvac_demand(t_i: f64, heating_sp: f64, cooling_sp: f64) -> f64 {
    if t_i < heating_sp {
        (heating_sp - t_i) / 1.0 // Heating (simplified)
    } else if t_i > cooling_sp {
        (t_i - cooling_sp) / 1.0 // Cooling (simplified)
    } else {
        0.0
    }
}

/// Get internal loads and solar gains for a timestep.
fn get_gains_for_timestep(spec: &CaseSpec, timestep: usize) -> (f64, f64, f64) {
    // Simplified gain extraction for diagnostic purposes
    // In real implementation, this would extract from validator

    let internal_load = if let Some(ref loads) = spec.internal_loads.get(0) {
        loads.total_load * 64.0 // Approximate floor area
    } else {
        0.0
    };

    let conv_fraction = if let Some(ref loads) = spec.internal_loads.get(0) {
        loads.convective_fraction
    } else {
        0.5
    };

    // Simplified solar gain profile (peak at noon)
    let hour = timestep % 24;
    let solar_base = 500.0; // Approximate peak solar
    let solar_fraction = if hour >= 6 && hour <= 18 {
        ((12.0 - (hour as f64 - 12.0).abs()) / 6.0).max(0.0)
    } else {
        0.0
    };
    let solar_gain = solar_base * solar_fraction;

    (internal_load, conv_fraction, solar_gain)
}

/// Calculate A_m factor from case specification.
fn calculate_a_m(spec: &CaseSpec) -> f64 {
    // This replicates the calculation from engine.rs
    if let Some(ref wall) = spec.construction.wall {
        let kappa = wall.iso_13790_effective_capacitance_per_area();
        let floor_area = spec.geometry.get(0).map(|g| g.floor_area()).unwrap_or(64.0);

        // Get mass class and A_m factor
        let mass_class = wall.iso_13790_mass_class();
        let a_m_factor = match mass_class {
            fluxion::sim::construction::MassClass::VeryLight => 2.5,
            fluxion::sim::construction::MassClass::Light => 2.5,
            fluxion::sim::construction::MassClass::Medium => 2.5,
            fluxion::sim::construction::MassClass::Heavy => 3.0,
            fluxion::sim::construction::MassClass::VeryHeavy => 3.5,
        };

        a_m_factor * floor_area
    } else {
        64.0 * 2.5 // Default
    }
}

/// Get hourly weather temperature (simplified Denver profile).
fn get_hourly_weather_temp(hour: usize) -> f64 {
    // Simplified Denver annual temperature profile
    // Peak summer: ~30°C, Peak winter: ~-5°C
    let day_of_year = (hour / 24) % 365;
    let season_factor = std::f64::consts::PI * 2.0 * (day_of_year as f64 / 365.0 - 0.25);
    let annual_mean = 12.0;
    let annual_amplitude = 17.5;
    let daily_factor = std::f64::consts::PI * 2.0 * (hour as f64 % 24.0 / 24.0);
    let daily_amplitude = 8.0;

    annual_mean + annual_amplitude * season_factor.cos() + daily_amplitude * daily_factor.cos()
}

/// Get ground temperature (simplified).
fn get_ground_temp(timestep: usize) -> f64 {
    // Ground temperature is roughly annual mean temperature
    12.0
}

/// Compare conductances across cases to identify patterns.
fn compare_conductances(all_diagnostics: &HashMap<String, ThermalNetworkDiagnostics>) {
    println!("\n=== Conductance Comparison Across Cases ===");
    println!(
        "{:<15} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Case", "h_tr_ms", "h_tr_is", "h_tr_w", "h_tr_em", "τ (hours)"
    );
    println!("{}", "-".repeat(70));

    for (case_id, diag) in all_diagnostics
        .iter()
        .filter(|(_, d)| d.hourly_flows.len() > 0)
        .collect::<Vec<_>>()
    {
        let cond = &diag.conductances;
        println!(
            "{:<15} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.2}",
            case_id,
            cond.h_tr_ms,
            cond.h_tr_is,
            cond.h_tr_w,
            cond.h_tr_em,
            cond.time_constant_hours()
        );
    }

    println!();
}

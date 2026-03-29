//! Diagnostic tool to trace energy flows and identify where simulation diverges
//!
//! This tool tracks:
//! 1. Hourly temperatures (Ti, Tm)
//! 2. HVAC energy (heating vs cooling)
//! 3. Solar and internal gains
//! 4. Mass energy changes
//! 5. Free-floating vs controlled comparison

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Energy Flow Trace for ASHRAE 140 Cases ===\n");

    // Test Case 600 (Low-Mass Baseline)
    println!("--- Case 600 (Low-Mass) ---");
    let spec_600 = ASHRAE140Case::Case600.spec();
    trace_energy_flows("600", &spec_600, false);

    println!();

    // Test Case 600FF (Free-Floating)
    println!("--- Case 600FF (Free-Floating) ---");
    let spec_600ff = ASHRAE140Case::Case600FF.spec();
    trace_energy_flows("600FF", &spec_600ff, true);
}

fn trace_energy_flows(
    _case_id: &str,
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
    is_free_floating: bool,
) {
    let mut model = ThermalModel::from_spec(spec);

    // Initial temperatures (should start at 20°C)
    println!("Initial State:");
    let temps = model.temperatures.as_ref();
    let mass_temps = model.mass_temperatures.as_ref();
    println!("  Ti (air temp): {:.2} °C", temps[0]);
    println!("  Tm (mass temp): {:.2} °C", mass_temps[0]);
    println!();

    // Conductances (extract before calling step_physics)
    let h_tr_w_vec = model.h_tr_w.as_ref().to_vec();
    let h_ve_vec = model.h_ve.as_ref().to_vec();
    let h_tr_is_vec = model.h_tr_is.as_ref().to_vec();
    let h_tr_ms_vec = model.h_tr_ms.as_ref().to_vec();
    let h_tr_em_vec = model.h_tr_em.as_ref().to_vec();
    let thermal_cap_vec = model.thermal_capacitance.as_ref().to_vec();
    let convective_fraction = model.convective_fraction;
    let solar_distribution_to_air = model.solar_distribution_to_air;
    let heating_setpoint = model.heating_setpoint;
    let cooling_setpoint = model.cooling_setpoint;

    println!("Conductances:");
    println!("  h_tr_w = {:.2} W/K", h_tr_w_vec[0]);
    println!("  h_ve = {:.2} W/K", h_ve_vec[0]);
    println!("  h_tr_is = {:.2} W/K", h_tr_is_vec[0]);
    println!("  h_tr_ms = {:.2} W/K", h_tr_ms_vec[0]);
    println!("  h_tr_em = {:.2} W/K", h_tr_em_vec[0]);
    println!("  Cm = {:.2e} J/K", thermal_cap_vec[0] / 1e6);
    println!();

    // Load parameters
    println!("Load Parameters:");
    println!("  Convective Fraction: {:.2}", convective_fraction);
    println!(
        "  Solar Distribution to Air: {:.2}",
        solar_distribution_to_air
    );
    println!("  Heating Setpoint: {:.1} °C", heating_setpoint);
    println!("  Cooling Setpoint: {:.1} °C", cooling_setpoint);
    println!();

    // Run 24 hours (first day) to trace energy flows
    let hours_to_trace = 24;
    let dt = 3600.0; // 1 hour in seconds

    let mut total_heating_joules = 0.0;
    let mut total_cooling_joules = 0.0;
    let mut total_hvac_joules = 0.0;
    let mut total_mass_energy_change_joules = 0.0;

    println!("=== Hourly Simulation (First 24 Hours) ===\n");

    for hour in 0..hours_to_trace {
        // Use synthetic outdoor temperature for analysis
        // Winter: -10°C, Summer: 30°C
        let outdoor_temp = if hour < 6 || hour > 18 {
            -10.0 // Winter night/morning
        } else if hour < 12 {
            0.0 // Winter day
        } else {
            10.0 // Winter afternoon
        };

        // Step physics
        let hvac_kwh = model.step_physics(hour, outdoor_temp);
        let hvac_joules = hvac_kwh * 3.6e6;

        // Separate heating vs cooling
        if hvac_joules > 0.0 {
            total_heating_joules += hvac_joules;
        } else {
            total_cooling_joules += -hvac_joules;
        }
        total_hvac_joules += hvac_joules.abs();

        // Track mass energy change
        let mass_energy_change_joules =
            model.mass_energy_change_cumulative - total_mass_energy_change_joules;
        total_mass_energy_change_joules = model.mass_energy_change_cumulative;

        // Get current state
        let ti = model.temperatures.as_ref()[0];
        let tm = model.mass_temperatures.as_ref()[0];

        // Calculate free-floating temp estimate (no HVAC)
        // ti_free should be calculated based on current mass temp
        let h_tr_ms = h_tr_ms_vec[0];
        let h_tr_is = h_tr_is_vec[0];
        let h_tr_w = h_tr_w_vec[0];
        let h_ve = h_ve_vec[0];
        let h_ext = h_tr_w + h_ve;
        let loads = model.loads.as_ref()[0] * spec.geometry[0].floor_area();
        let phi_ia = loads * convective_fraction;
        let phi_rad_total = loads * (1.0 - convective_fraction);
        let phi_st = phi_rad_total * solar_distribution_to_air;
        let phi_m = phi_rad_total * (1.0 - solar_distribution_to_air);

        let term_rest_1 = h_tr_ms + h_tr_is;
        let h_ms_is_prod = h_tr_ms * h_tr_is;
        let den = h_ms_is_prod + term_rest_1 * h_ext;
        let sensitivity = term_rest_1 / den;

        let t_i_free =
            (h_tr_ms * tm + h_tr_is * phi_st + term_rest_1 * (h_ext * outdoor_temp + phi_ia)) / den;

        // Print hourly trace
        println!("Hour {}: Outdoor={:6.1}°C  Ti={:6.2}°C  Tm={:6.2}°C  Free={:6.2}°C  HVAC={:8.2}kJ  H={:6.2}kJ  C={:6.2}kJ  ΔM={:8.1}kJ",
                  hour, outdoor_temp, ti, tm, t_i_free,
                  hvac_joules / 1000.0,
                  total_heating_joules / 1000.0,
                  total_cooling_joules / 1000.0,
                  mass_energy_change_joules / 1000.0);

        // Warn about unusual behavior
        if !is_free_floating && hvac_joules.abs() > 10000.0 {
            println!("  ⚠️  HIGH HVAC energy this hour: {:.2} kWh", hvac_kwh);
        }
        if mass_energy_change_joules.abs() > 5000.0 {
            println!(
                "  ℹ️  Large mass energy change: {:.1} kJ",
                mass_energy_change_joules / 1000.0
            );
        }
    }

    println!();
    println!("=== Summary (24 Hours) ===");
    println!("  Total Heating: {:.2} kWh", total_heating_joules / 3.6e6);
    println!("  Total Cooling: {:.2} kWh", total_cooling_joules / 3.6e6);
    println!("  Total HVAC: {:.2} kWh", total_hvac_joules / 3.6e6);
    println!(
        "  Net Mass ΔE: {:.2} MJ",
        total_mass_energy_change_joules / 1e6
    );
    println!();

    // Calculate effective heating/cooling ratio
    let total_heating_kwh = total_heating_joules / 3.6e6;
    let total_cooling_kwh = total_cooling_joules / 3.6e6;

    if total_heating_kwh > 0.0 {
        let ratio = total_cooling_kwh / total_heating_kwh;
        println!("  Cooling/Heating Ratio: {:.2}", ratio);

        if ratio > 5.0 {
            println!(
                "  ⚠️  WARNING: Cooling is {:.0}x higher than heating!",
                ratio
            );
            println!("   This suggests excessive cooling demand");
        }
    }

    println!();

    // Diagnostic analysis
    println!("=== Diagnostic Analysis ===");

    // Sensitivity check
    let h_tr_ms = h_tr_ms_vec[0];
    let h_tr_is = h_tr_is_vec[0];
    let h_tr_w = h_tr_w_vec[0];
    let h_ve = h_ve_vec[0];
    let term_rest_1 = h_tr_ms + h_tr_is;
    let h_ext = h_tr_w + h_ve;
    let h_ms_is_prod = h_tr_ms * h_tr_is;
    let den = h_ms_is_prod + term_rest_1 * h_ext;
    let sensitivity = term_rest_1 / den;

    println!("Sensitivity: {:.6} K/W", sensitivity);
    println!("  HVAC demand for 1°C error: {:.2} W", 1.0 / sensitivity);
    println!();

    // Check for potential issues
    if sensitivity < 0.001 {
        println!("⚠️  CRITICAL: Sensitivity is VERY LOW!");
        println!("   This causes excessive HVAC demand and may explain energy error");
    } else if sensitivity < 0.005 {
        println!("⚠️  WARNING: Sensitivity is LOW");
        println!("   This may contribute to energy error");
    } else {
        println!("✅ Sensitivity is in acceptable range");
    }

    println!();
}

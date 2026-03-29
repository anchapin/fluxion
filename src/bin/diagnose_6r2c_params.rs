//! 6R2C Thermal Network Parameter Diagnostic Tool
//!
//! This tool prints the actual thermal network parameters for Case 900
//! to investigate the root cause of heating overprediction.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};

fn main() {
    println!("=== 6R2C Thermal Network Parameters ===");
    println!();

    let spec = ASHRAE140Case::Case900.spec();

    println!("--- Geometry ---");
    println!(
        "Dimensions: {}m × {}m × {}m",
        spec.geometry[0].width, spec.geometry[0].depth, spec.geometry[0].height
    );
    println!("Floor Area: {:.2} m²", spec.geometry[0].floor_area());
    println!("Volume: {:.2} m³", spec.geometry[0].volume());
    println!();

    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("--- Thermal Capacitances ---");
    println!(
        "Total C_m: {:.2} MJ/K",
        model.thermal_capacitance[0] / 1_000_000.0
    );
    println!(
        "Envelope C_env: {:.2} MJ/K",
        model.envelope_thermal_capacitance[0] / 1_000_000.0
    );
    println!(
        "Internal C_int: {:.2} MJ/K",
        model.internal_thermal_capacitance[0] / 1_000_000.0
    );
    println!(
        "Split: {:.1}% envelope, {:.1}% internal",
        model.envelope_thermal_capacitance[0] / model.thermal_capacitance[0] * 100.0,
        model.internal_thermal_capacitance[0] / model.thermal_capacitance[0] * 100.0
    );
    println!();

    println!("--- Conductances (W/K) ---");
    println!("h_tr_em (envelope->exterior): {:.2} W/K", model.h_tr_em[0]);
    println!("h_tr_me (envelope->internal): {:.2} W/K", model.h_tr_me[0]);
    println!("h_tr_ms (mass->surface): {:.2} W/K", model.h_tr_ms[0]);
    println!("h_tr_is (surface->air): {:.2} W/K", model.h_tr_is[0]);
    println!("h_tr_w (windows): {:.2} W/K", model.h_tr_w[0]);
    println!("h_ve (ventilation): {:.2} W/K", model.h_ve[0]);
    println!();

    println!("--- Derived Values ---");
    println!("h_ext (h_tr_w + h_ve): {:.2} W/K", model.derived_h_ext[0]);
    println!("term_rest_1: {:.2} W/K", model.derived_term_rest_1[0]);
    println!("h_ms_is_prod: {:.2} W²/K²", model.derived_h_ms_is_prod[0]);
    println!("den: {:.2} W/K", model.derived_den[0]);
    println!("sensitivity: {:.4} K/(W/m²)", model.derived_sensitivity[0]);
    println!();

    println!("--- Time Constants ---");
    let tau_env = model.envelope_thermal_capacitance[0] / model.h_tr_em[0];
    let tau_env_hours = tau_env / 3600.0;
    println!("τ_env (C_env / h_tr_em): {:.2} hours", tau_env_hours);

    // Single mass equivalent time constant
    let tau_single = model.thermal_capacitance[0] / (model.h_tr_ms[0] + model.h_tr_is[0]);
    let tau_single_hours = tau_single / 3600.0;
    println!(
        "τ_single (C_m / (h_tr_ms + h_tr_is)): {:.2} hours",
        tau_single_hours
    );
    println!();

    println!("--- Loads ---");
    println!(
        "Internal loads (from spec): {:.2} W",
        model.loads[0] * model.zone_area[0]
    );
    println!("Internal load/m²: {:.2} W/m²", model.loads[0]);
    println!(
        "Solar distribution to air: {:.1}%",
        model.solar_distribution_to_air * 100.0
    );
    println!(
        "Convective fraction: {:.1}%",
        model.convective_fraction * 100.0
    );
    println!();

    println!("--- Case 900 Expected Values (ASHRAE 140) ---");
    println!("Heating: 1.17-2.04 MWh");
    println!("Cooling: 2.13-3.67 MWh");
    println!();
}

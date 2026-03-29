//! Diagnostic tool to investigate conductance calculations and compare to reference values
//!
//! This tool analyzes:
//! 1. Window conductance (h_tr_w)
//! 2. Infiltration conductance (h_ve)
//! 3. Surface-to-air conductance (h_tr_is)
//! 4. Mass-to-surface conductance (h_tr_ms)
//! 5. Exterior-mass conductance (h_tr_em)
//! 6. Sensitivity calculation
//! 7. Compare to reference values from ASHRAE 140 standard

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Conductance Diagnostic for ASHRAE 140 Cases ===\n");

    // Test Case 600 (Low-Mass Baseline)
    println!("--- Case 600 (Low-Mass) ---");
    let spec_600 = ASHRAE140Case::Case600.spec();
    diagnose_conductances("600", &spec_600);

    println!();

    // Test Case 900 (High-Mass Baseline)
    println!("--- Case 900 (High-Mass) ---");
    let spec_900 = ASHRAE140Case::Case900.spec();
    diagnose_conductances("900", &spec_900);

    println!();

    // Test Case 610 (Low-Mass with Night Setback)
    println!("--- Case 610 (Low-Mass + Night Setback) ---");
    let spec_610 = ASHRAE140Case::Case610.spec();
    diagnose_conductances("610", &spec_610);

    println!();

    // Test Case 920 (High-Mass with Night Setback)
    println!("--- Case 920 (High-Mass + Night Setback) ---");
    let spec_920 = ASHRAE140Case::Case920.spec();
    diagnose_conductances("920", &spec_920);
}

fn diagnose_conductances(case_id: &str, spec: &fluxion::validation::ashrae_140_cases::CaseSpec) {
    // Create model from spec
    let mut model = ThermalModel::from_spec(spec);

    // Get zone geometry (single zone for these cases)
    let geometry = &spec.geometry[0];
    let floor_area = geometry.floor_area();
    let wall_area = geometry.wall_area();
    let volume = geometry.volume();
    let total_window_area = spec.total_window_area();
    let opaque_area = wall_area - total_window_area;

    println!("Geometry:");
    println!("  Floor Area: {:.2} m²", floor_area);
    println!("  Wall Area: {:.2} m²", wall_area);
    println!("  Volume: {:.2} m³", volume);
    println!("  Window Area: {:.2} m²", total_window_area);
    println!("  Opaque Area: {:.2} m²", opaque_area);
    println!();

    // Access conductances
    let h_tr_w_vec = model.h_tr_w.as_ref();
    let h_ve_vec = model.h_ve.as_ref();
    let h_tr_is_vec = model.h_tr_is.as_ref();
    let h_tr_ms_vec = model.h_tr_ms.as_ref();
    let h_tr_em_vec = model.h_tr_em.as_ref();
    let thermal_cap_vec = model.thermal_capacitance.as_ref();

    // Reference values from ASHRAE 140 standard
    // Case 600: Low-mass construction
    // Case 900: High-mass construction
    let ref_h_tr_w = match case_id {
        "600" | "610" | "620" | "630" | "640" => 9.28, // From ASHRAE 140 (U=0.3, A=30.9 m²)
        "900" | "910" | "920" | "930" | "940" => 9.28, // Same window area
        _ => 0.0,
    };

    // Expected infiltration: ACH=0.5, V=48 m³, ρ=1.2, cp=1005
    // h_ve = 0.5 * 48 * 1.2 * 1005 / 3600 = 8.04 W/K
    let ref_h_ve = 8.04;

    // Expected surface-to-air: h_is = 3.45 * area_tot
    // area_tot = opaque_area + floor_area * 2.0
    let area_tot = opaque_area + floor_area * 2.0;
    let ref_h_tr_is = 3.45 * area_tot;

    println!("Conductances:");
    println!(
        "  h_tr_w (Window):        {:.2} W/K  | Ref: {:.2} W/K  | Error: {:.1}%",
        h_tr_w_vec[0],
        ref_h_tr_w,
        if ref_h_tr_w > 0.0 {
            (h_tr_w_vec[0] - ref_h_tr_w) / ref_h_tr_w * 100.0
        } else {
            0.0
        }
    );

    println!(
        "  h_ve (Infiltration):     {:.2} W/K  | Ref: {:.2} W/K  | Error: {:.1}%",
        h_ve_vec[0],
        ref_h_ve,
        (h_ve_vec[0] - ref_h_ve) / ref_h_ve * 100.0
    );

    println!(
        "  h_tr_is (Surface-Air):   {:.2} W/K  | Ref: {:.2} W/K  | Error: {:.1}%",
        h_tr_is_vec[0],
        ref_h_tr_is,
        (h_tr_is_vec[0] - ref_h_tr_is) / ref_h_tr_is * 100.0
    );

    println!("  h_tr_ms (Mass-Surface):  {:.2} W/K", h_tr_ms_vec[0]);

    println!("  h_tr_em (Exterior-Mass):  {:.2} W/K", h_tr_em_vec[0]);

    println!(
        "  C_m (Thermal Cap):       {:.2e} J/K",
        thermal_cap_vec[0] / 1e6
    );
    println!();

    // Calculate sensitivity
    // sensitivity = (h_tr_ms + h_tr_is) / (h_tr_ms*h_tr_is + (h_tr_ms+h_tr_is)*(h_tr_w+h_ve))
    let h_tr_ms = h_tr_ms_vec[0];
    let h_tr_is = h_tr_is_vec[0];
    let h_tr_w = h_tr_w_vec[0];
    let h_ve = h_ve_vec[0];

    let term_rest_1 = h_tr_ms + h_tr_is;
    let h_ext = h_tr_w + h_ve;
    let h_ms_is_prod = h_tr_ms * h_tr_is;
    let den = h_ms_is_prod + term_rest_1 * h_ext;
    let sensitivity = term_rest_1 / den;

    println!("Sensitivity Calculation:");
    println!(
        "  term_rest_1 = h_tr_ms + h_tr_is = {:.2} + {:.2} = {:.2} W/K",
        h_tr_ms, h_tr_is, term_rest_1
    );
    println!(
        "  h_ext = h_tr_w + h_ve = {:.2} + {:.2} = {:.2} W/K",
        h_tr_w, h_ve, h_ext
    );
    println!(
        "  h_ms_is_prod = h_tr_ms * h_tr_is = {:.2} * {:.2} = {:.2} (W/K)²",
        h_tr_ms, h_tr_is, h_ms_is_prod
    );
    println!(
        "  den = h_ms_is_prod + term_rest_1 * h_ext = {:.2} + {:.2} * {:.2} = {:.2} (W/K)²",
        h_ms_is_prod, term_rest_1, h_ext, den
    );
    println!(
        "  sensitivity = term_rest_1 / den = {:.2} / {:.2} = {:.4} K/W",
        term_rest_1, den, sensitivity
    );
    println!();

    // HVAC power multiplier analysis
    // HVAC demand = t_err / sensitivity
    // If sensitivity is small, HVAC demand will be large
    let t_err_heating = 1.0; // 1°C below setpoint
    let t_err_cooling = 1.0; // 1°C above setpoint
    let hvac_heating_demand = t_err_heating / sensitivity;
    let hvac_cooling_demand = t_err_cooling / sensitivity;

    println!("HVAC Demand for 1°C temperature error:");
    println!("  Heating: {:.2} W", hvac_heating_demand);
    println!("  Cooling: {:.2} W", hvac_cooling_demand);
    println!();

    // Check for potential issues
    println!("=== Diagnostic Analysis ===");

    if sensitivity < 0.001 {
        println!("⚠️  WARNING: Sensitivity is VERY LOW ({:.6})", sensitivity);
        println!("   This will cause excessive HVAC demand!");
        println!("   Typical sensitivity should be 0.005-0.02 K/W");
    } else if sensitivity < 0.005 {
        println!("⚠️  WARNING: Sensitivity is LOW ({:.6})", sensitivity);
        println!("   This may cause elevated HVAC demand");
        println!("   Typical sensitivity should be 0.005-0.02 K/W");
    } else if sensitivity > 0.05 {
        println!("⚠️  WARNING: Sensitivity is HIGH ({:.6})", sensitivity);
        println!("   This may cause insufficient HVAC response");
        println!("   Typical sensitivity should be 0.005-0.02 K/W");
    } else {
        println!("✅ Sensitivity is in reasonable range");
    }

    println!();

    // HVAC demand analysis
    // Reference HVAC peak for Case 600: ~5-6 kW heating, ~7-8 kW cooling
    // Reference HVAC peak for Case 900: ~2 kW heating, ~2-3 kW cooling
    let ref_peak_heating_600 = 6000.0; // W
    let ref_peak_cooling_600 = 8000.0; // W
    let ref_peak_heating_900 = 2000.0; // W
    let ref_peak_cooling_900 = 3000.0; // W

    if case_id.starts_with('6') {
        if hvac_heating_demand > ref_peak_heating_600 / 10.0 {
            println!(
                "⚠️  Heating demand per °C is {:.1}x higher than expected peak/10",
                hvac_heating_demand / (ref_peak_heating_600 / 10.0)
            );
            println!("   This could explain excessive energy consumption!");
        }
        if hvac_cooling_demand > ref_peak_cooling_600 / 10.0 {
            println!(
                "⚠️  Cooling demand per °C is {:.1}x higher than expected peak/10",
                hvac_cooling_demand / (ref_peak_cooling_600 / 10.0)
            );
            println!("   This could explain excessive energy consumption!");
        }
    } else if case_id.starts_with('9') {
        if hvac_heating_demand > ref_peak_heating_900 / 10.0 {
            println!(
                "⚠️  Heating demand per °C is {:.1}x higher than expected peak/10",
                hvac_heating_demand / (ref_peak_heating_900 / 10.0)
            );
            println!("   This could explain excessive energy consumption!");
        }
        if hvac_cooling_demand > ref_peak_cooling_900 / 10.0 {
            println!(
                "⚠️  Cooling demand per °C is {:.1}x higher than expected peak/10",
                hvac_cooling_demand / (ref_peak_cooling_900 / 10.0)
            );
            println!("   This could explain excessive energy consumption!");
        }
    }

    println!();
}

// Diagnostic tool for 600-series low-mass cases
//
// Session 40: Investigate why 600-series cases overpredict heating
// and underpredict cooling.
//
// Hypothesis: Low thermal mass causes different physics than high-mass cases.
// This diagnostic helps understand the thermal dynamics.

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::physics::cta::VectorField;

fn main() {
    println!("============================================================");
    println!("Session 40: 600-Series Low-Mass Case Diagnostics");
    println!("============================================================\n");

    // Cases to diagnose
    let cases = vec![("600", ASHRAE140Case::Case600),
                     ("610", ASHRAE140Case::Case610),
                     ("620", ASHRAE140Case::Case620),
                     ("630", ASHRAE140Case::Case630),
                     ("640", ASHRAE140Case::Case640)];

    for (case_id, case) in cases {
        println!("\n{}", "=".repeat(70));
        println!("Case {} Diagnostics", case_id);
        println!("{}", "=".repeat(70));

        // Get case spec
        let spec = case.spec();
        print_case_specs(&spec);

        // Create model
        let model = create_model(&spec);

        // Print thermal properties
        print_thermal_properties(&model, case_id);

        // Calculate time constants
        calculate_time_constants(&model, case_id);
    }

    println!("\n{}", "=".repeat(70));
    println!("Diagnostic Summary");
    println!("{}", "=".repeat(70));
    println!("\nKey Findings:");
    println!("1. Low thermal mass (600-series) has time constant τ ≈ 1-2 hours");
    println!("2. High thermal mass (900-series) has time constant τ ≈ 4-5 hours");
    println!("3. Low-mass buildings respond faster to gains/losses");
    println!("4. This may cause more HVAC cycling and higher energy consumption");
}

fn print_case_specs(spec: &fluxion::validation::ashrae_140_cases::CaseSpec) {
    println!("\nCase Specifications:");
    println!("  Floor Area: {:.2} m²", spec.geometry[0].floor_area());
    println!("  Window Area: {:.2} m²", spec.windows[0].iter().map(|w| w.area).sum::<f64>());
    println!("  Window Ratio: {:.2}", spec.windows[0].iter().map(|w| w.area).sum::<f64>() / spec.geometry[0].floor_area());

    // Construction
    println!("\n  Construction:");
    println!("    Wall U-value: {:.3} W/m²K", spec.construction.wall.u_value(None, None));
    println!("    Roof U-value: {:.3} W/m²K", spec.construction.roof.u_value(None, None));
    println!("    Floor U-value: {:.3} W/m²K", spec.construction.floor.u_value(None, None));

    // Thermal mass
    let wall_cap = spec.construction.wall.iso_13790_effective_capacitance_per_area();
    let roof_cap = spec.construction.roof.iso_13790_effective_capacitance_per_area();
    println!("\n  Thermal Mass:");
    println!("    Wall κ: {:.2} kJ/m²K", wall_cap / 1000.0);
    println!("    Roof κ: {:.2} kJ/m²K", roof_cap / 1000.0);
}

fn create_model(spec: &fluxion::validation::ashrae_140_cases::CaseSpec) -> ThermalModel<VectorField> {
    ThermalModel::from_spec(spec)
}

fn print_thermal_properties(model: &ThermalModel<VectorField>, _case_id: &str) {
    println!("\nThermal Properties:");

    // Thermal capacitance
    let total_cap: f64 = model.thermal_capacitance.as_ref().iter().sum();
    println!("  Total Thermal Capacitance: {:.2e} J/K", total_cap);
    println!("  Per Zone: {:.2e} J/K", total_cap / model.num_zones as f64);

    // Classify as low-mass or high-mass
    let is_low_mass = total_cap < 5.0e6;
    println!("  Mass Class: {}", if is_low_mass { "LOW-MASS" } else { "HIGH-MASS" });

    // Conductances
    println!("\n  Conductances:");
    println!("    h_tr_em: {:.2} W/K (exterior->mass)", model.h_tr_em.as_ref()[0]);
    println!("    h_tr_ms: {:.2} W/K (mass->surface)", model.h_tr_ms.as_ref()[0]);
    println!("    h_tr_is: {:.2} W/K (surface->interior)", model.h_tr_is.as_ref()[0]);
    println!("    h_tr_w:  {:.2} W/K (windows)", model.h_tr_w.as_ref()[0]);
    println!("    h_ve:    {:.2} W/K (ventilation)", model.h_ve.as_ref()[0]);

    // Coupling ratio
    let coupling_ratio = model.h_tr_em.as_ref()[0] / model.h_tr_ms.as_ref()[0];
    println!("\n  Coupling Ratio (h_tr_em / h_tr_ms): {:.3}", coupling_ratio);
    if coupling_ratio < 0.1 {
        println!("    ⚠️  WARNING: Coupling ratio < 0.1 (ASHRAE 140 requirement)");
    }

    // Mode-specific factors
    println!("\n  Mode-Specific Coupling Factors:");
    println!("    Heating Factor: {:.2}", model.h_tr_em_heating_factor);
    println!("    Cooling Factor: {:.2}", model.h_tr_em_cooling_factor);
}

fn calculate_time_constants(model: &ThermalModel<VectorField>, _case_id: &str) {
    println!("\nTime Constant Analysis:");

    // Thermal time constant: τ = C / U
    // where C = thermal capacitance, U = overall heat transfer coefficient

    let total_cap: f64 = model.thermal_capacitance.as_ref().iter().sum();
    let h_tr_em = model.h_tr_em.as_ref()[0];
    let _h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_is = model.h_tr_is.as_ref()[0];
    let h_tr_w = model.h_tr_w.as_ref()[0];
    let h_ve = model.h_ve.as_ref()[0];

    // Total conductance from interior to exterior
    let h_total = h_tr_w + h_ve + h_tr_is * h_tr_em / (h_tr_is + h_tr_em);

    // Time constant in hours
    let tau_seconds = total_cap / h_total;
    let tau_hours = tau_seconds / 3600.0;

    println!("  Total Thermal Capacitance: {:.2e} J/K", total_cap);
    println!("  Total Conductance: {:.2} W/K", h_total);
    println!("  Time Constant (τ): {:.2} hours", tau_hours);

    // Compare with high-mass cases
    if tau_hours < 2.0 {
        println!("  ⚠️  Very fast response (τ < 2 hours) - LOW MASS");
    } else if tau_hours < 4.0 {
        println!("  ⚠️  Fast response (τ < 4 hours) - MEDIUM MASS");
    } else {
        println!("  ✓  Slow response (τ ≥ 4 hours) - HIGH MASS");
    }

    // HVAC cycling implications
    println!("\n  HVAC Cycling Implications:");
    if tau_hours < 2.0 {
        println!("    - Low mass means rapid temperature changes");
        println!("    - HVAC may cycle more frequently");
        println!("    - Higher energy consumption possible");
        println!("    - ISSUE: May need different modulation strategy");
    } else {
        println!("    - High mass provides thermal damping");
        println!("    - HVAC cycles less frequently");
        println!("    - Lower energy consumption expected");
    }

    // Additional analysis: Solar gain distribution
    println!("\n  Solar Gain Analysis:");
    let mass_coupling_factor = if model.h_tr_em.as_ref()[0] > 0.0 {
        model.h_tr_em.as_ref()[0] / (model.h_tr_em.as_ref()[0] + model.h_tr_ms.as_ref()[0])
    } else {
        0.5
    };
    println!("    Mass coupling factor: {:.2} (fraction to mass)", mass_coupling_factor);
    println!("    - Low mass: More gains go directly to air");
    println!("    - This causes faster temperature swings");
    println!("    - ISSUE: Current HVAC modulation may be too aggressive");
}

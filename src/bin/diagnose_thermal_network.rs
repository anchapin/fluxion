//! Diagnostic tool to analyze 5R1C thermal network physics
//!
//! This tool analyzes:
//! 1. All conductances and thermal capacitances
//! 2. Thermal time constants
//! 3. Energy balance components
//! 4. h_tr_em calculation correctness
//!
//! Goal: Understand why heating energy is 4-13x higher than reference

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== 5R1C Thermal Network Analysis ===\n");

    // Analyze Case 600 (Low-Mass Baseline)
    println!("--- Case 600 (Low-Mass) ---");
    analyze_case("600", &ASHRAE140Case::Case600.spec());

    println!();

    // Analyze Case 900 (High-Mass Baseline)
    println!("--- Case 900 (High-Mass) ---");
    analyze_case("900", &ASHRAE140Case::Case900.spec());
}

fn analyze_case(case_id: &str, spec: &fluxion::validation::ashrae_140_cases::CaseSpec) {
    let mut model = ThermalModel::from_spec(spec);

    // Extract thermal parameters
    let h_tr_w = model.h_tr_w.as_ref()[0];
    let h_ve = model.h_ve.as_ref()[0];
    let h_tr_is = model.h_tr_is.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_em = model.h_tr_em.as_ref()[0];
    let cm = model.thermal_capacitance.as_ref()[0];

    println!("\n=== Thermal Network Parameters ===");
    println!("h_tr_w  (windows):          {:.2} W/K", h_tr_w);
    println!("h_ve    (infiltration):     {:.2} W/K", h_ve);
    println!("h_tr_is (surface->air):    {:.2} W/K", h_tr_is);
    println!("h_tr_ms (mass->surface):   {:.2} W/K", h_tr_ms);
    println!("h_tr_em (exterior->mass):  {:.2} W/K", h_tr_em);
    println!("C_m     (thermal mass):     {:.2} kJ/K", cm / 1000.0);

    // Calculate key derived values
    let h_ext = h_tr_w + h_ve; // Exterior conductance
    let term_rest_1 = h_tr_ms + h_tr_is; // Mass-surface-air path

    println!("\n=== Derived Conductances ===");
    println!("h_ext   (h_tr_w + h_ve):   {:.2} W/K", h_ext);
    println!("term_rest_1 (h_tr_ms + h_tr_is): {:.2} W/K", term_rest_1);

    // Thermal time constant analysis
    let tau_ms = cm / h_tr_ms; // Mass-surface time constant
    let tau_ext = cm / h_tr_em; // Mass-exterior time constant

    println!("\n=== Thermal Time Constants ===");
    println!("τ_ms (C_m / h_tr_ms):      {:.2} hours", tau_ms / 3600.0);
    println!("τ_ext (C_m / h_tr_em):     {:.2} hours", tau_ext / 3600.0);

    // Sensitivity calculation
    let h_ms_is_prod = h_tr_ms * h_tr_is;
    let den = h_ms_is_prod + term_rest_1 * h_ext;
    let sensitivity = term_rest_1 / den;

    println!("\n=== Sensitivity Analysis ===");
    println!(
        "h_ms_is_prod (h_tr_ms × h_tr_is):  {:.2e} (W/K)²",
        h_ms_is_prod
    );
    println!(
        "denominator (h_ms_is_prod + term_rest_1 × h_ext): {:.2e} (W/K)²",
        den
    );
    println!("sensitivity (term_rest_1 / den): {:.6} K/W", sensitivity);
    println!("HVAC demand for 1°C error: {:.2} W", 1.0 / sensitivity);

    // Check ASHRAE 140 reference ranges
    println!("\n=== ASHRAE 140 Reference Ranges ===");

    match case_id {
        "600" => {
            println!("Case 600 (Low-Mass):");
            println!("  Heating: 4.30-5.71 MWh (Reference)");
            println!("  Cooling: 6.14-8.45 MWh (Reference)");
        }
        "900" => {
            println!("Case 900 (High-Mass):");
            println!("  Heating: 1.17-2.04 MWh (Reference)");
            println!("  Cooling: 2.13-3.67 MWh (Reference)");
        }
        _ => {}
    }

    // Analyze h_tr_em calculation
    println!("\n=== h_tr_em Calculation Analysis ===");

    // Reconstruct the opaque conductance
    let floor_area = spec.geometry[0].floor_area();
    let wall_area = spec.geometry[0].wall_area() - 12.0; // Subtract window area (12.0 m²)
    let opaque_area = wall_area + floor_area * 2.0; // Walls + roof + floor

    let wall_u = spec.construction.wall.u_value(None);
    let roof_u = spec.construction.roof.u_value(None);
    let h_tr_op = opaque_area * wall_u + floor_area * roof_u + model.thermal_bridge_coefficient;

    println!("Opaque area: {:.2} m²", opaque_area);
    println!("Floor area:  {:.2} m²", floor_area);
    println!("Wall U-value: {:.4} W/m²K", wall_u);
    println!("Roof U-value: {:.4} W/m²K", roof_u);
    println!("h_tr_op (opaque conductance): {:.2} W/K", h_tr_op);

    // The problematic formula
    let h_ms_calc = 9.1; // W/m²K - ISO 13790
    let mass_class = spec.construction.wall.iso_13790_mass_class();
    let a_m_factor = mass_class.a_m_factor();
    let a_m = a_m_factor * floor_area;

    println!("\nISO 13790 Parameters:");
    println!("Mass class: {:?}", mass_class);
    println!("A_m factor: {:.1}", a_m_factor);
    println!("A_m (effective mass area): {:.2} m²", a_m);
    println!("h_ms (mass-surface coefficient): {:.1} W/m²K", h_ms_calc);
    println!("h_tr_ms (h_ms × A_m): {:.2} W/K", h_ms_calc * a_m);

    // Check the h_tr_em formula
    let h_tr_em_formula = 1.0 / ((1.0 / h_tr_op) - (1.0 / (h_ms_calc * a_m)));
    println!("\nh_tr_em Formula Check:");
    println!("  1 / h_tr_op = {:.6} K/W", 1.0 / h_tr_op);
    println!("  1 / (h_ms × A_m) = {:.6} K/W", 1.0 / (h_ms_calc * a_m));
    println!(
        "  Difference = {:.6} K/W",
        (1.0 / h_tr_op) - (1.0 / (h_ms_calc * a_m))
    );
    println!("  h_tr_em (calculated) = {:.2} W/K", h_tr_em_formula);
    println!("  h_tr_em (actual) = {:.2} W/K", h_tr_em);

    // Warning about subtraction
    if (1.0 / h_tr_op) < (1.0 / (h_ms_calc * a_m)) {
        println!("\n⚠️  WARNING: 1/h_tr_op < 1/(h_ms×A_m) causes negative h_tr_em!");
        println!("   This indicates a potential physics error in the formula.");
    }

    // Physics-based h_tr_em suggestion
    // For exterior to mass, should be based on thermal resistance from exterior to mass node
    // This depends on construction layers: exterior film -> materials -> interior film
    // The mass node represents the effective thermal mass of the construction

    println!("\n=== Physics-Based h_tr_em Suggestion ===");
    println!("The 5R1C model assumes:");
    println!("  1. Thermal mass is coupled to interior surface (h_tr_ms)");
    println!("  2. Exterior heat enters through opaque envelope (h_tr_op)");
    println!("  3. There is no direct exterior-to-mass path (h_tr_em = 0)");
    println!();
    println!("Current h_tr_em = {:.2} W/K", h_tr_em);
    println!("Alternative: Set h_tr_em = 0.0 (mass only coupled to surface)");

    // Run 24-hour simulation to see energy flows
    println!("\n=== 24-Hour Energy Flow Trace ===");
    run_energy_trace(&mut model, spec);
}

fn run_energy_trace(
    model: &mut ThermalModel<VectorField>,
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
) {
    let hours_to_trace = 24;
    let mut total_heating_joules = 0.0;
    let mut total_cooling_joules = 0.0;

    for hour in 0..hours_to_trace {
        // Use synthetic outdoor temperature for analysis
        let outdoor_temp = if hour < 6 || hour > 18 {
            -10.0 // Winter night
        } else if hour < 12 {
            0.0 // Winter morning
        } else {
            10.0 // Winter afternoon
        };

        let hvac_kwh = model.step_physics(hour, outdoor_temp);
        let hvac_joules = hvac_kwh * 3.6e6;

        if hvac_joules > 0.0 {
            total_heating_joules += hvac_joules;
        } else {
            total_cooling_joules += -hvac_joules;
        }
    }

    let heating_mwh = total_heating_joules / 3.6e9;
    let cooling_mwh = total_cooling_joules / 3.6e9;
    let total_mwh = heating_mwh + cooling_mwh;

    println!("24-Hour Energy:");
    println!("  Heating: {:.4} MWh", heating_mwh);
    println!("  Cooling: {:.4} MWh", cooling_mwh);
    println!("  Total:   {:.4} MWh", total_mwh);

    // Extrapolate to annual (× 365 for daily average)
    let annual_heating = heating_mwh * 365.0 / 24.0;
    let annual_cooling = cooling_mwh * 365.0 / 24.0;
    let annual_total = annual_heating + annual_cooling;

    println!("\nAnnual Extrapolation:");
    println!("  Heating: {:.2} MWh", annual_heating);
    println!("  Cooling: {:.2} MWh", annual_cooling);
    println!("  Total:   {:.2} MWh", annual_total);
}

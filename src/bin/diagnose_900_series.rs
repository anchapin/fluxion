// Compare 900-series high-mass cases with 600-series low-mass cases
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::physics::cta::VectorField;

fn main() {
    println!("============================================================");
    println!("Session 40: 900-Series vs 600-Series Comparison");
    println!("============================================================\n");

    // Compare Case 600 (low-mass) with Case 900 (high-mass)
    let cases = vec![
        ("600 (Low-Mass)", ASHRAE140Case::Case600),
        ("900 (High-Mass)", ASHRAE140Case::Case900),
    ];

    for (case_id, case) in cases {
        println!("\n{}", "=".repeat(70));
        println!("Case {} Analysis", case_id);
        println!("{}", "=".repeat(70));

        let spec = case.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);

        // Thermal capacitance
        let total_cap: f64 = model.thermal_capacitance.as_ref().iter().sum();
        println!("\nThermal Capacitance: {:.2e} J/K", total_cap);

        // Conductances
        let h_tr_em = model.h_tr_em.as_ref()[0];
        let h_tr_ms = model.h_tr_ms.as_ref()[0];
        let h_tr_is = model.h_tr_is.as_ref()[0];
        let h_tr_w = model.h_tr_w.as_ref()[0];
        let h_ve = model.h_ve.as_ref()[0];

        println!("\nConductances:");
        println!("  h_tr_em: {:.2} W/K (exterior->mass)", h_tr_em);
        println!("  h_tr_ms: {:.2} W/K (mass->surface)", h_tr_ms);
        println!("  h_tr_is: {:.2} W/K (surface->interior)", h_tr_is);
        println!("  h_tr_w:  {:.2} W/K (windows)", h_tr_w);
        println!("  h_ve:    {:.2} W/K (ventilation)", h_ve);

        // Coupling ratio
        let coupling_ratio = h_tr_em / h_tr_ms;
        println!("\nCoupling Ratio (h_tr_em / h_tr_ms): {:.3}", coupling_ratio);
        if coupling_ratio < 0.1 {
            println!("  ⚠️  BELOW ASHRAE 140 requirement (0.1)");
        } else {
            println!("  ✓  Meets ASHRAE 140 requirement (0.1)");
        }

        // Mode-specific factors
        println!("\nMode-Specific Coupling Factors:");
        println!("  Heating Factor: {:.2}", model.h_tr_em_heating_factor);
        println!("  Cooling Factor: {:.2}", model.h_tr_em_cooling_factor);

        // Time constant
        let h_total = h_tr_w + h_ve + h_tr_is * h_tr_em / (h_tr_is + h_tr_em);
        let tau_seconds = total_cap / h_total;
        let tau_hours = tau_seconds / 3600.0;
        println!("\nTime Constant (τ): {:.2} hours", tau_hours);

        // Mass coupling factor (solar gains)
        let mass_coupling = if h_tr_em > 0.0 {
            h_tr_em / (h_tr_em + h_tr_ms)
        } else {
            0.5
        };
        println!("Mass Coupling Factor: {:.2}", mass_coupling);
    }

    println!("\n{}", "=".repeat(70));
    println!("Key Differences");
    println!("{}", "=".repeat(70));
    println!("\n1. High-mass (900) has ~5x the thermal capacitance of low-mass (600)");
    println!("2. Both have coupling ratio < 0.1 (both need correction)");
    println!("3. High-mass has mode-specific factors (0.5-1.3)");
    println!("4. Low-mass has neutral factors (1.0, 1.0)");
    println!("\nISSUE: Low-mass buildings need coupling correction too!");
}

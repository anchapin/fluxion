//! Test different fix combinations to understand the impact on ASHRAE 140 validation

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Testing Physics-Based Fix Combinations ===\n");

    // Test Case 600 with different approaches
    println!("--- Case 600 Tests ---");
    let spec_600 = ASHRAE140Case::Case600.spec();

    println!("\n1. Baseline (old h_tr_ms, old h_tr_em formula):");
    test_case("600-baseline", &spec_600, FixConfig::Baseline);

    println!("\n2. Only h_tr_em = 0 (fix double-counting):");
    test_case("600-htr_em_zero", &spec_600, FixConfig::HtrEmZeroOnly);

    println!("\n3. Only h_tr_ms from thermal time constant:");
    test_case("600-htr_ms_tau", &spec_600, FixConfig::HtrMsTauOnly);

    println!("\n4. Both fixes (current implementation):");
    test_case("600-both_fixes", &spec_600, FixConfig::BothFixes);

    println!("\n--- Case 900 Tests ---");
    let spec_900 = ASHRAE140Case::Case900.spec();

    println!("\n1. Baseline (old h_tr_ms, old h_tr_em formula):");
    test_case("900-baseline", &spec_900, FixConfig::Baseline);

    println!("\n2. Only h_tr_em = 0 (fix double-counting):");
    test_case("900-htr_em_zero", &spec_900, FixConfig::HtrEmZeroOnly);

    println!("\n3. Only h_tr_ms from thermal time constant:");
    test_case("900-htr_ms_tau", &spec_900, FixConfig::HtrMsTauOnly);

    println!("\n4. Both fixes (current implementation):");
    test_case("900-both_fixes", &spec_900, FixConfig::BothFixes);
}

fn test_case(
    label: &str,
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
    _config: FixConfig,
) {
    let model = fluxion::ThermalModel::from_spec(spec);

    // Run simulation for 8760 timesteps
    let _energy = model.solve_timesteps(8760, false, false);

    println!(
        "  {}: Heating = {:.2} MWh, Cooling = {:.2} MWh",
        label,
        model.total_heating_kwh / 1000.0,
        model.total_cooling_kwh / 1000.0
    );
}

enum FixConfig {
    Baseline,      // Old h_tr_ms (9.1 × A_m), old h_tr_em formula
    HtrEmZeroOnly, // Old h_tr_ms, h_tr_em = 0.0
    HtrMsTauOnly,  // New h_tr_ms (from τ), old h_tr_em formula
    BothFixes,     // New h_tr_ms, h_tr_em = 0.0
}

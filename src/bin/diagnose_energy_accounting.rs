//! Diagnostic tool to investigate thermal mass energy accounting
//!
//! This tool tracks:
//! 1. HVAC energy vs mass energy changes
//! 2. Check thermal_mass_energy_accounting flag
//! 3. Check solar_distribution_to_air value
//! 4. Check HVAC enabled flags per zone

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Thermal Mass Energy Accounting Diagnostic ===\n");

    // Test Case 600 (Low-Mass Baseline)
    println!("--- Case 600 (Low-Mass) ---");
    let spec_600 = ASHRAE140Case::Case600.spec();
    diagnose_case("600", &spec_600);

    println!();

    // Test Case 900 (High-Mass Baseline)
    println!("--- Case 900 (High-Mass) ---");
    let spec_900 = ASHRAE140Case::Case900.spec();
    diagnose_case("900", &spec_900);

    println!();

    // Test Case 960 (Multi-Zone Sunspace)
    println!("--- Case 960 (Multi-Zone Sunspace) ---");
    let spec_960 = ASHRAE140Case::Case960.spec();
    diagnose_case("960", &spec_960);
}

fn diagnose_case(case_id: &str, spec: &fluxion::validation::ashrae_140_cases::CaseSpec) {
    // Create model from spec
    let model: ThermalModel<VectorField> = spec.into();

    // Check thermal mass energy accounting flag
    println!(
        "Thermal Mass Energy Accounting: {}",
        model.thermal_mass_energy_accounting
    );

    // Check solar distribution to air
    println!(
        "Solar Distribution to Air: {}",
        model.solar_distribution_to_air
    );

    // Check HVAC enabled per zone
    println!("HVAC Enabled per Zone:");
    model.hvac_enabled.iterate(|enabled| {
        println!(
            "  Enabled: {} ({})",
            enabled,
            if enabled > 0.5 { "Yes" } else { "No" }
        );
    });

    println!();

    // Run 10 timesteps for quick check
    let steps_to_analyze = 10;
    println!(
        "Running {} timesteps for quick analysis...",
        steps_to_analyze
    );
    println!();

    for step in 0..steps_to_analyze {
        // Store mass temperature before step
        let old_mass_temp_sum = model.mass_temperatures.iterate(|temp| temp).sum::<f64>();

        // Run one timestep
        let _energy_kwh = model.solve_timesteps(1, false, false);

        // Get mass temperature after step
        let new_mass_temp_sum = model.mass_temperatures.iterate(|temp| temp).sum::<f64>();

        // Calculate mass energy change
        let thermal_cap = model.thermal_capacitance.iterate(|cap| cap).sum::<f64>();
        let mass_energy_change = thermal_cap * (new_mass_temp_sum - old_mass_temp_sum);

        // Log details
        println!("Step {}:", step);
        println!("  Old Mass Temp (sum): {:.2} °C", old_mass_temp_sum);
        println!("  New Mass Temp (sum): {:.2} °C", new_mass_temp_sum);
        println!(
            "  Mass Energy Change: {:.2} kJ",
            mass_energy_change / 1000.0
        );
        println!(
            "  Cumulative Change: {:.2} kJ",
            model.mass_energy_change_cumulative / 1000.0
        );
        println!();
    }

    println!("=== Summary for {} ===", case_id);
    println!(
        "Final Cumulative Mass Energy Change: {:.2} MJ",
        model.mass_energy_change_cumulative / 1_000_000.0
    );
    println!();

    // If mass is net charging, HVAC is over-consuming
    let net_change_mj = model.mass_energy_change_cumulative / 1_000_000.0;
    if net_change_mj > 0.1 {
        println!(
            "⚠️  WARNING: Mass is net CHARGING by {:.2} MJ",
            net_change_mj
        );
        println!("   This suggests HVAC is over-consuming (energy stored in mass counted as consumption)");
        println!("   Thermal mass energy accounting IS enabled, but net charging indicates issue");
    } else if net_change_mj < -0.1 {
        println!(
            "ℹ️  INFO: Mass is net DISCHARGING by {:.2} MJ",
            net_change_mj.abs()
        );
        println!("   This suggests thermal mass is buffering (energy from mass is being used)");
    } else {
        println!(
            "✅ Mass energy is balanced ({:.2} MJ net change)",
            net_change_mj
        );
    }

    println!();
}

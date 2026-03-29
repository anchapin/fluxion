//! Simple diagnostic: Check thermal mass energy accounting status

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Thermal Mass Energy Accounting Status Check ===\n");

    let cases = [ASHRAE140Case::Case600, ASHRAE140Case::Case900];

    for case in cases {
        let spec = case.spec();
        println!("--- Case {} ---", spec.case_id);

        // Create model from spec
        let mut model: ThermalModel<VectorField> = ThermalModel::from_spec(&spec);

        println!(
            "Initial thermal_mass_energy_accounting: {}",
            model.thermal_mass_energy_accounting
        );
        println!(
            "Initial solar_distribution_to_air: {}",
            model.solar_distribution_to_air
        );
        println!("HVAC Enabled values: {:?}", model.hvac_enabled.as_slice());

        println!();

        // Test WITH thermal mass energy accounting ENABLED
        println!("=== Test WITH thermal mass energy accounting ENABLED ===");
        model.thermal_mass_energy_accounting = true;

        // Run simulation
        let surrogate = SurrogateManager::new().unwrap();
        let _energy = model.solve_timesteps(8760, &surrogate, false);

        // Note: ThermalModel doesn't expose individual heating/cooling energies
        // We'll need to look at the final cumulative mass energy change
        println!(
            "Final mass_energy_change_cumulative: {:.2} MJ",
            model.mass_energy_change_cumulative / 1_000_000.0
        );
        println!();

        // Reset model
        model = ThermalModel::from_spec(&spec);

        // Test WITHOUT thermal mass energy accounting (current ASHRAE 140 behavior)
        println!("=== Test WITHOUT thermal mass energy accounting (ASHRAE 140 default) ===");
        model.thermal_mass_energy_accounting = false;

        // Run simulation
        let _energy2 = model.solve_timesteps(8760, &surrogate, false);

        println!(
            "Final mass_energy_change_cumulative: {:.2} MJ",
            model.mass_energy_change_cumulative / 1_000_000.0
        );
        println!();

        // Check solar distribution (Issue #274 said this should vary)
        println!("=== Solar Distribution Check ===");
        if spec.case_id.starts_with('9') {
            println!(
                "Case {} is high-mass, expected solar_distribution_to_air = 0.5",
                spec.case_id
            );
            println!(
                "Actual solar_distribution_to_air: {}",
                model.solar_distribution_to_air
            );
            if model.solar_distribution_to_air > 0.6 {
                println!("⚠️  WARNING: solar_distribution_to_air is too high for high-mass case!");
            }
        } else {
            println!(
                "Case {} is low-mass, expected solar_distribution_to_air = 0.75",
                spec.case_id
            );
            println!(
                "Actual solar_distribution_to_air: {}",
                model.solar_distribution_to_air
            );
            if model.solar_distribution_to_air < 0.7 {
                println!("⚠️  WARNING: solar_distribution_to_air is too low for low-mass case!");
            }
        }
        println!();
    }
}

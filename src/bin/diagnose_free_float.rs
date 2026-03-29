use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Free-Floating Temperature Diagnostic ===\n");

    // Test Case 600FF (Free-Floating)
    println!("--- Case 600FF (Free-Floating, No HVAC) ---");
    let spec_600ff = ASHRAE140Case::Case600FF.spec();
    run_simulation("600FF", &spec_600ff);

    println!();

    // Test Case 900FF (Free-Floating)
    println!("--- Case 900FF (Free-Floating, No HVAC) ---");
    let spec_900ff = ASHRAE140Case::Case900FF.spec();
    run_simulation("900FF", &spec_900ff);

    println!();
}

fn run_simulation(case_id: &str, spec: &fluxion::validation::ashrae_140_cases::CaseSpec) {
    // Create model from spec
    let model = fluxion::ThermalModel::from_spec(spec);

    // Run for 8760 timesteps (1 year)
    let steps = 8760;
    let mut min_temp = f64::MAX;
    let mut max_temp = f64::MIN;

    println!("Running {} timesteps...", steps);

    for timestep in 0..steps {
        // Step physics (use surrogates=false for analytical physics)
        // We'll call solve_timesteps with single step each iteration
        let _energy = model.solve_timesteps(1, false, false);

        // Get temperature from the model (using iterate method)
        // Note: This is inefficient but we just need to check extremes
        let mut current_min = f64::MAX;
        let mut current_max = f64::MIN;
        model.temperatures.iterate(|temp| {
            current_min = current_min.min(temp);
            current_max = current_max.max(temp);
        });

        min_temp = min_temp.min(current_min);
        max_temp = max_temp.max(current_max);

        // Print progress every 1000 steps
        if timestep % 1000 == 0 {
            println!(
                "  Timestep {}: Min={:.2}°C, Max={:.2}°C",
                timestep, min_temp, max_temp
            );
        }
    }

    println!("\nResults:");
    println!("  Min Temperature: {:.2}°C", min_temp);
    println!("  Max Temperature: {:.2}°C", max_temp);
    println!("  Temperature Range: {:.2}°C", max_temp - min_temp);

    // Expected from ASHRAE 140
    if case_id == "600FF" {
        println!("\nReference (ASHRAE 140):");
        println!("  Min: -18.80 to -15.60°C");
        println!("  Max: 64.90 to 75.10°C");
    } else if case_id == "900FF" {
        println!("\nReference (ASHRAE 140):");
        println!("  Min: -6.40 to -1.60°C");
        println!("  Max: 41.80 to 46.40°C");
    }
}

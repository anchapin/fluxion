use fluxion::sim::construction::{
    exterior_film_coeff, interior_film_coeff, Assemblies, Construction, ConstructionLayer,
};

fn main() {
    // Example 1: Create a custom construction layer
    let fiberglass = ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.066);
    println!(
        "Fiberglass layer R-value: {:.4} m²K/W",
        fiberglass.r_value()
    );
    println!(
        "Fiberglass thermal capacitance: {:.2} J/m²K",
        fiberglass.thermal_capacitance_per_area()
    );

    // Example 2: Create a custom wall construction
    let custom_wall = Construction::new(vec![
        ConstructionLayer::new("Plasterboard", 0.16, 950.0, 840.0, 0.012), // Plasterboard
        ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.089),    // More insulation
        ConstructionLayer::new("Wood Siding", 0.14, 500.0, 1300.0, 0.009), // Siding
    ]);
    println!("\nCustom wall construction:");
    println!("  Total thickness: {:.3} m", custom_wall.total_thickness());
    println!(
        "  Total R-value: {:.4} m²K/W",
        custom_wall.r_value_total(None)
    );
    println!("  U-value: {:.4} W/m²K", custom_wall.u_value(None));
    println!(
        "  Thermal capacitance: {:.2} J/m²K",
        custom_wall.thermal_capacitance_per_area()
    );

    // Example 3: Use ASHRAE 140 standard assemblies
    let case_600_wall = Assemblies::low_mass_wall();
    println!("\nASHRAE 140 Case 600 (Low Mass) Wall:");
    println!("  Layers: {}", case_600_wall.layer_count());
    println!(
        "  U-value: {:.4} W/m²K (expected ~0.514)",
        case_600_wall.u_value(None)
    );
    println!("  R-value: {:.4} m²K/W", case_600_wall.r_value_total(None));

    let case_900_wall = Assemblies::high_mass_wall();
    println!("\nASHRAE 140 Case 900 (High Mass) Wall:");
    println!("  Layers: {}", case_900_wall.layer_count());
    println!("  U-value: {:.4} W/m²K", case_900_wall.u_value(None));
    println!("  R-value: {:.4} m²K/W", case_900_wall.r_value_total(None));
    println!(
        "  Thermal capacitance: {:.2} J/m²K",
        case_900_wall.thermal_capacitance_per_area()
    );

    // Example 4: Wind speed effect on U-value
    println!("\nWind speed effect on U-value (Case 600 wall):");
    for wind_speed in [0.0, 2.0, 5.0, 10.0] {
        let h_ext = exterior_film_coeff(wind_speed);
        let u = case_600_wall.u_value(Some(wind_speed));
        println!(
            "  {:.1} m/s wind → h_ext = {:.2} W/m²K → U = {:.4} W/m²K",
            wind_speed, h_ext, u
        );
    }

    // Example 5: ASHRAE film coefficients
    println!("\nASHRAE Film Coefficients:");
    println!("  Interior (h_int): {:.2} W/m²K", interior_film_coeff());
    println!(
        "  Exterior (h_ext) at 3.5 m/s: {:.2} W/m²K",
        exterior_film_coeff(3.5)
    );

    // Example 6: All ASHRAE 140 assemblies
    println!("\nASHRAE 140 Assembly U-values:");
    println!(
        "  Low mass wall: {:.4} W/m²K",
        Assemblies::low_mass_wall().u_value(None)
    );
    println!(
        "  Low mass roof: {:.4} W/m²K",
        Assemblies::low_mass_roof().u_value(None)
    );
    println!(
        "  Insulated floor: {:.4} W/m²K",
        Assemblies::insulated_floor().u_value(None)
    );
    println!(
        "  High mass wall: {:.4} W/m²K",
        Assemblies::high_mass_wall().u_value(None)
    );
    println!(
        "  High mass roof: {:.4} W/m²K",
        Assemblies::high_mass_roof().u_value(None)
    );
}

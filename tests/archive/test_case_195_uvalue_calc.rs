use fluxion::sim::construction::Assemblies;

#[test]
fn test_calculate_u_values() {
    let wall = Assemblies::low_mass_wall();
    let roof = Assemblies::low_mass_roof();
    let floor = Assemblies::insulated_floor();

    // Default film coefficients
    let h_int_wall = 7.69; // W/m²K (R_si = 0.13)
    let h_int_roof = 10.0; // W/m²K (R_si = 0.10)
    let h_int_floor = 5.88; // W/m²K (R_si = 0.17)
    let h_ext = 29.3; // W/m²K

    // Calculate R-values for each layer
    let wall_r = wall.layers.iter().map(|l| l.r_value()).sum::<f64>();
    let roof_r = roof.layers.iter().map(|l| l.r_value()).sum::<f64>();
    let floor_r = floor.layers.iter().map(|l| l.r_value()).sum::<f64>();

    println!("=== Layer R-values ===");
    println!("Wall materials R: {:.4} m²K/W", wall_r);
    println!("Roof materials R: {:.4} m²K/W", roof_r);
    println!("Floor materials R: {:.4} m²K/W", floor_r);

    let wall_r_total = 1.0 / h_int_wall + wall_r + 1.0 / h_ext;
    let roof_r_total = 1.0 / h_int_roof + roof_r + 1.0 / h_ext;
    let floor_r_total = 1.0 / h_int_floor + floor_r + 1.0 / h_ext;

    println!("\n=== Total R and U values ===");
    println!(
        "Wall: R={:.4} m²K/W, U={:.4} W/m²K",
        wall_r_total,
        1.0 / wall_r_total
    );
    println!(
        "Roof: R={:.4} m²K/W, U={:.4} W/m²K",
        roof_r_total,
        1.0 / roof_r_total
    );
    println!(
        "Floor: R={:.4} m²K/W, U={:.4} W/m²K",
        floor_r_total,
        1.0 / floor_r_total
    );

    // Check model values
    println!("\n=== Model values ===");
    println!("Wall U from model: 0.5144 W/m²K");
    println!("Roof U from model: 0.3177 W/m²K");
    println!("Floor U from model: 0.1902 W/m²K");

    // Calculate ASHRAE 140 specified U-values for Case 195
    println!("\n=== ASHRAE 140 specified U-values ===");
    println!("Wall: 0.514 W/m²K");
    println!("Roof: 0.318 W/m²K");
    println!("Floor: 0.190 W/m²K (ground coupling)");
}

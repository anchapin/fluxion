use fluxion::sim::construction::Assemblies;

#[test]
fn test_kappa_values() {
    let low_mass_wall = Assemblies::low_mass_wall();
    let high_mass_wall = Assemblies::high_mass_wall_standard();

    println!("\n=== Low Mass Wall ===");
    println!(
        "kappa: {:.0}",
        low_mass_wall.iso_13790_effective_capacitance_per_area()
    );
    println!("mass_class: {:?}", low_mass_wall.iso_13790_mass_class());
    println!(
        "a_m_factor: {}",
        low_mass_wall.iso_13790_mass_class().a_m_factor()
    );

    println!("\n=== High Mass Wall ===");
    println!(
        "kappa: {:.0}",
        high_mass_wall.iso_13790_effective_capacitance_per_area()
    );
    println!("mass_class: {:?}", high_mass_wall.iso_13790_mass_class());
    println!(
        "a_m_factor: {}",
        high_mass_wall.iso_13790_mass_class().a_m_factor()
    );
}

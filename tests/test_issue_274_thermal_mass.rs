//! Test to investigate thermal mass modeling differences between low-mass and high-mass series
//!
//! Issue #274: Investigation of thermal mass modeling differences

use fluxion::sim::construction::Assemblies;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseBuilder};

#[test]
fn test_thermal_capacitance_low_vs_high_mass() {
    let low_mass_spec = ASHRAE140Case::Case600.spec();
    let high_mass_spec = ASHRAE140Case::Case900.spec();

    // Calculate thermal capacitance per area for walls
    let low_wall_cap = low_mass_spec
        .construction
        .wall
        .thermal_capacitance_per_area();
    let high_wall_cap = high_mass_spec
        .construction
        .wall
        .thermal_capacitance_per_area();

    let low_roof_cap = low_mass_spec
        .construction
        .roof
        .thermal_capacitance_per_area();
    let high_roof_cap = high_mass_spec
        .construction
        .roof
        .thermal_capacitance_per_area();

    let low_floor_cap = low_mass_spec
        .construction
        .floor
        .thermal_capacitance_per_area();
    let high_floor_cap = high_mass_spec
        .construction
        .floor
        .thermal_capacitance_per_area();

    let floor_area = low_mass_spec.geometry[0].floor_area();
    let wall_area = low_mass_spec.geometry[0].wall_area();

    // Total thermal capacitance
    let low_total_wall = low_wall_cap * wall_area;
    let high_total_wall = high_wall_cap * wall_area;

    let low_total_roof = low_roof_cap * floor_area;
    let high_total_roof = high_roof_cap * floor_area;

    let low_total_floor = low_floor_cap * floor_area;
    let high_total_floor = high_floor_cap * floor_area;

    let low_total = low_total_wall + low_total_roof + low_total_floor;
    let high_total = high_total_wall + high_total_roof + high_total_floor;

    println!("=== Thermal Capacitance Comparison ===");
    println!("Floor Area: {} m²", floor_area);
    println!("Wall Area: {} m²", wall_area);
    println!();
    println!("Low Mass Construction:");
    println!("  Wall C/A: {:.2} kJ/m²K", low_wall_cap / 1000.0);
    println!("  Roof C/A: {:.2} kJ/m²K", low_roof_cap / 1000.0);
    println!("  Floor C/A: {:.2} kJ/m²K", low_floor_cap / 1000.0);
    println!("  Total Wall: {:.2} kJ/K", low_total_wall / 1000.0);
    println!("  Total Roof: {:.2} kJ/K", low_total_roof / 1000.0);
    println!("  Total Floor: {:.2} kJ/K", low_total_floor / 1000.0);
    println!("  Total Structure: {:.2} kJ/K", low_total / 1000.0);
    println!();
    println!("High Mass Construction:");
    println!("  Wall C/A: {:.2} kJ/m²K", high_wall_cap / 1000.0);
    println!("  Roof C/A: {:.2} kJ/m²K", high_roof_cap / 1000.0);
    println!("  Floor C/A: {:.2} kJ/m²K", high_floor_cap / 1000.0);
    println!("  Total Wall: {:.2} kJ/K", high_total_wall / 1000.0);
    println!("  Total Roof: {:.2} kJ/K", high_total_roof / 1000.0);
    println!("  Total Floor: {:.2} kJ/K", high_total_floor / 1000.0);
    println!("  Total Structure: {:.2} kJ/K", high_total / 1000.0);
    println!();
    println!("Ratio (High/Low):");
    println!("  Wall: {:.2}x", high_wall_cap / low_wall_cap);
    println!("  Roof: {:.2}x", high_roof_cap / low_roof_cap);
    println!("  Floor: {:.2}x", high_floor_cap / low_floor_cap);
    println!("  Total: {:.2}x", high_total / low_total);

    // High mass should have significantly higher thermal capacitance
    // Based on ASHRAE 140, high-mass buildings have ~3-5x more thermal mass
    assert!(
        high_total > 3.0 * low_total,
        "High-mass construction should have at least 3x thermal capacitance of low-mass"
    );
}

#[test]
fn test_solar_distribution_effect_on_thermal_mass() {
    let spec_600 = ASHRAE140Case::Case600.spec();
    let spec_900 = ASHRAE140Case::Case900.spec();

    // Both cases should have the same solar distribution setting
    // This is currently hardcoded to 0.1 in the model initialization
    // High-mass buildings should have lower solar_distribution_to_air
    // because more solar gains go to thermal mass for buffering

    println!("Solar distribution for Case 600: Not directly accessible in spec");
    println!("Solar distribution for Case 900: Not directly accessible in spec");
    println!();
    println!(
        "Note: solar_distribution_to_air is hardcoded to 0.1 in ThermalModel::from_case_spec()"
    );
    println!("This means 10% of radiative gains go to air, 90% go to thermal mass");
    println!();
    println!("Expected behavior:");
    println!("- Low-mass buildings: More solar gains to air (higher solar_distribution_to_air)");
    println!("- High-mass buildings: More solar gains to mass (lower solar_distribution_to_air)");
}

#[test]
fn test_5r1c_conductance_values() {
    // Check that the 5R1C conductances are correctly calculated
    let low_mass_spec = ASHRAE140Case::Case600.spec();
    let high_mass_spec = ASHRAE140Case::Case900.spec();

    // The 5R1C model uses these conductances:
    // h_tr_em: Exterior -> Mass
    // h_tr_ms: Mass -> Surface
    // h_tr_is: Surface -> Interior
    // h_tr_w: Exterior -> Interior (Windows)
    // h_ve: Ventilation -> Interior

    // For Case 600 and 900, the U-values should be approximately the same
    // The difference is in thermal capacitance, not conductance
    // Small differences (<1%) are acceptable due to different layer compositions
    let wall_u_low = low_mass_spec.construction.wall.u_value(None);
    let wall_u_high = high_mass_spec.construction.wall.u_value(None);
    let wall_u_diff = (wall_u_low - wall_u_high).abs() / wall_u_low;

    assert!(
        wall_u_diff < 0.01, // Allow <1% difference
        "Wall U-value difference ({:.4}%) should be < 1%",
        wall_u_diff * 100.0
    );

    let roof_u_low = low_mass_spec.construction.roof.u_value(None);
    let roof_u_high = high_mass_spec.construction.roof.u_value(None);
    let roof_u_diff = (roof_u_low - roof_u_high).abs() / roof_u_low;

    assert!(
        roof_u_diff < 0.01,
        "Roof U-value difference ({:.4}%) should be < 1%",
        roof_u_diff * 100.0
    );

    let floor_u_low = low_mass_spec.construction.floor.u_value(None);
    let floor_u_high = high_mass_spec.construction.floor.u_value(None);
    let floor_u_diff = (floor_u_low - floor_u_high).abs() / floor_u_low;

    assert!(
        floor_u_diff < 0.03, // Allow <3% difference (floor construction differs more)
        "Floor U-value difference ({:.4}%) should be < 3%",
        floor_u_diff * 100.0
    );

    println!("U-values are correctly equal between low-mass and high-mass cases");
    println!(
        "Wall U: {:.3} W/m²K",
        low_mass_spec.construction.wall.u_value(None)
    );
    println!(
        "Roof U: {:.3} W/m²K",
        low_mass_spec.construction.roof.u_value(None)
    );
    println!(
        "Floor U: {:.3} W/m²K",
        low_mass_spec.construction.floor.u_value(None)
    );
}

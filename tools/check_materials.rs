//! Diagnostic tool to check Case 900 material properties
use fluxion::sim::construction::Assemblies;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== ASHRAE 140 Case 900 Material Properties Verification ===\n");

    // Get Case 900 construction
    let spec = ASHRAE140Case::Case900.spec();
    let wall = &spec.construction.wall;
    let roof = &spec.construction.roof;
    let floor = &spec.construction.floor;

    println!("Current Fluxion Implementation:");
    println!("Wall layers:");
    for layer in &wall.layers {
        println!("  - {}: k={} W/mK, density={} kg/m³, cp={} J/kgK, thickness={}m",
                 layer.name, layer.thermal_conductivity, layer.density,
                 layer.specific_heat, layer.thickness);
    }
    println!("\nRoof layers:");
    for layer in &roof.layers {
        println!("  - {}: k={} W/mK, density={} kg/m³, cp={} J/kgK, thickness={}m",
                 layer.name, layer.thermal_conductivity, layer.density,
                 layer.specific_heat, layer.thickness);
    }
    println!("\nFloor layers:");
    for layer in &floor.layers {
        println!("  - {}: k={} W/mK, density={} kg/m³, cp={} J/kgK, thickness={}m",
                 layer.name, layer.thermal_conductivity, layer.density,
                 layer.specific_heat, layer.thickness);
    }

    // Calculate U-values
    let wall_u = 1.0 / wall.total_thermal_resistance();
    let roof_u = 1.0 / roof.total_thermal_resistance();
    let floor_u = 1.0 / floor.total_thermal_resistance();

    println!("\nCalculated U-values:");
    println!("  Wall U: {:.3} W/m²K", wall_u);
    println!("  Roof U: {:.3} W/m²K", roof_u);
    println!("  Floor U: {:.3} W/m²K", floor_u);

    println!("\n=== ASHRAE 140 Reference Values ===");
    println!("Per ASHRAE 140 User Manual (Table 7-27):");
    println!("\nWall Construction (EXT3_HW):");
    println!("  1. Concrete block: k=0.51 W/mK, thickness=0.100m, density=1400 kg/m³, cp=1000 J/kgK");
    println!("  2. Foam insulation: k=0.04 W/mK, thickness=0.0615m, density=10 kg/m³, cp=1400 J/kgK");
    println!("  3. Wood siding: k=0.16 W/mK, thickness=0.009m (from manual)");
    println!("  Expected U-value: ~0.509 W/m²K");

    println!("\nRoof Construction (ROOF_HW):");
    println!("  1. Concrete slab: k=1.13 W/mK, thickness=0.080m, density=1400 kg/m³, cp=1000 J/kgK");
    println!("  2. Foam insulation: k=0.04 W/mK, thickness=0.111m, density=10 kg/m³, cp=1400 J/kgK");
    println!("  3. Roof deck: (not specified in concrete block section)");
    println!("  Expected U-value: ~0.318 W/m²K");

    println!("\nFloor Construction (FLOOR_HW):");
    println!("  1. Concrete slab: k=1.13 W/mK, thickness=0.080m, density=1400 kg/m³, cp=1000 J/kgK");
    println!("  2. Foam insulation: k=0.04 W/mK, thickness=0.201m, density=10 kg/m³, cp=1400 J/kgK");
    println!("  Expected U-value: ~0.190 W/m²K");

    println!("\n=== CRITICAL FINDING ===");
    let wall_k = wall.layers[0].thermal_conductivity;
    println!("Current implementation uses:");
    println!("  - Concrete with k={} W/mK", wall_k);
    println!("ASHRAE 140 specifies:");
    println!("  - Concrete BLOCK with k=0.51 W/mK");
    println!("\nDifference: {:.1}% higher thermal conductivity!",
             (wall_k / 0.51 - 1.0) * 100.0);
    println!("This causes h_tr_em to be too high, resulting in:");
    println!("  - Too much heat flow through walls in winter");
    println!("  - Too much heat loss to exterior");
    println!("  - Annual heating energy over-prediction (236% above reference)");
}

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::Orientation;

#[test]
fn test_debug_case_195_model2() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Case 195 Model Debug 2 ===");

    // Calculate conductance manually
    let mut wall_area = 0.0;
    let mut wall_conductance = 0.0;
    let mut roof_area = 0.0;
    let mut roof_conductance = 0.0;
    let mut floor_area = 0.0;
    let mut floor_conductance = 0.0;

    for zone_surfaces in &model.surfaces {
        for surface in zone_surfaces {
            let conductance = surface.area * surface.u_value;
            match surface.orientation {
                Orientation::Up => {
                    roof_area = surface.area;
                    roof_conductance = conductance;
                }
                Orientation::Down => {
                    floor_area = surface.area;
                    floor_conductance = conductance;
                }
                _ => {
                    wall_area += surface.area;
                    wall_conductance += conductance;
                }
            }
            println!(
                "  {:?}: area={:.2}, u={:.4}, conductance={:.2}",
                surface.orientation, surface.area, surface.u_value, conductance
            );
        }
    }

    println!("\nManual calculation:");
    println!(
        "  Walls: {:.2} m² * {:.4} = {:.2} W/K",
        wall_area, 0.5144, wall_conductance
    );
    println!(
        "  Roof: {:.2} m² * {:.4} = {:.2} W/K",
        roof_area, 0.3177, roof_conductance
    );
    println!(
        "  Floor: {:.2} m² * {:.4} = {:.2} W/K",
        floor_area, 0.1902, floor_conductance
    );
    println!(
        "  Total: {:.2} W/K",
        wall_conductance + roof_conductance + floor_conductance
    );

    println!("\nh_tr_em from model: {:.2} W/K", model.h_tr_em[0]);
    println!("h_ve (infiltration): {:?}", model.h_ve);
    println!("h_tr_w (windows): {:?}", model.h_tr_w);
}

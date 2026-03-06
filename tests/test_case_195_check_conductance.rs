use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn check_conductance_values() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Check U-values from construction spec
    let wall_u = spec.construction.wall.u_value(None, None);
    let roof_u = spec.construction.roof.u_value(None, None);

    println!("=== U-Values from Construction ===");
    println!("wall_u_value: {:.4}", wall_u);
    println!("roof_u_value: {:.4}", roof_u);

    // Key conductances
    let h_tr_em = model.h_tr_em[0]; // External mass surface

    println!("\n=== h_tr_em ===");
    println!("Actual h_tr_em: {:.2} W/K", h_tr_em);

    // Expected: wall_u * opaque_wall_area + roof_u * roof_area
    // Case 195: 8m x 6m x 2.7m, no windows
    // Perimeter = 2*(8+6) = 28m, wall area = 28*2.7 = 75.6 m²
    let opaque_wall_area = 75.6;
    let roof_area = 48.0;
    let expected_h_tr_em = opaque_wall_area * wall_u + roof_area * roof_u;
    println!("Expected h_tr_em (manual): {:.2} W/K", expected_h_tr_em);
}

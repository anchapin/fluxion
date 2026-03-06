use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn debug_update_derived() {
    // Check what the U-values are in update_derived_parameters
    let spec = ASHRAE140Case::Case195.spec();

    // Get U-values from spec
    let wall_u = spec.construction.wall.u_value(None, None);
    let roof_u = spec.construction.roof.u_value(None, None);

    println!("=== From spec ===");
    println!("wall_u: {:.4}", wall_u);
    println!("roof_u: {:.4}", roof_u);

    // Check model U-values
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("\n=== From model ===");
    println!("wall_u_value: {:.4}", model.wall_u_value);
    println!("roof_u_value: {:.4}", model.roof_u_value);

    // Expected h_tr_em from simple calculation
    let zone_floor_area = 48.0;
    let zone_wall_area = 75.6;
    let roof_area = zone_floor_area;

    let expected_simple = zone_wall_area * model.wall_u_value + roof_area * model.roof_u_value;
    println!("\n=== Expected h_tr_em (simple) ===");
    println!(
        "h_tr_em = {:.2} * {:.4} + {:.2} * {:.4} = {:.2}",
        zone_wall_area, model.wall_u_value, roof_area, model.roof_u_value, expected_simple
    );

    println!("\n=== Actual h_tr_em ===");
    println!("h_tr_em: {:.2}", model.h_tr_em[0]);
}

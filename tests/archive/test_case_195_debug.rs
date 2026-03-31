use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn debug_case_195_h_tr_em() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Print all relevant conductances
    println!("=== Conductances after model init ===");
    println!("h_tr_w: {:.2}", model.h_tr_w[0]);
    println!("h_tr_em: {:.2}", model.h_tr_em[0]);
    println!("h_tr_ms: {:.2}", model.h_tr_ms[0]);
    println!("h_tr_is: {:.2}", model.h_tr_is[0]);
    println!("h_tr_floor: {:.2}", model.h_tr_floor[0]);
    println!("h_ve: {:.2}", model.h_ve[0]);

    // Calculate h_tr_op (opaque conductance)
    let wall_u = spec.construction.wall.u_value(None, None);
    let roof_u = spec.construction.roof.u_value(None, None);
    let zone_floor_area = 48.0;
    let zone_wall_area = 75.6; // Perimeter * height = 28 * 2.7
    let zone_window_area = 0.0; // No windows in Case 195
    let opaque_area = zone_wall_area - zone_window_area;

    println!("\n=== Manual calculation ===");
    println!("wall_u: {:.4}", wall_u);
    println!("roof_u: {:.4}", roof_u);
    println!("opaque_area: {:.2}", opaque_area);
    println!("zone_floor_area: {:.2}", zone_floor_area);

    let h_tr_op = opaque_area * wall_u + zone_floor_area * roof_u;
    println!("h_tr_op (simple): {:.2}", h_tr_op);

    // Calculate using ISO 13790 formula
    let a_m_factor = 2.0; // Light mass class
    let a_m = a_m_factor * zone_floor_area;
    let h_ms = 9.1;
    let h_ms_a_m = h_ms * a_m;
    println!("a_m: {:.2}", a_m);
    println!("h_ms * a_m: {:.2}", h_ms_a_m);

    let h_tr_em_iso = 1.0 / ((1.0 / h_tr_op) - (1.0 / h_ms_a_m));
    println!("h_tr_em (ISO formula): {:.2}", h_tr_em_iso);
}

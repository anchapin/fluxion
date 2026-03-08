use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_debug_900() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("\n=== Case 900 Debug ===");
    println!("h_tr_ms: {:.2}", model.h_tr_ms[0]);
    println!("h_tr_is: {:.2}", model.h_tr_is[0]);
    println!("h_tr_floor: {:.2}", model.h_tr_floor[0]);
    println!("derived_h_ext: {:.2}", model.derived_h_ext[0]);
    println!("derived_term_rest_1: {:.2}", model.derived_term_rest_1[0]);
    println!("derived_h_ms_is_prod: {:.2}", model.derived_h_ms_is_prod[0]);
    println!("derived_ground_coeff: {:.2}", model.derived_ground_coeff[0]);
    println!("derived_den: {:.2}", model.derived_den[0]);
    println!("derived_sensitivity: {:.4}", model.derived_sensitivity[0]);
    println!("zone_area: {:.2}", model.zone_area[0]);
    println!("num_zones: {}", model.num_zones);
}

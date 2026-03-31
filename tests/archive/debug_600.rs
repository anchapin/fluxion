use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_debug_600() {
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("\n=== Case 600 Debug ===");
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

    // Manual calculation - with derived values
    let term_rest_1 = model.derived_term_rest_1[0];
    let h_ms_is_prod = model.derived_h_ms_is_prod[0];
    let h_total = model.derived_h_ext[0];
    let ground_coeff = model.derived_ground_coeff[0];
    let den = h_ms_is_prod + term_rest_1 * h_total + ground_coeff;
    let sens = term_rest_1 / den;

    println!("\n=== Manual Calculation ===");
    println!("term_rest_1 = derived_term_rest_1 = {:.2}", term_rest_1);
    println!("h_ms_is_prod = derived_h_ms_is_prod = {:.2}", h_ms_is_prod);
    println!("h_total = derived_h_ext = {:.2}", h_total);
    println!("ground_coeff = derived_ground_coeff = {:.2}", ground_coeff);
    println!("den = h_ms_is_prod + term_rest_1 * h_total + ground_coeff");
    println!(
        "    = {:.2} + {:.2} * {:.2} + {:.2}",
        h_ms_is_prod, term_rest_1, h_total, ground_coeff
    );
    println!(
        "    = {:.2} + {:.2} + {:.2} = {:.2}",
        h_ms_is_prod,
        term_rest_1 * h_total,
        ground_coeff,
        den
    );
    println!(
        "sensitivity = term_rest_1 / den = {:.2} / {:.2} = {:.4}",
        term_rest_1, den, sens
    );
}

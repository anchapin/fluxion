use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn debug_sensitivity() {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Print derived parameters
    println!("=== Derived Parameters ===");
    println!("derived_h_ext: {:.4}", model.derived_h_ext[0]);
    println!("derived_term_rest_1: {:.4}", model.derived_term_rest_1[0]);
    println!("derived_h_ms_is_prod: {:.4}", model.derived_h_ms_is_prod[0]);
    println!("derived_ground_coeff: {:.4}", model.derived_ground_coeff[0]);
    println!("derived_den: {:.4}", model.derived_den[0]);
    println!("derived_sensitivity: {:.4}", model.derived_sensitivity[0]);

    // Expected:
    // derived_sensitivity = 0.8555
    // derived_h_ext = 57.71 (h_tr_w + h_ve + h_tr_em)

    // For Case 195:
    // - h_tr_w = 0 (no windows)
    // - h_ve = 0 (0 ACH)
    // - h_tr_em = 57.71 (walls + roof)
    println!("\n=== Individual Conductances ===");
    println!("h_tr_w: {:.4}", model.h_tr_w[0]);
    println!("h_tr_em: {:.4}", model.h_tr_em[0]);
    println!("h_tr_ms: {:.4}", model.h_tr_ms[0]);
    println!("h_tr_is: {:.4}", model.h_tr_is[0]);
    println!("h_tr_floor: {:.4}", model.h_tr_floor[0]);
    println!("h_ve: {:.4}", model.h_ve[0]);

    // Manual calculation of sensitivity
    let h_ext = model.h_tr_w[0] + model.h_ve[0] + model.h_tr_em[0];
    let term_rest_1 = model.h_tr_ms[0] + model.h_tr_is[0];
    let h_ms_is_prod = model.h_tr_ms[0] * model.h_tr_is[0];
    let ground_coeff = term_rest_1 * model.h_tr_floor[0] * 1.2;
    let den = h_ms_is_prod + term_rest_1 * h_ext + ground_coeff;
    let sensitivity = term_rest_1 / den;

    println!("\n=== Manual Calculation ===");
    println!("h_ext: {:.4}", h_ext);
    println!("term_rest_1: {:.4}", term_rest_1);
    println!("h_ms_is_prod: {:.4}", h_ms_is_prod);
    println!("ground_coeff: {:.4}", ground_coeff);
    println!("den: {:.4}", den);
    println!("sensitivity: {:.4}", sensitivity);
}

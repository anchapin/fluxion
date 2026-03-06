use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn verify_fix_correctness() {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Get conductances from model
    let h_tr_is = model.h_tr_is.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_em = model.h_tr_em[0];
    let h_tr_floor = model.h_tr_floor.as_ref()[0];
    let h_ve = model.h_ve.as_ref()[0];
    let h_tr_w = model.h_tr_w.as_ref()[0];

    println!("=== Model Conductances ===");
    println!("h_tr_is: {:.2} W/K", h_tr_is);
    println!("h_tr_ms: {:.2} W/K", h_tr_ms);
    println!("h_tr_em: {:.2} W/K", h_tr_em);
    println!("h_tr_floor: {:.2} W/K", h_tr_floor);
    println!("h_ve: {:.2} W/K", h_ve);
    println!("h_tr_w: {:.2} W/K", h_tr_w);

    // Check derived values
    println!("\n=== Derived Values ===");
    println!("derived_h_ext: {:.2} W/K", model.derived_h_ext.as_ref()[0]);
    println!(
        "derived_term_rest_1: {:.2} W/K",
        model.derived_term_rest_1.as_ref()[0]
    );
    println!(
        "derived_h_ms_is_prod: {:.2} W2/K2",
        model.derived_h_ms_is_prod.as_ref()[0]
    );
    println!(
        "derived_ground_coeff: {:.2} W/K",
        model.derived_ground_coeff.as_ref()[0]
    );
    println!("derived_den: {:.2} W2/K2", model.derived_den.as_ref()[0]);
    println!(
        "derived_sensitivity: {:.4}",
        model.derived_sensitivity.as_ref()[0]
    );

    // Manual calculation with CORRECT formula:
    // h_ext = h_tr_w + h_ve + h_tr_em = 0 + 0 + 57.71 = 57.71 W/K
    // term_rest_1 = h_tr_is + h_tr_ms = 592.02 + 873.60 = 1465.62 W/K
    // h_ms_is_prod = h_tr_is * h_tr_ms = 592.02 * 873.60 = 517188.67 W²/K²
    // ground_coeff = term_rest_1 * h_tr_floor * 1.2 = 1465.62 * 1.87 * 1.2 = 3291.11 W/K
    // den = h_ms_is_prod + term_rest_1 * h_ext + ground_coeff
    //    = 517188.67 + 1465.62 * 57.71 + 3291.11
    //    = 517188.67 + 84625.54 + 3291.11
    //    = 605105.32 W²/K²
    // sensitivity = term_rest_1 / den = 1465.62 / 605105.32 = 0.00242

    let h_ext_correct = h_tr_w + h_ve + h_tr_em; // Should be 57.71
    let term_rest_1_correct = h_tr_is + h_tr_ms; // Should be 1465.62
    let h_ms_is_prod_correct = h_tr_is * h_tr_ms; // Should be 517188.67
    let ground_coeff_correct = term_rest_1_correct * h_tr_floor * 1.2; // Should be ~3291
    let den_correct =
        h_ms_is_prod_correct + term_rest_1_correct * h_ext_correct + ground_coeff_correct;
    let sensitivity_correct = term_rest_1_correct / den_correct;

    println!("\n=== Manual Correct Calculation ===");
    println!("h_ext (correct): {:.2} W/K", h_ext_correct);
    println!("term_rest_1 (correct): {:.2} W/K", term_rest_1_correct);
    println!("h_ms_is_prod (correct): {:.2} W2/K2", h_ms_is_prod_correct);
    println!("ground_coeff (correct): {:.2} W/K", ground_coeff_correct);
    println!("den (correct): {:.2} W2/K2", den_correct);
    println!("sensitivity (correct): {:.6}", sensitivity_correct);

    // OLD (incorrect) formula that was being used:
    // h_ext_old = h_tr_w + h_ve = 0 + 0 = 0
    // den_old = h_ms_is_prod + term_rest_1 * 0 + ground_coeff
    //         = 517188. + 0 + 3291.
    //         = 520479.
    // sensitivity_old = term_rest_1 / den_old = 1465.62 / 520479 = 0.00282

    let h_ext_old = h_tr_w + h_ve; // Old (incorrect) = 0
    let den_old = h_ms_is_prod_correct + term_rest_1_correct * h_ext_old + ground_coeff_correct;
    let sensitivity_old = term_rest_1_correct / den_old;

    println!("\n=== OLD (Incorrect) Calculation ===");
    println!("h_ext (old, wrong): {:.2} W/K", h_ext_old);
    println!("den (old, wrong): {:.2} W2/K2", den_old);
    println!("sensitivity (old, wrong): {:.6}", sensitivity_old);

    // The derived_sensitivity should now match the correct value
    let model_sens = model.derived_sensitivity.as_ref()[0];
    println!("\n=== Comparison ===");
    println!("Model sensitivity: {:.6}", model_sens);
    println!("Expected (correct): {:.6}", sensitivity_correct);
    println!(
        "Match: {}",
        (model_sens - sensitivity_correct).abs() < 0.0001
    );
}

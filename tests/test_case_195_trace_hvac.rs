use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn trace_hvac_calculation() {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let t_out = -12.5;
    let timestep = 8715;

    // Get key values before stepping
    let sensitivity = model.derived_sensitivity.as_ref()[0];
    let term_rest_1 = model.derived_term_rest_1.as_ref()[0];
    let h_ms_is_prod = model.derived_h_ms_is_prod.as_ref()[0];
    let h_ext = model.derived_h_ext.as_ref()[0];
    let h_tr_floor = model.h_tr_floor.as_ref()[0];
    let ground_coeff = model.derived_ground_coeff.as_ref()[0];
    let h_tr_em = model.h_tr_em[0];

    println!("=== Input Values ===");
    println!("T_out: {:.1}C", t_out);
    println!("sensitivity: {:.6}", sensitivity);
    println!("term_rest_1: {:.2}", term_rest_1);
    println!("h_ms_is_prod: {:.2}", h_ms_is_prod);
    println!("h_ext: {:.2}", h_ext);
    println!("h_tr_floor: {:.2}", h_tr_floor);
    println!("ground_coeff: {:.2}", ground_coeff);
    println!("h_tr_em: {:.2}", h_tr_em);

    // Calculate t_i_free manually
    // First, calculate the ground temperature (simplified - use annual average)
    let t_g = 10.0; // Ground temp from test
    let h_sp = 20.0; // Heating setpoint

    // t_i_free = (h_ms_is_prod * T_m + term_rest_1 * (h_ext * T_e + h_tr_floor * T_g)) / den
    // At steady state with no internal gains and T_m ≈ T_s ≈ T_i, we can simplify
    // But actually let's look at what the model does:

    // For Case 195, T_m starts at 20C (initial condition)
    let t_m = 20.0;

    // num_tm = h_ms_is_prod * T_m = 517188.67 * 20 = 10343773
    let num_tm = h_ms_is_prod * t_m;

    // num_rest = term_rest_1 * (h_ext * T_e + h_tr_floor * T_g)
    //           = 1465.62 * (57.71 * -12.5 + 1.87 * 10.0)
    //           = 1465.62 * (-721.37 + 18.7)
    //           = 1465.62 * (-702.67)
    //           = -1030083
    let num_rest = term_rest_1 * (h_ext * t_out + h_tr_floor * t_g);

    // den = h_ms_is_prod + term_rest_1 * h_ext + ground_coeff
    //     = 517188.67 + 1465.62 * 57.71 + 3292.37
    //     = 517188.67 + 84625.54 + 3292.37
    //     = 605106.58
    let den = h_ms_is_prod + term_rest_1 * h_ext + ground_coeff;

    let t_i_free = (num_tm + num_rest) / den;

    println!("\n=== Manual Calculation ===");
    println!("num_tm (h_ms_is_prod * T_m): {:.2}", num_tm);
    println!(
        "num_rest (term_rest_1 * (h_ext * T_e + h_tr_floor * T_g)): {:.2}",
        num_rest
    );
    println!("den: {:.2}", den);
    println!("T_i_free (manual): {:.2}C", t_i_free);

    // Now calculate HVAC power
    // hvac_power = (T_set - T_i_free) / sensitivity
    let hvac_power = (h_sp - t_i_free) / sensitivity;

    println!("\n=== HVAC Power ===");
    println!("(T_set - T_i_free): {:.2}", h_sp - t_i_free);
    println!("HVAC power: {:.2} W", hvac_power);
    println!("HVAC power: {:.2} kW", hvac_power / 1000.0);

    // Run model
    let hvac_output = model.step_physics(timestep, t_out);
    println!("\n=== Model Output ===");
    println!("HVAC output: {:.2} kW", hvac_output);
    println!("Zone temp after: {:.2}C", model.temperatures.as_ref()[0]);
}

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn debug_zero_mass_formula() {
    // Simulate what happens if h_tr_ms = 0

    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let h_tr_is = model.h_tr_is[0];
    let h_tr_w = model.h_tr_w[0];
    let h_ve = model.h_ve[0];
    let h_tr_em = model.h_tr_em[0];
    let _h_tr_floor = model.h_tr_floor[0];

    // Current h_ext (includes h_tr_em)
    let h_ext_current = h_tr_w + h_ve + h_tr_em;
    println!("Current h_ext: {:.2}", h_ext_current);

    // Current formula with h_tr_ms
    let h_tr_ms = model.h_tr_ms[0];
    let term_rest_1 = h_tr_ms + h_tr_is;
    let h_ms_is_prod = h_tr_ms * h_tr_is;
    let den_current = h_ms_is_prod + term_rest_1 * h_ext_current;
    let sens_current = term_rest_1 / den_current;
    println!("Current sensitivity: {:.6}", sens_current);

    // With h_tr_ms = 0 (remove thermal mass)
    let h_tr_ms_zero = 0.0;
    let term_rest_1_zero = h_tr_ms_zero + h_tr_is;
    let h_ms_is_prod_zero = h_tr_ms_zero * h_tr_is;
    let den_zero = h_ms_is_prod_zero + term_rest_1_zero * h_ext_current;
    let sens_zero = term_rest_1_zero / den_zero;
    println!("\nWith h_tr_ms = 0:");
    println!("sensitivity: {:.6}", sens_zero);

    // But also need to bypass the mass coupling in h_tr_em
    // For zero-mass case: h_ext should be h_tr_w + h_ve + h_tr_op (not in series with mass)
    let wall_u = spec.construction.wall.u_value(None, None);
    let roof_u = spec.construction.roof.u_value(None, None);
    let zone_floor_area = 48.0;
    let zone_wall_area = 75.6;
    let zone_window_area = 0.0;
    let opaque_area = zone_wall_area - zone_window_area;
    let h_tr_op = opaque_area * wall_u + zone_floor_area * roof_u;
    let h_ext_simple = h_tr_w + h_ve + h_tr_op;

    let den_simple = term_rest_1_zero * h_ext_simple;
    let sens_simple = term_rest_1_zero / den_simple;
    println!("\nWith h_tr_ms = 0 AND h_tr_em = h_tr_op:");
    println!("sensitivity: {:.6}", sens_simple);

    // Expected: ~0.8555 according to user context
    println!("\nExpected: ~0.8555");
}

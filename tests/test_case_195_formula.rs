use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn debug_formula() {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let zone_floor_area = 48.0;
    let zone_wall_area = 75.6;
    let zone_window_area = 0.0;
    let opaque_area = zone_wall_area - zone_window_area;

    // Current approach (ISO 13790 formula)
    let wall_u = spec.construction.wall.u_value(None, None);
    let roof_u = spec.construction.roof.u_value(None, None);

    let h_tr_op = opaque_area * wall_u + zone_floor_area * roof_u;
    println!("h_tr_op (simple): {:.4}", h_tr_op);

    // Current h_tr_ms from model
    let h_tr_ms = model.h_tr_ms[0];
    let a_m = h_tr_ms / 9.1;
    println!("a_m: {:.2}", a_m);
    println!("h_tr_ms * a_m: {:.4}", h_tr_ms * a_m);

    let h_ms_a_m = h_tr_ms;
    let h_tr_em_current = 1.0 / ((1.0 / h_tr_op) - (1.0 / h_ms_a_m));
    println!("h_tr_em (ISO formula): {:.4}", h_tr_em_current);

    println!("\n=== For zero-mass limit (h_ms -> infinity) ===");
    let h_tr_em_zero_mass = h_tr_op;
    println!("h_tr_em (zero mass): {:.4}", h_tr_em_zero_mass);

    println!("\n=== Current model value ===");
    println!("h_tr_em: {:.4}", model.h_tr_em[0]);

    // The issue: with finite h_ms, the ISO formula gives h_tr_em slightly different from h_tr_op
    // But the derived_sensitivity calculation multiplies h_tr_ms * h_tr_is in the numerator
    // which gives huge values (873.6 * 592 = 517,188)

    // For low-mass case, the thermal mass shouldn't dominate the response
    // sensitivity = (h_tr_ms + h_tr_is) / den
    // If h_ms is small relative to h_tr_is, sensitivity -> 1

    println!("\n=== Sensitivity Analysis ===");
    let h_tr_is = model.h_tr_is[0];
    let h_ext = model.h_tr_w[0] + model.h_ve[0] + model.h_tr_em[0];

    // Current: uses full h_tr_ms (873.6)
    let den_current = h_tr_ms * h_tr_is + (h_tr_ms + h_tr_is) * h_ext;
    let sens_current = (h_tr_ms + h_tr_is) / den_current;
    println!("With h_tr_ms=873.6: sensitivity = {:.6}", sens_current);

    // If we set h_tr_ms = h_tr_is (equivalent mass), or higher
    let h_tr_ms_test = h_tr_is; // Equivalent
    let den_test = h_tr_ms_test * h_tr_is + (h_tr_ms_test + h_tr_is) * h_ext;
    let sens_test = (h_tr_ms_test + h_tr_is) / den_test;
    println!(
        "With h_tr_ms={:.2}: sensitivity = {:.6}",
        h_tr_is, sens_test
    );

    // If we use h_tr_em = h_tr_op directly (no mass coupling in series)
    let h_tr_em_simple = h_tr_op;
    let h_ext_simple = model.h_tr_w[0] + model.h_ve[0] + h_tr_em_simple;
    let den_simple = h_tr_ms * h_tr_is + (h_tr_ms + h_tr_is) * h_ext_simple;
    let sens_simple = (h_tr_ms + h_tr_is) / den_simple;
    println!(
        "With h_tr_em={:.2} (simple): sensitivity = {:.6}",
        h_tr_em_simple, sens_simple
    );
}

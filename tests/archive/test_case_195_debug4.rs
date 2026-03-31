use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_debug_h_tr_formula() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Debug h_tr_em formula ===");

    // Get areas
    let wall_area = 75.6;
    let roof_area = 48.0;
    let _floor_area = 48.0;
    let zone_area = 48.0;

    // Get U-values
    let wall_u = 0.5144;
    let roof_u = 0.3177;

    // Calculate h_tr_op (total opaque conductance)
    let h_tr_op = wall_area * wall_u + roof_area * roof_u;
    println!("h_tr_op (wall + roof): {:.2} W/K", h_tr_op);

    // Check model h_tr_ms
    println!("h_tr_ms from model: {:.2} W/K", model.h_tr_ms[0]);

    // For Case 195, thermal mass is effectively zero (low mass)
    // So h_ms * a_m should be ~0, making the formula problematic
    // h_tr_em = 1 / ((1/h_tr_op) - (1/(h_ms*a_m)))

    // Try series resistance only (no mass)
    let h_tr_em_series = h_tr_op;
    println!("\nh_tr_em (series only): {:.2} W/K", h_tr_em_series);
    println!("h_tr_em from model: {:.2} W/K", model.h_tr_em[0]);

    // Try alternative formula: h_tr_em = h_tr_op / 2 (for low mass)
    let h_tr_em_half = h_tr_op / 2.0;
    println!("h_tr_em (h_tr_op/2): {:.2} W/K", h_tr_em_half);

    // The issue is likely that h_tr_ms * a_m is too large
    // Let's check what a_m would give us h_tr_em = 57.71
    // 57.71 = 1 / ((1/54.14) - (1/x))
    // 1/57.71 = 1/54.14 - 1/x
    // 1/x = 1/54.14 - 1/57.71
    // 1/x = 0.01847 - 0.01733
    // 1/x = 0.00114
    // x = 877

    let required_h_ms_a_m = 1.0 / (1.0 / h_tr_op - 1.0 / model.h_tr_em[0]);
    println!(
        "\nRequired h_ms * a_m for h_tr_em={:.2}: {:.2}",
        model.h_tr_em[0], required_h_ms_a_m
    );

    // If h_ms = 9.1 (ISO 13790), then a_m = required / 9.1
    let required_a_m = required_h_ms_a_m / 9.1;
    println!("Required a_m: {:.2} m²", required_a_m);
    println!("Floor area: {:.2} m²", zone_area);
    println!("Required a_m factor: {:.2}", required_a_m / zone_area);
}

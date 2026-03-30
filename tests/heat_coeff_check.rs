//! Quick check of heat transfer coefficients for Case 600
//!
//! This test verifies the actual values of key heat loss coefficients
//! against ASHRAE 140 specifications.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_case_600_heat_coefficients() {
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("\n=== Case 600 Heat Transfer Coefficients ===");

    // Geometry
    println!("Geometry:");
    println!("  Floor area: {:.2} m²", model.zone_area[0]);
    println!("  Ceiling height: {:.2} m", model.ceiling_height[0]);
    println!("  Aspect ratio: {:.2}", model.aspect_ratio[0]);

    // Derived geometry
    let width = (model.zone_area[0] * model.aspect_ratio[0]).sqrt();
    let depth = model.zone_area[0] / width;
    let perimeter = 2.0 * (width + depth);
    let gross_wall_area = perimeter * model.ceiling_height[0];
    let window_area = gross_wall_area * model.window_ratio[0];
    let opaque_wall_area = gross_wall_area - window_area;
    let volume = model.zone_area[0] * model.ceiling_height[0];

    println!("\nDerived Geometry:");
    println!("  Width: {:.2} m", width);
    println!("  Depth: {:.2} m", depth);
    println!("  Perimeter: {:.2} m", perimeter);
    println!("  Gross wall area: {:.2} m²", gross_wall_area);
    println!("  Window area: {:.2} m²", window_area);
    println!("  Opaque wall area: {:.2} m²", opaque_wall_area);
    println!("  Volume: {:.2} m³", volume);

    // Construction U-values
    println!("\nConstruction U-values:");
    println!("  Window U: {:.3} W/m²K", model.window_u_value);
    println!("  Wall U: {:.3} W/m²K", model.wall_u_value);
    println!("  Roof U: {:.3} W/m²K", model.roof_u_value);
    println!("  Floor U: {:.3} W/m²K", model.floor_u_value);

    // Heat transfer coefficients (W/K)
    println!("\nHeat Transfer Coefficients (W/K):");
    println!("  h_tr_w (window): {:.2}", model.h_tr_w[0]);
    println!("  h_tr_em (opaque+roof): {:.2}", model.h_tr_em[0]);
    println!("  h_tr_floor (ground): {:.2}", model.h_tr_floor[0]);
    println!("  h_tr_ms (mass->surface): {:.2}", model.h_tr_ms[0]);
    println!("  h_tr_is (surface->interior): {:.2}", model.h_tr_is[0]);
    println!("  h_ve (ventilation): {:.2}", model.h_ve[0]);

    // Calculate expected values from ASHRAE 140 specs
    println!("\n=== ASHRAE 140 Expected Values ===");
    let expected_h_tr_w = 12.0 * 3.0; // 12 m² window * 3.0 W/m²K
    let expected_roof_area = 48.0; // Floor area
    let expected_h_tr_em_roof = 0.318 * 48.0; // Approx roof U * roof area

    println!(
        "Expected h_tr_w: {:.2} W/K (12 m² × 3.0 W/m²K)",
        expected_h_tr_w
    );
    println!("  Actual: {:.2} W/K", model.h_tr_w[0]);
    println!(
        "  Difference: {:.2} W/K ({:.1}%)",
        model.h_tr_w[0] - expected_h_tr_w,
        (model.h_tr_w[0] - expected_h_tr_w) / expected_h_tr_w * 100.0
    );

    // Ventilation expected
    let air_density = 1.2; // kg/m³
    let cp_air = 1000.0; // J/kg·K
    let expected_h_ve = air_density * cp_air * (0.5 * volume / 3600.0); // 0.5 ACH

    println!(
        "\nExpected h_ve: {:.2} W/K (0.5 ACH, ρcp=1200 J/m³K)",
        expected_h_ve
    );
    println!("  Actual: {:.2} W/K", model.h_ve[0]);
    println!(
        "  Difference: {:.2} W/K ({:.1}%)",
        model.h_ve[0] - expected_h_ve,
        (model.h_ve[0] - expected_h_ve) / expected_h_ve * 100.0
    );

    // Sensitivity
    println!("\nSensitivity:");
    println!("  derived_sensitivity: {:.4}", model.derived_sensitivity[0]);
    println!("  derived_term_rest_1: {:.2}", model.derived_term_rest_1[0]);
    println!("  derived_den: {:.2}", model.derived_den[0]);

    // Infiltration
    println!("\nInfiltration:");
    println!("  infiltration_rate: {:.2} ACH", model.infiltration_rate[0]);
}

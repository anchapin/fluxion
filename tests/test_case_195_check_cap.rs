use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn debug_capacitance() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Thermal Capacitance ===");
    println!(
        "thermal_capacitance: {:.0} J/K",
        model.thermal_capacitance[0]
    );

    // The issue is that even "very light" construction has thermal mass
    // For Case 195, we need essentially zero thermal mass

    // Let's calculate what the sensitivity would be with zero capacitance
    // When Cm -> 0, T_m -> T_s (no thermal mass effect)
    // The network reduces to: T_i = (h_tr_is * T_s + h_ext * T_ext) / (h_tr_is + h_ext)

    let h_tr_is = model.h_tr_is[0];
    let h_ext = model.h_tr_w[0] + model.h_ve[0] + model.h_tr_em[0];
    let h_total = h_tr_is + h_ext;

    // Zero mass limit
    let sens_zero_mass = h_tr_is / h_total;
    println!("\n=== Zero Mass Limit ===");
    println!("sensitivity: {:.4}", sens_zero_mass);

    // Current with mass
    println!("\n=== Current (with thermal mass) ===");
    println!("sensitivity: {:.4}", model.derived_sensitivity[0]);

    // The difference is dramatic! 0.91 vs 0.0024
    // The problem is in derived_sensitivity formula

    // Let's check: what should h_tr_ms be for zero mass?
    // If h_tr_ms -> infinity, then T_m -> T_s
    // But the derived_sensitivity formula includes h_tr_ms * h_tr_is in the denominator

    // Solution: For Case 195, we should set thermal capacitance to 0
    // This effectively removes the thermal mass node
}

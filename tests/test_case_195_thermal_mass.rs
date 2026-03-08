use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn debug_thermal_mass() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Check the construction thermal mass
    let wall_construction = &spec.construction.wall;
    let roof_construction = &spec.construction.roof;

    println!("=== Wall Construction ===");
    println!(
        "kappa: {:.0} J/m²K",
        wall_construction.iso_13790_effective_capacitance_per_area()
    );
    let wall_mass_class = wall_construction.iso_13790_mass_class();
    println!("mass class: {:?}", wall_mass_class);

    // Get a_m_factor from mass class
    let wall_a_m_factor = match wall_mass_class {
        fluxion::sim::construction::MassClass::VeryLight => 2.0,
        fluxion::sim::construction::MassClass::Light => 2.5,
        fluxion::sim::construction::MassClass::Medium => 3.0,
        fluxion::sim::construction::MassClass::Heavy => 3.5,
        fluxion::sim::construction::MassClass::VeryHeavy => 4.5,
    };
    println!("a_m_factor: {}", wall_a_m_factor);

    println!("\n=== Roof Construction ===");
    println!(
        "kappa: {:.0} J/m²K",
        roof_construction.iso_13790_effective_capacitance_per_area()
    );
    let roof_mass_class = roof_construction.iso_13790_mass_class();
    println!("mass class: {:?}", roof_mass_class);

    // For low-mass case, the mass temperature should track indoor air closely
    // Let's compute what happens if Cm -> 0
    // In that case: T_m -> T_s, so h_tr_ms * (T_m - T_s) -> 0
    // The network becomes: T_i = (h_tr_is * T_s + h_ext * T_ext + ...) / (h_tr_is + h_ext)

    let h_tr_is = model.h_tr_is[0];
    let h_ext = model.h_tr_w[0] + model.h_ve[0] + model.h_tr_em[0]; // = 57.71
    let h_total = h_tr_is + h_ext;

    println!("\n=== Zero Mass Limit (h_tr_ms -> infinity) ===");
    println!("h_tr_is: {:.2}", h_tr_is);
    println!("h_ext: {:.2}", h_ext);
    println!("h_total: {:.2}", h_total);
    println!("sensitivity (to T_s): {:.4}", h_tr_is / h_total);
    println!("sensitivity (to T_ext): {:.4}", h_ext / h_total);

    // Compare with actual values
    println!("\n=== Actual (5R1C with mass) ===");
    println!("derived_sensitivity: {:.4}", model.derived_sensitivity[0]);
}

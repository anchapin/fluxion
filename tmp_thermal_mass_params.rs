use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::ThermalModel;

fn main() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let model = ThermalModel::from_spec(&spec);

    println!("=== Case 900FF Thermal Mass Coupling Parameters ===");
    println!("Number of zones: {}", model.num_zones);
    println!();
    println!("Thermal capacitance (Cm):");
    for i in 0..model.num_zones.min(3) {
        println!("  Zone {}: {:.0} J/K", model.thermal_capacitance.as_ref()[i]);
    }
    println!();
    println!("Coupling conductances:");
    for i in 0..model.num_zones.min(3) {
        println!("  Zone {}: h_tr_em = {:.2} W/K, h_tr_ms = {:.2} W/K",
                i, model.h_tr_em.as_ref()[i], model.h_tr_ms.as_ref()[i]);
    }
    println!();
    println!("Other 5R1C parameters:");
    for i in 0..model.num_zones.min(3) {
        println!("  Zone {}: h_tr_is = {:.2} W/K, h_tr_w = {:.2} W/K, h_ve = {:.2} W/K",
                i, model.h_tr_is.as_ref()[i], model.h_tr_w.as_ref()[i], model.h_ve.as_ref()[i]);
    }
    println!();
    println!("Solar distribution:");
    println!("  solar_beam_to_mass_fraction: {:.2}", model.solar_beam_to_mass_fraction);
    println!("  solar_distribution_to_air: {:.2}", model.solar_distribution_to_air);
}

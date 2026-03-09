use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    
    println!("=== Case 900 Thermal Mass Coupling Parameters ===");
    println!("h_tr_em (default): {:.2} W/K", model.h_tr_em.as_ref().to_vec()[0]);
    println!("h_tr_em_heating: {:.2} W/K", model.h_tr_em_heating.as_ref().to_vec()[0]);
    println!("h_tr_em_cooling: {:.2} W/K", model.h_tr_em_cooling.as_ref().to_vec()[0]);
    println!("h_tr_em_heating_factor: {:.2}", model.h_tr_em_heating_factor);
    println!("h_tr_em_cooling_factor: {:.2}", model.h_tr_em_cooling_factor);
    println!("h_tr_ms: {:.2} W/K", model.h_tr_ms.as_ref().to_vec()[0]);
    println!("Thermal capacitance: {:.0} J/K", model.thermal_capacitance.as_ref().to_vec()[0]);
    println!("Coupling ratio (heating): {:.2}", model.h_tr_em_heating.as_ref().to_vec()[0] / model.h_tr_ms.as_ref().to_vec()[0]);
    println!("Coupling ratio (cooling): {:.2}", model.h_tr_em_cooling.as_ref().to_vec()[0] / model.h_tr_ms.as_ref().to_vec()[0]);
}

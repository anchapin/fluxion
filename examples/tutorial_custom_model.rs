//! Tutorial: Creating a Custom Thermal Model
//!
//! This example demonstrates how to create a custom thermal model
//! by extending the base ThermalModel with custom parameters.
//!
//! Note: This is a simplified version. The original example referenced
//! APIs that have been refactored.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Custom Thermal Model Tutorial ===\n");

    // Step 1: Create a base model from ASHRAE 140 Case 600
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("Base model created:");
    println!("  Zones: {}", model.num_zones);
    println!("  h_tr_em: {:.2} W/K", model.h_tr_em.as_ref()[0]);
    println!("  h_tr_ms: {:.2} W/K", model.h_tr_ms.as_ref()[0]);
    println!("  h_tr_is: {:.2} W/K", model.h_tr_is.as_ref()[0]);

    // Step 2: Access thermal capacitance
    let thermal_cap = model.thermal_capacitance.as_ref()[0];
    println!("  Thermal capacitance: {:.0} J/K", thermal_cap);

    println!("\n=== Tutorial Complete ===");
    println!("For more advanced customization, see the API documentation.");
}

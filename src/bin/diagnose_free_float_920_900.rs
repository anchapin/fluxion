//! Compare free-floating temperatures between Case 900 and 920
//!
//! This diagnostic checks if the free-floating temperature calculation
//! is working correctly for E/W vs South window configurations.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Free-Floating Temperature Comparison ===");
    println!("Comparing Case 900 (South) vs Case 920 (E/W)");
    println!();

    // Get case specifications
    let spec_900 = ASHRAE140Case::Case900.spec();
    let spec_920 = ASHRAE140Case::Case920.spec();

    println!("Case 900: {} ({})", spec_900.case_id, spec_900.description);
    println!("  Windows: South-facing 12m²");
    println!();

    println!("Case 920: {} ({})", spec_920.case_id, spec_920.description);
    println!("  Windows: East 6m² + West 6m²");
    println!();

    // Create models
    let mut model_900 = ThermalModel::<VectorField>::from_spec(&spec_900);
    let mut model_920 = ThermalModel::<VectorField>::from_spec(&spec_920);

    // Disable HVAC to test free-floating
    model_900.heating_setpoint = -999.0;
    model_900.cooling_setpoint = 999.0;
    model_920.heating_setpoint = -999.0;
    model_920.cooling_setpoint = 999.0;

    // Simulate a hot summer day (June 21, hour 12:00)
    let hour = 12;
    let outdoor_temp = 35.0; // Hot day

    // Manually calculate free-floating temps
    let ti_free_900 = model_900.calculate_free_float_temperature(hour, outdoor_temp);
    let ti_free_920 = model_920.calculate_free_float_temperature(hour, outdoor_temp);

    println!("=== Free-Floating Temperature at Hour {} ===", hour);
    println!("Outdoor temp: {}°C", outdoor_temp);
    println!();
    println!("Case 900 (South): {:.2}°C", ti_free_900);
    println!("Case 920 (E/W):   {:.2}°C", ti_free_920);
    println!();

    // Calculate expected cooling demand
    let cooling_setpoint = 27.0; // ASHRAE 140 standard cooling setpoint
    let cooling_excess_900 = ti_free_900 - cooling_setpoint;
    let cooling_excess_920 = ti_free_920 - cooling_setpoint;

    println!("=== Cooling Demand Analysis ===");
    println!("Cooling setpoint: {}°C", cooling_setpoint);
    println!();
    println!("Case 900 temp excess: {:.2}°C", cooling_excess_900);
    println!("Case 920 temp excess: {:.2}°C", cooling_excess_920);
    println!();

    if cooling_excess_900 > 0.0 && cooling_excess_920 > 0.0 {
        println!(
            "Ratio (920/900): {:.2}",
            cooling_excess_920 / cooling_excess_900
        );
        println!();
        println!("Expected cooling load ratio should match this temperature excess ratio");
    } else if cooling_excess_900 > 0.0 {
        println!("Case 900 needs cooling, Case 920 does not!");
    } else if cooling_excess_920 > 0.0 {
        println!("Case 920 needs cooling, Case 900 does not!");
    } else {
        println!("Neither case needs cooling at this timestep");
    }

    println!();
    println!("=== Key Insight ===");
    println!("If Case 920 has lower free-floating temps than Case 900,");
    println!("it will have lower cooling demand.");
    println!();
    println!("Possible causes:");
    println!("1. Solar gain timing (E vs S) affecting thermal mass");
    println!("2. Surface area differences (more exposed walls for E/W)");
    println!("3. View factors affecting solar distribution");
}

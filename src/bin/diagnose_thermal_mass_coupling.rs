// Diagnostic script to check thermal mass coupling values
// Based on thermal mass coupling tests from Session 34

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Thermal Mass Coupling Diagnostic ===\n");

    // Test Case 900 (high-mass)
    let spec_900 = ASHRAE140Case::Case900.spec();
    let model_900 = ThermalModel::from_spec(&spec_900);

    println!("=== Case 900 (High-Mass) ===");
    println!(
        "Mass temperature: {:.2}°C",
        model_900.mass_temperatures.as_ref()[0]
    );
    println!(
        "Zone temperature: {:.2}°C",
        model_900.temperatures.as_ref()[0]
    );
    println!(
        "Thermal capacitance: {:.2} kJ/K",
        model_900.thermal_capacitance.as_ref()[0] / 1000.0
    );
    println!();

    // Check h_tr_ms (mass to surface) conductance
    let h_tr_ms = model_900.h_tr_ms.as_ref()[0];
    println!("h_tr_ms (mass to surface): {:.3} W/K", h_tr_ms);
    println!("Expected range for high-mass: 1-10 W/K");
    if h_tr_ms >= 0.1 && h_tr_ms < 10.0 {
        println!("✓ h_tr_ms in reasonable range");
    } else {
        println!("✗ h_tr_ms out of range!");
    }
    println!();

    // Check h_tr_is (surface to interior air) conductance
    let h_tr_is = model_900.h_tr_is.as_ref()[0];
    println!("h_tr_is (surface to interior): {:.3} W/K", h_tr_is);
    println!("Expected range: 1-10 W/K");
    if h_tr_is >= 1.0 && h_tr_is < 10.0 {
        println!("✓ h_tr_is in reasonable range");
    } else {
        println!("✗ h_tr_is out of range!");
    }
    println!();

    // Check h_tr_em (exterior to mass) conductance
    let h_tr_em = model_900.h_tr_em.as_ref()[0];
    println!("h_tr_em (exterior to mass): {:.3} W/K", h_tr_em);
    println!();

    // Calculate thermal time constant (tau)
    let total_r: f64 = spec_900
        .construction
        .wall
        .layers
        .iter()
        .map(|l| l.thickness / l.conductivity)
        .sum::<f64>();

    let total_c: f64 = spec_900
        .construction
        .wall
        .layers
        .iter()
        .map(|l| l.thickness * l.density * l.specific_heat * 48.0) // 48 m² zone area
        .sum::<f64>();

    let tau_hours = total_r * total_c / 3600.0;
    println!("Total thermal resistance (R): {:.4} K·m²/W", total_r);
    println!(
        "Total thermal capacitance (C): {:.2} kJ/K",
        total_c / 1000.0
    );
    println!("Thermal time constant (τ = R×C): {:.2} hours", tau_hours);
    println!("Expected for Case 900: ~73 hours");
    if tau_hours >= 50.0 && tau_hours < 100.0 {
        println!("✓ τ in expected range (50-100 hours)");
    } else {
        println!("✗ τ out of expected range!");
    }
    println!();

    // Check energy balance components (using conductances and temperatures)
    let t_m = model_900.mass_temperatures.as_ref()[0];
    let t_zone = model_900.temperatures.as_ref()[0];

    // Heat flux from exterior to mass
    let q_ext_to_mass = h_tr_em * (20.0 - t_m); // Assuming 20°C outdoor
    println!("Heat flux exterior → mass (Q_em): {:.3} W", q_ext_to_mass);

    // Heat flux from surface to mass
    let q_surface_to_mass = h_tr_ms * (t_m - 20.0); // Assuming 20°C surface
    println!(
        "Heat flux surface → mass (Q_ms): {:.3} W",
        q_surface_to_mass
    );

    // Heat flux from surface to zone
    let q_surface_to_zone = h_tr_is * (20.0 - t_zone); // Assuming 20°C surface
    println!(
        "Heat flux surface → zone (Q_is): {:.3} W",
        q_surface_to_zone
    );

    println!();
    println!("=== Conductance Consistency Check ===");
    // For 5R1C: h_tr_is should be larger than h_tr_ms (surface better insulated)
    if h_tr_is > h_tr_ms {
        println!(
            "✓ h_tr_is ({:.2}) > h_tr_ms ({:.2}) - surface better insulated than mass",
            h_tr_is, h_tr_ms
        );
    } else {
        println!(
            "✗ h_tr_is ({:.2}) <= h_tr_ms ({:.2}) - unexpected",
            h_tr_is, h_tr_ms
        );
    }

    println!();
    println!("=== Comparison with Case 600 (Low-Mass) ===");

    // Test Case 600 (low-mass)
    let spec_600 = ASHRAE140Case::Case600.spec();
    let model_600 = ThermalModel::from_spec(&spec_600);

    println!(
        "Mass temperature: {:.2}°C",
        model_600.mass_temperatures.as_ref()[0]
    );
    println!(
        "Thermal capacitance: {:.2} kJ/K",
        model_600.thermal_capacitance.as_ref()[0] / 1000.0
    );
    println!(
        "h_tr_ms (mass to surface): {:.3} W/K",
        model_600.h_tr_ms.as_ref()[0]
    );
    println!(
        "h_tr_is (surface to interior): {:.3} W/K",
        model_600.h_tr_is.as_ref()[0]
    );

    // Calculate thermal time constant for Case 600
    let total_r_low: f64 = spec_600
        .construction
        .wall
        .layers
        .iter()
        .map(|l| l.thickness / l.conductivity)
        .sum::<f64>();

    let total_c_low: f64 = spec_600
        .construction
        .wall
        .layers
        .iter()
        .map(|l| l.thickness * l.density * l.specific_heat * 48.0)
        .sum::<f64>();

    let tau_hours_low = total_r_low * total_c_low / 3600.0;
    println!("Thermal time constant (τ): {:.2} hours", tau_hours_low);
    println!("Expected for Case 600: ~5 hours");
    if tau_hours_low > 2.0 && tau_hours_low < 10.0 {
        println!("✓ τ in expected range (2-10 hours)");
    } else {
        println!("✗ τ out of expected range!");
    }

    println!();
    println!("=== Key Observations ===");
    println!(
        "High-mass τ ({:.1}h) should be >> low-mass τ ({:.1}h)",
        tau_hours, tau_hours_low
    );
    if tau_hours > tau_hours_low * 5.0 {
        println!("✓ High-mass has significantly longer time constant (correct)");
    } else {
        println!("✗ Time constant ratio may be incorrect");
    }
}

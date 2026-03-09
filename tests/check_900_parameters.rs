//! Diagnostic test to check Case 900 parameters
//!
//! Purpose: Investigate ASHRAE 140 reference implementation by checking
//! what parameters are currently being used for Case 900.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_case_900_parameters() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Case 900 Parameters ===");
    println!();

    // Building geometry
    println!("Geometry:");
    println!("  Floor area: {:.2} m²", spec.geometry[0].floor_area());
    println!("  Wall area: {:.2} m²", spec.geometry[0].wall_area());
    println!("  Volume: {:.2} m³", spec.geometry[0].volume());
    println!();

    // Window properties
    println!("Window:");
    if !spec.windows.is_empty() && !spec.windows[0].is_empty() {
        println!("  Area: {:.2} m²", spec.windows[0][0].area);
    }
    println!();

    // Construction properties
    println!("Construction U-values:");
    let construction = &spec.construction;
    println!("  Wall: {:.3} W/m²K", construction.wall.u_value(None, None));
    println!("  Roof: {:.3} W/m²K", construction.roof.u_value(None, None));
    println!("  Floor: {:.3} W/m²K", construction.floor.u_value(None, None));
    println!();

    // HVAC setpoints
    println!("HVAC:");
    println!("  Heating setpoint: {:.1}°C", model.heating_setpoint);
    println!("  Cooling setpoint: {:.1}°C", model.cooling_setpoint);
    println!();

    // Infiltration
    println!("Infiltration:");
    println!("  ACH: {:.2} /h", model.infiltration_rate.as_ref()[0]);
    println!();

    // 5R1C parameters
    println!("5R1C Thermal Network Parameters:");
    let h_tr_em_avg = model.h_tr_em.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;
    let h_tr_ms_avg = model.h_tr_ms.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;
    let h_tr_is_avg = model.h_tr_is.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;
    let h_tr_w_avg = model.h_tr_w.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;
    let h_ve_avg = model.h_ve.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;

    println!("  h_tr_em (exterior-mass): {:.2} W/K", h_tr_em_avg);
    println!("  h_tr_ms (mass-surface): {:.2} W/K", h_tr_ms_avg);
    println!("  h_tr_is (surface-interior): {:.2} W/K", h_tr_is_avg);
    println!("  h_tr_w (exterior-interior): {:.2} W/K", h_tr_w_avg);
    println!("  h_ve (ventilation): {:.2} W/K", h_ve_avg);
    println!();

    // Coupling ratios
    let em_ms_ratio = h_tr_em_avg / h_tr_ms_avg;
    let em_total_ratio = h_tr_em_avg / (h_tr_em_avg + h_tr_ms_avg);
    println!("Coupling Ratios:");
    println!("  h_tr_em / h_tr_ms: {:.3}", em_ms_ratio);
    println!("  h_tr_em / (h_tr_em + h_tr_ms): {:.3}", em_total_ratio);
    println!("  Heat flow to exterior: {:.1}%", em_total_ratio * 100.0);
    println!("  Heat flow to surface: {:.1}%", (1.0 - em_total_ratio) * 100.0);
    println!();

    // Thermal mass
    let cm_avg = model.thermal_capacitance.as_ref().to_vec().iter().sum::<f64>() / model.num_zones as f64;
    let time_constant = cm_avg / (h_tr_em_avg + h_tr_ms_avg) / 3600.0; // hours
    println!("Thermal Mass:");
    println!("  Cm (average): {:.0} J/K", cm_avg);
    println!("  Time constant: {:.2} hours", time_constant);
    println!();

    // Solar distribution
    println!("Solar Distribution:");
    println!("  solar_beam_to_mass_fraction: {:.2}", model.solar_beam_to_mass_fraction);
    println!("  solar_distribution_to_air: {:.2}", model.solar_distribution_to_air);
    println!();

    // Environmental inputs
    println!("Environmental Inputs (from weather module):");
    println!("  Latitude: 39.83°N (hardcoded in DenverTmyWeather)");
    println!("  Elevation: 1655m (mentioned in comments)");
    println!("  Time zone: Mountain Time (not explicitly used)");
    println!("  Ground temperature: Dynamic (not checked)");
    println!();

    // Expected values from ASHRAE 140
    println!("=== ASHRAE 140 Reference Values (from docs) ===");
    println!("Annual Heating: [1.17, 2.04] MWh");
    println!("Annual Cooling: [2.13, 3.67] MWh");
    println!("Peak Heating: [1.10, 2.10] kW");
    println!("Peak Cooling: [2.10, 3.50] kW");
    println!();

    println!("✅ Case 900 parameters check complete");
}

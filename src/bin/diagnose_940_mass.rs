// Diagnostic tool for Case 940 thermal mass parameters
use fluxion::validation::ashrae_140_cases::CaseBuilder;
use fluxion::sim::engine::ThermalModel;

fn main() {
    println!("=== Case 940 Thermal Mass Diagnostics ===\n");

    // Create Case 940 (setback with high mass)
    let spec_940 = CaseBuilder::case_940_setback();
    let mut model_940 = ThermalModel::from_spec(&spec_940);

    // Create Case 900 (baseline high mass, no setback)
    let spec_900 = CaseBuilder::case_900_baseline();
    let model_900 = ThermalModel::from_spec(&spec_900);

    // Thermal capacitance comparison
    println!("1. THERMAL CAPACITANCE");
    println!("   Case 940 (setback):  {:.2e} J/K", model_940.thermal_capacitance[0]);
    println!("   Case 900 (baseline): {:.2e} J/K", model_900.thermal_capacitance[0]);
    println!("   Ratio (940/900):     {:.2}x", model_940.thermal_capacitance[0] / model_900.thermal_capacitance[0]);

    // Check if both exceed high-mass threshold
    let high_mass_threshold = 5.0e6; // From engine.rs
    println!("   High-mass threshold: {:.2e} J/K", high_mass_threshold);
    println!("   Case 940 exceeds threshold: {}", model_940.thermal_capacitance[0] > high_mass_threshold);
    println!("   Case 900 exceeds threshold: {}", model_900.thermal_capacitance[0] > high_mass_threshold);

    // Zone area
    println!("\n2. ZONE AREA");
    println!("   Case 940: {:.1} m²", model_940.zone_area[0]);
    println!("   Case 900: {:.1} m²", model_900.zone_area[0]);

    // Air capacitance (for reference)
    let air_cap_940 = model_940.zone_area[0] * 1.2 * 1005.0;
    let air_cap_900 = model_900.zone_area[0] * 1.2 * 1005.0;
    println!("\n3. AIR CAPACITANCE (ρ * V * cp)");
    println!("   Case 940: {:.2e} J/K", air_cap_940);
    println!("   Case 900: {:.2e} J/K", air_cap_900);

    // Structure capacitance (total - air)
    let struct_cap_940 = model_940.thermal_capacitance[0] - air_cap_940;
    let struct_cap_900 = model_900.thermal_capacitance[0] - air_cap_900;
    println!("\n4. STRUCTURE CAPACITANCE (total - air)");
    println!("   Case 940: {:.2e} J/K", struct_cap_940);
    println!("   Case 900: {:.2e} J/K", struct_cap_900);

    // 5R1C conductances
    println!("\n5. 5R1C CONDUCTANCES (Case 940)");
    println!("   h_tr_em (exterior→mass):  {:.2} W/K", model_940.h_tr_em.as_ref()[0]);
    println!("   h_tr_ms (mass→surface):  {:.2} W/K", model_940.h_tr_ms.as_ref()[0]);
    println!("   h_tr_is (surface→interior): {:.2} W/K", model_940.h_tr_is.as_ref()[0]);
    println!("   h_tr_w  (exterior→interior via windows): {:.2} W/K", model_940.h_tr_w.as_ref()[0]);
    println!("   h_ve    (ventilation):      {:.2} W/K", model_940.h_ve.as_ref()[0]);

    // Coupling ratio
    let coupling_ratio_940 = model_940.h_tr_em.as_ref()[0] / model_940.h_tr_ms.as_ref()[0];
    println!("\n6. THERMAL MASS COUPLING");
    println!("   Case 940 coupling ratio (h_tr_em / h_tr_ms): {:.3}", coupling_ratio_940);
    println!("   Target ratio: 0.1");
    println!("   Status: {}", if coupling_ratio_940 >= 0.1 { "✅ ADEQUATE" } else { "❌ TOO WEAK" });

    // Mode-specific conductances
    println!("\n7. MODE-SPECIFIC CONDUCTANCES");
    println!("   h_tr_em_heating: {:.2} W/K", model_940.h_tr_em_heating.as_ref()[0]);
    println!("   h_tr_em_cooling: {:.2} W/K", model_940.h_tr_em_cooling.as_ref()[0]);
    println!("   Heating coupling ratio: {:.3}", model_940.h_tr_em_heating.as_ref()[0] / model_940.h_tr_ms.as_ref()[0]);
    println!("   Cooling coupling ratio: {:.3}", model_940.h_tr_em_cooling.as_ref()[0] / model_940.h_tr_ms.as_ref()[0]);

    // Time constant
    let tau_940 = model_940.thermal_capacitance[0] / (model_940.h_tr_ms.as_ref()[0] + model_940.h_tr_em.as_ref()[0]);
    let tau_900 = model_900.thermal_capacitance[0] / (model_900.h_tr_ms.as_ref()[0] + model_900.h_tr_em.as_ref()[0]);
    println!("\n8. TIME CONSTANT (τ = Cm / (h_tr_ms + h_tr_em))");
    println!("   Case 940: {:.1} hours", tau_940 / 3600.0);
    println!("   Case 900: {:.1} hours", tau_900 / 3600.0);

    // Apply thermal mass correction and check again
    println!("\n9. AFTER THERMAL MASS CORRECTION");
    model_940.apply_thermal_mass_correction();
    println!("   h_tr_em (after correction): {:.2} W/K", model_940.h_tr_em.as_ref()[0]);
    println!("   h_tr_em_heating (after):    {:.2} W/K", model_940.h_tr_em_heating.as_ref()[0]);
    println!("   h_tr_em_cooling (after):    {:.2} W/K", model_940.h_tr_em_cooling.as_ref()[0]);
    let corrected_ratio = model_940.h_tr_em.as_ref()[0] / model_940.h_tr_ms.as_ref()[0];
    println!("   Corrected coupling ratio:  {:.3}", corrected_ratio);
    println!("   Status: {}", if corrected_ratio >= 0.1 { "✅ ADEQUATE" } else { "❌ STILL TOO WEAK" });

    // Setback schedule verification
    println!("\n10. SETBACK SCHEDULE");
    for hour in [0, 6, 7, 12, 23] {
        let heating_sp = model_940.heating_schedule.value(hour);
        let cooling_sp = model_940.cooling_schedule.value(hour);
        let status = if heating_sp == 10.0 { "SETBACK" } else { "NORMAL" };
        println!("   Hour {:2}: Heating SP = {:.1}°C, Cooling SP = {:.1}°C [{}]", hour, heating_sp, cooling_sp, status);
    }

    println!("\n=== DIAGNOSTIC COMPLETE ===");
}

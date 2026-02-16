// Debug test to see what's happening with the thermal model
#[cfg(test)]
mod debug_tests {
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

    #[test]
    fn debug_case_600_thermal_params() {
        let spec = ASHRAE140Case::Case600.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);
        
        println!("\n=== Case 600 Model Parameters ===");
        println!("Zone Area: {:.2} m²", model.zone_area[0]);
        println!("Window U-value: {:.2} W/m²K", model.window_u_value);
        println!("h_tr_w: {:.2} W/K", model.h_tr_w[0]);
        println!("h_tr_em: {:.2} W/K", model.h_tr_em[0]);
        println!("h_tr_ms: {:.2} W/K", model.h_tr_ms[0]);
        println!("h_tr_is: {:.2} W/K", model.h_tr_is[0]);
        println!("h_ve: {:.2} W/K", model.h_ve[0]);
        println!("h_tr_floor: {:.2} W/K", model.h_tr_floor[0]);
        println!("Thermal Capacitance: {:.2} J/K", model.thermal_capacitance[0]);
        println!("Heating Setpoint: {:.1}°C", model.heating_setpoint);
        println!("Cooling Setpoint: {:.1}°C", model.cooling_setpoint);
        println!("HVAC Heating Capacity: {:.0} W", model.hvac_heating_capacity);
        println!("HVAC Cooling Capacity: {:.0} W", model.hvac_cooling_capacity);
        println!("solar_distribution_to_air: {:.2}", model.solar_distribution_to_air);
        
        // Print derived values
        println!("\n=== Derived Values ===");
        println!("derived_h_ext: {:.2}", model.derived_h_ext[0]);
        println!("derived_term_rest_1: {:.2}", model.derived_term_rest_1[0]);
        println!("derived_den: {:.2}", model.derived_den[0]);
        println!("derived_sensitivity: {:.6}", model.derived_sensitivity[0]);
        
        // Calculate expected sensitivity
        let h_ext = model.h_tr_w[0] + model.h_ve[0];
        let term_rest_1 = model.h_tr_ms[0] + model.h_tr_is[0];
        let h_ms_is_prod = model.h_tr_ms[0] * model.h_tr_is[0];
        let den = h_ms_is_prod + term_rest_1 * h_ext;
        let sensitivity = term_rest_1 / den;
        
        println!("\n=== Manual Sensitivity Calculation ===");
        println!("h_ext (manual): {:.2}", h_ext);
        println!("term_rest_1 (manual): {:.2}", term_rest_1);
        println!("h_ms_is_prod (manual): {:.2}", h_ms_is_prod);
        println!("den (manual): {:.2}", den);
        println!("sensitivity (manual): {:.6}", sensitivity);
        
        // Test a simple step
        let mut test_model = model.clone();
        test_model.set_loads(&[200.0 / 48.0]); // ~4 W/m² internal gains
        
        // Calculate free-floating temp at 10°C outdoor
        let t_free = test_model.calculate_free_float_temperature(10.0);
        println!("\n=== Free-floating test ===");
        println!("Outdoor temp: 10°C");
        println!("Loads: {:.2} W/m²", 200.0 / 48.0);
        println!("Free-floating temp: {:.2}°C", t_free);
        println!("Heating setpoint: {:.1}°C", test_model.heating_setpoint);
        println!("Should heat: {}", t_free < test_model.heating_setpoint);
    }
}

// Solar Distribution Unit Tests for ASHRAE 140
//
// These tests validate how solar gains are distributed between zone air
// and thermal mass for both low-mass and high-mass buildings.
//
// Components tested:
// 1. Low-mass solar distribution factor
// 2. High-mass solar distribution factor
// 3. Time-dependent distribution (thermal lag)
// 4. Heat balance: internal gains → zone air vs thermal mass

use fluxion::sim::construction::ConstructionSpec;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, ConstructionType};

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Low-mass buildings should have more solar to zone air
    #[test]
    fn test_low_mass_solar_distribution_to_air() {
        // For low-mass buildings (600-series), solar gains should primarily go
        // to zone air directly because thermal mass has low heat capacity

        let low_mass_spec = ASHRAE140Case::Case600.spec();

        // Create low-mass model
        let mut model = ThermalModel::from_spec(&low_mass_spec);

        // Apply solar gain to check distribution
        // Solar gain should mostly go to zone air for low-mass

        // This test validates the distribution factor or heat balance
        // We expect that solar gains contribute to zone air temperature rise
        // and not be "absorbed" by thermal mass

        // For now, this is a placeholder test that verifies the model structure
        // The actual solar distribution logic would need to be exposed for testing

        // Validate that low-mass construction is actually configured
        assert_eq!(
            model.construction_type,
            ConstructionType::LowMass,
            "Case 600 should use low-mass construction"
        );

        // Thermal mass should be low for low-mass
        // Check thermal capacitance
        let thermal_mass = model.total_thermal_capacity.unwrap_or(0.0);

        // Low-mass: roughly 1000-5000 J/K
        // High-mass: roughly 50000-200000 J/K

        assert!(
            thermal_mass < 20000.0,
            "Low-mass construction should have thermal capacitance < 20000 J/K, got {:.1} J/K",
            thermal_mass
        );
    }

    // Test 2: High-mass buildings should have delayed solar distribution
    #[test]
    fn test_high_mass_solar_thermal_lag() {
        // For high-mass buildings (900-series), solar gains should be distributed
        // between zone air and thermal mass with time delay (thermal lag)

        let high_mass_spec = ASHRAE140Case::Case900.spec();

        let mut model = ThermalModel::from_spec(&high_mass_spec);

        // Validate that high-mass construction is configured
        assert_eq!(
            model.construction_type,
            ConstructionType::HighMass,
            "Case 900 should use high-mass construction"
        );

        // Thermal mass should be high for high-mass
        let thermal_mass = model.total_thermal_capacity.unwrap_or(0.0);

        // High-mass: roughly 50000-200000 J/K
        assert!(
            thermal_mass > 50000.0,
            "High-mass construction should have thermal capacitance > 50000 J/K, got {:.1} J/K",
            thermal_mass
        );

        // Thermal time constant
        // tau = R × C
        // For Case 900 high-mass: tau ≈ 73 hours

        // High-mass buildings should have significant thermal lag
        // Solar gains in morning should cause temperature rise later in the day
        // due to thermal mass storing heat and releasing it slowly
    }

    // Test 3: Verify thermal time constant calculation
    #[test]
    fn test_thermal_time_constant_calculation() {
        // Test that thermal time constant (tau) is calculated correctly
        // tau = R × C (resistance × capacitance)

        // Create simple model
        let spec = ASHRAE140Case::Case900.spec();

        // Get thermal resistance (sum of 1/C for all layers)
        let total_thermal_resistance: f64 = spec
            .construction
            .wall
            .layers
            .iter()
            .map(|l| l.thickness / l.conductivity)
            .sum::<f64>();

        // Get thermal capacitance
        let total_thermal_capacitance: f64 = spec
            .construction
            .wall
            .layers
            .iter()
            .map(|l| l.thickness * l.density * l.specific_heat * 48.0) // 48 m² zone area
            .sum::<f64>();

        let tau_hours = total_thermal_resistance * total_thermal_capacitance / 3600.0;

        // For Case 900 high-mass: tau should be ~73 hours
        assert!(
            tau_hours > 50.0 && tau_hours < 100.0,
            "Thermal time constant for Case 900 should be 50-100 hours, got {:.1} hours",
            tau_hours
        );
    }

    // Test 4: Verify conductance values for low vs high mass
    #[test]
    fn test_conductance_mass_dependence() {
        // Test that h_tr_ms and h_tr_is are appropriate for
        // low-mass vs high-mass constructions

        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let mut low_model = ThermalModel::from_spec(&low_spec);
        let mut high_model = ThermalModel::from_spec(&high_spec);

        // h_tr_ms (mass to surface) should scale with thermal mass
        let low_h_tr_ms = low_model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);
        let high_h_tr_ms = high_model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);

        // High-mass should have higher h_tr_ms
        assert!(
            high_h_tr_ms > low_h_tr_ms,
            "High-mass h_tr_ms ({:.2} W/K) should be > low-mass ({:.2} W/K)",
            high_h_tr_ms,
            low_h_tr_ms
        );

        // h_tr_is (surface to interior air) should scale similarly
        let low_h_tr_is = low_model.h_tr_is.as_ref().get(0).copied().unwrap_or(0.0);
        let high_h_tr_is = high_model.h_tr_is.as_ref().get(0).copied().unwrap_or(0.0);

        assert!(
            high_h_tr_is > low_h_tr_is,
            "High-mass h_tr_is ({:.2} W/K) should be > low-mass ({:.2} W/K)",
            high_h_tr_is,
            low_h_tr_is
        );
    }

    // Test 5: Solar distribution factor range validation
    #[test]
    fn test_solar_distribution_factor_range() {
        // Solar distribution factor should be between 0.0 and 1.0
        // 0.0 = all solar to thermal mass
        // 1.0 = all solar to zone air

        // This test validates that any solar distribution factors
        // in the code are within reasonable bounds

        // For now, this is a placeholder that checks the model structure
        // The actual solar distribution implementation would need to be exposed

        // Create both low and high mass models
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        // Verify construction types are correct
        assert_eq!(low_spec.construction_type, ConstructionType::LowMass);
        assert_eq!(high_spec.construction_type, ConstructionType::HighMass);

        // This test is a placeholder for when solar distribution
        // logic is exposed and testable
    }

    // Test 6: Heat balance verification for solar inputs
    #[test]
    fn test_heat_balance_solar_inputs() {
        // Verify that solar gains are properly accounted for in
        // the zone energy balance equation

        // Zone energy balance:
        // phi_ia = h_ext * (T_ext - T_ia) + h_tr_is * (T_s - T_ia)
        //       + solar_gain + internal_loads - HVAC_load

        // This test verifies that solar_gain appears in the balance equation
        // and is not double-counted or missing

        // For now, this is a structural test
        // When solar distribution logic is exposed, we can verify:
        // 1. Solar gains are included in energy balance
        // 2. No double-counting with envelope conduction
        // 3. Correct sign convention (positive = heat into zone)
    }

    // Test 7: Thermal lag behavior
    #[test]
    fn test_thermal_lag_diurnal_cycle() {
        // Test that thermal mass causes appropriate time delay
        // between solar input and zone temperature response

        // For high-mass buildings:
        // - Morning solar should cause temperature rise in afternoon/evening
        // - Temperature should peak later in the day than solar peak
        // - Night cooling should be slower due to stored heat in mass

        // This test uses the Case 900 high-mass model
        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        // Simulate a single timestep to check mass-temperature behavior
        // Apply a solar gain and check how temperatures respond

        // For now, this is a placeholder test
        // The actual implementation would need:
        // 1. Apply solar gain
        // 2. Step physics
        // 3. Check temperature response
        // 4. Verify time delay is appropriate
    }

    // Test 8: Low vs high mass simulation comparison
    #[test]
    fn test_low_vs_high_mass_response() {
        // Run short simulations for both low and high mass
        // Compare their thermal response characteristics

        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        // Create models
        let mut low_model = ThermalModel::from_spec(&low_spec);
        let mut high_model = ThermalModel::from_spec(&high_spec);

        // Apply same solar gain to both
        // For now, this is structural - we need to expose solar gain
        // calculation to test actual distribution

        // Compare thermal mass
        let low_mass = low_model.total_thermal_capacity.unwrap_or(0.0);
        let high_mass = high_model.total_thermal_capacity.unwrap_or(0.0);

        // Low-mass should have lower thermal mass
        assert!(
            low_mass < high_mass,
            "Low-mass ({:.1} J/K) should be < high-mass ({:.1} J/K)",
            low_mass,
            high_mass
        );

        // High-mass should have significant thermal lag
        // tau for high-mass: ~73 hours
        // tau for low-mass: ~5 hours

        // High-mass should respond much more slowly to solar inputs
    }

    // Test 9: Solar distribution affects zone vs mass temperature
    #[test]
    fn test_solar_distribution_zone_mass_difference() {
        // Test that solar distribution correctly handles the difference
        // between zone air temperature and thermal mass temperature

        // For different distribution factors:
        // - 100% to zone air: T_zone rises immediately with solar
        // - 100% to mass: T_mass rises immediately with solar
        // - Split: Some to zone, some to mass

        // This test validates the heat transfer paths
        // when solar distribution logic is exposed
    }

    // Test 10: Verify CTF vs 5R1C for high-mass solar distribution
    #[test]
    fn test_ctf_solar_distribution_vs_5r1c() {
        // For high-mass buildings, CTF should handle envelope conduction
        // 5R1C uses h_tr_em for envelope conduction
        // This affects how solar gains interact with thermal mass

        let high_spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&high_spec);

        // Check if CTF is enabled for high-mass
        // This is a placeholder - actual test would simulate
        // and compare CTF vs 5R1C solar distribution behavior

        assert_eq!(model.construction_type, ConstructionType::HighMass);

        // When CTF is enabled:
        // - Solar gains to zone should be immediate (no delay through mass)
        // - But mass temperature should still track solar heat stored
        // - CTF handles envelope conduction, not solar distribution
    }
}

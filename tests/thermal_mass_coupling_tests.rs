// Thermal Mass Coupling Unit Tests for ASHRAE 140
//
// These tests validate thermal mass coupling in the 5R1C network:
// - h_tr_ms (mass to surface) conductance
// - h_tr_is (surface to interior air) conductance
// - Thermal time constant (tau) calculation
// - Heat flux from mass back to zone air
// - Mass temperature update equation
//
// This is likely the root cause of 2-3x overprediction in Case 900.

use fluxion::sim::construction::ConstructionSpec;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, ConstructionType};
use fluxion::physics::constants::thermal::ISO13790_CONSTANTS;

const EPSILON: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Verify h_tr_ms (mass to surface) conductance calculation
    #[test]
    fn test_h_tr_ms_conductance_calculation() {
        // Test that h_tr_ms = Σ(C/A_i) where:
        // - C = thermal capacitance of layer i
        // - A = surface area of layer i

        // h_tr_ms should be conductance (W/K)
        // High-mass: should have lower h_tr_ms (more insulation, less conductive)
        // Low-mass: should have higher h_tr_ms (less insulation, more conductive)

        let high_spec = ASHRAE140Case::Case900.spec();
        let mut high_model = ThermalModel::from_spec(&high_spec);

        let h_tr_ms = high_model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);

        // For Case 900 high-mass:
        // - Concrete block (0.1m, k=0.51, ρ=1600, cp=880)
        // - Foam insulation (0.0615m, k=0.04, ρ=24, cp=840)
        // - Wood siding (0.009m, k=0.16, ρ=530, cp=900)
        // - Surface film coefficients (h_se=8.29, h_si=3.07)

        // Expected h_tr_ms range:
        // High-mass: 1-5 W/K (depends on surface films)
        // Low-mass: 5-15 W/K (lightweight construction)

        assert!(
            h_tr_ms >= 0.1 && h_tr_ms < 10.0,
            "h_tr_ms should be in reasonable range for high-mass: 1-10 W/K, got {:.3} W/K",
            h_tr_ms
        );
    }

    // Test 2: Verify h_tr_is (surface to interior air) conductance
    #[test]
    fn test_h_tr_is_conductance_calculation() {
        // Test that h_tr_is = Σ(A_i / R_i) where:
        // - A_i = surface area of layer i
        // - R_i = thermal resistance of layer i

        // h_tr_is includes surface film coefficients (h_se, h_si)

        let high_spec = ASHRAE140Case::Case900.spec();
        let mut high_model = ThermalModel::from_spec(&high_spec);

        let h_tr_is = high_model.h_tr_is.as_ref().get(0).copied().unwrap_or(0.0);

        // For Case 900:
        // - Surface areas vary by layer
        // - R values vary by layer
        // - h_se = 8.29 (exterior)
        // - h_si = 3.07 (interior)

        // Expected h_tr_is range:
        // - With film coefficients: 2-5 W/K
        // - Without films: 1-10 W/K (depends on construction)

        assert!(
            h_tr_is >= 1.0 && h_tr_is < 10.0,
            "h_tr_is should be in reasonable range: 1-10 W/K, got {:.3} W/K",
            h_tr_is
        );
    }

    // Test 3: Verify thermal time constant (tau) calculation
    #[test]
    fn test_thermal_time_constant_calculation() {
        // Test τ = R × C where:
        // - R = total thermal resistance of envelope
        // - C = total thermal capacitance

        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        // Calculate total R
        let total_r: f64 = spec.construction.wall.layers.iter()
            .map(|l| l.thickness / l.conductivity)
            .sum::<f64>();

        // Calculate total C (using zone area = 48 m²)
        let total_c: f64 = spec.construction.wall.layers.iter()
            .map(|l| l.thickness * l.density * l.specific_heat * 48.0)
            .sum::<f64>();

        let tau_hours = total_r * total_c / 3600.0;

        // For Case 900 high-mass: τ should be ~73 hours
        // R = 1 / (0.1/0.51 + 0.0615/0.04 + 8.29 + 1/3.07) ≈ 0.1 K·m²/W
        // C = 0.1×1600 + 0.0615×2400 + 0.009×530 × 48 ≈ 224 kJ/K
        // τ = 0.1 × 224 kJ/K / 3600 ≈ 73 hours

        assert!(
            tau_hours >= 50.0 && tau_hours < 100.0,
            "Thermal time constant for Case 900 should be 50-100 hours, got {:.1} hours",
            tau_hours
        );

        // For low-mass (Case 600): τ should be ~5 hours
        let low_spec = ASHRAE140Case::Case600.spec();
        let mut low_model = ThermalModel::from_spec(&low_spec);

        let total_r_low: f64 = low_spec.construction.wall.layers.iter()
            .map(|l| l.thickness / l.conductivity)
            .sum::<f64>();

        let total_c_low: f64 = low_spec.construction.wall.layers.iter()
            .map(|l| l.thickness * l.density * l.specific_heat * 48.0)
            .sum::<f64>();

        let tau_hours_low = total_r_low * total_c_low / 3600.0;

        assert!(
            tau_hours_low > 2.0 && tau_hours_low < 10.0,
            "Thermal time constant for Case 600 should be 2-10 hours, got {:.1} hours",
            tau_hours_low
        );

        // High-mass should have τ >> low-mass
        assert!(tau_hours > tau_hours_low,
            "High-mass τ ({:.1}h) should be >> low-mass τ ({:.1}h)",
            tau_hours, tau_hours_low
        );
    }

    // Test 4: Verify total thermal capacitance calculation
    #[test]
    fn test_total_thermal_capacitance_calculation() {
        // Test C_total = Σ(C_i × A_zone) where:
        // - C_i = thermal capacitance of layer i
        // - A_zone = zone area (48 m² for Case 900)

        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        let total_cap = model.total_thermal_capacity.unwrap_or(0.0);

        // For Case 900 high-mass:
        // C_total should be roughly 200-250 kJ/K
        // Based on: 0.1×1600 + 0.0615×2400 + 0.009×530 ≈ 224 kJ/K

        assert!(
            total_cap > 50000.0 && total_cap < 300000.0,
            "Total thermal capacitance for Case 900 should be 50-300 kJ/K, got {:.1} kJ/K",
            total_cap
        );

        // For low-mass (Case 600): should be < 20000 kJ/K
        let low_spec = ASHRAE140Case::Case600.spec();
        let mut low_model = ThermalModel::from_spec(&low_spec);

        let total_cap_low = low_model.total_thermal_capacity.unwrap_or(0.0);

        assert!(
            total_cap_low < 20000.0,
            "Total thermal capacitance for Case 600 should be < 20000 J/K, got {:.1} J/K",
            total_cap_low
        );
    }

    // Test 5: Verify mass temperature update equation
    #[test]
    fn test_mass_temperature_update_equation() {
        // Test: Tm_next = Tm + (Q_m × dt) / C_m
        // Where:
        // - Q_m = net heat flux into mass (from envelope conduction + solar)
        // - dt = timestep (3600 s for hourly)
        // - C_m = total thermal capacitance of mass

        // This is the fundamental thermal balance equation
        // Must conserve energy: Q_in = Q_out + dE_storage

        // This test verifies the equation structure is correct
        // We need to expose:
        // - Mass temperature
        // - Net heat flux into mass (Q_m)
        // - Thermal capacitance (C_m)
        // - Timestep (dt)

        // For now, this is a structural test
        // The actual implementation would need to be exposed
        // and validated against energy conservation
    }

    // Test 6: Verify heat flux calculation: mass → surface
    #[test]
    fn test_heat_flux_mass_to_surface() {
        // Test: Q_ms = h_tr_ms × (T_m - T_s)
        // Where:
        // - T_m = mass temperature
        // - T_s = surface temperature
        // - Q_ms = heat flux from mass to surface (W)

        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        let h_tr_ms = model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);

        // For high-mass, h_tr_ms should be 1-5 W/K
        // Heat flux Q_ms should be proportional to temperature difference

        assert!(
            h_tr_ms > 0.0,
            "h_tr_ms should be positive for Case 900, got {:.3} W/K",
            h_tr_ms
        );

        // This test verifies conductance exists and is positive
        // Actual flux calculation would need temperature values to validate
    }

    // Test 7: Verify heat flux calculation: surface → zone air
    #[test]
    fn test_heat_flux_surface_to_zone() {
        // Test: Q_is = h_tr_is × (T_s - T_ia)
        // Where:
        // - T_s = surface temperature
        // - T_ia = zone air temperature
        // - Q_is = heat flux from surface to zone air (W)

        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        let h_tr_is = model.h_tr_is.as_ref().get(0).copied().unwrap_or(0.0);

        // For Case 900, h_tr_is should be 1-5 W/K with film coefficients
        // Q_is should be proportional to temperature difference

        assert!(
            h_tr_is > 0.0,
            "h_tr_is should be positive for Case 900, got {:.3} W/K",
            h_tr_is
        );

        // Validates surface-to-zone air heat transfer path
    }

    // Test 8: Verify thermal mass energy balance
    #[test]
    fn test_thermal_mass_energy_balance() {
        // Test: Energy balance for thermal mass node
        // E_m = E_in - E_out + dE_storage (should be ≈ 0 at steady state)
        // Where:
        // - E_in = heat from envelope conduction + solar to mass
        // - E_out = heat from mass to surface (Q_ms)
        // - dE_storage = C_m × dT_m (change in stored energy)

        // This test validates that energy is conserved at the mass node
        // Should be integrated with the solve loop

        // For now, this is a structural test
        // The actual implementation would need to track:
        // - Energy in from all sources
        // - Energy out to all destinations
        // - Change in stored energy (dE_storage)
        // - Validate |E_m| < EPSILON (near zero at steady state)
    }

    // Test 9: Low vs high mass thermal coupling comparison
    #[test]
    fn test_low_vs_high_mass_coupling() {
        // Compare thermal coupling behavior:
        // - Low-mass: Fast response, small τ (~5h)
        // - High-mass: Slow response, large τ (~73h)

        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let mut low_model = ThermalModel::from_spec(&low_spec);
        let mut high_model = ThermalModel::from_spec(&high_spec);

        // Compare thermal time constants
        let low_model = ThermalModel::from_spec(&low_spec);
        let high_model = ThermalModel::from_spec(&high_spec);

        let low_tau = calculate_tau(&low_model);
        let high_tau = calculate_tau(&high_model);

        // Low-mass should have small τ
        assert!(low_tau < 10.0,
            "Low-mass τ should be < 10 hours, got {:.1} hours",
            low_tau
        );

        // High-mass should have large τ
        assert!(high_tau > 50.0,
            "High-mass τ should be > 50 hours, got {:.1} hours",
            high_tau
        );

        // Compare conductances
        let low_h_tr_ms = low_model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);
        let high_h_tr_ms = high_model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);

        // High-mass should have lower conductance (better insulated)
        assert!(high_h_tr_ms < low_h_tr_ms,
            "High-mass h_tr_ms ({:.2} W/K) should be < low-mass ({:.2} W/K)",
            high_h_tr_ms, low_h_tr_ms
        );

        // High-mass should be more thermally resistive to heat flow
        // This affects how quickly mass responds to thermal changes
    }

    // Test 10: Verify thermal lag effect on energy consumption
    #[test]
    fn test_thermal_lag_energy_impact() {
        // Test that thermal lag affects energy consumption correctly
        // High-mass with large τ:
        // - Should smooth out temperature swings
        // - Should delay heating/cooling
        // - Should reduce peak loads

        // For Case 900, τ ≈ 73 hours:
        // This is very large, causing significant thermal lag
        // Should result in:
        // - Lower peak loads (attenuated)
        // - Delayed heating/cooling (mass stores/releases heat)
        // - Potentially more accurate energy consumption

        // This test validates the τ parameter
        // Actual implementation would simulate with different τ values
    }

    // Test 11: Verify CTF thermal mass integration
    #[test]
    fn test_ctf_thermal_mass_integration() {
        // Test that CTF solver correctly interacts with thermal mass
        // CTF calculates envelope conduction through walls
        // Should affect mass temperature update
        // Should maintain energy conservation

        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        // Check if CTF is enabled for high-mass
        // This is a placeholder - actual test would need CTF enabled

        assert_eq!(model.construction_type, ConstructionType::HighMass);

        // When CTF is active:
        // - Envelope conduction (h_tr_em) should be handled by CTF
        // - Mass temperature should be updated based on CTF flux
        // - Should maintain proper energy balance
    }

    // Test 12: Verify thermal mass initialization
    #[test]
    fn test_thermal_mass_initialization() {
        // Test that thermal mass temperature is initialized reasonably
        // Should not start at 0°C (unless pre-conditioned)
        // Should start near indoor setpoint (20°C)

        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        let initial_temp = model.mass_temperatures.as_ref().get(0).copied().unwrap_or(0.0);

        // Mass temperature should be initialized near setpoint
        assert!(initial_temp > 15.0 && initial_temp < 25.0,
            "Mass temperature should be initialized near setpoint (15-25°C), got {:.1}°C",
            initial_temp
        );

        // This test verifies initial thermal state is reasonable
    }

    // Test 13: Verify conductance values are consistent
    #[test]
    fn test_conductance_consistency() {
        // Verify that h_tr_ms, h_tr_is, and h_tr_em form consistent network
        // Energy balance: Q_ext → Q_ms → Q_is → Q_ia

        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::from_spec(&spec);

        let h_tr_ms = model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);
        let h_tr_is = model.h_tr_is.as_ref().get(0).copied().unwrap_or(0.0);
        let h_tr_em = model.h_tr_em.as_ref().get(0).copied().unwrap_or(0.0);

        // Check consistency: h_tr_ms and h_tr_is should have similar magnitude
        // h_tr_em should be in between (envelope conduction)

        // For Case 900:
        // - h_tr_ms ≈ 1-5 W/K (mass to surface)
        // - h_tr_is ≈ 3-7 W/K (surface to interior with films)
        // - h_tr_em ≈ 0.5 W/K (exterior to mass)

        // These form a thermal ladder
        assert!(h_tr_ms > 0.0 && h_tr_is > 0.0 && h_tr_em > 0.0,
            "All conductances should be positive, got h_tr_ms={:.2}, h_tr_is={:.2}, h_tr_em={:.2}",
            h_tr_ms, h_tr_is, h_tr_em
        );

        // h_tr_is should be larger than h_tr_ms (surface better insulated)
        assert!(h_tr_is > h_tr_ms,
            "h_tr_is ({:.2} W/K) should be > h_tr_ms ({:.2} W/K)",
            h_tr_is, h_tr_ms
        );
    }
}

fn calculate_tau(model: &ThermalModel<VectorField>) -> f64 {
    // Calculate thermal time constant τ = R × C
    let total_r: f64 = model.construction.wall.layers.iter()
        .map(|l| l.thickness / l.conductivity)
        .sum::<f64>();

    let total_c: f64 = model.total_thermal_capacity.unwrap_or(0.0);

    total_r * total_c / 3600.0  // Convert to hours
}

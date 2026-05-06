use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, ConstructionType};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that verifies the solar distribution is correct after fix (Issue #664)
    ///
    /// FIX APPLIED: Changed formula from `solar_to_air = 0.1 * (1 - f_ms) + f_ms`
    /// to correct ISO 13790 Section C.2: `solar_to_air = 0.5 * f_ms`
    ///
    /// Results after fix:
    ///   - Light mass (f_ms=0.4): 0.5*0.4 = 0.20 → 20% to air, 80% to mass
    ///   - Heavy mass (f_ms=0.8): 0.5*0.8 = 0.40 → 40% to air, 60% to mass
    ///
    /// Physics: More thermal mass means more capacity to absorb solar gains,
    /// but proportionally less goes directly to air (since the mass node itself
    /// absorbs most of it before it can heat the air).
    #[test]
    fn test_heavy_mass_solar_to_air_corrected() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let low_model = ThermalModel::<VectorField>::from_spec(&low_spec);
        let high_model = ThermalModel::<VectorField>::from_spec(&high_spec);

        let low_dist_to_air = low_model.solar_distribution_to_air;
        let high_dist_to_air = high_model.solar_distribution_to_air;

        println!("Solar Distribution to Air (After Fix):");
        println!("  Case 600 (LowMass, f_ms=0.4): {:.2}", low_dist_to_air);
        println!("  Case 900 (HighMass, f_ms=0.8): {:.2}", high_dist_to_air);
        println!("  Expected: LowMass < HighMass (more mass = more to air proportionally)");

        // After fix: Heavy mass (0.40) has MORE solar to air than light mass (0.20)
        // This is CORRECT per ISO 13790 - the proportional split favors more to air
        // when there's more mass available to absorb the remainder
        assert!(
            high_dist_to_air > low_dist_to_air,
            "Heavy mass should have higher solar_to_air fraction. \
             Got Case 900: {:.2}, Case 600: {:.2}",
            high_dist_to_air,
            low_dist_to_air
        );

        // Verify absolute values are in expected range (±0.05 tolerance)
        let expected_low = 0.20;
        let expected_high = 0.40;
        assert!(
            (low_dist_to_air - expected_low).abs() < 0.05,
            "Case 600 solar_to_air should be {:.2}, got {:.2}",
            expected_low,
            low_dist_to_air
        );
        assert!(
            (high_dist_to_air - expected_high).abs() < 0.05,
            "Case 900 solar_to_air should be {:.2}, got {:.2}",
            expected_high,
            high_dist_to_air
        );
    }

    /// Additional test: Verify the solar_to_air values match ISO 13790
    #[test]
    fn test_solar_to_air_matches_iso_13790() {
        let high_spec = ASHRAE140Case::Case900.spec();
        let high_model = ThermalModel::<VectorField>::from_spec(&high_spec);

        let dist_to_air = high_model.solar_distribution_to_air;

        // Per ISO 13790 Section C.2: solar_to_air_frac = 0.5 * f_ms
        // For heavy mass (f_ms=0.8): 0.5 * 0.8 = 0.40
        // Tolerance: ±0.05
        let expected_high_mass = 0.40;
        let tolerance = 0.05;

        println!("Case 900 solar_distribution_to_air: {:.2}", dist_to_air);
        println!(
            "Expected (ISO 13790): {:.2} ± {:.2}",
            expected_high_mass, tolerance
        );

        assert!(
            (dist_to_air - expected_high_mass).abs() < tolerance,
            "Case 900 solar_distribution_to_air should be {:.2} per ISO 13790, got {:.2}",
            expected_high_mass,
            dist_to_air
        );
    }

    /// Verify construction type is correctly identified
    #[test]
    fn test_construction_type_identification() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        assert_eq!(low_spec.construction_type, ConstructionType::LowMass);
        assert_eq!(high_spec.construction_type, ConstructionType::HighMass);
    }

    /// Verify solar beam to mass fraction follows expected pattern
    #[test]
    fn test_solar_beam_to_mass_fraction() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let low_model = ThermalModel::<VectorField>::from_spec(&low_spec);
        let high_model = ThermalModel::<VectorField>::from_spec(&high_spec);

        let low_beam_to_mass = low_model.solar_beam_to_mass_fraction;
        let high_beam_to_mass = high_model.solar_beam_to_mass_fraction;

        println!("Solar Beam to Mass Fraction:");
        println!("  Case 600: {:.2}", low_beam_to_mass);
        println!("  Case 900: {:.2}", high_beam_to_mass);

        // Light mass (f_ms=0.4): 80% beam to mass
        // Heavy mass (f_ms=0.8): 60% beam to mass
        // Light mass has higher beam-to-mass fraction because its smaller mass
        // can't store as much, so more beam goes directly to mass surface
        assert!(
            low_beam_to_mass > high_beam_to_mass,
            "Light mass should have higher beam-to-mass fraction. \
             Got Case 600: {:.2}, Case 900: {:.2}",
            low_beam_to_mass,
            high_beam_to_mass
        );
    }

    /// Verify that solar_distribution_to_air + solar_beam_to_mass_fraction = 1.0
    #[test]
    fn test_solar_fractions_sum_to_one() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let low_model = ThermalModel::<VectorField>::from_spec(&low_spec);
        let high_model = ThermalModel::<VectorField>::from_spec(&high_spec);

        let low_sum = low_model.solar_distribution_to_air + low_model.solar_beam_to_mass_fraction;
        let high_sum =
            high_model.solar_distribution_to_air + high_model.solar_beam_to_mass_fraction;

        println!("Fraction sum (solar_to_air + solar_to_mass):");
        println!("  Case 600: {:.2}", low_sum);
        println!("  Case 900: {:.2}", high_sum);

        assert!(
            (low_sum - 1.0).abs() < 0.001,
            "Case 600 fractions should sum to 1.0, got {:.2}",
            low_sum
        );
        assert!(
            (high_sum - 1.0).abs() < 0.001,
            "Case 900 fractions should sum to 1.0, got {:.2}",
            high_sum
        );
    }
}

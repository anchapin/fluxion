use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, ConstructionType};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test ASHRAE 140-2023 Section 5.2.2 compliance for solar distribution.
    ///
    /// Per ASHRAE 140 Section 5.2.2:
    /// - 100% of transmitted solar goes to opaque interior surfaces (proportional to A×α)
    /// - ZERO fraction goes to the air node directly (solar_distribution_to_air = 0.0)
    /// - Windows (α ≈ 0 for ASHRAE 140 simplified model) are excluded
    ///
    /// This test verifies the correction from ISO 13790 approach (Issue #745).
    #[test]
    fn test_ashrae_140_solar_distribution_to_air_is_zero() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let low_model = ThermalModel::<VectorField>::from_spec_with_selector(
            &low_spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let high_model = ThermalModel::<VectorField>::from_spec_with_selector(
            &high_spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

        println!("ASHRAE 140 Section 5.2.2 Solar Distribution:");
        println!(
            "  Case 600 (LowMass): solar_distribution_to_air = {:.2}",
            low_model.solar.solar_distribution_to_air
        );
        println!(
            "  Case 900 (HighMass): solar_distribution_to_air = {:.2}",
            high_model.solar.solar_distribution_to_air
        );
        println!("  Expected: 0.0 for both (100% to opaque surfaces)");

        // ASHRAE 140: zero fraction goes to air node directly
        assert!(
            low_model.solar.solar_distribution_to_air.abs() < 0.001,
            "Case 600 solar_distribution_to_air should be 0.0 per ASHRAE 140, got {:.2}",
            low_model.solar.solar_distribution_to_air
        );
        assert!(
            high_model.solar.solar_distribution_to_air.abs() < 0.001,
            "Case 900 solar_distribution_to_air should be 0.0 per ASHRAE 140, got {:.2}",
            high_model.solar.solar_distribution_to_air
        );
    }

    /// Verify solar_beam_to_mass_fraction follows ASHRAE 140 (100% to mass for simplified model).
    #[test]
    fn test_ashrae_140_solar_beam_to_mass_fraction() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let low_model = ThermalModel::<VectorField>::from_spec_with_selector(
            &low_spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let high_model = ThermalModel::<VectorField>::from_spec_with_selector(
            &high_spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

        println!("ASHRAE 140 Solar Beam Distribution:");
        println!(
            "  Case 600: solar_beam_to_mass_fraction = {:.2}",
            low_model.solar.solar_beam_to_mass_fraction
        );
        println!(
            "  Case 900: solar_beam_to_mass_fraction = {:.2}",
            high_model.solar.solar_beam_to_mass_fraction
        );

        // ASHRAE 140: 100% of transmitted solar goes to opaque surfaces (thermal mass)
        assert!(
            (low_model.solar.solar_beam_to_mass_fraction - 1.0).abs() < 0.001,
            "Case 600 solar_beam_to_mass_fraction should be 1.0 per ASHRAE 140, got {:.2}",
            low_model.solar.solar_beam_to_mass_fraction
        );
        assert!(
            (high_model.solar.solar_beam_to_mass_fraction - 1.0).abs() < 0.001,
            "Case 900 solar_beam_to_mass_fraction should be 1.0 per ASHRAE 140, got {:.2}",
            high_model.solar.solar_beam_to_mass_fraction
        );
    }

    /// Verify construction type is correctly identified.
    #[test]
    fn test_construction_type_identification() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        assert_eq!(low_spec.construction_type, ConstructionType::LowMass);
        assert_eq!(high_spec.construction_type, ConstructionType::HighMass);
    }

    /// Verify that solar_distribution_to_air + solar_beam_to_mass_fraction = 1.0
    /// Per ASHRAE 140, all solar goes to opaque surfaces (mass).
    #[test]
    fn test_solar_fractions_sum_to_one() {
        let low_spec = ASHRAE140Case::Case600.spec();
        let high_spec = ASHRAE140Case::Case900.spec();

        let low_model = ThermalModel::<VectorField>::from_spec_with_selector(
            &low_spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let high_model = ThermalModel::<VectorField>::from_spec_with_selector(
            &high_spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

        let low_sum =
            low_model.solar.solar_distribution_to_air + low_model.solar.solar_beam_to_mass_fraction;
        let high_sum = high_model.solar.solar_distribution_to_air
            + high_model.solar.solar_beam_to_mass_fraction;

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

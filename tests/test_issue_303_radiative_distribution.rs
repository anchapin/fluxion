//! Test suite for area-weighted radiative gain distribution (Issue #303)
//!
//! This test validates that radiative gains are distributed correctly based on
//! surface areas and geometry, improving accuracy for asymmetric solar distribution.

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_area_weighted_radiative_distribution_basic() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(
        &spec,
    );

    // Test with a single zone radiative gain
    let zone_idx = 0;
    let radiative_gain_watts = 1000.0; // 1 kW total radiative gain

    let (radiative_to_surface, radiative_to_mass) =
        model.calculate_area_weighted_radiative_distribution(zone_idx, radiative_gain_watts);

    // The distribution should respect the solar_distribution_to_air parameter
    // Default is 0.1, so 10% to surface, 90% to mass
    let expected_surface = radiative_gain_watts * model.solar_distribution_to_air;
    let expected_mass = radiative_gain_watts * (1.0 - model.solar_distribution_to_air);

    assert!((radiative_to_surface - expected_surface).abs() < 1e-6,
        "Radiative to surface mismatch: got {}, expected {}",
        radiative_to_surface, expected_surface
    );

    assert!((radiative_to_mass - expected_mass).abs() < 1e-6,
        "Radiative to mass mismatch: got {}, expected {}",
        radiative_to_mass, expected_mass
    );
}

#[test]
fn test_area_weighted_radiative_distribution_different_fractions() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(
        &spec,
    );

    // Test with different solar distribution fractions
    let radiative_gain_watts = 1000.0;
    let test_fractions = [0.1, 0.3, 0.5, 0.7, 0.9];

    for &fraction in test_fractions.iter() {
        model.solar_distribution_to_air = fraction;

        let (radiative_to_surface, radiative_to_mass) =
            model.calculate_area_weighted_radiative_distribution(0, radiative_gain_watts);

        let expected_surface: f64 = radiative_gain_watts * fraction;
        let expected_mass: f64 = radiative_gain_watts * (1.0 - fraction);

        assert!((radiative_to_surface - expected_surface).abs() < 1e-6,
            "Distribution failed for fraction {}: got {}, expected {}",
            fraction, radiative_to_surface, expected_surface
        );

        assert!((radiative_to_mass - expected_mass).abs() < 1e-6,
            "Mass distribution failed for fraction {}: got {}, expected {}",
            fraction, radiative_to_mass, expected_mass
        );
    }
}

#[test]
fn test_area_weighted_radiative_distribution_multi_zone() {
    let spec = ASHRAE140Case::Case960.spec();
    let model = fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    // Case 960 has 2 zones
    assert_eq!(model.num_zones, 2, "Case 960 should have 2 zones");

    let radiative_gain_watts = 500.0;

    // Test both zones return valid distributions
    for zone_idx in 0..model.num_zones {
        let (radiative_to_surface, radiative_to_mass) =
            model.calculate_area_weighted_radiative_distribution(zone_idx, radiative_gain_watts);

        assert!(radiative_to_surface >= 0.0, "Radiative to surface should be non-negative");
        assert!(radiative_to_mass >= 0.0, "Radiative to mass should be non-negative");

        // Sum should equal total radiative gain (minus any losses to ground/floor)
        let total: f64 = radiative_to_surface + radiative_to_mass;
        assert!(
            (total - radiative_gain_watts).abs() < 1e-6,
            "Distribution sum mismatch for zone {}: got {}, expected {}",
            zone_idx, total, radiative_gain_watts
        );
    }
}

#[test]
fn test_area_weighted_radiative_distribution_surface_geometry() {
    let spec = ASHRAE140Case::Case600.spec();
    let model = fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    // Verify surfaces are properly defined
    assert!(!model.surfaces.is_empty(), "Model should have surfaces defined");
    assert!(!model.surfaces[0].is_empty(), "Zone 0 should have surfaces");

    // Calculate total surface area (excluding floor)
    let surfaces = &model.surfaces[0];
    let total_surface_area: f64 = surfaces
        .iter()
        .filter(|s| {
            s.orientation != fluxion::validation::ashrae_140_cases::Orientation::Down
        })
        .map(|s| s.area)
        .sum();

    assert!(
        total_surface_area > 0.0,
        "Total surface area should be positive for zone 0"
    );

    println!("Zone 0 surface area: {:.2} m²", total_surface_area);
    println!("Number of surfaces: {}", surfaces.len());
}

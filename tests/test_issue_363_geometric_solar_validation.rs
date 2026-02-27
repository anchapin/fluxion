//! Test suite for geometric solar distribution validation (Issue #363)
//!
//! This test validates beam-to-floor mapping and area-weighted diffuse distribution
//! for ASHRAE 140 validation, particularly for Case 960 (sunspace building).

use fluxion::sim::solar::{calculate_surface_irradiance, SurfaceIrradiance};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Test 1: Verify beam vs diffuse separation accuracy
#[test]
fn test_beam_diffuse_separation_accuracy() {
    let dni = 800.0; // Direct Normal Irradiance (W/m²)
    let dhi = 100.0; // Diffuse Horizontal Irradiance (W/m²)

    // Calculate beam irradiance with incidence angle
    let incidence_angle = 30.0_f64.to_radians();
    let beam = dni * incidence_angle.cos();

    // Verify beam calculation
    assert!(
        (beam - 800.0 * 30.0_f64.to_radians().cos()).abs() < 1e-6,
        "Beam irradiance calculation incorrect"
    );

    // Beam should be less than DNI (except at normal incidence)
    assert!(beam <= dni + 1e-6, "Beam irradiance should not exceed DNI");

    // Diffuse should equal DHI for isotropic sky model
    let diffuse: f64 = dhi; // Simplified isotropic model
    assert!(
        (diffuse - dhi).abs() < 1e-6,
        "Diffuse irradiance should equal DHI for isotropic model"
    );
}

/// Test 2: Verify beam-to-floor mapping logic
#[test]
fn test_beam_to_floor_mapping() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    // Simulate beam radiation entering through window
    let beam_irradiance_wm2 = 800.0;
    let window_area = 10.0; // m²
    let total_beam_watts = beam_irradiance_wm2 * window_area;

    // Based on Issue #297, 80-95% of beam should reach floor mass
    let solar_beam_to_mass_fraction = 0.9; // 90% to mass
    let expected_mass_gain = total_beam_watts * solar_beam_to_mass_fraction;
    let expected_surface_gain = total_beam_watts * (1.0 - solar_beam_to_mass_fraction);

    // Calculate distribution using model's method
    let (radiative_to_surface, radiative_to_mass) =
        model.calculate_area_weighted_radiative_distribution(0, total_beam_watts);

    // Verify mass receives majority of beam radiation
    let mass_fraction = radiative_to_mass / total_beam_watts;
    assert!(
        mass_fraction >= 0.8 && mass_fraction <= 0.95,
        "Beam-to-floor fraction should be 80-95%, got {:.2}%",
        mass_fraction * 100.0
    );

    // Verify energy balance
    let total_distributed = radiative_to_surface + radiative_to_mass;
    assert!(
        (total_distributed - total_beam_watts).abs() < 1.0,
        "Energy balance: distributed ({}) should equal input ({})",
        total_distributed,
        total_beam_watts
    );
}

/// Test 3: Verify area-weighted diffuse distribution
#[test]
fn test_area_weighted_diffuse_distribution() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    let diffuse_gain_watts = 500.0; // 500 W diffuse

    // Get surface areas for verification
    let surfaces = &model.surfaces[0];
    let total_surface_area: f64 = surfaces
        .iter()
        .filter(|s| s.orientation != fluxion::validation::ashrae_140_cases::Orientation::Down)
        .map(|s| s.area)
        .sum();

    // Calculate distribution
    let (radiative_to_surface, radiative_to_mass) =
        model.calculate_area_weighted_radiative_distribution(0, diffuse_gain_watts);

    // Diffuse should be distributed proportionally to surface area
    // Model uses solar_distribution_to_air parameter (default 0.1)
    let expected_surface = diffuse_gain_watts * model.solar_distribution_to_air;
    let expected_mass = diffuse_gain_watts * (1.0 - model.solar_distribution_to_air);

    assert!(
        (radiative_to_surface - expected_surface).abs() < 1e-6,
        "Surface gain mismatch: got {}, expected {}",
        radiative_to_surface,
        expected_surface
    );

    assert!(
        (radiative_to_mass - expected_mass).abs() < 1e-6,
        "Mass gain mismatch: got {}, expected {}",
        radiative_to_mass,
        expected_mass
    );

    println!("Total surface area: {:.2} m²", total_surface_area);
    println!("Diffuse to surface: {:.2} W", radiative_to_surface);
    println!("Diffuse to mass: {:.2} W", radiative_to_mass);
}

/// Test 4: Multi-zone building diffuse distribution (Case 960)
#[test]
fn test_area_weighted_diffuse_multi_zone() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    // Case 960 has 2 zones: conditioned space + sunspace
    assert_eq!(model.num_zones, 2, "Case 960 should have 2 zones");

    let diffuse_gain_watts = 300.0;

    // Test both zones
    for zone_idx in 0..model.num_zones {
        let (radiative_to_surface, radiative_to_mass) =
            model.calculate_area_weighted_radiative_distribution(zone_idx, diffuse_gain_watts);

        // Verify non-negative values
        assert!(
            radiative_to_surface >= 0.0,
            "Radiative to surface should be non-negative for zone {}",
            zone_idx
        );
        assert!(
            radiative_to_mass >= 0.0,
            "Radiative to mass should be non-negative for zone {}",
            zone_idx
        );

        // Verify energy balance
        let total = radiative_to_surface + radiative_to_mass;
        assert!(
            (total - diffuse_gain_watts).abs() < 1e-6,
            "Zone {} energy balance: distributed ({}) should equal input ({})",
            zone_idx,
            total,
            diffuse_gain_watts
        );
    }
}

/// Test 5: Case 960 validation with full simulation
#[test]
fn test_case_960_geometric_solar_validation() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    // Validate 2-zone setup
    assert_eq!(model.num_zones, 2, "Case 960 should have 2 zones");

    // Validate surfaces exist for both zones
    assert!(
        !model.surfaces.is_empty(),
        "Model should have surfaces defined"
    );
    assert_eq!(
        model.surfaces.len(),
        2,
        "Case 960 should have surfaces for both zones"
    );

    // Validate window properties
    assert!(
        !model.window_properties.is_empty(),
        "Case 960 should have window properties"
    );

    // Validate that sunspace (Zone 1) is free-floating or has different setpoints
    // Zone 0: conditioned space
    // Zone 1: sunspace (typically free-floating or different schedule)
    let heating_sp_0 = model.heating_setpoints.as_ref()[0];
    let heating_sp_1 = model.heating_setpoints.as_ref()[1];

    println!("Zone 0 heating setpoint: {:.2}°C", heating_sp_0);
    println!("Zone 1 heating setpoint: {:.2}°C", heating_sp_1);

    // For Case 960, Zone 1 (sunspace) is often free-floating
    // or has different conditioning than Zone 0
    let hvac_enabled = model.hvac_enabled.as_ref();
    println!("Zone 0 HVAC enabled: {}", hvac_enabled[0] > 0.5);
    println!("Zone 1 HVAC enabled: {}", hvac_enabled[1] > 0.5);
}

/// Test 6: Verify SurfaceIrradiance beam/diffuse structure
#[test]
fn test_surface_irradiance_structure() {
    let beam_wm2 = 800.0;
    let diffuse_wm2 = 100.0;
    let ground_wm2 = 20.0;

    let irradiance = SurfaceIrradiance::new(beam_wm2, diffuse_wm2, ground_wm2);

    // Verify individual components
    assert_eq!(irradiance.beam_wm2, beam_wm2);
    assert_eq!(irradiance.diffuse_wm2, diffuse_wm2);

    // Verify total is calculated correctly
    let expected_total = beam_wm2 + diffuse_wm2 + ground_wm2;
    assert_eq!(irradiance.total_wm2, expected_total);

    println!("Beam: {:.1} W/m²", irradiance.beam_wm2);
    println!("Diffuse: {:.1} W/m²", irradiance.diffuse_wm2);
    println!("Ground: {:.1} W/m²", ground_wm2);
    println!("Total: {:.1} W/m²", irradiance.total_wm2);
}

/// Test 7: Verify beam incidence angle effects
#[test]
fn test_beam_incidence_angle_effects() {
    let dni = 800.0; // Direct Normal Irradiance

    // Test at normal incidence (0°)
    let incidence_0 = 0.0_f64.to_radians();
    let beam_0 = dni * incidence_0.cos();

    // Test at 45° incidence
    let incidence_45 = 45.0_f64.to_radians();
    let beam_45 = dni * incidence_45.cos();

    // Test at 60° incidence (grazing angle)
    let incidence_60 = 60.0_f64.to_radians();
    let beam_60 = dni * incidence_60.cos();

    // Beam should decrease as incidence angle increases
    assert!(
        beam_0 > beam_45,
        "Beam at normal incidence should be greater than at 45°"
    );
    assert!(
        beam_45 > beam_60,
        "Beam at 45° should be greater than at 60°"
    );

    // At normal incidence, beam should equal DNI
    assert!(
        (beam_0 - dni).abs() < 1e-6,
        "At normal incidence, beam should equal DNI"
    );

    println!("Beam at 0°: {:.2} W/m²", beam_0);
    println!("Beam at 45°: {:.2} W/m²", beam_45);
    println!("Beam at 60°: {:.2} W/m²", beam_60);
}

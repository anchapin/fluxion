//! Test to verify multi-zone zone-specific parameters are correctly set.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_case_960_zone_specific_areas() {
    let spec = ASHRAE140Case::Case960.spec();

    assert_eq!(spec.num_zones, 2, "Case 960 should have 2 zones");

    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Zone 0: Back-zone (8m x 6m x 2.7m = 48 m²)
    let zone_0_area = model.zone_area.as_ref()[0];
    let expected_zone_0_area = 8.0 * 6.0; // 48 m²
    assert!(
        (zone_0_area - expected_zone_0_area).abs() < 0.1,
        "Zone 0 area should be ~48 m², got {:.2}",
        zone_0_area
    );

    // Zone 1: Sunspace (8m x 2m x 2.7m = 16 m²)
    let zone_1_area = model.zone_area.as_ref()[1];
    let expected_zone_1_area = 8.0 * 2.0; // 16 m²
    assert!(
        (zone_1_area - expected_zone_1_area).abs() < 0.1,
        "Zone 1 area should be ~16 m², got {:.2}",
        zone_1_area
    );

    println!("\n=== Case 960 Zone Areas ===");
    println!("Zone 0 (Back-zone): {:.2} m²", zone_0_area);
    println!("Zone 1 (Sunspace): {:.2} m²", zone_1_area);
    println!("=== End ===\n");
}

#[test]
fn test_case_960_hvac_enabled_flags() {
    let spec = ASHRAE140Case::Case960.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let hvac_enabled = model.hvac_enabled.as_ref();

    // Zone 0 should have HVAC enabled (back-zone)
    assert!(
        hvac_enabled[0] > 0.5,
        "Zone 0 HVAC should be enabled, got {:.2}",
        hvac_enabled[0]
    );

    // Zone 1 should be free-floating (sunspace)
    assert!(
        hvac_enabled[1] < 0.5,
        "Zone 1 HVAC should be disabled (free-floating), got {:.2}",
        hvac_enabled[1]
    );

    println!("\n=== Case 960 HVAC Enable Flags ===");
    println!(
        "Zone 0 (Back-zone): HVAC enabled = {}",
        hvac_enabled[0] > 0.5
    );
    println!(
        "Zone 1 (Sunspace): HVAC enabled = {}",
        hvac_enabled[1] > 0.5
    );
    println!("=== End ===\n");
}

#[test]
fn test_case_960_thermal_capacitance_per_zone() {
    let spec = ASHRAE140Case::Case960.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let thermal_cap = model.thermal_capacitance.as_ref();

    // Zone 0 and Zone 1 should have different thermal capacitances
    // because they have different floor areas and volumes
    let cap_0 = thermal_cap[0];
    let cap_1 = thermal_cap[1];

    // Zone 1 should have lower thermal capacitance (smaller volume)
    assert!(
        cap_1 < cap_0,
        "Zone 1 should have lower thermal capacitance than Zone 0, got {:.0} vs {:.0}",
        cap_1,
        cap_0
    );

    println!("\n=== Case 960 Thermal Capacitance ===");
    println!("Zone 0 (Back-zone): {:.0} J/K", cap_0);
    println!("Zone 1 (Sunspace): {:.0} J/K", cap_1);
    println!("Ratio: {:.2}x", cap_0 / cap_1);
    println!("=== End ===\n");
}

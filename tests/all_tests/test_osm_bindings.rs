// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Integration tests for OSM parsing and bindings.
//!
//! These tests verify that the OSM parser correctly loads the two_zone.osm
//! fixture and maps all geometric and material properties to Fluxion's
//! internal schema without loss.

use fluxion::interop::osm::{import_osm, OsmParser};

const TWO_ZONE_OSM_PATH: &str = "tests/fixtures/osm/two_zone.osm";

#[test]
fn test_osm_parser_loads_two_zone_fixture() {
    let result = import_osm(TWO_ZONE_OSM_PATH);
    assert!(
        result.is_ok(),
        "Should load two_zone.osm without error: {:?}",
        result.err()
    );
    let schema = result.unwrap();

    assert_eq!(
        schema.metadata.name, "Two Zone Fixture",
        "Building name should be 'Two Zone Fixture'"
    );
}

#[test]
fn test_osm_parser_extracts_zones() {
    let result = import_osm(TWO_ZONE_OSM_PATH);
    assert!(result.is_ok());
    let schema = result.unwrap();

    assert_eq!(
        schema.geometry.zones.len(),
        2,
        "Should have exactly 2 zones"
    );

    let zone_names: Vec<&str> = schema
        .geometry
        .zones
        .iter()
        .map(|z| z.name.as_str())
        .collect();
    assert!(zone_names.contains(&"Zone A"), "Should contain Zone A");
    assert!(zone_names.contains(&"Zone B"), "Should contain Zone B");
}

#[test]
fn test_osm_parser_extracts_zone_properties() {
    let result = import_osm(TWO_ZONE_OSM_PATH);
    assert!(result.is_ok());
    let schema = result.unwrap();

    let zone_a = schema
        .geometry
        .zones
        .iter()
        .find(|z| z.name == "Zone A")
        .unwrap();
    assert!(
        (zone_a.floor_area - 50.0).abs() < 1e-6,
        "Zone A floor area should be 50.0"
    );
    assert!(
        (zone_a.volume - 135.0).abs() < 1e-6,
        "Zone A volume should be 135.0"
    );

    let zone_b = schema
        .geometry
        .zones
        .iter()
        .find(|z| z.name == "Zone B")
        .unwrap();
    assert!(
        (zone_b.floor_area - 75.0).abs() < 1e-6,
        "Zone B floor area should be 75.0"
    );
    assert!(
        (zone_b.volume - 202.5).abs() < 1e-6,
        "Zone B volume should be 202.5"
    );
}

#[test]
fn test_osm_parser_extracts_building_properties() {
    let result = import_osm(TWO_ZONE_OSM_PATH);
    assert!(result.is_ok());
    let schema = result.unwrap();

    assert!(
        (schema.geometry.total_floor_area - 125.0).abs() < 1e-6,
        "Total floor area should be 125.0"
    );
    assert_eq!(schema.geometry.number_of_floors, 1, "Should have 1 floor");
    assert!(
        (schema.geometry.floor_height - 2.7).abs() < 1e-6,
        "Floor height should be 2.7"
    );
}

#[test]
fn test_osm_parser_extracts_materials() {
    let result = import_osm(TWO_ZONE_OSM_PATH);
    assert!(result.is_ok());
    let schema = result.unwrap();

    let wall_constr = &schema.constructions.wall;
    assert!(
        !wall_constr.layers.is_empty(),
        "Wall construction should have layers"
    );

    let plasterboard = wall_constr
        .layers
        .iter()
        .find(|l| l.name.contains("Plasterboard"));
    assert!(plasterboard.is_some(), "Should find Plasterboard layer");

    if let Some(mat) = plasterboard {
        assert!(
            (mat.thickness - 0.012).abs() < 1e-6,
            "Plasterboard thickness should be 0.012"
        );
        assert!(
            (mat.conductivity - 0.16).abs() < 1e-6,
            "Plasterboard conductivity should be 0.16"
        );
        assert!(
            (mat.density - 950.0).abs() < 1e-6,
            "Plasterboard density should be 950.0"
        );
        assert!(
            (mat.specific_heat - 840.0).abs() < 1e-6,
            "Plasterboard specific heat should be 840.0"
        );
    }
}

#[test]
fn test_osm_parser_extracts_constructions() {
    let result = import_osm(TWO_ZONE_OSM_PATH);
    assert!(result.is_ok());
    let schema = result.unwrap();

    assert!(
        !schema.constructions.wall.layers.is_empty(),
        "Wall should have layers"
    );
    assert!(
        !schema.constructions.roof.layers.is_empty(),
        "Roof should have layers"
    );
    assert!(
        !schema.constructions.floor.layers.is_empty(),
        "Floor should have layers"
    );

    assert_eq!(
        schema.constructions.wall.layers.len(),
        3,
        "ExtWall should have 3 layers (mat-w0, mat-w1, mat-w2)"
    );
    assert_eq!(
        schema.constructions.roof.layers.len(),
        3,
        "Roof should have 3 layers"
    );
    assert_eq!(
        schema.constructions.floor.layers.len(),
        3,
        "Floor should have 3 layers"
    );
}

#[test]
fn test_osm_parser_extracts_weather() {
    let result = import_osm(TWO_ZONE_OSM_PATH);
    assert!(result.is_ok());
    let schema = result.unwrap();

    match &schema.weather {
        fluxion::api::schema::WeatherData::TmyLocation { location } => {
            assert!(
                location.contains("40"),
                "Location should contain latitude 40"
            );
            assert!(
                location.contains("-105"),
                "Location should contain longitude -105"
            );
        }
        _ => panic!("Expected TmyLocation weather data"),
    }
}

#[test]
fn test_osm_parser_low_level_parse() {
    let osm_content =
        std::fs::read_to_string(TWO_ZONE_OSM_PATH).expect("Should read two_zone.osm fixture");

    let mut parser = OsmParser::new();
    let result = parser.parse_content(&osm_content);
    assert!(result.is_ok(), "Parser should succeed");

    let doc = result.unwrap();
    assert_eq!(doc.version, "3.6.0", "OSM version should be 3.6.0");

    let materials: Vec<_> = doc
        .objects
        .iter()
        .filter(|o| o.object_type == "OS:Material")
        .collect();
    assert_eq!(materials.len(), 9, "Should have 9 materials");

    let constructions: Vec<_> = doc
        .objects
        .iter()
        .filter(|o| o.object_type == "OS:Construction")
        .collect();
    assert_eq!(constructions.len(), 3, "Should have 3 constructions");

    let thermal_zones: Vec<_> = doc
        .objects
        .iter()
        .filter(|o| o.object_type == "OS:ThermalZone")
        .collect();
    assert_eq!(thermal_zones.len(), 2, "Should have 2 thermal zones");

    let spaces: Vec<_> = doc
        .objects
        .iter()
        .filter(|o| o.object_type == "OS:Space")
        .collect();
    assert_eq!(spaces.len(), 2, "Should have 2 spaces");

    let surfaces: Vec<_> = doc
        .objects
        .iter()
        .filter(|o| o.object_type == "OS:Surface")
        .collect();
    assert_eq!(surfaces.len(), 6, "Should have 6 surfaces");
}

#[test]
fn test_osm_parser_preserves_material_properties() {
    let osm_content =
        std::fs::read_to_string(TWO_ZONE_OSM_PATH).expect("Should read two_zone.osm fixture");

    let mut parser = OsmParser::new();
    let result = parser
        .parse_content(&osm_content)
        .expect("Parser should succeed");

    let materials: Vec<_> = result
        .objects
        .iter()
        .filter(|o| o.object_type == "OS:Material")
        .collect();

    let plasterboard = materials
        .iter()
        .find(|m| m.fields.get("name").map(|s| s.as_str()) == Some("Plasterboard"))
        .expect("Should find Plasterboard");

    assert_eq!(
        plasterboard.fields.get("thickness").map(|s| s.as_str()),
        Some("0.012")
    );
    assert_eq!(
        plasterboard.fields.get("conductivity").map(|s| s.as_str()),
        Some("0.16")
    );
    assert_eq!(
        plasterboard.fields.get("density").map(|s| s.as_str()),
        Some("950.0")
    );
    assert_eq!(
        plasterboard.fields.get("specific_heat").map(|s| s.as_str()),
        Some("840.0")
    );
}

#[test]
fn test_osm_parser_preserves_construction_layers() {
    let osm_content =
        std::fs::read_to_string(TWO_ZONE_OSM_PATH).expect("Should read two_zone.osm fixture");

    let mut parser = OsmParser::new();
    let result = parser
        .parse_content(&osm_content)
        .expect("Parser should succeed");

    let constructions: Vec<_> = result
        .objects
        .iter()
        .filter(|o| o.object_type == "OS:Construction")
        .collect();

    let ext_wall = constructions
        .iter()
        .find(|c| c.fields.get("name").map(|s| s.as_str()) == Some("ExtWall"))
        .expect("Should find ExtWall construction");

    assert_eq!(
        ext_wall.fields.get("layer_0").map(|s| s.as_str()),
        Some("{mat-w0}")
    );
    assert_eq!(
        ext_wall.fields.get("layer_1").map(|s| s.as_str()),
        Some("{mat-w1}")
    );
    assert_eq!(
        ext_wall.fields.get("layer_2").map(|s| s.as_str()),
        Some("{mat-w2}")
    );
}

#[test]
fn test_osm_parser_preserves_geometry() {
    let result = import_osm(TWO_ZONE_OSM_PATH).expect("Should load OSM");
    let schema = result;

    assert!((schema.geometry.total_floor_area - 125.0).abs() < 1e-6);
    assert!((schema.geometry.total_volume - (135.0 + 202.5)).abs() < 1e-6);
    assert_eq!(schema.geometry.number_of_floors, 1);
    assert!((schema.geometry.floor_height - 2.7).abs() < 1e-6);
}

#[test]
fn test_osm_roundtrip_zone_properties() {
    let result = import_osm(TWO_ZONE_OSM_PATH).expect("Should load OSM");
    let schema = result;

    for zone in &schema.geometry.zones {
        assert!(!zone.name.is_empty(), "Zone should have a name");
        assert!(
            zone.floor_area > 0.0,
            "Zone should have positive floor area"
        );
        assert!(zone.volume > 0.0, "Zone should have positive volume");
        assert!(zone.height > 0.0, "Zone should have positive height");
    }
}

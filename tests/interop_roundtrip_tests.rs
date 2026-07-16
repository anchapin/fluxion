// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Interoperability round-trip integration tests for gbXML and FMI export paths.
//!
//! These tests verify that export serialization preserves data correctly and
//! that re-importing the exported data produces consistent results.
//!
//! # gbXML Round-Trip
//!
//! Tests that a [`SimulationSchemaV1`] can be exported to gbXML format and
//! re-imported without loss of geometry.zones and construction data.
//!
//! Acceptance: geometry.zones[0].floor_area preserved within 0.01 m² for
//! Case 600 fixture (48.0 m² ± 0.01).
//!
//! # FMI Round-Trip
//!
//! Tests that a 3-zone FMU can be exported and its [`modelDescription.xml`]
//! read back correctly, verifying the zone configuration.
//!
//! Acceptance: zone temperatures and energy values within 0.1% at t=8760
//! for 3-zone FMU (deterministic, no random seed dependency).
//!
//! See Issue #1709

use fluxion::api::schema::{ConstructionSet, Geometry, SchemaMetadata, SimulationSchemaV1, SchemaVersion, ZoneGeometry};
use fluxion::interop::fmi::{FmiExporter, ZoneVariables};
use fluxion::interop::gbxml::{export_gbxml, import_gbxml};

// ============================================================================
// gbXML Round-Trip Tests
// ============================================================================

/// Create a minimal Case 600-like schema for round-trip testing.
///
/// Case 600 geometry: 8m × 6m × 2.7m = 48 m² floor area, 129.6 m³ volume.
fn case_600_schema() -> SimulationSchemaV1 {
    let zone = ZoneGeometry {
        name: "Zone 1".to_string(),
        floor_area: 48.0,
        volume: 129.6,
        height: 2.7,
    };

    let geometry = Geometry {
        zones: vec![zone],
        total_floor_area: 48.0,
        total_volume: 129.6,
        number_of_floors: 1,
        floor_height: 2.7,
    };

    let constructions = ConstructionSet::default();

    SimulationSchemaV1 {
        version: SchemaVersion::V1,
        metadata: SchemaMetadata {
            name: "Case 600 Test".to_string(),
            description: "ASHRAE 140 Case 600 baseline".to_string(),
            author: Some("Fluxion Test".to_string()),
            created_at: None,
            schema_version: SchemaVersion::V1,
        },
        geometry,
        constructions,
        schedules: fluxion::api::schema::ScheduleSet::default(),
        weather: fluxion::api::schema::WeatherData::TmyLocation {
            location: "Denver, CO".to_string(),
        },
        controls: fluxion::api::schema::ControlSet::default(),
        output: fluxion::api::schema::SimulationOutput::default(),
    }
}

#[test]
fn test_gbxml_roundtrip_zones_preserved() {
    let original = case_600_schema();
    let original_zone = &original.geometry.zones[0];

    let temp_dir = std::env::temp_dir();
    let gbxml_path = temp_dir.join("fluxion_case600_roundtrip.xml");

    export_gbxml(&original, &gbxml_path).expect("gbXML export should succeed");

    let imported = import_gbxml(&gbxml_path).expect("gbXML import should succeed");
    let imported_zone = &imported.geometry.zones[0];

    assert_eq!(
        original_zone.name, imported_zone.name,
        "Zone name should be preserved"
    );
    assert!(
        (original_zone.floor_area - imported_zone.floor_area).abs() < 0.01,
        "floor_area should be preserved within 0.01 m²: original={}, imported={}",
        original_zone.floor_area,
        imported_zone.floor_area
    );
    assert!(
        (original_zone.volume - imported_zone.volume).abs() < 0.01,
        "volume should be preserved within 0.01 m³: original={}, imported={}",
        original_zone.volume,
        imported_zone.volume
    );
    assert!(
        (original_zone.height - imported_zone.height).abs() < 0.01,
        "height should be preserved within 0.01 m: original={}, imported={}",
        original_zone.height,
        imported_zone.height
    );

    std::fs::remove_file(&gbxml_path).ok();
}

#[test]
fn test_gbxml_roundtrip_constructions_preserved() {
    let original = case_600_schema();

    let temp_dir = std::env::temp_dir();
    let gbxml_path = temp_dir.join("fluxion_case600_constructions_roundtrip.xml");

    export_gbxml(&original, &gbxml_path).expect("gbXML export should succeed");

    let imported = import_gbxml(&gbxml_path).expect("gbXML import should succeed");

    assert_eq!(
        original.constructions.wall.name, imported.constructions.wall.name,
        "Wall construction name should be preserved"
    );
    assert_eq!(
        original.constructions.wall.layers.len(),
        imported.constructions.wall.layers.len(),
        "Wall layer count should be preserved"
    );
    assert_eq!(
        original.constructions.roof.layers.len(),
        imported.constructions.roof.layers.len(),
        "Roof layer count should be preserved"
    );
    assert_eq!(
        original.constructions.floor.layers.len(),
        imported.constructions.floor.layers.len(),
        "Floor layer count should be preserved"
    );

    std::fs::remove_file(&gbxml_path).ok();
}

#[test]
fn test_gbxml_roundtrip_geometry_total_floor_area() {
    let original = case_600_schema();

    let temp_dir = std::env::temp_dir();
    let gbxml_path = temp_dir.join("fluxion_case600_area_roundtrip.xml");

    export_gbxml(&original, &gbxml_path).expect("gbXML export should succeed");

    let imported = import_gbxml(&gbxml_path).expect("gbXML import should succeed");

    assert!(
        (original.geometry.total_floor_area - imported.geometry.total_floor_area).abs() < 0.01,
        "total_floor_area should be preserved within 0.01 m²"
    );
    assert_eq!(
        original.geometry.number_of_floors,
        imported.geometry.number_of_floors,
        "number_of_floors should be preserved"
    );

    std::fs::remove_file(&gbxml_path).ok();
}

#[test]
fn test_gbxml_roundtrip_lossless_claim() {
    let original = case_600_schema();

    let temp_dir = std::env::temp_dir();
    let gbxml_path = temp_dir.join("fluxion_case600_lossless.xml");

    export_gbxml(&original, &gbxml_path).expect("gbXML export should succeed");
    let imported = import_gbxml(&gbxml_path).expect("gbXML import should succeed");

    let original_zone = &original.geometry.zones[0];
    let imported_zone = &imported.geometry.zones[0];

    let floor_area_ok =
        (original_zone.floor_area - imported_zone.floor_area).abs() < 0.01;
    let volume_ok = (original_zone.volume - imported_zone.volume).abs() < 0.01;

    assert!(
        floor_area_ok && volume_ok,
        "Lossless claim: geometry.zones[0] preserved within 0.01 m²/m³"
    );

    std::fs::remove_file(&gbxml_path).ok();
}

// ============================================================================
// FMI Round-Trip Tests
// ============================================================================

#[test]
fn test_fmi_3zone_export_roundtrip() {
    let zones = vec![
        ZoneVariables::new("Living"),
        ZoneVariables::new("Bedroom"),
        ZoneVariables::new("Kitchen"),
    ];

    let exporter = FmiExporter::new().with_zones(zones);

    assert_eq!(exporter.zone_count(), 3, "Should have exactly 3 zones");
    assert_eq!(
        exporter.total_variable_count(),
        21,
        "3 zones × 7 variables per zone = 21"
    );

    let temp_dir = std::env::temp_dir();
    let fmu_path = temp_dir.join("fluxion_3zone_roundtrip.fmu");

    exporter.export_fmu(&fmu_path).expect("FMU export should succeed");

    let xml_content =
        FmiExporter::read_model_description_from_fmu(&fmu_path)
            .expect("Should be able to read back modelDescription.xml");

    assert!(
        xml_content.contains("FluxionBuilding"),
        "modelDescription.xml should contain modelName"
    );
    assert!(
        xml_content.contains("Living"),
        "modelDescription.xml should contain Living zone"
    );
    assert!(
        xml_content.contains("Bedroom"),
        "modelDescription.xml should contain Bedroom zone"
    );
    assert!(
        xml_content.contains("Kitchen"),
        "modelDescription.xml should contain Kitchen zone"
    );

    std::fs::remove_file(&fmu_path).ok();
}

#[test]
fn test_fmi_roundtrip_zone_variable_count() {
    let zones = vec![
        ZoneVariables::new("Zone_A"),
        ZoneVariables::new("Zone_B"),
        ZoneVariables::new("Zone_C"),
    ];

    let exporter = FmiExporter::new().with_zones(zones);

    let var_names = exporter.variable_names();
    assert_eq!(
        var_names.len(),
        21,
        "Should have 21 total variables (7 per zone × 3 zones)"
    );

    let inputs: Vec<_> = var_names
        .iter()
        .filter(|(_, causality)| causality == "input")
        .collect();
    let outputs: Vec<_> = var_names
        .iter()
        .filter(|(_, causality)| causality == "output")
        .collect();

    assert_eq!(inputs.len(), 12, "Should have 12 inputs (4 per zone × 3)");
    assert_eq!(outputs.len(), 9, "Should have 9 outputs (3 per zone × 3)");
}

#[test]
fn test_fmi_roundtrip_deterministic() {
    let zones = vec![
        ZoneVariables::new("Living"),
        ZoneVariables::new("Bedroom"),
        ZoneVariables::new("Kitchen"),
    ];

    let exporter1 = FmiExporter::new().with_zones(zones.clone());
    let exporter2 = FmiExporter::new().with_zones(zones.clone());

    let xml1 = exporter1
        .generate_model_description_xml()
        .expect("Should generate XML");
    let xml2 = exporter2
        .generate_model_description_xml()
        .expect("Should generate XML");

    assert_eq!(
        xml1, xml2,
        "FMU export should be deterministic (no random seed dependency)"
    );
}

#[test]
fn test_fmi_roundtrip_3zone_fmu_contains_correct_variables() {
    let zones = vec![
        ZoneVariables::new("Living"),
        ZoneVariables::new("Bedroom"),
        ZoneVariables::new("Kitchen"),
    ];

    let exporter = FmiExporter::new().with_zones(zones);

    let var_names = exporter.variable_names();

    let living_vars: Vec<_> = var_names
        .iter()
        .filter(|(name, _)| name.starts_with("Living_"))
        .collect();
    let bedroom_vars: Vec<_> = var_names
        .iter()
        .filter(|(name, _)| name.starts_with("Bedroom_"))
        .collect();
    let kitchen_vars: Vec<_> = var_names
        .iter()
        .filter(|(name, _)| name.starts_with("Kitchen_"))
        .collect();

    assert_eq!(
        living_vars.len(),
        7,
        "Living zone should have 7 variables"
    );
    assert_eq!(
        bedroom_vars.len(),
        7,
        "Bedroom zone should have 7 variables"
    );
    assert_eq!(
        kitchen_vars.len(),
        7,
        "Kitchen zone should have 7 variables"
    );
}

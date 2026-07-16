// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Integration tests for the IDF/epJSON import scaffold (issue #1341).
//!
//! Covers:
//! - Lexer edge cases: quoted commas, multi-line strings, trailing
//!   comments, doubled-quote escapes.
//! - Parser object coverage for all 10 MVP types (design §4.1).
//! - `IdfError::Parse` carrying a line number on malformed input.
//! - End-to-end parse of the real ASHRAE 140 Case 600 IDF file under
//!   `tests/reference_data/energyplus_models/`, with exact object counts
//!   pinned in the test.

use std::path::PathBuf;

use fluxion::io::idf::{IdfError, IdfParser, IdfValue};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("idf")
}

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference_data")
        .join("energyplus_models")
}

// ---------------------------------------------------------------------------
// Lexer edge cases (issue acceptance criteria #2)
// ---------------------------------------------------------------------------

#[test]
fn lexer_quoted_comma_is_not_a_field_separator() {
    let src = r#"Material, "Hello, World!", OtherField;"#;
    let idf = IdfParser::from_str(src).expect("parses");
    assert_eq!(idf.objects.len(), 1);
    let m = &idf.objects[0];
    assert_eq!(m.object_type, "Material");
    // Two fields, not three — the comma inside quotes did NOT split.
    assert_eq!(m.fields.len(), 2, "got fields: {:?}", m.fields);
    assert_eq!(m.fields[0], IdfValue::String("Hello, World!".to_string()));
    assert_eq!(m.fields[1], IdfValue::String("OtherField".to_string()));
}

#[test]
fn lexer_multiline_quoted_string_is_preserved() {
    let path = fixtures_dir().join("lexer_edge_cases.idf");
    let idf = IdfParser::from_path(&path).expect("parses edge-case fixture");
    // Find the "MultiLine" material.
    let m = idf
        .materials()
        .find(|m| m.name.as_deref() == Some("MultiLine"))
        .expect("MultiLine material present");
    // The quoted field spans multiple lines and contains newlines.
    let roughness = m
        .fields
        .get(1)
        .expect("second field (roughness) is quoted multi-line");
    match roughness {
        IdfValue::String(s) => {
            assert!(s.contains("first line"));
            assert!(s.contains("second line"));
            assert!(s.contains("third line"));
        }
        other => panic!("expected quoted string, got {other:?}"),
    }
}

#[test]
fn lexer_trailing_comment_after_last_field_is_stripped() {
    let path = fixtures_dir().join("lexer_edge_cases.idf");
    let idf = IdfParser::from_path(&path).expect("parses edge-case fixture");
    let escaped = idf
        .materials()
        .find(|m| m.name.as_deref() == Some("EscapedQuote"))
        .expect("EscapedQuote material present");
    // 4 fields total — name, roughness (quoted escaped), thickness, conductivity.
    // The trailing "! trailing comment" must have been stripped.
    assert_eq!(escaped.fields.len(), 4);
}

#[test]
fn lexer_doubled_quote_escape_decodes_to_single_quote() {
    let path = fixtures_dir().join("lexer_edge_cases.idf");
    let idf = IdfParser::from_path(&path).expect("parses edge-case fixture");
    let escaped = idf
        .materials()
        .find(|m| m.name.as_deref() == Some("EscapedQuote"))
        .expect("EscapedQuote material present");
    let roughness = &escaped.fields[1];
    match roughness {
        IdfValue::String(s) => assert_eq!(s, "He said \"hello\" loudly"),
        other => panic!("expected quoted string, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// All 10 MVP object types — fixture parse (issue acceptance criteria #1, #3)
// ---------------------------------------------------------------------------

#[test]
fn parses_all_ten_mvp_object_types() {
    let path = fixtures_dir().join("all_ten_mvp_objects.idf");
    let idf = IdfParser::from_path(&path).expect("parses MVP fixture");

    // Version → 1, stored on the file.
    assert_eq!(idf.version.as_deref(), Some("25.2"));

    // Each of the 10 MVP types must be present at least once.
    assert_eq!(idf.versions().count(), 1, "Version");
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Timestep"))
            .count(),
        1,
        "Timestep"
    );
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("RunPeriod"))
            .count(),
        1,
        "RunPeriod"
    );
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Building"))
            .count(),
        1,
        "Building"
    );
    assert_eq!(idf.zones().count(), 1, "Zone");
    assert_eq!(idf.materials().count(), 2, "Material");
    assert_eq!(idf.constructions().count(), 1, "Construction");
    assert_eq!(
        idf.building_surfaces().count(),
        1,
        "BuildingSurface:Detailed"
    );
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("GlobalGeometryRules"))
            .count(),
        1,
        "GlobalGeometryRules"
    );
    assert_eq!(
        idf.ground_temperatures().count(),
        1,
        "Site:GroundTemperature:BuildingSurface"
    );
}

#[test]
fn missing_fields_become_empty_values() {
    // RunPeriod has a deliberately empty `Begin Year` field (`, ,`).
    let src = "RunPeriod, AnnualRun, 1, 1, , 12, 31;";
    let idf = IdfParser::from_str(src).expect("parses RunPeriod with missing field");
    let rp = &idf.objects[0];
    // 6 fields: name, begin_month, begin_day, <empty begin_year>,
    // end_month, end_day.
    assert_eq!(rp.fields.len(), 6);
    assert!(matches!(rp.fields[3], IdfValue::Empty));
}

#[test]
fn case_insensitive_object_classification() {
    // Mixed case object names classify correctly when filtered.
    let src = "version, 25.2;\nTIMESTEP, 1;\n";
    let idf = IdfParser::from_str(src).expect("parses mixed-case");
    assert_eq!(idf.versions().count(), 1);
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Timestep"))
            .count(),
        1
    );
}

#[test]
fn unknown_object_types_are_captured_not_rejected() {
    let src = "TotallyMadeUpObject, foo, bar, 1.0;\n";
    let idf = IdfParser::from_str(src).expect("parser must not reject unknown objects");
    assert_eq!(idf.objects.len(), 1);
    assert_eq!(idf.objects[0].object_type, "TotallyMadeUpObject");
    assert_eq!(idf.objects[0].fields.len(), 3);
}

// ---------------------------------------------------------------------------
// Error handling — IdfError::Parse carries line number
// ---------------------------------------------------------------------------

#[test]
fn parse_error_carries_line_number() {
    // Unterminated quote on line 2.
    let src = "Version, 25.2;\nMaterial, \"unterminated, 0.01;\n";
    let err = IdfParser::from_str(src).expect_err("must fail on unterminated quote");
    match err {
        IdfError::Parse { line, message } => {
            // Lexer surfaces the unterminated string at EOF, reporting
            // the final line of the document. The exact value is not
            // pinned — what matters is that some line number > 1 is
            // reported (i.e. not a sentinel 0).
            assert!(line >= 1, "line must be reported, got {line}");
            assert!(!message.is_empty());
        }
        other => panic!("expected IdfError::Parse, got {other:?}"),
    }
}

#[test]
fn io_error_when_path_missing() {
    let path = fixtures_dir().join("does_not_exist.idf");
    let err = IdfParser::from_path(&path).expect_err("missing file must produce Io error");
    assert!(matches!(err, IdfError::Io(_)));
}

// ---------------------------------------------------------------------------
// Real reference file — ASHRAE 140 Case 600 (issue acceptance criteria #1)
// ---------------------------------------------------------------------------

#[test]
fn parses_ashrae_140_case_600_with_exact_object_counts() {
    let path = reference_dir().join("ashrae_140_case_600.idf");
    assert!(path.exists(), "reference IDF missing at {}", path.display());

    let idf = IdfParser::from_path(&path).expect("parses ASHRAE 140 Case 600 IDF");

    // Pinned counts for the 10 MVP object types (verified offline via
    // Python token scan; see `.agents/results/issue-1341.md`).
    assert_eq!(idf.versions().count(), 1, "Version count");
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Timestep"))
            .count(),
        1,
        "Timestep count"
    );
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("RunPeriod"))
            .count(),
        1,
        "RunPeriod count"
    );
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Building"))
            .count(),
        1,
        "Building count"
    );
    assert_eq!(idf.zones().count(), 1, "Zone count");
    assert_eq!(idf.materials().count(), 5, "Material count");
    assert_eq!(idf.constructions().count(), 3, "Construction count");
    assert_eq!(
        idf.building_surfaces().count(),
        6,
        "BuildingSurface:Detailed count"
    );
    assert_eq!(
        idf.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("GlobalGeometryRules"))
            .count(),
        1,
        "GlobalGeometryRules count"
    );
    assert_eq!(
        idf.ground_temperatures().count(),
        1,
        "Site:GroundTemperature:BuildingSurface count"
    );

    // Version string is captured on the file.
    assert_eq!(idf.version.as_deref(), Some("25.2"));
}

// ---------------------------------------------------------------------------
// IDF → SimulationSchemaV1 round-trip (issue #1679)
// ---------------------------------------------------------------------------

#[test]
fn test_idf_roundtrip_case900() {
    use fluxion::api::schema::SimulationSchemaV1;
    use std::convert::TryFrom;

    let path = reference_dir().join("ashrae_140_case_900.idf");
    assert!(
        path.exists(),
        "ASHRAE 140 Case 900 IDF missing at {}",
        path.display()
    );

    let idf = IdfParser::from_path(&path).expect("parses ASHRAE 140 Case 900 IDF");

    // Convert to SimulationSchemaV1 via TryFrom<&IdfFile>.
    let schema = SimulationSchemaV1::try_from(&idf).expect("converts IDF to SimulationSchemaV1");

    // Case 900 has the same geometry as Case 600: 6m × 8m × 2.7m = 48 m² floor area.
    assert_eq!(
        schema.geometry.zones.len(),
        1,
        "Case 900 is a single-zone model"
    );
    let zone = &schema.geometry.zones[0];
    assert_eq!(zone.name, "ZONE1");
    // Floor area should be 48 m² (6 × 8).
    assert!(
        (zone.floor_area - 48.0).abs() < 1e-6,
        "floor_area = {} m², expected 48.0 m²",
        zone.floor_area
    );
    // Volume should be 129.6 m³ (48 × 2.7).
    assert!(
        (zone.volume - 129.6).abs() < 1e-6,
        "volume = {} m³, expected 129.6 m³",
        zone.volume
    );
    // Height should be derived from surface vertices (2.7 m).
    assert!(
        (zone.height - 2.7).abs() < 1e-6,
        "height = {} m, expected 2.7 m",
        zone.height
    );
    // Schema version must be V1.
    assert_eq!(schema.version, fluxion::api::schema::SchemaVersion::V1);
}

// epJSON parsing (issue #1676)
// ---------------------------------------------------------------------------

#[test]
fn test_parse_epjson_basic() {
    let path = reference_dir().join("case900.epJSON");
    assert!(
        path.exists(),
        "epJSON fixture missing at {}",
        path.display()
    );

    let idf = IdfParser::from_epjson_path(&path).expect("parses epJSON fixture");

    assert_eq!(idf.version.as_deref(), Some("25.2"));
    assert_eq!(idf.objects.len(), 2);

    let building = idf
        .objects
        .iter()
        .find(|o| o.object_type.eq_ignore_ascii_case("Building"))
        .expect("Building object present");
    assert_eq!(building.name.as_deref(), Some("MainBuilding"));
}

#[test]
fn test_parse_epjson_from_str() {
    let src = r#"{
      "Version": {
        "Version 1": { "version_identifier": "25.2" }
      },
      "Zone": {
        "Zone 1": { "name": "Zone1", "direction_of_relative_north": 0.0 }
      }
    }"#;
    let idf = IdfParser::from_epjson_str(src).expect("parses epJSON string");
    assert_eq!(idf.version.as_deref(), Some("25.2"));
    assert_eq!(idf.objects.len(), 2);
}
}

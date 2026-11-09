// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Round-trip tests for IDF export (issue #3991, export slice).
//!
//! Added by the supervisor while collecting a worker whose process died
//! mid-run: the worker's `src/io/idf/export.rs` shipped unit tests only,
//! so these integration tests pin the brief's actual acceptance criteria:
//!
//! - IDF -> export -> IDF preserves every parsed object of the all-ten
//!   MVP fixture and of a real reference model (Case 600).
//! - Export is idempotent: export(parse(export(x))) is byte-identical.
//! - IDF -> CaseSpec -> IDF preserves the envelope: the exported text
//!   re-parses and converts back to a CaseSpec with the same geometry.

use std::convert::TryFrom;
use std::path::PathBuf;

use fluxion::api::schema::{SimulationSchemaV1, WindowSpec, ZoneGeometry};
use fluxion::io::idf::export::export_schema;
use fluxion::io::idf::{
    case_spec_from_idf, export_case_spec_envelope, export_idf, IdfFile, IdfParser, IdfValue,
};
use fluxion::sim::hvac::HVACSystemType;
use fluxion::validation::ashrae_140_cases::Orientation;

/// The semantic content of a parsed file: object type, name and fields,
/// in order. `IdfObject::line` is source provenance (the line the object
/// started on in the text it was parsed from), not model content — the
/// exporter writes one object per line, so line numbers legitimately
/// change across a round-trip while the model must not.
fn semantic_objects(idf: &IdfFile) -> Vec<(String, Option<String>, Vec<IdfValue>)> {
    idf.objects
        .iter()
        .map(|o| (o.object_type.clone(), o.name.clone(), o.fields.clone()))
        .collect()
}

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

#[test]
fn all_ten_mvp_fixture_round_trips_object_for_object() {
    let path = fixtures_dir().join("all_ten_mvp_objects.idf");
    let idf = IdfParser::from_path(&path).expect("fixture parses");
    assert!(!idf.objects.is_empty(), "fixture must contain objects");

    let text = export_idf(&idf);
    let reparsed = IdfParser::from_str(&text).expect("exported text re-parses");
    assert_eq!(
        semantic_objects(&reparsed),
        semantic_objects(&idf),
        "round-trip must preserve every object and field"
    );

    // Idempotence: exporting the re-parse is byte-identical.
    assert_eq!(export_idf(&reparsed), text);
}

#[test]
fn case_600_reference_model_round_trips_object_for_object() {
    let path = reference_dir().join("ashrae_140_case_600.idf");
    let idf = IdfParser::from_path(&path).expect("case 600 parses");
    assert!(
        !idf.objects.is_empty(),
        "reference model must contain objects"
    );

    let text = export_idf(&idf);
    let reparsed = IdfParser::from_str(&text).expect("exported text re-parses");
    assert_eq!(
        semantic_objects(&reparsed),
        semantic_objects(&idf),
        "round-trip must preserve every object and field"
    );
    assert_eq!(export_idf(&reparsed), text);
}

#[test]
fn case_spec_envelope_export_reparses_to_the_same_geometry() {
    let path = reference_dir().join("ashrae_140_case_600.idf");
    let idf = IdfParser::from_path(&path).expect("case 600 parses");
    let case = case_spec_from_idf(&idf, "600").expect("case spec converts");
    assert!(!case.geometry.is_empty(), "case must carry geometry");

    let text = export_case_spec_envelope(&case);
    let reparsed = IdfParser::from_str(&text).expect("envelope export re-parses");

    // The envelope object types must all be present in the export.
    for object_type in [
        "Zone",
        "Material",
        "Construction",
        "BuildingSurface:Detailed",
    ] {
        assert!(
            reparsed
                .objects
                .iter()
                .any(|o| o.object_type == object_type),
            "envelope export is missing a {object_type} object; text was:\n{text}"
        );
    }

    // And the exported envelope converts back to a CaseSpec whose
    // geometry matches the original.
    let round_tripped =
        case_spec_from_idf(&reparsed, "600").expect("exported envelope converts back");
    assert_eq!(round_tripped.geometry.len(), case.geometry.len());
}

// ---------------------------------------------------------------------------
// Schema export (issue #4702)
// ---------------------------------------------------------------------------

/// A schema model with one zone, a 12 m² south-facing window
/// (U = 1.5 W/m²K, SHGC = 0.35) and the `Ideal` HVAC template.
fn schema_with_south_window() -> SimulationSchemaV1 {
    let mut schema = SimulationSchemaV1::default();
    schema.metadata.name = "FluxionSchemaExport".to_string();
    schema.geometry.zones = vec![ZoneGeometry {
        name: "ZONE1".to_string(),
        floor_area: 48.0,
        volume: 129.6,
        height: 2.7,
    }];
    schema.geometry.total_floor_area = 48.0;
    schema.geometry.total_volume = 129.6;
    schema.constructions.wall.window = Some(WindowSpec {
        window_area: 12.0,
        window_u_value: 1.5,
        window_shgc: 0.35,
        window_to_wall_ratio: None,
        surface: None,
        orientation: Some(Orientation::South),
    });
    // The roof / floor constructions carry no glazing in this model.
    schema.constructions.roof.window = None;
    schema.constructions.floor.window = None;
    schema.controls.hvac_system = Some(HVACSystemType::Ideal);
    schema
}

/// The schema export round-trips the window area and the U-value through
/// the **CaseSpec importer path** (`case_spec_from_idf`), which derives the
/// area from the `FenestrationSurface:Detailed` polygon and the U-value
/// from the construction's `WindowMaterial:SimpleGlazingSystem`.
#[test]
fn schema_export_round_trips_the_window_area_and_u_value() {
    let schema = schema_with_south_window();
    let text = export_schema(&schema);

    let reparsed = IdfParser::from_str(&text).expect("exported schema re-parses");
    assert!(
        reparsed.fenestration_surfaces().next().is_some(),
        "the export must contain a FenestrationSurface:Detailed; text was:\n{text}"
    );

    let case = case_spec_from_idf(&reparsed, "schema").expect("exported schema converts");
    let windows = &case.windows[0];
    let south = windows
        .iter()
        .find(|w| w.orientation == Orientation::South)
        .expect("the exported south window must re-import as a south window");
    assert!(
        (south.area - 12.0).abs() <= 1e-6,
        "window area must round-trip: expected 12.0, got {}",
        south.area
    );
    assert!(
        (case.window_properties.u_value - 1.5).abs() <= 1e-6,
        "window U-value must round-trip: expected 1.5, got {}",
        case.window_properties.u_value
    );
}

/// The schema export round-trips the `Ideal` HVAC assignment through the
/// **schema importer path** (`TryFrom<&IdfFile> for SimulationSchemaV1`),
/// which reads the `ZoneHVAC:IdealLoadsAirSystem` chain of the first zone.
#[test]
fn schema_export_round_trips_the_ideal_hvac_assignment() {
    let schema = schema_with_south_window();
    let text = export_schema(&schema);

    let reparsed = IdfParser::from_str(&text).expect("exported schema re-parses");
    let reimported = SimulationSchemaV1::try_from(&reparsed).expect("exported schema re-imports");
    assert_eq!(
        reimported.controls.hvac_system,
        Some(HVACSystemType::Ideal),
        "the re-imported schema must name the Ideal HVAC template"
    );
}

/// A non-`Ideal` HVAC template has no IDF equipment-chain mapping: the
/// export must say so explicitly instead of dropping it silently.
#[test]
fn non_ideal_hvac_template_emits_the_documented_loss_note() {
    let mut schema = schema_with_south_window();
    schema.controls.hvac_system = Some(HVACSystemType::VAV);

    let text = export_schema(&schema);
    assert!(
        text.contains(
            "! LOSS: HVAC template VAV — only the Ideal template maps to an IDF ZoneHVAC equipment chain; no HVAC objects were written"
        ),
        "the export must document the dropped HVAC template; text was:\n{text}"
    );
    assert!(
        !text.contains("ZoneHVAC:IdealLoadsAirSystem"),
        "a non-Ideal template must not write an ideal-loads chain; text was:\n{text}"
    );
}

/// A window whose glazing area cannot be determined is not representable:
/// the export must say so explicitly instead of dropping it silently.
#[test]
fn unexportable_window_emits_the_documented_loss_note() {
    let mut schema = schema_with_south_window();
    schema.constructions.wall.window = Some(WindowSpec {
        window_area: 0.0,
        window_u_value: 1.5,
        window_shgc: 0.35,
        window_to_wall_ratio: None,
        surface: None,
        orientation: Some(Orientation::South),
    });

    let text = export_schema(&schema);
    assert!(
        text.contains(
            "! LOSS: window on ZONE1-Wall-South — the glazing area is zero or undetermined (window_area <= 0 and no window_to_wall_ratio)"
        ),
        "the export must document the dropped window; text was:\n{text}"
    );
    assert!(
        !text.contains("FenestrationSurface:Detailed"),
        "an unexportable window must not be written as a fenestration surface; text was:\n{text}"
    );
}

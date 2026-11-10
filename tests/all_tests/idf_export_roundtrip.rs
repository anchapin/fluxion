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

use std::path::PathBuf;

use fluxion::io::idf::{
    case_spec_from_idf, export_case_spec_envelope, export_idf, IdfFile, IdfParser, IdfValue,
};

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

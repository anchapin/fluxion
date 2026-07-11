// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Acceptance tests for issue #1435 — `TryFrom<IdfFile> for SimulationSchemaV1`.
//!
//! The 18 reference IDF fixtures in `tests/reference_data/energyplus_models/`
//! are parsed by [`crate::io::idf::IdfParser::from_path`] and converted to
//! [`crate::api::schema::SimulationSchemaV1`] via the MVP converter
//! (`src/io/idf/convert.rs`, design §4.3). Each fixture's conversion is
//! expected to succeed without `IdfError` and yield a schema with the
//! expected geometry / construction summary.
//!
//! # Geometry expectations
//!
//! The ASHRAE 140-series fixtures (Cases 600, 900, 920, 950, 960) all model
//! a single 6 m × 8 m × 2.7 m zone (48 m² floor area, 129.6 m³ volume)
//! per `docs/idf-import-design.md` §4.1 and ASHRAE Standard 140 Annex B.
//! The lightweight step-change and ventilation fixtures use a single zone
//! with similar but smaller dimensions; the exact numbers below come from
//! the IDD / IDF files themselves.
//!
//! # Wall construction expectations
//!
//! Case 600 has a 3-layer wood-frame wall (`OUTR_WOOD` / `INSUL_R7` /
//! `GYP_13`) per the fixture comment at line 49; Case 900 has a single
//! 200 mm concrete layer (heavy mass). The conductivity check enforces
//! the issue's "within 1e-3 W/m·K" tolerance on the layer reading from
//! the IDF source.

use std::convert::TryFrom;
use std::path::{Path, PathBuf};

use fluxion::api::schema::SimulationSchemaV1;
use fluxion::io::idf::{IdfError, IdfFile, IdfParser};

/// All 18 reference IDF fixtures (per issue #1435).
const FIXTURES: &[&str] = &[
    "annual_solar_ventilation.idf",
    "ashrae_140_case_600.idf",
    "ashrae_140_case_900.idf",
    "ashrae_140_case_920.idf",
    "ashrae_140_case_950.idf",
    "ashrae_140_case_960.idf",
    "ashrae_140_solar_gain.idf",
    "fixed_inputs_zone_temp.idf",
    "step_change_composite.idf",
    "step_change_concrete.idf",
    "step_change_floor.idf",
    "step_change_lightweight.idf",
    "step_change_roof.idf",
    "ventilation_denver_01ach.idf",
    "ventilation_denver_05ach.idf",
    "ventilation_denver_10ach.idf",
    "ventilation_dulles_05ach.idf",
    "ventilation_tampa_05ach.idf",
];

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference_data")
        .join("energyplus_models")
}

/// Parse + convert a single fixture, returning the schema or the error.
fn convert_fixture(name: &str) -> Result<SimulationSchemaV1, IdfError> {
    let path = fixtures_dir().join(name);
    let idf = IdfParser::from_path(&path).expect("IDF parses");
    SimulationSchemaV1::try_from(idf)
}

/// Attempt to parse and convert every fixture, returning a list of
/// `(name, result)` pairs. Failures are collected (not propagated) so the
/// test summary can show which fixtures still need work.
fn convert_all() -> Vec<(&'static str, Result<SimulationSchemaV1, IdfError>)> {
    FIXTURES
        .iter()
        .map(|name| (*name, convert_fixture(name)))
        .collect()
}

// -----------------------------------------------------------------------------
// Bulk acceptance: every fixture must convert without IdfError
// -----------------------------------------------------------------------------

#[test]
fn all_eighteen_fixtures_convert_without_error() {
    let results = convert_all();
    let failures: Vec<_> = results
        .iter()
        .filter_map(|(name, r)| r.as_ref().err().map(|e| (*name, e.to_string())))
        .collect();
    if !failures.is_empty() {
        let summary: String = failures
            .iter()
            .map(|(name, err)| format!("  - {name}: {err}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "{} of {} fixtures failed to convert:\n{}",
            failures.len(),
            results.len(),
            summary
        );
    }
}

#[test]
fn case_600_schema_geometry() {
    let schema = convert_fixture("ashrae_140_case_600.idf").expect("Case 600 converts");
    assert_eq!(schema.geometry.zones.len(), 1, "Case 600 has 1 zone");
    let zone = &schema.geometry.zones[0];
    assert!(
        (zone.floor_area - 48.0).abs() < 1e-3,
        "floor_area ≈ 48 m², got {}",
        zone.floor_area
    );
    assert!(
        (zone.volume - 129.6).abs() < 1e-3,
        "volume ≈ 129.6 m³, got {}",
        zone.volume
    );
}

#[test]
fn case_600_wall_conductivity_matches_idf_within_1e_3() {
    // The IDF source (tests/reference_data/energyplus_models/ashrae_140_case_600.idf)
    // declares three layers with conductivities 0.115, 0.040, 0.160 W/m·K.
    let schema = convert_fixture("ashrae_140_case_600.idf").expect("Case 600 converts");
    // Wall construction — first Construction referenced by a Wall surface.
    let layers = &schema.constructions.wall.layers;
    assert_eq!(layers.len(), 3, "Case 600 wall has 3 layers");
    let expected_ks = [0.115, 0.040, 0.160];
    for (i, layer) in layers.iter().enumerate() {
        let diff = (layer.conductivity - expected_ks[i]).abs();
        assert!(
            diff < 1e-3,
            "layer {i} conductivity = {} (expected ≈ {}), diff = {diff}",
            layer.conductivity,
            expected_ks[i]
        );
    }
}

#[test]
fn case_600_metadata_includes_building_name() {
    let schema = convert_fixture("ashrae_140_case_600.idf").expect("Case 600 converts");
    assert_eq!(schema.metadata.name, "ASHRAE140_Case600");
    assert!(
        schema.metadata.description.contains("run_period"),
        "description should embed run-period summary, got: {}",
        schema.metadata.description
    );
}

#[test]
fn case_900_high_mass_detection() {
    let schema = convert_fixture("ashrae_140_case_900.idf").expect("Case 900 converts");
    assert_eq!(schema.geometry.zones.len(), 1, "Case 900 has 1 zone");
    // 200 mm concrete — should land in the wall layers with k ≈ 1.730.
    let wall = &schema.constructions.wall.layers;
    assert!(!wall.is_empty(), "Case 900 wall has at least one layer");
    let k = wall[0].conductivity;
    assert!(
        (k - 1.730).abs() < 1e-3,
        "concrete k = {k}, expected ≈ 1.730"
    );
}

#[test]
fn unsupported_version_is_rejected() {
    // Synthesize an IDF with an unsupported version.
    let idf = IdfParser::from_str("Version, 99.9;\n").unwrap();
    match SimulationSchemaV1::try_from(idf) {
        Err(IdfError::UnsupportedVersion(v)) => assert_eq!(v, "99.9"),
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }
}

#[test]
fn missing_version_is_rejected() {
    let idf: IdfFile = IdfParser::from_str("Timestep, 1;\n").unwrap();
    let err = SimulationSchemaV1::try_from(idf).unwrap_err();
    match err {
        IdfError::Conversion(_) => {}
        other => panic!("expected Conversion error, got {other:?}"),
    }
}

/// Smoke test: each fixture's schema serializes without panicking.
#[test]
fn all_schemas_serialize_to_json() {
    for (name, result) in convert_all() {
        let schema = result.unwrap_or_else(|e| {
            panic!("fixture {name} failed to convert: {e}");
        });
        serde_json::to_string(&schema)
            .unwrap_or_else(|e| panic!("fixture {name} JSON serialization failed: {e}"));
    }
}

/// Per-fixture report printed on test failure (or via `--nocapture`).
#[test]
fn report_per_fixture_outcome() {
    let results = convert_all();
    let n = results.len();
    let ok = results.iter().filter(|(_, r)| r.is_ok()).count();
    let bad = n - ok;
    println!("IDF→Schema conversion: {ok}/{n} succeeded, {bad} failed");
    for (name, r) in &results {
        match r {
            Ok(_) => println!("  [OK ] {name}"),
            Err(e) => println!("  [ERR] {name}: {e}"),
        }
    }
    assert_eq!(bad, 0, "{bad}/{n} fixtures failed to convert");
}

/// Helper used by the ASHRAE 140 acceptance integration test in
/// `tests/idf_ashrae_140_acceptance.rs` — exposed here so the same
/// fixture-parsing utility is reused.
pub fn load_idf(name: &str) -> Result<IdfFile, IdfError> {
    let path: &Path = &fixtures_dir().join(name);
    IdfParser::from_path(path)
}

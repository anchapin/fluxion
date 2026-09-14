//! Reference data loader with provenance tracking for ASHRAE 140 validation.
//!
//! This module loads reference data from `data/ashrae140_reference.json`,
//! which contains ASHRAE 140-2023 inter-program comparison ranges sourced from
//! published reference programs (EnergyPlus, TRNSYS, ESP-r, DOE-2).
//!
//! # Provenance
//!
//! - Source: ASHRAE 140-2023 Tables B8-1 through B8-5
//! - Programs: BSIMAC 9.0.74, CSE 0.861.1, DeST 2.0, EnergyPlus 9.0.1, ESP-r 13.3, TRNSYS 18.01.0001
//! - Reference: Std140_TF_Results.pdf (TESS, 19-Aug-2024)
//!
//! # Hash Verification
//!
//! SHA-256 checksum files (`.sha256`) are used to detect corruption or accidental
//! modification of reference data files.

use crate::util::sha256_hex::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Provenance block nested under `_schema.source` in the shipped reference
/// database.
///
/// Canonical shape: the nested-source layout emitted by the #667 generator
/// (`data/ashrae140_reference.json`). Issue #3759 introduced this struct when
/// the #748-era flat schema stopped matching the regenerated data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSource {
    /// Standard the data was sourced from (e.g. "ASHRAE 140-2023").
    pub standard: String,
    /// Section of the standard the test cases come from.
    pub section: String,
    /// Reference programs included in the inter-program comparison.
    pub programs: Vec<String>,
    /// Published results file the ranges were transcribed from.
    pub results_file: String,
    /// Mapping of metric name to its source table in the standard.
    pub tables: HashMap<String, String>,
    /// Mapping of metric name to its unit.
    pub units: HashMap<String, String>,
}

/// Metadata about the reference data source (`_schema` object).
///
/// Canonical shape: the nested-source layout emitted by the #667 generator;
/// Issue #3759 aligned this struct (and [`ReferenceSource`]) to it after the
/// shipped data stopped matching the #748-era flat schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSchema {
    /// Schema version (e.g. "1.0").
    pub version: String,
    /// Date the file was generated (e.g. "2026-05-12").
    pub generated: String,
    /// Provenance of the reference data.
    pub source: ReferenceSource,
    /// Human-readable description of the data set.
    pub description: String,
    /// Number of cases shipped in `cases`.
    pub total_cases: usize,
    /// Generator note about value sourcing.
    pub note: String,
}

/// Inter-program range with min, max, mean values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRange {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

/// Case reference data from ASHRAE 140-2023.
///
/// Load metrics are optional: free-float cases (`600FF`, `650FF`, `680FF`,
/// `900FF`, `950FF`, `980FF`) ship only free-float temperature ranges and no
/// heating/cooling loads, and case `960` ships only `ff_max_zone_temp_C`
/// (Issue #3759 — schema aligned to the #667-generated data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseReference {
    pub annual_heating_MWh: Option<MetricRange>,
    pub annual_cooling_MWh: Option<MetricRange>,
    pub peak_heating_kW: Option<MetricRange>,
    pub peak_cooling_kW: Option<MetricRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ff_max_zone_temp_C: Option<MetricRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ff_min_zone_temp_C: Option<MetricRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ff_mean_zone_temp_C: Option<MetricRange>,
}

/// Root structure of ashrae140_reference.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ashrae140ReferenceDb {
    #[serde(rename = "_schema")]
    pub schema: ReferenceSchema,
    pub cases: HashMap<String, CaseReference>,
}

/// Error types for reference data loading
#[derive(Debug, Clone)]
pub enum ReferenceLoaderError {
    FileNotFound(String),
    InvalidFormat(String),
    HashMismatch { expected: String, actual: String },
    ParseError(String),
}

impl std::fmt::Display for ReferenceLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReferenceLoaderError::FileNotFound(path) => {
                write!(f, "Reference data file not found: {}", path)
            }
            ReferenceLoaderError::InvalidFormat(msg) => {
                write!(f, "Invalid reference data format: {}", msg)
            }
            ReferenceLoaderError::HashMismatch { expected, actual } => {
                write!(f, "Hash mismatch: expected {}, got {}", expected, actual)
            }
            ReferenceLoaderError::ParseError(msg) => {
                write!(f, "Failed to parse reference data: {}", msg)
            }
        }
    }
}

impl std::error::Error for ReferenceLoaderError {}

/// Cached reference database load result.
///
/// Caching the full `Result` (rather than an `Option`) preserves the real
/// failure cause instead of laundering every error into "file not found"
/// (Issue #3721).
static REFERENCE_DB: std::sync::OnceLock<Result<Ashrae140ReferenceDb, ReferenceLoaderError>> =
    std::sync::OnceLock::new();

/// Default path to reference data
const DEFAULT_REFERENCE_PATH: &str = "data/ashrae140_reference.json";

/// Calculate SHA-256 hash of file contents
pub fn calculate_file_hash(path: &Path) -> Result<String, ReferenceLoaderError> {
    let content = fs::read(path).map_err(|e| {
        ReferenceLoaderError::FileNotFound(format!("Failed to read {}: {}", path.display(), e))
    })?;
    let hash = sha256_hex(Sha256::digest(&content));
    Ok(hash)
}

/// Verify file hash against expected SHA-256 checksum file
pub fn verify_file_hash(path: &Path) -> Result<(), ReferenceLoaderError> {
    let checksum_path = path.with_extension("sha256");
    if !checksum_path.exists() {
        return Ok(());
    }
    let expected = fs::read_to_string(&checksum_path)
        .map_err(|e| {
            ReferenceLoaderError::InvalidFormat(format!("Failed to read checksum: {}", e))
        })?
        .trim()
        .to_string();
    let actual = calculate_file_hash(path)?;
    if expected != actual {
        return Err(ReferenceLoaderError::HashMismatch { expected, actual });
    }
    Ok(())
}

/// Load reference database from JSON file
pub fn load_reference_database(path: &str) -> Result<Ashrae140ReferenceDb, ReferenceLoaderError> {
    let file_path = Path::new(path);
    if !file_path.exists() {
        return Err(ReferenceLoaderError::FileNotFound(path.to_string()));
    }
    verify_file_hash(file_path)?;
    let content = fs::read_to_string(file_path)
        .map_err(|e| ReferenceLoaderError::InvalidFormat(format!("Failed to read file: {}", e)))?;
    let db: Ashrae140ReferenceDb = serde_json::from_str(&content)
        .map_err(|e| ReferenceLoaderError::ParseError(format!("JSON parse error: {}", e)))?;
    Ok(db)
}

/// Get the global reference database, loading from default path if not cached.
///
/// # Error propagation
///
/// The result of the first load attempt is cached for the lifetime of the
/// process. On failure, the underlying [`load_reference_database`] error
/// variant (`FileNotFound`, `ParseError`, `InvalidFormat`, or `HashMismatch`)
/// is propagated verbatim instead of being collapsed into a generic
/// "file not found" message (Issue #3721).
///
/// # Cache-failure semantics
///
/// The cache initializes at most once, so a failed load is **not retried on
/// subsequent calls**: fixing or restoring the reference file on disk has no
/// effect on an already-running process; the process must be restarted to
/// re-attempt the load. Retry support was deliberately not added because it
/// would require per-call locking on a hot lookup path, and a missing or
/// corrupt reference database is treated as a start-up failure of a
/// validation run, not a transient condition.
pub fn get_reference_db() -> Result<&'static Ashrae140ReferenceDb, ReferenceLoaderError> {
    REFERENCE_DB
        .get_or_init(|| load_reference_database(DEFAULT_REFERENCE_PATH))
        .as_ref()
        .map_err(Clone::clone)
}

/// Get benchmark data for a specific case
pub fn get_reference_case(case_id: &str) -> Result<Option<CaseReference>, ReferenceLoaderError> {
    let db = get_reference_db()?;
    Ok(db.cases.get(case_id).cloned())
}

/// Check if reference data exists for a case
pub fn has_case(case_id: &str) -> bool {
    get_reference_db()
        .map(|db| db.cases.contains_key(case_id))
        .unwrap_or(false)
}

/// Get source information for documentation
pub fn get_source_info() -> Option<String> {
    get_reference_db().ok().map(|db| {
        format!(
            "{} (schema v{}, {} programs: {})",
            db.schema.source.standard,
            db.schema.version,
            db.schema.source.programs.len(),
            db.schema.source.programs.join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_db_loading() {
        // Issue #3759: the REAL shipped reference database must parse. Load it
        // directly (independent of the process-global cache and of the test
        // binary's working directory) via the manifest-relative path.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let shipped_path = format!("{manifest_dir}/data/ashrae140_reference.json");
        let db = load_reference_database(&shipped_path)
            .expect("shipped data/ashrae140_reference.json must parse (Issue #3759)");
        assert!(
            db.schema.total_cases == db.cases.len(),
            "schema total_cases ({}) must match cases map ({})",
            db.schema.total_cases,
            db.cases.len()
        );
        assert!(db.cases.contains_key("195"), "Should have case 195");
        assert!(db.cases.contains_key("600"), "Should have case 600");
        assert!(
            db.cases.contains_key("600FF"),
            "Should have free-float case 600FF"
        );

        // Through the cached public entry point. On success the known keys
        // must resolve; any failure must surface its real cause verbatim —
        // FileNotFound names the actual path rather than the laundered
        // "Reference database not available" stub (Issue #3721), and a
        // ParseError on the shipped data is the Issue #3759 regression and
        // must fail this test loudly.
        match get_reference_db() {
            Ok(db) => {
                assert!(db.schema.total_cases > 0, "Should have cases");
                assert!(db.cases.contains_key("195"), "Should have case 195");
                assert!(db.cases.contains_key("600"), "Should have case 600");
            }
            Err(ReferenceLoaderError::FileNotFound(path)) => {
                // Issue #3721: FileNotFound must carry the real path, not the
                // laundered "Reference database not available" stub message.
                assert!(
                    path.contains(DEFAULT_REFERENCE_PATH) && !path.contains("not available"),
                    "FileNotFound must name the actual path, got: {path}"
                );
                println!("Reference file not found - this is expected in test environment");
            }
            Err(e) => {
                panic!("Reference DB load failed with real cause surfaced: {e}");
            }
        }
    }

    #[test]
    fn test_has_case() {
        // Issue #837: An identifier that does not exist in the reference DB
        // (or any identifier when the DB is missing entirely) must report `false`.
        assert!(
            !has_case("INVALID"),
            "has_case must return false for an identifier not in the reference DB"
        );

        // Issue #3759: the shipped reference database parses, so real case
        // keys must resolve through the public lookup path.
        assert!(
            has_case("195"),
            "case 195 must resolve in the shipped reference DB"
        );
        assert!(
            has_case("600"),
            "case 600 must resolve in the shipped reference DB"
        );
    }

    #[test]
    fn test_get_source_info() {
        // Issue #3759: the shipped DB parses, so source info must be
        // available; `standard` lives in the nested `source` object (#667
        // shape) and the formatted string must still name the standard.
        let info = get_source_info().expect("shipped reference DB must load for source info");
        assert!(
            info.contains("ASHRAE 140"),
            "source info must name the standard, got: {info}"
        );
    }

    #[test]
    fn test_corrupt_json_surfaces_parse_error() {
        // Issue #3721: exercised through `load_reference_database` because the
        // process-global cache behind `get_reference_db` may already hold a
        // successful load from another test in the same binary; this is the
        // exact error `get_reference_db` now propagates verbatim.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "fluxion_ref_loader_corrupt_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(path.with_extension("sha256"));
        std::fs::write(&path, "{ this is not valid JSON !!").expect("write corrupt fixture");

        let result = load_reference_database(path.to_str().expect("utf-8 temp path"));

        let _ = std::fs::remove_file(&path);

        match result {
            Err(ReferenceLoaderError::ParseError(msg)) => {
                assert!(
                    msg.contains("JSON parse error"),
                    "error must name the JSON parse failure, got: {msg}"
                );
                assert!(
                    msg.contains("line 1"),
                    "serde's line/column detail (the real cause) must be preserved, got: {msg}"
                );
            }
            Err(other) => panic!("expected ParseError for corrupt JSON, got: {other:?}"),
            Ok(_) => panic!("corrupt JSON must not load successfully"),
        }
    }
}

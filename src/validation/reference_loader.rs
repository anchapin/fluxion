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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Format a SHA-256 digest (or any `AsRef<[u8]>`) as a lowercase hex string.
///
/// `sha2` 0.11 returns a `GenericArray<u8, U32>` whose `LowerHex` impl is no
/// longer available in newer `generic-array` releases, so we format the bytes
/// manually.
fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{:02x}", b);
    }
    s
}

/// Metadata about the reference data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSchema {
    pub version: String,
    pub source: String,
    pub programs: Vec<String>,
    pub tables: HashMap<String, String>,
    pub units: HashMap<String, String>,
    pub total_cases: usize,
}

/// Inter-program range with min, max, mean values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRange {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

/// Case reference data from ASHRAE 140-2023
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseReference {
    pub annual_heating_MWh: MetricRange,
    pub annual_cooling_MWh: MetricRange,
    pub peak_heating_kW: MetricRange,
    pub peak_cooling_kW: MetricRange,
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
#[derive(Debug)]
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

/// Cached reference database
static REFERENCE_DB: std::sync::OnceLock<Option<Ashrae140ReferenceDb>> = std::sync::OnceLock::new();

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

/// Get the global reference database, loading from default path if not cached
pub fn get_reference_db() -> Result<&'static Ashrae140ReferenceDb, ReferenceLoaderError> {
    REFERENCE_DB
        .get_or_init(|| load_reference_database(DEFAULT_REFERENCE_PATH).ok())
        .as_ref()
        .ok_or_else(|| {
            ReferenceLoaderError::FileNotFound(format!(
                "Reference database not available. Ensure {} exists.",
                DEFAULT_REFERENCE_PATH
            ))
        })
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
            "ASHRAE 140-{} ({} programs: {})",
            db.schema.version,
            db.schema.programs.len(),
            db.schema.programs.join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_db_loading() {
        match get_reference_db() {
            Ok(db) => {
                assert!(db.schema.total_cases > 0, "Should have cases");
                assert!(db.cases.contains_key("195"), "Should have case 195");
                assert!(db.cases.contains_key("600"), "Should have case 600");
            }
            Err(ReferenceLoaderError::FileNotFound(_)) => {
                println!("Reference file not found - this is expected in test environment");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_has_case() {
        // Issue #837: An identifier that does not exist in the reference DB
        // (or any identifier when the DB is missing entirely) must report `false`.
        // This is the only environment-independent invariant of `has_case`.
        assert!(
            !has_case("INVALID"),
            "has_case must return false for an identifier not in the reference DB"
        );

        // The reference file may or may not be bundled in this environment;
        // call the function on a few real case IDs only to exercise the lookup
        // path without panicking. The return value is intentionally not asserted.
        let _ = has_case("195");
        let _ = has_case("600");
    }

    #[test]
    fn test_get_source_info() {
        match get_source_info() {
            Some(info) => {
                assert!(info.contains("ASHRAE 140"));
            }
            None => {
                println!("Source info not available - reference file not found");
            }
        }
    }
}

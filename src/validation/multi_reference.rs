use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Reference range for a single program (EnergyPlus, ESP-r, TRNSYS).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProgramRange {
    pub min: f64,
    pub max: f64,
}

/// Reference ranges for all metrics of a single test case.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CaseRefs {
    #[serde(rename = "annual_heating")]
    pub annual_heating: Option<HashMap<String, ProgramRange>>,
    #[serde(rename = "annual_cooling")]
    pub annual_cooling: Option<HashMap<String, ProgramRange>>,
    #[serde(rename = "peak_heating")]
    pub peak_heating: Option<HashMap<String, ProgramRange>>,
    #[serde(rename = "peak_cooling")]
    pub peak_cooling: Option<HashMap<String, ProgramRange>>,
    #[serde(rename = "min_free_float")]
    pub min_free_float: Option<HashMap<String, ProgramRange>>,
    #[serde(rename = "max_free_float")]
    pub max_free_float: Option<HashMap<String, ProgramRange>>,
}

/// Multi-reference database containing versioned reference ranges from multiple programs.
#[derive(Debug, Deserialize, Serialize)]
pub struct MultiReferenceDB {
    pub version: String,
    pub source: Option<String>,
    pub cases: HashMap<String, CaseRefs>,
}

impl MultiReferenceDB {
    /// Loads a multi-reference database from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let db: MultiReferenceDB = serde_json::from_str(content.trim())?;
        Ok(db)
    }

    /// Gets the reference ranges for a specific case and metric.
    ///
    /// # Arguments
    /// - `case_id`: Case identifier (e.g., "600", "900")
    /// - `metric`: One of "annual_heating", "annual_cooling", "peak_heating", "peak_cooling", "min_free_float", "max_free_float"
    ///
    /// # Returns
    /// A HashMap mapping program names to their respective min/max ranges, or `None` if not found.
    pub fn get_ranges(
        &self,
        case_id: &str,
        metric: &str,
    ) -> Option<&HashMap<String, ProgramRange>> {
        let case = self.cases.get(case_id)?;
        match metric {
            "annual_heating" => case.annual_heating.as_ref(),
            "annual_cooling" => case.annual_cooling.as_ref(),
            "peak_heating" => case.peak_heating.as_ref(),
            "peak_cooling" => case.peak_cooling.as_ref(),
            "min_free_float" => case.min_free_float.as_ref(),
            "max_free_float" => case.max_free_float.as_ref(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_multireference_loading() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        // Verify case 600 exists and has at least EnergyPlus data
        let case_600 = db.cases.get("600").expect("Case 600 not found");
        let ah = case_600
            .annual_heating
            .as_ref()
            .expect("annual_heating should exist");
        assert!(
            ah.contains_key("EnergyPlus"),
            "EnergyPlus missing for case 600 annual_heating"
        );
        // Note: ESP-r and TRNSYS data may not be available for all cases
        // Only verify they exist if the reference file includes them
        for (_, range) in ah {
            assert!(range.min < range.max, "min must be less than max");
        }

        // Verify version present
        assert!(!db.version.is_empty());
    }

    #[test]
    fn test_multireference_get_ranges_valid() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        let ranges = db.get_ranges("600", "annual_heating");
        assert!(ranges.is_some());
        let ranges = ranges.unwrap();
        assert!(ranges.contains_key("EnergyPlus"));
    }

    #[test]
    fn test_multireference_get_ranges_invalid_case() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        let ranges = db.get_ranges("nonexistent", "annual_heating");
        assert!(ranges.is_none());
    }

    #[test]
    fn test_multireference_get_ranges_invalid_metric() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        let ranges = db.get_ranges("600", "invalid_metric");
        assert!(ranges.is_none());
    }

    #[test]
    fn test_multireference_get_ranges_all_metrics() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        let valid_metrics = [
            "annual_heating",
            "annual_cooling",
            "peak_heating",
            "peak_cooling",
        ];
        for metric in &valid_metrics {
            let ranges = db.get_ranges("600", metric);
            assert!(
                ranges.is_some(),
                "Metric {} should exist for case 600",
                metric
            );
        }
    }

    #[test]
    fn test_multireference_missing_file() {
        let result = MultiReferenceDB::from_file(Path::new("/nonexistent/path.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_multireference_invalid_json() {
        use std::io::Write;
        let temp_dir = env::temp_dir();
        let file_path = temp_dir.join("fluxion_test_invalid_ref.json");
        {
            let mut file = std::fs::File::create(&file_path).unwrap();
            writeln!(file, "not valid json").unwrap();
        }

        let result = MultiReferenceDB::from_file(&file_path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_program_range_validation() {
        let range = ProgramRange { min: 1.0, max: 5.0 };
        assert!(range.min < range.max);

        let range_reversed = ProgramRange { min: 5.0, max: 1.0 };
        assert!(range_reversed.min > range_reversed.max);
    }

    #[test]
    fn test_case_refs_structure() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        let case_600 = db.cases.get("600").expect("Case 600 not found");

        // All metric maps should have at least EnergyPlus (since they are now Option)
        assert!(
            case_600
                .annual_heating
                .as_ref()
                .map(|m| !m.is_empty())
                .unwrap_or(false),
            "annual_heating should exist for case 600"
        );
        assert!(
            case_600
                .annual_cooling
                .as_ref()
                .map(|m| !m.is_empty())
                .unwrap_or(false),
            "annual_cooling should exist for case 600"
        );
        assert!(
            case_600
                .peak_heating
                .as_ref()
                .map(|m| !m.is_empty())
                .unwrap_or(false),
            "peak_heating should exist for case 600"
        );
        assert!(
            case_600
                .peak_cooling
                .as_ref()
                .map(|m| !m.is_empty())
                .unwrap_or(false),
            "peak_cooling should exist for case 600"
        );
    }

    #[test]
    fn test_multireference_db_serialization() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        // Serialize to JSON
        let json = serde_json::to_string(&db).expect("Failed to serialize");
        assert!(!json.is_empty());

        // Deserialize back
        let db2: MultiReferenceDB = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(db.version, db2.version);
        assert_eq!(db.cases.len(), db2.cases.len());
    }

    #[test]
    fn test_multireference_source_field() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("docs/ashrae_140_references.json");
        let db = MultiReferenceDB::from_file(&path).expect("Failed to load reference data");

        // Source field is optional, verify it's handled correctly
        let _ = &db.source;
    }
}

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Reference range for a single program (EnergyPlus, ESP-r, TRNSYS).
#[derive(Debug, Deserialize)]
pub struct ProgramRange {
    pub min: f64,
    pub max: f64,
}

/// Reference ranges for all metrics of a single test case.
#[derive(Debug, Deserialize)]
pub struct CaseRefs {
    #[serde(rename = "annual_heating")]
    pub annual_heating: HashMap<String, ProgramRange>,
    #[serde(rename = "annual_cooling")]
    pub annual_cooling: HashMap<String, ProgramRange>,
    #[serde(rename = "peak_heating")]
    pub peak_heating: HashMap<String, ProgramRange>,
    #[serde(rename = "peak_cooling")]
    pub peak_cooling: HashMap<String, ProgramRange>,
}

/// Multi-reference database containing versioned reference ranges from multiple programs.
#[derive(Debug, Deserialize)]
pub struct MultiReferenceDB {
    pub version: String,
    pub source: Option<String>,
    pub cases: HashMap<String, CaseRefs>,
}

impl MultiReferenceDB {
    /// Loads a multi-reference database from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let db: MultiReferenceDB = serde_json::from_str(&content)?;
        Ok(db)
    }

    /// Gets the reference ranges for a specific case and metric.
    ///
    /// # Arguments
    /// - `case_id`: Case identifier (e.g., "600", "900")
    /// - `metric`: One of "annual_heating", "annual_cooling", "peak_heating", "peak_cooling"
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
            "annual_heating" => Some(&case.annual_heating),
            "annual_cooling" => Some(&case.annual_cooling),
            "peak_heating" => Some(&case.peak_heating),
            "peak_cooling" => Some(&case.peak_cooling),
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

        // Verify case 600 exists and has expected programs
        let case_600 = db.cases.get("600").expect("Case 600 not found");
        let ah = &case_600.annual_heating;
        assert!(
            ah.contains_key("EnergyPlus"),
            "EnergyPlus missing for case 600 annual_heating"
        );
        assert!(
            ah.contains_key("ESP-r"),
            "ESP-r missing for case 600 annual_heating"
        );
        assert!(
            ah.contains_key("TRNSYS"),
            "TRNSYS missing for case 600 annual_heating"
        );
        for (_, range) in ah {
            assert!(range.min < range.max, "min must be less than max");
        }

        // Verify version present
        assert!(!db.version.is_empty());
    }
}

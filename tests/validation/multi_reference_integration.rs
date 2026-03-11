use anyhow::Result;
use fluxion::validation::{
    ASHRAE140Validator, multi_reference::MultiReferenceDB, reporter::ValidationReportGenerator,
    BenchmarkReport, MetricType,
};
use std::{fs, path::{Path, PathBuf}};
use tempfile::tempdir;

#[test]
fn test_multi_reference_enrichment_and_report() -> Result<()> {
    // Load validator - should auto-load multi-reference DB from docs/
    let validator = ASHRAE140Validator::new();
    // Run full validation (this may take a few minutes)
    let report = validator.validate_analytical_engine();

    // Verify enrichment for energy and peak metrics
    let mut found_enriched = false;
    for result in &report.results {
        match result.metric {
            MetricType::AnnualHeating
            | MetricType::AnnualCooling
            | MetricType::PeakHeating
            | MetricType::PeakCooling => {
                found_enriched = true;
                assert!(
                    result.per_program.is_some(),
                    "Missing per_program for {} - {}",
                    result.case_id,
                    result.metric
                );
                let per_prog = result.per_program.as_ref().unwrap();
                assert!(
                    per_prog.contains_key("EnergyPlus"),
                    "EnergyPlus missing for {} - {}",
                    result.case_id,
                    result.metric
                );
                // At least one of ESP-r or TRNSYS should be present for typical cases
                // (some cases may not have both, but our reference includes both)
                assert!(
                    per_prog.contains_key("ESP-r") || per_prog.contains_key("TRNSYS"),
                    "No secondary reference programs for {} - {}",
                    result.case_id,
                    result.metric
                );
            }
            _ => {
                // Free-floating and other metrics without multi-ref should have None
                assert!(
                    result.per_program.is_none(),
                    "Unexpected per_program for {} - {:?}",
                    result.case_id,
                    result.metric
                );
            }
        }
    }
    assert!(
        found_enriched,
        "No energy/peak metrics found; enrichment not tested"
    );

    // Generate markdown report and verify multi-reference table appears
    let temp = tempdir()?;
    let output_path = temp.path().join("validation_report.md");
    let generator = ValidationReportGenerator::new(output_path.clone());
    generator
        .generate(&report, None, None)
        .map_err(|e| anyhow::anyhow!("Failed to generate report: {}", e))?;

    let markdown = fs::read_to_string(&output_path)?;
    assert!(
        markdown.contains("## Multi-Reference Comparison"),
        "Multi-Reference Comparison section missing"
    );
    // Check that table includes some cases
    assert!(markdown.contains("| 600 |"), "Case 600 not in multi-ref table");
    assert!(markdown.contains("EnergyPlus"), "EnergyPlus column missing");
    assert!(markdown.contains("ESP-r"), "ESP-r column missing");
    assert!(markdown.contains("TRNSYS"), "TRNSYS column missing");

    Ok(())
}

#[test]
fn test_update_references_with_remote() -> Result<()> {
    use fluxion::validation::commands::update_references;
    use mockito::mock;
    use serde_json::json;

    // Prepare a mock response with a different version
    let mock_db = json!({
        "version": "2025-02",
        "source": "Remote Test",
        "cases": {
            "600": {
                "annual_heating": {
                    "EnergyPlus": { "min": 5.0, "max": 7.0 },
                    "ESP-r": { "min": 4.9, "max": 7.1 }
                },
                "annual_cooling": {
                    "EnergyPlus": { "min": 8.0, "max": 10.0 }
                },
                "peak_heating": {
                    "EnergyPlus": { "min": 3.0, "max": 4.0 }
                },
                "peak_cooling": {
                    "EnergyPlus": { "min": 5.0, "max": 6.0 }
                }
            }
        }
    }).to_string();

    let _mock = mock("GET", "/")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(&mock_db)
        .create();

    let url = mockito::SERVER_URL;

    // Use a temporary directory to avoid affecting the real repository
    let temp = tempdir()?;
    let original_cwd = std::env::current_dir()?;
    let _guard = DirGuard::new(temp.path())?;

    // Create docs directory (function expects it to be able to write)
    fs::create_dir("docs")?;

    // Call update_references
    let result = update_references(Some(&url));
    assert!(
        result.is_ok(),
        "update_references failed: {:?}",
        result.err()
    );

    // Verify file written
    let output_path = std::path::Path::new("docs/ashrae_140_references.json");
    assert!(output_path.exists(), "Reference file not created");
    let content = fs::read_to_string(output_path)?;
    let parsed: MultiReferenceDB = serde_json::from_str(&content)?;
    assert_eq!(parsed.version, "2025-02");
    assert_eq!(parsed.cases.len(), 1);

    Ok(())
}

/// Guard to restore working directory on drop
struct DirGuard(PathBuf);
impl DirGuard {
    fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let cur = std::env::current_dir()?;
        std::env::set_current_dir(path)?;
        Ok(DirGuard(cur))
    }
}
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

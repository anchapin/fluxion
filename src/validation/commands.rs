use anyhow::{Context, Result};
use reqwest::blocking::Client;
use std::fs;
use std::path::Path;

use crate::validation::multi_reference::MultiReferenceDB;

/// Update or validate the multi-reference database.
///
/// If a URL is provided, fetches the reference data from that location.
/// If no URL is provided, validates the local reference data file.
pub fn update_references(url: Option<&str>) -> Result<()> {
    match url {
        Some(remote_url) => {
            eprintln!("Fetching reference data from {}...", remote_url);
            // Build blocking HTTP client
            let client = Client::builder()
                .build()
                .context("Failed to build HTTP client")?;
            // Send GET request
            let response = client
                .get(remote_url)
                .send()
                .context("Failed to send HTTP request")?;
            // Check status
            if !response.status().is_success() {
                anyhow::bail!("HTTP request failed with status: {}", response.status());
            }
            // Parse JSON into MultiReferenceDB
            let db: MultiReferenceDB = response.json().context("Failed to parse JSON response")?;
            // Validate structure
            if db.version.is_empty() {
                anyhow::bail!("Invalid reference data: version is empty");
            }
            if db.cases.is_empty() {
                anyhow::bail!("Invalid reference data: no cases found");
            }
            // Sample check: ensure at least one case has annual_heating with EnergyPlus
            let (_, sample_case) = db
                .cases
                .iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Cases map is empty"))?;
            if sample_case.annual_heating.is_empty() {
                anyhow::bail!("Invalid reference data: sample case has no annual_heating programs");
            }
            if !sample_case.annual_heating.contains_key("EnergyPlus") {
                anyhow::bail!("Invalid reference data: annual_heating missing EnergyPlus");
            }

            let output_path = Path::new("docs/ashrae_140_references.json");
            // If file exists, check if version matches
            if output_path.exists() {
                // Load existing
                let existing = match MultiReferenceDB::from_file(output_path) {
                    Ok(db) => db,
                    Err(e) => {
                        anyhow::bail!(
                            "Failed to load existing reference data at {}: {}",
                            output_path.display(),
                            e
                        );
                    }
                };
                if existing.version == db.version {
                    println!("Already up-to-date (version {})", db.version);
                    return Ok(());
                }
                // Backup existing
                let backup_path = output_path.with_extension("json.bak");
                fs::copy(output_path, &backup_path).with_context(|| {
                    format!(
                        "Failed to backup existing reference data to {}",
                        backup_path.display()
                    )
                })?;
                println!("Backed up existing file to {}", backup_path.display());
            }

            // Write new file
            let json =
                serde_json::to_string_pretty(&db).context("Failed to serialize reference data")?;
            fs::write(output_path, json).context("Failed to write reference data to file")?;
            println!(
                "Updated reference data to version {} from {}. Cases: {}.",
                db.version,
                db.source.as_deref().unwrap_or("unknown"),
                db.cases.len()
            );
        }
        None => {
            // Validate local file
            let default_path = Path::new("docs/ashrae_140_references.json");
            if !default_path.exists() {
                anyhow::bail!(
                    "Reference data file not found at {}",
                    default_path.display()
                );
            }
            match MultiReferenceDB::from_file(default_path) {
                Ok(db) => {
                    println!("Reference data is valid.");
                    println!("Version: {}", db.version);
                    if let Some(source) = &db.source {
                        println!("Source: {}", source);
                    }
                    println!("Number of cases: {}", db.cases.len());
                }
                Err(e) => {
                    anyhow::bail!("Failed to parse reference data: {}", e);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;
    use serde_json::json;
    use std::path::PathBuf;
    use tempfile::tempdir;

    /// Helper to change directory temporarily and restore on drop
    struct DirGuard(PathBuf);
    impl DirGuard {
        fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
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

    #[test]
    fn test_update_references_success() -> anyhow::Result<()> {
        // Prepare a mock JSON response with minimal valid data
        let mock_db = json!({
            "version": "2025-01",
            "source": "Test",
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
        })
        .to_string();

        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        // Run in a temporary directory to avoid affecting the real repository
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
        let output_path = Path::new("docs/ashrae_140_references.json");
        assert!(output_path.exists(), "Reference file not created");
        let content = fs::read_to_string(output_path)?;
        let parsed: MultiReferenceDB = serde_json::from_str(&content)?;
        assert_eq!(parsed.version, "2025-01");
        assert_eq!(parsed.cases.len(), 1);

        // Restore cwd automatically via guard
        Ok(())
    }

    #[test]
    fn test_update_references_invalid_json() {
        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("invalid json")
            .create();

        let result = update_references(Some(&url));
        assert!(result.is_err(), "Expected error for invalid JSON");
    }

    #[test]
    fn test_update_references_schema_validation_fails() {
        // Missing required 'version'
        let mock_db = json!({
            "source": "Test",
            "cases": {}
        })
        .to_string();

        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let result = update_references(Some(&url));
        assert!(result.is_err(), "Expected error for schema validation");
    }

    #[test]
    fn test_update_references_http_error() {
        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(404)
            .with_body("Not Found")
            .create();

        let result = update_references(Some(&url));
        assert!(result.is_err(), "Expected error for HTTP 404");
    }

    #[test]
    fn test_update_references_upgrade() -> anyhow::Result<()> {
        // Scenario: local file exists with old version, fetch new version
        let mock_db = json!({
            "version": "2025-01",
            "source": "Test",
            "cases": {
                "600": {
                    "annual_heating": { "EnergyPlus": { "min": 5.0, "max": 7.0 } },
                    "annual_cooling": { "EnergyPlus": { "min": 8.0, "max": 10.0 } },
                    "peak_heating": { "EnergyPlus": { "min": 3.0, "max": 4.0 } },
                    "peak_cooling": { "EnergyPlus": { "min": 5.0, "max": 6.0 } }
                }
            }
        })
        .to_string();

        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let temp = tempdir()?;
        let original_cwd = std::env::current_dir()?;
        let _guard = DirGuard::new(temp.path())?;

        fs::create_dir("docs")?;
        let output_path = Path::new("docs/ashrae_140_references.json");
        // Write an old version file
        let old_db = json!({
            "version": "2024-01",
            "source": "Old",
            "cases": {}
        });
        fs::write(output_path, serde_json::to_string_pretty(&old_db)?)?;

        let result = update_references(Some(&url));
        assert!(
            result.is_ok(),
            "update_references failed: {:?}",
            result.err()
        );

        // Verify file updated
        assert!(output_path.exists());
        let content = fs::read_to_string(output_path)?;
        let parsed: MultiReferenceDB = serde_json::from_str(&content)?;
        assert_eq!(parsed.version, "2025-01");

        // Verify backup exists
        let backup_path = Path::new("docs/ashrae_140_references.json.bak");
        assert!(backup_path.exists(), "Backup file not created");
        let backup_content = fs::read_to_string(backup_path)?;
        let backup: MultiReferenceDB = serde_json::from_str(&backup_content)?;
        assert_eq!(backup.version, "2024-01");

        Ok(())
    }

    #[test]
    fn test_update_references_already_up_to_date() -> anyhow::Result<()> {
        let mock_db = json!({
            "version": "2025-01",
            "source": "Test",
            "cases": {
                "600": {
                    "annual_heating": { "EnergyPlus": { "min": 5.0, "max": 7.0 } },
                    "annual_cooling": { "EnergyPlus": { "min": 8.0, "max": 10.0 } },
                    "peak_heating": { "EnergyPlus": { "min": 3.0, "max": 4.0 } },
                    "peak_cooling": { "EnergyPlus": { "min": 5.0, "max": 6.0 } }
                }
            }
        })
        .to_string();

        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let temp = tempdir()?;
        let _guard = DirGuard::new(temp.path())?;

        fs::create_dir("docs")?;
        let output_path = Path::new("docs/ashrae_140_references.json");

        // Write the same version file
        let existing_db = json!({
            "version": "2025-01",
            "source": "Existing",
            "cases": {}
        });
        fs::write(output_path, serde_json::to_string_pretty(&existing_db)?)?;

        let result = update_references(Some(&url));
        assert!(result.is_ok());

        // File is NOT updated when versions match (early return at line 66-67)
        let content = fs::read_to_string(output_path)?;
        let parsed: MultiReferenceDB = serde_json::from_str(&content)?;
        assert_eq!(parsed.version, "2025-01");
        // Source remains from existing file (no update happened)
        assert_eq!(parsed.source, Some("Existing".to_string()));
        // No backup should exist (no update happened)
        let backup_path = Path::new("docs/ashrae_140_references.json.bak");
        assert!(!backup_path.exists());

        Ok(())
    }

    #[test]
    fn test_update_references_empty_cases() {
        let mock_db = json!({
            "version": "2025-01",
            "source": "Test",
            "cases": {}
        })
        .to_string();

        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let result = update_references(Some(&url));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("no cases"));
    }

    #[test]
    fn test_update_references_missing_energyplus() {
        let mock_db = json!({
            "version": "2025-01",
            "source": "Test",
            "cases": {
                "600": {
                    "annual_heating": {
                        "ESP-r": { "min": 5.0, "max": 7.0 }
                    },
                    "annual_cooling": { "EnergyPlus": { "min": 8.0, "max": 10.0 } },
                    "peak_heating": { "EnergyPlus": { "min": 3.0, "max": 4.0 } },
                    "peak_cooling": { "EnergyPlus": { "min": 5.0, "max": 6.0 } }
                }
            }
        })
        .to_string();

        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let result = update_references(Some(&url));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("EnergyPlus"));
    }

    #[test]
    fn test_update_references_empty_annual_heating() {
        let mock_db = json!({
            "version": "2025-01",
            "source": "Test",
            "cases": {
                "600": {
                    "annual_heating": {},
                    "annual_cooling": { "EnergyPlus": { "min": 8.0, "max": 10.0 } },
                    "peak_heating": { "EnergyPlus": { "min": 3.0, "max": 4.0 } },
                    "peak_cooling": { "EnergyPlus": { "min": 5.0, "max": 6.0 } }
                }
            }
        })
        .to_string();

        let mut server = mockito::Server::new();
        let url = server.url();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let result = update_references(Some(&url));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("annual_heating"));
    }

    #[test]
    fn test_update_references_local_validation_success() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let _guard = DirGuard::new(temp.path())?;

        fs::create_dir("docs")?;
        let output_path = Path::new("docs/ashrae_140_references.json");

        let valid_db = json!({
            "version": "2025-01",
            "source": "Test",
            "cases": {
                "600": {
                    "annual_heating": { "EnergyPlus": { "min": 5.0, "max": 7.0 } },
                    "annual_cooling": { "EnergyPlus": { "min": 8.0, "max": 10.0 } },
                    "peak_heating": { "EnergyPlus": { "min": 3.0, "max": 4.0 } },
                    "peak_cooling": { "EnergyPlus": { "min": 5.0, "max": 6.0 } }
                }
            }
        });
        fs::write(output_path, serde_json::to_string_pretty(&valid_db)?)?;

        let result = update_references(None);
        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn test_update_references_local_file_not_found() {
        let temp = tempdir().unwrap();
        let _guard = DirGuard::new(temp.path()).unwrap();

        let result = update_references(None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_update_references_local_invalid_json() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let _guard = DirGuard::new(temp.path())?;

        fs::create_dir("docs")?;
        let output_path = Path::new("docs/ashrae_140_references.json");
        fs::write(output_path, "not valid json")?;

        let result = update_references(None);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_dir_guard_restores_directory() {
        let original = std::env::current_dir().unwrap();
        let temp = tempdir().unwrap();
        {
            let _guard = DirGuard::new(temp.path()).unwrap();
            assert_eq!(std::env::current_dir().unwrap(), temp.path());
        }
        assert_eq!(std::env::current_dir().unwrap(), original);
    }
}

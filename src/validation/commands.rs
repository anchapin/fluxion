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
    let output_path = Path::new("docs/ashrae_140_references.json");
    update_references_ext(url, output_path)
}

/// Update or validate with explicit path (useful for testing)
pub fn update_references_ext(url: Option<&str>, output_path: &Path) -> Result<()> {
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

            // Ensure directory exists
            if let Some(parent) = output_path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).context("Failed to create docs directory")?;
                }
            }

            // If file exists, check if version matches
            if output_path.exists() {
                // Load existing
                let existing = match MultiReferenceDB::from_file(output_path) {
                    Ok(db) => db,
                    Err(e) => {
                        // If it exists but invalid, we overwrite it anyway
                        eprintln!("Warning: Existing reference data is invalid: {}", e);
                        MultiReferenceDB {
                            version: "".to_string(),
                            source: None,
                            cases: std::collections::HashMap::new(),
                        }
                    }
                };
                if existing.version == db.version {
                    println!("Already up-to-date (version {})", db.version);
                    return Ok(());
                }
                // Backup existing
                let backup_path = output_path.with_extension("json.bak");
                fs::copy(output_path, &backup_path).ok();
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
            if !output_path.exists() {
                anyhow::bail!("Reference data file not found at {}", output_path.display());
            }
            match MultiReferenceDB::from_file(output_path) {
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
    use tempfile::tempdir;

    fn create_mock_db(version: &str, source: &str) -> serde_json::Value {
        json!({
            "version": version,
            "source": source,
            "cases": {
                "600": {
                    "annual_heating": { "EnergyPlus": { "min": 5.0, "max": 7.0 } },
                    "annual_cooling": { "EnergyPlus": { "min": 8.0, "max": 10.0 } },
                    "peak_heating": { "EnergyPlus": { "min": 3.0, "max": 4.0 } },
                    "peak_cooling": { "EnergyPlus": { "min": 5.0, "max": 6.0 } }
                }
            }
        })
    }

    #[test]
    fn test_update_references_success() -> anyhow::Result<()> {
        let mut server = mockito::Server::new();
        let version = "2025-01-success";
        let mock_db = create_mock_db(version, "Test").to_string();

        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let temp = tempdir()?;
        let output_path = temp.path().join("refs.json");

        let result = update_references_ext(Some(&server.url()), &output_path);
        assert!(result.is_ok(), "Error: {:?}", result.err());
        assert!(output_path.exists());
        Ok(())
    }

    #[test]
    fn test_update_references_upgrade() -> anyhow::Result<()> {
        let mut server = mockito::Server::new();
        let version = "2025-01-upgrade";
        let mock_db = create_mock_db(version, "New").to_string();

        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let temp = tempdir()?;
        let output_path = temp.path().join("refs.json");

        let old_db = create_mock_db("2024-01-old", "Old");
        fs::write(&output_path, serde_json::to_string_pretty(&old_db)?)?;

        let result = update_references_ext(Some(&server.url()), &output_path);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let content = fs::read_to_string(&output_path)?;
        let parsed: MultiReferenceDB = serde_json::from_str(&content)?;
        assert_eq!(parsed.version, version);

        let backup_path = output_path.with_extension("json.bak");
        assert!(backup_path.exists());
        Ok(())
    }

    #[test]
    fn test_update_references_already_up_to_date() -> anyhow::Result<()> {
        let mut server = mockito::Server::new();
        let version = "2025-01-same";
        let mock_db = create_mock_db(version, "Test").to_string();

        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let temp = tempdir()?;
        let output_path = temp.path().join("refs.json");

        let existing_db = create_mock_db(version, "Existing");
        fs::write(&output_path, serde_json::to_string_pretty(&existing_db)?)?;

        let result = update_references_ext(Some(&server.url()), &output_path);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let content = fs::read_to_string(&output_path)?;
        let parsed: MultiReferenceDB = serde_json::from_str(&content)?;
        assert_eq!(parsed.source, Some("Existing".to_string()));
        Ok(())
    }

    #[test]
    fn test_update_references_local_validation_success() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let output_path = temp.path().join("refs.json");
        let valid_db = create_mock_db("2025-01-local", "Test");
        fs::write(&output_path, serde_json::to_string_pretty(&valid_db)?)?;

        let result = update_references_ext(None, &output_path);
        assert!(result.is_ok(), "Error: {:?}", result.err());
        Ok(())
    }

    #[test]
    fn test_update_references_local_invalid_json() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let output_path = temp.path().join("refs.json");
        fs::write(&output_path, "{ invalid json")?;

        let result = update_references_ext(None, &output_path);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_update_references_local_file_not_found() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let output_path = temp.path().join("nonexistent.json");

        let result = update_references_ext(None, &output_path);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_update_references_invalid_json() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("invalid json")
            .create();

        let temp = tempdir().unwrap();
        let output_path = temp.path().join("refs.json");

        let result = update_references_ext(Some(&server.url()), &output_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_references_empty_cases() {
        let mut server = mockito::Server::new();
        let mock_db = json!({
            "version": "2025-01",
            "source": "Test",
            "cases": {}
        })
        .to_string();

        let _mock = server
            .mock("GET", Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(&mock_db)
            .create();

        let temp = tempdir().unwrap();
        let output_path = temp.path().join("refs.json");

        let result = update_references_ext(Some(&server.url()), &output_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no cases found"));
    }
}

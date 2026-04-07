//! TMY3 weather data download and caching.
//!
//! This module provides infrastructure for downloading TMY3 (Typical Meteorological Year)
//! weather data from NREL repository and caching it locally. It supports:
//!
//! - On-demand TMY3 file downloads
//! - Local caching in ~/.cache/fluxion/tmy3/
//! - SHA-256 checksum validation
//! - Weather location metadata from JSON
//!
//! # Example
//!
//! ```no_run
//! use fluxion::weather::tmy3::{Tmy3Cache, load_weather_locations};
//!
//! // Create cache
//! let cache = Tmy3Cache::new().unwrap();
//!
//! // Load weather locations
//! let locations = load_weather_locations("data/weather_locations.json").unwrap();
//!
//! // Download Denver TMY3 data
//! let denver = &locations["Denver"];
//! let filepath = cache.get_or_download(&denver.tmy3_url, "Denver").unwrap();
//! ```
//!
//! # Cache Location
//!
//! By default, TMY3 files are cached in:
//! - Linux/macOS: ~/.cache/fluxion/tmy3/
//! - Windows: %LOCALAPPDATA%\fluxion\tmy3\

use directories;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Weather location metadata.
///
/// Contains location information and URLs for weather data downloads.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WeatherLocation {
    /// Location name (e.g., "Denver", "Boston")
    pub name: String,

    /// Latitude (degrees)
    pub latitude: f64,

    /// Longitude (degrees)
    pub longitude: f64,

    /// Elevation (meters)
    pub elevation: f64,

    /// TMY3 download URL
    pub tmy3_url: String,

    /// EPW download URL
    pub epw_url: String,

    /// Climate zone
    #[serde(default)]
    pub climate_zone: Option<String>,
}

/// Load weather locations from JSON file.
///
/// # Arguments
///
/// * `path` - Path to weather_locations.json
///
/// # Returns
///
/// HashMap of location name to WeatherLocation or error
pub fn load_weather_locations(path: &str) -> Result<HashMap<String, WeatherLocation>, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read weather locations: {}", e))?;

    let locations: Vec<WeatherLocation> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse weather locations: {}", e))?;

    let mut map = HashMap::new();
    for location in locations {
        map.insert(location.name.clone(), location);
    }

    Ok(map)
}

/// Cache for TMY3 weather data downloads.
///
/// Manages downloading TMY3 files from NREL repository and caching them
/// locally in ~/.cache/fluxion/tmy3/ to avoid repeated network calls.
pub struct Tmy3Cache {
    cache_dir: PathBuf,
    client: Client,
}

impl Tmy3Cache {
    /// Create new TMY3 cache with default cache directory.
    ///
    /// Creates cache directory at ~/.cache/fluxion/tmy3/ if it doesn't exist.
    ///
    /// # Returns
    ///
    /// New Tmy3Cache instance or error if cache directory cannot be created
    pub fn new() -> Result<Self, String> {
        let proj_dirs = directories::ProjectDirs::from("com", "fluxion", "fluxion")
            .ok_or("Failed to determine cache directory")?;
        let cache_dir = proj_dirs.cache_dir().join("tmy3");

        fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;

        Ok(Tmy3Cache {
            cache_dir,
            client: Client::new(),
        })
    }

    /// Create new TMY3 cache with custom cache directory.
    ///
    /// Useful for testing or custom cache locations.
    ///
    /// # Arguments
    ///
    /// * `cache_dir` - Path to cache directory
    ///
    /// # Returns
    ///
    /// New Tmy3Cache instance with custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, String> {
        fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;

        Ok(Tmy3Cache {
            cache_dir,
            client: Client::new(),
        })
    }

    /// Get TMY3 file from cache or download from URL.
    ///
    /// Checks cache first. If file not in cache, downloads from URL and
    /// stores in cache with SHA-256 checksum validation.
    ///
    /// # Arguments
    ///
    /// * `url` - URL to download TMY3 file from
    /// * `location` - Location name for cache filename (e.g., "Denver", "Boston")
    ///
    /// # Returns
    ///
    /// Path to cached TMY3 file or error if download fails
    pub fn get_or_download(&self, url: &str, location: &str) -> Result<PathBuf, String> {
        let filename = format!("{}.tmy3", location.replace(' ', "_"));
        let filepath = self.cache_dir.join(&filename);

        // Check cache
        if filepath.exists() {
            // Verify checksum if checksum file exists
            let checksum_path = filepath.with_extension("sha256");
            if checksum_path.exists() {
                // Verify checksum
                let content = fs::read(&filepath)
                    .map_err(|e| format!("Failed to read cached file: {}", e))?;
                let checksum = format!("{:x}", Sha256::digest(&content));

                let expected_checksum = fs::read_to_string(&checksum_path)
                    .map_err(|e| format!("Failed to read checksum file: {}", e))?;

                if checksum != expected_checksum.trim() {
                    return Err(format!(
                        "Checksum mismatch: {} (expected: {})",
                        checksum, expected_checksum
                    ));
                }
            }
            return Ok(filepath);
        }

        // Download file
        let response = self
            .client
            .get(url)
            .send()
            .map_err(|e| format!("Failed to download TMY3: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let content = response
            .bytes()
            .map_err(|e| format!("Failed to read response: {}", e))?;

        // Calculate checksum
        let checksum = format!("{:x}", Sha256::digest(&content));

        // Write to cache
        let mut file = fs::File::create(&filepath)
            .map_err(|e| format!("Failed to create cache file: {}", e))?;
        file.write_all(&content)
            .map_err(|e| format!("Failed to write cache file: {}", e))?;

        // Write checksum file
        let checksum_path = filepath.with_extension("sha256");
        let mut checksum_file = fs::File::create(&checksum_path)
            .map_err(|e| format!("Failed to create checksum file: {}", e))?;
        checksum_file
            .write_all(checksum.as_bytes())
            .map_err(|e| format!("Failed to write checksum: {}", e))?;

        Ok(filepath)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weather_location_deserialize_all_fields() {
        let json = r#"{
            "name": "Denver",
            "latitude": 39.74,
            "longitude": -104.99,
            "elevation": 1634.0,
            "tmy3_url": "https://example.com/denver.tmy3",
            "epw_url": "https://example.com/denver.epw",
            "climate_zone": "5B"
        }"#;

        let location: WeatherLocation = serde_json::from_str(json).unwrap();
        assert_eq!(location.name, "Denver");
        assert_eq!(location.latitude, 39.74);
        assert_eq!(location.longitude, -104.99);
        assert_eq!(location.elevation, 1634.0);
        assert_eq!(location.tmy3_url, "https://example.com/denver.tmy3");
        assert_eq!(location.epw_url, "https://example.com/denver.epw");
        assert_eq!(location.climate_zone, Some("5B".to_string()));
    }

    #[test]
    fn test_weather_location_deserialize_optional_climate_zone_missing() {
        let json = r#"{
            "name": "Test",
            "latitude": 40.0,
            "longitude": -100.0,
            "elevation": 500.0,
            "tmy3_url": "https://example.com/test.tmy3",
            "epw_url": "https://example.com/test.epw"
        }"#;

        let location: WeatherLocation = serde_json::from_str(json).unwrap();
        assert_eq!(location.climate_zone, None);
    }

    #[test]
    fn test_weather_location_serialize_roundtrip() {
        let location = WeatherLocation {
            name: "Boston".to_string(),
            latitude: 42.36,
            longitude: -71.06,
            elevation: 6.0,
            tmy3_url: "https://example.com/boston.tmy3".to_string(),
            epw_url: "https://example.com/boston.epw".to_string(),
            climate_zone: Some("6A".to_string()),
        };

        let json = serde_json::to_string(&location).unwrap();
        let deserialized: WeatherLocation = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, location.name);
        assert_eq!(deserialized.latitude, location.latitude);
        assert_eq!(deserialized.longitude, location.longitude);
        assert_eq!(deserialized.elevation, location.elevation);
        assert_eq!(deserialized.tmy3_url, location.tmy3_url);
        assert_eq!(deserialized.epw_url, location.epw_url);
        assert_eq!(deserialized.climate_zone, location.climate_zone);
    }

    #[test]
    fn test_load_weather_locations_success() {
        let temp_dir = std::env::temp_dir();
        let json_path = temp_dir.join("test_weather_locations.json");

        let locations = vec![
            WeatherLocation {
                name: "Denver".to_string(),
                latitude: 39.74,
                longitude: -104.99,
                elevation: 1634.0,
                tmy3_url: "https://example.com/denver.tmy3".to_string(),
                epw_url: "https://example.com/denver.epw".to_string(),
                climate_zone: Some("5B".to_string()),
            },
            WeatherLocation {
                name: "Boston".to_string(),
                latitude: 42.36,
                longitude: -71.06,
                elevation: 6.0,
                tmy3_url: "https://example.com/boston.tmy3".to_string(),
                epw_url: "https://example.com/boston.epw".to_string(),
                climate_zone: Some("6A".to_string()),
            },
        ];

        let json = serde_json::to_string(&locations).unwrap();
        std::fs::write(&json_path, json).unwrap();

        let result = load_weather_locations(json_path.to_str().unwrap());
        assert!(result.is_ok());

        let map = result.unwrap();
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("Denver"));
        assert!(map.contains_key("Boston"));
        assert_eq!(map["Denver"].latitude, 39.74);
        assert_eq!(map["Boston"].climate_zone, Some("6A".to_string()));

        std::fs::remove_file(&json_path).ok();
    }

    #[test]
    fn test_load_weather_locations_file_not_found() {
        let result = load_weather_locations("/nonexistent/path/weather.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read"));
    }

    #[test]
    fn test_load_weather_locations_invalid_json() {
        let temp_dir = std::env::temp_dir();
        let json_path = temp_dir.join("invalid_weather.json");
        std::fs::write(&json_path, "not valid json").unwrap();

        let result = load_weather_locations(json_path.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse"));

        std::fs::remove_file(&json_path).ok();
    }

    #[test]
    fn test_load_weather_locations_empty_array() {
        let temp_dir = std::env::temp_dir();
        let json_path = temp_dir.join("empty_weather.json");
        std::fs::write(&json_path, "[]").unwrap();

        let result = load_weather_locations(json_path.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        std::fs::remove_file(&json_path).ok();
    }

    #[test]
    fn test_tmy3_cache_with_custom_dir() {
        let temp_dir = std::env::temp_dir().join("test_tmy3_cache");
        let result = Tmy3Cache::with_cache_dir(temp_dir.clone());
        assert!(result.is_ok());

        let cache = result.unwrap();
        assert!(temp_dir.exists());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tmy3_cache_get_or_download_uses_cache() {
        let temp_dir = std::env::temp_dir().join("test_tmy3_cache_2");
        let cache = Tmy3Cache::with_cache_dir(temp_dir.clone()).unwrap();

        let filename = "Test_Location.tmy3";
        let filepath = temp_dir.join(filename);
        let checksum_path = temp_dir.join("Test_Location.sha256");

        std::fs::write(&filepath, "test content").unwrap();
        let checksum = format!("{:x}", Sha256::digest(b"test content"));
        std::fs::write(&checksum_path, checksum).unwrap();

        let result = cache.get_or_download("https://example.com/test.tmy3", "Test Location");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), filepath);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tmy3_cache_get_or_download_checksum_mismatch() {
        let temp_dir = std::env::temp_dir().join("test_tmy3_cache_3");
        let cache = Tmy3Cache::with_cache_dir(temp_dir.clone()).unwrap();

        let filename = "Test_Location.tmy3";
        let filepath = temp_dir.join(filename);
        let checksum_path = temp_dir.join("Test_Location.sha256");

        std::fs::write(&filepath, "test content").unwrap();
        std::fs::write(&checksum_path, "wrong_checksum").unwrap();

        let result = cache.get_or_download("https://example.com/test.tmy3", "Test Location");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Checksum mismatch"));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tmy3_cache_filename_sanitization() {
        let temp_dir = std::env::temp_dir().join("test_tmy3_cache_4");
        let cache = Tmy3Cache::with_cache_dir(temp_dir.clone()).unwrap();

        let filename = "New_York_City.tmy3";
        let filepath = temp_dir.join(filename);
        let checksum_path = temp_dir.join("New_York_City.sha256");

        std::fs::write(&filepath, "nyc content").unwrap();
        let checksum = format!("{:x}", Sha256::digest(b"nyc content"));
        std::fs::write(&checksum_path, checksum).unwrap();

        let result = cache.get_or_download("https://example.com/nyc.tmy3", "New York City");
        assert!(result.is_ok());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tmy3_cache_filename_with_special_chars() {
        let temp_dir = std::env::temp_dir().join("test_tmy3_cache_5");
        let cache = Tmy3Cache::with_cache_dir(temp_dir.clone()).unwrap();

        let filename = "Los_Angeles.tmy3";
        let filepath = temp_dir.join(filename);
        let checksum_path = temp_dir.join("Los_Angeles.sha256");

        std::fs::write(&filepath, "la content").unwrap();
        let checksum = format!("{:x}", Sha256::digest(b"la content"));
        std::fs::write(&checksum_path, checksum).unwrap();

        let result = cache.get_or_download("https://example.com/la.tmy3", "Los Angeles");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), filepath);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tmy3_cache_new_creates_dir() {
        let temp_dir = std::env::temp_dir().join("test_tmy3_cache_new");
        std::fs::remove_dir_all(&temp_dir).ok();
        let result = Tmy3Cache::with_cache_dir(temp_dir.clone());
        assert!(result.is_ok());
        assert!(temp_dir.exists());
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_weather_location_debug() {
        let location = WeatherLocation {
            name: "Test".to_string(),
            latitude: 40.0,
            longitude: -100.0,
            elevation: 500.0,
            tmy3_url: "https://example.com/test.tmy3".to_string(),
            epw_url: "https://example.com/test.epw".to_string(),
            climate_zone: Some("4A".to_string()),
        };
        let debug_str = format!("{:?}", location);
        assert!(debug_str.contains("Test"));
        assert!(debug_str.contains("40.0"));
    }

    #[test]
    fn test_weather_location_clone() {
        let location = WeatherLocation {
            name: "Clone".to_string(),
            latitude: 35.0,
            longitude: -95.0,
            elevation: 300.0,
            tmy3_url: "https://example.com/clone.tmy3".to_string(),
            epw_url: "https://example.com/clone.epw".to_string(),
            climate_zone: None,
        };
        let cloned = location.clone();
        assert_eq!(cloned.name, location.name);
        assert_eq!(cloned.latitude, location.latitude);
        assert_eq!(cloned.climate_zone, None);
    }

    #[test]
    fn test_load_weather_locations_duplicate_names() {
        let temp_dir = std::env::temp_dir();
        let json_path = temp_dir.join("test_duplicate_locations.json");

        let locations = vec![
            WeatherLocation {
                name: "Denver".to_string(),
                latitude: 39.74,
                longitude: -104.99,
                elevation: 1634.0,
                tmy3_url: "https://example.com/denver1.tmy3".to_string(),
                epw_url: "https://example.com/denver1.epw".to_string(),
                climate_zone: Some("5B".to_string()),
            },
            WeatherLocation {
                name: "Denver".to_string(),
                latitude: 40.0,
                longitude: -105.0,
                elevation: 1700.0,
                tmy3_url: "https://example.com/denver2.tmy3".to_string(),
                epw_url: "https://example.com/denver2.epw".to_string(),
                climate_zone: Some("5B".to_string()),
            },
        ];

        let json = serde_json::to_string(&locations).unwrap();
        std::fs::write(&json_path, json).unwrap();

        let result = load_weather_locations(json_path.to_str().unwrap());
        assert!(result.is_ok());
        let map = result.unwrap();
        // Second entry should overwrite first
        assert_eq!(map.len(), 1);
        assert_eq!(map["Denver"].latitude, 40.0);

        std::fs::remove_file(&json_path).ok();
    }

    #[test]
    fn test_tmy3_cache_get_or_download_success() {
        let mut server = mockito::Server::new();
        let url = server.url();
        let mock_content = b"downloaded content";
        let _m = server
            .mock("GET", "/test.tmy3")
            .with_status(200)
            .with_body(mock_content)
            .create();

        let temp_dir = std::env::temp_dir().join("test_tmy3_download_success");
        std::fs::remove_dir_all(&temp_dir).ok();
        let cache = Tmy3Cache::with_cache_dir(temp_dir.clone()).unwrap();

        let full_url = format!("{}/test.tmy3", url);
        let result = cache.get_or_download(&full_url, "Downloaded Location");

        assert!(result.is_ok());
        let filepath = result.unwrap();
        assert!(filepath.exists());
        assert_eq!(std::fs::read(&filepath).unwrap(), mock_content);

        // Check checksum file
        let checksum_path = filepath.with_extension("sha256");
        assert!(checksum_path.exists());
        let expected_checksum = format!("{:x}", Sha256::digest(mock_content));
        assert_eq!(
            std::fs::read_to_string(checksum_path).unwrap(),
            expected_checksum
        );

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tmy3_cache_get_or_download_http_error() {
        let mut server = mockito::Server::new();
        let url = server.url();
        let _m = server.mock("GET", "/fail.tmy3").with_status(404).create();

        let temp_dir = std::env::temp_dir().join("test_tmy3_download_fail");
        std::fs::remove_dir_all(&temp_dir).ok();
        let cache = Tmy3Cache::with_cache_dir(temp_dir.clone()).unwrap();

        let full_url = format!("{}/fail.tmy3", url);
        let result = cache.get_or_download(&full_url, "Fail Location");

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("HTTP error: 404"));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tmy3_cache_new_default() {
        // Just test that it can be created without error
        // Note: this might fail in some CI environments if HOME is not set,
        // but directories crate usually handles it.
        let cache = Tmy3Cache::new();
        assert!(cache.is_ok());
    }
}

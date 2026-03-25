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
use std::path::PathBuf;

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

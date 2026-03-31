#[cfg(test)]
mod tests {
    use fluxion::weather::tmy3::{load_weather_locations, Tmy3Cache, WeatherLocation};
    use std::fs;

    #[test]
    fn test_load_weather_locations() {
        let locations = load_weather_locations("data/weather_locations.json")
            .expect("Failed to load weather locations");

        assert!(
            locations.contains_key("Denver"),
            "Should have Denver location"
        );
        assert!(
            locations.contains_key("Boston"),
            "Should have Boston location"
        );

        let denver = &locations["Denver"];
        assert_eq!(denver.name, "Denver");
        assert!(denver.latitude > 39.0 && denver.latitude < 40.0);
        assert!(denver.longitude < -104.0 && denver.longitude > -106.0);
    }

    #[test]
    fn test_load_weather_locations_denver_elevation() {
        let locations = load_weather_locations("data/weather_locations.json")
            .expect("Failed to load weather locations");

        let denver = &locations["Denver"];
        assert!(
            denver.elevation > 1500.0,
            "Denver elevation should be > 1500m"
        );
        assert!(
            denver.elevation < 2000.0,
            "Denver elevation should be < 2000m"
        );
    }

    #[test]
    fn test_load_weather_locations_urls() {
        let locations = load_weather_locations("data/weather_locations.json")
            .expect("Failed to load weather locations");

        let denver = &locations["Denver"];
        assert!(
            denver.tmy3_url.starts_with("http"),
            "TMY3 URL should be valid"
        );
        assert!(
            denver.epw_url.starts_with("http"),
            "EPW URL should be valid"
        );
    }

    #[test]
    fn test_load_weather_locations_climate_zone_optional() {
        let locations = load_weather_locations("data/weather_locations.json")
            .expect("Failed to load weather locations");

        // Climate zone is optional, check if it's present for at least one location
        let has_any_climate_zone = locations.values().any(|loc| loc.climate_zone.is_some());
        assert!(
            has_any_climate_zone,
            "At least one location should have climate zone"
        );
    }

    #[test]
    fn test_load_weather_locations_missing_file() {
        let result = load_weather_locations("nonexistent_file.json");
        assert!(result.is_err(), "Should return error for missing file");
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("Failed to read"),
            "Error should mention read failure"
        );
    }

    #[test]
    fn test_load_weather_locations_invalid_json() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let invalid_json_path = temp_dir.path().join("invalid.json");
        fs::write(&invalid_json_path, "not valid json").expect("Failed to write invalid JSON");

        let result = load_weather_locations(invalid_json_path.to_str().unwrap());
        assert!(result.is_err(), "Should return error for invalid JSON");
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("Failed to parse"),
            "Error should mention parse failure"
        );
    }

    #[test]
    fn test_tmy3_cache_custom_directory() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let cache_dir = temp_dir.path().join("tmy3");

        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        assert!(cache_dir.exists(), "Cache directory should exist");
        assert!(cache_dir.is_dir(), "Cache should be a directory");
    }

    #[test]
    fn test_tmy3_cache_nested_directory() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let cache_dir = temp_dir.path().join("nested").join("cache").join("tmy3");

        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        assert!(
            cache_dir.exists(),
            "Nested cache directory should be created"
        );
    }

    #[test]
    fn test_tmy3_cache_filename_format() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let cache_dir = temp_dir.path().join("tmy3");
        let _cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        // The cache should format filenames as "Location_Name.tmy3"
        // We can't test get_or_download without network, but we can verify
        // the cache directory structure
        assert!(cache_dir.exists());
    }

    #[test]
    fn test_weather_location_serialization() {
        let location = WeatherLocation {
            name: "Test City".to_string(),
            latitude: 40.0,
            longitude: -105.0,
            elevation: 1600.0,
            tmy3_url: "https://example.com/test.tmy3".to_string(),
            epw_url: "https://example.com/test.epw".to_string(),
            climate_zone: Some("5B".to_string()),
        };

        let json = serde_json::to_string(&location).expect("Failed to serialize");
        assert!(json.contains("Test City"));
        assert!(json.contains("5B"));

        let deserialized: WeatherLocation =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.name, "Test City");
        assert_eq!(deserialized.latitude, 40.0);
    }

    #[test]
    fn test_weather_location_without_climate_zone() {
        let json = r#"{
            "name": "Test City",
            "latitude": 40.0,
            "longitude": -105.0,
            "elevation": 1600.0,
            "tmy3_url": "https://example.com/test.tmy3",
            "epw_url": "https://example.com/test.epw"
        }"#;

        let location: WeatherLocation =
            serde_json::from_str(json).expect("Failed to deserialize without climate_zone");
        assert_eq!(location.name, "Test City");
        assert!(location.climate_zone.is_none());
    }

    #[test]
    fn test_weather_location_debug() {
        let location = WeatherLocation {
            name: "Debug Test".to_string(),
            latitude: 35.0,
            longitude: -100.0,
            elevation: 500.0,
            tmy3_url: "https://example.com/test.tmy3".to_string(),
            epw_url: "https://example.com/test.epw".to_string(),
            climate_zone: None,
        };

        let debug_str = format!("{:?}", location);
        assert!(debug_str.contains("Debug Test"));
        assert!(debug_str.contains("WeatherLocation"));
    }

    #[test]
    fn test_weather_location_clone() {
        let original = WeatherLocation {
            name: "Clone Test".to_string(),
            latitude: 42.0,
            longitude: -71.0,
            elevation: 100.0,
            tmy3_url: "https://example.com/test.tmy3".to_string(),
            epw_url: "https://example.com/test.epw".to_string(),
            climate_zone: Some("4A".to_string()),
        };

        let cloned = original.clone();
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.latitude, original.latitude);
    }
}

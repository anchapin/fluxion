#[cfg(test)]
mod tests {
    use fluxion::weather::tmy3::{load_weather_locations, Tmy3Cache};


    #[test]
    fn test_load_weather_locations() {
        // Test loading weather locations from JSON
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
    fn test_tmy3_cache_custom_directory() {
        // Test cache creation with custom directory
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let cache_dir = temp_dir.path().join("tmy3");

        let _cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        assert!(cache_dir.exists(), "Cache directory should exist");
        assert!(cache_dir.is_dir(), "Cache should be a directory");

        // Verify parent directory exists
        let parent = cache_dir.parent().unwrap();
        assert!(parent.exists(), "Parent directory should exist");
    }
}

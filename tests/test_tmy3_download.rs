// Issue #3467 — the TMY3 weather-download / on-disk-cache path lives behind
// the `tmy3-download` cargo feature. Without the feature, the
// `fluxion::weather::tmy3` re-export does not exist (see
// `fluxion-core/src/weather/mod.rs`), so this test binary is a no-op.
// Run with `--features tmy3-download` to actually exercise the TMY3 cache.
//
// Issue #3464 — the download-and-cache path (`Tmy3Cache::get_or_download`) now
// has hermetic coverage in this integration test binary via `mockito` (a local
// HTTP server bound to 127.0.0.1 on an ephemeral port, no real network). The
// previous version of this file at <https://github.com/anchapin/fluxion/issues/3464>
// explicitly conceded "We can't test get_or_download without network"; that
// gap is now closed.
#![cfg(feature = "tmy3-download")]

#[cfg(test)]
mod tests {
    use fluxion::weather::tmy3::{load_weather_locations, Tmy3Cache, WeatherLocation};
    use mockito::Server;
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    use std::fs;

    /// Compute the SHA-256 hex digest of `bytes`. Mirrors the helper used
    /// inside `fluxion_core::weather::tmy3` so the integration test can
    /// pre-populate cache state and verify stored checksums without
    /// depending on the private helper.
    fn sha256_hex(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        let mut s = String::with_capacity(digest.len() * 2);
        for b in digest {
            let _ = write!(s, "{:02x}", b);
        }
        s
    }

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

        let _cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        assert!(cache_dir.exists(), "Cache directory should exist");
        assert!(cache_dir.is_dir(), "Cache should be a directory");
    }

    #[test]
    fn test_tmy3_cache_nested_directory() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let cache_dir = temp_dir.path().join("nested").join("cache").join("tmy3");

        let _cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        assert!(
            cache_dir.exists(),
            "Nested cache directory should be created"
        );
    }

    #[test]
    fn test_tmy3_cache_filename_format() {
        // Issue #3464 — this is now exercised end-to-end by
        // `test_get_or_download_filename_sanitization` (and others) below
        // via the hermetic `mockito` server, so the on-disk cache directory
        // structure assertion is no longer the only thing we can verify here.
        // The minimal directory-creation assertion is kept for parity with
        // the previous coverage.
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let cache_dir = temp_dir.path().join("tmy3");
        let _cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

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

    // =====================================================================
    // Hermetic coverage of the TMY3 download-and-cache path (Issue #3464).
    //
    // These tests spin up a local `mockito` server bound to 127.0.0.1 on an
    // ephemeral port — no real network is required, so they pass in CI even
    // when the runner has no outbound connectivity. The cache directory is
    // always a `tempfile::tempdir()` scratch path that the OS reclaims on
    // test exit.
    // =====================================================================

    /// HTTP 200 success: download is written to cache and the SHA-256
    /// sidecar file is created with the expected digest. The mock must see
    /// the request hit `/denver.tmy3` exactly (no URL rewriting).
    #[test]
    fn test_get_or_download_http_200_writes_file_and_checksum() {
        let mut server = Server::new();
        let body: &[u8] = b"downloaded tmy3 bytes for Denver";
        let expected_checksum = sha256_hex(body);
        let _m = server
            .mock("GET", "/denver.tmy3")
            .with_status(200)
            .with_body(body)
            .expect(1)
            .create();

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        let url = format!("{}/denver.tmy3", server.url());
        let result = cache.get_or_download(&url, "Denver");
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

        let filepath = result.unwrap();
        let expected_path = cache_dir.join("Denver.tmy3");
        assert_eq!(filepath, expected_path);
        assert!(filepath.exists(), "cached .tmy3 file should exist");

        let stored_body = fs::read(&filepath).expect("Failed to read cached body");
        assert_eq!(stored_body, body, "cached body should match mock response");

        let checksum_path = filepath.with_extension("sha256");
        assert!(
            checksum_path.exists(),
            "checksum sidecar should be written on download"
        );
        let stored_checksum =
            fs::read_to_string(&checksum_path).expect("Failed to read checksum sidecar");
        assert_eq!(
            stored_checksum, expected_checksum,
            "stored SHA-256 should match the body"
        );
    }

    /// HTTP 404: download fails fast, no file or checksum is written to
    /// cache, and the error message surfaces the status code so callers can
    /// diagnose the missing endpoint.
    #[test]
    fn test_get_or_download_http_404_returns_error() {
        let mut server = Server::new();
        let _m = server
            .mock("GET", "/missing.tmy3")
            .with_status(404)
            .expect(1)
            .create();

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        let url = format!("{}/missing.tmy3", server.url());
        let result = cache.get_or_download(&url, "Nowhere");
        assert!(result.is_err(), "expected Err for HTTP 404");
        let err = result.unwrap_err();
        assert!(
            err.contains("HTTP error: 404"),
            "error should mention HTTP 404; got: {}",
            err
        );

        assert!(
            !cache_dir.join("Nowhere.tmy3").exists(),
            "no file should be cached on HTTP failure"
        );
        assert!(
            !cache_dir.join("Nowhere.sha256").exists(),
            "no checksum sidecar should be written on HTTP failure"
        );
    }

    /// HTTP 500: same fast-fail contract as 404 — propagates the status and
    /// leaves the cache untouched.
    #[test]
    fn test_get_or_download_http_500_returns_error() {
        let mut server = Server::new();
        let _m = server
            .mock("GET", "/server_error.tmy3")
            .with_status(500)
            .expect(1)
            .create();

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        let url = format!("{}/server_error.tmy3", server.url());
        let result = cache.get_or_download(&url, "Broken");
        assert!(result.is_err(), "expected Err for HTTP 500");
        let err = result.unwrap_err();
        assert!(
            err.contains("HTTP error: 500"),
            "error should mention HTTP 500; got: {}",
            err
        );
        assert!(
            !cache_dir.join("Broken.tmy3").exists(),
            "no file should be cached on HTTP 500"
        );
    }

    /// Cache hit with a valid SHA-256 sidecar: no HTTP request is sent
    /// (mock is registered with `expect(0)` semantics — using a bare server
    /// would 404 on any request, masking a regression in the cache-hit
    /// short-circuit).
    #[test]
    fn test_get_or_download_cache_hit_with_valid_checksum() {
        let body: &[u8] = b"previously downloaded tmy3 body";
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

        let filepath = cache_dir.join("Boston.tmy3");
        let checksum_path = cache_dir.join("Boston.sha256");
        fs::write(&filepath, body).expect("Failed to seed cached file");
        fs::write(&checksum_path, sha256_hex(body)).expect("Failed to seed checksum");

        // Bare server: any request would 404. If the cache-hit short-circuit
        // regresses, the test will fail with HTTP 404 from the server.
        let mut server = Server::new();
        let _unmocked = server.mock("GET", "/boston.tmy3"); // no `.create()` → never matched

        let cache = Tmy3Cache::with_cache_dir(cache_dir).expect("Failed to create cache");
        let url = format!("{}/boston.tmy3", server.url());
        let result = cache.get_or_download(&url, "Boston");

        assert!(
            result.is_ok(),
            "cache hit with valid checksum should succeed; got: {:?}",
            result.err()
        );
        assert_eq!(
            result.unwrap(),
            filepath,
            "cache hit should return the cached path"
        );
    }

    /// Cache hit with a corrupt sidecar: the cached body does not match the
    /// stored SHA-256 — the cache must surface a checksum mismatch rather
    /// than silently re-serving the stale bytes.
    #[test]
    fn test_get_or_download_cache_hit_with_invalid_checksum() {
        let body: &[u8] = b"corrupted tmy3 bytes";
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

        let filepath = cache_dir.join("Boston.tmy3");
        let checksum_path = cache_dir.join("Boston.sha256");
        fs::write(&filepath, body).expect("Failed to seed cached file");
        // 64-char hex string, deliberately not equal to sha256_hex(body).
        fs::write(&checksum_path, "deadbeef".repeat(8)).expect("Failed to seed bad checksum");

        let cache = Tmy3Cache::with_cache_dir(cache_dir).expect("Failed to create cache");
        let result = cache.get_or_download("http://127.0.0.1:1/boston.tmy3", "Boston");
        assert!(result.is_err(), "checksum mismatch should fail closed");
        let err = result.unwrap_err();
        assert!(
            err.contains("Checksum mismatch"),
            "error should report checksum mismatch; got: {}",
            err
        );
        // The corrupt file should still be on disk — `get_or_download` reports
        // the mismatch but does not delete the cached file (that is the
        // consumer's job if they want a forced refresh).
        assert!(
            filepath.exists(),
            "corrupt cache file is left untouched by the mismatch path"
        );
    }

    /// Cache hit with no checksum sidecar: the cache treats the absence of a
    /// `.sha256` as "unverified" and returns the cached path without making
    /// any HTTP request. This is the documented "legacy cache" branch.
    #[test]
    fn test_get_or_download_cache_hit_without_checksum_file() {
        let body: &[u8] = b"legacy cached tmy3 body";
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

        let filepath = cache_dir.join("Seattle.tmy3");
        fs::write(&filepath, body).expect("Failed to seed cached file");
        // Intentionally no Seattle.sha256 sidecar.

        let mut server = Server::new();
        let _unmocked = server.mock("GET", "/seattle.tmy3"); // would 404 if hit

        let cache = Tmy3Cache::with_cache_dir(cache_dir).expect("Failed to create cache");
        let url = format!("{}/seattle.tmy3", server.url());
        let result = cache.get_or_download(&url, "Seattle");
        assert!(
            result.is_ok(),
            "cache hit without checksum file should still succeed; got: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), filepath);
    }

    /// Filename sanitization: location names with whitespace become
    /// underscored cache filenames, and the checksum sidecar uses the same
    /// stem. This pins the `format!("{}.tmy3", location.replace(' ', "_"))`
    /// logic in `get_or_download`.
    #[test]
    fn test_get_or_download_filename_sanitization() {
        let mut server = Server::new();
        let body: &[u8] = b"new york tmy3 body";
        let _m = server
            .mock("GET", "/nyc.tmy3")
            .with_status(200)
            .with_body(body)
            .expect(1)
            .create();

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        let url = format!("{}/nyc.tmy3", server.url());
        let result = cache.get_or_download(&url, "New York City");
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

        let filepath = result.unwrap();
        assert_eq!(
            filepath.file_name().and_then(|n| n.to_str()),
            Some("New_York_City.tmy3"),
            "spaces in location must become underscores in the cache filename"
        );
        let checksum_path = filepath.with_extension("sha256");
        assert_eq!(
            checksum_path.file_name().and_then(|n| n.to_str()),
            Some("New_York_City.sha256"),
            "checksum sidecar uses the same sanitized stem"
        );
        assert_eq!(
            sha256_hex(body),
            fs::read_to_string(&checksum_path).unwrap()
        );
    }

    /// Two distinct locations share one cache directory: each gets its own
    /// `.tmy3` / `.sha256` pair, and the downloads are independent (the
    /// Boston mock would 404 if Boston's URL were misrouted to Denver).
    #[test]
    fn test_get_or_download_two_locations_same_cache() {
        let mut server = Server::new();
        let denver_body: &[u8] = b"denver bytes";
        let boston_body: &[u8] = b"boston bytes";
        let _m_denver = server
            .mock("GET", "/denver.tmy3")
            .with_status(200)
            .with_body(denver_body)
            .expect(1)
            .create();
        let _m_boston = server
            .mock("GET", "/boston.tmy3")
            .with_status(200)
            .with_body(boston_body)
            .expect(1)
            .create();

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        let denver_path = cache
            .get_or_download(&format!("{}/denver.tmy3", server.url()), "Denver")
            .expect("Denver download should succeed");
        let boston_path = cache
            .get_or_download(&format!("{}/boston.tmy3", server.url()), "Boston")
            .expect("Boston download should succeed");

        assert_ne!(
            denver_path, boston_path,
            "different locations should produce distinct cache files"
        );
        assert_eq!(fs::read(&denver_path).unwrap(), denver_body);
        assert_eq!(fs::read(&boston_path).unwrap(), boston_body);

        // Each location's checksum sidecar carries its own digest.
        let denver_checksum = fs::read_to_string(denver_path.with_extension("sha256")).unwrap();
        let boston_checksum = fs::read_to_string(boston_path.with_extension("sha256")).unwrap();
        assert_eq!(denver_checksum, sha256_hex(denver_body));
        assert_eq!(boston_checksum, sha256_hex(boston_body));
        assert_ne!(
            denver_checksum, boston_checksum,
            "different bodies must produce different SHA-256 digests"
        );
    }

    /// Connection error: pointing the cache at a port nothing is listening on
    /// must surface a `Failed to download TMY3: ...` error rather than
    /// panicking, hanging, or caching an empty file. Port 1 is a privileged
    /// port virtually never bound by user processes, so the connect attempt
    /// deterministically fails on a hermetic runner.
    #[test]
    fn test_get_or_download_connection_error() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("tmy3");
        let cache = Tmy3Cache::with_cache_dir(cache_dir.clone()).expect("Failed to create cache");

        let result = cache.get_or_download("http://127.0.0.1:1/nowhere.tmy3", "NoService");
        assert!(result.is_err(), "expected Err for unreachable host");
        let err = result.unwrap_err();
        assert!(
            err.contains("Failed to download TMY3"),
            "error should wrap the reqwest transport failure; got: {}",
            err
        );
        assert!(
            !cache_dir.join("NoService.tmy3").exists(),
            "no file should be cached on connection failure"
        );
        assert!(
            !cache_dir.join("NoService.sha256").exists(),
            "no checksum sidecar should be written on connection failure"
        );
    }

    /// Manifest round-trip: the canonical `data/weather_locations.json` entries
    /// expose HTTP(S) `.tmy3` / `.epw` URLs that the cache's `get_or_download`
    /// can consume. This pins the manifest contract so a regression in URL
    /// shape (e.g. dropping `.zip` / `.epw`) breaks here, not inside an
    /// ASHRAE-140 weather run.
    #[test]
    fn test_load_weather_locations_manifest_url_shape() {
        let locations = load_weather_locations("data/weather_locations.json")
            .expect("Failed to load weather locations");

        assert!(!locations.is_empty(), "manifest should not be empty");

        for (name, loc) in &locations {
            assert!(
                loc.tmy3_url.starts_with("http"),
                "{}: tmy3_url must start with http(s); got {}",
                name,
                loc.tmy3_url
            );
            assert!(
                loc.epw_url.starts_with("http"),
                "{}: epw_url must start with http(s); got {}",
                name,
                loc.epw_url
            );
            assert!(
                loc.tmy3_url.ends_with(".zip"),
                "{}: tmy3_url should be a .zip bundle; got {}",
                name,
                loc.tmy3_url
            );
            assert!(
                loc.epw_url.ends_with(".epw"),
                "{}: epw_url should be an .epw; got {}",
                name,
                loc.epw_url
            );
            assert!(
                loc.latitude.is_finite() && loc.latitude.abs() <= 90.0,
                "{}: latitude must be in [-90, 90]; got {}",
                name,
                loc.latitude
            );
            assert!(
                loc.longitude.is_finite() && loc.longitude.abs() <= 180.0,
                "{}: longitude must be in [-180, 180]; got {}",
                name,
                loc.longitude
            );
            assert!(
                loc.elevation.is_finite(),
                "{}: elevation must be finite; got {}",
                name,
                loc.elevation
            );
        }
    }
}

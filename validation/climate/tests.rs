// validation/climate/tests.rs
#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_climate_zone_definitions() {
        let zones = zones::get_all_climate_zones();
        assert!(!zones.is_empty(), "Climate zones should not be empty");
        assert!(
            zones.len() >= 8,
            "Should have at least 8 major climate zones"
        );
    }

    #[test]
    fn test_specific_climate_zones() {
        let zones = zones::get_all_climate_zones();

        // Test Zone 1A (Very Hot-Humid)
        assert!(zones.contains_key("1A"));
        let zone_1a = zones.get("1A").unwrap();
        assert_eq!(zone_1a.full_name, "Very Hot-Humid");
        assert!(zone_1a.heating_degree_days == 0);
        assert!(zone_1a.cooling_degree_days > 2000);

        // Test Zone 8 (Subarctic/Arctic)
        assert!(zones.contains_key("8"));
        let zone_8 = zones.get("8").unwrap();
        assert_eq!(zone_8.full_name, "Subarctic/Arctic");
        assert!(zone_8.heating_degree_days > 8000);
        assert!(zone_8.cooling_degree_days < 100);
    }

    #[test]
    fn test_major_climate_zones() {
        let major_zones = zones::get_major_climate_zones();
        assert!(
            major_zones.len() >= 8,
            "Should have at least 8 major climate zones"
        );
        assert!(major_zones.contains(&"1A".to_string()));
        assert!(major_zones.contains(&"8".to_string()));
    }

    #[test]
    fn test_validator_creation() {
        let validator = climate::ClimateZoneValidator::new();
        // Just test that it can be created without panicking
        assert!(true, "ClimateZoneValidator should be creatable");
    }
}

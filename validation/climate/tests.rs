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
        assert!(zone_1a.wind_speed_m_s > 0.0);
        assert!(zone_1a.precipitation_mm > 0.0);

        // Test Zone 8 (Subarctic/Arctic)
        assert!(zones.contains_key("8"));
        let zone_8 = zones.get("8").unwrap();
        assert_eq!(zone_8.full_name, "Subarctic/Arctic");
        assert!(zone_8.heating_degree_days > 8000);
        assert!(zone_8.cooling_degree_days < 100);
        assert!(zone_8.wind_speed_m_s > 0.0);
        assert!(zone_8.precipitation_mm > 0.0);
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

    #[test]
    fn test_zone_validation() {
        let validator = climate::ClimateZoneValidator::new();

        // Test validation for a known climate zone
        let result = validator.validate_zone("1A").unwrap();
        assert_eq!(result.zone_id, "1A");
        assert!(result.zone_description.contains("Very Hot-Humid"));
        assert!(!result.validation_metrics.is_empty());

        // Check that validation includes expected metrics
        let metric_names: Vec<String> = result
            .validation_metrics
            .iter()
            .map(|m| m.metric_name.clone())
            .collect();
        assert!(metric_names.contains(&"Temperature Range".to_string()));
        assert!(metric_names.contains(&"Humidity Range".to_string()));
        assert!(metric_names.contains(&"HDD/CDD Balance".to_string()));
    }

    #[test]
    fn test_all_zones_validation() {
        let validator = climate::ClimateZoneValidator::new();
        let results = validator.validate_all_zones();

        assert!(
            !results.is_empty(),
            "Should have validation results for major zones"
        );

        // Check that all results have the expected structure
        for result in results {
            assert!(!result.zone_id.is_empty());
            assert!(!result.zone_description.is_empty());
            assert!(!result.validation_metrics.is_empty());
        }
    }

    #[test]
    fn test_ashrae140_climate_zones() {
        let validator = climate::ClimateZoneValidator::new();
        let results = validator.validate_ashrae140_climate_zones();

        assert!(
            !results.is_empty(),
            "Should have validation results for ASHRAE 140 zones"
        );

        // Check that we have results for expected ASHRAE 140 zones
        let zone_ids: Vec<String> = results.iter().map(|r| r.zone_id.clone()).collect();

        assert!(zone_ids.contains(&"2B".to_string()));
        assert!(zone_ids.contains(&"5A".to_string()));
    }

    #[test]
    fn test_invalid_zone_validation() {
        let validator = climate::ClimateZoneValidator::new();

        // Test validation for an invalid climate zone
        let result = validator.validate_zone("INVALID");
        assert!(result.is_err(), "Should return error for invalid zone");
    }

    #[test]
    fn test_temperature_range_validation() {
        let validator = climate::ClimateZoneValidator::new();

        // Test validation for zones with different temperature characteristics
        let zone_1a = validator.validate_zone("1A").unwrap();
        let zone_8 = validator.validate_zone("8").unwrap();

        // Zone 1A should pass temperature validation (tropical climate)
        let temp_metric_1a = zone_1a
            .validation_metrics
            .iter()
            .find(|m| m.metric_name == "Temperature Range")
            .unwrap();
        assert!(matches!(
            temp_metric_1a.status,
            climate::ValidationStatus::Pass
        ));

        // Zone 8 should pass temperature validation (arctic climate)
        let temp_metric_8 = zone_8
            .validation_metrics
            .iter()
            .find(|m| m.metric_name == "Temperature Range")
            .unwrap();
        assert!(matches!(
            temp_metric_8.status,
            climate::ValidationStatus::Pass
        ));
    }

    #[test]
    fn test_hdd_cdd_balance_validation() {
        let validator = climate::ClimateZoneValidator::new();

        // Test HDD/CDD balance for different climate zones
        let zone_1a = validator.validate_zone("1A").unwrap();
        let zone_8 = validator.validate_zone("8").unwrap();

        // Zone 1A (hot climate) should have low HDD and high CDD
        let hdd_cdd_metric_1a = zone_1a
            .validation_metrics
            .iter()
            .find(|m| m.metric_name == "HDD/CDD Balance")
            .unwrap();
        assert!(matches!(
            hdd_cdd_metric_1a.status,
            climate::ValidationStatus::Pass
        ));
        assert!(hdd_cdd_metric_1a.value < 0.5); // Low HDD/CDD ratio for hot climates

        // Zone 8 (cold climate) should have high HDD and low CDD
        let hdd_cdd_metric_8 = zone_8
            .validation_metrics
            .iter()
            .find(|m| m.metric_name == "HDD/CDD Balance")
            .unwrap();
        assert!(matches!(
            hdd_cdd_metric_8.status,
            climate::ValidationStatus::Pass
        ));
        assert!(hdd_cdd_metric_8.value > 10.0); // High HDD/CDD ratio for cold climates
    }
}

// validation/climate/test_integration.rs
#[cfg(test)]
mod climate_integration_tests {
    use super::super::*;

    #[test]
    fn test_climate_zone_integration() {
        // Test that climate zones can be created and validated
        let zones = zones::get_all_climate_zones();
        assert!(!zones.is_empty(), "Climate zones should not be empty");

        // Test validator creation
        let validator = climate::ClimateZoneValidator::new();

        // Test validation for a specific zone
        let result = validator.validate_zone("1A");
        assert!(result.is_ok(), "Zone 1A validation should succeed");

        let validation_result = result.unwrap();
        assert_eq!(validation_result.zone_id, "1A");
        assert!(!validation_result.validation_metrics.is_empty());

        // Test all zones validation
        let all_results = validator.validate_all_zones();
        assert!(
            !all_results.is_empty(),
            "Should have results for major zones"
        );

        // Test ASHRAE 140 climate zones
        let ashrae_results = validator.validate_ashrae140_climate_zones();
        assert!(
            !ashrae_results.is_empty(),
            "Should have results for ASHRAE 140 zones"
        );
    }
}

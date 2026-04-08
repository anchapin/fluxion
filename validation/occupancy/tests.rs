// validation/occupancy/tests.rs
use crate::validation::occupancy::patterns::{get_occupancy_pattern, OccupancySchedule};
use crate::validation::occupancy::{OccupancyValidationResult, OccupancyValidator};

#[test]
fn test_occupancy_pattern_definitions() {
    let patterns = get_standard_occupancy_patterns();
    assert!(
        !patterns.is_empty(),
        "Occupancy patterns should not be empty"
    );
    assert!(
        patterns.len() >= 5,
        "Should have at least 5 standard occupancy patterns"
    );
}

#[test]
fn test_specific_occupancy_patterns() {
    let patterns = get_standard_occupancy_patterns();

    // Test residential pattern
    assert!(patterns.contains_key("residential"));
    let residential = patterns.get("residential").unwrap();
    assert_eq!(residential.name, "residential");
    assert!(
        residential.hourly_values[0] > 0.5,
        "Residential should have high night occupancy"
    );

    // Test commercial pattern
    assert!(patterns.contains_key("commercial"));
    let commercial = patterns.get("commercial").unwrap();
    assert_eq!(commercial.name, "commercial");
    assert!(
        commercial.hourly_values[8] > 0.8,
        "Commercial should have high daytime occupancy"
    );
}

#[test]
fn test_pattern_validation() {
    let patterns = get_standard_occupancy_patterns();
    for (name, pattern) in patterns.iter() {
        let result = pattern.validate();
        assert!(
            result.is_ok(),
            "Pattern {} should be valid: {:?}",
            name,
            result
        );
    }
}

#[test]
fn test_all_patterns_valid() {
    let result = patterns::validate_all_patterns();
    assert!(result.is_ok(), "All patterns should be valid");
}

#[test]
fn test_validator_creation() {
    let validator = OccupancyValidator::new();
    // Just test that it can be created without panicking
    assert!(true, "OccupancyValidator should be creatable");
}

#[test]
fn test_residential_pattern_validation() {
    let validator = OccupancyValidator::new();
    let result = validator.validate_pattern("residential");

    assert!(result.is_valid, "Residential pattern should be valid");
    assert_eq!(result.pattern_name, "residential");
    assert_eq!(
        result.errors.len(),
        0,
        "Residential pattern should have no errors"
    );
}

#[test]
fn test_commercial_pattern_validation() {
    let validator = OccupancyValidator::new();
    let result = validator.validate_pattern("commercial");

    assert!(result.is_valid, "Commercial pattern should be valid");
    assert_eq!(result.pattern_name, "commercial");
    assert_eq!(
        result.errors.len(),
        0,
        "Commercial pattern should have no errors"
    );
}

#[test]
fn test_school_pattern_validation() {
    let validator = OccupancyValidator::new();
    let result = validator.validate_pattern("school");

    assert!(result.is_valid, "School pattern should be valid");
    assert_eq!(result.pattern_name, "school");
    assert_eq!(
        result.errors.len(),
        0,
        "School pattern should have no errors"
    );
}

#[test]
fn test_hospital_pattern_validation() {
    let validator = OccupancyValidator::new();
    let result = validator.validate_pattern("hospital");

    assert!(result.is_valid, "Hospital pattern should be valid");
    assert_eq!(result.pattern_name, "hospital");
    assert_eq!(
        result.errors.len(),
        0,
        "Hospital pattern should have no errors"
    );
}

#[test]
fn test_retail_pattern_validation() {
    let validator = OccupancyValidator::new();
    let result = validator.validate_pattern("retail");

    assert!(result.is_valid, "Retail pattern should be valid");
    assert_eq!(result.pattern_name, "retail");
    assert_eq!(
        result.errors.len(),
        0,
        "Retail pattern should have no errors"
    );
}

#[test]
fn test_unknown_pattern_validation() {
    let validator = OccupancyValidator::new();
    let result = validator.validate_pattern("unknown");

    assert!(!result.is_valid, "Unknown pattern should be invalid");
    assert_eq!(result.pattern_name, "unknown");
    assert_eq!(
        result.errors.len(),
        1,
        "Unknown pattern should have one error"
    );
    assert!(
        result.errors[0].contains("Unknown occupancy pattern"),
        "Error should mention unknown pattern"
    );
}

#[test]
fn test_pattern_structure_validation() {
    let validator = OccupancyValidator::new();

    // Test all standard patterns
    let pattern_names = ["residential", "commercial", "school", "hospital", "retail"];

    for pattern_name in pattern_names.iter() {
        let result = validator.validate_pattern(pattern_name);

        // All standard patterns should be valid
        assert!(result.is_valid, "Pattern {} should be valid", pattern_name);

        // Should have no errors (warnings are okay)
        assert_eq!(
            result.errors.len(),
            0,
            "Pattern {} should have no errors",
            pattern_name
        );
    }
}

#[test]
fn test_occupancy_threshold_validation() {
    let validator = OccupancyValidator::new();

    // Test that all patterns have values within valid range
    let pattern_names = ["residential", "commercial", "school", "hospital", "retail"];

    for pattern_name in pattern_names.iter() {
        let pattern = get_occupancy_pattern(pattern_name).unwrap();
        let result = validator.validate_pattern(pattern_name);

        // Check that all values are within [0.0, 1.0] range
        for (hour, &value) in pattern.hourly_values.iter().enumerate() {
            assert!(
                value >= 0.0 && value <= 1.0,
                "Pattern {} hour {} has invalid occupancy value {}",
                pattern_name,
                hour,
                value
            );
        }

        // Validation should pass
        assert!(
            result.is_valid,
            "Pattern {} threshold validation should pass",
            pattern_name
        );
    }
}

#[test]
fn test_peak_hours_validation() {
    let validator = OccupancyValidator::new();

    // Test commercial pattern - should have high occupancy during business hours
    let result = validator.validate_pattern("commercial");

    // Commercial patterns should have high occupancy during business hours (8am-5pm)
    let pattern = get_occupancy_pattern("commercial").unwrap();
    for hour in 8..17 {
        // 8am to 5pm
        let occupancy = pattern.get_hourly_occupancy(hour);
        assert!(
            occupancy >= 0.7,
            "Commercial pattern should have high occupancy at hour {} (got {})",
            hour,
            occupancy
        );
    }
}

#[test]
fn test_pattern_variation_validation() {
    let validator = OccupancyValidator::new();

    // Test that patterns have reasonable variation
    let pattern_names = ["residential", "commercial", "school", "hospital", "retail"];

    for pattern_name in pattern_names.iter() {
        let pattern = get_occupancy_pattern(pattern_name).unwrap();
        let result = validator.validate_pattern(pattern_name);

        // Calculate variation
        let min_value = pattern
            .hourly_values
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_value = pattern
            .hourly_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let variation = max_value - min_value;

        // Patterns should have some variation (except possibly hospital)
        if pattern_name != "hospital" {
            assert!(
                variation > 0.3,
                "Pattern {} should have reasonable variation (got {})",
                pattern_name,
                variation
            );
        }
    }
}

#[test]
fn test_residential_evening_occupancy() {
    let validator = OccupancyValidator::new();
    let pattern = get_occupancy_pattern("residential").unwrap();
    let result = validator.validate_pattern("residential");

    // Residential patterns should have higher occupancy in evenings
    let evening_occupancy: f64 = (18..23).map(|h| pattern.get_hourly_occupancy(h)).sum();
    let morning_occupancy: f64 = (6..12).map(|h| pattern.get_hourly_occupancy(h)).sum();

    assert!(
        evening_occupancy >= morning_occupancy,
        "Residential pattern: evening occupancy ({}) should be >= morning occupancy ({})",
        evening_occupancy,
        morning_occupancy
    );

    // Validation should pass
    assert!(
        result.is_valid,
        "Residential pattern validation should pass"
    );
}

#[test]
fn test_commercial_business_hours() {
    let validator = OccupancyValidator::new();
    let pattern = get_occupancy_pattern("commercial").unwrap();
    let result = validator.validate_pattern("commercial");

    // Commercial patterns should have clear business hours
    let business_hours_occupancy: f64 = (8..18).map(|h| pattern.get_hourly_occupancy(h)).sum();
    let night_hours_occupancy: f64 = (0..6)
        .chain(18..24)
        .map(|h| pattern.get_hourly_occupancy(h))
        .sum();

    assert!(
        business_hours_occupancy > night_hours_occupancy,
        "Commercial pattern: business hours occupancy ({}) should be > night hours occupancy ({})",
        business_hours_occupancy,
        night_hours_occupancy
    );

    // Validation should pass
    assert!(result.is_valid, "Commercial pattern validation should pass");
}

#[test]
fn test_school_hours_validation() {
    let validator = OccupancyValidator::new();
    let pattern = get_occupancy_pattern("school").unwrap();
    let result = validator.validate_pattern("school");

    // School patterns should have occupancy during school hours (8am-4pm)
    let school_hours_occupancy: f64 = (8..16).map(|h| pattern.get_hourly_occupancy(h)).sum();
    let non_school_hours_occupancy: f64 = (0..8)
        .chain(16..24)
        .map(|h| pattern.get_hourly_occupancy(h))
        .sum();

    assert!(
        school_hours_occupancy > non_school_hours_occupancy,
        "School pattern: school hours occupancy ({}) should be > non-school hours occupancy ({})",
        school_hours_occupancy,
        non_school_hours_occupancy
    );

    // Validation should pass
    assert!(result.is_valid, "School pattern validation should pass");
}

#[test]
fn test_hospital_24_7_operation() {
    let validator = OccupancyValidator::new();
    let pattern = get_occupancy_pattern("hospital").unwrap();
    let result = validator.validate_pattern("hospital");

    // Hospital patterns should have 24/7 occupancy
    let min_occupancy = pattern
        .hourly_values
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));

    assert!(
        min_occupancy >= 0.5,
        "Hospital pattern: should maintain minimum occupancy >= 0.5 (got {})",
        min_occupancy
    );

    // Validation should pass
    assert!(result.is_valid, "Hospital pattern validation should pass");
}

#[test]
fn test_retail_extended_hours() {
    let validator = OccupancyValidator::new();
    let pattern = get_occupancy_pattern("retail").unwrap();
    let result = validator.validate_pattern("retail");

    // Retail patterns should have extended hours
    let first_occupied_hour = pattern
        .hourly_values
        .iter()
        .position(|&v| v > 0.1)
        .unwrap_or(24);
    let last_occupied_hour = pattern
        .hourly_values
        .iter()
        .rposition(|&v| v > 0.1)
        .unwrap_or(0);

    let operating_hours = if first_occupied_hour <= last_occupied_hour {
        last_occupied_hour - first_occupied_hour
    } else {
        0
    };

    assert!(
        operating_hours >= 12,
        "Retail pattern: should have extended operating hours >= 12 (got {})",
        operating_hours
    );

    // Validation should pass
    assert!(result.is_valid, "Retail pattern validation should pass");
}

#[test]
fn test_all_patterns_validation() {
    let results = OccupancyValidator::validate_all_patterns();

    // Should have results for all 5 standard patterns
    assert_eq!(
        results.len(),
        5,
        "Should have results for 5 standard patterns"
    );

    // All patterns should be valid
    for (name, result) in results.iter() {
        assert!(result.is_valid, "Pattern {} should be valid", name);
        assert_eq!(result.pattern_name, *name, "Pattern name should match");
    }
}

#[test]
fn test_pattern_validation_result_structure() {
    let validator = OccupancyValidator::new();
    let result = validator.validate_pattern("residential");

    // Test result structure
    assert_eq!(result.pattern_name, "residential");
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
}

#[test]
fn test_invalid_pattern_structure() {
    // Create an invalid pattern with out-of-range values
    let mut invalid_pattern = OccupancySchedule::new(
        "invalid",
        [2.0; 24], // All values > 1.0
        "Invalid test pattern",
    );

    // This should fail validation
    assert!(
        invalid_pattern.validate().is_err(),
        "Invalid pattern should fail validation"
    );
}

#[test]
fn test_pattern_hourly_values() {
    // Test that we can access hourly values correctly
    let pattern = get_occupancy_pattern("residential").unwrap();

    // Test a few specific hours
    assert!(
        pattern.get_hourly_occupancy(0) > 0.0,
        "Midnight should have some occupancy"
    );
    assert!(
        pattern.get_hourly_occupancy(12) > 0.0,
        "Noon should have some occupancy"
    );
    assert!(
        pattern.get_hourly_occupancy(18) > 0.0,
        "6 PM should have some occupancy"
    );

    // Test invalid hour
    assert_eq!(
        pattern.get_hourly_occupancy(24),
        0.0,
        "Invalid hour should return 0.0"
    );
}

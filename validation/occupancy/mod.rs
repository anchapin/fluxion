// validation/occupancy/mod.rs
/// Occupancy pattern validation module
///
/// This module provides validation capabilities for different occupancy patterns
/// and their impact on building energy performance
pub mod patterns;

use crate::validation::occupancy::patterns::{get_occupancy_pattern, OccupancySchedule};
use std::collections::HashMap;

/// Occupancy validation result
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyValidationResult {
    pub pattern_name: String,
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl OccupancyValidationResult {
    pub fn new(pattern_name: &str) -> Self {
        Self {
            pattern_name: pattern_name.to_string(),
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: &str) {
        self.is_valid = false;
        self.errors.push(error.to_string());
    }

    pub fn add_warning(&mut self, warning: &str) {
        self.warnings.push(warning.to_string());
    }
}

/// Occupancy validator
pub struct OccupancyValidator {
    // Validation rules and constraints
    min_occupancy_threshold: f64,
    max_occupancy_threshold: f64,
    required_peak_hours: Vec<usize>, // Hours that must have occupancy > threshold
}

impl OccupancyValidator {
    pub fn new() -> Self {
        Self {
            min_occupancy_threshold: 0.0,
            max_occupancy_threshold: 1.0,
            required_peak_hours: vec![8, 9, 10, 11, 12, 13, 14, 15, 16], // Typical business hours
        }
    }

    /// Validate a specific occupancy pattern
    pub fn validate_pattern(&self, pattern_name: &str) -> OccupancyValidationResult {
        let mut result = OccupancyValidationResult::new(pattern_name);

        // Get the occupancy pattern
        let pattern = match get_occupancy_pattern(pattern_name) {
            Some(p) => p,
            None => {
                result.add_error(&format!("Unknown occupancy pattern: {}", pattern_name));
                return result;
            }
        };

        // Validate basic pattern structure
        self.validate_pattern_structure(&pattern, &mut result);

        // Validate pattern-specific requirements
        self.validate_pattern_requirements(&pattern, &mut result);

        result
    }

    /// Validate basic pattern structure
    fn validate_pattern_structure(
        &self,
        pattern: &OccupancySchedule,
        result: &mut OccupancyValidationResult,
    ) {
        // Check that all values are within valid range
        for (hour, &value) in pattern.hourly_values.iter().enumerate() {
            if value < self.min_occupancy_threshold {
                result.add_error(&format!(
                    "Hour {} has occupancy {} below minimum threshold {}",
                    hour, value, self.min_occupancy_threshold
                ));
            }
            if value > self.max_occupancy_threshold {
                result.add_error(&format!(
                    "Hour {} has occupancy {} above maximum threshold {}",
                    hour, value, self.max_occupancy_threshold
                ));
            }
        }

        // Check that pattern has reasonable variation
        let min_value = pattern
            .hourly_values
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_value = pattern
            .hourly_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if max_value - min_value < 0.1 {
            result.add_warning("Pattern has very little variation between min and max occupancy");
        }
    }

    /// Validate pattern-specific requirements
    fn validate_pattern_requirements(
        &self,
        pattern: &OccupancySchedule,
        result: &mut OccupancyValidationResult,
    ) {
        // Check that required peak hours have sufficient occupancy
        for &hour in &self.required_peak_hours {
            let occupancy = pattern.get_hourly_occupancy(hour);
            if occupancy < 0.3 {
                result.add_warning(&format!(
                    "Peak hour {} has low occupancy {} (expected > 0.3)",
                    hour, occupancy
                ));
            }
        }

        // Pattern-specific validations
        match pattern.name.as_str() {
            "residential" => self.validate_residential_pattern(pattern, result),
            "commercial" => self.validate_commercial_pattern(pattern, result),
            "school" => self.validate_school_pattern(pattern, result),
            "hospital" => self.validate_hospital_pattern(pattern, result),
            "retail" => self.validate_retail_pattern(pattern, result),
            _ => {}
        }
    }

    /// Validate residential pattern
    fn validate_residential_pattern(
        &self,
        pattern: &OccupancySchedule,
        result: &mut OccupancyValidationResult,
    ) {
        // Residential patterns should have higher occupancy in evenings
        let evening_occupancy: f64 = (18..23).map(|h| pattern.get_hourly_occupancy(h)).sum();
        let morning_occupancy: f64 = (6..12).map(|h| pattern.get_hourly_occupancy(h)).sum();

        if evening_occupancy < morning_occupancy {
            result.add_warning(
                "Residential pattern: evening occupancy should typically be higher than morning",
            );
        }
    }

    /// Validate commercial pattern
    fn validate_commercial_pattern(
        &self,
        pattern: &OccupancySchedule,
        result: &mut OccupancyValidationResult,
    ) {
        // Commercial patterns should have clear business hours
        let business_hours_occupancy: f64 = (8..18).map(|h| pattern.get_hourly_occupancy(h)).sum();
        let night_hours_occupancy: f64 = (0..6)
            .chain(18..24)
            .map(|h| pattern.get_hourly_occupancy(h))
            .sum();

        if business_hours_occupancy <= night_hours_occupancy {
            result.add_error("Commercial pattern: business hours should have significantly higher occupancy than night hours");
        }
    }

    /// Validate school pattern
    fn validate_school_pattern(
        &self,
        pattern: &OccupancySchedule,
        result: &mut OccupancyValidationResult,
    ) {
        // School patterns should have occupancy during school hours (8am-4pm)
        let school_hours_occupancy: f64 = (8..16).map(|h| pattern.get_hourly_occupancy(h)).sum();
        let non_school_hours_occupancy: f64 = (0..8)
            .chain(16..24)
            .map(|h| pattern.get_hourly_occupancy(h))
            .sum();

        if school_hours_occupancy <= non_school_hours_occupancy {
            result.add_error("School pattern: school hours should have significantly higher occupancy than non-school hours");
        }
    }

    /// Validate hospital pattern
    fn validate_hospital_pattern(
        &self,
        pattern: &OccupancySchedule,
        result: &mut OccupancyValidationResult,
    ) {
        // Hospital patterns should have 24/7 occupancy
        let min_occupancy = pattern
            .hourly_values
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        if min_occupancy < 0.5 {
            result.add_warning(
                "Hospital pattern: should maintain higher minimum occupancy for 24/7 operation",
            );
        }
    }

    /// Validate retail pattern
    fn validate_retail_pattern(
        &self,
        pattern: &OccupancySchedule,
        result: &mut OccupancyValidationResult,
    ) {
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

        if operating_hours < 12 {
            result.add_warning(
                "Retail pattern: should typically have extended operating hours (> 12 hours)",
            );
        }
    }

    /// Validate all standard occupancy patterns
    pub fn validate_all_patterns() -> HashMap<String, OccupancyValidationResult> {
        let validator = Self::new();
        let pattern_names = ["residential", "commercial", "school", "hospital", "retail"];

        pattern_names
            .iter()
            .map(|&name| (name.to_string(), validator.validate_pattern(name)))
            .collect()
    }
}

/// Legacy compatibility types (kept for backward compatibility)
#[derive(Debug, Clone)]
pub struct OccupancyPatternValidationResult {
    pub pattern_name: String,
    pub pattern_description: String,
    pub validation_status: ValidationStatus,
    pub coverage_percentage: f64,
    pub energy_impact_analysis: EnergyImpactAnalysis,
}

/// Validation status
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Pass,
    Warning,
    Fail,
}

/// Energy impact analysis
#[derive(Debug, Clone)]
pub struct EnergyImpactAnalysis {
    pub heating_impact_percentage: f64,
    pub cooling_impact_percentage: f64,
    pub peak_load_impact_percentage: f64,
    pub overall_energy_variation: f64,
}

/// Legacy validator (kept for backward compatibility)
pub struct OccupancyPatternValidator {
    // Configuration and state will be added here
}

impl OccupancyPatternValidator {
    /// Create a new occupancy pattern validator
    pub fn new() -> Self {
        Self {
            // Initialize validator
        }
    }

    /// Validate a specific occupancy pattern
    pub fn validate_pattern(
        &self,
        pattern_name: &str,
    ) -> Result<OccupancyPatternValidationResult, String> {
        // Implementation will validate occupancy pattern performance
        Ok(OccupancyPatternValidationResult {
            pattern_name: pattern_name.to_string(),
            pattern_description: format!("Occupancy pattern: {}", pattern_name),
            validation_status: ValidationStatus::Pass,
            coverage_percentage: 100.0,
            energy_impact_analysis: EnergyImpactAnalysis {
                heating_impact_percentage: 0.0,
                cooling_impact_percentage: 0.0,
                peak_load_impact_percentage: 0.0,
                overall_energy_variation: 0.0,
            },
        })
    }

    /// Run validation for all standard occupancy patterns
    pub fn validate_all_patterns(&self) -> Vec<OccupancyPatternValidationResult> {
        // This will run validation for all standard occupancy patterns
        vec![]
    }

    /// Validate occupancy pattern integration with building simulation
    pub fn validate_integration(
        &self,
        pattern_name: &str,
    ) -> Result<IntegrationValidationResult, String> {
        // Implementation will validate how well the occupancy pattern integrates
        // with the building energy simulation
        Ok(IntegrationValidationResult {
            pattern_name: pattern_name.to_string(),
            integration_status: ValidationStatus::Pass,
            simulation_compatibility: 100.0,
            data_quality_score: 100.0,
        })
    }
}

/// Integration validation result
#[derive(Debug, Clone)]
pub struct IntegrationValidationResult {
    pub pattern_name: String,
    pub integration_status: ValidationStatus,
    pub simulation_compatibility: f64,
    pub data_quality_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_residential_pattern() {
        let validator = OccupancyValidator::new();
        let result = validator.validate_pattern("residential");
        assert!(result.is_valid, "Residential pattern should be valid");
        assert_eq!(result.pattern_name, "residential");
    }

    #[test]
    fn test_validate_commercial_pattern() {
        let validator = OccupancyValidator::new();
        let result = validator.validate_pattern("commercial");
        assert!(result.is_valid, "Commercial pattern should be valid");
        assert_eq!(result.pattern_name, "commercial");
    }

    #[test]
    fn test_validate_unknown_pattern() {
        let validator = OccupancyValidator::new();
        let result = validator.validate_pattern("unknown");
        assert!(!result.is_valid, "Unknown pattern should be invalid");
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_validate_all_patterns() {
        let results = OccupancyValidator::validate_all_patterns();
        assert_eq!(results.len(), 5);

        for (name, result) in results.iter() {
            // All standard patterns should be valid
            assert!(result.is_valid, "Pattern {} should be valid", name);
        }
    }

    #[test]
    fn test_pattern_structure_validation() {
        let validator = OccupancyValidator::new();
        let result = validator.validate_pattern("residential");

        // Should have no errors for standard patterns
        assert_eq!(result.errors.len(), 0);
    }
}

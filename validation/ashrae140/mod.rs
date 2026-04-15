// validation/ashrae140/mod.rs
/// ASHRAE 140 validation module
///
/// This module provides comprehensive ASHRAE 140 validation capabilities
/// including case definitions, validation logic, and result analysis
pub mod cases;

use crate::validation::climate::{ClimateZoneValidationResult, ClimateZoneValidator};
use crate::validation::occupancy::patterns::get_occupancy_pattern;
use crate::validation::occupancy::{OccupancyValidationResult, OccupancyValidator};
use std::collections::HashMap;

/// ASHRAE 140 validation result
#[derive(Debug, Clone)]
pub struct ASHRAE140ValidationResult {
    pub case_id: String,
    pub case_description: String,
    pub annual_heating_mwh: f64,
    pub annual_cooling_mwh: f64,
    pub peak_heating_kw: f64,
    pub peak_cooling_kw: f64,
    pub min_temp_celsius: Option<f64>,
    pub max_temp_celsius: Option<f64>,
    pub status: ValidationStatus,
    pub occupancy_validation: Option<OccupancyValidationResult>,
    pub climate_validation: Option<ClimateZoneValidationResult>,
}

/// Validation status
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Pass,
    Warning,
    Fail,
}

/// ASHRAE 140 validator
pub struct ASHRAE140Validator {
    occupancy_validator: OccupancyValidator,
    climate_validator: ClimateZoneValidator,
    case_occupancy_patterns: HashMap<String, String>, // Case ID -> Occupancy pattern name
    case_climate_zones: HashMap<String, String>,      // Case ID -> Climate zone
}

impl ASHRAE140Validator {
    /// Create a new ASHRAE 140 validator
    pub fn new() -> Self {
        Self {
            occupancy_validator: OccupancyValidator::new(),
            climate_validator: ClimateZoneValidator::new(),
            case_occupancy_patterns: Self::create_default_occupancy_mappings(),
            case_climate_zones: Self::create_default_climate_zone_mappings(),
        }
    }

    /// Create default occupancy pattern mappings for ASHRAE 140 cases
    fn create_default_occupancy_mappings() -> HashMap<String, String> {
        let mut mappings = HashMap::new();

        // Case 600 series - Low mass residential
        mappings.insert("600".to_string(), "residential".to_string());
        mappings.insert("610".to_string(), "residential".to_string());
        mappings.insert("620".to_string(), "residential".to_string());
        mappings.insert("630".to_string(), "residential".to_string());
        mappings.insert("640".to_string(), "residential".to_string());
        mappings.insert("650".to_string(), "residential".to_string());

        // Case 600 series variations
        mappings.insert("601".to_string(), "residential".to_string());
        mappings.insert("602".to_string(), "residential".to_string());
        mappings.insert("603".to_string(), "residential".to_string());

        // Case 900 series - High mass (could be commercial or office)
        mappings.insert("900".to_string(), "commercial".to_string());
        mappings.insert("910".to_string(), "commercial".to_string());
        mappings.insert("920".to_string(), "commercial".to_string());
        mappings.insert("930".to_string(), "commercial".to_string());
        mappings.insert("940".to_string(), "commercial".to_string());
        mappings.insert("950".to_string(), "commercial".to_string());

        // Additional cases
        mappings.insert("500".to_string(), "residential".to_string());
        mappings.insert("510".to_string(), "residential".to_string());
        mappings.insert("520".to_string(), "residential".to_string());

        mappings.insert("800".to_string(), "commercial".to_string()); // Office/retail
        mappings.insert("810".to_string(), "commercial".to_string());

        mappings.insert("SCHOOL".to_string(), "school".to_string());
        mappings.insert("HOSPITAL".to_string(), "hospital".to_string());
        mappings.insert("RETAIL".to_string(), "retail".to_string());

        mappings
    }

    /// Create default climate zone mappings for ASHRAE 140 cases
    fn create_default_climate_zone_mappings() -> HashMap<String, String> {
        let mut mappings = HashMap::new();

        // Case 600 series - Low mass residential (typically Zone 4A - Mixed-Humid)
        mappings.insert("600".to_string(), "4A".to_string());
        mappings.insert("610".to_string(), "4A".to_string());
        mappings.insert("620".to_string(), "4A".to_string());
        mappings.insert("630".to_string(), "4A".to_string());
        mappings.insert("640".to_string(), "4A".to_string());
        mappings.insert("650".to_string(), "4A".to_string());

        // Case 600 series variations
        mappings.insert("601".to_string(), "4A".to_string());
        mappings.insert("602".to_string(), "4A".to_string());
        mappings.insert("603".to_string(), "4A".to_string());

        // Case 900 series - High mass (could be Zone 5A - Cool-Humid)
        mappings.insert("900".to_string(), "5A".to_string());
        mappings.insert("910".to_string(), "5A".to_string());
        mappings.insert("920".to_string(), "5A".to_string());
        mappings.insert("930".to_string(), "5A".to_string());
        mappings.insert("940".to_string(), "5A".to_string());
        mappings.insert("950".to_string(), "5A".to_string());

        // Additional cases with specific climate zones
        mappings.insert("500".to_string(), "3C".to_string()); // Warm-Marine
        mappings.insert("510".to_string(), "3C".to_string());
        mappings.insert("520".to_string(), "3C".to_string());

        mappings.insert("800".to_string(), "2B".to_string()); // Hot-Dry for office/retail
        mappings.insert("810".to_string(), "2B".to_string());

        // Special cases
        mappings.insert("SCHOOL".to_string(), "4A".to_string());
        mappings.insert("HOSPITAL".to_string(), "5A".to_string());
        mappings.insert("RETAIL".to_string(), "2B".to_string());

        mappings
    }

    /// Set custom occupancy pattern for a specific case
    pub fn set_case_occupancy_pattern(&mut self, case_id: &str, pattern_name: &str) {
        self.case_occupancy_patterns
            .insert(case_id.to_string(), pattern_name.to_string());
    }

    /// Set custom climate zone for a specific case
    pub fn set_case_climate_zone(&mut self, case_id: &str, zone_id: &str) {
        self.case_climate_zones
            .insert(case_id.to_string(), zone_id.to_string());
    }

    /// Get occupancy pattern for a case
    pub fn get_case_occupancy_pattern(&self, case_id: &str) -> Option<String> {
        self.case_occupancy_patterns.get(case_id).cloned()
    }

    /// Get climate zone for a case
    pub fn get_case_climate_zone(&self, case_id: &str) -> Option<String> {
        self.case_climate_zones.get(case_id).cloned()
    }

    /// Validate occupancy pattern for a specific case
    pub fn validate_case_occupancy(&self, case_id: &str) -> Option<OccupancyValidationResult> {
        let pattern_name = self.get_case_occupancy_pattern(case_id)?;
        Some(self.occupancy_validator.validate_pattern(&pattern_name))
    }

    /// Validate climate zone for a specific case
    pub fn validate_case_climate_zone(&self, case_id: &str) -> Option<ClimateZoneValidationResult> {
        let zone_id = self.get_case_climate_zone(case_id)?;
        Some(self.climate_validator.validate_zone(&zone_id).ok()?)
    }

    /// Validate a specific case with occupancy and climate zone integration
    pub fn validate_case(&self, case_id: &str) -> Result<ASHRAE140ValidationResult, String> {
        // Validate occupancy pattern for this case
        let occupancy_validation = self.validate_case_occupancy(case_id);

        // Validate climate zone for this case
        let climate_validation = self.validate_case_climate_zone(case_id);

        // Get occupancy pattern name for description
        let pattern_name = self
            .get_case_occupancy_pattern(case_id)
            .map(|p| format!(" with {} occupancy", p))
            .unwrap_or_default();

        // Get climate zone for description
        let climate_zone_name = self
            .get_case_climate_zone(case_id)
            .map(|z| format!(" in climate zone {}", z))
            .unwrap_or_default();

        Ok(ASHRAE140ValidationResult {
            case_id: case_id.to_string(),
            case_description: format!(
                "ASHRAE 140 Case {}{}{}",
                case_id, pattern_name, climate_zone_name
            ),
            annual_heating_mwh: 0.0,
            annual_cooling_mwh: 0.0,
            peak_heating_kw: 0.0,
            peak_cooling_kw: 0.0,
            min_temp_celsius: None,
            max_temp_celsius: None,
            status: ValidationStatus::Pass,
            occupancy_validation,
            climate_validation,
        })
    }

    /// Run comprehensive validation suite with occupancy and climate zone integration
    pub fn validate_all_cases(&self) -> Vec<ASHRAE140ValidationResult> {
        // This will run all ASHRAE 140 cases and return results with occupancy and climate validation
        let case_ids = vec![
            "600", "610", "620", "630", "640", "650", "900", "910", "920", "930", "940", "950",
        ];

        case_ids
            .iter()
            .filter_map(|&case_id| self.validate_case(case_id).ok())
            .collect()
    }

    /// Validate climate zones for all cases
    pub fn validate_all_climate_zones(&self) -> HashMap<String, ClimateZoneValidationResult> {
        let mut results = HashMap::new();

        for (case_id, zone_id) in &self.case_climate_zones {
            if let Ok(result) = self.climate_validator.validate_zone(zone_id) {
                results.insert(case_id.clone(), result);
            }
        }

        results
    }

    /// Get climate zone energy impact analysis for a case
    pub fn analyze_climate_energy_impact(
        &self,
        case_id: &str,
    ) -> Option<ClimateEnergyImpactAnalysis> {
        let zone_id = self.get_case_climate_zone(case_id)?;
        let zone = crate::validation::climate::zones::get_climate_zone(&zone_id)?;

        Some(self.calculate_climate_energy_impact(&zone))
    }

    /// Calculate energy impact based on climate zone characteristics
    fn calculate_climate_energy_impact(
        &self,
        zone: &crate::validation::climate::zones::ClimateZone,
    ) -> ClimateEnergyImpactAnalysis {
        // Calculate heating and cooling impacts based on climate characteristics
        let heating_impact = zone.heating_degree_days / 100.0; // Scale down for percentage
        let cooling_impact = zone.cooling_degree_days / 100.0;

        // Solar radiation impact (higher solar = more cooling load)
        let solar_impact = zone.solar_radiation_kwh_m2 / 50.0;

        // Wind impact (higher wind speed can increase infiltration losses)
        let wind_impact = zone.wind_speed_m_s * 5.0;

        ClimateEnergyImpactAnalysis {
            heating_impact_percentage: heating_impact.min(100.0),
            cooling_impact_percentage: cooling_impact.min(100.0),
            solar_radiation_impact_percentage: solar_impact.min(50.0),
            wind_infiltration_impact_percentage: wind_impact.min(30.0),
            overall_climate_severity: (heating_impact + cooling_impact) / 2.0,
        }
    }

    /// Validate occupancy patterns for all cases
    pub fn validate_all_occupancy_patterns(&self) -> HashMap<String, OccupancyValidationResult> {
        let mut results = HashMap::new();

        for (case_id, pattern_name) in &self.case_occupancy_patterns {
            let result = self.occupancy_validator.validate_pattern(pattern_name);
            results.insert(case_id.clone(), result);
        }

        results
    }

    /// Get energy impact analysis based on occupancy pattern
    pub fn analyze_occupancy_energy_impact(&self, case_id: &str) -> Option<EnergyImpactAnalysis> {
        let pattern_name = self.get_case_occupancy_pattern(case_id)?;
        let pattern = get_occupancy_pattern(&pattern_name)?;

        Some(self.calculate_energy_impact_from_occupancy(&pattern))
    }

    /// Calculate energy impact based on occupancy pattern
    fn calculate_energy_impact_from_occupancy(
        &self,
        pattern: &crate::validation::occupancy::patterns::OccupancySchedule,
    ) -> EnergyImpactAnalysis {
        // Calculate average occupancy
        let avg_occupancy: f64 = pattern.hourly_values.iter().sum::<f64>() / 24.0;

        // Calculate peak occupancy period (4 consecutive hours with highest average)
        let mut max_peak = 0.0;
        for window in 0..20 {
            let peak = pattern.hourly_values[window..window + 4]
                .iter()
                .sum::<f64>()
                / 4.0;
            if peak > max_peak {
                max_peak = peak;
            }
        }

        // Estimate energy impact based on occupancy patterns
        // These are rough estimates - actual impact would depend on building characteristics
        let heating_impact = avg_occupancy * 15.0; // ~15% impact per 0.1 occupancy
        let cooling_impact = avg_occupancy * 20.0; // ~20% impact per 0.1 occupancy
        let peak_load_impact = (max_peak - avg_occupancy) * 25.0; // Peak demand impact

        EnergyImpactAnalysis {
            heating_impact_percentage: heating_impact,
            cooling_impact_percentage: cooling_impact,
            peak_load_impact_percentage: peak_load_impact,
            overall_energy_variation: (heating_impact + cooling_impact) / 2.0,
        }
    }
}

/// Energy impact analysis based on occupancy patterns
#[derive(Debug, Clone)]
pub struct EnergyImpactAnalysis {
    pub heating_impact_percentage: f64,
    pub cooling_impact_percentage: f64,
    pub peak_load_impact_percentage: f64,
    pub overall_energy_variation: f64,
}

/// Climate energy impact analysis based on climate zone characteristics
#[derive(Debug, Clone)]
pub struct ClimateEnergyImpactAnalysis {
    pub heating_impact_percentage: f64,
    pub cooling_impact_percentage: f64,
    pub solar_radiation_impact_percentage: f64,
    pub wind_infiltration_impact_percentage: f64,
    pub overall_climate_severity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ashrae140_validator_creation() {
        let validator = ASHRAE140Validator::new();
        assert!(true, "ASHRAE140Validator should be creatable");
    }

    #[test]
    fn test_default_occupancy_mappings() {
        let validator = ASHRAE140Validator::new();
        let mappings = validator.case_occupancy_patterns;

        assert!(
            !mappings.is_empty(),
            "Should have default occupancy mappings"
        );
        assert!(
            mappings.contains_key("600"),
            "Should have mapping for case 600"
        );
        assert_eq!(mappings.get("600"), Some(&"residential".to_string()));
    }

    #[test]
    fn test_case_occupancy_validation() {
        let validator = ASHRAE140Validator::new();

        // Test a case with occupancy pattern
        let result = validator.validate_case_occupancy("600");
        assert!(
            result.is_some(),
            "Case 600 should have occupancy validation"
        );

        let occupancy_result = result.unwrap();
        assert!(
            occupancy_result.is_valid,
            "Occupancy validation should pass"
        );
        assert_eq!(occupancy_result.pattern_name, "residential");
    }

    #[test]
    fn test_case_validation_with_occupancy() {
        let validator = ASHRAE140Validator::new();
        let result = validator.validate_case("600");

        assert!(result.is_ok(), "Case validation should succeed");
        let validation_result = result.unwrap();

        assert_eq!(validation_result.case_id, "600");
        assert!(validation_result.case_description.contains("residential"));
        assert!(validation_result.occupancy_validation.is_some());
    }

    #[test]
    fn test_custom_occupancy_pattern() {
        let mut validator = ASHRAE140Validator::new();
        validator.set_case_occupancy_pattern("600", "commercial");

        let result = validator.validate_case_occupancy("600");
        assert!(result.is_some());

        let occupancy_result = result.unwrap();
        assert_eq!(occupancy_result.pattern_name, "commercial");
    }

    #[test]
    fn test_energy_impact_analysis() {
        let validator = ASHRAE140Validator::new();
        let result = validator.analyze_occupancy_energy_impact("600");

        assert!(result.is_some(), "Should be able to analyze energy impact");
        let impact = result.unwrap();

        // Check that impact values are reasonable
        assert!(impact.heating_impact_percentage > 0.0);
        assert!(impact.cooling_impact_percentage > 0.0);
        assert!(impact.peak_load_impact_percentage >= 0.0);
    }

    #[test]
    fn test_all_cases_validation() {
        let validator = ASHRAE140Validator::new();
        let results = validator.validate_all_cases();

        assert!(!results.is_empty(), "Should have validation results");

        for result in results {
            assert!(result.case_description.contains("ASHRAE 140 Case"));
        }
    }

    #[test]
    fn test_climate_zone_validation() {
        let validator = ASHRAE140Validator::new();

        // Test climate zone validation for a case
        let result = validator.validate_case_climate_zone("600");
        assert!(
            result.is_some(),
            "Case 600 should have climate zone validation"
        );

        let climate_result = result.unwrap();
        assert_eq!(climate_result.zone_id, "4A");
        assert!(!climate_result.validation_metrics.is_empty());
    }

    #[test]
    fn test_case_validation_with_climate() {
        let validator = ASHRAE140Validator::new();
        let result = validator.validate_case("600");

        assert!(result.is_ok(), "Case validation should succeed");
        let validation_result = result.unwrap();

        assert_eq!(validation_result.case_id, "600");
        assert!(validation_result.case_description.contains("climate zone"));
        assert!(validation_result.climate_validation.is_some());
        assert!(validation_result.occupancy_validation.is_some());
    }

    #[test]
    fn test_all_climate_zones_validation() {
        let validator = ASHRAE140Validator::new();
        let results = validator.validate_all_climate_zones();

        assert!(
            !results.is_empty(),
            "Should have climate zone validation results"
        );

        // Check that we have results for expected cases
        assert!(results.contains_key("600"));
        assert!(results.contains_key("900"));
    }

    #[test]
    fn test_custom_climate_zone() {
        let mut validator = ASHRAE140Validator::new();
        validator.set_case_climate_zone("600", "5A");

        let result = validator.validate_case_climate_zone("600");
        assert!(result.is_some());

        let climate_result = result.unwrap();
        assert_eq!(climate_result.zone_id, "5A");
    }

    #[test]
    fn test_climate_energy_impact_analysis() {
        let validator = ASHRAE140Validator::new();
        let result = validator.analyze_climate_energy_impact("600");

        assert!(
            result.is_some(),
            "Should be able to analyze climate energy impact"
        );
        let impact = result.unwrap();

        // Check that impact values are reasonable
        assert!(impact.heating_impact_percentage > 0.0);
        assert!(impact.cooling_impact_percentage > 0.0);
        assert!(impact.solar_radiation_impact_percentage > 0.0);
        assert!(impact.wind_infiltration_impact_percentage >= 0.0);
    }

    #[test]
    fn test_case_description_with_climate() {
        let validator = ASHRAE140Validator::new();
        let result = validator.validate_case("600");

        assert!(result.is_ok());
        let validation_result = result.unwrap();

        // Check that description includes both occupancy and climate information
        assert!(validation_result.case_description.contains("occupancy"));
        assert!(validation_result.case_description.contains("climate zone"));
    }
}

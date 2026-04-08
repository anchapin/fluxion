//! High-mass thermal validation types.
//!
//! This module provides data structures for ASHRAE 140-2017 Addendum B
//! high-mass validation including construction types, validation cases,
//! and acceptance criteria metrics.

use std::fmt;

/// Construction type classification based on thermal mass.
///
/// Corresponds to ASHRAE 140-2017 Addendum B construction categories.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ConstructionType {
    /// Light construction: < 50 kg/m²
    Light,
    /// Medium construction: 50-150 kg/m²
    Medium,
    /// Heavy construction: 150-300 kg/m²
    Heavy,
    /// Very heavy construction: > 300 kg/m²
    #[default]
    VeryHeavy,
}

impl ConstructionType {
    /// Returns the typical mass per area for this construction type in kg/m².
    pub fn typical_mass_per_area(&self) -> f64 {
        match self {
            ConstructionType::Light => 25.0,
            ConstructionType::Medium => 100.0,
            ConstructionType::Heavy => 225.0,
            ConstructionType::VeryHeavy => 400.0,
        }
    }

    /// Returns the typical specific heat capacity in J/kg·K.
    pub fn typical_specific_heat(&self) -> f64 {
        match self {
            ConstructionType::Light => 900.0,
            ConstructionType::Medium => 840.0,
            ConstructionType::Heavy => 840.0,
            ConstructionType::VeryHeavy => 840.0,
        }
    }
}

impl fmt::Display for ConstructionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstructionType::Light => write!(f, "Light"),
            ConstructionType::Medium => write!(f, "Medium"),
            ConstructionType::Heavy => write!(f, "Heavy"),
            ConstructionType::VeryHeavy => write!(f, "VeryHeavy"),
        }
    }
}

/// High-mass validation case following ASHRAE 140-2017 Addendum B.
///
/// Contains all data needed to validate a high-mass thermal simulation
/// including reference load profiles and acceptance criteria.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HighMassCase {
    /// Construction type classification
    pub construction: ConstructionType,
    /// Thermal mass per unit area (kg/m²)
    pub mass_per_area: f64,
    /// Specific heat capacity (J/kg·K)
    pub specific_heat: f64,
    /// Reference hourly loading profile (W/m²)
    pub reference_loads: Vec<f64>,
    /// Acceptance tolerance percentage (%)
    pub tolerance: f64,
}

impl HighMassCase {
    /// Creates a new high-mass validation case.
    pub fn new(
        construction: ConstructionType,
        mass_per_area: f64,
        specific_heat: f64,
        reference_loads: Vec<f64>,
        tolerance: f64,
    ) -> Self {
        Self {
            construction,
            mass_per_area,
            specific_heat,
            reference_loads,
            tolerance,
        }
    }

    /// Creates a case using typical values for the given construction type.
    pub fn with_typical_values(construction: ConstructionType, tolerance: f64) -> Self {
        Self {
            construction: construction.clone(),
            mass_per_area: construction.typical_mass_per_area(),
            specific_heat: construction.typical_specific_heat(),
            reference_loads: Vec::new(),
            tolerance,
        }
    }

    /// Returns the number of hourly data points.
    pub fn data_points(&self) -> usize {
        self.reference_loads.len()
    }

    /// Checks if the case has valid reference data.
    pub fn has_reference_data(&self) -> bool {
        !self.reference_loads.is_empty()
    }
}

/// Validation result containing ASHRAE 140 acceptance criteria metrics.
///
/// Implements NMBE (Normalized Mean Bias Error) and CV(RMSE) (Coefficient
/// of Variation of Root Mean Square Error) as defined in ASHRAE 140-2017.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValidationResult {
    /// Normalized Mean Bias Error (%)
    pub nmbe: f64,
    /// Coefficient of Variation of RMSE (%)
    pub cv_rmse: f64,
    /// Whether the case passes acceptance criteria
    pub passes: bool,
}

impl ValidationResult {
    /// Creates a new validation result.
    pub fn new(nmbe: f64, cv_rmse: f64, tolerance: f64) -> Self {
        let passes = nmbe.abs() <= tolerance && cv_rmse <= tolerance;
        Self {
            nmbe,
            cv_rmse,
            passes,
        }
    }

    /// Creates a passing result.
    pub fn passing(nmbe: f64, cv_rmse: f64) -> Self {
        Self {
            nmbe,
            cv_rmse,
            passes: true,
        }
    }

    /// Creates a failing result.
    pub fn failing(nmbe: f64, cv_rmse: f64) -> Self {
        Self {
            nmbe,
            cv_rmse,
            passes: false,
        }
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NMBE: {:.2}%, CV(RMSE): {:.2}%, {}",
            self.nmbe,
            self.cv_rmse,
            if self.passes { "PASS" } else { "FAIL" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_type_display() {
        assert_eq!(format!("{}", ConstructionType::Light), "Light");
        assert_eq!(format!("{}", ConstructionType::Medium), "Medium");
        assert_eq!(format!("{}", ConstructionType::Heavy), "Heavy");
        assert_eq!(format!("{}", ConstructionType::VeryHeavy), "VeryHeavy");
    }

    #[test]
    fn test_construction_typical_values() {
        let light = ConstructionType::Light;
        assert_eq!(light.typical_mass_per_area(), 25.0);
        assert_eq!(light.typical_specific_heat(), 900.0);

        let heavy = ConstructionType::Heavy;
        assert_eq!(heavy.typical_mass_per_area(), 225.0);
    }

    #[test]
    fn test_high_mass_case_creation() {
        let case = HighMassCase::new(
            ConstructionType::Medium,
            100.0,
            840.0,
            vec![50.0, 55.0, 60.0],
            10.0,
        );
        assert_eq!(case.data_points(), 3);
        assert!(case.has_reference_data());
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult::new(5.0, 8.0, 10.0);
        assert!(result.passes);

        let fail_result = ValidationResult::new(15.0, 8.0, 10.0);
        assert!(!fail_result.passes);
    }
}

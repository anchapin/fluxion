//! Configuration Validation Module
//!
//! This module provides comprehensive validation for building assemblies, constants,
//! and thermal model parameters with structured JSON error output for CI integration.
//!
//! # Validation Philosophy
//!
//! - **Fail-fast:** Reject invalid configurations before simulation starts
//! - **Actionable errors:** Provide clear error messages with suggestions
//! - **Structured output:** JSON format for tooling and CI integration
//! - **Physical correctness:** Validate against ASHRAE/ISO standards

use crate::physics::constants::thermal::ashrae_140 as ashrae_140_thermal;
use crate::physics::constants::thermal::iso_13790;
use crate::sim::assembly::{BuildingAssembly, ConcreteMaterial, InsulationMaterial, MaterialLayer};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Configuration validation errors
///
/// These errors provide detailed information about what went wrong and where,
/// making it easy for users to fix configuration issues.
#[derive(Debug, Error)]
pub enum ConfigValidationError {
    /// Invalid value at specific path
    #[error("Invalid value at {path}: {field} = {value}")]
    InvalidValue {
        /// Path to the configuration file (e.g., "config.json:42")
        path: String,
        /// Field name that failed validation
        field: String,
        /// Invalid value that was provided
        value: serde_json::Value,
    },

    /// Missing required field
    #[error("Missing required field at {path}: {field}")]
    MissingField {
        /// Path to the configuration file
        path: String,
        /// Name of the missing field
        field: String,
    },

    /// General validation failure
    #[error("Validation failed at {path}: {message}")]
    ValidationError {
        /// Path to the configuration file
        path: String,
        /// Detailed error message
        message: String,
    },

    /// Value outside expected range
    #[error("Out of range at {path}: {field} = {value} (expected: {min} - {max})")]
    OutOfRange {
        /// Path to the configuration file
        path: String,
        /// Field name that failed validation
        field: String,
        /// Invalid value that was provided
        value: serde_json::Value,
        /// Minimum allowed value
        min: serde_json::Value,
        /// Maximum allowed value
        max: serde_json::Value,
    },

    /// Physical constraint violation
    #[error("Physical constraint violation at {path}: {message}")]
    PhysicalConstraintViolation {
        /// Path to the configuration file
        path: String,
        /// Detailed error message about the constraint violation
        message: String,
    },
}

/// Structured validation error for JSON output
///
/// This structure is designed for CI integration and automated processing,
/// providing all the information needed to locate and fix validation issues.
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationError {
    /// Path to the configuration file (e.g., "config.json:42")
    pub path: String,
    /// Field name that failed validation (e.g., "layer.thickness")
    pub field: String,
    /// Invalid value that was provided
    pub value: serde_json::Value,
    /// Human-readable error message
    pub message: String,
    /// Optional suggestion for fixing the error
    pub suggestion: Option<String>,
}

/// Complete validation result with errors and warnings
///
/// This structure provides the full validation outcome including both errors
/// (which prevent simulation) and warnings (which indicate potential issues).
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigValidationResult {
    /// Overall validation status: "passed" or "failed"
    pub validation: String,
    /// List of validation errors that must be fixed
    pub errors: Vec<ValidationError>,
    /// List of warnings that indicate potential issues
    pub warnings: Vec<ValidationError>,
}

impl ConfigValidationResult {
    /// Create a new successful validation result
    pub fn passed() -> Self {
        Self {
            validation: "passed".to_string(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a new failed validation result
    pub fn failed(errors: Vec<ValidationError>, warnings: Vec<ValidationError>) -> Self {
        Self {
            validation: "failed".to_string(),
            errors,
            warnings,
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.validation == "passed"
    }
}

/// Validate building assembly material properties
///
/// Checks all material layer properties for physical validity:
/// - Thickness must be positive
/// - Thermal conductivity must be positive
/// - Density must be positive
/// - Specific heat must be positive
/// - Emissivity must be in range [0, 1]
/// - Absorptance must be in range [0, 1]
/// - Thermal mass must be positive
///
/// # Arguments
/// * `assembly` - Building assembly to validate
/// * `path` - Path to configuration file for error reporting
///
/// # Returns
/// ConfigValidationResult with errors (blocking) and warnings (non-blocking)
///
/// # Examples
/// ```
/// use fluxion::validation::config::validate_assembly;
/// use fluxion::sim::assembly::{BuildingAssembly, AssemblyBuilder};
///
/// let assembly = AssemblyBuilder::new("test".to_string())
///     .add_layer(Box::new(ConcreteMaterial::new(0.1)))
///     .build()
///     .unwrap();
///
/// let result = validate_assembly(&assembly, "config.json");
/// assert!(result.is_valid());
/// ```
pub fn validate_assembly(assembly: &BuildingAssembly, path: &str) -> ConfigValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Validate each layer
    for (idx, layer) in assembly.layers.iter().enumerate() {
        let field_path = format!("{}.layers[{}]", path, idx);

        // Validate thickness
        if layer.thickness() <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "thickness".to_string(),
                value: serde_json::json!(layer.thickness()),
                message: "Thickness must be positive".to_string(),
                suggestion: Some("Use thickness > 0.0 meters".to_string()),
            });
        }

        // Validate conductivity
        if layer.conductivity() <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "conductivity".to_string(),
                value: serde_json::json!(layer.conductivity()),
                message: "Thermal conductivity must be positive".to_string(),
                suggestion: Some(
                    "Use conductivity > 0.0 W/mK (typical: 0.04-1.4 for building materials)"
                        .to_string(),
                ),
            });
        }

        // Validate density
        if layer.density() <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "density".to_string(),
                value: serde_json::json!(layer.density()),
                message: "Density must be positive".to_string(),
                suggestion: Some(
                    "Use density > 0.0 kg/m³ (typical: 50-2400 for building materials)".to_string(),
                ),
            });
        }

        // Validate specific heat
        if layer.specific_heat() <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "specific_heat".to_string(),
                value: serde_json::json!(layer.specific_heat()),
                message: "Specific heat must be positive".to_string(),
                suggestion: Some(
                    "Use specific_heat > 0.0 J/kgK (typical: 840 for most building materials)"
                        .to_string(),
                ),
            });
        }

        // Validate emissivity range [0, 1]
        if layer.emissivity() < 0.0 || layer.emissivity() > 1.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "emissivity".to_string(),
                value: serde_json::json!(layer.emissivity()),
                message: "Emissivity must be in range [0, 1]".to_string(),
                suggestion: Some(
                    "Use emissivity between 0.0 and 1.0 (typical: 0.9 for building surfaces)"
                        .to_string(),
                ),
            });
        }

        // Validate absorptance range [0, 1]
        if layer.absorptance() < 0.0 || layer.absorptance() > 1.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "absorptance".to_string(),
                value: serde_json::json!(layer.absorptance()),
                message: "Solar absorptance must be in range [0, 1]".to_string(),
                suggestion: Some(
                    "Use absorptance between 0.0 and 1.0 (typical: 0.3-0.9 for building materials)"
                        .to_string(),
                ),
            });
        }

        // Warning: Low emissivity (unusual for building materials)
        if layer.emissivity() < 0.8 && layer.emissivity() >= 0.0 {
            warnings.push(ValidationError {
                path: field_path,
                field: "emissivity".to_string(),
                value: serde_json::json!(layer.emissivity()),
                message: "Low emissivity (unusual for building materials)".to_string(),
                suggestion: Some("Typical building materials have emissivity 0.9. Low emissivity (<0.8) is unusual.".to_string()),
            });
        }
    }

    // Validate physical constraints
    let thermal_mass = assembly.thermal_mass();
    if thermal_mass <= 0.0 {
        errors.push(ValidationError {
            path: path.to_string(),
            field: "thermal_mass".to_string(),
            value: serde_json::json!(thermal_mass),
            message: "Thermal mass must be positive".to_string(),
            suggestion: Some("Check material density, specific heat, and thickness".to_string()),
        });
    }

    ConfigValidationResult {
        validation: if errors.is_empty() {
            "passed".to_string()
        } else {
            "failed".to_string()
        },
        errors,
        warnings,
    }
}

/// Validate physical constants for correctness and range checking
///
/// Checks all physical constants against expected ranges and documented values:
/// - ASHRAE 140 film coefficients (interior and exterior)
/// - Solar constant range (1300-1400 W/m² for Earth's mean distance)
/// - ISO 13790 Annex C thermal mass thresholds (positive values)
///
/// # Arguments
/// * `path` - Path to configuration file for error reporting
///
/// # Returns
/// ConfigValidationResult with errors (blocking) and warnings (non-blocking)
///
/// # Examples
/// ```
/// use fluxion::validation::config::validate_constants;
///
/// let result = validate_constants("config.json");
/// assert!(result.is_valid());
/// ```
pub fn validate_constants(path: &str) -> ConfigValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Validate ASHRAE 140 constants
    // Interior film coefficient
    let h_int = ashrae_140_thermal::INTERIOR_FILM_COEFF;
    if h_int <= 0.0 {
        errors.push(ValidationError {
            path: format!("{}.ashrae_140", path),
            field: "INTERIOR_FILM_COEFF".to_string(),
            value: serde_json::json!(h_int),
            message: "Interior film coefficient must be positive".to_string(),
            suggestion: Some("Typical range: 5-10 W/m²K".to_string()),
        });
    }

    // Exterior film coefficient
    let h_ext = ashrae_140_thermal::EXTERIOR_FILM_COEFF;
    if h_ext <= 0.0 {
        errors.push(ValidationError {
            path: format!("{}.ashrae_140", path),
            field: "EXTERIOR_FILM_COEFF".to_string(),
            value: serde_json::json!(h_ext),
            message: "Exterior film coefficient must be positive".to_string(),
            suggestion: Some("Typical range: 15-25 W/m²K".to_string()),
        });
    }

    // Solar constant
    let solar_constant = crate::physics::constants::solar::ashrae_140::SOLAR_CONSTANT;
    if solar_constant < 1300.0 || solar_constant > 1400.0 {
        warnings.push(ValidationError {
            path: format!("{}.solar", path),
            field: "SOLAR_CONSTANT".to_string(),
            value: serde_json::json!(solar_constant),
            message: "Solar constant outside typical range".to_string(),
            suggestion: Some("IPCC AR6 (2021) reports 1361.0 ±0.5 W/m²".to_string()),
        });
    }

    // Validate ISO 13790 Annex C thresholds
    let thresholds = [
        iso_13790::THERMAL_MASS_VERY_LIGHT,
        iso_13790::THERMAL_MASS_LIGHT,
        iso_13790::THERMAL_MASS_LIGHT_UPPER,
        iso_13790::THERMAL_MASS_MEDIUM,
        iso_13790::THERMAL_MASS_MEDIUM_UPPER,
        iso_13790::THERMAL_MASS_HEAVY,
        iso_13790::THERMAL_MASS_HEAVY_UPPER,
        iso_13790::THERMAL_MASS_VERY_HEAVY,
    ];

    for (idx, threshold) in thresholds.iter().enumerate() {
        if *threshold <= 0.0 {
            errors.push(ValidationError {
                path: format!("{}.iso_13790[{}]", path, idx),
                field: "thermal_mass_threshold".to_string(),
                value: serde_json::json!(threshold),
                message: "Thermal mass threshold must be positive".to_string(),
                suggestion: Some(
                    "ISO 13790 Annex C defines thresholds in kJ/m²K (50, 150, 260, 370)"
                        .to_string(),
                ),
            });
        }
    }

    ConfigValidationResult {
        validation: if errors.is_empty() {
            "passed".to_string()
        } else {
            "failed".to_string()
        },
        errors,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_passed() {
        let result = ConfigValidationResult::passed();
        assert!(result.is_valid());
        assert_eq!(result.validation, "passed");
        assert!(result.errors.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_validation_result_failed() {
        let error = ValidationError {
            path: "config.json:42".to_string(),
            field: "thickness".to_string(),
            value: serde_json::json!(-0.05),
            message: "Thickness must be positive".to_string(),
            suggestion: Some("Use thickness > 0.0 meters".to_string()),
        };

        let result = ConfigValidationResult::failed(vec![error], vec![]);
        assert!(!result.is_valid());
        assert_eq!(result.validation, "failed");
        assert_eq!(result.errors.len(), 1);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_validation_error_serialization() {
        let error = ValidationError {
            path: "config.json:42".to_string(),
            field: "thickness".to_string(),
            value: serde_json::json!(-0.05),
            message: "Thickness must be positive".to_string(),
            suggestion: Some("Use thickness > 0.0 meters".to_string()),
        };

        let json = serde_json::to_string(&error).unwrap();
        let parsed: ValidationError = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.path, "config.json:42");
        assert_eq!(parsed.field, "thickness");
        assert_eq!(parsed.value, serde_json::json!(-0.05));
        assert_eq!(parsed.message, "Thickness must be positive");
        assert_eq!(
            parsed.suggestion,
            Some("Use thickness > 0.0 meters".to_string())
        );
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = ConfigValidationResult::passed();
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ConfigValidationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.validation, "passed");
        assert!(parsed.errors.is_empty());
        assert!(parsed.warnings.is_empty());
    }
}

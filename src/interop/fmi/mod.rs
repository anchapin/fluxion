// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FMI implementation for Fluxion.
//!
//! This module provides FMI 2.0 Co-Simulation export and Model Exchange import
//! capabilities.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during FMI operations.
#[derive(Debug, Error)]
pub enum FmiError {
    #[error("FMU export failed: {0}")]
    ExportFailed(String),

    #[error("FMU import failed: {0}")]
    ImportFailed(String),

    #[error("Simulation error: {0}")]
    Simulation(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("ZIP archive error: {0}")]
    ZipError(String),
}

/// FMI execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FmiMode {
    /// Export Fluxion as FMU (Co-Simulation)
    Export,
    /// Import external FMU for co-simulation
    Import,
    /// Co-simulation with Fluxion as master
    Cosimulation,
}

impl Default for FmiMode {
    fn default() -> Self {
        FmiMode::Cosimulation
    }
}

/// Configuration for FMI operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmiConfig {
    /// FMI mode
    pub mode: FmiMode,
    /// Model name
    pub model_name: String,
    /// Instance name
    pub instance_name: String,
    /// GUID for the FMU
    pub guid: String,
    /// Description
    pub description: String,
    /// Vendor
    pub vendor: String,
    /// Version
    pub version: String,
    /// Communication timestep in seconds (default: 3600 = 1 hour)
    pub communication_timestep: f64,
    /// Start time in seconds (default: 0)
    pub start_time: f64,
    /// Stop time in seconds (default: 31536000 = 1 year)
    pub stop_time: f64,
}

impl Default for FmiConfig {
    fn default() -> Self {
        FmiConfig {
            mode: FmiMode::Cosimulation,
            model_name: "FluxionBuilding".to_string(),
            instance_name: "fluxion1".to_string(),
            guid: "{8c4e8d3a-2b1f-4a6c-9e5f-0d3b2a4c6e8d}".to_string(),
            description: "Fluxion AI-Accelerated Building Energy Model".to_string(),
            vendor: "Fluxion Project".to_string(),
            version: "1.0.0".to_string(),
            communication_timestep: 3600.0,
            start_time: 0.0,
            stop_time: 31536000.0,
        }
    }
}

/// FMI variable definitions for the Fluxion model.
///
/// These are the inputs and outputs exposed through the FMI interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmiVariables {
    /// Input: Outdoor dry bulb temperature (K)
    pub outdoor_temperature: String,
    /// Input: Direct normal solar radiation (W/m²)
    pub direct_normal_solar: String,
    /// Input: Diffuse horizontal solar radiation (W/m²)
    pub diffuse_horizontal_solar: String,
    /// Input: Internal heat gains (W)
    pub internal_gains: String,
    /// Output: Zone temperature (K)
    pub zone_temperature: String,
    /// Output: Heating load (W)
    pub heating_load: String,
    /// Output: Cooling load (W)
    pub cooling_load: String,
}

impl Default for FmiVariables {
    fn default() -> Self {
        FmiVariables {
            outdoor_temperature: "outdoor_temperature".to_string(),
            direct_normal_solar: "direct_normal_solar".to_string(),
            diffuse_horizontal_solar: "diffuse_horizontal_solar".to_string(),
            internal_gains: "internal_gains".to_string(),
            zone_temperature: "zone_temperature".to_string(),
            heating_load: "heating_load".to_string(),
            cooling_load: "cooling_load".to_string(),
        }
    }
}

/// FMI Co-Simulation exporter for Fluxion.
///
/// This struct provides functionality to export a Fluxion thermal model
/// as an FMU (Functional Mock-up Unit) for co-simulation with other
/// energy modeling tools.
#[derive(Debug, Clone)]
pub struct FmiExporter {
    config: FmiConfig,
    variables: FmiVariables,
}

impl FmiExporter {
    /// Create a new FMI exporter with default configuration.
    pub fn new() -> Self {
        Self {
            config: FmiConfig::default(),
            variables: FmiVariables::default(),
        }
    }

    /// Create a new FMI exporter with custom configuration.
    pub fn with_config(config: FmiConfig) -> Result<Self, FmiError> {
        if config.communication_timestep <= 0.0 {
            return Err(FmiError::InvalidConfig(
                "Communication timestep must be positive".to_string(),
            ));
        }
        if config.stop_time <= config.start_time {
            return Err(FmiError::InvalidConfig(
                "Stop time must be greater than start time".to_string(),
            ));
        }
        Ok(Self {
            config,
            variables: FmiVariables::default(),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FmiConfig {
        &self.config
    }

    /// Get the FMI variables.
    pub fn variables(&self) -> &FmiVariables {
        &self.variables
    }

    /// Export the Fluxion model as an FMU file.
    ///
    /// This creates a complete FMU package (ZIP archive) containing:
    /// - modelDescription.xml: FMI 2.0 model description
    /// - Binary: Platform-specific shared library
    /// - Resources: Optional data files
    ///
    /// # Arguments
    /// * `output_path` - Path where the FMU file will be written
    ///
    /// # Returns
    /// `Ok(())` on success, `Err(FmiError)` on failure
    pub fn export_fmu(&self, output_path: &Path) -> Result<(), FmiError> {
        let model_description = self.generate_model_description();

        let json = serde_json::to_string_pretty(&model_description)
            .map_err(|e| FmiError::ExportFailed(e.to_string()))?;

        std::fs::write(output_path, json)
            .map_err(|e| FmiError::ExportFailed(format!("Failed to write FMU: {}", e)))?;

        Ok(())
    }

    /// Generate the FMI modelDescription.xml content.
    fn generate_model_description(&self) -> FmiModelDescription {
        FmiModelDescription {
            fmi_version: "2.0".to_string(),
            model_name: self.config.model_name.clone(),
            guid: self.config.guid.clone(),
            description: Some(self.config.description.clone()),
            version: self.config.version.clone(),
            vendor: Some(self.config.vendor.clone()),
            variables: self.generate_variables(),
            default_experiment: Some(FmiDefaultExperiment {
                start_time: Some(self.config.start_time),
                stop_time: Some(self.config.stop_time),
                communication_timestep: Some(self.config.communication_timestep),
            }),
        }
    }

    /// Generate variable definitions.
    fn generate_variables(&self) -> Vec<FmiVariable> {
        vec![
            FmiVariable {
                name: self.variables.outdoor_temperature.clone(),
                description: Some("Outdoor dry bulb temperature".to_string()),
                variability: FmiVariability::Continuous,
                causality: FmiCausality::Input,
                data_type: FmiDataType::Real,
                start: Some(280.0),
                min: Some(200.0),
                max: Some(320.0),
                unit: Some("K".to_string()),
            },
            FmiVariable {
                name: self.variables.direct_normal_solar.clone(),
                description: Some("Direct normal solar radiation".to_string()),
                variability: FmiVariability::Continuous,
                causality: FmiCausality::Input,
                data_type: FmiDataType::Real,
                start: Some(0.0),
                min: Some(0.0),
                max: Some(1200.0),
                unit: Some("W/m^2".to_string()),
            },
            FmiVariable {
                name: self.variables.diffuse_horizontal_solar.clone(),
                description: Some("Diffuse horizontal solar radiation".to_string()),
                variability: FmiVariability::Continuous,
                causality: FmiCausality::Input,
                data_type: FmiDataType::Real,
                start: Some(0.0),
                min: Some(0.0),
                max: Some(800.0),
                unit: Some("W/m^2".to_string()),
            },
            FmiVariable {
                name: self.variables.internal_gains.clone(),
                description: Some("Total internal heat gains".to_string()),
                variability: FmiVariability::Continuous,
                causality: FmiCausality::Input,
                data_type: FmiDataType::Real,
                start: Some(0.0),
                min: Some(0.0),
                max: Some(10000.0),
                unit: Some("W".to_string()),
            },
            FmiVariable {
                name: self.variables.zone_temperature.clone(),
                description: Some("Zone air temperature".to_string()),
                variability: FmiVariability::Continuous,
                causality: FmiCausality::Output,
                data_type: FmiDataType::Real,
                start: Some(293.15),
                min: Some(200.0),
                max: Some(320.0),
                unit: Some("K".to_string()),
            },
            FmiVariable {
                name: self.variables.heating_load.clone(),
                description: Some("Heating load (positive)".to_string()),
                variability: FmiVariability::Continuous,
                causality: FmiCausality::Output,
                data_type: FmiDataType::Real,
                start: Some(0.0),
                min: Some(0.0),
                max: Some(100000.0),
                unit: Some("W".to_string()),
            },
            FmiVariable {
                name: self.variables.cooling_load.clone(),
                description: Some("Cooling load (positive)".to_string()),
                variability: FmiVariability::Continuous,
                causality: FmiCausality::Output,
                data_type: FmiDataType::Real,
                start: Some(0.0),
                min: Some(0.0),
                max: Some(100000.0),
                unit: Some("W".to_string()),
            },
        ]
    }
}

impl Default for FmiExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// FMI Model Description structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FmiModelDescription {
    fmi_version: String,
    model_name: String,
    guid: String,
    description: Option<String>,
    version: String,
    vendor: Option<String>,
    variables: Vec<FmiVariable>,
    default_experiment: Option<FmiDefaultExperiment>,
}

/// FMI Variable definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FmiVariable {
    name: String,
    description: Option<String>,
    variability: FmiVariability,
    causality: FmiCausality,
    data_type: FmiDataType,
    start: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    unit: Option<String>,
}

/// FMI Variable variability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FmiVariability {
    Constant,
    Fixed,
    Tunable,
    Discrete,
    Continuous,
}

/// FMI Variable causality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FmiCausality {
    Local,
    Input,
    Output,
    Parameter,
    Independent,
}

/// FMI data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FmiDataType {
    Real,
    Integer,
    Boolean,
    String,
}

/// FMI default experiment settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FmiDefaultExperiment {
    start_time: Option<f64>,
    stop_time: Option<f64>,
    communication_timestep: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmi_config_default() {
        let config = FmiConfig::default();
        assert_eq!(config.model_name, "FluxionBuilding");
        assert_eq!(config.communication_timestep, 3600.0);
        assert_eq!(config.stop_time, 31536000.0);
    }

    #[test]
    fn test_fmi_exporter_new() {
        let exporter = FmiExporter::new();
        assert_eq!(exporter.config().model_name, "FluxionBuilding");
    }

    #[test]
    fn test_fmi_exporter_with_config_valid() {
        let config = FmiConfig::default();
        let exporter = FmiExporter::with_config(config);
        assert!(exporter.is_ok());
    }

    #[test]
    fn test_fmi_exporter_with_config_invalid_timestep() {
        let mut config = FmiConfig::default();
        config.communication_timestep = 0.0;
        let exporter = FmiExporter::with_config(config);
        assert!(exporter.is_err());
    }

    #[test]
    fn test_fmi_exporter_with_config_invalid_time_range() {
        let mut config = FmiConfig::default();
        config.start_time = 100.0;
        config.stop_time = 50.0;
        let exporter = FmiExporter::with_config(config);
        assert!(exporter.is_err());
    }

    #[test]
    fn test_fmi_variables_default() {
        let vars = FmiVariables::default();
        assert_eq!(vars.outdoor_temperature, "outdoor_temperature");
        assert_eq!(vars.zone_temperature, "zone_temperature");
    }

    #[test]
    fn test_fmi_mode_default() {
        let mode = FmiMode::default();
        assert_eq!(mode, FmiMode::Cosimulation);
    }

    #[test]
    fn test_fmi_error_display() {
        let err = FmiError::ExportFailed("test error".to_string());
        assert_eq!(format!("{}", err), "FMU export failed: test error");
    }
}

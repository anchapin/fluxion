//! Thermal mass diagnostics module for high-mass building analysis.
//!
//! This module provides comprehensive thermal mass analysis capabilities
//! for building energy simulations, implementing ISO 13790 Annex C methods
//! for calculating effective capacitance, time constant, and damping factor.

use crate::physics::constants::thermal::iso_13790::annex_c::{
    calculate_effective_thermal_mass, THERMAL_MASS_HEAVY, THERMAL_MASS_MEDIUM,
    THERMAL_MASS_VERY_HEAVY, THERMAL_MASS_VERY_LIGHT,
};
use crate::sim::construction::ConstructionLayer;
use log::debug;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

/// Thermal mass properties calculated from building construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalMassProperties {
    /// Effective thermal capacitance in kJ/m²K
    pub effective_capacitance: f64,
    /// Time constant in hours
    pub time_constant: f64,
    /// Damping factor (unitless, 0.0-1.0)
    pub damping_factor: f64,
    /// Thermal mass classification per ISO 13790 Annex C
    pub classification: String,
}

impl Default for ThermalMassProperties {
    fn default() -> Self {
        Self {
            effective_capacitance: 150.0, // Medium-weight default
            time_constant: 6.0,
            damping_factor: 0.5,
            classification: "Medium".to_string(),
        }
    }
}

/// Thermal mass diagnostics analyzer.
///
/// Calculates key thermal mass metrics for building energy performance analysis.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ThermalMassDiagnostics {
    /// Simulation timestep in seconds (default: 3600 = 1 hour)
    pub timestep_seconds: u32,
    /// Total heat loss coefficient in W/m²K
    pub total_heat_loss_coefficient: f64,
    /// Building construction layers for analysis
    pub construction_layers: Option<Vec<ConstructionLayer>>,
}

impl ThermalMassDiagnostics {
    /// Create a new ThermalMassDiagnostics analyzer.
    ///
    /// # Arguments
    /// * `timestep_seconds` - Simulation timestep in seconds
    /// * `total_heat_loss_coefficient` - Building heat loss coefficient in W/m²K
    ///
    /// # Returns
    /// A new ThermalMassDiagnostics instance
    pub fn new(timestep_seconds: u32, total_heat_loss_coefficient: f64) -> Self {
        Self {
            timestep_seconds,
            total_heat_loss_coefficient,
            construction_layers: None,
        }
    }

    /// Create analyzer with construction layers.
    ///
    /// # Arguments
    /// * `construction_layers` - Construction layers to analyze
    /// * `total_heat_loss_coefficient` - Building heat loss coefficient in W/m²K
    ///
    /// # Returns
    /// A new ThermalMassDiagnostics instance with construction layers
    pub fn with_construction_layers(
        construction_layers: Vec<ConstructionLayer>,
        total_heat_loss_coefficient: f64,
    ) -> Self {
        Self {
            timestep_seconds: 3600, // Default 1-hour timestep
            total_heat_loss_coefficient,
            construction_layers: Some(construction_layers),
        }
    }

    /// Calculate effective thermal capacitance using ISO 13790 Annex C formula.
    ///
    /// # Formula
    /// ```text
    /// C_eff = Σ(density × specific_heat × thickness) / 1000
    ///
    /// Where:
    /// - density: kg/m³
    /// - specific_heat: J/kgK
    /// - thickness: m
    /// - Division by 1000 converts J/m²K to kJ/m²K
    /// ```
    ///
    /// # Returns
    /// Effective thermal capacitance in kJ/m²K
    pub fn calculate_effective_capacitance(&self) -> f64 {
        // If construction layers are available, calculate from actual layers
        if let Some(layers) = &self.construction_layers {
            let layer_tuples = layers
                .iter()
                .map(|layer| (layer.thickness, layer.density, layer.specific_heat))
                .collect::<Vec<_>>();

            calculate_effective_thermal_mass(&layer_tuples)
        } else {
            // Fallback: use medium-weight default
            150.0
        }
    }

    /// Calculate time constant in hours.
    ///
    /// # Formula
    /// ```text
    /// τ = C_eff / H
    ///
    /// Where:
    /// - C_eff: Effective thermal capacitance in kJ/m²K
    /// - H: Total heat loss coefficient in W/m²K
    /// - Result converted from seconds to hours
    /// ```
    ///
    /// # Returns
    /// Time constant in hours
    pub fn calculate_time_constant(&self) -> f64 {
        let effective_capacitance = self.calculate_effective_capacitance();

        // Convert kJ/m²K to J/m²K for unit consistency
        let capacitance_joules = effective_capacitance * 1000.0;

        // Calculate time constant in seconds, then convert to hours
        let time_constant_seconds = capacitance_joules / self.total_heat_loss_coefficient;
        time_constant_seconds / 3600.0
    }

    /// Calculate damping factor for the given timestep.
    ///
    /// # Formula
    /// ```text
    /// damping_factor = exp(-Δt / τ)
    ///
    /// Where:
    /// - Δt: Timestep in hours
    /// - τ: Time constant in hours
    /// ```
    ///
    /// # Returns
    /// Damping factor (unitless, 0.0-1.0)
    pub fn calculate_damping_factor(&self) -> f64 {
        let time_constant_hours = self.calculate_time_constant();
        let timestep_hours = self.timestep_seconds as f64 / 3600.0;

        (-timestep_hours / time_constant_hours).exp()
    }

    /// Classify thermal mass according to ISO 13790 Annex C thresholds.
    ///
    /// # Returns
    /// Thermal mass classification string
    pub fn classify_thermal_mass(&self) -> String {
        let capacitance = self.calculate_effective_capacitance();

        if capacitance < THERMAL_MASS_VERY_LIGHT {
            "VeryLight".to_string()
        } else if capacitance < THERMAL_MASS_MEDIUM {
            "Light".to_string()
        } else if capacitance < THERMAL_MASS_HEAVY {
            "Medium".to_string()
        } else if capacitance < THERMAL_MASS_VERY_HEAVY {
            "Heavy".to_string()
        } else {
            "VeryHeavy".to_string()
        }
    }

    /// Analyze building thermal mass and return comprehensive report.
    ///
    /// # Returns
    /// ThermalMassReport containing all calculated metrics
    pub fn analyze(&self) -> ThermalMassReport {
        let effective_capacitance = self.calculate_effective_capacitance();
        let time_constant = self.calculate_time_constant();
        let damping_factor = self.calculate_damping_factor();
        let classification = self.classify_thermal_mass();

        // Log diagnostic results
        debug!(
            "Thermal Mass Diagnostics - Capacitance: {:.1} kJ/m²K, Time Constant: {:.1} hours, Damping: {:.3}, Classification: {}",
            effective_capacitance, time_constant, damping_factor, classification
        );

        ThermalMassReport {
            effective_capacitance,
            time_constant,
            damping_factor,
            classification,
        }
    }

    /// Analyze building with custom timestep.
    ///
    /// # Arguments
    /// * `custom_timestep_seconds` - Custom timestep in seconds
    ///
    /// # Returns
    /// ThermalMassReport with metrics calculated for custom timestep
    pub fn analyze_with_timestep(&self, custom_timestep_seconds: u32) -> ThermalMassReport {
        let mut custom_analyzer = self.clone();
        custom_analyzer.timestep_seconds = custom_timestep_seconds;

        custom_analyzer.analyze()
    }
}

impl Clone for ThermalMassDiagnostics {
    fn clone(&self) -> Self {
        Self {
            timestep_seconds: self.timestep_seconds,
            total_heat_loss_coefficient: self.total_heat_loss_coefficient,
            construction_layers: self.construction_layers.clone(),
        }
    }
}

/// Comprehensive thermal mass analysis report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalMassReport {
    /// Effective thermal capacitance in kJ/m²K
    pub effective_capacitance: f64,
    /// Time constant in hours
    pub time_constant: f64,
    /// Damping factor (unitless, 0.0-1.0)
    pub damping_factor: f64,
    /// Thermal mass classification per ISO 13790 Annex C
    pub classification: String,
}

impl Default for ThermalMassReport {
    fn default() -> Self {
        Self {
            effective_capacitance: 150.0,
            time_constant: 6.0,
            damping_factor: 0.5,
            classification: "Medium".to_string(),
        }
    }
}

impl ThermalMassReport {
    /// Create a new ThermalMassReport with specified values.
    ///
    /// # Arguments
    /// * `effective_capacitance` - Effective thermal capacitance in kJ/m²K
    /// * `time_constant` - Time constant in hours
    /// * `damping_factor` - Damping factor (unitless, 0.0-1.0)
    /// * `classification` - Thermal mass classification
    ///
    /// # Returns
    /// A new ThermalMassReport instance
    pub fn new(
        effective_capacitance: f64,
        time_constant: f64,
        damping_factor: f64,
        classification: String,
    ) -> Self {
        Self {
            effective_capacitance,
            time_constant,
            damping_factor,
            classification,
        }
    }
}

impl Display for ThermalMassReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Thermal Mass Analysis Report\n{}\nEffective Capacitance: {:.1} kJ/m²K\nTime Constant: {:.1} hours\nDamping Factor: {:.3}\nClassification: {}",
            "=".repeat(30),
            self.effective_capacitance,
            self.time_constant,
            self.damping_factor,
            self.classification
        )
    }
}

impl ThermalMassReport {
    /// Check if thermal mass is sufficient for high-mass validation.
    ///
    /// # Returns
    /// true if thermal mass meets high-mass criteria (Heavy or VeryHeavy)
    pub fn is_high_mass(&self) -> bool {
        self.classification == "Heavy" || self.classification == "VeryHeavy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::construction::ConstructionLayer;

    #[test]
    fn test_thermal_mass_properties_default() {
        let props = ThermalMassProperties::default();
        assert_eq!(props.effective_capacitance, 150.0);
        assert_eq!(props.time_constant, 6.0);
        assert_eq!(props.damping_factor, 0.5);
        assert_eq!(props.classification, "Medium");
    }

    #[test]
    fn test_thermal_mass_diagnostics_new() {
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        assert_eq!(diagnostics.timestep_seconds, 3600);
        assert_eq!(diagnostics.total_heat_loss_coefficient, 10.0);
        assert!(diagnostics.construction_layers.is_none());
    }

    #[test]
    fn test_calculate_effective_capacitance_no_model() {
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let capacitance = diagnostics.calculate_effective_capacitance();
        // Should return medium-weight default when no construction layers
        assert_eq!(capacitance, 150.0);
    }

    #[test]
    fn test_calculate_time_constant() {
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let time_constant = diagnostics.calculate_time_constant();

        // With default capacitance of 150 kJ/m²K = 150,000 J/m²K
        // τ = 150,000 / 10 = 15,000 seconds = 4.166... hours
        let expected_hours = (150.0 * 1000.0) / 10.0 / 3600.0;
        assert!((time_constant - expected_hours).abs() < 0.01);
    }

    #[test]
    fn test_calculate_damping_factor() {
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let damping_factor = diagnostics.calculate_damping_factor();

        // With time constant of ~4.166 hours and 1-hour timestep
        // damping_factor = exp(-1/4.166) ≈ exp(-0.24) ≈ 0.786
        let time_constant = diagnostics.calculate_time_constant();
        let timestep_hours = 3600.0 / 3600.0; // 1 hour
        let expected = (-timestep_hours / time_constant).exp();
        assert!((damping_factor - expected).abs() < 0.01);
    }

    #[test]
    fn test_classify_thermal_mass() {
        // Test VeryLight
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        // Default is medium
        let classification = diagnostics.classify_thermal_mass();
        assert_eq!(classification, "Medium");
    }

    #[test]
    fn test_analyze() {
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let report = diagnostics.analyze();

        assert!(report.effective_capacitance > 0.0);
        assert!(report.time_constant > 0.0);
        assert!(report.damping_factor > 0.0 && report.damping_factor < 1.0);
        assert!(!report.classification.is_empty());
    }

    #[test]
    fn test_thermal_mass_report_is_high_mass() {}
}

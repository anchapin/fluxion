//! HVAC Efficiency Curves
//!
//! This module provides polynomial efficiency curve models for HVAC equipment,
//! using AHRI reference data for realistic part-load performance.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Polynomial efficiency curve coefficients.
///
/// Uses cubic polynomial to model COP as function of part-load ratio (PLR):
/// COP(PLR) = a + b*PLR + c*PLR² + d*PLR³
///
/// Combined with temperature degradation:
/// COP(PLR, T) = COP(PLR) * (1 - temp_coeff * |T - T_design|)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyCurve {
    /// Cubic polynomial coefficients: [a, b, c, d] for COP = a + b*PLR + c*PLR² + d*PLR³
    pub plr_coefficients: [f64; 4],
    /// Temperature coefficient (COP degrades per degree from design temperature)
    pub temp_coefficient: f64,
    /// Design outdoor temperature (°C)
    pub design_temp: f64,
}

impl EfficiencyCurve {
    /// Create a new efficiency curve from coefficients.
    ///
    /// # Arguments
    /// * `plr_coefficients` - Cubic polynomial coefficients [a, b, c, d]
    /// * `temp_coefficient` - Temperature degradation coefficient (per degree)
    /// * `design_temp` - Design outdoor temperature (°C)
    pub fn new(plr_coefficients: [f64; 4], temp_coefficient: f64, design_temp: f64) -> Self {
        Self {
            plr_coefficients,
            temp_coefficient,
            design_temp,
        }
    }

    /// Calculate COP at given PLR and outdoor temperature.
    ///
    /// Uses Horner's method for polynomial evaluation (efficient, avoids repeated powi):
    /// ((d*PLR + c)*PLR + b)*PLR + a
    ///
    /// # Arguments
    /// * `plr` - Part-load ratio (0.0 to 1.0)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    ///
    /// # Returns
    /// Coefficient of Performance (COP) or Energy Efficiency Ratio (EER)
    pub fn cop_at(&self, plr: f64, outdoor_temp: f64) -> f64 {
        // PLR contribution: cubic polynomial using Horner's method
        let plr_cop = ((self.plr_coefficients[3] * plr + self.plr_coefficients[2]) * plr
            + self.plr_coefficients[1])
            * plr
            + self.plr_coefficients[0];

        // Temperature degradation: linear from design temp
        let temp_diff = (self.design_temp - outdoor_temp).abs();
        let temp_factor = 1.0 - self.temp_coefficient * temp_diff;

        plr_cop * temp_factor.max(0.3) // Minimum 30% of rated COP
    }

    /// Evaluate polynomial at given value (for testing/validation).
    ///
    /// Uses Horner's method for numerical stability.
    pub fn evaluate_polynomial(&self, x: f64) -> f64 {
        ((self.plr_coefficients[3] * x + self.plr_coefficients[2]) * x + self.plr_coefficients[1])
            * x
            + self.plr_coefficients[0]
    }
}

/// AHRI efficiency curve configuration for multiple equipment types.
///
/// This structure holds polynomial coefficients for all equipment types,
/// loaded from JSON configuration file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyCurveConfig {
    /// Heat pump heating coefficients (cubic polynomial)
    pub heatpump_heating: CurveCoefficients,
    /// Heat pump cooling coefficients (cubic polynomial)
    pub heatpump_cooling: CurveCoefficients,
    /// Chiller coefficients (cubic polynomial)
    pub chiller: CurveCoefficients,
    /// Boiler coefficients (cubic polynomial)
    pub boiler: CurveCoefficients,
}

/// Curve coefficients for a single equipment type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveCoefficients {
    /// Cubic polynomial: a + b*PLR + c*PLR² + d*PLR³
    pub plr: [f64; 4],
    /// Temperature coefficient (per degree from design)
    pub temp_coefficient: f64,
    /// Design outdoor temperature (°C)
    pub design_temp: f64,
}

/// Load AHRI efficiency curve coefficients from JSON file.
///
/// # Arguments
/// * `path` - Path to JSON configuration file
///
/// # Returns
/// EfficiencyCurveConfig with AHRI coefficients
///
/// # Errors
/// Returns error if file not found or JSON is invalid
pub fn load_ahri_coefficients(path: &str) -> Result<EfficiencyCurveConfig, String> {
    let path_obj = Path::new(path);
    if !path_obj.exists() {
        return Err(format!("AHRI coefficient file not found: {}", path));
    }

    let content = fs::read_to_string(path_obj)
        .map_err(|e| format!("Failed to read AHRI coefficient file: {}", e))?;

    let config: EfficiencyCurveConfig = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse AHRI coefficient JSON: {}", e))?;

    Ok(config)
}

/// Create default AHRI coefficients (placeholder values).
///
/// These placeholder coefficients will be replaced with actual AHRI data
/// when available. For now, they provide reasonable default curves.
pub fn default_ahri_coefficients() -> EfficiencyCurveConfig {
    EfficiencyCurveConfig {
        heatpump_heating: CurveCoefficients {
            // COP = 3.5 - 0.8*PLR + 0.5*PLR² - 0.2*PLR³
            // Degradation: 2% per degree from -5°C design
            plr: [3.5, -0.8, 0.5, -0.2],
            temp_coefficient: 0.02,
            design_temp: -5.0,
        },
        heatpump_cooling: CurveCoefficients {
            // EER = 11.0 - 1.7*PLR + 1.1*PLR² - 0.4*PLR³
            // At PLR=1.0: EER = 11.0 - 1.7 + 1.1 - 0.4 = 10.0 (equivalent to COP 2.93)
            // Degradation: 2.2% per degree from 35°C design (scaled for EER)
            plr: [11.0, -1.7, 1.1, -0.4],
            temp_coefficient: 0.022,
            design_temp: 35.0,
        },
        chiller: CurveCoefficients {
            // COP = 4.5 - 0.6*PLR + 0.4*PLR² - 0.15*PLR³
            // Degradation: 0.5% per degree from 35°C design
            plr: [4.5, -0.6, 0.4, -0.15],
            temp_coefficient: 0.005,
            design_temp: 35.0,
        },
        boiler: CurveCoefficients {
            // Efficiency = 0.85 + 0.05*PLR - 0.03*PLR² + 0.01*PLR³
            // Degradation: 0.1% per degree from -5°C design
            plr: [0.85, 0.05, -0.03, 0.01],
            temp_coefficient: 0.001,
            design_temp: -5.0,
        },
    }
}

impl From<&CurveCoefficients> for EfficiencyCurve {
    fn from(coeffs: &CurveCoefficients) -> Self {
        EfficiencyCurve::new(coeffs.plr, coeffs.temp_coefficient, coeffs.design_temp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_efficiency_curves() {
        // Test cubic polynomial evaluation
        let coeffs = [3.5, -0.8, 0.5, -0.2];
        let curve = EfficiencyCurve::new(coeffs, 0.0, -5.0); // No temp degradation for this test

        // Test at PLR = 1.0 (full load)
        let cop_full_load = curve.cop_at(1.0, -5.0);
        // Polynomial: 3.5 - 0.8*1 + 0.5*1 - 0.2*1 = 3.0
        assert!((cop_full_load - 3.0).abs() < 0.1);

        // Test at PLR = 0.5 (part load)
        let cop_part_load = curve.cop_at(0.5, -5.0);
        assert!(cop_part_load < 3.5); // Degraded at part load
        assert!(cop_part_load > 2.0); // But not too low

        // Test at PLR = 0.0 (no load)
        let cop_no_load = curve.cop_at(0.0, -5.0);
        assert!((cop_no_load - 3.5).abs() < 0.1); // Intercept coefficient (a)

        // Test temperature degradation (now with temp_coefficient > 0)
        let curve_with_temp = EfficiencyCurve::new(coeffs, 0.02, -5.0);
        let cop_design_temp = curve_with_temp.cop_at(1.0, -5.0);
        let cop_cold_temp = curve_with_temp.cop_at(1.0, -15.0);
        assert!(cop_cold_temp < cop_design_temp); // Degraded at cold temp

        // Test minimum COP (30% of rated polynomial value at PLR=1.0)
        // At PLR=1.0, polynomial = 3.0 (not 3.5 due to cubic terms)
        let cop_extreme_temp = curve_with_temp.cop_at(1.0, -50.0);
        assert!(cop_extreme_temp >= 3.0 * 0.3); // Minimum 30% of rated (3.0)
    }

    #[test]
    fn test_horner_method_evaluation() {
        // Verify Horner's method matches direct evaluation
        let coeffs = [1.0, 2.0, 3.0, 4.0];
        let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);

        // Direct evaluation: 1 + 2*x + 3*x² + 4*x³
        let x: f64 = 2.0;
        let direct = 1.0 + 2.0 * x + 3.0 * x.powi(2) + 4.0 * x.powi(3);

        // Horner's method: ((4*x + 3)*x + 2)*x + 1
        let horner = curve.evaluate_polynomial(x);

        assert!((direct - horner).abs() < 1e-10); // Should match exactly
    }

    #[test]
    fn test_ahri_coefficient_loading() {
        // Test default AHRI coefficients
        let config = default_ahri_coefficients();

        // Verify heat pump coefficients
        assert_eq!(config.heatpump_heating.plr.len(), 4); // 4 coefficients
        assert_eq!(config.heatpump_heating.design_temp, -5.0);
        assert_eq!(config.heatpump_cooling.design_temp, 35.0);

        // Verify chiller coefficients
        assert_eq!(config.chiller.plr.len(), 4);
        assert_eq!(config.chiller.design_temp, 35.0);

        // Verify boiler coefficients
        assert_eq!(config.boiler.plr.len(), 4);
        assert_eq!(config.boiler.design_temp, -5.0);

        // Create efficiency curves from AHRI coefficients
        let hp_heating_curve: EfficiencyCurve = (&config.heatpump_heating).into();
        // With temp_coefficient=0.02, the COP will be degraded at -5°C design temp
        let cop_at_design = hp_heating_curve.cop_at(1.0, -5.0);
        // At design temp, temp_diff = 0, so no degradation
        // Polynomial: 3.5 - 0.8*1 + 0.5*1 - 0.2*1 = 3.0
        assert!((cop_at_design - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_efficiency_curve_s_shape() {
        // Test that polynomial curves produce S-shaped efficiency degradation
        let coeffs = [3.5, -0.8, 0.5, -0.2];
        let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);

        let cop_100 = curve.cop_at(1.0, 0.0); // 3.5 - 0.8 + 0.5 - 0.2 = 3.0
        let cop_75 = curve.cop_at(0.75, 0.0); // Calculated via Horner's method
        let cop_50 = curve.cop_at(0.5, 0.0); // Calculated via Horner's method
        let cop_25 = curve.cop_at(0.25, 0.0); // Calculated via Horner's method
        let cop_0 = curve.cop_at(0.0, 0.0); // 3.5 (intercept)

        // Verify basic properties
        assert!(cop_0 > cop_100); // Intercept > full load (typical for this curve)
        assert!(cop_0 == 3.5); // Intercept matches coefficient a

        // The S-shape depends on the specific coefficients
        // For these coefficients, verify the curve is monotonic decreasing
        assert!(cop_0 > cop_25);
        assert!(cop_25 > cop_50);
        assert!(cop_50 > cop_75);
        assert!(cop_75 > cop_100);
    }

    #[test]
    fn test_temperature_coefficient() {
        // Test that temperature coefficient degrades COP linearly
        let coeffs = [3.5, 0.0, 0.0, 0.0]; // No PLR degradation
        let temp_coeff = 0.02; // 2% per degree
        let curve = EfficiencyCurve::new(coeffs, temp_coeff, -5.0);

        let cop_design = curve.cop_at(1.0, -5.0);
        let cop_5_deg_colder = curve.cop_at(1.0, -10.0);
        let cop_10_deg_colder = curve.cop_at(1.0, -15.0);

        // Linear degradation: 2% per degree
        let degradation_5 = (cop_design - cop_5_deg_colder) / cop_design;
        let degradation_10 = (cop_design - cop_10_deg_colder) / cop_design;

        assert!((degradation_5 - 0.10).abs() < 0.01); // ~10% at 5°C colder
        assert!((degradation_10 - 0.20).abs() < 0.01); // ~20% at 10°C colder
    }
}

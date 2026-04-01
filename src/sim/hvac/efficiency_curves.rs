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
    pub fn new(plr_coefficients: [f64; 4], temp_coefficient: f64, design_temp: f64) -> Self {
        Self {
            plr_coefficients,
            temp_coefficient,
            design_temp,
        }
    }

    /// Calculate COP at given PLR and outdoor temperature.
    pub fn cop_at(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let plr_cop = ((self.plr_coefficients[3] * plr + self.plr_coefficients[2]) * plr
            + self.plr_coefficients[1])
            * plr
            + self.plr_coefficients[0];

        let temp_diff = (self.design_temp - outdoor_temp).abs();
        let temp_factor = 1.0 - self.temp_coefficient * temp_diff;

        plr_cop * temp_factor.max(0.3)
    }

    /// Evaluate polynomial at given value (for testing/validation).
    pub fn evaluate_polynomial(&self, x: f64) -> f64 {
        ((self.plr_coefficients[3] * x + self.plr_coefficients[2]) * x + self.plr_coefficients[1])
            * x
            + self.plr_coefficients[0]
    }
}

/// AHRI efficiency curve configuration for multiple equipment types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyCurveConfig {
    pub heatpump_heating: CurveCoefficients,
    pub heatpump_cooling: CurveCoefficients,
    pub chiller: CurveCoefficients,
    pub boiler: CurveCoefficients,
}

/// Curve coefficients for a single equipment type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveCoefficients {
    pub plr: [f64; 4],
    pub temp_coefficient: f64,
    pub design_temp: f64,
}

/// Load AHRI efficiency curve coefficients from JSON file.
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
pub fn default_ahri_coefficients() -> EfficiencyCurveConfig {
    EfficiencyCurveConfig {
        heatpump_heating: CurveCoefficients {
            plr: [3.5, -0.8, 0.5, -0.2],
            temp_coefficient: 0.02,
            design_temp: -5.0,
        },
        heatpump_cooling: CurveCoefficients {
            plr: [11.0, -1.7, 1.1, -0.4],
            temp_coefficient: 0.022,
            design_temp: 35.0,
        },
        chiller: CurveCoefficients {
            plr: [4.5, -0.6, 0.4, -0.15],
            temp_coefficient: 0.005,
            design_temp: 35.0,
        },
        boiler: CurveCoefficients {
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
        let coeffs = [3.5, -0.8, 0.5, -0.2];
        let curve = EfficiencyCurve::new(coeffs, 0.0, -5.0);

        let cop_full_load = curve.cop_at(1.0, -5.0);
        assert!((cop_full_load - 3.0).abs() < 0.1);

        let cop_part_load = curve.cop_at(0.5, -5.0);
        assert!(cop_part_load < 3.5);
        assert!(cop_part_load > 2.0);

        let cop_no_load = curve.cop_at(0.0, -5.0);
        assert!((cop_no_load - 3.5).abs() < 0.1);

        let curve_with_temp = EfficiencyCurve::new(coeffs, 0.02, -5.0);
        let cop_design_temp = curve_with_temp.cop_at(1.0, -5.0);
        let cop_cold_temp = curve_with_temp.cop_at(1.0, -15.0);
        assert!(cop_cold_temp < cop_design_temp);

        let cop_extreme_temp = curve_with_temp.cop_at(1.0, -50.0);
        assert!(cop_extreme_temp >= 3.0 * 0.3);
    }

    #[test]
    fn test_horner_method_evaluation() {
        let coeffs = [1.0, 2.0, 3.0, 4.0];
        let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);

        let x: f64 = 2.0;
        let direct = 1.0 + 2.0 * x + 3.0 * x.powi(2) + 4.0 * x.powi(3);
        let horner = curve.evaluate_polynomial(x);

        assert!((direct - horner).abs() < 1e-10);
    }

    #[test]
    fn test_ahri_coefficient_loading() {
        let config = default_ahri_coefficients();

        assert_eq!(config.heatpump_heating.plr.len(), 4);
        assert_eq!(config.heatpump_heating.design_temp, -5.0);
        assert_eq!(config.heatpump_cooling.design_temp, 35.0);
        assert_eq!(config.chiller.plr.len(), 4);
        assert_eq!(config.chiller.design_temp, 35.0);
        assert_eq!(config.boiler.plr.len(), 4);
        assert_eq!(config.boiler.design_temp, -5.0);

        let hp_heating_curve: EfficiencyCurve = (&config.heatpump_heating).into();
        let cop_at_design = hp_heating_curve.cop_at(1.0, -5.0);
        assert!((cop_at_design - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_efficiency_curve_s_shape() {
        let coeffs = [3.5, -0.8, 0.5, -0.2];
        let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);

        let cop_100 = curve.cop_at(1.0, 0.0);
        let cop_75 = curve.cop_at(0.75, 0.0);
        let cop_50 = curve.cop_at(0.5, 0.0);
        let cop_25 = curve.cop_at(0.25, 0.0);
        let cop_0 = curve.cop_at(0.0, 0.0);

        assert!(cop_0 > cop_100);
        assert!(cop_0 == 3.5);
        assert!(cop_0 > cop_25);
        assert!(cop_25 > cop_50);
        assert!(cop_50 > cop_75);
        assert!(cop_75 > cop_100);
    }

    #[test]
    fn test_temperature_coefficient() {
        let coeffs = [3.5, 0.0, 0.0, 0.0];
        let temp_coeff = 0.02;
        let curve = EfficiencyCurve::new(coeffs, temp_coeff, -5.0);

        let cop_design = curve.cop_at(1.0, -5.0);
        let cop_5_deg_colder = curve.cop_at(1.0, -10.0);
        let cop_10_deg_colder = curve.cop_at(1.0, -15.0);

        let degradation_5 = (cop_design - cop_5_deg_colder) / cop_design;
        let degradation_10 = (cop_design - cop_10_deg_colder) / cop_design;

        assert!((degradation_5 - 0.10).abs() < 0.01);
        assert!((degradation_10 - 0.20).abs() < 0.01);
    }

    #[test]
    fn test_efficiency_curve_new_and_fields() {
        let coeffs = [2.0, 0.5, -0.1, 0.05];
        let curve = EfficiencyCurve::new(coeffs, 0.01, 20.0);
        assert_eq!(curve.plr_coefficients, coeffs);
        assert!((curve.temp_coefficient - 0.01).abs() < 0.001);
        assert!((curve.design_temp - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_efficiency_curve_clone() {
        let coeffs = [3.5, -0.8, 0.5, -0.2];
        let curve = EfficiencyCurve::new(coeffs, 0.02, -5.0);
        let cloned = curve.clone();
        assert_eq!(cloned.plr_coefficients, curve.plr_coefficients);
        assert_eq!(cloned.temp_coefficient, curve.temp_coefficient);
        assert_eq!(cloned.design_temp, curve.design_temp);
    }

    #[test]
    fn test_evaluate_polynomial_zero() {
        let coeffs = [5.0, 0.0, 0.0, 0.0];
        let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);
        assert!((curve.evaluate_polynomial(0.0) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_polynomial_linear() {
        let coeffs = [0.0, 2.0, 0.0, 0.0];
        let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);
        assert!((curve.evaluate_polynomial(3.0) - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_cop_at_zero_plr() {
        let coeffs = [3.5, -0.8, 0.5, -0.2];
        let curve = EfficiencyCurve::new(coeffs, 0.0, 20.0);
        assert!((curve.cop_at(0.0, 20.0) - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_cop_at_extreme_temperature_minimum_floor() {
        let coeffs = [3.5, 0.0, 0.0, 0.0];
        let curve = EfficiencyCurve::new(coeffs, 0.05, 20.0);
        let cop = curve.cop_at(1.0, 100.0);
        assert!(cop >= 3.5 * 0.3);
    }

    #[test]
    fn test_curve_coefficients_clone_debug() {
        let coeffs = CurveCoefficients {
            plr: [1.0, 2.0, 3.0, 4.0],
            temp_coefficient: 0.02,
            design_temp: 30.0,
        };
        let cloned = coeffs.clone();
        assert_eq!(cloned.plr, coeffs.plr);
        let debug_str = format!("{:?}", coeffs);
        assert!(debug_str.contains("plr"));
    }

    #[test]
    fn test_efficiency_curve_config_clone_debug() {
        let config = default_ahri_coefficients();
        let cloned = config.clone();
        assert_eq!(cloned.heatpump_heating.plr, config.heatpump_heating.plr);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("EfficiencyCurveConfig"));
    }

    #[test]
    fn test_from_curve_coefficients_to_efficiency_curve() {
        let coeffs = CurveCoefficients {
            plr: [2.0, 0.5, -0.1, 0.02],
            temp_coefficient: 0.015,
            design_temp: 30.0,
        };
        let curve: EfficiencyCurve = (&coeffs).into();
        assert_eq!(curve.plr_coefficients, coeffs.plr);
        assert_eq!(curve.temp_coefficient, coeffs.temp_coefficient);
        assert_eq!(curve.design_temp, coeffs.design_temp);
    }

    #[test]
    fn test_load_ahri_coefficients_missing_file() {
        let result = load_ahri_coefficients("/nonexistent/path/ahri.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_default_ahri_coefficients_all_present() {
        let config = default_ahri_coefficients();
        assert_eq!(config.heatpump_heating.plr.len(), 4);
        assert_eq!(config.heatpump_cooling.plr.len(), 4);
        assert_eq!(config.chiller.plr.len(), 4);
        assert_eq!(config.boiler.plr.len(), 4);
    }

    #[test]
    fn test_efficiency_curve_debug_format() {
        let coeffs = [1.0, 2.0, 3.0, 4.0];
        let curve = EfficiencyCurve::new(coeffs, 0.01, 25.0);
        let debug_str = format!("{:?}", curve);
        assert!(debug_str.contains("EfficiencyCurve"));
        assert!(debug_str.contains("plr_coefficients"));
    }

    #[test]
    fn test_load_ahri_coefficients_valid_file() {
        let temp_path = "/tmp/test_ahri_coefficients.json";
        let config = default_ahri_coefficients();
        let json = serde_json::to_string(&config).unwrap();
        std::fs::write(temp_path, &json).unwrap();

        let result = load_ahri_coefficients(temp_path);
        assert!(result.is_ok());
        let loaded = result.unwrap();
        assert_eq!(loaded.heatpump_heating.plr.len(), 4);
        assert_eq!(loaded.heatpump_cooling.design_temp, 35.0);

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_load_ahri_coefficients_invalid_json() {
        let temp_path = "/tmp/test_ahri_invalid.json";
        std::fs::write(temp_path, "not valid json").unwrap();

        let result = load_ahri_coefficients(temp_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse"));

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_efficiency_curve_config_serialization() {
        let config = default_ahri_coefficients();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EfficiencyCurveConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.heatpump_heating.plr,
            config.heatpump_heating.plr
        );
        assert_eq!(deserialized.boiler.design_temp, config.boiler.design_temp);
    }

    #[test]
    fn test_efficiency_curve_serialization() {
        let coeffs = [3.5, -0.8, 0.5, -0.2];
        let curve = EfficiencyCurve::new(coeffs, 0.02, -5.0);
        let json = serde_json::to_string(&curve).unwrap();
        let deserialized: EfficiencyCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.plr_coefficients, coeffs);
        assert_eq!(deserialized.design_temp, -5.0);
    }
}

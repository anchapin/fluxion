//! EnergyPlus Oracle Validation Framework
//!
//! Provides utilities for validating Fluxion physics calculations
//! against EnergyPlus reference data.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Maximum allowed absolute error (for temperatures in K, fluxes in W/m²)
pub const DEFAULT_MAX_ABS_ERROR: f64 = 1.0;

/// Maximum allowed relative error (as fraction, e.g., 0.05 = 5%)
pub const DEFAULT_MAX_REL_ERROR: f64 = 0.05;

/// Minimum required correlation coefficient
pub const DEFAULT_MIN_CORRELATION: f64 = 0.95;

/// Maximum allowed RMSE (root mean squared error)
pub const DEFAULT_MAX_RMSE: f64 = 0.5;

/// EnergyPlus reference results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EPReference {
    /// Test case identifier
    pub case_id: String,

    /// Zone air temperatures (hourly, in °C)
    pub zone_temperatures: Option<Vec<f64>>,

    /// Outdoor air temperatures (hourly, in °C)
    pub outdoor_temperatures: Option<Vec<f64>>,

    /// Heating energy demand (kWh)
    pub heating_energy_kwh: Option<f64>,

    /// Cooling energy demand (kWh)
    pub cooling_energy_kwh: Option<f64>,

    /// Surface outside face temperatures (hourly, in °C)
    pub surface_outside_temps: Option<Vec<f64>>,

    /// Surface inside face temperatures (hourly, in °C)
    pub surface_inside_temps: Option<Vec<f64>>,

    /// Heat fluxes through surfaces (hourly, in W/m²)
    pub surface_fluxes: Option<Vec<f64>>,
}

/// Fluxion simulation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxionResults {
    /// Test case identifier
    pub case_id: String,

    /// Zone air temperatures (hourly, in °C)
    pub zone_temperatures: Vec<f64>,

    /// Outdoor air temperatures (hourly, in °C)
    pub outdoor_temperatures: Vec<f64>,

    /// Heating energy demand (kWh)
    pub heating_energy_kwh: f64,

    /// Cooling energy demand (kWh)
    pub cooling_energy_kwh: f64,

    /// Annual carbon emissions (kg CO2eq)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_carbon_kg: Option<f64>,

    /// Surface outside face temperatures (hourly, in °C)
    pub surface_outside_temps: Option<Vec<f64>>,

    /// Surface inside face temperatures (hourly, in °C)
    pub surface_inside_temps: Option<Vec<f64>>,

    /// Heat fluxes through surfaces (hourly, in W/m²)
    pub surface_fluxes: Option<Vec<f64>>,
}

/// Validation criteria
#[derive(Debug, Clone, Copy)]
pub struct ValidationCriteria {
    /// Maximum allowed absolute error
    pub max_abs_error: f64,

    /// Maximum allowed relative error
    pub max_rel_error: f64,

    /// Minimum required correlation coefficient
    pub min_correlation: f64,

    /// Maximum allowed RMSE
    pub max_rmse: f64,
}

impl Default for ValidationCriteria {
    fn default() -> Self {
        Self {
            max_abs_error: DEFAULT_MAX_ABS_ERROR,
            max_rel_error: DEFAULT_MAX_REL_ERROR,
            min_correlation: DEFAULT_MIN_CORRELATION,
            max_rmse: DEFAULT_MAX_RMSE,
        }
    }
}

impl ValidationCriteria {
    /// Create strict validation criteria (for critical physics)
    pub fn strict() -> Self {
        Self {
            max_abs_error: 0.5,
            max_rel_error: 0.02, // 2%
            min_correlation: 0.98,
            max_rmse: 0.2,
        }
    }

    /// Create lenient validation criteria (for exploratory tests)
    pub fn lenient() -> Self {
        Self {
            max_abs_error: 2.0,
            max_rel_error: 0.10, // 10%
            min_correlation: 0.90,
            max_rmse: 1.0,
        }
    }
}

/// Validation report for a single comparison
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Overall pass/fail status
    pub passed: bool,

    /// Temperature validation
    pub temperature: Option<ValidationDetails>,

    /// Heating energy validation
    pub heating_energy: Option<ValidationDetails>,

    /// Cooling energy validation
    pub cooling_energy: Option<ValidationDetails>,

    /// Flux validation
    pub flux: Option<ValidationDetails>,
}

/// Detailed validation metrics
#[derive(Debug, Clone)]
pub struct ValidationDetails {
    /// Maximum absolute error
    pub max_abs_error: f64,

    /// Maximum relative error
    pub max_rel_error: f64,

    /// Root mean squared error
    pub rmse: f64,

    /// Correlation coefficient
    pub correlation: f64,

    /// Pass status
    pub passed: bool,

    /// Additional details
    pub notes: Vec<String>,
}

impl ValidationDetails {
    /// Create a passed validation
    pub fn pass() -> Self {
        Self {
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            rmse: 0.0,
            correlation: 1.0,
            passed: true,
            notes: vec![],
        }
    }

    /// Create a failed validation
    pub fn fail(reason: &str) -> Self {
        Self {
            max_abs_error: f64::INFINITY,
            max_rel_error: f64::INFINITY,
            rmse: f64::INFINITY,
            correlation: 0.0,
            passed: false,
            notes: vec![reason.to_string()],
        }
    }
}

/// EnergyPlus Oracle Validator
pub struct EPOracle {
    /// Reference data directory
    ref_dir: PathBuf,

    /// Validation criteria
    criteria: ValidationCriteria,
}

impl EPOracle {
    /// Create a new EP oracle validator
    pub fn new() -> Result<Self> {
        let ref_dir = PathBuf::from("refdata/ep");

        Ok(Self {
            ref_dir,
            criteria: ValidationCriteria::default(),
        })
    }

    /// Create EP oracle with custom reference directory
    pub fn with_ref_dir<P: AsRef<Path>>(ref_dir: P) -> Result<Self> {
        Ok(Self {
            ref_dir: ref_dir.as_ref().to_path_buf(),
            criteria: ValidationCriteria::default(),
        })
    }

    /// Set custom validation criteria
    pub fn with_criteria(mut self, criteria: ValidationCriteria) -> Self {
        self.criteria = criteria;
        self
    }

    /// Load EP reference data for a test case
    pub fn load_reference(&self, case_id: &str) -> Result<EPReference> {
        let ref_file = self.ref_dir.join(format!("Case_{}_results.json", case_id));

        let content = fs::read_to_string(&ref_file)
            .with_context(|| format!("Failed to read EP reference file: {}", ref_file.display()))?;

        let mut reference: EPReference = serde_json::from_str(&content).with_context(|| {
            format!("Failed to parse EP reference JSON: {}", ref_file.display())
        })?;

        reference.case_id = case_id.to_string();
        Ok(reference)
    }

    /// Validate Fluxion results against EP reference
    pub fn validate(&self, fluxion_results: &FluxionResults) -> ValidationReport {
        let reference = match self.load_reference(&fluxion_results.case_id) {
            Ok(ref_data) => ref_data,
            Err(e) => {
                log::warn!(
                    "Failed to load EP reference for case {}: {}",
                    fluxion_results.case_id,
                    e
                );
                return ValidationReport {
                    passed: false,
                    temperature: None,
                    heating_energy: None,
                    cooling_energy: None,
                    flux: None,
                };
            }
        };

        let mut report = ValidationReport {
            passed: true,
            temperature: None,
            heating_energy: None,
            cooling_energy: None,
            flux: None,
        };

        // Validate temperatures
        if let (Some(ep_temps), Some(flux_temps)) = (
            &reference.zone_temperatures,
            Some(&fluxion_results.zone_temperatures),
        ) {
            report.temperature = Some(self.validate_temperatures(flux_temps, ep_temps));
            if !report.temperature.as_ref().unwrap().passed {
                report.passed = false;
            }
        }

        // Validate heating energy
        if let (Some(ep_heat), flux_heat) = (
            reference.heating_energy_kwh,
            fluxion_results.heating_energy_kwh,
        ) {
            report.heating_energy = Some(self.validate_energy(flux_heat, ep_heat));
            if !report.heating_energy.as_ref().unwrap().passed {
                report.passed = false;
            }
        }

        // Validate cooling energy
        if let (Some(ep_cool), flux_cool) = (
            reference.cooling_energy_kwh,
            fluxion_results.cooling_energy_kwh,
        ) {
            report.cooling_energy = Some(self.validate_energy(flux_cool, ep_cool));
            if !report.cooling_energy.as_ref().unwrap().passed {
                report.passed = false;
            }
        }

        report
    }

    /// Validate temperature traces
    fn validate_temperatures(&self, fluxion: &[f64], ep: &[f64]) -> ValidationDetails {
        let n = fluxion.len().min(ep.len());

        if n == 0 {
            return ValidationDetails::fail("No temperature data");
        }

        // Calculate RMSE
        let sum_sq: f64 = fluxion
            .iter()
            .zip(ep.iter())
            .take(n)
            .map(|(f, e)| (f - e).powi(2))
            .sum();
        let rmse = (sum_sq / n as f64).sqrt();

        // Calculate max absolute error
        let max_abs = fluxion
            .iter()
            .zip(ep.iter())
            .take(n)
            .map(|(f, e)| (f - e).abs())
            .fold(0.0_f64, f64::max);

        // Calculate max relative error
        let max_rel = fluxion
            .iter()
            .zip(ep.iter())
            .take(n)
            .map(|(f, e)| {
                if e.abs() > 0.001 {
                    (f - e).abs() / e.abs()
                } else {
                    f.abs()
                }
            })
            .fold(0.0_f64, f64::max);

        // Calculate correlation
        let mean_f = fluxion[..n].iter().sum::<f64>() / n as f64;
        let mean_e = ep[..n].iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denom_f = 0.0;
        let mut denom_e = 0.0;

        for i in 0..n {
            let df = fluxion[i] - mean_f;
            let de = ep[i] - mean_e;
            numerator += df * de;
            denom_f += df * df;
            denom_e += de * de;
        }

        let correlation = if denom_f > 0.0 && denom_e > 0.0 {
            numerator / (denom_f.sqrt() * denom_e.sqrt())
        } else {
            0.0
        };

        // Check criteria
        let passed = rmse <= self.criteria.max_rmse
            && max_abs <= self.criteria.max_abs_error
            && max_rel <= self.criteria.max_rel_error
            && correlation >= self.criteria.min_correlation;

        let mut notes = vec![];
        if !passed {
            if rmse > self.criteria.max_rmse {
                notes.push(format!(
                    "RMSE {:.3} exceeds limit {:.3}",
                    rmse, self.criteria.max_rmse
                ));
            }
            if max_abs > self.criteria.max_abs_error {
                notes.push(format!(
                    "Max abs error {:.3} exceeds limit {:.3}",
                    max_abs, self.criteria.max_abs_error
                ));
            }
            if max_rel > self.criteria.max_rel_error {
                notes.push(format!(
                    "Max rel error {:.3} exceeds limit {:.3}",
                    max_rel, self.criteria.max_rel_error
                ));
            }
            if correlation < self.criteria.min_correlation {
                notes.push(format!(
                    "Correlation {:.3} below limit {:.3}",
                    correlation, self.criteria.min_correlation
                ));
            }
        }

        ValidationDetails {
            max_abs_error: max_abs,
            max_rel_error: max_rel,
            rmse,
            correlation,
            passed,
            notes,
        }
    }

    /// Validate energy consumption
    fn validate_energy(&self, fluxion: f64, ep: f64) -> ValidationDetails {
        let abs_error = (fluxion - ep).abs();
        let rel_error = if ep.abs() > 0.001 {
            abs_error / ep.abs()
        } else {
            fluxion.abs()
        };

        let passed =
            abs_error <= self.criteria.max_abs_error && rel_error <= self.criteria.max_rel_error;

        let mut notes = vec![];
        if !passed {
            if abs_error > self.criteria.max_abs_error {
                notes.push(format!(
                    "Abs error {:.3} exceeds limit {:.3}",
                    abs_error, self.criteria.max_abs_error
                ));
            }
            if rel_error > self.criteria.max_rel_error {
                notes.push(format!(
                    "Rel error {:.1}% exceeds limit {:.1}%",
                    rel_error * 100.0,
                    self.criteria.max_rel_error * 100.0
                ));
            }
        }

        ValidationDetails {
            max_abs_error: abs_error,
            max_rel_error: rel_error,
            rmse: abs_error,              // For single value, RMSE = abs error
            correlation: 1.0 - rel_error, // Approximate correlation
            passed,
            notes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_criteria_default() {
        let criteria = ValidationCriteria::default();
        assert_eq!(criteria.max_abs_error, DEFAULT_MAX_ABS_ERROR);
        assert_eq!(criteria.max_rel_error, DEFAULT_MAX_REL_ERROR);
        assert_eq!(criteria.min_correlation, DEFAULT_MIN_CORRELATION);
        assert_eq!(criteria.max_rmse, DEFAULT_MAX_RMSE);
    }

    #[test]
    fn test_validation_criteria_strict() {
        let criteria = ValidationCriteria::strict();
        assert!(criteria.max_abs_error < DEFAULT_MAX_ABS_ERROR);
        assert!(criteria.max_rel_error < DEFAULT_MAX_REL_ERROR);
        assert!(criteria.min_correlation > DEFAULT_MIN_CORRELATION);
    }

    #[test]
    fn test_validation_criteria_lenient() {
        let criteria = ValidationCriteria::lenient();
        assert!(criteria.max_abs_error > DEFAULT_MAX_ABS_ERROR);
        assert!(criteria.max_rel_error > DEFAULT_MAX_REL_ERROR);
        assert!(criteria.min_correlation < DEFAULT_MIN_CORRELATION);
    }

    #[test]
    fn test_temperature_validation_perfect_match() {
        let oracle = EPOracle::new().unwrap();
        let fluxion = vec![20.0, 21.0, 22.0];
        let ep = vec![20.0, 21.0, 22.0];

        let details = oracle.validate_temperatures(&fluxion, &ep);

        assert!(details.passed);
        assert_eq!(details.rmse, 0.0);
        assert_eq!(details.max_abs_error, 0.0);
        assert_eq!(details.max_rel_error, 0.0);
        // Use approximate comparison for correlation due to floating point precision
        assert!(
            (details.correlation - 1.0).abs() < 1e-10,
            "Expected correlation ≈ 1.0, got {}",
            details.correlation
        );
    }

    #[test]
    fn test_temperature_validation_within_tolerance() {
        let oracle = EPOracle::new().unwrap();
        let fluxion = vec![20.0, 21.5, 22.0];
        let ep = vec![20.0, 21.0, 22.0];

        let details = oracle.validate_temperatures(&fluxion, &ep);

        // Max error is 0.5, which is within default tolerance of 1.0
        assert!(details.passed);
        assert!(details.max_abs_error <= 1.0);
    }

    #[test]
    fn test_temperature_validation_outside_tolerance() {
        let oracle = EPOracle::new().unwrap();
        let fluxion = vec![20.0, 25.0, 22.0];
        let ep = vec![20.0, 21.0, 22.0];

        let details = oracle.validate_temperatures(&fluxion, &ep);

        // Max error is 4.0, which exceeds default tolerance of 1.0
        assert!(!details.passed);
        assert!(details.max_abs_error > 1.0);
        assert!(!details.notes.is_empty());
    }

    #[test]
    fn test_energy_validation_within_tolerance() {
        let oracle = EPOracle::new().unwrap();
        // Use values where both abs_error (<=1.0) and rel_error (<=5%) are within tolerance
        // abs_error = 0.5, rel_error = 0.5/100.5 = 0.497%
        let fluxion = 100.0;
        let ep = 100.5;

        let details = oracle.validate_energy(fluxion, ep);

        // Rel error = 0.5/100.5 = 0.497%, within 5% tolerance
        // Abs error = 0.5, within 1.0 tolerance
        assert!(details.passed);
    }

    #[test]
    fn test_energy_validation_outside_tolerance() {
        let oracle = EPOracle::new().unwrap();
        let fluxion = 1000.0;
        let ep = 800.0;

        let details = oracle.validate_energy(fluxion, ep);

        // Rel error = 200/800 = 25%, exceeds 5% tolerance
        assert!(!details.passed);
        assert!(!details.notes.is_empty());
    }

    #[test]
    fn test_validation_details_pass_and_fail() {
        let pass = ValidationDetails::pass();
        assert!(pass.passed);
        assert_eq!(pass.max_abs_error, 0.0);
        assert_eq!(pass.correlation, 1.0);
        assert!(pass.notes.is_empty());

        let fail = ValidationDetails::fail("test failure reason");
        assert!(!fail.passed);
        assert_eq!(fail.max_abs_error, f64::INFINITY);
        assert_eq!(fail.notes.len(), 1);
        assert_eq!(fail.notes[0], "test failure reason");
    }

    #[test]
    fn test_ep_reference_clone() {
        let ep = EPReference {
            case_id: "600".to_string(),
            zone_temperatures: Some(vec![20.0, 21.0]),
            outdoor_temperatures: Some(vec![10.0, 11.0]),
            heating_energy_kwh: Some(100.0),
            cooling_energy_kwh: Some(50.0),
            surface_outside_temps: Some(vec![5.0, 6.0]),
            surface_inside_temps: Some(vec![19.0, 20.0]),
            surface_fluxes: Some(vec![10.0, 11.0]),
        };
        let cloned = ep.clone();
        assert_eq!(cloned.case_id, "600");
        assert_eq!(cloned.heating_energy_kwh, Some(100.0));
    }

    #[test]
    fn test_fluxion_results_clone() {
        let results = FluxionResults {
            case_id: "600".to_string(),
            zone_temperatures: vec![20.0, 21.0],
            outdoor_temperatures: vec![10.0, 11.0],
            heating_energy_kwh: 100.0,
            cooling_energy_kwh: 50.0,
            surface_outside_temps: Some(vec![5.0, 6.0]),
            surface_inside_temps: Some(vec![19.0, 20.0]),
            surface_fluxes: Some(vec![10.0, 11.0]),
        };
        let cloned = results.clone();
        assert_eq!(cloned.heating_energy_kwh, 100.0);
    }

    #[test]
    fn test_ep_oracle_new_default_ref_dir() {
        let oracle = EPOracle::new().unwrap();
        assert_eq!(oracle.ref_dir, PathBuf::from("refdata/ep"));
    }

    #[test]
    fn test_validate_returns_none_when_no_reference_data() {
        let oracle = EPOracle::new().unwrap();
        let results = FluxionResults {
            case_id: "NONEXISTENT_CASE".to_string(),
            zone_temperatures: vec![20.0, 21.0],
            outdoor_temperatures: vec![10.0, 11.0],
            heating_energy_kwh: 100.0,
            cooling_energy_kwh: 50.0,
            surface_outside_temps: None,
            surface_inside_temps: None,
            surface_fluxes: None,
        };
        let report = oracle.validate(&results);
        assert!(!report.passed);
        assert!(report.temperature.is_none());
        assert!(report.heating_energy.is_none());
        assert!(report.cooling_energy.is_none());
        assert!(report.flux.is_none());
    }

    #[test]
    fn test_validation_details_fail_with_reason() {
        let fail = ValidationDetails::fail("custom failure reason");
        assert!(!fail.passed);
        assert!(fail.max_abs_error.is_infinite());
        assert!(fail.max_rel_error.is_infinite());
        assert!(fail.rmse.is_infinite());
        assert_eq!(fail.correlation, 0.0);
        assert_eq!(fail.notes.len(), 1);
        assert_eq!(fail.notes[0], "custom failure reason");
    }

    #[test]
    fn test_validation_criteria_strict_values() {
        let criteria = ValidationCriteria::strict();
        assert_eq!(criteria.max_abs_error, 0.5);
        assert_eq!(criteria.max_rel_error, 0.02);
        assert_eq!(criteria.min_correlation, 0.98);
        assert_eq!(criteria.max_rmse, 0.2);
    }

    #[test]
    fn test_validation_criteria_lenient_values() {
        let criteria = ValidationCriteria::lenient();
        assert_eq!(criteria.max_abs_error, 2.0);
        assert_eq!(criteria.max_rel_error, 0.10);
        assert_eq!(criteria.min_correlation, 0.90);
        assert_eq!(criteria.max_rmse, 1.0);
    }

    #[test]
    fn test_ep_reference_debug_format() {
        let ep = EPReference {
            case_id: "600".to_string(),
            zone_temperatures: Some(vec![20.0]),
            outdoor_temperatures: None,
            heating_energy_kwh: None,
            cooling_energy_kwh: None,
            surface_outside_temps: None,
            surface_inside_temps: None,
            surface_fluxes: None,
        };
        let debug_str = format!("{:?}", ep);
        assert!(debug_str.contains("EPReference"));
        assert!(debug_str.contains("600"));
    }

    #[test]
    fn test_fluxion_results_debug_format() {
        let results = FluxionResults {
            case_id: "600".to_string(),
            zone_temperatures: vec![20.0],
            outdoor_temperatures: vec![10.0],
            heating_energy_kwh: 100.0,
            cooling_energy_kwh: 50.0,
            surface_outside_temps: None,
            surface_inside_temps: None,
            surface_fluxes: None,
        };
        let debug_str = format!("{:?}", results);
        assert!(debug_str.contains("FluxionResults"));
        assert!(debug_str.contains("600"));
    }

    #[test]
    fn test_validation_details_debug_format() {
        let details = ValidationDetails {
            max_abs_error: 0.5,
            max_rel_error: 0.05,
            rmse: 0.3,
            correlation: 0.97,
            passed: true,
            notes: vec!["test note".to_string()],
        };
        let debug_str = format!("{:?}", details);
        assert!(debug_str.contains("ValidationDetails"));
    }

    #[test]
    fn test_validation_report_debug_format() {
        let report = ValidationReport {
            passed: true,
            temperature: Some(ValidationDetails::pass()),
            heating_energy: None,
            cooling_energy: None,
            flux: None,
        };
        let debug_str = format!("{:?}", report);
        assert!(debug_str.contains("ValidationReport"));
    }

    #[test]
    fn test_ep_reference_clone_all_fields() {
        let ep = EPReference {
            case_id: "900".to_string(),
            zone_temperatures: Some(vec![20.0, 21.0, 22.0]),
            outdoor_temperatures: Some(vec![10.0, 11.0, 12.0]),
            heating_energy_kwh: Some(150.0),
            cooling_energy_kwh: Some(75.0),
            surface_outside_temps: Some(vec![5.0, 6.0, 7.0]),
            surface_inside_temps: Some(vec![19.0, 20.0, 21.0]),
            surface_fluxes: Some(vec![10.0, 11.0, 12.0]),
        };
        let cloned = ep.clone();
        assert_eq!(cloned.case_id, "900");
        assert_eq!(cloned.zone_temperatures, Some(vec![20.0, 21.0, 22.0]));
        assert_eq!(cloned.heating_energy_kwh, Some(150.0));
    }

    #[test]
    fn test_fluxion_results_all_fields() {
        let results = FluxionResults {
            case_id: "900".to_string(),
            zone_temperatures: vec![20.0, 21.0, 22.0],
            outdoor_temperatures: vec![10.0, 11.0, 12.0],
            heating_energy_kwh: 150.0,
            cooling_energy_kwh: 75.0,
            surface_outside_temps: Some(vec![5.0, 6.0, 7.0]),
            surface_inside_temps: Some(vec![19.0, 20.0, 21.0]),
            surface_fluxes: Some(vec![10.0, 11.0, 12.0]),
        };
        assert_eq!(results.case_id, "900");
        assert_eq!(results.zone_temperatures.len(), 3);
        assert_eq!(results.heating_energy_kwh, 150.0);
    }

    #[test]
    fn test_ep_oracle_with_ref_dir_custom() {
        let oracle = EPOracle::with_ref_dir("/custom/path").unwrap();
        assert_eq!(oracle.ref_dir, PathBuf::from("/custom/path"));
    }

    #[test]
    fn test_validate_with_criteria_chaining() {
        let oracle = EPOracle::new().unwrap().with_criteria(ValidationCriteria {
            max_abs_error: 3.0,
            max_rel_error: 0.15,
            min_correlation: 0.85,
            max_rmse: 1.5,
        });
        let fluxion = vec![20.0, 21.0];
        let ep = vec![20.5, 21.5];
        let details = oracle.validate_temperatures(&fluxion, &ep);
        assert!(details.max_abs_error <= 3.0);
    }

    #[test]
    fn test_temperature_validation_single_element() {
        let oracle = EPOracle::new().unwrap();
        let fluxion = vec![20.0];
        let ep = vec![20.5];
        let details = oracle.validate_temperatures(&fluxion, &ep);
        assert!((details.max_abs_error - 0.5).abs() < 0.001);
        assert!((details.rmse - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_energy_validation_very_small_reference() {
        let oracle = EPOracle::new().unwrap();
        let details = oracle.validate_energy(0.001, 0.0005);
        assert!(details.max_abs_error.is_finite());
    }

    #[test]
    fn test_temperature_validation_zero_correlation_case() {
        let oracle = EPOracle::new().unwrap();
        let fluxion = vec![20.0, 20.0, 20.0];
        let ep = vec![20.0, 25.0, 15.0];
        let details = oracle.validate_temperatures(&fluxion, &ep);
        assert!(details.correlation.abs() <= 1.0);
    }

    #[test]
    fn test_temperature_validation_empty_data() {
        let oracle = EPOracle::new().unwrap();
        let fluxion: Vec<f64> = vec![];
        let ep: Vec<f64> = vec![];
        let details = oracle.validate_temperatures(&fluxion, &ep);
        assert!(!details.passed);
        assert!(details
            .notes
            .iter()
            .any(|n| n.contains("No temperature data")));
    }

    #[test]
    fn test_energy_validation_zero_reference() {
        let oracle = EPOracle::new().unwrap();
        let details = oracle.validate_energy(0.5, 0.0);
        assert!(details.max_abs_error.is_finite());
        assert!(details.max_rel_error.is_finite());
    }

    #[test]
    fn test_validation_report_clone() {
        let report = ValidationReport {
            passed: true,
            temperature: Some(ValidationDetails::pass()),
            heating_energy: Some(ValidationDetails::pass()),
            cooling_energy: None,
            flux: None,
        };
        let cloned = report.clone();
        assert!(cloned.passed);
        assert!(cloned.temperature.is_some());
        assert!(cloned.heating_energy.is_some());
    }

    #[test]
    fn test_validation_details_clone() {
        let details = ValidationDetails {
            max_abs_error: 0.5,
            max_rel_error: 0.05,
            rmse: 0.3,
            correlation: 0.97,
            passed: true,
            notes: vec!["test note".to_string()],
        };
        let cloned = details.clone();
        assert_eq!(cloned.max_abs_error, 0.5);
        assert_eq!(cloned.notes.len(), 1);
    }

    #[test]
    fn test_energy_validation_exact_match() {
        let oracle = EPOracle::new().unwrap();
        let details = oracle.validate_energy(100.0, 100.0);
        assert!(details.passed);
        assert_eq!(details.max_abs_error, 0.0);
        assert_eq!(details.max_rel_error, 0.0);
    }

    #[test]
    fn test_temperature_validation_unequal_lengths() {
        let oracle = EPOracle::new().unwrap();
        let fluxion = vec![20.0, 21.0, 22.0, 23.0];
        let ep = vec![20.0, 21.0];
        let details = oracle.validate_temperatures(&fluxion, &ep);
        assert!(details.max_abs_error.is_finite());
        assert!(details.rmse.is_finite());
    }

    #[test]
    fn test_validation_report_partial_failures() {
        let report = ValidationReport {
            passed: false,
            temperature: Some(ValidationDetails::fail("temperature mismatch")),
            heating_energy: Some(ValidationDetails::pass()),
            cooling_energy: None,
            flux: None,
        };
        assert!(!report.passed);
        assert!(report.temperature.is_some());
        assert!(!report.temperature.as_ref().unwrap().passed);
        assert!(report.heating_energy.as_ref().unwrap().passed);
        assert!(report.cooling_energy.is_none());
    }

    #[test]
    fn test_validation_details_multiple_failure_notes() {
        let criteria = ValidationCriteria {
            max_abs_error: 0.5,
            max_rel_error: 0.02,
            min_correlation: 0.98,
            max_rmse: 0.2,
        };
        let oracle = EPOracle::new().unwrap().with_criteria(criteria);
        let fluxion = vec![20.0, 30.0, 40.0];
        let ep = vec![20.0, 21.0, 22.0];
        let details = oracle.validate_temperatures(&fluxion, &ep);
        assert!(!details.passed);
        assert!(details.notes.len() >= 2);
    }

    #[test]
    fn test_ep_reference_serialize_deserialize() {
        let ep = EPReference {
            case_id: "600".to_string(),
            zone_temperatures: Some(vec![20.0, 21.0, 22.0]),
            outdoor_temperatures: Some(vec![10.0, 11.0, 12.0]),
            heating_energy_kwh: Some(100.0),
            cooling_energy_kwh: Some(50.0),
            surface_outside_temps: Some(vec![5.0, 6.0, 7.0]),
            surface_inside_temps: Some(vec![19.0, 20.0, 21.0]),
            surface_fluxes: Some(vec![10.0, 11.0, 12.0]),
        };
        let json = serde_json::to_string(&ep).unwrap();
        let parsed: EPReference = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.case_id, "600");
        assert_eq!(parsed.heating_energy_kwh, Some(100.0));
    }

    #[test]
    fn test_validate_with_all_fields_populated() {
        let temp_dir = std::env::temp_dir().join("fluxion_ep_oracle_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let ref_data = EPReference {
            case_id: "TEST_CASE".to_string(),
            zone_temperatures: Some(vec![20.0, 21.0, 22.0, 23.0]),
            outdoor_temperatures: Some(vec![10.0, 11.0, 12.0, 13.0]),
            heating_energy_kwh: Some(100.0),
            cooling_energy_kwh: Some(50.0),
            surface_outside_temps: Some(vec![5.0, 6.0, 7.0, 8.0]),
            surface_inside_temps: Some(vec![19.0, 20.0, 21.0, 22.0]),
            surface_fluxes: Some(vec![10.0, 11.0, 12.0, 13.0]),
        };
        let json = serde_json::to_string(&ref_data).unwrap();
        std::fs::write(temp_dir.join("Case_TEST_CASE_results.json"), &json).unwrap();

        let oracle = EPOracle::with_ref_dir(&temp_dir).unwrap();
        let results = FluxionResults {
            case_id: "TEST_CASE".to_string(),
            zone_temperatures: vec![20.1, 21.1, 22.1, 23.1],
            outdoor_temperatures: vec![10.0, 11.0, 12.0, 13.0],
            heating_energy_kwh: 100.5,
            cooling_energy_kwh: 50.2,
            surface_outside_temps: Some(vec![5.1, 6.1, 7.1, 8.1]),
            surface_inside_temps: Some(vec![19.1, 20.1, 21.1, 22.1]),
            surface_fluxes: Some(vec![10.1, 11.1, 12.1, 13.1]),
        };

        let report = oracle.validate(&results);
        assert!(report.temperature.is_some());
        assert!(report.heating_energy.is_some());
        assert!(report.cooling_energy.is_some());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_validate_with_partial_reference_data() {
        let temp_dir = std::env::temp_dir().join("fluxion_ep_partial_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let ref_data = EPReference {
            case_id: "PARTIAL".to_string(),
            zone_temperatures: Some(vec![20.0, 21.0]),
            outdoor_temperatures: None,
            heating_energy_kwh: Some(100.0),
            cooling_energy_kwh: None,
            surface_outside_temps: None,
            surface_inside_temps: None,
            surface_fluxes: None,
        };
        let json = serde_json::to_string(&ref_data).unwrap();
        std::fs::write(temp_dir.join("Case_PARTIAL_results.json"), &json).unwrap();

        let oracle = EPOracle::with_ref_dir(&temp_dir).unwrap();
        let results = FluxionResults {
            case_id: "PARTIAL".to_string(),
            zone_temperatures: vec![20.5, 21.5],
            outdoor_temperatures: vec![10.0, 11.0],
            heating_energy_kwh: 101.0,
            cooling_energy_kwh: 0.0,
            surface_outside_temps: None,
            surface_inside_temps: None,
            surface_fluxes: None,
        };

        let report = oracle.validate(&results);
        assert!(report.temperature.is_some());
        assert!(report.heating_energy.is_some());
        assert!(report.cooling_energy.is_none());
        assert!(report.flux.is_none());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

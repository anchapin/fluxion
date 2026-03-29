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
        assert_eq!(details.correlation, 1.0);
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
        let fluxion = 1000.0;
        let ep = 1050.0;

        let details = oracle.validate_energy(fluxion, ep);

        // Rel error = 50/1050 = 4.76%, within 5% tolerance
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
}

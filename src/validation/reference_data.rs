//! Reference data loading utilities for ASHRAE 140 validation
//!
//! This module provides utilities for loading and processing reference data
//! from various sources (benchmark constants, CSV files) for multi-zone
//! validation cases like ASHRAE 140 Case 960 and Case 970.

use crate::validation::ashrae_140_multi_zone::{Case960Reference, Case970Reference};
use crate::validation::benchmark::Case960Benchmark;
use csv::Reader;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;

/// Error type for reference data loading operations
#[derive(Debug)]
pub enum ReferenceDataError {
    /// File not found
    FileNotFound(String),
    /// Invalid file format
    InvalidFormat(String),
    /// CSV parsing error
    CsvError(csv::Error),
    /// Missing required field
    MissingField(String),
    /// Invalid data value
    InvalidValue(String),
    /// Unsupported case
    UnsupportedCase(String),
}

impl fmt::Display for ReferenceDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReferenceDataError::FileNotFound(path) => write!(f, "File not found: {}", path),
            ReferenceDataError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ReferenceDataError::CsvError(err) => write!(f, "CSV error: {}", err),
            ReferenceDataError::MissingField(field) => write!(f, "Missing field: {}", field),
            ReferenceDataError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
            ReferenceDataError::UnsupportedCase(case_id) => {
                write!(f, "Unsupported case: {}", case_id)
            }
        }
    }
}

impl Error for ReferenceDataError {}

impl From<csv::Error> for ReferenceDataError {
    fn from(err: csv::Error) -> Self {
        ReferenceDataError::CsvError(err)
    }
}

/// Reference data structure for validation
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceData {
    /// Case identifier (e.g., "960", "970")
    pub case_id: String,

    /// Annual heating energy reference (MWh)
    pub annual_heating: f64,

    /// Annual cooling energy reference (MWh)
    pub annual_cooling: f64,

    /// Peak heating load reference (kW)
    pub peak_heating: f64,

    /// Peak cooling load reference (kW)
    pub peak_cooling: f64,

    /// Zone temperatures at key timesteps (hour -> temperatures)
    #[serde(skip_deserializing)]
    pub zone_temperatures: HashMap<usize, Vec<f64>>,

    /// Energy tolerance (fraction)
    pub energy_tolerance: f64,

    /// Load tolerance (fraction)
    pub load_tolerance: f64,

    /// Temperature tolerance (°C)
    pub temperature_tolerance: f64,
}

impl Default for ReferenceData {
    fn default() -> Self {
        Self {
            case_id: "960".to_string(),
            annual_heating: 12.4,
            annual_cooling: 8.7,
            peak_heating: 5.2,
            peak_cooling: 4.8,
            zone_temperatures: HashMap::new(),
            energy_tolerance: 0.15,
            load_tolerance: 0.10,
            temperature_tolerance: 1.0,
        }
    }
}

impl ReferenceData {
    /// Validate that reference data is complete and reasonable
    pub fn validate_reference_format(&self) -> Result<(), ReferenceDataError> {
        if self.annual_heating < 0.0 {
            return Err(ReferenceDataError::InvalidValue(
                "annual_heating cannot be negative".to_string(),
            ));
        }

        if self.annual_cooling < 0.0 {
            return Err(ReferenceDataError::InvalidValue(
                "annual_cooling cannot be negative".to_string(),
            ));
        }

        if self.peak_heating < 0.0 {
            return Err(ReferenceDataError::InvalidValue(
                "peak_heating cannot be negative".to_string(),
            ));
        }

        if self.peak_cooling < 0.0 {
            return Err(ReferenceDataError::InvalidValue(
                "peak_cooling cannot be negative".to_string(),
            ));
        }

        if self.energy_tolerance <= 0.0 || self.energy_tolerance > 1.0 {
            return Err(ReferenceDataError::InvalidValue(
                "energy_tolerance must be between 0 and 1".to_string(),
            ));
        }

        if self.load_tolerance <= 0.0 || self.load_tolerance > 1.0 {
            return Err(ReferenceDataError::InvalidValue(
                "load_tolerance must be between 0 and 1".to_string(),
            ));
        }

        if self.temperature_tolerance <= 0.0 {
            return Err(ReferenceDataError::InvalidValue(
                "temperature_tolerance must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Load Case 960 reference data from benchmark constants
pub fn load_case_960_reference() -> Result<Case960Reference, ReferenceDataError> {
    let benchmark = Case960Benchmark::default();

    let reference = Case960Reference {
        zone_temperatures: HashMap::from([
            (4380, vec![15.2, 8.1]),  // Winter design day
            (5000, vec![26.8, 38.4]), // Summer design day
            (8760, vec![20.1, 18.7]), // Annual average
        ]),
        annual_heating: benchmark.annual_heating_ref,
        annual_cooling: benchmark.annual_cooling_ref,
        peak_heating: benchmark.peak_heating_ref,
        peak_cooling: benchmark.peak_cooling_ref,
        min_temperature: 5.0,
        max_temperature: 45.0,
        temperature_tolerance: 1.0,
        energy_tolerance: benchmark.energy_tolerance,
        load_tolerance: benchmark.peak_tolerance,
    };

    Ok(reference)
}

/// Load Case 970 reference data (placeholder for future implementation)
pub fn load_case_970_reference() -> Result<Case970Reference, ReferenceDataError> {
    let reference = Case970Reference {
        zone_temperatures: HashMap::from([
            (4380, vec![18.5, 16.2]), // Winter design day
            (5000, vec![24.8, 22.4]), // Summer design day
            (8760, vec![21.1, 19.7]), // Annual average
        ]),
        annual_heating: 15.0,
        annual_cooling: 12.0,
        peak_heating: 7.5,
        peak_cooling: 6.8,
        min_temperature: 8.0,
        max_temperature: 42.0,
        temperature_tolerance: 1.5,
        energy_tolerance: 0.15,
        load_tolerance: 0.10,
    };

    Ok(reference)
}

/// Load multi-zone reference data by case ID
pub fn load_multi_zone_reference(case_id: &str) -> Result<ReferenceData, ReferenceDataError> {
    match case_id {
        "960" => {
            let case960_ref = load_case_960_reference()?;
            let mut data = ReferenceData::default();
            data.case_id = "960".to_string();
            data.annual_heating = case960_ref.annual_heating;
            data.annual_cooling = case960_ref.annual_cooling;
            data.peak_heating = case960_ref.peak_heating;
            data.peak_cooling = case960_ref.peak_cooling;
            data.zone_temperatures = case960_ref.zone_temperatures;
            data.energy_tolerance = case960_ref.energy_tolerance;
            data.load_tolerance = case960_ref.load_tolerance;
            data.temperature_tolerance = case960_ref.temperature_tolerance;
            Ok(data)
        }
        "970" => {
            let case970_ref = load_case_970_reference()?;
            let mut data = ReferenceData::default();
            data.case_id = "970".to_string();
            data.annual_heating = case970_ref.annual_heating;
            data.annual_cooling = case970_ref.annual_cooling;
            data.peak_heating = case970_ref.peak_heating;
            data.peak_cooling = case970_ref.peak_cooling;
            data.zone_temperatures = case970_ref.zone_temperatures;
            data.energy_tolerance = case970_ref.energy_tolerance;
            data.load_tolerance = case970_ref.load_tolerance;
            data.temperature_tolerance = case970_ref.temperature_tolerance;
            Ok(data)
        }
        _ => Err(ReferenceDataError::UnsupportedCase(case_id.to_string())),
    }
}

/// Load reference data from CSV file
pub fn load_csv_reference<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<ReferenceData>, ReferenceDataError> {
    if !path.as_ref().exists() {
        return Err(ReferenceDataError::FileNotFound(
            path.as_ref().to_string_lossy().into_owned(),
        ));
    }

    let mut reader = Reader::from_path(path)?;
    let mut results = Vec::new();

    for result in reader.deserialize() {
        let mut record: ReferenceData = result?;

        // Validate the record
        record.validate_reference_format()?;

        // Parse zone temperatures if present (CSV format would need custom handling)
        // For now, we'll use default zone temperatures
        if record.zone_temperatures.is_empty() {
            // Add default zone temperatures based on case
            if record.case_id == "960" {
                record.zone_temperatures = HashMap::from([
                    (4380, vec![15.2, 8.1]),
                    (5000, vec![26.8, 38.4]),
                    (8760, vec![20.1, 18.7]),
                ]);
            } else if record.case_id == "970" {
                record.zone_temperatures = HashMap::from([
                    (4380, vec![18.5, 16.2]),
                    (5000, vec![24.8, 22.4]),
                    (8760, vec![21.1, 19.7]),
                ]);
            }
        }

        results.push(record);
    }

    Ok(results)
}

/// Parse hourly data from CSV content
pub fn parse_hourly_data(csv_content: &str) -> Result<Vec<f64>, ReferenceDataError> {
    let mut reader = Reader::from_reader(csv_content.as_bytes());
    let mut values = Vec::new();

    for result in reader.records() {
        let record = result?;
        if record.len() >= 1 {
            if let Ok(value) = record[0].parse::<f64>() {
                values.push(value);
            }
        }
    }

    if values.is_empty() {
        return Err(ReferenceDataError::InvalidFormat(
            "No valid numeric data found in CSV".to_string(),
        ));
    }

    Ok(values)
}

/// Calculate percentage difference between actual and reference values
pub fn calculate_percentage_difference(actual: f64, reference: f64) -> f64 {
    if reference == 0.0 {
        if actual == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        ((actual - reference).abs() / reference) * 100.0
    }
}

/// Calculate Root Mean Square Error (RMSE)
pub fn calculate_rmse(actual: &[f64], reference: &[f64]) -> Result<f64, ReferenceDataError> {
    if actual.len() != reference.len() {
        return Err(ReferenceDataError::InvalidValue(format!(
            "Length mismatch: actual {} vs reference {}",
            actual.len(),
            reference.len()
        )));
    }

    if actual.is_empty() {
        return Err(ReferenceDataError::InvalidValue(
            "Empty arrays provided".to_string(),
        ));
    }

    let sum_squared: f64 = actual
        .iter()
        .zip(reference.iter())
        .map(|(&a, &r)| {
            let diff = a - r;
            diff * diff
        })
        .sum();

    let mean_squared = sum_squared / actual.len() as f64;
    Ok(mean_squared.sqrt())
}

/// Calculate Mean Bias Error (MBE)
pub fn calculate_mbe(actual: &[f64], reference: &[f64]) -> Result<f64, ReferenceDataError> {
    if actual.len() != reference.len() {
        return Err(ReferenceDataError::InvalidValue(format!(
            "Length mismatch: actual {} vs reference {}",
            actual.len(),
            reference.len()
        )));
    }

    if actual.is_empty() {
        return Err(ReferenceDataError::InvalidValue(
            "Empty arrays provided".to_string(),
        ));
    }

    let sum_errors: f64 = actual
        .iter()
        .zip(reference.iter())
        .map(|(&a, &r)| a - r)
        .sum();

    Ok(sum_errors / actual.len() as f64)
}

/// Check if a value is within tolerance
pub fn within_tolerance(actual: f64, reference: f64, tolerance: f64) -> bool {
    if reference == 0.0 {
        return actual == 0.0;
    }

    let difference = (actual - reference).abs();
    let percentage_diff = (difference / reference) * 100.0;
    percentage_diff <= tolerance * 100.0 + 1e-10 // Add small epsilon for floating point
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_case_960_reference() {
        let reference = load_case_960_reference().unwrap();

        assert_eq!(reference.annual_heating, 2.05);
        assert_eq!(reference.annual_cooling, 2.165);
        assert_eq!(reference.peak_heating, 5.0);
        assert_eq!(reference.peak_cooling, 2.0);
        assert_eq!(reference.energy_tolerance, 0.15);
        assert_eq!(reference.load_tolerance, 0.10);
        assert_eq!(reference.temperature_tolerance, 1.0);
        assert!(!reference.zone_temperatures.is_empty());
    }

    #[test]
    fn test_load_case_970_reference() {
        let reference = load_case_970_reference().unwrap();

        assert_eq!(reference.annual_heating, 15.0);
        assert_eq!(reference.annual_cooling, 12.0);
        assert_eq!(reference.peak_heating, 7.5);
        assert_eq!(reference.peak_cooling, 6.8);
        assert_eq!(reference.energy_tolerance, 0.15);
        assert_eq!(reference.load_tolerance, 0.10);
        assert_eq!(reference.temperature_tolerance, 1.5);
        assert!(!reference.zone_temperatures.is_empty());
    }

    #[test]
    fn test_load_multi_zone_reference() {
        let reference_960 = load_multi_zone_reference("960").unwrap();
        assert_eq!(reference_960.case_id, "960");
        assert_eq!(reference_960.annual_heating, 2.05);

        let reference_970 = load_multi_zone_reference("970").unwrap();
        assert_eq!(reference_970.case_id, "970");
        assert_eq!(reference_970.annual_heating, 15.0);

        let unsupported = load_multi_zone_reference("999");
        assert!(unsupported.is_err());
    }

    #[test]
    fn test_reference_data_validation() {
        let valid_data = ReferenceData::default();
        assert!(valid_data.validate_reference_format().is_ok());

        let mut invalid_data = ReferenceData::default();
        invalid_data.annual_heating = -1.0;
        assert!(invalid_data.validate_reference_format().is_err());

        let mut invalid_data = ReferenceData::default();
        invalid_data.energy_tolerance = 1.5;
        assert!(invalid_data.validate_reference_format().is_err());
    }

    #[test]
    fn test_csv_reference_loading() {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            "case_id,annual_heating,annual_cooling,peak_heating,peak_cooling,energy_tolerance,load_tolerance,temperature_tolerance"
        )
        .unwrap();
        writeln!(temp_file, "960,12.4,8.7,5.2,4.8,0.15,0.10,1.0").unwrap();

        let path = temp_file.path().to_str().unwrap().to_string();
        let references = load_csv_reference(&path).unwrap();

        assert_eq!(references.len(), 1);
        assert_eq!(references[0].case_id, "960");
        assert_eq!(references[0].annual_heating, 12.4);
        assert_eq!(references[0].annual_cooling, 8.7);
    }

    #[test]
    fn test_csv_reference_loading_nonexistent() {
        let result = load_csv_reference("/nonexistent/path.csv");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ReferenceDataError::FileNotFound(_)
        ));
    }

    #[test]
    fn test_parse_hourly_data() {
        let csv_content = "temperature\n20.5\n21.3\n19.8\n";
        let values = parse_hourly_data(csv_content).unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], 20.5);
        assert_eq!(values[1], 21.3);
        assert_eq!(values[2], 19.8);
    }

    #[test]
    fn test_parse_hourly_data_invalid() {
        let csv_content = "temperature\ninvalid\n20.5\n";
        let values = parse_hourly_data(csv_content).unwrap();
        assert_eq!(values.len(), 1); // Only the valid value
        assert_eq!(values[0], 20.5);
    }

    #[test]
    fn test_calculate_percentage_difference() {
        let pct = calculate_percentage_difference(2.2, 2.0);
        assert!(pct >= 9.99 && pct <= 10.01);

        let pct = calculate_percentage_difference(1.8, 2.0);
        assert!(pct >= 9.99 && pct <= 10.01);

        let pct = calculate_percentage_difference(0.0, 0.0);
        assert_eq!(pct, 0.0);

        let pct = calculate_percentage_difference(5.0, 0.0);
        assert!(pct.is_infinite());
    }

    #[test]
    fn test_calculate_rmse() {
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let reference = vec![1.1, 1.9, 3.1, 3.9];
        let rmse = calculate_rmse(&actual, &reference).unwrap();
        assert!(rmse >= 0.099 && rmse <= 0.101);
    }

    #[test]
    fn test_calculate_rmse_length_mismatch() {
        let actual = vec![1.0, 2.0];
        let reference = vec![1.0];
        let result = calculate_rmse(&actual, &reference);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_mbe() {
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let reference = vec![1.1, 1.9, 3.1, 3.9];
        let mbe = calculate_mbe(&actual, &reference).unwrap();
        assert!(mbe >= -0.001 && mbe <= 0.001); // Should be close to 0.0
    }

    #[test]
    fn test_within_tolerance() {
        assert!(within_tolerance(2.05, 2.05, 0.15));
        assert!(within_tolerance(2.3575, 2.05, 0.15));
        assert!(within_tolerance(1.7425, 2.05, 0.15));
        assert!(!within_tolerance(2.4, 2.05, 0.15));
        assert!(!within_tolerance(1.7, 2.05, 0.15));
    }

    #[test]
    fn test_within_tolerance_zero_reference() {
        assert!(within_tolerance(0.0, 0.0, 0.15));
        assert!(!within_tolerance(1.0, 0.0, 0.15));
    }
}

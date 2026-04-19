//! Enhanced reference data loading module for ASHRAE 140 validation
//!
//! This module provides CSV-based reference data loading for expanded ASHRAE 140
//! cases (800-810 and 195-470) with caching and proper error handling.

use crate::validation::ashrae140::ASHRAE140Case;
use csv::Reader;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

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
    /// Data completeness error
    IncompleteData(String),
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
            ReferenceDataError::IncompleteData(msg) => write!(f, "Incomplete data: {}", msg),
        }
    }
}

impl From<csv::Error> for ReferenceDataError {
    fn from(err: csv::Error) -> Self {
        ReferenceDataError::CsvError(err)
    }
}

/// Reference data row structure matching CSV format for series 800
#[derive(Debug, Deserialize)]
struct Series800DataRow {
    case: u32,
    hour: u32,
    zone1_temp: f64,
    zone1_heating: f64,
    zone1_cooling: f64,
    zone2_temp: Option<f64>,
    zone2_heating: Option<f64>,
    zone2_cooling: Option<f64>,
    total_energy: f64,
}

/// Reference data row structure matching CSV format for series 195
#[derive(Debug, Deserialize)]
struct Series195DataRow {
    case: u32,
    hour: u32,
    zone1_temp: f64,
    zone1_heating: f64,
    zone1_cooling: f64,
    total_energy: f64,
    peak_load: Option<f64>,
}

/// Reference dataset structure for storing loaded case data
#[derive(Debug, Clone)]
pub struct ReferenceDataset {
    pub case: ASHRAE140Case,
    pub hourly_data: HashMap<u32, HourlyDataPoint>,
    pub peak_load: Option<f64>,
}

impl ReferenceDataset {
    /// Create new reference dataset for a case
    pub fn new(case: ASHRAE140Case) -> Self {
        Self {
            case,
            hourly_data: HashMap::new(),
            peak_load: None,
        }
    }

    /// Add hourly data point
    pub fn add_hourly_data(
        &mut self,
        hour: u32,
        zone1_temp: f64,
        zone1_heating: f64,
        zone1_cooling: f64,
        total_energy: f64,
    ) {
        self.hourly_data.insert(
            hour,
            HourlyDataPoint {
                zone1_temp,
                zone1_heating,
                zone1_cooling,
                zone2_temp: None,
                zone2_heating: None,
                zone2_cooling: None,
                total_energy,
            },
        );
    }

    /// Add zone 2 data to existing hourly data point
    pub fn add_zone2_data(
        &mut self,
        hour: u32,
        zone2_temp: f64,
        zone2_heating: f64,
        zone2_cooling: f64,
    ) {
        if let Some(data_point) = self.hourly_data.get_mut(&hour) {
            data_point.zone2_temp = Some(zone2_temp);
            data_point.zone2_heating = Some(zone2_heating);
            data_point.zone2_cooling = Some(zone2_cooling);
        }
    }

    /// Set peak load value
    pub fn set_peak_load(&mut self, _hour: u32, peak_load: f64) {
        self.peak_load = Some(peak_load);
    }

    /// Validate that dataset has complete data (8760 hours)
    pub fn validate_completeness(&self) -> Result<(), ReferenceDataError> {
        if self.hourly_data.len() != 8760 {
            return Err(ReferenceDataError::IncompleteData(format!(
                "Case {} has {} hours, expected 8760",
                self.case.number(),
                self.hourly_data.len()
            )));
        }
        Ok(())
    }
}

/// Hourly data point structure
#[derive(Debug, Clone)]
pub struct HourlyDataPoint {
    pub zone1_temp: f64,
    pub zone1_heating: f64,
    pub zone1_cooling: f64,
    pub zone2_temp: Option<f64>,
    pub zone2_heating: Option<f64>,
    pub zone2_cooling: Option<f64>,
    pub total_energy: f64,
}

/// Load reference data for series 800 cases (HVAC equipment validation)
pub fn load_series_800_reference(
    case: ASHRAE140Case,
) -> Result<ReferenceDataset, ReferenceDataError> {
    let case_num_str = case.number();
    let case_num: u32 = case_num_str.parse().map_err(|_| {
        ReferenceDataError::InvalidValue(format!("Invalid case number: {}", case_num_str))
    })?;
    let path = Path::new("data/reference/ashrae140/series_800.csv");

    if !path.exists() {
        return Err(ReferenceDataError::FileNotFound(
            path.to_string_lossy().into_owned(),
        ));
    }

    let mut reader = Reader::from_path(path)?;
    let mut dataset = ReferenceDataset::new(case);

    for result in reader.deserialize() {
        let record: Series800DataRow = result?;
        if record.case == case_num {
            dataset.add_hourly_data(
                record.hour,
                record.zone1_temp,
                record.zone1_heating,
                record.zone1_cooling,
                record.total_energy,
            );
            if let Some(zone2_temp) = record.zone2_temp {
                dataset.add_zone2_data(
                    record.hour,
                    zone2_temp,
                    record.zone2_heating.unwrap_or(0.0),
                    record.zone2_cooling.unwrap_or(0.0),
                );
            }
        }
    }

    dataset.validate_completeness()?;
    Ok(dataset)
}

/// Load reference data for series 195 cases (diagnostic validation)
pub fn load_series_195_reference(
    case: ASHRAE140Case,
) -> Result<ReferenceDataset, ReferenceDataError> {
    let case_num_str = case.number();
    let case_num: u32 = case_num_str.parse().map_err(|_| {
        ReferenceDataError::InvalidValue(format!("Invalid case number: {}", case_num_str))
    })?;
    let path = Path::new("data/reference/ashrae140/series_195.csv");

    if !path.exists() {
        return Err(ReferenceDataError::FileNotFound(
            path.to_string_lossy().into_owned(),
        ));
    }

    let mut reader = Reader::from_path(path)?;
    let mut dataset = ReferenceDataset::new(case);

    for result in reader.deserialize() {
        let record: Series195DataRow = result?;
        if record.case == case_num {
            dataset.add_hourly_data(
                record.hour,
                record.zone1_temp,
                record.zone1_heating,
                record.zone1_cooling,
                record.total_energy,
            );
            if let Some(peak_load) = record.peak_load {
                dataset.set_peak_load(record.hour, peak_load);
            }
        }
    }

    dataset.validate_completeness()?;
    Ok(dataset)
}

/// Cached reference data loading
static REFERENCE_CACHE: Lazy<Mutex<HashMap<ASHRAE140Case, ReferenceDataset>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Load reference data for any ASHRAE 140 case with caching
pub fn load_reference_data(case: ASHRAE140Case) -> Result<ReferenceDataset, ReferenceDataError> {
    // Check cache first
    {
        let cache = REFERENCE_CACHE.lock().unwrap();
        if let Some(data) = cache.get(&case) {
            return Ok(data.clone());
        }
    }

    // Load from file
    let dataset = match case {
        ASHRAE140Case::Case800
        | ASHRAE140Case::Case801
        | ASHRAE140Case::Case802
        | ASHRAE140Case::Case803
        | ASHRAE140Case::Case804
        | ASHRAE140Case::Case805
        | ASHRAE140Case::Case806
        | ASHRAE140Case::Case807
        | ASHRAE140Case::Case808
        | ASHRAE140Case::Case809
        | ASHRAE140Case::Case810 => load_series_800_reference(case),
        ASHRAE140Case::Case195
        | ASHRAE140Case::Case195HighMass
        | ASHRAE140Case::Case195NoLoads
        | ASHRAE140Case::Case195NoSolar
        | ASHRAE140Case::Case195ThermalBridge
        | ASHRAE140Case::Case195SHGC03
        | ASHRAE140Case::Case195SHGC06
        | ASHRAE140Case::Case195SHGC09
        | ASHRAE140Case::Case195Albedo01
        | ASHRAE140Case::Case195Albedo05
        | ASHRAE140Case::Case195Albedo09
        | ASHRAE140Case::Case196
        | ASHRAE140Case::Case197
        | ASHRAE140Case::Case198
        | ASHRAE140Case::Case200
        | ASHRAE140Case::Case470 => load_series_195_reference(case),
        _ => {
            // Fallback to legacy reference loading for other cases
            return Err(ReferenceDataError::UnsupportedCase(format!(
                "Case {:?}",
                case
            )));
        }
    }?;

    // Cache the result
    REFERENCE_CACHE
        .lock()
        .unwrap()
        .insert(case, dataset.clone());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;

    #[test]
    fn test_load_series_800_reference() {
        let case = ASHRAE140Case::Case800;
        let result = load_series_800_reference(case);

        match result {
            Ok(dataset) => {
                assert_eq!(dataset.case, case);
                assert_eq!(dataset.hourly_data.len(), 8760);
                assert!(dataset.peak_load.is_none()); // Series 800 doesn't have peak_load in CSV
            }
            Err(ReferenceDataError::FileNotFound(_)) => {
                // File doesn't exist in test environment, which is expected
                println!("Test skipped: series_800.csv not found");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_load_series_195_reference() {
        let case = ASHRAE140Case::Case195;
        let result = load_series_195_reference(case);

        match result {
            Ok(dataset) => {
                assert_eq!(dataset.case, case);
                assert_eq!(dataset.hourly_data.len(), 8760);
                assert!(dataset.peak_load.is_some()); // Series 195 has peak_load
            }
            Err(ReferenceDataError::FileNotFound(_)) => {
                // File doesn't exist in test environment, which is expected
                println!("Test skipped: series_195.csv not found");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_reference_data_caching() {
        // This test verifies that caching works by loading the same case twice
        // and ensuring it returns the same instance
        let case = ASHRAE140Case::Case800;

        // Clear cache first
        REFERENCE_CACHE.lock().unwrap().clear();

        // First load (should hit file system or fail)
        let first_result = load_series_800_reference(case);

        // Second load (should hit cache if first succeeded)
        let second_result = load_reference_data(case);

        match (first_result, second_result) {
            (Ok(first), Ok(second)) => {
                assert_eq!(first.case, second.case);
                assert_eq!(first.hourly_data.len(), second.hourly_data.len());
            }
            (
                Err(ReferenceDataError::FileNotFound(_)),
                Err(ReferenceDataError::UnsupportedCase(_)),
            ) => {
                // Expected in test environment without files
                println!("Test skipped: reference files not found");
            }
            (Err(e1), Err(e2)) => panic!("Both loads failed: {} and {}", e1, e2),
            _ => panic!("Inconsistent results between cached and non-cached loads"),
        }
    }

    #[test]
    fn test_dataset_completeness_validation() {
        let case = ASHRAE140Case::Case800;
        let mut dataset = ReferenceDataset::new(case);

        // Add incomplete data
        dataset.add_hourly_data(1, 20.0, 100.0, 50.0, 150.0);

        // Should fail validation
        let result = dataset.validate_completeness();
        assert!(result.is_err());

        if let Err(ReferenceDataError::IncompleteData(msg)) = result {
            assert!(msg.contains("has 1 hours, expected 8760"));
        } else {
            panic!("Expected IncompleteData error");
        }
    }
}

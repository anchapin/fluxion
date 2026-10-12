//! Reference data types for ASHRAE 140 validation
//!
//! Previously this module loaded CSV fixtures from data/synthetic/ashrae140/
//! (series_195.csv, series_800.csv). Those generated fixtures were removed —
//! they were never engine outputs and are not usable as validation reference.
//! The dataset types remain for compatibility; see the quarantine README in
//! data/synthetic/ashrae140/.

use crate::validation::ashrae140::ASHRAE140Case;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
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
    //
    // The synthetic CSV fixtures (data/synthetic/ashrae140/series_*.csv) and
    // their loaders were removed: generated fixtures, never engine outputs,
    // not usable as validation reference. See the quarantine README there.
    let dataset = Err::<ReferenceDataset, ReferenceDataError>(
        ReferenceDataError::UnsupportedCase(format!("{:?}", case)),
    )?;

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

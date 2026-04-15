//! Reference data module for validation
//!
//! This module provides functionality for loading and managing reference data
//! for ASHRAE 140 validation cases.

use std::error::Error;
use std::fmt;

/// Error type for reference data operations
#[derive(Debug)]
pub enum ReferenceDataError {
    /// File not found
    FileNotFound(String),
    /// Invalid file format
    InvalidFormat(String),
    /// Missing required field
    MissingField(String),
    /// Invalid data value
    InvalidValue(String),
}

impl fmt::Display for ReferenceDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReferenceDataError::FileNotFound(path) => write!(f, "File not found: {}", path),
            ReferenceDataError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ReferenceDataError::MissingField(field) => write!(f, "Missing field: {}", field),
            ReferenceDataError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
        }
    }
}

impl Error for ReferenceDataError {}

/// A single hourly data point from reference data
#[derive(Debug, Clone)]
pub struct HourlyDataPoint {
    pub timestamp: String,
    pub value: f64,
}

/// Reference dataset containing hourly data
#[derive(Debug, Clone)]
pub struct ReferenceDataset {
    pub case_id: String,
    pub hourly_data: Vec<HourlyDataPoint>,
}

/// Load reference data for a specific case
pub fn load_reference_data(case_id: &str) -> Result<ReferenceDataset, ReferenceDataError> {
    // TODO: Implement actual loading logic
    Ok(ReferenceDataset {
        case_id: case_id.to_string(),
        hourly_data: vec![],
    })
}

/// Load series 195 reference data
pub fn load_series_195_reference() -> Result<ReferenceDataset, ReferenceDataError> {
    load_reference_data("series_195")
}

/// Load series 800 reference data
pub fn load_series_800_reference() -> Result<ReferenceDataset, ReferenceDataError> {
    load_reference_data("series_800")
}

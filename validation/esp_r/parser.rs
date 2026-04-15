// validation/esp_r/parser.rs
/// ESP-r output parser
///
/// Parses ESP-r CSV output files containing zone temperature and load data
use csv::Reader;
use serde::Deserialize;
use std::error::Error;
use std::path::Path;

/// ESP-r zone data structure
#[derive(Debug, Deserialize)]
pub struct EspRZoneData {
    /// Zone identifier
    pub zone_id: String,
    /// Zone temperature in °C
    pub temperature: f64,
    /// Heating load in W
    pub heating_load: f64,
    /// Cooling load in W
    pub cooling_load: f64,
}

/// Parse ESP-r CSV output file
///
/// # Arguments
/// * `path` - Path to ESP-r CSV output file
///
/// # Returns
/// Vector of parsed zone data
///
/// # Example
/// ```
/// use std::path::Path;
/// let data = parse_esp_r_output(Path::new("esp_r_output.csv"))?;
/// ```
pub fn parse_esp_r_output(path: &Path) -> Result<Vec<EspRZoneData>, Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let mut results = Vec::new();

    for record in reader.deserialize() {
        results.push(record?);
    }

    Ok(results)
}

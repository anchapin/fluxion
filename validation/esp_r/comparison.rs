// validation/esp_r/comparison.rs
/// Cross-validation comparison logic
///
/// Compares Fluxion simulation results with ESP-r reference data
/// using configurable tolerance bands for temperature and load comparisons.
use serde::Serialize;

/// Comparison result for a single zone
#[derive(Debug, Serialize)]
pub struct ComparisonResult {
    /// Zone identifier
    pub zone_id: String,
    /// Whether temperature is within tolerance
    pub temp_within_tolerance: bool,
    /// Whether heating load is within tolerance
    pub heating_within_tolerance: bool,
    /// Absolute temperature difference
    pub temp_difference: f64,
    /// Absolute heating load difference
    pub heating_difference: f64,
}

/// Compare Fluxion results with ESP-r reference data
///
/// # Arguments
/// * `fluxion_results` - Fluxion multi-zone validation results
/// * `esp_r_data` - Parsed ESP-r zone data
/// * `tolerance` - Temperature tolerance in °C
///
/// # Returns
/// Vector of comparison results for each zone
pub fn compare_results(
    fluxion_results: &crate::validation::MultiZoneValidationResults,
    esp_r_data: &[crate::validation::esp_r::parser::EspRZoneData],
    tolerance: f64,
) -> Vec<ComparisonResult> {
    fluxion_results
        .zones
        .iter()
        .map(|zone| {
            // Find matching ESP-r zone
            let esp_r_zone = esp_r_data.iter().find(|e| e.zone_id == zone.id);

            if let Some(esp_r_zone) = esp_r_zone {
                let temp_diff = (zone.average_temp - esp_r_zone.temperature).abs();
                let heating_diff = (zone.total_heating - esp_r_zone.heating_load).abs();

                ComparisonResult {
                    zone_id: zone.id.clone(),
                    temp_within_tolerance: temp_diff <= tolerance,
                    heating_within_tolerance: heating_diff <= tolerance,
                    temp_difference: temp_diff,
                    heating_difference: heating_diff,
                }
            } else {
                // No matching ESP-r zone found
                ComparisonResult {
                    zone_id: zone.id.clone(),
                    temp_within_tolerance: false,
                    heating_within_tolerance: false,
                    temp_difference: f64::INFINITY,
                    heating_difference: f64::INFINITY,
                }
            }
        })
        .collect()
}

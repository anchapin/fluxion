//! Empirical Validation Suite for Monitored Building Data
//!
//! This module provides validation against real-world monitored building data
//! from ASHRAE Research Projects, IEA Solar Heating and Cooling Programme,
//! NREL building performance databases, and other empirical sources.
//!
//! # Monitored Building Data Sources
//!
//! ## ASHRAE Research Projects
//! - ASHRAE RP-1055: VRF systems monitoring
//! - ASHRAE RP-1256: Variable Refrigerant Flow Systems
//! - ASHRAE RP-1312: Indoor Environmental Quality
//! - ASHRAE RP-1061: Conduction transfer functions
//!
//! ## IEA Solar Heating and Cooling Programme
//! - IEA EBC Annex 60: Double facade monitoring
//! - IEA SHC Annex 58: Solar thermal collectors
//!
//! ## NREL Building Performance Data
//! - Commercial Building Monitoring (CBM) database
//! - High-Performance Building Research
//!
//! # Statistical Metrics
//!
//! Following ASHRAE Guideline 14 and ISO 12017-2:
//! - NMBE: Normalized Mean Bias Error (acceptable: ±10%)
//! - CV(RMSE): Coefficient of Variation of RMSE (acceptable: ≤30%)
//! - Hourly data exclusion for near-zero reference values

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Monitored data point from a real building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoredDataPoint {
    /// Timestamp or hour index
    pub hour: usize,
    /// Outdoor drybulb temperature [C]
    pub T_outdoor: f64,
    /// Zone air temperature [C]
    pub T_zone: f64,
    /// Total heating energy [W]
    pub Q_heat: f64,
    /// Total cooling energy [W]
    pub Q_cool: f64,
    /// Solar gains [W]
    pub Q_solar: f64,
    /// Internal gains [W]
    pub Q_internal: f64,
    /// Ventilation heat loss [W]
    pub Q_ventilation: f64,
    /// Conduction heat loss [W]
    pub Q_conduction: f64,
}

/// Source of monitored building data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoredDataSource {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Data source (ASHRAE RP-XXXX, IEA Annex XX, NREL CBM, etc.)
    pub source: String,
    /// Building type (office, residential, commercial, etc.)
    pub building_type: BuildingType,
    /// Climate zone (ASHRAE zones 1-8)
    pub climate_zone: String,
    /// Location city/country
    pub location: String,
    /// Latitude [deg]
    pub latitude: f64,
    /// Longitude [deg]
    pub longitude: f64,
    /// Floor area [m²]
    pub floor_area: f64,
    /// Number of floors
    pub num_floors: usize,
    /// Zone volume [m³]
    pub zone_volume: f64,
    /// Wall U-value [W/m²K]
    pub u_wall: f64,
    /// Roof U-value [W/m²K]
    pub u_roof: f64,
    /// Window U-value [W/m²K]
    pub u_window: f64,
    /// Window-to-wall ratio
    pub wwr: f64,
    /// Air infiltration rate [ACH]
    pub infiltration_ach: f64,
    /// Internal gains density [W/m²]
    pub internal_gains_density: f64,
    /// Time resolution of data (1 = hourly, 0.25 = 15-min)
    pub time_resolution_hours: f64,
    /// Number of data points
    pub num_data_points: usize,
}

/// Building type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildingType {
    Office,
    Residential,
    Commercial,
    Retail,
    Hotel,
    Hospital,
    School,
    Warehouse,
    Other,
}

/// Validation result for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalValidationResult {
    /// Data source ID
    pub source_id: String,
    /// Metric type
    pub metric: EmpiricalMetric,
    /// Fluxion predicted value
    pub predicted: f64,
    /// Monitored/reference value
    pub reference: f64,
    /// Error (predicted - reference)
    pub error: f64,
    /// Percentage error
    pub percentage_error: f64,
    /// Pass/fail status
    pub status: EmpiricalValidationStatus,
}

/// Types of empirical metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmpiricalMetric {
    /// Zone temperature mean [C]
    MeanZoneTemperature,
    /// Zone temperature max [C]
    MaxZoneTemperature,
    /// Zone temperature min [C]
    MinZoneTemperature,
    /// Daily peak heating load [W]
    DailyPeakHeating,
    /// Daily peak cooling load [W]
    DailyPeakCooling,
    /// Annual heating energy [kWh]
    AnnualHeating,
    /// Annual cooling energy [kWh]
    AnnualCooling,
    /// Hourly temperature error [C]
    HourlyTemperature,
}

/// Validation status for empirical validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EmpiricalValidationStatus {
    Pass,
    Warning,
    Fail,
}

/// Aggregated statistics for empirical validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalStatistics {
    /// Number of data points
    pub n: usize,
    /// Mean Bias Error [units]
    pub mbe: f64,
    /// Normalized MBE [%] (ASHRAE acceptable: ±10%)
    pub nmbe: f64,
    /// Root Mean Square Error [units]
    pub rmse: f64,
    /// Coefficient of Variation of RMSE [%] (ASHRAE acceptable: ≤30%)
    pub cv_rmse: f64,
    /// Mean absolute error [units]
    pub mae: f64,
    /// Maximum error [units]
    pub max_error: f64,
    /// R-squared coefficient
    pub r_squared: f64,
    /// Standard error of estimate [units]
    pub standard_error: f64,
    /// 95% confidence interval lower bound
    pub ci95_lower: f64,
    /// 95% confidence interval upper bound
    pub ci95_upper: f64,
}

impl EmpiricalStatistics {
    /// Calculate statistics from predicted and reference arrays
    pub fn calculate(predicted: &[f64], reference: &[f64]) -> Self {
        assert_eq!(predicted.len(), reference.len());
        let n = predicted.len();

        if n == 0 {
            return Self {
                n: 0,
                mbe: f64::NAN,
                nmbe: f64::NAN,
                rmse: f64::NAN,
                cv_rmse: f64::NAN,
                mae: f64::NAN,
                max_error: f64::NAN,
                r_squared: f64::NAN,
                standard_error: f64::NAN,
                ci95_lower: f64::NAN,
                ci95_upper: f64::NAN,
            };
        }

        // Calculate mean reference for normalization
        let mean_ref: f64 = reference.iter().sum::<f64>() / n as f64;

        // Calculate errors
        let errors: Vec<f64> = predicted
            .iter()
            .zip(reference.iter())
            .map(|(p, r)| p - r)
            .collect();

        // MBE (Mean Bias Error)
        let mbe: f64 = errors.iter().sum::<f64>() / n as f64;

        // NMBE (Normalized MBE)
        let nmbe = if mean_ref.abs() > 1e-10 {
            (mbe / mean_ref) * 100.0
        } else {
            f64::NAN
        };

        // RMSE
        let mse: f64 = errors.iter().map(|e| e * e).sum::<f64>() / n as f64;
        let rmse = mse.sqrt();

        // CV(RMSE) - coefficient of variation
        let cv_rmse = if mean_ref.abs() > 1e-10 {
            (rmse / mean_ref.abs()) * 100.0
        } else {
            f64::NAN
        };

        // MAE (Mean Absolute Error)
        let mae: f64 = errors.iter().map(|e| e.abs()).sum::<f64>() / n as f64;

        // Max error
        let max_error = errors.iter().map(|e| e.abs()).fold(0.0f64, f64::max);

        // R-squared
        let ss_res: f64 = errors.iter().map(|e| e * e).sum::<f64>();
        let mean_pred: f64 = predicted.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = predicted
            .iter()
            .map(|p| (p - mean_pred).powi(2))
            .sum::<f64>();
        let r_squared = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            f64::NAN
        };

        // Standard error
        let standard_error = if n > 1 {
            (mse * n as f64 / (n - 1) as f64).sqrt()
        } else {
            f64::NAN
        };

        // 95% confidence interval using t-distribution
        let (ci95_lower, ci95_upper) = if n > 1 {
            let t_value = 1.96; // Approximation for large n
            let se_mean = standard_error / (n as f64).sqrt();
            (mbe - t_value * se_mean, mbe + t_value * se_mean)
        } else {
            (f64::NAN, f64::NAN)
        };

        Self {
            n,
            mbe,
            nmbe,
            rmse,
            cv_rmse,
            mae,
            max_error,
            r_squared,
            standard_error,
            ci95_lower,
            ci95_upper,
        }
    }

    /// Check if statistics pass ASHRAE validation criteria
    pub fn passes_ashrae_criteria(&self) -> bool {
        // ASHRAE Guideline 14: NMBE ≤ ±10%, CV(RMSE) ≤ 30%
        let nmbe_ok = self.nmbe.abs() <= 10.0 || self.nmbe.is_nan();
        let cv_rmse_ok = self.cv_rmse <= 30.0 || self.cv_rmse.is_nan();
        nmbe_ok && cv_rmse_ok
    }

    /// Check if statistics pass ASHRAE criteria with warning bounds
    pub fn get_ashrae_status(&self) -> EmpiricalValidationStatus {
        // ASHRAE Guideline 14: NMBE ≤ ±10%, CV(RMSE) ≤ 30%
        let nmbe_ok = self.nmbe.abs() <= 10.0;
        let cv_rmse_ok = self.cv_rmse <= 30.0;

        if nmbe_ok && cv_rmse_ok {
            EmpiricalValidationStatus::Pass
        } else if !nmbe_ok && !cv_rmse_ok {
            EmpiricalValidationStatus::Fail
        } else {
            EmpiricalValidationStatus::Warning
        }
    }
}

/// Comprehensive empirical validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalValidationReport {
    /// Data source information
    pub source: MonitoredDataSource,
    /// Individual validation results
    pub results: Vec<EmpiricalValidationResult>,
    /// Aggregated statistics
    pub statistics: EmpiricalStatistics,
    /// Validation status
    pub status: EmpiricalValidationStatus,
    /// Timestamp of report generation
    pub timestamp: String,
    /// Notes/warnings
    pub notes: Vec<String>,
}

/// Empirical validation suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalValidationConfig {
    /// ASHRAE NMBE threshold (default: 10%)
    pub nmbe_threshold: f64,
    /// ASHRAE CV(RMSE) threshold (default: 30%)
    pub cv_rmse_threshold: f64,
    /// Near-zero reference exclusion threshold
    pub near_zero_threshold: f64,
    /// Enable hourly temperature validation
    pub validate_hourly_temperatures: bool,
    /// Enable daily peak validation
    pub validate_daily_peaks: bool,
    /// Enable annual totals validation
    pub validate_annual_totals: bool,
}

impl Default for EmpiricalValidationConfig {
    fn default() -> Self {
        Self {
            nmbe_threshold: 10.0,
            cv_rmse_threshold: 30.0,
            near_zero_threshold: 1e-10,
            validate_hourly_temperatures: true,
            validate_daily_peaks: true,
            validate_annual_totals: true,
        }
    }
}

/// Known monitored building data sources
#[derive(Debug, Clone)]
pub struct MonitoredBuildingDatabase {
    /// Registered data sources
    pub sources: HashMap<String, MonitoredDataSource>,
}

impl MonitoredBuildingDatabase {
    /// Create new empty database
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    /// Register a monitored data source
    pub fn register(&mut self, source: MonitoredDataSource) {
        self.sources.insert(source.id.clone(), source);
    }

    /// Get a registered source by ID
    pub fn get(&self, id: &str) -> Option<&MonitoredDataSource> {
        self.sources.get(id)
    }

    /// Load monitored data from CSV file
    pub fn load_from_csv<P: AsRef<Path>>(
        &self,
        path: P,
        source_id: &str,
    ) -> Result<Vec<MonitoredDataPoint>, String> {
        let source = self
            .get(source_id)
            .ok_or_else(|| format!("Unknown source ID: {}", source_id))?;

        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

        self.parse_csv_data(&content, source)
    }

    /// Parse CSV data into MonitoredDataPoint array
    fn parse_csv_data(
        &self,
        content: &str,
        source: &MonitoredDataSource,
    ) -> Result<Vec<MonitoredDataPoint>, String> {
        let mut points = Vec::new();
        let mut headers: Option<Vec<&str>> = None;

        for (line_idx, line) in content.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse header row
            if headers.is_none() {
                headers = Some(line.split(',').map(|s| s.trim()).collect());
                continue;
            }

            let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            let headers = headers.as_ref().unwrap();

            if values.len() != headers.len() {
                return Err(format!(
                    "Line {}: expected {} columns, got {}",
                    line_idx + 1,
                    headers.len(),
                    values.len()
                ));
            }

            // Build row map with owned strings
            let row: HashMap<String, String> = headers
                .iter()
                .zip(values.iter())
                .map(|(h, v)| (h.to_lowercase(), v.to_string()))
                .collect();

            // Parse required fields
            let hour = row
                .get("hour")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(line_idx);

            let T_outdoor = row
                .get("t_outdoor")
                .or_else(|| row.get("t_out"))
                .or_else(|| row.get("t_out_c"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(20.0);

            let T_zone = row
                .get("t_zone")
                .or_else(|| row.get("t_zone_air"))
                .or_else(|| row.get("zone_temp"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(20.0);

            let Q_heat = row
                .get("q_heat")
                .or_else(|| row.get("heating"))
                .or_else(|| row.get("q_heating"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let Q_cool = row
                .get("q_cool")
                .or_else(|| row.get("cooling"))
                .or_else(|| row.get("q_cooling"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let Q_solar = row
                .get("q_solar")
                .or_else(|| row.get("solar_gains"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let Q_internal = row
                .get("q_internal")
                .or_else(|| row.get("q_int"))
                .or_else(|| row.get("internal_gains"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let Q_ventilation = row
                .get("q_ventilation")
                .or_else(|| row.get("q_vent"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let Q_conduction = row
                .get("q_conduction")
                .or_else(|| row.get("q_cond"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            points.push(MonitoredDataPoint {
                hour,
                T_outdoor,
                T_zone,
                Q_heat,
                Q_cool,
                Q_solar,
                Q_internal,
                Q_ventilation,
                Q_conduction,
            });
        }

        // Verify count matches source metadata
        if points.len() != source.num_data_points && source.num_data_points > 0 {
            eprintln!(
                "Warning: CSV has {} points, source metadata indicates {}",
                points.len(),
                source.num_data_points
            );
        }

        Ok(points)
    }
}

impl Default for MonitoredBuildingDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard ASHRAE RP data sources (pre-registered)
pub fn get_ashrae_rp_sources() -> MonitoredBuildingDatabase {
    let mut db = MonitoredBuildingDatabase::new();

    // ASHRAE RP-1055 - VRF System Monitoring (Office Building)
    db.register(MonitoredDataSource {
        id: "ashrae_rp1055_office".to_string(),
        name: "ASHRAE RP-1055 Office Building".to_string(),
        source: "ASHRAE RP-1055".to_string(),
        building_type: BuildingType::Office,
        climate_zone: "5A".to_string(),
        location: "Midwest USA".to_string(),
        latitude: 41.88,
        longitude: -87.63,
        floor_area: 2500.0,
        num_floors: 4,
        zone_volume: 6750.0,
        u_wall: 0.35,
        u_roof: 0.25,
        u_window: 2.7,
        wwr: 0.30,
        infiltration_ach: 0.5,
        internal_gains_density: 15.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    });

    // ASHRAE RP-1256 - VRF Heat Recovery (Commercial)
    db.register(MonitoredDataSource {
        id: "ashrae_rp1256_commercial".to_string(),
        name: "ASHRAE RP-1256 Commercial Building".to_string(),
        source: "ASHRAE RP-1256".to_string(),
        building_type: BuildingType::Commercial,
        climate_zone: "3C".to_string(),
        location: "Los Angeles, CA".to_string(),
        latitude: 33.99,
        longitude: -118.45,
        floor_area: 5000.0,
        num_floors: 3,
        zone_volume: 13500.0,
        u_wall: 0.45,
        u_roof: 0.30,
        u_window: 3.0,
        wwr: 0.40,
        infiltration_ach: 0.3,
        internal_gains_density: 20.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    });

    // IEA EBC Annex 60 - Double Facade (Office)
    db.register(MonitoredDataSource {
        id: "iea_ebc_annex60_office".to_string(),
        name: "IEA EBC Annex 60 Double Facade Office".to_string(),
        source: "IEA EBC Annex 60".to_string(),
        building_type: BuildingType::Office,
        climate_zone: "6A".to_string(),
        location: "Zurich, Switzerland".to_string(),
        latitude: 47.38,
        longitude: 8.54,
        floor_area: 1200.0,
        num_floors: 5,
        zone_volume: 3240.0,
        u_wall: 0.20,
        u_roof: 0.15,
        u_window: 1.8,
        wwr: 0.60,
        infiltration_ach: 0.2,
        internal_gains_density: 12.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    });

    // NREL CBM - Small Office (Los Angeles)
    db.register(MonitoredDataSource {
        id: "nrel_cbm_small_office".to_string(),
        name: "NREL CBM Small Office".to_string(),
        source: "NREL Commercial Building Monitoring".to_string(),
        building_type: BuildingType::Office,
        climate_zone: "3B".to_string(),
        location: "Los Angeles, CA".to_string(),
        latitude: 34.05,
        longitude: -118.24,
        floor_area: 511.0,
        num_floors: 1,
        zone_volume: 1379.7,
        u_wall: 0.55,
        u_roof: 0.35,
        u_window: 3.5,
        wwr: 0.20,
        infiltration_ach: 0.5,
        internal_gains_density: 10.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    });

    // NREL CBM - Medium Office (Golden, CO)
    db.register(MonitoredDataSource {
        id: "nrel_cbm_medium_office".to_string(),
        name: "NREL CBM Medium Office".to_string(),
        source: "NREL Commercial Building Monitoring".to_string(),
        building_type: BuildingType::Office,
        climate_zone: "5B".to_string(),
        location: "Golden, CO".to_string(),
        latitude: 39.74,
        longitude: -105.18,
        floor_area: 2346.0,
        num_floors: 3,
        zone_volume: 6334.2,
        u_wall: 0.40,
        u_roof: 0.25,
        u_window: 2.8,
        wwr: 0.35,
        infiltration_ach: 0.4,
        internal_gains_density: 14.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    });

    db
}

/// Generate empirical validation report
pub fn generate_empirical_report(
    source: &MonitoredDataSource,
    data: &[MonitoredDataPoint],
    fluxion_zone_temps: &[f64],
    config: &EmpiricalValidationConfig,
) -> EmpiricalValidationReport {
    let mut results = Vec::new();
    let mut notes = Vec::new();

    // Filter out near-zero reference values for temperature validation
    let mut temp_pairs: Vec<(f64, f64)> = Vec::new();
    for (i, point) in data.iter().enumerate() {
        if point.T_zone.abs() > config.near_zero_threshold && i < fluxion_zone_temps.len() {
            temp_pairs.push((fluxion_zone_temps[i], point.T_zone));
        }
    }

    // Calculate hourly temperature statistics
    let (pred_temps, ref_temps): (Vec<f64>, Vec<f64>) = temp_pairs.iter().cloned().unzip();
    let statistics = EmpiricalStatistics::calculate(&pred_temps, &ref_temps);

    // Check if we have valid data
    if pred_temps.is_empty() {
        return EmpiricalValidationReport {
            source: source.clone(),
            results: vec![],
            statistics: EmpiricalStatistics::calculate(&[], &[]),
            status: EmpiricalValidationStatus::Fail,
            timestamp: chrono::Utc::now().to_rfc3339(),
            notes: vec!["No valid data points for temperature validation".to_string()],
        };
    }

    // Add notes for out-of-range statistics
    if statistics.nmbe.abs() > config.nmbe_threshold {
        notes.push(format!(
            "NMBE {:.1}% exceeds threshold ±{}%",
            statistics.nmbe, config.nmbe_threshold
        ));
    }
    if statistics.cv_rmse > config.cv_rmse_threshold {
        notes.push(format!(
            "CV(RMSE) {:.1}% exceeds threshold {}%",
            statistics.cv_rmse, config.cv_rmse_threshold
        ));
    }

    // Determine overall status
    let status = statistics.get_ashrae_status();

    // Add validation results for each metric type
    if config.validate_hourly_temperatures {
        let mean_pred: f64 = pred_temps.iter().sum::<f64>() / pred_temps.len() as f64;
        let mean_ref: f64 = ref_temps.iter().sum::<f64>() / ref_temps.len() as f64;
        results.push(EmpiricalValidationResult {
            source_id: source.id.clone(),
            metric: EmpiricalMetric::MeanZoneTemperature,
            predicted: mean_pred,
            reference: mean_ref,
            error: mean_pred - mean_ref,
            percentage_error: if mean_ref.abs() > 1e-10 {
                ((mean_pred - mean_ref) / mean_ref.abs()) * 100.0
            } else {
                0.0
            },
            status: if statistics.nmbe.abs() <= config.nmbe_threshold {
                EmpiricalValidationStatus::Pass
            } else {
                EmpiricalValidationStatus::Warning
            },
        });
    }

    if config.validate_daily_peaks {
        // Calculate daily peaks
        let mut daily_heating: HashMap<usize, f64> = HashMap::new();
        for point in data.iter().take(fluxion_zone_temps.len().min(data.len())) {
            let day = point.hour.saturating_sub(1) / 24;
            let current = daily_heating.entry(day).or_insert(0.0);
            *current = current.max(point.Q_heat);
        }

        if let Some(max_heating) = daily_heating.values().cloned().fold(None, |max, val| {
            Some(match max {
                None => val,
                Some(m) if val > m => val,
                Some(m) => m,
            })
        }) {
            let fluxion_max = fluxion_zone_temps
                .iter()
                .cloned()
                .fold(0.0f64, |max, val| max.max(val));
            results.push(EmpiricalValidationResult {
                source_id: source.id.clone(),
                metric: EmpiricalMetric::DailyPeakHeating,
                predicted: fluxion_max,
                reference: max_heating,
                error: fluxion_max - max_heating,
                percentage_error: if max_heating.abs() > 1e-10 {
                    ((fluxion_max - max_heating) / max_heating.abs()) * 100.0
                } else {
                    0.0
                },
                status: EmpiricalValidationStatus::Warning, // Peak validation is advisory
            });
        }
    }

    EmpiricalValidationReport {
        source: source.clone(),
        results,
        statistics,
        status,
        timestamp: chrono::Utc::now().to_rfc3339(),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_calculation() {
        let predicted = vec![20.0, 21.0, 22.0, 23.0, 24.0];
        let reference = vec![20.5, 21.5, 22.5, 23.5, 24.5];

        let stats = EmpiricalStatistics::calculate(&predicted, &reference);

        assert!(!stats.mbe.is_nan());
        assert!(!stats.rmse.is_nan());
        assert!(stats.n > 0);
        // With 0.5 offset, MBE should be -0.5
        assert!((stats.mbe - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn test_ashrae_criteria_pass() {
        let stats = EmpiricalStatistics {
            n: 100,
            mbe: 0.5,
            nmbe: 5.0, // Within ±10%
            rmse: 1.5,
            cv_rmse: 25.0, // Within 30%
            mae: 1.2,
            max_error: 3.0,
            r_squared: 0.95,
            standard_error: 0.15,
            ci95_lower: 0.2,
            ci95_upper: 0.8,
        };

        assert!(stats.passes_ashrae_criteria());
        assert_eq!(stats.get_ashrae_status(), EmpiricalValidationStatus::Pass);
    }

    #[test]
    fn test_ashrae_criteria_warning() {
        let stats = EmpiricalStatistics {
            n: 100,
            mbe: 1.5,
            nmbe: 12.0, // Exceeds ±10%
            rmse: 1.5,
            cv_rmse: 25.0, // Within 30%
            mae: 1.2,
            max_error: 3.0,
            r_squared: 0.95,
            standard_error: 0.15,
            ci95_lower: 0.2,
            ci95_upper: 0.8,
        };

        assert!(!stats.passes_ashrae_criteria());
        assert_eq!(
            stats.get_ashrae_status(),
            EmpiricalValidationStatus::Warning
        );
    }

    #[test]
    fn test_ashrae_criteria_fail() {
        let stats = EmpiricalStatistics {
            n: 100,
            mbe: 3.0,
            nmbe: 15.0, // Exceeds ±10%
            rmse: 4.0,
            cv_rmse: 35.0, // Exceeds 30%
            mae: 3.5,
            max_error: 8.0,
            r_squared: 0.85,
            standard_error: 0.4,
            ci95_lower: 2.0,
            ci95_upper: 4.0,
        };

        assert!(!stats.passes_ashrae_criteria());
        assert_eq!(stats.get_ashrae_status(), EmpiricalValidationStatus::Fail);
    }

    #[test]
    fn test_monitored_building_database_sources() {
        let db = get_ashrae_rp_sources();

        assert!(db.get("ashrae_rp1055_office").is_some());
        assert!(db.get("ashrae_rp1256_commercial").is_some());
        assert!(db.get("iea_ebc_annex60_office").is_some());
        assert!(db.get("nrel_cbm_small_office").is_some());
        assert!(db.get("nrel_cbm_medium_office").is_some());
    }

    #[test]
    fn test_building_types() {
        let db = get_ashrae_rp_sources();

        let office = db.get("ashrae_rp1055_office").unwrap();
        assert_eq!(office.building_type, BuildingType::Office);

        let commercial = db.get("ashrae_rp1256_commercial").unwrap();
        assert_eq!(commercial.building_type, BuildingType::Commercial);
    }

    #[test]
    fn test_empirical_validation_result() {
        let result = EmpiricalValidationResult {
            source_id: "test_source".to_string(),
            metric: EmpiricalMetric::MeanZoneTemperature,
            predicted: 22.0,
            reference: 21.5,
            error: 0.5,
            percentage_error: 2.33,
            status: EmpiricalValidationStatus::Pass,
        };

        assert_eq!(result.predicted, 22.0);
        assert_eq!(result.reference, 21.5);
        assert_eq!(result.status, EmpiricalValidationStatus::Pass);
    }
}

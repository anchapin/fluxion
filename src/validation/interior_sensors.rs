//! Interior Temperature Sensor Data Pipeline
//!
//! Ingests sub-hourly interior temperature sensor logs for empirical
//! validation of zone temperature predictions. Captures per-sensor
//! metadata (location, accuracy, mounting type) and time-series readings
//! so that ground-truth interior temperatures can be compared directly
//! against Fluxion simulation outputs.
//!
//! # Data Format
//!
//! CSV with columns:
//! - `timestamp` — ISO-8601 or epoch seconds
//! - `sensor_id` — unique sensor identifier
//! - `temperature_c` — measured interior temperature [°C]
//!
//! Optional columns: `zone_id`, `relative_humidity_pct`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Physical mounting location of the sensor inside the building
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorPlacement {
    /// Mounted on interior wall surface
    InteriorWall,
    /// Suspended in room air (free-air)
    FreeAir,
    /// Mounted on ceiling
    Ceiling,
    /// Mounted near floor
    FloorLevel,
    /// Mounted on exterior wall interior surface
    ExteriorWallInterior,
    /// Other / unknown placement
    Other,
}

impl SensorPlacement {
    /// Parse from common string variants found in CSV data
    pub fn from_str_loose(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("wall") && lower.contains("ext") {
            Self::ExteriorWallInterior
        } else if lower.contains("wall") {
            Self::InteriorWall
        } else if lower.contains("free") || lower.contains("air") || lower.contains("room") {
            Self::FreeAir
        } else if lower.contains("ceil") || lower.contains("roof") {
            Self::Ceiling
        } else if lower.contains("floor") {
            Self::FloorLevel
        } else {
            Self::Other
        }
    }
}

/// Metadata for a single interior temperature sensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteriorSensorMeta {
    /// Unique sensor identifier (e.g. "TH-001")
    pub sensor_id: String,
    /// Zone this sensor monitors
    pub zone_id: String,
    /// Physical placement type
    pub placement: SensorPlacement,
    /// Manufacturer / model string
    pub model: String,
    /// Measurement accuracy [±°C]
    pub accuracy_c: f64,
    /// Measurement resolution [°C]
    pub resolution_c: f64,
    /// Height above floor [m]
    pub mounting_height_m: f64,
    /// Sampling interval that the sensor was configured for [seconds]
    pub configured_interval_s: u64,
    /// Calibration date (ISO-8601), empty if uncalibrated
    pub calibration_date: String,
    /// Free-text notes
    pub notes: String,
}

impl InteriorSensorMeta {
    /// Create a new sensor metadata record with sensible defaults
    pub fn new(sensor_id: &str, zone_id: &str) -> Self {
        Self {
            sensor_id: sensor_id.to_string(),
            zone_id: zone_id.to_string(),
            placement: SensorPlacement::FreeAir,
            model: String::new(),
            accuracy_c: 0.2,
            resolution_c: 0.01,
            mounting_height_m: 1.1,
            configured_interval_s: 300,
            calibration_date: String::new(),
            notes: String::new(),
        }
    }
}

/// A single interior temperature reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteriorSensorReading {
    /// Timestamp as epoch seconds
    pub timestamp: f64,
    /// Sensor that produced this reading
    pub sensor_id: String,
    /// Measured interior temperature [°C]
    pub temperature_c: f64,
    /// Optional relative humidity [%]
    pub relative_humidity_pct: Option<f64>,
}

/// Complete dataset of interior sensor readings with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteriorSensorDataset {
    /// Human-readable dataset name
    pub name: String,
    /// Source description (instrument, research project, etc.)
    pub source: String,
    /// Per-sensor metadata keyed by sensor_id
    pub sensors: HashMap<String, InteriorSensorMeta>,
    /// All readings across all sensors, chronologically sorted
    pub readings: Vec<InteriorSensorReading>,
}

impl InteriorSensorDataset {
    /// Create an empty dataset
    pub fn new(name: &str, source: &str) -> Self {
        Self {
            name: name.to_string(),
            source: source.to_string(),
            sensors: HashMap::new(),
            readings: Vec::new(),
        }
    }

    /// Register a sensor's metadata
    pub fn register_sensor(&mut self, meta: InteriorSensorMeta) {
        self.sensors.insert(meta.sensor_id.clone(), meta);
    }

    /// Add a batch of readings; auto-rejects readings for unknown sensors
    pub fn add_readings(&mut self, readings: Vec<InteriorSensorReading>) -> usize {
        let mut accepted = 0;
        for r in readings {
            if self.sensors.contains_key(&r.sensor_id) {
                self.readings.push(r);
                accepted += 1;
            }
        }
        self.readings.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        accepted
    }

    /// Return readings for a specific sensor
    pub fn readings_for_sensor(&self, sensor_id: &str) -> Vec<&InteriorSensorReading> {
        self.readings
            .iter()
            .filter(|r| r.sensor_id == sensor_id)
            .collect()
    }

    /// Return the mean temperature across all readings
    pub fn mean_temperature(&self) -> Option<f64> {
        if self.readings.is_empty() {
            return None;
        }
        let sum: f64 = self.readings.iter().map(|r| r.temperature_c).sum();
        Some(sum / self.readings.len() as f64)
    }

    /// Return per-sensor mean temperatures
    pub fn per_sensor_means(&self) -> HashMap<String, f64> {
        let mut sums: HashMap<String, f64> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for r in &self.readings {
            *sums.entry(r.sensor_id.clone()).or_insert(0.0) += r.temperature_c;
            *counts.entry(r.sensor_id.clone()).or_insert(0) += 1;
        }
        sums.into_iter()
            .zip(counts.into_iter())
            .map(|((k, sum), (_, cnt))| (k, sum / cnt as f64))
            .collect()
    }

    /// Number of readings
    pub fn len(&self) -> usize {
        self.readings.len()
    }

    /// Whether the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.readings.is_empty()
    }
}

/// Loader for interior sensor CSV logs
pub struct InteriorSensorLoader;

impl InteriorSensorLoader {
    /// Load sensor metadata from a CSV file.
    ///
    /// Expected columns: `sensor_id`, `zone_id`, `placement`, `model`,
    /// `accuracy_c`, `resolution_c`, `mounting_height_m`,
    /// `configured_interval_s`, `calibration_date`, `notes`.
    pub fn load_metadata<P: AsRef<Path>>(path: P) -> Result<Vec<InteriorSensorMeta>, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read metadata: {}", e))?;
        Self::parse_metadata_csv(&content)
    }

    /// Parse metadata CSV content into sensor metadata records
    pub fn parse_metadata_csv(content: &str) -> Result<Vec<InteriorSensorMeta>, String> {
        let mut sensors = Vec::new();
        let mut headers: Option<Vec<String>> = None;

        for (line_idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if headers.is_none() {
                headers = Some(line.split(',').map(|s| s.trim().to_lowercase()).collect());
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

            let row: HashMap<&str, &str> = headers
                .iter()
                .zip(values.iter())
                .map(|(h, v)| (h.as_str(), *v))
                .collect();

            let sensor_id = row
                .get("sensor_id")
                .ok_or_else(|| format!("Line {}: missing sensor_id", line_idx + 1))?
                .to_string();
            let zone_id = row
                .get("zone_id")
                .ok_or_else(|| format!("Line {}: missing zone_id", line_idx + 1))?
                .to_string();

            let placement = row
                .get("placement")
                .map(|s| SensorPlacement::from_str_loose(s))
                .unwrap_or(SensorPlacement::FreeAir);

            let model = row
                .get("model")
                .map_or("", |v| v)
                .to_string();

            let accuracy_c = row
                .get("accuracy_c")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.2);

            let resolution_c = row
                .get("resolution_c")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.01);

            let mounting_height_m = row
                .get("mounting_height_m")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1.1);

            let configured_interval_s = row
                .get("configured_interval_s")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(300);

            let calibration_date = row
                .get("calibration_date")
                .map_or("", |v| v)
                .to_string();

            let notes = row
                .get("notes")
                .map_or("", |v| v)
                .to_string();

            sensors.push(InteriorSensorMeta {
                sensor_id,
                zone_id,
                placement,
                model,
                accuracy_c,
                resolution_c,
                mounting_height_m,
                configured_interval_s,
                calibration_date,
                notes,
            });
        }

        Ok(sensors)
    }

    /// Load readings from a CSV file.
    ///
    /// Expected columns: `timestamp`, `sensor_id`, `temperature_c`.
    /// Optional: `relative_humidity_pct`.
    /// Timestamps may be ISO-8601 strings or numeric epoch seconds.
    pub fn load_readings<P: AsRef<Path>>(path: P) -> Result<Vec<InteriorSensorReading>, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read readings: {}", e))?;
        Self::parse_readings_csv(&content)
    }

    /// Parse readings CSV content
    pub fn parse_readings_csv(content: &str) -> Result<Vec<InteriorSensorReading>, String> {
        let mut readings = Vec::new();
        let mut headers: Option<Vec<String>> = None;

        for (line_idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if headers.is_none() {
                headers = Some(line.split(',').map(|s| s.trim().to_lowercase()).collect());
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

            let row: HashMap<&str, &str> = headers
                .iter()
                .zip(values.iter())
                .map(|(h, v)| (h.as_str(), *v))
                .collect();

            let timestamp = row
                .get("timestamp")
                .ok_or_else(|| format!("Line {}: missing timestamp", line_idx + 1))?;

            // Parse timestamp: try numeric first, then ISO-8601
            let timestamp_f64 = if let Ok(v) = timestamp.parse::<f64>() {
                v
            } else if let Some(v) = parse_iso8601_approx(timestamp) {
                v
            } else {
                return Err(format!(
                    "Line {}: invalid timestamp '{}'",
                    line_idx + 1,
                    timestamp
                ));
            };

            let sensor_id = row
                .get("sensor_id")
                .ok_or_else(|| format!("Line {}: missing sensor_id", line_idx + 1))?
                .to_string();

            let temperature_c = row
                .get("temperature_c")
                .or_else(|| row.get("temp_c"))
                .or_else(|| row.get("temperature"))
                .ok_or_else(|| format!("Line {}: missing temperature_c", line_idx + 1))?
                .parse::<f64>()
                .map_err(|e| format!("Line {}: invalid temperature: {}", line_idx + 1, e))?;

            let relative_humidity_pct = row
                .get("relative_humidity_pct")
                .or_else(|| row.get("humidity_pct"))
                .or_else(|| row.get("rh_pct"))
                .and_then(|s| s.parse::<f64>().ok());

            readings.push(InteriorSensorReading {
                timestamp: timestamp_f64,
                sensor_id,
                temperature_c,
                relative_humidity_pct,
            });
        }

        Ok(readings)
    }

    /// Load a complete dataset from a directory containing
    /// `metadata.csv` and `readings.csv`.
    pub fn load_dataset<P: AsRef<Path>>(
        dir: P,
        name: &str,
        source: &str,
    ) -> Result<InteriorSensorDataset, String> {
        let dir = dir.as_ref();
        let meta_path = dir.join("metadata.csv");
        let readings_path = dir.join("readings.csv");

        let mut dataset = InteriorSensorDataset::new(name, source);

        // Load and register sensor metadata
        let sensors = Self::load_metadata(&meta_path)?;
        for s in sensors {
            dataset.register_sensor(s);
        }

        // Load readings
        let readings = Self::load_readings(&readings_path)?;
        dataset.add_readings(readings);

        Ok(dataset)
    }

    /// Build a `MonitoredBuildingDatabase` entry from a sensor dataset,
    /// using the mean interior temperature as the zone temperature
    /// and averaging the per-sensor metadata into a single source record.
    pub fn dataset_to_source(dataset: &InteriorSensorDataset) -> Option<crate::validation::empirical::MonitoredDataSource> {
        if dataset.sensors.is_empty() {
            return None;
        }

        // Collect zone ids
        let _zone_ids: Vec<&str> = dataset.sensors.values().map(|s| s.zone_id.as_str()).collect();

        // Average configured interval
        let avg_interval_s: f64 = dataset
            .sensors
            .values()
            .map(|s| s.configured_interval_s as f64)
            .sum::<f64>()
            / dataset.sensors.len() as f64;
        let time_resolution_hours = avg_interval_s / 3600.0;

        Some(crate::validation::empirical::MonitoredDataSource {
            id: format!("interior_sensor_{}", dataset.name.replace(' ', "_")),
            name: dataset.name.clone(),
            source: dataset.source.clone(),
            building_type: crate::validation::empirical::BuildingType::Office,
            climate_zone: String::new(),
            location: String::new(),
            latitude: 0.0,
            longitude: 0.0,
            floor_area: 0.0,
            num_floors: 0,
            zone_volume: 0.0,
            u_wall: 0.0,
            u_roof: 0.0,
            u_window: 0.0,
            wwr: 0.0,
            infiltration_ach: 0.0,
            internal_gains_density: 0.0,
            time_resolution_hours,
            num_data_points: dataset.readings.len(),
        })
    }

    /// Convert sensor readings into `MonitoredDataPoint` entries suitable
    /// for the existing empirical validation pipeline. Readings are
    /// aggregated into one point per time step using the zone-mean
    /// interior temperature.
    pub fn dataset_to_data_points(
        dataset: &InteriorSensorDataset,
    ) -> Vec<crate::validation::empirical::MonitoredDataPoint> {
        // Group readings by timestamp
        let mut by_ts: HashMap<u64, Vec<&InteriorSensorReading>> = HashMap::new();
        for r in &dataset.readings {
            let ts_key = r.timestamp as u64;
            by_ts.entry(ts_key).or_default().push(r);
        }

        let mut points: Vec<crate::validation::empirical::MonitoredDataPoint> = by_ts
            .into_iter()
            .enumerate()
            .map(|(idx, (_ts, mut readings_at_ts))| {
                readings_at_ts.sort_by(|a, b| {
                    a.sensor_id
                        .partial_cmp(&b.sensor_id)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mean_temp: f64 = readings_at_ts.iter().map(|r| r.temperature_c).sum::<f64>()
                    / readings_at_ts.len() as f64;

                crate::validation::empirical::MonitoredDataPoint {
                    hour: idx,
                    T_outdoor: 0.0,
                    T_zone: mean_temp,
                    Q_heat: 0.0,
                    Q_cool: 0.0,
                    Q_solar: 0.0,
                    Q_internal: 0.0,
                    Q_ventilation: 0.0,
                    Q_conduction: 0.0,
                }
            })
            .collect();

        points.sort_by_key(|p| p.hour);
        points
    }
}

/// Approximate parsing of ISO-8601 datetime strings to epoch seconds.
/// Handles common formats: `2024-01-15T10:30:00Z`, `2024-01-15 10:30:00`.
/// Returns `None` for unrecognized formats.
fn parse_iso8601_approx(s: &str) -> Option<f64> {
    let s = s.trim().replace('Z', "+00:00");
    // Split date and time
    let (date_part, time_part) = if let Some(pos) = s.find('T') {
        (&s[..pos], Some(&s[pos + 1..]))
    } else if let Some(pos) = s.find(' ') {
        (&s[..pos], Some(&s[pos + 1..]))
    } else {
        (s.as_str(), None)
    };

    let date_parts: Vec<&str> = date_part.split('-').collect();
    if date_parts.len() != 3 {
        return None;
    }
    let year: i64 = date_parts[0].parse().ok()?;
    let month: i64 = date_parts[1].parse().ok()?;
    let day: i64 = date_parts[2].parse().ok()?;

    let mut total_seconds = 0.0;
    // Days to months (non-leap)
    let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    for y in 1970..year {
        total_seconds += if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
            366.0 * 86400.0
        } else {
            365.0 * 86400.0
        };
    }
    for m in 1..month {
        let days = if m == 2 && leap { 29 } else { days_in_month[m as usize] };
        total_seconds += days as f64 * 86400.0;
    }
    total_seconds += (day - 1) as f64 * 86400.0;

    if let Some(tp) = time_part {
        // Strip timezone suffix for simple parsing
        let tp = tp
            .split('+')
            .next()
            .and_then(|t| t.split('-').next())
            .unwrap_or(tp);
        let time_parts: Vec<&str> = tp.split(':').collect();
        if time_parts.len() >= 2 {
            let hours: f64 = time_parts[0].parse().ok()?;
            let minutes: f64 = time_parts[1].parse().ok()?;
            let seconds: f64 = if time_parts.len() > 2 {
                time_parts[2].parse().ok().unwrap_or(0.0)
            } else {
                0.0
            };
            total_seconds += hours * 3600.0 + minutes * 60.0 + seconds;
        }
    }

    Some(total_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_meta_defaults() {
        let meta = InteriorSensorMeta::new("TH-001", "zone_1");
        assert_eq!(meta.sensor_id, "TH-001");
        assert_eq!(meta.zone_id, "zone_1");
        assert_eq!(meta.placement, SensorPlacement::FreeAir);
        assert!((meta.accuracy_c - 0.2).abs() < 1e-10);
        assert!((meta.resolution_c - 0.01).abs() < 1e-10);
        assert!((meta.mounting_height_m - 1.1).abs() < 1e-10);
        assert_eq!(meta.configured_interval_s, 300);
    }

    #[test]
    fn test_sensor_placement_from_str() {
        assert_eq!(SensorPlacement::from_str_loose("Interior Wall"), SensorPlacement::InteriorWall);
        assert_eq!(SensorPlacement::from_str_loose("free air"), SensorPlacement::FreeAir);
        assert_eq!(SensorPlacement::from_str_loose("ceiling"), SensorPlacement::Ceiling);
        assert_eq!(SensorPlacement::from_str_loose("floor level"), SensorPlacement::FloorLevel);
        assert_eq!(
            SensorPlacement::from_str_loose("Exterior Wall Interior"),
            SensorPlacement::ExteriorWallInterior
        );
    }

    #[test]
    fn test_dataset_register_and_read() {
        let mut ds = InteriorSensorDataset::new("test", "unit_test");
        ds.register_sensor(InteriorSensorMeta::new("S1", "Z1"));
        ds.register_sensor(InteriorSensorMeta::new("S2", "Z1"));

        let readings = vec![
            InteriorSensorReading { timestamp: 0.0, sensor_id: "S1".into(), temperature_c: 21.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 0.0, sensor_id: "S2".into(), temperature_c: 22.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 300.0, sensor_id: "S1".into(), temperature_c: 21.5, relative_humidity_pct: None },
        ];

        let accepted = ds.add_readings(readings);
        assert_eq!(accepted, 3);
        assert_eq!(ds.len(), 3);
    }

    #[test]
    fn test_dataset_rejects_unknown_sensor() {
        let mut ds = InteriorSensorDataset::new("test", "unit_test");
        ds.register_sensor(InteriorSensorMeta::new("S1", "Z1"));

        let readings = vec![
            InteriorSensorReading { timestamp: 0.0, sensor_id: "UNKNOWN".into(), temperature_c: 21.0, relative_humidity_pct: None },
        ];

        let accepted = ds.add_readings(readings);
        assert_eq!(accepted, 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_per_sensor_means() {
        let mut ds = InteriorSensorDataset::new("test", "unit_test");
        ds.register_sensor(InteriorSensorMeta::new("S1", "Z1"));
        ds.register_sensor(InteriorSensorMeta::new("S2", "Z1"));

        let readings = vec![
            InteriorSensorReading { timestamp: 0.0, sensor_id: "S1".into(), temperature_c: 20.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 0.0, sensor_id: "S2".into(), temperature_c: 22.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 300.0, sensor_id: "S1".into(), temperature_c: 22.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 300.0, sensor_id: "S2".into(), temperature_c: 24.0, relative_humidity_pct: None },
        ];

        ds.add_readings(readings);

        let means = ds.per_sensor_means();
        assert!((means["S1"] - 21.0).abs() < 1e-10);
        assert!((means["S2"] - 23.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_temperature() {
        let mut ds = InteriorSensorDataset::new("test", "unit_test");
        ds.register_sensor(InteriorSensorMeta::new("S1", "Z1"));

        let readings = vec![
            InteriorSensorReading { timestamp: 0.0, sensor_id: "S1".into(), temperature_c: 20.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 1.0, sensor_id: "S1".into(), temperature_c: 24.0, relative_humidity_pct: None },
        ];

        ds.add_readings(readings);
        assert!((ds.mean_temperature().unwrap() - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_dataset() {
        let ds = InteriorSensorDataset::new("empty", "unit_test");
        assert!(ds.is_empty());
        assert!(ds.mean_temperature().is_none());
    }

    #[test]
    fn test_parse_metadata_csv() {
        let csv = "\
sensor_id,zone_id,placement,model,accuracy_c,resolution_c,mounting_height_m,configured_interval_s,calibration_date,notes
TH-001,zone_1,Interior Wall,Sensirion SHT31,0.2,0.01,1.1,300,2024-01-15,Primary sensor
TH-002,zone_2,Free Air,HS-S1,0.3,0.02,1.5,600,,Backup";
        let sensors = InteriorSensorLoader::parse_metadata_csv(csv).unwrap();
        assert_eq!(sensors.len(), 2);
        assert_eq!(sensors[0].sensor_id, "TH-001");
        assert_eq!(sensors[0].placement, SensorPlacement::InteriorWall);
        assert!((sensors[0].accuracy_c - 0.2).abs() < 1e-10);
        assert_eq!(sensors[1].sensor_id, "TH-002");
        assert_eq!(sensors[1].zone_id, "zone_2");
    }

    #[test]
    fn test_parse_readings_csv() {
        let csv = "\
timestamp,sensor_id,temperature_c,relative_humidity_pct
1700000000,TH-001,21.5,45.0
1700000300,TH-001,21.6,44.8
1700000000,TH-002,22.1,42.0";
        let readings = InteriorSensorLoader::parse_readings_csv(csv).unwrap();
        assert_eq!(readings.len(), 3);
        assert!((readings[0].temperature_c - 21.5).abs() < 1e-10);
        assert_eq!(readings[0].relative_humidity_pct, Some(45.0));
    }

    #[test]
    fn test_parse_readings_csv_iso8601() {
        let csv = "\
timestamp,sensor_id,temperature_c
2024-01-15T10:30:00Z,TH-001,21.5
2024-01-15T11:00:00Z,TH-001,22.0";
        let readings = InteriorSensorLoader::parse_readings_csv(csv).unwrap();
        assert_eq!(readings.len(), 2);
        // Second timestamp should be after first
        assert!(readings[1].timestamp > readings[0].timestamp);
    }

    #[test]
    fn test_dataset_to_data_points() {
        let mut ds = InteriorSensorDataset::new("test", "unit_test");
        ds.register_sensor(InteriorSensorMeta::new("S1", "Z1"));
        ds.register_sensor(InteriorSensorMeta::new("S2", "Z1"));

        let readings = vec![
            InteriorSensorReading { timestamp: 0.0, sensor_id: "S1".into(), temperature_c: 20.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 0.0, sensor_id: "S2".into(), temperature_c: 22.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 300.0, sensor_id: "S1".into(), temperature_c: 21.0, relative_humidity_pct: None },
            InteriorSensorReading { timestamp: 300.0, sensor_id: "S2".into(), temperature_c: 23.0, relative_humidity_pct: None },
        ];
        ds.add_readings(readings);

        let points = InteriorSensorLoader::dataset_to_data_points(&ds);
        assert_eq!(points.len(), 2);
        // First point: mean of 20.0 and 22.0 = 21.0
        assert!((points[0].T_zone - 21.0).abs() < 1e-10);
        // Second point: mean of 21.0 and 23.0 = 22.0
        assert!((points[1].T_zone - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_dataset_to_source() {
        let mut ds = InteriorSensorDataset::new("My Building", "field study");
        ds.register_sensor(InteriorSensorMeta::new("S1", "Z1"));
        ds.register_sensor(InteriorSensorMeta::new("S2", "Z1"));

        let source = InteriorSensorLoader::dataset_to_source(&ds).unwrap();
        assert_eq!(source.id, "interior_sensor_My_Building");
        assert_eq!(source.name, "My Building");
        assert!((source.time_resolution_hours - 300.0 / 3600.0).abs() < 1e-10);
    }
}

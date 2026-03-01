//! ML Surrogate Data Collector for ASHRAE 140 Validation
//!
//! This module connects Rust data collection hooks to automatically generate
//! training data for ONNX models upon successful ASHRAE 140 validation runs.
//!
//! # Purpose
//!
//! Ensures AI/ML models are physically valid for use as surrogates in the
//! fluxion engine by collecting data from successful validation runs.
//!
//! # Usage
//!
//! ```rust,ignore
//! use fluxion::validation::ml_data_collector::MLDataCollector;
//!
//! let mut collector = MLDataCollector::new("data/training");
//! collector.enable_auto_collection(true);
//! // Run ASHRAE 140 validation...
//! collector.collect_from_validation_results(&case_id, &results);
//! collector.save_to_disk()?;
//! ```

use super::ashrae_140_cases::CaseSpec;
use super::diagnostic::HourlyData;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Configuration for ML data collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLDataCollectorConfig {
    /// Enable automatic data collection after successful validations
    pub auto_collect: bool,
    /// Output directory for collected training data
    pub output_dir: String,
    /// Minimum R² threshold for accepting validation data
    pub min_r_squared: f64,
    /// Maximum relative error threshold
    pub max_relative_error: f64,
    /// Collect hourly data for detailed training
    pub collect_hourly: bool,
    /// Collect energy breakdown for physics-informed training
    pub collect_energy_breakdown: bool,
}

impl Default for MLDataCollectorConfig {
    fn default() -> Self {
        Self {
            auto_collect: true,
            output_dir: "data/training".to_string(),
            min_r_squared: 0.98,
            max_relative_error: 0.05,
            collect_hourly: true,
            collect_energy_breakdown: true,
        }
    }
}

/// Training sample collected from a validation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Case identifier
    pub case_id: String,
    /// Hour index (0-8759)
    pub hour: usize,
    /// Input features for ML model
    #[serde(rename = "inputs")]
    pub features: TrainingFeatures,
    /// Target outputs (actual physics engine results)
    #[serde(rename = "targets")]
    pub targets: TrainingTargets,
}

/// Input features for surrogate model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFeatures {
    /// Outdoor air temperature (°C)
    pub outdoor_temp: f64,
    /// Zone temperatures (°C)
    pub zone_temps: Vec<f64>,
    /// Solar gains per zone (W/m²)
    pub solar_gains: Vec<f64>,
    /// Internal loads per zone (W/m²)
    pub internal_loads: Vec<f64>,
    /// HVAC heating setpoint (°C)
    pub heating_setpoint: f64,
    /// HVAC cooling setpoint (°C)
    pub cooling_setpoint: f64,
    /// Hour of day (0-23)
    pub hour_of_day: u8,
    /// Day of year (1-366)
    pub day_of_year: u16,
    /// Month (1-12)
    pub month: u8,
    /// U-value of building envelope (W/m²K)
    pub u_value: f64,
    /// Window to wall ratio
    pub wwr: f64,
}

/// Target outputs for supervised training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTargets {
    /// Required heating load per zone (W/m²)
    pub heating_load: Vec<f64>,
    /// Required cooling load per zone (W/m²)
    pub cooling_load: Vec<f64>,
    /// Net HVAC power (W)
    pub total_hvac_power: f64,
}

/// Validation metrics for assessing training data quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// R² score against deterministic physics
    pub r_squared: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Maximum absolute error
    pub max_error: f64,
    /// Mean relative error
    pub mean_relative_error: f64,
}

/// ML data collector for ASHRAE 140 validation
pub struct MLDataCollector {
    config: MLDataCollectorConfig,
    collected_samples: Vec<TrainingSample>,
    case_metrics: HashMap<String, ValidationMetrics>,
    hourly_data_buffer: HashMap<String, Vec<HourlyData>>,
}

impl MLDataCollector {
    pub fn new(output_dir: &str) -> Self {
        Self::with_config(MLDataCollectorConfig {
            output_dir: output_dir.to_string(),
            ..Default::default()
        })
    }

    pub fn with_config(config: MLDataCollectorConfig) -> Self {
        Self {
            config,
            collected_samples: Vec::new(),
            case_metrics: HashMap::new(),
            hourly_data_buffer: HashMap::new(),
        }
    }

    pub fn enable_auto_collection(&mut self, enabled: bool) {
        self.config.auto_collect = enabled;
    }

    pub fn collect_from_hourly_data(
        &mut self,
        case_id: &str,
        spec: &CaseSpec,
        hourly_data: &[HourlyData],
    ) -> Result<usize, String> {
        if !self.config.auto_collect {
            return Ok(0);
        }

        let u_value = spec.construction.wall.u_value(None, None);
        let wwr = spec
            .windows
            .first()
            .and_then(|w| w.first())
            .map(|win_area| win_area.area / spec.geometry[0].wall_area())
            .unwrap_or(0.2);

        let heating_setpoint = spec
            .hvac
            .first()
            .map(|h| h.heating_setpoint)
            .unwrap_or(20.0);
        let cooling_setpoint = spec
            .hvac
            .first()
            .map(|h| h.cooling_setpoint)
            .unwrap_or(27.0);

        let mut samples_collected = 0;

        for data in hourly_data {
            let num_zones = data.zone_temps.len();

            let features = TrainingFeatures {
                outdoor_temp: data.outdoor_temp,
                zone_temps: data.zone_temps.clone(),
                solar_gains: data
                    .solar_gains
                    .iter()
                    .zip(&data.zone_temps)
                    .map(|(g, temp)| {
                        let floor_area = spec.geometry.get(0).map_or(20.0, |g| g.floor_area());
                        g / floor_area
                    })
                    .collect(),
                internal_loads: data.internal_loads.clone(),
                heating_setpoint,
                cooling_setpoint,
                hour_of_day: data.hour_of_day as u8,
                day_of_year: (data.month as u16 * 32 + data.day as u16),
                month: data.month as u8,
                u_value,
                wwr,
            };

            let targets = TrainingTargets {
                heating_load: data.hvac_heating.iter().map(|&h| h / 1000.0).collect(),
                cooling_load: data.hvac_cooling.iter().map(|&c| (-c) / 1000.0).collect(),
                total_hvac_power: data.total_hvac_power(),
            };

            let sample = TrainingSample {
                case_id: case_id.to_string(),
                hour: data.hour,
                features,
                targets,
            };

            self.collected_samples.push(sample);
            samples_collected += 1;
        }

        Ok(samples_collected)
    }

    pub fn record_validation_metrics(&mut self, case_id: &str, metrics: ValidationMetrics) {
        self.case_metrics.insert(case_id.to_string(), metrics);
    }

    pub fn compute_validation_metrics(predicted: &[f64], actual: &[f64]) -> ValidationMetrics {
        assert_eq!(predicted.len(), actual.len(), "Arrays must be same length");

        let n = predicted.len() as f64;

        let mean_actual = actual.iter().sum::<f64>() / n;
        let ss_tot = actual
            .iter()
            .map(|&y| (y - mean_actual).powi(2))
            .sum::<f64>();

        let mae = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).abs())
            .sum::<f64>()
            / n;

        let mse = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum::<f64>()
            / n;

        let rmse = mse.sqrt();

        let max_error = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).abs())
            .fold(0.0_f64, f64::max);

        let mean_relative_error = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| {
                if a.abs() > 1e-10 {
                    (p - a).abs() / a.abs()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n;

        let r_squared = if ss_tot > 1e-10 {
            1.0 - (mse * n) / ss_tot
        } else {
            1.0
        };

        ValidationMetrics {
            r_squared,
            mae,
            rmse,
            max_error,
            mean_relative_error,
        }
    }

    pub fn is_data_quality_acceptable(&self, case_id: &str) -> bool {
        if let Some(metrics) = self.case_metrics.get(case_id) {
            metrics.r_squared >= self.config.min_r_squared
                && metrics.mean_relative_error <= self.config.max_relative_error
        } else {
            true
        }
    }

    pub fn save_to_disk(&self) -> Result<(), String> {
        let output_dir = Path::new(&self.config.output_dir);
        if !output_dir.exists() {
            fs::create_dir_all(output_dir)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string();

        let metadata_path = output_dir.join(format!("metadata_{}.json", timestamp));
        let metadata = serde_json::to_string_pretty(&self.config)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
        let metadata_file = File::create(&metadata_path)
            .map_err(|e| format!("Failed to create metadata file: {}", e))?;
        let mut metadata_writer = BufWriter::new(metadata_file);
        metadata_writer
            .write_all(metadata.as_bytes())
            .map_err(|e| format!("Failed to write metadata: {}", e))?;

        let samples_path = output_dir.join(format!("samples_{}.csv", timestamp));
        let samples_file = File::create(&samples_path)
            .map_err(|e| format!("Failed to create samples file: {}", e))?;
        let mut samples_writer = BufWriter::new(samples_file);

        let header = "case_id,hour,outdoor_temp,heating_setpoint,cooling_setpoint,hour_of_day,day_of_year,month,u_value,wwr,heating_load,cooling_load,total_hvac_power\n";
        samples_writer
            .write_all(header.as_bytes())
            .map_err(|e| format!("Failed to write CSV header: {}", e))?;

        for sample in &self.collected_samples {
            let mut line = String::new();

            line.push_str(&sample.case_id);
            line.push_str(&format!(",{}", sample.hour));
            line.push_str(&format!(",{:.4}", sample.features.outdoor_temp));
            line.push_str(&format!(",{:.4}", sample.features.heating_setpoint));
            line.push_str(&format!(",{:.4}", sample.features.cooling_setpoint));
            line.push_str(&format!(",{}", sample.features.hour_of_day));
            line.push_str(&format!(",{}", sample.features.day_of_year));
            line.push_str(&format!(",{}", sample.features.month));
            line.push_str(&format!(",{:.4}", sample.features.u_value));
            line.push_str(&format!(",{:.4}", sample.features.wwr));

            let mean_heating_load: f64 = if sample.targets.heating_load.is_empty() {
                0.0
            } else {
                sample.targets.heating_load.iter().sum::<f64>()
                    / sample.targets.heating_load.len() as f64
            };
            let mean_cooling_load: f64 = if sample.targets.cooling_load.is_empty() {
                0.0
            } else {
                sample.targets.cooling_load.iter().sum::<f64>()
                    / sample.targets.cooling_load.len() as f64
            };

            line.push_str(&format!(",{:.6}", mean_heating_load));
            line.push_str(&format!(",{:.6}", mean_cooling_load));
            line.push_str(&format!(",{:.6}", sample.targets.total_hvac_power));
            line.push('\n');

            samples_writer
                .write_all(line.as_bytes())
                .map_err(|e| format!("Failed to write sample: {}", e))?;
        }

        let metrics_path = output_dir.join(format!("metrics_{}.json", timestamp));
        let metrics_json = serde_json::to_string_pretty(&self.case_metrics)
            .map_err(|e| format!("Failed to serialize metrics: {}", e))?;
        let metrics_file = File::create(&metrics_path)
            .map_err(|e| format!("Failed to create metrics file: {}", e))?;
        let mut metrics_writer = BufWriter::new(metrics_file);
        metrics_writer
            .write_all(metrics_json.as_bytes())
            .map_err(|e| format!("Failed to write metrics: {}", e))?;

        println!(
            "Saved {} training samples to {}",
            self.collected_samples.len(),
            output_dir.display()
        );
        println!("  Samples: {}", samples_path.display());
        println!("  Metrics: {}", metrics_path.display());
        println!("  Metadata: {}", metadata_path.display());

        Ok(())
    }

    pub fn get_sample_count(&self) -> usize {
        self.collected_samples.len()
    }

    pub fn clear(&mut self) {
        self.collected_samples.clear();
        self.case_metrics.clear();
        self.hourly_data_buffer.clear();
    }

    pub fn merge(&mut self, other: MLDataCollector) {
        self.collected_samples.extend(other.collected_samples);
        self.case_metrics.extend(other.case_metrics);
        self.hourly_data_buffer.extend(other.hourly_data_buffer);
    }
}

impl Default for MLDataCollector {
    fn default() -> Self {
        Self::new("data/training")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_validation_metrics_perfect() {
        let predicted = vec![1.0, 2.0, 3.0, 4.0];
        let actual = vec![1.0, 2.0, 3.0, 4.0];

        let metrics = MLDataCollector::compute_validation_metrics(&predicted, &actual);

        assert!((metrics.r_squared - 1.0).abs() < 1e-6);
        assert!((metrics.mae - 0.0).abs() < 1e-6);
        assert!((metrics.rmse - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_validation_metrics_with_error() {
        let predicted = vec![1.1, 2.2, 2.9, 4.1];
        let actual = vec![1.0, 2.0, 3.0, 4.0];

        let metrics = MLDataCollector::compute_validation_metrics(&predicted, &actual);

        assert!(metrics.mae > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.r_squared > 0.9);
    }

    #[test]
    fn test_data_quality_acceptable() {
        let mut collector = MLDataCollector::new("test_output");
        collector.enable_auto_collection(true);

        let metrics = ValidationMetrics {
            r_squared: 0.99,
            mae: 0.01,
            rmse: 0.02,
            max_error: 0.05,
            mean_relative_error: 0.03,
        };

        collector.record_validation_metrics("case_600", metrics);

        assert!(collector.is_data_quality_acceptable("case_600"));
    }

    #[test]
    fn test_data_quality_rejected() {
        let mut collector = MLDataCollector::new("test_output");
        collector.enable_auto_collection(true);

        let metrics = ValidationMetrics {
            r_squared: 0.85,
            mae: 0.5,
            rmse: 0.7,
            max_error: 2.0,
            mean_relative_error: 0.15,
        };

        collector.record_validation_metrics("case_600", metrics);

        assert!(!collector.is_data_quality_acceptable("case_600"));
    }
}

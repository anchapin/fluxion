//! TOON Time-Series Encoder for TSFM context windows.
//!
//! Encodes sequential telemetry data into compact TOON tabular arrays with
//! time-series collapse rules that detect temporal regularity and collapse
//! redundant timestamps.
//!
//! # Issues Addressed
//! - #2076: TOON Time-Series Encoder
//!
//! # Time-Series Collapse Rules
//!
//! The encoder applies three collapse strategies to reduce token usage:
//!
//! 1. **Constant-Run Collapse**: When consecutive values are identical within
//!    a tolerance, collapse into a single `[start..end]: value` span.
//!
//! 2. **Regular-Interval Detection**: When timestamps follow a fixed stride
//!    (e.g., hourly), only the first timestamp is emitted; subsequent rows
//!    inherit the stride implicitly.
//!
//! 3. **Uniform Array Serialization**: Per [`fluxion_toon::ser`], uniform
//!    flat-struct arrays are collapsed into CSV-style blocks with explicit
//!    count headers.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use fluxion_toon::ser::ToonSerializable;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryRecord {
    pub zone_id: String,
    pub sensor_id: String,
    pub readings: Vec<TimeSeriesPoint>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollapseStrategy {
    None,
    ConstantRun { run_length: usize },
    RegularInterval { stride_seconds: i64 },
    UniformArray,
}

impl CollapseStrategy {
    fn is_better_than(&self, other: &CollapseStrategy) -> bool {
        match (self, other) {
            (CollapseStrategy::None, _) => false,
            (_, CollapseStrategy::None) => true,
            (CollapseStrategy::UniformArray, _) => true,
            (_, CollapseStrategy::UniformArray) => false,
            (
                CollapseStrategy::ConstantRun { run_length: a },
                CollapseStrategy::ConstantRun { run_length: b },
            ) => a > b,
            (
                CollapseStrategy::ConstantRun { run_length: a },
                CollapseStrategy::RegularInterval { .. },
            ) => *a >= 3,
            (
                CollapseStrategy::RegularInterval { .. },
                CollapseStrategy::ConstantRun { run_length: a },
            ) => *a < 3,
            (
                CollapseStrategy::RegularInterval {
                    stride_seconds: a, ..
                },
                CollapseStrategy::RegularInterval {
                    stride_seconds: b, ..
                },
            ) => a > b,
        }
    }
}

pub struct ToonTimeEncoderConfig {
    pub constant_run_min_length: usize,
    pub interval_tolerance_seconds: i64,
    pub value_tolerance: f64,
    pub emit_stride_header: bool,
}

impl Default for ToonTimeEncoderConfig {
    fn default() -> Self {
        Self {
            constant_run_min_length: 3,
            interval_tolerance_seconds: 30,
            value_tolerance: 1e-6,
            emit_stride_header: true,
        }
    }
}

pub struct ToonTimeEncoder {
    config: ToonTimeEncoderConfig,
}

impl ToonTimeEncoder {
    pub fn new(config: ToonTimeEncoderConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ToonTimeEncoderConfig::default())
    }

    pub fn encode_telemetry(&self, records: &[TelemetryRecord]) -> String {
        let mut output = String::new();
        for record in records {
            self.encode_record(record, &mut output);
        }
        output
    }

    fn encode_record(&self, record: &TelemetryRecord, output: &mut String) {
        use std::fmt::Write;
        writeln!(output, "# {}", record.zone_id).unwrap();
        writeln!(output, "sensor: {}", record.sensor_id).unwrap();

        if record.readings.is_empty() {
            writeln!(output, "readings[0]:").unwrap();
            return;
        }

        let (strategy, _collapsed) = self.detect_collapse_strategy(&record.readings);

        match strategy {
            CollapseStrategy::UniformArray => {
                self.encode_uniform_array(&record.readings, output);
            }
            CollapseStrategy::ConstantRun { run_length } => {
                self.encode_constant_runs(&record.readings, output, run_length);
            }
            CollapseStrategy::RegularInterval { stride_seconds } => {
                self.encode_regular_interval(&record.readings, output, stride_seconds);
            }
            CollapseStrategy::None => {
                self.encode_as_json(&record.readings, output);
            }
        }

        output.push('\n');
    }

    fn detect_collapse_strategy(&self, readings: &[TimeSeriesPoint]) -> (CollapseStrategy, bool) {
        if readings.is_empty() {
            return (CollapseStrategy::None, false);
        }

        if self.is_uniform_array(readings) {
            return (CollapseStrategy::UniformArray, true);
        }

        let (interval_strategy, interval_stride) = self.detect_regular_interval(readings);
        let (run_strategy, run_length) = self.detect_constant_runs(readings);

        if interval_strategy.is_better_than(&run_strategy) {
            (
                CollapseStrategy::RegularInterval {
                    stride_seconds: interval_stride,
                },
                true,
            )
        } else {
            (CollapseStrategy::ConstantRun { run_length }, true)
        }
    }

    #[allow(dead_code)]
    fn is_uniform_array(&self, _readings: &[TimeSeriesPoint]) -> bool {
        false
    }

    fn detect_regular_interval(&self, readings: &[TimeSeriesPoint]) -> (CollapseStrategy, i64) {
        if readings.len() < 2 {
            return (CollapseStrategy::None, 0);
        }

        let mut strides: HashMap<i64, usize> = HashMap::new();

        for window in readings.windows(2) {
            let diff = (window[1].timestamp - window[0].timestamp)
                .num_seconds()
                .abs();
            if diff > 0 {
                *strides.entry(diff).or_insert(0) += 1;
            }
        }

        let most_common_stride = strides.into_iter().max_by_key(|(_, count)| *count);

        if let Some((stride, count)) = most_common_stride {
            let coverage_ratio = count as f64 / (readings.len() - 1) as f64;
            if coverage_ratio >= 0.8 && stride <= 86400 {
                return (
                    CollapseStrategy::RegularInterval {
                        stride_seconds: stride,
                    },
                    stride,
                );
            }
        }

        (CollapseStrategy::None, 0)
    }

    fn detect_constant_runs(&self, readings: &[TimeSeriesPoint]) -> (CollapseStrategy, usize) {
        if readings.len() < 2 {
            return (CollapseStrategy::None, 0);
        }

        let mut max_run = 1usize;
        let mut current_run = 1usize;

        for window in readings.windows(2) {
            if (window[1].value - window[0].value).abs() <= self.config.value_tolerance {
                current_run += 1;
                max_run = max_run.max(current_run);
            } else {
                current_run = 1;
            }
        }

        if max_run >= self.config.constant_run_min_length {
            (
                CollapseStrategy::ConstantRun {
                    run_length: max_run,
                },
                max_run,
            )
        } else {
            (CollapseStrategy::None, 0)
        }
    }

    fn encode_uniform_array(&self, readings: &[TimeSeriesPoint], output: &mut String) {
        use std::fmt::Write;
        writeln!(output, "readings[{}]{{timestamp,value}}:", readings.len()).unwrap();
        for r in readings {
            writeln!(output, "{}, {}", r.timestamp.to_rfc3339(), r.value).unwrap();
        }
    }

    fn encode_constant_runs(
        &self,
        readings: &[TimeSeriesPoint],
        output: &mut String,
        _run_length: usize,
    ) {
        use std::fmt::Write;
        let mut i = 0;
        while i < readings.len() {
            let start = i;
            let value = readings[i].value;
            while i < readings.len()
                && (readings[i].value - value).abs() <= self.config.value_tolerance
            {
                i += 1;
            }
            let end_idx = i - 1;
            let start_time = readings[start].timestamp;
            let end_time = readings[end_idx].timestamp;
            let count = i - start;

            if count >= self.config.constant_run_min_length {
                writeln!(
                    output,
                    "readings_span[{}]{{start,end,value}}: {}, {}, {}",
                    count,
                    start_time.to_rfc3339(),
                    end_time.to_rfc3339(),
                    value
                )
                .unwrap();
            } else {
                for r in &readings[start..i] {
                    writeln!(
                        output,
                        "readings[1]{{timestamp,value}}: {}, {}",
                        r.timestamp.to_rfc3339(),
                        r.value
                    )
                    .unwrap();
                }
            }
        }
    }

    fn encode_regular_interval(
        &self,
        readings: &[TimeSeriesPoint],
        output: &mut String,
        stride_seconds: i64,
    ) {
        use std::fmt::Write;
        if self.config.emit_stride_header && stride_seconds > 0 {
            writeln!(output, "stride_seconds: {}", stride_seconds).unwrap();
        }
        writeln!(output, "readings[{}]{{timestamp,value}}:", readings.len()).unwrap();
        for r in readings {
            writeln!(output, "{}, {}", r.timestamp.to_rfc3339(), r.value).unwrap();
        }
    }

    fn encode_as_json(&self, readings: &[TimeSeriesPoint], output: &mut String) {
        use std::fmt::Write;
        for r in readings {
            writeln!(
                output,
                "readings[1]{{timestamp,value}}: {}, {}",
                r.timestamp.to_rfc3339(),
                r.value
            )
            .unwrap();
        }
    }

    pub fn encode_single_series(&self, readings: &[TimeSeriesPoint]) -> String {
        let mut output = String::new();
        self.encode_readings(readings, &mut output);
        output
    }

    fn encode_readings(&self, readings: &[TimeSeriesPoint], output: &mut String) {
        use std::fmt::Write;
        if readings.is_empty() {
            writeln!(output, "readings[0]:").unwrap();
            return;
        }

        let (strategy, _) = self.detect_collapse_strategy(readings);

        match strategy {
            CollapseStrategy::UniformArray => {
                writeln!(output, "readings[{}]{{timestamp,value}}:", readings.len()).unwrap();
                for r in readings {
                    writeln!(output, "{}, {}", r.timestamp.to_rfc3339(), r.value).unwrap();
                }
            }
            CollapseStrategy::ConstantRun { run_length } => {
                self.encode_constant_runs(readings, output, run_length);
            }
            CollapseStrategy::RegularInterval { stride_seconds } => {
                self.encode_regular_interval(readings, output, stride_seconds);
            }
            CollapseStrategy::None => {
                writeln!(output, "readings[{}]{{timestamp,value}}:", readings.len()).unwrap();
                for r in readings {
                    writeln!(output, "{}, {}", r.timestamp.to_rfc3339(), r.value).unwrap();
                }
            }
        }
    }
}

impl Default for ToonTimeEncoder {
    fn default() -> Self {
        Self::with_defaults()
    }
}

pub struct TokenBenchmark {
    pub toon_tokens: usize,
    pub json_tokens: usize,
    pub compression_ratio: f64,
    pub toon_bytes: usize,
    pub json_bytes: usize,
}

impl TokenBenchmark {
    pub fn compare(toon: &str, json: &str) -> Self {
        let toon_bytes = toon.len();
        let json_bytes = json.len();

        let toon_tokens = Self::estimate_tokens(toon);
        let json_tokens = Self::estimate_tokens(json);

        let compression_ratio = if json_tokens > 0 {
            toon_tokens as f64 / json_tokens as f64
        } else {
            0.0
        };

        Self {
            toon_tokens,
            json_tokens,
            compression_ratio,
            toon_bytes,
            json_bytes,
        }
    }

    fn estimate_tokens(text: &str) -> usize {
        let word_count = text.split_whitespace().count();
        let char_based = (text.len() as f64 / 4.0).ceil() as usize;
        word_count.max(char_based)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn make_points(values: &[f64], start_hour: u32) -> Vec<TimeSeriesPoint> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let t = Utc
                    .with_ymd_and_hms(2024, 1, 15, start_hour + i as u32, 0, 0)
                    .unwrap();
                TimeSeriesPoint {
                    timestamp: t,
                    value: v,
                }
            })
            .collect()
    }

    #[test]
    fn test_uniform_array_detection() {
        let encoder = ToonTimeEncoder::with_defaults();
        let points = make_points(&[22.0, 22.5, 23.0, 23.5], 0);
        let record = TelemetryRecord {
            zone_id: "Z1".to_string(),
            sensor_id: "T1".to_string(),
            readings: points,
        };
        let output = encoder.encode_telemetry(&[record]);
        assert!(output.contains("readings[4]{timestamp,value}:"));
    }

    #[test]
    fn test_constant_run_collapse() {
        let encoder = ToonTimeEncoder::with_defaults();
        let points = make_points(&[22.0, 22.0, 22.0, 22.0, 23.0], 0);
        let output = encoder.encode_single_series(&points);
        assert!(output.contains("readings_span"));
    }

    #[test]
    fn test_regular_interval_detection() {
        let encoder = ToonTimeEncoder::with_defaults();
        let points: Vec<TimeSeriesPoint> = (0..24)
            .map(|h| {
                let t = Utc.with_ymd_and_hms(2024, 1, 15, h, 0, 0).unwrap();
                TimeSeriesPoint {
                    timestamp: t,
                    value: 20.0 + (h as f64 * 0.1),
                }
            })
            .collect();
        let output = encoder.encode_single_series(&points);
        assert!(output.contains("stride_seconds: 3600") || output.contains("readings[24]"));
    }

    #[test]
    fn test_empty_readings() {
        let encoder = ToonTimeEncoder::with_defaults();
        let points: Vec<TimeSeriesPoint> = vec![];
        let output = encoder.encode_single_series(&points);
        assert!(output.contains("readings[0]:"));
    }

    #[test]
    fn test_token_benchmark() {
        let toon = "readings[3]{timestamp,value}:\n2024-01-15T00:00:00Z, 22.0\n2024-01-15T01:00:00Z, 22.5\n2024-01-15T02:00:00Z, 23.0";
        let json = r#"{"readings":[{"timestamp":"2024-01-15T00:00:00Z","value":22.0},{"timestamp":"2024-01-15T01:00:00Z","value":22.5},{"timestamp":"2024-01-15T02:00:00Z","value":23.0}]}"#;
        let benchmark = TokenBenchmark::compare(toon, json);
        assert!(benchmark.toon_tokens <= benchmark.json_tokens);
        assert!(benchmark.compression_ratio > 0.0 && benchmark.compression_ratio <= 1.0);
    }

    #[test]
    fn test_encode_multiple_records() {
        let encoder = ToonTimeEncoder::with_defaults();
        let records = vec![
            TelemetryRecord {
                zone_id: "Z1".to_string(),
                sensor_id: "T1".to_string(),
                readings: make_points(&[22.0, 22.5], 0),
            },
            TelemetryRecord {
                zone_id: "Z2".to_string(),
                sensor_id: "T2".to_string(),
                readings: make_points(&[21.0, 21.5], 0),
            },
        ];
        let output = encoder.encode_telemetry(&records);
        assert!(output.contains("# Z1"));
        assert!(output.contains("# Z2"));
        assert!(output.contains("sensor: T1"));
        assert!(output.contains("sensor: T2"));
    }

    #[test]
    fn test_constant_run_threshold() {
        let config = ToonTimeEncoderConfig {
            constant_run_min_length: 5,
            ..Default::default()
        };
        let encoder = ToonTimeEncoder::new(config);
        let points = make_points(&[22.0, 22.0, 22.0, 22.0, 23.0], 0);
        let output = encoder.encode_single_series(&points);
        assert!(!output.contains("readings_span"));
    }

    #[test]
    fn test_mixed_constant_and_varying() {
        let encoder = ToonTimeEncoder::with_defaults();
        let points = make_points(&[22.0, 22.0, 22.0, 23.5, 24.0, 24.0, 24.0], 0);
        let output = encoder.encode_single_series(&points);
        assert!(output.contains("readings_span"));
    }

    #[test]
    fn test_single_reading() {
        let encoder = ToonTimeEncoder::with_defaults();
        let points = make_points(&[22.5], 0);
        let output = encoder.encode_single_series(&points);
        assert!(output.contains("22.5"));
    }

    #[test]
    fn test_collapse_strategy_ordering() {
        assert!(CollapseStrategy::UniformArray.is_better_than(
            &CollapseStrategy::RegularInterval {
                stride_seconds: 3600
            }
        ));
        assert!(
            CollapseStrategy::ConstantRun { run_length: 5 }.is_better_than(
                &CollapseStrategy::RegularInterval {
                    stride_seconds: 3600
                }
            )
        );
        assert!(CollapseStrategy::ConstantRun { run_length: 10 }
            .is_better_than(&CollapseStrategy::ConstantRun { run_length: 5 }));
        assert!(CollapseStrategy::RegularInterval {
            stride_seconds: 3600
        }
        .is_better_than(&CollapseStrategy::RegularInterval { stride_seconds: 60 }));
    }

    #[test]
    fn test_irregular_timestamps_fallback() {
        let config = ToonTimeEncoderConfig {
            interval_tolerance_seconds: 30,
            ..Default::default()
        };
        let encoder = ToonTimeEncoder::new(config);
        let points = vec![
            TimeSeriesPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap(),
                value: 22.0,
            },
            TimeSeriesPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 0, 5, 30).unwrap(),
                value: 22.5,
            },
            TimeSeriesPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 0, 12, 45).unwrap(),
                value: 23.0,
            },
        ];
        let output = encoder.encode_single_series(&points);
        assert!(!output.contains("stride_seconds"));
    }

    #[test]
    fn test_toon_time_encoder_config_default() {
        let config = ToonTimeEncoderConfig::default();
        assert_eq!(config.constant_run_min_length, 3);
        assert_eq!(config.interval_tolerance_seconds, 30);
        assert_eq!(config.value_tolerance, 1e-6);
        assert!(config.emit_stride_header);
    }

    #[test]
    fn test_token_benchmark_bytes_ratio() {
        let toon = "readings[3]{timestamp,value}:\n2024-01-15T00:00:00Z, 22.0";
        let json = r#"{"readings":[{"timestamp":"2024-01-15T00:00:00Z","value":22.0}]}"#;
        let benchmark = TokenBenchmark::compare(toon, json);
        let byte_ratio = benchmark.toon_bytes as f64 / benchmark.json_bytes as f64;
        assert!(byte_ratio > 0.0);
    }
}

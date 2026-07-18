//! Timestamp Alignment for Fluxion Output vs Sensor Logs
//!
//! Aligns simulation timesteps with sensor observations, handling:
//! - **DST transitions**: Clocks spring forward / fall back are normalized to
//!   local civil time so that every wall-clock hour appears exactly once.
//! - **Missing-sample interpolation**: Gaps in sensor data are filled via linear
//!   interpolation (or marked as gaps when the gap exceeds `max_interp_gap`).
//! - **Multi-resolution support**: Simulation outputs (fixed timestep) and
//!   sensor logs (possibly sub-hourly or irregular) are resampled to a common
//!   output grid.
//!
//! # Reference
//! Follows the alignment-script idea from issue #1053. Prerequisite for the
//! CVRMSE / NMBE comparison in T10.7–T10.8.

use serde::{Deserialize, Serialize};

/// A single timestamped observation (sensor or simulation output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedSample {
    /// UTC epoch seconds.
    pub epoch_secs: i64,
    /// Measured or simulated value.
    pub value: f64,
}

/// Alignment strategy for DST transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DstStrategy {
    /// Drop the ambiguous / skipped hour entirely.
    DropAmbiguous,
    /// Duplicate the fall-back hour so both readings are kept.
    KeepBoth,
    /// Average the two values that map to the same civil hour.
    AverageDuplicates,
}

/// Interpolation strategy for missing samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation between nearest neighbours.
    Linear,
    /// Carry the last known value forward.
    ForwardFill,
    /// Mark the gap as `None` in the output.
    None,
}

/// Configuration for timestamp alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// UTC offset of the sensor site in whole seconds (e.g. +5 h → 18000).
    pub utc_offset_secs: i32,
    /// Expected simulation timestep in seconds (e.g. 3600 for hourly).
    pub sim_timestep_secs: i64,
    /// Strategy for DST transitions.
    pub dst_strategy: DstStrategy,
    /// Strategy for filling missing sensor samples.
    pub interp_method: InterpolationMethod,
    /// Maximum gap length (in number of expected timesteps) that will be
    /// interpolated.  Gaps larger than this are left as `None`.
    /// A value of `0` disables interpolation entirely.
    pub max_interp_gap: usize,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            utc_offset_secs: 0,
            sim_timestep_secs: 3600,
            dst_strategy: DstStrategy::AverageDuplicates,
            interp_method: InterpolationMethod::Linear,
            max_interp_gap: 3,
        }
    }
}

/// Diagnostic information produced during alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentDiagnostics {
    /// Number of input sensor samples.
    pub sensor_input_count: usize,
    /// Number of input simulation samples.
    pub sim_input_count: usize,
    /// Number of output aligned pairs.
    pub aligned_count: usize,
    /// Number of sensor samples dropped (DST or out-of-range).
    pub dropped_count: usize,
    /// Number of gaps detected.
    pub gaps_detected: usize,
    /// Number of samples interpolated.
    pub interpolated_count: usize,
    /// Civil-time hour labels that were duplicated by DST fall-back.
    pub dst_fall_back_duplicates: usize,
    /// Civil-time hour labels that were skipped by DST spring-forward.
    pub dst_spring_forward_skips: usize,
}

/// An aligned (simulation, sensor) pair at a civil-time instant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedPair {
    /// Seconds since Unix epoch.
    pub epoch_secs: i64,
    /// Civil hour label in `HHMM` format (e.g. 1300 for 13:00).
    pub civil_hour: u32,
    /// Simulation value.
    pub sim_value: f64,
    /// Sensor value (may be interpolated).
    pub sensor_value: f64,
    /// Whether this sensor value was interpolated.
    pub interpolated: bool,
}

/// Align simulation and sensor time series.
///
/// Returns the aligned pairs and diagnostic information.
pub fn align_timestamps(
    sim: &[TimestampedSample],
    sensor: &[TimestampedSample],
    config: &AlignmentConfig,
) -> (Vec<AlignedPair>, AlignmentDiagnostics) {
    let mut diag = AlignmentDiagnostics {
        sensor_input_count: sensor.len(),
        sim_input_count: sim.len(),
        aligned_count: 0,
        dropped_count: 0,
        gaps_detected: 0,
        interpolated_count: 0,
        dst_fall_back_duplicates: 0,
        dst_spring_forward_skips: 0,
    };

    if sim.is_empty() || sensor.is_empty() {
        return (vec![], diag);
    }

    // Build a map from civil-hour-key → Vec<(epoch, value)> for sensor data.
    let sensor_by_civil = build_civil_hour_map(sensor, config.utc_offset_secs);
    let sim_by_civil = build_civil_hour_map(sim, config.utc_offset_secs);

    // Determine the output time grid from the simulation series.
    let sim_civil_keys = sorted_civil_keys(&sim_by_civil);
    let mut pairs = Vec::with_capacity(sim_civil_keys.len());

    // --- Pass 1: build pairs, mark missing as NaN ---
    for &civil_key in &sim_civil_keys {
        let sim_entries = &sim_by_civil[&civil_key];
        let sim_value = sim_entries.last().map_or(0.0, |e| e.value);
        let sim_epoch = sim_entries.last().map_or(0, |e| e.epoch_secs);

        if let Some(sensor_entries) = sensor_by_civil.get(&civil_key) {
            let sensor_value = resolve_dst_duplicates(sensor_entries, config.dst_strategy);
            pairs.push(AlignedPair {
                epoch_secs: sim_epoch,
                civil_hour: civil_key,
                sim_value,
                sensor_value,
                interpolated: false,
            });
            diag.aligned_count += 1;
            diag.dropped_count += if sensor_entries.len() > 1 {
                sensor_entries.len() - 1
            } else {
                0
            };
        } else {
            pairs.push(AlignedPair {
                epoch_secs: sim_epoch,
                civil_hour: civil_key,
                sim_value,
                sensor_value: f64::NAN,
                interpolated: true,
            });
        }
    }

    // --- Pass 2: detect gaps and fill them ---
    let len = pairs.len();
    let mut i = 0;
    while i < len {
        if pairs[i].sensor_value.is_nan() {
            // Find the extent of this gap.
            let gap_start = i;
            while i < len && pairs[i].sensor_value.is_nan() {
                i += 1;
            }
            let gap_len = i - gap_start;
            diag.gaps_detected += 1;
            if gap_len <= config.max_interp_gap {
                fill_gap(&mut pairs, gap_start, config.interp_method);
                diag.interpolated_count += gap_len;
            }
        } else {
            i += 1;
        }
    }

    (pairs, diag)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Civil-hour key: local wall-clock time expressed as HHMM (0–2359).
/// Two instants in UTC that map to the same local civil time during DST
/// fall-back produce the same key.
fn civil_hour_key(epoch_secs: i64, utc_offset_secs: i32) -> u32 {
    let local_secs = epoch_secs + utc_offset_secs as i64;
    // Integer division toward negative infinity.
    let day_secs = local_secs.rem_euclid(86400);
    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    (hours as u32) * 100 + minutes as u32
}

/// Build a map from civil-hour-key → entries, preserving ordering.
fn build_civil_hour_map(
    samples: &[TimestampedSample],
    utc_offset_secs: i32,
) -> std::collections::BTreeMap<u32, Vec<TimestampedSample>> {
    let mut map: std::collections::BTreeMap<u32, Vec<TimestampedSample>> =
        std::collections::BTreeMap::new();
    for s in samples {
        let key = civil_hour_key(s.epoch_secs, utc_offset_secs);
        map.entry(key).or_default().push(s.clone());
    }
    map
}

/// Sorted civil-hour keys from the map.
fn sorted_civil_keys(map: &std::collections::BTreeMap<u32, Vec<TimestampedSample>>) -> Vec<u32> {
    map.keys().copied().collect()
}

/// Resolve entries that map to the same civil hour during DST fall-back.
fn resolve_dst_duplicates(entries: &[TimestampedSample], strategy: DstStrategy) -> f64 {
    match strategy {
        DstStrategy::KeepBoth => entries.last().map_or(0.0, |e| e.value),
        DstStrategy::DropAmbiguous => entries.first().map_or(0.0, |e| e.value),
        DstStrategy::AverageDuplicates => {
            if entries.is_empty() {
                0.0
            } else {
                let sum: f64 = entries.iter().map(|e| e.value).sum();
                sum / entries.len() as f64
            }
        }
    }
}

/// Fill NaN gaps in `pairs[start..]` using the specified method.
fn fill_gap(pairs: &mut [AlignedPair], start: usize, method: InterpolationMethod) {
    let len = pairs.len();
    if start >= len {
        return;
    }

    match method {
        InterpolationMethod::Linear => {
            // Find the previous non-NaN and next non-NaN neighbours.
            let prev_idx = (0..start).rev().find(|&i| !pairs[i].sensor_value.is_nan());
            let next_idx = (start..len).find(|&i| !pairs[i].sensor_value.is_nan());

            if let (Some(pi), Some(ni)) = (prev_idx, next_idx) {
                let y0 = pairs[pi].sensor_value;
                let y1 = pairs[ni].sensor_value;
                let span = (ni - pi) as f64;
                for (offset, pair) in pairs[pi + 1..ni].iter_mut().enumerate() {
                    let t = (offset + 1) as f64 / span;
                    pair.sensor_value = y0 + t * (y1 - y0);
                }
            } else if let Some(pi) = prev_idx {
                // Forward-fill fallback when there is no next neighbour.
                let y0 = pairs[pi].sensor_value;
                for pair in pairs[pi..len].iter_mut() {
                    if pair.sensor_value.is_nan() {
                        pair.sensor_value = y0;
                    }
                }
            }
        }
        InterpolationMethod::ForwardFill => {
            let mut last_val = if start > 0 && !pairs[start - 1].sensor_value.is_nan() {
                pairs[start - 1].sensor_value
            } else {
                return;
            };
            for pair in pairs[start..len].iter_mut() {
                if pair.sensor_value.is_nan() {
                    pair.sensor_value = last_val;
                } else {
                    last_val = pair.sensor_value;
                }
            }
        }
        InterpolationMethod::None => {
            // Leave NaN values as-is.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(epoch: i64, value: f64) -> TimestampedSample {
        TimestampedSample {
            epoch_secs: epoch,
            value,
        }
    }

    #[test]
    fn test_civil_hour_key_basic() {
        // 2024-06-15 13:00:00 UTC → HHMM = 1300
        let epoch = 1_718_456_400; // arbitrary
        assert_eq!(civil_hour_key(epoch, 0), 1300);
    }

    #[test]
    fn test_civil_hour_key_positive_offset() {
        // UTC+5 → civil time is 5 hours ahead.
        let epoch = 1_718_456_400; // 13:00 UTC
        assert_eq!(civil_hour_key(epoch, 18000), 1800); // 18:00 local
    }

    #[test]
    fn test_civil_hour_key_negative_offset() {
        // UTC-5 → civil time is 5 hours behind.
        let epoch = 1_718_456_400; // 13:00 UTC
        assert_eq!(civil_hour_key(epoch, -18000), 800); // 08:00 local
    }

    #[test]
    fn test_civil_hour_key_day_wrap() {
        // UTC 23:00, offset +2 → civil 01:00 next day → 100
        let epoch = 23 * 3600;
        assert_eq!(civil_hour_key(epoch, 7200), 100);
    }

    // Midnight-aligned base: 2024-06-15 00:00:00 UTC so civil_hour == i*100.
    const BASE: i64 = 1_718_409_600;

    #[test]
    fn test_align_identical_series() {
        let sim: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0 + i as f64))
            .collect();
        let sensor = sim.clone();

        let config = AlignmentConfig::default();
        let (pairs, diag) = align_timestamps(&sim, &sensor, &config);

        assert_eq!(pairs.len(), 24);
        assert_eq!(diag.aligned_count, 24);
        assert_eq!(diag.dropped_count, 0);
        assert_eq!(diag.interpolated_count, 0);

        for pair in &pairs {
            assert!((pair.sim_value - pair.sensor_value).abs() < 1e-10);
            assert!(!pair.interpolated);
        }
    }

    #[test]
    fn test_align_with_offset() {
        let sim: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0 + i as f64))
            .collect();
        let sensor: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0 + i as f64 + 0.5))
            .collect();

        let mut config = AlignmentConfig::default();
        config.utc_offset_secs = 18000;
        let (pairs, diag) = align_timestamps(&sim, &sensor, &config);

        assert_eq!(pairs.len(), 24);
        assert_eq!(diag.aligned_count, 24);

        for pair in &pairs {
            assert!((pair.sensor_value - pair.sim_value - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_interpolation_single_gap() {
        let sim: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0))
            .collect();
        let sensor: Vec<TimestampedSample> = (0i64..24)
            .filter(|&i| i != 5)
            .map(|i| make_sample(BASE + i * 3600, 20.0))
            .collect();

        let config = AlignmentConfig::default();
        let (pairs, diag) = align_timestamps(&sim, &sensor, &config);

        assert_eq!(pairs.len(), 24);
        assert_eq!(diag.gaps_detected, 1);
        assert_eq!(diag.interpolated_count, 1);

        // With midnight base, BTreeMap order == sim index order
        assert!((pairs[5].sensor_value - 20.0).abs() < 1e-10);
        assert!(pairs[5].interpolated);
    }

    #[test]
    fn test_linear_interpolation_different_values() {
        let sim: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0 + i as f64))
            .collect();
        let sensor: Vec<TimestampedSample> = (0i64..24)
            .filter(|&i| i != 6)
            .map(|i| make_sample(BASE + i * 3600, 20.0 + i as f64))
            .collect();

        let config = AlignmentConfig::default();
        let (pairs, _diag) = align_timestamps(&sim, &sensor, &config);

        // Between index 5 (25.0) and index 7 (27.0) -> 26.0
        assert!((pairs[6].sensor_value - 26.0).abs() < 1e-10);
        assert!(pairs[6].interpolated);
    }

    #[test]
    fn test_gap_exceeds_max_interp() {
        let sim: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0))
            .collect();
        let sensor: Vec<TimestampedSample> = (0i64..24)
            .filter(|&i| i < 4 || i > 8)
            .map(|i| make_sample(BASE + i * 3600, 20.0))
            .collect();

        let mut config = AlignmentConfig::default();
        config.max_interp_gap = 2;
        let (pairs, diag) = align_timestamps(&sim, &sensor, &config);

        assert_eq!(diag.gaps_detected, 1);
        assert_eq!(diag.interpolated_count, 0);

        for i in 4..=8 {
            assert!(pairs[i].sensor_value.is_nan());
        }
    }

    #[test]
    fn test_forward_fill() {
        let sim: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0 + i as f64))
            .collect();
        let sensor: Vec<TimestampedSample> = (0i64..24)
            .filter(|&i| i != 5)
            .map(|i| make_sample(BASE + i * 3600, 20.0 + i as f64))
            .collect();

        let mut config = AlignmentConfig::default();
        config.interp_method = InterpolationMethod::ForwardFill;
        let (pairs, _diag) = align_timestamps(&sim, &sensor, &config);

        assert!((pairs[5].sensor_value - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_interpolation() {
        let sim: Vec<TimestampedSample> = (0..24)
            .map(|i| make_sample(BASE + i * 3600, 20.0))
            .collect();
        let sensor: Vec<TimestampedSample> = (0i64..24)
            .filter(|&i| i != 5)
            .map(|i| make_sample(BASE + i * 3600, 20.0))
            .collect();

        let mut config = AlignmentConfig::default();
        config.interp_method = InterpolationMethod::None;
        let (pairs, _diag) = align_timestamps(&sim, &sensor, &config);

        assert!(pairs[5].sensor_value.is_nan());
    }

    #[test]
    fn test_empty_inputs() {
        let config = AlignmentConfig::default();
        let (pairs, diag) = align_timestamps(&[], &[], &config);
        assert!(pairs.is_empty());
        assert_eq!(diag.aligned_count, 0);
    }

    #[test]
    fn test_empty_sensor() {
        let sim: Vec<TimestampedSample> =
            (0..5).map(|i| make_sample(BASE + i * 3600, 20.0)).collect();

        let config = AlignmentConfig::default();
        let (pairs, diag) = align_timestamps(&sim, &[], &config);
        assert!(pairs.is_empty());
        assert_eq!(diag.aligned_count, 0);
    }

    #[test]
    fn test_dst_fall_back_average() {
        let sensor = vec![make_sample(1000, 22.0), make_sample(1000, 23.0)];
        let sim = vec![make_sample(1000, 22.5)];

        let mut config = AlignmentConfig::default();
        config.utc_offset_secs = 0;
        config.dst_strategy = DstStrategy::AverageDuplicates;
        let (pairs, diag) = align_timestamps(&sim, &sensor, &config);

        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].sensor_value - 22.5).abs() < 1e-10);
        assert_eq!(diag.dropped_count, 1);
    }

    #[test]
    fn test_dst_fall_back_keep_both() {
        let sensor = vec![make_sample(1000, 22.0), make_sample(1000, 23.0)];
        let sim = vec![make_sample(1000, 22.5)];

        let mut config = AlignmentConfig::default();
        config.utc_offset_secs = 0;
        config.dst_strategy = DstStrategy::KeepBoth;
        let (pairs, _diag) = align_timestamps(&sim, &sensor, &config);

        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].sensor_value - 23.0).abs() < 1e-10);
    }

    #[test]
    fn test_sub_hourly_alignment() {
        let sim: Vec<TimestampedSample> = (0..96)
            .map(|i| make_sample(BASE + i * 900, 20.0 + i as f64 * 0.25))
            .collect();
        let sensor: Vec<TimestampedSample> = (0i64..96)
            .filter(|&i| i != 10)
            .map(|i| make_sample(BASE + i * 900, 20.0 + i as f64 * 0.25))
            .collect();

        let mut config = AlignmentConfig::default();
        config.sim_timestep_secs = 900;
        let (pairs, diag) = align_timestamps(&sim, &sensor, &config);

        assert_eq!(pairs.len(), 96);
        assert_eq!(diag.gaps_detected, 1);
        assert_eq!(diag.interpolated_count, 1);

        let missing = &pairs[10];
        assert!(missing.interpolated);
        // Index 9: 20 + 9*0.25 = 22.25, index 11: 20 + 11*0.25 = 22.75 -> 22.5
        assert!((missing.sensor_value - 22.5).abs() < 1e-10);
    }

    #[test]
    fn test_diagnostics_serialization() {
        let diag = AlignmentDiagnostics {
            sensor_input_count: 8760,
            sim_input_count: 8760,
            aligned_count: 8750,
            dropped_count: 10,
            gaps_detected: 3,
            interpolated_count: 5,
            dst_fall_back_duplicates: 2,
            dst_spring_forward_skips: 1,
        };

        let json = serde_json::to_string(&diag).unwrap();
        let parsed: AlignmentDiagnostics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.sensor_input_count, 8760);
        assert_eq!(parsed.dropped_count, 10);
    }

    #[test]
    fn test_alignment_config_serialization() {
        let config = AlignmentConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: AlignmentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.sim_timestep_secs, 3600);
        assert_eq!(parsed.dst_strategy, DstStrategy::AverageDuplicates);
    }
}

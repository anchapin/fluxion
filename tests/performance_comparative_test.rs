use crate::validation::performance::comparative::{ComparativeAnalyzer, ConfigurationResult};
use crate::validation::performance::historical::{HistoricalRecord, HistoricalTracker};
use crate::validation::performance::reports::PerformanceMetrics;
use chrono::Utc;
use serde_json::json;

#[test]
fn test_comparative_analysis_basic() {
    let baseline_metrics = PerformanceMetrics {
        timestep_duration_ms: 50.0,
        memory_usage_bytes: 10000,
        iterations_per_timestep: 10,
    };

    let optimized_metrics = PerformanceMetrics {
        timestep_duration_ms: 40.0,
        memory_usage_bytes: 8000,
        iterations_per_timestep: 8,
    };

    let baseline = ConfigurationResult {
        name: "baseline".to_string(),
        metrics: baseline_metrics,
        configuration: json!({ "zones": 1, "construction": "standard" }),
    };

    let optimized = ConfigurationResult {
        name: "optimized".to_string(),
        metrics: optimized_metrics,
        configuration: json!({ "zones": 1, "construction": "standard", "solver": "optimized" }),
    };

    let analyzer = ComparativeAnalyzer::new(baseline);
    let deltas = analyzer.compare_two(&baseline, &optimized);

    assert_eq!(deltas.len(), 2);
}

#[test]
fn test_historical_tracking_basic() {
    let mut tracker = HistoricalTracker::new();

    let record1 = HistoricalRecord {
        timestamp: Utc::now(),
        commit_hash: "abc123".to_string(),
        version: "1.0.0".to_string(),
        metrics: PerformanceMetrics {
            timestep_duration_ms: 50.0,
            memory_usage_bytes: 10000,
            iterations_per_timestep: 10,
        },
        configuration: json!({}),
    };

    tracker.add_record("single-zone", record1);

    // Just verify we can add records without panicking
    assert_eq!(tracker.records.len(), 1);
}

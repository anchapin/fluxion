use chrono::Utc;
use fluxion::validation::performance::comparative::{
    ComparativeAnalyzer, ConfigurationResult, PerformanceDelta,
};
use fluxion::validation::performance::historical::{
    HistoricalRecord, HistoricalTracker, TrendDirection,
};
use fluxion::validation::performance::reports::PerformanceMetrics;
use serde_json::json;

#[test]
fn test_comparative_analysis() {
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

    // Check timestep duration delta
    let time_delta = deltas
        .iter()
        .find(|d| d.metric == "timestep_duration_ms")
        .unwrap();
    assert_eq!(time_delta.delta, -10.0);
    assert_eq!(time_delta.percent_change, -20.0);

    // Check memory usage delta
    let memory_delta = deltas
        .iter()
        .find(|d| d.metric == "memory_usage_bytes")
        .unwrap();
    assert_eq!(memory_delta.delta, -2000.0);
    assert_eq!(memory_delta.percent_change, -20.0);
}

#[test]
fn test_historical_tracking() {
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

    let record2 = HistoricalRecord {
        timestamp: Utc::now(),
        commit_hash: "def456".to_string(),
        version: "1.1.0".to_string(),
        metrics: PerformanceMetrics {
            timestep_duration_ms: 45.0,
            memory_usage_bytes: 9000,
            iterations_per_timestep: 9,
        },
        configuration: json!({}),
    };

    tracker.add_record("single-zone", record1);
    tracker.add_record("single-zone", record2);

    let trend = tracker
        .analyze_trend("single-zone", "timestep_duration_ms")
        .unwrap();
    assert_eq!(trend.trend, TrendDirection::Improving);
    assert!(trend.average_change < 0.0); // Negative change means improvement
}

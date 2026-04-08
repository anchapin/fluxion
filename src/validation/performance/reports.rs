use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Comparison {
    pub baseline_metrics: PerformanceMetrics,
    pub current_metrics: PerformanceMetrics,
    pub improvement_percent: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PerformanceReport {
    pub timestamp: DateTime<Utc>,
    pub metrics: PerformanceMetrics,
    pub baseline_comparison: Option<Comparison>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PerformanceMetrics {
    pub timestep_duration_ms: f64,
    pub memory_usage_bytes: usize,
    pub iterations_per_timestep: u32,
}

pub fn generate_performance_report(metrics: metrics::PerformanceMetrics) -> PerformanceReport {
    PerformanceReport {
        timestamp: Utc::now(),
        metrics: PerformanceMetrics {
            timestep_duration_ms: metrics.timestep_duration.as_secs_f64() * 1000.0,
            memory_usage_bytes: metrics.memory_usage,
            iterations_per_timestep: metrics.iterations_per_timestep,
        },
        baseline_comparison: None,
    }
}

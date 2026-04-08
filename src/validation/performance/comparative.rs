use crate::validation::performance::reports::PerformanceMetrics;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct ComparativeAnalysis {
    pub configurations: Vec<ConfigurationResult>,
    pub best_performer: String,
    pub performance_deltas: Vec<PerformanceDelta>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConfigurationResult {
    pub name: String,
    pub metrics: PerformanceMetrics,
    pub configuration: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceDelta {
    pub config_a: String,
    pub config_b: String,
    pub metric: String,
    pub delta: f64,
    pub percent_change: f64,
}

pub struct ComparativeAnalyzer {
    baseline_config: ConfigurationResult,
}

impl ComparativeAnalyzer {
    pub fn new(baseline: ConfigurationResult) -> Self {
        Self {
            baseline_config: baseline,
        }
    }

    pub fn add_configuration(&mut self, config: ConfigurationResult) {
        // Store configuration for comparison
    }

    pub fn analyze(&self) -> ComparativeAnalysis {
        // Implement comparative analysis logic
        ComparativeAnalysis {
            configurations: vec![],
            best_performer: String::new(),
            performance_deltas: vec![],
        }
    }

    pub fn compare_two(
        &self,
        config_a: &ConfigurationResult,
        config_b: &ConfigurationResult,
    ) -> Vec<PerformanceDelta> {
        let mut deltas = vec![];

        // Compare timestep duration
        let time_delta =
            config_b.metrics.timestep_duration_ms - config_a.metrics.timestep_duration_ms;
        let time_percent = (time_delta / config_a.metrics.timestep_duration_ms) * 100.0;
        deltas.push(PerformanceDelta {
            config_a: config_a.name.clone(),
            config_b: config_b.name.clone(),
            metric: "timestep_duration_ms".to_string(),
            delta: time_delta,
            percent_change: time_percent,
        });

        // Compare memory usage
        let memory_delta =
            config_b.metrics.memory_usage_bytes as f64 - config_a.metrics.memory_usage_bytes as f64;
        let memory_percent = (memory_delta / config_a.metrics.memory_usage_bytes as f64) * 100.0;
        deltas.push(PerformanceDelta {
            config_a: config_a.name.clone(),
            config_b: config_b.name.clone(),
            metric: "memory_usage_bytes".to_string(),
            delta: memory_delta,
            percent_change: memory_percent,
        });

        deltas
    }
}

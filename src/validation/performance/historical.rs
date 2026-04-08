use crate::validation::performance::reports::PerformanceMetrics;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
pub struct HistoricalRecord {
    pub timestamp: DateTime<Utc>,
    pub commit_hash: String,
    pub version: String,
    pub metrics: PerformanceMetrics,
    pub configuration: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PerformanceTrend {
    pub metric: String,
    pub values: Vec<(DateTime<Utc>, f64)>,
    pub trend: TrendDirection,
    pub average_change: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

pub struct HistoricalTracker {
    records: HashMap<String, Vec<HistoricalRecord>>, // Keyed by benchmark name
}

impl HistoricalTracker {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    pub fn add_record(&mut self, benchmark_name: &str, record: HistoricalRecord) {
        self.records
            .entry(benchmark_name.to_string())
            .or_insert_with(Vec::new)
            .push(record);
    }

    pub fn analyze_trend(&self, benchmark_name: &str, metric: &str) -> Option<PerformanceTrend> {
        let records = self.records.get(benchmark_name)?;

        if records.len() < 2 {
            return None; // Not enough data for trend analysis
        }

        let mut values = vec![];
        let metric_values: Vec<f64> = records
            .iter()
            .map(|r| match metric {
                "timestep_duration_ms" => r.metrics.timestep_duration_ms,
                "memory_usage_bytes" => r.metrics.memory_usage_bytes as f64,
                _ => 0.0,
            })
            .collect();

        for (i, record) in records.iter().enumerate() {
            values.push((record.timestamp, metric_values[i]));
        }

        // Calculate trend direction
        let first = metric_values[0];
        let last = metric_values[metric_values.len() - 1];
        let change = ((last - first) / first) * 100.0;

        let trend = if change < -5.0 {
            TrendDirection::Improving
        } else if change > 5.0 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        Some(PerformanceTrend {
            metric: metric.to_string(),
            values,
            trend,
            average_change: change,
        })
    }

    pub fn generate_historical_report(&self) -> HistoricalPerformanceReport {
        let mut report = HistoricalPerformanceReport {
            generated_at: Utc::now(),
            benchmarks: vec![],
            overall_trend: TrendDirection::Stable,
        };

        for (benchmark_name, records) in &self.records {
            let latest = records.last().unwrap();
            report.benchmarks.push(BenchmarkHistory {
                name: benchmark_name.clone(),
                latest: latest.metrics.clone(),
                record_count: records.len(),
            });
        }

        report
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HistoricalPerformanceReport {
    pub generated_at: DateTime<Utc>,
    pub benchmarks: Vec<BenchmarkHistory>,
    pub overall_trend: TrendDirection,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BenchmarkHistory {
    pub name: String,
    pub latest: PerformanceMetrics,
    pub record_count: usize,
}

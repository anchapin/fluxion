use crate::validation::performance::metrics;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Comparison {
    pub baseline_metrics: PerformanceMetrics,
    pub current_metrics: PerformanceMetrics,
    pub improvement_percent: f64,
    pub regression_warnings: Option<Vec<String>>,
    pub trend_analysis: Option<TrendAnalysis>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrendAnalysis {
    pub historical_data: Vec<HistoricalDataPoint>,
    pub average_improvement: f64,
    pub stability_score: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HistoricalDataPoint {
    pub timestamp: DateTime<Utc>,
    pub performance_score: f64,
    pub notes: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceReport {
    pub timestamp: DateTime<Utc>,
    pub metrics: PerformanceMetrics,
    pub baseline_comparison: Option<Comparison>,
    pub regression_warnings: Option<Vec<String>>,
    pub trend_analysis: Option<TrendAnalysis>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PerformanceMetrics {
    pub timestep_duration_ms: f64,
    pub memory_usage_bytes: usize,
    pub iterations_per_timestep: u32,
    pub cpu_utilization: f32,
    pub throughput_tps: f32,
    pub zone_coupling_time_ms: f64,
}

pub fn generate_performance_report(metrics: metrics::PerformanceMetrics) -> PerformanceReport {
    PerformanceReport {
        timestamp: Utc::now(),
        metrics: PerformanceMetrics {
            timestep_duration_ms: metrics.timestep_duration.as_secs_f64() * 1000.0,
            memory_usage_bytes: metrics.memory_usage,
            iterations_per_timestep: metrics.iterations_per_timestep,
            cpu_utilization: metrics.cpu_utilization,
            throughput_tps: metrics.throughput_tps,
            zone_coupling_time_ms: metrics.zone_coupling_time.as_secs_f64() * 1000.0,
        },
        baseline_comparison: None,
        regression_warnings: None,
        trend_analysis: None,
    }
}

pub fn generate_comparison_report(
    baseline: metrics::PerformanceMetrics,
    current: metrics::PerformanceMetrics,
) -> Comparison {
    let baseline_duration = baseline.timestep_duration.as_secs_f64() * 1000.0;
    let current_duration = current.timestep_duration.as_secs_f64() * 1000.0;

    let improvement_percent = if baseline_duration > 0.0 {
        ((baseline_duration - current_duration) / baseline_duration) * 100.0
    } else {
        0.0
    };

    let regression_warnings = detect_regressions(&current, &baseline);
    let trend_analysis = generate_trend_analysis(&[baseline.clone(), current.clone()]);

    Comparison {
        baseline_metrics: PerformanceMetrics {
            timestep_duration_ms: baseline_duration,
            memory_usage_bytes: baseline.memory_usage,
            iterations_per_timestep: baseline.iterations_per_timestep,
            cpu_utilization: baseline.cpu_utilization,
            throughput_tps: baseline.throughput_tps,
            zone_coupling_time_ms: baseline.zone_coupling_time.as_secs_f64() * 1000.0,
        },
        current_metrics: PerformanceMetrics {
            timestep_duration_ms: current_duration,
            memory_usage_bytes: current.memory_usage,
            iterations_per_timestep: current.iterations_per_timestep,
            cpu_utilization: current.cpu_utilization,
            throughput_tps: current.throughput_tps,
            zone_coupling_time_ms: current.zone_coupling_time.as_secs_f64() * 1000.0,
        },
        improvement_percent,
        regression_warnings: if regression_warnings.is_empty() {
            None
        } else {
            Some(regression_warnings)
        },
        trend_analysis: Some(trend_analysis),
    }
}

pub fn detect_regressions(
    current: &metrics::PerformanceMetrics,
    baseline: &metrics::PerformanceMetrics,
) -> Vec<String> {
    let mut warnings = Vec::new();

    let baseline_duration = baseline.timestep_duration.as_secs_f64() * 1000.0;
    let current_duration = current.timestep_duration.as_secs_f64() * 1000.0;

    // Check for performance regressions (5% threshold)
    if current_duration > baseline_duration * 1.05 {
        warnings.push(format!(
            "Timestep duration regression: {:.2}% increase",
            ((current_duration - baseline_duration) / baseline_duration) * 100.0
        ));
    }

    if current.memory_usage > baseline.memory_usage * 110 / 100 {
        warnings.push(format!(
            "Memory usage regression: {:.2}% increase",
            (current.memory_usage as f64 - baseline.memory_usage as f64)
                / baseline.memory_usage as f64
                * 100.0
        ));
    }

    if current.iterations_per_timestep > baseline.iterations_per_timestep * 110 / 100 {
        warnings.push(format!(
            "Solver iterations regression: {:.2}% increase",
            (current.iterations_per_timestep as f64 - baseline.iterations_per_timestep as f64)
                / baseline.iterations_per_timestep as f64
                * 100.0
        ));
    }

    warnings
}

pub fn generate_trend_analysis(historical: &[metrics::PerformanceMetrics]) -> TrendAnalysis {
    if historical.is_empty() {
        return TrendAnalysis {
            historical_data: Vec::new(),
            average_improvement: 0.0,
            stability_score: 0.0,
        };
    }

    let mut data_points = Vec::new();
    let mut total_improvement = 0.0;

    for (i, metrics) in historical.iter().enumerate() {
        let duration_ms = metrics.timestep_duration.as_secs_f64() * 1000.0;
        let performance_score = calculate_performance_score(metrics);

        data_points.push(HistoricalDataPoint {
            timestamp: Utc::now(),
            performance_score,
            notes: format!("Measurement {}", i + 1),
        });

        if i > 0 {
            let prev_duration = historical[i - 1].timestep_duration.as_secs_f64() * 1000.0;
            if prev_duration > 0.0 {
                total_improvement += ((prev_duration - duration_ms) / prev_duration) * 100.0;
            }
        }
    }

    let average_improvement = if historical.len() > 1 {
        total_improvement / (historical.len() - 1) as f64
    } else {
        0.0
    };

    let stability_score = calculate_stability_score(&data_points);

    TrendAnalysis {
        historical_data: data_points,
        average_improvement,
        stability_score,
    }
}

fn calculate_performance_score(metrics: &metrics::PerformanceMetrics) -> f64 {
    // Higher score is better (lower duration, lower memory, fewer iterations)
    let duration_ms = metrics.timestep_duration.as_secs_f64() * 1000.0;
    let memory_mb = metrics.memory_usage as f64 / (1024.0 * 1024.0);

    // Normalize metrics and calculate score (inverse of bad metrics)
    let duration_score = 100.0 / (1.0 + duration_ms);
    let memory_score = 100.0 / (1.0 + memory_mb);
    let iteration_score = 100.0 / (1.0 + metrics.iterations_per_timestep as f64);

    (duration_score + memory_score + iteration_score) / 3.0
}

fn calculate_stability_score(data_points: &[HistoricalDataPoint]) -> f64 {
    if data_points.len() < 2 {
        return 100.0;
    }

    let mean: f64 =
        data_points.iter().map(|p| p.performance_score).sum::<f64>() / data_points.len() as f64;
    let variance: f64 = data_points
        .iter()
        .map(|p| (p.performance_score - mean).powi(2))
        .sum::<f64>()
        / data_points.len() as f64;
    let std_dev = variance.sqrt();

    // Stability score: higher is better (less variation)
    if mean > 0.0 {
        (1.0 - (std_dev / mean)).max(0.0) * 100.0
    } else {
        100.0
    }
}

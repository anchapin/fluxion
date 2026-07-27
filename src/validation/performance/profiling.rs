//! Performance profiling module
//!
//! This module provides functionality for profiling the performance
//! of validation cases.

use crate::validation::ashrae140::ASHRAE140Case;
use crate::validation::ashrae_140_validator::ASHRAE140Validator;
use crate::validation::performance::metrics::PerformanceMetrics;
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use std::process::Command;

fn measure_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    if let Ok(output) = Command::new("sh")
        .arg("-c")
        .arg("ps -o rss= -p $$")
        .output()
    {
        if let Ok(size_str) = String::from_utf8(output.stdout) {
            if let Ok(size_kb) = size_str.trim().parse::<usize>() {
                return size_kb * 1024;
            }
        }
    }
    8 * 1024 * 1024
}

fn measure_cpu_utilization() -> f32 {
    let start_time = Instant::now();
    let start_cpu = get_process_cpu_time();
    std::thread::sleep(Duration::from_millis(100));
    let end_cpu = get_process_cpu_time();
    let elapsed = start_time.elapsed();
    let cpu_time = end_cpu - start_cpu;
    if elapsed.as_secs_f64() > 0.0 {
        (cpu_time / elapsed.as_secs_f64() * 100.0) as f32
    } else {
        0.0
    }
}

fn get_process_cpu_time() -> f64 {
    #[cfg(target_os = "linux")]
    if let Ok(output) = Command::new("sh")
        .arg("-c")
        .arg("ps -o time= -p $$")
        .output()
    {
        if let Ok(time_str) = String::from_utf8(output.stdout) {
            let parts: Vec<&str> = time_str.trim().split(':').collect();
            if parts.len() == 3 {
                let minutes = parts[1].parse::<f64>().unwrap_or(0.0);
                let seconds = parts[2].parse::<f64>().unwrap_or(0.0);
                return minutes * 60.0 + seconds;
            }
        }
    }
    0.0
}

/// Profile a single validation case
pub fn profile_case(case: ASHRAE140Case, iterations: usize) -> PerformanceMetrics {
    let start_time = Instant::now();
    let coupling_start = Instant::now();

    let mut validator = ASHRAE140Validator::new();
    let (_benchmark_report, _diagnostic_report) =
        validator.validate_single_case_with_diagnostics(case);

    let coupling_duration = coupling_start.elapsed();
    let execution_time = start_time.elapsed();

    let memory_usage = measure_memory_usage();
    let cpu_utilization = measure_cpu_utilization();
    let throughput_tps = if execution_time.as_secs_f64() > 0.0 {
        (iterations as f64 / execution_time.as_secs_f64()) as f32
    } else {
        0.0
    };

    PerformanceMetrics {
        timestep_duration: execution_time,
        memory_usage,
        iterations_per_timestep: iterations as u32,
        cpu_utilization,
        throughput_tps,
        zone_coupling_time: coupling_duration,
    }
}

/// Generate a performance report from metrics
pub fn generate_performance_report(metrics: &[PerformanceMetrics]) -> serde_json::Value {
    let total_time_ms: f64 = metrics
        .iter()
        .map(|m| m.timestep_duration.as_secs_f64() * 1000.0)
        .sum();
    let avg_time_ms = total_time_ms / metrics.len() as f64;
    let total_memory_bytes: usize = metrics.iter().map(|m| m.memory_usage).sum();

    serde_json::json!({
        "total_cases": metrics.len(),
        "total_execution_time_ms": total_time_ms,
        "average_execution_time_ms": avg_time_ms,
        "total_memory_usage_bytes": total_memory_bytes,
        "average_memory_usage_bytes": total_memory_bytes as f64 / metrics.len() as f64,
        "cases": metrics.iter().map(|metric| {
            serde_json::json!({
                "timestep_duration_ms": metric.timestep_duration.as_secs_f64() * 1000.0,
                "memory_usage_bytes": metric.memory_usage
            })
        }).collect::<Vec<_>>()
    })
}

/// Generate a detailed performance report from metrics
pub fn generate_detailed_performance_report(metrics: &[PerformanceMetrics]) -> serde_json::Value {
    generate_performance_report(metrics)
}

/// Analyze performance bottlenecks
pub fn analyze_bottlenecks(metrics: &[PerformanceMetrics]) -> serde_json::Value {
    let mut bottlenecks = Vec::new();
    let mut recommendations = Vec::new();

    if metrics.is_empty() {
        return serde_json::json!({
            "bottlenecks": [],
            "recommendations": []
        });
    }

    let avg_cpu: f32 =
        metrics.iter().map(|m| m.cpu_utilization).sum::<f32>() / metrics.len() as f32;
    let avg_memory: usize = metrics.iter().map(|m| m.memory_usage).sum::<usize>() / metrics.len();
    let avg_duration_ms: f64 = metrics
        .iter()
        .map(|m| m.timestep_duration.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_throughput: f32 =
        metrics.iter().map(|m| m.throughput_tps).sum::<f32>() / metrics.len() as f32;

    if avg_cpu > 80.0 {
        bottlenecks.push("High CPU utilization detected");
        recommendations
            .push("Consider optimizing solver iterations or enabling parallel execution");
    }

    if avg_memory > 100 * 1024 * 1024 {
        bottlenecks.push("High memory usage detected");
        recommendations.push("Consider reducing model complexity or batch processing");
    }

    if avg_duration_ms > 1000.0 {
        bottlenecks.push("Slow execution time detected");
        recommendations.push("Profile individual solver components for optimization opportunities");
    }

    if avg_throughput < 1.0 && !metrics.is_empty() {
        bottlenecks.push("Low throughput detected");
        recommendations
            .push("Consider using a more efficient solver or reducing timestep frequency");
    }

    let coupling_ratio: f64 = if avg_duration_ms > 0.0 {
        metrics
            .iter()
            .map(|m| {
                m.zone_coupling_time.as_secs_f64() * 1000.0 / m.timestep_duration.as_secs_f64()
            })
            .sum::<f64>()
            / metrics.len() as f64
    } else {
        0.0
    };

    if coupling_ratio > 0.3 {
        bottlenecks.push("High zone coupling overhead");
        recommendations.push("Consider optimizing zone coupling algorithm or reducing zone count");
    }

    serde_json::json!({
        "bottlenecks": bottlenecks,
        "recommendations": recommendations
    })
}

/// Log performance metrics
pub fn log_performance_metrics(metrics: &[PerformanceMetrics]) {
    for metric in metrics {
        log::info!(
            "Timestep: {:.2}ms, Memory: {} bytes, CPU: {:.1}%, Throughput: {:.1} tps",
            metric.timestep_duration.as_secs_f64() * 1000.0,
            metric.memory_usage,
            metric.cpu_utilization * 100.0,
            metric.throughput_tps
        );
    }
}

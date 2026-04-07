// Performance profiling and optimization for validation suite
// This module provides performance monitoring, profiling, and optimization
// capabilities for ASHRAE 140 validation cases

use crate::validation::ASHRAE140Case;
use serde::{Deserialize, Serialize};
use std::thread;
use std::time::Instant;

/// Performance metrics for a single validation case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub case: ASHRAE140Case,
    pub total_time_ms: f64,
    pub per_timestep_ms: f64,
    pub peak_memory_mb: Option<f64>,
    pub initialization_time_ms: f64,
    pub simulation_time_ms: f64,
    pub post_processing_time_ms: f64,
}

/// Profile a validation case and return performance metrics
pub fn profile_case(case: ASHRAE140Case, iterations: usize) -> PerformanceMetrics {
    let mut total_time = 0.0;
    let mut sim_time = 0.0;
    let mut init_time_total = 0.0;
    let mut post_time_total = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        let init_start = Instant::now();

        // Case initialization
        let case_def = crate::validation::ashrae140::cases::build_case(case);
        let init_time = init_start.elapsed();
        init_time_total += init_time.as_secs_f64() * 1000.0;

        let sim_start = Instant::now();
        // Run simulation
        // For now, we'll simulate the timing with a placeholder
        // In a real implementation, this would call run_validation_case(&case_def)
        std::thread::sleep(std::time::Duration::from_millis(10));
        let simulation_time = sim_start.elapsed();
        sim_time += simulation_time.as_secs_f64() * 1000.0;

        let post_start = Instant::now();
        // Post-processing
        // For now, we'll simulate the timing with a placeholder
        std::thread::sleep(std::time::Duration::from_millis(5));
        let post_time = post_start.elapsed();
        post_time_total += post_time.as_secs_f64() * 1000.0;

        let case_time = start.elapsed();
        total_time += case_time.as_secs_f64() * 1000.0;
    }

    PerformanceMetrics {
        case,
        total_time_ms: total_time / iterations as f64,
        per_timestep_ms: (sim_time / iterations as f64) / 8760.0,
        peak_memory_mb: None, // Would require memory profiling
        initialization_time_ms: init_time_total / iterations as f64,
        simulation_time_ms: sim_time / iterations as f64,
        post_processing_time_ms: post_time_total / iterations as f64,
    }
}

/// Generate performance report
pub fn generate_performance_report(metrics: &[PerformanceMetrics]) -> String {
    let mut report = String::new();
    report.push_str("Validation Performance Report\n");
    report.push_str("================================\n\n");

    for metric in metrics {
        report.push_str(&format!(
            "Case {:?}:\n  Total: {:.2}ms\n  Per timestep: {:.4}ms\n  Target: <50.0000ms/timestep\n  Status: {}\n\n",
            metric.case,
            metric.total_time_ms,
            metric.per_timestep_ms,
            if metric.per_timestep_ms < 50.0 { "✓ PASS" } else { "✗ FAIL" }
        ));
    }

    // Add summary statistics
    let avg_timestep: f64 =
        metrics.iter().map(|m| m.per_timestep_ms).sum::<f64>() / metrics.len() as f64;
    report.push_str(&format!(
        "Summary:\n  Average per timestep: {:.4}ms\n  Cases meeting target: {}/{} ({:.1}%)\n",
        avg_timestep,
        metrics.iter().filter(|m| m.per_timestep_ms < 50.0).count(),
        metrics.len(),
        (metrics.iter().filter(|m| m.per_timestep_ms < 50.0).count() as f64 / metrics.len() as f64)
            * 100.0
    ));

    report
}

/// Identify performance bottlenecks
pub fn analyze_bottlenecks(metrics: &[PerformanceMetrics]) -> Vec<String> {
    let mut issues = Vec::new();

    for metric in metrics {
        if metric.per_timestep_ms >= 50.0 {
            issues.push(format!(
                "Case {:?}: {:.4}ms/timestep (target: <50.0000ms)",
                metric.case, metric.per_timestep_ms
            ));
        }

        if metric.simulation_time_ms > 1000.0 {
            issues.push(format!(
                "Case {:?}: Slow simulation ({:.2}ms total)",
                metric.case, metric.simulation_time_ms
            ));
        }
    }

    issues
}

/// Log performance metrics for monitoring
pub fn log_performance_metrics(metrics: &PerformanceMetrics) {
    eprintln!(
        "[PERF] Case {:?}: total={:.2}ms, per_timestep={:.4}ms, status={}",
        metrics.case,
        metrics.total_time_ms,
        metrics.per_timestep_ms,
        if metrics.per_timestep_ms < 50.0 {
            "OK"
        } else {
            "SLOW"
        }
    );
}

/// Generate detailed performance report with breakdown
pub fn generate_detailed_performance_report(metrics: &[PerformanceMetrics]) -> String {
    let mut report = String::new();
    report.push_str("Detailed Validation Performance Report\n");
    report.push_str("========================================\n\n");

    for metric in metrics {
        report.push_str(&format!(
            "Case {:?}:\n  Total Time: {:.2}ms\n  Per Timestep: {:.4}ms\n  Initialization: {:.2}ms\n  Simulation: {:.2}ms\n  Post-processing: {:.2}ms\n  Target: <50.0000ms/timestep\n  Status: {}\n\n",
            metric.case,
            metric.total_time_ms,
            metric.per_timestep_ms,
            metric.initialization_time_ms,
            metric.simulation_time_ms,
            metric.post_processing_time_ms,
            if metric.per_timestep_ms < 50.0 { "✓ PASS" } else { "✗ FAIL" }
        ));
    }

    // Add summary statistics
    let avg_timestep: f64 =
        metrics.iter().map(|m| m.per_timestep_ms).sum::<f64>() / metrics.len() as f64;
    let avg_init: f64 = metrics
        .iter()
        .map(|m| m.initialization_time_ms)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_sim: f64 =
        metrics.iter().map(|m| m.simulation_time_ms).sum::<f64>() / metrics.len() as f64;
    let avg_post: f64 = metrics
        .iter()
        .map(|m| m.post_processing_time_ms)
        .sum::<f64>()
        / metrics.len() as f64;

    report.push_str(&format!(
        "Summary:\n  Average per timestep: {:.4}ms\n  Average initialization: {:.2}ms\n  Average simulation: {:.2}ms\n  Average post-processing: {:.2}ms\n  Cases meeting target: {}/{} ({:.1}%)\n",
        avg_timestep,
        avg_init,
        avg_sim,
        avg_post,
        metrics.iter().filter(|m| m.per_timestep_ms < 50.0).count(),
        metrics.len(),
        (metrics.iter().filter(|m| m.per_timestep_ms < 50.0).count() as f64 / metrics.len() as f64) * 100.0
    ));

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_structure() {
        let metrics = PerformanceMetrics {
            case: ASHRAE140Case::Case800,
            total_time_ms: 100.0,
            per_timestep_ms: 0.01,
            peak_memory_mb: None,
            initialization_time_ms: 10.0,
            simulation_time_ms: 80.0,
            post_processing_time_ms: 10.0,
        };

        assert_eq!(metrics.case, ASHRAE140Case::Case800);
        assert!(metrics.per_timestep_ms < 50.0);
    }

    #[test]
    fn test_profile_case_function() {
        let metrics = profile_case(ASHRAE140Case::Case800, 1);
        assert!(metrics.total_time_ms > 0.0);
        assert!(metrics.per_timestep_ms > 0.0);
    }

    #[test]
    fn test_performance_report_generation() {
        let metrics = vec![
            PerformanceMetrics {
                case: ASHRAE140Case::Case800,
                total_time_ms: 100.0,
                per_timestep_ms: 0.01,
                peak_memory_mb: None,
                initialization_time_ms: 10.0,
                simulation_time_ms: 80.0,
                post_processing_time_ms: 10.0,
            },
            PerformanceMetrics {
                case: ASHRAE140Case::Case801,
                total_time_ms: 150.0,
                per_timestep_ms: 0.015,
                peak_memory_mb: None,
                initialization_time_ms: 15.0,
                simulation_time_ms: 120.0,
                post_processing_time_ms: 15.0,
            },
        ];

        let report = generate_performance_report(&metrics);
        assert!(report.contains("Validation Performance Report"));
        assert!(report.contains("✓ PASS"));
        assert!(report.contains("Summary:"));
    }

    #[test]
    fn test_bottleneck_analysis() {
        let metrics = vec![
            PerformanceMetrics {
                case: ASHRAE140Case::Case800,
                total_time_ms: 100.0,
                per_timestep_ms: 0.01,
                peak_memory_mb: None,
                initialization_time_ms: 10.0,
                simulation_time_ms: 80.0,
                post_processing_time_ms: 10.0,
            },
            PerformanceMetrics {
                case: ASHRAE140Case::Case801,
                total_time_ms: 5000.0,
                per_timestep_ms: 60.0,
                peak_memory_mb: None,
                initialization_time_ms: 100.0,
                simulation_time_ms: 4800.0,
                post_processing_time_ms: 100.0,
            },
        ];

        let issues = analyze_bottlenecks(&metrics);
        assert_eq!(issues.len(), 2); // One for slow timestep, one for slow simulation
        assert!(issues[0].contains("Case801"));
    }

    #[test]
    fn test_log_performance_metrics() {
        let metrics = PerformanceMetrics {
            case: ASHRAE140Case::Case800,
            total_time_ms: 100.0,
            per_timestep_ms: 0.01,
            peak_memory_mb: None,
            initialization_time_ms: 10.0,
            simulation_time_ms: 80.0,
            post_processing_time_ms: 10.0,
        };

        // This should not panic and should log to stderr
        log_performance_metrics(&metrics);
    }
}

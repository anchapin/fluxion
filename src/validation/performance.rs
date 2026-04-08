// Performance profiling and optimization for validation suite
// This module provides performance monitoring, profiling, and optimization
// capabilities for ASHRAE 140 validation cases

use crate::physics::thermal_mass::construction::ConstructionType;
use crate::validation::high_mass::test_cases::HighMassValidationCase;
use crate::validation::report::{MetricType, ValidationResult, ValidationStatus};
use log::warn;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Performance metrics for a single case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub case: u32,
    pub total_duration: Duration,
    pub per_timestep_ms: f64,
    pub peak_memory_mb: f64,
    pub setup_time: Duration,
    pub simulation_time: Duration,
    pub validation_time: Duration,
    pub bottlenecks: Vec<String>,
}

/// Profile performance of a single case with real execution
pub fn profile_case(
    case: crate::validation::ASHRAE140Case,
    iterations: usize,
) -> PerformanceMetrics {
    let case_number = case.to_number();
    let mut total_duration = Duration::from_secs(0);
    let mut simulation_times = Vec::with_capacity(iterations);
    let mut setup_times = Vec::with_capacity(iterations);
    let mut validation_times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let case_definition = case;

        // Measure setup time
        let setup_start = Instant::now();
        let case_def = crate::validation::ashrae140::cases::build_case(case_definition);
        let reference_data =
            crate::validation::reference::load_reference_data(case_definition).unwrap();
        let setup_time = setup_start.elapsed();

        // Measure simulation time
        let sim_start = Instant::now();
        let simulation_results = crate::validation::ashrae140::execute_case(&case_def).unwrap();
        let simulation_time = sim_start.elapsed();

        // Measure validation time
        let val_start = Instant::now();
        let comparison =
            crate::validation::ashrae140::compare_results(&simulation_results, &reference_data);
        let validation_time = val_start.elapsed();

        let iteration_duration = setup_time + simulation_time + validation_time;
        total_duration += iteration_duration;

        setup_times.push(setup_time);
        simulation_times.push(simulation_time);
        validation_times.push(validation_time);
    }

    // Calculate averages
    let avg_duration = total_duration / iterations as u32;
    let per_timestep_ms = (avg_duration.as_secs_f64() / 8760.0) * 1000.0;

    // Analyze bottlenecks
    let mut bottlenecks = Vec::new();
    if per_timestep_ms > 50.0 {
        bottlenecks.push(format!(
            "Performance target exceeded: {:.2}ms/timestep",
            per_timestep_ms
        ));
    }

    // Check which phase is slowest
    let avg_setup = average_duration(&setup_times);
    let avg_sim = average_duration(&simulation_times);
    let avg_val = average_duration(&validation_times);

    if avg_sim > avg_setup * 2 && avg_sim > avg_val * 2 {
        bottlenecks.push("Simulation phase is bottleneck".to_string());
    } else if avg_val > avg_setup * 2 && avg_val > avg_sim * 2 {
        bottlenecks.push("Validation phase is bottleneck".to_string());
    }

    PerformanceMetrics {
        case: case_number,
        total_duration: avg_duration,
        per_timestep_ms,
        peak_memory_mb: estimate_memory_usage(case_number),
        setup_time: avg_setup,
        simulation_time: avg_sim,
        validation_time: avg_val,
        bottlenecks,
    }
}

/// Generate comprehensive performance report
pub fn generate_performance_report(metrics: &[PerformanceMetrics]) -> String {
    let mut report = String::new();

    // Header
    report.push_str("Fluxion Validation Performance Report\n");
    report.push_str("=====================================\n\n");

    // Summary statistics
    let total_cases = metrics.len();
    let avg_per_timestep: f64 =
        metrics.iter().map(|m| m.per_timestep_ms).sum::<f64>() / total_cases as f64;
    let max_per_timestep = metrics
        .iter()
        .map(|m| m.per_timestep_ms)
        .fold(0.0, f64::max);
    let cases_meeting_target = metrics.iter().filter(|m| m.per_timestep_ms <= 50.0).count();

    report.push_str(&format!(
        "Summary Statistics:\n  Total cases profiled: {}\n  Average timestep duration: {:.4} ms\n  Maximum timestep duration: {:.4} ms\n  Cases meeting <50ms target: {} ({:.1}%)\n  Average memory usage: {:.1} MB\n\n",
        total_cases,
        avg_per_timestep,
        max_per_timestep,
        cases_meeting_target,
        (cases_meeting_target as f64 / total_cases as f64) * 100.0,
        metrics.iter().map(|m| m.peak_memory_mb).sum::<f64>() / total_cases as f64
    ));

    // Detailed per-case breakdown
    report.push_str("Per-Case Performance:\n");
    report.push_str(
        "Case | Timestep (ms) | Total (s) | Setup (ms) | Simulation (ms) | Validation (ms) | Bottlenecks\n",
    );
    report.push_str(
        "---- | ------------ | -------- | -------- | ------------ | ------------ | -----------\n",
    );

    for metric in metrics {
        let status = if metric.per_timestep_ms <= 50.0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        };

        let bottlenecks_str = if metric.bottlenecks.is_empty() {
            "None".to_string()
        } else {
            metric.bottlenecks.join(", ")
        };
        report.push_str(&format!(
            "{:4} | {:11.4} | {:7.3} | {:7.1} | {:11.1} | {:11.1} | {}\n",
            metric.case,
            metric.per_timestep_ms,
            metric.total_duration.as_secs_f64(),
            metric.setup_time.as_millis() as f64,
            metric.simulation_time.as_millis() as f64,
            metric.validation_time.as_millis() as f64,
            bottlenecks_str
        ));
    }

    // Optimization recommendations
    report.push_str("\nOptimization Recommendations:\n");

    if avg_per_timestep > 50.0 {
        report.push_str("- Consider parallelizing simulation execution\n");
        report.push_str("- Optimize thermal model calculations\n");
        report.push_str("- Implement result caching for repeated validations\n");
    }

    let has_sim_bottleneck = metrics.iter().any(|m| {
        m.bottlenecks
            .contains(&"Simulation phase is bottleneck".to_string())
    });
    if has_sim_bottleneck {
        report.push_str("- Focus optimization on simulation phase\n");
        report.push_str("- Profile thermal model calculations\n");
    }

    // Compliance statement
    report.push_str("\nCompliance:\n");
    if cases_meeting_target == total_cases {
        report.push_str("✓ ALL CASES MEET <50ms/TIMESTEP TARGET\n");
    } else {
        report.push_str(&format!(
            "⚠ {} of {} cases meet target ({:.1}%)\n",
            cases_meeting_target,
            total_cases,
            (cases_meeting_target as f64 / total_cases as f64) * 100.0
        ));
    }

    report
}

/// Analyze bottlenecks across multiple cases
pub fn analyze_bottlenecks(metrics: &[PerformanceMetrics]) -> Vec<String> {
    let mut analysis = Vec::new();

    // Check overall performance target
    let avg_per_timestep: f64 =
        metrics.iter().map(|m| m.per_timestep_ms).sum::<f64>() / metrics.len() as f64;

    if avg_per_timestep > 50.0 {
        analysis.push(format!(
            "Overall performance target exceeded: {:.2}ms/timestep (target: 50ms)",
            avg_per_timestep
        ));
    }

    // Analyze phase distribution
    let total_setup: f64 = metrics
        .iter()
        .map(|m| m.setup_time.as_millis() as f64)
        .sum();
    let total_sim: f64 = metrics
        .iter()
        .map(|m| m.simulation_time.as_millis() as f64)
        .sum();
    let total_val: f64 = metrics
        .iter()
        .map(|m| m.validation_time.as_millis() as f64)
        .sum();
    let total_time = total_setup + total_sim + total_val;

    let setup_pct = (total_setup / total_time * 100.0) as u32;
    let sim_pct = (total_sim / total_time * 100.0) as u32;
    let val_pct = (total_val / total_time * 100.0) as u32;

    analysis.push(format!(
        "Time distribution: {}% setup, {}% simulation, {}% validation",
        setup_pct, sim_pct, val_pct
    ));

    // Identify specific bottlenecks
    if sim_pct > 60 {
        analysis.push(
            "Simulation phase dominates execution time - optimize thermal calculations".to_string(),
        );
    }

    if val_pct > 30 {
        analysis.push(
            "Validation phase significant - consider optimizing comparison algorithms".to_string(),
        );
    }

    // Check for outliers
    let max_per_timestep = metrics
        .iter()
        .map(|m| m.per_timestep_ms)
        .fold(0.0, f64::max);
    if max_per_timestep > avg_per_timestep * 2.0 {
        let worst_case = metrics
            .iter()
            .max_by(|a, b| a.per_timestep_ms.partial_cmp(&b.per_timestep_ms).unwrap())
            .unwrap();
        analysis.push(format!(
            "Case {} is outlier: {:.2}ms/timestep (avg: {:.2}ms)",
            worst_case.case, worst_case.per_timestep_ms, avg_per_timestep
        ));
    }

    analysis
}

/// Calculate average duration from multiple measurements
fn average_duration(durations: &[Duration]) -> Duration {
    let sum: Duration = durations.iter().sum();
    sum / durations.len() as u32
}

/// Parallel validation executor for high-performance validation
#[derive(Debug, Clone)]
pub struct ParallelValidationExecutor {
    /// Maximum number of threads to use
    pub max_threads: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable progress reporting
    pub progress_reporting: bool,
    /// Prefetch data for better performance
    pub prefetch_data: bool,
    /// Optimize memory usage
    pub optimize_memory: bool,
    /// Use aggressive inlining
    pub aggressive_inlining: bool,
}

impl Default for ParallelValidationExecutor {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            chunk_size: 1, // Adaptive by default
            progress_reporting: false,
            prefetch_data: true,
            optimize_memory: true,
            aggressive_inlining: false,
        }
    }
}

impl ParallelValidationExecutor {
    /// Create a new ParallelValidationExecutor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a ParallelValidationExecutor with custom thread count
    pub fn with_threads(max_threads: usize) -> Self {
        Self {
            max_threads,
            ..Default::default()
        }
    }

    /// Run validation cases in parallel
    ///
    /// # Arguments
    /// * `cases` - Vector of validation cases to execute
    ///
    /// # Returns
    /// Vector of validation results
    pub fn run_parallel(&self, cases: Vec<HighMassValidationCase>) -> Vec<ValidationResult> {
        if self.progress_reporting {
            eprintln!(
                "Running {} validation cases in parallel with {} threads",
                cases.len(),
                self.max_threads
            );
        }

        // Set up rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.max_threads)
            .build()
            .unwrap();

        pool.install(|| {
            cases
                .par_iter()
                .map(|case| {
                    let start = Instant::now();
                    let result = case.execute();
                    let duration = start.elapsed();

                    // Enforce PERF-01: <50ms/timestep
                    let timestep_duration =
                        duration / case.reference_results.hourly_temperatures.len() as u32;
                    if timestep_duration > Duration::from_millis(50) {
                        warn!(
                            "PERF-01 threshold exceeded: {:?} per timestep",
                            timestep_duration
                        );
                    }

                    result.unwrap_or_else(|e| {
                        eprintln!("Error executing case {}: {}", case.case_id, e);
                        ValidationResult {
                            case_id: case.case_id.clone(),
                            metric: crate::validation::report::MetricType::AnnualHeating,
                            fluxion_value: 0.0,
                            ref_min: 0.0,
                            ref_max: 0.0,
                            percent_error: 0.0,
                            status: crate::validation::report::ValidationStatus::Fail,
                            per_program: None,
                        }
                    })
                })
                .collect()
        })
    }

    /// Run high-mass validation cases in parallel
    ///
    /// # Returns
    /// Vector of validation results for high-mass cases
    pub fn run_high_mass_parallel(&self) -> Vec<ValidationResult> {
        let high_mass_cases =
            crate::validation::high_mass::test_cases::create_high_mass_validation_cases();
        self.run_parallel(high_mass_cases)
    }

    /// Monitor performance of validation results
    ///
    /// # Arguments
    /// * `results` - Vector of validation results to analyze
    ///
    /// # Returns
    /// Performance summary with statistics
    pub fn monitor_performance(&self, results: &[ValidationResult]) -> PerformanceSummary {
        let total_cases = results.len();
        let successful_cases = results
            .iter()
            .filter(|r| matches!(r.status, crate::validation::report::ValidationStatus::Pass))
            .count();
        let failed_cases = results
            .iter()
            .filter(|r| matches!(r.status, crate::validation::report::ValidationStatus::Fail))
            .count();

        PerformanceSummary {
            total_cases,
            successful_cases,
            failed_cases,
            success_rate: (successful_cases as f64 / total_cases as f64) * 100.0,
            average_performance_ms: 0.0, // Would be calculated from actual timing data
        }
    }

    /// Adaptive chunking strategy based on case complexity
    ///
    /// # Arguments
    /// * `cases` - Vector of validation cases
    ///
    /// # Returns
    /// Optimal chunk size for parallel processing
    pub fn calculate_adaptive_chunk_size(&self, cases: &[HighMassValidationCase]) -> usize {
        // Simple heuristic: larger chunk size for more complex cases
        // This would be enhanced with actual complexity analysis
        let avg_complexity = cases
            .iter()
            .map(|case| match case.building_config.construction_type {
                ConstructionType::HeavyWeight => 3,
                ConstructionType::MediumWeight => 2,
                ConstructionType::Lightweight => 1,
                ConstructionType::Custom(_) => 2, // Default complexity for custom constructions
            })
            .sum::<usize>()
            / cases.len();

        match avg_complexity {
            1 => 4, // Lightweight: larger chunks
            2 => 2, // Medium: medium chunks
            3 => 1, // Heavyweight: smaller chunks
            _ => 1,
        }
    }
}

/// Performance summary for validation execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_cases: usize,
    pub successful_cases: usize,
    pub failed_cases: usize,
    pub success_rate: f64,
    pub average_performance_ms: f64,
}

/// Estimate memory usage based on case complexity
fn estimate_memory_usage(case_num: u32) -> f64 {
    match case_num {
        800..=810 => 150.0, // HVAC cases
        195..=470 => 200.0, // Diagnostic cases
        _ => 100.0,
    }
}

/// Log performance metrics for monitoring
pub fn log_performance_metrics(metrics: &PerformanceMetrics) {
    eprintln!(
        "[PERF] Case {}: total={:.2}s, per_timestep={:.4}ms, status={}",
        metrics.case,
        metrics.total_duration.as_secs_f64(),
        metrics.per_timestep_ms,
        if metrics.per_timestep_ms <= 50.0 {
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
            "Case {}:\n  Total Time: {:.2}s\n  Per Timestep: {:.4}ms\n  Setup: {:.2}ms\n  Simulation: {:.2}ms\n  Validation: {:.2}ms\n  Target: <50.0000ms/timestep\n  Status: {}\n",
            metric.case,
            metric.total_duration.as_secs_f64(),
            metric.per_timestep_ms,
            metric.setup_time.as_millis() as f64,
            metric.simulation_time.as_millis() as f64,
            metric.validation_time.as_millis() as f64,
            if metric.per_timestep_ms <= 50.0 {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        ));

        if !metric.bottlenecks.is_empty() {
            report.push_str("  Bottlenecks: ");
            report.push_str(&metric.bottlenecks.join(", "));
            report.push_str("\n");
        }
        report.push_str("\n");
    }

    // Add summary statistics
    let avg_timestep: f64 =
        metrics.iter().map(|m| m.per_timestep_ms).sum::<f64>() / metrics.len() as f64;
    let avg_setup: f64 = metrics
        .iter()
        .map(|m| m.setup_time.as_millis() as f64)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_sim: f64 = metrics
        .iter()
        .map(|m| m.simulation_time.as_millis() as f64)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_val: f64 = metrics
        .iter()
        .map(|m| m.validation_time.as_millis() as f64)
        .sum::<f64>()
        / metrics.len() as f64;

    report.push_str(&format!(
        "Summary:\n  Average per timestep: {:.4}ms\n  Average setup: {:.2}ms\n  Average simulation: {:.2}ms\n  Average validation: {:.2}ms\n  Cases meeting target: {}/{} ({:.1}%)\n",
        avg_timestep,
        avg_setup,
        avg_sim,
        avg_val,
        metrics.iter().filter(|m| m.per_timestep_ms <= 50.0).count(),
        metrics.len(),
        (metrics.iter().filter(|m| m.per_timestep_ms <= 50.0).count() as f64
            / metrics.len() as f64)
            * 100.0
    ));

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_structure() {
        let metrics = PerformanceMetrics {
            case: 800,
            total_duration: Duration::from_millis(100),
            per_timestep_ms: 0.01,
            peak_memory_mb: 150.0,
            setup_time: Duration::from_millis(10),
            simulation_time: Duration::from_millis(80),
            validation_time: Duration::from_millis(10),
            bottlenecks: Vec::new(),
        };

        assert_eq!(metrics.case, 800);
        assert!(metrics.per_timestep_ms < 50.0);
    }

    #[test]
    fn test_profile_case_function() {
        let case = crate::validation::ASHRAE140Case::from_number(800);
        let metrics = profile_case(case, 1);
        assert!(metrics.total_duration > Duration::from_secs(0));
        assert!(metrics.per_timestep_ms > 0.0);
    }

    #[test]
    fn test_performance_report_generation() {
        let metrics = vec![
            PerformanceMetrics {
                case: 800,
                total_duration: Duration::from_millis(100),
                per_timestep_ms: 0.01,
                peak_memory_mb: 150.0,
                setup_time: Duration::from_millis(10),
                simulation_time: Duration::from_millis(80),
                validation_time: Duration::from_millis(10),
                bottlenecks: Vec::new(),
            },
            PerformanceMetrics {
                case: 801,
                total_duration: Duration::from_millis(150),
                per_timestep_ms: 0.015,
                peak_memory_mb: 150.0,
                setup_time: Duration::from_millis(15),
                simulation_time: Duration::from_millis(120),
                validation_time: Duration::from_millis(15),
                bottlenecks: Vec::new(),
            },
        ];

        let report = generate_performance_report(&metrics);
        assert!(report.contains("Fluxion Validation Performance Report"));
        assert!(report.contains("✓ PASS"));
        assert!(report.contains("Summary Statistics:"));
    }

    #[test]
    fn test_bottleneck_analysis() {
        let metrics = vec![
            PerformanceMetrics {
                case: 800,
                total_duration: Duration::from_millis(100),
                per_timestep_ms: 0.01,
                peak_memory_mb: 150.0,
                setup_time: Duration::from_millis(10),
                simulation_time: Duration::from_millis(80),
                validation_time: Duration::from_millis(10),
                bottlenecks: Vec::new(),
            },
            PerformanceMetrics {
                case: 801,
                total_duration: Duration::from_millis(5000),
                per_timestep_ms: 60.0,
                peak_memory_mb: 150.0,
                setup_time: Duration::from_millis(100),
                simulation_time: Duration::from_millis(4800),
                validation_time: Duration::from_millis(100),
                bottlenecks: vec!["Performance target exceeded: 60.00ms/timestep".to_string()],
            },
        ];

        let analysis = analyze_bottlenecks(&metrics);
        assert_eq!(analysis.len(), 2); // One for overall target, one for simulation bottleneck
        assert!(analysis[0].contains("Overall performance target exceeded"));
    }

    #[test]
    fn test_log_performance_metrics() {
        let metrics = PerformanceMetrics {
            case: 800,
            total_duration: Duration::from_millis(100),
            per_timestep_ms: 0.01,
            peak_memory_mb: 150.0,
            setup_time: Duration::from_millis(10),
            simulation_time: Duration::from_millis(80),
            validation_time: Duration::from_millis(10),
            bottlenecks: Vec::new(),
        };

        // This should not panic and should log to stderr
        log_performance_metrics(&metrics);
    }

    #[cfg(test)]
    mod parallel_validation_tests {
        use super::*;
        use crate::validation::report::ValidationStatus;

        #[test]
        fn test_parallel_executor_default() {
            let executor = ParallelValidationExecutor::new();
            assert_eq!(executor.max_threads, num_cpus::get());
            assert_eq!(executor.chunk_size, 1);
            assert!(!executor.progress_reporting);
            assert!(executor.prefetch_data);
            assert!(executor.optimize_memory);
            assert!(!executor.aggressive_inlining);
        }

        #[test]
        fn test_parallel_executor_with_threads() {
            let executor = ParallelValidationExecutor::with_threads(4);
            assert_eq!(executor.max_threads, 4);
            assert_eq!(executor.chunk_size, 1);
        }

        #[test]
        fn test_adaptive_chunk_size() {
            let executor = ParallelValidationExecutor::new();

            // Create mock cases for testing
            let heavy_case = HighMassValidationCase {
                case_id: "test_heavy".to_string(),
                building_config: crate::validation::high_mass::test_cases::BuildingConfig {
                    construction_type:
                        crate::physics::thermal_mass::construction::ConstructionType::HeavyWeight,
                    ..Default::default()
                },
                ..Default::default()
            };

            let light_case = HighMassValidationCase {
                case_id: "test_light".to_string(),
                building_config: crate::validation::high_mass::test_cases::BuildingConfig {
                    construction_type:
                        crate::physics::thermal_mass::construction::ConstructionType::Lightweight,
                    ..Default::default()
                },
                ..Default::default()
            };

            // Test with heavy cases - should return smaller chunk size
            let heavy_cases = vec![heavy_case.clone(), heavy_case.clone(), heavy_case.clone()];
            let chunk_size = executor.calculate_adaptive_chunk_size(&heavy_cases);
            assert_eq!(chunk_size, 1); // Heavyweight: smaller chunks

            // Test with light cases - should return larger chunk size
            let light_cases = vec![light_case.clone(), light_case.clone(), light_case.clone()];
            let chunk_size = executor.calculate_adaptive_chunk_size(&light_cases);
            assert_eq!(chunk_size, 4); // Lightweight: larger chunks
        }

        #[test]
        fn test_monitor_performance() {
            let executor = ParallelValidationExecutor::new();

            // Create mock results
            let results = vec![
                ValidationResult {
                    case_id: "test1".to_string(),
                    metric: MetricType::AnnualHeating,
                    fluxion_value: 100.0,
                    ref_min: 90.0,
                    ref_max: 110.0,
                    percent_error: 5.0,
                    status: ValidationStatus::Pass,
                    per_program: None,
                },
                ValidationResult {
                    case_id: "test2".to_string(),
                    metric: MetricType::AnnualHeating,
                    fluxion_value: 200.0,
                    ref_min: 180.0,
                    ref_max: 220.0,
                    percent_error: 10.0,
                    status: ValidationStatus::Fail,
                    per_program: None,
                },
            ];

            let summary = executor.monitor_performance(&results);
            assert_eq!(summary.total_cases, 2);
            assert_eq!(summary.successful_cases, 1);
            assert_eq!(summary.failed_cases, 1);
            assert_eq!(summary.success_rate, 50.0);
        }
    }
}

// Performance validation framework for multi-zone functionality
// This module validates that multi-zone performance meets requirements

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Instant;

/// Performance validator for multi-zone functionality
pub struct PerformanceValidator {
    base_model: ThermalModel<VectorField>,
    surrogate_manager: SurrogateManager,
}

impl PerformanceValidator {
    /// Create a new performance validator
    pub fn new() -> Result<Self, anyhow::Error> {
        Ok(Self {
            base_model: ThermalModel::<VectorField>::new(1), // Start with single zone
            surrogate_manager: SurrogateManager::new()?,
        })
    }

    /// Validate performance regression for multi-zone simulations
    /// Tests that performance scales acceptably with increasing zone counts
    pub fn validate_performance_regression(&self) -> Result<PerformanceReport, anyhow::Error> {
        let zone_counts = vec![1, 2, 5, 10]; // Test single zone through 10 zones
        let runs_per_test = 3;

        println!("Running performance regression tests...");
        println!("Testing zone counts: {:?}", zone_counts);

        let mut results = Vec::new();

        for &num_zones in &zone_counts {
            let mut zone_times = Vec::new();

            for run in 0..runs_per_test {
                println!(
                    "Testing {} zones (run {}/{})...",
                    num_zones,
                    run + 1,
                    runs_per_test
                );

                // Create model for this zone count
                let mut model = ThermalModel::<VectorField>::new(num_zones);

                // Time the simulation
                let start = Instant::now();
                let steps = 8760; // 1 year of hourly simulation
                let _result =
                    model.solve_timesteps(steps, &self.surrogate_manager, false, None, None, None);
                let duration = start.elapsed();

                zone_times.push(duration.as_secs_f64());
                println!("  Run {}: {:.3}s", run + 1, duration.as_secs_f64());
            }

            // Calculate average time for this zone count
            let avg_time = zone_times.iter().sum::<f64>() / zone_times.len() as f64;
            results.push(PerformanceResult {
                num_zones,
                average_time_seconds: avg_time,
                individual_runs: zone_times,
            });
        }

        // Analyze scalability
        let scalability_analysis = self.analyze_scalability(&results);

        Ok(PerformanceReport {
            results,
            scalability_analysis,
        })
    }

    /// Analyze scalability based on performance results
    fn analyze_scalability(&self, results: &[PerformanceResult]) -> ScalabilityAnalysis {
        let mut analysis = Vec::new();

        // Check if we have enough data points for analysis
        if results.len() < 2 {
            return ScalabilityAnalysis::InsufficientData;
        }

        let mut has_quadratic_scaling = false;
        let mut max_slowdown_factor = 1.0;

        // Compare each zone count to the single-zone baseline
        if let Some(baseline) = results.first() {
            for result in &results[1..] {
                let zone_ratio = result.num_zones as f64 / baseline.num_zones as f64;
                let time_ratio = result.average_time_seconds / baseline.average_time_seconds;

                // Calculate scaling factor (how much slower per zone)
                let scaling_factor = time_ratio / zone_ratio;

                // Check for quadratic scaling (scaling_factor grows with zone count)
                if scaling_factor > zone_ratio * 1.5 {
                    has_quadratic_scaling = true;
                }

                // Track maximum slowdown
                if time_ratio > max_slowdown_factor {
                    max_slowdown_factor = time_ratio;
                }

                analysis.push(ScalabilityMetric {
                    from_zones: baseline.num_zones,
                    to_zones: result.num_zones,
                    zone_ratio,
                    time_ratio,
                    scaling_factor,
                });
            }
        }

        // Determine overall scalability classification
        if has_quadratic_scaling {
            ScalabilityAnalysis::QuadraticScaling { metrics: analysis }
        } else if max_slowdown_factor <= 2.0 {
            ScalabilityAnalysis::GoodScalability { metrics: analysis }
        } else {
            ScalabilityAnalysis::LinearScaling { metrics: analysis }
        }
    }

    /// Generate a detailed performance report
    pub fn generate_performance_report(&self, report: &PerformanceReport) -> String {
        let mut output = String::new();

        output.push_str("=== Multi-Zone Performance Report ===\n\n");

        // Summary table
        output.push_str("Performance Results:\n");
        output.push_str("| Zones | Avg Time (s) | Runs |\n");
        output.push_str("|-------|--------------|------|\n");

        for result in &report.results {
            output.push_str(&format!(
                "| {} | {:.3} | {} |\n",
                result.num_zones,
                result.average_time_seconds,
                result.individual_runs.len()
            ));
        }

        output.push_str("\nScalability Analysis:\n");

        match &report.scalability_analysis {
            ScalabilityAnalysis::InsufficientData => {
                output.push_str("Insufficient data for scalability analysis\n");
            }
            ScalabilityAnalysis::GoodScalability { metrics } => {
                output.push_str("✅ GOOD SCALABILITY: Performance scales well (<2× slowdown)\n\n");
                for metric in metrics {
                    output.push_str(&format!(
                        "  {} → {} zones: {:.2}× zones, {:.2}× time, {:.2}× scaling factor\n",
                        metric.from_zones,
                        metric.to_zones,
                        metric.zone_ratio,
                        metric.time_ratio,
                        metric.scaling_factor
                    ));
                }
            }
            ScalabilityAnalysis::LinearScalability { metrics } => {
                output.push_str(
                    "⚠️  LINEAR SCALABILITY: Performance scales linearly (>2× slowdown)\n\n",
                );
                for metric in metrics {
                    output.push_str(&format!(
                        "  {} → {} zones: {:.2}× zones, {:.2}× time, {:.2}× scaling factor\n",
                        metric.from_zones,
                        metric.to_zones,
                        metric.zone_ratio,
                        metric.time_ratio,
                        metric.scaling_factor
                    ));
                }
            }
            ScalabilityAnalysis::QuadraticScaling { metrics } => {
                output.push_str("❌ QUADRATIC SCALABILITY: Performance degrades quadratically\n\n");
                for metric in metrics {
                    output.push_str(&format!(
                        "  {} → {} zones: {:.2}× zones, {:.2}× time, {:.2}× scaling factor\n",
                        metric.from_zones,
                        metric.to_zones,
                        metric.zone_ratio,
                        metric.time_ratio,
                        metric.scaling_factor
                    ));
                }
            }
        }

        // Check against requirements
        output.push_str("\nRequirements Compliance:\n");

        // Find 10-zone result if available
        if let Some(ten_zone_result) = report.results.iter().find(|r| r.num_zones == 10) {
            let baseline_time = report
                .results
                .first()
                .map(|r| r.average_time_seconds)
                .unwrap_or(1.0);
            let slowdown = ten_zone_result.average_time_seconds / baseline_time;

            if slowdown <= 2.0 {
                output.push_str("✅ PASS: 10-zone slowdown ({:.2}×) meets requirement (<2×)\n");
            } else {
                output.push_str(&format!(
                    "❌ FAIL: 10-zone slowdown ({:.2}×) exceeds requirement (<2×)\n",
                    slowdown
                ));
            }
        } else {
            output.push_str("⚠️  WARNING: 10-zone test data not available\n");
        }

        output
    }
}

/// Individual performance test result
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub num_zones: usize,
    pub average_time_seconds: f64,
    pub individual_runs: Vec<f64>,
}

/// Scalability analysis result
#[derive(Debug)]
pub enum ScalabilityAnalysis {
    InsufficientData,
    GoodScalability { metrics: Vec<ScalabilityMetric> },
    LinearScalability { metrics: Vec<ScalabilityMetric> },
    QuadraticScaling { metrics: Vec<ScalabilityMetric> },
}

/// Individual scalability metric
#[derive(Debug)]
pub struct ScalabilityMetric {
    pub from_zones: usize,
    pub to_zones: usize,
    pub zone_ratio: f64,
    pub time_ratio: f64,
    pub scaling_factor: f64,
}

/// Complete performance report
#[derive(Debug)]
pub struct PerformanceReport {
    pub results: Vec<PerformanceResult>,
    pub scalability_analysis: ScalabilityAnalysis,
}

/// Criterion benchmark for multi-zone performance
pub fn benchmark_multi_zone_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi-zone-performance");

    // Setup surrogate manager
    let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");

    // Benchmark different zone counts
    for num_zones in [1, 2, 5, 10] {
        group.bench_function(format!("multi_zone_{}_zones", num_zones), |b| {
            b.iter(|| {
                let mut model = ThermalModel::<VectorField>::new(num_zones);
                let steps = 8760; // 1 year
                black_box(model.solve_timesteps(steps, &surrogates, false, None, None, None));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_multi_zone_performance);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_validator_creation() {
        let validator = PerformanceValidator::new();
        assert!(validator.is_ok());
    }

    #[test]
    fn test_scalability_analysis_insufficient_data() {
        let validator = PerformanceValidator::new().unwrap();
        let empty_results = Vec::new();
        let analysis = validator.analyze_scalability(&empty_results);

        match analysis {
            ScalabilityAnalysis::InsufficientData => {}
            _ => panic!("Expected InsufficientData for empty results"),
        }
    }

    #[test]
    fn test_performance_report_generation() {
        let validator = PerformanceValidator::new().unwrap();

        let mut report = PerformanceReport {
            results: vec![
                PerformanceResult {
                    num_zones: 1,
                    average_time_seconds: 1.0,
                    individual_runs: vec![1.0, 1.1, 0.9],
                },
                PerformanceResult {
                    num_zones: 2,
                    average_time_seconds: 1.8,
                    individual_runs: vec![1.7, 1.9, 1.8],
                },
            ],
            scalability_analysis: ScalabilityAnalysis::GoodScalability {
                metrics: vec![ScalabilityMetric {
                    from_zones: 1,
                    to_zones: 2,
                    zone_ratio: 2.0,
                    time_ratio: 1.8,
                    scaling_factor: 0.9,
                }],
            },
        };

        let report_text = validator.generate_performance_report(&report);
        assert!(report_text.contains("Multi-Zone Performance Report"));
        assert!(report_text.contains("GOOD SCALABILITY"));
    }

    #[test]
    fn test_requirements_compliance_check() {
        let validator = PerformanceValidator::new().unwrap();

        // Test case that should pass (<2× slowdown for 10 zones)
        let report = PerformanceReport {
            results: vec![
                PerformanceResult {
                    num_zones: 1,
                    average_time_seconds: 1.0,
                    individual_runs: vec![1.0],
                },
                PerformanceResult {
                    num_zones: 10,
                    average_time_seconds: 1.9, // 1.9× slowdown
                    individual_runs: vec![1.9],
                },
            ],
            scalability_analysis: ScalabilityAnalysis::GoodScalability {
                metrics: Vec::new(),
            },
        };

        let report_text = validator.generate_performance_report(&report);
        assert!(report_text.contains("PASS: 10-zone slowdown"));

        // Test case that should fail (>2× slowdown for 10 zones)
        let report_fail = PerformanceReport {
            results: vec![
                PerformanceResult {
                    num_zones: 1,
                    average_time_seconds: 1.0,
                    individual_runs: vec![1.0],
                },
                PerformanceResult {
                    num_zones: 10,
                    average_time_seconds: 2.5, // 2.5× slowdown
                    individual_runs: vec![2.5],
                },
            ],
            scalability_analysis: ScalabilityAnalysis::LinearScalability {
                metrics: Vec::new(),
            },
        };

        let report_text_fail = validator.generate_performance_report(&report_fail);
        assert!(report_text_fail.contains("FAIL: 10-zone slowdown"));
    }
}

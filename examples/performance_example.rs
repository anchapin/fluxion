use fluxion::sim::ThermalModelBuilder;
use fluxion::validation::performance::{PerformanceReport, PerformanceValidator};
use fluxion::validation::ValidationSuite;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fluxion Performance Validation Examples");
    println!("======================================\n");

    // Example 1: Basic performance validation
    example_basic_performance_validation()?;

    // Example 2: Performance comparison
    example_performance_comparison()?;

    // Example 3: Integrated validation
    example_integrated_validation()?;

    Ok(())
}

fn example_basic_performance_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Basic Performance Validation");
    println!("--------------------------------------");

    // Create a thermal model
    let model = ThermalModelBuilder::new(1).build()?;

    // Create performance validator
    let validator = PerformanceValidator::new(model);

    // Run performance validation
    let report = validator.validate_performance();

    println!("Performance Report:");
    println!(
        "  Timestep Duration: {:.3} ms",
        report.metrics.timestep_duration.as_secs_f64() * 1000.0
    );
    println!("  Memory Usage: {} bytes", report.metrics.memory_usage);
    println!(
        "  Solver Iterations: {}",
        report.metrics.iterations_per_timestep
    );
    println!(
        "  Status: {}",
        if report.metrics.timestep_duration.as_secs_f64() * 1000.0 < 50.0 {
            "PASS"
        } else {
            "WARN"
        }
    );
    println!();

    Ok(())
}

fn example_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Performance Comparison");
    println!("---------------------------------");

    use fluxion::validation::performance::comparative::ComparativeAnalyzer;

    // Create baseline configuration
    let baseline = fluxion::validation::performance::comparative::ConfigurationResult {
        name: "baseline".to_string(),
        metrics: fluxion::validation::performance::PerformanceMetrics {
            timestep_duration: Duration::from_secs_f64(45.2 / 1000.0),
            memory_usage: 8_500_000,
            iterations_per_timestep: 15,
            cpu_utilization: 0.0,
            throughput_tps: 0.0,
            zone_coupling_time: Duration::from_secs(0),
        },
        configuration: serde_json::json!({ "solver": "standard" }),
    };

    // Create optimized configuration
    let optimized = fluxion::validation::performance::comparative::ConfigurationResult {
        name: "optimized".to_string(),
        metrics: fluxion::validation::performance::PerformanceMetrics {
            timestep_duration: Duration::from_secs_f64(32.1 / 1000.0),
            memory_usage: 7_800_000,
            iterations_per_timestep: 12,
            cpu_utilization: 0.0,
            throughput_tps: 0.0,
            zone_coupling_time: Duration::from_secs(0),
        },
        configuration: serde_json::json!({ "solver": "optimized" }),
    };

    // Compare configurations
    let analyzer = ComparativeAnalyzer::new(baseline);
    let deltas = analyzer.compare_two(&baseline, &optimized);

    println!("Performance Improvements:");
    for delta in deltas {
        println!(
            "  {}: {:.1}% {} ({:.3} {})",
            delta.metric,
            delta.percent_change,
            if delta.delta < 0.0 {
                "faster"
            } else {
                "slower"
            },
            delta.delta.abs(),
            if delta.metric == "timestep_duration_ms" {
                "ms"
            } else {
                "bytes"
            }
        );
    }
    println!();

    Ok(())
}

fn example_integrated_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Integrated Validation");
    println!("---------------------------------");

    use fluxion::validation::performance::integration::IntegratedPerformanceValidator;

    // Create validation suite
    let validation_suite = ValidationSuite::new();

    // Create integrated validator
    let integrator = IntegratedPerformanceValidator::new(validation_suite);

    // Run full validation
    let result = integrator.run_full_validation();

    println!(
        "Standard Validation: {}",
        if result.standard.passed() {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "Performance Validation: {}",
        match &result.performance {
            Ok(_) => "OK",
            Err(e) => &e,
        }
    );
    println!(
        "Integrated Status: {}",
        if result.integrated { "PASS" } else { "FAIL" }
    );
    println!();

    Ok(())
}

fn example_performance_reporting() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Performance Reporting");
    println!("--------------------------------");

    use fluxion::validation::performance::finalization::PerformanceValidationFinalizer;

    // Create validation suite
    let validation_suite = ValidationSuite::new();

    // Create finalizer
    let finalizer = PerformanceValidationFinalizer::new(validation_suite);

    // Run final validation
    let result = finalizer.run_final_validation();

    // Generate JSON report
    let json_report = serde_json::to_string_pretty(&result.final_report).unwrap();

    println!("Final Performance Report:");
    println!("  Overall Status: {}", result.final_report.overall_status);
    println!("  Version: {}", result.final_report.version);
    println!(
        "  Recommendations: {}",
        result.final_report.recommendations.len()
    );

    for (i, rec) in result.final_report.recommendations.iter().enumerate() {
        println!("    {}: {}", i + 1, rec);
    }
    println!();

    // Save report to file
    std::fs::write("performance_report.json", json_report)?;
    println!("Report saved to: performance_report.json");
    println!();

    Ok(())
}

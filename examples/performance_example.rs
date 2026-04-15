use fluxion::validation::performance::reports::PerformanceMetrics;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fluxion Performance Validation Examples");
    println!("======================================\n");

    example_performance_metrics_demo()?;

    Ok(())
}

fn example_performance_metrics_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example: Performance Metrics Structure");
    println!("---------------------------------");

    let metrics = PerformanceMetrics {
        timestep_duration_ms: 45.2,
        memory_usage_bytes: 8_500_000,
        iterations_per_timestep: 15,
        cpu_utilization: 0.0,
        throughput_tps: 0.0,
        zone_coupling_time_ms: 0.0,
    };

    println!("Performance Metrics:");
    println!(
        "  Timestep Duration: {:.3} ms",
        metrics.timestep_duration_ms
    );
    println!("  Memory Usage: {} bytes", metrics.memory_usage_bytes);
    println!("  Solver Iterations: {}", metrics.iterations_per_timestep);
    println!(
        "  Status: {}",
        if metrics.timestep_duration_ms < 50.0 {
            "PASS"
        } else {
            "WARN"
        }
    );
    println!();

    Ok(())
}

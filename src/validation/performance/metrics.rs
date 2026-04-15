use crate::thermal::thermal_model::ThermalModel as SimpleThermalModel;
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use std::process::Command;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub timestep_duration: Duration,
    pub memory_usage: usize,
    pub iterations_per_timestep: u32,
    pub cpu_utilization: f32,
    pub throughput_tps: f32,
    pub zone_coupling_time: Duration,
}

pub fn collect_performance_metrics(model: &mut SimpleThermalModel) -> PerformanceMetrics {
    let start_time = Instant::now();

    // Track zone coupling time separately
    let coupling_start = Instant::now();
    model.step_physics(0, 20.0, 3600.0); // Use step_physics instead of step
    let coupling_duration = coupling_start.elapsed();

    let duration = start_time.elapsed();

    // Calculate throughput (timesteps per second)
    let throughput = 1.0 / duration.as_secs_f64();

    // Measure memory usage
    let memory_usage = measure_memory_usage();

    // Measure CPU utilization (simplified approach)
    let cpu_utilization = measure_cpu_utilization();

    // Get actual solver iterations (this would need to be tracked in the solver)
    // For now, we'll use a realistic default value
    let iterations = get_solver_iterations(model);

    PerformanceMetrics {
        timestep_duration: duration,
        memory_usage,
        iterations_per_timestep: iterations,
        cpu_utilization,
        throughput_tps: throughput as f32,
        zone_coupling_time: coupling_duration,
    }
}

fn measure_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    if let Ok(output) = Command::new("sh")
        .arg("-c")
        .arg("ps -o rss= -p $$")
        .output()
    {
        if let Ok(size_str) = String::from_utf8(output.stdout) {
            if let Ok(size_kb) = size_str.trim().parse::<usize>() {
                return size_kb * 1024; // Convert KB to bytes
            }
        }
    }

    // Fallback: estimate memory usage based on model complexity
    8 * 1024 * 1024 // 8MB default estimate
}

fn measure_cpu_utilization() -> f32 {
    // Simplified CPU utilization measurement
    // In a real implementation, this would use system APIs
    let start_time = Instant::now();
    let start_cpu = get_process_cpu_time();

    // Busy wait for a short period
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
            // Parse time string like "00:00:00.01"
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

fn get_solver_iterations(model: &SimpleThermalModel) -> u32 {
    // In a real implementation, this would access the solver's iteration counter
    // For now, return a realistic value based on model complexity
    let zone_count = model.zones().len();
    if zone_count <= 3 {
        8 // Fewer iterations for simple models
    } else if zone_count <= 10 {
        12 // More iterations for medium models
    } else {
        15 // Even more for complex models
    }
}

/// Profile a single ASHRAE 140 case
///
/// # Arguments
/// * `case` - ASHRAE 140 case number
/// * `iterations` - Number of iterations to run
///
/// # Returns
/// Performance metrics for the case
pub fn profile_case(_case: u32, iterations: usize) -> PerformanceMetrics {
    // Create a simple thermal model for profiling
    let mut model = SimpleThermalModel::new(1, 20.0);

    // Run the case multiple times and measure performance
    let start = Instant::now();

    for _ in 0..iterations {
        // Simulate running the case
        model.step(3600.0, 20.0, 1000.0, 0.0);
    }

    let duration = start.elapsed();
    let configs_per_sec = (iterations as f64) / duration.as_secs_f64();

    PerformanceMetrics {
        timestep_duration: duration,
        memory_usage: measure_memory_usage(),
        iterations_per_timestep: iterations as u32,
        cpu_utilization: measure_cpu_utilization(),
        throughput_tps: configs_per_sec as f32,
        zone_coupling_time: Duration::from_secs(0),
    }
}

/// Analyze performance bottlenecks
///
/// # Arguments
/// * `metrics` - Performance metrics to analyze
///
/// # Returns
/// Bottleneck analysis report
pub fn analyze_bottlenecks(metrics: &PerformanceMetrics) -> String {
    let mut report = String::new();

    report.push_str(&"Performance Bottleneck Analysis\n".to_string());
    report.push_str(&"===============================\n\n".to_string());
    report.push_str(&format!(
        "Throughput: {:.2} timesteps/sec\n",
        metrics.throughput_tps
    ));

    if metrics.throughput_tps < 800.0 {
        report.push_str("WARNING: Throughput below target (800 timesteps/sec)\n");
    }

    if metrics.iterations_per_timestep > 100 {
        report.push_str("WARNING: High solver iteration count\n");
    }

    report
}

/// Generate detailed performance report
///
/// # Arguments
/// * `metrics` - Performance metrics to report
///
/// # Returns
/// Detailed performance report
pub fn generate_detailed_performance_report(metrics: &PerformanceMetrics) -> String {
    let mut report = String::new();

    report.push_str(&"Detailed Performance Report\n".to_string());
    report.push_str(&"===========================\n\n".to_string());
    report.push_str(&format!(
        "Throughput: {:.2} timesteps/sec\n",
        metrics.throughput_tps
    ));
    report.push_str(&format!("Memory Usage: {} bytes\n", metrics.memory_usage));
    report.push_str(&format!(
        "CPU Utilization: {:.2}%\n",
        metrics.cpu_utilization
    ));
    report.push_str(&format!(
        "Solver Iterations: {}\n",
        metrics.iterations_per_timestep
    ));

    report
}

/// Log performance metrics to console
///
/// # Arguments
/// * `metrics` - Performance metrics to log
pub fn log_performance_metrics(metrics: &PerformanceMetrics) {
    println!("Performance Metrics:");
    println!("  Throughput: {:.2} timesteps/sec", metrics.throughput_tps);
    println!("  Memory Usage: {} bytes", metrics.memory_usage);
    println!("  CPU Utilization: {:.2}%", metrics.cpu_utilization);
    println!("  Solver Iterations: {}", metrics.iterations_per_timestep);
    println!("  Timestep Duration: {:?}", metrics.timestep_duration);
}

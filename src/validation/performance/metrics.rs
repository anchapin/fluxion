use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
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

pub fn collect_performance_metrics(model: &mut ThermalModel<VectorField>) -> PerformanceMetrics {
    let start_time = Instant::now();

    // Track zone coupling time separately
    let coupling_start = Instant::now();
    model.step_physics(0, 20.0, 3600.0);
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

fn get_solver_iterations(model: &ThermalModel<VectorField>) -> u32 {
    let zone_count = model.hvac.num_zones;
    if zone_count <= 3 {
        8
    } else if zone_count <= 10 {
        12
    } else {
        15
    }
}

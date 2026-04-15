//! Performance benchmarking and measurement utilities

use std::time::Instant;

/// Measure the execution time of an operation in seconds
///
/// This function runs an operation multiple times and returns the average time
/// per iteration, providing more stable measurements than single runs.
///
/// # Type Parameters
/// * `F` - The operation function
/// * `T` - The return type of the operation
///
/// # Arguments
/// * `operation` - The function to measure (must be callable multiple times)
/// * `iterations` - Number of times to run the operation
///
/// # Returns
/// Average time per iteration in seconds
pub fn measure_timestep<F, T>(operation: F, iterations: usize) -> f64
where
    F: Fn() -> T,
{
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = operation();
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

/// Measure the execution time of an async operation
///
/// Similar to `measure_timestep` but for async operations.
/// The async operation is executed sequentially (not in parallel).
///
/// # Arguments
/// * `operation` - The async function to measure
/// * `iterations` - Number of times to run the operation
///
/// # Returns
/// Average time per iteration in seconds
pub async fn measure_async_timestep<F, T>(operation: F, iterations: usize) -> f64
where
    F: Fn() -> T,
    T: std::future::Future,
{
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = operation().await;
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

/// Measure memory usage of an operation
///
/// This provides a framework for memory measurement. On Linux, it reads
/// from /proc/self/status. On other platforms, returns 0 as a placeholder.
///
/// # Type Parameters
/// * `F` - The operation function
/// * `T` - The return type of the operation
///
/// # Arguments
/// * `operation` - The function to measure
///
/// # Returns
/// Estimated memory usage in bytes (0 on unsupported platforms)
pub fn measure_memory_usage<F, T>(operation: F) -> usize
where
    F: Fn() -> T,
{
    // Platform-specific memory measurement
    #[cfg(target_os = "linux")]
    {
        // Use /proc/self/status for memory measurement
        // This is a simplified version - full implementation would parse the file
        let _ = operation();

        // Read from /proc/self/status
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    // Extract RSS value
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = operation();
        0 // Placeholder for non-Linux platforms
    }
}

/// Benchmark result containing timing information
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Number of iterations run
    pub iterations: usize,
    /// Total time in seconds
    pub total_time: f64,
    /// Average time per iteration in seconds
    pub average_time: f64,
    /// Minimum time in seconds
    pub min_time: f64,
    /// Maximum time in seconds
    pub max_time: f64,
    /// Operations per second
    pub ops_per_sec: f64,
}

impl BenchmarkResult {
    /// Create a new BenchmarkResult
    pub fn new(
        name: String,
        iterations: usize,
        total_time: f64,
        min_time: f64,
        max_time: f64,
    ) -> Self {
        let average_time = total_time / iterations as f64;
        let ops_per_sec = if average_time > 0.0 {
            1.0 / average_time
        } else {
            0.0
        };

        BenchmarkResult {
            name,
            iterations,
            total_time,
            average_time,
            min_time,
            max_time,
            ops_per_sec,
        }
    }

    /// Format the benchmark result as a string
    pub fn format(&self) -> String {
        format!(
            "{}: {:.3} us/op ({} ops/sec), min: {:.3}, max: {:.3}",
            self.name,
            self.average_time * 1_000_000.0,
            self.ops_per_sec as u64,
            self.min_time * 1_000_000.0,
            self.max_time * 1_000_000.0
        )
    }
}

/// Run a benchmark with multiple iterations and return statistics
///
/// This provides min/max/avg timing for more comprehensive benchmarking.
///
/// # Arguments
/// * `name` - Name of the benchmark
/// * `operation` - The function to benchmark
/// * `iterations` - Number of iterations to run
///
/// # Returns
/// BenchmarkResult with timing statistics
pub fn run_benchmark<F, T>(name: String, operation: F, iterations: usize) -> BenchmarkResult
where
    F: Fn() -> T,
{
    let mut times: Vec<f64> = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = operation();
        times.push(start.elapsed().as_secs_f64());
    }

    let total_time: f64 = times.iter().sum();
    let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    BenchmarkResult::new(name, iterations, total_time, min_time, max_time)
}

/// Calculate throughput in configurations per second
///
/// # Arguments
/// * `configurations` - Number of configurations evaluated
/// * `duration_seconds` - Time taken in seconds
///
/// # Returns
/// Throughput in configs per second
pub fn calculate_throughput(configurations: usize, duration_seconds: f64) -> f64 {
    if duration_seconds > 0.0 {
        configurations as f64 / duration_seconds
    } else {
        0.0
    }
}

/// Benchmark performance metrics for validation runs
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    /// Total cases evaluated
    pub total_cases: usize,
    /// Total execution time in seconds
    pub execution_time: f64,
    /// Cases per second
    pub cases_per_second: f64,
    /// Average time per case in milliseconds
    pub avg_time_per_case_ms: f64,
    /// Whether performance target was met (<50ms per case)
    pub meets_perf_target: bool,
}

impl BenchmarkMetrics {
    /// Create metrics from validation run
    pub fn from_run(total_cases: usize, execution_time: f64) -> Self {
        let cases_per_second = calculate_throughput(total_cases, execution_time);
        let avg_time_per_case_ms = if total_cases > 0 {
            (execution_time / total_cases as f64) * 1000.0
        } else {
            0.0
        };
        let meets_perf_target = avg_time_per_case_ms < 50.0; // PERF-01 target

        BenchmarkMetrics {
            total_cases,
            execution_time,
            cases_per_second,
            avg_time_per_case_ms,
            meets_perf_target,
        }
    }

    /// Format metrics as string
    pub fn format(&self) -> String {
        format!(
            "Performance: {} cases in {:.3}s ({:.1} cases/sec, {:.2}ms/case) - {}",
            self.total_cases,
            self.execution_time,
            self.cases_per_second,
            self.avg_time_per_case_ms,
            if self.meets_perf_target {
                "MET TARGET"
            } else {
                "BELOW TARGET"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_timestep() {
        let result = measure_timestep(|| 1 + 2, 1000);
        assert!(result >= 0.0);
    }

    #[test]
    fn test_run_benchmark() {
        let result = run_benchmark("test".to_string(), || 1 + 2, 100);
        assert_eq!(result.name, "test");
        assert_eq!(result.iterations, 100);
        assert!(result.average_time >= 0.0);
    }

    #[test]
    fn test_calculate_throughput() {
        let throughput = calculate_throughput(1000, 1.0);
        assert_eq!(throughput, 1000.0);

        let throughput_zero = calculate_throughput(1000, 0.0);
        assert_eq!(throughput_zero, 0.0);
    }

    #[test]
    fn test_benchmark_metrics() {
        let metrics = BenchmarkMetrics::from_run(100, 4.9);
        assert_eq!(metrics.total_cases, 100);
        assert!((metrics.execution_time - 4.9).abs() < 0.001);
        // 4.9s / 100 = 49ms per case, target is <50ms per case
        assert!(metrics.meets_perf_target);
    }
}

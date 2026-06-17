//! Dynamic batch inference optimization for improved throughput.
//!
//! This module provides utilities for dynamic batching of inference requests,
//! optimizing batch size selection for maximum GPU/CPU utilization.

use std::sync::Mutex;
use std::time::Instant;

/// Configuration for dynamic batching.
#[derive(Clone, Debug)]
pub struct DynamicBatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size before forcing processing
    pub min_batch_size: usize,
    /// Maximum wait time in milliseconds before forcing batch processing
    pub max_wait_ms: u64,
    /// Target batch size (will adapt towards this)
    pub target_batch_size: usize,
    /// Enable dynamic size adaptation
    pub enable_adaptation: bool,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        DynamicBatchConfig {
            max_batch_size: 256,
            min_batch_size: 1,
            max_wait_ms: 10,
            target_batch_size: 64,
            enable_adaptation: true,
        }
    }
}

impl DynamicBatchConfig {
    /// Create config optimized for low latency
    pub fn low_latency() -> Self {
        DynamicBatchConfig {
            max_batch_size: 32,
            min_batch_size: 1,
            max_wait_ms: 1,
            target_batch_size: 16,
            enable_adaptation: false,
        }
    }

    /// Create config optimized for high throughput
    pub fn high_throughput() -> Self {
        DynamicBatchConfig {
            max_batch_size: 512,
            min_batch_size: 16,
            max_wait_ms: 50,
            target_batch_size: 128,
            enable_adaptation: true,
        }
    }
}

/// Statistics for batch processing.
#[derive(Clone, Debug, Default)]
pub struct BatchStats {
    /// Total number of requests processed
    pub total_requests: usize,
    /// Total number of batches processed
    pub total_batches: usize,
    /// Total inference time in milliseconds
    pub total_inference_ms: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Peak batch size
    pub peak_batch_size: usize,
    /// Number of adaptive size adjustments
    pub adaptation_count: usize,
}

impl BatchStats {
    fn new() -> Self {
        BatchStats::default()
    }

    fn record_batch(&mut self, batch_size: usize, inference_ms: u64) {
        self.total_requests += batch_size;
        self.total_batches += 1;
        self.total_inference_ms += inference_ms;

        let n = self.total_batches as f64;
        self.avg_batch_size = (self.avg_batch_size * (n - 1.0) + batch_size as f64) / n;

        if batch_size > self.peak_batch_size {
            self.peak_batch_size = batch_size;
        }
    }
}

/// Batch processor that handles inference with optimizations.
pub struct BatchProcessor {
    /// Configuration
    config: DynamicBatchConfig,
    /// Statistics for batch processing
    stats: Mutex<BatchStats>,
}

impl BatchProcessor {
    /// Create a new batch processor.
    pub fn new(config: DynamicBatchConfig) -> Self {
        BatchProcessor {
            config,
            stats: Mutex::new(BatchStats::new()),
        }
    }

    /// Process a single input (non-batched, for compatibility).
    pub fn process_single<F>(&self, input: &[f64], inference_fn: F) -> Vec<f64>
    where
        F: Fn(&[Vec<f64>]) -> Vec<Vec<f64>>,
    {
        let batch_result = self.process_batch(&[input.to_vec()], inference_fn);
        batch_result.into_iter().next().unwrap_or_else(|| vec![1.2])
    }

    /// Process a batch of inputs with optimized batching.
    pub fn process_batch<F>(&self, inputs: &[Vec<f64>], inference_fn: F) -> Vec<Vec<f64>>
    where
        F: Fn(&[Vec<f64>]) -> Vec<Vec<f64>>,
    {
        if inputs.is_empty() {
            return Vec::new();
        }

        let start = Instant::now();

        let optimized_batch = self.optimize_batch_size(inputs);
        let batch_size = optimized_batch.len();

        let results = inference_fn(&optimized_batch);

        let elapsed = start.elapsed().as_millis() as u64;

        let mut stats = self.stats.lock().unwrap();
        stats.record_batch(batch_size, elapsed);

        results
    }

    /// Optimize batch size based on input characteristics.
    fn optimize_batch_size(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let input_size = inputs.len();

        if !self.config.enable_adaptation || input_size >= self.config.target_batch_size {
            return inputs.to_vec();
        }

        if input_size >= self.config.min_batch_size {
            inputs.to_vec()
        } else {
            let mut batch = inputs.to_vec();
            while batch.len() < self.config.min_batch_size {
                batch.push(inputs[0].clone());
            }
            batch
        }
    }

    /// Get statistics.
    pub fn get_stats(&self) -> BatchStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = BatchStats::new();
    }
}

/// Benchmark result for batch inference.
#[derive(Clone, Debug)]
pub struct BatchBenchmarkResult {
    /// Batch size used
    pub batch_size: usize,
    /// Total time in milliseconds
    pub total_time_ms: f64,
    /// Average time per inference in microseconds
    pub avg_time_per_inference_us: f64,
    /// Throughput (inferences per second)
    pub throughput: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
}

impl BatchBenchmarkResult {
    /// Print benchmark results.
    pub fn print(&self) {
        println!("Batch Size: {}", self.batch_size);
        println!("Total Time: {:.2} ms", self.total_time_ms);
        println!(
            "Avg Time/Inference: {:.2} μs",
            self.avg_time_per_inference_us
        );
        println!("Throughput: {:.0} inferences/sec", self.throughput);
    }
}

/// Run benchmark for different batch sizes.
pub fn benchmark_batch_inference<F>(
    inference_fn: F,
    max_batch_size: usize,
) -> Vec<BatchBenchmarkResult>
where
    F: Fn(&[Vec<f64>]) -> Vec<Vec<f64>> + Clone,
{
    let mut results = Vec::new();

    let input_dim = 10;

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        .iter()
        .copied()
        .filter(|&b| b <= max_batch_size)
    {
        let inputs: Vec<Vec<f64>> = (0..batch_size)
            .map(|i| (0..input_dim).map(|j| (i + j) as f64 * 0.1).collect())
            .collect();

        let _ = inference_fn(&inputs);

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = inference_fn(&inputs);
        }
        let elapsed = start.elapsed();

        let total_time_ms = elapsed.as_secs_f64() * 1000.0;
        let avg_time_us = (elapsed.as_nanos() as f64 / iterations as f64) / 1000.0;
        let throughput = (batch_size * iterations) as f64 / elapsed.as_secs_f64();

        results.push(BatchBenchmarkResult {
            batch_size,
            total_time_ms,
            avg_time_per_inference_us: avg_time_us,
            throughput,
            peak_memory_mb: 0.0,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_batch_config_defaults() {
        let config = DynamicBatchConfig::default();
        assert_eq!(config.max_batch_size, 256);
        assert_eq!(config.min_batch_size, 1);
        assert_eq!(config.max_wait_ms, 10);
        assert_eq!(config.target_batch_size, 64);
        assert!(config.enable_adaptation);
    }

    #[test]
    fn test_dynamic_batch_config_presets() {
        let low_latency = DynamicBatchConfig::low_latency();
        assert_eq!(low_latency.max_batch_size, 32);
        assert_eq!(low_latency.max_wait_ms, 1);
        assert!(!low_latency.enable_adaptation);

        let high_throughput = DynamicBatchConfig::high_throughput();
        assert_eq!(high_throughput.max_batch_size, 512);
        assert_eq!(high_throughput.min_batch_size, 16);
        assert_eq!(high_throughput.target_batch_size, 128);
    }

    #[test]
    fn test_batch_stats_record() {
        let mut stats = BatchStats::new();
        stats.record_batch(10, 5);
        stats.record_batch(20, 10);

        assert_eq!(stats.total_requests, 30);
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_inference_ms, 15);
        assert_eq!(stats.peak_batch_size, 20);
    }

    #[test]
    fn test_batch_processor() {
        let config = DynamicBatchConfig::default();
        let processor = BatchProcessor::new(config);

        let mock_inference = |inputs: &[Vec<f64>]| -> Vec<Vec<f64>> {
            inputs
                .iter()
                .map(|v| v.iter().map(|x| x * 2.0).collect())
                .collect()
        };

        let inputs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let results = processor.process_batch(&inputs, mock_inference);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_batch_benchmark() {
        let mock_inference = |inputs: &[Vec<f64>]| -> Vec<Vec<f64>> {
            inputs
                .iter()
                .map(|v| v.iter().map(|x| x * 2.0).collect())
                .collect()
        };

        let results = benchmark_batch_inference(mock_inference, 16);

        assert!(!results.is_empty());
        for result in &results {
            assert!(result.throughput > 0.0);
            assert!(result.batch_size > 0);
        }
    }

    #[test]
    fn test_dynamic_batch_config_clone_debug() {
        let config = DynamicBatchConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_batch_size, config.max_batch_size);

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("max_batch_size"));
    }

    #[test]
    fn test_batch_stats_default() {
        let stats = BatchStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.total_inference_ms, 0);
        assert_eq!(stats.peak_batch_size, 0);
        assert_eq!(stats.adaptation_count, 0);
    }

    #[test]
    fn test_batch_stats_clone_debug() {
        let stats = BatchStats {
            total_requests: 100,
            total_batches: 10,
            total_inference_ms: 50,
            avg_batch_size: 10.0,
            peak_batch_size: 20,
            adaptation_count: 3,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.peak_batch_size, stats.peak_batch_size);

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("total_requests"));
    }

    #[test]
    fn test_batch_processor_process_single() {
        let config = DynamicBatchConfig::default();
        let processor = BatchProcessor::new(config);

        let mock_inference = |inputs: &[Vec<f64>]| -> Vec<Vec<f64>> {
            inputs.iter().map(|v| vec![v.iter().sum()]).collect()
        };

        let result = processor.process_single(&[1.0, 2.0, 3.0], mock_inference);
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_batch_processor_empty_batch() {
        let config = DynamicBatchConfig::default();
        let processor = BatchProcessor::new(config);

        let mock_inference = |_: &[Vec<f64>]| -> Vec<Vec<f64>> { vec![] };

        let result = processor.process_batch(&[], mock_inference);
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_processor_optimize_batch_size_adaptation_disabled() {
        let config = DynamicBatchConfig {
            enable_adaptation: false,
            ..DynamicBatchConfig::default()
        };
        let processor = BatchProcessor::new(config);

        let inputs: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0]];
        let result = processor.process_batch(&inputs, |batch| batch.to_vec());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_batch_processor_optimize_batch_size_padding() {
        let config = DynamicBatchConfig {
            min_batch_size: 4,
            enable_adaptation: true,
            ..DynamicBatchConfig::default()
        };
        let processor = BatchProcessor::new(config);

        let inputs: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
        let result = processor.process_batch(&inputs, |batch| batch.to_vec());

        assert!(result.len() >= 4);
        assert_eq!(result[0], vec![1.0, 2.0]);
    }

    #[test]
    fn test_batch_processor_stats_and_reset() {
        let config = DynamicBatchConfig::default();
        let processor = BatchProcessor::new(config);

        let mock_inference = |inputs: &[Vec<f64>]| -> Vec<Vec<f64>> { inputs.to_vec() };

        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        processor.process_batch(&inputs, mock_inference);

        let stats = processor.get_stats();
        assert_eq!(stats.total_batches, 1);
        assert_eq!(stats.total_requests, 3);

        processor.reset_stats();
        let stats_after = processor.get_stats();
        assert_eq!(stats_after.total_batches, 0);
        assert_eq!(stats_after.total_requests, 0);
    }

    #[test]
    fn test_batch_benchmark_result_print() {
        let result = BatchBenchmarkResult {
            batch_size: 32,
            total_time_ms: 100.0,
            avg_time_per_inference_us: 3125.0,
            throughput: 320.0,
            peak_memory_mb: 256.0,
        };

        result.print();
    }

    #[test]
    fn test_batch_benchmark_result_clone_debug() {
        let result = BatchBenchmarkResult {
            batch_size: 16,
            total_time_ms: 50.0,
            avg_time_per_inference_us: 3000.0,
            throughput: 320.0,
            peak_memory_mb: 128.0,
        };

        let cloned = result.clone();
        assert_eq!(cloned.batch_size, result.batch_size);

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("batch_size"));
    }

    #[test]
    fn test_batch_benchmark_large_sizes() {
        let mock_inference = |inputs: &[Vec<f64>]| -> Vec<Vec<f64>> { inputs.to_vec() };

        let results = benchmark_batch_inference(mock_inference, 1024);
        assert!(!results.is_empty());

        for r in &results {
            assert!(r.batch_size <= 1024);
            assert!(r.total_time_ms >= 0.0);
            assert!(r.throughput > 0.0);
        }
    }
}

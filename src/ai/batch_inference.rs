//! Dynamic batch inference optimization for improved throughput.
//!
//! This module provides utilities for dynamic batching of inference requests,
//! optimizing batch size selection, and managing batch queues for maximum
//! GPU/CPU utilization.

use std::sync::{Arc, Mutex};
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
            max_wait_ms: 10, // 10ms max wait
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

/// A single inference request.
#[derive(Clone, Debug)]
pub struct BatchRequest {
    /// Input data for the request
    pub input: Vec<f64>,
    /// Channel to send the response back
    pub response_tx: std::sync::mpsc::Sender<Vec<f64>>,
}

/// Dynamic batch manager that collects requests and processes them in batches.
pub struct DynamicBatchManager {
    config: DynamicBatchConfig,
    pending_requests: Mutex<Vec<BatchRequest>>,
    stats: Mutex<BatchStats>,
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

        // Update average batch size
        let n = self.total_batches as f64;
        self.avg_batch_size = (self.avg_batch_size * (n - 1.0) + batch_size as f64) / n;

        // Update peak
        if batch_size > self.peak_batch_size {
            self.peak_batch_size = batch_size;
        }
    }
}

impl DynamicBatchManager {
    /// Create a new dynamic batch manager.
    pub fn new(config: DynamicBatchConfig) -> Self {
        DynamicBatchManager {
            config,
            pending_requests: Mutex::new(Vec::new()),
            stats: Mutex::new(BatchStats::new()),
        }
    }

    /// Add a request to the batch queue.
    ///
    /// Returns true if the batch should be processed immediately.
    pub fn add_request(&self, request: BatchRequest) -> bool {
        let mut pending = self.pending_requests.lock().unwrap();
        pending.push(request);

        // Check if we should process immediately
        pending.len() >= self.config.max_batch_size
    }

    /// Get pending requests for processing.
    pub fn get_pending(&self, force: bool) -> Vec<BatchRequest> {
        let mut pending = self.pending_requests.lock().unwrap();

        if force || pending.len() >= self.config.min_batch_size {
            let batch = pending.drain(..).collect();
            batch
        } else {
            Vec::new()
        }
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending_requests.lock().unwrap().len()
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> BatchStats {
        self.stats.lock().unwrap().clone()
    }

    /// Record batch processing for statistics.
    pub fn record_batch(&self, batch_size: usize, inference_ms: u64) {
        let mut stats = self.stats.lock().unwrap();
        stats.record_batch(batch_size, inference_ms);
    }
}

/// Batch processor that handles inference with optimizations.
pub struct BatchProcessor {
    /// Configuration
    config: DynamicBatchConfig,
    /// Pending batch queue
    queue: Arc<DynamicBatchManager>,
}

impl BatchProcessor {
    /// Create a new batch processor.
    pub fn new(config: DynamicBatchConfig) -> Self {
        BatchProcessor {
            config,
            queue: Arc::new(DynamicBatchManager::new(DynamicBatchConfig::default())),
        }
    }

    /// Process a single input (non-batched, for compatibility).
    pub fn process_single<F>(&self, input: &[f64], inference_fn: F) -> Vec<f64>
    where
        F: Fn(&[Vec<f64>]) -> Vec<Vec<f64>>,
    {
        // Process single input through batch path for consistency
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

        // Apply dynamic batching optimization
        let optimized_batch = self.optimize_batch_size(inputs);
        let batch_size = optimized_batch.len();

        // Run inference
        let results = inference_fn(&optimized_batch);

        let elapsed = start.elapsed().as_millis() as u64;

        // Record statistics
        self.queue.record_batch(batch_size, elapsed);

        results
    }

    /// Optimize batch size based on input characteristics.
    fn optimize_batch_size(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let input_size = inputs.len();

        if !self.config.enable_adaptation || input_size >= self.config.target_batch_size {
            // Use all inputs if we meet target or adaptation disabled
            return inputs.to_vec();
        }

        // For small inputs, consider if we should pad or wait
        // This is a simplified implementation - in production you'd track queue depth
        if input_size >= self.config.min_batch_size {
            inputs.to_vec()
        } else {
            // Pad with duplicates to meet min batch size (for GPU efficiency)
            let mut batch = inputs.to_vec();
            while batch.len() < self.config.min_batch_size {
                batch.push(inputs[0].clone());
            }
            batch
        }
    }

    /// Get queue statistics.
    pub fn get_stats(&self) -> BatchStats {
        self.queue.get_stats()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        let mut stats = self.queue.stats.lock().unwrap();
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

    // Generate test inputs (simulating different batch sizes)
    let input_dim = 10; // Typical number of zones

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        .iter()
        .copied()
        .filter(|&b| b <= max_batch_size)
    {
        // Generate test inputs
        let inputs: Vec<Vec<f64>> = (0..batch_size)
            .map(|i| (0..input_dim).map(|j| (i + j) as f64 * 0.1).collect())
            .collect();

        // Warmup
        let _ = inference_fn(&inputs);

        // Benchmark
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
            peak_memory_mb: 0.0, // Would need memory profiler
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
    }

    #[test]
    fn test_dynamic_batch_config_presets() {
        let low_latency = DynamicBatchConfig::low_latency();
        assert_eq!(low_latency.max_batch_size, 32);
        assert_eq!(low_latency.max_wait_ms, 1);

        let high_throughput = DynamicBatchConfig::high_throughput();
        assert_eq!(high_throughput.max_batch_size, 512);
        assert_eq!(high_throughput.target_batch_size, 128);
    }

    #[test]
    fn test_batch_stats() {
        let mut stats = BatchStats::new();
        stats.record_batch(10, 5);
        stats.record_batch(20, 10);

        assert_eq!(stats.total_requests, 30);
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.peak_batch_size, 20);
    }

    #[test]
    fn test_batch_processor() {
        let config = DynamicBatchConfig::default();
        let processor = BatchProcessor::new(config);

        // Simple mock inference function
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
        // Larger batches should generally have higher throughput
        let first = &results[0];
        let last = &results[results.len() - 1];
        assert!(last.throughput >= first.throughput * 0.5); // At least 50% of linear scaling
    }
}

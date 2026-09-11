//! Inference metrics: latency/accuracy tracking for surrogate inference.
//!
//! [`InferenceMetrics`] keeps the rolling average inference latency,
//! throughput, and peak-memory counters surfaced via
//! [`SurrogateManager::inference_metrics`](crate::ai::surrogate::SurrogateManager::inference_metrics).

#[derive(Clone, Debug, Default)]
pub struct InferenceMetrics {
    pub avg_inference_time_ms: f64,
    pub num_inferences: usize,
    pub peak_memory_mb: f64,
    pub throughput: f64,
}

impl InferenceMetrics {
    pub fn record_inference(&mut self, time_ms: f64) {
        let n = self.num_inferences as f64;
        self.avg_inference_time_ms = (self.avg_inference_time_ms * n + time_ms) / (n + 1.0);
        self.num_inferences += 1;
        if self.avg_inference_time_ms > 0.0 {
            self.throughput = 1000.0 / self.avg_inference_time_ms;
        }
    }
    pub fn reset(&mut self) {
        self.avg_inference_time_ms = 0.0;
        self.num_inferences = 0;
        self.peak_memory_mb = 0.0;
        self.throughput = 0.0;
    }
}

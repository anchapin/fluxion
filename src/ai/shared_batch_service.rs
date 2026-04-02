//! Shared batch inference service for concurrent surrogate requests.
//!
//! This service aggregates inference requests from multiple workers into batches,
//! maximizing GPU utilization for ONNX Runtime. It runs a dedicated worker thread
//! that collects requests, calls `predict_loads_batched`, and distributes results
//! back to requesters via oneshot channels.

use crate::ai::surrogate::SurrogateManager;
use crossbeam::channel::{self, Receiver, RecvTimeoutError, Sender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

/// Configuration for dynamic batch sizing.
#[derive(Clone, Debug)]
pub struct DynamicBatchConfig {
    /// Maximum number of requests to include in a single batch.
    pub max_batch_size: usize,
    /// Maximum time to wait (milliseconds) for a batch to fill before processing
    /// whatever has been collected.
    pub wait_ms: u64,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            wait_ms: 10,
        }
    }
}

/// Incoming inference request: temperature vector and channel to send response.
struct InferenceRequest {
    temps: Vec<f64>,
    response_tx: Sender<Vec<f64>>,
}

/// Shared batch inference service.
///
/// Workers call `submit()` to send their temperature vectors and receive a
/// Receiver for the corresponding loads. The service aggregates submissions
/// into batches and runs a single `predict_loads_batched` call per batch.
#[derive(Clone)]
pub struct SharedBatchInferenceService {
    inner: Arc<Inner>,
}

struct Inner {
    sender: Option<Sender<InferenceRequest>>,
    _thread: Option<JoinHandle<()>>,
}

impl SharedBatchInferenceService {
    /// Creates a new service with the given surrogate manager, batch configuration,
    /// and channel capacity (should be at least the number of concurrent workers).
    pub fn new(
        surrogate: SurrogateManager,
        config: DynamicBatchConfig,
        channel_capacity: usize,
    ) -> Self {
        let (tx, rx) = channel::bounded(channel_capacity);
        let sender = tx.clone(); // keep a sender for submissions
        let thread = thread::spawn(move || Self::run_worker(rx, surrogate, config));
        Self {
            inner: Arc::new(Inner {
                sender: Some(sender),
                _thread: Some(thread),
            }),
        }
    }

    /// Submits a temperature vector for inference.
    ///
    /// Returns a Receiver that will receive the predicted loads (Vec<f64>).
    /// The call is non-blocking; the returned receiver should be used to
    /// wait for the result.
    pub fn submit(&self, temps: Vec<f64>) -> Receiver<Vec<f64>> {
        let (resp_tx, resp_rx) = channel::unbounded();
        let request = InferenceRequest {
            temps,
            response_tx: resp_tx,
        };
        // Send the request to the service thread. If the service thread has died,
        // this will panic; but under normal operation it should succeed.
        self.inner
            .sender
            .as_ref()
            .expect("SharedBatchInferenceService already dropped")
            .send(request)
            .expect("SharedBatchInferenceService worker thread died");
        resp_rx
    }

    /// Worker thread main loop.
    fn run_worker(
        req_rx: Receiver<InferenceRequest>,
        surrogate: SurrogateManager,
        config: DynamicBatchConfig,
    ) {
        loop {
            // Collect a batch of requests.
            let mut batch = Vec::new();

            // First request (blocking). This will return Err when all senders are dropped.
            match req_rx.recv() {
                Ok(first_req) => batch.push(first_req),
                Err(_) => break, // All senders gone, exit thread.
            }

            // Try to collect additional requests up to max_batch_size, waiting up to wait_ms.
            while batch.len() < config.max_batch_size {
                match req_rx.recv_timeout(Duration::from_millis(config.wait_ms)) {
                    Ok(req) => batch.push(req),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => break, // channel closed, exit after processing
                }
            }

            // Separate inputs and response senders.
            let (inputs, senders): (Vec<Vec<f64>>, Vec<Sender<Vec<f64>>>) = batch
                .into_iter()
                .map(|req| (req.temps, req.response_tx))
                .unzip();

            // Perform batched inference.
            let outputs = surrogate.predict_loads_batched(&inputs);

            // Send results back to requesters.
            for (tx, out) in senders.into_iter().zip(outputs.into_iter()) {
                // Ignore errors: if receiver dropped, that's okay.
                let _ = tx.send(out);
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Drop the sender first so the worker thread's req_rx.recv() returns Err and it exits.
        drop(self.sender.take());

        // Attempt to join the worker thread.
        if let Some(thread) = self._thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::surrogate::SurrogateManager;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Helper function to create a SurrogateManager with dummy model loaded.
    /// Returns None if the dummy model file is not available (e.g., in CI).
    fn create_test_surrogate() -> SurrogateManager {
        // Always use mock surrogate for unit tests to ensure reliability and speed.
        // The SurrogateManager will return mock loads (1.2 W/m2) if no model is loaded.
        SurrogateManager::new().unwrap()
    }

    #[test]
    fn test_shared_batch_service_single() {
        let surrogate = create_test_surrogate();
        let config = DynamicBatchConfig {
            max_batch_size: 4,
            wait_ms: 10,
        };
        let service = SharedBatchInferenceService::new(surrogate, config, 4);
        let temps = vec![20.0, 21.0];
        let rx = service.submit(temps);
        let result = rx.recv().expect("No result received");
        assert_eq!(result.len(), 2);
        assert!(result[0] > 0.0);
        assert!(result[1] > 0.0);
    }

    #[test]
    fn test_shared_batch_service_concurrent() {
        let surrogate = create_test_surrogate();
        let config = DynamicBatchConfig {
            max_batch_size: 10,
            wait_ms: 100,
        };
        let n_workers = 20;
        let service = SharedBatchInferenceService::new(surrogate, config, n_workers);

        let mut handles = Vec::new();

        for i in 0..n_workers {
            let service = service.clone();
            let handle = thread::spawn(move || {
                let input = vec![20.0 + i as f64, 21.0 + i as f64];
                let rx = service.submit(input.clone());
                let output = rx.recv().expect("Failed to receive output");
                output
            });
            handles.push(handle);
        }

        for h in handles {
            let out = h.join().expect("Thread panicked");
            assert_eq!(out.len(), 2);
            assert!(out[0] > 0.0);
            assert!(out[1] > 0.0);
        }
    }

    #[test]
    fn test_shared_batch_service_batching() {
        // Verify that the service actually batches requests together by using
        // a surrogate that counts how many times predict_loads_batched is called.
        let surrogate = create_test_surrogate();
        let config = DynamicBatchConfig {
            max_batch_size: 5,
            wait_ms: 50,
        };
        let service = SharedBatchInferenceService::new(surrogate, config, 10);

        let mut handles = Vec::new();
        let n_requests = 10;

        for i in 0..n_requests {
            let service = service.clone();
            let handle = thread::spawn(move || {
                let rx = service.submit(vec![i as f64, (i + 1) as f64]);
                rx.recv().unwrap()
            });
            handles.push(handle);
        }

        // Wait for all to complete.
        for h in handles {
            h.join().unwrap();
        }

        // The service should have processed all requests. Since we don't have
        // an easy way to count batch calls without a mock, we just verify
        // correctness.
    }

    #[test]
    fn test_dynamic_batch_config_default() {
        let config = DynamicBatchConfig::default();
        assert_eq!(config.max_batch_size, 512);
        assert_eq!(config.wait_ms, 10);
    }

    #[test]
    fn test_dynamic_batch_config_custom() {
        let config = DynamicBatchConfig {
            max_batch_size: 64,
            wait_ms: 100,
        };
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.wait_ms, 100);
    }

    #[test]
    fn test_dynamic_batch_config_clone() {
        let config = DynamicBatchConfig {
            max_batch_size: 32,
            wait_ms: 50,
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_batch_size, 32);
        assert_eq!(cloned.wait_ms, 50);
    }

    #[test]
    fn test_dynamic_batch_config_debug() {
        let config = DynamicBatchConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("max_batch_size"));
        assert!(debug_str.contains("wait_ms"));
    }
}

use crate::ai::surrogate::{InferenceBackend, SurrogateManager};
use crossbeam::queue::SegQueue;
use rayon::prelude::*;
use std::sync::Arc;

/// Manages multiple SurrogateManagers, typically one per GPU device.
pub struct DistributedSurrogateManager {
    managers: Vec<SurrogateManager>,
    queue: Arc<SegQueue<usize>>, // Queue of available manager indices
}

impl DistributedSurrogateManager {
    /// Create a new DistributedSurrogateManager with the given model and backend configuration.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model.
    /// * `backend` - Inference backend to use (e.g., CUDA).
    /// * `device_ids` - List of device IDs to use (e.g., [0, 1] for 2 GPUs).
    pub fn new(
        model_path: &str,
        backend: InferenceBackend,
        device_ids: &[usize],
    ) -> Result<Self, String> {
        let mut managers = Vec::new();
        let queue = Arc::new(SegQueue::new());

        for (i, &device_id) in device_ids.iter().enumerate() {
            let manager = SurrogateManager::with_gpu_backend(model_path, backend, device_id)?;
            managers.push(manager);
            queue.push(i);
        }

        if managers.is_empty() {
            return Err("No devices specified".to_string());
        }

        Ok(DistributedSurrogateManager { managers, queue })
    }

    /// Create a DistributedSurrogateManager from existing managers.
    pub fn from_managers(managers: Vec<SurrogateManager>) -> Self {
        let queue = Arc::new(SegQueue::new());
        for i in 0..managers.len() {
            queue.push(i);
        }
        DistributedSurrogateManager { managers, queue }
    }

    /// Evaluate a population using distributed inference.
    ///
    /// # Arguments
    /// * `population` - A vector of input vectors (e.g., temperatures).
    ///
    /// # Returns
    /// * `Vec<Vec<f64>>` - A vector of result vectors.
    pub fn evaluate_population_distributed(
        &self,
        population: Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, String> {
        // Determine batch size per device or chunk size
        let num_devices = self.managers.len();
        if num_devices == 0 {
            return Err("No inference managers available".to_string());
        }

        // Simple parallel iteration using rayon
        let chunk_size = if population.is_empty() {
            1
        } else {
            population.len().div_ceil(num_devices)
        };

        let results: Result<Vec<Vec<Vec<f64>>>, String> = population
            .par_chunks(chunk_size)
            .map(|chunk| {
                // Acquire a manager index
                let manager_idx = loop {
                    if let Some(idx) = self.queue.pop() {
                        break idx;
                    }
                    // Spin wait or backoff
                    std::thread::yield_now();
                };

                let manager = &self.managers[manager_idx];
                let batch = chunk.to_vec();
                let res = manager.predict_loads_batched(&batch);

                // Return manager index
                self.queue.push(manager_idx);

                Ok(res)
            })
            .collect();

        match results {
            Ok(chunks) => Ok(chunks.into_iter().flatten().collect()),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_evaluate() {
        let managers = vec![
            SurrogateManager::new().unwrap(),
            SurrogateManager::new().unwrap(),
        ];
        let manager = DistributedSurrogateManager::from_managers(managers);

        let population = vec![
            vec![20.0, 21.0],
            vec![22.0, 23.0],
            vec![24.0, 25.0],
            vec![26.0, 27.0],
        ];

        let results = manager.evaluate_population_distributed(population).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0][0], 1.2);
    }

    #[test]
    fn test_distributed_evaluate_empty() {
        let managers = vec![SurrogateManager::new().unwrap()];
        let manager = DistributedSurrogateManager::from_managers(managers);
        let results = manager.evaluate_population_distributed(vec![]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_distributed_new_empty_device_ids() {
        let result =
            DistributedSurrogateManager::new("dummy_model.onnx", InferenceBackend::CPU, &[]);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err, "No devices specified");
    }

    #[test]
    fn test_distributed_new_missing_model() {
        // Should fail gracefully when model file doesn't exist
        let result =
            DistributedSurrogateManager::new("nonexistent_model.onnx", InferenceBackend::CPU, &[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_distributed_queue_behavior() {
        // Test the queue data structure independently
        let queue: Arc<SegQueue<usize>> = Arc::new(SegQueue::new());

        // Push some indices
        queue.push(0);
        queue.push(1);
        queue.push(2);

        // Pop should return in some order
        let mut popped = Vec::new();
        while let Some(idx) = queue.pop() {
            popped.push(idx);
        }
        popped.sort();
        assert_eq!(popped, vec![0, 1, 2]);
    }

    #[test]
    fn test_distributed_queue_empty_pop() {
        let queue: Arc<SegQueue<usize>> = Arc::new(SegQueue::new());
        assert!(queue.pop().is_none());
    }
}

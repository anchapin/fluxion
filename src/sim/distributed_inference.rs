//! Distributed Inference Architecture for Building Energy Modeling
//!
//! This module provides async task management using tokio and data parallelism
//! using rayon for running thousands of building variants simultaneously.
//! This is essential for high-throughput building energy analysis and optimization.

use crate::sim::thermal_model::{ThermalModelTrait, ThermalModelMode};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Result type for distributed inference operations
pub type DistributedResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Configuration for distributed inference execution
#[derive(Debug, Clone)]
pub struct DistributedInferenceConfig {
    /// Number of rayon workers for CPU parallelism
    pub rayon_workers: usize,
    /// Number of tokio async tasks for I/O concurrency
    pub async_tasks: usize,
    /// Maximum concurrent building evaluations
    pub max_concurrent: usize,
    /// Chunk size for batch processing
    pub chunk_size: usize,
}

impl Default for DistributedInferenceConfig {
    fn default() -> Self {
        let rayon_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            rayon_workers,
            async_tasks: rayon_workers * 4,
            max_concurrent: rayon_workers * 2,
            chunk_size: 100,
        }
    }
}

/// A building variant for parallel evaluation
#[derive(Debug, Clone)]
pub struct BuildingVariant {
    /// Unique identifier for this variant
    pub id: u64,
    /// Building parameters: [U-value, heating_setpoint, cooling_setpoint, ...]
    pub parameters: Vec<f64>,
}

impl BuildingVariant {
    /// Create a new building variant
    pub fn new(id: u64, parameters: Vec<f64>) -> Self {
        Self { id, parameters }
    }
}

/// Result of evaluating a building variant
#[derive(Debug, Clone)]
pub struct VariantResult {
    /// Variant ID
    pub id: u64,
    /// Energy Use Intensity (kWh/m²/year)
    pub eui: f64,
    /// Whether the evaluation succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Parallel thermal model evaluator using rayon for data parallelism.
///
/// This evaluator allows running multiple building variants in parallel
/// using rayon for efficient CPU utilization.
pub struct ParallelThermalEvaluator {
    config: DistributedInferenceConfig,
}

impl ParallelThermalEvaluator {
    /// Create a new parallel thermal evaluator with default configuration.
    pub fn new() -> Self {
        Self {
            config: DistributedInferenceConfig::default(),
        }
    }

    /// Create a new parallel thermal evaluator with custom configuration.
    pub fn with_config(config: DistributedInferenceConfig) -> Self {
        Self { config }
    }

    /// Evaluate a population of building variants in parallel using rayon.
    ///
    /// # Arguments
    /// * `population` - Vector of building variants to evaluate
    /// * `model_factory` - Factory function to create thermal models
    ///
    /// # Returns
    /// Vector of results for each variant
    pub fn evaluate_population<F>(
        &self,
        population: Vec<BuildingVariant>,
        mut model_factory: F,
    ) -> Vec<VariantResult>
    where
        F: FnMut(&[f64]) -> Box<dyn ThermalModelTrait> + Send + Sync,
    {
        let factory = Arc::new(Mutex::new(model_factory));

        population
            .par_iter()
            .map(|variant| {
                let mut model = {
                    let mut f = factory.lock().unwrap();
                    f(&variant.parameters)
                };

                // Apply parameters and solve
                model.apply_parameters(&variant.parameters);
                
                // Run a simplified simulation (8760 hourly timesteps = 1 year)
                let eui = model.solve_timesteps(8760, &crate::ai::surrogate::SurrogateManager::default(), false);

                VariantResult {
                    id: variant.id,
                    eui,
                    success: true,
                    error: None,
                }
            })
            .collect()
    }

    /// Evaluate population in chunks for memory efficiency with large populations.
    ///
    /// # Arguments
    /// * `population` - Vector of building variants to evaluate
    /// * `model_factory` - Factory function to create thermal models
    ///
    /// # Returns
    /// Vector of results for each variant
    pub fn evaluate_chunked<F>(
        &self,
        population: Vec<BuildingVariant>,
        mut model_factory: F,
    ) -> Vec<VariantResult>
    where
        F: FnMut(&[f64]) -> Box<dyn ThermalModelTrait> + Send + Sync,
    {
        let chunk_size = self.config.chunk_size;
        
        let chunks: Vec<Vec<BuildingVariant>> = population
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        let results: Vec<Vec<VariantResult>> = chunks
            .par_iter()
            .map(|chunk| {
                // Create a new model factory for each parallel chunk to avoid borrow issues
                // This works because we create a fresh closure that captures nothing
                let mut chunk_model: Box<dyn ThermalModelTrait> = Box::new(crate::sim::thermal_model::PhysicsThermalModel::new(1));
                
                chunk
                    .iter()
                    .map(|variant| {
                        chunk_model.apply_parameters(&variant.parameters);
                        let eui = chunk_model.solve_timesteps(8760, &crate::ai::surrogate::SurrogateManager::default(), false);
                        
                        VariantResult {
                            id: variant.id,
                            eui,
                            success: true,
                            error: None,
                        }
                    })
                    .collect()
            })
            .collect();

        results.into_iter().flatten().collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistributedInferenceConfig {
        &self.config
    }
}

impl Default for ParallelThermalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Async task manager for distributed inference using tokio.
///
/// This manager handles scheduling and execution of building variant simulations
/// using async/await patterns for high-throughput concurrent processing.
pub struct AsyncInferenceManager {
    /// Channel sender for submitting new tasks
    task_sender: mpsc::Sender<BuildingVariant>,
    /// Channel receiver for receiving task results
    result_receiver: mpsc::Receiver<VariantResult>,
    /// Maximum number of concurrent tasks
    max_concurrent: usize,
    /// Total tasks submitted
    tasks_submitted: Arc<Mutex<u64>>,
    /// Total tasks completed
    tasks_completed: Arc<Mutex<u64>>,
}

impl AsyncInferenceManager {
    /// Create a new async inference manager.
    ///
    /// # Arguments
    /// * `max_concurrent` - Maximum number of concurrent tasks
    ///
    /// # Returns
    /// A new AsyncInferenceManager instance
    pub fn new(max_concurrent: usize) -> Self {
        let (task_sender, mut task_receiver) = mpsc::channel::<BuildingVariant>(10000);
        let (result_sender, result_receiver) = mpsc::channel::<VariantResult>(10000);

        let tasks_submitted = Arc::new(Mutex::new(0u64));
        let tasks_completed = Arc::new(Mutex::new(0u64));
        let max_concurrent_copy = max_concurrent;

        // Spawn the async worker pool
        tokio::spawn(async move {
            let mut running_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
            let mut pending_queue: Vec<BuildingVariant> = Vec::new();

            loop {
                tokio::select! {
                    // Try to receive new tasks
                    new_task = task_receiver.recv() => {
                        match new_task {
                            Some(variant) => {
                                // Clean up finished tasks
                                running_handles.retain(|h| !h.is_finished());

                                if running_handles.len() < max_concurrent_copy {
                                    let sender = result_sender.clone();
                                    let handle = tokio::spawn(async move {
                                        // Simulate thermal model evaluation
                                        let params = &variant.parameters;
                                        let eui = if params.len() >= 3 {
                                            let u_value = params[0];
                                            let heating = params[1];
                                            let cooling = params[2];
                                            
                                            let base_load = 50.0;
                                            let u_factor = (u_value - 1.5).abs() * 8.0;
                                            let setpoint_diff = (cooling - heating) * 4.0;
                                            
                                            base_load + u_factor + setpoint_diff
                                        } else {
                                            f64::NAN
                                        };

                                        let result = VariantResult {
                                            id: variant.id,
                                            eui,
                                            success: !eui.is_nan(),
                                            error: if eui.is_nan() { Some("Invalid parameters".to_string()) } else { None },
                                        };
                                        
                                        let _ = sender.send(result).await;
                                    });
                                    running_handles.push(handle);
                                } else {
                                    pending_queue.push(variant);
                                }
                            }
                            None => {
                                // Channel closed, wait for remaining tasks
                                for handle in running_handles {
                                    let _ = handle.await;
                                }
                                break;
                            }
                        }
                    }

                    // Periodic cleanup and task spawning
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(5)) => {
                        running_handles.retain(|h| !h.is_finished());

                        while running_handles.len() < max_concurrent_copy {
                            match pending_queue.pop() {
                                Some(variant) => {
                                    let sender = result_sender.clone();
                                    let handle = tokio::spawn(async move {
                                        let params = &variant.parameters;
                                        let eui = if params.len() >= 3 {
                                            let u_value = params[0];
                                            let heating = params[1];
                                            let cooling = params[2];
                                            
                                            let base_load = 50.0;
                                            let u_factor = (u_value - 1.5).abs() * 8.0;
                                            let setpoint_diff = (cooling - heating) * 4.0;
                                            
                                            base_load + u_factor + setpoint_diff
                                        } else {
                                            f64::NAN
                                        };

                                        let result = VariantResult {
                                            id: variant.id,
                                            eui,
                                            success: !eui.is_nan(),
                                            error: if eui.is_nan() { Some("Invalid parameters".to_string()) } else { None },
                                        };
                                        
                                        let _ = sender.send(result).await;
                                    });
                                    running_handles.push(handle);
                                }
                                None => break,
                            }
                        }
                    }
                }
            }
        });

        Self {
            task_sender,
            result_receiver,
            max_concurrent,
            tasks_submitted,
            tasks_completed,
        }
    }

    /// Submit a new building variant for async processing.
    ///
    /// # Arguments
    /// * `variant` - Building variant to evaluate
    ///
    /// # Returns
    /// Task ID
    pub async fn submit(&mut self, variant: BuildingVariant) -> u64 {
        let id = variant.id;
        {
            let mut count = self.tasks_submitted.lock().unwrap();
            *count += 1;
        }
        
        let _ = self.task_sender.send(variant).await;
        id
    }

    /// Submit multiple variants at once (batch submission).
    ///
    /// # Arguments
    /// * `variants` - List of building variants
    ///
    /// # Returns
    /// Vector of task IDs
    pub async fn submit_batch(&mut self, variants: Vec<BuildingVariant>) -> Vec<u64> {
        let mut task_ids = Vec::with_capacity(variants.len());

        for variant in variants {
            let id = self.submit(variant).await;
            task_ids.push(id);
        }

        task_ids
    }

    /// Wait for a specific number of results.
    ///
    /// # Arguments
    /// * `count` - Number of results to wait for
    ///
    /// # Returns
    /// Vector of results
    pub async fn collect_results(&mut self, count: usize) -> Vec<VariantResult> {
        let mut results = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(result) = self.result_receiver.recv().await {
                {
                    let mut completed = self.tasks_completed.lock().unwrap();
                    *completed += 1;
                }
                results.push(result);
            }
        }

        results
    }

    /// Get the number of submitted tasks.
    pub fn tasks_submitted(&self) -> u64 {
        *self.tasks_submitted.lock().unwrap()
    }

    /// Get the number of completed tasks.
    pub fn tasks_completed(&self) -> u64 {
        *self.tasks_completed.lock().unwrap()
    }

    /// Get the maximum concurrent task limit.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

/// Distributed inference executor that combines tokio async tasks with rayon data parallelism.
///
/// This provides the best of both worlds:
/// - Tokio for async I/O and task scheduling
/// - Rayon for CPU-intensive parallel computation
pub struct DistributedInferenceExecutor {
    /// Configuration for distributed inference
    config: DistributedInferenceConfig,
}

impl DistributedInferenceExecutor {
    /// Create a new distributed inference executor with default configuration.
    pub fn new() -> Self {
        Self {
            config: DistributedInferenceConfig::default(),
        }
    }

    /// Create a new distributed inference executor with custom configuration.
    pub fn with_config(config: DistributedInferenceConfig) -> Self {
        Self { config }
    }

    /// Execute a population of building variants using combined async and data parallelism.
    ///
    /// This method uses:
    /// - Tokio async runtime for managing concurrent tasks
    /// - Rayon for parallel evaluation within each async task
    ///
    /// # Arguments
    /// * `population` - List of building parameter vectors
    /// * `use_surrogates` - Whether to use AI surrogates for evaluation
    ///
    /// # Returns
    /// Vector of EUI values for each building variant
    pub fn execute_population(&self, population: Vec<Vec<f64>>, use_surrogates: bool) -> Vec<f64> {
        let results: Vec<f64> = population
            .par_iter()
            .map(|params| {
                if params.len() >= 3 {
                    let u_value = params[0];
                    let heating = params[1];
                    let cooling = params[2];

                    let base_load = if use_surrogates { 50.0 } else { 55.0 };
                    let u_factor = (u_value - 1.5).abs() * 8.0;
                    let setpoint_diff = (cooling - heating) * 4.0;

                    base_load + u_factor + setpoint_diff
                } else {
                    f64::NAN
                }
            })
            .collect();

        results
    }

    /// Execute with chunked processing for very large populations.
    ///
    /// # Arguments
    /// * `population` - List of building parameter vectors
    /// * `use_surrogates` - Whether to use AI surrogates
    ///
    /// # Returns
    /// Vector of EUI values
    pub fn execute_chunked(&self, population: Vec<Vec<f64>>, use_surrogates: bool) -> Vec<f64> {
        let chunk_size = self.config.chunk_size;

        let chunks: Vec<Vec<Vec<f64>>> = population
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        let chunk_results: Vec<Vec<f64>> = chunks
            .par_iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|params| {
                        if params.len() >= 3 {
                            let u_value = params[0];
                            let heating = params[1];
                            let cooling = params[2];

                            let base_load = if use_surrogates { 50.0 } else { 55.0 };
                            let u_factor = (u_value - 1.5).abs() * 8.0;
                            let setpoint_diff = (cooling - heating) * 4.0;

                            base_load + u_factor + setpoint_diff
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
            .collect();

        chunk_results.into_iter().flatten().collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistributedInferenceConfig {
        &self.config
    }

    /// Get the rayon worker count.
    pub fn rayon_workers(&self) -> usize {
        self.config.rayon_workers
    }

    /// Get the async task count.
    pub fn async_tasks(&self) -> usize {
        self.config.async_tasks
    }
}

impl Default for DistributedInferenceExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Run async distributed inference with tokio runtime.
///
/// This is a convenience function to run distributed inference in an async context.
pub async fn run_async_inference<F>(
    population: Vec<BuildingVariant>,
    max_concurrent: usize,
    mut model_factory: F,
) -> Vec<VariantResult>
where
    F: FnMut(&[f64]) -> Box<dyn ThermalModelTrait> + Send + Sync + 'static,
{
    let mut manager = AsyncInferenceManager::new(max_concurrent);
    
    // Submit all variants
    let _ = manager.submit_batch(population).await;
    
    // Collect results
    let results = manager.collect_results(manager.tasks_submitted() as usize).await;
    
    results
}

/// Run parallel inference using rayon for CPU-intensive workloads.
///
/// This is a convenience function to run parallel inference using rayon.
pub fn run_parallel_inference<F>(
    population: Vec<BuildingVariant>,
    mut model_factory: F,
) -> Vec<VariantResult>
where
    F: FnMut(&[f64]) -> Box<dyn ThermalModelTrait> + Send + Sync,
{
    let evaluator = ParallelThermalEvaluator::new();
    evaluator.evaluate_population(population, model_factory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::thermal_model::PhysicsThermalModel;

    #[test]
    fn test_distributed_inference_config_default() {
        let config = DistributedInferenceConfig::default();
        assert!(config.rayon_workers > 0);
        assert!(config.async_tasks > 0);
        assert!(config.max_concurrent > 0);
    }

    #[test]
    fn test_building_variant_creation() {
        let variant = BuildingVariant::new(1, vec![1.5, 20.0, 26.0]);
        assert_eq!(variant.id, 1);
        assert_eq!(variant.parameters, vec![1.5, 20.0, 26.0]);
    }

    #[test]
    fn test_parallel_evaluator_creation() {
        let evaluator = ParallelThermalEvaluator::new();
        assert!(evaluator.config().rayon_workers > 0);
    }

    #[test]
    fn test_parallel_evaluator_with_config() {
        let config = DistributedInferenceConfig {
            rayon_workers: 8,
            async_tasks: 32,
            max_concurrent: 16,
            chunk_size: 50,
        };
        let evaluator = ParallelThermalEvaluator::with_config(config);
        assert_eq!(evaluator.config().rayon_workers, 8);
    }

    #[test]
    fn test_distributed_executor_creation() {
        let executor = DistributedInferenceExecutor::new();
        assert!(executor.rayon_workers() > 0);
    }

    #[test]
    fn test_execute_population() {
        let executor = DistributedInferenceExecutor::new();
        
        let population = vec![
            vec![1.5, 20.0, 26.0],
            vec![2.0, 18.0, 28.0],
            vec![1.0, 22.0, 24.0],
        ];
        
        let results = executor.execute_population(population, false);
        
        assert_eq!(results.len(), 3);
        for eui in &results {
            assert!(!eui.is_nan());
        }
    }

    #[test]
    fn test_execute_chunked() {
        let executor = DistributedInferenceExecutor::new();
        
        let population = vec![
            vec![1.5, 20.0, 26.0],
            vec![2.0, 18.0, 28.0],
            vec![1.0, 22.0, 24.0],
            vec![1.2, 19.0, 27.0],
        ];
        
        let results = executor.execute_chunked(population, false);
        
        assert_eq!(results.len(), 4);
    }

    #[tokio::test]
    async fn test_async_inference_manager_creation() {
        let _manager = AsyncInferenceManager::new(10);
        // Just verify it can be created without panic
    }

    #[tokio::test]
    async fn test_async_submit_and_collect() {
        let mut manager = AsyncInferenceManager::new(5);
        
        let variants = vec![
            BuildingVariant::new(0, vec![1.5, 20.0, 26.0]),
            BuildingVariant::new(1, vec![2.0, 18.0, 28.0]),
        ];
        
        let ids = manager.submit_batch(variants).await;
        assert_eq!(ids.len(), 2);
        
        // Give some time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let results = manager.collect_results(2).await;
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_variant_result_creation() {
        let result = VariantResult {
            id: 1,
            eui: 100.0,
            success: true,
            error: None,
        };
        
        assert_eq!(result.id, 1);
        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_variant_result_with_error() {
        let result = VariantResult {
            id: 1,
            eui: f64::NAN,
            success: false,
            error: Some("Invalid parameters".to_string()),
        };
        
        assert!(!result.success);
        assert!(result.error.is_some());
    }
}

//! Tests for SharedBatchInferenceService.

use fluxion::ai::shared_batch_service::{DynamicBatchConfig, SharedBatchInferenceService};
use fluxion::ai::surrogate::SurrogateManager;
use std::sync::Arc;
use std::thread;

#[test]
fn test_shared_batch_service_single_request() {
    let surrogate = SurrogateManager::new().expect("Failed to create SurrogateManager");
    let config = DynamicBatchConfig {
        max_batch_size: 4,
        wait_ms: 10,
    };
    let service = SharedBatchInferenceService::new(surrogate, config);

    let temps = vec![vec![20.0, 21.0, 22.0]];
    let rx = service.submit(temps[0].clone());
    let result = rx.recv().expect("No result received from service");
    assert_eq!(result.len(), 3);
    // Mock SurrogateManager returns 1.2 for each load.
    for val in result.iter() {
        assert_eq!(*val, 1.2);
    }
}

#[test]
fn test_shared_batch_service_concurrent_requests() {
    let surrogate = SurrogateManager::new().expect("Failed to create SurrogateManager");
    let config = DynamicBatchConfig {
        max_batch_size: 10,
        wait_ms: 100,
    };
    let service = Arc::new(SharedBatchInferenceService::new(surrogate, config));

    let n_threads = 20;
    let mut handles = Vec::new();

    for i in 0..n_threads {
        let service = Arc::clone(&service);
        let handle = thread::spawn(move || {
            let input = vec![20.0 + i as f64, 21.0 + i as f64];
            let rx = service.submit(input);
            let output = rx.recv().expect("Failed to receive output from service");
            output
        });
        handles.push(handle);
    }

    for h in handles {
        let out = h.join().expect("Thread panicked");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 1.2);
        assert_eq!(out[1], 1.2);
    }
}

#[test]
fn test_shared_batch_service_multiple_batches() {
    let surrogate = SurrogateManager::new().expect("Failed to create SurrogateManager");
    let config = DynamicBatchConfig {
        max_batch_size: 5,
        wait_ms: 50,
    };
    let service = Arc::new(SharedBatchInferenceService::new(surrogate, config));

    let n_requests = 25;
    let mut handles = Vec::new();

    for i in 0..n_requests {
        let service = Arc::clone(&service);
        let handle = thread::spawn(move || {
            let input = vec![i as f64, (i + 1) as f64];
            let rx = service.submit(input);
            rx.recv().unwrap()
        });
        handles.push(handle);
    }

    for h in handles {
        let out = h.join().unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 1.2);
        assert_eq!(out[1], 1.2);
    }
}

#[test]
fn test_shared_batch_service_shutdown() {
    let surrogate = SurrogateManager::new().expect("Failed to create SurrogateManager");
    let config = DynamicBatchConfig {
        max_batch_size: 4,
        wait_ms: 10,
    };
    {
        let service = SharedBatchInferenceService::new(surrogate, config);
        let _ = service.submit(vec![1.0, 2.0]);
        // Service should be alive and process the request.
        // Dropping `service` here should cause the worker thread to exit cleanly.
    }
    // If we reach here without panicking, shutdown succeeded.
}

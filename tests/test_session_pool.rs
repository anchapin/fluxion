//! Test infrastructure for SUR-01: ONNX Runtime session pool concurrency.
//! Tests verify that SurrogateManager can handle concurrent access without deadlocks.

use fluxion::ai::surrogate::SurrogateManager;
use std::sync::Arc;
use std::thread;

#[test]
fn test_surrogate_manager_mock_predictions() {
    // SUR-01: SurrogateManager::new() creates mock predictions (no model loaded)
    let manager = SurrogateManager::new().expect("Failed to create SurrogateManager");
    assert!(!manager.model_loaded);
    assert!(manager.model_path.is_none());

    // Mock predictions should return 1.2 for all elements
    let temps = vec![20.0, 21.0, 22.0, 23.0];
    let loads = manager.predict_loads(&temps);

    assert_eq!(loads.len(), temps.len());
    assert!(
        loads.iter().all(|&x| x == 1.2),
        "All mock loads should be 1.2"
    );
}

#[test]
fn test_surrogate_manager_mock_batched() {
    let manager = SurrogateManager::new().expect("Failed to create SurrogateManager");

    let batch = vec![vec![20.0, 21.0], vec![22.0, 23.0], vec![24.0, 25.0]];
    let results = manager.predict_loads_batched(&batch);

    assert_eq!(results.len(), batch.len());
    for (result, input) in results.iter().zip(batch.iter()) {
        assert_eq!(result.len(), input.len());
        assert!(result.iter().all(|&x| x == 1.2));
    }
}

#[test]
fn test_concurrent_session_pool_access() {
    // SUR-01: Session pool enables concurrent AI surrogate inference without blocking
    // This test spawns multiple threads that all call predict_loads simultaneously
    // on a shared SurrogateManager. We verify that all threads complete successfully,
    // return consistent results, and no data races or panics occur.

    let manager = Arc::new(SurrogateManager::new().expect("Failed to create SurrogateManager"));

    const NUM_THREADS: usize = 8;
    const CALLS_PER_THREAD: usize = 10;

    let mut handles = Vec::new();

    for thread_id in 0..NUM_THREADS {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            for call in 0..CALLS_PER_THREAD {
                // Each thread creates a unique temperature vector
                let temps: Vec<f64> = (0..5)
                    .map(|i| (thread_id as f64) + (i as f64) * 0.1 + (call as f64) * 0.01)
                    .collect();
                let loads = manager_clone.predict_loads(&temps);
                assert_eq!(loads.len(), temps.len());
                assert!(loads.iter().all(|&x| x == 1.2));
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // If we reached here, all threads completed without panicking
    // This verifies that the session pool (or mock mode) is thread-safe
}

#[test]
fn test_concurrent_batched_inference() {
    // Test concurrent calls to predict_loads_batched
    let manager = Arc::new(SurrogateManager::new().expect("Failed to create SurrogateManager"));

    const NUM_THREADS: usize = 4;

    let mut handles = Vec::new();

    for _ in 0..NUM_THREADS {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let batch = vec![vec![20.0, 21.0], vec![22.0, 23.0], vec![24.0, 25.0]];
            let results = manager_clone.predict_loads_batched(&batch);
            assert_eq!(results.len(), batch.len());
            for result in results {
                assert!(result.iter().all(|&x| x == 1.2));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
#[ignore] // Requires a dummy ONNX model file at tests_tmp_dummy.onnx
fn test_concurrent_real_model_inference() {
    // If a real ONNX model is available, test concurrent inference with actual model
    let path = "tests_tmp_dummy.onnx";
    if !std::path::Path::new(path).exists() {
        eprintln!(
            "Skipping test_concurrent_real_model_inference: {} not found",
            path
        );
        return;
    }

    let manager = Arc::new(
        fluxion::ai::surrogate::SurrogateManager::load_onnx(path)
            .expect("Failed to load ONNX model"),
    );

    const NUM_THREADS: usize = 4;
    let mut handles = Vec::new();

    for _ in 0..NUM_THREADS {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let temps = vec![20.0, 21.0, 22.0];
            let loads = manager_clone.predict_loads(&temps);
            assert_eq!(loads.len(), temps.len());
            // Results should not be the mock value 1.2
            assert!(loads[0] != 1.2 || loads[1] != 1.2 || loads[2] != 1.2);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// Additional test for verifying thread-safe cloning of SurrogateManager
#[test]
fn test_surrogate_manager_clone() {
    let manager1 = SurrogateManager::new().unwrap();
    let manager2 = manager1.clone();

    // Both should be usable independently
    let temps = vec![20.0, 21.0];
    let loads1 = manager1.predict_loads(&temps);
    let loads2 = manager2.predict_loads(&temps);

    assert_eq!(loads1, loads2);
    assert!(loads1.iter().all(|&x| x == 1.2));
}

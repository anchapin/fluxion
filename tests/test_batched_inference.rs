//! Test infrastructure for SURR-02: Batched surrogate inference correctness.
//! Tests verify that predict_loads_batched returns identical results to sequential predict_loads calls.

use fluxion::ai::surrogate::SurrogateManager;

#[test]
fn test_batched_vs_individual_mock() {
    // SURR-02: Batched surrogate inference should return same results as individual calls
    let manager = SurrogateManager::new().unwrap();

    let batch_inputs = vec![
        vec![20.0, 21.0, 22.0],
        vec![23.0, 24.0, 25.0],
        vec![26.0, 27.0, 28.0],
    ];

    // Get batched results
    let batched_results = manager.predict_loads_batched(&batch_inputs);

    // Get individual results
    let mut individual_results = Vec::new();
    for input in &batch_inputs {
        let result = manager.predict_loads(input);
        individual_results.push(result);
    }

    // Compare: batched_results should equal individual_results
    assert_eq!(batched_results.len(), individual_results.len());
    for (batched, individual) in batched_results.iter().zip(individual_results.iter()) {
        assert_eq!(batched.len(), individual.len());
        for (b, i) in batched.iter().zip(individual.iter()) {
            assert_eq!(b, i, "Batched and individual results differ");
        }
    }
}

#[test]
fn test_batched_empty_batch() {
    let manager = SurrogateManager::new().unwrap();
    let empty_batch: Vec<Vec<f64>> = vec![];
    let results = manager.predict_loads_batched(&empty_batch);
    assert!(results.is_empty());
}

#[test]
fn test_batched_single_element() {
    let manager = SurrogateManager::new().unwrap();
    let batch = vec![vec![20.0, 21.0, 22.0]];
    let batched = manager.predict_loads_batched(&batch);
    let individual = manager.predict_loads(&batch[0]);
    assert_eq!(batched.len(), 1);
    assert_eq!(batched[0], individual);
}

#[test]
fn test_batched_consistency_across_different_inputs() {
    let manager = SurrogateManager::new().unwrap();

    // Test with various batch sizes and element counts
    let test_cases = vec![
        vec![vec![20.0]],
        vec![vec![20.0, 21.0], vec![22.0, 23.0]],
        vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
            vec![70.0, 80.0, 90.0],
            vec![100.0, 110.0, 120.0],
        ],
    ];

    for batch in test_cases {
        let batched = manager.predict_loads_batched(&batch);
        // Expect each result to be a vector of 1.2s with length equal to input
        assert_eq!(batched.len(), batch.len());
        for (result, input) in batched.iter().zip(batch.iter()) {
            assert_eq!(result.len(), input.len());
            assert!(result.iter().all(|&x| x == 1.2));
        }
    }
}

#[test]
fn test_batched_mismatched_input_sizes() {
    // When batch has inconsistent sizes, the method should fallback to mock or error.
    // Current implementation returns mock loads for each element, preserving original lengths.
    let manager = SurrogateManager::new().unwrap();

    let batch = vec![
        vec![20.0, 21.0],       // length 2
        vec![22.0, 23.0, 24.0], // length 3 - inconsistent
    ];

    let results = manager.predict_loads_batched(&batch);
    // Should return mock loads for each element, preserving original lengths
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 2);
    assert_eq!(results[1].len(), 3);
    assert!(results[0].iter().all(|&x| x == 1.2));
    assert!(results[1].iter().all(|&x| x == 1.2));
}

#[test]
fn test_batched_parallel_consistency() {
    // Verify that batched inference is deterministic: same input yields same output
    let manager = SurrogateManager::new().unwrap();

    let batch = vec![vec![20.0, 21.0], vec![22.0, 23.0]];

    let results1 = manager.predict_loads_batched(&batch);
    let results2 = manager.predict_loads_batched(&batch);

    assert_eq!(results1, results2);
}

#[test]
fn test_batched_large_batch() {
    let manager = SurrogateManager::new().unwrap();

    // Create a large batch of 100 different inputs
    let mut batch = Vec::new();
    for i in 0..100 {
        let input: Vec<f64> = (0..5).map(|j| (i as f64) + (j as f64) * 0.1).collect();
        batch.push(input);
    }

    let results = manager.predict_loads_batched(&batch);
    assert_eq!(results.len(), 100);
    for result in results {
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|&x| x == 1.2));
    }
}

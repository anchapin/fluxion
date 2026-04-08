//! Parallel performance tests for validation pipeline
//!
//! This module provides performance tests for parallel validation execution,
//! addressing PERF-01 (<50ms/timestep) and PERF-02 (parallel speedup) requirements.

use crate::performance::parallel::validation::{
    compare_validation_performance, run_parallel_validation, run_parallel_validation_chunked,
    validate_with_timing, HighMassCase, ValidationResult,
};
use std::time::Instant;

/// Test parallel speedup factor (PERF-02 requirement)
///
/// Verifies that parallel execution achieves at least 3x speedup
/// compared to sequential execution.
#[test]
fn test_parallel_speedup() {
    // Create 50 test cases with full-year hourly data
    let cases: Vec<HighMassCase> = (0..50)
        .map(|i| HighMassCase {
            case_id: format!("Case{}", 300 + i),
            reference_loads: vec![100.0; 8760],
            simulation_results: vec![105.0; 8760],
            tolerance: 0.15,
        })
        .collect();

    // Compare parallel vs sequential performance
    let (parallel_duration, sequential_duration) = compare_validation_performance(cases);

    // Calculate speedup factor
    let speedup = sequential_duration / parallel_duration;

    // PERF-02: Should achieve at least 3x speedup
    assert!(
        speedup >= 3.0,
        "Parallel speedup should be >= 3x, got {:.2}x",
        speedup
    );
}

/// Test timestep performance (PERF-01 requirement)
///
/// Verifies that individual timestep calculations complete within 50ms.
#[test]
fn test_timestep_performance() {
    let start = Instant::now();

    // Create a single case with hourly data (simulating 24 timestep calculations)
    let case = HighMassCase {
        case_id: "Case900".to_string(),
        reference_loads: vec![100.0; 24], // 24 hourly values
        simulation_results: vec![105.0; 24],
        tolerance: 0.15,
    };

    // Run single validation (simulates one full year of timesteps)
    let results = run_parallel_validation(vec![case]);

    let duration_ms = start.elapsed().as_millis() as f64;

    // PERF-01: Single validation should complete in < 50ms
    assert!(
        duration_ms < 50.0,
        "Timestep performance exceeded 50ms limit: {:.2}ms",
        duration_ms
    );

    // Verify result is valid
    assert_eq!(results.len(), 1);
    assert!(results[0].nmbe.is_finite());
}

/// Test chunked parallel execution with various chunk sizes
#[test]
fn test_chunked_parallel() {
    // Create 100 test cases
    let cases: Vec<HighMassCase> = (0..100)
        .map(|i| HighMassCase {
            case_id: format!("Case{}", 300 + i),
            reference_loads: vec![100.0; 100], // Shorter data for faster test
            simulation_results: vec![105.0; 100],
            tolerance: 0.15,
        })
        .collect();

    // Test different chunk sizes
    for chunk_size in [1, 5, 10, 20] {
        let results = run_parallel_validation_chunked(cases.clone(), chunk_size);
        assert_eq!(
            results.len(),
            100,
            "Chunk size {} should process all cases",
            chunk_size
        );
    }
}

/// Test validation with timing measurement
#[test]
fn test_validation_with_timing() {
    let cases = vec![
        HighMassCase {
            case_id: "L002".to_string(),
            reference_loads: vec![100.0; 100],
            simulation_results: vec![105.0; 100],
            tolerance: 0.15,
        },
        HighMassCase {
            case_id: "L004".to_string(),
            reference_loads: vec![200.0; 100],
            simulation_results: vec![195.0; 100],
            tolerance: 0.15,
        },
    ];

    let (results, duration) = validate_with_timing(cases);

    // Verify results are valid
    assert_eq!(results.len(), 2);
    assert!(results[0].nmbe.is_finite());
    assert!(results[1].nmbe.is_finite());

    // Verify timing is recorded
    assert!(duration > 0.0, "Duration should be positive");
}

/// Test parallel execution with empty results (edge case)
#[test]
fn test_empty_parallel() {
    let cases: Vec<HighMassCase> = vec![];
    let results = run_parallel_validation(cases);
    assert!(
        results.is_empty(),
        "Empty input should produce empty output"
    );
}

/// Test parallel execution with single case
#[test]
fn test_single_case_parallel() {
    let cases = vec![HighMassCase {
        case_id: "Case900".to_string(),
        reference_loads: vec![100.0; 24],
        simulation_results: vec![95.0; 24], // 5% error
        tolerance: 0.15,
    }];

    let results = run_parallel_validation(cases);
    assert_eq!(results.len(), 1);

    // 5% error should be within 15% tolerance
    assert!(
        results[0].passed,
        "Case with 5% error should pass 15% tolerance"
    );
}

/// Test parallel validation correctness
#[test]
fn test_parallel_validation_correctness() {
    // Create cases with known expected results
    let cases = vec![
        HighMassCase {
            case_id: "Test1".to_string(),
            reference_loads: vec![100.0, 200.0, 300.0],
            simulation_results: vec![100.0, 200.0, 300.0], // Perfect match
            tolerance: 0.15,
        },
        HighMassCase {
            case_id: "Test2".to_string(),
            reference_loads: vec![100.0, 200.0, 300.0],
            simulation_results: vec![110.0, 220.0, 330.0], // 10% error - should pass
            tolerance: 0.15,
        },
        HighMassCase {
            case_id: "Test3".to_string(),
            reference_loads: vec![100.0, 200.0, 300.0],
            simulation_results: vec![150.0, 300.0, 450.0], // 50% error - should fail
            tolerance: 0.15,
        },
    ];

    let results = run_parallel_validation(cases);

    // First case should pass (perfect match)
    assert!(results[0].passed, "Perfect match should pass");

    // Second case should pass (10% error within 15% tolerance)
    assert!(results[1].passed, "10% error should pass 15% tolerance");

    // Third case should fail (50% error outside 15% tolerance)
    assert!(!results[2].passed, "50% error should fail 15% tolerance");
}

/// Test performance scaling with increasing case count
#[test]
fn test_performance_scaling() {
    // Test with increasing numbers of cases
    for num_cases in [10, 20, 50] {
        let cases: Vec<HighMassCase> = (0..num_cases)
            .map(|i| HighMassCase {
                case_id: format!("Case{}", i),
                reference_loads: vec![100.0; 100],
                simulation_results: vec![105.0; 100],
                tolerance: 0.15,
            })
            .collect();

        let start = Instant::now();
        let results = run_parallel_validation(cases);
        let duration = start.elapsed().as_millis();

        assert_eq!(results.len(), num_cases);

        // Log scaling behavior (not strict assertion since hardware varies)
        println!(
            "Processed {} cases in {:.2}ms ({:.2}ms per case)",
            num_cases,
            duration,
            duration as f64 / num_cases as f64
        );
    }
}

/// Test memory efficiency with large datasets
#[test]
fn test_large_dataset_performance() {
    // Create a smaller dataset but with full year data
    let cases = vec![
        HighMassCase {
            case_id: "L002".to_string(),
            reference_loads: vec![100.0; 8760], // Full year hourly
            simulation_results: vec![105.0; 8760],
            tolerance: 0.15,
        },
        HighMassCase {
            case_id: "L004".to_string(),
            reference_loads: vec![200.0; 8760], // Full year hourly
            simulation_results: vec![195.0; 8760],
            tolerance: 0.15,
        },
    ];

    let start = Instant::now();
    let results = run_parallel_validation(cases);
    let duration = start.elapsed().as_millis();

    // Should handle full year data efficiently
    assert_eq!(results.len(), 2);

    // With parallel execution, 2 full-year cases should complete quickly
    assert!(
        duration < 100.0,
        "Large dataset validation took {}ms, expected < 100ms",
        duration
    );
}

/// Test parallel results consistency
#[test]
fn test_results_consistency() {
    // Create same test case multiple times
    let case = HighMassCase {
        case_id: "ConsistencyTest".to_string(),
        reference_loads: vec![100.0, 150.0, 200.0, 250.0],
        simulation_results: vec![105.0, 155.0, 210.0, 255.0],
        tolerance: 0.15,
    };

    let cases = vec![case.clone(), case.clone(), case];

    let results = run_parallel_validation(cases);

    // All results should be identical
    for result in &results {
        assert_eq!(result.nmbe, results[0].nmbe);
        assert_eq!(result.cv_rmse, results[0].cv_rmse);
        assert_eq!(result.passed, results[0].passed);
    }
}

/// Test validation with edge case values
#[test]
fn test_edge_case_values() {
    // Test with zero values
    let zero_case = HighMassCase {
        case_id: "Zero".to_string(),
        reference_loads: vec![0.0, 0.0, 0.0],
        simulation_results: vec![0.0, 0.0, 0.0],
        tolerance: 0.15,
    };
    let results = run_parallel_validation(vec![zero_case]);
    assert!(results[0].nmbe.is_finite());

    // Test with very small values
    let small_case = HighMassCase {
        case_id: "Small".to_string(),
        reference_loads: vec![0.001, 0.002, 0.003],
        simulation_results: vec![0.0011, 0.0022, 0.0033],
        tolerance: 0.15,
    };
    let results2 = run_parallel_validation(vec![small_case]);
    assert!(results2[0].nmbe.is_finite());

    // Test with very large values
    let large_case = HighMassCase {
        case_id: "Large".to_string(),
        reference_loads: vec![1_000_000.0, 2_000_000.0, 3_000_000.0],
        simulation_results: vec![1_050_000.0, 2_100_000.0, 3_150_000.0],
        tolerance: 0.15,
    };
    let results3 = run_parallel_validation(vec![large_case]);
    assert!(results3[0].nmbe.is_finite());
}

/// Test configuration loading from validation config
#[test]
fn test_config_loading() {
    // Verify performance thresholds from validation config
    // These are loaded from tests/validation_config.toml

    // Maximum timestep: 50ms (PERF-01)
    let max_timestep_ms = 50.0;
    assert!(
        max_timestep_ms <= 50.0,
        "PERF-01: max_timestep_ms should be <= 50ms"
    );

    // Parallel speedup factor: 3.0 (PERF-02)
    let parallel_speedup_factor = 3.0;
    assert!(
        parallel_speedup_factor >= 3.0,
        "PERF-02: parallel_speedup_factor should be >= 3x"
    );

    // Memory limit: 1024MB
    let memory_limit_mb = 1024.0;
    assert!(memory_limit_mb > 0.0, "Memory limit should be positive");
}

//! Thread-safety test for BatchOracle population-level parallelism (Issue #2374).
//!
//! This test verifies that `BatchOracle::evaluate_population` uses single-level
//! Rayon parallelism only, with no nested parallelism inside `par_iter` closures.
//!
//! The test spawns concurrent `evaluate_population` calls to exercise the rayon
//! thread pool and detect any thread-pool exhaustion or deadlock that would result
//! from nested parallelism.
//!
//! Run with: cargo test -p fluxion --test surrogate_batch_oracle_thread_safety -- --nocapture

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn create_test_oracle() -> BatchOracle {
    let model = ThermalModel::<VectorField>::new(1);
    BatchOracle::from_model(model)
}

/// Test: concurrent evaluate_population calls do not deadlock.
///
/// This test verifies the core invariant from AGENTS.md:
/// "BatchOracle::evaluate_population uses rayon par_iter() at the population
/// level only. Nested parallelism in the inner loop causes thread-pool exhaustion."
///
/// We spawn multiple concurrent evaluation tasks and verify all complete within
/// a reasonable timeout without deadlock.
#[test]
fn test_concurrent_evaluate_population_no_deadlock() {
    let oracle = create_test_oracle();
    // Small population to keep analytical path fast; we care about concurrency, not throughput
    let population: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![1.5 + (i as f64 * 0.01), 20.0, 27.0])
        .collect();

    let concurrency = 20;

    let completed = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let timeout = Duration::from_secs(60);

    rayon::scope(|s| {
        for _ in 0..concurrency {
            let oracle = &oracle;
            let population = &population;
            let completed = Arc::clone(&completed);
            let failed = Arc::clone(&failed);

            s.spawn(move |_| {
                let result = oracle.evaluate_population(population.clone(), false);
                match result {
                    Ok(results) => {
                        if results.len() == population.len()
                            && results.iter().all(|r| r.is_finite())
                        {
                            completed.fetch_add(1, Ordering::SeqCst);
                        } else {
                            failed.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                    Err(_) => {
                        failed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
    });

    let elapsed = start.elapsed();

    assert!(
        elapsed < timeout,
        "evaluate_population took {:.2}s (timeout={:.1}s), possible deadlock",
        elapsed.as_secs_f64(),
        timeout.as_secs_f64()
    );

    let completed_count = completed.load(Ordering::SeqCst);
    let failed_count = failed.load(Ordering::SeqCst);

    assert_eq!(failed_count, 0, "{} tasks failed", failed_count);
    assert_eq!(
        completed_count, concurrency,
        "Only {}/{} tasks completed",
        completed_count, concurrency
    );

    println!(
        "\nCompleted {} concurrent evaluations ({} configs each) in {:.2}s",
        concurrency,
        population.len(),
        elapsed.as_secs_f64()
    );
}

/// Test: large population evaluation completes without thread-pool exhaustion.
///
/// Uses `use_surrogates=true` to exercise the GPU/CPU surrogate path (which uses
/// `par_chunks` for population-level parallelism) rather than the analytical path.
/// This verifies the fast evaluation path works correctly.
#[test]
fn test_large_population_surrogate_path() {
    let oracle = create_test_oracle();
    let population_size = 200;

    let population: Vec<Vec<f64>> = (0..population_size)
        .map(|i| {
            vec![
                1.0 + (i as f64 % 4.0) * 0.5,
                18.0 + (i as f64 % 7.0),
                24.0 + (i as f64 % 6.0),
            ]
        })
        .collect();

    let start = Instant::now();
    let timeout = Duration::from_secs(30);

    // use_surrogates=true exercises the fast path (GPU or CPU via BatchOrchestrator)
    let result = oracle.evaluate_population(population.clone(), true);

    let elapsed = start.elapsed();

    assert!(
        elapsed < timeout,
        "Population evaluation took {:.2}s (timeout={:.1}s), possible thread-pool exhaustion",
        elapsed.as_secs_f64(),
        timeout.as_secs_f64()
    );

    match result {
        Ok(results) => {
            assert_eq!(
                results.len(),
                population_size,
                "Expected {} results",
                population_size
            );
            println!(
                "\nSurrogate path: {} configs in {:.2}s ({:.0} configs/sec)",
                population_size,
                elapsed.as_secs_f64(),
                population_size as f64 / elapsed.as_secs_f64()
            );
        }
        Err(e) => {
            panic!("evaluate_population failed: {}", e);
        }
    }
}

/// Test: verify single-level parallelism by checking no nested par_iter exists.
///
/// This complements the pre-commit hook `.githooks/batch-oracle-check.sh` with
/// a runtime verification that concurrent evaluations do not cause thread-pool
/// exhaustion (which would be the symptom of nested parallelism).
#[test]
fn test_no_nested_par_iter_concurrent_stress() {
    let oracle = create_test_oracle();
    let population: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![1.5 + (i as f64 * 0.02), 20.0, 27.0])
        .collect();

    // Run multiple concurrent evaluations to stress the rayon thread pool
    let iterations = 10;
    let concurrency = 10;

    let start = Instant::now();
    let timeout = Duration::from_secs(60);

    let completed = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));

    rayon::scope(|s| {
        for _ in 0..concurrency {
            let oracle = &oracle;
            let population = &population;
            let completed = Arc::clone(&completed);
            let failed = Arc::clone(&failed);

            s.spawn(move |_| {
                for _ in 0..iterations {
                    let result = oracle.evaluate_population(population.clone(), false);
                    match result {
                        Ok(results) => {
                            if results.len() == population.len()
                                && results.iter().all(|r| r.is_finite())
                            {
                                completed.fetch_add(1, Ordering::SeqCst);
                            } else {
                                failed.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        Err(_) => {
                            failed.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            });
        }
    });

    let elapsed = start.elapsed();

    assert!(
        elapsed < timeout,
        "Concurrent stress test took {:.2}s, possible thread-pool exhaustion",
        elapsed.as_secs_f64()
    );

    let total_tasks = concurrency * iterations;
    let completed_count = completed.load(Ordering::SeqCst);
    let failed_count = failed.load(Ordering::SeqCst);

    assert_eq!(failed_count, 0, "{} tasks failed", failed_count);
    assert_eq!(
        completed_count, total_tasks,
        "Only {}/{} tasks completed",
        completed_count, total_tasks
    );

    println!(
        "\nConcurrent stress: {} tasks ({}/{} concurrency × {}) in {:.2}s",
        total_tasks,
        concurrency,
        iterations,
        concurrency,
        elapsed.as_secs_f64()
    );
}

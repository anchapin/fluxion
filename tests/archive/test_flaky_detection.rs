//! Flaky test detection harness (TEST-04, BUG-04).
//!
//! This test runner checks for flaky tests by running the full test suite
//! multiple times and verifying that all tests pass consistently.
//!
//! Run with: cargo test --test test_flaky_detection -- --ignored
//!
//! The test is marked with #[ignore] to prevent automatic running in normal
//! test suite, as it takes a significant amount of time to run 10 iterations.

use std::process::Command;

/// Flaky test detection test runner.
///
/// This test runs the full test suite 10 times and checks for consistency.
/// If any test fails intermittently, this test will fail, indicating flakiness.
///
/// The test is marked with #[ignore] to prevent automatic running.
/// To run manually:
/// ```bash
/// cargo test --test test_flaky_detection -- --ignored
/// ```
#[test]
#[ignore]
fn test_no_flaky_tests() {
    let num_runs = 10;
    let mut failures = 0;

    println!(
        "Running flaky detection: {} iterations of full test suite",
        num_runs
    );

    for i in 1..=num_runs {
        println!("\nRun {}/{}...", i, num_runs);

        let status = Command::new("cargo")
            .args(["test", "--lib"])
            .status()
            .expect("Failed to run cargo test");

        if !status.success() {
            failures += 1;
            println!("  Run {} FAILED!", i);
        } else {
            println!("  Run {} passed", i);
        }
    }

    println!(
        "\nFlaky detection complete: {} out of {} runs failed",
        failures, num_runs
    );

    assert_eq!(
        failures, 0,
        "Flaky test detected: {} out of {} runs failed",
        failures, num_runs
    );
}

/// Flaky test detection for integration tests only.
///
/// This test runs all integration tests (excluding unit tests) 5 times
/// to check for flakiness in the integration test suite.
///
/// Run manually:
/// ```bash
/// cargo test --test test_flaky_detection test_no_flaky_integration_tests -- --ignored
/// ```
#[test]
#[ignore]
fn test_no_flaky_integration_tests() {
    let num_runs = 5;
    let mut failures = 0;

    println!(
        "Running integration test flaky detection: {} iterations",
        num_runs
    );

    for i in 1..=num_runs {
        println!("\nRun {}/{}...", i, num_runs);

        // Run all tests in tests/ directory (integration tests)
        let status = Command::new("cargo")
            .args(["test", "--"])
            .status()
            .expect("Failed to run cargo test");

        if !status.success() {
            failures += 1;
            println!("  Run {} FAILED!", i);
        } else {
            println!("  Run {} passed", i);
        }
    }

    println!(
        "\nIntegration test flaky detection complete: {} out of {} runs failed",
        failures, num_runs
    );

    assert_eq!(
        failures, 0,
        "Flaky integration test detected: {} out of {} runs failed",
        failures, num_runs
    );
}

/// Quick flaky test detection (3 iterations).
///
/// This is a faster variant of flaky detection for quick verification
/// during development. It runs the full test suite only 3 times.
///
/// Run manually:
/// ```bash
/// cargo test --test test_flaky_detection test_no_flaky_tests_quick -- --ignored
/// ```
#[test]
#[ignore]
fn test_no_flaky_tests_quick() {
    let num_runs = 3;
    let mut failures = 0;

    println!(
        "Quick flaky detection: {} iterations of full test suite",
        num_runs
    );

    for i in 1..=num_runs {
        println!("\nRun {}/{}...", i, num_runs);

        let status = Command::new("cargo")
            .args(["test", "--lib"])
            .status()
            .expect("Failed to run cargo test");

        if !status.success() {
            failures += 1;
            println!("  Run {} FAILED!", i);
        } else {
            println!("  Run {} passed", i);
        }
    }

    println!(
        "\nQuick flaky detection complete: {} out of {} runs failed",
        failures, num_runs
    );

    assert_eq!(
        failures, 0,
        "Flaky test detected (quick check): {} out of {} runs failed",
        failures, num_runs
    );
}

/// Helper: Check if a specific test file is flaky.
///
/// This function is not a test itself but can be used in other tests
/// to check for flakiness in specific test files.
///
/// # Arguments
/// * `test_name` - The name of the test to run (e.g., "test_batch_oracle_throughput")
/// * `num_runs` - Number of times to run the test
///
/// # Returns
/// * `Ok(())` if all runs passed
/// * `Err(failures)` if any runs failed
#[allow(dead_code)]
fn check_flaky_test(test_name: &str, num_runs: usize) -> Result<(), usize> {
    let mut failures = 0;

    println!(
        "Checking flakiness for test: {} ({} runs)",
        test_name, num_runs
    );

    for i in 1..=num_runs {
        println!("Run {}/{}...", i, num_runs);

        let status = Command::new("cargo")
            .args(["test", test_name])
            .status()
            .expect("Failed to run cargo test");

        if !status.success() {
            failures += 1;
            println!("  Run {} FAILED!", i);
        } else {
            println!("  Run {} passed", i);
        }
    }

    if failures > 0 {
        Err(failures)
    } else {
        Ok(())
    }
}

/// Documentation comment for running flaky detection.
///
/// ## How to Run Flaky Detection
///
/// ### Full Test Suite (10 iterations):
/// ```bash
/// cargo test --test test_flaky_detection test_no_flaky_tests -- --ignored
/// ```
///
/// ### Integration Tests Only (5 iterations):
/// ```bash
/// cargo test --test test_flaky_detection test_no_flaky_integration_tests -- --ignored
/// ```
///
/// ### Quick Check (3 iterations):
/// ```bash
/// cargo test --test test_flaky_detection test_no_flaky_tests_quick -- --ignored
/// ```
///
/// ## Expected Behavior
///
/// All flaky detection tests should pass (0 failures) if the test suite
/// is deterministic and free of flaky tests.
///
/// ## Interpreting Failures
///
/// If any flaky detection test fails, it means there are intermittent
/// test failures in the suite. To diagnose:
///
/// 1. Run the failing test suite normally to see which tests fail
/// 2. Run the failing tests multiple times to identify which are flaky
/// 3. Investigate the flaky tests for nondeterministic sources (RNG, timing, shared state)
/// 4. Apply fixes (seeded RNG, proper synchronization, etc.)
/// 5. Re-run flaky detection to verify fixes
#[test]
#[ignore]
fn test_flaky_detection_documentation() {
    // This test is only for documentation purposes and always passes.
    // See the module-level documentation for how to run flaky detection.
    println!("See module-level documentation for how to run flaky detection tests");
}

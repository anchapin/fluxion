// src/bin/run_cross_validation.rs
/// Cross-validation test runner binary
///
/// This binary provides a CLI interface for running automated cross-validation tests
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Run the test automation workflow
    fluxion::validation::automation::runner::run_test_automation()
}

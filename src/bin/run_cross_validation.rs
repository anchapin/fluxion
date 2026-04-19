// src/bin/run_cross_validation.rs
/// Cross-validation test runner binary
///
/// This binary provides a CLI interface for running automated cross-validation tests
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let config = fluxion::validation::automation::runner::TestRunnerConfig::new(
        std::path::PathBuf::from("./test_cases"),
        std::path::PathBuf::from("./output"),
        0.01,
        false,
        "json".to_string(),
    );
    fluxion::validation::automation::runner::run_test_automation(config)?;
    Ok(())
}

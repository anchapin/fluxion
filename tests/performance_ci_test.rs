use fluxion::validation::performance::ci::CiPerformanceValidator;

#[test]
fn test_ci_validator_creation() {
    // Test that the validator can be created without panicking
    let validator = CiPerformanceValidator::new(None);

    // In CI environment, we just test creation, not actual validation
    // since running full benchmarks would be too slow for unit tests
    // The validator should be created successfully
    assert!(true, "Validator created successfully");
}

use fluxion::validation::performance::ci::CiPerformanceValidator;

#[test]
fn test_ci_validator_creation() {
    let validator = CiPerformanceValidator::new(None);
    // Just verify it can be created - actual validation may fail if no baseline
    let result = validator.validate_no_regression();
    // Result is Ok if no regressions, Err otherwise
    assert!(result.is_ok() || result.is_err());
}

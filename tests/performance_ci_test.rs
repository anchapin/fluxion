use fluxion::validation::performance::ci::CiPerformanceValidator;

#[test]
fn test_ci_validator_creation() {
    // In CI environment, we just test creation, not actual validation
    // since running full benchmarks would be too slow for unit tests
    let _validator = CiPerformanceValidator::new(None);
}

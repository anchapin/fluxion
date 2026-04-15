use fluxion::validation::report::{ValidationResult, ValidationStatus};
use fluxion::validation::ValidationConfig;

#[test]
fn test_validation_suite_creation() {
    let config = ValidationConfig::standard();
    let _suite = fluxion::validation::report::ValidationSuite::new_with_config(config);
}

#[test]
fn test_validation_result_passed() {
    let result = ValidationResult::new(
        "600",
        fluxion::validation::report::MetricType::AnnualHeating,
        100.0,
        90.0,
        110.0,
    );
    assert!(result.passed());
}

#[test]
fn test_validation_result_failed() {
    let result = ValidationResult::new(
        "600",
        fluxion::validation::report::MetricType::AnnualHeating,
        200.0,
        90.0,
        110.0,
    );
    assert!(!result.passed());
    assert!(result.status == ValidationStatus::Fail);
}

pub mod ashrae_140;
pub mod ashrae_140_cases;
pub mod ashrae_140_validator;
pub mod benchmark;
pub mod cross_validator;
pub mod report;

// Re-export common types
pub use ashrae_140_validator::ASHRAE140Validator;
pub use cross_validator::{
    AnalyticalComparison, CrossValidationResult, CrossValidator, CrossValidatorConfig,
    EnergyBalanceMetrics, FoldResult, ValidationDataPoint,
};

pub use ashrae_140_cases::Orientation;
pub use ashrae_140_cases::{
    ASHRAE140Case, CaseBuilder, CaseSpec, ConstructionSpec, ConstructionType, GeometrySpec,
    HvacSchedule, InternalLoads, NightVentilation, ShadingDevice, ShadingType, WindowArea,
};
pub use benchmark::{get_all_benchmark_data, get_all_case_ids, get_benchmark_data};
pub use report::{
    BenchmarkData, BenchmarkReport, MetricType, ReferenceProgram, ValidationResult,
    ValidationStatus, ValidationSuite,
};

#[cfg(test)]
mod tests {
    use super::ashrae_140_validator::ASHRAE140Validator;

    #[test]
    fn test_ashrae_140_validation() {
        let mut validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();
        report.print_summary();

        // Check for Case 600
        assert!(report.results.iter().any(|r| r.case_id == "600"));

        // Ensure MAE is calculated
        assert!(!report.mae().is_nan());
    }
}

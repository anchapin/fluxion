pub mod analyzer;
pub mod ashrae_140;
pub mod ashrae_140_cases;
pub mod ashrae_140_validator;
pub mod assembly_library;
pub mod benchmark;
pub mod commands;
pub mod cross_validator;
pub mod diagnostic;
pub mod diagnostics;
pub mod export;
pub mod fdd;
pub mod guardrails;
pub mod multi_reference;
pub mod physics_validator;
pub mod report;
pub mod reporter;

// Re-export common types
pub use analyzer::{Analyzer, AnalyzerConfig, AnalyzerError, QualityMetrics};
pub use ashrae_140_validator::{validate_case_with_diagnostics, ASHRAE140Validator};
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
pub use commands::update_references;
pub use diagnostic::{
    ComparisonRow, DiagnosticCollector, DiagnosticConfig, DiagnosticReport, EnergyBreakdown,
    HourlyData, PeakTiming, TemperatureProfile,
};
pub use physics_validator::{
    generate_validation_report, PhysicsValidationResult, PhysicsValidator, TemperatureViolation,
};
pub use report::{
    BenchmarkData, BenchmarkReport, MetricType, ReferenceProgram, ValidationResult,
    ValidationStatus, ValidationSuite,
};
pub use reporter::{SystematicIssue, SystematicIssueMap, ValidationReportGenerator};

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

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
    use std::collections::HashMap;

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

    #[test]
    fn test_multireference_status() {
        use super::multi_reference::{CaseRefs, MultiReferenceDB, ProgramRange};
        use super::report::{BenchmarkReport, ValidationStatus};
        use super::MetricType;

        // Build a minimal multi-reference DB with two programs
        let mut cases = HashMap::new();
        let mut annual_heating = HashMap::new();
        annual_heating.insert(
            "EnergyPlus".to_string(),
            ProgramRange { min: 5.0, max: 5.5 },
        );
        annual_heating.insert("ESP-r".to_string(), ProgramRange { min: 6.0, max: 6.5 });

        let case_refs = CaseRefs {
            annual_heating: annual_heating,
            annual_cooling: HashMap::new(),
            peak_heating: HashMap::new(),
            peak_cooling: HashMap::new(),
        };
        cases.insert("600".to_string(), case_refs);

        let db = MultiReferenceDB {
            version: "test".to_string(),
            source: None,
            cases,
        };

        // Case 1: Fluxion value within EnergyPlus range -> overall PASS
        let mut report1 = BenchmarkReport::new();
        report1.add_result_with_multi("600", MetricType::AnnualHeating, 5.2, &db);
        let res1 = &report1.results[0];
        let per1 = res1.per_program.as_ref().unwrap();
        assert_eq!(per1["EnergyPlus"], ValidationStatus::Pass);
        assert_eq!(per1["ESP-r"], ValidationStatus::Fail);
        assert_eq!(res1.status, ValidationStatus::Pass);

        // Case 2: Fluxion within ESP-r but outside EnergyPlus -> overall WARN
        let mut report2 = BenchmarkReport::new();
        report2.add_result_with_multi("600", MetricType::AnnualHeating, 6.2, &db);
        let res2 = &report2.results[0];
        let per2 = res2.per_program.as_ref().unwrap();
        assert_eq!(per2["EnergyPlus"], ValidationStatus::Fail);
        assert_eq!(per2["ESP-r"], ValidationStatus::Pass);
        assert_eq!(res2.status, ValidationStatus::Warning);

        // Case 3: Fluxion outside all programs -> overall FAIL
        let mut report3 = BenchmarkReport::new();
        report3.add_result_with_multi("600", MetricType::AnnualHeating, 4.0, &db);
        let res3 = &report3.results[0];
        let per3 = res3.per_program.as_ref().unwrap();
        assert_eq!(per3["EnergyPlus"], ValidationStatus::Fail);
        assert_eq!(per3["ESP-r"], ValidationStatus::Fail);
        assert_eq!(res3.status, ValidationStatus::Fail);
    }
}

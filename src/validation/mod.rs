pub mod ab_testing;
pub mod analyzer;
pub mod ashrae_140;
pub mod ashrae_140_cases;
pub mod ashrae_140_validator;
pub mod assembly_library;
pub mod benchmark;
pub mod commands;
pub mod config;
pub mod cross_validator;
pub mod diagnostic;
pub mod diagnostics;
pub mod ep_oracle;
pub mod export;
pub mod fdd;
pub mod guardrails;
pub mod multi_reference;

pub mod physics_validator;
pub mod report;
pub mod reporter;

pub mod statistical;
pub mod thermal_mass;
pub mod thermal_mass_energy_accounting;

// Re-export common types
pub use ab_testing::{ABTestRunner, ComparisonReport, TestResults, ThermalNetworkVariant};
pub use analyzer::{Analyzer, AnalyzerConfig, AnalyzerError, QualityMetrics};
pub use ashrae_140_validator::{validate_case_with_diagnostics, ASHRAE140Validator};
pub use config::{validate_assembly, validate_constants, ConfigValidationResult, ValidationError};
pub use cross_validator::{
    AnalyticalComparison, CrossValidationResult, CrossValidator, CrossValidatorConfig,
    EnergyBalanceMetrics, FoldResult, ValidationDataPoint,
};
pub use ep_oracle::{
    EPOracle, EPReference, FluxionResults, ValidationCriteria, ValidationDetails, ValidationReport,
    DEFAULT_MAX_ABS_ERROR, DEFAULT_MAX_RMSE,
};
pub use statistical::{
    calculate_ci_cv_rmse, calculate_ci_nmbe, calculate_cohens_d, calculate_cv_rmse, calculate_nmbe,
    calculate_standard_error, validate_group_80_percent, validate_group_hybrid,
    validate_group_single_case, validate_groups, BenjaminiHochberg, EffectDirection,
    StatisticalMetrics, StatisticalReport, StatisticalValidator, ValidationGroup,
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
pub use thermal_mass_energy_accounting::{
    calculate_mass_energy, validate_energy_balance_over_year, EnergyBalanceReport,
};

#[cfg(test)]
mod tests {
    use super::ashrae_140_validator::ASHRAE140Validator;
    use std::collections::HashMap;

    #[test]
    fn test_ashrae_140_validation() {
        let validator = ASHRAE140Validator::new();
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

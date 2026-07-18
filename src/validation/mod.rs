pub mod ab_testing;
pub mod analyzer;
pub mod ashrae140;
pub mod ashrae_140;
pub mod ashrae_140_cases;
pub mod ashrae_140_validator;
pub mod assembly_library;
pub mod automation;
pub mod benchmark;
pub mod commands;
pub mod config;
pub mod copilot;
pub mod cross_validation;
pub mod cross_validator;
pub mod diagnostic;
pub mod diagnostics;
pub mod ep_oracle;
pub mod export;
pub mod fdd;
pub mod flexlab_test_cell;
pub mod guardrails;
pub mod hvac_bestest;
pub mod issue_classifier;
pub mod multi_reference;
pub mod performance;
pub mod reference;
pub mod reference_data;
pub mod reference_loader;
pub mod reporting;
pub mod tolerance;

pub mod physics_validator;
pub mod report;
pub mod reporter;

pub mod adaptive_calibration;
pub mod ashrae_140_multi_zone;
pub mod case_195_calibration;
pub mod case_960;
pub mod empirical;
pub mod energy_balance;
pub mod high_mass;
pub mod statistical;
pub mod thermal_mass;
pub mod thermal_mass_energy_accounting;

// Re-export common types
pub use ab_testing::{ABTestRunner, ComparisonReport, TestResults, ThermalNetworkVariant};
pub use analyzer::{Analyzer, AnalyzerConfig, AnalyzerError, QualityMetrics};
pub use ashrae_140_validator::{
    validate_ashrae_140, validate_case_with_diagnostics, ASHRAE140Validator,
    FreeFloatValidationResult,
};
pub use config::{validate_assembly, validate_constants, ConfigValidationResult, ValidationError};
pub use cross_validation::adapters::{EnergyPlusAdapter, TRNSYSAdapter};
pub use cross_validation::{CrossValidationAdapter, ValidationResults};
pub use cross_validator::{
    AnalyticalComparison, CrossValidationResult, CrossValidator, CrossValidatorConfig,
    EnergyBalanceMetrics, FoldResult, ValidationDataPoint,
};
pub use ep_oracle::{
    EPOracle, EPReference, FluxionResults, ValidationCriteria, ValidationDetails, ValidationReport,
    DEFAULT_MAX_ABS_ERROR, DEFAULT_MAX_RMSE,
};
pub use performance::{
    CiPerformanceReport, CiPerformanceValidator, ComparativeAnalysis, ComparativeAnalyzer,
    ConfigurationResult, PerformanceDelta, PerformanceMetrics, Phase47CompletionValidator,
    PhaseCompletionReport, PhaseCompletionResult, RequirementResult,
};
pub use performance::{
    IntegratedPerformanceValidator, IntegratedReport, IntegratedValidationResult,
};
pub use statistical::{
    calculate_ci_cv_rmse, calculate_ci_nmbe, calculate_cohens_d, calculate_cv_rmse, calculate_nmbe,
    calculate_standard_error, validate_group_80_percent, validate_group_hybrid,
    validate_group_single_case, validate_groups, BenjaminiHochberg, EffectDirection,
    StatisticalMetrics, StatisticalReport, StatisticalValidator, ValidationGroup,
};

pub use adaptive_calibration::{
    AdaptiveCalibrationResult, AdaptiveHourlyCalibrator, BiasPattern, CalibrationIteration,
    CalibrationState, CalibrationTrigger, HourlyObservation, SmartMeterPatternAnalyzer,
    TriggerDetector,
};
pub use ashrae140::cases::build_case;
pub use ashrae140::ASHRAE140Case;
pub use ashrae140::ASHRAE140CaseDefinition;
pub use ashrae_140_cases::Orientation;
pub use ashrae_140_cases::{
    CaseBuilder, CaseSpec, ConstructionSpec, ConstructionType, GeometrySpec, HvacSchedule,
    InternalLoads, NightVentilation, ShadingDevice, ShadingType, WindowArea,
};
pub use ashrae_140_multi_zone::{ASHRAE140MultiZoneValidator, Case960Reference};
pub use benchmark::{get_all_benchmark_data, get_all_case_ids, get_benchmark_data};
pub use case_195_calibration::{
    run_case_195_calibration, CalibrationParameters, CalibrationResult, Case195Calibrator,
};
pub use case_960::{
    run_complete_case_960_validation, Case960ReferenceImplementation, Case960Result,
};
pub use commands::update_references;
pub use diagnostic::{
    ComparisonRow, DiagnosticCollector, DiagnosticConfig, DiagnosticReport, EnergyBreakdown,
    HourlyData, PeakTiming, TemperatureProfile,
};
pub use energy_balance::EnergyBalanceValidator;
pub use high_mass::{
    generate_combined_report, run_all_high_mass_cases, validate_construction_type,
};
pub use high_mass::{
    CombinedHighMassReport, HighMassMetrics, HighMassSummary, HighMassValidationCase,
    HighMassValidationReport,
};
pub use hvac_bestest::{
    run_hvac_bestest, validate_results, EquipmentType, HVACBestestCase, HVACBestestCaseDefinition,
    HVACBestestResult, HVACBestestRunner, OperatingMode,
};
pub use physics_validator::{
    generate_validation_report, PhysicsValidationResult, PhysicsValidator, TemperatureViolation,
};
pub use reference::{
    load_reference_data, load_series_195_reference, load_series_800_reference, HourlyDataPoint,
    ReferenceDataError, ReferenceDataset,
};
pub use reference_data::{
    calculate_mbe, calculate_percentage_difference, calculate_rmse, load_case_960_reference,
    load_case_970_reference, load_csv_reference, load_multi_zone_reference, parse_hourly_data,
    within_tolerance, ReferenceData,
};
pub use report::{
    BenchmarkReport, Interpretation, MetricType, ReferenceProgram, ReportHeader, ValidationResult,
    ValidationStatus, ValidationSuite,
};

// Re-export copilot types
pub use copilot::{
    BemChecker, BemIssue, BemIssueSeverity, Copilot, CopilotConfig, CopilotResult,
    ValidationChecks, OLLAMA_DEFAULT_URL,
};

/// Validation configuration for different validation scenarios
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Validation mode (standard, ashrae140, etc.)
    pub mode: ValidationMode,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone)]
pub enum ValidationMode {
    Standard,
    ASHRAE140(u32), // Case number for ASHRAE 140 validation
    PerformanceOnly,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_timestep_duration_ms: f64,
    pub max_memory_usage_bytes: usize,
}

impl ValidationConfig {
    /// Create standard validation configuration
    pub fn standard() -> Self {
        Self {
            mode: ValidationMode::Standard,
            performance_thresholds: PerformanceThresholds {
                max_timestep_duration_ms: 50.0,
                max_memory_usage_bytes: 10_000_000,
            },
        }
    }

    /// Create ASHRAE 140 validation configuration
    pub fn ashrae140(case_number: u32) -> Self {
        Self {
            mode: ValidationMode::ASHRAE140(case_number),
            performance_thresholds: PerformanceThresholds {
                max_timestep_duration_ms: 100.0, // More lenient for ASHRAE 140
                max_memory_usage_bytes: 20_000_000,
            },
        }
    }
}

use serde::{Deserialize, Serialize};

/// Comparison metrics for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub rmse: f64,
    pub percentage_difference: f64,
    pub max_deviation: f64,
    pub within_tolerance: bool,
}

impl Default for ComparisonMetrics {
    fn default() -> Self {
        Self {
            rmse: 0.0,
            percentage_difference: 0.0,
            max_deviation: 0.0,
            within_tolerance: true,
        }
    }
}
pub use reporter::{SystematicIssue, SystematicIssueMap, ValidationReportGenerator};
pub use thermal_mass_energy_accounting::{
    calculate_mass_energy, validate_energy_balance_over_year, BuildingBalanceSummary,
    EnergyBalanceReport, ZoneBalanceEntry,
};

// Empirical validation re-exports
pub use empirical::{
    generate_empirical_report, get_ashrae_rp_sources, BuildingType, EmpiricalMetric,
    EmpiricalStatistics, EmpiricalValidationConfig, EmpiricalValidationReport,
    EmpiricalValidationResult, EmpiricalValidationStatus, MonitoredBuildingDatabase,
    MonitoredDataPoint, MonitoredDataSource,
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
            annual_heating: Some(annual_heating),
            annual_cooling: Some(HashMap::new()),
            peak_heating: Some(HashMap::new()),
            peak_cooling: Some(HashMap::new()),
            min_free_float: None,
            max_free_float: None,
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

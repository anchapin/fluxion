pub mod ci;
pub mod comparative;
pub mod completion;
pub mod finalization;
pub mod historical;
pub mod integration;
pub mod metrics;
pub mod optimization;
pub mod parallel_executor;
pub mod profiling;
pub mod reports;

use crate::thermal::thermal_model::ThermalModel;

pub use ci::{CiPerformanceReport, CiPerformanceValidator};
pub use comparative::{
    ComparativeAnalysis, ComparativeAnalyzer, ConfigurationResult, PerformanceDelta,
};
pub use completion::{
    Phase47CompletionValidator, PhaseCompletionReport, PhaseCompletionResult, RequirementResult,
};
pub use finalization::{
    ComparativeAnalysisResult, FinalPerformanceReport, FinalValidationResult,
    PerformanceValidationFinalizer,
};
pub use historical::{
    BenchmarkHistory, HistoricalPerformanceReport, HistoricalRecord, HistoricalTracker,
    PerformanceTrend, TrendDirection,
};
pub use integration::{
    IntegratedPerformanceValidator, IntegratedReport, IntegratedValidationResult,
};
pub use metrics::PerformanceMetrics;
pub use optimization::{
    generate_optimization_report, SolverOptimization, ZoneCouplingOptimization,
};
pub use parallel_executor::ParallelValidationExecutor;
pub use profiling::{
    analyze_bottlenecks, generate_detailed_performance_report, generate_performance_report,
    log_performance_metrics, profile_case,
};
pub use reports::PerformanceReport;

pub struct PerformanceValidator {
    model: ThermalModel,
}

impl PerformanceValidator {
    pub fn new(model: ThermalModel) -> Self {
        Self { model }
    }

    pub fn validate_performance(&self) -> PerformanceReport {
        let metrics = metrics::collect_performance_metrics(&self.model);
        reports::generate_performance_report(metrics)
    }
}

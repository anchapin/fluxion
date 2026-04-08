pub mod comparative;
pub mod finalization;
pub mod historical;
pub mod integration;
pub mod metrics;
pub mod optimization;
pub mod reports;

use crate::thermal::thermal_model::ThermalModel;

pub use comparative::{
    ComparativeAnalysis, ComparativeAnalyzer, ConfigurationResult, PerformanceDelta,
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

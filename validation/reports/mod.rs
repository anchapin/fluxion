// validation/reports/mod.rs
/// Reports module
///
/// This module provides reporting functionality for validation results
pub mod cross_validation;

// Re-export key types for easy access
pub use cross_validation::generate_markdown_report;
pub use cross_validation::generate_report;
pub use cross_validation::CrossValidationReport;
pub use cross_validation::SummaryStatistics;

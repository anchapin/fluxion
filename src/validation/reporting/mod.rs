// validation/reporting/mod.rs
/// Validation reporting module
///
/// This module provides reporting capabilities for validation results
/// including CLI commands, report generation, and examples.
/// CLI commands for reporting
pub mod cli;

/// Examples for reporting functionality
pub mod examples;

/// Report generator
pub mod generator;

/// Aggregation module for monthly/daily reporting
pub mod aggregation;

// Re-export key types
pub use cli::ReportingCommand;
pub use generator::ValidationReportGenerator;

// Re-export aggregation types
pub use aggregation::{
    hourly_to_daily, hourly_to_monthly, verify_monthly_aggregation, CustomReportFilter,
    DailyAggregation, MonthlyAggregation, SqlReporter,
};

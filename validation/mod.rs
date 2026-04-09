// validation/mod.rs
/// Validation module
///
/// This module provides validation infrastructure for Fluxion
pub mod esp_r;
pub mod reports;
pub mod automation;

/// ASHRAE 140 validation module
pub mod ashrae140;

/// Climate zone validation module
pub mod climate;

/// Occupancy pattern validation module
pub mod occupancy;

/// Comprehensive reporting module
pub mod reporting;

/// Framework integration module
pub mod integration;

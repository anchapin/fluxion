//! HVAC BESTEST Validation Module
//!
//! ASHRAE RP-865 HVAC BESTEST for airside equipment, controls, and part-load performance validation.
//!
//! # Overview
//!
//! This module implements the HVAC BESTEST cases from ASHRAE RP-865 to validate:
//! - Equipment energy consumption within acceptable tolerance
//! - Zone temperature maintenance
//! - Part-load performance ratios
//! - Control strategy performance
//!
//! # Test Cases
//!
//! - `Case600`: Chiller part-load performance
//! - `Case610`: Boiler part-load performance
//! - `Case620`: Heat pump performance
//! - `Case630`: VAV system performance
//! - `Case640`: CAV system performance
//!
//! # Usage
//!
//! ```rust
//! use fluxion::validation::hvac_bestest::{run_hvac_bestest, validate_results};
//!
//! // Run all HVAC BESTEST cases
//! let results = run_hvac_bestest();
//!
//! // Validate results
//! let (passed, failed, mean_error) = validate_results(&results);
//! println!("Passed: {}, Failed: {}, Mean Error: {:.2}%", passed, failed, mean_error);
//! ```

pub mod cases;
pub mod reporting;
pub mod runner;

pub use cases::{
    get_bestest_cases, get_reference_data, EquipmentType, HVACBestestCase,
    HVACBestestCaseDefinition, HVACBestestReferenceData, OperatingMode,
};
pub use reporting::{
    assert_within_bounds, assert_within_bounds_full, check_within_bounds, BoundStatus,
    CaseMetricReport, HvacBestestReport, HvacBestestToleranceConfig, ReportSummary, ToleranceCheck,
    REFERENCE_ZERO_EPSILON,
};
pub use runner::{run_hvac_bestest, validate_results, HVACBestestResult, HVACBestestRunner};

/// Module version
pub const VERSION: &str = "1.0.0";

/// Tolerance for validation (%)
pub const DEFAULT_TOLERANCE: f64 = 10.0;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_module_integration() {
        let results = run_hvac_bestest();
        assert_eq!(results.len(), 5);

        for result in &results {
            println!(
                "{:?}: Energy={:.1} kWh, Peak={:.0} W, Error={:.1}%",
                result.case_id,
                result.annual_energy_kwh,
                result.peak_demand_w,
                result.energy_error_percent
            );
        }
    }

    #[test]
    fn test_case_coverage() {
        let cases = get_bestest_cases();
        let mut case_ids: Vec<HVACBestestCase> = cases.iter().map(|c| c.case_id).collect();
        case_ids.sort();
        case_ids.dedup();

        assert_eq!(case_ids.len(), 5);
    }

    #[test]
    fn test_reference_data_coverage() {
        let cases = get_bestest_cases();
        for case in cases {
            let ref_data = get_reference_data(case.case_id);
            assert!(
                ref_data.is_some(),
                "Missing reference data for {:?}",
                case.case_id
            );
        }
    }
}

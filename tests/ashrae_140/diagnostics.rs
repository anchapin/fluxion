//! Consolidated validation logic for ASHRAE 140 diagnostic cases
//!
//! This module provides helper functions and shared validation logic for diagnostic
//! case testing, including Cases 195-470 (in-depth diagnostics), Cases 800-810
//! (HVAC equipment), and diagnostic variants (solid conduction, solar gain).
//!
//! # Design Philosophy
//!
//! The consolidated validation pattern centralizes common validation logic while
//! keeping case-specific implementations in separate test files. This reduces
//! duplication and makes it easier to maintain consistent validation behavior
//! across all diagnostic case ranges.
//!
//! # Usage
//!
//! ```rust,no_run
//! use tests::ashrae_140::diagnostics;
//!
//! // Validate a range of cases
//! let result = diagnostics::validate_diagnostic_range(195, 470, &validator);
//! println!("Passed: {}/{}", result.passed, result.total_cases);
//!
//! // Run pre-configured diagnostic suites
//! let result_195_470 = diagnostics::run_cases_195_470();
//! let result_800_810 = diagnostics::run_cases_800_810();
//! ```
//!
//! # Integration with ASHRAE140Validator
//!
//! This module uses the existing ASHRAE140Validator framework from
//! src/validation/ashrae_140_validator.rs, which provides:
//! - Standardized tolerance checking (±15% annual, ±10% monthly, ±1°C free-float)
//! - Multi-reference comparison (EnergyPlus, ESP-r, TRNSYS)
//! - Pass/warning/fail criteria with detailed diagnostic output

use fluxion::validation::ashrae_140_validator::{ASHRAE140Validator, ValidationResult};

/// Result summary for a range of diagnostic cases
///
/// This struct provides a consolidated view of validation results across multiple
/// diagnostic cases, making it easy to assess overall compliance and identify
/// specific cases that need attention.
#[derive(Debug, Clone)]
pub struct DiagnosticRangeResult {
    /// Case range identifier (e.g., "195-470", "800-810")
    pub range: String,
    /// Total number of cases validated
    pub total_cases: usize,
    /// Number of cases that passed validation
    pub passed: usize,
    /// Detailed results for each case (case_id, validation_result)
    pub results: Vec<(String, ValidationResult)>,
}

impl DiagnosticRangeResult {
    /// Calculates the pass rate as a percentage (0.0-100.0)
    pub fn pass_rate(&self) -> f64 {
        if self.total_cases == 0 {
            0.0
        } else {
            (self.passed as f64 / self.total_cases as f64) * 100.0
        }
    }

    /// Returns true if all cases passed validation
    pub fn all_passed(&self) -> bool {
        self.passed == self.total_cases
    }

    /// Returns a list of case IDs that failed validation
    pub fn failed_cases(&self) -> Vec<String> {
        self.results
            .iter()
            .filter(|(_, result)| !result.in_range)
            .map(|(case_id, _)| case_id.clone())
            .collect()
    }
}

/// Validates a range of diagnostic cases and returns summary results
///
/// This generic function iterates through a case range, validates each case
/// using the provided ASHRAE140Validator, and returns a consolidated summary.
///
/// # Arguments
///
/// * `start` - Starting case number (inclusive)
/// * `end` - Ending case number (inclusive)
/// * `validator` - ASHRAE140Validator instance to use for validation
///
/// # Returns
///
/// DiagnosticRangeResult containing validation summary and detailed results
///
/// # Error Handling
///
/// Individual case validation errors are caught and logged as warnings but do
/// not stop processing of remaining cases. This ensures that we get a complete
/// picture even if some cases have issues.
///
/// # Example
///
/// ```rust,no_run
/// use fluxion::validation::ashrae_140_validator::ASHRAE140Validator;
/// use tests::ashrae_140::diagnostics::validate_diagnostic_range;
///
/// let mut validator = ASHRAE140Validator::new();
/// let result = validate_diagnostic_range(195, 470, &validator);
/// println!("Pass rate: {:.1}%", result.pass_rate());
/// ```
pub fn validate_diagnostic_range(
    start: u32,
    end: u32,
    validator: &ASHRAE140Validator,
) -> DiagnosticRangeResult {
    let mut results = Vec::new();

    for case_num in start..=end {
        let case_id = case_num.to_string();

        // Validate the case - handle errors gracefully
        let validation_result = match validator.validate_case(&case_id) {
            Ok(result) => result,
            Err(e) => {
                // Log warning but continue processing
                eprintln!("Warning: Failed to validate case {}: {}", case_id, e);
                // Create a failed validation result as placeholder
                ValidationResult {
                    in_range: false,
                    error_pct: f64::NAN,
                }
            }
        };

        results.push((case_id, validation_result));
    }

    let passed = results.iter().filter(|(_, r)| r.in_range).count();

    DiagnosticRangeResult {
        range: format!("{}-{}", start, end),
        total_cases: results.len(),
        passed,
        results,
    }
}

/// Runs ASHRAE 140 Cases 195-470 diagnostic suite
///
/// Cases 195-470 are in-depth diagnostics for testing specific components:
/// - Lighting diagnostics (heat gain, scheduling)
/// - Equipment diagnostics (power consumption, thermal output)
/// - Thermal mass behavior (heat storage, thermal inertia)
/// - Internal load variations (different schedules, occupancies)
///
/// # Returns
///
/// DiagnosticRangeResult with summary and detailed results
///
/// # Note
///
/// This is a Wave 0 implementation that validates the framework. Full case
/// specifications will be implemented in Plan 18-02 after ASHRAE 140 reference
/// data is available.
pub fn run_cases_195_470() -> DiagnosticRangeResult {
    let validator = ASHRAE140Validator::new();
    validate_diagnostic_range(195, 470, &validator)
}

/// Runs ASHRAE 140 Cases 800-810 HVAC equipment diagnostic suite
///
/// Cases 800-810 validate HVAC equipment performance and control strategies:
/// - Heat pump systems (Case 800-809)
/// - Chiller plant systems (Case 810)
/// - Equipment efficiency curves (polynomial degradation at part-load)
/// - Cycling losses (startup penalties, minimum runtime)
/// - Predictive control (thermal inertia, dT/dt-based control)
///
/// # Returns
///
/// DiagnosticRangeResult with summary and detailed results
///
/// # Note
///
/// This is a Wave 0 implementation that validates the framework. Full case
/// specifications will be implemented in Plan 18-03 after ASHRAE 140 reference
/// data is available.
pub fn run_cases_800_810() -> DiagnosticRangeResult {
    let validator = ASHRAE140Validator::new();
    validate_diagnostic_range(800, 810, &validator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_diagnostic_range_empty() {
        let validator = ASHRAE140Validator::new();
        let result = validate_diagnostic_range(1, 0, &validator);

        assert_eq!(result.total_cases, 0);
        assert_eq!(result.passed, 0);
        assert_eq!(result.pass_rate(), 0.0);
        assert!(results.all_passed());
    }

    #[test]
    fn test_diagnostic_range_result_pass_rate() {
        let result = DiagnosticRangeResult {
            range: "195-470".to_string(),
            total_cases: 10,
            passed: 8,
            results: vec![],
        };

        assert_eq!(result.pass_rate(), 80.0);
        assert!(!result.all_passed());
        assert_eq!(result.failed_cases().len(), 2);
    }

    #[test]
    fn test_diagnostic_range_result_all_passed() {
        let result = DiagnosticRangeResult {
            range: "800-810".to_string(),
            total_cases: 5,
            passed: 5,
            results: vec![],
        };

        assert_eq!(result.pass_rate(), 100.0);
        assert!(result.all_passed());
        assert!(result.failed_cases().is_empty());
    }

    #[test]
    fn test_diagnostic_range_result_failed_cases() {
        let results = vec![
            (
                "195".to_string(),
                ValidationResult {
                    in_range: true,
                    error_pct: 1.0,
                },
            ),
            (
                "196".to_string(),
                ValidationResult {
                    in_range: false,
                    error_pct: 20.0,
                },
            ),
            (
                "197".to_string(),
                ValidationResult {
                    in_range: true,
                    error_pct: 0.5,
                },
            ),
            (
                "198".to_string(),
                ValidationResult {
                    in_range: false,
                    error_pct: 25.0,
                },
            ),
        ];

        let result = DiagnosticRangeResult {
            range: "195-198".to_string(),
            total_cases: 4,
            passed: 2,
            results,
        };

        let failed = result.failed_cases();
        assert_eq!(failed.len(), 2);
        assert!(failed.contains(&"196".to_string()));
        assert!(failed.contains(&"198".to_string()));
    }

    /// Integration test for Cases 800-810 HVAC equipment cases
    ///
    /// Runs all 11 HVAC equipment cases (800-810) and validates that
    /// at least 80% pass the ASHRAE 140 acceptance criteria.
    /// This test validates the full implementation of Task 2-4:
    /// - CaseBuilder methods (Task 2)
    /// - Reference data (Task 3)
    /// - Test implementations (Task 4)
    #[test]
    fn test_cases_800_810_integration() {
        let results = run_cases_800_810();

        // Validate that at least 80% of cases pass
        assert!(
            results.passed >= results.total_cases * 0.8,
            "Cases 800-810 pass rate {:.1}% ({}/{}) below 80% threshold",
            results.pass_rate() * 100.0,
            results.passed,
            results.total_cases
        );

        println!(
            "Cases 800-810: {}/{} passed ({:.1}%)",
            results.passed,
            results.total_cases,
            results.pass_rate() * 100.0
        );

        // Validate that we tested all 11 HVAC equipment cases
        assert_eq!(
            results.total_cases, 11,
            "Expected 11 cases (800-810), got {}",
            results.total_cases
        );

        // Check for specific case failures
        let failed = results.failed_cases();
        if !failed.is_empty() {
            println!("Failed cases: {:?}", failed);
        }
    }
}

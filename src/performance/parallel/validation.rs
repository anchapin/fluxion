//! Parallel validation pipeline using Rayon for concurrent case execution

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Validation result for a single case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Case identifier
    pub case_id: String,
    /// Whether the case passed validation
    pub passed: bool,
    /// NMBE (Normalized Mean Bias Error) percentage
    pub nmbe: f64,
    /// CV(RMSE) (Coefficient of Variation of Root Mean Square Error) percentage
    pub cv_rmse: f64,
    /// Maximum deviation from reference
    pub max_deviation: f64,
}

impl Default for ValidationResult {
    fn default() -> Self {
        ValidationResult {
            case_id: String::new(),
            passed: false,
            nmbe: 0.0,
            cv_rmse: 0.0,
            max_deviation: 0.0,
        }
    }
}

/// High-mass validation case with reference loads and expected results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighMassCase {
    /// Case identifier
    pub case_id: String,
    /// Reference loads from ASHRAE 140 reference data
    pub reference_loads: Vec<f64>,
    /// Simulation results to validate
    pub simulation_results: Vec<f64>,
    /// Tolerance for validation (default 0.15 = 15%)
    pub tolerance: f64,
}

impl Default for HighMassCase {
    fn default() -> Self {
        HighMassCase {
            case_id: String::new(),
            reference_loads: vec![0.0; 8760], // Full year hourly
            simulation_results: vec![0.0; 8760],
            tolerance: 0.15,
        }
    }
}

/// Run parallel validation across multiple high-mass cases
///
/// This function demonstrates parallel case execution using Rayon's par_iter.
/// Each case is validated independently in parallel, leveraging all available CPU cores.
///
/// # Arguments
/// * `cases` - Vector of high-mass validation cases
///
/// # Returns
/// Vector of validation results in the same order as input cases
pub fn run_parallel_validation(cases: Vec<HighMassCase>) -> Vec<ValidationResult> {
    cases
        .into_par_iter()
        .map(|case| validate_single_case(case))
        .collect()
}

/// Run parallel validation with configurable chunk size
///
/// This version allows tuning the minimum chunk size for optimal performance
/// on different hardware configurations.
///
/// # Arguments
/// * `cases` - Vector of high-mass validation cases
/// * `chunk_size` - Minimum number of items per chunk (0 = auto)
pub fn run_parallel_validation_chunked(
    cases: Vec<HighMassCase>,
    chunk_size: usize,
) -> Vec<ValidationResult> {
    cases
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|case| validate_single_case(case))
        .collect()
}

/// Validate a single high-mass case against reference data
fn validate_single_case(case: HighMassCase) -> ValidationResult {
    let reference = &case.reference_loads;
    let results = &case.simulation_results;

    if reference.is_empty() || results.is_empty() {
        return ValidationResult {
            case_id: case.case_id,
            passed: false,
            nmbe: 0.0,
            cv_rmse: 0.0,
            max_deviation: 0.0,
        };
    }

    // Calculate NMBE (Normalized Mean Bias Error)
    let sum_diff: f64 = results
        .iter()
        .zip(reference.iter())
        .map(|(r, ref_)| r - ref_)
        .sum();
    let mean_ref = reference.iter().sum::<f64>() / reference.len() as f64;
    let nmbe = if mean_ref.abs() > 1e-10 {
        (sum_diff / reference.len() as f64) / mean_ref * 100.0
    } else {
        0.0
    };

    // Calculate CV(RMSE)
    let sum_sq_diff: f64 = results
        .iter()
        .zip(reference.iter())
        .map(|(r, ref_)| (r - ref_).powi(2))
        .sum();
    let rmse = (sum_sq_diff / results.len() as f64).sqrt();
    let cv_rmse = if mean_ref.abs() > 1e-10 {
        (rmse / mean_ref) * 100.0
    } else {
        0.0
    };

    // Calculate maximum deviation
    let max_deviation: f64 = results
        .iter()
        .zip(reference.iter())
        .map(|(r, ref_)| (r - ref_).abs())
        .fold(0.0_f64, |a, b| a.max(b));

    // Determine pass/fail based on ASHRAE 140 tolerance bands
    let tolerance_pct = case.tolerance * 100.0;
    let passed = nmbe.abs() <= tolerance_pct && cv_rmse <= tolerance_pct;

    ValidationResult {
        case_id: case.case_id,
        passed,
        nmbe,
        cv_rmse,
        max_deviation,
    }
}

/// Run parallel validation with timing measurement
///
/// Returns both the validation results and the execution time in seconds.
///
/// # Arguments
/// * `cases` - Vector of high-mass validation cases
///
/// # Returns
/// Tuple of (validation_results, execution_duration_seconds)
pub fn validate_with_timing(cases: Vec<HighMassCase>) -> (Vec<ValidationResult>, f64) {
    let start = std::time::Instant::now();
    let results = run_parallel_validation(cases);
    let duration = start.elapsed().as_secs_f64();
    (results, duration)
}

/// Compare parallel vs sequential validation performance
///
/// Runs both paths and returns timing information for each.
///
/// # Arguments
/// * `cases` - Vector of high-mass validation cases
///
/// # Returns
/// Tuple of (parallel_duration, sequential_duration)
pub fn compare_validation_performance(cases: Vec<HighMassCase>) -> (f64, f64) {
    // Parallel timing
    let start_parallel = std::time::Instant::now();
    let _parallel_results = run_parallel_validation(cases.clone());
    let parallel_duration = start_parallel.elapsed().as_secs_f64();

    // Sequential timing (for comparison)
    let start_sequential = std::time::Instant::now();
    let _sequential_results: Vec<ValidationResult> = cases
        .iter()
        .map(|case| validate_single_case(case.clone()))
        .collect();
    let sequential_duration = start_sequential.elapsed().as_secs_f64();

    (parallel_duration, sequential_duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_validation_basic() {
        let cases = vec![
            HighMassCase {
                case_id: "L002".to_string(),
                reference_loads: vec![100.0; 8760],
                simulation_results: vec![105.0; 8760],
                tolerance: 0.15,
            },
            HighMassCase {
                case_id: "L004".to_string(),
                reference_loads: vec![200.0; 8760],
                simulation_results: vec![195.0; 8760],
                tolerance: 0.15,
            },
        ];

        let results = run_parallel_validation(cases);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].case_id, "L002");
        assert_eq!(results[1].case_id, "L004");
    }

    #[test]
    fn test_parallel_validation_chunked() {
        let cases = vec![
            HighMassCase {
                case_id: "L002".to_string(),
                reference_loads: vec![100.0; 8760],
                simulation_results: vec![105.0; 8760],
                tolerance: 0.15,
            };
            10
        ];

        let results = run_parallel_validation_chunked(cases, 2);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_validation_with_timing() {
        let cases = vec![
            HighMassCase {
                case_id: "L002".to_string(),
                reference_loads: vec![100.0; 8760],
                simulation_results: vec![105.0; 8760],
                tolerance: 0.15,
            };
            5
        ];

        let (results, duration) = validate_with_timing(cases);
        assert_eq!(results.len(), 5);
        assert!(duration >= 0.0);
    }

    #[test]
    fn test_validation_tolerance() {
        // Case that should pass
        let case_pass = HighMassCase {
            case_id: "PASS".to_string(),
            reference_loads: vec![100.0; 100],
            simulation_results: vec![105.0; 100], // 5% difference
            tolerance: 0.15,                      // 15% tolerance
        };

        // Case that should fail
        let case_fail = HighMassCase {
            case_id: "FAIL".to_string(),
            reference_loads: vec![100.0; 100],
            simulation_results: vec![200.0; 100], // 100% difference
            tolerance: 0.15,
        };

        let result_pass = validate_single_case(case_pass);
        let result_fail = validate_single_case(case_fail);

        assert!(result_pass.passed);
        assert!(!result_fail.passed);
    }
}

use crate::BatchOracle;
use anyhow::Result;
use csv::{Reader, Writer};
use rayon::prelude::*;
use serde::Deserialize;
use std::path::Path;

/// Parameter range for sensitivity analysis
#[derive(Debug, Clone, Deserialize)]
pub struct ParameterRange {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

/// Generate an One-Factor-At-a-Time (OAT) design matrix.
///
/// For each parameter, `levels` samples are taken where that parameter varies linearly
/// from min to max while all other parameters are held at their baseline (midpoint).
///
/// The returned design matrix has rows = ranges.len() * levels, and columns = ranges.len().
pub fn generate_oat_design(ranges: &[ParameterRange], levels: usize) -> Vec<Vec<f64>> {
    if levels == 0 {
        return Vec::new();
    }
    let n_params = ranges.len();
    let mut design = Vec::with_capacity(n_params * levels);
    for (i, range) in ranges.iter().enumerate() {
        // baseline for other params: midpoint
        let baseline: Vec<f64> = ranges.iter().map(|r| (r.min + r.max) / 2.0).collect();
        for j in 0..levels {
            let t = if levels > 1 {
                j as f64 / (levels - 1) as f64
            } else {
                0.5
            };
            let value = range.min + t * (range.max - range.min);
            let mut row = baseline.clone();
            row[i] = value;
            design.push(row);
        }
    }
    design
}

/// Generate a Sobol quasi-random design matrix.
///
/// Generates `num_samples` parameter vectors using Sobol sequence for good space-filling coverage.
/// Each column is scaled to the corresponding parameter range.
pub fn generate_sobol_design(ranges: &[ParameterRange], num_samples: usize) -> Vec<Vec<f64>> {
    let n_params = ranges.len();
    if n_params == 0 || num_samples == 0 {
        return Vec::new();
    }
    // Note: The `sobol` dependency is included for future proper Sobol sequence generation.
    // For now, we use uniform random sampling as a placeholder that still covers the ranges.
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let mut row = Vec::with_capacity(n_params);
        for range in ranges {
            let u: f64 = rng.gen_range(0.0..1.0);
            row.push(range.min + u * (range.max - range.min));
        }
        samples.push(row);
    }
    samples
}

/// Metric set for a single parameter's sensitivity
#[derive(Debug, Clone)]
pub struct MetricSet {
    pub normalized_coeff: f64,
    pub cvrmse: f64,
    pub nmbe: f64,
    pub slope: f64,
}

/// Sensitivity report containing results for each parameter
#[derive(Debug, Clone)]
pub struct SensitivityReport {
    pub parameters: Vec<String>,
    pub metrics: Vec<MetricSet>,
}

// DESIGN REFACTOR (Task 2 - 07-11):
// This function will be refactored to use BatchOracle for batch evaluation.
// New signature: run_sensitivity(design: &[Vec<f64>], oracle: &BatchOracle, use_surrogates: bool) -> Vec<f64>
// The oracle's base model should be configured for the specific ASHRAE case.
// This aligns with the two-class API (BatchOracle for population, Model for single).
// The internal rayon parallel loop will be replaced by oracle.evaluate().
// Call sites will be updated accordingly.

/// Evaluate a design matrix using a BatchOracle.
///
/// The oracle must be pre-configured with a base model (e.g., from an ASHRAE case).
/// This function forwards the design matrix to the oracle's batch evaluation method.
///
/// # Arguments
///
/// * `design` - Design matrix (each inner Vec is a parameter vector).
/// * `oracle` - BatchOracle instance configured for the specific case.
/// * `use_surrogates` - Whether to use AI surrogates for faster evaluation.
///
/// # Returns
///
/// Vector of EUI values (kWh/m²/yr) for each design point.
pub fn run_sensitivity(
    design: &[Vec<f64>],
    oracle: &BatchOracle,
    use_surrogates: bool,
) -> Vec<f64> {
    oracle
        .evaluate_population(design.to_vec(), use_surrogates)
        .expect("Evaluation failed")
}

/// Compute sensitivity metrics for each parameter in the design matrix.
///
/// The design matrix is assumed to have `k` columns (parameters). For each column `i`,
/// this function extracts the values and performs a simple linear regression against the
/// outputs to compute slope, CVRMSE, NMBE, and a normalized coefficient.
///
/// The resulting report's parameters are named "Parameter 0", "Parameter 1", etc., and
/// the metrics are sorted by descending absolute normalized coefficient.
pub fn compute_metrics(design: &[Vec<f64>], outputs: &[f64]) -> SensitivityReport {
    let n = outputs.len();
    if n == 0 {
        return SensitivityReport {
            parameters: Vec::new(),
            metrics: Vec::new(),
        };
    }
    let n_params = design.get(0).map(|row| row.len()).unwrap_or(0);
    let mut param_metrics: Vec<(String, MetricSet)> = Vec::new();

    let y_mean = outputs.iter().copied().sum::<f64>() / n as f64;
    let y_min = outputs.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = outputs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let epsilon = 1e-10;

    for i in 0..n_params {
        // Extract column i as x
        let mut x = Vec::with_capacity(n);
        for row in design.iter().take(n) {
            if i < row.len() {
                x.push(row[i]);
            } else {
                x.push(0.0);
            }
        }

        // Compute linear regression manually: y = a + b*x
        // Calculate means
        let x_mean = x.iter().sum::<f64>() / n as f64;
        // Covariance and variance
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for j in 0..n {
            let dx = x[j] - x_mean;
            let dy = outputs[j] - y_mean;
            cov += dx * dy;
            var_x += dx * dx;
        }
        let slope = if var_x.abs() > epsilon {
            cov / var_x
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        // Compute residuals
        let residuals: Vec<f64> = outputs
            .iter()
            .zip(x.iter())
            .map(|(&y, &xi)| y - (intercept + slope * xi))
            .collect();

        let mse = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
        let rmse = mse.sqrt();
        let cvrmse = rmse / (y_mean.abs() + epsilon) * 100.0;
        let nmbe = residuals.iter().sum::<f64>() / (n as f64 * (y_mean.abs() + epsilon)) * 100.0;

        let normalized_coeff = (y_max - y_min) / y_mean.abs() * 100.0;

        let metric = MetricSet {
            normalized_coeff,
            cvrmse,
            nmbe,
            slope,
        };
        param_metrics.push((format!("Parameter {}", i), metric));
    }

    // Sort by descending absolute normalized_coeff
    param_metrics.sort_by(|a, b| {
        b.1.normalized_coeff
            .abs()
            .partial_cmp(&a.1.normalized_coeff.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (parameters, metrics): (Vec<String>, Vec<MetricSet>) = param_metrics.into_iter().unzip();

    SensitivityReport {
        parameters,
        metrics,
    }
}

/// Export a sensitivity report to a CSV file.
///
/// The CSV includes headers: Rank,Parameter,NormalizedCoeff,CVRMSE,NMBE,Slope.
/// Rows are in rank order (1-indexed).
pub fn export_to_csv(report: &SensitivityReport, path: &Path) -> Result<()> {
    let mut wtr = Writer::from_path(path)?;
    // Write header
    wtr.write_record(&[
        "Rank",
        "Parameter",
        "NormalizedCoeff",
        "CVRMSE",
        "NMBE",
        "Slope",
    ])?;

    // Collect parameters and metrics into a single vector and sort by normalized_coeff descending
    let mut entries: Vec<(&String, &MetricSet)> = report
        .parameters
        .iter()
        .zip(report.metrics.iter())
        .collect();
    entries.sort_by(|a, b| {
        b.1.normalized_coeff
            .abs()
            .partial_cmp(&a.1.normalized_coeff.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, (param, metric)) in entries.iter().enumerate() {
        wtr.write_record(&[
            (rank + 1).to_string(),
            (*param).clone(),
            metric.normalized_coeff.to_string(),
            metric.cvrmse.to_string(),
            metric.nmbe.to_string(),
            metric.slope.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_oat_generates_correct_matrix() {
        let ranges = vec![
            ParameterRange {
                name: "window_u".to_string(),
                min: 0.1,
                max: 5.0,
            },
            ParameterRange {
                name: "heating_setpoint".to_string(),
                min: 15.0,
                max: 25.0,
            },
        ];
        let levels = 5;
        let design = generate_oat_design(&ranges, levels);
        // Total rows = 2 * 5 = 10
        assert_eq!(design.len(), 10);
        // Each row has 2 columns
        for row in &design {
            assert_eq!(row.len(), 2);
        }
        // Check first block (parameter 0 varies)
        let baseline_heating = (15.0 + 25.0) / 2.0; // 20.0
        for (j, row) in design[0..levels].iter().enumerate() {
            let expected_u =
                ranges[0].min + (j as f64 / (levels - 1) as f64) * (ranges[0].max - ranges[0].min);
            assert!((row[0] - expected_u).abs() < 1e-10);
            assert!((row[1] - baseline_heating).abs() < 1e-10);
        }
        // Check second block (parameter 1 varies)
        let baseline_u = (0.1 + 5.0) / 2.0; // 2.6
        for (j, row) in design[levels..].iter().enumerate() {
            let expected_hp =
                ranges[1].min + (j as f64 / (levels - 1) as f64) * (ranges[1].max - ranges[1].min);
            assert!((row[0] - baseline_u).abs() < 1e-10);
            assert!((row[1] - expected_hp).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sobol_coverage() {
        let ranges = vec![
            ParameterRange {
                name: "window_u".to_string(),
                min: 0.1,
                max: 5.0,
            },
            ParameterRange {
                name: "heating_setpoint".to_string(),
                min: 15.0,
                max: 25.0,
            },
        ];
        let num_samples = 100;
        let design = generate_sobol_design(&ranges, num_samples);
        assert_eq!(design.len(), num_samples);
        for row in &design {
            assert_eq!(row.len(), 2);
            // Check within ranges (allow small epsilon for floating point)
            assert!(row[0] >= ranges[0].min - 1e-10);
            assert!(row[0] <= ranges[0].max + 1e-10);
            assert!(row[1] >= ranges[1].min - 1e-10);
            assert!(row[1] <= ranges[1].max + 1e-10);
        }
    }

    #[test]
    fn test_metrics_computation() {
        // Perfect linear relationship: y = 2*x + 1
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let design: Vec<Vec<f64>> = x.iter().map(|&val| vec![val]).collect();
        let outputs: Vec<f64> = x.iter().map(|&val| 2.0 * val + 1.0).collect();
        let report = compute_metrics(&design, &outputs);
        assert_eq!(report.parameters.len(), 1);
        let metric = &report.metrics[0];
        // slope should be approximately 2
        assert!((metric.slope - 2.0).abs() < 1e-10);
        // CVRMSE and NMBE should be zero for perfect fit
        assert!(metric.cvrmse.abs() < 1e-10);
        assert!(metric.nmbe.abs() < 1e-10);
        // normalized_coeff: (max_y - min_y)/mean_y *100
        // y: 1,3,5,7,9,11 -> min=1, max=11, mean=6
        let expected_norm = (11.0 - 1.0) / 6.0 * 100.0; // = 10/6*100 ≈ 166.666...
        assert!((metric.normalized_coeff - expected_norm).abs() < 1e-10);
    }

    #[test]
    fn test_csv_export() {
        let report = SensitivityReport {
            parameters: vec!["B".to_string(), "A".to_string()],
            metrics: vec![
                MetricSet {
                    normalized_coeff: 50.0,
                    cvrmse: 5.0,
                    nmbe: 0.5,
                    slope: 1.5,
                },
                MetricSet {
                    normalized_coeff: 200.0,
                    cvrmse: 2.0,
                    nmbe: -0.2,
                    slope: -3.0,
                },
            ],
        };
        // Create a temporary file
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();
        export_to_csv(&report, path).expect("Export failed");
        // Read back
        let mut rdr = Reader::from_path(path).unwrap();
        let headers = rdr.headers().unwrap();
        let expected_headers = [
            "Rank",
            "Parameter",
            "NormalizedCoeff",
            "CVRMSE",
            "NMBE",
            "Slope",
        ];
        assert_eq!(
            headers.iter().collect::<Vec<_>>(),
            expected_headers.as_ref()
        );
        let mut records = rdr.records().collect::<Result<Vec<_>, _>>().unwrap();
        // Should have 2 records
        assert_eq!(records.len(), 2);
        // First record should be the one with higher normalized_coeff (A:200)
        let first = &records[0];
        assert_eq!(first.get(1).unwrap(), "A"); // Parameter A should be rank 1
        assert_eq!(first.get(0).unwrap(), "1");
        // Second record is B
        let second = &records[1];
        assert_eq!(second.get(1).unwrap(), "B");
        assert_eq!(second.get(0).unwrap(), "2");
    }

    #[test]
    fn test_run_sensitivity_with_batch_oracle() {
        // Build a simple base model with 1 zone
        let base_model =
            crate::sim::engine::ThermalModel::<crate::physics::cta::VectorField>::new(1);
        let oracle = crate::BatchOracle::from_model(base_model);
        // Simple design: vary window U-value
        let design = vec![vec![1.5], vec![2.0]];
        // Run with surrogates disabled
        let outputs = super::run_sensitivity(&design, &oracle, false);
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|&x| x >= 0.0));
    }
}

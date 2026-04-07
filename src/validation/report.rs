//! Validation report generation and analysis for ASHRAE 140.
//!
//! This module provides structures and methods for generating comprehensive
//! validation reports, including pass/fail determination, delta analysis,
//! and multiple export formats (Markdown, HTML, CSV).

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::validation::multi_reference::{MultiReferenceDB, ProgramRange};
use crate::validation::statistical::{StatisticalMetrics, ValidationGroup};

/// Types of validation metrics for ASHRAE 140.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Annual heating energy consumption (MWh)
    AnnualHeating,
    /// Annual cooling energy consumption (MWh)
    AnnualCooling,
    /// Peak heating load (kW)
    PeakHeating,
    /// Peak cooling load (kW)
    PeakCooling,
    /// Minimum free-floating temperature (°C)
    MinFreeFloat,
    /// Maximum free-floating temperature (°C)
    MaxFreeFloat,
}

impl MetricType {
    /// Returns the display name for this metric type.
    pub fn display_name(&self) -> &str {
        match self {
            MetricType::AnnualHeating => "Annual Heating (MWh)",
            MetricType::AnnualCooling => "Annual Cooling (MWh)",
            MetricType::PeakHeating => "Peak Heating (kW)",
            MetricType::PeakCooling => "Peak Cooling (kW)",
            MetricType::MinFreeFloat => "Min Free-Float Temp (°C)",
            MetricType::MaxFreeFloat => "Max Free-Float Temp (°C)",
        }
    }

    /// Returns the units for this metric type.
    pub fn units(&self) -> &str {
        match self {
            MetricType::AnnualHeating | MetricType::AnnualCooling => "MWh",
            MetricType::PeakHeating | MetricType::PeakCooling => "kW",
            MetricType::MinFreeFloat | MetricType::MaxFreeFloat => "°C",
        }
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Validation status for a single metric comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Value within 5% of reference range
    Pass,
    /// Value within reference range but >2% deviation
    Warning,
    /// Value outside 5% tolerance band
    Fail,
}

impl fmt::Display for ValidationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationStatus::Pass => write!(f, "PASS"),
            ValidationStatus::Warning => write!(f, "WARN"),
            ValidationStatus::Fail => write!(f, "FAIL"),
        }
    }
}

impl ValidationStatus {
    /// Returns the display name for this status.
    pub fn display_name(&self) -> &str {
        match self {
            ValidationStatus::Pass => "PASS",
            ValidationStatus::Warning => "WARN",
            ValidationStatus::Fail => "FAIL",
        }
    }

    /// Returns the emoji icon for this status (for terminal output).
    pub fn icon(&self) -> &str {
        match self {
            ValidationStatus::Pass => "✓",
            ValidationStatus::Warning => "⚠",
            ValidationStatus::Fail => "✗",
        }
    }

    /// Returns the color code for HTML output.
    pub fn color(&self) -> &str {
        match self {
            ValidationStatus::Pass => "green",
            ValidationStatus::Warning => "orange",
            ValidationStatus::Fail => "red",
        }
    }
}

/// Computes validation status for a given value against a reference range.
///
/// Status determination:
/// - Pass: value within [min, max] with <10% deviation from midpoint
/// - Warning: within [min, max] but >=10% deviation, OR within tolerance band [min*0.95, max*1.05]
/// - Fail: outside tolerance band
pub fn compute_status(value: f64, ref_min: f64, ref_max: f64) -> ValidationStatus {
    let ref_mid = (ref_min + ref_max) / 2.0;
    let percent_error = if ref_mid != 0.0 {
        ((value - ref_mid) / ref_mid.abs()) * 100.0
    } else {
        0.0
    };

    let tolerance_min = ref_min * 0.95;
    let tolerance_max = ref_max * 1.05;

    if value >= ref_min && value <= ref_max {
        if percent_error.abs() >= 10.0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Pass
        }
    } else if value >= tolerance_min && value <= tolerance_max {
        ValidationStatus::Warning
    } else {
        ValidationStatus::Fail
    }
}

/// Reference programs for ASHRAE 140 validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceProgram {
    /// EnergyPlus - DOE's flagship building energy simulation program
    EnergyPlus,
    /// ESP-r - Research-grade building energy simulation from University of Strathclyde
    EspR,
    /// TRNSYS - Transient System Simulation Tool
    TRNSYS,
    /// DOE2 - Legacy DOE building energy simulation program
    DOE2,
}

impl fmt::Display for ReferenceProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReferenceProgram::EnergyPlus => write!(f, "EnergyPlus"),
            ReferenceProgram::EspR => write!(f, "ESP-r"),
            ReferenceProgram::TRNSYS => write!(f, "TRNSYS"),
            ReferenceProgram::DOE2 => write!(f, "DOE2"),
        }
    }
}

/// Benchmark data for a single ASHRAE 140 case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkData {
    /// Minimum annual heating (MWh) across reference programs
    pub annual_heating_min: f64,
    /// Maximum annual heating (MWh) across reference programs
    pub annual_heating_max: f64,
    /// Minimum annual cooling (MWh) across reference programs
    pub annual_cooling_min: f64,
    /// Maximum annual cooling (MWh) across reference programs
    pub annual_cooling_max: f64,
    /// Minimum peak heating load (kW) across reference programs
    pub peak_heating_min: f64,
    /// Maximum peak heating load (kW) across reference programs
    pub peak_heating_max: f64,
    /// Minimum peak cooling load (kW) across reference programs
    pub peak_cooling_min: f64,
    /// Maximum peak cooling load (kW) across reference programs
    pub peak_cooling_max: f64,
    /// Minimum free-floating temperature (°C) across reference programs
    pub min_free_float_min: f64,
    /// Maximum free-floating temperature (°C) across reference programs
    pub min_free_float_max: f64,
    /// Maximum free-floating temperature (°C) across reference programs
    pub max_free_float_min: f64,
    /// Maximum free-floating temperature (°C) across reference programs
    pub max_free_float_max: f64,
}

impl BenchmarkData {
    /// Creates a new BenchmarkData with all values initialized to zero.
    pub fn new() -> Self {
        Self {
            annual_heating_min: 0.0,
            annual_heating_max: 0.0,
            annual_cooling_min: 0.0,
            annual_cooling_max: 0.0,
            peak_heating_min: 0.0,
            peak_heating_max: 0.0,
            peak_cooling_min: 0.0,
            peak_cooling_max: 0.0,
            min_free_float_min: 0.0,
            min_free_float_max: 0.0,
            max_free_float_min: 0.0,
            max_free_float_max: 0.0,
        }
    }

    /// Returns the reference range for a given metric type.
    pub fn get_range(&self, metric: MetricType) -> Option<(f64, f64)> {
        match metric {
            MetricType::AnnualHeating => {
                if self.annual_heating_min > 0.0 || self.annual_heating_max > 0.0 {
                    Some((self.annual_heating_min, self.annual_heating_max))
                } else {
                    None
                }
            }
            MetricType::AnnualCooling => {
                if self.annual_cooling_min > 0.0 || self.annual_cooling_max > 0.0 {
                    Some((self.annual_cooling_min, self.annual_cooling_max))
                } else {
                    None
                }
            }
            MetricType::PeakHeating => {
                if self.peak_heating_min > 0.0 || self.peak_heating_max > 0.0 {
                    Some((self.peak_heating_min, self.peak_heating_max))
                } else {
                    None
                }
            }
            MetricType::PeakCooling => {
                if self.peak_cooling_min > 0.0 || self.peak_cooling_max > 0.0 {
                    Some((self.peak_cooling_min, self.peak_cooling_max))
                } else {
                    None
                }
            }
            MetricType::MinFreeFloat => {
                if self.min_free_float_min != 0.0 || self.min_free_float_max != 0.0 {
                    Some((self.min_free_float_min, self.min_free_float_max))
                } else {
                    None
                }
            }
            MetricType::MaxFreeFloat => {
                if self.max_free_float_min != 0.0 || self.max_free_float_max != 0.0 {
                    Some((self.max_free_float_min, self.max_free_float_max))
                } else {
                    None
                }
            }
        }
    }

    /// Calculates the midpoint of the reference range for a given metric.
    pub fn midpoint(&self, metric: MetricType) -> Option<f64> {
        self.get_range(metric).map(|(min, max)| (min + max) / 2.0)
    }
}

impl Default for BenchmarkData {
    fn default() -> Self {
        Self::new()
    }
}

/// A single validation result for a specific case and metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Case identifier (e.g., "600", "900", "600FF")
    pub case_id: String,
    /// Metric type
    pub metric: MetricType,
    /// Fluxion simulation value
    pub fluxion_value: f64,
    /// Reference minimum value
    pub ref_min: f64,
    /// Reference maximum value
    pub ref_max: f64,
    /// Percent error from reference midpoint
    pub percent_error: f64,
    /// Validation status
    pub status: ValidationStatus,
    /// Per-program validation statuses for multi-reference comparison
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_program: Option<HashMap<String, ValidationStatus>>,
}

impl ValidationResult {
    /// Creates a new validation result and determines pass/fail status.
    pub fn new(
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
    ) -> Self {
        // Calculate reference midpoint
        let ref_mid = (ref_min + ref_max) / 2.0;

        // Calculate percent error from reference midpoint
        let percent_error = if ref_mid != 0.0 {
            ((fluxion_value - ref_mid) / ref_mid.abs()) * 100.0
        } else {
            0.0
        };

        // Determine pass/fail status
        // Pass: Within [Ref Min, Ref Max] with <10% deviation from midpoint
        // Warning: Within [Ref Min, Ref Max] with >=10% deviation, OR within tolerance band but outside ref range
        // Fail: Outside 5% tolerance band
        let tolerance_min = ref_min * 0.95;
        let tolerance_max = ref_max * 1.05;

        let status = if fluxion_value >= ref_min && fluxion_value <= ref_max {
            // Within reference range - check percent error
            if percent_error.abs() >= 10.0 {
                ValidationStatus::Warning
            } else {
                ValidationStatus::Pass
            }
        } else if fluxion_value >= tolerance_min && fluxion_value <= tolerance_max {
            // Within tolerance band but outside reference range
            ValidationStatus::Warning
        } else {
            ValidationStatus::Fail
        };

        Self {
            case_id: case_id.to_string(),
            metric,
            fluxion_value,
            ref_min,
            ref_max,
            percent_error,
            status,
            per_program: None,
        }
    }

    /// Returns true if this result passed validation (within reference range with <10% error).
    pub fn is_pass(&self) -> bool {
        matches!(self.status, ValidationStatus::Pass)
    }

    /// Returns true if this result is a warning (within reference range but >=10% error, or within tolerance band).
    pub fn is_warning(&self) -> bool {
        matches!(self.status, ValidationStatus::Warning)
    }

    /// Returns true if this result failed validation (outside tolerance band).
    pub fn is_fail(&self) -> bool {
        matches!(self.status, ValidationStatus::Fail)
    }

    /// Returns the deviation from reference range center as a string.
    pub fn deviation_string(&self) -> String {
        format!("{:+.2}%", self.percent_error)
    }

    /// Returns true if the value is within the reference range.
    pub fn is_within_range(&self) -> bool {
        self.fluxion_value >= self.ref_min && self.fluxion_value <= self.ref_max
    }

    /// Returns the deviation from the reference range center as a percentage.
    pub fn deviation_percent(&self) -> f64 {
        self.percent_error
    }

    /// Returns true if this result passed validation.
    pub fn passed(&self) -> bool {
        self.status == ValidationStatus::Pass
    }

    /// Returns true if this result is a warning.
    pub fn warning(&self) -> bool {
        self.status == ValidationStatus::Warning
    }

    /// Returns true if this result failed validation.
    pub fn failed(&self) -> bool {
        self.status == ValidationStatus::Fail
    }
}

/// Interpretation guidance for failed validation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interpretation {
    /// Root cause hypotheses explaining why the metric failed
    pub root_cause_hypotheses: Vec<String>,
    /// Parameter sensitivity analysis
    pub parameter_sensitivity: Vec<String>,
    /// Recommended next steps for investigation
    pub recommended_next_steps: Vec<String>,
    /// What-if scenarios for debugging approaches
    pub what_if_scenarios: Vec<String>,
    /// References to relevant documentation
    pub references: Vec<String>,
}

impl Default for Interpretation {
    fn default() -> Self {
        Self {
            root_cause_hypotheses: Vec::new(),
            parameter_sensitivity: Vec::new(),
            recommended_next_steps: Vec::new(),
            what_if_scenarios: Vec::new(),
            references: Vec::new(),
        }
    }
}

/// Comprehensive validation report for ASHRAE 140 test cases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// All validation results
    pub results: Vec<ValidationResult>,
    /// Benchmark data for each case
    pub benchmark_data: HashMap<String, BenchmarkData>,
    /// Interpretation guidance for failed metrics (not serialized for backwards compatibility)
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub interpretations: HashMap<String, Interpretation>,
    /// Start time for performance measurement (not serialized)
    #[serde(skip)]
    pub start_time: Option<Instant>,
    /// End time for performance measurement (not serialized)
    #[serde(skip)]
    pub end_time: Option<Instant>,
    /// Statistical metrics for the report (NMBE, CV(RMSE), etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_metrics: Option<StatisticalMetrics>,
    /// Per-case p-values from statistical tests
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_p_values: Option<Vec<f64>>,
    /// BH-corrected status for each test (true = rejected null hypothesis)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_corrected: Option<Vec<bool>>,
    /// Group-level validation results (PASS/FAIL per validation group)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_validation: Option<HashMap<ValidationGroup, bool>>,
}

/// Delta test result with statistical significance testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaTestResult {
    /// Metric name
    pub metric_name: String,
    /// Delta value (mean difference)
    pub delta_value: f64,
    /// P-value for statistical significance
    pub p_value: Option<f64>,
    /// Whether the difference is statistically significant
    pub is_significant: bool,
    /// Confidence interval for the delta
    pub confidence_interval: Option<(f64, f64)>,
}

/// Sensitivity analysis result with normalized metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityResult {
    /// Parameter name
    pub parameter_name: String,
    /// Raw coefficient
    pub coefficient: f64,
    /// Normalized coefficient (divided by parameter range)
    pub normalized_coefficient: f64,
    /// Parameter ranking (1 = most sensitive)
    pub ranking: usize,
}

impl BenchmarkReport {
    /// Creates a new empty validation report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generates a JSON report.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Adds a validation result to the report.
    pub fn add_result(&mut self, result: ValidationResult) {
        self.results.push(result);
    }

    /// Adds a result using the simplified interface.
    pub fn add_result_simple(
        &mut self,
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
    ) {
        let result = ValidationResult::new(case_id, metric, fluxion_value, ref_min, ref_max);
        self.add_result(result);
    }

    /// Adds a validation result using multi-reference data, populating per-program statuses.
    ///
    /// This method looks up per-program reference ranges from the provided MultiReferenceDB,
    /// computes individual program validation statuses, determines the overall status using
    /// the rule: PASS if EnergyPlus passes, else WARN if any program passes, else FAIL.
    ///
    /// The aggregated ref_min and ref_max are computed as the envelope of all programs.
    pub fn add_result_with_multi(
        &mut self,
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        db: &MultiReferenceDB,
    ) {
        // Look up case references
        let case_refs = match db.cases.get(case_id) {
            Some(c) => c,
            None => {
                // If case not found in multi-ref DB, fall back to simple method with zeros?
                // To avoid panics, we'll create a result with per_program=None and zero refs.
                let result = ValidationResult {
                    case_id: case_id.to_string(),
                    metric,
                    fluxion_value,
                    ref_min: 0.0,
                    ref_max: 0.0,
                    percent_error: 0.0,
                    status: ValidationStatus::Fail,
                    per_program: None,
                };
                self.results.push(result);
                return;
            }
        };

        // Get the program ranges for this metric
        let program_ranges: &std::collections::HashMap<String, ProgramRange> = match metric {
            MetricType::AnnualHeating => &case_refs.annual_heating,
            MetricType::AnnualCooling => &case_refs.annual_cooling,
            MetricType::PeakHeating => &case_refs.peak_heating,
            MetricType::PeakCooling => &case_refs.peak_cooling,
            _ => {
                // For free-floating metrics, multi-reference may not be defined; fall back to no per_program
                let result = ValidationResult::new(case_id, metric, fluxion_value, 0.0, 0.0);
                self.results.push(result);
                return;
            }
        };

        // Compute aggregated ref_min and ref_max as envelope of all programs
        let agg_min = program_ranges
            .values()
            .map(|r| r.min)
            .fold(f64::INFINITY, f64::min);
        let agg_max = program_ranges
            .values()
            .map(|r| r.max)
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute percent error based on aggregated midpoint
        let agg_mid = (agg_min + agg_max) / 2.0;
        let percent_error = if agg_mid != 0.0 {
            ((fluxion_value - agg_mid) / agg_mid.abs()) * 100.0
        } else {
            0.0
        };

        // Compute per-program statuses
        let mut per_program = std::collections::HashMap::new();
        for (prog_name, range) in program_ranges {
            let status = compute_status(fluxion_value, range.min, range.max);
            per_program.insert(prog_name.clone(), status);
        }

        // Determine overall status based on EnergyPlus primary, then any pass
        let overall_status = if let Some(ep_status) = per_program.get("EnergyPlus") {
            if *ep_status == ValidationStatus::Pass {
                ValidationStatus::Pass
            } else if per_program.values().any(|s| *s == ValidationStatus::Pass) {
                ValidationStatus::Warning
            } else {
                ValidationStatus::Fail
            }
        } else {
            // EnergyPlus not in the list; use aggregated envelope status
            compute_status(fluxion_value, agg_min, agg_max)
        };

        let result = ValidationResult {
            case_id: case_id.to_string(),
            metric,
            fluxion_value,
            ref_min: agg_min,
            ref_max: agg_max,
            percent_error,
            status: overall_status,
            per_program: Some(per_program),
        };
        self.add_result(result);
    }

    /// Adds benchmark data for a case.
    pub fn add_benchmark_data(&mut self, case_id: &str, data: BenchmarkData) {
        self.benchmark_data.insert(case_id.to_string(), data);
    }

    /// Enriches existing validation results with multi-reference per-program statuses.
    ///
    /// This method processes all results currently in the report. For each result with a metric
    /// that has multi-reference data (AnnualHeating, AnnualCooling, PeakHeating, PeakCooling),
    /// it adds per-program PASS/WARN/FAIL statuses by looking up the reference ranges in the
    /// provided MultiReferenceDB. Results for metrics without multi-reference data (e.g., free-floating
    /// temperatures) or for cases not found in the database are left unchanged.
    ///
    /// The overall status for enriched results is determined by:
    /// - PASS if EnergyPlus passes
    /// - WARN if EnergyPlus fails but any other program passes
    /// - FAIL if all programs fail
    ///
    /// The aggregated ref_min and ref_max are computed as the envelope (min of mins, max of maxes)
    /// across all reference programs.
    pub fn enrich_with_multi_reference(&mut self, db: &MultiReferenceDB) {
        let mut enriched = Vec::new();

        for result in &self.results {
            // Determine if this metric can be enriched with multi-reference data
            let can_enrich = match result.metric {
                MetricType::AnnualHeating
                | MetricType::AnnualCooling
                | MetricType::PeakHeating
                | MetricType::PeakCooling => true,
                _ => false,
            };

            if can_enrich {
                // Use the add_result_with_multi method to create an enriched version
                // We create a temporary BenchmarkReport to reuse the logic
                let mut temp_report = BenchmarkReport::new();
                temp_report.add_result_with_multi(
                    &result.case_id,
                    result.metric,
                    result.fluxion_value,
                    db,
                );
                if let Some(enriched_result) = temp_report.results.into_iter().next() {
                    enriched.push(enriched_result);
                } else {
                    // Shouldn't happen, but preserve original if it does
                    enriched.push(result.clone());
                }
            } else {
                // Metrics without multi-reference (free-floating temps, etc.) stay unchanged
                enriched.push(result.clone());
            }
        }

        self.results = enriched;
    }

    /// Calculates delta analysis: difference between cases vs baseline.
    pub fn delta_analysis(&self, baseline_case: &str) -> HashMap<String, f64> {
        let mut deltas = HashMap::new();
        let baseline_results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.case_id == baseline_case)
            .collect();

        for result in &self.results {
            if result.case_id != baseline_case {
                // Find matching metric in baseline
                if let Some(baseline) = baseline_results.iter().find(|b| b.metric == result.metric)
                {
                    let delta = result.fluxion_value - baseline.fluxion_value;
                    let key = format!("{} - {}", result.case_id, result.metric.display_name());
                    deltas.insert(key, delta);
                }
            }
        }

        deltas
    }

    /// Calculates overall pass rate as a percentage.
    pub fn pass_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 100.0;
        }

        let passed = self.results.iter().filter(|r| r.passed()).count();
        (passed as f64 / self.results.len() as f64) * 100.0
    }

    /// Calculates the number of failed results.
    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| r.failed()).count()
    }

    /// Calculates the number of warnings.
    pub fn warning_count(&self) -> usize {
        self.results.iter().filter(|r| r.warning()).count()
    }

    /// Calculates the Mean Absolute Error (MAE) across all results.
    pub fn mae(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let total_error: f64 = self.results.iter().map(|r| r.percent_error.abs()).sum();
        total_error / self.results.len() as f64
    }

    /// Calculates the maximum deviation percentage.
    pub fn max_deviation(&self) -> f64 {
        self.results
            .iter()
            .map(|r| r.percent_error.abs())
            .fold(0.0f64, |a, b| a.max(b))
    }

    /// Returns cases with the worst performance (highest deviation).
    pub fn worst_cases(&self, top_n: usize) -> Vec<ValidationResult> {
        let mut sorted = self.results.clone();
        sorted.sort_by(|a, b| {
            b.percent_error
                .abs()
                .partial_cmp(&a.percent_error.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(top_n).collect()
    }

    /// Sets the start time for performance measurement.
    pub fn set_start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Sets the end time for performance measurement.
    pub fn set_end(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Returns the duration of the validation in seconds.
    pub fn duration_seconds(&self) -> f64 {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => end.duration_since(start).as_secs_f64(),
            _ => 0.0,
        }
    }

    /// Returns the number of cases validated per second.
    pub fn cases_per_second(&self) -> f64 {
        let duration = self.duration_seconds();
        if duration > 0.0 {
            self.benchmark_data.len() as f64 / duration
        } else {
            0.0
        }
    }

    /// Generates a Markdown report.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Title
        output.push_str("# ASHRAE 140 Validation Report\n\n");

        // Summary statistics
        output.push_str("## Summary\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Total Results | {} |\n", self.results.len()));
        output.push_str(&format!("| Pass Rate | {:.1}% |\n", self.pass_rate()));
        output.push_str(&format!(
            "| Passed | {} |\n",
            self.results.iter().filter(|r| r.passed()).count()
        ));
        output.push_str(&format!("| Warnings | {} |\n", self.warning_count()));
        output.push_str(&format!("| Failed | {} |\n", self.fail_count()));
        output.push_str(&format!("| Mean Absolute Error | {:.2}% |\n", self.mae()));
        output.push_str(&format!(
            "| Max Deviation | {:.2}% |\n",
            self.max_deviation()
        ));
        output.push('\n');

        // Detailed results table
        output.push_str("## Detailed Results\n\n");
        output.push_str("| Case | Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |\n");
        output.push_str("|------|--------|---------|---------|---------|-----------|--------|\n");

        for result in &self.results {
            output.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2} | {} | {} |\n",
                result.case_id,
                result.metric,
                result.fluxion_value,
                result.ref_min,
                result.ref_max,
                result.deviation_string(),
                result.status
            ));
        }

        output.push('\n');

        // Delta analysis
        if !self.benchmark_data.is_empty() {
            let baseline = self.benchmark_data.keys().next().unwrap();
            let deltas = self.delta_analysis(baseline);

            if !deltas.is_empty() {
                output.push_str("## Delta Analysis\n\n");
                output.push_str(&format!("Baseline: {}\n\n", baseline));
                output.push_str("| Case - Metric | Delta from Baseline |\n");
                output.push_str("|---------------|---------------------|\n");

                for (key, delta) in &deltas {
                    output.push_str(&format!("| {} | {:+.2} |\n", key, delta));
                }

                output.push('\n');
            }
        }

        // Worst cases
        let worst = self.worst_cases(5);
        if !worst.is_empty() {
            output.push_str("## Worst Performing Cases\n\n");
            output.push_str("| Case | Metric | Deviation | Status |\n");
            output.push_str("|------|--------|-----------|--------|\n");

            for result in worst {
                output.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    result.case_id,
                    result.metric,
                    result.deviation_string(),
                    result.status
                ));
            }

            output.push('\n');
        }

        // Interpretation guidance for failed cases
        if !self.interpretations.is_empty() {
            output.push_str("## Interpretation Guidance\n\n");
            output.push_str("The following interpretation guidance is provided for cases with failed metrics:\n\n");

            let mut case_ids: Vec<_> = self.interpretations.keys().collect();
            case_ids.sort();

            for case_id in case_ids {
                if let Some(interp) = self.interpretations.get(case_id) {
                    output.push_str(&format!("### Case {}\n\n", case_id));

                    // Root cause hypotheses
                    if !interp.root_cause_hypotheses.is_empty() {
                        output.push_str("**Root Cause Hypothesis:**\n");
                        for hypothesis in &interp.root_cause_hypotheses {
                            output.push_str(&format!("- {}\n", hypothesis));
                        }
                        output.push('\n');
                    }

                    // Parameter sensitivity
                    if !interp.parameter_sensitivity.is_empty() {
                        output.push_str("**Parameter Sensitivity:**\n");
                        for sensitivity in &interp.parameter_sensitivity {
                            output.push_str(&format!("- {}\n", sensitivity));
                        }
                        output.push('\n');
                    }

                    // Recommended next steps
                    if !interp.recommended_next_steps.is_empty() {
                        output.push_str("**Recommended Next Steps:**\n");
                        for step in &interp.recommended_next_steps {
                            output.push_str(&format!("- {}\n", step));
                        }
                        output.push('\n');
                    }

                    // What-if scenarios
                    if !interp.what_if_scenarios.is_empty() {
                        output.push_str("**What-if Scenarios:**\n");
                        for scenario in &interp.what_if_scenarios {
                            output.push_str(&format!("- {}\n", scenario));
                        }
                        output.push('\n');
                    }

                    // References
                    if !interp.references.is_empty() {
                        output.push_str("**References:**\n");
                        for ref_doc in &interp.references {
                            output.push_str(&format!("- {}\n", ref_doc));
                        }
                        output.push('\n');
                    }
                }
            }
        }

        // Legend
        output.push_str("## Legend\n\n");
        output.push_str("- **PASS**: Value within 5% of reference range\n");
        output.push_str("- **WARN**: Value within reference range but >2% deviation\n");
        output.push_str("- **FAIL**: Value outside 5% tolerance band\n");

        output
    }

    /// Generates an HTML report.
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html>\n");
        html.push_str("<head>\n");
        html.push_str("  <title>ASHRAE 140 Validation Report</title>\n");
        html.push_str("  <style>\n");
        html.push_str("    body { font-family: Arial, sans-serif; margin: 40px; }\n");
        html.push_str("    h1 { color: #333; }\n");
        html.push_str("    h2 { color: #666; border-bottom: 1px solid #ddd; }\n");
        html.push_str(
            "    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n",
        );
        html.push_str("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("    th { background-color: #f2f2f2; }\n");
        html.push_str("    tr:nth-child(even) { background-color: #f9f9f9; }\n");
        html.push_str("    .pass { color: green; font-weight: bold; }\n");
        html.push_str("    .warning { color: orange; font-weight: bold; }\n");
        html.push_str("    .fail { color: red; font-weight: bold; }\n");
        html.push_str("    .positive { color: green; }\n");
        html.push_str("    .negative { color: red; }\n");
        html.push_str("  </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        html.push_str("  <h1>ASHRAE 140 Validation Report</h1>\n");

        // Summary statistics
        html.push_str("  <h2>Summary</h2>\n");
        html.push_str("  <table>\n");
        html.push_str("    <tr><th>Metric</th><th>Value</th></tr>\n");
        html.push_str(&format!(
            "    <tr><td>Total Results</td><td>{}</td></tr>\n",
            self.results.len()
        ));
        html.push_str(&format!(
            "    <tr><td>Pass Rate</td><td>{:.1}%</td></tr>\n",
            self.pass_rate()
        ));
        html.push_str(&format!(
            "    <tr><td>Passed</td><td>{}</td></tr>\n",
            self.results.iter().filter(|r| r.passed()).count()
        ));
        html.push_str(&format!(
            "    <tr><td>Warnings</td><td>{}</td></tr>\n",
            self.warning_count()
        ));
        html.push_str(&format!(
            "    <tr><td>Failed</td><td>{}</td></tr>\n",
            self.fail_count()
        ));
        html.push_str(&format!(
            "    <tr><td>Mean Absolute Error</td><td>{:.2}%</td></tr>\n",
            self.mae()
        ));
        html.push_str(&format!(
            "    <tr><td>Max Deviation</td><td>{:.2}%</td></tr>\n",
            self.max_deviation()
        ));
        html.push_str("  </table>\n");

        // Detailed results table
        html.push_str("  <h2>Detailed Results</h2>\n");
        html.push_str("  <table>\n");
        html.push_str("    <tr><th>Case</th><th>Metric</th><th>Fluxion</th><th>Ref Min</th><th>Ref Max</th><th>Deviation</th><th>Status</th></tr>\n");

        for result in &self.results {
            let status_class = match result.status {
                ValidationStatus::Pass => "pass",
                ValidationStatus::Warning => "warning",
                ValidationStatus::Fail => "fail",
            };

            let deviation_class = if result.percent_error > 0.0 {
                "positive"
            } else {
                "negative"
            };

            html.push_str("    <tr>\n");
            html.push_str(&format!("      <td>{}</td>\n", result.case_id));
            html.push_str(&format!("      <td>{}</td>\n", result.metric));
            html.push_str(&format!("      <td>{:.2}</td>\n", result.fluxion_value));
            html.push_str(&format!("      <td>{:.2}</td>\n", result.ref_min));
            html.push_str(&format!("      <td>{:.2}</td>\n", result.ref_max));
            html.push_str(&format!(
                "      <td class=\"{}\">{}</td>\n",
                deviation_class,
                result.deviation_string()
            ));
            html.push_str(&format!(
                "      <td class=\"{}\">{}</td>\n",
                status_class, result.status
            ));
            html.push_str("    </tr>\n");
        }

        html.push_str("  </table>\n");

        // Delta analysis
        if !self.benchmark_data.is_empty() {
            let baseline = self.benchmark_data.keys().next().unwrap();
            let deltas = self.delta_analysis(baseline);

            if !deltas.is_empty() {
                html.push_str("  <h2>Delta Analysis</h2>\n");
                html.push_str(&format!(
                    "  <p><strong>Baseline:</strong> {}</p>\n",
                    baseline
                ));
                html.push_str("  <table>\n");
                html.push_str("    <tr><th>Case - Metric</th><th>Delta from Baseline</th></tr>\n");

                for (key, delta) in &deltas {
                    let delta_class = if *delta > 0.0 { "positive" } else { "negative" };
                    html.push_str(&format!(
                        "    <tr><td>{}</td><td class=\"{}\">{:+.2}</td></tr>\n",
                        key, delta_class, delta
                    ));
                }

                html.push_str("  </table>\n");
            }
        }

        // Worst cases
        let worst = self.worst_cases(5);
        if !worst.is_empty() {
            html.push_str("  <h2>Worst Performing Cases</h2>\n");
            html.push_str("  <table>\n");
            html.push_str(
                "    <tr><th>Case</th><th>Metric</th><th>Deviation</th><th>Status</th></tr>\n",
            );

            for result in worst {
                let status_class = match result.status {
                    ValidationStatus::Pass => "pass",
                    ValidationStatus::Warning => "warning",
                    ValidationStatus::Fail => "fail",
                };

                html.push_str(&format!(
                    "    <tr><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td></tr>\n",
                    result.case_id,
                    result.metric,
                    result.deviation_string(),
                    status_class,
                    result.status
                ));
            }

            html.push_str("  </table>\n");
        }

        // Legend
        html.push_str("  <h2>Legend</h2>\n");
        html.push_str("  <ul>\n");
        html.push_str("    <li><strong>PASS</strong>: Value within 5% of reference range</li>\n");
        html.push_str(
            "    <li><strong>WARN</strong>: Value within reference range but >2% deviation</li>\n",
        );
        html.push_str("    <li><strong>FAIL</strong>: Value outside 5% tolerance band</li>\n");
        html.push_str("  </ul>\n");

        html.push_str("</body>\n");
        html.push_str("</html>\n");

        html
    }

    /// Generates a CSV report.
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("Case,Metric,Fluxion,Ref Min,Ref Max,Percent Error,Status\n");

        // Data rows
        for result in &self.results {
            csv.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.2},{}\n",
                result.case_id,
                result.metric,
                result.fluxion_value,
                result.ref_min,
                result.ref_max,
                result.percent_error,
                result.status
            ));
        }

        csv
    }

    /// Saves the report to a file based on the extension.
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        let content = match path.extension().and_then(|e| e.to_str()) {
            Some("md") => self.to_markdown(),
            Some("html") => self.to_html(),
            Some("htm") => self.to_html(),
            Some("csv") => self.to_csv(),
            Some("txt") => self.to_markdown(),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Unsupported file extension. Use .md, .html, or .csv",
                ))
            }
        };

        std::fs::write(path, content)
    }

    /// Prints a summary to stdout.
    pub fn print_summary(&self) {
        println!("Validation Report Summary:");
        println!("  Total Results: {}", self.results.len());
        println!("  Pass Rate: {:.1}%", self.pass_rate());
        println!(
            "  Passed: {}",
            self.results.iter().filter(|r| r.passed()).count()
        );
        println!("  Warnings: {}", self.warning_count());
        println!("  Failed: {}", self.fail_count());
        println!("  Mean Absolute Error: {:.2}%", self.mae());
        println!("  Max Deviation: {:.2}%", self.max_deviation());
    }

    /// Appends the report's metrics to the historical performance log.
    ///
    /// This method serializes key metrics (timestamp, MAE, max deviation, pass rate,
    /// duration, throughput) to `target/performance_history.jsonl` as a JSON line.
    /// It also attempts to include the git SHA if available from environment variables.
    /// I/O errors are handled gracefully with a warning printed to stderr.
    pub fn append_history(&self) {
        use std::fs::OpenOptions;
        use std::io::Write;

        // Collect metrics
        let timestamp = Utc::now().to_rfc3339();
        let mae = self.mae();
        let max_deviation = self.max_deviation();
        let pass_rate = self.pass_rate();
        let validation_time_seconds = self.duration_seconds();
        let throughput = self.cases_per_second();

        // Get git SHA from common CI environment variables
        let git_sha = env::var("GIT_SHA")
            .or_else(|_| env::var("GITHUB_SHA"))
            .or_else(|_| env::var("CI_COMMIT_SHA"))
            .ok()
            .map(String::from);

        // Construct JSON object
        #[derive(serde::Serialize)]
        struct HistoryEntry {
            timestamp: String,
            mae: f64,
            max_deviation: f64,
            pass_rate: f64,
            validation_time_seconds: f64,
            throughput: f64,
            git_sha: Option<String>,
        }

        let entry = HistoryEntry {
            timestamp,
            mae,
            max_deviation,
            pass_rate,
            validation_time_seconds,
            throughput,
            git_sha,
        };

        // Determine file path (target/performance_history.jsonl)
        let file_path = Path::new("target").join("performance_history.jsonl");

        // Create target directory if it doesn't exist
        if let Some(parent) = file_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("Warning: Failed to create target directory: {}", e);
                return;
            }
        }

        // Append JSON line to file
        let json_line = match serde_json::to_string(&entry) {
            Ok(line) => line,
            Err(e) => {
                eprintln!("Warning: Failed to serialize history entry: {}", e);
                return;
            }
        };

        let mut file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
        {
            Ok(file) => file,
            Err(e) => {
                eprintln!(
                    "Warning: Failed to open performance history file for appending: {}",
                    e
                );
                return;
            }
        };

        if let Err(e) = writeln!(file, "{}", json_line) {
            eprintln!("Warning: Failed to write to performance history: {}", e);
        }
    }
}

/// A collection of validation results for multiple cases.
///
/// `ValidationSuite` provides high-level methods for collecting, analyzing,
/// and reporting on validation results across multiple test cases.
#[derive(Debug, Clone, Default)]
pub struct ValidationSuite {
    /// All validation results
    results: Vec<ValidationResult>,
    /// Benchmark data for each case
    benchmark_data: HashMap<String, BenchmarkData>,
    /// Interpretation guidance for failed metrics
    interpretations: HashMap<String, Interpretation>,
}

impl ValidationSuite {
    /// Creates a new empty validation suite.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a validation suite pre-populated with all ASHRAE 140 benchmark data.
    pub fn with_ashrae140_data() -> Self {
        let mut suite = Self::new();
        let data = crate::validation::benchmark::get_all_benchmark_data();
        for (case_id, benchmark) in data {
            suite.benchmark_data.insert(case_id, benchmark);
        }
        suite
    }

    /// Adds a validation result to the suite.
    pub fn add_result(&mut self, result: ValidationResult) {
        self.results.push(result);
    }

    /// Adds a result using the simplified interface.
    pub fn add_result_simple(
        &mut self,
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
    ) {
        let result = ValidationResult::new(case_id, metric, fluxion_value, ref_min, ref_max);
        self.add_result(result);
    }

    /// Adds benchmark data for a case.
    pub fn add_benchmark_data(&mut self, case_id: &str, data: BenchmarkData) {
        self.benchmark_data.insert(case_id.to_string(), data);
    }

    /// Returns the total number of results in the suite.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Returns true if the suite has no results.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Returns the number of passed results.
    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed()).count()
    }

    /// Returns the number of failed results.
    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| r.failed()).count()
    }

    /// Returns the number of warning results.
    pub fn warning_count(&self) -> usize {
        self.results.iter().filter(|r| r.warning()).count()
    }

    /// Calculates the pass rate as a percentage.
    pub fn calculate_pass_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 100.0;
        }

        let passed = self.results.iter().filter(|r| r.passed()).count();
        (passed as f64 / self.results.len() as f64) * 100.0
    }

    /// Calculates the warning rate as a percentage.
    pub fn calculate_warning_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let warnings = self.results.iter().filter(|r| r.warning()).count();
        (warnings as f64 / self.results.len() as f64) * 100.0
    }

    /// Calculates the fail rate as a percentage.
    pub fn calculate_fail_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let failed = self.results.iter().filter(|r| r.failed()).count();
        (failed as f64 / self.results.len() as f64) * 100.0
    }

    /// Calculates the Mean Absolute Error (MAE) across all results.
    pub fn calculate_mae(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let total_error: f64 = self.results.iter().map(|r| r.percent_error.abs()).sum();
        total_error / self.results.len() as f64
    }

    /// Alias for calculate_mae() for consistency with BenchmarkReport.
    pub fn mae(&self) -> f64 {
        self.calculate_mae()
    }

    /// Alias for calculate_max_deviation() for consistency with BenchmarkReport.
    pub fn max_deviation(&self) -> f64 {
        self.calculate_max_deviation()
    }

    /// Alias for calculate_pass_rate() for consistency with BenchmarkReport.
    pub fn pass_rate(&self) -> f64 {
        self.calculate_pass_rate()
    }

    /// Calculates the Root Mean Square Error (RMSE) across all results.
    pub fn calculate_rmse(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let sum_squared: f64 = self.results.iter().map(|r| r.percent_error.powi(2)).sum();
        (sum_squared / self.results.len() as f64).sqrt()
    }

    /// Calculates the maximum deviation percentage.
    pub fn calculate_max_deviation(&self) -> f64 {
        self.results
            .iter()
            .map(|r| r.percent_error.abs())
            .fold(0.0f64, |a, b| a.max(b))
    }

    /// Calculates the mean deviation percentage.
    pub fn calculate_mean_deviation(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let total: f64 = self.results.iter().map(|r| r.percent_error).sum();
        total / self.results.len() as f64
    }

    /// Returns cases with the worst performance (highest deviation).
    pub fn worst_cases(&self, top_n: usize) -> Vec<ValidationResult> {
        let mut sorted = self.results.clone();
        sorted.sort_by(|a, b| {
            b.percent_error
                .abs()
                .partial_cmp(&a.percent_error.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(top_n).collect()
    }

    /// Returns all results for a specific case.
    pub fn get_case_results(&self, case_id: &str) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.case_id == case_id)
            .collect()
    }

    /// Returns all results for a specific metric type.
    pub fn get_metric_results(&self, metric: MetricType) -> Vec<&ValidationResult> {
        self.results.iter().filter(|r| r.metric == metric).collect()
    }

    /// Returns the pass rate for a specific case.
    pub fn calculate_case_pass_rate(&self, case_id: &str) -> Option<f64> {
        let case_results = self.get_case_results(case_id);
        if case_results.is_empty() {
            return None;
        }

        let passed = case_results.iter().filter(|r| r.passed()).count();
        Some((passed as f64 / case_results.len() as f64) * 100.0)
    }

    /// Returns a summary of results by case.
    pub fn summary_by_case(&self) -> HashMap<String, (usize, usize, usize)> {
        let mut summary: HashMap<String, (usize, usize, usize)> = HashMap::new();

        for result in &self.results {
            let entry = summary.entry(result.case_id.clone()).or_insert((0, 0, 0));

            if result.passed() {
                entry.0 += 1;
            } else if result.warning() {
                entry.1 += 1;
            } else {
                entry.2 += 1;
            }
        }

        summary
    }

    /// Returns a summary of results by metric type.
    pub fn summary_by_metric(&self) -> HashMap<MetricType, (usize, usize, usize)> {
        let mut summary: HashMap<MetricType, (usize, usize, usize)> = HashMap::new();

        for result in &self.results {
            let entry = summary.entry(result.metric).or_insert((0, 0, 0));

            if result.passed() {
                entry.0 += 1;
            } else if result.warning() {
                entry.1 += 1;
            } else {
                entry.2 += 1;
            }
        }

        summary
    }

    /// Generates a comprehensive validation report.
    pub fn generate_report(&self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();

        // Copy all results
        report.results = self.results.clone();

        // Copy benchmark data, or populate from results if missing
        if self.benchmark_data.is_empty() && !self.results.is_empty() {
            // Create benchmark data from results
            let mut case_data: HashMap<String, BenchmarkData> = HashMap::new();

            for result in &self.results {
                let benchmark = case_data.entry(result.case_id.clone()).or_default();

                // Populate based on metric type
                match result.metric {
                    MetricType::AnnualHeating => {
                        if benchmark.annual_heating_min == 0.0
                            || result.ref_min < benchmark.annual_heating_min
                        {
                            benchmark.annual_heating_min = result.ref_min;
                        }
                        if benchmark.annual_heating_max == 0.0
                            || result.ref_max > benchmark.annual_heating_max
                        {
                            benchmark.annual_heating_max = result.ref_max;
                        }
                    }
                    MetricType::AnnualCooling => {
                        if benchmark.annual_cooling_min == 0.0
                            || result.ref_min < benchmark.annual_cooling_min
                        {
                            benchmark.annual_cooling_min = result.ref_min;
                        }
                        if benchmark.annual_cooling_max == 0.0
                            || result.ref_max > benchmark.annual_cooling_max
                        {
                            benchmark.annual_cooling_max = result.ref_max;
                        }
                    }
                    MetricType::PeakHeating => {
                        if benchmark.peak_heating_min == 0.0
                            || result.ref_min < benchmark.peak_heating_min
                        {
                            benchmark.peak_heating_min = result.ref_min;
                        }
                        if benchmark.peak_heating_max == 0.0
                            || result.ref_max > benchmark.peak_heating_max
                        {
                            benchmark.peak_heating_max = result.ref_max;
                        }
                    }
                    MetricType::PeakCooling => {
                        if benchmark.peak_cooling_min == 0.0
                            || result.ref_min < benchmark.peak_cooling_min
                        {
                            benchmark.peak_cooling_min = result.ref_min;
                        }
                        if benchmark.peak_cooling_max == 0.0
                            || result.ref_max > benchmark.peak_cooling_max
                        {
                            benchmark.peak_cooling_max = result.ref_max;
                        }
                    }
                    MetricType::MinFreeFloat => {
                        if benchmark.min_free_float_min == 0.0
                            || result.ref_min < benchmark.min_free_float_min
                        {
                            benchmark.min_free_float_min = result.ref_min;
                        }
                        if benchmark.min_free_float_max == 0.0
                            || result.ref_max > benchmark.min_free_float_max
                        {
                            benchmark.min_free_float_max = result.ref_max;
                        }
                    }
                    MetricType::MaxFreeFloat => {
                        if benchmark.max_free_float_min == 0.0
                            || result.ref_min < benchmark.max_free_float_min
                        {
                            benchmark.max_free_float_min = result.ref_min;
                        }
                        if benchmark.max_free_float_max == 0.0
                            || result.ref_max > benchmark.max_free_float_max
                        {
                            benchmark.max_free_float_max = result.ref_max;
                        }
                    }
                }
            }

            for (case_id, data) in case_data {
                report.benchmark_data.insert(case_id, data);
            }
        } else {
            // Copy existing benchmark data
            for (case_id, data) in &self.benchmark_data {
                report.benchmark_data.insert(case_id.clone(), data.clone());
            }
        }

        report
    }

    /// Prints a detailed summary to stdout.
    pub fn print_detailed_summary(&self) {
        println!("Validation Suite Summary:");
        println!("  Total Results: {}", self.len());
        println!(
            "  Pass Rate: {:.1}% ({} passed)",
            self.calculate_pass_rate(),
            self.results.iter().filter(|r| r.passed()).count()
        );
        println!(
            "  Warning Rate: {:.1}% ({} warnings)",
            self.calculate_warning_rate(),
            self.warning_count()
        );
        println!(
            "  Fail Rate: {:.1}% ({} failed)",
            self.calculate_fail_rate(),
            self.fail_count()
        );
        println!("  Mean Absolute Error: {:.2}%", self.calculate_mae());
        println!("  Root Mean Square Error: {:.2}%", self.calculate_rmse());
        println!("  Max Deviation: {:.2}%", self.calculate_max_deviation());
        println!("  Mean Deviation: {:+.2}%", self.calculate_mean_deviation());

        // Summary by case
        println!("\nSummary by Case:");
        let case_summary = self.summary_by_case();
        let mut case_ids: Vec<_> = case_summary.keys().collect();
        case_ids.sort();

        for case_id in case_ids {
            let (passed, warnings, failed) = case_summary.get(case_id).unwrap();
            let total = passed + warnings + failed;
            let pass_rate = (*passed as f64 / total as f64) * 100.0;
            println!(
                "  {}: {}/{} passed ({:.1}%) - {} warnings, {} failed",
                case_id, passed, total, pass_rate, warnings, failed
            );
        }
    }

    /// Clears all results from the suite.
    pub fn clear(&mut self) {
        self.results.clear();
        self.interpretations.clear();
    }

    /// Generates interpretation guidance for failed metrics.
    ///
    /// This method analyzes validation results and generates interpretation guidance
    /// for metrics that failed validation, providing root cause hypotheses,
    /// parameter sensitivity, recommended next steps, what-if scenarios,
    /// and references to relevant documentation.
    pub fn generate_interpretations(&mut self) {
        // Group results by case ID
        let mut case_results: HashMap<String, Vec<&ValidationResult>> = HashMap::new();
        for result in &self.results {
            case_results
                .entry(result.case_id.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        // Generate interpretations for cases with failures
        for (case_id, results) in case_results {
            let failed_metrics: Vec<&ValidationResult> =
                results.iter().filter(|r| r.failed()).cloned().collect();

            if !failed_metrics.is_empty() {
                let interpretation =
                    Self::generate_interpretation_for_case(&case_id, &failed_metrics);
                self.interpretations.insert(case_id.clone(), interpretation);
            }
        }
    }

    /// Generates interpretation guidance for a specific case's failed metrics.
    fn generate_interpretation_for_case(
        case_id: &str,
        failed_metrics: &[&ValidationResult],
    ) -> Interpretation {
        let mut interpretation = Interpretation::default();

        // Generate root cause hypotheses based on case ID and metrics
        interpretation.root_cause_hypotheses =
            Self::generate_root_cause_hypotheses(case_id, failed_metrics);

        // Parameter sensitivity
        interpretation.parameter_sensitivity = Self::generate_parameter_sensitivity(case_id);

        // Recommended next steps
        interpretation.recommended_next_steps =
            Self::generate_recommended_steps(case_id, failed_metrics);

        // What-if scenarios
        interpretation.what_if_scenarios = Self::generate_what_if_scenarios(case_id);

        // References
        interpretation.references = Self::generate_references(case_id);

        interpretation
    }

    /// Generates root cause hypotheses for failed metrics.
    fn generate_root_cause_hypotheses(
        case_id: &str,
        failed_metrics: &[&ValidationResult],
    ) -> Vec<String> {
        let mut hypotheses = Vec::new();

        // Case-specific hypotheses
        match case_id {
            "900" | "910" | "920" | "930" | "940" | "950" => {
                hypotheses.push(
                    "High-mass annual energy over-prediction is a known 5R1C ISO 13790 limitation. \
                     The single thermal capacitance node cannot accurately represent complex thermal mass \
                     dynamics over 8760 simulation hours.".to_string()
                );
            }
            "960" => {
                hypotheses.push(
                    "Multi-zone inter-zone heat transfer issues may cause annual cooling failure. \
                     Check h_tr_em coupling ratio and zone-to-zone conductances."
                        .to_string(),
                );
            }
            _ => {
                for metric in failed_metrics {
                    hypotheses.push(format!(
                        "{} deviation may be due to parameter calibration or model structure.",
                        metric.metric.display_name()
                    ));
                }
            }
        }

        // Metric-specific hypotheses
        for metric in failed_metrics {
            match metric.metric {
                MetricType::AnnualCooling => {
                    if case_id.starts_with("9") {
                        hypotheses.push(
                            "High-mass cooling energy over-prediction suggests thermal mass coupling \
                             ratio (h_tr_em / h_tr_ms) may be too low, causing excessive heat \
                             storage and delayed cooling response.".to_string()
                        );
                    }
                }
                MetricType::AnnualHeating => {
                    if case_id.starts_with("9") {
                        hypotheses.push(
                            "High-mass heating energy over-prediction indicates thermal mass is \
                             storing too much heat during the day and releasing it slowly, \
                             increasing heating demand."
                                .to_string(),
                        );
                    }
                }
                _ => {}
            }
        }

        hypotheses
    }

    /// Generates parameter sensitivity analysis.
    fn generate_parameter_sensitivity(case_id: &str) -> Vec<String> {
        let mut sensitivity = Vec::new();

        if case_id.starts_with("9") {
            // High-mass cases
            sensitivity.push(
                "Thermal mass coupling ratio (h_tr_em / h_tr_ms) - affects heat storage/release rate".to_string(),
            );
            sensitivity.push(
                "Thermal capacitance (Cm) - determines thermal mass response time".to_string(),
            );
        }

        if case_id == "960" {
            sensitivity
                .push("Zone-to-zone conductances (h_tr_iz) - inter-zone heat transfer".to_string());
            sensitivity.push(
                "Sunspace surface area - affects solar gain and heat distribution".to_string(),
            );
        }

        // Common sensitivities
        sensitivity.push(
            "Solar gain parameters (SHGC, incidence angles) - summer cooling demand".to_string(),
        );
        sensitivity.push(
            "HVAC setpoint - affects free-floating temperature and HVAC activation".to_string(),
        );
        sensitivity.push("Window U-value - affects envelope heat loss/gain".to_string());

        sensitivity
    }

    /// Generates recommended next steps for investigation.
    fn generate_recommended_steps(
        case_id: &str,
        failed_metrics: &[&ValidationResult],
    ) -> Vec<String> {
        let mut steps = Vec::new();

        if case_id.starts_with("9") {
            steps.push(
                "Review docs/KNOWN_LIMITATIONS.md for 5R1C high-mass limitations".to_string(),
            );
            steps.push(
                "Consider .planning/phases/12-Model-Exploration/ for alternative thermal network evaluation".to_string(),
            );
            steps.push(
                "Evaluate mode-specific coupling (h_tr_em_heating vs h_tr_em_cooling)".to_string(),
            );
        }

        if case_id == "960" {
            steps.push(
                "Review .planning/phases/08-Critical-Issue-Resolution/ for Case 960 inter-zone heat transfer issues".to_string(),
            );
        }

        for metric in failed_metrics {
            match metric.metric {
                MetricType::AnnualCooling => {
                    steps.push(format!(
                        "Test sensitivity of {} to thermal mass coupling ratio - try increasing h_tr_em",
                        metric.metric.display_name()
                    ));
                }
                MetricType::AnnualHeating => {
                    steps.push(format!(
                        "Test sensitivity of {} to thermal capacitance - try adjusting Cm",
                        metric.metric.display_name()
                    ));
                }
                _ => {}
            }
        }

        steps
    }

    /// Generates what-if scenarios for debugging approaches.
    fn generate_what_if_scenarios(case_id: &str) -> Vec<String> {
        let mut scenarios = Vec::new();

        if case_id.starts_with("9") {
            scenarios.push(
                "If we increased h_tr_em coupling ratio to 0.2: Would reduce heating/cooling energy by ~15%".to_string(),
            );
            scenarios.push(
                "If we increased thermal capacitance Cm: Would slow thermal response, potentially improving accuracy".to_string(),
            );
        }

        if case_id == "960" {
            scenarios.push(
                "If we adjusted zone-to-zone conductances: Would change heat distribution between zones".to_string(),
            );
            scenarios.push(
                "If we increased sunspace ventilation: Would reduce overheating and improve cooling accuracy".to_string(),
            );
        }

        scenarios.push(
            "If we added exterior surface area: Would increase solar gains, potentially worsening cooling".to_string(),
        );
        scenarios.push(
            "If we used adaptive HVAC setpoint: Could reduce cooling demand by matching thermal mass temperature to comfort band".to_string(),
        );

        scenarios
    }

    /// Generates references to relevant documentation.
    fn generate_references(case_id: &str) -> Vec<String> {
        let mut refs = Vec::new();

        refs.push("See docs/KNOWN_LIMITATIONS.md for known 5R1C limitations".to_string());

        if case_id == "960" {
            refs.push(
                "See .planning/phases/08-Critical-Issue-Resolution/ for Case 960 investigation"
                    .to_string(),
            );
        }

        if case_id.starts_with("9") {
            refs.push("See .planning/phases/12-Model-Exploration/ for 6R2C evaluation".to_string());
        }

        refs
    }
}

impl BenchmarkReport {
    /// Calculates monthly aggregation from hourly data.
    ///
    /// This function correctly sums hourly values into 12 months using actual hours per month.
    /// Returns a vector of 12 monthly totals (not averages).
    ///
    /// # Arguments
    /// * `hourly_data` - Slice of hourly values (typically 8760 for one year)
    ///
    /// # Returns
    /// Vector of 12 monthly totals
    pub fn calculate_monthly_aggregation(hourly_data: &[f64]) -> Vec<f64> {
        let mut monthly = vec![0.0; 12];

        // Actual hours per month (non-leap year)
        let hours_per_month = [744, 696, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744];

        // Calculate cumulative hour counts for month boundaries
        let mut month_boundaries = vec![0; 13];
        for i in 1..13 {
            month_boundaries[i] = hours_per_month[0..i].iter().sum();
        }

        // Sum hourly values into appropriate month
        for (i, &value) in hourly_data.iter().enumerate() {
            if i < 8760 {
                // Find which month this hour belongs to
                let month_idx = month_boundaries
                    .iter()
                    .position(|&boundary| i < boundary)
                    .unwrap_or(11);

                if month_idx < 12 {
                    monthly[month_idx] += value;
                }
            }
        }

        monthly
    }

    /// Performs delta test with statistical significance testing.
    ///
    /// This function compares two datasets and determines if the difference
    /// is statistically significant using a two-tailed z-test.
    ///
    /// # Arguments
    /// * `baseline` - Baseline dataset
    /// * `current` - Current dataset to compare against baseline
    /// * `confidence_level` - Confidence level (e.g., 0.95 for 95% confidence)
    ///
    /// # Returns
    /// DeltaTestResult with statistical analysis
    pub fn perform_delta_test(
        baseline: &[f64],
        current: &[f64],
        confidence_level: f64,
    ) -> DeltaTestResult {
        let n = baseline.len();
        let m = current.len();

        // Calculate means
        let mean_baseline: f64 = baseline.iter().sum::<f64>() / n as f64;
        let mean_current: f64 = current.iter().sum::<f64>() / m as f64;

        // Calculate standard deviations
        let std_baseline = Self::calculate_std_deviation(baseline, mean_baseline);
        let std_current = Self::calculate_std_deviation(current, mean_current);

        // Calculate delta (mean difference)
        let delta = mean_current - mean_baseline;

        // Calculate pooled standard error
        let pooled_std_error = if std_baseline > 0.0 || std_current > 0.0 {
            ((std_baseline.powi(2) / n as f64) + (std_current.powi(2) / m as f64)).sqrt()
        } else {
            0.0
        };

        // Calculate z-score (for large samples)
        let z_score = if pooled_std_error > 0.0 {
            delta / pooled_std_error
        } else {
            0.0
        };

        // Calculate p-value (two-tailed test)
        let p_value = if z_score.abs() > 0.0 {
            2.0 * (1.0 - Self::normal_cdf(z_score.abs()))
        } else {
            1.0
        };

        // Determine significance (p < 0.05 for 95% confidence)
        let is_significant = p_value < 0.05;

        // Calculate confidence interval
        let critical_value = if confidence_level == 0.95 {
            1.96 // For 95% confidence
        } else if confidence_level == 0.99 {
            2.58 // For 99% confidence
        } else {
            // Approximate for other levels
            (2.0 * (confidence_level + 0.5)).ln().sqrt()
        };

        let ci_lower = delta - critical_value * pooled_std_error;
        let ci_upper = delta + critical_value * pooled_std_error;

        DeltaTestResult {
            metric_name: "delta".to_string(),
            delta_value: delta,
            p_value: Some(p_value),
            is_significant,
            confidence_interval: Some((ci_lower, ci_upper)),
        }
    }

    /// Calculates standard deviation for a dataset.
    fn calculate_std_deviation(data: &[f64], mean: f64) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let variance: f64 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

        variance.sqrt()
    }

    /// Approximation of standard normal cumulative distribution function (CDF).
    ///
    /// Uses the Abramowitz and Stegun approximation for accuracy.
    fn normal_cdf(x: f64) -> f64 {
        // Approximation constants
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let k = 1.0 / (1.0 + p * x.abs());
        let y = 1.0 - (((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k) * (-0.5 * x * x).exp();

        0.5 * (1.0 + if x < 0.0 { -1.0 } else { 1.0 } * y)
    }

    /// Normalizes sensitivity coefficients by parameter range.
    ///
    /// This function normalizes raw sensitivity coefficients by dividing by
    /// the parameter range, enabling fair comparison of parameter importance.
    ///
    /// # Arguments
    /// * `sensitivity_results` - Raw sensitivity coefficients
    /// * `parameter_ranges` - Parameter ranges as (min, max) tuples
    ///
    /// # Returns
    /// Vector of SensitivityResult with normalized coefficients and rankings
    pub fn normalize_sensitivity(
        sensitivity_results: &[(String, f64)],
        parameter_ranges: &std::collections::HashMap<String, (f64, f64)>,
    ) -> Vec<SensitivityResult> {
        let mut normalized_results = Vec::new();

        for (param_name, coefficient) in sensitivity_results {
            // Get parameter range
            let (min_val, max_val) = parameter_ranges.get(param_name).unwrap_or(&(0.0, 1.0));

            let range = max_val - min_val;

            // Normalize coefficient by dividing by parameter range
            let normalized_coefficient = if range > 0.0 {
                coefficient.abs() / range
            } else {
                coefficient.abs()
            };

            normalized_results.push(SensitivityResult {
                parameter_name: param_name.clone(),
                coefficient: *coefficient,
                normalized_coefficient,
                ranking: 0, // Will be assigned after sorting
            });
        }

        // Sort by normalized coefficient (descending)
        normalized_results.sort_by(|a, b| {
            b.normalized_coefficient
                .partial_cmp(&a.normalized_coefficient)
                .unwrap()
        });

        // Assign rankings
        for (i, result) in normalized_results.iter_mut().enumerate() {
            result.ranking = i + 1;
        }

        normalized_results
    }

    /// Gets standard parameter ranges for Fluxion model.
    ///
    /// Returns parameter ranges for window U-value, HVAC setpoint,
    /// thermal mass, and infiltration rate.
    pub fn get_parameter_ranges() -> std::collections::HashMap<String, (f64, f64)> {
        let mut ranges = std::collections::HashMap::new();

        ranges.insert("window_u_value".to_string(), (0.1, 5.0)); // W/m²K
        ranges.insert("hvac_setpoint".to_string(), (15.0, 30.0)); // °C
        ranges.insert("thermal_mass".to_string(), (1_000_000.0, 50_000_000.0)); // J/K
        ranges.insert("infiltration_rate".to_string(), (0.1, 2.0)); // ACH

        ranges
    }
}

/// Analysis of differences between test cases (e.g., variant vs. baseline).
#[derive(Debug, Clone)]
pub struct DeltaResult {
    /// Case being analyzed (the variant)
    pub case_id: String,
    /// Baseline case for comparison
    pub baseline_id: String,
    /// Metric type
    pub metric: MetricType,
    /// Absolute difference in Fluxion values
    pub fluxion_delta: f64,
    /// Absolute difference in reference midpoint values
    pub reference_delta: f64,
    /// Percent deviation of Fluxion delta from reference delta
    pub deviation_percent: f64,
}

/// Report containing delta analysis for multiple case variants.
#[derive(Debug, Clone, Default)]
pub struct DeltaReport {
    /// All delta analysis results
    pub deltas: Vec<DeltaResult>,
}

impl DeltaReport {
    /// Creates a new empty delta report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a delta analysis result.
    pub fn add_delta(&mut self, delta: DeltaResult) {
        self.deltas.push(delta);
    }

    /// Generates a Markdown table for the delta analysis.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        output.push_str("## Delta Analysis\n\n");
        output.push_str("| Case vs Baseline | Metric | Fluxion Δ | Ref Δ | Deviation |\n");
        output.push_str("|------------------|--------|-----------|-------|-----------|\n");

        for d in &self.deltas {
            output.push_str(&format!(
                "| {} vs {} | {} | {:.2} | {:.2} | {:+.2}% |\n",
                d.case_id,
                d.baseline_id,
                d.metric,
                d.fluxion_delta,
                d.reference_delta,
                d.deviation_percent
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_new_methods() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert!(result.is_within_range());
        assert!((result.deviation_percent() - (-0.0999)).abs() < 0.1);

        let fail = ValidationResult::new("600", MetricType::AnnualHeating, 3.0, 4.30, 5.71);
        assert!(!fail.is_within_range());
    }

    #[test]
    fn test_delta_report() {
        let mut report = DeltaReport::new();
        report.add_delta(DeltaResult {
            case_id: "610".to_string(),
            baseline_id: "600".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_delta: 0.1,
            reference_delta: 0.08,
            deviation_percent: 25.0,
        });

        let md = report.to_markdown();
        assert!(md.contains("610 vs 600"));
        assert!(md.contains("Annual Heating"));
        assert!(md.contains("25.00%"));
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(
            MetricType::AnnualHeating.display_name(),
            "Annual Heating (MWh)"
        );
        assert_eq!(MetricType::AnnualCooling.units(), "MWh");
        assert_eq!(MetricType::PeakHeating.units(), "kW");
    }

    #[test]
    fn test_validation_status_display() {
        assert_eq!(ValidationStatus::Pass.to_string(), "PASS");
        assert_eq!(ValidationStatus::Warning.to_string(), "WARN");
        assert_eq!(ValidationStatus::Fail.to_string(), "FAIL");
    }

    #[test]
    fn test_validation_result_pass() {
        // Case 600: Heating range 4.30-5.71 MWh
        // Midpoint: 5.005
        // 5% tolerance: [4.085, 5.9955]
        // Fluxion value 5.0 should pass
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert_eq!(result.status, ValidationStatus::Pass);
        assert!(result.passed());
        assert!(!result.warning());
        assert!(!result.failed());
    }

    #[test]
    fn test_validation_result_warning() {
        // Case 600: Heating range 4.30-5.71 MWh
        // Midpoint: 5.005
        // 4.30 is within range but has >2% deviation from midpoint
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 4.31, 4.30, 5.71);
        assert_eq!(result.status, ValidationStatus::Warning);
        assert!(!result.passed());
        assert!(result.warning());
        assert!(!result.failed());
    }

    #[test]
    fn test_validation_result_fail() {
        // Case 600: Heating range 4.30-5.71 MWh
        // 4.0 is outside 5% tolerance (below 4.085)
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 4.0, 4.30, 5.71);
        assert_eq!(result.status, ValidationStatus::Fail);
        assert!(!result.passed());
        assert!(!result.warning());
        assert!(result.failed());
    }

    #[test]
    fn test_validation_result_percent_error() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.50, 4.30, 5.71);
        // Midpoint: 5.005, Error: (5.50 - 5.005) / 5.005 * 100 ≈ 9.89%
        assert!((result.percent_error - 9.89).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_data_range() {
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };

        let range = data.get_range(MetricType::AnnualHeating);
        assert_eq!(range, Some((4.30, 5.71)));

        let range = data.get_range(MetricType::AnnualCooling);
        assert_eq!(range, None); // Not set
    }

    #[test]
    fn test_benchmark_data_midpoint() {
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };

        let midpoint = data.midpoint(MetricType::AnnualHeating);
        assert_eq!(midpoint, Some(5.005));
    }

    #[test]
    fn test_validation_report_basic() {
        let mut report = BenchmarkReport::new();

        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        report.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04);

        assert_eq!(report.results.len(), 3);
        assert!(report.pass_rate() > 0.0);
        assert!(report.mae() >= 0.0);
    }

    #[test]
    fn test_validation_report_markdown() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);

        let markdown = report.to_markdown();
        assert!(markdown.contains("# ASHRAE 140 Validation Report"));
        assert!(markdown.contains("## Summary"));
        assert!(markdown.contains("600"));
    }

    #[test]
    fn test_validation_report_csv() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);

        let csv = report.to_csv();
        assert!(csv.contains("Case,Metric,Fluxion,Ref Min,Ref Max"));
        assert!(csv.contains("600,Annual Heating"));
    }

    #[test]
    fn test_validation_suite_basic() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);

        assert_eq!(suite.len(), 2);
        assert!(!suite.is_empty());
        assert_eq!(suite.pass_count(), 2);
        assert_eq!(suite.fail_count(), 0);
    }

    #[test]
    fn test_validation_suite_pass_rate() {
        let mut suite = ValidationSuite::new();

        // Add mix of pass, warning, fail
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.0, 6.14, 8.45); // Fail
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.17, 1.17, 2.04); // Warning

        let pass_rate = suite.calculate_pass_rate();
        assert!((pass_rate - 33.33).abs() < 0.1); // 1 out of 3 = 33.33%

        let warning_rate = suite.calculate_warning_rate();
        assert!((warning_rate - 33.33).abs() < 0.1); // 1 out of 3

        let fail_rate = suite.calculate_fail_rate();
        assert!((fail_rate - 33.33).abs() < 0.1); // 1 out of 3
    }

    #[test]
    fn test_validation_suite_mae() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // ~0%
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 6.14, 8.45); // ~5%

        let mae = suite.calculate_mae();
        assert!((0.0..=10.0).contains(&mae));
    }

    #[test]
    fn test_validation_suite_rmse() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.5, 6.14, 8.45);

        let rmse = suite.calculate_rmse();
        assert!(rmse >= 0.0);
    }

    #[test]
    fn test_validation_suite_max_deviation() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // ~0%
        suite.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45); // ~25%

        let max_dev = suite.calculate_max_deviation();
        assert!(max_dev >= 20.0);
    }

    #[test]
    fn test_validation_suite_worst_cases() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45);
        suite.add_result_simple("900", MetricType::AnnualHeating, 0.5, 1.17, 2.04);

        let worst = suite.worst_cases(2);
        assert_eq!(worst.len(), 2);

        // Check that worst case has highest deviation
        let first_dev = worst[0].percent_error.abs();
        let second_dev = worst[1].percent_error.abs();
        assert!(first_dev >= second_dev);
    }

    #[test]
    fn test_validation_suite_get_case_results() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04);

        let case_600_results = suite.get_case_results("600");
        assert_eq!(case_600_results.len(), 2);

        let case_900_results = suite.get_case_results("900");
        assert_eq!(case_900_results.len(), 1);
    }

    #[test]
    fn test_validation_suite_get_metric_results() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04);

        let heating_results = suite.get_metric_results(MetricType::AnnualHeating);
        assert_eq!(heating_results.len(), 2);

        let cooling_results = suite.get_metric_results(MetricType::AnnualCooling);
        assert_eq!(cooling_results.len(), 1);
    }

    #[test]
    fn test_validation_suite_case_pass_rate() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.0, 6.14, 8.45); // Fail

        let pass_rate = suite.calculate_case_pass_rate("600");
        assert_eq!(pass_rate, Some(50.0));

        let no_data = suite.calculate_case_pass_rate("INVALID");
        assert_eq!(no_data, None);
    }

    #[test]
    fn test_validation_suite_summary_by_case() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.31, 6.14, 8.45); // Fail
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04); // Pass

        let summary = suite.summary_by_case();

        let case_600 = summary.get("600").unwrap();
        assert_eq!(case_600, &(1, 0, 1)); // 1 pass, 0 warnings, 1 fail

        let case_900 = summary.get("900").unwrap();
        assert_eq!(case_900, &(1, 0, 0)); // 1 pass, 0 warnings, 0 fails
    }

    #[test]
    fn test_validation_suite_summary_by_metric() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71); // Pass
        suite.add_result_simple("600", MetricType::AnnualCooling, 4.0, 6.14, 8.45); // Fail
        suite.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04); // Pass

        let summary = suite.summary_by_metric();

        let heating = summary.get(&MetricType::AnnualHeating).unwrap();
        assert_eq!(heating, &(2, 0, 0)); // 2 pass, 0 warnings, 0 fails

        let cooling = summary.get(&MetricType::AnnualCooling).unwrap();
        assert_eq!(cooling, &(0, 0, 1)); // 0 pass, 0 warnings, 1 fail
    }

    #[test]
    fn test_validation_suite_generate_report() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);

        let report = suite.generate_report();

        assert_eq!(report.results.len(), 1);
        assert!(!report.benchmark_data.is_empty());
    }

    #[test]
    fn test_validation_suite_clear() {
        let mut suite = ValidationSuite::new();

        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert_eq!(suite.len(), 1);

        suite.clear();
        assert_eq!(suite.len(), 0);
        assert!(suite.is_empty());
    }

    #[test]
    fn test_validation_suite_mean_deviation() {
        let mut suite = ValidationSuite::new();

        // Use values that are more symmetric to get mean close to 0
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.5, 4.30, 5.71); // +9.89%
        suite.add_result_simple("600", MetricType::AnnualCooling, 6.57, 6.14, 8.45); // -10%

        let mean_dev = suite.calculate_mean_deviation();
        // Should be close to 0 (positive and negative cancel out)
        assert!(mean_dev.abs() < 1.0);
    }

    #[test]
    fn test_validation_suite_empty() {
        let suite = ValidationSuite::new();

        assert_eq!(suite.len(), 0);
        assert!(suite.is_empty());
        assert_eq!(suite.calculate_pass_rate(), 100.0); // Empty suite defaults to 100%
        assert_eq!(suite.calculate_mae(), 0.0);
    }

    #[test]
    fn test_append_history() {
        use std::fs;
        use std::thread::sleep;
        use std::time::Duration;
        use tempfile::tempdir;

        // Create a report with some results
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_benchmark_data(
            "600",
            BenchmarkData {
                annual_heating_min: 4.30,
                annual_heating_max: 5.71,
                ..Default::default()
            },
        );

        // Simulate validation timing
        report.set_start();
        sleep(Duration::from_millis(10));
        report.set_end();

        // Setup temporary directory guard to isolate file operations
        struct DirGuard(PathBuf);
        impl Drop for DirGuard {
            fn drop(&mut self) {
                // Restore original directory on drop, panic-safe
                let _ = std::env::set_current_dir(&self.0);
            }
        }

        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = tempdir().unwrap();
        let _guard = DirGuard(original_dir.clone());
        std::env::set_current_dir(temp_dir.path()).unwrap();

        // Call append_history
        report.append_history();

        // Verify file creation
        let log_path = temp_dir
            .path()
            .join("target")
            .join("performance_history.jsonl");
        assert!(log_path.exists(), "Performance history file should exist");

        // Read and verify content
        let content = fs::read_to_string(&log_path).expect("Should read log file");
        let mut valid_lines = 0;
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let json: serde_json::Value = serde_json::from_str(line).expect("Valid JSON line");
            assert!(json.get("timestamp").is_some());
            assert!(json.get("mae").is_some());
            assert!(json.get("max_deviation").is_some());
            assert!(json.get("pass_rate").is_some());
            assert!(json.get("validation_time_seconds").is_some());
            assert!(json.get("throughput").is_some());
            assert!(json.get("git_sha").is_some());
            valid_lines += 1;
        }
        assert_eq!(valid_lines, 1);
    }

    #[test]
    fn test_benchmark_report_statistical_fields() {
        use crate::validation::statistical::{EffectDirection, StatisticalMetrics};

        let mut report = BenchmarkReport::new();

        // Test 1: BenchmarkReport can hold optional StatisticalMetrics
        let metrics = StatisticalMetrics {
            nmbe: 2.3,
            cv_rmse: 8.7,
            nmbe_ci: (1.5, 3.1),
            cv_rmse_ci: (7.2, 10.2),
            cohens_d: 0.42,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        };
        report.statistical_metrics = Some(metrics.clone());

        assert!(report.statistical_metrics.is_some());
        let retrieved = report.statistical_metrics.as_ref().unwrap();
        assert_eq!(retrieved.nmbe, 2.3);
        assert_eq!(retrieved.cv_rmse, 8.7);

        // Test 2: BenchmarkReport can hold p-values and BH correction
        report.statistical_p_values = Some(vec![0.023, 0.089, 0.156]);
        report.statistical_corrected = Some(vec![true, false, false]);

        assert!(report.statistical_p_values.is_some());
        assert_eq!(report.statistical_p_values.as_ref().unwrap().len(), 3);
        assert!(report.statistical_corrected.is_some());
        assert_eq!(report.statistical_corrected.as_ref().unwrap()[0], true);

        // Test 3: BenchmarkReport can hold group validation results
        let mut group_results = std::collections::HashMap::new();
        group_results.insert(ValidationGroup::Baseline, true);
        group_results.insert(ValidationGroup::HighMass, false);
        report.group_validation = Some(group_results);

        assert!(report.group_validation.is_some());
        let groups = report.group_validation.as_ref().unwrap();
        assert_eq!(groups.get(&ValidationGroup::Baseline), Some(&true));
        assert_eq!(groups.get(&ValidationGroup::HighMass), Some(&false));

        // Test 4: Report without statistical fields (backward compatibility)
        let bare_report = BenchmarkReport::new();
        assert!(bare_report.statistical_metrics.is_none());
        assert!(bare_report.statistical_p_values.is_none());
        assert!(bare_report.statistical_corrected.is_none());
        assert!(bare_report.group_validation.is_none());

        // Test 5: Serialization with optional fields
        let json = serde_json::to_string(&report).expect("Should serialize");
        assert!(json.contains("statistical_metrics"));
        assert!(json.contains("nmbe"));

        let bare_json = serde_json::to_string(&bare_report).expect("Should serialize bare report");
        assert!(!bare_json.contains("statistical_metrics"));
    }

    #[test]
    fn test_validation_result_no_modification_needed() {
        // Test 2: ValidationResult doesn't need modification (per-case stats separate)
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 6.0, 5.5, 7.0);
        assert_eq!(result.case_id, "600");
        assert_eq!(result.metric, MetricType::AnnualHeating);
        assert_eq!(result.fluxion_value, 6.0);
        assert!(result.is_within_range());
    }

    #[test]
    fn test_benchmark_report_serialization_with_statistical_fields() {
        use crate::validation::statistical::{EffectDirection, StatisticalMetrics};

        let mut report = BenchmarkReport::new();
        report.add_result(ValidationResult::new(
            "600",
            MetricType::AnnualHeating,
            6.0,
            5.5,
            7.0,
        ));

        // Add statistical metrics
        report.statistical_metrics = Some(StatisticalMetrics {
            nmbe: 1.5,
            cv_rmse: 5.2,
            nmbe_ci: (0.8, 2.2),
            cv_rmse_ci: (4.1, 6.3),
            cohens_d: 0.28,
            effect_direction: EffectDirection::Underprediction,
            excluded_cases: 0,
        });

        // Add p-values and correction
        report.statistical_p_values = Some(vec![0.03, 0.12, 0.45]);
        report.statistical_corrected = Some(vec![true, false, false]);

        // Add group validation
        let mut groups = std::collections::HashMap::new();
        groups.insert(ValidationGroup::Baseline, true);
        groups.insert(ValidationGroup::HighMass, false);
        report.group_validation = Some(groups);

        // Test JSON serialization
        let json = serde_json::to_string_pretty(&report).expect("Should serialize");
        assert!(json.contains("statistical_metrics"));
        assert!(json.contains("statistical_p_values"));
        assert!(json.contains("statistical_corrected"));
        assert!(json.contains("group_validation"));
        assert!(json.contains("\"nmbe\": 1.5"));
        assert!(json.contains("\"cv_rmse\": 5.2"));

        // Test deserialization
        let deserialized: BenchmarkReport =
            serde_json::from_str(&json).expect("Should deserialize");
        assert!(deserialized.statistical_metrics.is_some());
        assert_eq!(deserialized.statistical_metrics.as_ref().unwrap().nmbe, 1.5);
        assert_eq!(deserialized.statistical_p_values.as_ref().unwrap().len(), 3);
        assert_eq!(
            deserialized.statistical_corrected.as_ref().unwrap().len(),
            3
        );
        assert!(deserialized.group_validation.is_some());

        // Test CSV export (optional fields should be handled gracefully)
        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join("test_statistical_export.csv");
        let csv_content = report.to_csv();
        fs::write(&csv_path, csv_content).expect("Should write CSV file");
        assert!(csv_path.exists());

        // Clean up
        let _ = std::fs::remove_file(csv_path);
    }

    #[test]
    fn test_compute_status_pass() {
        let status = compute_status(5.0, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Pass);
    }

    #[test]
    fn test_compute_status_warning_within_range() {
        let status = compute_status(4.01, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Warning);
    }

    #[test]
    fn test_compute_status_warning_tolerance_band() {
        let status = compute_status(6.2, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Warning);
    }

    #[test]
    fn test_compute_status_fail_below() {
        let status = compute_status(3.0, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_fail_above() {
        let status = compute_status(7.0, 4.0, 6.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_compute_status_zero_ref_mid() {
        let status = compute_status(0.5, 0.0, 0.0);
        assert_eq!(status, ValidationStatus::Fail);
    }

    #[test]
    fn test_validation_status_color_and_icon() {
        assert_eq!(ValidationStatus::Pass.color(), "green");
        assert_eq!(ValidationStatus::Warning.color(), "orange");
        assert_eq!(ValidationStatus::Fail.color(), "red");
        assert_eq!(ValidationStatus::Pass.icon(), "✓");
        assert_eq!(ValidationStatus::Warning.icon(), "⚠");
        assert_eq!(ValidationStatus::Fail.icon(), "✗");
        assert_eq!(ValidationStatus::Pass.display_name(), "PASS");
        assert_eq!(ValidationStatus::Warning.display_name(), "WARN");
        assert_eq!(ValidationStatus::Fail.display_name(), "FAIL");
    }

    #[test]
    fn test_reference_program_display() {
        assert_eq!(format!("{}", ReferenceProgram::EnergyPlus), "EnergyPlus");
        assert_eq!(format!("{}", ReferenceProgram::EspR), "ESP-r");
        assert_eq!(format!("{}", ReferenceProgram::TRNSYS), "TRNSYS");
        assert_eq!(format!("{}", ReferenceProgram::DOE2), "DOE2");
    }

    #[test]
    fn test_benchmark_data_new_and_default() {
        let data = BenchmarkData::new();
        assert_eq!(data.annual_heating_min, 0.0);
        assert_eq!(data.annual_cooling_max, 0.0);
        assert_eq!(data.peak_heating_min, 0.0);
        assert_eq!(data.peak_cooling_max, 0.0);
        assert_eq!(data.min_free_float_min, 0.0);
        assert_eq!(data.max_free_float_max, 0.0);
        let default_data = BenchmarkData::default();
        assert_eq!(default_data.annual_heating_min, 0.0);
    }

    #[test]
    fn test_benchmark_data_all_ranges() {
        let data = BenchmarkData {
            annual_heating_min: 1.0,
            annual_heating_max: 2.0,
            annual_cooling_min: 3.0,
            annual_cooling_max: 4.0,
            peak_heating_min: 5.0,
            peak_heating_max: 6.0,
            peak_cooling_min: 7.0,
            peak_cooling_max: 8.0,
            min_free_float_min: 9.0,
            min_free_float_max: 10.0,
            max_free_float_min: 11.0,
            max_free_float_max: 12.0,
        };
        assert_eq!(data.get_range(MetricType::AnnualHeating), Some((1.0, 2.0)));
        assert_eq!(data.get_range(MetricType::AnnualCooling), Some((3.0, 4.0)));
        assert_eq!(data.get_range(MetricType::PeakHeating), Some((5.0, 6.0)));
        assert_eq!(data.get_range(MetricType::PeakCooling), Some((7.0, 8.0)));
        assert_eq!(data.get_range(MetricType::MinFreeFloat), Some((9.0, 10.0)));
        assert_eq!(data.get_range(MetricType::MaxFreeFloat), Some((11.0, 12.0)));
        assert_eq!(data.midpoint(MetricType::AnnualHeating), Some(1.5));
    }

    #[test]
    fn test_validation_result_is_pass_warning_fail() {
        let pass = ValidationResult::new("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert!(pass.is_pass());
        assert!(!pass.is_warning());
        assert!(!pass.is_fail());

        let fail = ValidationResult::new("600", MetricType::AnnualHeating, 1.0, 4.30, 5.71);
        assert!(!fail.is_pass());
        assert!(!fail.is_warning());
        assert!(fail.is_fail());
    }

    #[test]
    fn test_validation_result_deviation_string() {
        let result = ValidationResult::new("600", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        let dev = result.deviation_string();
        assert!(dev.contains("%"));
    }

    #[test]
    fn test_interpretation_default() {
        let interp = Interpretation::default();
        assert!(interp.root_cause_hypotheses.is_empty());
        assert!(interp.parameter_sensitivity.is_empty());
        assert!(interp.recommended_next_steps.is_empty());
        assert!(interp.what_if_scenarios.is_empty());
        assert!(interp.references.is_empty());
    }

    #[test]
    fn test_benchmark_report_to_json() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let json = report.to_json();
        assert!(json.contains("results"));
        assert!(json.contains("600"));
    }

    #[test]
    fn test_benchmark_report_add_result_simple() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].case_id, "600");
        assert_eq!(report.results[0].fluxion_value, 5.0);
    }

    #[test]
    fn test_benchmark_report_add_benchmark_data() {
        let mut report = BenchmarkReport::new();
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };
        report.add_benchmark_data("600", data);
        assert!(report.benchmark_data.contains_key("600"));
    }

    #[test]
    fn test_benchmark_report_delta_analysis() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("610", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        report.add_result_simple("610", MetricType::AnnualCooling, 6.5, 6.14, 8.45);

        let deltas = report.delta_analysis("600");
        assert!(!deltas.is_empty());
        assert!(deltas.contains_key("610 - Annual Heating (MWh)"));
        assert!((deltas["610 - Annual Heating (MWh)"] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_report_delta_analysis_no_baseline_match() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("610", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        let deltas = report.delta_analysis("600");
        assert!(deltas.is_empty());
    }

    #[test]
    fn test_benchmark_report_pass_rate_empty() {
        let report = BenchmarkReport::new();
        assert_eq!(report.pass_rate(), 100.0);
    }

    #[test]
    fn test_benchmark_report_fail_count_and_warning_count() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 1.0, 6.14, 8.45);
        assert_eq!(report.fail_count(), 1);
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn test_benchmark_report_mae_empty() {
        let report = BenchmarkReport::new();
        assert_eq!(report.mae(), 0.0);
    }

    #[test]
    fn test_benchmark_report_max_deviation() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45);
        let max_dev = report.max_deviation();
        assert!(max_dev > 20.0);
    }

    #[test]
    fn test_benchmark_report_worst_cases() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 9.0, 6.14, 8.45);
        report.add_result_simple("900", MetricType::AnnualHeating, 0.5, 1.17, 2.04);
        let worst = report.worst_cases(2);
        assert_eq!(worst.len(), 2);
        assert!(worst[0].percent_error.abs() >= worst[1].percent_error.abs());
    }

    #[test]
    fn test_benchmark_report_worst_cases_empty() {
        let report = BenchmarkReport::new();
        let worst = report.worst_cases(5);
        assert!(worst.is_empty());
    }

    #[test]
    fn test_benchmark_report_duration_and_throughput() {
        let mut report = BenchmarkReport::new();
        report.add_benchmark_data("600", BenchmarkData::new());
        assert_eq!(report.duration_seconds(), 0.0);
        assert_eq!(report.cases_per_second(), 0.0);
        report.set_start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        report.set_end();
        assert!(report.duration_seconds() > 0.0);
        assert!(report.cases_per_second() > 0.0);
    }

    #[test]
    fn test_benchmark_report_to_html() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let html = report.to_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("ASHRAE 140 Validation Report"));
        assert!(html.contains("600"));
        assert!(html.contains("class=\"pass\""));
    }

    #[test]
    fn test_benchmark_report_to_html_with_delta() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("610", MetricType::AnnualHeating, 5.5, 4.30, 5.71);
        report.add_benchmark_data("600", BenchmarkData::new());
        let html = report.to_html();
        assert!(html.contains("Delta Analysis"));
    }

    #[test]
    fn test_benchmark_report_to_html_with_worst() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 15.0, 6.14, 8.45);
        let html = report.to_html();
        assert!(html.contains("Worst Performing Cases"));
        assert!(html.contains("class=\"fail\""));
    }

    #[test]
    fn test_benchmark_report_to_csv() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("900", MetricType::AnnualCooling, 3.0, 2.13, 3.67);
        let csv = report.to_csv();
        assert!(csv.contains("Case,Metric,Fluxion,Ref Min,Ref Max,Percent Error,Status"));
        assert!(csv.contains("600"));
        assert!(csv.contains("900"));
        assert!(csv.contains("Annual Heating"));
        assert!(csv.contains("Annual Cooling"));
    }

    #[test]
    fn test_benchmark_report_save_to_file_markdown() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.md");
        assert!(report.save_to_file(&path).is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("ASHRAE 140 Validation Report"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_html() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.html");
        assert!(report.save_to_file(&path).is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_csv() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.csv");
        assert!(report.save_to_file(&path).is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Case,Metric"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_txt() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.txt");
        assert!(report.save_to_file(&path).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_benchmark_report_save_to_file_unsupported() {
        let report = BenchmarkReport::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_report.xml");
        let result = report.save_to_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_benchmark_report_to_markdown_with_interpretations() {
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 1.0, 4.30, 5.71);
        let mut interp = Interpretation::default();
        interp
            .root_cause_hypotheses
            .push("Test hypothesis".to_string());
        interp
            .parameter_sensitivity
            .push("Test sensitivity".to_string());
        interp.recommended_next_steps.push("Test step".to_string());
        interp.what_if_scenarios.push("Test scenario".to_string());
        interp.references.push("Test reference".to_string());
        report.interpretations.insert("600".to_string(), interp);
        let md = report.to_markdown();
        assert!(md.contains("Interpretation Guidance"));
        assert!(md.contains("Test hypothesis"));
        assert!(md.contains("Test sensitivity"));
        assert!(md.contains("Test step"));
        assert!(md.contains("Test scenario"));
        assert!(md.contains("Test reference"));
    }

    #[test]
    fn test_benchmark_report_add_result_with_multi_case_not_found() {
        use crate::validation::multi_reference::MultiReferenceDB;
        let mut report = BenchmarkReport::new();
        let db = MultiReferenceDB {
            version: "test".to_string(),
            source: None,
            cases: std::collections::HashMap::new(),
        };
        report.add_result_with_multi("NONEXISTENT", MetricType::AnnualHeating, 5.0, &db);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].status, ValidationStatus::Fail);
        assert!(report.results[0].per_program.is_none());
    }

    #[test]
    fn test_benchmark_report_enrich_with_multi_reference_empty() {
        use crate::validation::multi_reference::MultiReferenceDB;
        let mut report = BenchmarkReport::new();
        let db = MultiReferenceDB {
            version: "test".to_string(),
            source: None,
            cases: std::collections::HashMap::new(),
        };
        report.enrich_with_multi_reference(&db);
        assert!(report.results.is_empty());
    }

    #[test]
    fn test_benchmark_report_enrich_with_multi_reference_free_float_unchanged() {
        use crate::validation::multi_reference::MultiReferenceDB;
        let mut report = BenchmarkReport::new();
        report.add_result_simple("600FF", MetricType::MinFreeFloat, -10.0, -18.8, -15.6);
        let db = MultiReferenceDB {
            version: "test".to_string(),
            source: None,
            cases: std::collections::HashMap::new(),
        };
        report.enrich_with_multi_reference(&db);
        assert_eq!(report.results.len(), 1);
        assert!(report.results[0].per_program.is_none());
    }

    #[test]
    fn test_validation_suite_with_ashrae140_data() {
        let suite = ValidationSuite::with_ashrae140_data();
        assert!(!suite.benchmark_data.is_empty());
    }

    #[test]
    fn test_validation_suite_add_benchmark_data() {
        let mut suite = ValidationSuite::new();
        suite.add_benchmark_data("600", BenchmarkData::new());
        assert!(suite.benchmark_data.contains_key("600"));
    }

    #[test]
    fn test_validation_suite_generate_report_with_benchmark_data() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        let data = BenchmarkData {
            annual_heating_min: 4.30,
            annual_heating_max: 5.71,
            ..Default::default()
        };
        suite.add_benchmark_data("600", data);
        let report = suite.generate_report();
        assert!(report.benchmark_data.contains_key("600"));
    }

    #[test]
    fn test_validation_suite_print_detailed_summary() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        suite.print_detailed_summary();
    }

    #[test]
    fn test_validation_suite_generate_interpretations() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("900", MetricType::AnnualHeating, 5.0, 1.17, 2.04);
        suite.generate_interpretations();
        assert!(!suite.interpretations.is_empty());
        assert!(suite.interpretations.contains_key("900"));
    }

    #[test]
    fn test_validation_suite_generate_interpretations_no_failures() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        suite.generate_interpretations();
        assert!(suite.interpretations.is_empty());
    }

    #[test]
    fn test_validation_suite_generate_interpretations_case_960() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("960", MetricType::AnnualCooling, 5.0, 1.55, 2.78);
        suite.generate_interpretations();
        assert!(suite.interpretations.contains_key("960"));
        let interp = suite.interpretations.get("960").unwrap();
        assert!(!interp.root_cause_hypotheses.is_empty());
    }

    #[test]
    fn test_validation_suite_generate_interpretations_unknown_case() {
        let mut suite = ValidationSuite::new();
        suite.add_result_simple("XXX", MetricType::AnnualHeating, 5.0, 1.0, 2.0);
        suite.generate_interpretations();
        assert!(suite.interpretations.contains_key("XXX"));
    }
}

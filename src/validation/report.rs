//! Validation report generation and analysis for ASHRAE 140.
//!
//! This module provides structures and methods for generating comprehensive
//! validation reports, including pass/fail determination, delta analysis,
//! and multiple export formats (Markdown, HTML, CSV).

use std::collections::HashMap;
use std::fmt;

/// Types of validation metrics for ASHRAE 140.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
        }
    }

    /// Returns the deviation from reference range center as a string.
    pub fn deviation_string(&self) -> String {
        format!("{:+.2}%", self.percent_error)
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

/// Comprehensive validation report for ASHRAE 140 test cases.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All validation results
    pub results: Vec<ValidationResult>,
    /// Benchmark data for each case
    pub benchmark_data: HashMap<String, BenchmarkData>,
}

impl ValidationReport {
    /// Creates a new empty validation report.
    pub fn new() -> Self {
        Self::default()
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

    /// Adds benchmark data for a case.
    pub fn add_benchmark_data(&mut self, case_id: &str, data: BenchmarkData) {
        self.benchmark_data.insert(case_id.to_string(), data);
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

    /// Alias for calculate_mae() for consistency with ValidationReport.
    pub fn mae(&self) -> f64 {
        self.calculate_mae()
    }

    /// Alias for calculate_max_deviation() for consistency with ValidationReport.
    pub fn max_deviation(&self) -> f64 {
        self.calculate_max_deviation()
    }

    /// Alias for calculate_pass_rate() for consistency with ValidationReport.
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
    pub fn generate_report(&self) -> ValidationReport {
        let mut report = ValidationReport::new();

        // Copy all results
        report.results = self.results.clone();

        // Copy benchmark data, or populate from results if missing
        if self.benchmark_data.is_empty() && !self.results.is_empty() {
            // Create benchmark data from results
            let mut case_data: HashMap<String, BenchmarkData> = HashMap::new();

            for result in &self.results {
                let benchmark = case_data
                    .entry(result.case_id.clone())
                    .or_default();

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut report = ValidationReport::new();

        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);
        report.add_result_simple("600", MetricType::AnnualCooling, 7.0, 6.14, 8.45);
        report.add_result_simple("900", MetricType::AnnualHeating, 1.5, 1.17, 2.04);

        assert_eq!(report.results.len(), 3);
        assert!(report.pass_rate() > 0.0);
        assert!(report.mae() >= 0.0);
    }

    #[test]
    fn test_validation_report_markdown() {
        let mut report = ValidationReport::new();
        report.add_result_simple("600", MetricType::AnnualHeating, 5.0, 4.30, 5.71);

        let markdown = report.to_markdown();
        assert!(markdown.contains("# ASHRAE 140 Validation Report"));
        assert!(markdown.contains("## Summary"));
        assert!(markdown.contains("600"));
    }

    #[test]
    fn test_validation_report_csv() {
        let mut report = ValidationReport::new();
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
        assert!(mae >= 0.0 && mae <= 10.0);
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
}

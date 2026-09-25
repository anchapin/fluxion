//! Multi-zone validation report family — `MultiZoneValidationReport`,
//! `Case960Report`, `Case970Report`, `MultiZoneSummary`, and
//! `ValidationSuite`. Extracted from `src/validation/report.rs` at Issue
//! #3788 decomposition time (the parent landed at 4136/4136 lines, see
//! `tests/reference_data/module_size/report_ratchet.json`) so the parent
//! `report/mod.rs` is small enough to satisfy the Issue #3457
//! module-size ratchet. Public API is preserved unchanged; every
//! `crate::validation::report::X` path continues to work because
//! `mod.rs` re-exports the public symbols defined here.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    BenchmarkData, BenchmarkReport, Interpretation, MetricType, ValidationResult, ValidationStatus,
};

impl Default for MultiZoneValidationReport {
    fn default() -> Self {
        Self {
            case_960_report: Case960Report::default(),
            case_970_report: Case970Report::default(),
            case_980_report: Case960Report::default(),
            summary: MultiZoneSummary::default(),
        }
    }
}

impl Default for Case960Report {
    fn default() -> Self {
        let default_result = ValidationResult {
            case_id: "960".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 0.0,
            ref_min: 0.0,
            ref_max: 0.0,
            percent_error: 0.0,
            status: ValidationStatus::Fail,
            per_program: None,
            peak_date: None,
            peak_hour: None,
            peak_timestamp: None,
        };

        Self {
            annual_heating: default_result.clone(),
            annual_cooling: default_result.clone(),
            peak_heating: default_result.clone(),
            peak_cooling: default_result.clone(),
            temperature_profile: default_result.clone(),
            inter_zone_heat_transfer: default_result,
        }
    }
}

impl Default for Case970Report {
    fn default() -> Self {
        let default_result = ValidationResult {
            case_id: "970".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 0.0,
            ref_min: 0.0,
            ref_max: 0.0,
            percent_error: 0.0,
            status: ValidationStatus::Fail,
            per_program: None,
            peak_date: None,
            peak_hour: None,
            peak_timestamp: None,
        };

        Self {
            annual_heating: default_result.clone(),
            annual_cooling: default_result.clone(),
            peak_heating: default_result.clone(),
            peak_cooling: default_result.clone(),
            multi_zone_coupling: default_result,
        }
    }
}

impl Default for MultiZoneSummary {
    fn default() -> Self {
        Self {
            total_tests: 0,
            passed_tests: 0,
            warning_tests: 0,
            failed_tests: 0,
            pass_rate: 0.0,
            mean_absolute_error: 0.0,
            max_deviation: 0.0,
            overall_status: ValidationStatus::Fail,
        }
    }
}

/// Multi-zone validation report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiZoneValidationReport {
    /// Case 960 validation results
    pub case_960_report: Case960Report,
    /// Case 970 validation results
    pub case_970_report: Case970Report,
    /// Case 980 validation results (stub)
    pub case_980_report: Case960Report,
    /// Overall multi-zone validation summary
    pub summary: MultiZoneSummary,
}

/// Case 960 specific validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case960Report {
    /// Annual heating validation results
    pub annual_heating: ValidationResult,
    /// Annual cooling validation results
    pub annual_cooling: ValidationResult,
    /// Peak heating validation results
    pub peak_heating: ValidationResult,
    /// Peak cooling validation results
    pub peak_cooling: ValidationResult,
    /// Temperature profile validation results
    pub temperature_profile: ValidationResult,
    /// Inter-zone heat transfer validation
    pub inter_zone_heat_transfer: ValidationResult,
}

/// Case 970 specific validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case970Report {
    /// Annual heating validation results
    pub annual_heating: ValidationResult,
    /// Annual cooling validation results
    pub annual_cooling: ValidationResult,
    /// Peak heating validation results
    pub peak_heating: ValidationResult,
    /// Peak cooling validation results
    pub peak_cooling: ValidationResult,
    /// Multi-zone coupling validation
    pub multi_zone_coupling: ValidationResult,
}

/// Multi-zone validation summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiZoneSummary {
    /// Total number of validation tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Number of warning tests
    pub warning_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Overall pass rate percentage
    pub pass_rate: f64,
    /// Mean absolute error across all tests
    pub mean_absolute_error: f64,
    /// Maximum deviation percentage
    pub max_deviation: f64,
    /// Overall validation status
    pub overall_status: ValidationStatus,
}

/// A collection of validation results for multiple cases.
///
/// `ValidationSuite` provides high-level methods for collecting, analyzing,
/// and reporting on validation results across multiple test cases.
#[derive(Debug, Clone)]
pub struct ValidationSuite {
    /// All validation results
    results: Vec<ValidationResult>,
    /// Benchmark data for each case
    ///
    /// Field is `pub` so the extracted `tests` child module of
    /// `mod.rs` can read it for assertions, matching the pre-#3788
    /// visibility from `report.rs`'s inline `mod tests` block.
    /// Production callers should treat this as opaque and use
    /// `add_benchmark_data` / `with_ashrae140_data` instead.
    pub benchmark_data: HashMap<String, BenchmarkData>,
    /// Interpretation guidance for failed metrics
    ///
    /// Field is `pub` (not just module-private) so the extracted
    /// `tests` child module of `mod.rs` can read it for assertions,
    /// matching the pre-Issue-#3788 visibility from `report.rs`'s
    /// inline `mod tests` block. Production callers should treat
    /// this as opaque and use `generate_interpretations` /
    /// `interpretation_for_case` instead.
    pub interpretations: HashMap<String, Interpretation>,
}

impl Default for ValidationSuite {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            benchmark_data: HashMap::new(),
            interpretations: HashMap::new(),
        }
    }
}

impl ValidationSuite {
    /// Creates a new empty validation suite with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new validation suite with specified configuration.
    ///
    /// The configuration is currently unused (reserved for future use);
    /// this constructor is retained for API compatibility.
    pub fn new_with_config(_config: crate::validation::ValidationConfig) -> Self {
        Self::new()
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
            let entry = summary.entry(result.metric.clone()).or_insert((0, 0, 0));

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
                    MetricType::IncidentSolar { .. } => {
                        // Incident solar metrics are not aggregated into benchmark data
                        // They are per-orientation and handled separately
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

    /// Emits a detailed summary via `tracing`. (Issue #2500)
    pub fn print_detailed_summary(&self) {
        tracing::info!(
            total_results = self.len(),
            pass_rate_pct = self.calculate_pass_rate(),
            passed = self.results.iter().filter(|r| r.passed()).count(),
            warning_rate_pct = self.calculate_warning_rate(),
            warnings = self.warning_count(),
            fail_rate_pct = self.calculate_fail_rate(),
            failed = self.fail_count(),
            mae_pct = self.calculate_mae(),
            rmse_pct = self.calculate_rmse(),
            max_deviation_pct = self.calculate_max_deviation(),
            mean_deviation_pct = self.calculate_mean_deviation(),
            "validation suite summary",
        );

        // Summary by case
        let case_summary = self.summary_by_case();
        let mut case_ids: Vec<_> = case_summary.keys().collect();
        case_ids.sort();

        for case_id in case_ids {
            let (passed, warnings, failed) = case_summary.get(case_id).unwrap();
            let total = passed + warnings + failed;
            let pass_rate = (*passed as f64 / total as f64) * 100.0;
            let span = tracing::info_span!("ashrae140_case", case_id = %case_id);
            let _guard = span.enter();
            tracing::info!(
                case_id = %case_id,
                passed = *passed,
                total = total,
                pass_rate_pct = pass_rate,
                warnings = *warnings,
                failed = *failed,
                "summary by case",
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
                .or_default()
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
                MetricType::AnnualCooling if case_id.starts_with("9") => {
                    hypotheses.push(
                        "High-mass cooling energy over-prediction suggests thermal mass coupling \
                         ratio (h_tr_em / h_tr_ms) may be too low, causing excessive heat \
                         storage and delayed cooling response."
                            .to_string(),
                    );
                }
                MetricType::AnnualHeating if case_id.starts_with("9") => {
                    hypotheses.push(
                        "High-mass heating energy over-prediction indicates thermal mass is \
                         storing too much heat during the day and releasing it slowly, \
                         increasing heating demand."
                            .to_string(),
                    );
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

    /// Run standard validation and return a ValidationResult
    pub fn run_validation(&self) -> ValidationResult {
        // For now, return a mock result
        // In a real implementation, this would run all validations
        ValidationResult {
            case_id: "integrated".to_string(),
            metric: MetricType::AnnualHeating,
            fluxion_value: 100.0,
            ref_min: 95.0,
            ref_max: 105.0,
            percent_error: 0.0,
            status: ValidationStatus::Pass,
            per_program: None,
            peak_date: None,
            peak_hour: None,
            peak_timestamp: None,
        }
    }

    /// Run performance validation and return a PerformanceReport
    pub fn run_performance_validation(
        &self,
    ) -> Result<crate::validation::performance::PerformanceReport, String> {
        // For now, return a mock performance report
        // In a real implementation, this would run actual performance tests
        Ok(crate::validation::performance::reports::PerformanceReport {
            timestamp: Utc::now(),
            metrics: crate::validation::performance::reports::PerformanceMetrics {
                timestep_duration_ms: 25.0,
                memory_usage_bytes: 5_000_000,
                iterations_per_timestep: 50,
                cpu_utilization: 0.75,
                throughput_tps: 1000.0,
                zone_coupling_time_ms: 5.0,
            },
            baseline_comparison: None,
            regression_warnings: None,
            trend_analysis: None,
        })
    }
}

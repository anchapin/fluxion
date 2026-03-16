//! A/B testing framework for thermal network variant comparison.
//!
//! This module provides statistical comparison framework for evaluating thermal network
//! variants (5R1C baseline, 6R2C opt-in, 8R3C new, targeted fixes). Uses NMBE, CV(RMSE),
//! and pass rates to quantify improvement, enabling data-driven decisions about adopting
//! 8R3C or other fixes.
//!
//! # Design
//!
//! - Manual-only execution (no CI) - triggered via `cargo test ab_testing -- --nocapture`
//! - Supports multiple thermal network variants for comparative analysis
//! - Calculates statistical metrics (NMBE, CV(RMSE)) against reference data
//! - Generates markdown reports with improvement metrics and recommendations
//!
//! # Usage
//!
//! ```rust,no_run
//! use fluxion::validation::ab_testing::{ABTestRunner, ThermalNetworkVariant};
//!
//! let runner = ABTestRunner::new()
//!     .with_variants(vec![ThermalNetworkVariant::FiveR1C, ThermalNetworkVariant::SixR2C])
//!     .with_cases(vec!["600", "900"]);
//!
//! let baseline = runner.run_all_variants(ThermalNetworkVariant::FiveR1C);
//! let test = runner.run_all_variants(ThermalNetworkVariant::SixR2C);
//!
//! let report = runner.compare_results(&baseline, &test);
//! println!("{}", report.to_markdown());
//! ```

use crate::validation::benchmark::get_benchmark_data;

/// Thermal network variants available for A/B testing.
///
/// Represents different thermal network configurations that can be compared
/// for accuracy and performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ThermalNetworkVariant {
    /// Current default 5R1C thermal network (ISO 13790 compliant)
    FiveR1C,
    /// Existing opt-in 6R2C thermal network (via ThermalModel::configure_6r2c_model())
    SixR2C,
    /// New 8R3C thermal network (future implementation for Phase 22 evaluation)
    EightR3C,
    /// Targeted refinement A (thermal mass coupling adjustment)
    ThermalMassFixA,
    /// Targeted refinement B (alternative thermal mass approach)
    ThermalMassFixB,
}

impl std::fmt::Display for ThermalNetworkVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThermalNetworkVariant::FiveR1C => write!(f, "5R1C"),
            ThermalNetworkVariant::SixR2C => write!(f, "6R2C"),
            ThermalNetworkVariant::EightR3C => write!(f, "8R3C"),
            ThermalNetworkVariant::ThermalMassFixA => write!(f, "ThermalMassFixA"),
            ThermalNetworkVariant::ThermalMassFixB => write!(f, "ThermalMassFixB"),
        }
    }
}

/// Test results for a single thermal network variant and ASHRAE 140 case.
///
/// Contains simulation metrics and reference ranges for comparison.
#[derive(Debug, Clone)]
pub struct TestResults {
    /// Thermal network variant used for this test
    pub variant: ThermalNetworkVariant,
    /// ASHRAE 140 case ID (e.g., "600", "900", "960")
    pub case_id: String,
    /// Annual heating energy consumption (MWh)
    pub annual_heating_mwh: f64,
    /// Annual cooling energy consumption (MWh)
    pub annual_cooling_mwh: f64,
    /// Peak heating load (kW)
    pub peak_heating_kw: f64,
    /// Peak cooling load (kW)
    pub peak_cooling_kw: f64,
    /// Annual heating reference minimum (MWh)
    pub annual_heating_ref_min: f64,
    /// Annual heating reference maximum (MWh)
    pub annual_heating_ref_max: f64,
    /// Annual cooling reference minimum (MWh)
    pub annual_cooling_ref_min: f64,
    /// Annual cooling reference maximum (MWh)
    pub annual_cooling_ref_max: f64,
}

impl TestResults {
    /// Check if heating energy is within tolerance.
    pub fn heating_ok(&self) -> bool {
        self.annual_heating_mwh >= self.annual_heating_ref_min * 0.85
            && self.annual_heating_mwh <= self.annual_heating_ref_max * 1.15
    }

    /// Check if cooling energy is within tolerance.
    pub fn cooling_ok(&self) -> bool {
        self.annual_cooling_mwh >= self.annual_cooling_ref_min * 0.85
            && self.annual_cooling_mwh <= self.annual_cooling_ref_max * 1.15
    }

    /// Check if all metrics are within tolerance.
    pub fn all_ok(&self) -> bool {
        self.heating_ok() && self.cooling_ok()
    }
}

/// A/B test results for a thermal network variant across all cases.
///
/// Aggregates test results with statistical metrics (NMBE, CV(RMSE), pass rate).
#[derive(Debug, Clone)]
pub struct ABTestResult {
    /// Thermal network variant being tested
    pub variant: ThermalNetworkVariant,
    /// Individual test results for each case
    pub cases: Vec<TestResults>,
    /// Normalized Mean Bias Error for heating (percentage)
    pub nmbe_heating: f64,
    /// Normalized Mean Bias Error for cooling (percentage)
    pub nmbe_cooling: f64,
    /// Coefficient of Variation of RMSE for heating (percentage)
    pub cv_rmse_heating: f64,
    /// Coefficient of Variation of RMSE for cooling (percentage)
    pub cv_rmse_cooling: f64,
    /// Pass rate (percentage of cases within tolerance)
    pub pass_rate: f64,
}

impl ABTestResult {
    /// Calculate pass rate for a given tolerance percentage.
    ///
    /// # Arguments
    /// * `tolerance_pct` - Tolerance percentage (e.g., 15.0 for ±15%)
    ///
    /// # Returns
    /// Percentage of cases within tolerance (0.0 to 100.0)
    pub fn pass_rate(&self, tolerance_pct: f64) -> f64 {
        if self.cases.is_empty() {
            return 0.0;
        }

        let passed = self
            .cases
            .iter()
            .filter(|case| {
                let heating_ok = case.annual_heating_mwh
                    >= case.annual_heating_ref_min * (1.0 - tolerance_pct / 100.0)
                    && case.annual_heating_mwh
                        <= case.annual_heating_ref_max * (1.0 + tolerance_pct / 100.0);
                let cooling_ok = case.annual_cooling_mwh
                    >= case.annual_cooling_ref_min * (1.0 - tolerance_pct / 100.0)
                    && case.annual_cooling_mwh
                        <= case.annual_cooling_ref_max * (1.0 + tolerance_pct / 100.0);
                heating_ok && cooling_ok
            })
            .count();

        (passed as f64 / self.cases.len() as f64) * 100.0
    }

    /// Compare this test result against a baseline.
    ///
    /// # Arguments
    /// * `baseline` - Baseline ABTestResult to compare against
    ///
    /// # Returns
    /// String describing the comparison (NMBE improvement, CV(RMSE) improvement, pass rate improvement)
    pub fn compare(&self, baseline: &ABTestResult) -> String {
        let nmbe_heating_improvement = baseline.nmbe_heating - self.nmbe_heating;
        let nmbe_cooling_improvement = baseline.nmbe_cooling - self.nmbe_cooling;
        let cv_rmse_heating_improvement = baseline.cv_rmse_heating - self.cv_rmse_heating;
        let cv_rmse_cooling_improvement = baseline.cv_rmse_cooling - self.cv_rmse_cooling;
        let pass_rate_improvement = self.pass_rate - baseline.pass_rate;

        format!(
            "{} vs {}:\n\
             - Heating NMBE: {:.2}% → {:.2}% (improvement: {:+.2}%)\n\
             - Cooling NMBE: {:.2}% → {:.2}% (improvement: {:+.2}%)\n\
             - Heating CV(RMSE): {:.2}% → {:.2}% (improvement: {:+.2}%)\n\
             - Cooling CV(RMSE): {:.2}% → {:.2}% (improvement: {:+.2}%)\n\
             - Pass rate: {:.1}% → {:.1}% (improvement: {:+.1}%)",
            baseline.variant,
            self.variant,
            baseline.nmbe_heating,
            self.nmbe_heating,
            nmbe_heating_improvement,
            baseline.nmbe_cooling,
            self.nmbe_cooling,
            nmbe_cooling_improvement,
            baseline.cv_rmse_heating,
            self.cv_rmse_heating,
            cv_rmse_heating_improvement,
            baseline.cv_rmse_cooling,
            self.cv_rmse_cooling,
            cv_rmse_cooling_improvement,
            baseline.pass_rate,
            self.pass_rate,
            pass_rate_improvement
        )
    }
}

/// Comparison report for two thermal network variants.
///
/// Contains improvement metrics and a recommendation based on statistical analysis.
#[derive(Debug)]
pub struct ComparisonReport {
    /// Baseline thermal network variant
    pub baseline_variant: ThermalNetworkVariant,
    /// Test thermal network variant being evaluated
    pub test_variant: ThermalNetworkVariant,
    /// Heating NMBE improvement (baseline - test, in percentage)
    pub heating_nmbe_improvement: f64,
    /// Cooling NMBE improvement (baseline - test, in percentage)
    pub cooling_nmbe_improvement: f64,
    /// Pass rate improvement (test - baseline, in percentage)
    pub pass_rate_improvement: f64,
    /// Heating CV(RMSE) improvement (baseline - test, in percentage)
    pub heating_cv_rmse_improvement: f64,
    /// Cooling CV(RMSE) improvement (baseline - test, in percentage)
    pub cooling_cv_rmse_improvement: f64,
    /// Recommendation based on metrics (ADOPT, DEFER, or REJECT)
    pub recommendation: String,
    /// Detailed explanation of the recommendation
    pub explanation: String,
}

impl ComparisonReport {
    /// Generate a markdown report for this comparison.
    ///
    /// # Returns
    /// Markdown string with improvement metrics and recommendation
    pub fn to_markdown(&self) -> String {
        let baseline_nmbe_heating = self.heating_nmbe_improvement + self.test_nmbe_heating();
        let baseline_nmbe_cooling = self.cooling_nmbe_improvement + self.test_nmbe_cooling();
        let baseline_cv_rmse_heating =
            self.heating_cv_rmse_improvement + self.test_cv_rmse_heating();
        let baseline_cv_rmse_cooling =
            self.cooling_cv_rmse_improvement + self.test_cv_rmse_cooling();
        let baseline_pass_rate = self.pass_rate_improvement + self.test_pass_rate();

        format!(
            "# A/B Test Comparison Report\n\n\
             **Baseline:** {} vs **Test:** {}\n\n\
             ## Improvement Metrics\n\n\
             | Metric | Baseline | Test | Improvement |\n\
             |--------|----------|------|-------------|\n\
             | Heating NMBE | {:.2}% | {:.2}% | {:+.2}% |\n\
             | Cooling NMBE | {:.2}% | {:.2}% | {:+.2}% |\n\
             | Heating CV(RMSE) | {:.2}% | {:.2}% | {:+.2}% |\n\
             | Cooling CV(RMSE) | {:.2}% | {:.2}% | {:+.2}% |\n\
             | Pass Rate | {:.1}% | {:.1}% | {:+.1}% |\n\n\
             ## Recommendation\n\n\
             **{}**\n\n\
             {}\n",
            self.baseline_variant,
            self.test_variant,
            baseline_nmbe_heating,
            self.test_nmbe_heating(),
            self.heating_nmbe_improvement,
            baseline_nmbe_cooling,
            self.test_nmbe_cooling(),
            self.cooling_nmbe_improvement,
            baseline_cv_rmse_heating,
            self.test_cv_rmse_heating(),
            self.heating_cv_rmse_improvement,
            baseline_cv_rmse_cooling,
            self.test_cv_rmse_cooling(),
            self.cooling_cv_rmse_improvement,
            baseline_pass_rate,
            self.test_pass_rate(),
            self.pass_rate_improvement,
            self.recommendation,
            self.explanation
        )
    }

    // Helper methods to extract test values (these are placeholders)
    fn test_nmbe_heating(&self) -> f64 {
        0.0
    }

    fn test_nmbe_cooling(&self) -> f64 {
        0.0
    }

    fn test_cv_rmse_heating(&self) -> f64 {
        0.0
    }

    fn test_cv_rmse_cooling(&self) -> f64 {
        0.0
    }

    fn test_pass_rate(&self) -> f64 {
        self.pass_rate_improvement + 50.0 // Simplified baseline of 50%
    }
}

/// A/B test runner for comparing thermal network variants.
///
/// Manages test execution, metric calculation, and report generation.
pub struct ABTestRunner {
    /// Thermal network variants to test
    pub variants: Vec<ThermalNetworkVariant>,
    /// ASHRAE 140 cases to run (case IDs)
    pub cases: Vec<&'static str>,
}

impl Default for ABTestRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl ABTestRunner {
    /// Create a new A/B test runner with default configuration.
    ///
    /// Default configuration:
    /// - Variants: [FiveR1C, SixR2C]
    /// - Cases: All 18 ASHRAE 140 cases (600, 610-650, 800-810, 900-960)
    pub fn new() -> Self {
        ABTestRunner {
            variants: vec![
                ThermalNetworkVariant::FiveR1C,
                ThermalNetworkVariant::SixR2C,
            ],
            cases: vec![
                // 600 series (lightweight)
                "600", "610", "620", "630", "640", "650",
                // 800 series (medium weight, free-floating)
                "800", "810", // 900 series (heavyweight)
                "900", "920", "930", "940", "950", "960",
            ],
        }
    }

    /// Set the thermal network variants to test.
    ///
    /// # Arguments
    /// * `variants` - List of thermal network variants
    ///
    /// # Returns
    /// Self for builder pattern chaining
    pub fn with_variants(mut self, variants: Vec<ThermalNetworkVariant>) -> Self {
        self.variants = variants;
        self
    }

    /// Set the ASHRAE 140 cases to run.
    ///
    /// # Arguments
    /// * `cases` - List of case IDs (e.g., ["600", "900"])
    ///
    /// # Returns
    /// Self for builder pattern chaining
    pub fn with_cases(mut self, cases: Vec<&'static str>) -> Self {
        self.cases = cases;
        self
    }

    /// Run a single thermal network variant for a specific case.
    ///
    /// # Arguments
    /// * `variant` - Thermal network variant to test
    /// * `case_id` - ASHRAE 140 case ID
    ///
    /// # Returns
    /// TestResults with simulation metrics and reference ranges
    ///
    /// # Panics
    /// If case configuration is invalid or simulation fails
    ///
    /// # Note
    /// This is a placeholder implementation. In a real implementation, this would:
    /// 1. Create a ThermalModel for the specified case
    /// 2. Configure the model for the specified variant (5R1C, 6R2C, etc.)
    /// 3. Run the simulation for one year (8760 hours)
    /// 4. Extract results and return TestResults
    ///
    /// For now, this returns mock data based on the variant to demonstrate the framework structure.
    pub fn run_variant(&self, variant: ThermalNetworkVariant, case_id: &str) -> TestResults {
        // Get benchmark data for reference ranges
        let benchmark = get_benchmark_data(case_id)
            .unwrap_or_else(|| panic!("No benchmark data for case: {}", case_id));

        // TODO: Implement actual simulation
        // For now, return mock data that varies by variant
        let (heating_factor, cooling_factor) = match variant {
            ThermalNetworkVariant::FiveR1C => (1.0, 1.0),
            ThermalNetworkVariant::SixR2C => (0.95, 0.97), // Slight improvement
            ThermalNetworkVariant::EightR3C => (0.90, 0.95), // More improvement
            ThermalNetworkVariant::ThermalMassFixA => (0.93, 0.96),
            ThermalNetworkVariant::ThermalMassFixB => (0.97, 0.98),
        };

        // Use midpoint of reference range as base value
        let ref_heating_mid = (benchmark.annual_heating_min + benchmark.annual_heating_max) / 2.0;
        let ref_cooling_mid = (benchmark.annual_cooling_min + benchmark.annual_cooling_max) / 2.0;

        let annual_heating_mwh = ref_heating_mid * heating_factor;
        let annual_cooling_mwh = ref_cooling_mid * cooling_factor;

        // Mock peak loads
        let peak_heating_kw = (ref_heating_mid * 1000.0) / 8760.0 * 2.0;
        let peak_cooling_kw = (ref_cooling_mid * 1000.0) / 8760.0 * 2.5;

        TestResults {
            variant,
            case_id: case_id.to_string(),
            annual_heating_mwh,
            annual_cooling_mwh,
            peak_heating_kw,
            peak_cooling_kw,
            annual_heating_ref_min: benchmark.annual_heating_min,
            annual_heating_ref_max: benchmark.annual_heating_max,
            annual_cooling_ref_min: benchmark.annual_cooling_min,
            annual_cooling_ref_max: benchmark.annual_cooling_max,
        }
    }

    /// Run all configured cases for a thermal network variant.
    ///
    /// # Arguments
    /// * `variant` - Thermal network variant to test
    ///
    /// # Returns
    /// ABTestResult with aggregated metrics (NMBE, CV(RMSE), pass rate)
    pub fn run_all_variants(&self, variant: ThermalNetworkVariant) -> ABTestResult {
        let cases: Vec<TestResults> = self
            .cases
            .iter()
            .map(|case_id| self.run_variant(variant, case_id))
            .collect();

        // Calculate statistical metrics against reference values
        let (nmbe_heating, nmbe_cooling, cv_rmse_heating, cv_rmse_cooling) =
            self.calculate_metrics(&cases);

        // Calculate pass rate (within ±15% tolerance)
        let pass_rate = self.calculate_pass_rate(&cases, 15.0);

        ABTestResult {
            variant,
            cases,
            nmbe_heating,
            nmbe_cooling,
            cv_rmse_heating,
            cv_rmse_cooling,
            pass_rate,
        }
    }

    /// Calculate statistical metrics for a set of test results.
    ///
    /// # Arguments
    /// * `cases` - Test results to analyze
    ///
    /// # Returns
    /// Tuple of (nmbe_heating, nmbe_cooling, cv_rmse_heating, cv_rmse_cooling) in percentage
    fn calculate_metrics(&self, cases: &[TestResults]) -> (f64, f64, f64, f64) {
        if cases.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mut nmbe_heating_sum = 0.0;
        let mut nmbe_cooling_sum = 0.0;
        let mut rmse_heating_sum = 0.0;
        let mut rmse_cooling_sum = 0.0;

        for case in cases {
            // Use midpoint of reference range as reference value
            let ref_heating_mid = (case.annual_heating_ref_min + case.annual_heating_ref_max) / 2.0;
            let ref_cooling_mid = (case.annual_cooling_ref_min + case.annual_cooling_ref_max) / 2.0;

            // Calculate bias error
            let heating_bias = case.annual_heating_mwh - ref_heating_mid;
            let cooling_bias = case.annual_cooling_mwh - ref_cooling_mid;

            // Sum for NMBE calculation
            nmbe_heating_sum += heating_bias / ref_heating_mid;
            nmbe_cooling_sum += cooling_bias / ref_cooling_mid;

            // Sum for RMSE calculation
            rmse_heating_sum += (heating_bias / ref_heating_mid).powi(2);
            rmse_cooling_sum += (cooling_bias / ref_cooling_mid).powi(2);
        }

        // Calculate NMBE (Normalized Mean Bias Error)
        let nmbe_heating = (nmbe_heating_sum / cases.len() as f64) * 100.0;
        let nmbe_cooling = (nmbe_cooling_sum / cases.len() as f64) * 100.0;

        // Calculate CV(RMSE) (Coefficient of Variation of Root Mean Square Error)
        let rmse_heating = (rmse_heating_sum / cases.len() as f64).sqrt();
        let rmse_cooling = (rmse_cooling_sum / cases.len() as f64).sqrt();

        let cv_rmse_heating = rmse_heating * 100.0;
        let cv_rmse_cooling = rmse_cooling * 100.0;

        (nmbe_heating, nmbe_cooling, cv_rmse_heating, cv_rmse_cooling)
    }

    /// Calculate pass rate for a set of test results.
    ///
    /// # Arguments
    /// * `cases` - Test results to analyze
    /// * `tolerance_pct` - Tolerance percentage (e.g., 15.0 for ±15%)
    ///
    /// # Returns
    /// Pass rate as percentage (0.0 to 100.0)
    fn calculate_pass_rate(&self, cases: &[TestResults], tolerance_pct: f64) -> f64 {
        if cases.is_empty() {
            return 0.0;
        }

        let passed = cases
            .iter()
            .filter(|case| {
                let heating_ok = case.annual_heating_mwh
                    >= case.annual_heating_ref_min * (1.0 - tolerance_pct / 100.0)
                    && case.annual_heating_mwh
                        <= case.annual_heating_ref_max * (1.0 + tolerance_pct / 100.0);
                let cooling_ok = case.annual_cooling_mwh
                    >= case.annual_cooling_ref_min * (1.0 - tolerance_pct / 100.0)
                    && case.annual_cooling_mwh
                        <= case.annual_cooling_ref_max * (1.0 + tolerance_pct / 100.0);
                heating_ok && cooling_ok
            })
            .count();

        (passed as f64 / cases.len() as f64) * 100.0
    }

    /// Compare two thermal network variants and generate a comparison report.
    ///
    /// # Arguments
    /// * `baseline` - Baseline ABTestResult
    /// * `test` - Test ABTestResult to compare against baseline
    ///
    /// # Returns
    /// ComparisonReport with improvement metrics and recommendation
    pub fn compare_results(
        &self,
        baseline: &ABTestResult,
        test: &ABTestResult,
    ) -> ComparisonReport {
        let heating_nmbe_improvement = baseline.nmbe_heating - test.nmbe_heating;
        let cooling_nmbe_improvement = baseline.nmbe_cooling - test.nmbe_cooling;
        let heating_cv_rmse_improvement = baseline.cv_rmse_heating - test.cv_rmse_heating;
        let cooling_cv_rmse_improvement = baseline.cv_rmse_cooling - test.cv_rmse_cooling;
        let pass_rate_improvement = test.pass_rate - baseline.pass_rate;

        // Determine recommendation based on metrics
        let (recommendation, explanation) = self.determine_recommendation(
            heating_nmbe_improvement,
            cooling_nmbe_improvement,
            pass_rate_improvement,
            heating_cv_rmse_improvement,
            cooling_cv_rmse_improvement,
        );

        ComparisonReport {
            baseline_variant: baseline.variant,
            test_variant: test.variant,
            heating_nmbe_improvement,
            cooling_nmbe_improvement,
            pass_rate_improvement,
            heating_cv_rmse_improvement,
            cooling_cv_rmse_improvement,
            recommendation,
            explanation,
        }
    }

    /// Determine recommendation based on improvement metrics.
    ///
    /// # Arguments
    /// * `heating_nmbe_improvement` - Heating NMBE improvement (baseline - test)
    /// * `cooling_nmbe_improvement` - Cooling NMBE improvement (baseline - test)
    /// * `pass_rate_improvement` - Pass rate improvement (test - baseline)
    /// * `heating_cv_rmse_improvement` - Heating CV(RMSE) improvement (baseline - test)
    /// * `cooling_cv_rmse_improvement` - Cooling CV(RMSE) improvement (baseline - test)
    ///
    /// # Returns
    /// Tuple of (recommendation string, explanation string)
    fn determine_recommendation(
        &self,
        heating_nmbe_improvement: f64,
        cooling_nmbe_improvement: f64,
        pass_rate_improvement: f64,
        heating_cv_rmse_improvement: f64,
        cooling_cv_rmse_improvement: f64,
    ) -> (String, String) {
        // Check if test variant shows clear improvement
        let has_improvement = (heating_nmbe_improvement > 0.5
            || cooling_nmbe_improvement > 0.5
            || pass_rate_improvement > 5.0
            || heating_cv_rmse_improvement > 0.5
            || cooling_cv_rmse_improvement > 0.5);

        if !has_improvement {
            return (
                "DEFER".to_string(),
                "Test variant does not show statistically significant improvement over baseline. \
                 Consider deferring adoption until clear benefits are demonstrated or \
                 alternative improvements are explored."
                    .to_string(),
            );
        }

        // Check if improvement is substantial
        let has_substantial_improvement = (heating_nmbe_improvement > 2.0
            || cooling_nmbe_improvement > 2.0
            || pass_rate_improvement > 15.0
            || heating_cv_rmse_improvement > 2.0
            || cooling_cv_rmse_improvement > 2.0);

        if has_substantial_improvement {
            return (
                "ADOPT".to_string(),
                format!(
                    "Test variant shows substantial improvement over baseline (heating NMBE: {:+.2}%, \
                     cooling NMBE: {:+.2}%, pass rate: {:+.1}%). Recommended for adoption.",
                    heating_nmbe_improvement, cooling_nmbe_improvement, pass_rate_improvement
                ),
            );
        }

        // Moderate improvement - adopt with caution
        (
            "ADOPT".to_string(),
            format!(
                "Test variant shows moderate improvement over baseline (heating NMBE: {:+.2}%, \
                 cooling NMBE: {:+.2}%, pass rate: {:+.1}%). Adoption recommended, but monitor \
                 performance in production.",
                heating_nmbe_improvement, cooling_nmbe_improvement, pass_rate_improvement
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_network_variant_display() {
        assert_eq!(ThermalNetworkVariant::FiveR1C.to_string(), "5R1C");
        assert_eq!(ThermalNetworkVariant::SixR2C.to_string(), "6R2C");
        assert_eq!(ThermalNetworkVariant::EightR3C.to_string(), "8R3C");
        assert_eq!(
            ThermalNetworkVariant::ThermalMassFixA.to_string(),
            "ThermalMassFixA"
        );
        assert_eq!(
            ThermalNetworkVariant::ThermalMassFixB.to_string(),
            "ThermalMassFixB"
        );
    }

    #[test]
    fn test_test_results_checks() {
        let result = TestResults {
            variant: ThermalNetworkVariant::FiveR1C,
            case_id: "600".to_string(),
            annual_heating_mwh: 5.0,
            annual_cooling_mwh: 3.0,
            peak_heating_kw: 10.0,
            peak_cooling_kw: 15.0,
            annual_heating_ref_min: 5.0,
            annual_heating_ref_max: 5.5,
            annual_cooling_ref_min: 3.0,
            annual_cooling_ref_max: 3.5,
        };

        assert!(result.heating_ok());
        assert!(result.cooling_ok());
        assert!(result.all_ok());
    }

    #[test]
    fn test_ab_test_runner_default() {
        let runner = ABTestRunner::new();
        assert_eq!(runner.variants.len(), 2);
        assert_eq!(runner.cases.len(), 14);
    }
}

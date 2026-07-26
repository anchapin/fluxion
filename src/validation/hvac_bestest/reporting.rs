//! ASHRAE-compliant tolerance assertions and reporting for HVAC BESTEST (RP-865).
//!
//! Implements the acceptance criteria for issue #1759:
//! - ASHRAE-style per-case % deviation with within/outside bound flags.
//! - Configurable tolerances that default to the RP-865 published bands.
//! - Machine-readable JSON and human-readable Markdown report output.
//!
//! # Physics / standards basis
//!
//! ASHRAE RP-865 ("HVAC BESTEST", Secondary-Side Equipment Validation) compares
//! simulation results against published reference ranges and reference-program
//! point values. A metric is considered *within bound* (PASS) when EITHER:
//!   (a) the absolute signed % deviation from the reference point value is at or
//!       below the configured tolerance band, OR
//!   (b) the simulated value falls inside the case's published min/max reference
//!       range (the multi-program envelope). This mirrors the envelope-based
//!       pass logic used in ASHRAE Standard 140 §7 and ASHRAE Guideline 14 §6.
//!
//! The signed percent difference is defined as:
//! ```text
//!   %diff = (simulated - reference) / reference * 100     (reference != 0)
//! ```
//! When `reference == 0`, the relative difference is undefined; the bound is then
//! decided solely by the published range check (if available) or by exact
//! equality of the two zeros. This guard avoids division-by-zero and keeps the
//! report mathematically consistent (no negative-absolute or NaN states).

use serde::{Deserialize, Serialize};

use crate::validation::hvac_bestest::cases::{
    get_reference_data, HVACBestestCase, HVACBestestCaseDefinition,
};
use crate::validation::hvac_bestest::runner::HVACBestestResult;

use chrono::Utc;

/// ASHRAE-style within/outside bound flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundStatus {
    /// Simulated value is inside the acceptance envelope (PASS).
    Within,
    /// Simulated value is outside the acceptance envelope (FAIL).
    Outside,
}

impl BoundStatus {
    /// `true` when the metric passes the acceptance criterion.
    #[must_use]
    pub fn is_passing(self) -> bool {
        matches!(self, Self::Within)
    }

    /// Human-readable status token used in reports.
    #[must_use]
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Within => "PASS",
            Self::Outside => "FAIL",
        }
    }
}

/// Tolerance configuration. Fields are percentages (e.g. `10.0` = ±10%).
///
/// Defaults encode the **RP-865 published bands**: ±10% for annual energy and
/// part-load COP, ±15% for peak demand. Tolerances are overridable per-suite
/// so external tooling can tighten or relax the bands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HvacBestestToleranceConfig {
    /// Acceptance band for annual energy consumption (% of reference).
    pub energy_tolerance_percent: f64,
    /// Acceptance band for peak demand (% of reference).
    pub demand_tolerance_percent: f64,
    /// Acceptance band for part-load COP values (% of reference).
    pub plr_tolerance_percent: f64,
    /// When `true`, a metric also passes if it lies inside the case's
    /// published reference `[min, max]` range, even when the % band is
    /// exceeded. This is the ASHRAE envelope semantics.
    pub honor_published_range: bool,
}

impl Default for HvacBestestToleranceConfig {
    fn default() -> Self {
        Self::rp865_defaults()
    }
}

impl HvacBestestToleranceConfig {
    /// RP-865 published acceptance bands.
    ///
    /// - Annual energy: ±10% (RP-865 §5 acceptance criterion).
    /// - Peak demand:   ±15% (RP-865 §5, peak loads carry larger program spread).
    /// - Part-load COP: ±10% (RP-865 §6 part-load curve acceptance).
    /// - Published reference range honoured as a secondary pass path.
    #[must_use]
    pub const fn rp865_defaults() -> Self {
        Self {
            energy_tolerance_percent: 10.0,
            demand_tolerance_percent: 15.0,
            plr_tolerance_percent: 10.0,
            honor_published_range: true,
        }
    }

    /// Construct a strict single-band config (one % applied to every metric),
    /// with the published range *not* honoured. Useful for tight regression gates.
    #[must_use]
    pub const fn strict(band_percent: f64) -> Self {
        Self {
            energy_tolerance_percent: band_percent,
            demand_tolerance_percent: band_percent,
            plr_tolerance_percent: band_percent,
            honor_published_range: false,
        }
    }
}

/// Threshold below which a reference value is treated as zero (avoids
/// division by a vanishing denominator). 1e-9 is well below any real
/// HVAC metric magnitude (kWh, W, or COP are all >> 1e-9).
pub const REFERENCE_ZERO_EPSILON: f64 = 1e-9;

/// Result of a single tolerance check.
///
/// `percent_diff` carries the **signed** deviation so reports can show
/// over-/under-prediction; `abs_percent_diff` drives the bound test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceCheck {
    /// Simulated value.
    pub simulated: f64,
    /// Reference point value.
    pub reference: f64,
    /// Signed percent difference `(sim - ref) / ref * 100`. May be
    /// non-finite when `reference ≈ 0`.
    pub percent_diff: f64,
    /// Absolute value of `percent_diff`.
    pub abs_percent_diff: f64,
    /// The tolerance band (%) applied for this check.
    pub tolerance_band_percent: f64,
    /// Published reference lower bound (`None` when unavailable).
    pub range_min: Option<f64>,
    /// Published reference upper bound (`None` when unavailable).
    pub range_max: Option<f64>,
    /// `true` when the metric is within the acceptance envelope.
    pub within_bound: bool,
    /// Categorical status derived from `within_bound`.
    pub status: BoundStatus,
}

impl ToleranceCheck {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.within_bound
    }
}

/// Core tolerance computation (pure, no I/O).
///
/// Computes the signed and absolute percent differences and the within-bound
/// flag. When `reference` is within `REFERENCE_ZERO_EPSILON` of zero, the
/// relative difference is treated as `0.0` if `simulated` is also ~0, else as
/// `f64::INFINITY`; in that degenerate case the bound is decided by the range
/// check only.
#[must_use]
pub fn check_within_bounds(
    simulated: f64,
    reference: f64,
    tolerance_band_percent: f64,
    range_min: Option<f64>,
    range_max: Option<f64>,
    honor_published_range: bool,
) -> ToleranceCheck {
    let percent_diff = if reference.abs() < REFERENCE_ZERO_EPSILON {
        if simulated.abs() < REFERENCE_ZERO_EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (simulated - reference) / reference * 100.0
    };
    let abs_percent_diff = percent_diff.abs();

    let within_pct = abs_percent_diff <= tolerance_band_percent;

    let within_range = if honor_published_range {
        match (range_min, range_max) {
            (Some(lo), Some(hi)) => simulated >= lo && simulated <= hi,
            _ => false,
        }
    } else {
        false
    };

    let within_bound = within_pct || within_range;
    let status = if within_bound {
        BoundStatus::Within
    } else {
        BoundStatus::Outside
    };

    ToleranceCheck {
        simulated,
        reference,
        percent_diff,
        abs_percent_diff,
        tolerance_band_percent,
        range_min,
        range_max,
        within_bound,
        status,
    }
}

/// Test-helper assertion: panic with an ASHRAE-style message if `simulated`
/// is outside the configured band. Returns the [`ToleranceCheck`] on success so
/// callers can chain additional assertions.
///
/// # Panics
/// Panics if the metric is outside the acceptance envelope.
#[track_caller]
#[must_use]
pub fn assert_within_bounds(
    simulated: f64,
    reference: f64,
    tolerance_band_percent: f64,
    context: &str,
) -> ToleranceCheck {
    assert_within_bounds_full(
        simulated,
        reference,
        tolerance_band_percent,
        None,
        None,
        true,
        context,
    )
}

/// Full-parameter tolerance assertion (with published range support).
///
/// # Panics
/// Panics if the metric is outside the acceptance envelope.
#[track_caller]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn assert_within_bounds_full(
    simulated: f64,
    reference: f64,
    tolerance_band_percent: f64,
    range_min: Option<f64>,
    range_max: Option<f64>,
    honor_published_range: bool,
    context: &str,
) -> ToleranceCheck {
    let check = check_within_bounds(
        simulated,
        reference,
        tolerance_band_percent,
        range_min,
        range_max,
        honor_published_range,
    );
    if !check.within_bound {
        let lo = range_min
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "—".to_string());
        let hi = range_max
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "—".to_string());
        panic!(
            "{context}: simulated {simulated} is OUTSIDE acceptance envelope.\n  \
             reference={reference}, ±{tolerance_band_percent}% band, \
             published range [{lo}, {hi}]\n  \
             signed %diff = {pct}%, |%diff| = {abs}% > {tol}%",
            pct = fmt_pct_signed(check.percent_diff),
            abs = fmt_pct_abs(check.abs_percent_diff),
            tol = tolerance_band_percent,
        );
    }
    check
}

/// One metric row in the per-case report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseMetricReport {
    /// Case identifier.
    pub case_id: HVACBestestCase,
    /// Descriptive case name (mirrors the case definition).
    pub case_name: String,
    /// Metric name, e.g. "Annual Energy (kWh)".
    pub metric: String,
    /// Simulated value.
    pub simulated: f64,
    /// Reference point value.
    pub reference: f64,
    /// Signed percent difference.
    pub percent_diff: f64,
    /// Absolute percent difference.
    pub abs_percent_diff: f64,
    /// Tolerance band (%) applied.
    pub tolerance_band_percent: f64,
    /// Published reference range lower bound.
    pub range_min: Option<f64>,
    /// Published reference range upper bound.
    pub range_max: Option<f64>,
    /// Within/outside bound flag.
    pub status: BoundStatus,
}

impl CaseMetricReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.status.is_passing()
    }

    /// Build a metric row from a raw [`ToleranceCheck`] plus metadata.
    #[must_use]
    fn from_check(
        case_id: HVACBestestCase,
        case_name: &str,
        metric: &str,
        check: &ToleranceCheck,
    ) -> Self {
        Self {
            case_id,
            case_name: case_name.to_string(),
            metric: metric.to_string(),
            simulated: check.simulated,
            reference: check.reference,
            percent_diff: check.percent_diff,
            abs_percent_diff: check.abs_percent_diff,
            tolerance_band_percent: check.tolerance_band_percent,
            range_min: check.range_min,
            range_max: check.range_max,
            status: check.status,
        }
    }
}

/// Aggregate suite summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total number of metric rows evaluated.
    pub total_metrics: usize,
    /// Metric rows that passed (within bound).
    pub passed: usize,
    /// Metric rows that failed (outside bound).
    pub failed: usize,
    /// Pass rate as a percentage in `[0, 100]`.
    pub pass_rate_percent: f64,
    /// Mean absolute percent deviation across all rows.
    pub mean_abs_percent_diff: f64,
    /// Maximum absolute percent deviation across all rows.
    pub max_abs_percent_diff: f64,
}

/// Full ASHRAE-compliant report for the HVAC BESTEST suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HvacBestestReport {
    /// Report title.
    pub title: String,
    /// UTC generation timestamp (RFC 3339).
    pub generated_at: String,
    /// HVAC BESTEST module version (`hvac_bestest::VERSION`).
    pub suite_version: String,
    /// Tolerance configuration used to produce this report.
    pub tolerance_config: HvacBestestToleranceConfig,
    /// Aggregate summary.
    pub summary: ReportSummary,
    /// Per-case per-metric rows.
    pub cases: Vec<CaseMetricReport>,
}

impl HvacBestestReport {
    /// Build a report from a set of runner results and case definitions.
    ///
    /// Results are matched to definitions by `case_id`. Missing reference data
    /// for a case simply omits that case's rows (no panic).
    #[must_use]
    pub fn from_results(
        results: &[HVACBestestResult],
        case_defs: &[HVACBestestCaseDefinition],
        config: &HvacBestestToleranceConfig,
    ) -> Self {
        let mut rows: Vec<CaseMetricReport> = Vec::new();

        for result in results {
            let Some(def) = case_defs.iter().find(|d| d.case_id == result.case_id) else {
                continue;
            };
            let case_id = result.case_id;
            let case_name = def.name.as_str();
            let Some(ref_data) = get_reference_data(case_id) else {
                continue;
            };

            // Annual energy
            let energy = check_within_bounds(
                result.annual_energy_kwh,
                ref_data.annual_energy_kwh,
                config.energy_tolerance_percent,
                Some(def.ref_energy_min),
                Some(def.ref_energy_max),
                config.honor_published_range,
            );
            rows.push(CaseMetricReport::from_check(
                case_id,
                case_name,
                "Annual Energy (kWh)",
                &energy,
            ));

            // Peak demand
            let demand = check_within_bounds(
                result.peak_demand_w,
                ref_data.peak_demand_w,
                config.demand_tolerance_percent,
                Some(def.ref_demand_min),
                Some(def.ref_demand_max),
                config.honor_published_range,
            );
            rows.push(CaseMetricReport::from_check(
                case_id,
                case_name,
                "Peak Demand (W)",
                &demand,
            ));

            // Part-load COP at 50 / 75 / 100 %
            for (plr_label, sim_cop, ref_cop) in [
                ("PLR 50% COP", result.plr_50_cop, ref_data.plr_50_cop),
                ("PLR 75% COP", result.plr_75_cop, ref_data.plr_75_cop),
                ("PLR 100% COP", result.plr_100_cop, ref_data.plr_100_cop),
            ] {
                let check = check_within_bounds(
                    sim_cop,
                    ref_cop,
                    config.plr_tolerance_percent,
                    None,
                    None,
                    config.honor_published_range,
                );
                rows.push(CaseMetricReport::from_check(
                    case_id, case_name, plr_label, &check,
                ));
            }
        }

        let summary = build_summary(&rows);

        Self {
            title: "HVAC BESTEST (RP-865) Validation Report".to_string(),
            generated_at: Utc::now().to_rfc3339(),
            suite_version: crate::validation::hvac_bestest::VERSION.to_string(),
            tolerance_config: config.clone(),
            summary,
            cases: rows,
        }
    }

    /// Serialize to a JSON string with non-finite floats replaced by `null`
    /// (serde_json cannot encode `±inf`/`NaN` directly).
    ///
    /// # Errors
    /// Returns an error only if the underlying JSON value cannot be serialized.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let value = self.to_json_value();
        serde_json::to_string_pretty(&value)
    }

    /// Build a `serde_json::Value` with all non-finite floats sanitized to `null`.
    fn to_json_value(&self) -> serde_json::Value {
        // Serialize normally first; this works because we replace non-finite
        // fields at the report level. We re-serialize through a tolerant mapper.
        let json = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        sanitize_floats(json)
    }

    /// Render a human-readable Markdown report (tables of case/metric/sim/ref/
    /// %diff/tolerance/status), mirroring the ASHRAE Standard 140 results layout.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", self.title));
        md.push_str(&format!("*Generated: {}*\n\n", self.generated_at));
        md.push_str(&format!(
            "Suite version: {} | Tolerance preset: RP-865 defaults \
             (energy ±{e}%, demand ±{d}%, PLR ±{p}%)\n\n",
            self.suite_version,
            e = self.tolerance_config.energy_tolerance_percent,
            d = self.tolerance_config.demand_tolerance_percent,
            p = self.tolerance_config.plr_tolerance_percent,
        ));

        // Summary table
        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Total Metrics | {} |\n",
            self.summary.total_metrics
        ));
        md.push_str(&format!("| Passed | {} |\n", self.summary.passed));
        md.push_str(&format!("| Failed | {} |\n", self.summary.failed));
        md.push_str(&format!(
            "| Pass Rate | {:.1}% |\n",
            self.summary.pass_rate_percent
        ));
        md.push_str(&format!(
            "| Mean Absolute % Deviation | {} |\n",
            fmt_pct_abs(self.summary.mean_abs_percent_diff)
        ));
        md.push_str(&format!(
            "| Max Absolute % Deviation | {} |\n\n",
            fmt_pct_abs(self.summary.max_abs_percent_diff)
        ));

        // Detailed per-case table
        md.push_str("## Detailed Results\n\n");
        md.push_str("| Case | Metric | Simulated | Reference | % Diff | Tolerance | Status |\n");
        md.push_str("|------|--------|-----------|-----------|--------|-----------|--------|\n");
        for row in &self.cases {
            let status_glyph = if row.passed() { "PASS" } else { "FAIL" };
            md.push_str(&format!(
                "| {:?} | {} | {} | {} | {} | ±{}% | {} |\n",
                row.case_id,
                row.metric,
                fmt_val(row.simulated),
                fmt_val(row.reference),
                fmt_pct_signed(row.percent_diff),
                row.tolerance_band_percent,
                status_glyph,
            ));
        }

        md
    }
}

/// Build the aggregate summary from the per-case rows.
fn build_summary(rows: &[CaseMetricReport]) -> ReportSummary {
    let total_metrics = rows.len();
    let passed = rows.iter().filter(|r| r.passed()).count();
    let failed = total_metrics - passed;
    let pass_rate_percent = if total_metrics == 0 {
        0.0
    } else {
        passed as f64 / total_metrics as f64 * 100.0
    };

    let finite_abs: Vec<f64> = rows
        .iter()
        .map(|r| r.abs_percent_diff)
        .filter(|v| v.is_finite())
        .collect();

    let mean_abs_percent_diff = if finite_abs.is_empty() {
        0.0
    } else {
        finite_abs.iter().sum::<f64>() / finite_abs.len() as f64
    };
    let max_abs_percent_diff = finite_abs.iter().cloned().fold(0.0_f64, f64::max);

    ReportSummary {
        total_metrics,
        passed,
        failed,
        pass_rate_percent,
        mean_abs_percent_diff,
        max_abs_percent_diff,
    }
}

/// Format a value, replacing `±inf`/`NaN` with `"N/A"`.
fn fmt_val(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.2}")
    } else {
        "N/A".to_string()
    }
}

/// Format a signed percent, replacing non-finite with `"N/A"`.
fn fmt_pct_signed(v: f64) -> String {
    if v.is_finite() {
        format!("{v:+.2}%")
    } else {
        "N/A".to_string()
    }
}

/// Format an absolute percent, replacing non-finite with `"N/A"`.
fn fmt_pct_abs(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.2}%")
    } else {
        "N/A".to_string()
    }
}

/// Recursively replace non-finite JSON floats with `null` so the document is
/// always valid JSON.
fn sanitize_floats(value: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    return Value::Null;
                }
            }
            Value::Number(n)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(sanitize_floats).collect()),
        Value::Object(map) => {
            let new_map: serde_json::Map<String, Value> = map
                .into_iter()
                .map(|(k, v)| (k, sanitize_floats(v)))
                .collect();
            Value::Object(new_map)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::hvac_bestest::cases::{get_bestest_cases, HVACBestestCase};
    use crate::validation::hvac_bestest::run_hvac_bestest;

    #[test]
    fn test_check_within_bounds_pass() {
        let c = check_within_bounds(30500.0, 30000.0, 10.0, None, None, false);
        assert!(c.within_bound);
        assert_eq!(c.status, BoundStatus::Within);
        assert!((c.percent_diff - 1.6667).abs() < 0.01);
    }

    #[test]
    fn test_check_within_bounds_fail() {
        let c = check_within_bounds(40000.0, 30000.0, 10.0, None, None, false);
        assert!(!c.within_bound);
        assert_eq!(c.status, BoundStatus::Outside);
    }

    #[test]
    fn test_honor_published_range_passes() {
        // 15% pct diff but inside published range -> Within (envelope honored).
        let c = check_within_bounds(34500.0, 30000.0, 10.0, Some(25000.0), Some(35000.0), true);
        assert!(c.within_bound);
    }

    #[test]
    fn test_disabling_published_range_fails() {
        // Same numbers, but range NOT honored -> Outside.
        let c = check_within_bounds(34500.0, 30000.0, 10.0, Some(25000.0), Some(35000.0), false);
        assert!(!c.within_bound);
    }

    #[test]
    fn test_zero_reference_exact_match() {
        let c = check_within_bounds(0.0, 0.0, 10.0, None, None, false);
        assert!(c.within_bound);
        assert_eq!(c.percent_diff, 0.0);
    }

    #[test]
    fn test_zero_reference_nonzero_sim_is_outside() {
        let c = check_within_bounds(5.0, 0.0, 10.0, None, None, false);
        assert!(!c.within_bound);
        assert!(c.percent_diff.is_infinite());
    }

    #[test]
    fn test_assert_within_bounds_panics_on_fail() {
        let result = std::panic::catch_unwind(|| {
            assert_within_bounds(40000.0, 30000.0, 10.0, "energy test");
        });
        assert!(result.is_err(), "should panic when outside band");
    }

    #[test]
    fn test_assert_within_bounds_returns_check_on_pass() {
        let c = assert_within_bounds(30500.0, 30000.0, 10.0, "energy test");
        assert!(c.passed());
    }

    #[test]
    fn test_rp865_defaults_and_strict() {
        let d = HvacBestestToleranceConfig::rp865_defaults();
        assert!((d.energy_tolerance_percent - 10.0).abs() < 1e-9);
        assert!((d.demand_tolerance_percent - 15.0).abs() < 1e-9);
        assert!((d.plr_tolerance_percent - 10.0).abs() < 1e-9);
        assert!(d.honor_published_range);

        let s = HvacBestestToleranceConfig::strict(5.0);
        assert!((s.energy_tolerance_percent - 5.0).abs() < 1e-9);
        assert!(!s.honor_published_range);
    }

    #[test]
    fn test_report_from_results_structure() {
        let results = run_hvac_bestest();
        let defs = get_bestest_cases();
        let report = HvacBestestReport::from_results(
            &results,
            &defs,
            &HvacBestestToleranceConfig::default(),
        );

        // 5 cases * 5 metrics (energy, demand, plr50, plr75, plr100) = 25 rows.
        assert_eq!(report.cases.len(), 25);
        assert_eq!(report.summary.total_metrics, 25);
        assert_eq!(report.summary.passed + report.summary.failed, 25);

        // Each case contributes exactly 5 rows.
        for case_id in [
            HVACBestestCase::Case600,
            HVACBestestCase::Case610,
            HVACBestestCase::Case620,
            HVACBestestCase::Case630,
            HVACBestestCase::Case640,
        ] {
            let count = report.cases.iter().filter(|r| r.case_id == case_id).count();
            assert_eq!(count, 5, "case {:?} should have 5 metric rows", case_id);
        }
    }

    #[test]
    fn test_report_json_is_valid() {
        let results = run_hvac_bestest();
        let defs = get_bestest_cases();
        let report = HvacBestestReport::from_results(
            &results,
            &defs,
            &HvacBestestToleranceConfig::default(),
        );
        let json = report.to_json().expect("JSON serialization must succeed");
        // Must be parseable back, and contain no raw Infinity/NaN tokens.
        let reparsed: serde_json::Value =
            serde_json::from_str(&json).expect("JSON must round-trip");
        assert!(json.find("Infinity").is_none());
        assert!(json.find("NaN").is_none());
        assert!(reparsed.get("summary").is_some());
    }

    #[test]
    fn test_report_markdown_contains_tables() {
        let results = run_hvac_bestest();
        let defs = get_bestest_cases();
        let report = HvacBestestReport::from_results(
            &results,
            &defs,
            &HvacBestestToleranceConfig::default(),
        );
        let md = report.to_markdown();
        assert!(md.contains("# HVAC BESTEST (RP-865) Validation Report"));
        assert!(md.contains("## Summary"));
        assert!(md.contains("## Detailed Results"));
        assert!(md.contains("Annual Energy (kWh)"));
        assert!(md.contains("Peak Demand (W)"));
        assert!(md.contains("PLR 100% COP"));
    }

    #[test]
    fn test_bound_status_tokens() {
        assert_eq!(BoundStatus::Within.as_token(), "PASS");
        assert_eq!(BoundStatus::Outside.as_token(), "FAIL");
        assert!(BoundStatus::Within.is_passing());
        assert!(!BoundStatus::Outside.is_passing());
    }
}

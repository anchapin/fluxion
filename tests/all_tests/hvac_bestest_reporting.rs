//! Integration tests for HVAC BESTEST tolerance assertions + ASHRAE reporting.
//!
//! Issue #1759: per-case % deviation, within/outside bound flags, configurable
//! tolerances defaulting to the RP-865 published bands, and JSON + Markdown
//! report generation.

use fluxion::validation::hvac_bestest::{
    assert_within_bounds, assert_within_bounds_full, check_within_bounds, get_bestest_cases,
    run_hvac_bestest, BoundStatus, HVACBestestCase, HvacBestestReport, HvacBestestToleranceConfig,
    REFERENCE_ZERO_EPSILON,
};

/// A clearly-in-band value passes the assertion and returns a passing check.
#[test]
fn assert_within_bounds_passes_for_in_band_value() {
    let check = assert_within_bounds(102.0, 100.0, 5.0, "sanity in-band");
    assert!(check.passed(), "2% deviation must pass a ±5% band");
    assert_eq!(check.status, BoundStatus::Within);
    // Signed % diff preserved: +2%.
    assert!((check.percent_diff - 2.0).abs() < 1e-9);
}

/// An out-of-band value panics with an ASHRAE-style message.
#[test]
#[should_panic(expected = "OUTSIDE acceptance envelope")]
fn assert_within_bounds_panics_for_out_of_band_value() {
    let _ = assert_within_bounds(150.0, 100.0, 5.0, "sanity out-of-band");
}

/// The published-range envelope rescues a value that is outside the % band
/// but inside the reference min/max window.
#[test]
fn published_range_envelope_rescues_pct_failure() {
    // 20% over reference, far outside ±10% band...
    let check = assert_within_bounds_full(
        120.0,
        100.0,
        10.0,
        Some(110.0),
        Some(130.0),
        true,
        "range-rescued",
    );
    assert!(
        check.passed(),
        "in-range value must pass when range is honored"
    );
}

/// Same numbers, but with the range envelope disabled -> FAIL.
#[test]
fn disabling_range_envelope_reverts_to_pure_pct_band() {
    let check = check_within_bounds(120.0, 100.0, 10.0, Some(110.0), Some(130.0), false);
    assert!(!check.within_bound);
    assert_eq!(check.status, BoundStatus::Outside);
}

/// Zero-reference degenerate cases must not divide by zero.
#[test]
fn zero_reference_does_not_produce_nan_or_divide_by_zero() {
    // Exact zero match -> Within, finite 0% diff.
    let exact = check_within_bounds(0.0, 0.0, 10.0, None, None, false);
    assert!(exact.within_bound);
    assert!(exact.percent_diff.is_finite());
    assert!(exact.percent_diff.abs() < REFERENCE_ZERO_EPSILON);

    // Nonzero sim against zero reference -> Outside, +inf diff.
    let nonzero = check_within_bounds(5.0, 0.0, 10.0, None, None, false);
    assert!(!nonzero.within_bound);
    assert!(nonzero.percent_diff.is_infinite());
}

/// RP-865 default bands match the published acceptance criteria.
#[test]
fn rp865_default_tolerances_match_published_bands() {
    let cfg = HvacBestestToleranceConfig::default();
    assert_eq!(cfg.energy_tolerance_percent, 10.0);
    assert_eq!(cfg.demand_tolerance_percent, 15.0);
    assert_eq!(cfg.plr_tolerance_percent, 10.0);
    assert!(cfg.honor_published_range);
}

/// The full suite produces a well-formed report with the expected row count.
#[test]
fn report_covers_all_cases_and_metrics() {
    let results = run_hvac_bestest();
    let defs = get_bestest_cases();
    let report =
        HvacBestestReport::from_results(&results, &defs, &HvacBestestToleranceConfig::default());

    // 5 cases × 5 metrics (energy, demand, plr50/75/100).
    assert_eq!(report.cases.len(), 25);
    assert_eq!(report.summary.total_metrics, 25);
    assert_eq!(report.summary.passed + report.summary.failed, 25);

    // Every declared case is represented.
    for case in [
        HVACBestestCase::Case600,
        HVACBestestCase::Case610,
        HVACBestestCase::Case620,
        HVACBestestCase::Case630,
        HVACBestestCase::Case640,
    ] {
        assert!(
            report.cases.iter().any(|r| r.case_id == case),
            "case {:?} missing from report",
            case
        );
    }
}

/// The pass-rate is in a valid [0, 100] range and metrics are consistent.
#[test]
fn report_summary_is_self_consistent() {
    let results = run_hvac_bestest();
    let defs = get_bestest_cases();
    let report =
        HvacBestestReport::from_results(&results, &defs, &HvacBestestToleranceConfig::default());

    assert!(report.summary.pass_rate_percent >= 0.0 && report.summary.pass_rate_percent <= 100.0);
    assert!(report.summary.mean_abs_percent_diff >= 0.0);
    assert!(report.summary.max_abs_percent_diff >= report.summary.mean_abs_percent_diff);
}

/// JSON output is valid JSON (no raw Infinity/NaN tokens) and round-trips.
#[test]
fn report_json_is_valid_and_round_trips() {
    let results = run_hvac_bestest();
    let defs = get_bestest_cases();
    let report =
        HvacBestestReport::from_results(&results, &defs, &HvacBestestToleranceConfig::default());

    let json = report.to_json().expect("JSON serialization must succeed");
    assert!(
        !json.contains("Infinity") && !json.contains("NaN"),
        "JSON must not contain non-finite float tokens"
    );

    let reparsed: serde_json::Value =
        serde_json::from_str(&json).expect("JSON must parse back into a Value");
    assert!(reparsed.get("summary").is_some());
    assert!(reparsed.get("cases").is_some());
    let summary = reparsed.get("summary").unwrap();
    assert_eq!(
        summary.get("total_metrics").and_then(|v| v.as_u64()),
        Some(25)
    );
}

/// Markdown output contains the ASHRAE-style per-case pass/fail table.
#[test]
fn report_markdown_has_ashrae_style_tables() {
    let results = run_hvac_bestest();
    let defs = get_bestest_cases();
    let report =
        HvacBestestReport::from_results(&results, &defs, &HvacBestestToleranceConfig::default());

    let md = report.to_markdown();
    assert!(md.contains("# HVAC BESTEST (RP-865) Validation Report"));
    assert!(md.contains("## Summary"));
    assert!(md.contains("## Detailed Results"));
    assert!(md.contains("| Case | Metric | Simulated | Reference | % Diff | Tolerance | Status |"));
    assert!(md.contains("Annual Energy (kWh)"));
    assert!(md.contains("Peak Demand (W)"));
    assert!(md.contains("PLR 100% COP"));
    // Every status cell is either PASS or FAIL.
    assert!(md.contains("PASS") || md.contains("FAIL"));
}

/// A stricter tolerance preset lowers (or keeps) the pass rate vs the RP-865
/// defaults — i.e. tightening never increases passes.
#[test]
fn stricter_tolerance_never_increases_pass_count() {
    let results = run_hvac_bestest();
    let defs = get_bestest_cases();

    let default_report =
        HvacBestestReport::from_results(&results, &defs, &HvacBestestToleranceConfig::default());
    let strict_report =
        HvacBestestReport::from_results(&results, &defs, &HvacBestestToleranceConfig::strict(1.0));

    assert!(
        strict_report.summary.passed <= default_report.summary.passed,
        "1% strict band must not pass more rows than RP-865 defaults (strict={}, default={})",
        strict_report.summary.passed,
        default_report.summary.passed
    );
}

/// Tolerances are overridable: a custom config flows through to the per-row band.
#[test]
fn custom_tolerance_config_flows_to_rows() {
    let results = run_hvac_bestest();
    let defs = get_bestest_cases();
    let cfg = HvacBestestToleranceConfig {
        energy_tolerance_percent: 7.5,
        demand_tolerance_percent: 12.5,
        plr_tolerance_percent: 4.0,
        honor_published_range: false,
    };
    let report = HvacBestestReport::from_results(&results, &defs, &cfg);

    // Energy rows carry the configured 7.5% band.
    for row in report
        .cases
        .iter()
        .filter(|r| r.metric == "Annual Energy (kWh)")
    {
        assert!(
            (row.tolerance_band_percent - 7.5).abs() < 1e-9,
            "energy row band must be 7.5%"
        );
    }
    // Demand rows carry the configured 12.5% band.
    for row in report
        .cases
        .iter()
        .filter(|r| r.metric == "Peak Demand (W)")
    {
        assert!(
            (row.tolerance_band_percent - 12.5).abs() < 1e-9,
            "demand row band must be 12.5%"
        );
    }
    // PLR rows carry the configured 4.0% band.
    for row in report.cases.iter().filter(|r| r.metric.contains("PLR")) {
        assert!(
            (row.tolerance_band_percent - 4.0).abs() < 1e-9,
            "PLR row band must be 4.0%"
        );
    }
}

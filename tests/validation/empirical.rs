//! Empirical Validation Harness — Case Registration & Reporting Skeleton
//!
//! Issue #1803 / Plan T10.1
//!
//! Scaffolds the case-registration and reporting interfaces for the
//! **FLEXLAB empirical validation chain** (T10.2..T10.8). The follow-up
//! issues will register concrete FLEXLAB test-cell cases against this
//! harness; until then, the module is intentionally no-op / green.
//!
//! ## Scope (this issue)
//!
//! - Define `EmpiricalCase` (registration record) and
//!   `EmpiricalCaseRegistry` (the in-memory registry).
//! - Define `EmpiricalCaseReport` + `EmpiricalCaseStatus` (the per-run
//!   reporting record) and `EmpiricalHarnessSummary` (aggregate counts).
//! - Provide `render_markdown_report` so a CI job can emit a stable,
//!   deterministic summary for human + machine consumption.
//! - Compile and pass under `cargo test --release --lib` and
//!   `cargo test --test validation_empirical_harness`.
//!
//! ## Non-scope (deferred)
//!
//! - Concrete FLEXLAB test-cell data ingestion (T10.2).
//! - Time-alignment between Fluxion hourly output and FLEXLAB 1-min/15-min
//!   logged sensors (T10.3).
//! - ASHRAE Guideline 14 NMBE / CV(RMSE) computation against FLEXLAB
//!   reference (T10.4).
//! - Statistical-pass / fail thresholds per FLEXLAB test cell (T10.5..8).
//!
//! These are intentionally not implemented here so this PR remains a
//! skeleton and does not silently tune anything (see AGENTS.md:
//! "no parameter tuning to make system tests pass").

#![allow(dead_code)]

use std::collections::BTreeMap;

/// Stable identifier for a registered empirical validation case.
///
/// Convention: `<plan>.<case>` (e.g. `T10.2.flexlab_cell_12w`). Concrete
/// cases are registered by their owning issue.
pub type EmpiricalCaseId = String;

/// Lifecycle status of a single empirical-validation run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmpiricalCaseStatus {
    /// Case is registered but no run has been recorded yet.
    Pending,
    /// Run was intentionally skipped (missing data, environment gate, etc.).
    Skipped,
    /// Run completed and all registered checks passed.
    Passed,
    /// Run completed and at least one registered check failed.
    Failed,
}

/// Registration record for one empirical-validation case (e.g. one FLEXLAB
/// test cell / period pair).
///
/// Field set is intentionally minimal — follow-up issues (T10.2..8) will
/// extend this struct as concrete FLEXLAB metadata becomes known.
#[derive(Debug, Clone)]
pub struct EmpiricalCase {
    pub id: EmpiricalCaseId,
    pub facility: &'static str,
    pub case_name: &'static str,
    pub reference_source: &'static str,
    pub description: &'static str,
}

/// Per-run reporting record.
#[derive(Debug, Clone)]
pub struct EmpiricalCaseReport {
    pub id: EmpiricalCaseId,
    pub status: EmpiricalCaseStatus,
    pub note: Option<String>,
}

/// Aggregate counts derived from a registry's report history.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EmpiricalHarnessSummary {
    pub registered: usize,
    pub pending: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

/// In-memory empirical-case registry. Cases are registered up front; each
/// run appends an `EmpiricalCaseReport` keyed by case id.
#[derive(Debug, Default)]
pub struct EmpiricalCaseRegistry {
    cases: Vec<EmpiricalCase>,
    reports: BTreeMap<EmpiricalCaseId, Vec<EmpiricalCaseReport>>,
}

impl EmpiricalCaseRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, case: EmpiricalCase) {
        self.cases.push(case);
    }

    pub fn record(&mut self, report: EmpiricalCaseReport) {
        self.reports
            .entry(report.id.clone())
            .or_default()
            .push(report);
    }

    pub fn cases(&self) -> &[EmpiricalCase] {
        &self.cases
    }

    pub fn reports_for(&self, id: &str) -> Option<&[EmpiricalCaseReport]> {
        self.reports.get(id).map(Vec::as_slice)
    }

    pub fn summary(&self) -> EmpiricalHarnessSummary {
        let mut s = EmpiricalHarnessSummary::default();
        s.registered = self.cases.len();
        for reports in self.reports.values() {
            for r in reports {
                match r.status {
                    EmpiricalCaseStatus::Pending => s.pending += 1,
                    EmpiricalCaseStatus::Skipped => s.skipped += 1,
                    EmpiricalCaseStatus::Passed => s.passed += 1,
                    EmpiricalCaseStatus::Failed => s.failed += 1,
                }
            }
        }
        s
    }
}

/// Render the harness summary as a deterministic Markdown report. The
/// output is stable so CI can grep / diff it across runs.
pub fn render_markdown_report(s: &EmpiricalHarnessSummary) -> String {
    format!(
        "# Empirical Validation Harness\n\n\
         - Registered cases: {registered}\n\
         - Pending:          {pending}\n\
         - Passed:           {passed}\n\
         - Failed:           {failed}\n\
         - Skipped:          {skipped}\n",
        registered = s.registered,
        pending = s.pending,
        passed = s.passed,
        failed = s.failed,
        skipped = s.skipped,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_summarizes_to_zero() {
        let registry = EmpiricalCaseRegistry::default();
        let s = registry.summary();
        assert_eq!(s.registered, 0);
        assert_eq!(s.pending, 0);
        assert_eq!(s.passed, 0);
        assert_eq!(s.failed, 0);
        assert_eq!(s.skipped, 0);
    }

    #[test]
    fn registration_and_reporting_round_trip() {
        let mut r = EmpiricalCaseRegistry::new();
        r.register(EmpiricalCase {
            id: "T10.2".into(),
            facility: "FLEXLAB",
            case_name: "placeholder",
            reference_source: "LBNL FLEXLAB (TBD by T10.2)",
            description: "placeholder registration; concrete case lands in T10.2",
        });
        assert_eq!(r.cases().len(), 1);

        r.record(EmpiricalCaseReport {
            id: "T10.2".into(),
            status: EmpiricalCaseStatus::Skipped,
            note: Some("skeleton — case lands in T10.2".into()),
        });

        let s = r.summary();
        assert_eq!(s.registered, 1);
        assert_eq!(s.skipped, 1);
        assert_eq!(s.passed, 0);
        assert_eq!(s.failed, 0);
        assert_eq!(s.pending, 0);

        let reports = r
            .reports_for("T10.2")
            .expect("T10.2 report must be present");
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].status, EmpiricalCaseStatus::Skipped);
    }

    #[test]
    fn markdown_report_is_deterministic_and_complete() {
        let s = EmpiricalHarnessSummary {
            registered: 3,
            pending: 1,
            passed: 2,
            failed: 0,
            skipped: 0,
        };
        let md = render_markdown_report(&s);
        assert!(md.contains("# Empirical Validation Harness"));
        assert!(md.contains("Registered cases: 3"));
        assert!(md.contains("Pending:          1"));
        assert!(md.contains("Passed:           2"));
        assert!(md.contains("Failed:           0"));
        assert!(md.contains("Skipped:          0"));
    }

    #[test]
    fn status_classification_covers_all_variants() {
        let mut r = EmpiricalCaseRegistry::new();
        for (id, st) in [
            ("a", EmpiricalCaseStatus::Pending),
            ("b", EmpiricalCaseStatus::Skipped),
            ("c", EmpiricalCaseStatus::Passed),
            ("d", EmpiricalCaseStatus::Failed),
        ] {
            r.record(EmpiricalCaseReport {
                id: id.into(),
                status: st,
                note: None,
            });
        }
        let s = r.summary();
        assert_eq!(s.pending, 1);
        assert_eq!(s.skipped, 1);
        assert_eq!(s.passed, 1);
        assert_eq!(s.failed, 1);
    }
}

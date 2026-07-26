//! HVAC BESTEST CI runner stub (issue #1756).
//!
//! Establishes the runner plumbing for the HVAC BESTEST CI-check pipeline so
//! that follow-on issues (#1755, #1757, #1758, #1759) can register real
//! analytical and comparative cases without touching CI wiring.
//!
//! ## Design
//!
//! The runner emits a structured **pass / skip / fail** report. During rollout
//! the runner registers zero cases, producing an always-passing empty report
//! that keeps CI green while real assertions are still TODO. When a case is
//! registered as [`CaseStatus::Skip`] it does **not** fail the CI gate — only
//! an explicit [`CaseStatus::Fail`] does.
//!
//! ## Future integration
//!
//! Follow-on issues will:
//! 1. Implement case runners in `analytical.rs` / `comparative.rs`.
//! 2. Call [`HvacBestestRunner::register`] for each case.
//! 3. The `#[test]` functions below already enforce the gate semantics.

use std::fmt;

// ---------------------------------------------------------------------------
// Report data model
// ---------------------------------------------------------------------------

/// Outcome status for a single HVAC BESTEST case.
///
/// The three-valued enum allows follow-on issues to mark a case as `Skip`
/// during incremental rollout without failing the CI gate. Only `Fail` is
/// treated as a hard failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseStatus {
    /// Case passed within its acceptance tolerance.
    Pass,
    /// Case is registered but not yet implemented; permitted during rollout.
    Skip,
    /// Case failed an explicit assertion; fails the CI gate.
    Fail,
}

/// Structured outcome for a single registered case.
#[derive(Debug, Clone)]
pub struct CaseOutcome {
    /// Case identifier (e.g. `"AE101"`, `"E100"`).
    pub case_id: String,
    /// Pass / Skip / Fail status.
    pub status: CaseStatus,
    /// Human-readable detail: tolerance band, skip reason, or failure message.
    pub detail: String,
}

/// Aggregated pass/skip/fail report for the entire HVAC BESTEST suite.
#[derive(Debug, Clone, Default)]
pub struct BestestReport {
    /// Ordered list of per-case outcomes.
    pub outcomes: Vec<CaseOutcome>,
}

impl BestestReport {
    /// Returns `(pass_count, skip_count, fail_count)`.
    pub fn counts(&self) -> (usize, usize, usize) {
        let pass = self
            .outcomes
            .iter()
            .filter(|o| o.status == CaseStatus::Pass)
            .count();
        let skip = self
            .outcomes
            .iter()
            .filter(|o| o.status == CaseStatus::Skip)
            .count();
        let fail = self
            .outcomes
            .iter()
            .filter(|o| o.status == CaseStatus::Fail)
            .count();
        (pass, skip, fail)
    }

    /// Returns `true` if any case has [`CaseStatus::Fail`].
    ///
    /// This is the single predicate the CI gate asserts against.
    pub fn has_failures(&self) -> bool {
        self.outcomes.iter().any(|o| o.status == CaseStatus::Fail)
    }

    /// Total number of registered cases.
    pub fn total(&self) -> usize {
        self.outcomes.len()
    }
}

impl fmt::Display for BestestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (pass, skip, fail) = self.counts();
        writeln!(f, "=== HVAC BESTEST CI Report ===")?;
        writeln!(
            f,
            "Registered: {}  |  Pass: {}  Skip: {}  Fail: {}",
            self.total(),
            pass,
            skip,
            fail
        )?;
        if self.outcomes.is_empty() {
            writeln!(f, "  (no cases registered — dummy stub mode, issue #1756)")?;
        } else {
            for o in &self.outcomes {
                writeln!(f, "  [{:?}] {:<10} {}", o.status, o.case_id, o.detail)?;
            }
        }
        writeln!(f, "==============================")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Runner stub
// ---------------------------------------------------------------------------

/// HVAC BESTEST runner stub.
///
/// Iterates registered cases and emits a structured pass/skip/fail report
/// ([`BestestReport`]). During initial rollout (this issue) the runner
/// registers zero cases, producing an always-passing empty report.
///
/// Follow-on issues register cases via [`Self::register`] once real
/// simulation logic lands in `analytical.rs` / `comparative.rs`.
pub struct HvacBestestRunner {
    cases: Vec<CaseOutcome>,
}

impl Default for HvacBestestRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl HvacBestestRunner {
    /// Create a runner with no registered cases (dummy stub mode).
    pub fn new() -> Self {
        Self { cases: Vec::new() }
    }

    /// Register a case outcome.
    ///
    /// Called by follow-on case modules (#1755–#1759) to report their result.
    /// The runner collects these and [`Self::run`] assembles the final report.
    pub fn register(&mut self, case_id: &str, status: CaseStatus, detail: &str) {
        self.cases.push(CaseOutcome {
            case_id: case_id.to_string(),
            status,
            detail: detail.to_string(),
        });
    }

    /// Produce the structured pass/skip/fail report.
    ///
    /// In the current stub this returns the pre-registered outcomes directly.
    /// When real simulation cases are added, this method will execute each
    /// case runner and evaluate acceptance tolerances before assembling the
    /// report.
    pub fn run(&self) -> BestestReport {
        BestestReport {
            outcomes: self.cases.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// CI entry-point tests
//
// `cargo test --test hvac_bestest` discovers these. During rollout every test
// passes; the suite verifies pipeline plumbing and gate semantics only.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Marker test: confirms the dummy runner stub pipeline is wired and
    /// `cargo test --test hvac_bestest` executes end-to-end. This is the
    /// always-green CI anchor.
    #[test]
    fn test_hvac_bestest_pipeline_smoke() {
        let runner = HvacBestestRunner::new();
        let report = runner.run();

        // Dummy stub: zero registered cases, always passes.
        assert_eq!(report.total(), 0);
        assert!(!report.has_failures());

        let (pass, skip, fail) = report.counts();
        assert_eq!((pass, skip, fail), (0, 0, 0));

        // Print the report artifact for CI inspection.
        println!("{report}");
    }

    /// Verifies that an empty runner (rollout mode) always passes the gate.
    #[test]
    fn test_empty_runner_passes_gate() {
        let runner = HvacBestestRunner::new();
        let report = runner.run();

        assert!(
            !report.has_failures(),
            "empty report must not fail the gate"
        );
    }

    /// Verifies that `Skip` cases do **not** fail the CI gate. This is the
    /// core rollout guarantee: follow-on issues can register incomplete cases
    /// as `Skip` without breaking CI.
    #[test]
    fn test_skip_does_not_fail_gate() {
        let mut runner = HvacBestestRunner::new();
        runner.register("AE101", CaseStatus::Skip, "not yet implemented");
        runner.register("E100", CaseStatus::Skip, "pending reference data");

        let report = runner.run();
        let (pass, skip, fail) = report.counts();

        assert_eq!((pass, skip, fail), (0, 2, 0));
        assert!(!report.has_failures(), "skips must not fail the gate");

        println!("{report}");
    }

    /// Verifies that `Pass` cases are counted correctly and do not fail.
    #[test]
    fn test_pass_case_counted() {
        let mut runner = HvacBestestRunner::new();
        runner.register(
            "AE101",
            CaseStatus::Pass,
            "energy error 1.2% within 10% tolerance",
        );

        let report = runner.run();
        let (pass, skip, fail) = report.counts();

        assert_eq!((pass, skip, fail), (1, 0, 0));
        assert!(!report.has_failures());
    }

    /// Verifies that a `Fail` case is detected by the gate. The test asserts
    /// `has_failures()` returns `true` — it does **not** itself fail.
    #[test]
    fn test_fail_detected_by_gate() {
        let mut runner = HvacBestestRunner::new();
        runner.register("AE445", CaseStatus::Fail, "sensible load 25% out of band");

        let report = runner.run();
        let (pass, skip, fail) = report.counts();

        assert_eq!((pass, skip, fail), (0, 0, 1));
        assert!(
            report.has_failures(),
            "a Fail case must be detected by the gate"
        );
    }

    /// Verifies the `Display` implementation renders the report for CI
    /// artifact inspection.
    #[test]
    fn test_report_display() {
        let mut runner = HvacBestestRunner::new();
        runner.register("AE101", CaseStatus::Pass, "ok");
        runner.register("AE200", CaseStatus::Skip, "TODO");

        let report = runner.run();
        let rendered = format!("{report}");

        assert!(rendered.contains("HVAC BESTEST CI Report"));
        assert!(rendered.contains("Pass: 1"));
        assert!(rendered.contains("Skip: 1"));
        assert!(rendered.contains("Fail: 0"));
    }
}

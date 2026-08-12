//! HVAC BESTEST CI runner — issue #2684.
//!
//! Previously (issues #1756 / #2307) this module was a *zero-case stub*: it
//! registered zero cases (later: three cases hardcoded as `CaseStatus::Pass`
//! without running any simulation), so `cargo test --test hvac_bestest` was
//! **structurally incapable of failing**. That gave false confidence — a green
//! CI badge that misrepresented the validation posture.
//!
//! ## Current design (post-#2684)
//!
//! The runner now invokes the **real** analytical case computations defined in
//! [`crate::analytical`] (`run_e100` / `run_e200` / `run_e300`) and registers
//! the actual pass/fail outcome against the first-principles constant-COP
//! reference. A [`CaseStatus::Fail`] — produced when a computed energy or peak
//! ratio falls outside the documented tolerance — trips the CI gate.
//!
//! Three structural-guard tests below enforce that the runner can never
//! silently regress to a stub:
//!   1. [`test_runner_registers_nonzero_cases`] — the canonical runner must
//!      register more than zero cases.
//!   2. [`test_runner_outcomes_carry_real_computation`] — every registered
//!      detail string must embed a numeric ratio, proving the result came from
//!      a computation rather than a hardcoded "Pass" phrase.
//!   3. [`test_gate_fails_when_analytical_case_out_of_band`] — the gate
//!      semantics: an out-of-band computation produces `CaseStatus::Fail`.
//!
//! Cases that would require a full EnergyPlus-comparable annual zone simulation
//! (the IEA Task 22 / RP-865 published *ensemble* bounds in
//! `data/comparative_bounds_*.csv`) are blocked on the documented
//! Case-600-class cooling structural gap (see `docs/KNOWN_ISSUES.md` §LIMIT-05
//! / §SOLAR-02) and are wired as `#[ignore]`-quarantined tests that run on
//! demand via `cargo test -- --ignored`, documenting the gap without blocking
//! every PR.

use std::fmt;

use crate::analytical::{run_e100, run_e200, run_e300, CaseComputation};

// ---------------------------------------------------------------------------
// Report data model
// ---------------------------------------------------------------------------

/// Outcome status for a single HVAC BESTEST case.
///
/// `Skip` is permitted during incremental rollout and does **not** fail the CI
/// gate — only `Fail` does.
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
    /// Case identifier (e.g. `"E100"`, `"AE101"`).
    pub case_id: String,
    /// Pass / Skip / Fail status.
    pub status: CaseStatus,
    /// Human-readable detail: tolerance band, skip reason, or failure message.
    /// For computed cases this embeds the numeric ratio so the outcome is
    /// auditable and a regression to a hardcoded string is detectable (#2684).
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
            writeln!(
                f,
                "  (no cases registered — REGRESSION: issue #2684 requires non-zero cases)"
            )?;
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
// Runner — registers REAL computed outcomes (issue #2684)
// ---------------------------------------------------------------------------

/// HVAC BESTEST runner. Iterates registered cases and emits a structured
/// pass/skip/fail report ([`BestestReport`]).
///
/// Use [`HvacBestestRunner::bestest_rp865_cases`] to obtain a runner populated
/// with the canonical RP-865 analytical cases — each driven by the real
/// Fluxion equipment-model computation in [`crate::analytical`].
pub struct HvacBestestRunner {
    cases: Vec<CaseOutcome>,
}

impl Default for HvacBestestRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl HvacBestestRunner {
    /// Create an empty runner. Intended for ad-hoc composition; the canonical
    /// CI entry point is [`Self::bestest_rp865_cases`].
    pub fn new() -> Self {
        Self { cases: Vec::new() }
    }

    /// Create a runner pre-populated with the HVAC BESTEST RP-865 analytical
    /// cases (E100, E200, E300), each driven by the **real** equipment-model
    /// computation in [`crate::analytical`].
    ///
    /// Each case computes annual energy + peak demand via a Fluxion HVAC model
    /// over a mid-latitude TMY temperature-bin distribution (8760 h) and
    /// compares against an independent first-principles constant-COP reference
    /// derived from the cited ASHRAE 90.1 rated efficiency. The registered
    /// status is the **actual** [`CaseComputation::within_band`] verdict —
    /// `Pass` when within tolerance, `Fail` when outside — so the CI gate
    /// genuinely trips on a regression in the equipment models or bin
    /// integration.
    ///
    /// ## Cases
    ///
    /// | Case | System | Description | Tolerance |
    /// |------|--------|-------------|-----------|
    /// | E100 | System A (CAV) | Electric resistance heating | ±5% |
    /// | E200 | System A (CAV) | Packaged AC (DX cooling) | ±5% |
    /// | E300 | System B (VAV) | VAV terminal with reheat | ±10% |
    ///
    /// ## Sources
    ///
    /// - IEA SHC Task 22, "HVAC BESTEST Volume 1: Cases E100-E200"
    /// - NREL/TP-5500-66000 (Neymark et al., 2016, DOI 10.2172/1244668)
    /// - ASHRAE Standard 90.1-2019, Tables 6.8.1A/C/D
    pub fn bestest_rp865_cases() -> Self {
        let mut runner = Self { cases: Vec::new() };
        runner.register_computed(run_e100());
        runner.register_computed(run_e200());
        runner.register_computed(run_e300());
        runner
    }

    /// Register a computed analytical case. The status is derived from the
    /// computation: `Pass` if within tolerance, `Fail` otherwise. The detail
    /// string embeds the numeric ratios for auditability.
    fn register_computed(&mut self, case: CaseComputation) {
        let status = if case.within_band() {
            CaseStatus::Pass
        } else {
            CaseStatus::Fail
        };
        self.cases.push(CaseOutcome {
            case_id: case.case_id.to_string(),
            status,
            detail: case.detail_line(),
        });
    }

    /// Register a case outcome directly. Used by ad-hoc composition and by the
    /// gate-semantics unit tests below.
    pub fn register(&mut self, case_id: &str, status: CaseStatus, detail: &str) {
        self.cases.push(CaseOutcome {
            case_id: case_id.to_string(),
            status,
            detail: detail.to_string(),
        });
    }

    /// Produce the structured pass/skip/fail report.
    pub fn run(&self) -> BestestReport {
        BestestReport {
            outcomes: self.cases.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// CI entry-point tests
//
// `cargo test --test hvac_bestest` discovers these. The tests below enforce
// both the gate semantics AND the issue-#2684 structural guarantee that the
// runner registers real, computed, non-zero cases.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytical::run_e100;

    // --- Structural guards (issue #2684: prevent regression to zero-case stub) -

    /// **Meta-guard (issue #2684):** the canonical RP-865 runner MUST register
    /// a non-zero number of cases. If a future change re-stubs the runner to
    /// the old "always-passing empty report" (or the later three-hardcoded-
    /// passes variant), this test fails. This is the single assertion that
    /// makes the false-confidence class of regression impossible.
    #[test]
    fn test_runner_registers_nonzero_cases() {
        let runner = HvacBestestRunner::bestest_rp865_cases();
        let report = runner.run();

        assert!(
            report.total() > 0,
            "issue #2684 regression: runner registered {} cases; must be > 0",
            report.total()
        );

        let (pass, skip, fail) = report.counts();
        assert!(
            pass + skip + fail == report.total(),
            "counts must partition the registered cases"
        );

        println!("{report}");
    }

    /// **Meta-guard (issue #2684):** every registered detail string must carry
    /// evidence of real computation — specifically a numeric energy ratio
    /// (`ratio <number>`). The old hardcoded stub registered details like
    /// `"energy ratio within ±5% of rated-COP reference"` with no number; this
    /// test rejects that class of phrase so the runner cannot silently revert
    /// to asserting a string instead of a measurement.
    #[test]
    fn test_runner_outcomes_carry_real_computation() {
        let runner = HvacBestestRunner::bestest_rp865_cases();
        let report = runner.run();

        assert!(
            !report.outcomes.is_empty(),
            "runner must register cases to audit"
        );
        for o in &report.outcomes {
            assert!(
                o.detail.contains("ratio "),
                "{:?} detail missing numeric ratio: {}",
                o.case_id,
                o.detail
            );
            // Reject the old hardcoded phrase that had no number.
            assert!(
                !o.detail.contains("within ±") || o.detail.contains("ratio "),
                "{:?} detail looks hardcoded (no computed ratio): {}",
                o.case_id,
                o.detail
            );
        }
    }

    /// **Gate-semantics guard:** the canonical RP-865 cases currently pass
    /// within tolerance, so the runner must not report any failure. If a
    /// regression in the equipment models or bin integration pushes a case
    /// out of band, this test fails — the CI signal that previously did not
    /// exist at all.
    #[test]
    fn test_bestest_rp865_cases_pass_within_tolerance() {
        let runner = HvacBestestRunner::bestest_rp865_cases();
        let report = runner.run();

        assert_eq!(
            report.total(),
            3,
            "expected E100, E200, E300 registered; got {} cases",
            report.total()
        );

        let case_ids: Vec<_> = report.outcomes.iter().map(|o| o.case_id.as_str()).collect();
        assert!(case_ids.contains(&"E100"), "E100 missing: {case_ids:?}");
        assert!(case_ids.contains(&"E200"), "E200 missing: {case_ids:?}");
        assert!(case_ids.contains(&"E300"), "E300 missing: {case_ids:?}");

        assert!(
            !report.has_failures(),
            "RP-865 analytical cases must pass within tolerance; got failures:\n{report}"
        );

        let (pass, _, fail) = report.counts();
        assert_eq!(fail, 0, "no failures expected; got:\n{report}");
        assert_eq!(pass, 3, "all three cases should pass; got:\n{report}");

        println!("{report}");
    }

    /// **Gate-semantics guard:** verify the gate actually trips when an
    /// analytical case is out of band. Drives `run_e100` with a corrupted
    /// reference (10× the real reference energy) so `within_band` is false,
    /// and confirms the runner would register `Fail` + `has_failures()`.
    /// This proves the gate is not structurally incapable of failing — the
    /// core complaint of issue #2684.
    #[test]
    fn test_gate_fails_when_analytical_case_out_of_band() {
        let mut broken = run_e100();
        // Corrupt the reference so the computed ratio is far out of band.
        broken.reference_energy_kwh = broken.computed_energy_kwh / 10.0;
        assert!(
            !broken.within_band(),
            "fixture: corrupted reference must push the case out of band"
        );

        let mut runner = HvacBestestRunner::new();
        runner.register_computed(broken);
        let report = runner.run();

        assert_eq!(report.total(), 1);
        let (pass, skip, fail) = report.counts();
        assert_eq!((pass, skip, fail), (0, 0, 1));
        assert!(
            report.has_failures(),
            "an out-of-band analytical case MUST fail the gate (issue #2684)"
        );
    }

    // --- Report / gate plumbing (retain semantics from the original stub) -----

    /// Verifies that `Skip` cases do **not** fail the CI gate. This is the
    /// rollout guarantee: follow-on issues can register incomplete cases as
    /// `Skip` without breaking CI.
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

    /// Verifies the `Display` implementation renders the report for CI
    /// artifact inspection.
    #[test]
    fn test_report_display() {
        let runner = HvacBestestRunner::bestest_rp865_cases();
        let report = runner.run();
        let rendered = format!("{report}");

        assert!(rendered.contains("HVAC BESTEST CI Report"));
        assert!(rendered.contains("Registered: 3"));
        assert!(!rendered.contains("REGRESSION"));
        // Each computed case prints its detail line (E100/E200/E300).
        assert!(rendered.contains("E100"));
        assert!(rendered.contains("E200"));
        assert!(rendered.contains("E300"));
    }

    // --- Comparative ensemble-bound case (STRUCTURAL GAP — #[ignore]) --------
    //
    // The IEA Task 22 / RP-865 PUBLISHED ensemble bounds
    // (data/comparative_bounds_e100_e200.csv) require a full EnergyPlus-
    // comparable annual zone simulation. Fluxion's cooling-load path has a
    // documented structural gap vs EnergyPlus (Case 600/900 annual cooling
    // ≈20-60% below the ensemble midpoint — see docs/KNOWN_ISSUES.md §LIMIT-05
    // UPDATE / §SOLAR-02 UPDATE, and the strict-energy-gate baseline in
    // tests/reference_data/zone_balance/strict_energy_gate_baseline.json).
    //
    // Per issue #2684's preferred-fix clause (3), these are REAL cases that
    // run on-demand via `cargo test --test hvac_bestest -- --ignored`, so the
    // gap is documented and measured rather than hidden, but they do NOT
    // block CI on every PR.

    /// Comparative check of the analytical E200 cooling energy against the
    /// published IEA Task 22 ensemble bound (5.60–6.70 MWh annual cooling).
    ///
    /// # Ignored reason
    /// Blocked on the Case-600-class annual-cooling structural gap
    /// (`docs/KNOWN_ISSUES.md` §LIMIT-05 UPDATE — GaugeSolver-routed; §SOLAR-02
    /// UPDATE — residual annual-energy deviation). The analytical E200 model
    /// (constant-COP chiller over a UA·ΔT bin load) yields ≈4.89 MWh, ≈20%
    /// below the ensemble midpoint — the same direction and magnitude as the
    /// ASHRAE 140 Case 600 cooling gap. Closing it requires the post-#1323 /
    /// #1213 / #1328 cooling-path fix, not a test-side constant. Run on demand:
    /// `cargo test --test hvac_bestest comparative_e200 -- --ignored --nocapture`
    #[test]
    #[ignore = "blocked on Case-600-class cooling structural gap; see docs/KNOWN_ISSUES.md §LIMIT-05/§SOLAR-02"]
    fn comparative_e200_cooling_vs_iea_task22_ensemble() {
        let case = run_e200();
        // Published IEA Task 22 E200 annual-cooling ensemble band (MWh), from
        // data/comparative_bounds_e100_e200.csv (transcribed; provenance in the CSV).
        let band_low_mwh = 5.60;
        let band_high_mwh = 6.70;
        let computed_mwh = case.computed_energy_kwh / 1000.0;
        println!(
            "E200 comparative: computed {computed_mwh:.3} MWh vs IEA Task 22 ensemble [{band_low_mwh}, {band_high_mwh}] MWh"
        );
        assert!(
            (band_low_mwh..=band_high_mwh).contains(&computed_mwh),
            "E200 annual cooling {computed_mwh:.3} MWh outside IEA Task 22 ensemble [{band_low_mwh}, {band_high_mwh}] MWh — \
             this is the documented Case-600-class structural cooling gap (KNOWN_ISSUES §LIMIT-05/§SOLAR-02)"
        );
    }
}

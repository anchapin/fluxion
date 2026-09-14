//! Tests for `fluxion::validation::performance::completion::Phase47CompletionValidator`.
//!
//! Replaces a near-empty stub that asserted only that
//! `completion_percentage >= 0.0` (trivially always true) with substantive
//! coverage of the validator's API shape and the consistency of the
//! returned `PhaseCompletionResult` / `PhaseCompletionReport`. See GitHub
//! issue #2564.
//!
//! The validator's internal "checks" inspect the working directory for
//! specific files (e.g. `benches/performance.rs`,
//! `tests/performance_integration_test.rs`); these are tested as
//! *structural* properties — the count of requirements, the shape of the
//! IDs, the math between passed/total and `completion_percentage` — not
//! as a CI dependency on every file existing.

use fluxion::validation::performance::completion::{
    Phase47CompletionValidator, PhaseCompletionReport, PhaseCompletionResult, RequirementResult,
};

#[test]
fn phase_47_validator_defines_expected_number_of_requirements() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    assert_eq!(
        result.requirements.len(),
        14,
        "Phase 47 must track exactly 14 PERF-NN requirements"
    );
}

#[test]
fn phase_47_requirement_ids_match_perf_nn_pattern() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    let mut ids: Vec<&str> = result.requirements.iter().map(|r| r.id.as_str()).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![
            "PERF-01", "PERF-02", "PERF-03", "PERF-04", "PERF-05", "PERF-06", "PERF-07", "PERF-08",
            "PERF-09", "PERF-10", "PERF-11", "PERF-12", "PERF-13", "PERF-14",
        ]
    );
}

#[test]
fn phase_47_requirement_descriptions_are_non_empty() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    for req in &result.requirements {
        assert!(
            !req.description.trim().is_empty(),
            "Requirement {} has an empty description",
            req.id
        );
    }
}

#[test]
fn phase_47_completion_percentage_is_in_closed_unit_interval() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    let pct = result.completion_percentage;
    assert!(
        (0.0..=100.0).contains(&pct),
        "completion_percentage={pct} must be in [0, 100]"
    );
}

#[test]
fn phase_47_completion_percentage_matches_passed_count() {
    // (passed / total) * 100 — exact equality with `completion_percentage`
    // verifies the formula is wired through `validate_all_requirements`.
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    let total = result.requirements.len() as f64;
    let passed = result.requirements.iter().filter(|r| r.passed).count() as f64;
    let expected_pct = (passed / total) * 100.0;
    assert!(
        (result.completion_percentage - expected_pct).abs() < 1e-9,
        "completion_percentage={} != expected {} (passed={passed}, total={total})",
        result.completion_percentage,
        expected_pct
    );
}

#[test]
fn phase_47_all_passed_implies_100_percent() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    if result.all_passed {
        assert!(
            (result.completion_percentage - 100.0).abs() < 1e-9,
            "all_passed=true but completion_percentage={}",
            result.completion_percentage
        );
        assert_eq!(
            result.requirements.iter().filter(|r| r.passed).count(),
            result.requirements.len()
        );
    }
}

#[test]
fn phase_47_any_failed_implies_less_than_100_percent() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    if !result.all_passed {
        assert!(
            result.completion_percentage < 100.0,
            "all_passed=false but completion_percentage=100.0"
        );
        assert!(
            result.requirements.iter().any(|r| !r.passed),
            "all_passed=false but every requirement is passed"
        );
    }
}

#[test]
fn phase_47_passed_count_plus_failed_count_equals_total() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    let passed = result.requirements.iter().filter(|r| r.passed).count();
    let failed = result.requirements.iter().filter(|r| !r.passed).count();
    assert_eq!(
        passed + failed,
        result.requirements.len(),
        "passed ({passed}) + failed ({failed}) must equal total ({})",
        result.requirements.len()
    );
}

#[test]
fn phase_47_each_requirement_carries_an_id_matching_its_position() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    for (idx, req) in result.requirements.iter().enumerate() {
        let expected_id = format!("PERF-{:02}", idx + 1);
        assert_eq!(
            req.id, expected_id,
            "Requirement at index {idx} has id {} but expected {expected_id}",
            req.id
        );
    }
}

#[test]
fn phase_47_completion_report_status_reflects_all_passed() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    let report: PhaseCompletionReport = validator.generate_completion_report(&result);
    let expected_status = if result.all_passed {
        "COMPLETE"
    } else {
        "INCOMPLETE"
    };
    assert_eq!(report.status, expected_status);
}

#[test]
fn phase_47_completion_report_carries_through_result_fields() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    let report = validator.generate_completion_report(&result);

    assert_eq!(report.requirements.len(), result.requirements.len());
    assert!(
        (report.completion_percentage - result.completion_percentage).abs() < 1e-9,
        "report.completion_percentage={} != result.completion_percentage={}",
        report.completion_percentage,
        result.completion_percentage
    );
    assert_eq!(report.phase, "47-performance-validation-optimization");
}

#[test]
fn phase_47_completion_report_summary_mentions_total_when_complete() {
    // When all requirements pass, the summary must explicitly mention
    // the total count.
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    if result.all_passed {
        let report = validator.generate_completion_report(&result);
        assert!(
            report
                .summary
                .contains(&result.requirements.len().to_string()),
            "complete-phase summary must mention total requirement count, got: {}",
            report.summary
        );
    }
}

#[test]
fn phase_47_completion_report_summary_mentions_passed_and_failed_when_incomplete() {
    let validator = Phase47CompletionValidator::new();
    let result = validator.validate_all_requirements();
    if !result.all_passed {
        let passed = result.requirements.iter().filter(|r| r.passed).count();
        let failed = result.requirements.len() - passed;
        let report = validator.generate_completion_report(&result);
        assert!(
            report.summary.contains(&passed.to_string()),
            "incomplete summary must mention passed count {passed}: {}",
            report.summary
        );
        assert!(
            report.summary.contains(&failed.to_string()),
            "incomplete summary must mention failed count {failed}: {}",
            report.summary
        );
    }
}

#[test]
fn phase_47_requirement_result_struct_fields_round_trip() {
    // The `RequirementResult` struct is `Debug + Clone + Serialize +
    // Deserialize` — exercise that boundary explicitly via Debug
    // formatting so any future field-removal is caught here.
    let sample = RequirementResult {
        id: "PERF-XX".to_string(),
        description: "sample".to_string(),
        passed: true,
    };
    let cloned = sample.clone();
    assert_eq!(cloned.id, "PERF-XX");
    assert!(cloned.passed);
    let dbg = format!("{sample:?}");
    assert!(dbg.contains("PERF-XX"));
    assert!(dbg.contains("sample"));
}

#[test]
fn phase_47_result_struct_round_trip_is_deterministic() {
    // The structural shape of the validator is deterministic: same
    // number of requirements, same IDs, same descriptions. We do NOT
    // assert that `passed` flags are identical across calls because
    // PERF-14 (run_comprehensive_tests) shells out to `cargo test
    // --test performance_*`, whose outcome depends on external state
    // (which test binaries have been built in the current target
    // directory) and so can legitimately differ between invocations.
    let validator = Phase47CompletionValidator::new();
    let r1 = validator.validate_all_requirements();
    let r2 = validator.validate_all_requirements();

    assert_eq!(r1.requirements.len(), r2.requirements.len());
    for (a, b) in r1.requirements.iter().zip(r2.requirements.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.description, b.description);
    }
}

#[test]
fn phase_47_validate_is_idempotent_for_structural_fields() {
    // Same rationale as `phase_47_result_struct_round_trip_is_deterministic`:
    // assert structural invariance, not the file-system-derived
    // `passed` flag.
    let validator = Phase47CompletionValidator::new();
    let r1: PhaseCompletionResult = validator.validate_all_requirements();
    let r2: PhaseCompletionResult = validator.validate_all_requirements();
    assert_eq!(r1.requirements.len(), r2.requirements.len());
    for (a, b) in r1.requirements.iter().zip(r2.requirements.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.description, b.description);
    }
}

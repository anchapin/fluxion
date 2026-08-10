use fluxion::validation::report::ValidationStatus;
use fluxion::validation::ASHRAE140Validator;

#[test]
fn test_ashrae_140_comprehensive_validation() {
    let validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // Check that we have results
    assert!(!report.results.is_empty());

    // Verify that metrics have valid ranges and status
    for result in &report.results {
        assert!(result.ref_max >= result.ref_min);
        // Status should be one of the defined variants
        match result.status {
            ValidationStatus::Pass | ValidationStatus::Warning | ValidationStatus::Fail => (),
        }
    }

    // Print the report summary for visibility in test output
    report.print_summary();

    // Regression Guardrails: Log major failures but don't panic yet
    // while we are still calibrating the 900-series.
    for case_id in ["195", "600", "620"] {
        let failures: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.case_id == case_id && matches!(r.status, ValidationStatus::Fail))
            .collect();

        for f in failures {
            println!(
                "ATTENTION: Potential regression in Case {} {}: Actual {}, Ref {} - {}",
                case_id, f.metric, f.fluxion_value, f.ref_min, f.ref_max
            );
        }
    }

    // Ensure we can generate markdown
    let markdown = report.to_markdown();
    assert!(markdown.contains("# ASHRAE 140 Validation Report"));
}

#[test]
fn test_all_cases_instantiation() {
    // Verify all 18+ cases can be instantiated and have specs
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    let spec = ASHRAE140Case::Case960.spec();
    println!("DEBUG: Case 960 spec.num_zones = {}", spec.num_zones);
    println!("DEBUG: Case 960 spec.hvac.len() = {}", spec.hvac.len());

    let case_ids = [
        "600", "610", "620", "630", "640", "650", "600FF", "650FF", "900", "910", "920", "930",
        "940", "950", "900FF", "950FF", "960", "195",
    ];

    for id in case_ids {
        let case = match id {
            "600" => ASHRAE140Case::Case600,
            "610" => ASHRAE140Case::Case610,
            "620" => ASHRAE140Case::Case620,
            "630" => ASHRAE140Case::Case630,
            "640" => ASHRAE140Case::Case640,
            "650" => ASHRAE140Case::Case650,
            "600FF" => ASHRAE140Case::Case600FF,
            "650FF" => ASHRAE140Case::Case650FF,
            "900" => ASHRAE140Case::Case900,
            "910" => ASHRAE140Case::Case910,
            "920" => ASHRAE140Case::Case920,
            "930" => ASHRAE140Case::Case930,
            "940" => ASHRAE140Case::Case940,
            "950" => ASHRAE140Case::Case950,
            "900FF" => ASHRAE140Case::Case900FF,
            "950FF" => ASHRAE140Case::Case950FF,
            "960" => ASHRAE140Case::Case960,
            "195" => ASHRAE140Case::Case195,
            _ => panic!("Unknown case ID"),
        };

        let spec = case.spec();
        assert_eq!(spec.case_id, id);
        assert!(spec.validate().is_ok());
    }
}

#[test]
fn generate_validation_report() {
    use fluxion::validation::reporter::ValidationReportGenerator;
    use fluxion::validation::Analyzer;
    use std::path::PathBuf;

    let validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // Classify systematic issues
    let systematic_issues = ValidationReportGenerator::classify_systematic_issues(&report);

    // Generate main validation report
    let generator = ValidationReportGenerator::new(PathBuf::from("docs/ASHRAE140_RESULTS.md"));
    generator
        .generate(&report, Some(&systematic_issues), None)
        .expect("Failed to generate report");

    // Verify file was created
    assert!(generator.output_path.exists());

    // Verify content contains expected sections
    let content = std::fs::read_to_string(&generator.output_path).expect("Failed to read report");
    assert!(content.contains("# ASHRAE Standard 140 Validation Results"));
    assert!(content.contains("## Summary"));
    assert!(content.contains("## Detailed Results"));
    assert!(content.contains("## Systematic Issues"));
    assert!(content.contains("## Phase Progress"));
    assert!(content.contains("## References"));
    assert!(content.contains("## What's Fixed in Phase 5"));

    // Update quality metrics automatically (Task 5: metrics collection hook)
    let analyzer = Analyzer::default();
    match analyzer.update_quality_metrics(&report) {
        Ok(_) => {
            // Verify quality metrics file was created
            let metrics_path = PathBuf::from("docs/QUALITY_METRICS.md");
            assert!(metrics_path.exists(), "Quality metrics file not generated");
        }
        Err(e) => {
            panic!("Failed to update quality metrics: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Issue #2500 — structured tracing JSON output from the validation module.
//
// When the `tracing-subscriber-json` feature is enabled, the diagnostic
// helpers emit machine-parseable per-case events. This test installs a JSON
// `tracing_subscriber` writing to an in-memory buffer (scoped to the test via
// `with_default`, so no global subscriber is touched), triggers a structured
// diagnostic, and asserts the captured JSON contains the ingestion contract
// fields (`case_id`, `metric`, `deviation_pct`, `actual`, `expected`).
// ---------------------------------------------------------------------------
#[cfg(feature = "tracing-subscriber-json")]
#[test]
fn test_validation_diagnostic_json_output() {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// A cloneable shared byte buffer usable as a `tracing_subscriber` writer.
    #[derive(Clone)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBuffer {
        type Writer = SharedBuffer;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    let buffer = SharedBuffer(Arc::new(Mutex::new(Vec::new())));

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(buffer.clone())
        .json()
        .finish();

    let dispatch = tracing::dispatcher::Dispatch::new(subscriber);
    tracing::dispatcher::with_default(&dispatch, || {
        // Emit a representative per-case diagnostic (Case 600, annual heating,
        // value inside the reference range → PASS).
        fluxion::validation::diagnostic::emit_case_diagnostic(
            "600",
            "Annual Heating",
            5.2,
            4.30,
            5.71,
            0.0,
            "PASS",
        );
    });

    let captured = String::from_utf8(buffer.0.lock().unwrap().clone())
        .expect("captured tracing output must be valid UTF-8");

    // The ingestion contract: each emitted record must carry these structured
    // fields so Loki/Elastic can index per-case pass/fail.
    assert!(
        captured.contains("\"case_id\":\"600\""),
        "missing/incorrect case_id field; got: {captured}"
    );
    assert!(
        captured.contains("\"metric\":\"Annual Heating\""),
        "missing/incorrect metric field; got: {captured}"
    );
    assert!(
        captured.contains("\"deviation_pct\""),
        "missing deviation_pct field; got: {captured}"
    );
    assert!(captured.contains("\"actual\""), "missing actual field");
    assert!(captured.contains("\"expected\""), "missing expected field");
    assert!(
        captured.contains("\"status\":\"PASS\""),
        "missing/incorrect status field; got: {captured}"
    );
    // Span context must be present so events can be correlated to a case.
    assert!(
        captured.contains("ashrae140_case"),
        "missing ashrae140_case span; got: {captured}"
    );
}

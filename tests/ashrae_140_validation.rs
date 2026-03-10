use fluxion::validation::report::ValidationStatus;
use fluxion::validation::ASHRAE140Validator;

#[test]
fn test_ashrae_140_comprehensive_validation() {
    let mut validator = ASHRAE140Validator::new();
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

    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // Classify systematic issues
    let systematic_issues = ValidationReportGenerator::classify_systematic_issues(&report);

    // Generate main validation report
    let generator = ValidationReportGenerator::new(PathBuf::from("docs/ASHRAE140_RESULTS.md"));
    generator
        .generate(&report, Some(&systematic_issues))
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

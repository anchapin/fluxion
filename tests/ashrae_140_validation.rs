use fluxion::validation::report::ValidationStatus;
use fluxion::validation::{ASHRAE140Case, ASHRAE140Validator};

#[test]
fn test_ashrae_140_comprehensive_validation() {
    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // Check that we have results
    assert!(!report.results.is_empty());

    // Check Case 600 specific results
    let case_600_results: Vec<_> = report
        .results
        .iter()
        .filter(|r| r.case_id == "600")
        .collect();
    assert!(!case_600_results.is_empty());

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

    // Ensure we can generate markdown
    let markdown = report.to_markdown();
    assert!(markdown.contains("# ASHRAE 140 Validation Report"));
}

#[test]
fn test_all_cases_instantiation() {
    // Verify all 18+ cases can be instantiated and have specs
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

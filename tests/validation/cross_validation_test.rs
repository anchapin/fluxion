// tests/validation/cross_validation_test.rs
use fluxion::validation::esp_r::comparison::ComparisonResult;
/// Cross-validation tests
///
/// Test suite for cross-validation functionality and report generation
use fluxion::validation::reports::cross_validation::{generate_markdown_report, generate_report};

#[test]
fn test_cross_validation_report_with_perfect_match() {
    let comparison_results = vec![
        ComparisonResult {
            zone_id: "zone_1".to_string(),
            temp_difference: 0.0,
            heating_difference: 0.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
        ComparisonResult {
            zone_id: "zone_2".to_string(),
            temp_difference: 0.0,
            heating_difference: 0.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
    ];

    let report = generate_report(comparison_results, 0.5);

    assert!(report.overall_pass);
    assert_eq!(report.summary_statistics.pass_rate, 1.0);
    assert_eq!(report.summary_statistics.mean_temp_difference, 0.0);
}

#[test]
fn test_cross_validation_report_with_tolerance_violations() {
    let comparison_results = vec![
        ComparisonResult {
            zone_id: "zone_1".to_string(),
            temp_difference: 0.2,
            heating_difference: 10.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
        ComparisonResult {
            zone_id: "zone_2".to_string(),
            temp_difference: 0.6,      // Exceeds 0.5 tolerance
            heating_difference: 150.0, // Exceeds tolerance
            temp_within_tolerance: false,
            heating_within_tolerance: false,
        },
    ];

    let report = generate_report(comparison_results, 0.5);

    assert!(!report.overall_pass);
    assert_eq!(report.summary_statistics.pass_rate, 0.5);
    assert!(report.summary_statistics.mean_temp_difference > 0.0);
}

#[test]
fn test_cross_validation_report_statistics() {
    let comparison_results = vec![
        ComparisonResult {
            zone_id: "zone_1".to_string(),
            temp_difference: 0.1,
            heating_difference: 5.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
        ComparisonResult {
            zone_id: "zone_2".to_string(),
            temp_difference: 0.3,
            heating_difference: 15.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
        ComparisonResult {
            zone_id: "zone_3".to_string(),
            temp_difference: 0.2,
            heating_difference: 10.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
    ];

    let report = generate_report(comparison_results, 0.5);

    // Mean should be average of [0.1, 0.3, 0.2] = 0.2
    assert!((report.summary_statistics.mean_temp_difference - 0.2).abs() < 0.001);

    // Max should be 0.3
    assert_eq!(report.summary_statistics.max_temp_difference, 0.3);

    // Should pass with 100% pass rate
    assert!(report.overall_pass);
    assert_eq!(report.summary_statistics.pass_rate, 1.0);
}

#[test]
fn test_markdown_report_structure() {
    let comparison_results = vec![
        ComparisonResult {
            zone_id: "living_room".to_string(),
            temp_difference: 0.15,
            heating_difference: 8.5,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
        ComparisonResult {
            zone_id: "bedroom".to_string(),
            temp_difference: 0.45,
            heating_difference: 45.0,
            temp_within_tolerance: true,
            heating_within_tolerance: true,
        },
    ];

    let report = generate_report(comparison_results, 0.5);
    let markdown = generate_markdown_report(&report);

    // Check markdown structure
    assert!(markdown.contains("# Cross-Validation Report"));
    assert!(markdown.contains("**Overall Status:**"));
    assert!(markdown.contains("**Summary Statistics:**"));
    assert!(markdown.contains("- Mean Temperature Difference:"));
    assert!(markdown.contains("- Max Temperature Difference:"));
    assert!(markdown.contains("- Pass Rate:"));
    assert!(markdown.contains("**Zone Results:**"));

    // Check table structure
    assert!(markdown.contains("| Zone | Temp Within Tolerance | Heating Within Tolerance | Temp Diff (°C) | Heating Diff (W) |"));
    assert!(markdown.contains("living_room"));
    assert!(markdown.contains("bedroom"));

    // Check pass indicators
    assert!(markdown.matches("✅").count() >= 4); // At least 4 checkmarks
}

#[test]
fn test_cross_validation_report_edge_cases() {
    // Empty results
    let empty_results = vec![];
    let empty_report = generate_report(empty_results, 0.5);

    assert!(!empty_report.overall_pass); // Should fail with no results
    assert!(empty_report.summary_statistics.pass_rate.is_nan());

    // Single zone failure
    let single_failure = vec![ComparisonResult {
        zone_id: "single".to_string(),
        temp_difference: 1.0, // Way outside tolerance
        heating_difference: 1000.0,
        temp_within_tolerance: false,
        heating_within_tolerance: false,
    }];

    let failure_report = generate_report(single_failure, 0.5);
    assert!(!failure_report.overall_pass);
    assert_eq!(failure_report.summary_statistics.pass_rate, 0.0);
}

#[test]
fn test_tolerance_boundary_conditions() {
    // Exactly at tolerance boundary
    let boundary_results = vec![
        ComparisonResult {
            zone_id: "boundary_pass".to_string(),
            temp_difference: 0.5, // Exactly at tolerance
            heating_difference: 50.0,
            temp_within_tolerance: true, // Should pass (<= tolerance)
            heating_within_tolerance: true,
        },
        ComparisonResult {
            zone_id: "boundary_fail".to_string(),
            temp_difference: 0.51, // Just over tolerance
            heating_difference: 50.0,
            temp_within_tolerance: false, // Should fail (> tolerance)
            heating_within_tolerance: true,
        },
    ];

    let report = generate_report(boundary_results, 0.5);

    // Should fail overall due to boundary_fail
    assert!(!report.overall_pass);
    assert_eq!(report.summary_statistics.pass_rate, 0.5);
}

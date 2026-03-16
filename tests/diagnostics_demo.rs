use fluxion::validation::{validate_case_with_diagnostics, ASHRAE140Case};

#[test]
fn demo_diagnostics_collection() {
    // Run Case 900 with diagnostics enabled
    let (report, diagnostics) = validate_case_with_diagnostics(ASHRAE140Case::Case900, true);

    // Ensure diagnostics were collected
    assert!(diagnostics.is_some(), "Diagnostics should be collected");
    let diag = diagnostics.unwrap();

    // Print summary to stderr (visible with --nocapture)
    diag.print_summary();

    // Export to CSV
    let output_dir = std::path::Path::new("tmp");
    std::fs::create_dir_all(output_dir).expect("Failed to create tmp directory");
    let csv_path = output_dir.join("case900_hourly.csv");
    diag.export_csv(&csv_path).expect("Failed to export CSV");
    eprintln!("Diagnostics CSV exported to {:?}", csv_path);

    // Basic sanity checks
    assert!(
        report.annual_heating_mwh > 0.0,
        "Heating energy should be positive"
    );
    assert!(
        report.annual_cooling_mwh >= 0.0,
        "Cooling energy should be non-negative"
    );
    assert!(
        report.peak_heating_kw > 0.0,
        "Peak heating should be positive"
    );
    assert!(
        report.peak_cooling_kw >= 0.0,
        "Peak cooling should be non-negative"
    );
}

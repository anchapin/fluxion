//! Integration test for ASHRAE 140 diagnostic features.
//!
//! This test verifies that the diagnostic output and debugging tools
//! work correctly as specified in Issue #282.

use fluxion::validation::{
    diagnostic::{
        ComparisonRow, DiagnosticCollector, DiagnosticConfig, DiagnosticReport, EnergyBreakdown,
        HourlyData, PeakTiming, TemperatureProfile,
    },
    ASHRAE140Case, ASHRAE140Validator,
};

#[test]
fn test_issue_282_hourly_output_logging() {
    // Verify hourly data can be collected and exported
    let config = DiagnosticConfig {
        enabled: true,
        output_hourly: true,
        hourly_output_path: None,
        output_energy_breakdown: false,
        output_peak_timing: false,
        output_temperature_profiles: false,
        output_comparison_table: false,
        verbose: false,
    };
    let mut collector = DiagnosticCollector::new(config);

    collector.start_case("600", 1);

    // Record 24 hours of data
    for hour in 0..24 {
        let mut data = HourlyData::new(hour, 1);
        data.outdoor_temp = 10.0 + (hour as f64);
        data.zone_temps[0] = 20.0 + (hour as f64) * 0.1;
        data.solar_gains[0] = if hour >= 6 && hour <= 18 { 500.0 } else { 0.0 };
        data.internal_loads[0] = 200.0;
        data.hvac_heating[0] = if hour < 12 { 1000.0 } else { 0.0 };
        collector.record_hour(data);
    }

    // Verify data was collected
    assert_eq!(collector.hourly_data.len(), 24);

    // Test CSV export
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_hourly.csv");
    let path_str = temp_path.to_string_lossy().into_owned();

    let result = collector.export_hourly_csv(&path_str);
    assert!(result.is_ok());

    // Verify file content
    let content = std::fs::read_to_string(&temp_path).unwrap();
    assert!(content.contains("Hour,Month,Day,HourOfDay"));
    assert!(content.contains("OutdoorTemp"));
    assert!(content.contains("ZoneTemp"));
    assert!(content.contains("SolarGain"));
    assert!(content.contains("HVACHeating"));
    assert!(content.contains("HVACCooling"));

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_issue_282_component_energy_breakdown() {
    // Verify energy breakdown provides component-level analysis
    let config = DiagnosticConfig::full();
    let mut collector = DiagnosticCollector::new(config);

    collector.start_case("600", 1);

    // Record some data
    for hour in 0..100 {
        let mut data = HourlyData::new(hour, 1);
        data.outdoor_temp = 10.0;
        data.zone_temps[0] = 20.0;
        data.solar_gains[0] = 100.0;
        data.internal_loads[0] = 200.0;
        data.hvac_heating[0] = 500.0;
        data.infiltration_loss[0] = 50.0;
        data.envelope_conduction[0] = 25.0;
        collector.record_hour(data);
    }

    collector.finalize_case(5.0, 3.0);

    // Verify energy breakdown was computed
    assert!(collector.energy_breakdowns.contains_key("600"));
    let breakdown = collector.energy_breakdowns.get("600").unwrap();

    // Verify all components have values
    assert!(breakdown.heating_mwh > 0.0);
    assert!(breakdown.cooling_mwh > 0.0);
    assert!(breakdown.solar_gains_mwh >= 0.0);
    assert!(breakdown.internal_gains_mwh >= 0.0);

    // Test formatted output
    let formatted = breakdown.to_formatted_string("600");
    assert!(formatted.contains("Case 600 Energy Breakdown"));
    assert!(formatted.contains("Envelope Conduction"));
    assert!(formatted.contains("Infiltration"));
    assert!(formatted.contains("Solar Gains"));
    assert!(formatted.contains("Internal Gains"));
}

#[test]
fn test_issue_282_peak_load_timing() {
    // Verify peak load timing is reported correctly
    let config = DiagnosticConfig::full();
    let mut collector = DiagnosticCollector::new(config);

    collector.start_case("600", 1);

    // Record data with known peaks
    for hour in 0..100 {
        let mut data = HourlyData::new(hour, 1);
        data.outdoor_temp = 10.0;
        data.zone_temps[0] = 20.0;
        data.solar_gains[0] = 100.0;
        data.internal_loads[0] = 200.0;
        // Set peak heating at hour 50
        data.hvac_heating[0] = if hour == 50 { 5500.0 } else { 500.0 };
        // Set peak cooling at hour 75
        data.hvac_cooling[0] = if hour == 75 { 3200.0 } else { 300.0 };
        collector.record_hour(data);
    }

    collector.finalize_case(5.0, 3.0);

    // Verify peak timing was computed
    assert!(collector.peak_timings.contains_key("600"));
    let timing = collector.peak_timings.get("600").unwrap();

    // Verify peaks were captured
    assert_eq!(timing.peak_heating_hour, 50);
    assert_eq!(timing.peak_heating_kw, 5.5); // 5500W = 5.5kW
    assert_eq!(timing.peak_cooling_hour, 75);
    assert_eq!(timing.peak_cooling_kw, 3.2); // 3200W = 3.2kW

    // Test formatted output
    let formatted = timing.to_formatted_string("600");
    assert!(formatted.contains("Case 600 Peak Load Timing"));
    assert!(formatted.contains("5.50")); // Peak heating kW
    assert!(formatted.contains("3.20")); // Peak cooling kW
}

#[test]
fn test_issue_282_temperature_profile_export() {
    // Verify temperature profile for free-floating cases
    let mut profile = TemperatureProfile::new("600FF");

    // Add temperature readings
    for temp in [15.0, 18.0, 20.0, 22.0, 25.0] {
        profile.update(temp);
    }
    profile.finalize();

    // Verify statistics
    assert_eq!(profile.case_id, "600FF");
    assert_eq!(profile.min_temp, 15.0);
    assert_eq!(profile.max_temp, 25.0);
    assert_eq!(profile.avg_temp, 20.0);
    assert_eq!(profile.swing, 10.0);

    // Test CSV export
    let csv = profile.to_csv();
    assert!(csv.contains("Hour,Zone_Temp,Outdoor_Temp"));
    assert!(csv.contains("15.00")); // Min temp
    assert!(csv.contains("25.00")); // Max temp
}

#[test]
fn test_issue_282_validation_comparison_table() {
    // Verify validation comparison table generation
    let config = DiagnosticConfig {
        enabled: true,
        output_hourly: false,
        hourly_output_path: None,
        output_energy_breakdown: false,
        output_peak_timing: false,
        output_temperature_profiles: false,
        output_comparison_table: true,
        verbose: false,
    };
    let mut collector = DiagnosticCollector::new(config);

    // Add comparison rows
    collector.add_comparison("600", "Heating", 5.0, 4.30, 5.71, "PASS", -0.1);
    collector.add_comparison("600", "Cooling", 7.0, 6.14, 8.45, "PASS", 5.0);
    collector.add_comparison("900", "Heating", 1.5, 1.17, 2.04, "PASS", 8.5);

    // Verify comparison rows were added
    assert_eq!(collector.comparison_rows.len(), 3);

    // Test table output
    collector.print_comparison_table();

    // Verify each row
    assert_eq!(collector.comparison_rows[0].case_id, "600");
    assert_eq!(collector.comparison_rows[0].metric, "Heating");
    assert_eq!(collector.comparison_rows[0].status, "PASS");
}

#[test]
fn test_issue_282_diagnostic_report_generation() {
    // Verify comprehensive diagnostic report generation
    let config = DiagnosticConfig::full();
    let output_energy_breakdown = config.output_energy_breakdown;
    let output_peak_timing = config.output_peak_timing;
    let output_temperature_profiles = config.output_temperature_profiles;
    let output_comparison_table = config.output_comparison_table;
    let mut report = DiagnosticReport::new(config);

    // Add energy breakdown
    let breakdown = EnergyBreakdown {
        envelope_conduction_mwh: 2.5,
        infiltration_mwh: 1.0,
        solar_gains_mwh: 5.0,
        internal_gains_mwh: 3.0,
        heating_mwh: 4.0,
        cooling_mwh: 6.0,
        net_balance_mwh: 5.0,
    };
    report.add_energy_breakdown("600", breakdown);

    // Add peak timing
    let timing = PeakTiming {
        peak_heating_kw: 5.5,
        peak_heating_hour: 123,
        peak_cooling_kw: 3.2,
        peak_cooling_hour: 4567,
    };
    report.add_peak_timing("600", timing);

    // Add temperature profile
    let mut profile = TemperatureProfile::new("600FF");
    profile.update(15.0);
    profile.update(25.0);
    profile.finalize();
    report.add_temperature_profile(profile);

    // Add comparison row
    let row = ComparisonRow::new("600", "Heating", 5.0, 4.30, 5.71);
    report.add_comparison_row(row);

    // Test Markdown generation
    let markdown = report.to_markdown();

    // Verify all sections are present
    assert!(markdown.contains("# ASHRAE 140 Diagnostic Report"));
    // Note: Sections only appear if data is present and config flag is set
    if output_energy_breakdown && !report.energy_breakdowns.is_empty() {
        assert!(markdown.contains("## Energy Breakdowns"));
    }
    if output_peak_timing && !report.peak_timings.is_empty() {
        assert!(markdown.contains("## Peak Load Timing"));
    }
    if output_temperature_profiles && !report.temperature_profiles.is_empty() {
        assert!(markdown.contains("## Temperature Profiles"));
    }
    if output_comparison_table && !report.comparison_rows.is_empty() {
        assert!(markdown.contains("## Validation Comparison Table"));
    }
}

#[test]
fn test_issue_282_environment_variable_support() {
    // Verify environment variable configuration
    // Test that config can be created from environment
    let config = DiagnosticConfig::from_env();

    // By default, diagnostics should be disabled unless env var is set
    let is_debug_set = std::env::var("ASHRAE_140_DEBUG").is_ok();

    assert_eq!(config.enabled, is_debug_set);

    // Set env var and test again
    std::env::set_var("ASHRAE_140_DEBUG", "1");
    let config_with_debug = DiagnosticConfig::from_env();
    assert!(config_with_debug.enabled);

    // Clean up
    std::env::remove_var("ASHRAE_140_DEBUG");
}

#[test]
fn test_issue_282_validator_integration() {
    // Verify validator integrates correctly with diagnostics
    let config = DiagnosticConfig {
        enabled: true,
        output_hourly: false,
        hourly_output_path: None,
        output_energy_breakdown: true,
        output_peak_timing: true,
        output_temperature_profiles: true,
        output_comparison_table: true,
        verbose: false,
    };

    let output_energy_breakdown = config.output_energy_breakdown;
    let output_peak_timing = config.output_peak_timing;

    let mut validator = ASHRAE140Validator::with_diagnostics(config);

    // Run validation with diagnostics
    let (report, diagnostic_report) = validator.validate_with_diagnostics();

    // Verify we got results
    assert!(!report.results.is_empty());

    // Verify diagnostic report has data
    assert!(!diagnostic_report.comparison_rows.is_empty());
    assert!(!diagnostic_report.energy_breakdowns.is_empty() || !output_energy_breakdown);
    assert!(!diagnostic_report.peak_timings.is_empty() || !output_peak_timing);
}

#[test]
fn test_issue_282_performance_impact() {
    // Verify diagnostic features don't impact performance when disabled
    let config_disabled = DiagnosticConfig::disabled();
    let mut collector_disabled = DiagnosticCollector::new(config_disabled);

    collector_disabled.start_case("600", 1);

    // Record data (should be no-op when disabled)
    for hour in 0..100 {
        let data = HourlyData::new(hour, 1);
        collector_disabled.record_hour(data);
    }

    // Verify no data was collected when disabled
    assert_eq!(collector_disabled.hourly_data.len(), 0);
    assert!(!collector_disabled.energy_breakdowns.contains_key("600"));
}

#[test]
fn test_issue_282_single_case_validation() {
    // Verify single case validation with diagnostics
    let mut validator = ASHRAE140Validator::new();

    let (report, collector) = validator.validate_single_case_with_diagnostics(ASHRAE140Case::Case600);

    // Verify we got validation results
    assert!(!report.results.is_empty());

    // Verify collector exists (may or may not have collected data depending on config)
    assert!(!collector.hourly_data.is_empty() || !collector.config.enabled);
}

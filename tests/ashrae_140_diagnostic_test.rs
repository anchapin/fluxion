//! Diagnostic output and debugging tools test for ASHRAE 140 validation.
//!
//! This test verifies that the diagnostic features work correctly as specified in Issue #282.

use fluxion::validation::{
    diagnostic::{
        ComparisonRow, DiagnosticCollector, DiagnosticConfig, DiagnosticReport, EnergyBreakdown,
        HourlyData, PeakTiming, TemperatureProfile,
    },
    ASHRAE140Case, ASHRAE140Validator,
};

#[test]
fn test_diagnostic_config_from_env() {
    // Test that config can be created from environment
    let config = DiagnosticConfig::from_env();
    // Should read from env vars, but default to disabled if not set
    assert!(!config.enabled || std::env::var("ASHRAE_140_DEBUG").is_ok());
}

#[test]
fn test_diagnostic_config_full() {
    let config = DiagnosticConfig::full();
    assert!(config.enabled);
    assert!(config.output_hourly);
    assert!(config.output_energy_breakdown);
    assert!(config.output_peak_timing);
    assert!(config.output_temperature_profiles);
    assert!(config.output_comparison_table);
    assert!(config.verbose);
}

#[test]
fn test_diagnostic_config_disabled() {
    let config = DiagnosticConfig::disabled();
    assert!(!config.enabled);
    assert!(!config.output_hourly);
    assert!(!config.output_energy_breakdown);
    assert!(!config.output_peak_timing);
    assert!(!config.output_temperature_profiles);
    assert!(!config.output_comparison_table);
    assert!(!config.verbose);
}

#[test]
fn test_hourly_data_creation() {
    let data = HourlyData::new(0, 2);
    assert_eq!(data.hour, 0);
    assert_eq!(data.month, 1);
    assert_eq!(data.day, 1);
    assert_eq!(data.hour_of_day, 0);
    assert_eq!(data.zone_temps.len(), 2);
    assert_eq!(data.mass_temps.len(), 2);
    assert_eq!(data.solar_gains.len(), 2);
    assert_eq!(data.hvac_heating.len(), 2);
    assert_eq!(data.hvac_cooling.len(), 2);
    assert_eq!(data.internal_loads.len(), 2);
}

#[test]
fn test_hourly_data_to_csv_row() {
    let mut data = HourlyData::new(0, 1);
    data.outdoor_temp = 10.5;
    data.zone_temps[0] = 20.1;
    data.solar_gains[0] = 100.0;
    data.hvac_heating[0] = 500.0;
    data.hvac_cooling[0] = 0.0;

    let csv = data.to_csv_row();
    assert!(csv.contains("0,1,1,0")); // Hour, Month, Day, HourOfDay
    assert!(csv.contains("10.50")); // Outdoor temp
    assert!(csv.contains("20.10")); // Zone temp
}

#[test]
fn test_energy_breakdown() {
    let breakdown = EnergyBreakdown {
        envelope_conduction_mwh: 2.5,
        infiltration_mwh: 1.0,
        solar_gains_mwh: 5.0,
        internal_gains_mwh: 3.0,
        heating_mwh: 4.0,
        cooling_mwh: 6.0,
        net_balance_mwh: 5.0,
    };

    // Test total calculations
    assert_eq!(breakdown.total_input_mwh(), 8.0); // solar + internal
    assert_eq!(breakdown.total_loss_mwh(), 3.5); // conduction + infiltration

    // Test formatted output
    let formatted = breakdown.to_formatted_string("600");
    assert!(formatted.contains("Case 600"));
    assert!(formatted.contains("2.50")); // Envelope conduction
    assert!(formatted.contains("1.00")); // Infiltration
    assert!(formatted.contains("5.00")); // Solar gains
    assert!(formatted.contains("3.00")); // Internal gains
    assert!(formatted.contains("4.00")); // Heating
    assert!(formatted.contains("6.00")); // Cooling
}

#[test]
fn test_peak_timing() {
    let timing = PeakTiming {
        peak_heating_kw: 5.5,
        peak_heating_hour: 123,
        peak_cooling_kw: 3.2,
        peak_cooling_hour: 4567,
    };

    // Test datetime conversion
    let heating_time = timing.peak_heating_time_str();
    let cooling_time = timing.peak_cooling_time_str();
    assert!(heating_time.contains("Month"));
    assert!(cooling_time.contains("Month"));

    // Test formatted output
    let formatted = timing.to_formatted_string("600");
    assert!(formatted.contains("Case 600"));
    assert!(formatted.contains("5.50")); // Peak heating
    assert!(formatted.contains("3.20")); // Peak cooling
                                         // The formatted output uses table format with | separators
    assert!(formatted.contains("| Heating |") || formatted.contains("|Heating|"));
    assert!(formatted.contains("| Cooling |") || formatted.contains("|Cooling|"));
}

#[test]
fn test_temperature_profile() {
    let mut profile = TemperatureProfile::new("600FF");
    profile.update(15.0);
    profile.update(20.0);
    profile.update(25.0);
    profile.finalize();

    assert_eq!(profile.case_id, "600FF");
    assert_eq!(profile.min_temp, 15.0);
    assert_eq!(profile.max_temp, 25.0);
    assert_eq!(profile.avg_temp, 20.0);
    assert_eq!(profile.swing, 10.0);
}

#[test]
fn test_comparison_row() {
    let row = ComparisonRow::new("600", "Heating", 5.0, 4.30, 5.71);

    assert_eq!(row.case_id, "600");
    assert_eq!(row.metric, "Heating");
    assert_eq!(row.fluxion_value, 5.0);
    assert_eq!(row.ref_min, 4.30);
    assert_eq!(row.ref_max, 5.71);
    // Within range, so should be PASS
    assert_eq!(row.status, "PASS");

    // Test markdown row output
    let md = row.to_markdown_row();
    assert!(md.contains("| 600 | Heating"));
    assert!(md.contains("5.00"));
}

#[test]
fn test_diagnostic_collector() {
    let config = DiagnosticConfig::full();
    let mut collector = DiagnosticCollector::new(config);

    // Start a case
    collector.start_case("600", 1);

    // Record some hourly data
    let mut data = HourlyData::new(0, 1);
    data.outdoor_temp = 10.0;
    data.zone_temps[0] = 20.0;
    data.solar_gains[0] = 100.0;
    data.internal_loads[0] = 200.0;
    data.hvac_heating[0] = 500.0;
    collector.record_hour(data.clone());

    let mut data2 = HourlyData::new(1, 1);
    data2.outdoor_temp = 11.0;
    data2.zone_temps[0] = 21.0;
    data2.solar_gains[0] = 150.0;
    data2.internal_loads[0] = 200.0;
    data2.hvac_cooling[0] = 300.0;
    collector.record_hour(data2);

    // Finalize the case
    collector.finalize_case(5.0, 3.0);

    // Verify data was collected
    assert_eq!(collector.current_case, "600");
    assert_eq!(collector.hourly_data.len(), 2);
    assert!(collector.energy_breakdowns.contains_key("600"));
    assert!(collector.peak_timings.contains_key("600"));

    // Verify energy breakdown
    let breakdown = collector.energy_breakdowns.get("600").unwrap();
    assert_eq!(breakdown.heating_mwh, 5.0);
    assert_eq!(breakdown.cooling_mwh, 3.0);

    // Verify peak timing
    let timing = collector.peak_timings.get("600").unwrap();
    assert_eq!(timing.peak_heating_kw, 0.5); // 500W = 0.5kW
    assert_eq!(timing.peak_heating_hour, 0);
}

#[test]
fn test_diagnostic_collector_disabled() {
    let config = DiagnosticConfig::disabled();
    let mut collector = DiagnosticCollector::new(config);

    collector.start_case("600", 1);

    let data = HourlyData::new(0, 1);
    collector.record_hour(data);
    collector.finalize_case(5.0, 3.0);

    // Should not collect any data when disabled
    assert!(collector.hourly_data.is_empty());
    assert!(!collector.energy_breakdowns.contains_key("600"));
}

#[test]
fn test_diagnostic_report() {
    let config = DiagnosticConfig::full();
    let mut report = DiagnosticReport::new(config);

    // Add energy breakdown
    let breakdown = EnergyBreakdown {
        heating_mwh: 5.0,
        cooling_mwh: 7.0,
        envelope_conduction_mwh: 2.0,
        infiltration_mwh: 1.0,
        solar_gains_mwh: 5.0,
        internal_gains_mwh: 3.0,
        net_balance_mwh: 5.0,
    };
    report.add_energy_breakdown("600", breakdown);

    // Add peak timing
    let timing = PeakTiming {
        peak_heating_kw: 5.5,
        peak_heating_hour: 500,
        peak_cooling_kw: 3.2,
        peak_cooling_hour: 4500,
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

    // Test markdown generation
    let markdown = report.to_markdown();
    assert!(markdown.contains("# ASHRAE 140 Diagnostic Report"));
    assert!(markdown.contains("## Energy Breakdowns"));
    assert!(markdown.contains("## Peak Load Timing"));
    assert!(markdown.contains("## Temperature Profiles"));
    assert!(markdown.contains("## Validation Comparison Table"));
}

#[test]
fn test_validator_with_diagnostics() {
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
    let mut validator = ASHRAE140Validator::with_diagnostics(config);

    // Run validation with diagnostics
    let (report, diagnostic_report) = validator.validate_with_diagnostics();

    // Check that we got results
    assert!(!report.results.is_empty());

    // Check diagnostic report
    assert!(!diagnostic_report.comparison_rows.is_empty());
}

#[test]
fn test_validator_single_case_with_diagnostics() {
    let mut validator = ASHRAE140Validator::new();

    // Test single case validation with diagnostics
    let (report, collector) =
        validator.validate_single_case_with_diagnostics(ASHRAE140Case::Case600);

    // Check that we got results
    assert!(!report.results.is_empty());

    // Check that collector was used
    assert!(!collector.hourly_data.is_empty() || !collector.config.enabled);
}

#[test]
fn test_export_hourly_csv() {
    let config = DiagnosticConfig::full();
    let mut collector = DiagnosticCollector::new(config);

    collector.start_case("600", 1);

    // Add some hourly data
    for i in 0..10 {
        let mut data = HourlyData::new(i, 1);
        data.outdoor_temp = 10.0 + (i as f64);
        data.zone_temps[0] = 20.0 + (i as f64);
        data.solar_gains[0] = 100.0 * (i as f64);
        data.internal_loads[0] = 200.0;
        collector.record_hour(data);
    }

    // Export to a temp file
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_hourly_export.csv");
    let path_str = temp_path.to_string_lossy().into_owned();

    let result = collector.export_hourly_csv(&path_str);
    assert!(result.is_ok());

    // Verify file exists and has content
    assert!(temp_path.exists());
    let content = std::fs::read_to_string(&temp_path).unwrap();
    assert!(content.contains("Hour,Month,Day,HourOfDay"));
    assert!(content.contains("OutdoorTemp,ZoneTemp"));

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_peak_timing_datetime_conversion() {
    // Test various hour conversions
    assert_eq!(PeakTiming::hour_to_datetime(0), "Jan 1 00:00");
    assert_eq!(PeakTiming::hour_to_datetime(12), "Jan 1 12:00");
    assert_eq!(PeakTiming::hour_to_datetime(24), "Jan 2 00:00");
    assert_eq!(PeakTiming::hour_to_datetime(744), "Feb 1 00:00"); // 31*24

    // Mid-year: July 1 (day 182) at noon
    let mid_year_hour = 181 * 24 + 12;
    let dt = PeakTiming::hour_to_datetime(mid_year_hour);
    assert!(dt.contains("Jul"));
    assert!(dt.contains("12:00"));

    // End of year: Dec 31 (day 365) at 23:00
    let end_of_year_hour = 364 * 24 + 23;
    let dt = PeakTiming::hour_to_datetime(end_of_year_hour);
    assert_eq!(dt, "Dec 31 23:00");
}

#[test]
fn test_diagnostic_report_print_summary() {
    let config = DiagnosticConfig {
        enabled: true,
        output_hourly: false,
        hourly_output_path: None,
        output_energy_breakdown: true,
        output_peak_timing: true,
        output_temperature_profiles: true,
        output_comparison_table: true,
        verbose: true,
    };
    let mut report = DiagnosticReport::new(config);

    // Add some data
    let breakdown = EnergyBreakdown {
        heating_mwh: 5.0,
        cooling_mwh: 7.0,
        ..Default::default()
    };
    report.add_energy_breakdown("600", breakdown);

    let timing = PeakTiming {
        peak_heating_kw: 5.5,
        peak_heating_hour: 500,
        peak_cooling_kw: 3.2,
        peak_cooling_hour: 4500,
    };
    report.add_peak_timing("600", timing);

    let mut profile = TemperatureProfile::new("600FF");
    profile.update(15.0);
    profile.update(25.0);
    profile.finalize();
    report.add_temperature_profile(profile);

    // Print summary (should not panic)
    report.print_summary();
}

#[test]
fn test_comparison_row_fail_case() {
    // Test a case that should fail
    let row = ComparisonRow::new("600", "Heating", 10.0, 4.30, 5.71);
    assert_eq!(row.status, "FAIL");
    // Deviation is (10.0 - 5.005) / 5.005 * 100 ≈ 99.8%
    assert!(row.deviation_percent > 90.0); // Should be >90% outside range
}

#[test]
fn test_energy_breakdown_net_balance() {
    let mut breakdown = EnergyBreakdown::default();
    breakdown.solar_gains_mwh = 5.0;
    breakdown.internal_gains_mwh = 3.0;
    breakdown.heating_mwh = 4.0;
    breakdown.cooling_mwh = 6.0;

    // Net balance = gains - heating + cooling (cooling is negative energy)
    // = 5 + 3 - 4 + 6 = 10.0
    breakdown.net_balance_mwh = breakdown.solar_gains_mwh + breakdown.internal_gains_mwh
        - breakdown.heating_mwh
        + breakdown.cooling_mwh;

    assert_eq!(breakdown.net_balance_mwh, 10.0);
}

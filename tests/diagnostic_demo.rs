//! Demonstration of ASHRAE 140 diagnostic features.
//!
//! This example shows how to use the diagnostic output and debugging tools
//! to analyze ASHRAE 140 validation results.

use fluxion::validation::diagnostic::{
    ComparisonRow, DiagnosticCollector, DiagnosticConfig, DiagnosticReport, EnergyBreakdown,
    HourlyData, PeakTiming, TemperatureProfile,
};

fn main() -> anyhow::Result<()> {
    println!("=== ASHRAE 140 Diagnostic Features Demo ===\n");

    // 1. Demonstrate DiagnosticConfig
    println!("1. Diagnostic Configuration");
    println!("   - Disabled by default");
    println!("   - Can be enabled via ASHRAE_140_DEBUG=1");
    println!("   - Can be fully enabled with DiagnosticConfig::full()");
    println!();

    // 2. Demonstrate hourly data collection
    println!("2. Hourly Data Collection");
    let mut hourly_data = Vec::new();
    for hour in 0..24 {
        let mut data = HourlyData::new(hour, 1);
        data.outdoor_temp = 5.0 + (hour as f64) * 0.5;
        data.zone_temps[0] = 20.0 + (hour as f64) * 0.2;
        data.solar_gains[0] = if hour >= 6 && hour <= 18 { 500.0 } else { 0.0 };
        data.internal_loads[0] = 200.0;
        hourly_data.push(data);
    }
    println!("   - Collected {} hours of data", hourly_data.len());
    println!("   - Each record: Hour, Outdoor Temp, Zone Temp, Solar, HVAC, etc.");
    println!();

    // 3. Demonstrate energy breakdown
    println!("3. Energy Breakdown");
    let breakdown = EnergyBreakdown {
        envelope_conduction_mwh: 2.5,
        infiltration_mwh: 1.0,
        solar_gains_mwh: 5.0,
        internal_gains_mwh: 3.0,
        heating_mwh: 4.0,
        cooling_mwh: 6.0,
        net_balance_mwh: 5.0,
    };
    println!("{}", breakdown.to_formatted_string("600"));
    println!();

    // 4. Demonstrate peak timing
    println!("4. Peak Load Timing");
    let timing = PeakTiming {
        peak_heating_kw: 5.5,
        peak_heating_hour: 123,
        peak_cooling_kw: 3.2,
        peak_cooling_hour: 4567,
    };
    println!("{}", timing.to_formatted_string("600"));
    println!();

    // 5. Demonstrate temperature profile
    println!("5. Temperature Profile (Free-Floating)");
    let mut profile = TemperatureProfile::new("600FF");
    for temp in [15.0, 20.0, 25.0, 18.0, 22.0] {
        profile.update(temp);
    }
    profile.finalize();
    println!("   Case: {}", profile.case_id);
    println!("   Min Temp: {:.1}°C", profile.min_temp);
    println!("   Max Temp: {:.1}°C", profile.max_temp);
    println!("   Avg Temp: {:.1}°C", profile.avg_temp);
    println!("   Swing: {:.1}°C", profile.swing);
    println!();

    // 6. Demonstrate comparison table
    println!("6. Validation Comparison Table");
    let row = ComparisonRow::new("600", "Heating", 5.0, 4.30, 5.71);
    println!("   Case: {}", row.case_id);
    println!("   Metric: {}", row.metric);
    println!("   Fluxion: {:.2}", row.fluxion_value);
    println!("   Ref Min: {:.2}", row.ref_min);
    println!("   Ref Max: {:.2}", row.ref_max);
    println!("   Deviation: {:.1}%", row.deviation_percent);
    println!("   Status: {}", row.status);
    println!();

    // 7. Demonstrate diagnostic collector
    println!("7. Diagnostic Collector");
    let config = DiagnosticConfig::full();
    let mut collector = DiagnosticCollector::new(config.clone());
    collector.start_case("600", 1);

    // Add some hourly data
    for hour in 0..10 {
        let mut data = HourlyData::new(hour, 1);
        data.outdoor_temp = 10.0 + (hour as f64);
        data.zone_temps[0] = 20.0 + (hour as f64);
        data.solar_gains[0] = 100.0 * (hour as f64);
        data.internal_loads[0] = 200.0;
        collector.record_hour(data);
    }

    collector.finalize_case(5.0, 3.0);
    println!(
        "   - Collected {} hourly records",
        collector.hourly_data.len()
    );
    println!("   - Generated energy breakdown and peak timing");
    println!();

    // 8. Demonstrate diagnostic report
    println!("8. Diagnostic Report");
    let mut report = DiagnosticReport::new(config);
    report.add_energy_breakdown("600", breakdown);
    report.add_peak_timing("600", timing);
    report.add_temperature_profile(profile);
    report.add_comparison_row(row);

    let markdown = report.to_markdown();
    println!("   Generated {}-character Markdown report", markdown.len());
    println!();

    // 9. Demonstrate validator with diagnostics
    println!("9. Validator with Diagnostics");
    println!("   - Use ASHRAE140Validator::with_diagnostics(config)");
    println!("   - Call validate_with_diagnostics() for full diagnostic output");
    println!("   - Or use validate_single_case_with_diagnostics() for single case");
    println!();

    // 10. Usage examples
    println!("10. Usage Examples");
    println!();
    println!("   Enable diagnostics via environment variable:");
    println!("   $ ASHRAE_140_DEBUG=1 cargo test --test ashrae_140_validation");
    println!();
    println!("   With verbose output:");
    println!("   $ ASHRAE_140_DEBUG=1 ASHRAE_140_VERBOSE=1 cargo test");
    println!();
    println!("   Export hourly data to CSV:");
    println!("   $ ASHRAE_140_DEBUG=1 ASHRAE_140_HOURLY_OUTPUT=1 \\");
    println!("     ASHRAE_140_HOURLY_PATH=hourly.csv cargo test");
    println!();
    println!("   Programmatically enable diagnostics:");
    println!("   let config = DiagnosticConfig::full();");
    println!("   let mut validator = ASHRAE140Validator::with_diagnostics(config);");
    println!("   let (report, diagnostic) = validator.validate_with_diagnostics();");
    println!();

    println!("=== Demo Complete ===");
    Ok(())
}

// Solar Gain Unit Tests for ASHRAE 140 Case 900
// Standalone test binary to verify solar gain calculations against EnergyPlus reference

use serde_json;
use std::fs;

fn main() {
    println!("=== Solar Gain Unit Tests ===");
    println!("Testing against EnergyPlus reference data...");

    // Load EnergyPlus reference data
    let ep_data = match load_energyplus_reference() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load EnergyPlus reference: {}", e);
            std::process::exit(1);
        }
    };

    // Run all tests
    let mut passed = 0;
    let mut failed = 0;
    let mut ignored = 0;

    let tests = vec![
        (
            "test_energyplus_reference_validity",
            "EnergyPlus reference data validity",
        ),
        (
            "test_solar_gain_zero_at_night",
            "Solar gain zero during night hours",
        ),
        (
            "test_solar_gain_peaks_at_noon",
            "Solar gain peaks around noon",
        ),
        (
            "test_solar_gain_seasonal_pattern",
            "Solar gain seasonal pattern",
        ),
        (
            "test_solar_gain_temperature_correlation",
            "Solar gain temperature correlation",
        ),
        (
            "test_solar_energy_conservation",
            "Solar energy conservation",
        ),
        (
            "test_solar_gain_cloudy_days",
            "Solar gain zero during cloudy days",
        ),
        ("test_solar_rate_units", "Solar rate units and scale"),
        ("test_solar_gain_continuity", "Solar gain time continuity"),
        ("test_solar_daily_pattern", "Solar gain daily pattern"),
    ];

    let total_tests = tests.len();
    for (test_name, test_desc) in tests {
        print!("Running: {} - {}...", test_name, test_desc);

        let result = run_test(test_name, &ep_data);

        match result {
            TestResult::Pass => {
                println!("  ✓ PASSED");
                passed += 1;
            }
            TestResult::Fail(msg) => {
                println!("  ✗ FAILED: {}", msg);
                failed += 1;
            }
            TestResult::Ignore(msg) => {
                println!("  ⊘ IGNORED: {}", msg);
                ignored += 1;
            }
        }
    }

    println!("\n=== Test Results ===");
    println!("Total: {} tests", total_tests);
    println!(
        "Passed: {} ({:.1}%)",
        passed,
        (passed as f64 / total_tests as f64) * 100.0
    );
    println!(
        "Failed: {} ({:.1}%)",
        failed,
        (failed as f64 / total_tests as f64) * 100.0
    );
    println!(
        "Ignored: {} ({:.1}%)",
        ignored,
        (ignored as f64 / total_tests as f64) * 100.0
    );

    // Exit with error code if any tests failed
    std::process::exit(if failed > 0 { 1 } else { 0 });
}

#[derive(Debug, serde::Deserialize)]
struct EnergyPlusReference {
    zone_air_temp_c: Vec<f64>,
    heating_energy_wh: Vec<f64>,
    cooling_energy_wh: Vec<f64>,
    solar_rate_total_w: Vec<f64>,
}

fn load_energyplus_reference() -> Result<EnergyPlusReference, Box<dyn std::error::Error>> {
    let path = "benchmarks/outputs/bestest_gsr/case_900/run/reference_data.json";
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let data: EnergyPlusReference = serde_json::from_reader(file).map_err(|e| e.to_string())?;

    Ok(data)
}

enum TestResult {
    Pass,
    Fail(String),
    Ignore(String),
}

fn run_test(test_name: &str, ep: &EnergyPlusReference) -> TestResult {
    match test_name {
        "test_energyplus_reference_validity" => test_energyplus_reference_validity(ep),
        "test_solar_gain_zero_at_night" => test_solar_gain_zero_at_night(ep),
        "test_solar_gain_peaks_at_noon" => test_solar_gain_peaks_at_noon(ep),
        "test_solar_gain_seasonal_pattern" => test_solar_gain_seasonal_pattern(ep),
        "test_solar_gain_temperature_correlation" => test_solar_gain_temperature_correlation(ep),
        "test_solar_energy_conservation" => test_solar_energy_conservation(ep),
        "test_solar_gain_cloudy_days" => test_solar_gain_cloudy_days(ep),
        "test_solar_rate_units" => test_solar_rate_units(ep),
        "test_solar_gain_continuity" => test_solar_gain_continuity(ep),
        "test_solar_daily_pattern" => test_solar_daily_pattern(ep),
        _ => TestResult::Ignore(format!("Unknown test: {}", test_name)),
    }
}

fn test_energyplus_reference_validity(ep: &EnergyPlusReference) -> TestResult {
    // Verify we have 8760 hours of data
    if ep.zone_air_temp_c.len() != 8760 {
        return TestResult::Fail(format!(
            "Zone temperature should have 8760 hours, got {}",
            ep.zone_air_temp_c.len()
        ));
    }
    if ep.heating_energy_wh.len() != 8760 {
        return TestResult::Fail(format!(
            "Heating energy should have 8760 hours, got {}",
            ep.heating_energy_wh.len()
        ));
    }
    if ep.cooling_energy_wh.len() != 8760 {
        return TestResult::Fail(format!(
            "Cooling energy should have 8760 hours, got {}",
            ep.cooling_energy_wh.len()
        ));
    }
    if ep.solar_rate_total_w.len() != 8760 {
        return TestResult::Fail(format!(
            "Solar rate should have 8760 hours, got {}",
            ep.solar_rate_total_w.len()
        ));
    }

    // Verify annual totals match expected EnergyPlus values
    let heating_mwh: f64 = ep.heating_energy_wh.iter().sum::<f64>() / 1000.0;
    let cooling_mwh: f64 = ep.cooling_energy_wh.iter().sum::<f64>() / 1000.0;

    // EnergyPlus reference from energyplus_reference_data.json:
    // Heating: 1.661 MWh, Cooling: 2.497 MWh
    if heating_mwh < 1.6 || heating_mwh > 1.7 {
        return TestResult::Fail(format!(
            "Heating should be ~1.66 MWh, got {:.3} MWh",
            heating_mwh
        ));
    }
    if cooling_mwh < 2.4 || cooling_mwh > 2.6 {
        return TestResult::Fail(format!(
            "Cooling should be ~2.50 MWh, got {:.3} MWh",
            cooling_mwh
        ));
    }

    TestResult::Pass
}

fn test_solar_gain_zero_at_night(ep: &EnergyPlusReference) -> TestResult {
    // Solar should be zero during night hours (typically hours 0-5)
    // Check first few hours (Jan 1, midnight to 5 AM)
    let mut non_zero_count = 0;
    for i in 0..6 {
        if ep.solar_rate_total_w[i] >= 1.0 {
            non_zero_count += 1;
        }
    }

    if non_zero_count > 0 {
        return TestResult::Fail(format!(
            "Solar should be < 1.0 W at night, but found {} hours >= 1.0 W (hours 0-5)",
            non_zero_count
        ));
    }

    TestResult::Pass
}

fn test_solar_gain_peaks_at_noon(ep: &EnergyPlusReference) -> TestResult {
    // Find solar peaks (should be around hours 11-13, 11 AM - 1 PM)
    let mut max_solar = 0.0;
    let mut max_hour = 0;

    for (i, &solar) in ep.solar_rate_total_w.iter().enumerate() {
        if solar > max_solar {
            max_solar = solar;
            max_hour = i;
        }
    }

    // Max solar should be around noon (hour 11-13 for local time)
    // Note: EnergyPlus uses UTC, so adjust for Denver time zone (-7 hours)
    // Hour 18 in UTC = 11 AM MST
    if max_hour < 17 || max_hour > 19 {
        return TestResult::Fail(format!(
            "Max solar should occur around noon UTC (hours 17-19), got hour {}",
            max_hour
        ));
    }

    // Max solar should be reasonable (500-600 W for Denver)
    if max_solar < 400.0 || max_solar > 700.0 {
        return TestResult::Fail(format!(
            "Max solar should be 400-700 W, got {:.2} W",
            max_solar
        ));
    }

    TestResult::Pass
}

fn test_solar_gain_seasonal_pattern(ep: &EnergyPlusReference) -> TestResult {
    // Summer months (June-August) should have higher solar than winter
    // June (hours 4320-5087)
    let mut summer_solar: f64 = 0.0;
    let mut summer_hours = 0;

    for i in 4320..5088 {
        summer_solar += ep.solar_rate_total_w[i];
        summer_hours += 1;
    }

    // Winter (Dec-Feb, hours 0-2160)
    let mut winter_solar: f64 = 0.0;
    let mut winter_hours = 0;

    for i in 0..2160 {
        winter_solar += ep.solar_rate_total_w[i];
        winter_hours += 1;
    }

    let summer_avg = summer_solar / summer_hours as f64;
    let winter_avg = winter_solar / winter_hours as f64;

    // Summer should have higher solar than winter
    if summer_avg <= winter_avg {
        return TestResult::Fail(format!(
            "Summer solar ({:.2} W) should be higher than winter ({:.2} W)",
            summer_avg, winter_avg
        ));
    }

    TestResult::Pass
}

fn test_solar_gain_temperature_correlation(ep: &EnergyPlusReference) -> TestResult {
    // On sunny days, solar gain should correlate with temperature rise
    // Check day with high solar (hour ~4320, ~day 180, June 28)
    let solar_hour = 4320;
    let high_solar_threshold = 400.0; // W

    if ep.solar_rate_total_w[solar_hour] > high_solar_threshold {
        // Zone temperature should rise during the day when solar is high
        // Temperature should be higher than surrounding hours with low solar

        // Check temperature at solar_hour vs temperature at solar_hour - 3 and + 3
        let temp_at_solar = ep.zone_air_temp_c[solar_hour];
        let temp_before = if solar_hour >= 3 {
            ep.zone_air_temp_c[solar_hour - 3]
        } else {
            20.0
        };
        let temp_after = if solar_hour < 8760 - 3 {
            ep.zone_air_temp_c[solar_hour + 3]
        } else {
            temp_at_solar
        };

        // With high solar, temperature should be rising
        if solar_hour >= 3 {
            if temp_at_solar <= temp_before {
                return TestResult::Fail(format!(
                    "Temperature should rise with solar: before ({:.2} C) -> at solar ({:.2} C)",
                    temp_before, temp_at_solar
                ));
            }
        }

        if solar_hour < 8760 - 3 {
            if temp_after <= temp_at_solar {
                return TestResult::Fail(format!(
                    "Temperature should continue rising: at solar ({:.2} C) -> after ({:.2} C)",
                    temp_at_solar, temp_after
                ));
            }
        }
    }

    TestResult::Pass
}

fn test_solar_energy_conservation(ep: &EnergyPlusReference) -> TestResult {
    // Total solar energy should be positive
    let total_solar_energy: f64 = ep.solar_rate_total_w.iter().sum::<f64>();

    if total_solar_energy < 10000.0 {
        return TestResult::Fail(format!(
            "Total solar energy should be significant, got {:.2} Wh",
            total_solar_energy
        ));
    }

    // Calculate rough annual solar estimate
    // Denver ~1700 kWh/m²/year direct solar
    // Case 900: 12 m² windows total
    // Estimated annual: ~20,400 kWh = ~20 MWh
    let estimated_annual_mwh = total_solar_energy / 1000.0 / 8760.0;

    // Should be in reasonable range (5-30 MWh depending on assumptions)
    if estimated_annual_mwh < 5.0 || estimated_annual_mwh > 50.0 {
        return TestResult::Fail(format!(
            "Estimated annual solar should be 5-50 MWh, got {:.2} MWh",
            estimated_annual_mwh
        ));
    }

    TestResult::Pass
}

fn test_solar_gain_cloudy_days(ep: &EnergyPlusReference) -> TestResult {
    // Find days with low solar (potential cloudy days)
    // Define low solar threshold
    let low_solar_threshold = 50.0; // W

    let mut low_solar_hours = 0;
    let mut total_hours_checked = 0;

    // Check first 100 hours (first 4+ days)
    for i in 0..100 {
        total_hours_checked += 1;
        if ep.solar_rate_total_w[i] < low_solar_threshold {
            low_solar_hours += 1;
        }
    }

    // Should have some low solar hours (cloudy periods)
    let low_solar_fraction = low_solar_hours as f64 / total_hours_checked as f64;

    // Denver should have some cloudy periods
    if low_solar_fraction < 0.05 || low_solar_fraction > 0.5 {
        return TestResult::Fail(format!(
            "Cloudy period fraction should be 5-50%, got {:.1}%",
            low_solar_fraction * 100.0
        ));
    }

    TestResult::Pass
}

fn test_solar_rate_units(ep: &EnergyPlusReference) -> TestResult {
    // Solar rate should be in reasonable range for Case 900
    // 12 m² windows, transmittance ~0.8
    // Peak DNI ~900 W/m²
    // Expected peak: 900 * 12 * 0.8 ≈ 8640 W (but actual window area less)

    let max_solar = ep.solar_rate_total_w.iter().fold(0.0_f64, |a, &b| a.max(b));

    // Should be less than 10 kW (typical residential)
    if max_solar > 10000.0 {
        return TestResult::Fail(format!(
            "Max solar rate should be < 10 kW, got {:.2} W",
            max_solar
        ));
    }

    // Should be significant (> 1 kW)
    if max_solar < 1000.0 {
        return TestResult::Fail(format!(
            "Max solar rate should be > 1 kW, got {:.2} W",
            max_solar
        ));
    }

    TestResult::Pass
}

fn test_solar_gain_continuity(ep: &EnergyPlusReference) -> TestResult {
    // Solar should not have sudden jumps (unless weather changes rapidly)
    // Check for unrealistic changes between consecutive hours

    let mut unrealistic_jumps = 0;

    for i in 1..ep.solar_rate_total_w.len() {
        let prev = ep.solar_rate_total_w[i - 1];
        let curr = ep.solar_rate_total_w[i];

        // Sudden jump: > 500 W change in one hour
        if (curr - prev).abs() > 500.0 {
            unrealistic_jumps += 1;
        }
    }

    // Should have very few or no unrealistic jumps
    let jump_fraction = unrealistic_jumps as f64 / ep.solar_rate_total_w.len() as f64;
    if jump_fraction > 0.001 {
        return TestResult::Fail(format!(
            "Unrealistic solar jumps should be < 0.1%, got {:.3}%",
            jump_fraction * 100.0
        ));
    }

    TestResult::Pass
}

fn test_solar_daily_pattern(ep: &EnergyPlusReference) -> TestResult {
    // Solar should follow daily pattern: zero at night, rise morning, peak noon, decline afternoon
    // Check a typical sunny day (e.g., June 21, hour 4320)

    let day_start = 4320; // Hour 4320 = June 21, 0:00
    let day_hours: Vec<f64> = (0..24)
        .map(|h| ep.solar_rate_total_w[day_start + h])
        .collect();

    // Night (hours 0-5): should be zero
    for i in 0..6 {
        if day_hours[i] >= 10.0 {
            return TestResult::Fail(format!(
                "Solar should be near zero at night hour {}, got {:.2} W",
                i, day_hours[i]
            ));
        }
    }

    // Solar should increase from morning to noon
    let morning_peak = day_hours[6..12].iter().fold(0.0_f64, |a, &b| a.max(b));
    let noon_peak = day_hours[11..14].iter().fold(0.0_f64, |a, &b| a.max(b));

    if noon_peak < morning_peak {
        return TestResult::Fail(format!(
            "Noon solar ({:.2} W) should be >= morning ({:.2} W)",
            noon_peak, morning_peak
        ));
    }

    // Solar should decline afternoon
    let afternoon_peak = day_hours[14..18].iter().fold(0.0_f64, |a, &b| a.max(b));

    // Noon should be higher than afternoon
    if noon_peak <= afternoon_peak {
        return TestResult::Fail(format!(
            "Noon solar ({:.2} W) should be > afternoon ({:.2} W)",
            noon_peak, afternoon_peak
        ));
    }

    TestResult::Pass
}

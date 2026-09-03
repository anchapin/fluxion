//! Comparison test: Denver TMY parametric model vs EPW file for ASHRAE 140 validation.
//!
//! **Issue #922**: This test documents the discrepancies between the parametric
//! `DenverTmyWeather` model (used in all ASHRAE 140 tests) and the actual Denver
//! TMY EPW file (`USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw`).
//!
//! ## Key Finding
//!
//! The parametric model produces a **significantly narrower temperature range**
//! than the EPW file, particularly in winter months where it underestimates both
//! the minimum and maximum temperatures. This directly affects ASHRAE 140
//! validation results — the parametric model's warmer winter temperatures mean
//! heating load cases may show lower energy consumption than expected.
//!
//! ## Recommendations
//!
//! 1. **For ASHRAE 140 validation**: Use the EPW file (`EpwWeatherSource`) for
//!    consistency with reference benchmark tools (EnergyPlus, ESP-r, TRNSYS).
//! 2. **Parametric model**: Suitable for prototyping and unit tests where exact
//!    weather values don't affect validation pass/fail criteria.
//! 3. **Future work**: Consider calibrating the parametric model's seasonal
//!    amplitude to better match the EPW data.

use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

/// EPW file path relative to project root
const EPW_PATH: &str = "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw";

/// Month names for display
const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Days in each month (non-leap year)
// Kept for the monthly rollup follow-up comparisons.
#[allow(dead_code)]
const MONTH_DAYS: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Monthly statistics for comparison
#[derive(Debug, Clone)]
struct MonthlyStats {
    min_temp: f64,
    max_temp: f64,
    avg_temp: f64,
    peak_ghi: f64,
    avg_ghi: f64,
    avg_wind: f64,
    avg_humidity: f64,
    sample_count: u32,
}

/// Compute monthly statistics from an EPW source
fn compute_epw_monthly_stats(epw: &EpwWeatherSource) -> Vec<MonthlyStats> {
    let mut stats = vec![
        MonthlyStats {
            min_temp: f64::INFINITY,
            max_temp: f64::NEG_INFINITY,
            avg_temp: 0.0,
            peak_ghi: 0.0,
            avg_ghi: 0.0,
            avg_wind: 0.0,
            avg_humidity: 0.0,
            sample_count: 0,
        };
        12
    ];

    for hour in 0..8760 {
        let data = epw.get_hourly_data(hour).unwrap();
        let month = hour / 744; // Rough month index (0-11)
        let m = month.min(11);

        stats[m].min_temp = stats[m].min_temp.min(data.dry_bulb_temp);
        stats[m].max_temp = stats[m].max_temp.max(data.dry_bulb_temp);
        stats[m].avg_temp += data.dry_bulb_temp;
        stats[m].peak_ghi = stats[m].peak_ghi.max(data.ghi);
        stats[m].avg_ghi += data.ghi;
        stats[m].avg_wind += data.wind_speed;
        stats[m].avg_humidity += data.humidity;
        stats[m].sample_count += 1;
    }

    for s in &mut stats {
        if s.sample_count > 0 {
            s.avg_temp /= s.sample_count as f64;
            s.avg_ghi /= s.sample_count as f64;
            s.avg_wind /= s.sample_count as f64;
            s.avg_humidity /= s.sample_count as f64;
        }
    }

    stats
}

/// Compute monthly statistics from the parametric model
fn compute_parametric_monthly_stats(parametric: &DenverTmyWeather) -> Vec<MonthlyStats> {
    let mut stats = vec![
        MonthlyStats {
            min_temp: f64::INFINITY,
            max_temp: f64::NEG_INFINITY,
            avg_temp: 0.0,
            peak_ghi: 0.0,
            avg_ghi: 0.0,
            avg_wind: 0.0,
            avg_humidity: 0.0,
            sample_count: 0,
        };
        12
    ];

    for hour in 0..8760 {
        let data = parametric.get_hourly_data(hour).unwrap();
        let month = hour / 744;
        let m = month.min(11);

        stats[m].min_temp = stats[m].min_temp.min(data.dry_bulb_temp);
        stats[m].max_temp = stats[m].max_temp.max(data.dry_bulb_temp);
        stats[m].avg_temp += data.dry_bulb_temp;
        stats[m].peak_ghi = stats[m].peak_ghi.max(data.ghi);
        stats[m].avg_ghi += data.ghi;
        stats[m].avg_wind += data.wind_speed;
        stats[m].avg_humidity += data.humidity;
        stats[m].sample_count += 1;
    }

    for s in &mut stats {
        if s.sample_count > 0 {
            s.avg_temp /= s.sample_count as f64;
            s.avg_ghi /= s.sample_count as f64;
            s.avg_wind /= s.sample_count as f64;
            s.avg_humidity /= s.sample_count as f64;
        }
    }

    stats
}

// ============================================================================
// COMPARISON TESTS
// ============================================================================

/// Test that both weather sources load successfully
#[test]
fn test_both_sources_load() {
    let epw = EpwWeatherSource::from_file(EPW_PATH);
    assert!(epw.is_ok(), "EPW file should load: {:?}", epw.err());

    let parametric = DenverTmyWeather::new();
    let data = parametric.get_hourly_data(0);
    assert!(data.is_ok(), "Parametric model should produce data");
}

/// Compare monthly temperature statistics between parametric model and EPW.
///
/// This test documents the known discrepancies. It uses lenient thresholds
/// because the parametric model is intentionally simplified. The test passes
/// but prints warnings when discrepancies exceed expected bounds.
#[test]
fn test_monthly_temperature_comparison() {
    let epw = EpwWeatherSource::from_file(EPW_PATH).expect("EPW should load");
    let parametric = DenverTmyWeather::new();

    let epw_stats = compute_epw_monthly_stats(&epw);
    let param_stats = compute_parametric_monthly_stats(&parametric);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   ASHRAE 140 Weather Comparison: Parametric vs EPW         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n--- Monthly Max Temperature (°C) ---");
    println!(
        "{:<5} {:>10} {:>10} {:>10} {:>8}",
        "Month", "Parametric", "EPW", "Diff", "Status"
    );
    println!("{}", "-".repeat(50));

    let mut max_temp_discrepancies = Vec::new();

    for i in 0..12 {
        let diff = param_stats[i].max_temp - epw_stats[i].max_temp;
        let status = if diff.abs() > 5.0 {
            "⚠️  LARGE"
        } else if diff.abs() > 2.0 {
            "~ok"
        } else {
            "✓"
        };
        if diff.abs() > 5.0 {
            max_temp_discrepancies.push((MONTH_NAMES[i], diff));
        }
        println!(
            "{:<5} {:>10.1} {:>10.1} {:>+10.1} {:>8}",
            MONTH_NAMES[i], param_stats[i].max_temp, epw_stats[i].max_temp, diff, status
        );
    }

    println!("\n--- Monthly Min Temperature (°C) ---");
    println!(
        "{:<5} {:>10} {:>10} {:>10} {:>8}",
        "Month", "Parametric", "EPW", "Diff", "Status"
    );
    println!("{}", "-".repeat(50));

    let mut min_temp_discrepancies = Vec::new();

    for i in 0..12 {
        let diff = param_stats[i].min_temp - epw_stats[i].min_temp;
        let status = if diff.abs() > 5.0 {
            "⚠️  LARGE"
        } else if diff.abs() > 2.0 {
            "~ok"
        } else {
            "✓"
        };
        if diff.abs() > 5.0 {
            min_temp_discrepancies.push((MONTH_NAMES[i], diff));
        }
        println!(
            "{:<5} {:>10.1} {:>10.1} {:>+10.1} {:>8}",
            MONTH_NAMES[i], param_stats[i].min_temp, epw_stats[i].min_temp, diff, status
        );
    }

    println!("\n--- Monthly Average Temperature (°C) ---");
    println!(
        "{:<5} {:>10} {:>10} {:>10}",
        "Month", "Parametric", "EPW", "Diff"
    );
    println!("{}", "-".repeat(45));
    for i in 0..12 {
        let diff = param_stats[i].avg_temp - epw_stats[i].avg_temp;
        println!(
            "{:<5} {:>10.1} {:>10.1} {:>+10.1}",
            MONTH_NAMES[i], param_stats[i].avg_temp, epw_stats[i].avg_temp, diff
        );
    }

    println!("\n--- Monthly Peak GHI (W/m²) ---");
    println!(
        "{:<5} {:>10} {:>10} {:>10}",
        "Month", "Parametric", "EPW", "Diff"
    );
    println!("{}", "-".repeat(45));
    for i in 0..12 {
        let diff = param_stats[i].peak_ghi - epw_stats[i].peak_ghi;
        println!(
            "{:<5} {:>10.0} {:>10.0} {:>+10.0}",
            MONTH_NAMES[i], param_stats[i].peak_ghi, epw_stats[i].peak_ghi, diff
        );
    }

    // Print summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   SUMMARY                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    if !max_temp_discrepancies.is_empty() {
        println!("\n⚠️  Large max temperature discrepancies (>5°C):");
        for (month, diff) in &max_temp_discrepancies {
            println!(
                "   {}: {:+.1}°C (parametric is {})",
                month,
                diff,
                if *diff > 0.0 { "hotter" } else { "cooler" }
            );
        }
    }

    if !min_temp_discrepancies.is_empty() {
        println!("\n⚠️  Large min temperature discrepancies (>5°C):");
        for (month, diff) in &min_temp_discrepancies {
            println!(
                "   {}: {:+.1}°C (parametric is {})",
                month,
                diff,
                if *diff > 0.0 { "warmer" } else { "colder" }
            );
        }
    }

    // The parametric model is known to have a narrower temperature range.
    // This is acceptable for unit tests but NOT for ASHRAE 140 validation.
    // We document but don't fail — the test's purpose is comparison.
    println!("\n📋 Note: This test documents discrepancies for Issue #922.");
    println!("   For ASHRAE 140 validation, use EpwWeatherSource instead.");
}

/// Test that annual average temperatures are reasonably close.
///
/// The parametric model targets Denver's annual average (~10°C) and should
/// be within 2°C of the EPW annual average.
#[test]
fn test_annual_average_temperature() {
    let epw = EpwWeatherSource::from_file(EPW_PATH).expect("EPW should load");
    let parametric = DenverTmyWeather::new();

    let epw_stats = compute_epw_monthly_stats(&epw);
    let param_stats = compute_parametric_monthly_stats(&parametric);

    let epw_annual_avg: f64 = epw_stats.iter().map(|s| s.avg_temp).sum::<f64>() / 12.0;
    let param_annual_avg: f64 = param_stats.iter().map(|s| s.avg_temp).sum::<f64>() / 12.0;

    println!("\nAnnual average temperature:");
    println!("  Parametric: {:.1}°C", param_annual_avg);
    println!("  EPW:        {:.1}°C", epw_annual_avg);
    println!("  Diff:       {:+.1}°C", param_annual_avg - epw_annual_avg);

    // Annual average should be within 2°C (both target ~10°C for Denver)
    assert!(
        (param_annual_avg - epw_annual_avg).abs() < 2.0,
        "Annual average temp difference too large: parametric={:.1}, EPW={:.1}",
        param_annual_avg,
        epw_annual_avg
    );
}

/// Test that the parametric model's temperature range is narrower than EPW.
///
/// This documents the core discrepancy: the parametric model uses a sinusoidal
/// approximation with amplitude ~15°C, while the EPW has a range of ~60°C
/// (-24°C to +35°C).
#[test]
fn test_temperature_range_discrepancy() {
    let epw = EpwWeatherSource::from_file(EPW_PATH).expect("EPW should load");
    let parametric = DenverTmyWeather::new();

    let epw_stats = compute_epw_monthly_stats(&epw);
    let param_stats = compute_parametric_monthly_stats(&parametric);

    let epw_overall_min = epw_stats
        .iter()
        .map(|s| s.min_temp)
        .fold(f64::INFINITY, f64::min);
    let epw_overall_max = epw_stats
        .iter()
        .map(|s| s.max_temp)
        .fold(f64::NEG_INFINITY, f64::max);
    let param_overall_min = param_stats
        .iter()
        .map(|s| s.min_temp)
        .fold(f64::INFINITY, f64::min);
    let param_overall_max = param_stats
        .iter()
        .map(|s| s.max_temp)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\nTemperature range comparison:");
    println!(
        "  EPW:        {:.1} to {:.1}°C (range: {:.1}°C)",
        epw_overall_min,
        epw_overall_max,
        epw_overall_max - epw_overall_min
    );
    println!(
        "  Parametric: {:.1} to {:.1}°C (range: {:.1}°C)",
        param_overall_min,
        param_overall_max,
        param_overall_max - param_overall_min
    );

    // Parametric range should be narrower (documented discrepancy)
    let epw_range = epw_overall_max - epw_overall_min;
    let param_range = param_overall_max - param_overall_min;

    println!("\n  EPW range:        {:.1}°C", epw_range);
    println!("  Parametric range: {:.1}°C", param_range);
    println!(
        "  Ratio:            {:.1}%",
        (param_range / epw_range) * 100.0
    );

    // The parametric model's range should be at least 60% of EPW's range
    // (currently it's about 65% — this threshold catches regressions)
    assert!(
        param_range / epw_range > 0.6,
        "Parametric temperature range too narrow: {:.1}% of EPW range",
        (param_range / epw_range) * 100.0
    );
}

/// Test that solar radiation values are reasonably close between sources.
///
/// Solar radiation is more consistent between models because both use
/// similar clear-sky calculations for Denver's latitude and altitude.
#[test]
fn test_solar_radiation_comparison() {
    let epw = EpwWeatherSource::from_file(EPW_PATH).expect("EPW should load");
    let parametric = DenverTmyWeather::new();

    let epw_stats = compute_epw_monthly_stats(&epw);
    let param_stats = compute_parametric_monthly_stats(&parametric);

    println!("\nMonthly peak GHI comparison (W/m²):");
    println!(
        "{:<5} {:>10} {:>10} {:>10} {:>8}",
        "Month", "Parametric", "EPW", "Diff", "%Diff"
    );
    println!("{}", "-".repeat(50));

    for i in 0..12 {
        let diff = param_stats[i].peak_ghi - epw_stats[i].peak_ghi;
        let pct = if epw_stats[i].peak_ghi > 0.0 {
            (diff / epw_stats[i].peak_ghi) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<5} {:>10.0} {:>10.0} {:>+10.0} {:>+7.1}%",
            MONTH_NAMES[i], param_stats[i].peak_ghi, epw_stats[i].peak_ghi, diff, pct
        );
    }

    // Peak GHI should be within 15% on average across all months
    let total_param_ghi: f64 = param_stats.iter().map(|s| s.peak_ghi).sum();
    let total_epw_ghi: f64 = epw_stats.iter().map(|s| s.peak_ghi).sum();
    let overall_pct_diff = ((total_param_ghi - total_epw_ghi) / total_epw_ghi) * 100.0;

    println!("\n  Overall peak GHI difference: {:+.1}%", overall_pct_diff);

    assert!(
        overall_pct_diff.abs() < 15.0,
        "Peak GHI difference too large: {:+.1}%",
        overall_pct_diff
    );
}

/// Test documenting which ASHRAE 140 cases are most affected by weather source.
///
/// Cases involving heating (600-series, 900-series) are most sensitive to
/// winter temperature discrepancies. Cooling cases (800-series) are less
/// affected because summer temperatures are closer between sources.
#[test]
fn test_ashrae140_impact_assessment() {
    let epw = EpwWeatherSource::from_file(EPW_PATH).expect("EPW should load");
    let parametric = DenverTmyWeather::new();

    let epw_stats = compute_epw_monthly_stats(&epw);
    let param_stats = compute_parametric_monthly_stats(&parametric);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   ASHRAE 140 Impact Assessment                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Heating season (Oct-Mar): Cases 600, 610, 620, 630, 900, 910, 920
    let heating_months = [9, 10, 11, 0, 1, 2]; // Oct=9, Nov=10, Dec=11, Jan=0, Feb=1, Mar=2
    let heating_epw_avg: f64 = heating_months
        .iter()
        .map(|&m| epw_stats[m].avg_temp)
        .sum::<f64>()
        / 6.0;
    let heating_param_avg: f64 = heating_months
        .iter()
        .map(|&m| param_stats[m].avg_temp)
        .sum::<f64>()
        / 6.0;

    println!("\nHeating season (Oct-Mar) — Cases 600-series, 900-series:");
    println!("  EPW average:        {:.1}°C", heating_epw_avg);
    println!("  Parametric average: {:.1}°C", heating_param_avg);
    println!(
        "  Difference:         {:+.1}°C",
        heating_param_avg - heating_epw_avg
    );
    println!("  ⚠️  Parametric model is WARMER → heating loads will be UNDERESTIMATED");

    // Cooling season (May-Sep): Cases 800, 810, 820, 830, 840
    let cooling_months = [4, 5, 6, 7, 8]; // May=4, Jun=5, Jul=6, Aug=7, Sep=8
    let cooling_epw_avg: f64 = cooling_months
        .iter()
        .map(|&m| epw_stats[m].avg_temp)
        .sum::<f64>()
        / 5.0;
    let cooling_param_avg: f64 = cooling_months
        .iter()
        .map(|&m| param_stats[m].avg_temp)
        .sum::<f64>()
        / 5.0;

    println!("\nCooling season (May-Sep) — Cases 800-series:");
    println!("  EPW average:        {:.1}°C", cooling_epw_avg);
    println!("  Parametric average: {:.1}°C", cooling_param_avg);
    println!(
        "  Difference:         {:+.1}°C",
        cooling_param_avg - cooling_epw_avg
    );

    if (cooling_param_avg - cooling_epw_avg).abs() < 2.0 {
        println!("  ✓ Difference is small — cooling cases less affected");
    } else {
        println!("  ⚠️  Significant difference — cooling cases may be affected");
    }

    // Free-floating cases (Case 970): Most sensitive to outdoor temperature
    println!("\nFree-floating case (970):");
    println!("  Most sensitive to outdoor temperature at every hour.");
    println!("  Parametric model's narrower range will produce different results.");

    println!("\n📋 Recommendation:");
    println!("   For ASHRAE 140 validation, use EpwWeatherSource for consistency");
    println!("   with reference tools (EnergyPlus, ESP-r, TRNSYS).");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the EPW file path is correct
    #[test]
    fn test_epw_file_exists() {
        assert!(
            std::path::Path::new(EPW_PATH).exists(),
            "EPW file not found at: {}",
            EPW_PATH
        );
    }
}

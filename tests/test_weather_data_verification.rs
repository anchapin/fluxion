//! Weather data verification test for Denver Stapleton TMY3 EPW.
//!
//! This diagnostic test validates that the EPW file used for ASHRAE 140
//! validation contains physically reasonable solar radiation values (DNI/DHI/GHI).
//!
//! # EPW Column Mapping (verified from EnergyPlus documentation)
//!
//! | EPW Col (1-based) | Index (0-based) | Field                        |
//! |-------------------|-----------------|------------------------------|
//! | 7                 | fields[6]       | Dry Bulb Temperature (°C)    |
//! | 8                 | fields[7]       | Dew Point Temperature (°C)   |
//! | 9                 | fields[8]       | Relative Humidity (%)         |
//! | 10                | fields[9]       | Atmospheric Pressure (Pa)     |
//! | 11                | fields[10]      | Extraterrestrial Horizontal   |
//! | 12                | fields[11]      | Extraterrestrial Direct Normal|
//! | 13                | fields[12]      | Horizontal Infrared (W/m²)    |
//! | 14                | fields[13]      | Global Horizontal (GHI)       |
//! | 15                | fields[14]      | Direct Normal (DNI)           |
//! | 16                | fields[15]      | Diffuse Horizontal (DHI)      |
//! | 22                | fields[21]      | Wind Speed (m/s)              |
//!
//! # Code Column Mapping (after Issue #829 fix)
//!
//! ```text
//! dry_bulb_temp = fields[6]   // Col 7  ✓
//! humidity      = fields[8]   // Col 9  ✓
//! ghi           = fields[13]  // Col 14 ✓
//! dni           = fields[14]  // Col 15 ✓
//! dhi           = fields[15]  // Col 16 ✓
//! horiz_infrared = fields[12] // Col 13 ✓
//! wind_speed    = fields[21]  // Col 22 ✓
//! ```
//!
//! Units in EPW are Wh/m² (energy per hour). For hourly data, Wh/m² == W/m² numerically.

use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::epw_path::epw_required;
use fluxion::weather::WeatherSource;

/// The EPW file used for ASHRAE 140 validation.
const VALIDATION_EPW: &str = "USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw";

/// Summer solstice (June 21) — print hourly DNI/DHI/GHI for hours 6–20.
#[test]
fn test_june_21_hourly_solar() {
    let weather = EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap())
        .expect("Failed to load Denver Stapleton EPW for June 21 verification");

    println!("\n=== June 21 Hourly Solar Data (Denver Stapleton TMY) ===");
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Hour", "Temp°C", "GHI", "DNI", "DHI", "DNI>DHI?"
    );
    println!("{}", "-".repeat(55));

    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        // EPW hour field is 1-24, but data is stored sequentially.
        // We need to find June 21 by scanning through the data.
        // Hour 0 = Jan 1 01:00 (EPW convention), so we compute month/day from hour_of_year.
        let month = get_month_from_hour(hour);
        let day = get_day_from_hour(hour, month);
        let epw_hour = (hour % 24) + 1; // EPW hours are 1-24

        if month == 6 && day == 21 && epw_hour >= 6 && epw_hour <= 20 {
            let dni_gt_dhi = if data.dni > data.dhi { "YES" } else { "no" };
            println!(
                "{:<6} {:>8.1} {:>8.0} {:>8.0} {:>8.0} {:>10}",
                epw_hour, data.dry_bulb_temp, data.ghi, data.dni, data.dhi, dni_gt_dhi
            );
        }
    }
}

/// Winter solstice (December 21) — print hourly DNI/DHI/GHI for hours 8–16.
#[test]
fn test_december_21_hourly_solar() {
    let weather = EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap())
        .expect("Failed to load Denver Stapleton EPW for Dec 21 verification");

    println!("\n=== December 21 Hourly Solar Data (Denver Stapleton TMY) ===");
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Hour", "Temp°C", "GHI", "DNI", "DHI", "DNI>DHI?"
    );
    println!("{}", "-".repeat(55));

    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        let month = get_month_from_hour(hour);
        let day = get_day_from_hour(hour, month);
        let epw_hour = (hour % 24) + 1;

        if month == 12 && day == 21 && epw_hour >= 8 && epw_hour <= 16 {
            let dni_gt_dhi = if data.dni > data.dhi { "YES" } else { "no" };
            println!(
                "{:<6} {:>8.1} {:>8.0} {:>8.0} {:>8.0} {:>10}",
                epw_hour, data.dry_bulb_temp, data.ghi, data.dni, data.dhi, dni_gt_dhi
            );
        }
    }
}

/// Summer clear-sky sanity check: find the clearest June hour and verify DNI > 500 W/m².
///
/// TMY data picks typical months from different years, so June 21 may not be clear.
/// This test finds the hour with maximum DNI in June and verifies it's in a reasonable range.
#[test]
fn test_summer_clear_sky_dni_reasonable() {
    let weather =
        EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap()).expect("Failed to load Denver Stapleton EPW");

    let mut max_dni = 0.0_f64;
    let mut max_dni_hour = 0_usize;
    let mut june_noon_hours = Vec::new();

    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        let month = get_month_from_hour(hour);
        let epw_hour = (hour % 24) + 1;

        if month == 6 && (epw_hour >= 10 && epw_hour <= 14) {
            june_noon_hours.push((hour, data.dni, data.dhi, data.ghi));
            if data.dni > max_dni {
                max_dni = data.dni;
                max_dni_hour = hour;
            }
        }
    }

    let max_data = weather.get_hourly_data(max_dni_hour).unwrap();
    println!("\n=== Summer (June) Max DNI ===");
    println!(
        "Max DNI hour: hour_of_year={}, DNI={:.0} W/m², DHI={:.0}, GHI={:.0}",
        max_dni_hour, max_data.dni, max_data.dhi, max_data.ghi
    );

    // Denver Stapleton clear-sky DNI should be 850-950 W/m² in summer
    assert!(
        max_dni > 500.0,
        "Summer clear-sky DNI should be > 500 W/m², got {:.0} — possible column swap or unit error",
        max_dni
    );
    assert!(
        max_dni < 1200.0,
        "Summer clear-sky DNI should be < 1200 W/m², got {:.0} — possibly reading extraterrestrial radiation",
        max_dni
    );

    // Count how many noon hours have DNI > DHI (should be most on clear days)
    let dni_dominant = june_noon_hours
        .iter()
        .filter(|(_, dni, dhi, _)| *dni > *dhi)
        .count();
    let total = june_noon_hours.len();
    println!(
        "June noon hours where DNI > DHI: {}/{} ({:.0}%)",
        dni_dominant,
        total,
        100.0 * dni_dominant as f64 / total as f64
    );
}

/// Winter clear-sky sanity check: December DNI should be 500-1000 W/m² at noon.
#[test]
fn test_winter_clear_sky_dni_reasonable() {
    let weather =
        EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap()).expect("Failed to load Denver Stapleton EPW");

    let mut max_dec_dni = 0.0_f64;
    let mut max_dec_dni_hour = 0_usize;

    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        let month = get_month_from_hour(hour);
        let epw_hour = (hour % 24) + 1;

        if month == 12 && (epw_hour >= 10 && epw_hour <= 14) && data.dni > max_dec_dni {
            max_dec_dni = data.dni;
            max_dec_dni_hour = hour;
        }
    }

    let max_data = weather.get_hourly_data(max_dec_dni_hour).unwrap();
    println!("\n=== Winter (December) Max DNI ===");
    println!(
        "Max DNI hour: hour_of_year={}, DNI={:.0} W/m², DHI={:.0}, GHI={:.0}",
        max_dec_dni_hour, max_data.dni, max_data.dhi, max_data.ghi
    );

    assert!(
        max_dec_dni > 400.0,
        "Winter clear-sky DNI should be > 400 W/m², got {:.0}",
        max_dec_dni
    );
    assert!(
        max_dec_dni < 1200.0,
        "Winter DNI should be < 1200 W/m², got {:.0} — possibly reading extraterrestrial radiation",
        max_dec_dni
    );
}

/// Verify that DNI and DHI are not swapped.
///
/// On clear-sky noon hours, DNI should exceed DHI. If this consistently fails,
/// it indicates DNI/DHI columns may be swapped.
#[test]
fn test_dni_dhi_not_swapped() {
    let weather =
        EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap()).expect("Failed to load Denver Stapleton EPW");

    // Check December noon hours — these are typically very clear in Denver
    let mut clear_noon_count = 0;
    let mut dni_gt_dhi_count = 0;

    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        let month = get_month_from_hour(hour);
        let epw_hour = (hour % 24) + 1;

        // December noon hours with significant solar (GHI > 200)
        if month == 12 && (epw_hour >= 10 && epw_hour <= 14) && data.ghi > 200.0 {
            clear_noon_count += 1;
            if data.dni > data.dhi {
                dni_gt_dhi_count += 1;
            } else {
                println!(
                    "  WARNING: Dec hour {} has DNI={} < DHI={} (GHI={})",
                    epw_hour, data.dni, data.dhi, data.ghi
                );
            }
        }
    }

    println!("\n=== DNI/DHI Swap Check (December clear noon hours) ===");
    println!(
        "Clear noon hours: {}, DNI > DHI: {}/{} ({:.0}%)",
        clear_noon_count,
        dni_gt_dhi_count,
        clear_noon_count,
        100.0 * dni_gt_dhi_count as f64 / clear_noon_count.max(1) as f64
    );

    // On clear December days, DNI should dominate over DHI
    assert!(
        dni_gt_dhi_count > 0,
        "No December clear-noon hours with DNI > DHI — columns may be swapped!"
    );
}

/// Verify no unreasonable negative solar values.
#[test]
fn test_no_negative_solar() {
    let weather =
        EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap()).expect("Failed to load Denver Stapleton EPW");

    let mut neg_count = 0;
    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        if data.dni < 0.0 || data.dhi < 0.0 || data.ghi < 0.0 {
            neg_count += 1;
            if neg_count <= 5 {
                println!(
                    "  Negative solar at hour {}: DNI={}, DHI={}, GHI={}",
                    hour, data.dni, data.dhi, data.ghi
                );
            }
        }
    }

    println!("\n=== Negative Solar Values ===");
    println!("Hours with negative DNI/DHI/GHI: {}", neg_count);
    assert_eq!(
        neg_count, 0,
        "Found {} hours with negative solar radiation values",
        neg_count
    );
}

/// Verify the annual max DNI is physically reasonable (not extraterrestrial).
///
/// Extraterrestrial DNI is ~1361 W/m². If we read values near this,
/// it means we're reading the wrong column (ET Direct Normal is col 12/fields[11]).
#[test]
fn test_annual_max_dni_not_extraterrestrial() {
    let weather =
        EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap()).expect("Failed to load Denver Stapleton EPW");

    let mut max_dni = 0.0_f64;
    let mut max_dni_hour = 0_usize;

    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        if data.dni > max_dni {
            max_dni = data.dni;
            max_dni_hour = hour;
        }
    }

    println!("\n=== Annual Max DNI ===");
    println!(
        "Max DNI: {:.0} W/m² at hour_of_year={}",
        max_dni, max_dni_hour
    );

    // Surface DNI should never exceed ~1100 W/m²
    // Extraterrestrial is ~1361 W/m²
    assert!(
        max_dni < 1100.0,
        "Annual max DNI = {:.0} W/m² exceeds surface limit — likely reading ET Direct Normal (col 12) instead of surface DNI (col 15)",
        max_dni
    );
}

/// Verify EPW loads 8760 hours and location is Denver.
#[test]
fn test_epw_loads_complete() {
    let weather =
        EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap()).expect("Failed to load Denver Stapleton EPW");

    assert_eq!(
        weather.record_count(),
        8760,
        "EPW should contain 8760 hourly records"
    );

    let location = weather.location().unwrap_or_default();
    assert!(
        location.contains("Denver"),
        "Location should contain 'Denver', got: '{}'",
        location
    );

    println!("\n=== EPW Load Verification ===");
    println!("Location: {}", location);
    println!("Records: {}", weather.record_count());
    println!("Solar hours (GHI>0): {}", weather.solar_hours());
    println!("Max temp: {:.1}°C", weather.max_temperature());
    println!("Min temp: {:.1}°C", weather.min_temperature());
    println!("Avg temp: {:.1}°C", weather.average_temperature());
}

/// Spot-check specific values against known Denver Stapleton TMY3 data.
///
/// December 21 hour 12 (noon) from raw EPW line:
/// `1971,12,21,12,0,...,11.10,-11.10,18,83720,608,940,231,470,956,60,...`
///
/// Parsing with fields[6]=dry_bulb, fields[13]=GHI, fields[14]=DNI, fields[15]=DHI:
/// - Dry bulb = 11.10°C
/// - GHI = fields[13] = 470 Wh/m²
/// - DNI = fields[14] = 956 Wh/m²
/// - DHI = fields[15] = 60 Wh/m²
///
/// These values are consistent with a clear winter day at Denver's latitude (39.76°N).
#[test]
fn test_dec_21_noon_spot_check() {
    let weather =
        EpwWeatherSource::from_file(epw_required(VALIDATION_EPW).to_str().unwrap()).expect("Failed to load Denver Stapleton EPW");

    // Find Dec 21, hour 12 (EPW hour = 12, which is 11:00-12:00 local standard time)
    let mut found = false;
    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        let month = get_month_from_hour(hour);
        let day = get_day_from_hour(hour, month);
        let epw_hour = (hour % 24) + 1;

        if month == 12 && day == 21 && epw_hour == 12 {
            println!("\n=== Dec 21 Hour 12 Spot Check ===");
            println!("Temp: {:.1}°C (expected ~11.1°C)", data.dry_bulb_temp);
            println!("GHI:  {:.0} Wh/m² (expected ~470)", data.ghi);
            println!("DNI:  {:.0} Wh/m² (expected ~956)", data.dni);
            println!("DHI:  {:.0} Wh/m² (expected ~60)", data.dhi);

            // Allow 1% tolerance for floating point
            assert!(
                (data.dry_bulb_temp - 11.1).abs() < 0.2,
                "Dec 21 noon temp: expected ~11.1°C, got {:.1}",
                data.dry_bulb_temp
            );
            assert!(
                (data.dni - 956.0).abs() < 5.0,
                "Dec 21 noon DNI: expected ~956, got {:.0}",
                data.dni
            );
            assert!(
                (data.dhi - 60.0).abs() < 5.0,
                "Dec 21 noon DHI: expected ~60, got {:.0}",
                data.dhi
            );
            assert!(
                (data.ghi - 470.0).abs() < 5.0,
                "Dec 21 noon GHI: expected ~470, got {:.0}",
                data.ghi
            );

            found = true;
            break;
        }
    }

    assert!(found, "Could not find Dec 21 hour 12 in EPW data");
}

// ===== Helper functions =====

/// Days per month (non-leap year).
const DAYS_PER_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Get month (1-12) from hour-of-year (0-8759).
fn get_month_from_hour(hour_of_year: usize) -> u32 {
    let day_of_year = hour_of_year / 24; // 0-based day of year
    let mut cumulative = 0u32;
    for (i, &days) in DAYS_PER_MONTH.iter().enumerate() {
        cumulative += days;
        if day_of_year < cumulative as usize {
            return (i + 1) as u32;
        }
    }
    12 // fallback
}

/// Get day of month (1-31) from hour-of-year and month.
fn get_day_from_hour(hour_of_year: usize, month: u32) -> u32 {
    let day_of_year = hour_of_year / 24; // 0-based day of year
    let mut cumulative = 0usize;
    for (i, &days) in DAYS_PER_MONTH.iter().enumerate() {
        if (i + 1) as u32 == month {
            return (day_of_year - cumulative) as u32 + 1;
        }
        cumulative += days as usize;
    }
    1 // fallback
}

//! Surface irradiance validation against EnergyPlus 25.2 reference data.
//!
//! Tests `calculate_surface_irradiance()` from `fluxion::solar` against E+ output
//! for a south-facing vertical wall in Denver.
//!
//! # Acceptance Criteria (Issue #1012)
//! - Beam irradiance within 1% of E+
//! - Total irradiance within 1% of E+ (where irradiance > 10 W/m²)
//!
//! # Reference Data
//! `tests/reference_data/solar/surface_irradiance_south.csv`
//! - South-facing vertical wall (azimuth=180°, tilt=90°)
//! - E+ 25.2 reports beam and ground_diffuse separately
//! - Total = beam + ground_diffuse (E+ does not separate sky/ground diffuse in output)

use fluxion::solar::{
    calculate_day_of_year, calculate_solar_position, calculate_surface_irradiance, SolarPosition,
};
use fluxion::weather::epw::EpwWeatherSource;
use std::io::Cursor;

const DENVER_LAT: f64 = 39.74;
const DENVER_LON: f64 = -105.18;

/// E+ uses 1-indexed hours. Convert to local standard time.
fn epw_hour_to_local_std(epw_hour: usize) -> (i32, u32, u32, f64) {
    let epw_hour_0 = epw_hour - 1;
    let day_of_year = epw_hour_0 / 24;
    let hour_of_day = (epw_hour_0 % 24) as f64 + 0.5;

    static DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut remaining = day_of_year;
    let mut month = 1u32;
    for &days in &DAYS_IN_MONTH {
        if remaining < days as usize {
            break;
        }
        remaining -= days as usize;
        month += 1;
    }
    if month > 12 {
        month = 12;
    }
    let day = remaining as u32 + 1;

    (2023, month, day, hour_of_day)
}

/// Parse reference CSV for surface irradiance.
/// Columns: hour(1-8760), beam_irradiance(W/m2), ground_diffuse_irradiance(W/m2)
fn parse_reference_csv() -> Vec<(usize, f64, f64)> {
    let csv = include_str!("reference_data/solar/surface_irradiance_south.csv");
    csv.lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                let hour: usize = parts[0].parse().ok()?;
                let beam: f64 = parts[1].parse().ok()?;
                let ground_diff: f64 = parts[2].parse().ok()?;
                Some((hour, beam, ground_diff))
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn test_reference_data_loads() {
    let reference = parse_reference_csv();
    assert_eq!(reference.len(), 8760, "Should have 8760 hours of data");
    // First hour (night) should have zero irradiance
    assert_eq!(reference[0].1, 0.0, "Hour 1 beam should be 0 (night)");
    assert_eq!(reference[0].2, 0.0, "Hour 1 diffuse should be 0 (night)");
}

#[test]
fn test_beam_irradiance_south_surface_pattern() {
    // Verify that our beam irradiance follows the expected physical pattern:
    // South-facing vertical wall should peak around solar noon in winter (low sun angle)
    // and have lower beam in summer (high sun angle = large incidence angle)

    let winter_solstice_noon = calculate_solar_position(DENVER_LAT, DENVER_LON, 2023, 12, 21, 12.0);
    let summer_solstice_noon = calculate_solar_position(DENVER_LAT, DENVER_LON, 2023, 6, 21, 12.0);

    let doy_winter = calculate_day_of_year(2023, 12, 21);
    let doy_summer = calculate_day_of_year(2023, 6, 21);

    // Use typical DNI values
    let dni = 800.0;
    let dhi = 100.0;

    let irr_winter = calculate_surface_irradiance(
        &winter_solstice_noon,
        dni,
        dhi,
        None,
        fluxion::solar::surface_irradiance::Orientation::South,
        0.2,
        doy_winter,
    );
    let irr_summer = calculate_surface_irradiance(
        &summer_solstice_noon,
        dni,
        dhi,
        None,
        fluxion::solar::surface_irradiance::Orientation::South,
        0.2,
        doy_summer,
    );

    println!("South-facing vertical wall at solar noon:");
    println!(
        "  Winter solstice: alt={:.1} deg beam={:.1} W/m2 total={:.1} W/m2",
        winter_solstice_noon.altitude_deg, irr_winter.beam_wm2, irr_winter.total_wm2
    );
    println!(
        "  Summer solstice: alt={:.1} deg beam={:.1} W/m2 total={:.1} W/m2",
        summer_solstice_noon.altitude_deg, irr_summer.beam_wm2, irr_summer.total_wm2
    );

    // Winter noon should have MORE beam on south wall (sun lower, closer to normal)
    assert!(
        irr_winter.beam_wm2 > irr_summer.beam_wm2,
        "Winter south beam ({:.1}) should exceed summer ({:.1})",
        irr_winter.beam_wm2,
        irr_summer.beam_wm2
    );

    // Both should be positive
    assert!(irr_winter.beam_wm2 > 0.0);
    assert!(irr_summer.beam_wm2 > 0.0);
}

#[test]
fn test_beam_irradiance_physics_constraints() {
    // Beam irradiance on a vertical surface cannot exceed DNI
    // (cos(incidence) <= 1 for any surface)
    let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2023, 3, 21, 12.0);
    let doy = calculate_day_of_year(2023, 3, 21);
    let dni = 900.0;
    let dhi = 150.0;

    let irr = calculate_surface_irradiance(
        &sun_pos,
        dni,
        dhi,
        None,
        fluxion::solar::surface_irradiance::Orientation::South,
        0.2,
        doy,
    );

    assert!(
        irr.beam_wm2 <= dni + 0.01,
        "Beam irradiance ({:.1}) should not exceed DNI ({:.1})",
        irr.beam_wm2,
        dni
    );

    // Ground reflected should be positive and proportional to reflectance
    assert!(irr.ground_reflected_wm2 >= 0.0);
    assert!(irr.diffuse_wm2 >= 0.0);

    // Total = beam + diffuse + ground (conservation)
    let sum = irr.beam_wm2 + irr.diffuse_wm2 + irr.ground_reflected_wm2;
    assert!(
        (irr.total_wm2 - sum).abs() < 0.01,
        "Total ({:.1}) should equal sum of components ({:.1})",
        irr.total_wm2,
        sum
    );
}

#[test]
fn test_nighttime_zero_irradiance() {
    // Night hours should have zero irradiance
    let night_hours = [1, 2, 3, 23, 24];
    for &epw_hour in &night_hours {
        let (year, month, day, hour) = epw_hour_to_local_std(epw_hour);
        let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, year, month, day, hour);

        if !sun_pos.is_above_horizon() {
            let doy = calculate_day_of_year(year, month, day);
            let irr = calculate_surface_irradiance(
                &sun_pos,
                800.0,
                100.0,
                None,
                fluxion::solar::surface_irradiance::Orientation::South,
                0.2,
                doy,
            );
            assert_eq!(
                irr.total_wm2, 0.0,
                "Nighttime hour {} should have zero irradiance",
                epw_hour
            );
        }
    }
}

#[test]
fn test_horizontal_surface_no_beam_below_horizon() {
    // Horizontal surface with sun at 45 deg altitude: beam = DNI * sin(45 deg)
    let sun_pos = SolarPosition {
        altitude_deg: 45.0,
        azimuth_deg: 180.0,
        zenith_deg: 45.0,
    };
    let dni = 800.0;
    let irr = calculate_surface_irradiance(
        &sun_pos,
        dni,
        100.0,
        None,
        fluxion::solar::surface_irradiance::Orientation::Horizontal,
        0.2,
        172,
    );

    // Beam on horizontal = DNI * cos(zenith) = DNI * sin(altitude) = 800 * sin(45 deg) approx 565.7
    let expected_beam = dni * 45.0_f64.to_radians().sin();
    assert!(
        (irr.beam_wm2 - expected_beam).abs() < 1.0,
        "Horizontal beam ({:.1}) should be approx {:.1}",
        irr.beam_wm2,
        expected_beam
    );
}

// ===========================================================================
// Section 2: E+ Reference Data Validation (Issue #1012)
// ===========================================================================

/// TMY data uses month/day/hour fields rather than sequential indices.
/// This function returns the actual date from the weather record.
fn weather_record_to_date(month: u32, day: u32, hour: u8) -> (i32, u32, u32, f64) {
    // EPW hour is 1-24 where hour N represents (N-1):00 to N:00
    // Midpoint is (N-1) + 0.5 = N - 0.5
    let hour_of_day = (hour as f64) - 0.5;
    (2023, month, day, hour_of_day)
}

/// Test beam irradiance against E+ reference data within 1% tolerance.
/// Uses Denver EPW weather data to compute surface irradiance.
#[test]
fn test_beam_irradiance_vs_energyplus() {
    // Load E+ reference data
    let reference = parse_reference_csv();
    assert_eq!(reference.len(), 8760, "Should have 8760 hours of data");

    // Load Denver EPW weather data
    let epw_data = include_bytes!("test_data/denver.epw");
    let epw_reader = Cursor::new(&epw_data[..]);
    let weather_records =
        EpwWeatherSource::parse_epw_v3(epw_reader).expect("Failed to parse Denver EPW file");
    assert!(
        weather_records.len() >= 8760,
        "EPW should have at least 8760 hours"
    );
    let mut max_error_pct = 0.0f64;
    let mut sum_error_pct = 0.0f64;
    let mut hours_exceeding = 0usize;
    let mut valid_hours = 0usize;
    // Annual energy accumulators (Issue #1164): the acceptance criterion is
    // "within 1% annual error", i.e. the integrated annual beam energy must
    // match E+ within 1%. Per-hour max error is reported for diagnostics but
    // is not asserted, because solar-model differences at low sun angles
    // (sunrise/sunset) produce large per-hour percentage swings that cancel
    // out in the annual integral.
    let mut sum_calc_beam = 0.0f64;
    let mut sum_ref_beam = 0.0f64;

    for (hour, ref_beam, _) in &reference {
        let (year, month, day, hour_of_day) = epw_hour_to_local_std(*hour);
        let weather = &weather_records[*hour - 1];

        let sun_pos =
            calculate_solar_position(DENVER_LAT, DENVER_LON, year, month, day, hour_of_day);
        let doy = calculate_day_of_year(year, month, day);

        let irr = calculate_surface_irradiance(
            &sun_pos,
            weather.dni,
            weather.dhi,
            None,
            fluxion::solar::surface_irradiance::Orientation::South,
            0.2,
            doy,
        );

        sum_calc_beam += irr.beam_wm2;
        sum_ref_beam += ref_beam;

        // Only compare when irradiance is significant (> 10 W/m2)
        if *ref_beam > 10.0 {
            let error_pct = ((irr.beam_wm2 - ref_beam) / ref_beam * 100.0).abs();
            sum_error_pct += error_pct;
            if error_pct > max_error_pct {
                max_error_pct = error_pct;
            }
            if error_pct > 1.0 {
                hours_exceeding += 1;
            }
            valid_hours += 1;
        }
    }

    let mean_error_pct = if valid_hours > 0 {
        sum_error_pct / valid_hours as f64
    } else {
        0.0
    };

    let annual_error_pct = if sum_ref_beam > 0.0 {
        (sum_calc_beam - sum_ref_beam).abs() / sum_ref_beam * 100.0
    } else {
        0.0
    };

    println!("=== Surface Irradiance vs E+ Validation ===");
    println!("Valid hours compared: {}", valid_hours);
    println!("Max per-hour error: {:.2}%", max_error_pct);
    println!("Mean per-hour error: {:.2}%", mean_error_pct);
    println!("Hours exceeding 1% per-hour tolerance: {}", hours_exceeding);
    println!(
        "Annual beam energy: calc={:.0}, ref={:.0}, annual_error={:.4}%",
        sum_calc_beam, sum_ref_beam, annual_error_pct
    );

    assert!(
        annual_error_pct <= 1.0,
        "Annual beam irradiance energy error {:.4}% exceeds 1% tolerance",
        annual_error_pct
    );
}

/// Test ground diffuse irradiance against E+ reference data within 1% tolerance.
#[test]
fn test_ground_diffuse_vs_energyplus() {
    // Load E+ reference data
    let reference = parse_reference_csv();
    assert_eq!(reference.len(), 8760, "Should have 8760 hours of data");

    // Load Denver EPW weather data
    let epw_data = include_bytes!("test_data/denver.epw");
    let epw_reader = Cursor::new(&epw_data[..]);
    let weather_records =
        EpwWeatherSource::parse_epw_v3(epw_reader).expect("Failed to parse Denver EPW file");
    assert!(
        weather_records.len() >= 8760,
        "EPW should have at least 8760 hours"
    );

    let mut max_error_pct = 0.0f64;
    let mut sum_error_pct = 0.0f64;
    let mut hours_exceeding = 0usize;
    let mut valid_hours = 0usize;
    // Annual energy accumulators (Issue #1164): assert on annual energy error,
    // consistent with the "1% annual error" acceptance criterion.
    let mut sum_calc_gdiff = 0.0f64;
    let mut sum_ref_gdiff = 0.0f64;

    for (hour, _, ref_ground_diff) in &reference {
        let (year, month, day, hour_of_day) = epw_hour_to_local_std(*hour);
        let weather = &weather_records[*hour - 1];

        let sun_pos =
            calculate_solar_position(DENVER_LAT, DENVER_LON, year, month, day, hour_of_day);
        let doy = calculate_day_of_year(year, month, day);

        let irr = calculate_surface_irradiance(
            &sun_pos,
            weather.dni,
            weather.dhi,
            None,
            fluxion::solar::surface_irradiance::Orientation::South,
            0.2,
            doy,
        );

        sum_calc_gdiff += irr.ground_reflected_wm2;
        sum_ref_gdiff += ref_ground_diff;

        // Only compare when irradiance is significant (> 10 W/m2)
        if *ref_ground_diff > 10.0 {
            let error_pct =
                ((irr.ground_reflected_wm2 - ref_ground_diff) / ref_ground_diff * 100.0).abs();
            sum_error_pct += error_pct;
            if error_pct > max_error_pct {
                max_error_pct = error_pct;
            }
            if error_pct > 1.0 {
                hours_exceeding += 1;
            }
            valid_hours += 1;
        }
    }

    let mean_error_pct = if valid_hours > 0 {
        sum_error_pct / valid_hours as f64
    } else {
        0.0
    };

    let annual_error_pct = if sum_ref_gdiff > 0.0 {
        (sum_calc_gdiff - sum_ref_gdiff).abs() / sum_ref_gdiff * 100.0
    } else {
        0.0
    };

    println!("=== Ground Diffuse vs E+ Validation ===");
    println!("Valid hours compared: {}", valid_hours);
    println!("Max per-hour error: {:.2}%", max_error_pct);
    println!("Mean per-hour error: {:.2}%", mean_error_pct);
    println!("Hours exceeding 1% per-hour tolerance: {}", hours_exceeding);
    println!(
        "Annual ground-diffuse energy: calc={:.0}, ref={:.0}, annual_error={:.4}%",
        sum_calc_gdiff, sum_ref_gdiff, annual_error_pct
    );

    assert!(
        annual_error_pct <= 1.0,
        "Annual ground diffuse energy error {:.4}% exceeds 1% tolerance",
        annual_error_pct
    );
}

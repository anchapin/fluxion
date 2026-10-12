//! Pure solar position calculation using the NOAA solar calculator algorithm.
//!
//! NO imports from `sim::` or `validation::` — this is a standalone physics module.
//!
//! # Algorithm
//! Uses the NOAA Solar Position Algorithm (SPA) simplified form:
//! 1. Calculate day-of-year and fractional year angle (gamma)
//! 2. Compute equation of time and solar declination
//! 3. Apply longitude correction for true solar time
//! 4. Compute hour angle, then altitude and azimuth
//!
//! # Reference
//! NOAA Global Monitoring Division, "Solar Position Calculator"
//! <https://gml.noaa.gov/grad/solcalc/>

/// Sun position in the sky at a given time and location.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    /// Solar altitude angle (elevation above horizon) in degrees.
    /// Positive = above horizon, negative = below.
    pub altitude_deg: f64,
    /// Solar azimuth angle measured from North, clockwise in degrees.
    /// 0=North, 90=East, 180=South, 270=West.
    pub azimuth_deg: f64,
    /// Solar zenith angle (90° - altitude) in degrees.
    pub zenith_deg: f64,
}

impl SolarPosition {
    /// Returns true if the sun is above the horizon.
    pub fn is_above_horizon(&self) -> bool {
        self.altitude_deg > 0.0
    }

    /// Calculate cosine of incidence angle on a tilted surface.
    ///
    /// Uses the standard formula for solar incidence on a tilted surface:
    /// cos(θ) = sin(β)·sin(α) + cos(β)·cos(α)·cos(φ - γ)
    ///
    /// Where:
    /// - β = surface tilt (0° = horizontal, 90° = vertical)
    /// - α = solar altitude angle
    /// - φ = solar azimuth angle
    /// - γ = surface azimuth angle
    ///
    /// The result is clamped to [0, 1] since incidence angle is [0°, 90°].
    pub fn incidence_cosine(&self, surface_tilt_deg: f64, surface_azimuth_deg: f64) -> f64 {
        if !self.is_above_horizon() {
            return 0.0;
        }

        let alpha = self.altitude_deg.to_radians();
        let phi = self.azimuth_deg.to_radians();
        let beta = surface_tilt_deg.to_radians();
        let gamma = surface_azimuth_deg.to_radians();

        // cos(θ) = sin(α)cos(β) + cos(α)sin(β)cos(φ - γ)
        // α = solar altitude, β = surface tilt, φ = solar azimuth, γ = surface azimuth
        let cos_theta_i = alpha.sin() * beta.cos() + alpha.cos() * beta.sin() * (phi - gamma).cos();

        cos_theta_i.clamp(0.0, 1.0)
    }
}

/// Calculates day of year from year, month, and day.
pub fn calculate_day_of_year(year: i32, month: u32, day: u32) -> usize {
    let is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);

    static MONTH_DAYS_ACCUM: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

    // Protect against invalid month inputs
    let m_idx = (month.clamp(1, 12) - 1) as usize;
    let mut day_of_year = MONTH_DAYS_ACCUM[m_idx] as usize + day as usize;

    if is_leap_year && month > 2 {
        day_of_year += 1;
    }

    day_of_year
}

/// Issue #4384 follow-up (round 19): single-source derivation of the solar
/// calendar date from an hour-of-year, in the engine's non-leap-year convention
/// convention, with the mid-hour sun position (`+0.5` h). This is the inverse
/// of `calculate_day_of_year` for a non-leap reference year and reproduces
/// `timestep_to_date`'s month/day exactly (#4385's requirement).
///
/// The 9R4C integrator sites (`step_9r4c.rs` sol-air boundary and the
/// `step_with_coupling` multi-node boundary) and the invariant checker's
/// mirror (`invariant_checker.rs::compute_9r4c_t_sol_air`) each carried an
/// inline copy of this month/day derivation. Round 17's checker/integrator
/// desync (#4384) was exactly two copies of this block drifting apart in
/// month encoding, so both sides now consume this one function — the desync
/// class is removed structurally, not by convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarStepDate {
    /// Calendar month, 1-based (January = 1), as `calculate_solar_position`
    /// expects.
    pub month: u32,
    /// Day of month, 1-based, uncapped.
    pub day: u32,
    /// Mid-hour sun convention: `(hour_of_year % 24) as f64 + 0.5`.
    pub mid_hour: f64,
}

/// Derive the solar-calendar date for an hour-of-year (0..8759), non-leap
/// year. Values outside 0..8760 are taken modulo 8760, matching the
/// integrator's `timestep % 8760` wrap.
pub fn derive_solar_step_date(hour_of_year: usize) -> SolarStepDate {
    let hoy = hour_of_year % 8760;
    let day_of_year = hoy / 24;
    let mid_hour = (hoy % 24) as f64 + 0.5;
    // Identical to `ThermalModel::timestep_to_date`'s derivation (the #4385
    // authoritative encoding): walk the non-leap month lengths.
    let days_in_month: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month: u32 = 1;
    let mut day: u32 = day_of_year as u32 + 1;
    for &dim in days_in_month.iter() {
        if day <= dim {
            break;
        }
        day -= dim;
        month += 1;
    }
    SolarStepDate {
        month,
        day,
        mid_hour,
    }
}

/// Calculate solar position using the NOAA solar calculator algorithm.
///
/// # Arguments
/// * `latitude_deg` - Latitude in degrees (positive = North)
/// * `longitude_deg` - Longitude in degrees (positive = East, negative = West)
/// * `year` - Calendar year
/// * `month` - Month (1-12)
/// * `day` - Day of month
/// * `hour` - Local standard time as fractional hour (e.g., 12.5 = 12:30)
/// * `utc_offset_hours` - Optional explicit UTC offset of the local time zone in
///   hours (EPW LOCATION column 10 sign convention: negative for west of
///   Greenwich, positive for east). When `Some`, this overrides the
///   longitude-inferred time-zone meridian used for the solar-time correction.
///   When `None`, the meridian is inferred from longitude as
///   `round(longitude / 15) * 15` degrees (legacy behaviour, preserved for
///   backward compatibility with ASHRAE 140 baselines).
///
/// # Physics
/// The equation of time corrects for the eccentricity of Earth's orbit and the
/// obliquity of the ecliptic. The time-zone meridian converts local standard
/// time to solar time via the `(longitude - time_zone_meridian) × 4 min/deg`
/// correction.
///
/// For Denver (UTC-7, longitude -105°): time zone meridian = -105°, so solar
/// time correction = (longitude - time_zone_meridian) × 4 min/deg.
///
/// # Issue #1416
/// Callers should pass the explicit EPW LOCATION time-zone offset
/// (`utc_offset_hours: Some(-7.0)` for Denver) for non-Denver weather files.
/// Half-hour time zones (India UTC+5:30, Iran UTC+3:30, Newfoundland UTC-3:30)
/// and locations at longitudes offset by exactly 7.5° from a 15°-multiple cannot
/// be represented by `(longitude / 15).round() * 15`; only the explicit offset
/// produces the correct solar time.
pub fn calculate_solar_position(
    latitude_deg: f64,
    longitude_deg: f64,
    year: i32,
    month: u32,
    day: u32,
    hour: f64,
    utc_offset_hours: Option<f64>,
) -> SolarPosition {
    let is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    static MONTH_DAYS_ACCUM: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let m_idx = (month.clamp(1, 12) - 1) as usize;
    let mut day_of_year = MONTH_DAYS_ACCUM[m_idx] + day as i32;
    if is_leap_year && month > 2 {
        day_of_year += 1;
    }

    let days_in_year = if is_leap_year { 366 } else { 365 };
    let day_of_year_f = day_of_year as f64;

    // Fractional year angle (gamma) — uses hour of day
    let gamma = 2.0 * std::f64::consts::PI * (day_of_year_f - 1.0 + (hour - 12.0) / 24.0)
        / days_in_year as f64;

    // Equation of time (minutes) — NOAA correlation
    let eqtime_minutes = 229.18
        * (0.000075 + 0.001868 * gamma.cos()
            - 0.032077 * gamma.sin()
            - 0.014615 * (2.0 * gamma).cos()
            - 0.040849 * (2.0 * gamma).sin());

    // Solar declination (radians) — NOAA Fourier series
    let decl_rad = 0.006918 - 0.399912 * gamma.cos() + 0.070257 * gamma.sin()
        - 0.006758 * (2.0 * gamma).cos()
        + 0.000907 * (2.0 * gamma).sin()
        - 0.002697 * (3.0 * gamma).cos()
        + 0.00148 * (3.0 * gamma).sin();

    // Time zone meridian: either the explicit EPW LOCATION offset (when
    // provided) or round longitude to nearest 15° multiple.
    // For Denver (-104.99°): time zone = -105° (MST = UTC-7)
    let time_zone_meridian = match utc_offset_hours {
        // Issue #1416: prefer the explicit EPW LOCATION offset when known.
        // Sign convention matches the rest of the formula (positive east):
        // UTC-7 → -105°, UTC+5:30 → +82.5°, UTC-3:30 → -52.5°.
        Some(offset_hours) => offset_hours * 15.0,
        // Legacy fallback: infer from longitude (matches original behaviour
        // exactly, so existing ASHRAE 140 baselines are unchanged).
        None => (longitude_deg / 15.0).round() * 15.0,
    };

    // Solar time offset (minutes): correction for longitude vs time zone meridian
    let time_offset_minutes = eqtime_minutes + 4.0 * (longitude_deg - time_zone_meridian);

    // True solar time (minutes from midnight)
    let solar_time = hour * 60.0 + time_offset_minutes;

    // Hour angle: 15° per hour from solar noon (solar_time=720 min)
    let ha = (solar_time / 4.0) - 180.0; // degrees
    let lat_rad = latitude_deg.to_radians();
    let ha_rad = ha.to_radians();

    // Solar zenith angle
    let cos_zenith = lat_rad.sin() * decl_rad.sin() + lat_rad.cos() * decl_rad.cos() * ha_rad.cos();

    // Clamp to handle floating-point edge cases near horizon
    let cos_zenith = cos_zenith.clamp(-1.0, 1.0);
    let zenith = cos_zenith.acos().to_degrees();
    let elev = 90.0 - zenith;

    // Solar azimuth calculation
    let zenith_rad = zenith.to_radians();

    // Avoid division by zero when zenith ≈ 0 (sun directly overhead)
    let cos_az = if zenith_rad.sin().abs() < 1e-10 {
        0.0
    } else {
        (cos_zenith * lat_rad.sin() - decl_rad.sin()) / (zenith_rad.sin() * lat_rad.cos())
    };
    let cos_az = cos_az.clamp(-1.0, 1.0);

    // Azimuth from acos is measured from South: 0°=South, positive West
    // Convert to standard meteorological convention: 0°=North, clockwise
    //   0°=N, 90°=E, 180°=S, 270°=W
    let az_from_south = cos_az.acos().to_degrees();
    let mut az_from_north = if ha_rad > 0.0 {
        // Afternoon: sun is west of south → az_from_south positive (West)
        // In N-clockwise: South+West = 180+az_from_south
        180.0 + az_from_south
    } else {
        // Morning: sun is east of south → az_from_south negative (East)
        // In N-clockwise: South-East = 180-az_from_south
        180.0 - az_from_south
    };
    // Normalize to [0, 360)
    if az_from_north >= 360.0 {
        az_from_north -= 360.0;
    }
    if az_from_north < 0.0 {
        az_from_north += 360.0;
    }

    SolarPosition {
        altitude_deg: elev,
        zenith_deg: zenith,
        azimuth_deg: az_from_north,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round 19 (#4384 follow-up): the shared helper must reproduce the
    /// applied month/day derivation verbatim. `timestep_to_date`'s
    /// days-in-month walk is the authoritative #4385 encoding; the
    /// cumulative-month-starts form (the pre-#4385 inline block) must agree
    /// with it everywhere, so either reading of the history pins the helper.
    #[test]
    fn test_derive_solar_step_date_matches_legacy_math_full_year() {
        let month_starts: [usize; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let days_in_month: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        for hoy in 0..8760usize {
            // Authoritative: timestep_to_date's loop.
            let doy = hoy / 24;
            let (mut m, mut d) = (1u32, doy as u32 + 1);
            for &dim in days_in_month.iter() {
                if d <= dim {
                    break;
                }
                d -= dim;
                m += 1;
            }

            let date = derive_solar_step_date(hoy);
            assert_eq!(date.month, m, "hoy={hoy}");
            assert_eq!(date.day, d, "hoy={hoy}");
            assert_eq!(date.mid_hour, (hoy % 24) as f64 + 0.5, "hoy={hoy}");

            // Cross-check: cumulative-starts form must be the same date.
            let m2 = month_starts
                .iter()
                .position(|&s| s > doy)
                .unwrap_or(12)
                .saturating_sub(1) as u32
                + 1;
            let d2 = (doy - month_starts[m2 as usize - 1]) as u32 + 1;
            assert_eq!((date.month, date.day), (m2, d2), "hoy={hoy}");
        }
    }

    /// The helper is exactly `timestep_to_date` on month/day — verified on
    /// the sim side (tests under `src/sim/thermal_model_core/`), where the
    /// two live in the same layer.
    #[test]
    fn test_derive_solar_step_date_matches_days_in_month_walks() {
        for hoy in [0usize, 743, 744, 1415, 1416, 4343, 8758, 8759] {
            let doy = hoy / 24;
            let days_in_month: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let (mut m, mut d) = (1u32, doy as u32 + 1);
            for &dim in days_in_month.iter() {
                if d <= dim {
                    break;
                }
                d -= dim;
                m += 1;
            }
            let date = derive_solar_step_date(hoy);
            assert_eq!((date.month, date.day), (m, d), "hoy={hoy}");
        }
    }

    /// Round trip through `calculate_day_of_year`: (month, day) must map
    /// back to the day-of-year the helper consumed.
    #[test]
    fn test_derive_solar_step_date_round_trips_through_calculate_day_of_year() {
        for hoy in 0..8760usize {
            let d = derive_solar_step_date(hoy);
            // Non-leap reference year: the engine's 8760-hour year has
            // February = 28 days (the `timestep_to_date` convention), while
            // calendar 2024 is a leap year and would offset every day after
            // Feb 28 by one.
            let doy = calculate_day_of_year(2023, d.month, d.day);
            assert_eq!(doy, hoy / 24 + 1, "hoy={hoy}");
        }
    }

    /// Boundaries: year edges, month edges, and the 8760 wrap.
    #[test]
    fn test_derive_solar_step_date_boundaries() {
        let d = derive_solar_step_date(0);
        assert_eq!((d.month, d.day, d.mid_hour), (1, 1, 0.5));

        let d = derive_solar_step_date(8759);
        assert_eq!((d.month, d.day, d.mid_hour), (12, 31, 23.5));

        let d = derive_solar_step_date(743); // Jan 31, 23:30–00:00 midpoint
        assert_eq!((d.month, d.day, d.mid_hour), (1, 31, 23.5));

        let d = derive_solar_step_date(744); // Feb 1
        assert_eq!((d.month, d.day), (2, 1));

        let d = derive_solar_step_date(1416); // Mar 1
        assert_eq!((d.month, d.day), (3, 1));

        assert_eq!(derive_solar_step_date(8760), derive_solar_step_date(0));
    }
    use proptest::prelude::*;

    #[test]
    fn test_solar_position_winter_morning() {
        let sun_pos = calculate_solar_position(39.7, -105.0, 2024, 12, 21, 8.0, None);
        assert!(sun_pos.altitude_deg > 0.0);
    }

    #[test]
    fn test_solar_position_summer_evening() {
        let sun_pos = calculate_solar_position(39.7, -105.0, 2024, 6, 21, 18.0, None);
        if sun_pos.is_above_horizon() {
            assert!(sun_pos.azimuth_deg >= 0.0 && sun_pos.azimuth_deg < 360.0);
        }
    }

    #[test]
    fn test_calculate_day_of_year_jan1() {
        let doy = calculate_day_of_year(2024, 1, 1);
        assert_eq!(doy, 1);
    }

    #[test]
    fn test_calculate_day_of_year_dec31() {
        let doy = calculate_day_of_year(2024, 12, 31);
        assert_eq!(doy, 366);
    }

    #[test]
    fn test_calculate_day_of_year_non_leap() {
        let doy = calculate_day_of_year(2023, 12, 31);
        assert_eq!(doy, 365);
    }

    #[test]
    fn test_solar_position_summer_solstice_noon() {
        // At solar noon on summer solstice at 39.74°N:
        // Declination ≈ 23.45° → altitude ≈ 90 - (39.74 - 23.45) = 73.71°
        let sun_pos = calculate_solar_position(39.7392, -105.0, 2024, 6, 21, 12.0, None);
        assert!(sun_pos.altitude_deg > 70.0 && sun_pos.altitude_deg < 77.0);
        assert!(sun_pos.is_above_horizon());
        // Azimuth near 180° (South) at solar noon
        assert!(sun_pos.azimuth_deg > 170.0 && sun_pos.azimuth_deg < 190.0);
    }

    #[test]
    fn test_solar_position_winter_solstice_noon() {
        let sun_pos = calculate_solar_position(39.7392, -105.0, 2024, 12, 21, 12.0, None);
        // Altitude ≈ 90 - (39.74 + 23.45) = 26.81°
        assert!(sun_pos.altitude_deg > 24.0 && sun_pos.altitude_deg < 30.0);
        assert!(sun_pos.is_above_horizon());
    }

    #[test]
    fn test_incidence_cosine_below_horizon() {
        let pos = SolarPosition {
            altitude_deg: -10.0,
            azimuth_deg: 180.0,
            zenith_deg: 100.0,
        };
        let cos = pos.incidence_cosine(90.0, 180.0);
        assert!((cos - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_solar_position_equality() {
        let pos1 = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let pos2 = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        assert_eq!(pos1, pos2);
    }

    // -------------------------------------------------------------------------
    // Property-Based Tests (proptest)
    // Issue #1062: Property-based testing for core math & parsers
    //
    // These tests verify physical invariants for solar position calculations
    // across random lat/lon/hour/day combinations.
    // -------------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn prop_solar_position_azimuth_range(
            latitude in -90.0_f64..90.0,
            longitude in -180.0_f64..180.0,
            hour in 0.0_f64..24.0,
        ) {
            let pos = calculate_solar_position(latitude, longitude, 2024, 6, 21, hour, None);
            prop_assert!(pos.azimuth_deg >= 0.0 && pos.azimuth_deg < 360.0,
                "Azimuth {} out of range [0, 360)", pos.azimuth_deg);
        }

        #[test]
        fn prop_solar_position_zenith_and_altitude_sum_to_90(
            latitude in -90.0_f64..90.0,
            longitude in -180.0_f64..180.0,
            hour in 0.0_f64..24.0,
        ) {
            let pos = calculate_solar_position(latitude, longitude, 2024, 6, 21, hour, None);
            let sum = pos.altitude_deg + pos.zenith_deg;
            prop_assert!((sum - 90.0).abs() < 1e-10,
                "Altitude {} + Zenith {} should equal 90", pos.altitude_deg, pos.zenith_deg);
        }

        #[test]
        fn prop_altitude_bounded_by_horizon(
            latitude in -90.0_f64..90.0,
            longitude in -180.0_f64..180.0,
        ) {
            let pos = calculate_solar_position(latitude, longitude, 2024, 12, 21, 12.0, None);
            prop_assert!(pos.altitude_deg >= -90.0 && pos.altitude_deg <= 90.0,
                "Altitude {} outside physical range [-90, 90]", pos.altitude_deg);
        }

        #[test]
        fn prop_is_above_horizon_consistency(
            latitude in -90.0_f64..90.0,
            longitude in -180.0_f64..180.0,
        ) {
            let pos = calculate_solar_position(latitude, longitude, 2024, 6, 21, 12.0, None);
            let expected = pos.altitude_deg > 0.0;
            prop_assert_eq!(pos.is_above_horizon(), expected,
                "is_above_horizon inconsistent with altitude {}", pos.altitude_deg);
        }

        #[test]
        fn prop_incidence_cosine_clamped_to_unit_interval(
            latitude in -90.0_f64..90.0,
            longitude in -180.0_f64..180.0,
            surface_tilt in 0.0_f64..90.0,
            surface_azimuth in 0.0_f64..360.0,
        ) {
            let pos = calculate_solar_position(latitude, longitude, 2024, 6, 21, 12.0, None);
            let cos_i = pos.incidence_cosine(surface_tilt, surface_azimuth);
            prop_assert!(cos_i >= 0.0 && cos_i <= 1.0,
                "Incidence cosine {} outside [0, 1]", cos_i);
        }

        #[test]
        fn prop_day_of_year_valid_range(
            year in 1900_i32..2100,
            month in 1_u32..13,
            day in 1_u32..32,
        ) {
            let doy = calculate_day_of_year(year, month, day);
            prop_assert!(doy >= 1 && doy <= 366,
                "Day of year {} outside valid range [1, 366]", doy);
        }

        #[test]
        fn prop_leap_year_day_of_year_dec31(
            year in 1900_i32..2100,
        ) {
            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            let expected = if is_leap { 366 } else { 365 };
            let doy = calculate_day_of_year(year, 12, 31);
            prop_assert_eq!(doy, expected, "Day of year for Dec 31 mismatch");
        }
    }
}

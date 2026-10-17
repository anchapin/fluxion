//! ASHRAE 140 §6 Weather Driver Tests — first smoke tests (Issue #4269).
//!
//! The six WD files are real-location weather chosen to break weather-file
//! interpretation in six different ways. These tests pin the engine
//! behaviours §6 depends on that do not require the §6.3 emitter (#4283) or
//! the conformance table (#4276):
//!
//! 1. WD100 vs WD600 differ in exactly EPW field 33 (0-indexed 32, albedo);
//!    the engine reads it per hour and it drives ground-reflected shortwave.
//! 2. WD500 (+5.5 h UTC offset): solar position honours the half-hour offset;
//!    truncating to +5 shifts solar noon by ~30 minutes and must not happen.
//! 3. WD300 (Santiago, lat −33.4): southern-hemisphere declination — at
//!    December solar noon the sun sits in the *northern* sky.
//! 4. WD400 (Barrow, lat 71.3): polar night/day hours produce finite,
//!    non-negative irradiance.

use fluxion::solar::solar_position::calculate_solar_position;
use fluxion::solar::surface_irradiance::{calculate_surface_irradiance, Orientation};
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;
use std::path::PathBuf;

fn wd_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/weather")
        .join(name)
}

fn wd_source(name: &str) -> EpwWeatherSource {
    EpwWeatherSource::from_file(
        wd_path(name)
            .to_str()
            .expect("WD weather path must be valid UTF-8"),
    )
    .unwrap_or_else(|e| panic!("failed to parse {name}: {e:?}"))
}

/// §6 acceptance: albedo is read per hour from EPW field 33 (0-indexed 32)
/// and drives ground-reflected shortwave. WD100 and WD600 are byte-identical
/// except that field (0 vs 0.2), so any difference in ground-reflected
/// irradiance is attributable to it and nothing else.
#[test]
fn wd100_wd600_albedo_discriminates_ground_reflected() {
    let wd100 = wd_source("WD100.epw");
    let wd600 = wd_source("WD600.epw");

    // Spot-check the parsed albedo at several hours spread over the year.
    for hour in [0, 1000, 4380, 7000, 8759] {
        let a100 = wd100
            .get_hourly_data(hour)
            .expect("WD100 row")
            .albedo
            .expect("WD100 reports albedo 0 explicitly");
        let a600 = wd600
            .get_hourly_data(hour)
            .expect("WD600 row")
            .albedo
            .expect("WD600 reports albedo 0.2 explicitly");
        assert_eq!(
            a100, 0.0,
            "WD100 hour {hour}: albedo must be the authored 0"
        );
        assert_eq!(
            a600, 0.2,
            "WD600 hour {hour}: albedo must be the authored 0.2"
        );
    }

    // The albedo must actually drive ground-reflected shortwave: same sky,
    // different ground, different irradiance on a tilted surface.
    let sun = calculate_solar_position(39.833, -104.650, 2023, 6, 21, 12.0, Some(-7.0));
    assert!(sun.is_above_horizon());
    let flat = Orientation::North;
    let rho0 = calculate_surface_irradiance(&sun, 800.0, 150.0, None, flat, 0.0, 172);
    let rho02 = calculate_surface_irradiance(&sun, 800.0, 150.0, None, flat, 0.2, 172);
    assert!(
        rho02.ground_reflected_wm2 > rho0.ground_reflected_wm2,
        "albedo 0.2 must produce more ground-reflected shortwave than albedo 0"
    );
    assert!(
        rho0.ground_reflected_wm2.abs() < 1e-9,
        "WD100 albedo 0 must give zero ground-reflected shortwave"
    );
}

/// §6 acceptance: solar position accepts non-integer UTC offsets. WD500 is
/// New Delhi (UTC+5.5); integer-truncating to +5 shifts solar noon by about
/// half an hour. The correct offset puts the altitude peak within 20 minutes
/// of 12:00 local standard time (equation-of-time + longitude correction);
/// the truncated offset must be measurably worse.
#[test]
fn wd500_halfhour_utc_offset_peaks_at_solar_noon() {
    let lat = 28.567_f64;
    let lon = 77.103_f64;
    let day = (2023, 4, 15);

    fn peak_error_near_noon(
        lat: f64,
        lon: f64,
        year: i32,
        month: u32,
        day: u32,
        utc_offset: f64,
    ) -> (f64, f64) {
        // Sample ±90 min around local noon at 5-minute steps; return the
        // offset (minutes) of the altitude peak from 12:00 and its altitude.
        let mut best = (f64::MAX, f64::MIN);
        for step in -18_i32..=18 {
            let hour = 12.0 + f64::from(step) * (5.0 / 60.0);
            let sun = calculate_solar_position(lat, lon, year, month, day, hour, Some(utc_offset));
            if sun.altitude_deg > best.1 {
                best = (step as f64 * 5.0, sun.altitude_deg);
            }
        }
        best
    }

    let (err_correct, _alt_correct) = peak_error_near_noon(lat, lon, day.0, day.1, day.2, 5.5);
    let (err_truncated, _alt_truncated) = peak_error_near_noon(lat, lon, day.0, day.1, day.2, 5.0);

    assert!(
        err_correct.abs() <= 20.0,
        "UTC+5.5 peak must be within 20 min of 12:00 local, got {err_correct} min"
    );
    assert!(
        (err_correct - err_truncated).abs() >= 15.0,
        "truncating to UTC+5 must shift the peak by ≥15 min (got {err_truncated} vs {err_correct})"
    );
}

/// §6 acceptance: southern-hemisphere declination. WD300 is Santiago
/// (lat −33.393). December is southern summer: the sun rides high (≈77° at
/// solar noon on the December solstice, declination −23.4° being *north* of
/// the site) and passes through the *northern* sky; June inverts this. An
/// engine that flips the declination sign puts the December sun at ≈33°
/// elevation and south — the failure mode this test pins.
#[test]
fn wd300_southern_hemisphere_december_noon_sun_is_north() {
    let lat = -33.393_f64;
    let lon = -70.786_f64;

    fn noon_state(lat: f64, lon: f64, month: u32, day: u32) -> (f64, f64) {
        // Scan 10:00–14:00 local at 5-minute steps; return the altitude and
        // azimuth at the altitude peak (solar noon).
        let mut best = (f64::MIN, 0.0_f64);
        for step in 0..48 {
            let hour = 10.0 + step as f64 * (5.0 / 60.0);
            let sun = calculate_solar_position(lat, lon, 2023, month, day, hour, Some(-4.0));
            if sun.altitude_deg > best.0 {
                best = (sun.altitude_deg, sun.azimuth_deg);
            }
        }
        best
    }

    let (alt_dec, az_dec) = noon_state(lat, lon, 12, 21);
    let (alt_jun, _az_jun) = noon_state(lat, lon, 6, 21);

    assert!(
        (70.0..=82.0).contains(&alt_dec),
        "December solar-noon altitude ≈ 90 − |lat − decl| ≈ 80°, got {alt_dec}"
    );
    let north_offset = az_dec.min(360.0 - az_dec);
    assert!(
        north_offset <= 30.0,
        "December sun must pass through the northern sky at noon (azimuth ≈ 0/360), got {az_dec}"
    );
    assert!(
        alt_jun < alt_dec - 30.0,
        "June solar-noon sun must be far lower than December's (got {alt_jun} vs {alt_dec})"
    );
}

/// §6 acceptance: polar cases produce finite, non-negative irradiance
/// through polar night and polar day. WD400 is Barrow, AK (lat 71.286).
#[test]
fn wd400_polar_night_and_day_irradiance_finite_nonnegative() {
    let wd400 = wd_source("WD400.epw");
    let lat = 71.286_f64;
    let lon = -156.767_f64;

    // Sample hours across polar night (late Dec), polar day (late Jun) and
    // the equinox, driving the irradiance model from the file's own DNI/DHI.
    for &(month, day) in &[(12u32, 21u32), (12, 22), (6, 21), (3, 20)] {
        let day_of_year = match month {
            3 => 79,
            6 => 172,
            _ => 355,
        };
        for hour_step in 0..24 {
            let hour = hour_step as f64;
            let sun = calculate_solar_position(lat, lon, 2023, month, day, hour, Some(-9.0));
            assert!(sun.altitude_deg.is_finite() && sun.azimuth_deg.is_finite());

            // Hour-of-year index for the weather row (0-based).
            let (m0, d0) = (month, day);
            let hoy = days_before(m0) * 24 + hour_step;
            let rec = wd400
                .get_hourly_data(hoy.min(8759))
                .unwrap_or_else(|e| panic!("WD400 row for {m0}-{d0} h{hour_step}: {e:?}"));
            let irr = calculate_surface_irradiance(
                &sun,
                rec.dni,
                rec.dhi,
                None,
                Orientation::Horizontal,
                rec.albedo.unwrap_or(0.2),
                day_of_year,
            );
            let vals = [
                irr.beam_wm2,
                irr.diffuse_wm2,
                irr.ground_reflected_wm2,
                irr.total_wm2,
            ];
            for v in vals {
                assert!(
                    v.is_finite(),
                    "non-finite irradiance {m0}-{d0} h{hour_step}"
                );
                assert!(
                    v >= -1e-9,
                    "negative irradiance {v} at {m0}-{d0} h{hour_step}"
                );
            }
            if !sun.is_above_horizon() {
                assert!(
                    irr.total_wm2.abs() < 1e-9,
                    "sun below horizon must give zero irradiance at {m0}-{d0} h{hour_step}"
                );
            }
        }
    }
}

fn days_before(month: u32) -> usize {
    const ACC: [usize; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    ACC[(month as usize - 1).min(11)]
}

// ===========================================================================
// §6 case harness (Issue #4269): the six WD files driven through the engine
// on the low-mass case configuration, producing the per-case outputs the
// #4283 emitter and #4276 conformance table consume.
// ===========================================================================

use fluxion::validation::section6::{
    run_all_section6_cases, section6_workflow_csv, Section6CaseResult,
};

/// One engine pass for the whole binary: each test reads the same results.
fn harness_results() -> &'static Vec<Section6CaseResult> {
    static RESULTS: std::sync::OnceLock<Vec<Section6CaseResult>> = std::sync::OnceLock::new();
    RESULTS.get_or_init(|| run_all_section6_cases().expect("all six §6 cases must run"))
}

/// §6.2.1.1 outputs: every case runs the full year and reports finite
/// heating, cooling, and peaks. A NaN or infinite value anywhere is a
/// harness failure before any comparison is meaningful.
#[test]
fn section6_all_six_cases_run_with_finite_outputs() {
    let results = harness_results();
    assert_eq!(results.len(), 6, "six §6 cases must run");
    for r in results {
        assert!(
            r.is_finite(),
            "{} produced a non-finite output: {r:?}",
            r.case_id
        );
        assert!(
            r.annual_heating_mwh >= 0.0 && r.annual_cooling_mwh >= 0.0,
            "{}: reported loads must be non-negative (§6.2.1.1.5)",
            r.case_id
        );
    }
}

/// Site metadata must come from each file's own EPW LOCATION header, not
/// the fabric cohort's Denver default — WD300/WD400/WD500 would otherwise
/// run with the wrong sun entirely.
#[test]
fn section6_site_metadata_follows_each_epw_header() {
    let results = harness_results();
    let by_id = |id: &str| {
        results
            .iter()
            .find(|r| r.case_id == id)
            .unwrap_or_else(|| panic!("{id} missing"))
    };
    let wd100 = by_id("WD100");
    let wd300 = by_id("WD300");
    let wd400 = by_id("WD400");
    let wd500 = by_id("WD500");

    assert!((wd100.latitude_deg - 39.833).abs() < 1e-2, "WD100 lat");
    assert!((wd100.longitude_deg - (-104.650)).abs() < 1e-2, "WD100 lon");
    assert_eq!(wd100.utc_offset_hours, Some(-7.0));
    assert!(
        (wd300.latitude_deg - (-33.393)).abs() < 1e-2,
        "WD300 southern latitude"
    );
    assert!(
        (wd400.latitude_deg - 71.286).abs() < 1e-2,
        "WD400 arctic latitude"
    );
    assert_eq!(wd500.utc_offset_hours, Some(5.5), "WD500 half-hour offset");
}

/// The sharpest test in the section: WD100 and WD600 are byte-identical
/// except EPW field 33 (albedo 0 vs 0.2). Through the full engine path,
/// with the §6 authored-albedo override, the pair must still produce
/// different outputs — with the 0.0→0.2 fabric fallback they would come
/// out identical and the isolation would silently vanish.
#[test]
fn section6_wd100_wd600_differ_through_full_engine_path() {
    let results = harness_results();
    let wd100 = results.iter().find(|r| r.case_id == "WD100").unwrap();
    let wd600 = results.iter().find(|r| r.case_id == "WD600").unwrap();

    let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
    assert!(
        rel(wd100.annual_heating_mwh, wd600.annual_heating_mwh) > 1e-4
            || rel(wd100.annual_cooling_mwh, wd600.annual_cooling_mwh) > 1e-4,
        "WD100 and WD600 must differ through the engine: albedo is not being read \
         (heating {}/{} MWh, cooling {}/{} MWh)",
        wd100.annual_heating_mwh,
        wd600.annual_heating_mwh,
        wd100.annual_cooling_mwh,
        wd600.annual_cooling_mwh
    );
    // Direction: WD600's 0.2 ground reflects more shortwave onto the
    // surfaces, so it cannot cool less than WD100.
    assert!(
        wd600.annual_cooling_mwh >= wd100.annual_cooling_mwh - 1e-6,
        "WD600 (albedo 0.2) must not cool less than WD100 (albedo 0): \
         {wd100:?} vs {wd600:?}"
    );
}

/// WD300 (Santiago, lat −33.4): seasons are inverted — heating must
/// concentrate in the southern winter (Jun–Aug) and cooling in the
/// southern summer (Dec–Feb). A northern-hemisphere reading of the file
/// inverts both.
#[test]
fn section6_wd300_southern_hemisphere_seasonality() {
    let results = harness_results();
    let wd300 = results.iter().find(|r| r.case_id == "WD300").unwrap();
    let sum = |a: &[f64; 12], idx: &[usize]| idx.iter().map(|&i| a[i]).sum::<f64>();
    let sh_winter = [5, 6, 7]; // Jun, Jul, Aug
    let sh_summer = [11, 0, 1]; // Dec, Jan, Feb
    let heat_winter = sum(&wd300.monthly_heating_mwh, &sh_winter);
    let heat_summer = sum(&wd300.monthly_heating_mwh, &sh_summer);
    let cool_winter = sum(&wd300.monthly_cooling_mwh, &sh_winter);
    let cool_summer = sum(&wd300.monthly_cooling_mwh, &sh_summer);
    assert!(
        heat_winter > heat_summer,
        "WD300 heating must peak in southern winter (Jun-Aug {heat_winter} MWh) \
         over southern summer (Dec-Feb {heat_summer} MWh)"
    );
    assert!(
        cool_summer > cool_winter,
        "WD300 cooling must peak in southern summer (Dec-Feb {cool_summer} MWh) \
         over southern winter (Jun-Aug {cool_winter} MWh)"
    );
}

/// WD400 (Barrow, lat 71.3): arctic — the lowest annual cooling of the six
/// and heating at or above every temperate case. Also the polar-finiteness
/// acceptance at engine level (polar night/day through the full year).
#[test]
fn section6_wd400_arctic_extremes() {
    let results = harness_results();
    let wd400 = results.iter().find(|r| r.case_id == "WD400").unwrap();
    for r in results {
        if r.case_id != "WD400" {
            assert!(
                wd400.annual_cooling_mwh <= r.annual_cooling_mwh,
                "WD400 must have the lowest annual cooling: {r:?}"
            );
        }
    }
    let wd200 = results.iter().find(|r| r.case_id == "WD200").unwrap();
    assert!(
        wd400.annual_heating_mwh > wd200.annual_heating_mwh,
        "arctic Barrow must heat more than Atlanta: {} vs {} MWh",
        wd400.annual_heating_mwh,
        wd200.annual_heating_mwh
    );
}

/// WD500 (New Delhi, +5.5 h): hot climate — cooling-dominant annual loads.
#[test]
fn section6_wd500_hot_climate_cooling_dominant() {
    let results = harness_results();
    let wd500 = results.iter().find(|r| r.case_id == "WD500").unwrap();
    assert!(
        wd500.annual_cooling_mwh > wd500.annual_heating_mwh,
        "New Delhi must be cooling-dominant: cooling {} vs heating {} MWh",
        wd500.annual_cooling_mwh,
        wd500.annual_heating_mwh
    );
}

/// The interchange artifact: one flat CSV row per case, case id first,
/// matching the workflow-results shape #4276 adopts from BESTEST-GSR.
#[test]
fn section6_workflow_csv_shape() {
    let csv = section6_workflow_csv().expect("workflow CSV");
    let mut lines = csv.lines();
    assert!(lines
        .next()
        .unwrap()
        .starts_with("case,description,weather_file"));
    let mut rows = 0;
    for line in lines {
        let first = line.split(',').next().unwrap();
        assert!(
            ["WD100", "WD200", "WD300", "WD400", "WD500", "WD600"].contains(&first),
            "row must key on the case id first, got {first:?}"
        );
        rows += 1;
    }
    assert_eq!(rows, 6);
}

/// Writes the §6 workflow-results CSV into the repo (`results/`) so the
/// #4283 emitter and #4276 conformance table consume a fixed artifact.
/// Ignored: run explicitly with
/// `cargo test --release --test ashrae_140_section6_weather_driver -- --ignored emit_section6_workflow_csv --nocapture`.
#[test]
#[ignore]
fn emit_section6_workflow_csv() {
    let results = harness_results();
    let mut csv = String::from(Section6CaseResult::csv_header());
    csv.push('\n');
    for r in results.iter() {
        csv.push_str(&r.csv_row());
        csv.push('\n');
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results");
    std::fs::create_dir_all(&dir).expect("create results dir");
    let path = dir.join("section6_workflow_results.csv");
    std::fs::write(&path, &csv).expect("write section6 workflow CSV");
    println!("wrote {}", path.display());
    println!("{csv}");
}

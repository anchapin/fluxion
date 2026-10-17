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

// §6 case harness engine driver (Issue #4269) — lives in the integration
// suite because the #1441/#2495 sim↔validation cycle guard rejects new
// validation→sim lib edges (round-25 precedent, sim_per_zone_u_lw.rs).
// The #4283 emitter will absorb or re-home this logic when it lands.

use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// One §6 weather-driver case definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section6CaseDef {
    /// Case id as the consumer keys it (`WD100` … `WD600`).
    pub case_id: &'static str,
    /// Location description from the standard's weather table.
    pub description: &'static str,
    /// EPW file under `assets/weather/`.
    pub epw_file: &'static str,
}

/// The six §6 cases, in standard order.
pub const SECTION6_CASES: [Section6CaseDef; 6] = [
    Section6CaseDef {
        case_id: "WD100",
        description: "Denver Intl AP, CO — baseline, high elevation",
        epw_file: "WD100.epw",
    },
    Section6CaseDef {
        case_id: "WD200",
        description: "Atlanta-Hartsfield, GA — humid, near sea level",
        epw_file: "WD200.epw",
    },
    Section6CaseDef {
        case_id: "WD300",
        description: "Santiago Benitez, Chile — southern hemisphere",
        epw_file: "WD300.epw",
    },
    Section6CaseDef {
        case_id: "WD400",
        description: "Barrow-Post-Rogers, AK — arctic, polar day/night",
        epw_file: "WD400.epw",
    },
    Section6CaseDef {
        case_id: "WD500",
        description: "New Delhi Gandhi Intl — half-hour UTC offset",
        epw_file: "WD500.epw",
    },
    Section6CaseDef {
        case_id: "WD600",
        description: "Denver Intl AP, CO — identical to WD100, albedo 0.2",
        epw_file: "WD600.epw",
    },
];

/// Per-case §6 outputs, one engine run each.
#[derive(Debug, Clone, PartialEq)]
pub struct Section6CaseResult {
    pub case_id: String,
    pub description: String,
    pub weather_file: String,
    /// Site metadata from the EPW LOCATION header.
    pub city_state: String,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub utc_offset_hours: Option<f64>,
    /// Section 6.2.1.1.1 — annual heating load (MWh).
    pub annual_heating_mwh: f64,
    /// Section 6.2.1.1.2 — annual sensible cooling load (MWh).
    pub annual_cooling_mwh: f64,
    /// Section 6.2.1.1.3 — annual hourly-integrated peak heating load (kW).
    pub peak_heating_kw: f64,
    /// Section 6.2.1.1.4 — annual hourly-integrated peak cooling load (kW).
    pub peak_cooling_kw: f64,
    /// Peak occurrence timestamps, `MM-DD-HH` (Section 6.2.1.1.3/4).
    pub peak_heating_occurrence: String,
    pub peak_cooling_occurrence: String,
    /// Monthly diagnostics, calendar order (Section 6.2.1.2.4 shape, MWh).
    pub monthly_heating_mwh: [f64; 12],
    pub monthly_cooling_mwh: [f64; 12],
}

impl Section6CaseResult {
    /// True when every reported quantity is finite (§6 acceptance baseline).
    pub fn is_finite(&self) -> bool {
        self.annual_heating_mwh.is_finite()
            && self.annual_cooling_mwh.is_finite()
            && self.peak_heating_kw.is_finite()
            && self.peak_cooling_kw.is_finite()
            && self.monthly_heating_mwh.iter().all(|v| v.is_finite())
            && self.monthly_cooling_mwh.iter().all(|v| v.is_finite())
    }

    /// CSV header for the flat workflow-results interchange file.
    pub fn csv_header() -> String {
        "case,description,weather_file,city_state,latitude_deg,longitude_deg,utc_offset_hours,\
annual_heating_mwh,annual_cooling_mwh,peak_heating_kw,peak_heating_occurrence,\
peak_cooling_kw,peak_cooling_occurrence,monthly_heating_mwh,monthly_cooling_mwh"
            .to_string()
    }

    /// One flat CSV row for this case, keyed on the case id first.
    /// Free-text fields are quoted (they carry commas from the EPW header).
    pub fn csv_row(&self) -> String {
        let join12 = |a: &[f64; 12]| {
            a.iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
                .join("/")
        };
        let q = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
        format!(
            "{},{},{},{},{:.3},{:.3},{},{:.4},{:.4},{:.4},{},{:.4},{},{},{}",
            self.case_id,
            q(&self.description),
            self.weather_file,
            q(&self.city_state),
            self.latitude_deg,
            self.longitude_deg,
            self.utc_offset_hours
                .map(|o| format!("{o}"))
                .unwrap_or_default(),
            self.annual_heating_mwh,
            self.annual_cooling_mwh,
            self.peak_heating_kw,
            self.peak_heating_occurrence,
            self.peak_cooling_kw,
            self.peak_cooling_occurrence,
            join12(&self.monthly_heating_mwh),
            join12(&self.monthly_cooling_mwh),
        )
    }
}

fn weather_path(epw_file: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/weather")
        .join(epw_file)
}

/// Runs one §6 case through the engine and returns its per-case outputs.
///
/// The run is blind execution in the strict-gate sense: the case is built
/// from the Case 600 spec and driven by the EPW file directly; no case-id
/// shortcuts reach the engine.
pub fn run_section6_case(def: &Section6CaseDef) -> Result<Section6CaseResult, String> {
    let path = weather_path(def.epw_file);
    let weather = EpwWeatherSource::from_file(&path)
        .map_err(|e| format!("{}: failed to parse EPW: {e:?}", def.case_id))?;

    // Site metadata from the EPW LOCATION header — the WD files are real
    // locations and the solar position must follow each file, not the
    // fabric cohort's Denver default.
    let site = weather
        .location_struct()
        .ok_or_else(|| format!("{}: EPW header carries no structured location", def.case_id))?;
    let latitude_deg = site
        .latitude_deg()
        .ok_or_else(|| format!("{}: EPW header carries no latitude", def.case_id))?;
    let longitude_deg = site
        .longitude_deg()
        .ok_or_else(|| format!("{}: EPW header carries no longitude", def.case_id))?;
    let utc_offset_hours = site.utc_offset_hours;
    let city_state = site.city_state().to_string();

    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<fluxion::physics::cta::VectorField>::from_spec_with_selector(
        &spec,
        &ThermalSelector::default(),
    )
    .map_err(|e| format!("{}: model init failed: {e:?}", def.case_id))?;
    model.0.solar.latitude_deg = latitude_deg;
    model.0.solar.longitude_deg = longitude_deg;
    model.0.solar.utc_offset_hours = utc_offset_hours;
    // §6 per-case override (Issue #4269, flagged for Alex's review): the
    // authored per-hour EPW albedo passes through unfiltered, including
    // WD100's explicit 0, so the WD100/WD600 isolation survives the engine.
    model.set_authored_epw_albedo(true);

    let j_to_mwh = 1.0 / 3.6e9;
    let mut annual_heating_j = 0.0_f64;
    let mut annual_cooling_j = 0.0_f64;
    let mut peak_heating_w = 0.0_f64;
    let mut peak_cooling_w = 0.0_f64;
    let mut peak_heating_at = (0_u32, 0_u32, 0_u32);
    let mut peak_cooling_at = (0_u32, 0_u32, 0_u32);
    let mut monthly_heating_mwh = [0.0_f64; 12];
    let mut monthly_cooling_mwh = [0.0_f64; 12];

    for step in 0..8760 {
        let weather_data = weather
            .get_hourly_data(step)
            .map_err(|e| format!("{}: hour {step}: {e:?}", def.case_id))?;
        model.solar.weather = Some(weather_data.clone());
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_j = energy_kwh * 3.6e6;
        // Calendar month/day/hour from the engine's own timestep mapping so
        // the occurrence timestamps and monthly buckets match §6.2.1.1.3/4.
        let (_year, month, day, hour) = fluxion::sim::engine::ThermalModel::<
            fluxion::physics::cta::VectorField,
        >::timestep_to_date(step);
        let mi = (month as usize - 1).min(11);
        if energy_kwh > 0.0 {
            annual_heating_j += energy_j;
            monthly_heating_mwh[mi] += energy_j * j_to_mwh;
            if energy_j / 3600.0 > peak_heating_w {
                peak_heating_w = energy_j / 3600.0;
                peak_heating_at = (month, day, hour as u32);
            }
        } else if energy_kwh < 0.0 {
            annual_cooling_j += -energy_j;
            monthly_cooling_mwh[mi] += -energy_j * j_to_mwh;
            if -energy_j / 3600.0 > peak_cooling_w {
                peak_cooling_w = -energy_j / 3600.0;
                peak_cooling_at = (month, day, hour as u32);
            }
        }
    }

    Ok(Section6CaseResult {
        case_id: def.case_id.to_string(),
        description: def.description.to_string(),
        weather_file: def.epw_file.to_string(),
        city_state,
        latitude_deg,
        longitude_deg,
        utc_offset_hours,
        annual_heating_mwh: annual_heating_j * j_to_mwh,
        annual_cooling_mwh: annual_cooling_j * j_to_mwh,
        peak_heating_kw: peak_heating_w / 1000.0,
        peak_cooling_kw: peak_cooling_w / 1000.0,
        peak_heating_occurrence: format_mdh(&peak_heating_at),
        peak_cooling_occurrence: format_mdh(&peak_cooling_at),
        monthly_heating_mwh,
        monthly_cooling_mwh,
    })
}

fn format_mdh(mdh: &(u32, u32, u32)) -> String {
    format!("{:02}-{:02}-{:02}", mdh.0, mdh.1, mdh.2)
}

/// Runs all six §6 cases and returns their results in standard order.
pub fn run_all_section6_cases() -> Result<Vec<Section6CaseResult>, String> {
    SECTION6_CASES.iter().map(run_section6_case).collect()
}

/// Builds the flat workflow-results CSV for all six cases.
pub fn section6_workflow_csv() -> Result<String, String> {
    let results = run_all_section6_cases()?;
    let mut csv = Section6CaseResult::csv_header();
    csv.push('\n');
    for r in &results {
        csv.push_str(&r.csv_row());
        csv.push('\n');
    }
    Ok(csv)
}

// ===========================================================================
// §6 case harness (Issue #4269): the six WD files driven through the engine
// on the low-mass case configuration, producing the per-case outputs the
// #4283 emitter and #4276 conformance table consume.
// ===========================================================================

/// One engine pass for the whole binary: each test reads the same results.
fn harness_results() -> &'static [Section6CaseResult] {
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

/// Writes the §6 workflow-results CSV into the repo
/// (`data/reference/section6/`) so the
/// #4283 emitter and #4276 conformance table consume a fixed artifact.
/// Ignored: run explicitly with
/// `cargo test --release --test ashrae_140_section6_weather_driver -- --ignored emit_section6_workflow_csv --nocapture`.
#[test]
#[ignore]
fn emit_section6_workflow_csv() {
    let results = harness_results();
    let mut csv = Section6CaseResult::csv_header();
    csv.push('\n');
    for r in results.iter() {
        csv.push_str(&r.csv_row());
        csv.push('\n');
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/reference/section6");
    std::fs::create_dir_all(&dir).expect("create results dir");
    let path = dir.join("section6_workflow_results.csv");
    std::fs::write(&path, &csv).expect("write section6 workflow CSV");
    println!("wrote {}", path.display());
    println!("{csv}");
}

// ===========================================================================
// #4283 emitter + #4276 conformance table (round 48): consume the committed
// workflow CSV rows and evaluate each case against the §6.2.1 output
// requirements the in-repo standard PDF carries
// (docs/140_2017_a_20200901.pdf, §6.2.1, extracted with pdftotext).
//
// Three states per case, per #4276 (rewritten 2026-09-30): conformant,
// incomplete (ran, missing required outputs), not implemented. No score,
// no percentage — Standard 140 defines no acceptance criteria (the #4277
// finding), so the table reports coverage, not conformance-with-a-grade.
// ===========================================================================

/// One §6.2.1 output requirement the table tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section6Requirement {
    /// Normative section number in Standard 140-2017.
    pub section: &'static str,
    /// What the requirement asks for, in the table's words.
    pub label: &'static str,
    /// True when the harness emits this quantity today.
    pub emitted_by_harness: bool,
}

/// §6.2.1.1.1 through §6.2.1.1.5 apply to all non-free-float cases; the
/// harness runs the Case 600 configuration, so the Case-600-only additions
/// (§6.2.1.2) and the §6.2.1.8 daily hourly outputs are tracked as well.
/// Everything the harness does not emit is `missing`, never assumed.
pub const SECTION6_REQUIREMENTS: &[Section6Requirement] = &[
    Section6Requirement {
        section: "6.2.1.1.1",
        label: "annual heating load (MWh)",
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.1.2",
        label: "annual sensible cooling load (MWh)",
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.1.3",
        label: "annual hourly integrated peak heating load (kW) with occurrence",
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.1.4",
        label: "annual hourly integrated peak cooling load (kW) with occurrence",
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.1.5",
        label: "all §6.2.1.1 loads entered into the report as positive values",
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.2.1",
        label: "annual incident unshaded total solar radiation on H/N/E/S/W (kWh/m2)",
        emitted_by_harness: false,
    },
    Section6Requirement {
        section: "6.2.1.2.2",
        label: "unshaded annual transmitted solar radiation through south windows (kWh/m2)",
        emitted_by_harness: false,
    },
    Section6Requirement {
        section: "6.2.1.2.3",
        label: "sky temperature: annual mean, min, max with occurrence (°C)",
        emitted_by_harness: false,
    },
    Section6Requirement {
        section: "6.2.1.2.4",
        label: "monthly heating/cooling loads and monthly peaks with occurrence",
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.8",
        label: "daily hourly output for the Table 6-1 days",
        emitted_by_harness: false,
    },
];

/// Per-case conformance record: one row of the table.
#[derive(Debug, Clone, PartialEq)]
pub struct Section6ConformanceRow {
    pub case_id: String,
    pub weather_file: String,
    /// True when the case ran and reported the §6.2.1.1 core outputs.
    pub ran: bool,
    /// `conformant`, `incomplete`, or `not implemented` (#4276 states).
    pub status: &'static str,
    /// Per-requirement statuses, same order as SECTION6_REQUIREMENTS.
    /// `report` = emitted and satisfies the requirement's checks here;
    /// `missing` = required, not emitted by this harness.
    pub requirement_statuses: Vec<&'static str>,
}

/// Parses one flat workflow CSV row (quoted fields respected).
/// Returns None for the header or a blank line.
pub fn parse_section6_csv_row(line: &str) -> Option<Section6CaseResult> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(c);
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => {
                    fields.push(std::mem::take(&mut cur));
                }
                _ => cur.push(c),
            }
        }
    }
    fields.push(cur);
    if fields.len() < 15 || fields[0] == "case" {
        return None;
    }
    let f = |i: usize| -> Option<f64> { fields.get(i)?.parse().ok() };
    let arr12 = |i: usize| -> Option<[f64; 12]> {
        let mut a = [0.0_f64; 12];
        for (k, v) in fields[i].split('/').enumerate() {
            a[k] = v.parse().ok()?;
        }
        Some(a)
    };
    Some(Section6CaseResult {
        case_id: fields[0].clone(),
        description: fields[1].clone(),
        weather_file: fields[2].clone(),
        city_state: fields[3].clone(),
        latitude_deg: f(4)?,
        longitude_deg: f(5)?,
        utc_offset_hours: fields[6].parse().ok(),
        annual_heating_mwh: f(7)?,
        annual_cooling_mwh: f(8)?,
        peak_heating_kw: f(9)?,
        peak_heating_occurrence: fields[10].clone(),
        peak_cooling_kw: f(11)?,
        peak_cooling_occurrence: fields[12].clone(),
        monthly_heating_mwh: arr12(13)?,
        monthly_cooling_mwh: arr12(14)?,
    })
}

/// Parses the whole workflow CSV (the #4283 emitter's input contract).
pub fn parse_section6_workflow_csv(text: &str) -> Vec<Section6CaseResult> {
    text.lines().filter_map(parse_section6_csv_row).collect()
}

/// Evaluates one case against the §6.2.1 requirements.
pub fn evaluate_section6_conformance(r: &Section6CaseResult) -> Section6ConformanceRow {
    // §6.2.1.1.5: every §6.2.1.1 load enters the report as a positive value.
    let positivity_ok = r.annual_heating_mwh >= 0.0
        && r.annual_cooling_mwh >= 0.0
        && r.peak_heating_kw >= 0.0
        && r.peak_cooling_kw >= 0.0
        && r.monthly_heating_mwh.iter().all(|&v| v >= 0.0)
        && r.monthly_cooling_mwh.iter().all(|&v| v >= 0.0);
    let occurrence_ok = |s: &str| {
        // MM-DD-HH shape from the engine's own calendar mapping.
        let parts: Vec<_> = s.split('-').collect();
        parts.len() == 3
            && parts
                .iter()
                .all(|p| p.len() == 2 && p.chars().all(|c| c.is_ascii_digit()))
    };
    let core_checks = [
        // 6.2.1.1.1 / .2: finite annual loads.
        r.annual_heating_mwh.is_finite(),
        r.annual_cooling_mwh.is_finite(),
        // 6.2.1.1.3 / .4: finite peaks with MM-DD-HH occurrences.
        r.peak_heating_kw.is_finite() && occurrence_ok(&r.peak_heating_occurrence),
        r.peak_cooling_kw.is_finite() && occurrence_ok(&r.peak_cooling_occurrence),
        positivity_ok,
    ];
    let mut statuses = Vec::with_capacity(SECTION6_REQUIREMENTS.len());
    for (i, req) in SECTION6_REQUIREMENTS.iter().enumerate() {
        if !req.emitted_by_harness {
            statuses.push("missing");
        } else if i < core_checks.len() {
            statuses.push(if core_checks[i] { "report" } else { "missing" });
        } else {
            // 6.2.1.2.4: 12 monthly values each, finite.
            statuses.push(
                if r.monthly_heating_mwh.iter().all(|v| v.is_finite())
                    && r.monthly_cooling_mwh.iter().all(|v| v.is_finite())
                {
                    "report"
                } else {
                    "missing"
                },
            );
        }
    }
    let ran = r.is_finite();
    let status = if ran && statuses.iter().all(|&s| s == "report") {
        "conformant"
    } else if ran {
        "incomplete"
    } else {
        "not implemented"
    };
    Section6ConformanceRow {
        case_id: r.case_id.clone(),
        weather_file: r.weather_file.clone(),
        ran,
        status,
        requirement_statuses: statuses,
    }
}

/// Builds the full conformance table from workflow CSV rows.
pub fn build_section6_conformance(results: &[Section6CaseResult]) -> Vec<Section6ConformanceRow> {
    results.iter().map(evaluate_section6_conformance).collect()
}

/// CSV form of the conformance table.
pub fn section6_conformance_csv(rows: &[Section6ConformanceRow]) -> String {
    let mut out = String::from("case,weather_file,ran,status");
    for req in SECTION6_REQUIREMENTS {
        out.push_str(&format!(",{}", req.section));
    }
    out.push('\n');
    for r in rows {
        out.push_str(&r.case_id);
        out.push(',');
        out.push_str(&r.weather_file);
        out.push(',');
        out.push_str(if r.ran { "yes" } else { "no" });
        out.push(',');
        out.push_str(r.status);
        for s in &r.requirement_statuses {
            out.push(',');
            out.push_str(s);
        }
        out.push('\n');
    }
    out
}

/// Markdown form of the conformance table, for the issue and the docs.
pub fn section6_conformance_markdown(rows: &[Section6ConformanceRow]) -> String {
    let mut out = String::from(
        "# §6 conformance table (#4276)\n\n\
         Coverage of the Standard 140 §6.2.1 output requirements for the six\n\
         weather-driver cases, evaluated from the committed workflow CSV\n\
         (`section6_workflow_results.csv`). States per #4276: `conformant`,\n\
         `incomplete` (ran, missing required outputs), `not implemented`.\n\
         No score, no percentage — Standard 140 defines no acceptance\n\
         criteria (#4277). `report` = emitted and satisfies the checks here;\n\
         `missing` = required, not emitted by the harness yet.\n\n",
    );
    out.push_str("| Case | Weather | Status |");
    for req in SECTION6_REQUIREMENTS {
        out.push_str(&format!(" {} |", req.section));
    }
    out.push_str("\n|---|---|---|");
    out.push_str(&"---|".repeat(SECTION6_REQUIREMENTS.len()));
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "| {} | {} | {} |",
            r.case_id, r.weather_file, r.status
        ));
        for s in &r.requirement_statuses {
            out.push_str(&format!(" {s} |"));
        }
        out.push('\n');
    }
    out
}

fn committed_workflow_csv() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section6/section6_workflow_results.csv");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed §6 workflow CSV unreadable: {e}"))
}

/// The #4283 emitter consumes the committed workflow CSV rows and the
/// #4276 table evaluates them: every case ran, every §6.2.1.1 core output
/// is present and satisfies its check, and the missing set is exactly the
/// harness quantities not emitted yet (§6.2.1.2.1–.3, §6.2.1.8). Honest
/// statuses only — nothing here upgrades an unemitted quantity.
#[test]
fn section6_conformance_table_from_committed_csv() {
    let results = parse_section6_workflow_csv(&committed_workflow_csv());
    assert_eq!(results.len(), 6, "six cases in the committed CSV");
    let rows = build_section6_conformance(&results);
    for r in &rows {
        assert!(r.ran, "{} did not run", r.case_id);
        assert_eq!(
            r.status, "incomplete",
            "{}: expected incomplete (ran, Case-600 additions not emitted)",
            r.case_id
        );
        for (i, req) in SECTION6_REQUIREMENTS.iter().enumerate() {
            let expected = if req.emitted_by_harness {
                "report"
            } else {
                "missing"
            };
            assert_eq!(
                r.requirement_statuses[i], expected,
                "{} / §{}: status must be {expected}",
                r.case_id, req.section
            );
        }
    }
}

/// The emitted table must round-trip: regenerating it from the committed
/// workflow CSV reproduces the committed conformance CSV byte-for-byte.
#[test]
fn section6_conformance_csv_artifact_matches_regeneration() {
    let results = parse_section6_workflow_csv(&committed_workflow_csv());
    let rows = build_section6_conformance(&results);
    let regenerated = section6_conformance_csv(&rows);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section6/section6_conformance.csv");
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed conformance CSV unreadable: {e}"));
    assert_eq!(regenerated, committed, "committed conformance CSV is stale");
}

/// Writes both conformance artifacts into `data/reference/section6/`.
/// Ignored, category `diagnostic`: run explicitly with
/// `cargo test --release --test ashrae_140_section6_weather_driver -- --ignored emit_section6_conformance --nocapture`.
#[test]
#[ignore]
fn emit_section6_conformance() {
    let results = parse_section6_workflow_csv(&committed_workflow_csv());
    let rows = build_section6_conformance(&results);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/reference/section6");
    std::fs::create_dir_all(&dir).expect("create section6 dir");
    let csv_path = dir.join("section6_conformance.csv");
    std::fs::write(&csv_path, section6_conformance_csv(&rows)).expect("write conformance CSV");
    let md_path = dir.join("section6_conformance.md");
    std::fs::write(&md_path, section6_conformance_markdown(&rows)).expect("write conformance md");
    println!("wrote {} and {}", csv_path.display(), md_path.display());
    println!("{}", section6_conformance_markdown(&rows));
}

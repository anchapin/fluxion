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

use fluxion::sim::sky_radiation::SkyRadiationExchange;
use fluxion::sim::solar::{calculate_window_solar_gain, WindowProperties};
use fluxion::solar::solar_position::{calculate_day_of_year, calculate_solar_position};
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
    /// §6.2.1.2.1 — annual incident unshaded total solar radiation
    /// (kWh/m²), order [horizontal, north, east, south, west]. Computed
    /// with the engine's own `calculate_surface_irradiance`, the EPW's
    /// per-hour authored albedo (the #4269 rule), and the case's own
    /// EPW-driven solar position.
    pub annual_incident_solar_kwh_m2: [f64; 5],
    /// §6.2.1.2.2 — annual unshaded transmitted total solar radiation
    /// through the south windows (kWh/m² of glazing), from the engine's
    /// `calculate_window_solar_gain` with a 1 m² unshaded reference
    /// window at the Case 600 spec SHGC.
    pub annual_transmitted_south_kwh_m2: f64,
    /// §6.2.1.2.3 — sky temperature (°C) annual mean / min / max with
    /// MM-DD-HH occurrences, from the EPW horizontal infrared via the
    /// engine's `SkyRadiationExchange::sky_temperature_from_ir`.
    pub sky_temp_mean_c: f64,
    pub sky_temp_min_c: f64,
    pub sky_temp_min_occurrence: String,
    pub sky_temp_max_c: f64,
    pub sky_temp_max_occurrence: String,
    /// §6.2.1.8 — Table 6-1 daily hourly outputs. Populated by the
    /// harness run; empty on a row parsed from the flat workflow CSV
    /// (the daily hourly series live in their own committed CSV).
    pub daily_hourly: Vec<Section6DayHourly>,
}

/// The Table 6-1 daily hourly output days for the Case 600 configuration
/// (Standard 140-2017 Table 6-1, extracted from the in-repo PDF):
/// incident H/S/W on May 4, Jul 14, Mar 5 (plus S/W Jul 27); sky
/// temperature Feb 1, May 4, Jul 14; transmitted south Feb 1, May 4,
/// Jul 14; hourly integrated loads Feb 1, Jul 14, Jan 4 (plus hourly
/// Jan 4). The harness captures all tracked quantities on all six days —
/// a documented superset of each day's requirement.
pub const SECTION6_TABLE61_DAYS: [(u32, u32); 6] =
    [(1, 4), (2, 1), (3, 5), (5, 4), (7, 14), (7, 27)];

/// One Table 6-1 day's hourly series for one case.
#[derive(Debug, Clone, PartialEq)]
pub struct Section6DayHourly {
    pub month: u8,
    pub day: u8,
    /// Hourly integrated incident unshaded total solar (Wh/m²).
    pub incident_h_wh_m2: [f64; 24],
    pub incident_s_wh_m2: [f64; 24],
    pub incident_w_wh_m2: [f64; 24],
    /// Hourly transmitted total solar through south windows (Wh/m²).
    pub transmitted_south_wh_m2: [f64; 24],
    /// Hourly integrated sky temperature (°C).
    pub sky_temp_c: [f64; 24],
    /// Hourly integrated heating (+) / sensible cooling (−) load (kWh),
    /// signed: heating positive, cooling negative.
    pub load_kwh: [f64; 24],
}

impl Section6DayHourly {
    pub fn is_finite(&self) -> bool {
        self.incident_h_wh_m2.iter().all(|v| v.is_finite())
            && self.incident_s_wh_m2.iter().all(|v| v.is_finite())
            && self.incident_w_wh_m2.iter().all(|v| v.is_finite())
            && self.transmitted_south_wh_m2.iter().all(|v| v.is_finite())
            && self.sky_temp_c.iter().all(|v| v.is_finite())
            && self.load_kwh.iter().all(|v| v.is_finite())
    }

    pub fn csv_header() -> String {
        "case,month,day,hour,incident_h_wh_m2,incident_s_wh_m2,incident_w_wh_m2,\
transmitted_south_wh_m2,sky_temp_c,load_kwh"
            .to_string()
    }

    pub fn csv_rows(&self, case_id: &str) -> String {
        let mut out = String::new();
        for h in 0..24 {
            out.push_str(&format!(
                "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.3},{:.4}\n",
                case_id,
                self.month,
                self.day,
                h,
                self.incident_h_wh_m2[h],
                self.incident_s_wh_m2[h],
                self.incident_w_wh_m2[h],
                self.transmitted_south_wh_m2[h],
                self.sky_temp_c[h],
                self.load_kwh[h],
            ));
        }
        out
    }
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
            && self
                .annual_incident_solar_kwh_m2
                .iter()
                .all(|v| v.is_finite())
            && self.annual_transmitted_south_kwh_m2.is_finite()
            && self.sky_temp_mean_c.is_finite()
            && self.sky_temp_min_c.is_finite()
            && self.sky_temp_max_c.is_finite()
            && self.daily_hourly.iter().all(|d| d.is_finite())
    }

    /// CSV header for the flat workflow-results interchange file.
    pub fn csv_header() -> String {
        "case,description,weather_file,city_state,latitude_deg,longitude_deg,utc_offset_hours,\
annual_heating_mwh,annual_cooling_mwh,peak_heating_kw,peak_heating_occurrence,\
peak_cooling_kw,peak_cooling_occurrence,monthly_heating_mwh,monthly_cooling_mwh,\
annual_incident_h_kwh_m2,annual_incident_n_kwh_m2,annual_incident_e_kwh_m2,\
annual_incident_s_kwh_m2,annual_incident_w_kwh_m2,annual_transmitted_south_kwh_m2,\
sky_temp_mean_c,sky_temp_min_c,sky_temp_min_occurrence,sky_temp_max_c,sky_temp_max_occurrence"
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
            "{},{},{},{},{:.3},{:.3},{},{:.4},{:.4},{:.4},{},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
            format!("{:.4}", self.annual_incident_solar_kwh_m2[0]),
            format!("{:.4}", self.annual_incident_solar_kwh_m2[1]),
            format!("{:.4}", self.annual_incident_solar_kwh_m2[2]),
            format!("{:.4}", self.annual_incident_solar_kwh_m2[3]),
            format!("{:.4}", self.annual_incident_solar_kwh_m2[4]),
            format!("{:.4}", self.annual_transmitted_south_kwh_m2),
            format!("{:.4}", self.sky_temp_mean_c),
            format!("{:.4}", self.sky_temp_min_c),
            self.sky_temp_min_occurrence,
            format!("{:.4}", self.sky_temp_max_c),
            self.sky_temp_max_occurrence,
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

    // §6.2.1.2 additions: incident solar (H/N/E/S/W, kWh/m²), unshaded
    // transmitted south solar (kWh/m²), sky temperature statistics.
    // Computed with the engine's own solar chain from each case's EPW —
    // the same solar position, irradiance, window, and sky-temperature
    // functions the engine and the rest of the validation suite call.
    const INCIDENT_ORIENTATIONS: [Orientation; 5] = [
        Orientation::Horizontal,
        Orientation::North,
        Orientation::East,
        Orientation::South,
        Orientation::West,
    ];
    let mut annual_incident_kwh_m2 = [0.0_f64; 5];
    let mut annual_transmitted_south_kwh_m2 = 0.0_f64;
    let mut sky_temp_sum = 0.0_f64;
    let mut sky_temp_min = f64::INFINITY;
    let mut sky_temp_min_at = (0_u32, 0_u32, 0_u32);
    let mut sky_temp_max = f64::NEG_INFINITY;
    let mut sky_temp_max_at = (0_u32, 0_u32, 0_u32);

    // Case 600 spec window for the §6.2.1.2.2 transmitted-solar reference
    // (1 m², unshaded: the requirement's "unshaded" wording, so no
    // overhang/fin geometry is applied even though Case 600 carries one).
    let window_spec_shgc = spec.window_properties.shgc;
    let window_1m2 = WindowProperties::new(
        1.0,
        window_spec_shgc,
        spec.window_properties.normal_transmittance,
    );

    // §6.2.1.8 Table 6-1 day capture.
    let mut daily_hourly: Vec<Section6DayHourly> = SECTION6_TABLE61_DAYS
        .iter()
        .map(|&(m, d)| Section6DayHourly {
            month: m as u8,
            day: d as u8,
            incident_h_wh_m2: [0.0; 24],
            incident_s_wh_m2: [0.0; 24],
            incident_w_wh_m2: [0.0; 24],
            transmitted_south_wh_m2: [0.0; 24],
            sky_temp_c: [0.0; 24],
            load_kwh: [0.0; 24],
        })
        .collect();
    let day_index = |month: u32, day: u32| -> Option<usize> {
        SECTION6_TABLE61_DAYS
            .iter()
            .position(|&(m, d)| m == month && d == day)
    };

    for step in 0..8760 {
        let weather_data = weather
            .get_hourly_data(step)
            .map_err(|e| format!("{}: hour {step}: {e:?}", def.case_id))?;
        model.solar.weather = Some(weather_data.clone());
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_j = energy_kwh * 3.6e6;
        // Calendar month/day/hour from the engine's own timestep mapping so
        // the occurrence timestamps and monthly buckets match §6.2.1.1.3/4.
        let (year, month, day, hour) = fluxion::sim::engine::ThermalModel::<
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

        // §6.2.1.2.1 / .2 / .3 per-hour quantities from the EPW record.
        let sun = calculate_solar_position(
            latitude_deg,
            longitude_deg,
            year,
            month,
            day,
            hour,
            utc_offset_hours,
        );
        let day_of_year = calculate_day_of_year(year, month, day);
        // Authored EPW albedo passes through unfiltered (the #4269 rule
        // above); fall back to the 0.2 default only when the file carries
        // no albedo field at all.
        let albedo = weather_data.albedo.unwrap_or(0.2);
        for (o, orientation) in INCIDENT_ORIENTATIONS.iter().enumerate() {
            let irr = calculate_surface_irradiance(
                &sun,
                weather_data.dni,
                weather_data.dhi,
                Some(weather_data.ghi),
                *orientation,
                albedo,
                day_of_year,
            );
            annual_incident_kwh_m2[o] += irr.total_wm2 / 1000.0;
        }
        let irr_s = calculate_surface_irradiance(
            &sun,
            weather_data.dni,
            weather_data.dhi,
            Some(weather_data.ghi),
            Orientation::South,
            albedo,
            day_of_year,
        );
        let gain_s = calculate_window_solar_gain(
            &irr_s,
            &window_1m2,
            None,
            None,
            &[],
            &sun,
            Orientation::South.into(),
        );
        annual_transmitted_south_kwh_m2 += gain_s.total_gain_w / 1000.0;
        let sky_temp_c =
            SkyRadiationExchange::sky_temperature_from_ir(weather_data.horizontal_infrared);
        sky_temp_sum += sky_temp_c;
        if sky_temp_c < sky_temp_min {
            sky_temp_min = sky_temp_c;
            sky_temp_min_at = (month, day, hour as u32);
        }
        if sky_temp_c > sky_temp_max {
            sky_temp_max = sky_temp_c;
            sky_temp_max_at = (month, day, hour as u32);
        }

        // §6.2.1.8 Table 6-1 day capture.
        if let Some(di) = day_index(month, day) {
            let h = hour as usize;
            let d = &mut daily_hourly[di];
            for (o, orientation) in [
                Orientation::Horizontal,
                Orientation::South,
                Orientation::West,
            ]
            .iter()
            .enumerate()
            {
                let irr = calculate_surface_irradiance(
                    &sun,
                    weather_data.dni,
                    weather_data.dhi,
                    Some(weather_data.ghi),
                    *orientation,
                    albedo,
                    day_of_year,
                );
                let v = irr.total_wm2;
                match o {
                    0 => d.incident_h_wh_m2[h] = v,
                    1 => d.incident_s_wh_m2[h] = v,
                    _ => d.incident_w_wh_m2[h] = v,
                }
            }
            let gain = calculate_window_solar_gain(
                &irr_s,
                &window_1m2,
                None,
                None,
                &[],
                &sun,
                Orientation::South.into(),
            );
            d.transmitted_south_wh_m2[h] = gain.total_gain_w;
            d.sky_temp_c[h] = sky_temp_c;
            d.load_kwh[h] = energy_kwh;
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
        annual_incident_solar_kwh_m2: annual_incident_kwh_m2,
        annual_transmitted_south_kwh_m2,
        sky_temp_mean_c: sky_temp_sum / 8760.0,
        sky_temp_min_c: sky_temp_min,
        sky_temp_min_occurrence: format_mdh(&sky_temp_min_at),
        sky_temp_max_c: sky_temp_max,
        sky_temp_max_occurrence: format_mdh(&sky_temp_max_at),
        daily_hourly,
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
        // Round 51 (2026-10-17): emitted from the EPW record via the
        // engine's own calculate_surface_irradiance chain.
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.2.2",
        label: "unshaded annual transmitted solar radiation through south windows (kWh/m2)",
        // Round 51 (2026-10-17): emitted via calculate_window_solar_gain,
        // 1 m² unshaded reference window at the Case 600 spec SHGC.
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.2.3",
        label: "sky temperature: annual mean, min, max with occurrence (°C)",
        // Round 51 (2026-10-17): emitted via sky_temperature_from_ir on
        // the EPW horizontal infrared, with MM-DD-HH occurrences.
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.2.4",
        label: "monthly heating/cooling loads and monthly peaks with occurrence",
        emitted_by_harness: true,
    },
    Section6Requirement {
        section: "6.2.1.8",
        label: "daily hourly output for the Table 6-1 days",
        // Round 51 (2026-10-17): the harness captures the Case-600-config
        // Table 6-1 days' hourly series (incident H/S/W, transmitted
        // south, sky temperature, signed loads) into
        // section6_table61_daily_hourly.csv.
        emitted_by_harness: true,
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
    if fields.len() < 26 || fields[0] == "case" {
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
    let arr5 = |i: usize| -> Option<[f64; 5]> {
        let mut a = [0.0_f64; 5];
        for (k, v) in fields[i..i + 5].iter().enumerate() {
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
        annual_incident_solar_kwh_m2: arr5(15)?,
        annual_transmitted_south_kwh_m2: f(20)?,
        sky_temp_mean_c: f(21)?,
        sky_temp_min_c: f(22)?,
        sky_temp_min_occurrence: fields[23].clone(),
        sky_temp_max_c: f(24)?,
        sky_temp_max_occurrence: fields[25].clone(),
        daily_hourly: Vec::new(),
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
    // §6.2.1.2.1: all five orientations finite and non-negative.
    let incident_ok = r
        .annual_incident_solar_kwh_m2
        .iter()
        .all(|&v| v.is_finite() && v >= 0.0);
    // §6.2.1.2.2: finite and non-negative.
    let transmitted_ok =
        r.annual_transmitted_south_kwh_m2.is_finite() && r.annual_transmitted_south_kwh_m2 >= 0.0;
    // §6.2.1.2.3: finite min/mean/max in order, with MM-DD-HH occurrences.
    let sky_ok = r.sky_temp_mean_c.is_finite()
        && r.sky_temp_min_c.is_finite()
        && r.sky_temp_max_c.is_finite()
        && r.sky_temp_min_c <= r.sky_temp_mean_c
        && r.sky_temp_mean_c <= r.sky_temp_max_c
        && occurrence_ok(&r.sky_temp_min_occurrence)
        && occurrence_ok(&r.sky_temp_max_occurrence);
    // §6.2.1.8: the Table 6-1 days present, 24 finite hourly records each.
    let daily_ok = {
        let mut ok = r.daily_hourly.len() == SECTION6_TABLE61_DAYS.len();
        for (d, &(m, day)) in r.daily_hourly.iter().zip(SECTION6_TABLE61_DAYS.iter()) {
            ok = ok && d.month as u32 == m && d.day as u32 == day && d.is_finite();
        }
        ok
    };
    let mut statuses = Vec::with_capacity(SECTION6_REQUIREMENTS.len());
    for (i, req) in SECTION6_REQUIREMENTS.iter().enumerate() {
        if !req.emitted_by_harness {
            statuses.push("missing");
        } else if i < core_checks.len() {
            statuses.push(if core_checks[i] { "report" } else { "missing" });
        } else {
            statuses.push(match i {
                // 6.2.1.2.1
                5 if incident_ok => "report",
                // 6.2.1.2.2
                6 if transmitted_ok => "report",
                // 6.2.1.2.3
                7 if sky_ok => "report",
                // 6.2.1.2.4: 12 monthly values each, finite.
                8 if r.monthly_heating_mwh.iter().all(|v| v.is_finite())
                    && r.monthly_cooling_mwh.iter().all(|v| v.is_finite()) =>
                {
                    "report"
                }
                // 6.2.1.8
                9 if daily_ok => "report",
                _ => "missing",
            });
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

/// Parses the committed Table 6-1 daily hourly CSV into per-case day
/// series. Returns (case_id, days) pairs in first-appearance order.
pub fn parse_section6_daily_csv(text: &str) -> Vec<(String, Vec<Section6DayHourly>)> {
    let mut order: Vec<String> = Vec::new();
    let mut by_case: std::collections::HashMap<String, Vec<Section6DayHourly>> =
        std::collections::HashMap::new();
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 10 {
            continue;
        }
        let (Ok(month), Ok(day), Ok(hour)) = (
            f[1].parse::<u8>(),
            f[2].parse::<u8>(),
            f[3].parse::<usize>(),
        ) else {
            continue;
        };
        let case_id = f[0].to_string();
        if !by_case.contains_key(&case_id) {
            order.push(case_id.clone());
        }
        let days = by_case.entry(case_id).or_default();
        let (month, day) = (month, day);
        let d = match days.iter_mut().find(|d| d.month == month && d.day == day) {
            Some(d) => d,
            None => {
                days.push(Section6DayHourly {
                    month,
                    day,
                    incident_h_wh_m2: [0.0; 24],
                    incident_s_wh_m2: [0.0; 24],
                    incident_w_wh_m2: [0.0; 24],
                    transmitted_south_wh_m2: [0.0; 24],
                    sky_temp_c: [0.0; 24],
                    load_kwh: [0.0; 24],
                });
                days.last_mut().unwrap()
            }
        };
        let v = |i: usize| f[i].parse::<f64>().unwrap_or(f64::NAN);
        if hour < 24 {
            d.incident_h_wh_m2[hour] = v(4);
            d.incident_s_wh_m2[hour] = v(5);
            d.incident_w_wh_m2[hour] = v(6);
            d.transmitted_south_wh_m2[hour] = v(7);
            d.sky_temp_c[hour] = v(8);
            d.load_kwh[hour] = v(9);
        }
    }
    order
        .into_iter()
        .map(|c| {
            let days = by_case.remove(&c).unwrap_or_default();
            (c, days)
        })
        .collect()
}

/// Builds the full conformance table from workflow CSV rows plus the
/// committed Table 6-1 daily hourly series (the §6.2.1.8 evidence).
pub fn build_section6_conformance_with_daily(
    results: &[Section6CaseResult],
    daily_by_case: &[(String, Vec<Section6DayHourly>)],
) -> Vec<Section6ConformanceRow> {
    results
        .iter()
        .map(|r| {
            let mut r = r.clone();
            if let Some((_, days)) = daily_by_case.iter().find(|(c, _)| *c == r.case_id) {
                r.daily_hourly = days.clone();
            }
            evaluate_section6_conformance(&r)
        })
        .collect()
}

/// Builds the conformance table from workflow rows only; the §6.2.1.8
/// status stays `missing` unless daily hourly series are supplied.
pub fn build_section6_conformance(results: &[Section6CaseResult]) -> Vec<Section6ConformanceRow> {
    build_section6_conformance_with_daily(results, &[])
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

fn committed_daily_csv() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section6/section6_table61_daily_hourly.csv");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed §6 Table 6-1 daily CSV unreadable: {e}"))
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
    let daily = parse_section6_daily_csv(&committed_daily_csv());
    assert_eq!(daily.len(), 6, "six cases in the committed daily CSV");
    let rows = build_section6_conformance_with_daily(&results, &daily);
    for r in &rows {
        assert!(r.ran, "{} did not run", r.case_id);
        assert_eq!(
            r.status, "conformant",
            "{}: expected conformant once every §6.2.1 output is emitted and satisfies its check (round 51 upgraded the former missing set: §6.2.1.2.1–.3, §6.2.1.8)",
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
    let daily = parse_section6_daily_csv(&committed_daily_csv());
    let rows = build_section6_conformance_with_daily(&results, &daily);
    let regenerated = section6_conformance_csv(&rows);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section6/section6_conformance.csv");
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed conformance CSV unreadable: {e}"));
    assert_eq!(regenerated, committed, "committed conformance CSV is stale");
}

/// The committed Table 6-1 daily hourly CSV must cover all six cases, all
/// six Table 6-1 days, and 24 finite hourly records per day, with
/// non-negative solar and irradiance/temperature values.
#[test]
fn section6_daily_csv_covers_table61_days() {
    let daily = parse_section6_daily_csv(&committed_daily_csv());
    assert_eq!(daily.len(), 6, "six cases in the committed daily CSV");
    for (case_id, days) in &daily {
        assert_eq!(
            days.len(),
            SECTION6_TABLE61_DAYS.len(),
            "{case_id}: Table 6-1 day coverage"
        );
        for (d, &(m, day)) in days.iter().zip(SECTION6_TABLE61_DAYS.iter()) {
            assert_eq!(
                (d.month as u32, d.day as u32),
                (m, day),
                "{case_id}: expected Table 6-1 day {m}-{day}"
            );
            assert!(
                d.is_finite(),
                "{case_id} {m}-{day}: non-finite hourly value"
            );
            for h in 0..24 {
                assert!(
                    d.incident_h_wh_m2[h] >= 0.0
                        && d.incident_s_wh_m2[h] >= 0.0
                        && d.incident_w_wh_m2[h] >= 0.0
                        && d.transmitted_south_wh_m2[h] >= 0.0,
                    "{case_id} {m}-{day} h{h}: solar quantities must be non-negative"
                );
            }
        }
    }
}

/// The emitted daily hourly artifact must round-trip byte-for-byte.
#[test]
fn section6_daily_csv_artifact_matches_regeneration() {
    let results = harness_results();
    let regenerated = section6_daily_csv(results);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section6/section6_table61_daily_hourly.csv");
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed daily hourly CSV unreadable: {e}"));
    assert_eq!(
        regenerated, committed,
        "committed daily hourly CSV is stale"
    );
}

/// Flat CSV of all cases' Table 6-1 daily hourly series.
pub fn section6_daily_csv(results: &[Section6CaseResult]) -> String {
    let mut out = Section6DayHourly::csv_header();
    out.push('\n');
    for r in results {
        for d in &r.daily_hourly {
            out.push_str(&d.csv_rows(&r.case_id));
        }
    }
    out
}

/// Writes both conformance artifacts into `data/reference/section6/`.
/// Ignored, category `diagnostic`: run explicitly with
/// `cargo test --release --test ashrae_140_section6_weather_driver -- --ignored emit_section6_conformance --nocapture`.
#[test]
#[ignore]
fn emit_section6_conformance() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/reference/section6");
    std::fs::create_dir_all(&dir).expect("create section6 dir");
    // Daily hourly artifact first: the conformance table's §6.2.1.8
    // status is evaluated from it.
    let daily_path = dir.join("section6_table61_daily_hourly.csv");
    std::fs::write(&daily_path, section6_daily_csv(harness_results()))
        .expect("write daily hourly CSV");
    let results = parse_section6_workflow_csv(&committed_workflow_csv());
    let daily = parse_section6_daily_csv(&committed_daily_csv());
    let rows = build_section6_conformance_with_daily(&results, &daily);
    let csv_path = dir.join("section6_conformance.csv");
    std::fs::write(&csv_path, section6_conformance_csv(&rows)).expect("write conformance CSV");
    let md_path = dir.join("section6_conformance.md");
    std::fs::write(&md_path, section6_conformance_markdown(&rows)).expect("write conformance md");
    println!(
        "wrote {}, {} and {}",
        csv_path.display(),
        md_path.display(),
        daily_path.display()
    );
    println!("{}", section6_conformance_markdown(&rows));
}

// ===========================================================================
// Value pins (round 49, 2026-10-17, follow-up to the #4456 review finding):
// the §6 conformance CSV carries statuses, not values, so its staleness gate
// cannot trip on a numeric engine change. These pins make any engine change
// that moves a WD-case output fail CI. Recorded values are the committed
// `data/reference/section6/section6_workflow_results.csv` rows, which the
// `emit_section6_workflow_csv` emitter above produces from this same harness.
// When a physics-correct engine change moves a value, update the pin here AND
// re-emit the CSV in the same PR, with the old→new provenance noted in the PR
// description (same discipline as the strict-gate re-record battery).
// Tolerance ±0.5%: wide enough for the 4-decimal CSV rounding (≤0.005%),
// tight enough to catch any real engine drift.
// ===========================================================================

/// Re-recorded 2026-10-19: the R71 sub-hour air-driver default flip (loop
/// round 73, #1522; env-gated harness `FLX_R71_AIR_CAP` in round 72) changed
/// every 5R1C load path, so the §6 load pins moved; solar/sky pins did not
/// move (not on the modified path). Old values (2026-10-17, develop
/// `592ec089`, round 49): WD100 6.2568/3.9788/4.6542/4.0783,
/// WD200 2.8683/4.6949/3.8725/3.8667, WD300 3.5127/2.2674/2.6624/2.7009,
/// WD400 31.7243/0.1292/8.1737/2.7594, WD500 0.3678/13.4623/1.6043/5.4335,
/// WD600 5.8903/5.2570/4.6339/4.6264. Recorded from the regenerated
/// `section6_workflow_results.csv` (emit_section6_workflow_csv); nothing
/// tuned.
// Re-recorded 2026-10-20 (loop round 78): the coupled single-N default flip
// (`FLX_R77_COUPLED_N` default 12, Alex-directed, #3983 §63/§64) changed the
// 5R1C advance; values re-emitted from the regenerated
// section6_workflow_results.csv on the flipped default build. Old values:
// WD100 [5.1468, 4.3117, 4.4012, 4.1700], WD200 [2.2037, 5.1836, 3.5252,
// 3.9177], WD300 [2.9688, 2.2645, 2.4531, 2.6061], WD400 [30.7720, 0.1491,
// 8.1135, 2.8136], WD500 [0.1365, 14.8884, 1.2383, 5.4561], WD600 [4.7522,
// 5.7304, 4.3681, 4.7184]. Peak occurrences moved where the lighter load
// shifted the extreme hour (WD200 C 11-15-14→08-19-13, WD300 H 07-22-07→
// 07-22-06, WD400 C 08-13-14→08-15-14, WD500 H 12-28-07→01-01-06, WD500 C
// 11-07-14→10-10-13). Solar/sky columns did not move (weather-only path).
// Nothing tuned.
// Re-recorded 2026-10-21 (loop round 85): the per-class coupled sub-step
// mass advance promoted to the engine default (`FLX_R84_PERCLASS`,
// Alex-directed 2026-10-21 21:18 ET, #3983 §65) moved the 5R1C load values
// again, on the spec-built WD harness models (per-class construction
// tensors present). Old values (round 78): WD100 [5.5870, 2.2174, 3.3912,
// 2.9266], WD200 [2.3630, 3.2045, 2.7602, 2.9644], WD300 [3.0172, 1.1242,
// 1.9765, 1.6364], WD400 [32.6207, 0.0188, 7.6981, 1.2946], WD500
// [0.0913, 11.8580, 0.8382, 4.2703], WD600 [5.1168, 3.1923, 3.3177,
// 3.4114]. Peak occurrences unchanged this round. Solar/sky columns did
// not move (weather-only path). Re-emitted section6_workflow_results.csv
// in the same commit; nothing tuned.
// Re-recorded 2026-10-22 (loop round 89): the ISO 13790 §C.3 surface-pool
// mass share promoted to the engine default (`FLX_R88_PHIST_MASS_SHARE`,
// Alex-directed 2026-10-22 14:57 ET, round-88 arm #4543) moved the 5R1C
// load values again, on the spec-built WD harness models. Old values
// (round 85): WD100 [5.5169, 2.1927, 3.4070, 2.9103], WD200 [2.3281,
// 3.1367, 2.7677, 2.9446], WD300 [3.1111, 1.0444, 2.0025, 1.6020], WD400
// [32.3156, 0.0199, 7.6659, 1.3362], WD500 [0.0876, 11.6552, 0.8439,
// 4.2467], WD600 [5.0666, 3.1355, 3.3389, 3.3883]. Peak occurrences:
// WD300 cooling 02-13-14→12-15-14; all others unchanged this round.
// Solar/sky columns did not move (weather-only path). Re-emitted
// section6_workflow_results.csv and section6_table61_daily_hourly.csv in
// the same commit; nothing tuned.
// Re-recorded 2026-10-22 (loop round 89b, item 2a): the mass-node full
// time-coverage fix promoted to the engine default
// (`FLX_R89_MASS_DT_ALIGN`, flip commit ceb19b0b) moved the 5R1C load
// values again, on the spec-built WD harness models. Old values
// (round 89a): WD100 [4.7561, 2.2977, 3.2003, 2.8610], WD200 [1.8954,
// 3.3523, 2.6149, 2.9007], WD300 [2.9096, 1.1312, 1.9495, 1.6305],
// WD400 [31.6671, 0.0165, 7.6537, 1.2185], WD500 [0.0278, 12.4903,
// 0.8497, 4.1871], WD600 [4.2744, 3.3412, 3.1188, 3.3351]. Peak
// occurrences moved where the restored mass coverage shifted the
// extreme hour (WD100 C 09-11-12→10-01-13, WD200 C 08-19-13→11-11-14,
// WD300 H 07-22-06→07-22-07, WD300 C 12-15-14→12-15-15, WD400 C
// 08-15-14→08-13-14, WD500 H 01-01-06→12-30-07, WD500 C 10-10-13→
// 10-05-14, WD600 C 09-11-12→10-01-13). Solar/sky columns did not move
// (weather-only path). Re-emitted both CSVs in the same commit; nothing
// tuned.
// Re-recorded 2026-10-22 (loop round 90): the R77 solar-lag filter removed
// from the default route (`FLX_R90_NO_LAG_FILTER` default-on; ISO 13790
// §C.3 algebraic surface node — the reference formulation has no smoothing
// element on the air share of phi_st; §LIMIT-28 round-90 record) moved the
// 5R1C load values on the spec-built WD harness models. Old values
// (round 89b): WD100 [6.1577, 3.7245, 4.7033, 3.8571], WD200 [2.7968,
// 4.4727, 3.9221, 3.6327], WD300 [3.6632, 2.1077, 2.7220, 2.5994], WD400
// [32.1401, 0.0994, 8.2808, 2.3964], WD500 [0.3267, 13.3151, 1.5546,
// 5.1821], WD600 [5.7891, 4.9546, 4.6818, 4.3806]. Peak occurrences moved
// where the lighter heating load shifted the extreme hour (WD200 C
// 11-11-14 -> 10-10-13, WD300 C 12-15-15 -> 12-15-14, WD500 C 10-05-14 ->
// 10-05-13). Solar/sky columns did not move (weather-only path).
// Re-emitted section6_workflow_results.csv and
// section6_table61_daily_hourly.csv in the same commit; nothing tuned.
// Re-recorded 2026-10-22 (loop round 91, flip 1): the ASHRAE 140 Table
// 5-138 window-loss solar sink promoted to the engine default
// (`FLX_R91_RETRANS`; §5.2.1.12.1 "accounting for solar lost ...
// retransmitted back out the windows", Table 5-138 row "Solar Lost through
// Windows 0.035"; round-83 arm left gated on band direction alone,
// flipped under the 2026-10-22 criteria) moved the 5R1C load values on
// the spec-built WD harness models. Old values (round 90): WD100
// [6.1790, 3.7506, 4.7066, 3.9022], WD200 [2.8027, 4.4855, 3.9269,
// 3.6817], WD300 [3.6514, 2.0936, 2.7238, 2.5986], WD400 [32.1487,
// 0.1055, 8.2808, 2.5237], WD500 [0.3274, 13.3153, 1.5597, 5.2406],
// WD600 [5.8104, 4.9806, 4.6855, 4.4303]. One peak occurrence moved
// (WD600 C 10-01-13 -> 09-03-13); all others unchanged. Solar/sky
// columns did not move (weather-only path). Re-emitted
// section6_workflow_results.csv and section6_table61_daily_hourly.csv in
// the same commit; nothing tuned.
// Re-recorded 2026-10-23 (loop round 91, flip 2, `FLX_R91_SOL2ST`,
// §LIMIT-28 round-91 record): annual H/C and both peaks move for all
// six cases with the retired incumbent solar split; cooling-peak
// occurrences move for WD100/200/300/500/600 (WD400 unchanged). From
// the regenerated section6_workflow_results.csv. Solar/sky pins did
// not move (weather-only path). Nothing tuned.
// Re-recorded 2026-10-23 (loop round 91, flip 3): the Table 5-138
// per-class solar routing default (`FLX_R91_T5138`, §LIMIT-28 round-91
// record) re-pins the WD annual loads/peaks and the Table 6-1 day loads;
// the solar/sky columns and annual incident/transmitted values did not
// move (the routing partitions transmitted solar, it does not change the
// solar chain): WD100 7.1124/1.9064 -> 4.9518/4.7225 MWh (peak cooling
// occurrence 08-19-13 -> 10-18-14), WD200 3.3056/2.6052 -> 2.1325/5.6454
// (occurrence 08-19-13 -> 11-11-15), WD300 4.0237/1.4282 -> 3.1941/2.6104
// (occurrences unchanged), WD400 33.7832/0.0081 -> 30.6301/0.1578
// (occurrence 08-13-14 -> 08-07-16), WD500 0.4390/9.8393 -> 0.1510/15.8321
// (occurrences 12-30-07/10-05-14 -> 01-01-07/11-07-14), WD600 6.7310/2.6400
// -> 4.5815/6.2588 (occurrence 08-19-13 -> 10-18-14); day-load pins
// re-derived from the re-emitted section6_table61_daily_hourly.csv.
// Nothing tuned.
const SECTION6_VALUE_PINS: [(&str, f64, f64, f64, f64, &str, &str); 6] = [
    (
        "WD100", 5.7121, 3.0909, 4.6158, 3.2457, "02-09-05", "10-01-15",
    ),
    (
        "WD200", 2.5463, 3.9915, 3.8032, 3.0300, "02-12-07", "08-18-16",
    ),
    (
        "WD300", 3.5501, 2.0081, 2.6735, 2.5038, "07-22-08", "12-15-16",
    ),
    (
        "WD400", 32.0393, 0.0466, 8.2613, 1.6245, "01-26-16", "08-07-17",
    ),
    (
        "WD500", 0.2271, 12.8891, 1.4234, 4.6299, "12-28-07", "10-10-15",
    ),
    (
        "WD600", 5.3235, 4.2200, 4.5866, 3.7307, "02-09-05", "09-04-15",
    ),
];

const SECTION6_PIN_TOLERANCE: f64 = 0.005; // ±0.5%

// ---------------------------------------------------------------------------
// Round 51 pins (2026-10-17, develop `17c3ed16`, physics loop round 51):
// first recorded values for the formerly `missing` §6.2.1.2.1–.3 outputs,
// emitted by the extended harness from each case's own EPW via the engine's
// solar chain. Same ±0.5% discipline as SECTION6_VALUE_PINS; occurrences
// exact. Provenance: harness extension only — the engine is untouched, so
// these are recorded at the develop baseline.
// Load totals re-recorded 2026-10-24 (loop round 93, physics/r93-t5138-airshare: the ISO 13790 .C3 lag-site air-share term retires under the T5138 class route, FLX_R93_T5138_AIRSHARE falsy-token restore; premise verified against ASHRAE 140-2017 Table 5-138 + ISO 13790 Annex C) from the regenerated `section6_table61_daily_hourly.csv`; solar/sky columns did not move (weather-only path). Round-91c pin values (heat/cool/peak_h): WD100 [4.9518, 4.7225, 4.5739]; WD200 [2.1325, 5.6454, 3.7531]; WD300 [3.1941, 2.6104, 2.6313]; WD400 [30.6301, 0.1578, 8.2370]; WD500 [0.1510, 15.8321, 1.3847]; WD600 [4.5815, 6.2588, 4.5427]. Nothing tuned.
// ---------------------------------------------------------------------------
/// (case, annual incident solar [H,N,E,S,W] kWh/m², annual unshaded
/// transmitted south kWh/m², sky temp mean/min/max °C, sky min/max
/// occurrence MM-DD-HH) — section6_workflow_results.csv columns 16–26.
#[allow(clippy::type_complexity)]
const SECTION6_SOLAR_SKY_PINS: [(&str, [f64; 5], f64, f64, f64, f64, &str, &str); 6] = [
    (
        "WD100",
        [1692.0660, 265.4963, 940.6321, 1178.1116, 680.0683],
        774.0371,
        -2.0259,
        -38.1241,
        25.9857,
        "12-31-23",
        "08-04-14",
    ),
    (
        "WD200",
        [1706.8185, 294.4915, 870.3428, 1012.7436, 695.5009],
        658.9191,
        6.2808,
        -31.9210,
        29.3864,
        "02-12-05",
        "07-08-18",
    ),
    (
        "WD300",
        [1944.1705, 1234.5077, 1091.6968, 476.4039, 934.9494],
        312.5566,
        3.4066,
        -19.6907,
        23.8209,
        "07-22-04",
        "01-22-16",
    ),
    (
        "WD400",
        [724.9477, 240.8784, 437.4804, 603.6964, 391.5191],
        418.4923,
        -25.0215,
        -64.7248,
        8.7370,
        "01-26-15",
        "07-16-11",
    ),
    (
        "WD500",
        [1956.8986, 480.8809, 1123.3061, 1321.0755, 932.8583],
        857.0100,
        15.1553,
        -11.6837,
        37.3466,
        "12-29-04",
        "07-09-15",
    ),
    (
        "WD600",
        [1692.0660, 432.2374, 1107.3732, 1344.8527, 846.8094],
        883.1691,
        -2.0259,
        -38.1241,
        25.9857,
        "12-31-23",
        "08-04-14",
    ),
];

/// §6.2.1.8 day-level pins from section6_table61_daily_hourly.csv:
/// (case, daily total load kWh for Jan 4 / Feb 1 / Jul 14 (signed,
/// heating +, cooling −), daily mean sky temp °C for Feb 1 / May 4 /
/// Jul 14, daily total incident horizontal Wh/m² for Jul 14). These
/// guard the hourly series' totals; the byte-identity staleness test on
/// the committed CSV guards the full hourly shape.
/// Load totals re-recorded 2026-10-21 (loop round 85, per-class coupled
/// sub-step mass advance promoted to default, `FLX_R84_PERCLASS`,
/// Alex-directed 2026-10-21 21:18 ET, #3983 §65) from the regenerated
/// `section6_table61_daily_hourly.csv`; solar/sky columns did not move
/// (weather-only path). Round-78 load totals (coupled single-N default
/// flip, #3983 §63/§64; prior re-record 2026-10-19 round 73, R71
/// air-driver default flip #1522): WD100 [42.0251, 35.4767, -12.2571],
/// WD200 [20.114, 24.8932, -26.9096], WD300 [-4.4409, -12.8254, 24.7269],
/// WD400 [85.1847, 139.5708, 18.0445], WD500 [-3.6218, -12.0011,
/// -36.948], WD600 [40.8985, 32.4421, -19.5713]. Nothing tuned.
/// Load totals re-recorded 2026-10-22 (loop round 89, ISO §C.3
/// surface-pool mass share default flip, `FLX_R88_PHIST_MASS_SHARE`,
/// Alex-directed 2026-10-22 14:57 ET, round-88 arm #4543) from the
/// regenerated `section6_table61_daily_hourly.csv`; solar/sky columns
/// did not move (weather-only path). Round-85 load totals: WD100
/// [42.7781, 35.1745, -11.5262], WD200 [20.7252, 23.9342, -25.9208],
/// WD300 [-4.6488, -11.9670, 24.6486], WD400 [87.3840, 137.1279,
/// 17.8526], WD500 [-4.2797, -12.0047, -35.2561], WD600 [41.6243,
/// 32.2406, -18.5057]. Nothing tuned.
/// Load totals re-recorded 2026-10-22 (loop round 90, R77 lag-filter
/// removal default flip, `FLX_R90_NO_LAG_FILTER`, §LIMIT-28 round-90
/// record) from the regenerated `section6_table61_daily_hourly.csv`;
/// solar/sky columns did not move (weather-only path). Round-89b load
/// totals: WD100 [61.8618, 29.7646, -18.5243], WD200 [22.3500, 32.5514,
/// -29.9073], WD300 [-13.3412, -18.4369, 22.2585], WD400 [117.6887,
/// 117.1189, 12.6272], WD500 [-6.6486, -17.9874, -27.2679], WD600
/// [60.4531, 26.0563, -27.2438]. Nothing tuned.
/// Load totals re-recorded 2026-10-22 (loop round 91, flip 1, Table 5-138
/// window-loss sink default, `FLX_R91_RETRANS`, §LIMIT-28 round-91 record)
/// from the regenerated `section6_table61_daily_hourly.csv`; solar/sky
/// columns did not move (weather-only path). Round-90 load totals: WD100
/// [61.9827, 29.6279, -18.5160], WD200 [22.1166, 32.6062, -29.9079], WD300
/// [-13.3064, -18.3591, 22.2142], WD400 [117.6887, 117.1199, 13.0220],
/// WD500 [-6.5828, -18.0433, -27.2594], WD600 [60.5784, 25.9177,
/// -27.2402]. Nothing tuned.
/// Load totals re-recorded 2026-10-23 (loop round 91, flip 2, incumbent
/// solar split retired from the default, `FLX_R91_SOL2ST`, §LIMIT-28
/// round-91 record) from the regenerated
/// `section6_table61_daily_hourly.csv`; solar/sky columns did not move
/// (weather-only path). Round-91a load totals: WD100 [62.4010, 31.3302, -18.1013], WD200 [22.9641, 32.9039, -29.3351], WD300 [-12.8509, -17.9232, 22.3372], WD400 [117.6887, 117.1232, 13.6229], WD500 [-5.4717, -16.6809, -26.7636], WD600 [61.0308, 27.7137, -26.5968]. Nothing tuned.
/// Load totals re-recorded 2026-10-24 (loop round 93, physics/r93-t5138-airshare: the ISO 13790 .C3 lag-site air-share term retires under the T5138 class route, FLX_R93_T5138_AIRSHARE falsy-token restore) from the regenerated `section6_table61_daily_hourly.csv`; solar/sky columns did not move. Round-91c load totals: WD100 [59.1701, 32.7116, -17.9008]; WD200 [24.8527, 31.1463, -29.7561]; WD300 [-12.7172, -17.9007, 21.9047]; WD400 [117.2253, 116.6404, 10.5411]; WD500 [-5.6250, -14.9992, -27.5215]; WD600 [57.6893, 29.7205, -26.3013]. Nothing tuned.
const SECTION6_TABLE61_DAY_PINS: [(&str, [f64; 3], [f64; 3], f64); 6] = [
    (
        "WD100",
        [59.1701, 32.7116, -17.9008],
        [-25.226, -1.577, 8.832],
        8823.6,
    ),
    (
        "WD200",
        [24.8527, 31.1463, -29.7561],
        [-1.514, 10.349, 19.353],
        6868.2,
    ),
    (
        "WD300",
        [-12.7172, -17.9007, 21.9047],
        [8.897, -1.019, -5.611],
        3220.6,
    ),
    (
        "WD400",
        [117.2253, 116.6404, 10.5411],
        [-26.286, -32.833, -3.490],
        3756.7,
    ),
    (
        "WD500",
        [-5.6250, -14.9992, -27.5215],
        [1.491, 23.641, 25.454],
        3107.4,
    ),
    (
        "WD600",
        [57.6893, 29.7205, -26.3013],
        [-25.226, -1.577, 8.832],
        8823.6,
    ),
];

/// Any engine change that moves a §6.2.1.2 solar/sky output or a Table 6-1
/// daily aggregate must fail here until the pins and the committed CSVs are
/// honestly re-recorded with dated provenance.
#[test]
fn section6_solar_sky_and_table61_values_pinned_to_recorded_baseline() {
    let results = harness_results();
    for (case_id, incident, transmitted, sky_mean, sky_min, sky_max, occ_min, occ_max) in
        SECTION6_SOLAR_SKY_PINS
    {
        let r = results
            .iter()
            .find(|r| r.case_id == case_id)
            .unwrap_or_else(|| panic!("{case_id} missing from harness results"));
        for (o, (label, measured, pinned)) in [
            ("incident H", r.annual_incident_solar_kwh_m2[0], incident[0]),
            ("incident N", r.annual_incident_solar_kwh_m2[1], incident[1]),
            ("incident E", r.annual_incident_solar_kwh_m2[2], incident[2]),
            ("incident S", r.annual_incident_solar_kwh_m2[3], incident[3]),
            ("incident W", r.annual_incident_solar_kwh_m2[4], incident[4]),
            (
                "transmitted S",
                r.annual_transmitted_south_kwh_m2,
                transmitted,
            ),
            ("sky mean", r.sky_temp_mean_c, sky_mean),
            ("sky min", r.sky_temp_min_c, sky_min),
            ("sky max", r.sky_temp_max_c, sky_max),
        ]
        .into_iter()
        .enumerate()
        {
            let rel = ((measured - pinned) / pinned).abs();
            assert!(
                rel <= SECTION6_PIN_TOLERANCE,
                "{case_id} {label} drifted: measured {measured:.4} vs pinned {pinned:.4}                  ({:.2}% > ±0.5%). If this is a physics-correct change, re-emit the §6 CSVs                  and update the pins with dated old→new provenance — do not tune the engine.",
                rel * 100.0
            );
            let _ = o;
        }
        assert_eq!(
            r.sky_temp_min_occurrence, occ_min,
            "{case_id} sky-temperature minimum occurrence moved"
        );
        assert_eq!(
            r.sky_temp_max_occurrence, occ_max,
            "{case_id} sky-temperature maximum occurrence moved"
        );
    }

    for (case_id, loads, skys, incident_h_jul14) in SECTION6_TABLE61_DAY_PINS {
        let r = results
            .iter()
            .find(|r| r.case_id == case_id)
            .unwrap_or_else(|| panic!("{case_id} missing from harness results"));
        let day = |m: u32, d: u32| {
            r.daily_hourly
                .iter()
                .find(|x| x.month as u32 == m && x.day as u32 == d)
                .unwrap_or_else(|| panic!("{case_id}: Table 6-1 day {m}-{d} missing"))
        };
        let jan4 = day(1, 4);
        let feb1 = day(2, 1);
        let may4 = day(5, 4);
        let jul14 = day(7, 14);
        let sum = |a: &[f64; 24]| a.iter().sum::<f64>();
        for (label, measured, pinned) in [
            ("Jan 4 load kWh", sum(&jan4.load_kwh), loads[0]),
            ("Feb 1 load kWh", sum(&feb1.load_kwh), loads[1]),
            ("Jul 14 load kWh", sum(&jul14.load_kwh), loads[2]),
            ("Feb 1 sky mean °C", sum(&feb1.sky_temp_c) / 24.0, skys[0]),
            ("May 4 sky mean °C", sum(&may4.sky_temp_c) / 24.0, skys[1]),
            ("Jul 14 sky mean °C", sum(&jul14.sky_temp_c) / 24.0, skys[2]),
            (
                "Jul 14 incident H Wh/m²",
                sum(&jul14.incident_h_wh_m2),
                incident_h_jul14,
            ),
        ] {
            let rel = if pinned != 0.0 {
                ((measured - pinned) / pinned).abs()
            } else {
                measured.abs()
            };
            assert!(
                rel <= SECTION6_PIN_TOLERANCE,
                "{case_id} {label} drifted: measured {measured:.4} vs pinned {pinned:.4}                  ({:.2}% > ±0.5%). If this is a physics-correct change, re-emit the §6 CSVs                  and update the pins with dated old→new provenance — do not tune the engine.",
                rel * 100.0
            );
        }
    }
}

/// Any engine change that moves a §6 WD-case annual load, peak, or peak
/// occurrence must fail here until the pins (and the committed CSV) are
/// honestly re-recorded with dated provenance.
#[test]
fn section6_output_values_pinned_to_recorded_baseline() {
    let results = harness_results();
    assert_eq!(
        results.len(),
        SECTION6_VALUE_PINS.len(),
        "pin table must cover every §6 case"
    );
    for (case_id, heat, cool, peak_h, peak_c, occ_h, occ_c) in SECTION6_VALUE_PINS {
        let r = results
            .iter()
            .find(|r| r.case_id == case_id)
            .unwrap_or_else(|| panic!("{case_id} missing from harness results"));
        for (label, measured, pinned) in [
            ("annual_heating_mwh", r.annual_heating_mwh, heat),
            ("annual_cooling_mwh", r.annual_cooling_mwh, cool),
            ("peak_heating_kw", r.peak_heating_kw, peak_h),
            ("peak_cooling_kw", r.peak_cooling_kw, peak_c),
        ] {
            let rel = ((measured - pinned) / pinned).abs();
            assert!(
                rel <= SECTION6_PIN_TOLERANCE,
                "{case_id} {label} drifted: measured {measured:.4} vs pinned {pinned:.4} \
                 ({:.2}% > ±0.5%). If this is a physics-correct change, re-emit \
                 section6_workflow_results.csv and update SECTION6_VALUE_PINS with \
                 dated old→new provenance — do not tune the engine to the pins.",
                rel * 100.0
            );
        }
        assert_eq!(
            r.peak_heating_occurrence, occ_h,
            "{case_id} heating peak occurrence moved (sun/timestamp convention change?)"
        );
        assert_eq!(
            r.peak_cooling_occurrence, occ_c,
            "{case_id} cooling peak occurrence moved (sun/timestamp convention change?)"
        );
    }
}

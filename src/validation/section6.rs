//! ASHRAE 140 §6 Weather Driver case harness (Issue #4269).
//!
//! Runs the six WD100–WD600 weather files through the engine on the §5.2.1
//! low-mass case configuration and produces the per-case outputs the §6
//! reporting chain consumes (#4283 emitter, #4276 conformance table).
//!
//! # What the harness runs
//!
//! Each WD case is the standard's low-mass envelope case (Case 600
//! configuration) driven for a full 8760-hour year by the location's EPW
//! file, with the site (latitude, longitude, UTC offset) and the authored
//! per-hour ground albedo taken from the EPW header/data rather than the
//! fabric-cohort defaults. The authored-albedo override exists because the
//! 0.0→0.2 TMY3 fabric fallback (Issue #4269, flagged for Alex's review)
//! would otherwise flatten WD100's explicit 0 onto WD600's 0.2 and erase the
//! section's single-variable ground-reflected-shortwave isolation.
//!
//! # Required outputs (grounded in the in-repo standard text)
//!
//! The harness produces the Class I non-free-float outputs of standard
//! Section 6.2.1.1 (docs/140_2017_a_20200901.pdf, lines 5950–5961):
//! annual heating load (MWh), annual sensible cooling load (MWh), and the
//! annual hourly-integrated peak heating/cooling loads (kW) with their
//! month/day/hour of occurrence. Monthly heating/cooling series are carried
//! as diagnostics (Section 6.2.1.2.4 shape) for the §6 seasonality checks.
//!
//! # Consumer contract
//!
//! [`Section6CaseResult::csv_header`] / [`Section6CaseResult::csv_row`]
//! emit a flat workflow-results row keyed on the case id first — the same
//! "one flat CSV, one row per case" interchange shape #4276 adopts from
//! BESTEST-GSR — so the #4283 emitter and the conformance table read these
//! rows without re-deriving them from the engine.

use crate::sim::engine::ThermalModel;
use crate::sim::thermal_selector::ThermalSelector;
use crate::validation::ashrae_140_cases::ASHRAE140Case;
use crate::weather::epw::EpwWeatherSource;
use crate::weather::WeatherSource;

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
    let mut model = ThermalModel::<crate::physics::cta::VectorField>::from_spec_with_selector(
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
        let (_year, month, day, hour) = crate::sim::engine::ThermalModel::<
            crate::physics::cta::VectorField,
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
    let mut csv = String::from(Section6CaseResult::csv_header());
    csv.push('\n');
    for r in &results {
        csv.push_str(&r.csv_row());
        csv.push('\n');
    }
    Ok(csv)
}

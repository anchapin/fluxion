//! ASHRAE 140 §7 results-CSV emitter (Issue #4283).
//!
//! Emits one CSV row per implemented §7 (Building Thermal Envelope and
//! Fabric Load) case, satisfying the BESTEST-GSR consumer contract in
//! `data/reference/ashrae140/bestest_gsr_csv_contract.json` (#4282). The
//! consumer is `NatLabRockies/BESTEST-GSR results/bestest_populate_report.rb`,
//! which keys rows on the first whitespace token of column 1 and parses
//! timestamps by character offset — both fail silently, so the contract and
//! `scripts/check_bestest_csv.py` are the acceptance gate.
//!
//! Scope (per #4283): §7 cases run ideal heating/cooling with unlimited
//! capacity, so the emitted quantities are envelope loads — annual energy
//! (MWh), hourly-integrated peaks (kW), monthly energy (MWh) with per-month
//! peak values (kW) and `DD-MMM-HH:MM` timestamps — plus free-float zone
//! temperatures, incident/transmitted solar, sky temperature, day-profile
//! hourly series, and the `temp_bins` hourly temperature-bin histogram.
//! Equipment consumption is §9/§10 and is not part of this.
//!
//! Engine route: blind, spec-only — `ASHRAE140Case::*.spec()` through
//! `ThermalModel::from_spec_with_selector` + `step_physics` on the canonical
//! strict-gate weather (`strict_gate_weather_epw()`, DRYCOLD 725650TY by
//! default, Golden opt-in). No case-id shortcuts reach the engine. The
//! solar/sky chain is the engine's own (`calculate_surface_irradiance`,
//! `calculate_window_solar_gain`, `SkyRadiationExchange::sky_temperature_from_ir`)
//! with each EPW's authored albedo, following the #4269 rule and the §6
//! weather-driver pattern (rounds 48/51).
//!
//! Only implemented cases are emitted (contract rule: the consumer reads its
//! case list off the template, so omitted cases fail one lookup rather than
//! corrupt layout). The pinned §7 case set is the 18 ASHRAE140Case variants
//! below; free-float rows carry zero loads by physics (HVAC disabled), and
//! their `status` column says so.
//!
//! Hour convention: the engine's `timestep_to_date` hour is 0-based; the CSV
//! emits hour-ending form `HH = hour + 1` (1..24) with `:00` minutes, so the
//! consumer's minute-rollover rule (increment hour when minutes nonzero)
//! never produces hour 25.
//!
//! Packed-array convention (consumer re-splits on ',' after skipping the
//! leading label elements): day profiles carry two label elements then 24
//! hourly values (hour 1..24), except `temp_<date>` with one label element;
//! `temp_bins` carries two label elements then 90 bin counts for the 1 °C
//! bins −19…+70.

use fluxion::ashrae_cases::ShadingType;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::shading::{Overhang, ShadeFin, Side};
use fluxion::sim::sky_radiation::SkyRadiationExchange;
use fluxion::sim::solar::{calculate_window_solar_gain, WindowProperties};
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::solar::solar_position::{calculate_day_of_year, calculate_solar_position};
use fluxion::solar::surface_irradiance::{calculate_surface_irradiance, Orientation};
use fluxion::validation::ashrae_140_cases::{strict_gate_weather_epw, ASHRAE140Case};
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

const MONTH_ABBR: [&str; 12] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];

/// One §7 case definition: the pinned case set (contract rule: emit only
/// implemented cases; the consumer fails one lookup per omitted case).
pub struct Section7CaseDef {
    pub case_id: &'static str,
    pub variant: ASHRAE140Case,
    /// Free-float cases run with HVAC disabled (`run_blind_free_float`
    /// pattern, #4170): loads are zero by physics, temps are the output.
    pub free_float: bool,
    pub description: &'static str,
}

/// The pinned §7 case set: every ASHRAE140Case variant fluxion implements,
/// in series order (600/900 series, then 960/970, then the free floats).
pub const SECTION7_CASES: [Section7CaseDef; 18] = [
    Section7CaseDef {
        case_id: "600",
        variant: ASHRAE140Case::Case600,
        free_float: false,
        description: "Low mass baseline",
    },
    Section7CaseDef {
        case_id: "610",
        variant: ASHRAE140Case::Case610,
        free_float: false,
        description: "Low mass with south shading",
    },
    Section7CaseDef {
        case_id: "620",
        variant: ASHRAE140Case::Case620,
        free_float: false,
        description: "Low mass with east/west windows",
    },
    Section7CaseDef {
        case_id: "630",
        variant: ASHRAE140Case::Case630,
        free_float: false,
        description: "Low mass with east/west shading",
    },
    Section7CaseDef {
        case_id: "640",
        variant: ASHRAE140Case::Case640,
        free_float: false,
        description: "Low mass with thermostat setback",
    },
    Section7CaseDef {
        case_id: "650",
        variant: ASHRAE140Case::Case650,
        free_float: false,
        description: "Low mass with night ventilation",
    },
    Section7CaseDef {
        case_id: "900",
        variant: ASHRAE140Case::Case900,
        free_float: false,
        description: "High mass baseline",
    },
    Section7CaseDef {
        case_id: "910",
        variant: ASHRAE140Case::Case910,
        free_float: false,
        description: "High mass with south shading",
    },
    Section7CaseDef {
        case_id: "920",
        variant: ASHRAE140Case::Case920,
        free_float: false,
        description: "High mass with east/west windows",
    },
    Section7CaseDef {
        case_id: "930",
        variant: ASHRAE140Case::Case930,
        free_float: false,
        description: "High mass with east/west shading",
    },
    Section7CaseDef {
        case_id: "940",
        variant: ASHRAE140Case::Case940,
        free_float: false,
        description: "High mass with thermostat setback",
    },
    Section7CaseDef {
        case_id: "950",
        variant: ASHRAE140Case::Case950,
        free_float: false,
        description: "High mass with night ventilation",
    },
    Section7CaseDef {
        case_id: "960",
        variant: ASHRAE140Case::Case960,
        free_float: false,
        description: "Sunspace",
    },
    Section7CaseDef {
        case_id: "970",
        variant: ASHRAE140Case::Case970,
        free_float: false,
        description: "Five zone cross coupling",
    },
    Section7CaseDef {
        case_id: "600FF",
        variant: ASHRAE140Case::Case600FF,
        free_float: true,
        description: "Low mass free float",
    },
    Section7CaseDef {
        case_id: "650FF",
        variant: ASHRAE140Case::Case650FF,
        free_float: true,
        description: "Low mass night vent free float",
    },
    Section7CaseDef {
        case_id: "900FF",
        variant: ASHRAE140Case::Case900FF,
        free_float: true,
        description: "High mass free float",
    },
    Section7CaseDef {
        case_id: "950FF",
        variant: ASHRAE140Case::Case950FF,
        free_float: true,
        description: "High mass night vent free float",
    },
];

/// The §7 day-profile days (contract `day_profile_array` names):
/// Feb 1, May 4, Jul 14 — the same days the §6 Table 6-1 daily CSV covers.
pub const SECTION7_PROFILE_DAYS: [(u32, u32); 3] = [(2, 1), (5, 4), (7, 14)];

/// Per-case §7 outputs, one engine run each.
#[derive(Debug, Clone, PartialEq)]
pub struct Section7CaseResult {
    pub case_id: String,
    pub description: String,
    pub free_float: bool,
    /// Annual heating (MWh), ideal-load integral. Zero on free-float rows.
    pub annual_heating_mwh: f64,
    /// Annual sensible cooling (MWh). Zero on free-float rows.
    pub annual_cooling_mwh: f64,
    /// Annual hourly-integrated peak heating/cooling (kW).
    pub peak_heating_kw: f64,
    pub peak_cooling_kw: f64,
    /// Peak occurrence (month, day, hour0) with hour0 the engine's 0-based
    /// hour; rendered `DD-MMM-HH:MM` at emission.
    pub peak_heating_at: (u32, u32, u32),
    pub peak_cooling_at: (u32, u32, u32),
    /// Monthly ideal-load energy (MWh), calendar order.
    pub monthly_heating_mwh: [f64; 12],
    pub monthly_cooling_mwh: [f64; 12],
    /// Per-month peak value (kW) and (month, day, hour0) occurrence.
    /// A month with no heating (cooling) hours carries 0.0 and the sentinel
    /// occurrence (0, 1, 0) — rendered `01-JAN-01:00`-style for that month.
    pub monthly_heating_peak_kw: [f64; 12],
    pub monthly_heating_peak_at: [(u32, u32, u32); 12],
    pub monthly_cooling_peak_kw: [f64; 12],
    pub monthly_cooling_peak_at: [(u32, u32, u32); 12],
    /// Annual mean / min / max zone temperature (°C, zone 1) with the
    /// 0-based 8760 index of each extremum's first occurrence. Multi-zone
    /// cases report zone 1 only (documented superset decision; the §7
    /// temperature outputs are defined on the single-zone test cell).
    pub avg_temp_c: f64,
    pub min_temp_c: f64,
    pub min_index_position: usize,
    pub max_temp_c: f64,
    pub max_index_position: usize,
    /// Annual incident unshaded total solar (kWh/m²), order
    /// [horizontal(=consumer "roof"), north, east, south, west].
    pub annual_incident_kwh_m2: [f64; 5],
    /// Annual unshaded transmitted total solar through the south windows
    /// (kWh/m² of glazing; §6.2.1.2.2 reference-window route).
    pub annual_transmitted_south_kwh_m2: f64,
    /// Sky temperature (°C) mean / min / max with 8760 indices.
    pub sky_avg_temp_c: f64,
    pub sky_min_temp_c: f64,
    pub sky_min_index_position: usize,
    pub sky_max_temp_c: f64,
    pub sky_max_index_position: usize,
    /// Day-profile hourly series for each SECTION7_PROFILE_DAYS day.
    pub day_profiles: Vec<Section7DayHourly>,
    /// Hourly zone-temperature bin histogram (`temp_bins`): counts of hours
    /// whose zone-1 temperature floor falls in each 1 °C bin from −19 to 69
    /// inclusive (90 bins; consumer reads bins −19…+70). Hours below −19 °C
    /// count into the first bin and hours ≥ 70 °C into the last, so the
    /// 8760-hour total is preserved.
    pub temp_bins: [u64; 90],
    /// §6.2.1.3 (Case 610 only): annual transmitted solar (diffuse +
    /// direct-beam) through the case's shaded south window with its
    /// horizontal overhang, kWh/m². Standard 140-2017 §6.2.1.3.1 notes
    /// the quantity "does not include radiation that is absorbed in the
    /// glass and conducted inward as heat" — emitted via the engine's own
    /// `calculate_window_solar_gain` chain with the case window geometry and
    /// the engine's spec→shading conversion (round 62; see the runner).
    pub req_6213_shaded_south_transmitted_kwh_m2: Option<f64>,
    /// §6.2.1.4 (Case 620 only): unshaded annual transmitted solar
    /// through the case's west window, kWh/m² (§6.2.1.4.1).
    pub req_6214_west_transmitted_kwh_m2: Option<f64>,
    /// §6.2.1.5 (Case 630 only): annual transmitted solar through the
    /// case's shaded west window with horizontal overhang and vertical
    /// fins, kWh/m² (§6.2.1.5.1).
    pub req_6215_shaded_west_transmitted_kwh_m2: Option<f64>,
    /// §6.2.1.6 sun-zone free-float temperatures for Case 960 only
    /// (140-2017 §5.2.2.2.7.7.b: "Sun zone only, all free-float case
    /// outputs in accordance with Section 6.2.1.6"). The sun zone is the
    /// spec's zone index 1 (sunspace; zone 0 is the conditioned back
    /// zone). `None` for every other case — the FF cases' single zone is
    /// already the recorded zone-1 output.
    pub sun_zone: Option<Section7SunZoneStats>,
}

/// Case 960's sun-zone free-float temperature statistics (§6.2.1.6 form).
#[derive(Debug, Clone, PartialEq)]
pub struct Section7SunZoneStats {
    pub min_temp_c: f64,
    /// 0-based 8760 index of the minimum's first occurrence.
    pub min_index: usize,
    pub max_temp_c: f64,
    /// 0-based 8760 index of the maximum's first occurrence.
    pub max_index: usize,
    pub mean_temp_c: f64,
}

/// One profile day's hourly series for one case.
#[derive(Debug, Clone, PartialEq)]
pub struct Section7DayHourly {
    pub month: u8,
    pub day: u8,
    /// Hourly integrated incident unshaded total solar (Wh/m²) on the
    /// horizontal plane (consumer "roof" surface), south, and west.
    pub incident_h_wh_m2: [f64; 24],
    pub incident_s_wh_m2: [f64; 24],
    pub incident_w_wh_m2: [f64; 24],
    /// Hourly transmitted total solar through the south windows (Wh/m²).
    pub transmitted_south_wh_m2: [f64; 24],
    /// Hourly integrated sky temperature (°C).
    pub sky_temp_c: [f64; 24],
    /// Hourly ideal load (kWh), signed: heating positive, cooling negative.
    pub load_kwh: [f64; 24],
    /// Hourly zone-1 temperature (°C).
    pub zone_temp_c: [f64; 24],
}

fn strict_gate_epw_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(strict_gate_weather_epw())
}

/// `DD-MMM-HH:MM` from (month, day, hour0). Hour-ending form: `HH = hour0+1`
/// (1..24), minutes always `00`, so the consumer's minute-rollover rule
/// (increment the hour when minutes are nonzero) never fires and can never
/// produce hour 25.
fn format_dd_mmm_hhmm(at: (u32, u32, u32)) -> String {
    // `at.0` is 1-based (1..=12); the "no peak" sentinel carries month 12 so
    // this subtraction stays in range in debug builds too. Hour-ending form:
    // `HH = hour0+1` (1..24), minutes always `00`, so the consumer's
    // minute-rollover rule (increment the hour when minutes are nonzero)
    // never fires and can never produce hour 25.
    let mon = MONTH_ABBR[(at.0 as usize - 1).clamp(0, 11)];
    format!(
        "{:02}-{}-{:02}:00",
        at.1,
        mon,
        at.2.saturating_add(1).clamp(1, 24)
    )
}

/// Runs one §7 case through the engine and returns its per-case outputs.
///
/// Blind execution: only the CaseSpec reaches the engine, on the canonical
/// §7 blind route (run_blind_annual_energy / run_blind_free_float pattern —
/// see the coordinates note inside).
pub fn run_section7_case(def: &Section7CaseDef) -> Result<Section7CaseResult, String> {
    let path = strict_gate_epw_path();
    let weather = EpwWeatherSource::from_file(&path)
        .map_err(|e| format!("{}: EPW parse: {e:?}", def.case_id))?;

    let spec = def.variant.spec();
    let mut model = ThermalModel::<fluxion::physics::cta::VectorField>::from_spec_with_selector(
        &spec,
        &ThermalSelector::default(),
    )
    .map_err(|e| format!("{}: model init failed: {e:?}", def.case_id))?;

    // Solar-chain coordinates come from the model's own site defaults —
    // the same values the engine's solar code uses — so the emitter's
    // incident-solar quantities and the engine's loads see one sky.
    // Deliberately NO EPW-location override and NO authored-EPW albedo flag:
    // DRYCOLD's albedo field is an unfilled 0 (all 8760 rows), and #4269's
    // authored-albedo rule is a §6 WD-file rule — forcing it here would
    // silently zero ground-reflected shortwave and move §7 off the
    // engine's canonical blind §7 route (measured round 53: with the flag
    // set, 600FF Tmax fell 55.26 → 51.62 °C against the strict-gate value).
    let latitude_deg = model.0.solar.latitude_deg;
    let longitude_deg = model.0.solar.longitude_deg;
    let utc_offset_hours = model.0.solar.utc_offset_hours;
    let _ = (latitude_deg, longitude_deg, utc_offset_hours); // recorded for provenance, not injected

    // Free-float mode (mirrors run_blind_free_float, #4170): HVAC disabled.
    if def.free_float {
        model.0.setpoints.heating_setpoint = -999.0;
        model.0.setpoints.cooling_setpoint = 999.0;
        model.0.hvac.hvac_heating_capacity = 0.0;
        model.0.hvac.hvac_cooling_capacity = 0.0;
    }

    const INCIDENT_ORIENTATIONS: [Orientation; 5] = [
        Orientation::Horizontal,
        Orientation::North,
        Orientation::East,
        Orientation::South,
        Orientation::West,
    ];
    const PROFILE_ORIENTATIONS: [Orientation; 3] = [
        Orientation::Horizontal,
        Orientation::South,
        Orientation::West,
    ];

    let j_to_mwh = 1.0 / 3.6e9;
    // Zero-load months keep the "no peak" sentinel, which renders as
    // `01-DEC-01:00` in the emitted timestamps (month 12, day 1, hour 0).
    // Earlier rounds used (0, 1, 0) here and depended on release-mode
    // wraparound of `month - 1` in format_dd_mmm_hhmm to land on DEC —
    // debug builds panicked instead. Same rendered bytes, valid arithmetic.
    let sentinel_at: (u32, u32, u32) = (12, 1, 0); // renders 01-DEC-01:00
    let mut annual_heating_j = 0.0_f64;
    let mut annual_cooling_j = 0.0_f64;
    let mut peak_heating_w = 0.0_f64;
    let mut peak_cooling_w = 0.0_f64;
    let mut peak_heating_at = sentinel_at;
    let mut peak_cooling_at = sentinel_at;
    let mut monthly_heating_mwh = [0.0_f64; 12];
    let mut monthly_cooling_mwh = [0.0_f64; 12];
    let mut monthly_heating_peak_kw = [0.0_f64; 12];
    let mut monthly_cooling_peak_kw = [0.0_f64; 12];
    let mut monthly_heating_peak_at = [sentinel_at; 12];
    let mut monthly_cooling_peak_at = [sentinel_at; 12];

    let mut temp_sum = 0.0_f64;
    let mut temp_min = f64::INFINITY;
    let mut temp_min_index = 0_usize;
    let mut temp_max = f64::NEG_INFINITY;
    let mut temp_max_index = 0_usize;
    let mut temp_bins = [0_u64; 90];

    let mut annual_incident_kwh_m2 = [0.0_f64; 5];
    let mut annual_transmitted_south_kwh_m2 = 0.0_f64;
    let mut sky_temp_sum = 0.0_f64;
    let mut sky_temp_min = f64::INFINITY;
    let mut sky_temp_min_index = 0_usize;
    let mut sky_temp_max = f64::NEG_INFINITY;
    let mut sky_temp_max_index = 0_usize;

    // Case-spec window for the §6.2.1.2.2 transmitted-solar reference
    // (1 m², unshaded: the requirement's wording, so no overhang/fin
    // geometry is applied even when the case carries one).
    let window_1m2 = WindowProperties::new(
        1.0,
        spec.window_properties.shgc,
        spec.window_properties.normal_transmittance,
    );

    // Section 6.2.1.3/.4/.5 case-window route (round 62): the case's own
    // window at its spec area/orientation, shaded per the engine's own
    // spec-to-shading conversion (`from_spec_with_selector`, the post-
    // LIMIT-39 wiring): overhang mounted at the window top
    // (distance_above 0, "infinite" extension 10 m), fins at both window
    // edges spanning the full window height. Requirements (in-repo
    // standard, verified 2026-10-18): 6.2.1.3.1 (Case 610) annual
    // transmitted through the shaded south window with horizontal
    // overhang; 6.2.1.4.1 (Case 620) unshaded annual transmitted through
    // the west window; 6.2.1.5.1 (Case 630) annual transmitted through
    // the shaded west window with overhang and vertical fins.
    let case_req: Option<(&'static str, Orientation)> = match def.case_id {
        "610" => Some((
            "req_6213_shaded_south_transmitted_kwh_m2",
            Orientation::South,
        )),
        "620" => Some(("req_6214_west_transmitted_kwh_m2", Orientation::West)),
        "630" => Some(("req_6215_shaded_west_transmitted_kwh_m2", Orientation::West)),
        _ => None,
    };
    let case_window_geom = case_req.and_then(|(_, ori)| {
        spec.windows
            .first()
            .and_then(|ws| ws.iter().find(|w| w.orientation == ori.into()))
            .copied()
    });
    let case_window_props = case_window_geom.map(|g| {
        WindowProperties::new(
            g.area,
            spec.window_properties.shgc,
            spec.window_properties.normal_transmittance,
        )
    });
    let case_overhang = spec
        .shading
        .as_ref()
        .filter(|s| {
            matches!(
                s.shading_type,
                ShadingType::Overhang | ShadingType::OverhangAndFins
            )
        })
        .map(|s| Overhang {
            depth: s.overhang_depth,
            distance_above: 0.0,
            extension: 10.0, // "Infinite" (engine wiring, LIMIT-39)
        });
    let case_fins: Vec<ShadeFin> = match spec.shading.as_ref() {
        Some(s)
            if matches!(
                s.shading_type,
                ShadingType::Fins | ShadingType::OverhangAndFins
            ) =>
        {
            let h = case_window_geom.map(|g| g.height).unwrap_or(2.0);
            vec![
                ShadeFin {
                    depth: s.fin_width,
                    distance_from_edge: 0.0,
                    side: Side::Left,
                    height: h,
                },
                ShadeFin {
                    depth: s.fin_width,
                    distance_from_edge: 0.0,
                    side: Side::Right,
                    height: h,
                },
            ]
        }
        _ => Vec::new(),
    };
    let mut case_transmitted_kwh_m2 = 0.0_f64;

    // Case 960 sun-zone free-float stats (zone index 1, sunspace). The
    // back zone (index 0) is the conditioned zone recorded above; the sun
    // zone carries `HvacSchedule::free_floating()` in the spec, so its
    // temperature floats by physics without any emitter override.
    let is_960 = def.case_id == "960";
    let mut sun_zone_sum = 0.0_f64;
    let mut sun_zone_min = f64::INFINITY;
    let mut sun_zone_min_index = 0_usize;
    let mut sun_zone_max = f64::NEG_INFINITY;
    let mut sun_zone_max_index = 0_usize;

    let mut day_profiles: Vec<Section7DayHourly> = SECTION7_PROFILE_DAYS
        .iter()
        .map(|&(m, d)| Section7DayHourly {
            month: m as u8,
            day: d as u8,
            incident_h_wh_m2: [0.0; 24],
            incident_s_wh_m2: [0.0; 24],
            incident_w_wh_m2: [0.0; 24],
            transmitted_south_wh_m2: [0.0; 24],
            sky_temp_c: [0.0; 24],
            load_kwh: [0.0; 24],
            zone_temp_c: [0.0; 24],
        })
        .collect();
    let day_index = |month: u32, day: u32| -> Option<usize> {
        SECTION7_PROFILE_DAYS
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
        let (year, month, day, hour) =
            ThermalModel::<fluxion::physics::cta::VectorField>::timestep_to_date(step);
        let hour0 = hour as u32;
        let mi = (month as usize - 1).min(11);
        if energy_kwh > 0.0 {
            annual_heating_j += energy_j;
            monthly_heating_mwh[mi] += energy_j * j_to_mwh;
            let kw = energy_j / 3600.0;
            if kw > peak_heating_w {
                peak_heating_w = kw;
                peak_heating_at = (month, day, hour0);
            }
            if kw > monthly_heating_peak_kw[mi] {
                monthly_heating_peak_kw[mi] = kw;
                monthly_heating_peak_at[mi] = (month, day, hour0);
            }
        } else if energy_kwh < 0.0 {
            annual_cooling_j += -energy_j;
            monthly_cooling_mwh[mi] += -energy_j * j_to_mwh;
            let kw = -energy_j / 3600.0;
            if kw > peak_cooling_w {
                peak_cooling_w = kw;
                peak_cooling_at = (month, day, hour0);
            }
            if kw > monthly_cooling_peak_kw[mi] {
                monthly_cooling_peak_kw[mi] = kw;
                monthly_cooling_peak_at[mi] = (month, day, hour0);
            }
        }

        // Zone temperature (zone 1; multi-zone cases report zone 1 only —
        // documented superset decision, §7's temperature outputs are defined
        // on the single-zone test cell).
        let zone_temp = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(f64::NAN);
        temp_sum += zone_temp;
        if zone_temp < temp_min {
            temp_min = zone_temp;
            temp_min_index = step;
        }
        if zone_temp > temp_max {
            temp_max = zone_temp;
            temp_max_index = step;
        }
        // temp_bins: floor into the −19..69 clamp window (see struct docs).
        let bin = ((zone_temp.floor() as i64).clamp(-19, 69) + 19) as usize;
        temp_bins[bin] += 1;

        // Case 960 sun zone (zone index 1): free-float by spec.
        if is_960 {
            if let Some(t) = model.setpoints.temperatures.as_slice().get(1).copied() {
                sun_zone_sum += t;
                if t < sun_zone_min {
                    sun_zone_min = t;
                    sun_zone_min_index = step;
                }
                if t > sun_zone_max {
                    sun_zone_max = t;
                    sun_zone_max_index = step;
                }
            }
        }

        // Per-hour solar/sky quantities from the EPW record (engine chain).
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
        let albedo = weather_data.albedo.unwrap_or(0.2);
        let mut irr_s = None;
        let mut irr_w = None;
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
            match *orientation {
                Orientation::South => irr_s = Some(irr),
                Orientation::West => irr_w = Some(irr),
                _ => {}
            }
        }
        let irr_s = irr_s.expect("South is in INCIDENT_ORIENTATIONS");
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
        if let (Some(g), Some(props), Some((_, ori))) =
            (case_window_geom, case_window_props, case_req)
        {
            let irr_case = if ori == Orientation::South {
                irr_s
            } else {
                irr_w.expect("West is in INCIDENT_ORIENTATIONS")
            };
            let gain_case = calculate_window_solar_gain(
                &irr_case,
                &props,
                Some(&g),
                case_overhang.as_ref(),
                &case_fins,
                &sun,
                ori.into(),
            );
            // Requirement form is kWh/m² of glazing (6.2.1.3.1/.4.1/.5.1);
            // the engine gain is area-scaled watts, so normalise by the
            // case window area.
            case_transmitted_kwh_m2 += gain_case.total_gain_w / g.area / 1000.0;
        }
        let sky_temp_c =
            SkyRadiationExchange::sky_temperature_from_ir(weather_data.horizontal_infrared);
        sky_temp_sum += sky_temp_c;
        if sky_temp_c < sky_temp_min {
            sky_temp_min = sky_temp_c;
            sky_temp_min_index = step;
        }
        if sky_temp_c > sky_temp_max {
            sky_temp_max = sky_temp_c;
            sky_temp_max_index = step;
        }

        // Day-profile capture.
        if let Some(di) = day_index(month, day) {
            let h = hour0 as usize;
            let d = &mut day_profiles[di];
            for (o, orientation) in PROFILE_ORIENTATIONS.iter().enumerate() {
                let irr = calculate_surface_irradiance(
                    &sun,
                    weather_data.dni,
                    weather_data.dhi,
                    Some(weather_data.ghi),
                    *orientation,
                    albedo,
                    day_of_year,
                );
                match o {
                    0 => d.incident_h_wh_m2[h] = irr.total_wm2,
                    1 => d.incident_s_wh_m2[h] = irr.total_wm2,
                    _ => d.incident_w_wh_m2[h] = irr.total_wm2,
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
            d.zone_temp_c[h] = zone_temp;
        }
    }

    Ok(Section7CaseResult {
        case_id: def.case_id.to_string(),
        description: def.description.to_string(),
        free_float: def.free_float,
        annual_heating_mwh: annual_heating_j * j_to_mwh,
        annual_cooling_mwh: annual_cooling_j * j_to_mwh,
        peak_heating_kw: peak_heating_w / 1000.0,
        peak_cooling_kw: peak_cooling_w / 1000.0,
        peak_heating_at,
        peak_cooling_at,
        monthly_heating_mwh,
        monthly_cooling_mwh,
        monthly_heating_peak_kw,
        monthly_heating_peak_at,
        monthly_cooling_peak_kw,
        monthly_cooling_peak_at,
        avg_temp_c: temp_sum / 8760.0,
        min_temp_c: temp_min,
        min_index_position: temp_min_index,
        max_temp_c: temp_max,
        max_index_position: temp_max_index,
        annual_incident_kwh_m2,
        annual_transmitted_south_kwh_m2,
        sky_avg_temp_c: sky_temp_sum / 8760.0,
        sky_min_temp_c: sky_temp_min,
        sky_min_index_position: sky_temp_min_index,
        sky_max_temp_c: sky_temp_max,
        sky_max_index_position: sky_temp_max_index,
        day_profiles,
        temp_bins,
        req_6213_shaded_south_transmitted_kwh_m2: if def.case_id == "610" {
            Some(case_transmitted_kwh_m2)
        } else {
            None
        },
        req_6214_west_transmitted_kwh_m2: if def.case_id == "620" {
            Some(case_transmitted_kwh_m2)
        } else {
            None
        },
        req_6215_shaded_west_transmitted_kwh_m2: if def.case_id == "630" {
            Some(case_transmitted_kwh_m2)
        } else {
            None
        },
        sun_zone: if is_960 {
            Some(Section7SunZoneStats {
                min_temp_c: sun_zone_min,
                min_index: sun_zone_min_index,
                max_temp_c: sun_zone_max,
                max_index: sun_zone_max_index,
                mean_temp_c: sun_zone_sum / 8760.0,
            })
        } else {
            None
        },
    })
}

/// Runs the full pinned §7 case set in standard order.
pub fn run_all_section7_cases() -> Result<Vec<Section7CaseResult>, String> {
    SECTION7_CASES.iter().map(run_section7_case).collect()
}

/// One engine pass for the whole test binary: each test reads the same
/// results (§6 `harness_results` pattern).
fn harness_results() -> &'static [Section7CaseResult] {
    static RESULTS: std::sync::OnceLock<Vec<Section7CaseResult>> = std::sync::OnceLock::new();
    RESULTS.get_or_init(|| run_all_section7_cases().expect("all §7 cases must run"))
}

// ===========================================================================
// CSV emission — the BESTEST-GSR contract form.
// ===========================================================================

fn fmt(v: f64, places: usize) -> String {
    format!("{:.*}", places, v)
}

fn join2(labels: &str, vals: &[f64], places: usize) -> String {
    let mut out = labels.to_string();
    for v in vals {
        out.push(',');
        out.push_str(&fmt(*v, places));
    }
    out
}

fn day_profile<'a>(r: &'a Section7CaseResult, month: u32, day: u32) -> &'a Section7DayHourly {
    r.day_profiles
        .iter()
        .find(|d| d.month as u32 == month && d.day as u32 == day)
        .expect("every SECTION7_PROFILE_DAYS day captured")
}

/// CSV header: `bestest.case_num`, the 48 contract-prefixed columns in
/// contract order, then the trailing unprefixed columns.
pub fn section7_gsr_csv_header() -> String {
    const P: &str = "bestest_building_thermal_envelope_and_fabric_load_reporting";
    let scalars = [
        "annual_cooling",
        "annual_heating",
        "avg_temp",
        "east_incident_solar_radiation",
        "horizontal_incident_solar_radiation",
        "max_index_position",
        "max_temp",
        "min_index_position",
        "min_temp",
        "north_incident_solar_radiation",
        "peak_cooling_value",
        "peak_heating_value",
        "sky_avg_temp",
        "sky_max_index_position",
        "sky_max_temp",
        "sky_min_index_position",
        "sky_min_temp",
        "south_incident_solar_radiation",
        "surf_out_inst_slr_rad_0201_zone_surface_roof",
        "surf_out_inst_slr_rad_0201_zone_surface_south",
        "surf_out_inst_slr_rad_0201_zone_surface_west",
        "surf_out_inst_slr_rad_0504_zone_surface_roof",
        "surf_out_inst_slr_rad_0504_zone_surface_south",
        "surf_out_inst_slr_rad_0504_zone_surface_west",
        "surf_out_inst_slr_rad_0714_zone_surface_roof",
        "surf_out_inst_slr_rad_0714_zone_surface_south",
        "surf_out_inst_slr_rad_0714_zone_surface_west",
        "surf_win_trans_rad_0201_zone_surface_south",
        "surf_win_trans_rad_0504_zone_surface_south",
        "surf_win_trans_rad_0714_zone_surface_south",
        "west_incident_solar_radiation",
        "zone_total_transmitted_solar_radiation",
    ];
    let timestamps = ["peak_cooling_time", "peak_heating_time"];
    let monthly = [
        "monthly_clg_cons",
        "monthly_clg_peak_time",
        "monthly_clg_peak_val",
        "monthly_htg_cons",
        "monthly_htg_peak_time",
        "monthly_htg_peak_val",
    ];
    let day_profiles = [
        "sens_htg_clg_0201",
        "sens_htg_clg_0714",
        "site_sky_temp_0201",
        "site_sky_temp_0504",
        "site_sky_temp_0714",
        "temp_0201",
        "temp_0714",
    ];
    let mut cols: Vec<String> = vec!["bestest.case_num".to_string()];
    for f in scalars.iter().chain(timestamps.iter()) {
        cols.push(format!("{P}.{f}"));
    }
    for f in monthly.iter().chain(day_profiles.iter()) {
        cols.push(format!("{P}.{f}"));
    }
    cols.push(format!("{P}.temp_bins"));
    for f in ["building_name", "dir_name", "status"] {
        cols.push(f.to_string());
    }
    cols.join(",")
}

/// One CSV row for this case. Column 1 is `<case> - <description>` for
/// conditioned cases and bare `<case>` for free-float cases (contract
/// `key_column.value_format`); the consumer keys on the first whitespace
/// token, so the case id is always first.
pub fn section7_gsr_csv_row(r: &Section7CaseResult) -> String {
    let key = if r.free_float {
        r.case_id.clone()
    } else {
        format!("{} - {}", r.case_id, r.description)
    };
    let status = if r.free_float {
        "emitted_free_float"
    } else {
        "emitted"
    };
    let join12_3 = |a: &[f64; 12]| a.iter().map(|v| fmt(*v, 3)).collect::<Vec<_>>().join(",");
    let join12_ts = |a: &[(u32, u32, u32); 12]| {
        a.iter()
            .map(|at| format_dd_mmm_hhmm(*at))
            .collect::<Vec<_>>()
            .join(",")
    };
    let d0201 = day_profile(r, 2, 1);
    let d0504 = day_profile(r, 5, 4);
    let d0714 = day_profile(r, 7, 14);
    let bins = {
        let mut out = "hourly_counts,C_bins".to_string();
        for c in r.temp_bins.iter() {
            out.push(',');
            out.push_str(&c.to_string());
        }
        out
    };
    let fields = [
        // 32 scalars
        fmt(r.annual_cooling_mwh, 3),
        fmt(r.annual_heating_mwh, 3),
        fmt(r.avg_temp_c, 2),
        fmt(r.annual_incident_kwh_m2[2], 2), // east
        fmt(r.annual_incident_kwh_m2[0], 2), // horizontal
        r.max_index_position.to_string(),
        fmt(r.max_temp_c, 2),
        r.min_index_position.to_string(),
        fmt(r.min_temp_c, 2),
        fmt(r.annual_incident_kwh_m2[1], 2), // north
        fmt(r.peak_cooling_kw, 3),
        fmt(r.peak_heating_kw, 3),
        fmt(r.sky_avg_temp_c, 2),
        r.sky_max_index_position.to_string(),
        fmt(r.sky_max_temp_c, 2),
        r.sky_min_index_position.to_string(),
        fmt(r.sky_min_temp_c, 2),
        fmt(r.annual_incident_kwh_m2[3], 2), // south
        join2("0201,roof", &d0201.incident_h_wh_m2, 2),
        join2("0201,south", &d0201.incident_s_wh_m2, 2),
        join2("0201,west", &d0201.incident_w_wh_m2, 2),
        join2("0504,roof", &d0504.incident_h_wh_m2, 2),
        join2("0504,south", &d0504.incident_s_wh_m2, 2),
        join2("0504,west", &d0504.incident_w_wh_m2, 2),
        join2("0714,roof", &d0714.incident_h_wh_m2, 2),
        join2("0714,south", &d0714.incident_s_wh_m2, 2),
        join2("0714,west", &d0714.incident_w_wh_m2, 2),
        join2("0201,south", &d0201.transmitted_south_wh_m2, 2),
        join2("0504,south", &d0504.transmitted_south_wh_m2, 2),
        join2("0714,south", &d0714.transmitted_south_wh_m2, 2),
        fmt(r.annual_incident_kwh_m2[2], 2), // east
        fmt(r.annual_transmitted_south_kwh_m2, 2),
        // 2 timestamps
        format_dd_mmm_hhmm(r.peak_cooling_at),
        format_dd_mmm_hhmm(r.peak_heating_at),
        // 6 monthly arrays
        join12_3(&r.monthly_cooling_mwh),
        join12_ts(&r.monthly_cooling_peak_at),
        join12_3(&r.monthly_cooling_peak_kw),
        join12_3(&r.monthly_heating_mwh),
        join12_ts(&r.monthly_heating_peak_at),
        join12_3(&r.monthly_heating_peak_kw),
        // 7 day-profile arrays
        join2("0201,signed_kWh", &d0201.load_kwh, 4),
        join2("0714,signed_kWh", &d0714.load_kwh, 4),
        join2("0201,sky_temp_C", &d0201.sky_temp_c, 2),
        join2("0504,sky_temp_C", &d0504.sky_temp_c, 2),
        join2("0714,sky_temp_C", &d0714.sky_temp_c, 2),
        join2("temp_C", &d0201.zone_temp_c, 2),
        join2("temp_C", &d0714.zone_temp_c, 2),
        // 1 other packed
        bins,
    ];
    format!(
        "{},\"{}\",{},{},{}",
        key,
        fields.join("\",\""),
        "Fluxion",
        r.case_id,
        status
    )
}

/// Full CSV text for all cases, contract form.
pub fn section7_gsr_csv(results: &[Section7CaseResult]) -> String {
    let mut out = section7_gsr_csv_header();
    out.push('\n');
    for r in results {
        out.push_str(&section7_gsr_csv_row(r));
        out.push('\n');
    }
    out
}

fn committed_gsr_csv() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section7/workflow_results.csv");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed §7 workflow_results.csv unreadable: {e}"))
}

// ===========================================================================
// Tests.
// ===========================================================================

/// The §7 case set is pinned: the 18 implemented ASHRAE140Case variants, no
/// case-id shortcuts, free-float flags exactly on the FF variants.
#[test]
fn section7_case_set_is_pinned() {
    assert_eq!(SECTION7_CASES.len(), 18, "pinned §7 case set");
    let ids: Vec<&str> = SECTION7_CASES.iter().map(|c| c.case_id).collect();
    assert_eq!(
        ids,
        [
            "600", "610", "620", "630", "640", "650", "900", "910", "920", "930", "940", "950",
            "960", "970", "600FF", "650FF", "900FF", "950FF"
        ]
    );
    for c in &SECTION7_CASES {
        assert_eq!(
            c.free_float,
            c.case_id.ends_with("FF"),
            "{}: free_float flag must match the FF suffix",
            c.case_id
        );
    }
}

/// Every case runs the full year on the canonical strict-gate weather and
/// reports finite outputs. Free-float rows carry zero loads by physics.
#[test]
fn section7_all_cases_run_with_finite_outputs() {
    let results = harness_results();
    assert_eq!(results.len(), 18, "all pinned §7 cases must run");
    for r in results {
        assert!(r.annual_heating_mwh.is_finite() && r.annual_cooling_mwh.is_finite());
        assert!(r.peak_heating_kw.is_finite() && r.peak_cooling_kw.is_finite());
        assert!(r.avg_temp_c.is_finite() && r.min_temp_c.is_finite() && r.max_temp_c.is_finite());
        assert!(r
            .annual_incident_kwh_m2
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
        assert!(r.annual_transmitted_south_kwh_m2.is_finite());
        assert!(r.sky_avg_temp_c.is_finite());
        assert_eq!(
            r.temp_bins.iter().sum::<u64>(),
            8760,
            "{}: temp_bins must preserve all 8760 hours",
            r.case_id
        );
        assert_eq!(r.day_profiles.len(), 3);
        if r.free_float {
            assert_eq!(
                r.annual_heating_mwh, 0.0,
                "{}: FF loads must be zero",
                r.case_id
            );
            assert_eq!(
                r.annual_cooling_mwh, 0.0,
                "{}: FF loads must be zero",
                r.case_id
            );
        } else {
            // Conditioned cases carry some ideal load. Cases 650/950 report
            // exactly zero heating on the canonical blind route (observed,
            // round 53; 950's heating-off is §B8.5 by spec, 650's zero is
            // the engine's current blind-path behavior) — so no >0
            // assertion on heating alone.
            assert!(
                r.annual_heating_mwh + r.annual_cooling_mwh > 0.0,
                "{}: conditioned case must carry ideal load",
                r.case_id
            );
        }
    }
}

/// Contract shape: 52 columns, dotted headers, key-first rule, timestamp
/// regex, packed-array lengths (12 monthly, 26 day-profile with two labels,
/// 25 for the one-label temp profiles, 92 temp_bins).
#[test]
fn section7_gsr_csv_shape() {
    let csv = section7_gsr_csv(harness_results());
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    assert_eq!(
        header.len(),
        52,
        "52 columns: key + 48 prefixed + 3 trailing"
    );
    assert_eq!(header[0], "bestest.case_num");
    assert_eq!(
        header[1],
        "bestest_building_thermal_envelope_and_fabric_load_reporting.annual_cooling"
    );
    assert_eq!(header.last().copied(), Some("status"));
    let ts_re = |s: &str| -> bool {
        let b = s.as_bytes();
        b.len() == 12
            && b[0..2].iter().all(u8::is_ascii_digit)
            && b[2] == b'-'
            && b[6] == b'-'
            && b[9] == b':'
            && b[3..6].iter().all(u8::is_ascii_uppercase)
            && b[7..9].iter().all(u8::is_ascii_digit)
            && b[10..12].iter().all(u8::is_ascii_digit)
    };
    let mut seen = std::collections::HashSet::new();
    for line in lines {
        // Packed arrays carry commas, so cells are reassembled by scanning
        // the quoted fields.
        let mut cells: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut in_quotes = false;
        for ch in line.chars() {
            match ch {
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => {
                    cells.push(std::mem::take(&mut cur));
                }
                c => cur.push(c),
            }
        }
        cells.push(cur);
        assert_eq!(cells.len(), 52, "row must have 52 cells: {line}");
        let key = cells[0].split(' ').next().unwrap().to_string();
        assert!(
            !key.is_empty(),
            "consumer keys on the first whitespace token"
        );
        assert!(seen.insert(key.clone()), "duplicate case key {key}");
        // Timestamp cells: peak times (cells 33,34) and the two 12-value
        // monthly peak-time arrays (cells 36, 39).
        assert!(
            ts_re(&cells[33]) && ts_re(&cells[34]),
            "peak times DD-MMM-HH:MM"
        );
        for col in [36usize, 39usize] {
            for ts in cells[col].split(',') {
                assert!(ts_re(ts), "monthly peak time {ts:?} must be DD-MMM-HH:MM");
            }
            assert_eq!(cells[col].split(',').count(), 12);
        }
        for col in [35usize, 37usize, 38usize, 40usize] {
            assert_eq!(
                cells[col].split(',').count(),
                12,
                "monthly array col {}",
                header[col]
            );
        }
        // Day profiles: two labels + 24 values = 26 (cells 41-45);
        // temp profiles: one label + 24 = 25 (cells 46, 47);
        // temp_bins: two labels + 90 bins = 92 (cell 48).
        for col in 41..=45usize {
            assert_eq!(
                cells[col].split(',').count(),
                26,
                "day profile col {}",
                header[col]
            );
        }
        for col in [46usize, 47usize] {
            assert_eq!(
                cells[col].split(',').count(),
                25,
                "temp profile col {}",
                header[col]
            );
        }
        assert_eq!(
            cells[48].split(',').count(),
            92,
            "temp_bins: 2 labels + 90 bins"
        );
    }
    assert_eq!(seen.len(), 18, "18 unique case keys");
}

/// Byte-identity staleness gate: the committed artifact must equal a fresh
/// regeneration from this same harness. Any engine change that moves a §7
/// output fails here until the CSV is re-emitted in the same PR with dated
/// old→new provenance (§6 round-51 discipline).
#[test]
fn section7_gsr_csv_artifact_matches_regeneration() {
    let regenerated = section7_gsr_csv(harness_results());
    assert_eq!(
        regenerated,
        committed_gsr_csv(),
        "committed data/reference/section7/workflow_results.csv is stale — \
         re-emit with cargo test --release --test ashrae_140_section7_gsr_emitter \
         -- --ignored emit_section7_gsr_csv and note the old→new values in the PR"
    );
}

/// Value pins (round 53, 2026-10-17): headline §7 tripwires over the
/// byte-identity gate, recorded from the first emission at `2622b154`+
/// (recorded values read off the committed
/// `data/reference/section7/workflow_results.csv`). Tolerance ±0.5%, peak
/// timestamps exact — same discipline as the §6 pins. When a physics-correct
/// engine change moves a value, update the pin AND re-emit the CSV in the
/// same PR, with the old→new provenance noted in the PR description.
fn pin_check(case_id: &str, actual: f64, expected: f64) {
    // Tolerance: ±0.5% of the pinned value, floored at 0.01 so pins taken
    // from the CSV's 2-decimal fields absorb the field rounding itself.
    let tol = (expected.abs() * 0.005).max(0.01);
    assert!(
        (actual - expected).abs() <= tol.max(1e-9),
        "{case_id}: {actual} drifted outside ±0.5% of pinned {expected}"
    );
}

#[test]
fn section7_headline_value_pins() {
    let results = harness_results();
    let by_id = |id: &str| results.iter().find(|r| r.case_id == id).unwrap();
    // Re-recorded 2026-10-22 (loop round 87): the mass-side night-vent
    // coupling removed from the 5R1C default (air-only vent routing,
    // Alex-approved 2026-10-22 06:41 ET "I approve the vent arm";
    // FLX_R86_VENT_AIR_ONLY; #821 double-count removal, ISO 13790 §C.4;
    // #3983 §66) moved the 650FF rows: −17.41/41.89 → −16.20/46.11
    // (Tmin honestly leaves [−17.8, −16.7], §LIMIT-43). Values cross-check
    // the re-recorded strict_energy_gate_baseline.json (600 5.271/2.939
    // MWh, 600FF −3.93/47.24; 900/900FF/950FF unchanged) and the
    // re-emitted data/reference/section7/workflow_results.csv. Round-85
    // pins (per-class flip, #3983 §65): 650FF −17.41/41.89. Round-78
    // pins (coupled single-N flip, #3983 §63/§64): 600 5.353/2.966, 600FF
    // −3.78/47.42, 650FF −17.40/41.74. Nothing tuned.
    // Re-recorded 2026-10-22 (loop round 89): the ISO 13790 §C.3
    // surface-pool mass share promoted to the engine default
    // (`FLX_R88_PHIST_MASS_SHARE`, Alex-directed 2026-10-22 14:57 ET,
    // round-88 arm #4543) moved the 600-family rows: 600 5.271/2.939 →
    // 4.467/3.124 MWh (heating enters its band), 600FF −3.93/47.24 →
    // −2.56/46.84, 650FF −16.20/46.11 → −15.97/45.46 (cohort drifts
    // slightly away from band — recorded, not tuned, §LIMIT-28). Values
    // cross-check the re-recorded strict_energy_gate_baseline.json
    // (_doc_round89). 900/900FF/950FF unchanged (9R4C inert control).
    // Re-recorded 2026-10-22 (loop round 89b, item 2a): the mass-node
    // full time-coverage fix promoted to the engine default
    // (`FLX_R89_MASS_DT_ALIGN`, flip commit ceb19b0b) moved the
    // 600-family rows: 600 4.467/3.124 -> 5.937/4.700 MWh (heating
    // honestly leaves its band — recorded, §LIMIT-28), 600FF
    // -2.56/46.84 -> -12.85/53.40 (Tmin ENTERS [-13.80, -9.90] for the
    // first time), 650FF -15.97/45.46 -> -17.62/52.79 (Tmin ENTERS
    // [-17.80, -16.70] for the first time). Values cross-check the
    // re-recorded strict_energy_gate_baseline.json (_doc_round89b).
    // 900/900FF/950FF unchanged (9R4C inert control).
    // Re-recorded 2026-10-22 (loop round 90): the R77 solar-lag filter
    // removed from the default route (`FLX_R90_NO_LAG_FILTER` default-on,
    // ISO 13790 §C.3 algebraic surface node, §LIMIT-28 round-90 record)
    // moved the 600-family rows: 600 5.937/4.700 -> 5.959/4.728 MWh,
    // 600FF -12.85/53.40 -> -12.87/53.84 (Tmax gap 18.35 -> 17.68 pp),
    // 650FF -17.62/52.79 -> -17.63/53.09 (Tmax gap 17.45 -> 16.98 pp;
    // both Tmin rows stay in band). Values cross-check the re-recorded
    // strict_energy_gate_baseline.json (_doc_round90). 900/900FF/950FF
    // unchanged (9R4C inert control).
    // Re-recorded 2026-10-22 (loop round 91, flip 1): the ASHRAE 140
    // Table 5-138 window-loss solar sink promoted to the engine default
    // (`FLX_R91_RETRANS`; §5.2.1.12.1 "accounting for solar lost ...
    // retransmitted back out the windows", Table 5-138 row "Solar Lost
    // through Windows 0.035"; round-83 arm left gated on band direction
    // alone, flip under the 2026-10-22 criteria) moved the 600-family
    // rows: 600 5.959/4.728 -> 6.054/4.490 MWh, 600FF -12.87/53.84 ->
    // -12.90/53.05, 650FF -17.63/53.09 -> -17.65/52.25 (both Tmin rows
    // stay in band). Values cross-check the re-recorded
    // strict_energy_gate_baseline.json (_doc_round91a).
    // 900/900FF/950FF unchanged (9R4C inert control).
    // Re-recorded 2026-10-23 (loop round 91, flip 2): the incumbent
    // density-driven solar split retired from the engine default
    // (`FLX_R91_SOL2ST`, §5.2.1.12 + ISO 13790 §C.1-C.3, §LIMIT-28
    // round-91 record) moved the 600-family rows: 600 6.054/4.490 ->
    // 6.915/2.444 MWh, 600FF -12.90/53.05 -> -13.07/44.83 (Tmin stays
    // in band; Tmax gap 18.88 -> 31.45 pp), 650FF -17.65/52.25 ->
    // -17.77/44.32 (Tmin stays in band, 0.03 K from the edge; Tmax
    // gap 18.30 -> 30.70 pp). 620/630 cooling honestly leave their
    // bands — recorded, not tuned. Values cross-check the re-recorded
    // strict_energy_gate_baseline.json (_doc_round91b).
    // 900/900FF/950FF unchanged (9R4C inert control).
    let pins: &[(&str, fn(&Section7CaseResult) -> f64, f64)] = &[
        ("600", |r| r.annual_heating_mwh, 6.915),
        ("600", |r| r.annual_cooling_mwh, 2.444),
        ("900", |r| r.annual_heating_mwh, 1.225),
        ("900", |r| r.annual_cooling_mwh, 2.249),
        ("600FF", |r| r.min_temp_c, -13.07),
        ("600FF", |r| r.max_temp_c, 44.83),
        ("650FF", |r| r.min_temp_c, -17.77),
        ("650FF", |r| r.max_temp_c, 44.32),
        ("900FF", |r| r.min_temp_c, -0.26),
        ("900FF", |r| r.max_temp_c, 43.28),
        ("950FF", |r| r.min_temp_c, -15.67),
        ("950FF", |r| r.max_temp_c, 38.78),
    ];
    for (id, get, expected) in pins {
        pin_check(id, get(by_id(id)), *expected);
    }
}

/// Writes the committed artifact into `data/reference/section7/`.
/// Ignored, category `diagnostic` (QUARANTINE.md row): run explicitly with
/// `cargo test --release --test ashrae_140_section7_gsr_emitter -- --ignored
/// emit_section7_gsr_csv --nocapture`.
#[test]
#[ignore]
fn emit_section7_gsr_csv() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/reference/section7");
    std::fs::create_dir_all(&dir).expect("create section7 dir");
    let csv = section7_gsr_csv(harness_results());
    let path = dir.join("workflow_results.csv");
    std::fs::write(&path, &csv).expect("write §7 workflow_results.csv");
    println!("wrote {}", path.display());
    println!("{csv}");
}

// ===========================================================================
// §6.2.1 conformance-output companion CSV (round 62).
//
// The GSR contract CSV (workflow_results.csv) is the BESTEST-GSR consumer's
// contract surface and stays byte-stable in its 52-column shape; the
// §6.2.1.3/.4/.5 case-window transmitted-solar quantities and Case 960's
// §6.2.1.6 sun-zone free-float temperatures are emitted here instead, in a
// companion artifact with its own byte-identity gate. The §7 conformance
// table (ashrae_140_section7_conformance.rs) reads this file to upgrade the
// 6.2.1.3 (610) / 6.2.1.4 (620) / 6.2.1.5 (630) and 960's 6.2.1.6 cells.
// ===========================================================================

/// Header of the companion CSV. One row per pinned §7 case, in series
/// order; inapplicable cells are empty.
pub fn section621_outputs_csv_header() -> String {
    [
        "case",
        "req_6213_shaded_south_transmitted_kwh_m2",
        "req_6214_west_transmitted_kwh_m2",
        "req_6215_shaded_west_transmitted_kwh_m2",
        "sun_zone_min_temp_C",
        "sun_zone_min_index",
        "sun_zone_max_temp_C",
        "sun_zone_max_index",
        "sun_zone_mean_temp_C",
    ]
    .join(",")
}

fn section621_outputs_csv_row(r: &Section7CaseResult) -> String {
    let f4 = |v: Option<f64>| v.map(|x| fmt(x, 4)).unwrap_or_default();
    let sun = match &r.sun_zone {
        Some(s) => format!(
            "{:.2},{},{:.2},{},{:.2}",
            s.min_temp_c, s.min_index, s.max_temp_c, s.max_index, s.mean_temp_c
        ),
        None => ",,,,".to_string(),
    };
    [
        r.case_id.clone(),
        f4(r.req_6213_shaded_south_transmitted_kwh_m2),
        f4(r.req_6214_west_transmitted_kwh_m2),
        f4(r.req_6215_shaded_west_transmitted_kwh_m2),
        sun,
    ]
    .join(",")
}

/// Full companion CSV text for all cases, series order.
pub fn section621_outputs_csv(results: &[Section7CaseResult]) -> String {
    let mut out = section621_outputs_csv_header();
    out.push('\n');
    for r in results {
        out.push_str(&section621_outputs_csv_row(r));
        out.push('\n');
    }
    out
}

fn committed_section621_csv() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section7/section621_outputs.csv");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed §6.2.1 companion CSV unreadable: {e}"))
}

/// Shape + physics sanity of the companion CSV: the three shaded/west
/// transmitted cells are present exactly on cases 610/620/630 and finite
/// non-negative there; the sun-zone stats are present exactly on 960,
/// finite, with 8760-range indices and min ≤ mean ≤ max.
#[test]
fn section621_outputs_shape_and_cases() {
    let results = harness_results();
    assert_eq!(results.len(), 18);
    for r in results {
        let reqs = [
            r.req_6213_shaded_south_transmitted_kwh_m2,
            r.req_6214_west_transmitted_kwh_m2,
            r.req_6215_shaded_west_transmitted_kwh_m2,
        ];
        let expected_some: [bool; 3] = [r.case_id == "610", r.case_id == "620", r.case_id == "630"];
        for (v, want) in reqs.iter().zip(expected_some.iter()) {
            assert_eq!(
                v.is_some(),
                *want,
                "{}: §6.2.1.3/.4/.5 emitted-cell presence",
                r.case_id
            );
            if let Some(v) = v {
                assert!(
                    v.is_finite() && *v >= 0.0,
                    "{}: transmitted kWh/m²",
                    r.case_id
                );
            }
        }
        match &r.sun_zone {
            Some(s) => {
                assert_eq!(r.case_id, "960", "sun-zone stats only on Case 960");
                assert!(s.min_temp_c.is_finite() && s.max_temp_c.is_finite());
                assert!(s.mean_temp_c.is_finite());
                assert!(s.min_index <= 8759 && s.max_index <= 8759);
                assert!(
                    s.min_temp_c <= s.mean_temp_c + 1e-9 && s.mean_temp_c <= s.max_temp_c + 1e-9,
                    "960: sun-zone min ≤ mean ≤ max"
                );
            }
            None => assert_ne!(r.case_id, "960", "960 must carry sun-zone stats"),
        }
    }
}

/// The companion CSV must round-trip against the committed artifact
/// byte-for-byte.
#[test]
fn section621_outputs_csv_artifact_matches_regeneration() {
    let csv = section621_outputs_csv(harness_results());
    assert_eq!(
        csv,
        committed_section621_csv(),
        "committed §6.2.1 companion CSV is stale — re-emit"
    );
}

/// Value pins for the round-62 emitted quantities (provenance: recorded
/// 2026-10-18, loop round 62, develop `88da4b1d`, canonical DRYCOLD blind
/// §7 route — first emission of these cells). Record pins, not validator
/// bands: drift means the emission route moved and is re-recorded
/// honestly, never tuned.
#[test]
fn section621_value_pins() {
    let results = harness_results();
    let by_id = |id: &str| results.iter().find(|r| r.case_id == id).unwrap();
    // Case-window annual transmitted solar, kWh/m² of glazing.
    pin_check(
        "610",
        by_id("610")
            .req_6213_shaded_south_transmitted_kwh_m2
            .unwrap(),
        529.44,
    );
    pin_check(
        "620",
        by_id("620").req_6214_west_transmitted_kwh_m2.unwrap(),
        457.30,
    );
    pin_check(
        "630",
        by_id("630")
            .req_6215_shaded_west_transmitted_kwh_m2
            .unwrap(),
        324.44,
    );
    // Cross-checks against the round-48/53 reference quantities: the
    // shaded-south figure must sit below the 6.2.1.2.2 unshaded south
    // reference (774 kWh/m²), and shading must reduce the west figure
    // (630 vs 620).
    let ref_south = by_id("600").annual_transmitted_south_kwh_m2;
    assert!(
        by_id("610")
            .req_6213_shaded_south_transmitted_kwh_m2
            .unwrap()
            < ref_south,
        "610 shaded south must be below the unshaded {ref_south} reference"
    );
    assert!(
        by_id("630")
            .req_6215_shaded_west_transmitted_kwh_m2
            .unwrap()
            < by_id("620").req_6214_west_transmitted_kwh_m2.unwrap(),
        "630 shaded west must be below 620 unshaded west"
    );
    // Case 960 sun zone (°C, 8760-index occurrences).
    let sun = by_id("960").sun_zone.as_ref().unwrap();
    pin_check("960", sun.min_temp_c, 5.42);
    pin_check("960", sun.max_temp_c, 50.23);
    pin_check("960", sun.mean_temp_c, 29.51);
    assert_eq!(sun.min_index, 941);
    assert_eq!(sun.max_index, 6086);
}

/// Writes the committed §6.2.1 companion CSV. Run explicitly:
/// `cargo test --release --test ashrae_140_section7_gsr_emitter -- --ignored
/// emit_section621_outputs --nocapture`.
#[test]
#[ignore]
fn emit_section621_outputs() {
    let csv = section621_outputs_csv(harness_results());
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section7/section621_outputs.csv");
    std::fs::write(&path, &csv).expect("write §6.2.1 companion CSV");
    println!("wrote {}", path.display());
    println!("{csv}");
}

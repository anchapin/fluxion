//! ASHRAE 140-2023 §9 CE100–CE200 engine-run harness + per-case acceptance
//! band scoring — Issue #4296 (slice: "engine-run harness + per-case
//! acceptance bands").
//!
//! # What this module does
//!
//! `section9_ce.rs` landed the §9 equipment layer: the CE100 performance map
//! (`fluxion::sim::hvac::CePerformanceMap`, Tables 9-5…9-10 of 140-2023) and
//! one steady-state dispatch (`dispatch_ce_steady_state`). It dispatched a
//! single case (CE200) because the other 13 have no transcribed entering
//! wet-bulb, and it asserted physics invariants only — no band, because the
//! §9 analytical slot is `null` and the Task-22 comparative bounds are
//! labelled "evidence range …, not a pass/fail criterion".
//!
//! This module closes the remaining slice:
//!
//! 1. **Every** CE case (CE100–CE200, 14 of them) is dispatched through the
//!    same engine equipment path (`CePerformanceMap::evaluate` via
//!    `dispatch_ce_steady_state`) — deterministically, with no weather file
//!    and no synthetic data.
//! 2. Each case's result is scored against **that case's own** band, read
//!    from the generated case table
//!    (`data/reference/ashrae140/section9_ce_cases.json` → `acceptance`).
//!    Every scored row records the band kind (`analytical` vs `comparative`)
//!    and the band's source string, exactly as committed.
//! 3. Rows that fall outside their band are **recorded as known-fail rows
//!    with the real numbers** (engine value, band edges, gap) in
//!    `data/section9_engine_run_results.json`, which this module verifies
//!    against the live engine run on every test execution.
//!
//! Nothing here fits a band to the engine output and nothing widens a band:
//! the bands are read verbatim from the committed artefacts.
//!
//! # The entering wet-bulb: what is and is not known
//!
//! The map is tabulated over (ODB × EWB) with EDB columns. A CE case fixes
//! its ODB (and its setpoint = EDB), but **its own EWB is not transcribed**
//! for 13 of the 14 cases: the standard's §9 input specification lives in the
//! licensed normative materials (Issue #4285), and the tree has no
//! per-case entering condition. The single exception is CE200, whose
//! condition *is* the map's own AHRI rating point (EWB 19.4 °C).
//!
//! Rather than invent an entering condition (RULES.md: nothing invented), the
//! harness dispatches each case at **every entering wet-bulb column the map
//! itself tabulates** (15.0 / 17.2 / 19.4 / 21.7 °C, read from the committed
//! map artefact, not hard-coded) and reports the resulting envelope. The
//! per-case EWB remains an open transcription item (methodology follow-up F4,
//! blocked on #4285) — the harness does not claim any column is "the" case
//! condition.
//!
//! # Band metrics and why most of them cannot be scored
//!
//! The Task-22 comparative bounds carry two metrics per case:
//! `annual_cooling` (MWh) and `peak_cooling` (kW); CE150 adds
//! `max_zone_temp` (°C).
//!
//! * `annual_cooling` is an **annual** quantity. It needs the normative TMY2
//!   weather file the standard binds the case to (`CE100A`, `CE110A`,
//!   `CE165A`, `CE200A`) — licensed supplemental material that is never
//!   committed to this repository (see `fluxion-core/src/weather/tm2.rs`).
//!   A steady-state equipment dispatch has no duty cycle, so this metric is
//!   recorded as **not reproducible**, with the band values shown.
//! * `max_zone_temp` is a zone-temperature trajectory output of an annual
//!   run; a steady-state dispatch holds the case setpoint by construction.
//!   Also **not reproducible**.
//! * `peak_cooling` (kW) is the only band metric a steady-state equipment
//!   dispatch can produce. It is scored against the engine's **full-load
//!   cooling power at the case condition** (compressor power + the map's
//!   230 W indoor fan heat, adjusted-net basis). The ensemble's own metric
//!   semantics (peak cooling *load* vs peak cooling *power*) are **not
//!   verifiable from this tree** — the bounds CSV records `source_page` as
//!   `pending` — so every scored row carries that caveat, and the engine's
//!   required cooling load (the alternative reading) is recorded alongside.
//!
//! # Known-fail convention
//!
//! The recorded results file follows the repository's strict-gate convention
//! (`tests/reference_data/zone_balance/strict_energy_gate_baseline.json`):
//! per metric, the measured value, the published band, the gap as a
//! percentage of the band midpoint (`gap_pct_of_mid`, 0 inside the band),
//! and a status. `pass` / `known_fail` / `not_reproducible` / `no_band` are
//! the four statuses used here. A `known_fail` row is a *documented*
//! structural gap, not a hidden failure: the numbers are in the file, in
//! `docs/KNOWN_ISSUES.md` (§LIMIT-42) and in the §9a row of
//! `docs/compliance/section_coverage.md`.
//!
//! # Re-recording the results file
//!
//! ```text
//! FLUXION_SECTION9_WRITE_RESULTS=tests/validation/hvac_equipment_rated_performance/data/section9_engine_run_results.json \
//!   cargo test --test hvac_equipment_rated_performance section9_engine_run
//! ```
//!
//! The re-record path is explicit and opt-in; the ordinary test run always
//! compares the live engine run against the committed file.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use fluxion::sim::hvac::{CePerformanceMap, CeTable};

use super::section9_ce::dispatch_ce_steady_state;

const SECTION9_CASES_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/reference/ashrae140/section9_ce_cases.json"
));
const SECTION9_MAP_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/reference/ashrae140/section9_ce100_performance_map.json"
));
/// The recorded per-case engine-run rows (the known-fail record).
const SECTION9_RESULTS_JSON: &str = include_str!("data/section9_engine_run_results.json");

/// Opt-in path for re-recording the results file (see the module docs).
const WRITE_RESULTS_ENV: &str = "FLUXION_SECTION9_WRITE_RESULTS";

/// Relative tolerance for the live-vs-recorded comparison (the recorded file
/// stores values rounded to 1e-6).
const RECORDED_RTOL: f64 = 1e-6;
/// Absolute tolerance for the live-vs-recorded comparison.
const RECORDED_ATOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Committed artefacts
// ---------------------------------------------------------------------------

fn load_cases() -> Vec<Value> {
    let root: Value = serde_json::from_str(SECTION9_CASES_JSON).expect("parse §9 CE cases");
    root["cases"].as_array().expect("cases array").clone()
}

fn load_map() -> CePerformanceMap {
    CePerformanceMap::from_json_str(SECTION9_MAP_JSON).expect("parse §9 CE100 performance map")
}

fn map_json() -> Value {
    serde_json::from_str(SECTION9_MAP_JSON).expect("parse §9 CE100 performance map json")
}

/// The map's own tabulated entering wet-bulb columns (°C), read from the
/// committed map artefact rather than restated in code.
fn map_ewb_columns() -> Vec<f64> {
    let root = map_json();
    let table = &root["tables"]["adjusted_net_Table_26e"];
    let row = table
        .as_object()
        .expect("adjusted-net table object")
        .values()
        .next()
        .expect("adjusted-net table has an ODB row");
    let mut cols: Vec<f64> = row
        .as_object()
        .expect("EWB row object")
        .keys()
        .filter_map(|k| k.parse::<f64>().ok())
        .collect();
    cols.sort_by(|a, b| a.partial_cmp(b).expect("finite EWB"));
    assert!(!cols.is_empty(), "map exposes no EWB column");
    cols
}

/// Indoor fan heat (W) on the adjusted-net basis, from the map artefact.
fn fan_heat_w() -> f64 {
    map_json()["fan_heat"]["actual_W"]
        .as_f64()
        .expect("fan_heat.actual_W")
}

// ---------------------------------------------------------------------------
// One engine dispatch
// ---------------------------------------------------------------------------

/// One §9 CE case dispatched at one tabulated entering wet-bulb.
#[derive(Debug, Clone)]
struct CaseRun {
    entering_wet_bulb_c: f64,
    outdoor_dry_bulb_c: f64,
    entering_dry_bulb_c: f64,
    /// Full-load total (sensible + latent) capacity at the condition [W].
    total_capacity_w: f64,
    /// Full-load sensible capacity at the condition [W].
    sensible_capacity_w: f64,
    /// Full-load latent capacity (total − sensible) at the condition [W].
    latent_capacity_w: f64,
    /// Full-load compressor power at the condition [W].
    compressor_power_w: f64,
    /// Full-load cooling power (compressor + indoor fan) [kW] — the quantity
    /// scored against the `peak_cooling` band.
    peak_cooling_power_kw: f64,
    /// The case's required cooling load (sensible + latent gains) [W].
    required_load_w: f64,
    /// The case's required load [kW] — the alternative reading of the
    /// `peak_cooling` band metric, recorded for transparency.
    required_load_kw: f64,
    /// Part-load ratio from the load balance (clamped to [0, 1]).
    plr: f64,
    /// Delivered capacity at the operating point [W].
    delivered_capacity_w: f64,
    /// Delivered compressor power at the operating point [W].
    delivered_power_w: f64,
    /// Shortfall of the delivered load against the required load [W]; zero
    /// when the unit meets the case load at this entering condition.
    capacity_shortfall_w: f64,
    /// Sensible heat ratio at the operating point.
    shr: f64,
    /// COP at the operating point (total capacity / (compressor + fan)).
    cop: f64,
}

/// Dispatch one CE case at one entering wet-bulb through the engine's DX
/// performance map (the same path `dispatch_ce_steady_state` uses).
fn run_case(case: &Value, map: &CePerformanceMap, ewb_c: f64, fan_w: f64) -> CaseRun {
    let cond = &case["conditions"];
    let odb_c = cond["outdoor_dry_bulb_C"].as_f64().expect("ODB");
    let edb_c = cond["cooling_setpoint_C"].as_f64().expect("setpoint");
    let full = map.evaluate(CeTable::AdjustedNet26e, ewb_c, odb_c, edb_c, 1.0, fan_w);
    let dispatch = dispatch_ce_steady_state(case, map, ewb_c, fan_w);
    CaseRun {
        entering_wet_bulb_c: ewb_c,
        outdoor_dry_bulb_c: odb_c,
        entering_dry_bulb_c: edb_c,
        total_capacity_w: full.total_capacity_w,
        sensible_capacity_w: full.sensible_capacity_w,
        latent_capacity_w: full.total_capacity_w - full.sensible_capacity_w,
        compressor_power_w: full.compressor_power_w,
        peak_cooling_power_kw: (full.compressor_power_w + fan_w) / 1000.0,
        required_load_w: dispatch.required_load_w,
        required_load_kw: dispatch.required_load_w / 1000.0,
        plr: dispatch.plr,
        delivered_capacity_w: dispatch.delivered_capacity_w,
        delivered_power_w: dispatch.delivered_power_w,
        capacity_shortfall_w: (dispatch.required_load_w - dispatch.delivered_capacity_w).max(0.0),
        shr: dispatch.shr,
        cop: dispatch.cop,
    }
}

// ---------------------------------------------------------------------------
// Band scoring
// ---------------------------------------------------------------------------

/// Gap as a percentage of the band midpoint — the repository's strict-gate
/// convention (`scripts/check_strict_energy_gate_regression.py`): 0 inside
/// the band, otherwise the distance outside it as a percentage of the
/// midpoint.
fn gap_pct_of_mid(value: f64, low: f64, high: f64) -> f64 {
    let mid = 0.5 * (low + high);
    if value < low {
        if mid <= 0.0 {
            f64::INFINITY
        } else {
            (low - value) / mid * 100.0
        }
    } else if value > high {
        if mid <= 0.0 {
            f64::INFINITY
        } else {
            (value - high) / mid * 100.0
        }
    } else {
        0.0
    }
}

fn direction(value: f64, low: f64, high: f64) -> &'static str {
    if value < low {
        "below"
    } else if value > high {
        "above"
    } else {
        "inside"
    }
}

/// Reason recorded on every `not_reproducible` row: the band metric is an
/// output of the annual case run, which needs the normative weather file.
fn not_reproducible_reason(metric: &str, case_id: &str, weather_file: &str) -> String {
    match metric {
        "annual_cooling" => format!(
            "annual total: the case's normative TMY2 weather file ({weather_file}) is licensed \
             ASHRAE supplemental material and is never committed to this repository \
             (fluxion-core/src/weather/tm2.rs); a steady-state equipment dispatch has no duty \
             cycle, so the annual band cannot be reproduced from the engine run. Tracked as a \
             coverage gap for {case_id} (methodology follow-up F1, licensed gate #4285)."
        ),
        "max_zone_temp" | "min_zone_temp" => format!(
            "zone-temperature trajectory output of the annual case run; the steady-state dispatch \
             holds the case setpoint by construction, so no trajectory exists to compare. \
             Tracked as a coverage gap for {case_id} (methodology follow-up F1, #4285)."
        ),
        other => format!(
            "band metric '{other}' is not produced by the steady-state equipment dispatch \
             (case {case_id}); it is an output of the annual case run and needs the normative \
             weather file (methodology follow-up F1, #4285)."
        ),
    }
}

/// Build the scored band rows for one case from its committed `acceptance`
/// block. Returns one row per band metric, so no metric is silently dropped.
fn band_rows(case: &Value, runs: &[CaseRun]) -> Vec<Value> {
    let case_id = case["case_id"].as_str().expect("case_id").to_string();
    let weather_file = case["weather_file"]
        .as_str()
        .unwrap_or("(none)")
        .to_string();
    let acceptance = &case["acceptance"];
    let mut rows: Vec<Value> = Vec::new();

    // Analytical slot first: it is the primary criterion when the standard
    // publishes a quasi-analytical solution. For §9 it is `null` today (the
    // 140-2023 §9 normative materials are licensed, #4285), so the
    // comparative slot below is what actually scores.
    for (kind, slot) in [
        ("analytical", &acceptance["analytical"]),
        ("comparative", &acceptance["comparative"]),
    ] {
        if slot.is_null() {
            continue;
        }
        let source = slot["source"].as_str().unwrap_or("").to_string();
        let role = slot["role"].as_str().unwrap_or("").to_string();
        let bounds = slot["bounds"]
            .as_object()
            .unwrap_or_else(|| panic!("{kind} band for {case_id} has no bounds object"));
        for (metric, band) in bounds {
            let low = band[0].as_f64().expect("band low");
            let high = band[1].as_f64().expect("band high");
            let unit = band[2].as_str().unwrap_or("").to_string();
            let mut row = Map::new();
            row.insert("metric".into(), json!(metric));
            row.insert("band_kind".into(), json!(kind));
            row.insert("band_source".into(), json!(source));
            row.insert("band_role".into(), json!(role));
            row.insert("band_low".into(), json!(round6(low)));
            row.insert("band_high".into(), json!(round6(high)));
            row.insert("band_unit".into(), json!(unit));

            if metric == "peak_cooling" {
                // The only band metric a steady-state equipment dispatch can
                // produce: compare the engine's full-load cooling power at
                // the case condition against the ensemble range.
                let mut inside: Vec<f64> = Vec::new();
                let mut outside: Vec<f64> = Vec::new();
                let mut worst: Option<&CaseRun> = None;
                let mut worst_gap = -1.0_f64;
                for run in runs {
                    let gap = gap_pct_of_mid(run.peak_cooling_power_kw, low, high);
                    if gap == 0.0 {
                        inside.push(run.entering_wet_bulb_c);
                    } else {
                        outside.push(run.entering_wet_bulb_c);
                    }
                    if gap > worst_gap {
                        worst_gap = gap;
                        worst = Some(run);
                    }
                }
                let worst = worst.expect("at least one entering condition");
                let values: Vec<f64> = runs.iter().map(|r| r.peak_cooling_power_kw).collect();
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let status = if outside.is_empty() {
                    "pass"
                } else {
                    "known_fail"
                };
                row.insert("engine_metric".into(), json!("peak_cooling_power"));
                row.insert(
                    "engine_metric_definition".into(),
                    json!(
                        "full-load cooling power at the case condition = compressor power at \
                         PLR=1 + the map's 230 W indoor fan heat (adjusted-net basis)"
                    ),
                );
                row.insert(
                    "engine_value_kw".into(),
                    json!(round6(worst.peak_cooling_power_kw)),
                );
                row.insert(
                    "engine_value_range_kw".into(),
                    json!([round6(min), round6(max)]),
                );
                row.insert(
                    "engine_required_load_kw".into(),
                    json!(round6(worst.required_load_kw)),
                );
                row.insert("gap_pct_of_mid".into(), json!(round6(worst_gap)));
                row.insert(
                    "direction".into(),
                    json!(direction(worst.peak_cooling_power_kw, low, high)),
                );
                row.insert("inside_at_ewb_C".into(), json!(inside));
                row.insert("outside_at_ewb_C".into(), json!(outside));
                row.insert("status".into(), json!(status));
                row.insert(
                    "reason".into(),
                    json!(
                        "comparative band metric scored against the steady-state engine run. The \
                         ensemble metric semantics (peak cooling load vs peak cooling power) are \
                         not verifiable from this tree — the bounds CSV records source_page \
                         'pending' — and the value ladder in that file is placeholder-patterned \
                         (the CE100 peak_cooling row is byte-identical to the AE300 row of \
                         comparative_bounds_ae101_ae445.csv). The band was read verbatim and not \
                         widened; the case's own entering wet-bulb is untranscribed, so the row \
                         covers every entering condition the map tabulates."
                    ),
                );
            } else {
                row.insert("engine_metric".into(), json!(Value::Null));
                row.insert("engine_value_kw".into(), json!(Value::Null));
                row.insert("engine_value_range_kw".into(), json!(Value::Null));
                row.insert("engine_required_load_kw".into(), json!(Value::Null));
                row.insert("gap_pct_of_mid".into(), json!(Value::Null));
                row.insert("direction".into(), json!("not_reproducible"));
                row.insert("inside_at_ewb_C".into(), json!([]));
                row.insert("outside_at_ewb_C".into(), json!([]));
                row.insert("status".into(), json!("not_reproducible"));
                row.insert(
                    "reason".into(),
                    json!(not_reproducible_reason(metric, &case_id, &weather_file)),
                );
            }
            rows.push(Value::Object(row));
        }
    }
    rows
}

/// Round to 1e-6 so the recorded file stays readable and stable.
fn round6(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    (x * 1e6).round() / 1e6
}

/// The whole §9 engine run: every case, every tabulated entering condition,
/// every band row. This is the single source of the recorded results file.
fn build_results() -> Value {
    let cases = load_cases();
    let map = load_map();
    let fan_w = fan_heat_w();
    let ewbs = map_ewb_columns();

    let mut case_rows: Vec<Value> = Vec::new();
    let mut totals = BTreeMap::new();
    let mut engine_runs = 0_u64;
    let mut scored_rows = 0_u64;
    let mut pass_rows = 0_u64;
    let mut known_fail_rows = 0_u64;
    let mut not_reproducible_rows = 0_u64;
    let mut no_band_cases = 0_u64;
    let mut capacity_bound_runs = 0_u64;

    for case in &cases {
        let case_id = case["case_id"].as_str().expect("case_id").to_string();
        let runs: Vec<CaseRun> = ewbs
            .iter()
            .map(|ewb| run_case(case, &map, *ewb, fan_w))
            .collect();
        engine_runs += runs.len() as u64;
        capacity_bound_runs += runs.iter().filter(|r| r.capacity_shortfall_w > 0.0).count() as u64;

        let runs_json: Vec<Value> = runs
            .iter()
            .map(|r| {
                json!({
                    "entering_wet_bulb_C": round6(r.entering_wet_bulb_c),
                    "outdoor_dry_bulb_C": round6(r.outdoor_dry_bulb_c),
                    "entering_dry_bulb_C": round6(r.entering_dry_bulb_c),
                    "total_capacity_w": round6(r.total_capacity_w),
                    "sensible_capacity_w": round6(r.sensible_capacity_w),
                    "latent_capacity_w": round6(r.latent_capacity_w),
                    "compressor_power_w": round6(r.compressor_power_w),
                    "peak_cooling_power_kw": round6(r.peak_cooling_power_kw),
                    "required_load_w": round6(r.required_load_w),
                    "required_load_kw": round6(r.required_load_kw),
                    "plr": round6(r.plr),
                    "delivered_capacity_w": round6(r.delivered_capacity_w),
                    "delivered_power_w": round6(r.delivered_power_w),
                    "capacity_shortfall_w": round6(r.capacity_shortfall_w),
                    "shr": round6(r.shr),
                    "cop": round6(r.cop),
                })
            })
            .collect();

        let rows = band_rows(case, &runs);
        for row in &rows {
            scored_rows += 1;
            match row["status"].as_str().unwrap_or("") {
                "pass" => pass_rows += 1,
                "known_fail" => known_fail_rows += 1,
                "not_reproducible" => not_reproducible_rows += 1,
                _ => {}
            }
        }
        let status = if rows.is_empty() {
            no_band_cases += 1;
            "no_band"
        } else if rows
            .iter()
            .any(|r| r["status"].as_str() == Some("known_fail"))
        {
            "known_fail"
        } else if rows
            .iter()
            .all(|r| r["status"].as_str() == Some("not_reproducible"))
        {
            "not_reproducible"
        } else {
            "pass"
        };

        case_rows.push(json!({
            "case_id": case_id,
            "series": case["series"].clone(),
            "conditions": case["conditions"].clone(),
            "weather_file": case["weather_file"].clone(),
            "entering_wet_bulb_sweep_C": ewbs.iter().map(|e| round6(*e)).collect::<Vec<f64>>(),
            "runs": runs_json,
            "bands": rows,
            "status": status,
        }));
    }

    totals.insert("cases".to_string(), json!(case_rows.len()));
    totals.insert("engine_runs".to_string(), json!(engine_runs));
    totals.insert("band_rows".to_string(), json!(scored_rows));
    totals.insert("pass_rows".to_string(), json!(pass_rows));
    totals.insert("known_fail_rows".to_string(), json!(known_fail_rows));
    totals.insert(
        "not_reproducible_rows".to_string(),
        json!(not_reproducible_rows),
    );
    totals.insert("no_band_cases".to_string(), json!(no_band_cases));
    totals.insert(
        "capacity_bound_runs".to_string(),
        json!(capacity_bound_runs),
    );

    json!({
        "_doc": "Recorded ASHRAE 140-2023 Section 9 CE100-CE200 engine-run results and per-case \
                 acceptance-band rows (Issue #4296). GENERATED by the §9 engine-run harness \
                 (tests/validation/hvac_equipment_rated_performance/section9_engine_run.rs) — \
                 re-record with FLUXION_SECTION9_WRITE_RESULTS; never hand-edit.",
        "_case_table": "data/reference/ashrae140/section9_ce_cases.json (generated by \
                        scripts/generate_section9_ce_cases.py)",
        "_performance_map": "data/reference/ashrae140/section9_ce100_performance_map.json \
                             (Tables 9-5..9-10 of 140-2023)",
        "_entering_condition_policy": "The per-case entering wet-bulb is not transcribed for 13 \
                                       of the 14 CE cases (the 140-2023 §9 input specification is \
                                       licensed, #4285). The harness dispatches every case at \
                                       every entering wet-bulb column the map tabulates and \
                                       reports the envelope; it does not attribute a column to a \
                                       case.",
        "_band_policy": "Bands are read verbatim from each case's `acceptance` block in the \
                        generated case table: the `analytical` slot is the primary criterion \
                        where the standard publishes a quasi-analytical solution (null for every \
                        §9 case today), the `comparative` slot is the Task-22 ensemble evidence \
                        range otherwise. No band is fitted to the engine output and no band is \
                        widened.",
        "_metric_mapping": {
            "peak_cooling": "scored against the engine's full-load cooling power at the case \
                             condition (compressor power + 230 W indoor fan, adjusted-net basis). \
                             The ensemble semantics (peak cooling load vs peak cooling power) \
                             are unverified (bounds CSV source_page = pending).",
            "annual_cooling": "not reproducible: needs the normative TMY2 weather file (licensed, \
                               never committed); a steady-state dispatch has no duty cycle.",
            "max_zone_temp": "not reproducible: needs the annual zone-temperature trajectory; the \
                              steady-state dispatch holds the case setpoint."
        },
        "_status_legend": {
            "pass": "the engine value falls inside the band",
            "known_fail": "the engine value falls outside the band — recorded with the real \
                           numbers, tracked in docs/KNOWN_ISSUES.md §LIMIT-42",
            "not_reproducible": "the band metric cannot be produced by this run (coverage gap)",
            "no_band": "the case carries no band at all (comparative bounds missing)"
        },
        "totals": Value::Object(totals.into_iter().collect()),
        "cases": case_rows,
    })
}

// ---------------------------------------------------------------------------
// Live-vs-recorded comparison
// ---------------------------------------------------------------------------

/// Recursive comparison of two JSON documents; numbers are compared with a
/// tolerance (the recorded file rounds to 1e-6), everything else exactly.
fn json_diff(live: &Value, recorded: &Value, path: &str) -> Vec<String> {
    let mut diffs = Vec::new();
    match (live, recorded) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (
                a.as_f64().unwrap_or(f64::NAN),
                b.as_f64().unwrap_or(f64::NAN),
            );
            let tol = RECORDED_ATOL + RECORDED_RTOL * a.abs().max(b.abs());
            if !((a - b).abs() <= tol) {
                diffs.push(format!("{path}: live {a} vs recorded {b}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                match (a.get(key), b.get(key)) {
                    (Some(x), Some(y)) => diffs.extend(json_diff(x, y, &format!("{path}.{key}"))),
                    (Some(_), None) => {
                        diffs.push(format!("{path}.{key}: missing in recorded file"))
                    }
                    (None, Some(_)) => diffs.push(format!("{path}.{key}: missing in live run")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                diffs.push(format!(
                    "{path}: live has {} entries, recorded has {}",
                    a.len(),
                    b.len()
                ));
            } else {
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    diffs.extend(json_diff(x, y, &format!("{path}[{i}]")));
                }
            }
        }
        (x, y) => {
            if x != y {
                diffs.push(format!("{path}: live {x} vs recorded {y}"));
            }
        }
    }
    diffs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Every CE case in the generated table is exercised by the engine run, at
/// every entering condition the map tabulates, and the generated table's own
/// `engine_coverage.cases_run` agrees with what actually ran.
#[test]
fn test_section9_engine_run_covers_every_ce_case() {
    let cases = load_cases();
    let live = build_results();
    let rows = live["cases"].as_array().expect("cases array");
    assert_eq!(
        rows.len(),
        cases.len(),
        "every §9 CE case must be exercised"
    );
    assert_eq!(rows.len(), 14, "the §9 CE family is CE100–CE200 (14 cases)");

    let case_ids: Vec<&str> = cases
        .iter()
        .map(|c| c["case_id"].as_str().expect("case_id"))
        .collect();
    for id in &case_ids {
        assert!(
            rows.iter().any(|r| r["case_id"] == *id),
            "case {id} missing from the engine run"
        );
    }

    // Each case is dispatched at every tabulated entering wet-bulb.
    let ewbs = map_ewb_columns();
    assert_eq!(ewbs.len(), 4, "the map tabulates four EWB columns");
    for row in rows {
        let runs = row["runs"].as_array().expect("runs array");
        assert_eq!(
            runs.len(),
            ewbs.len(),
            "case {} was not dispatched at every tabulated entering condition",
            row["case_id"]
        );
        for (run, ewb) in runs.iter().zip(ewbs.iter()) {
            assert!(
                (run["entering_wet_bulb_C"].as_f64().expect("ewb") - ewb).abs() < 1e-9,
                "case {} dispatched at an unexpected entering condition",
                row["case_id"]
            );
        }
    }

    // The generated case table's coverage claim must match the run.
    let root: Value = serde_json::from_str(SECTION9_CASES_JSON).expect("parse §9 CE cases");
    let claimed = root["engine_coverage"]["cases_run"]
        .as_u64()
        .expect("engine_coverage.cases_run");
    assert_eq!(
        claimed,
        rows.len() as u64,
        "section9_ce_cases.json engine_coverage.cases_run must match the engine run"
    );
}

/// The engine run is deterministic: two independent runs produce byte-identical
/// results.
#[test]
fn test_section9_engine_run_is_deterministic() {
    let first = build_results();
    let second = build_results();
    assert_eq!(
        serde_json::to_string(&first).expect("serialize"),
        serde_json::to_string(&second).expect("serialize"),
        "the §9 engine run must be deterministic"
    );
}

/// The recorded results file matches the live engine run — every recorded
/// number is re-measured, so a known-fail row cannot go stale silently.
#[test]
fn test_section9_recorded_results_match_live_engine_run() {
    let live = build_results();
    if let Ok(path) = std::env::var(WRITE_RESULTS_ENV) {
        let text = serde_json::to_string_pretty(&live).expect("serialize") + "\n";
        std::fs::write(&path, text).expect("write §9 engine-run results");
        println!("re-recorded §9 engine-run results to {path}");
        return;
    }
    let recorded: Value =
        serde_json::from_str(SECTION9_RESULTS_JSON).expect("parse recorded §9 engine-run results");
    let diffs = json_diff(&live, &recorded, "results");
    assert!(
        diffs.is_empty(),
        "recorded §9 engine-run results differ from the live run (re-record with \
         FLUXION_SECTION9_WRITE_RESULTS):\n  {}",
        diffs.join("\n  ")
    );
}

/// Every scored row declares its band kind (analytical vs comparative) and the
/// band's source, and the §9 analytical slot is still `null` for every case —
/// if a quasi-analytical criterion is transcribed (#4285), this test fails
/// loudly so the harness can be extended to score it.
#[test]
fn test_section9_band_rows_declare_kind_and_source() {
    let cases = load_cases();
    let live = build_results();
    for case in &cases {
        let case_id = case["case_id"].as_str().expect("case_id");
        assert!(
            case["acceptance"]["analytical"].is_null(),
            "case {case_id} grew an analytical acceptance slot: score it with kind \
             'analytical' before comparing (the harness only scores the comparative slot today)"
        );
    }
    for row in live["cases"].as_array().expect("cases array") {
        let case_id = row["case_id"].as_str().expect("case_id");
        let bands = row["bands"].as_array().expect("bands array");
        for band in bands {
            let kind = band["band_kind"].as_str().expect("band_kind");
            assert!(
                kind == "analytical" || kind == "comparative",
                "case {case_id}: band kind {kind} is neither analytical nor comparative"
            );
            let source = band["band_source"].as_str().expect("band_source");
            assert!(
                !source.trim().is_empty(),
                "case {case_id}: band row has no source"
            );
        }
    }
}

/// Every band metric of every case is scored or explicitly marked
/// not-reproducible — no metric is silently dropped, and the cases with no
/// band at all are recorded as such.
#[test]
fn test_section9_every_band_metric_is_accounted_for() {
    let cases = load_cases();
    let live = build_results();
    for case in &cases {
        let case_id = case["case_id"].as_str().expect("case_id");
        let expected_metrics = case["acceptance"]["comparative"]["bounds"]
            .as_object()
            .map(|b| b.len())
            .unwrap_or(0);
        let row = live["cases"]
            .as_array()
            .expect("cases array")
            .iter()
            .find(|r| r["case_id"] == case_id)
            .expect("case row");
        let bands = row["bands"].as_array().expect("bands array");
        assert_eq!(
            bands.len(),
            expected_metrics,
            "case {case_id}: every committed band metric must be accounted for"
        );
        if expected_metrics == 0 {
            assert_eq!(
                row["status"].as_str(),
                Some("no_band"),
                "case {case_id} has no comparative band and must be recorded as no_band"
            );
        }
        for band in bands {
            let status = band["status"].as_str().expect("status");
            match status {
                "pass" | "known_fail" => {
                    let value = band["engine_value_kw"].as_f64().expect("engine value");
                    assert!(value.is_finite(), "case {case_id}: non-finite engine value");
                    let low = band["band_low"].as_f64().expect("band low");
                    let high = band["band_high"].as_f64().expect("band high");
                    let gap = band["gap_pct_of_mid"].as_f64().expect("gap");
                    if status == "pass" {
                        assert!(
                            value >= low && value <= high,
                            "case {case_id}: a pass row must sit inside the band"
                        );
                        assert!(gap.abs() < 1e-9, "case {case_id}: pass row must have gap 0");
                    } else {
                        assert!(
                            value < low || value > high,
                            "case {case_id}: a known_fail row must sit outside the band"
                        );
                        assert!(
                            gap > 0.0,
                            "case {case_id}: a known_fail row must carry a positive gap"
                        );
                    }
                }
                "not_reproducible" => {
                    let reason = band["reason"].as_str().expect("reason");
                    assert!(
                        reason.len() > 40,
                        "case {case_id}: a not-reproducible row must say why"
                    );
                    assert!(
                        band["engine_value_kw"].is_null(),
                        "case {case_id}: a not-reproducible row must not carry an engine value"
                    );
                }
                other => panic!("case {case_id}: unexpected band status {other}"),
            }
        }
    }
}

/// Physics identity of the dispatch: where the unit can meet the case load,
/// the delivered load closes the case load; where it cannot, the shortfall is
/// recorded (the CE100 unit is undersized for CE165/CE185 at the driest
/// tabulated entering condition).
#[test]
fn test_section9_dispatch_closes_the_case_load_or_records_the_shortfall() {
    let live = build_results();
    let mut capacity_bound: Vec<String> = Vec::new();
    for row in live["cases"].as_array().expect("cases array") {
        let case_id = row["case_id"].as_str().expect("case_id");
        for run in row["runs"].as_array().expect("runs array") {
            let required = run["required_load_w"].as_f64().expect("required load");
            let delivered = run["delivered_capacity_w"].as_f64().expect("delivered");
            let shortfall = run["capacity_shortfall_w"].as_f64().expect("shortfall");
            let plr = run["plr"].as_f64().expect("plr");
            assert!(
                (0.0..=1.0).contains(&plr),
                "case {case_id}: PLR {plr} out of range"
            );
            if shortfall > 0.0 {
                capacity_bound.push(format!(
                    "{case_id} @ EWB {} °C: short {:.0} W",
                    run["entering_wet_bulb_C"].as_f64().unwrap_or(f64::NAN),
                    shortfall
                ));
                assert!(
                    (plr - 1.0).abs() < 1e-9,
                    "case {case_id}: a capacity shortfall must show PLR = 1"
                );
                assert!(
                    (required - delivered - shortfall).abs() < 1e-6,
                    "case {case_id}: the recorded shortfall must match the load balance"
                );
            } else {
                let rel = (delivered - required).abs() / required.max(1.0);
                assert!(
                    rel < 1e-9,
                    "case {case_id}: the dispatch must close the case load (rel {rel})"
                );
            }
        }
    }
    // The known capacity-bound corners, recorded rather than hidden.
    assert!(
        capacity_bound.iter().any(|s| s.starts_with("CE165")),
        "CE165 must record its capacity shortfall at the driest tabulated entering condition: \
         {capacity_bound:?}"
    );
    assert!(
        capacity_bound.iter().any(|s| s.starts_with("CE185")),
        "CE185 must record its capacity shortfall at the driest tabulated entering condition: \
         {capacity_bound:?}"
    );
}

/// The scored rows are real numbers against the case's own band: the engine
/// value and the band edges are present, the band came from the committed
/// artefact, and the band was not widened to fit the engine.
#[test]
fn test_section9_scored_rows_carry_real_numbers_against_their_own_band() {
    let cases = load_cases();
    let live = build_results();
    let mut scored = 0_usize;
    for case in &cases {
        let case_id = case["case_id"].as_str().expect("case_id");
        let committed = &case["acceptance"]["comparative"]["bounds"];
        let row = live["cases"]
            .as_array()
            .expect("cases array")
            .iter()
            .find(|r| r["case_id"] == case_id)
            .expect("case row");
        for band in row["bands"].as_array().expect("bands array") {
            let metric = band["metric"].as_str().expect("metric");
            let committed_band = committed.get(metric).unwrap_or_else(|| {
                panic!("case {case_id}: band metric {metric} is not in the case table")
            });
            let low = committed_band[0].as_f64().expect("low");
            let high = committed_band[1].as_f64().expect("high");
            let recorded_low = band["band_low"].as_f64().expect("band_low");
            let recorded_high = band["band_high"].as_f64().expect("band_high");
            assert!(
                (recorded_low - low).abs() < 1e-9 && (recorded_high - high).abs() < 1e-9,
                "case {case_id}: the scored band must be the committed band, verbatim"
            );
            if band["status"].as_str() == Some("known_fail") {
                scored += 1;
                let value = band["engine_value_kw"].as_f64().expect("engine value");
                assert!(
                    value.is_finite() && value > 0.0,
                    "case {case_id}: the recorded engine value must be a real number"
                );
            }
        }
    }
    assert!(
        scored > 0,
        "at least one §9 case must be scored against its own band"
    );
}

/// The generated §9 case table wires the CE100 performance map through every
/// case and states each case's acceptance-band status explicitly (Issue
/// #4296).
///
/// * The root `performance_map` block must describe the map artefact the
///   engine actually dispatches against: 24 tabulated points (6 ODB rows x 4
///   EWB columns) and the same ODB/EWB columns the map artefact exposes.
/// * Every case must repeat that wiring, so a case can never claim a
///   performance basis the map does not tabulate.
/// * Every case must name its band kinds explicitly: the `analytical` slot
///   (null today, with its reason) and the `comparative` slot (the Task-22
///   ensemble band with its source, or an explicit note naming the missing
///   source for the cases the bounds CSV does not cover).
#[test]
fn test_section9_case_table_wires_the_performance_map_and_states_its_bands() {
    let root: Value = serde_json::from_str(SECTION9_CASES_JSON).expect("parse §9 CE cases");

    // --- the performance-map wiring -------------------------------------
    let map_meta = &root["performance_map"];
    assert_eq!(
        map_meta["point_count"].as_u64(),
        Some(24),
        "the CE100 performance map tabulates 24 points per table (6 ODB rows x 4 EWB columns)"
    );
    assert_eq!(
        map_meta["scoring_table"].as_str(),
        Some("adjusted_net_Table_26e"),
        "the case table must name the map table the engine scores against"
    );

    let map_table = &map_json()["tables"]["adjusted_net_Table_26e"];
    let mut map_odb: Vec<f64> = map_table
        .as_object()
        .expect("adjusted-net table object")
        .keys()
        .filter_map(|k| k.parse::<f64>().ok())
        .collect();
    map_odb.sort_by(|a, b| a.partial_cmp(b).expect("finite ODB"));
    let table_odb: Vec<f64> = map_meta["odb_columns_C"]
        .as_array()
        .expect("odb_columns_C array")
        .iter()
        .map(|v| v.as_f64().expect("ODB column"))
        .collect();
    assert_eq!(
        table_odb, map_odb,
        "the case table's ODB columns must be the map artefact's ODB columns"
    );
    let table_ewb: Vec<f64> = map_meta["ewb_columns_C"]
        .as_array()
        .expect("ewb_columns_C array")
        .iter()
        .map(|v| v.as_f64().expect("EWB column"))
        .collect();
    assert_eq!(
        table_ewb,
        map_ewb_columns(),
        "the case table's EWB columns must be the map artefact's EWB columns"
    );

    // --- per-case wiring and band status -------------------------------
    let cases = root["cases"].as_array().expect("cases array");
    for case in cases {
        let case_id = case["case_id"].as_str().expect("case_id");
        let perf = &case["equipment_performance"];
        assert_eq!(
            perf["map_point_count"].as_u64(),
            Some(24),
            "case {case_id}: the case must be wired to the 24-point performance map"
        );
        assert_eq!(
            perf["source"].as_str(),
            Some("data/reference/ashrae140/section9_ce100_performance_map.json"),
            "case {case_id}: the case must name the performance-map artefact it dispatches against"
        );
        assert!(
            !perf["source_sha256"].as_str().unwrap_or("").is_empty(),
            "case {case_id}: the performance-map source must be pinned by digest"
        );
        let case_odb: Vec<f64> = perf["map_odb_columns_C"]
            .as_array()
            .expect("map_odb_columns_C array")
            .iter()
            .map(|v| v.as_f64().expect("ODB column"))
            .collect();
        assert_eq!(
            case_odb, map_odb,
            "case {case_id}: the case must be wired to the map's ODB columns"
        );
        assert_eq!(
            perf["case_odb_is_tabulated"].as_bool(),
            Some(true),
            "case {case_id}: the case's outdoor dry-bulb must be a tabulated map row"
        );
        assert!(
            perf["entering_wet_bulb_C"].is_null(),
            "case {case_id}: the per-case entering wet-bulb is not transcribed — it must stay \
             explicitly null rather than be invented"
        );

        // Band kinds are explicit for every case.
        let acceptance = &case["acceptance"];
        assert!(
            acceptance["analytical"].is_null(),
            "case {case_id}: the §9.2 quasi-analytical slot must stay null until the licensed \
             140-2023 §9 materials are transcribed (#4285)"
        );
        assert!(
            !acceptance["analytical_note"]
                .as_str()
                .unwrap_or("")
                .trim()
                .is_empty(),
            "case {case_id}: the null analytical slot must carry its reason"
        );
        if acceptance["comparative"].is_null() {
            assert!(
                !acceptance["comparative_note"]
                    .as_str()
                    .unwrap_or("")
                    .trim()
                    .is_empty(),
                "case {case_id}: a case with no comparative band must say so explicitly and name \
                 the missing source"
            );
        } else {
            let comparative = &acceptance["comparative"];
            let source = comparative["source"].as_str().expect("comparative source");
            assert!(
                source.contains("comparative_bounds_ce100_ce200.csv"),
                "case {case_id}: the comparative band must name the bounds file it was read from"
            );
            // The Task-22 identifier quoted in the source string is the
            // published one ("E100"), not a doubled "EE100".
            let task22_id = format!("identifier {}", &case_id[1..]);
            assert!(
                source.contains(&task22_id),
                "case {case_id}: the band source must quote the published Task-22 identifier \
                 {task22_id}"
            );
            assert!(
                !comparative["bounds"].as_object().expect("bounds object").is_empty(),
                "case {case_id}: a comparative band must carry its bounds"
            );
            assert!(
                !comparative["role"].as_str().unwrap_or("").trim().is_empty(),
                "case {case_id}: a comparative band must state its role"
            );
        }
    }
}

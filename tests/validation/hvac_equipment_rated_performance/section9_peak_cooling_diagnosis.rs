//! §9a `peak_cooling` known-fail **diagnosis** — Issue #4271 (Refs; does NOT
//! close #4271).
//!
//! # What this module does
//!
//! The §9a engine-run harness (`section9_engine_run.rs`, #4296) scores every
//! CE case's `peak_cooling` band row against the engine's **full-load cooling
//! power** (compressor power at PLR = 1 + the map's 230 W indoor fan heat,
//! adjusted-net basis). Nine of the fourteen §9a rows land outside the
//! Task-22 comparative `peak_cooling` band and are recorded as known-fail
//! rows (`docs/KNOWN_ISSUES.md` §LIMIT-42).
//!
//! The ensemble's own metric semantics — peak cooling **load** (kW thermal)
//! vs peak cooling **power** (kW electrical) — are *not* verifiable from
//! this tree: the bounds CSV records `source_page` as `pending` for every
//! row, and the CE100 `peak_cooling` band bounds (2.70–3.20 kW) coincide
//! numerically with the AE300 `peak_cooling` band bounds in
//! `comparative_bounds_ae101_ae445.csv` (the rows differ in case/table
//! fields; the coincidence is noted, not interpreted). This module therefore **quantifies**
//! which candidate reading of `peak_cooling` — if any — explains the nine
//! gaps, using only the committed artefacts and the public
//! `fluxion::sim::hvac` API. It changes nothing: no band is widened, no
//! engine number is altered, no result is re-recorded.
//!
//! # The candidate readings (all computed live)
//!
//! For each known-fail case and each entering wet-bulb (EWB) column the map
//! tabulates, this module computes six candidate readings of the band metric
//! and the gap of each to the case's own band (the harness's `gap_pct_of_mid`
//! definition: 0 inside the band, else the distance outside as a percentage
//! of the band midpoint):
//!
//! | candidate | definition |
//! |---|---|
//! | `power_adjnet_230W` | the recorded reading: full-load compressor power (adjusted-net table) + the map's 230 W actual indoor fan heat |
//! | `load_required` | the engine's required cooling load (the case's sensible + latent internal gains) — the *load* reading |
//! | `power_fan_default_329W` | full-load compressor power + the map's 329 W manufacturer-default fan heat (the fan-heat basis variant) |
//! | `power_gross_table` | full-load compressor power read from the **gross** capacity table (Table 26c) + 230 W fan (the gross-basis variant) |
//! | `capacity_gross` | the full-load **total capacity** on the gross table — the engine's cooling *capacity* read as the metric |
//! | `power_partload` | the delivered compressor power at the dispatch's operating point + 230 W fan (the *part-load* power reading) |
//!
//! The gross-basis variant (d) **is** computable from the public API —
//! `CeTable::Gross26c` and `CePerformanceMap::fan_heat_default_w()` /
//! `fan_heat_actual_w()` are public, and `evaluate` returns the gross
//! table's own compressor power and capacity — so no new `src/` code is
//! needed and none was added.
//!
//! # Verdicts
//!
//! Per (case, candidate):
//!
//! * `explains` — the candidate value is inside the band at **every**
//!   tabulated EWB column;
//! * `partially_explains` — not inside everywhere, but the candidate's gap is
//!   strictly smaller than the recorded reading's gap at **every** column
//!   (it moves towards the band without reaching it);
//! * `explains_nothing` — the gap is unchanged or larger at every column
//!   (including candidates that move *past* the band to the other side).
//!
//! The verdicts are computed live from the engine and asserted against the
//! pinned expectation table below, so the diagnosis cannot drift from the
//! engine state it measured: if the map, the case table, the recorded
//! results or the engine change, this test fails.
//!
//! # What this module does NOT show
//!
//! * It does **not** claim that any case passes §9a. The comparative band is
//!   an ensemble *evidence range*, not a pass/fail criterion (see #4277).
//! * It does **not** score against the standard's own §9.2 quasi-analytical
//!   criteria: that slot is `null` for every §9 case and stays blocked on
//!   #4285 (licensed 140-2023 §9 normative materials — per-case entering
//!   wet-bulb, quasi-analytical slots, normative CE weather files).
//! * It does **not** invent a per-case entering wet-bulb: every case is
//!   dispatched at every EWB column the map tabulates, and no column is
//!   attributed to a case.
//! * It does **not** assert anything about `annual_cooling` or
//!   `max_zone_temp` (both not reproducible without the licensed weather
//!   files).

use serde_json::Value;

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

/// Tolerance on the pinned candidate values [kW].
const VALUE_TOL: f64 = 1e-6;
/// Tolerance on the pinned gap percentages [percentage points of band mid].
const GAP_TOL: f64 = 1e-4;

/// The candidate readings of the `peak_cooling` band metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Candidate {
    /// Recorded reading: full-load compressor power + 230 W actual fan heat.
    PowerAdjNet230,
    /// The engine's required cooling load (the *load* reading).
    LoadRequired,
    /// Full-load compressor power + 329 W manufacturer-default fan heat.
    PowerFanDefault329,
    /// Full-load compressor power read from the gross capacity table.
    PowerGrossTable,
    /// Full-load total capacity on the gross table (capacity read as metric).
    CapacityGross,
    /// Delivered compressor power at the dispatch operating point + 230 W.
    PowerPartLoad,
}

impl Candidate {
    /// Stable name (also the column order of the pinned expectation table).
    fn name(self) -> &'static str {
        match self {
            Candidate::PowerAdjNet230 => "power_adjnet_230W",
            Candidate::LoadRequired => "load_required",
            Candidate::PowerFanDefault329 => "power_fan_default_329W",
            Candidate::PowerGrossTable => "power_gross_table",
            Candidate::CapacityGross => "capacity_gross",
            Candidate::PowerPartLoad => "power_partload",
        }
    }

    /// Every candidate, in the pinned order.
    const ALL: [Candidate; 6] = [
        Candidate::PowerAdjNet230,
        Candidate::LoadRequired,
        Candidate::PowerFanDefault329,
        Candidate::PowerGrossTable,
        Candidate::CapacityGross,
        Candidate::PowerPartLoad,
    ];
}

/// One (case, EWB, candidate) measurement.
#[derive(Debug, Clone)]
struct Measurement {
    case_id: String,
    ewb_c: f64,
    candidate: Candidate,
    value_kw: f64,
    gap_pct: f64,
    inside: bool,
}

/// One case's whole diagnosis.
#[derive(Debug, Clone)]
struct CaseDiagnosis {
    case_id: String,
    band_low: f64,
    band_high: f64,
    /// The case's required cooling load [kW] (EWB-independent).
    required_load_kw: f64,
    /// The map's full-load COP at each EWB column.
    cop: Vec<f64>,
    /// The load / full-load-power ratio at each EWB column.
    load_power_ratio: Vec<f64>,
    /// Per candidate, per EWB column: the measurement.
    rows: Vec<Measurement>,
}

fn load_cases() -> Vec<Value> {
    let root: Value = serde_json::from_str(SECTION9_CASES_JSON).expect("parse §9 CE cases");
    root["cases"].as_array().expect("cases array").clone()
}

fn load_map() -> CePerformanceMap {
    CePerformanceMap::from_json_str(SECTION9_MAP_JSON).expect("parse §9 CE100 performance map")
}

/// The map's own tabulated entering wet-bulb columns (°C), read from the
/// committed map artefact (the public API exposes no EWB axis).
fn map_ewb_columns() -> Vec<f64> {
    let root: Value = serde_json::from_str(SECTION9_MAP_JSON).expect("parse map json");
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

/// The harness's gap definition: 0 inside the band, otherwise the distance
/// outside it as a percentage of the band midpoint.
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

/// The recorded results file (the known-fail record).
fn load_recorded_results() -> Value {
    serde_json::from_str(SECTION9_RESULTS_JSON).expect("parse recorded §9 engine-run results")
}

/// The known-fail cases, re-derived from the committed results file: every
/// case with a `peak_cooling` band row whose recorded status is
/// `known_fail`. Nothing is hard-coded.
fn recorded_known_fail_cases() -> Vec<String> {
    let recorded = load_recorded_results();
    let mut ids: Vec<String> = recorded["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .filter(|case| {
            case["bands"]
                .as_array()
                .expect("bands array")
                .iter()
                .any(|b| b["metric"] == "peak_cooling" && b["status"] == "known_fail")
        })
        .map(|case| case["case_id"].as_str().expect("case_id").to_string())
        .collect();
    ids.sort();
    ids
}

/// The comparative `peak_cooling` band of a case, if it has one.
fn peak_cooling_band(case: &Value) -> Option<(f64, f64)> {
    let band = case["acceptance"]["comparative"]["bounds"]["peak_cooling"].as_array()?;
    Some((band[0].as_f64()?, band[1].as_f64()?))
}

/// Compute the whole diagnosis: every case with a band, every tabulated EWB
/// column, every candidate reading.
fn build_diagnosis() -> Vec<CaseDiagnosis> {
    let cases = load_cases();
    let map = load_map();
    let ewbs = map_ewb_columns();
    let fan_actual_w = map.fan_heat_actual_w();
    let fan_default_w = map.fan_heat_default_w();

    let mut out: Vec<CaseDiagnosis> = Vec::new();
    for case in &cases {
        let case_id = case["case_id"].as_str().expect("case_id").to_string();
        let Some((band_low, band_high)) = peak_cooling_band(case) else {
            continue;
        };
        let cond = &case["conditions"];
        let odb_c = cond["outdoor_dry_bulb_C"].as_f64().expect("ODB");
        let edb_c = cond["cooling_setpoint_C"].as_f64().expect("setpoint");

        let mut rows: Vec<Measurement> = Vec::new();
        let mut cop: Vec<f64> = Vec::new();
        let mut load_power_ratio: Vec<f64> = Vec::new();
        let mut required_load_kw = f64::NAN;

        for ewb_c in &ewbs {
            // Full load on the adjusted-net basis (the recorded reading's
            // power) and on the gross basis (the gross-capacity variant).
            let adj = map.evaluate(
                CeTable::AdjustedNet26e,
                *ewb_c,
                odb_c,
                edb_c,
                1.0,
                fan_actual_w,
            );
            let gross = map.evaluate(CeTable::Gross26c, *ewb_c, odb_c, edb_c, 1.0, fan_actual_w);
            // The steady-state dispatch (same path as the harness).
            let dispatch = dispatch_ce_steady_state(case, &map, *ewb_c, fan_actual_w);

            required_load_kw = dispatch.required_load_w / 1000.0;
            cop.push(adj.cop);
            load_power_ratio
                .push(dispatch.required_load_w / (adj.compressor_power_w + fan_actual_w));

            let values = [
                (
                    Candidate::PowerAdjNet230,
                    (adj.compressor_power_w + fan_actual_w) / 1000.0,
                ),
                (Candidate::LoadRequired, dispatch.required_load_w / 1000.0),
                (
                    Candidate::PowerFanDefault329,
                    (adj.compressor_power_w + fan_default_w) / 1000.0,
                ),
                (
                    Candidate::PowerGrossTable,
                    (gross.compressor_power_w + fan_actual_w) / 1000.0,
                ),
                (Candidate::CapacityGross, gross.total_capacity_w / 1000.0),
                (
                    Candidate::PowerPartLoad,
                    (dispatch.delivered_power_w + fan_actual_w) / 1000.0,
                ),
            ];
            for (candidate, value_kw) in values {
                let gap_pct = gap_pct_of_mid(value_kw, band_low, band_high);
                rows.push(Measurement {
                    case_id: case_id.clone(),
                    ewb_c: *ewb_c,
                    candidate,
                    value_kw,
                    gap_pct,
                    inside: gap_pct == 0.0,
                });
            }
        }
        out.push(CaseDiagnosis {
            case_id,
            band_low,
            band_high,
            required_load_kw,
            cop,
            load_power_ratio,
            rows,
        });
    }
    out
}

/// The verdict of one candidate for one case, from the live measurements.
///
/// Pairing is per EWB column: a candidate is compared against the recorded
/// reading at the SAME column, never against another column's gap.
fn verdict(diag: &CaseDiagnosis, candidate: Candidate) -> &'static str {
    let rows_for = |c: Candidate| -> Vec<&Measurement> {
        diag.rows.iter().filter(|r| r.candidate == c).collect()
    };
    let target = rows_for(candidate);
    let base = rows_for(Candidate::PowerAdjNet230);
    assert_eq!(
        target.len(),
        base.len(),
        "candidate and recorded readings must share EWB columns"
    );
    if target.iter().all(|r| r.inside) {
        return "explains";
    }
    let all_smaller = target
        .iter()
        .zip(base.iter())
        .all(|(t, b)| (t.ewb_c - b.ewb_c).abs() < 1e-9 && t.gap_pct < b.gap_pct);
    if all_smaller {
        "partially_explains"
    } else {
        "explains_nothing"
    }
}

/// The pinned expectation table: one row per (case, EWB column), holding the
/// six candidate values in kW and the six candidate gaps in % of the band
/// midpoint, in `Candidate::ALL` order.
///
/// Format: `"v1,v2,v3,v4,v5,v6|g1,g2,g3,g4,g5,g6"`.
///
/// Pinned to the engine state this diagnosis measured; a change in the map,
/// the case table, the recorded results or the engine fails the test.
const EXPECTED: &[(&str, f64, &str)] = &[
    ("CE100", 15.0, "2.34,5.4,2.439,2.34,6.25,2.1684078312084685|12.203389830508485,74.57627118644068,8.847457627118647,12.203389830508485,103.38983050847457,18.020073518357005"),
    ("CE100", 17.2, "2.37,5.4,2.469,2.37,6.72,2.081807338964392|11.186440677966104,74.57627118644068,7.8305084745762805,11.186440677966104,119.32203389830507,20.955683424935867"),
    ("CE100", 19.4, "2.41,5.4,2.509,2.41,7.21,2.008736752400207|9.830508474576272,74.57627118644068,6.47457627118645,9.830508474576272,135.9322033898305,23.43265246100994"),
    ("CE100", 21.7, "2.44,5.4,2.539,2.44,7.71,1.9339613460352272|8.813559322033905,74.57627118644068,5.457627118644068,8.813559322033905,152.88135593220338,25.96741199880586"),
    ("CE110", 15.0, "1.85,5.4,1.949,1.85,7.42,1.5202483595350504|27.58620689655172,77.58620689655174,24.172413793103445,27.58620689655172,147.24137931034483,38.95695311948102"),
    ("CE110", 17.2, "1.89,5.4,1.989,1.89,8.01,1.468980619158588|26.20689655172414,77.58620689655174,22.793103448275858,26.20689655172414,167.58620689655172,40.72480623591076"),
    ("CE110", 19.4, "1.94,5.4,2.039,1.94,8.65,1.4248108410257205|24.482758620689655,77.58620689655174,21.068965517241374,24.482758620689655,189.6551724137931,42.24790203359584"),
    ("CE110", 21.7, "1.99,5.4,2.089,1.99,9.29,1.3859428158591849|22.75862068965517,77.58620689655174,19.344827586206893,22.75862068965517,211.72413793103445,43.58817876347638"),
    ("CE120", 15.0, "1.85,5.4,1.949,1.85,7.89,1.4547880982182608|31.147540983606554,68.85245901639347,27.901639344262293,31.147540983606554,150.4918032786885,44.105308255138986"),
    ("CE120", 17.2, "1.89,5.4,1.989,1.89,8.01,1.468980619158588|29.836065573770494,68.85245901639347,26.59016393442622,29.836065573770494,154.42622950819674,43.63997969971843"),
    ("CE120", 19.4, "1.94,5.4,2.039,1.94,8.65,1.4248108410257205|28.196721311475407,68.85245901639347,24.95081967213114,28.196721311475407,175.40983606557378,45.08816914669768"),
    ("CE120", 21.7, "1.99,5.4,2.089,1.99,9.29,1.3859428158591849|26.557377049180324,68.85245901639347,23.311475409836063,26.557377049180324,196.3934426229508,46.36253062756771"),
    ("CE130", 15.0, "2.34,0.27,2.439,2.34,6.25,0.35112900653803225|9.122807017543867,81.75438596491227,5.6491228070175445,9.122807017543867,110.52631578947367,78.90775415656027"),
    ("CE130", 17.2, "2.37,0.27,2.469,2.37,6.72,0.344063036706267|8.07017543859649,81.75438596491227,4.5964912280701835,8.07017543859649,127.01754385964912,79.15568292258712"),
    ("CE130", 19.4, "2.41,0.27,2.509,2.41,7.21,0.3381307489239282|6.666666666666665,81.75438596491227,3.1929824561403573,6.666666666666665,144.21052631578945,79.36383337109024"),
    ("CE130", 21.7, "2.44,0.27,2.539,2.44,7.71,0.33236905356232976|5.614035087719303,81.75438596491227,2.1403508771929802,5.614035087719303,161.75438596491225,79.56599812062002"),
    ("CE140", 15.0, "1.85,0.27,1.949,1.85,7.42,0.3080330114962214|30,82.66666666666667,26.699999999999996,30,139,81.39889961679262"),
    ("CE140", 17.2, "1.89,0.27,1.989,1.89,8.01,0.30395783314439595|28.666666666666668,82.66666666666667,25.366666666666664,28.666666666666668,158.66666666666666,81.5347388951868"),
    ("CE140", 19.4, "1.94,0.27,2.039,1.94,8.65,0.3004492916161223|27,82.66666666666667,23.699999999999996,27,180,81.65169027946258"),
    ("CE140", 21.7, "1.99,0.27,2.089,1.99,9.29,0.2974320889899235|25.333333333333336,82.66666666666667,22.033333333333335,25.333333333333336,201.33333333333331,81.75226370033587"),
    ("CE150", 15.0, "1.85,6.5,1.949,1.85,7.42,1.7274424025062607|36.764705882352935,82.35294117647058,33.85294117647059,36.764705882352935,109.41176470588235,40.36934110275704"),
    ("CE150", 17.2, "1.89,6.5,1.989,1.89,8.01,1.6711877844976812|35.588235294117645,82.35294117647058,32.67647058823529,35.588235294117645,126.76470588235291,42.023888691244665"),
    ("CE150", 19.4, "1.94,6.5,2.039,1.94,8.65,1.6228012911762937|34.11764705882353,82.35294117647058,31.205882352941174,34.11764705882353,145.58823529411765,43.447020847756065"),
    ("CE150", 21.7, "1.99,6.5,2.089,1.99,9.29,1.5800500368195465|32.64705882352941,82.35294117647058,29.735294117647058,32.64705882352941,164.4117647058823,44.70441068177804"),
    ("CE160", 15.0, "1.85,6.5,1.949,1.85,7.89,1.6540583363536534|35.82089552238806,85.07462686567166,32.86567164179104,35.82089552238806,126.5671641791045,41.66990040735363"),
    ("CE160", 17.2, "1.89,6.5,1.989,1.89,8.01,1.6711877844976812|34.626865671641795,85.07462686567166,31.67164179104477,34.626865671641795,130.14925373134326,41.15857359708414"),
    ("CE160", 19.4, "1.94,6.5,2.039,1.94,8.65,1.6228012911762937|33.134328358208954,85.07462686567166,30.179104477611933,33.134328358208954,149.2537313432836,42.602946532050936"),
    ("CE160", 21.7, "1.99,6.5,2.089,1.99,9.29,1.5800500368195465|31.64179104477612,85.07462686567166,28.686567164179106,31.64179104477612,168.35820895522386,43.879103378521"),
    ("CE170", 15.0, "1.85,3.2,1.949,1.85,7.42,1.0559656580559535|37.68115942028985,0,34.81159420289855,37.68115942028985,106.37681159420289,60.696647592581044"),
    ("CE170", 17.2, "1.89,3.2,1.989,1.89,8.01,1.0191632670646158|36.52173913043478,0,33.65217391304347,36.52173913043478,123.47826086956522,61.763383563344455"),
    ("CE170", 19.4, "1.94,3.2,2.039,1.94,8.65,0.9874100783155377|35.07246376811594,0,32.202898550724626,35.07246376811594,142.02898550724638,62.68376584592643"),
    ("CE170", 21.7, "1.99,3.2,2.089,1.99,9.29,0.9597170532491325|33.6231884057971,0,30.753623188405793,33.6231884057971,160.5797101449275,63.48646222466282"),
    ("CE200", 15.0, "2,7.937,2.099,2,7.57,2|34.28571428571429,118.20000000000002,31.457142857142856,34.28571428571429,107.71428571428572,34.28571428571429"),
    ("CE200", 17.2, "2.04,7.937,2.139,2.04,7.6,2.04|33.142857142857146,118.20000000000002,30.314285714285727,33.142857142857146,108.57142857142857,33.142857142857146"),
    ("CE200", 19.4, "2.09,7.937,2.189,2.09,8.18,2.0876541177746684|31.714285714285722,118.20000000000002,28.885714285714286,31.714285714285722,125.14285714285714,31.781310920723765"),
    ("CE200", 21.7, "2.14,7.937,2.239,2.14,8.8,2.0293581753826064|30.28571428571429,118.20000000000002,27.457142857142863,30.28571428571429,142.8571428571429,33.446909274782676"),
];

/// The pinned per-case verdicts: (case, candidate name, verdict), in the
/// order `verdict()` defines them. Only two cells are not
/// `explains_nothing`: the load reading explains CE170, and the
/// manufacturer-default fan-heat basis partially explains every case.
const EXPECTED_VERDICTS: &[(&str, &str, &str)] = &[
    ("CE100", "power_adjnet_230W", "explains_nothing"),
    ("CE100", "load_required", "explains_nothing"),
    ("CE100", "power_fan_default_329W", "partially_explains"),
    ("CE100", "power_gross_table", "explains_nothing"),
    ("CE100", "capacity_gross", "explains_nothing"),
    ("CE100", "power_partload", "explains_nothing"),
    ("CE110", "power_adjnet_230W", "explains_nothing"),
    ("CE110", "load_required", "explains_nothing"),
    ("CE110", "power_fan_default_329W", "partially_explains"),
    ("CE110", "power_gross_table", "explains_nothing"),
    ("CE110", "capacity_gross", "explains_nothing"),
    ("CE110", "power_partload", "explains_nothing"),
    ("CE120", "power_adjnet_230W", "explains_nothing"),
    ("CE120", "load_required", "explains_nothing"),
    ("CE120", "power_fan_default_329W", "partially_explains"),
    ("CE120", "power_gross_table", "explains_nothing"),
    ("CE120", "capacity_gross", "explains_nothing"),
    ("CE120", "power_partload", "explains_nothing"),
    ("CE130", "power_adjnet_230W", "explains_nothing"),
    ("CE130", "load_required", "explains_nothing"),
    ("CE130", "power_fan_default_329W", "partially_explains"),
    ("CE130", "power_gross_table", "explains_nothing"),
    ("CE130", "capacity_gross", "explains_nothing"),
    ("CE130", "power_partload", "explains_nothing"),
    ("CE140", "power_adjnet_230W", "explains_nothing"),
    ("CE140", "load_required", "explains_nothing"),
    ("CE140", "power_fan_default_329W", "partially_explains"),
    ("CE140", "power_gross_table", "explains_nothing"),
    ("CE140", "capacity_gross", "explains_nothing"),
    ("CE140", "power_partload", "explains_nothing"),
    ("CE150", "power_adjnet_230W", "explains_nothing"),
    ("CE150", "load_required", "explains_nothing"),
    ("CE150", "power_fan_default_329W", "partially_explains"),
    ("CE150", "power_gross_table", "explains_nothing"),
    ("CE150", "capacity_gross", "explains_nothing"),
    ("CE150", "power_partload", "explains_nothing"),
    ("CE160", "power_adjnet_230W", "explains_nothing"),
    ("CE160", "load_required", "explains_nothing"),
    ("CE160", "power_fan_default_329W", "partially_explains"),
    ("CE160", "power_gross_table", "explains_nothing"),
    ("CE160", "capacity_gross", "explains_nothing"),
    ("CE160", "power_partload", "explains_nothing"),
    ("CE170", "power_adjnet_230W", "explains_nothing"),
    ("CE170", "load_required", "explains"),
    ("CE170", "power_fan_default_329W", "partially_explains"),
    ("CE170", "power_gross_table", "explains_nothing"),
    ("CE170", "capacity_gross", "explains_nothing"),
    ("CE170", "power_partload", "explains_nothing"),
    ("CE200", "power_adjnet_230W", "explains_nothing"),
    ("CE200", "load_required", "explains_nothing"),
    ("CE200", "power_fan_default_329W", "partially_explains"),
    ("CE200", "power_gross_table", "explains_nothing"),
    ("CE200", "capacity_gross", "explains_nothing"),
    ("CE200", "power_partload", "explains_nothing"),
];

/// One row of the pinned table, in the order the diagnosis computes it.
type ExpectedRow = (String, f64, String, f64, f64);

fn live_rows(diags: &[CaseDiagnosis]) -> Vec<ExpectedRow> {
    let mut rows: Vec<ExpectedRow> = Vec::new();
    for diag in diags {
        for r in &diag.rows {
            rows.push((
                r.case_id.clone(),
                r.ewb_c,
                r.candidate.name().to_string(),
                r.value_kw,
                r.gap_pct,
            ));
        }
    }
    rows
}

/// The nine known-fail cases are re-derived from the committed results file,
/// and the live diagnosis must reproduce exactly that set (the recorded
/// reading is the only one that can be "known-fail" in the harness).
#[test]
fn test_section9_peak_cooling_diagnosis_reports_the_known_fail_cases() {
    let recorded = recorded_known_fail_cases();
    assert_eq!(
        recorded.len(),
        9,
        "the committed results file must record 9 known-fail `peak_cooling` rows, got {recorded:?}"
    );

    let diags = build_diagnosis();
    let mut live_known_fail: Vec<String> = diags
        .iter()
        .filter(|d| {
            d.rows
                .iter()
                .any(|r| r.candidate == Candidate::PowerAdjNet230 && !r.inside)
        })
        .map(|d| d.case_id.clone())
        .collect();
    live_known_fail.sort();
    assert_eq!(
        live_known_fail, recorded,
        "the live known-fail set must equal the committed one"
    );

    // The recorded reading at the worst column must reproduce the recorded
    // engine value and gap of the committed file (the diagnosis is pinned to
    // the recorded artefact, not to a private computation).
    let results = load_recorded_results();
    for diag in &diags {
        let case_row = results["cases"]
            .as_array()
            .expect("cases array")
            .iter()
            .find(|c| c["case_id"] == diag.case_id)
            .expect("case row in recorded results");
        let band_row = case_row["bands"]
            .as_array()
            .expect("bands array")
            .iter()
            .find(|b| b["metric"] == "peak_cooling")
            .expect("peak_cooling band row");
        let recorded_value = band_row["engine_value_kw"].as_f64().expect("engine value");
        let recorded_gap = band_row["gap_pct_of_mid"].as_f64().expect("gap");
        let live_worst = diag
            .rows
            .iter()
            .filter(|r| r.candidate == Candidate::PowerAdjNet230)
            .max_by(|a, b| a.gap_pct.total_cmp(&b.gap_pct))
            .expect("recorded reading measured");
        assert!(
            (live_worst.value_kw - recorded_value).abs() <= VALUE_TOL,
            "case {}: live recorded reading {} kW vs committed {recorded_value} kW",
            diag.case_id,
            live_worst.value_kw
        );
        assert!(
            (live_worst.gap_pct - recorded_gap).abs() <= GAP_TOL,
            "case {}: live gap {} vs committed {recorded_gap}",
            diag.case_id,
            live_worst.gap_pct
        );
    }
}

/// The candidate table is pinned: every case × EWB column × candidate value
/// and gap, computed live, must match the recorded expectation.
#[test]
fn test_section9_peak_cooling_candidate_table_is_pinned() {
    let diags = build_diagnosis();
    let live = live_rows(&diags);
    let mut pinned: Vec<ExpectedRow> = Vec::new();
    for (case_id, ewb, packed) in EXPECTED {
        let (values, gaps) = packed.split_once('|').expect("packed row has values|gaps");
        let values: Vec<f64> = values
            .split(',')
            .map(|v| v.parse::<f64>().expect("pinned value"))
            .collect();
        let gaps: Vec<f64> = gaps
            .split(',')
            .map(|v| v.parse::<f64>().expect("pinned gap"))
            .collect();
        assert_eq!(
            values.len(),
            Candidate::ALL.len(),
            "pinned row for {case_id} @ {ewb} must carry one value per candidate"
        );
        assert_eq!(gaps.len(), values.len(), "pinned gaps per candidate");
        for (i, candidate) in Candidate::ALL.iter().enumerate() {
            pinned.push((
                case_id.to_string(),
                *ewb,
                candidate.name().to_string(),
                values[i],
                gaps[i],
            ));
        }
    }
    assert_eq!(
        live.len(),
        pinned.len(),
        "the pinned candidate table must cover every measurement"
    );
    let mut diffs: Vec<String> = Vec::new();
    for (got, want) in live.iter().zip(pinned.iter()) {
        let (case_id, ewb, cand, value, gap) = got;
        let (w_case, w_ewb, w_cand, w_value, w_gap) = want;
        if case_id != w_case
            || (ewb - w_ewb).abs() > 1e-9
            || cand != w_cand
            || (value - w_value).abs() > VALUE_TOL
            || (gap - w_gap).abs() > GAP_TOL
        {
            diffs.push(format!(
                "live {case_id} @ {ewb} °C {cand}: value {value} kW gap {gap}% \
                 vs pinned {w_case} @ {w_ewb} °C {w_cand}: value {w_value} kW gap {w_gap}%"
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "the §9a `peak_cooling` candidate table drifted from the pinned diagnosis:\n  {}",
        diffs.join("\n  ")
    );
}

/// The diagnosis is deterministic: two independent computations agree.
#[test]
fn test_section9_peak_cooling_diagnosis_is_deterministic() {
    let first = build_diagnosis();
    let second = build_diagnosis();
    assert_eq!(
        live_rows(&first),
        live_rows(&second),
        "the §9a `peak_cooling` diagnosis must be deterministic"
    );
}

/// The per-case verdict per candidate, as quoted in the investigation note.
#[test]
fn test_section9_peak_cooling_candidate_verdicts() {
    let diags = build_diagnosis();
    let known_fail = recorded_known_fail_cases();
    // Pin every verdict: the diagnosis's conclusion must fail loudly if the
    // engine moves under it.
    for (case_id, candidate_name, want) in EXPECTED_VERDICTS {
        let diag = diags
            .iter()
            .find(|d| d.case_id == *case_id)
            .expect("diagnosed case");
        let candidate = Candidate::ALL
            .iter()
            .find(|c| c.name() == *candidate_name)
            .expect("known candidate");
        let got = verdict(diag, *candidate);
        assert_eq!(
            got, *want,
            "case {case_id} candidate {candidate_name}: verdict {got}, expected {want}"
        );
    }
    for diag in &diags {
        if !known_fail.contains(&diag.case_id) {
            continue;
        }
        for candidate in Candidate::ALL {
            let v = verdict(diag, candidate);
            assert!(
                matches!(v, "explains" | "partially_explains" | "explains_nothing"),
                "case {} candidate {}: unexpected verdict {v}",
                diag.case_id,
                candidate.name()
            );
            println!(
                "{:6} {:>22}  band [{:.2}, {:.2}] kW  -> {}",
                diag.case_id,
                candidate.name(),
                diag.band_low,
                diag.band_high,
                v
            );
        }
    }
}

/// Print the live candidate table (the table quoted in the investigation
/// note) and the per-case ratios.
#[test]
fn test_section9_peak_cooling_diagnosis_table_is_printed() {
    let diags = build_diagnosis();
    let known_fail = recorded_known_fail_cases();
    println!("case  ewb_C  band_low band_high candidate            value_kW    gap_pct  inside");
    for diag in &diags {
        if !known_fail.contains(&diag.case_id) {
            continue;
        }
        for r in &diag.rows {
            println!(
                "{:6} {:>5.1}  {:>8.2} {:>9.2} {:<22} {:>8.4} {:>10.4}  {}",
                diag.case_id,
                r.ewb_c,
                diag.band_low,
                diag.band_high,
                r.candidate.name(),
                r.value_kw,
                r.gap_pct,
                r.inside
            );
        }
    }
    println!("--- pinned-table format (case, ewb, \"v1..v6|g1..g6\") ---");
    for diag in &diags {
        if !known_fail.contains(&diag.case_id) {
            continue;
        }
        for ewb in map_ewb_columns() {
            let values: Vec<String> = Candidate::ALL
                .iter()
                .map(|c| {
                    diag.rows
                        .iter()
                        .find(|r| r.candidate == *c && (r.ewb_c - ewb).abs() < 1e-9)
                        .expect("measurement")
                        .value_kw
                        .to_string()
                })
                .collect();
            let gaps: Vec<String> = Candidate::ALL
                .iter()
                .map(|c| {
                    diag.rows
                        .iter()
                        .find(|r| r.candidate == *c && (r.ewb_c - ewb).abs() < 1e-9)
                        .expect("measurement")
                        .gap_pct
                        .to_string()
                })
                .collect();
            println!(
                "    (\"{}\", {:.1}, \"{}|{}\"),",
                diag.case_id,
                ewb,
                values.join(","),
                gaps.join(",")
            );
        }
    }
    for diag in &diags {
        if !known_fail.contains(&diag.case_id) {
            continue;
        }
        for (i, ewb) in map_ewb_columns().iter().enumerate() {
            println!(
                "ratio {:6} @ {:>4.1} °C  required_load_kW={:.4}  load/power_ratio={:.4}  cop={:.4}",
                diag.case_id,
                ewb,
                diag.required_load_kw,
                diag.load_power_ratio[i],
                diag.cop[i]
            );
        }
    }
}

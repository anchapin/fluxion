//! §7 conformance table (Issue #4276) — three-state evaluation of the
//! committed §7 workflow CSV.
//!
//! Extends the #4276 conformance table (round 48 built the §6 counterpart,
//! `data/reference/section6/section6_conformance.{csv,md}`) to the §7
//! Building Thermal Envelope and Fabric Load case set. Per #4276 and the
//! #4277 finding: states are `conformant` / `incomplete` (ran, missing
//! required outputs) / `not implemented` — no score, no percentage.
//!
//! Requirements are transcribed from the in-repo standard
//! (`docs/140_2017_a_20200901.pdf`), Section 6.2.1 "Output Requirements for
//! Building Thermal Envelope and Fabric Load Tests" — the Class I output
//! section the §7 cases must report against. Per-requirement cell states:
//! `report` (required, emitted, satisfies the check here), `partial`
//! (emitted for a subset of the required days; see 6.2.1.8 note), `missing`
//! (required, not emitted), `na` (not required for this case family).
//! Honest statuses only — nothing upgrades an unemitted quantity.
//!
//! Known honest gaps recorded by this table (not engine defects — scope
//! decisions to close in later rounds, each needing a harness change):
//! - 6.2.1.3/.4/.5 (annual transmitted solar through the *shaded south* /
//!   *west* / *shaded west* windows for Cases 610/620/630): the emitter's
//!   `zone_total_transmitted_solar_radiation` is the 6.2.1.2.2 reference
//!   quantity — a 1 m² *unshaded* south window — which does not satisfy
//!   these by construction.
//! - 6.2.1.6 for Case 960 requires free-float outputs for the *sun zone*
//!   (5.2.2.2.7.7.b); the emitter records zone 1 only.
//! - 6.2.1.8 daily hourly: the emitted day-profile days are the GSR
//!   contract days (Feb 1, May 4, Jul 14). The addendum PDF's Table 6-1
//!   day cells carry unresolved tracked changes (candidate days Jan 4 /
//!   Mar 5 / Jul 27 are not machine-readable from the scanned tracked
//!   copy), so every case with a Table 6-1 row is held at `partial`
//!   until the day list is confirmed against a clean copy of the standard.
//! - Case 970 is not defined in 140-2017 (no 5.2.2.x section, no case in
//!   the PDF text); its row is evaluated against 6.2.1.1 as the envelope
//!   analogue and flagged in the markdown.

use std::collections::BTreeMap;

/// One §7 case record parsed back from the committed workflow CSV.
#[derive(Debug, Clone)]
struct Section7CsvRow {
    case_id: String,
    annual_heating_mwh: Option<f64>,
    annual_cooling_mwh: Option<f64>,
    peak_heating_kw: Option<f64>,
    peak_cooling_kw: Option<f64>,
    peak_heating_time: String,
    peak_cooling_time: String,
    monthly_htg_cons: Option<[f64; 12]>,
    monthly_clg_cons: Option<[f64; 12]>,
    monthly_htg_peak_val: Option<[f64; 12]>,
    monthly_clg_peak_val: Option<[f64; 12]>,
    monthly_htg_peak_time: Option<[String; 12]>,
    monthly_clg_peak_time: Option<[String; 12]>,
    incident_east: Option<f64>,
    incident_horizontal: Option<f64>,
    incident_north: Option<f64>,
    incident_south: Option<f64>,
    incident_west: Option<f64>,
    transmitted_south_annual: Option<f64>,
    sky_min_temp: Option<f64>,
    sky_min_index: Option<f64>,
    sky_max_temp: Option<f64>,
    sky_max_index: Option<f64>,
    avg_temp: Option<f64>,
    min_temp: Option<f64>,
    min_index: Option<f64>,
    max_temp: Option<f64>,
    max_index: Option<f64>,
    /// packed `label...,v...` fields keyed by their `MMDD,<label>` head.
    sens_load_0201: Option<[f64; 24]>,
    sens_load_0714: Option<[f64; 24]>,
    sky_temp_0201: Option<[f64; 24]>,
    sky_temp_0504: Option<[f64; 24]>,
    sky_temp_0714: Option<[f64; 24]>,
    temp_0201: Option<[f64; 24]>,
    temp_0714: Option<[f64; 24]>,
    temp_bins: Option<Vec<u64>>,
    status: String,
}

/// Minimal RFC-4180-ish CSV field splitter (quotes, embedded commas), the
/// §6 harness's parser pattern.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        cur.push('"');
                    } else {
                        in_quotes = false;
                    }
                }
                _ => cur.push(c),
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => fields.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            }
        }
    }
    fields.push(cur);
    fields
}

fn parse_f(s: &str) -> Option<f64> {
    s.trim().parse().ok()
}

/// Packed array: `label...,v1,v2,...` — drops leading label elements that
/// are non-numeric, parses the rest.
fn parse_packed(s: &str) -> Option<Vec<f64>> {
    // `label...,v1,...` form: keep the trailing numeric run; label tokens
    // (e.g. `signed_kWh`) are non-numeric and dropped.
    let toks: Vec<Option<f64>> = s.split(',').map(|p| p.trim().parse().ok()).collect();
    let mut out = Vec::new();
    for t in toks.into_iter().rev() {
        match t {
            Some(v) => out.push(v),
            None => break,
        }
    }
    out.reverse();
    Some(out)
}

fn parse_packed24(s: &str) -> Option<[f64; 24]> {
    let v = parse_packed(s)?;
    if v.len() < 24 {
        return None;
    }
    let mut out = [0.0; 24];
    out.copy_from_slice(&v[v.len() - 24..]);
    Some(out)
}

fn parse_packed_bins(s: &str) -> Option<Vec<u64>> {
    // leading label elements ("hourly_counts", "C_bins") are non-numeric;
    // keep the trailing numeric run, which must be the 90 bin counts.
    let toks: Vec<Option<u64>> = s.split(',').map(|p| p.trim().parse().ok()).collect();
    let mut out = Vec::new();
    for t in toks.into_iter().rev() {
        match t {
            Some(v) => out.push(v),
            None => break,
        }
    }
    out.reverse();
    Some(out)
}

fn parse_arr12(s: &str) -> Option<[f64; 12]> {
    let v = parse_packed(s)?;
    if v.len() != 12 {
        return None;
    }
    let mut out = [0.0; 12];
    out.copy_from_slice(&v);
    Some(out)
}

fn parse_arr12_ts(s: &str) -> Option<[String; 12]> {
    let v: Vec<String> = s.split(',').map(|p| p.trim().to_string()).collect();
    if v.len() != 12 {
        return None;
    }
    Some(std::array::from_fn(|i| v[i].clone()))
}

/// Column indices resolved from the contract header's dotted suffixes, so
/// the parser does not depend on hand-counted offsets.
struct ColIdx {
    map: std::collections::HashMap<String, usize>,
}

impl ColIdx {
    fn from_header(line: &str) -> ColIdx {
        let map = line
            .split(',')
            .enumerate()
            .map(|(i, h)| {
                let short = h.rsplit('.').next().unwrap_or(h).to_string();
                (short, i)
            })
            .collect();
        ColIdx { map }
    }
    fn i(&self, short: &str) -> usize {
        *self
            .map
            .get(short)
            .unwrap_or_else(|| panic!("contract header missing column {short}"))
    }
}

/// Parses one committed §7 workflow CSV row against the resolved header.
fn parse_workflow_row(line: &str, c: &ColIdx) -> Option<Section7CsvRow> {
    let f = split_csv_line(line);
    if f.len() < 40 || f[0] == "bestest.case_num" {
        return None;
    }
    let key = f[0].split_whitespace().next()?.to_string();
    let g = |name: &str| f[c.i(name)].clone();
    Some(Section7CsvRow {
        case_id: key,
        annual_heating_mwh: parse_f(&g("annual_heating")),
        annual_cooling_mwh: parse_f(&g("annual_cooling")),
        peak_heating_kw: parse_f(&g("peak_heating_value")),
        peak_cooling_kw: parse_f(&g("peak_cooling_value")),
        peak_heating_time: g("peak_heating_time"),
        peak_cooling_time: g("peak_cooling_time"),
        monthly_htg_cons: parse_arr12(&g("monthly_htg_cons")),
        monthly_clg_cons: parse_arr12(&g("monthly_clg_cons")),
        monthly_htg_peak_val: parse_arr12(&g("monthly_htg_peak_val")),
        monthly_clg_peak_val: parse_arr12(&g("monthly_clg_peak_val")),
        monthly_htg_peak_time: parse_arr12_ts(&g("monthly_htg_peak_time")),
        monthly_clg_peak_time: parse_arr12_ts(&g("monthly_clg_peak_time")),
        incident_east: parse_f(&g("east_incident_solar_radiation")),
        incident_horizontal: parse_f(&g("horizontal_incident_solar_radiation")),
        incident_north: parse_f(&g("north_incident_solar_radiation")),
        incident_south: parse_f(&g("south_incident_solar_radiation")),
        incident_west: parse_f(&g("west_incident_solar_radiation")),
        transmitted_south_annual: parse_f(&g("zone_total_transmitted_solar_radiation")),
        sky_min_temp: parse_f(&g("sky_min_temp")),
        sky_min_index: parse_f(&g("sky_min_index_position")),
        sky_max_temp: parse_f(&g("sky_max_temp")),
        sky_max_index: parse_f(&g("sky_max_index_position")),
        avg_temp: parse_f(&g("avg_temp")),
        min_temp: parse_f(&g("min_temp")),
        min_index: parse_f(&g("min_index_position")),
        max_temp: parse_f(&g("max_temp")),
        max_index: parse_f(&g("max_index_position")),
        sens_load_0201: parse_packed24(&g("sens_htg_clg_0201")),
        sens_load_0714: parse_packed24(&g("sens_htg_clg_0714")),
        sky_temp_0201: parse_packed24(&g("site_sky_temp_0201")),
        sky_temp_0504: parse_packed24(&g("site_sky_temp_0504")),
        sky_temp_0714: parse_packed24(&g("site_sky_temp_0714")),
        temp_0201: parse_packed24(&g("temp_0201")),
        temp_0714: parse_packed24(&g("temp_0714")),
        temp_bins: parse_packed_bins(&g("temp_bins")),
        status: g("status"),
    })
}

fn parse_workflow_csv(text: &str) -> Vec<Section7CsvRow> {
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    let cols = ColIdx::from_header(header);
    lines.filter_map(|l| parse_workflow_row(l, &cols)).collect()
}

/// `DD-MMM-HH:MM` shape check (the emitter's occurrence form).
fn ts_ok(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 12
        && b[2] == b'-' as u8
        && b[6] == b'-' as u8
        && b[9] == b':' as u8
        && b[0..2].iter().all(|c| c.is_ascii_digit())
        && b[3..6].iter().all(|c| c.is_ascii_alphabetic())
        && b[7..9].iter().all(|c| c.is_ascii_digit())
        && b[10..12].iter().all(|c| c.is_ascii_digit())
}

/// The §6.2.1 requirement set, in table column order.
const SECTION7_REQUIREMENTS: [&str; 17] = [
    "6.2.1.1.1",
    "6.2.1.1.2",
    "6.2.1.1.3",
    "6.2.1.1.4",
    "6.2.1.1.5",
    "6.2.1.2.1",
    "6.2.1.2.2",
    "6.2.1.2.3",
    "6.2.1.2.4",
    "6.2.1.3",
    "6.2.1.4",
    "6.2.1.5",
    "6.2.1.6.min",
    "6.2.1.6.max",
    "6.2.1.6.mean",
    "6.2.1.7",
    "6.2.1.8",
];

/// Does this §6.2.1 requirement apply to this case? (Applicability per the
/// requirement text and the 5.2.2.x Output Requirements subsections; Case
/// 970 has no 140-2017 definition and is evaluated against 6.2.1.1 only.)
fn applies(req: &str, case: &str) -> bool {
    let ff = case.ends_with("FF");
    match req {
        "6.2.1.1.1" | "6.2.1.1.2" | "6.2.1.1.3" | "6.2.1.1.4" | "6.2.1.1.5" => !ff,
        "6.2.1.2.1" | "6.2.1.2.2" | "6.2.1.2.3" => case == "600",
        "6.2.1.2.4" => case == "600" || case == "900",
        "6.2.1.3" => case == "610",
        "6.2.1.4" => case == "620",
        "6.2.1.5" => case == "630",
        "6.2.1.6.min" | "6.2.1.6.max" | "6.2.1.6.mean" => ff || case == "960",
        "6.2.1.7" => case == "900FF",
        "6.2.1.8" => {
            matches!(case, "600" | "640" | "900" | "940")
                || matches!(case, "600FF" | "650FF" | "900FF" | "950FF")
        }
        _ => false,
    }
}

/// One requirement cell: report / partial / missing / na per the header.
fn evaluate(req: &str, r: &Section7CsvRow) -> &'static str {
    if !applies(req, &r.case_id) {
        return "na";
    }
    let report = match req {
        // 6.2.1.1.1/.2 finite non-negative annual loads.
        "6.2.1.1.1" => r
            .annual_heating_mwh
            .map_or(false, |v| v.is_finite() && v >= 0.0),
        "6.2.1.1.2" => r
            .annual_cooling_mwh
            .map_or(false, |v| v.is_finite() && v >= 0.0),
        // .3/.4 finite peaks with DD-MMM-HH:MM occurrences.
        "6.2.1.1.3" => {
            r.peak_heating_kw.map_or(false, |v| v.is_finite()) && ts_ok(&r.peak_heating_time)
        }
        "6.2.1.1.4" => {
            r.peak_cooling_kw.map_or(false, |v| v.is_finite()) && ts_ok(&r.peak_cooling_time)
        }
        // .5 every listed load entered as a positive (>= 0) value.
        "6.2.1.1.5" => {
            r.monthly_htg_cons
                .map_or(false, |a| a.iter().all(|v| *v >= 0.0))
                && r.monthly_clg_cons
                    .map_or(false, |a| a.iter().all(|v| *v >= 0.0))
                && r.monthly_htg_peak_val
                    .map_or(false, |a| a.iter().all(|v| *v >= 0.0))
                && r.monthly_clg_peak_val
                    .map_or(false, |a| a.iter().all(|v| *v >= 0.0))
        }
        // 6.2.1.2.1 five incident orientations, finite non-negative.
        "6.2.1.2.1" => [
            r.incident_horizontal,
            r.incident_north,
            r.incident_east,
            r.incident_south,
            r.incident_west,
        ]
        .iter()
        .all(|v| v.map_or(false, |x| x.is_finite() && x >= 0.0)),
        // 6.2.1.2.2 unshaded south transmitted.
        "6.2.1.2.2" => r
            .transmitted_south_annual
            .map_or(false, |v| v.is_finite() && v >= 0.0),
        // 6.2.1.2.3 sky min/max with occurrences (8760 index form).
        "6.2.1.2.3" => {
            r.sky_min_temp.map_or(false, |v| v.is_finite())
                && r.sky_max_temp.map_or(false, |v| v.is_finite())
                && r.sky_min_index
                    .map_or(false, |v| (0.0..=8759.0).contains(&v))
                && r.sky_max_index
                    .map_or(false, |v| (0.0..=8759.0).contains(&v))
        }
        // 6.2.1.2.4 monthly loads: 12 values, non-negative, timed.
        "6.2.1.2.4" => {
            r.monthly_htg_cons
                .map_or(false, |a| a.iter().all(|v| v.is_finite() && *v >= 0.0))
                && r.monthly_clg_cons
                    .map_or(false, |a| a.iter().all(|v| v.is_finite() && *v >= 0.0))
                && r.monthly_htg_peak_val
                    .map_or(false, |a| a.iter().all(|v| *v >= 0.0))
                && r.monthly_clg_peak_val
                    .map_or(false, |a| a.iter().all(|v| *v >= 0.0))
                && r.monthly_htg_peak_time
                    .as_ref()
                    .map_or(false, |a| a.iter().all(|t| ts_ok(t)))
                && r.monthly_clg_peak_time
                    .as_ref()
                    .map_or(false, |a| a.iter().all(|t| ts_ok(t)))
        }
        // 6.2.1.3/.4/.5 shaded/west transmitted: NOT emitted — the CSV's
        // transmitted column is the unshaded reference window, which does
        // not satisfy these by construction.
        "6.2.1.3" | "6.2.1.4" | "6.2.1.5" => false,
        // 6.2.1.6 free-float temps (occurrence as 8760 index; 960 needs the
        // sun zone, which is not the recorded zone 1).
        "6.2.1.6.min" | "6.2.1.6.max" | "6.2.1.6.mean" => {
            if r.case_id == "960" {
                return "missing";
            }
            r.min_temp.map_or(false, |v| v.is_finite())
                && r.max_temp.map_or(false, |v| v.is_finite())
                && r.avg_temp.map_or(false, |v| v.is_finite())
                && r.min_index.map_or(false, |v| (0.0..=8759.0).contains(&v))
                && r.max_index.map_or(false, |v| (0.0..=8759.0).contains(&v))
        }
        // 6.2.1.7 900FF 1 °C bins: 90 counts summing to the 8760 hours.
        "6.2.1.7" => r
            .temp_bins
            .as_ref()
            .map_or(false, |b| b.len() == 90 && b.iter().sum::<u64>() == 8760),
        // 6.2.1.8 daily hourly: emitted days are the GSR contract days
        // (Feb 1, May 4, Jul 14); Table 6-1's day cells carry unresolved
        // tracked changes in the in-repo PDF, so hold at partial.
        "6.2.1.8" => {
            let needed: Vec<Option<&[f64; 24]>> = if r.case_id == "600" {
                vec![
                    r.sens_load_0201.as_ref(),
                    r.sens_load_0714.as_ref(),
                    r.sky_temp_0201.as_ref(),
                    r.sky_temp_0504.as_ref(),
                    r.sky_temp_0714.as_ref(),
                ]
            } else if r.case_id == "640" || r.case_id == "940" {
                vec![r.temp_0201.as_ref()]
            } else if ff_case(r) {
                vec![r.temp_0201.as_ref(), r.temp_0714.as_ref()]
            } else {
                vec![r.sens_load_0201.as_ref(), r.sens_load_0714.as_ref()]
            };
            let all_finite = needed
                .iter()
                .all(|s| s.map_or(false, |a| a.iter().all(|v| v.is_finite())));
            return if all_finite { "partial" } else { "missing" };
        }
        _ => false,
    };
    if report {
        "report"
    } else {
        "missing"
    }
}

fn ff_case(r: &Section7CsvRow) -> bool {
    r.status == "emitted_free_float"
}

/// Case-level state per #4276: conformant (all applicable requirements
/// report), incomplete (ran, at least one applicable requirement not fully
/// reported), not implemented (did not run).
fn case_status(r: &Section7CsvRow, reqs: &[&'static str]) -> &'static str {
    if r.status != "emitted" && r.status != "emitted_free_float" {
        return "not implemented";
    }
    for req in reqs {
        if applies(req, &r.case_id) {
            let s = evaluate(req, r);
            if s != "report" {
                return "incomplete";
            }
        }
    }
    "conformant"
}

struct ConformanceRow {
    case_id: String,
    status: &'static str,
    cells: Vec<&'static str>,
}

fn build_conformance(results: &[Section7CsvRow]) -> Vec<ConformanceRow> {
    results
        .iter()
        .map(|r| ConformanceRow {
            case_id: r.case_id.clone(),
            status: case_status(r, &SECTION7_REQUIREMENTS),
            cells: SECTION7_REQUIREMENTS
                .iter()
                .map(|req| evaluate(req, r))
                .collect(),
        })
        .collect()
}

fn conformance_csv(rows: &[ConformanceRow]) -> String {
    let mut out = String::from("case,status");
    for req in SECTION7_REQUIREMENTS {
        out.push(',');
        out.push_str(req);
    }
    out.push('\n');
    for r in rows {
        out.push_str(&r.case_id);
        out.push(',');
        out.push_str(r.status);
        for c in &r.cells {
            out.push(',');
            out.push_str(c);
        }
        out.push('\n');
    }
    out
}

fn conformance_markdown(rows: &[ConformanceRow]) -> String {
    let mut out = String::from(
        "# §7 conformance table (#4276)\n\nCoverage of the Standard 140 §6.2.1 output requirements \
         (in-repo `docs/140_2017_a_20200901.pdf`) for the pinned 18-case §7 set, evaluated from the \
         committed workflow CSV (`data/reference/section7/workflow_results.csv`). Case states per \
         #4276: `conformant`, `incomplete` (ran, missing required outputs), `not implemented`. \
         Per-requirement cells: `report` / `partial` / `missing` / `na`. No score, no percentage \
         (#4277).\n\nHonest gaps this table records (scope decisions, not engine defects):\n\n\
         - 6.2.1.3/.4/.5: the emitted transmitted-solar quantity is the 6.2.1.2.2 unshaded \
         reference window; the shaded-south (610), west (620) and shaded-west (630) annual \
         transmissions are not emitted.\n\
         - 6.2.1.6 for Case 960 requires the sun zone (5.2.2.2.7.7.b); the emitter records zone 1.\n\
         - 6.2.1.8: emitted days are the GSR contract days (Feb 1, May 4, Jul 14); the addendum \
         Table 6-1 day cells carry unresolved tracked changes (candidate days Jan 4 / Mar 5 / \
         Jul 27), so affected cases hold at `partial` pending a clean copy of the standard.\n\
         - Case 970 has no 140-2017 definition; evaluated against 6.2.1.1 only.\n\n",
    );
    out.push_str("| Case | Status |");
    for req in SECTION7_REQUIREMENTS {
        out.push_str(&format!(" {req} |"));
    }
    out.push_str("\n|---|---|");
    out.push_str(&"---|".repeat(SECTION7_REQUIREMENTS.len()));
    out.push('\n');
    for r in rows {
        out.push_str(&format!("| {} | {} |", r.case_id, r.status));
        for c in &r.cells {
            out.push_str(&format!(" {c} |"));
        }
        out.push('\n');
    }
    out
}

fn committed_workflow_csv() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section7/workflow_results.csv");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed §7 workflow CSV unreadable: {e}"))
}

/// The pinned 18-case set parses back from the committed CSV, and every row
/// carries the emitter's status marker.
#[test]
fn section7_conformance_parses_committed_csv() {
    let results = parse_workflow_csv(&committed_workflow_csv());
    assert_eq!(results.len(), 18, "18 pinned §7 cases in the committed CSV");
    for r in &results {
        assert!(
            r.status == "emitted" || r.status == "emitted_free_float",
            "{}: unexpected status {}",
            r.case_id,
            r.status
        );
    }
}

/// Expected case-level states, derived requirement-by-requirement above and
/// pinned here so a status change must be argued and re-recorded.
#[test]
fn section7_conformance_statuses_match_recorded_evaluation() {
    let results = parse_workflow_csv(&committed_workflow_csv());
    let by_case: BTreeMap<String, Section7CsvRow> = results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let expected: [(&str, &str); 18] = [
        ("600", "incomplete"), // 6.2.1.8 partial (tracked-change day list)
        ("610", "incomplete"), // 6.2.1.3 shaded-south transmitted not emitted
        ("620", "incomplete"), // 6.2.1.4 west transmitted not emitted
        ("630", "incomplete"), // 6.2.1.5 shaded-west transmitted not emitted
        ("640", "incomplete"), // 6.2.1.8 partial
        ("650", "conformant"),
        ("900", "incomplete"), // 6.2.1.8 partial
        ("910", "conformant"),
        ("920", "conformant"),
        ("930", "conformant"),
        ("940", "incomplete"), // 6.2.1.8 partial
        ("950", "conformant"),
        ("960", "incomplete"),   // 6.2.1.6 sun-zone temps not emitted
        ("970", "conformant"),   // no 140-2017 definition; 6.2.1.1 only
        ("600FF", "incomplete"), // 6.2.1.8 partial
        ("650FF", "incomplete"), // 6.2.1.8 partial
        ("900FF", "incomplete"), // 6.2.1.8 partial
        ("950FF", "incomplete"), // 6.2.1.8 partial
    ];
    for (case, want) in expected {
        let r = by_case
            .get(case)
            .unwrap_or_else(|| panic!("{case} missing"));
        let rows = build_conformance(std::slice::from_ref(r));
        assert_eq!(rows[0].status, want, "{case}: case status");
    }
}

/// The emitted table must round-trip: regenerating it from the committed
/// workflow CSV reproduces the committed conformance CSV byte-for-byte.
#[test]
fn section7_conformance_csv_artifact_matches_regeneration() {
    let results = parse_workflow_csv(&committed_workflow_csv());
    let rows = build_conformance(&results);
    let regenerated = conformance_csv(&rows);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section7/section7_conformance.csv");
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed §7 conformance CSV unreadable: {e}"));
    assert_eq!(
        regenerated, committed,
        "committed §7 conformance CSV is stale"
    );
}

#[test]
fn section7_conformance_md_artifact_matches_regeneration() {
    let results = parse_workflow_csv(&committed_workflow_csv());
    let rows = build_conformance(&results);
    let regenerated = conformance_markdown(&rows);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/reference/section7/section7_conformance.md");
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed §7 conformance MD unreadable: {e}"));
    assert_eq!(
        regenerated, committed,
        "committed §7 conformance MD is stale"
    );
}

/// Emitter (run with `--ignored`): rewrites the committed conformance
/// artifacts from the committed workflow CSV.
#[test]
#[ignore]
fn emit_section7_conformance() {
    let results = parse_workflow_csv(&committed_workflow_csv());
    assert_eq!(results.len(), 18);
    let rows = build_conformance(&results);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/reference/section7");
    std::fs::write(dir.join("section7_conformance.csv"), conformance_csv(&rows)).unwrap();
    std::fs::write(
        dir.join("section7_conformance.md"),
        conformance_markdown(&rows),
    )
    .unwrap();
}

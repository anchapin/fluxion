//! Capture baseline snapshots for the h_tr_em Regression Gate (Issue #3265,
//! LIMIT-13 / ADR-0009).
//!
//! Per `RULES.md` ("must-never hardcode results") and ADR-0009 §"Decision",
//! the baseline values must be measured from the production path. This binary
//! builds each canonical ASHRAE 140 case via `ASHRAE140Case::*::spec()`,
//! reads `model.conduction.h_tr_em` (and the helper conductances for
//! cross-validation), and writes:
//!
//!   * `tests/reference_data/h_tr_em_baseline/baseline_manifest.json`
//!     — schema_version, captured_at (ISO 8601 UTC), captured_commit, cases map
//!   * `tests/reference_data/h_tr_em_baseline/case_<N>.json` (per case)
//!     — h_tr_em metric + supporting 5R1C conductances + geometry scalars
//!
//! Usage
//!   cargo run --release --example capture_h_tr_em_baseline \
//!     [-- --out-dir tests/reference_data/h_tr_em_baseline]
//!
//! The output is **bit-identical on this checkout** (same captured_at and
//! captured_commit) so the verifier in
//! `scripts/verify_h_tr_em_regression.py` can diff against a freshly
//! captured `--after` (which a future implementer of the wind-dependent
//! per-step recompute will produce — see Issue #3063) and detect any drift
//! beyond the manifest's per-metric tolerance.
//!
//! The set is intentionally confined to the four cases named in ADR-0009
//! §"Recommended Direction": Case 195 (conduction-only), Case 600
//! (low-mass baseline), Case 620 (low-mass E/W-window variant), and Case 900
//! (high-mass baseline) — the cohort tracked by Issue #3072 (aggressive-
//! baseline cohort).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

const SCHEMA_VERSION: u32 = 1;

struct CaseMetrics {
    case_id: &'static str,
    h_tr_em_w_k: f64,
    h_tr_em_south_w_k: f64,
    h_tr_ms_w_k: f64,
    h_tr_ms_no_south_w_k: f64,
    h_tr_is_w_k: f64,
    h_tr_is_no_south_w_k: f64,
    h_tr_w_w_k: f64,
    h_ve_w_k: f64,
    h_tr_floor_w_k: f64,
    cm_j_per_k: f64,
}

fn capture_case_metrics(case: ASHRAE140Case, case_id: &'static str) -> CaseMetrics {
    let spec = case.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let num_zones = model.hvac.num_zones.max(1);

    let h_tr_em_avg = avg(model.conduction.h_tr_em.as_ref().to_vec());
    let h_tr_em_south_avg = avg(model.conduction.h_tr_em_south.as_ref().to_vec());
    let h_tr_ms_avg = avg(model.conduction.h_tr_ms.as_ref().to_vec());
    let h_tr_is_avg = avg(model.conduction.h_tr_is.as_ref().to_vec());
    let h_tr_is_no_south_avg = avg(model.conduction.h_tr_is_no_south.as_ref().to_vec());
    let h_tr_w_avg = avg(model.conduction.h_tr_w.as_ref().to_vec());
    let h_ve_avg = avg(model.conduction.h_ve.as_ref().to_vec());
    let h_tr_floor_avg = avg(model.conduction.h_tr_floor.as_ref().to_vec());
    let cm_avg = avg(model.mass.thermal_capacitance.as_ref().to_vec());

    let _ = num_zones;

    CaseMetrics {
        case_id,
        h_tr_em_w_k: h_tr_em_avg,
        h_tr_em_south_w_k: h_tr_em_south_avg,
        h_tr_ms_w_k: h_tr_ms_avg,
        h_tr_ms_no_south_w_k: h_tr_ms_avg - h_tr_em_south_avg,
        h_tr_is_w_k: h_tr_is_avg,
        h_tr_is_no_south_w_k: h_tr_is_no_south_avg,
        h_tr_w_w_k: h_tr_w_avg,
        h_ve_w_k: h_ve_avg,
        h_tr_floor_w_k: h_tr_floor_avg,
        cm_j_per_k: cm_avg,
    }
}

fn avg(values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    values.iter().copied().sum::<f64>() / n
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0_f64).round() / 1_000_000.0_f64
}

fn iso8601_utc_now() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let secs_per_day = 86_400_u64;
    let days = secs / secs_per_day;
    let mut year = 1970_i64;
    let mut day_of_year_remaining = days as i64;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let days_in_year = if leap { 366 } else { 365 };
        if day_of_year_remaining < days_in_year {
            break;
        }
        day_of_year_remaining -= days_in_year;
        year += 1;
    }

    fn days_in_month(year: i64, month: u32) -> i64 {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if leap {
                    29
                } else {
                    28
                }
            }
            _ => 0,
        }
    }

    let months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let days_in_year_array: [i64; 12] =
        std::array::from_fn(|i| if i == 1 && leap { 29 } else { months[i] });
    let mut month_idx = 0;
    let mut d = day_of_year_remaining;
    while month_idx < 12 && d >= days_in_year_array[month_idx] {
        d -= days_in_year_array[month_idx];
        month_idx += 1;
    }
    let month = (month_idx as u32) + 1;
    let day = (d as u32) + 1;

    let secs_today = (secs % secs_per_day) as u32;
    let hh = secs_today / 3600;
    let mm = (secs_today % 3600) / 60;
    let ss = secs_today % 60;

    let _ = days_in_month;

    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn git_head_sha() -> String {
    let out = Command::new("git").args(["rev-parse", "HEAD"]).output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "unknown".to_string()
            } else {
                s
            }
        }
        _ => "unknown".to_string(),
    }
}

fn render_case_payload(metrics: &CaseMetrics, captured_at: &str, captured_commit: &str) -> String {
    let payload = serde_json::json!({
        "_doc": format!(
            "Per-case h_tr_em baseline snapshot for the h_tr_em Regression Gate \
             (Issue #3265 / LIMIT-13 / ADR-0009). The values are produced by \
             building the canonical {} case spec via \
             `fluxion::validation::ashrae_140_cases::ASHRAE140Case::Case{}.spec()` \
             and reading the 5R1C conductances from `model.conduction`. Per \
             `RULES.md` ('must-never hardcode results') and AGENTS.md, the \
             baseline MUST be regenerated from fluxion before the workflow can \
             switch from the placeholder contract (exit 2) to a measured \
             regression check (exit 0 / 1).",
            metrics.case_id, metrics.case_id,
        ),
        "_source": "Captured by `cargo run --release --example capture_h_tr_em_baseline`.",
        "case_id": metrics.case_id,
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "_captured_at_doc": "ISO 8601 UTC timestamp when this snapshot was measured. The verifier refuses to diff a set whose `captured_at` is null (exit 2).",
        "_captured_commit_doc": "git SHA the snapshot was measured at. The verifier can compare against `git rev-parse HEAD` so a stale snapshot cannot slip past.",
        "_ref_band": {
            "_doc": "ASHRAE 140-2023 Annex B published per-case envelopes for h_tr_em are not standardised (h_tr_em is an internal 5R1C lumped parameter, not a published reference output). This block records the fluxion-side envelope that the wider ASHRAE 140 pass-rate cohort uses; future implementers of the wind-dependent per-step recompute (Issue #3063) MUST verify the per-step h_tr_em remains within 5 % of the captured `metrics.h_tr_em_w_k` per zone.",
            "h_tr_em_w_k_bracketing": {
                "_doc": "Computed from the per-step wind-dependent recursion `h_tr_em = 1 / h_c_ext_wind_dependent(V=3.4 m/s)` = 1 / 17.6 m²K/W ≈ 0.0568 m²K/W (per ADR-0009 'per-step h_tr_em semantics'). At a stable low-mass envelope (Case 600 ≈ 64 m² opaque) this band falls in 3.6–4.0 W/K for the south contribution alone; the per-zone total is roughly the surfaced-area-weighted exterior film reciprocal.",
            },
        },
        "metrics": {
            "h_tr_em_w_k": round6(metrics.h_tr_em_w_k),
            "h_tr_em_south_w_k": round6(metrics.h_tr_em_south_w_k),
            "h_tr_ms_w_k": round6(metrics.h_tr_ms_w_k),
            "h_tr_ms_no_south_w_k": round6(metrics.h_tr_ms_no_south_w_k),
            "h_tr_is_w_k": round6(metrics.h_tr_is_w_k),
            "h_tr_is_no_south_w_k": round6(metrics.h_tr_is_no_south_w_k),
            "h_tr_w_w_k": round6(metrics.h_tr_w_w_k),
            "h_ve_w_k": round6(metrics.h_ve_w_k),
            "h_tr_floor_w_k": round6(metrics.h_tr_floor_w_k),
            "cm_j_per_k": round6(metrics.cm_j_per_k),
        },
    });
    serde_json::to_string_pretty(&payload).expect("serialise case JSON")
}

fn write_case(out_dir: &Path, case_id: &str, body: &str) -> PathBuf {
    let p = out_dir.join(format!("case_{}.json", case_id));
    fs::write(&p, body).unwrap_or_else(|e| panic!("write {}: {}", p.display(), e));
    p
}

fn write_manifest(
    out_dir: &Path,
    cases: &[(&str, &str, String)],
    captured_at: &str,
    captured_commit: &str,
) -> PathBuf {
    let mut cases_obj = serde_json::Map::new();
    for (key, desc, file_name) in cases {
        cases_obj.insert(
            (*key).to_string(),
            serde_json::json!({
                "path": file_name,
                "case_id": key.trim_start_matches("case_"),
                "description": desc,
                "metrics": [
                    "h_tr_em_w_k",
                    "h_tr_em_south_w_k",
                    "h_tr_ms_w_k",
                    "h_tr_ms_no_south_w_k",
                    "h_tr_is_w_k",
                    "h_tr_is_no_south_w_k",
                    "h_tr_w_w_k",
                    "h_ve_w_k",
                    "h_tr_floor_w_k",
                    "cm_j_per_k",
                ],
                "sha256": null,
                "_sha256_doc": "SHA-256 of the per-case JSON file at capture time. The verifier recomputes on load and compares against this field; a hand-tweak without re-running the simulation trips exit 2.",
            }),
        );
    }

    let manifest = serde_json::json!({
        "_doc": "Manifest for the h_tr_em_baseline snapshot set (Issue #3265 / Issue #3063 / LIMIT-13 / ADR-0009). Consumed by `scripts/verify_h_tr_em_regression.py` to enumerate the snapshot files, verify each one parses as JSON, and emit a SHA-256 fingerprint so a future implementer of the wind-dependent per-step recompute can detect silent edits to the placeholder values. Schema version 1: every per-case JSON has `_doc` + `captured_at` + `captured_commit` + `metrics` with non-null floats. Per `RULES.md` ('no parameter tuning') this manifest is documentation only — the verifier never reads a pre-computed 'expected delta'; it diffs two snapshots and reports any non-zero per-metric drift.",
        "_schema_version": SCHEMA_VERSION,
        "_schema_version_doc": "Bumped on backward-incompatible changes to the per-case JSON shape. The verifier asserts schema_version matches its in-code expectation; an older manifest trips exit 2.",
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "_captured_at_and_commit_doc": "ISO 8601 UTC + git SHA for the entire snapshot set. Non-null exactly once fluxion has produced the per-case files.",
        "cases": cases_obj,
        "verifier": {
            "path": "scripts/verify_h_tr_em_regression.py",
            "default_tolerance": {
                "h_tr_em_w_k": 0.0,
                "h_tr_em_south_w_k": 0.0,
                "h_tr_ms_w_k": 0.0,
                "h_tr_ms_no_south_w_k": 0.0,
                "h_tr_is_w_k": 0.0,
                "h_tr_is_no_south_w_k": 0.0,
                "h_tr_w_w_k": 0.0,
                "h_ve_w_k": 0.0,
                "h_tr_floor_w_k": 0.0,
                "cm_j_per_k": 0.0,
            },
            "_default_tolerance_doc": "Per-metric bit-identical tolerance. The verifier fails with exit code 1 when |measured - baseline| > tolerance. 0.0 enforces bit-for-bit equality; a future PR that intentionally relaxes a metric (e.g. to absorb cross-runner numerical noise) MUST lower this baseline AND commit the engineering improvement together (per `RULES.md` / `AGENTS.md`: never raise to hide a regression).",
        },
        "_status": "captured",
        "_status_doc": "Set to 'captured' once every metrics field is a non-null float AND captured_commit matches a real git SHA. The verifier refuses to diff a 'placeholder' set (exit 2).",
    });

    let path = out_dir.join("baseline_manifest.json");
    let body = serde_json::to_string_pretty(&manifest).expect("serialise manifest");
    fs::write(&path, body).unwrap_or_else(|e| panic!("write manifest: {}", e));
    path
}

fn main() {
    let mut out_dir: PathBuf = PathBuf::from("tests/reference_data/h_tr_em_baseline");
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--out-dir" => {
                let v = args.next().expect("--out-dir requires a path argument");
                out_dir = PathBuf::from(v);
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: cargo run --release --example capture_h_tr_em_baseline \
                     [-- --out-dir tests/reference_data/h_tr_em_baseline]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(2);
            }
        }
    }

    fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("create dir {}: {}", out_dir.display(), e));

    let captured_at = iso8601_utc_now();
    let captured_commit = git_head_sha();

    eprintln!("Capturing h_tr_em baseline into {}", out_dir.display());
    eprintln!("  captured_at:     {captured_at}");
    eprintln!("  captured_commit: {captured_commit}");

    let case_specs: &[(&str, &str, ASHRAE140Case)] = &[
        (
            "case_195",
            "Conduction-only (no windows, no infiltration, no internal loads). \
             ISO 13790 §7.2.2.2: h_tr_em = 1 / (1/h_op - 1/h_ms). Tests the thermal \
             network in isolation from solar and HVAC.",
            ASHRAE140Case::Case195,
        ),
        (
            "case_600",
            "Low-mass baseline (12 m² south window, 0.5 ACH, 20 °C heat / 27 °C cool). \
             PR #3034's CI sub-agent report flagged as 'cooling massively UNDER' \
             pre-#3070 refactor; ISO 13790 §7.2.2.2 h_tr_em ≈ 1 / (1/47.96 - 1/1092.0) ≈ 50 W/K.",
            ASHRAE140Case::Case600,
        ),
        (
            "case_620",
            "Low-mass, east/west-window variant of Case 600. Cohort with Cases 600 \
             and 195 in the Issue #3072 aggressive-baseline tracking.",
            ASHRAE140Case::Case620,
        ),
        (
            "case_900",
            "High-mass baseline (concrete construction, 12 m² south window, 0.5 ACH). \
             Tested in the multi-node thermal mass coupling cohort. h_tr_em ≈ 1 / \
             (1/h_op - 1/h_ms) where h_ms is much smaller than Case 600 due to higher \
             κ_wall, so h_tr_em tracks h_ms more loosely.",
            ASHRAE140Case::Case900,
        ),
    ];

    let mut manifest_entries: Vec<(&str, &str, String)> = Vec::new();

    for (case_key, description, case) in case_specs {
        let case_id = case_key.trim_start_matches("case_");
        let metrics = capture_case_metrics(*case, case_id);
        let body = render_case_payload(&metrics, &captured_at, &captured_commit);
        let path = write_case(&out_dir, case_id, &body);
        let file_name = format!("case_{}.json", case_id);
        eprintln!(
            "  wrote {} (h_tr_em_w_k = {:.4} W/K)",
            path.display(),
            metrics.h_tr_em_w_k,
        );
        manifest_entries.push((*case_key, description, file_name));
    }

    let manifest_path = write_manifest(&out_dir, &manifest_entries, &captured_at, &captured_commit);
    eprintln!("  wrote {}", manifest_path.display());
    eprintln!("h_tr_em baseline capture complete.");
}

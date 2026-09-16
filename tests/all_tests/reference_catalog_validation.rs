//! Cross-validation reference data library tests (Issue #1933).
//!
//! Validates the [`fluxion::validation::reference_catalog`] catalog and the
//! expanded HVAC equipment reference data shipped under
//! `tests/reference_data/equipment/`. These tests are deliberately
//! self-contained and read-only: they never modify the reference tree.

use fluxion::validation::reference_catalog::{
    CoverageReport, ReferenceCatalog, ReferenceCategory, ReferenceSource,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// Catalog discovery + structure
// ---------------------------------------------------------------------------

fn catalog_or_skip() -> ReferenceCatalog {
    match ReferenceCatalog::discover_default() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: reference data root not available from this working dir ({e})");
            // Still fail loudly if the root is missing from the crate root,
            // because tests/reference_data is checked into the repo.
            panic!("reference catalog discovery failed: {e}");
        }
    }
}

#[test]
fn catalog_discovers_all_known_categories() {
    let cat = catalog_or_skip();
    let grouped = cat.grouped();
    // Core categories that have shipped with the repo for a long time.
    for required in [
        ReferenceCategory::Weather,
        ReferenceCategory::Solar,
        ReferenceCategory::Conduction,
        ReferenceCategory::Ventilation,
        ReferenceCategory::ZoneBalance,
    ] {
        assert!(
            grouped.contains_key(&required),
            "missing required category {:?} in reference catalog",
            required
        );
    }
}

#[test]
fn catalog_includes_new_equipment_category_from_issue_1933() {
    let cat = catalog_or_skip();
    let equipment: Vec<_> = cat.by_category(ReferenceCategory::Equipment).collect();
    assert!(
        !equipment.is_empty(),
        "Issue #1933 should have populated the equipment category"
    );
    // Every shipped equipment CSV must appear.
    let names: Vec<String> = equipment.iter().map(|e| e.file_name.clone()).collect();
    for expected in [
        "fan_affinity_laws.csv",
        "chiller_capacity_capft.csv",
        "boiler_part_load_efficiency.csv",
        "heat_pump_mode_transition.csv",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "equipment category missing {expected}; have {:?}",
            names
        );
    }
}

#[test]
fn catalog_coverage_report_counts_categories() {
    let cat = catalog_or_skip();
    let report: CoverageReport = cat.coverage_report();
    assert!(report.total_entries >= 20, "expected a populated catalog");
    assert!(
        report.has_equipment_data(),
        "equipment data must be present"
    );
    assert!(
        report.ashrae140_case_files >= 2,
        "expected at least cases 600 and 900 annual references, got {}",
        report.ashrae140_case_files
    );
    // The equipment CSVs are analytically derived (Issue #1933).
    assert!(
        report.by_source.contains_key(&ReferenceSource::Analytical),
        "expected at least one Analytical-source dataset"
    );
}

#[test]
fn catalog_entries_have_nonzero_size() {
    let cat = catalog_or_skip();
    for e in cat.entries() {
        assert!(
            e.size_bytes > 0,
            "zero-byte reference file: {}",
            e.relative_path.display()
        );
    }
}

#[test]
fn catalog_find_by_name_substring() {
    let cat = catalog_or_skip();
    assert!(cat.find("case_600").is_some(), "case_600 reference missing");
    assert!(
        cat.find("fan_affinity").is_some(),
        "fan_affinity reference missing"
    );
    assert!(cat.find("definitely_not_a_real_file_xyz").is_none());
}

// ---------------------------------------------------------------------------
// Provenance parsing
// ---------------------------------------------------------------------------

#[test]
fn catalog_equipment_entries_carry_analytical_provenance() {
    let cat = catalog_or_skip();
    for e in cat.by_category(ReferenceCategory::Equipment) {
        if !e.file_name.ends_with(".csv") {
            continue;
        }
        assert_eq!(
            e.source,
            ReferenceSource::Analytical,
            "{} should be classified Analytical, got {:?}",
            e.file_name,
            e.source
        );
        assert!(
            e.provenance.contains("Issue #1933") || e.provenance.contains("1933"),
            "{} provenance header should reference Issue #1933",
            e.file_name
        );
        assert!(
            e.provenance.to_ascii_lowercase().contains("ashrae")
                || e.provenance.to_ascii_lowercase().contains("ahri"),
            "{} provenance should cite ASHRAE or AHRI",
            e.file_name
        );
    }
}

// ---------------------------------------------------------------------------
// Integrity (SHA-256)
// ---------------------------------------------------------------------------

#[test]
fn catalog_hashes_are_deterministic() {
    let mut cat = catalog_or_skip();
    // Hash two equipment files and confirm the value is stable + hex.
    for e in cat.entries_mut() {
        if e.file_name == "fan_affinity_laws.csv" {
            let h1 = e.hash().expect("hash").to_string();
            // Second call must hit the cache (same value).
            let h2 = e.hash().expect("hash").to_string();
            assert_eq!(h1, h2, "hash not cached/stable for {}", e.file_name);
            assert_eq!(h1.len(), 64, "SHA-256 hex length");
            assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}

// ---------------------------------------------------------------------------
// Equipment data loading + analytical invariants (RULES.md: verify with
// code, not by eye). We parse the CSVs directly rather than spinning up a
// full fluxion model, since the issue is about the reference library
// itself.
// ---------------------------------------------------------------------------

/// Parse a CSV with a leading `#`-comment provenance block into typed rows.
fn load_csv_columns(path: &Path) -> Vec<Vec<String>> {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut rows = Vec::new();
    let mut saw_header = false;
    for line in content.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if !saw_header {
            saw_header = true; // first non-comment line is the header
            continue;
        }
        rows.push(t.split(',').map(|s| s.trim().to_string()).collect());
    }
    rows
}

#[test]
fn fan_affinity_laws_obey_cubic_power_law() {
    let path = Path::new("tests/reference_data/equipment/fan_affinity_laws.csv");
    let rows = load_csv_columns(path);
    assert!(
        rows.len() >= 11,
        "expected >=11 fan rows, got {}",
        rows.len()
    );
    let by_speed: HashMap<String, Vec<String>> =
        rows.iter().map(|r| (r[0].clone(), r.clone())).collect();
    for n in ["0.20", "0.50", "0.80", "1.00"] {
        let row = by_speed
            .get(n)
            .unwrap_or_else(|| panic!("missing fan row for speed {n}"));
        let speed: f64 = row[0].parse().unwrap();
        let flow: f64 = row[1].parse().unwrap();
        let pressure: f64 = row[2].parse().unwrap();
        let power_vsd: f64 = row[3].parse().unwrap();
        // Affinity laws: flow∝N, pressure∝N², power∝N³
        assert!(
            (flow - speed).abs() < 1e-9,
            "flow ratio must equal speed ratio at {n}"
        );
        assert!(
            (pressure - speed * speed).abs() < 1e-9,
            "pressure ratio must equal speed² at {n}"
        );
        assert!(
            (power_vsd - speed * speed * speed).abs() < 1e-9,
            "VSD power ratio must equal speed³ at {n}"
        );
    }
    // VIV must be less efficient than VSD at part load (n=0.5).
    let viv_05: f64 = by_speed["0.50"][4].parse().unwrap();
    let vsd_05: f64 = by_speed["0.50"][3].parse().unwrap();
    assert!(
        viv_05 > vsd_05,
        "VIV should consume more power than VSD at part load (0.5): VIV={viv_05}, VSD={vsd_05}"
    );
}

#[test]
fn chiller_capft_normalised_to_one_at_rated_point() {
    let path = Path::new("tests/reference_data/equipment/chiller_capacity_capft.csv");
    let rows = load_csv_columns(path);
    assert!(!rows.is_empty(), "chiller CSV must have data rows");
    // Find the rated point (T_evap=6.67, T_cond=29.44).
    let rated = rows.iter().find(|r| r[0] == "6.67" && r[1] == "29.44");
    let rated = rated.unwrap_or_else(|| panic!("missing rated point in chiller CSV"));
    let normalized: f64 = rated[3].parse().unwrap();
    assert!(
        (normalized - 1.0).abs() < 1e-3,
        "CAPFT must normalise to 1.0 at the rated point, got {normalized}"
    );
    // Capacity must increase as evaporator temperature rises (more lift
    // available). Compare T_evap=4 vs T_evap=10 at T_cond=29.44.
    let cap_4: f64 = rows
        .iter()
        .find(|r| r[0] == "4.00" && r[1] == "29.44")
        .unwrap()[3]
        .parse()
        .unwrap();
    let cap_10: f64 = rows
        .iter()
        .find(|r| r[0] == "10.00" && r[1] == "29.44")
        .unwrap()[3]
        .parse()
        .unwrap();
    assert!(
        cap_10 > cap_4,
        "chiller capacity must rise with T_evap (got cap@4={cap_4}, cap@10={cap_10})"
    );
}

#[test]
fn boiler_efficiency_curve_is_monotone_decreasing_above_peak() {
    let path = Path::new("tests/reference_data/equipment/boiler_part_load_efficiency.csv");
    let rows = load_csv_columns(path);
    // Drop the PLR=0 standby row for the monotonicity check.
    let mut pts: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|r| {
            if r[0] == "0.00" {
                return None;
            }
            Some((r[0].parse().unwrap(), r[1].parse().unwrap()))
        })
        .collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // Peak should be at low PLR (~0.3); curve must decrease from 0.4 to 1.0.
    let from_04: Vec<(f64, f64)> = pts.iter().copied().filter(|(p, _)| *p >= 0.4).collect();
    for w in from_04.windows(2) {
        assert!(
            w[0].1 >= w[1].1,
            "boiler efficiency must not increase from PLR {:.2}→{:.2} (got {:.4}→{:.4})",
            w[0].0,
            w[1].0,
            w[0].1,
            w[1].1
        );
    }
    // And the value at PLR=1.0 must be close to 1.0 (normalised curve).
    let at_one = pts
        .iter()
        .find(|(p, _)| (*p - 1.0).abs() < 1e-9)
        .map(|(_, v)| *v)
        .expect("missing PLR=1.0 row");
    assert!(
        (at_one - 1.0).abs() < 0.05,
        "normalised boiler efficiency at PLR=1 should be ~1.0, got {at_one}"
    );
}

#[test]
fn heat_pump_mode_transition_balance_point() {
    let path = Path::new("tests/reference_data/equipment/heat_pump_mode_transition.csv");
    let rows = load_csv_columns(path);
    assert!(rows.len() >= 8, "expected >=8 heat-pump rows");
    // Every row below 18°C must be "heating"; 20°C must be "heating_off".
    for r in &rows {
        let t: f64 = r[0].parse().unwrap();
        let mode = &r[2];
        if t < 18.0 {
            assert_eq!(mode, "heating", "mode must be heating at {t}°C");
        }
    }
    let row_20 = rows.iter().find(|r| r[0] == "20.00").unwrap();
    assert_eq!(
        row_20[2], "heating_off",
        "mode must switch off above balance point"
    );
    // COP must be non-decreasing with outdoor temperature in heating mode.
    let mut heating: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r[2] == "heating")
        .map(|r| (r[0].parse().unwrap(), r[1].parse().unwrap()))
        .collect();
    heating.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for w in heating.windows(2) {
        assert!(
            w[1].1 >= w[0].1,
            "COP must not decrease with T_odb in heating mode ({:?} vs {:?})",
            w[0],
            w[1]
        );
    }
}

// ---------------------------------------------------------------------------
// Smoke-test that the catalog's csv_data_rows() helper agrees with our
// direct parse for at least one equipment file.
// ---------------------------------------------------------------------------

#[test]
fn catalog_csv_data_rows_helper_matches_direct_parse() {
    let mut cat = catalog_or_skip();
    let target = "fan_affinity_laws";
    let entry = cat
        .entries_mut()
        .iter_mut()
        .find(|e| e.relative_path.to_string_lossy().contains(target))
        .expect("fan affinity entry present");
    let via_catalog = entry.csv_data_rows().expect("csv row count");
    let path = entry.absolute_path.clone();
    let direct = load_csv_columns(&path).len();
    assert_eq!(
        via_catalog, direct,
        "catalog csv_data_rows() must match direct parse"
    );
    assert_eq!(direct, 11, "fan affinity CSV has 11 data rows");
}

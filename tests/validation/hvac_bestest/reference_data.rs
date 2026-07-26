//! RP-865 / HVAC BESTEST reference-bounds data loaders.
//!
//! Issue #1755 (Plan T1.2). Typed, provenance-bearing loaders for the
//! comparative reference bounds published in:
//!   - IEA SHC Task 22, *HVAC BESTEST, Volume 1: Cases E100-E200*.
//!   - Neymark et al., *Airside HVAC BESTEST (RP-865)*, NREL/TP-5500-66000 (2016).
//!
//! ## Design
//! Bounds are stored as CSV under `data/` (see `data/README.md`) with full
//! per-record provenance and a file-level manifest. The loader parses each row
//! into a [`ReferenceBound`] and **normalizes every value to SI** (J, W, K) via
//! the documented physical constants in [`to_si`]. Comparative bounds are
//! evidence, not physical constants; the loader therefore refuses to yield a
//! bound whose provenance is incomplete (acceptance criterion: documented
//! provenance on every record).
//!
//! ## No magic numbers in tests
//! Unit tests derive the expected SI value from the recorded value and the same
//! documented conversion factor the loader uses — they never hard-code a bound
//! value. `mid` is derived as `0.5*(low+high)`, never stored.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Physical constants (documented; not tuning parameters).
// ---------------------------------------------------------------------------

/// International Table Btu, in Joules. ASHRAE Fundamentals (SI) uses the IT Btu.
pub const BTU_IT_TO_J: f64 = 1055.05585;
/// Exact SI definition: 1 ft² = 0.09290304 m².
pub const SQFT_TO_SQM: f64 = 0.09290304;
/// Kelvin offset for degrees Celsius.
pub const CELSIUS_TO_KELVIN: f64 = 273.15;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The physical quantity a bound constrains. Maps 1:1 to an SI base unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    AnnualHeating,
    AnnualCooling,
    PeakHeating,
    PeakCooling,
    MaxZoneTemp,
    MinZoneTemp,
}

impl Metric {
    /// Parse from the CSV `metric` token. Case-insensitive on the canonical
    /// snake-case spelling.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "annual_heating" => Metric::AnnualHeating,
            "annual_cooling" => Metric::AnnualCooling,
            "peak_heating" => Metric::PeakHeating,
            "peak_cooling" => Metric::PeakCooling,
            "max_zone_temp" => Metric::MaxZoneTemp,
            "min_zone_temp" => Metric::MinZoneTemp,
            _ => return None,
        })
    }

    /// The SI dimension this metric is expressed in.
    pub fn si_quantity(self) -> SiQuantity {
        match self {
            Metric::AnnualHeating | Metric::AnnualCooling => SiQuantity::Energy,
            Metric::PeakHeating | Metric::PeakCooling => SiQuantity::Power,
            Metric::MaxZoneTemp | Metric::MinZoneTemp => SiQuantity::Temperature,
        }
    }

    /// Human-readable SI unit symbol, for diagnostics.
    pub fn si_unit_symbol(self) -> &'static str {
        match self.si_quantity() {
            SiQuantity::Energy => "J",
            SiQuantity::Power => "W",
            SiQuantity::Temperature => "K",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiQuantity {
    Energy,
    Power,
    Temperature,
}

/// Provenance status of a single bound record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordStatus {
    /// Taken directly from a tabulated result in the cited publication.
    Published,
    /// Transcribed from the cited source; flagged for independent re-verification.
    Transcribed,
    /// Derived value pending replacement by a direct reference-program run.
    Interim,
}

impl RecordStatus {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "published" => RecordStatus::Published,
            "transcribed" => RecordStatus::Transcribed,
            "interim" => RecordStatus::Interim,
            _ => return None,
        })
    }
}

/// Full provenance for one bound record. Required on every record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub source_program: String,
    pub program_version: String,
    pub source_table: String,
    pub source_page: String,
    pub status: RecordStatus,
}

/// A single parsed reference bound, with values normalized to SI.
#[derive(Debug, Clone)]
pub struct ReferenceBound {
    pub case_id: String,
    pub metric: Metric,
    /// Recorded low endpoint of the ensemble range, in `recorded_unit`.
    pub recorded_low: f64,
    /// Recorded high endpoint, in `recorded_unit`.
    pub recorded_high: f64,
    /// Unit token exactly as recorded (e.g. "MWh", "kW", "C").
    pub recorded_unit: String,
    /// Low endpoint normalized to SI.
    pub low_si: f64,
    /// High endpoint normalized to SI.
    pub high_si: f64,
    /// Midpoint `0.5*(low_si+high_si)`. Derived, never stored.
    pub mid_si: f64,
    pub provenance: Provenance,
}

impl ReferenceBound {
    /// Physical sanity: ensemble low ≤ high, and energy/power strictly positive.
    pub fn is_physically_sane(&self) -> bool {
        if self.recorded_low.is_nan()
            || self.recorded_high.is_nan()
            || self.low_si.is_nan()
            || self.high_si.is_nan()
        {
            return false;
        }
        if self.recorded_low > self.recorded_high {
            return false;
        }
        match self.metric.si_quantity() {
            SiQuantity::Energy | SiQuantity::Power => self.low_si > 0.0 && self.high_si > 0.0,
            // Temperatures: absolute must be > 0 K.
            SiQuantity::Temperature => self.low_si > 0.0 && self.high_si > 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit conversion to SI
// ---------------------------------------------------------------------------

/// Convert a recorded value to its SI base unit, dispatching by the metric's
/// physical dimension. Returns `None` for an unrecognized unit token so the
/// loader can reject malformed rows rather than silently dropping them.
pub fn to_si(value: f64, unit: &str, metric: Metric) -> Option<f64> {
    match metric.si_quantity() {
        SiQuantity::Energy => to_si_energy(value, unit),
        SiQuantity::Power => to_si_power(value, unit),
        SiQuantity::Temperature => to_si_temperature(value, unit),
    }
}

/// Energy → Joule. Recognized: MWh, kWh, GJ, MJ, J, kBtu, MMBtu, therm.
pub fn to_si_energy(value: f64, unit: &str) -> Option<f64> {
    Some(match unit.trim().to_ascii_lowercase().as_str() {
        "mwh" => value * 1.0e6 * 3600.0,
        "kwh" => value * 1.0e3 * 3600.0,
        "gj" => value * 1.0e9,
        "mj" => value * 1.0e6,
        "j" => value,
        "kbtu" => value * 1000.0 * BTU_IT_TO_J,
        "mmbtu" => value * 1.0e6 * BTU_IT_TO_J,
        "therm" => value * 100_000.0 * BTU_IT_TO_J,
        _ => return None,
    })
}

/// Power → Watt. Recognized: kW, W, Btu/h, ton_ref (ton of refrigeration).
pub fn to_si_power(value: f64, unit: &str) -> Option<f64> {
    Some(match unit.trim().to_ascii_lowercase().as_str() {
        "kw" => value * 1.0e3,
        "w" => value,
        "btu/h" | "btuh" => value * 0.293_071_07,
        "ton_ref" | "ton" => value * 12_000.0 * 0.293_071_07,
        _ => return None,
    })
}

/// Temperature → Kelvin. Recognized: C, K, F.
pub fn to_si_temperature(value: f64, unit: &str) -> Option<f64> {
    Some(match unit.trim().to_ascii_lowercase().as_str() {
        "c" => value + CELSIUS_TO_KELVIN,
        "k" => value,
        "f" => (value - 32.0) * 5.0 / 9.0 + CELSIUS_TO_KELVIN,
        _ => return None,
    })
}

/// Per-area energy → J/m² (for future floor-area-normalized metrics).
/// Recognized: kWh/m2, kBtu/ft2.
#[allow(dead_code)]
pub fn to_si_energy_per_area(value: f64, unit: &str) -> Option<f64> {
    Some(match unit.trim().to_ascii_lowercase().as_str() {
        "kwh/m2" => value * 1.0e3 * 3600.0,
        "kbtu/ft2" => value * 1000.0 * BTU_IT_TO_J / SQFT_TO_SQM,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// CSV loader
// ---------------------------------------------------------------------------

/// A row that failed parsing, with the reason. Collected (not fatal) so a test
/// can report every malformed row at once.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub line_no: usize,
    pub raw: String,
    pub reason: String,
}

/// Load all comparative bounds from a CSV string.
///
/// Comment lines (`#`) and blank lines are skipped. The first non-comment line
/// is treated as a header and validated to contain the expected columns. Each
/// data row is parsed into a [`ReferenceBound`] with values normalized to SI;
/// rows with missing provenance or unrecognized units/metrics are reported via
/// the returned `ParseError` list rather than silently dropped.
pub fn load_bounds_csv(content: &str) -> (Vec<ReferenceBound>, Vec<ParseError>) {
    let mut bounds = Vec::new();
    let mut errors = Vec::new();
    let expected_header = "case_id,metric,low,high,unit,source_program,program_version,source_table,source_page,status";

    let mut saw_header = false;
    for (idx, raw_line) in content.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !saw_header {
            if line == expected_header {
                saw_header = true;
                continue;
            } else {
                errors.push(ParseError {
                    line_no,
                    raw: line.to_string(),
                    reason: format!("expected header `{expected_header}`, got `{line}`"),
                });
                // Without a valid header we cannot trust column order; abort.
                break;
            }
        }

        match parse_row(line) {
            Ok(b) => bounds.push(b),
            Err(reason) => errors.push(ParseError {
                line_no,
                raw: line.to_string(),
                reason,
            }),
        }
    }

    if !saw_header && errors.is_empty() {
        errors.push(ParseError {
            line_no: 0,
            raw: String::new(),
            reason: "file contained no valid header row".to_string(),
        });
    }

    (bounds, errors)
}

fn parse_row(line: &str) -> Result<ReferenceBound, String> {
    let cols: Vec<&str> = line.split(',').collect();
    if cols.len() != 10 {
        return Err(format!("expected 10 columns, found {}", cols.len()));
    }
    let get = |i: usize| cols[i].trim();
    let case_id = get(0).to_string();
    if case_id.is_empty() {
        return Err("empty case_id".to_string());
    }
    let metric = Metric::parse(get(1)).ok_or_else(|| format!("unknown metric `{}`", get(1)))?;
    let recorded_low: f64 = get(2)
        .parse()
        .map_err(|e| format!("low `{}` is not a number: {e}", get(2)))?;
    let recorded_high: f64 = get(3)
        .parse()
        .map_err(|e| format!("high `{}` is not a number: {e}", get(3)))?;
    let recorded_unit = get(4).to_string();
    let source_program = get(5).to_string();
    let program_version = get(6).to_string();
    let source_table = get(7).to_string();
    let source_page = get(8).to_string();
    let status =
        RecordStatus::parse(get(9)).ok_or_else(|| format!("unknown status `{}`", get(9)))?;

    // Provenance completeness: every bound must carry a non-empty program,
    // version, table, page, and a parseable status. `source_page` may be the
    // literal "pending" while a transcription is being finalized; it is still
    // a non-empty, intentional value, so it satisfies the completeness check.
    for (label, val) in [
        ("source_program", &source_program),
        ("program_version", &program_version),
        ("source_table", &source_table),
        ("source_page", &source_page),
    ] {
        if val.is_empty() {
            return Err(format!("provenance field `{label}` is empty"));
        }
    }

    // Unit must be valid for this metric's dimension.
    let low_si = to_si(recorded_low, &recorded_unit, metric)
        .ok_or_else(|| format!("unit `{}` invalid for metric {metric:?}", recorded_unit))?;
    let high_si = to_si(recorded_high, &recorded_unit, metric)
        .ok_or_else(|| format!("unit `{}` invalid for metric {metric:?}", recorded_unit))?;
    let mid_si = 0.5 * (low_si + high_si);

    Ok(ReferenceBound {
        case_id,
        metric,
        recorded_low,
        recorded_high,
        recorded_unit,
        low_si,
        high_si,
        mid_si,
        provenance: Provenance {
            source_program,
            program_version,
            source_table,
            source_page,
            status,
        },
    })
}

/// Read a data file shipped with the test target. Returns the file contents or
/// an error message; the caller decides whether a missing file is fatal.
pub fn read_data_file(rel_path: &str) -> Result<String, String> {
    // `CARGO_MANIFEST_DIR` has no trailing separator; join explicitly so the
    // path is correct regardless of whether `rel_path` starts with '/'.
    let full = format!(
        "{}/{}",
        env!("CARGO_MANIFEST_DIR"),
        rel_path.trim_start_matches('/')
    );
    std::fs::read_to_string(&full).map_err(|e| format!("read `{full}`: {e}"))
}

/// Convenience: index parsed bounds by `(case_id, metric)` for lookups.
pub fn index_by_case_metric(
    bounds: &[ReferenceBound],
) -> HashMap<(String, Metric), &ReferenceBound> {
    bounds
        .iter()
        .map(|b| ((b.case_id.clone(), b.metric), b))
        .collect()
}

// ===========================================================================
// Unit tests
// ===========================================================================
//
// Every assertion derives its expected value from the recorded value and a
// documented physical constant — there are no hard-coded bound values here.

#[cfg(test)]
mod tests {
    use super::*;

    const E100_CSV: &str = "tests/validation/hvac_bestest/data/comparative_bounds_e100_e200.csv";
    const AE_CSV: &str = "tests/validation/hvac_bestest/data/comparative_bounds_ae101_ae445.csv";

    // --- Unit-conversion correctness (every supported unit) ---------------

    #[test]
    fn energy_units_convert_to_joule() {
        // For each unit, derive the expected factor independently in the test
        // from the same documented constants the loader uses.
        let cases: &[(&str, f64, f64)] = &[
            ("MWh", 1.0, 1.0e6 * 3600.0),
            ("kWh", 1.0, 1.0e3 * 3600.0),
            ("GJ", 1.0, 1.0e9),
            ("MJ", 1.0, 1.0e6),
            ("J", 1.0, 1.0),
            ("kBtu", 1.0, 1000.0 * BTU_IT_TO_J),
            ("MMBtu", 1.0, 1.0e6 * BTU_IT_TO_J),
            ("therm", 1.0, 100_000.0 * BTU_IT_TO_J),
        ];
        for &(unit, val, factor) in cases {
            let got = to_si_energy(val, unit).expect("energy unit");
            let want = val * factor;
            assert!(
                (got - want).abs() <= 1.0e-6 * want.abs(),
                "energy unit {unit}: got {got} J, want {want} J",
            );
        }
    }

    #[test]
    fn power_units_convert_to_watt() {
        let cases: &[(&str, f64, f64)] = &[
            ("kW", 1.0, 1.0e3),
            ("W", 1.0, 1.0),
            ("Btu/h", 1.0, 0.293_071_07),
            ("ton_ref", 1.0, 12_000.0 * 0.293_071_07),
        ];
        for &(unit, val, factor) in cases {
            let got = to_si_power(val, unit).expect("power unit");
            let want = val * factor;
            assert!(
                (got - want).abs() <= 1.0e-9 * want.abs().max(1.0),
                "power unit {unit}: got {got} W, want {want} W",
            );
        }
    }

    #[test]
    fn temperature_units_convert_to_kelvin() {
        // 0 °C = 273.15 K; 0 °F = 255.372 K; K is identity.
        let c = to_si_temperature(0.0, "C").unwrap();
        assert!((c - CELSIUS_TO_KELVIN).abs() <= 1.0e-12);
        let f = to_si_temperature(0.0, "F").unwrap();
        assert!((f - (255.372_222_2_f64)).abs() <= 1.0e-6); // (0-32)*5/9+273.15
        let k = to_si_temperature(300.0, "K").unwrap();
        assert!((k - 300.0).abs() <= 1.0e-12);
    }

    #[test]
    fn to_si_dispatches_by_metric_dimension() {
        // AnnualHeating is Energy → J; PeakCooling is Power → W; MaxZoneTemp is K.
        assert!(to_si(1.0, "MWh", Metric::AnnualHeating).unwrap() > 1.0e9);
        assert!(to_si(1.0, "kW", Metric::PeakCooling).unwrap() == 1000.0);
        assert!(to_si(0.0, "C", Metric::MaxZoneTemp).unwrap() == CELSIUS_TO_KELVIN);
    }

    #[test]
    fn unknown_unit_is_rejected_not_silently_dropped() {
        assert!(to_si_energy(1.0, "erg").is_none());
        assert!(to_si_power(1.0, "hp").is_none());
        assert!(to_si_temperature(1.0, "R").is_none());
    }

    // --- Metric / status parsing -----------------------------------------

    #[test]
    fn metric_parse_is_case_insensitive_and_strict() {
        assert_eq!(Metric::parse("Annual_Cooling"), Some(Metric::AnnualCooling));
        assert_eq!(Metric::parse("peak_heating"), Some(Metric::PeakHeating));
        assert!(Metric::parse("annual_gas").is_none());
    }

    #[test]
    fn metric_si_quantity_and_symbol_are_consistent() {
        assert_eq!(Metric::AnnualHeating.si_quantity(), SiQuantity::Energy);
        assert_eq!(Metric::PeakCooling.si_quantity(), SiQuantity::Power);
        assert_eq!(Metric::MinZoneTemp.si_quantity(), SiQuantity::Temperature);
        assert_eq!(Metric::AnnualCooling.si_unit_symbol(), "J");
        assert_eq!(Metric::PeakHeating.si_unit_symbol(), "W");
        assert_eq!(Metric::MaxZoneTemp.si_unit_symbol(), "K");
    }

    #[test]
    fn status_parse_covers_all_manifest_legend_values() {
        assert_eq!(
            RecordStatus::parse("published"),
            Some(RecordStatus::Published)
        );
        assert_eq!(
            RecordStatus::parse("Transcribed"),
            Some(RecordStatus::Transcribed)
        );
        assert_eq!(RecordStatus::parse("interim"), Some(RecordStatus::Interim));
        assert!(RecordStatus::parse("draft").is_none());
    }

    // --- Loader: real data files -----------------------------------------

    #[test]
    fn e100_csv_loads_with_no_errors() {
        let content = read_data_file(E100_CSV).expect("E100 csv present");
        let (bounds, errors) = load_bounds_csv(&content);
        assert!(errors.is_empty(), "parse errors: {errors:?}");
        assert!(!bounds.is_empty(), "E100 csv produced zero bounds");
    }

    #[test]
    fn ae_csv_loads_with_no_errors() {
        let content = read_data_file(AE_CSV).expect("AE csv present");
        let (bounds, errors) = load_bounds_csv(&content);
        assert!(errors.is_empty(), "parse errors: {errors:?}");
        assert!(!bounds.is_empty(), "AE csv produced zero bounds");
    }

    #[test]
    fn every_bound_is_physically_sane() {
        for &path in &[E100_CSV, AE_CSV] {
            let content = read_data_file(path).expect("csv present");
            let (bounds, _errs) = load_bounds_csv(&content);
            for b in &bounds {
                assert!(
                    b.is_physically_sane(),
                    "{} {:?}: physically insane (low={}, high={}, unit={})",
                    b.case_id,
                    b.metric,
                    b.recorded_low,
                    b.recorded_high,
                    b.recorded_unit,
                );
            }
        }
    }

    #[test]
    fn every_bound_has_complete_provenance() {
        // Acceptance criterion: documented provenance on every record.
        for &path in &[E100_CSV, AE_CSV] {
            let content = read_data_file(path).expect("csv present");
            let (bounds, errors) = load_bounds_csv(&content);
            assert!(errors.is_empty(), "{}: {errors:?}", path);
            for b in &bounds {
                let p = &b.provenance;
                assert!(!p.source_program.is_empty(), "{}: empty program", b.case_id);
                assert!(
                    !p.program_version.is_empty(),
                    "{}: empty version",
                    b.case_id
                );
                assert!(!p.source_table.is_empty(), "{}: empty table", b.case_id);
                assert!(!p.source_page.is_empty(), "{}: empty page", b.case_id);
            }
        }
    }

    #[test]
    fn si_conversion_matches_recorded_value_times_documented_factor() {
        // No magic numbers: derive expected SI from the recorded value and the
        // documented factor for the recorded unit, per metric dimension.
        for &path in &[E100_CSV, AE_CSV] {
            let content = read_data_file(path).expect("csv present");
            let (bounds, _) = load_bounds_csv(&content);
            for b in &bounds {
                let factor = match (b.metric.si_quantity(), b.recorded_unit.as_str()) {
                    (SiQuantity::Energy, "MWh") => 1.0e6 * 3600.0,
                    (SiQuantity::Energy, "kWh") => 1.0e3 * 3600.0,
                    (SiQuantity::Energy, "GJ") => 1.0e9,
                    (SiQuantity::Power, "kW") => 1.0e3,
                    (SiQuantity::Power, "W") => 1.0,
                    (SiQuantity::Temperature, "C") => 1.0, // offset handled below
                    (SiQuantity::Temperature, "K") => 1.0,
                    _ => panic!(
                        "test does not yet cover unit `{}` for {:?} — add the factor",
                        b.recorded_unit, b.metric
                    ),
                };
                let want_low = if b.metric.si_quantity() == SiQuantity::Temperature {
                    // C/K offset is a documented constant, not a magic number.
                    b.recorded_low * factor + CELSIUS_TO_KELVIN
                } else {
                    b.recorded_low * factor
                };
                let want_high = if b.metric.si_quantity() == SiQuantity::Temperature {
                    b.recorded_high * factor + CELSIUS_TO_KELVIN
                } else {
                    b.recorded_high * factor
                };
                let rel = |a: f64, e: f64| (a - e).abs() <= 1.0e-9 * e.abs().max(1.0);
                assert!(
                    rel(b.low_si, want_low) && rel(b.high_si, want_high),
                    "{} {:?}: SI low/high = {}/{}, expected {}/{} (unit {})",
                    b.case_id,
                    b.metric,
                    b.low_si,
                    b.high_si,
                    want_low,
                    want_high,
                    b.recorded_unit,
                );
            }
        }
    }

    #[test]
    fn mid_is_exactly_halfway_between_low_and_high() {
        for &path in &[E100_CSV, AE_CSV] {
            let content = read_data_file(path).expect("csv present");
            let (bounds, _) = load_bounds_csv(&content);
            for b in &bounds {
                let expected_mid = 0.5 * (b.low_si + b.high_si);
                assert!(
                    (b.mid_si - expected_mid).abs() <= 1.0e-9 * expected_mid.abs(),
                    "{} {:?}: mid {} != expected {}",
                    b.case_id,
                    b.metric,
                    b.mid_si,
                    expected_mid,
                );
            }
        }
    }

    #[test]
    fn case_ids_belong_to_expected_families() {
        let e100 = read_data_file(E100_CSV).unwrap();
        let (bounds, _) = load_bounds_csv(&e100);
        for b in &bounds {
            assert!(
                b.case_id.starts_with('E') && b.case_id[1..].chars().all(|c| c.is_ascii_digit()),
                "E100 csv contains non-E-family case `{}`",
                b.case_id,
            );
        }
        let ae = read_data_file(AE_CSV).unwrap();
        let (bounds, _) = load_bounds_csv(&ae);
        for b in &bounds {
            assert!(
                b.case_id.starts_with("AE") && b.case_id[2..].chars().all(|c| c.is_ascii_digit()),
                "AE csv contains non-AE-family case `{}`",
                b.case_id,
            );
        }
    }

    #[test]
    fn index_supports_case_metric_lookup() {
        let content = read_data_file(AE_CSV).unwrap();
        let (bounds, _) = load_bounds_csv(&content);
        let idx = index_by_case_metric(&bounds);
        let key = ("AE101".to_string(), Metric::AnnualCooling);
        let b = idx.get(&key).expect("AE101 annual_cooling present");
        assert!(b.recorded_unit == "MWh");
        assert!(b.recorded_low < b.recorded_high);
    }

    // --- Loader robustness: malformed input ------------------------------

    #[test]
    fn missing_provenance_field_is_a_parse_error() {
        let csv = "case_id,metric,low,high,unit,source_program,program_version,source_table,source_page,status\n\
                   E100,annual_cooling,1.0,2.0,MWh,,multi,Table,pending,transcribed\n";
        let (bounds, errors) = load_bounds_csv(csv);
        assert!(bounds.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("source_program"));
    }

    #[test]
    fn unknown_metric_is_reported_not_silently_dropped() {
        let csv = "case_id,metric,low,high,unit,source_program,program_version,source_table,source_page,status\n\
                   E100,annual_gas,1.0,2.0,MWh,ensemble,multi,Table,pending,transcribed\n";
        let (bounds, errors) = load_bounds_csv(csv);
        assert!(bounds.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("annual_gas"));
    }

    #[test]
    fn wrong_column_count_is_reported() {
        let csv = "case_id,metric,low,high,unit,source_program,program_version,source_table,source_page,status\n\
                   E100,annual_cooling,1.0,2.0,MWh,ensemble\n";
        let (bounds, errors) = load_bounds_csv(csv);
        assert!(bounds.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("10 columns"));
    }

    #[test]
    fn invalid_unit_for_metric_dimension_is_rejected() {
        // kW is a power unit; annual_cooling is Energy → must reject.
        let csv = "case_id,metric,low,high,unit,source_program,program_version,source_table,source_page,status\n\
                   E100,annual_cooling,1.0,2.0,kW,ensemble,multi,Table,pending,transcribed\n";
        let (bounds, errors) = load_bounds_csv(csv);
        assert!(bounds.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("invalid"));
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let csv = "# a comment\n\n\
                   case_id,metric,low,high,unit,source_program,program_version,source_table,source_page,status\n\
                   # another comment\n\
                   E100,annual_cooling,1.0,2.0,MWh,ensemble,multi,Table,pending,transcribed\n";
        let (bounds, errors) = load_bounds_csv(csv);
        assert!(errors.is_empty());
        assert_eq!(bounds.len(), 1);
    }

    #[test]
    fn wrong_header_aborts_with_error() {
        let csv = "case_id,foo,bar\nE100,x,1.0\n";
        let (bounds, errors) = load_bounds_csv(csv);
        assert!(bounds.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("expected header"));
    }
}

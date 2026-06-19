//! Blind Validation Test Suite for ASHRAE 140
//!
//! This test suite measures the baseline failure state when all corrections
//! are disabled (ValidationMode::Blind). This is part of the ASHRAE 140
//! Blind Validation Plan (v1.3) Phase A.2.
//!
//! In addition to the annual / peak / free-floating metrics, this suite also
//! measures **monthly** heating/cooling energy for Cases 600 and 900 against
//! the Phase D ±10% criterion (issue #1165). The monthly metric is
//! **reporting-only**: it never fails the build — the physics fixes that would
//! make it pass are tracked separately in #1163 and #1168. See
//! `tests/reference_data/ashrae140/monthly/README.md` for the interim
//! reference-data provenance.
//!
//! # Expected Result
//! ~0% pass rate when corrections are disabled - the corrections are what
//! make the current numbers look acceptable.
//!
//! # Usage
//! ```bash
//! cargo test --test ashrae_140_blind_validation -- --nocapture
//! ```

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::benchmark;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

// ─────────────────────────────────────────────────────────────────────────────
// Monthly energy aggregation helpers (issue #1165)
// ─────────────────────────────────────────────────────────────────────────────
/// Inclusive start hour (0-based, hour-of-year) of each calendar month for a
/// non-leap TMY representative year (365 d × 24 h = 8760 h). Derived from the
/// cumulative day counts [31,28,31,30,31,30,31,31,30,31,30,31,31].
const MONTH_START_HOUR: [usize; 12] = [
    0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
];

/// Three-letter month labels, aligned with [`MONTH_START_HOUR`].
const MONTH_LABELS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Map an hour-of-year (0..8759) to a month index (0..11) using
/// [`MONTH_START_HOUR`]. Linear scan is trivially cheap (12 entries).
fn month_index_for_hour(hour: usize) -> usize {
    let mut idx = 0;
    for (i, &start) in MONTH_START_HOUR.iter().enumerate() {
        if hour >= start {
            idx = i;
        } else {
            break;
        }
    }
    idx
}

struct BlindValidationResult {
    case_id: String,
    metric: String,
    simulated_value: f64,
    reference_min: f64,
    reference_max: f64,
    percent_error: f64,
    within_tolerance: bool,
}

fn simulate_case_blind(
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
) -> CaseResultsBlinded {
    let mut model = ThermalModel::<VectorField>::from_spec(spec);

    model.reset_peak_power();
    model.reset_heating_cooling_energy();

    const STEPS: usize = 8760;
    let num_zones = model.num_zones;
    let is_free_floating = spec.is_free_floating();

    if is_free_floating {
        model.heating_setpoint = -999.0;
        model.cooling_setpoint = 999.0;
        model.hvac_heating_capacity = 0.0;
        model.hvac_cooling_capacity = 0.0;
    }

    let mut hvac_enabled_vals = vec![1.0; num_zones];
    if !spec.hvac.is_empty() {
        for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
            if zone_idx < num_zones {
                hvac_enabled_vals[zone_idx] = if hvac.is_enabled() { 1.0 } else { 0.0 };
            }
        }
    }
    model.hvac_enabled = VectorField::new(hvac_enabled_vals);

    let mut min_temp_celsius: f64 = f64::INFINITY;
    let mut max_temp_celsius: f64 = f64::NEG_INFINITY;

    // Monthly energy accumulators (kWh per month, Jan..Dec). We sample the
    // model's cumulative kWh accessors before/after each step and bucket the
    // delta into the current month. kWh / 1000 -> MWh at the end. Because we
    // reuse the same cumulative fields the annual values are derived from,
    // Σ(monthly_mwh) == annual_mwh by construction. (issue #1165)
    let mut monthly_heating_kwh: [f64; 12] = [0.0; 12];
    let mut monthly_cooling_kwh: [f64; 12] = [0.0; 12];

    for step in 0..STEPS {
        let hour_of_day = step % 24;

        let weather_data = DenverTmyWeather::new().get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());

        if let Some(hvac_schedule) = spec.hvac.first() {
            let hour = hour_of_day as u8;
            let heating_sp = hvac_schedule
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac_schedule.heating_setpoint);
            let cooling_sp = model.cooling_schedule.value(hour as usize);
            model.heating_setpoint = heating_sp;
            model.cooling_setpoint = cooling_sp;
        }

        // Snapshot cumulative energy before the physics step so the delta can
        // be bucketed into the current month.
        let heat_before = model.get_heating_energy_kwh();
        let cool_before = model.get_cooling_energy_kwh();

        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        let m = month_index_for_hour(step);
        monthly_heating_kwh[m] += model.get_heating_energy_kwh() - heat_before;
        monthly_cooling_kwh[m] += model.get_cooling_energy_kwh() - cool_before;

        if is_free_floating {
            if let Some(&zone_temp) = model.temperatures.as_slice().first() {
                min_temp_celsius = min_temp_celsius.min(zone_temp);
                max_temp_celsius = max_temp_celsius.max(zone_temp);
            }
        }
    }

    let mut monthly_heating_mwh: [f64; 12] = [0.0; 12];
    let mut monthly_cooling_mwh: [f64; 12] = [0.0; 12];
    for i in 0..12 {
        monthly_heating_mwh[i] = monthly_heating_kwh[i] / 1000.0;
        monthly_cooling_mwh[i] = monthly_cooling_kwh[i] / 1000.0;
    }

    CaseResultsBlinded {
        min_temp_celsius: if is_free_floating && min_temp_celsius != f64::INFINITY {
            Some(min_temp_celsius)
        } else {
            None
        },
        max_temp_celsius: if is_free_floating && max_temp_celsius != f64::NEG_INFINITY {
            Some(max_temp_celsius)
        } else {
            None
        },
        annual_heating_mwh: model.annual_heating_energy / 1000.0,
        annual_cooling_mwh: model.annual_cooling_energy / 1000.0,
        peak_heating_kw: model.get_peak_heating_power_kw(),
        peak_cooling_kw: model.get_peak_cooling_power_kw(),
        monthly_heating_mwh,
        monthly_cooling_mwh,
    }
}

struct CaseResultsBlinded {
    min_temp_celsius: Option<f64>,
    max_temp_celsius: Option<f64>,
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
    /// Monthly heating energy (MWh) for Jan..Dec (issue #1165).
    /// Sum equals `annual_heating_mwh` (delta accumulation, same units).
    monthly_heating_mwh: [f64; 12],
    /// Monthly cooling energy (MWh) for Jan..Dec (issue #1165).
    /// Sum equals `annual_cooling_mwh` (delta accumulation, same units).
    monthly_cooling_mwh: [f64; 12],
}

fn run_blind_validation() -> Vec<BlindValidationResult> {
    let mut results = Vec::new();
    let benchmark_data = benchmark::get_all_benchmark_data();

    let cases = vec![
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case610,
        ASHRAE140Case::Case620,
        ASHRAE140Case::Case630,
        ASHRAE140Case::Case640,
        ASHRAE140Case::Case650,
        ASHRAE140Case::Case600FF,
        ASHRAE140Case::Case650FF,
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case910,
        ASHRAE140Case::Case920,
        ASHRAE140Case::Case930,
        ASHRAE140Case::Case940,
        ASHRAE140Case::Case950,
        ASHRAE140Case::Case900FF,
        ASHRAE140Case::Case950FF,
        ASHRAE140Case::Case960,
        ASHRAE140Case::Case195,
    ];

    for case in cases {
        let case_id = case.number();
        let spec = case.spec();
        let is_free_floating = spec.is_free_floating();

        let sim_results = simulate_case_blind(&spec);

        if let Some(data) = benchmark_data.get(&case_id) {
            if is_free_floating {
                if let Some(min_temp) = sim_results.min_temp_celsius {
                    let ref_mid = (data.min_free_float_min + data.min_free_float_max) / 2.0;
                    let percent_error = ((min_temp - ref_mid) / ref_mid * 100.0).abs();
                    let within_tolerance =
                        min_temp >= data.min_free_float_min && min_temp <= data.min_free_float_max;

                    results.push(BlindValidationResult {
                        case_id: case_id.clone(),
                        metric: "MinFreeFloat".to_string(),
                        simulated_value: min_temp,
                        reference_min: data.min_free_float_min,
                        reference_max: data.min_free_float_max,
                        percent_error,
                        within_tolerance,
                    });
                }

                if let Some(max_temp) = sim_results.max_temp_celsius {
                    let ref_mid = (data.max_free_float_min + data.max_free_float_max) / 2.0;
                    let percent_error = ((max_temp - ref_mid) / ref_mid * 100.0).abs();
                    let within_tolerance =
                        max_temp >= data.max_free_float_min && max_temp <= data.max_free_float_max;

                    results.push(BlindValidationResult {
                        case_id: case_id.clone(),
                        metric: "MaxFreeFloat".to_string(),
                        simulated_value: max_temp,
                        reference_min: data.max_free_float_min,
                        reference_max: data.max_free_float_max,
                        percent_error,
                        within_tolerance,
                    });
                }
            } else {
                let metrics = vec![
                    (
                        "AnnualHeating",
                        sim_results.annual_heating_mwh,
                        data.annual_heating_min,
                        data.annual_heating_max,
                    ),
                    (
                        "AnnualCooling",
                        sim_results.annual_cooling_mwh,
                        data.annual_cooling_min,
                        data.annual_cooling_max,
                    ),
                    (
                        "PeakHeating",
                        sim_results.peak_heating_kw,
                        data.peak_heating_min,
                        data.peak_heating_max,
                    ),
                    (
                        "PeakCooling",
                        sim_results.peak_cooling_kw,
                        data.peak_cooling_min,
                        data.peak_cooling_max,
                    ),
                ];

                for (metric_name, value, ref_min, ref_max) in metrics {
                    if ref_min > 0.0 || ref_max > 0.0 {
                        let ref_mid = (ref_min + ref_max) / 2.0;
                        let percent_error = if ref_mid.abs() > 1e-10 {
                            ((value - ref_mid) / ref_mid * 100.0).abs()
                        } else {
                            0.0
                        };
                        let within_tolerance = value >= ref_min && value <= ref_max;

                        results.push(BlindValidationResult {
                            case_id: case_id.clone(),
                            metric: metric_name.to_string(),
                            simulated_value: value,
                            reference_min: ref_min,
                            reference_max: ref_max,
                            percent_error,
                            within_tolerance,
                        });
                    }
                }
            }
        }
    }

    results
}

fn print_results_table(results: &[BlindValidationResult]) {
    println!("\n====================================================================================================");
    println!("ASHRAE 140 BLIND VALIDATION BASELINE RESULTS");
    println!("(All corrections disabled)");
    println!("====================================================================================================");
    println!();

    println!(
        "{:>8} {:>15} {:>15} {:>15} {:>15} {:>12} {:>10}",
        "Case", "Metric", "Simulated", "Ref Min", "Ref Max", "% Error", "Status"
    );
    println!("----------------------------------------------------------------------------------------------------");

    for r in results {
        let status = if r.within_tolerance { "PASS" } else { "FAIL" };
        println!(
            "{:>8} {:>15} {:>15.4} {:>15.4} {:>15.4} {:>12.2} {:>10}",
            r.case_id,
            r.metric,
            r.simulated_value,
            r.reference_min,
            r.reference_max,
            r.percent_error,
            status
        );
    }

    println!("----------------------------------------------------------------------------------------------------");
}

fn compute_summary_statistics(results: &[BlindValidationResult]) -> (usize, usize, f64, f64) {
    let total = results.len();
    let passed = results.iter().filter(|r| r.within_tolerance).count();
    let failed = total - passed;
    let pass_rate = if total > 0 {
        passed as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    let mae = if !results.is_empty() {
        results.iter().map(|r| r.percent_error).sum::<f64>() / total as f64
    } else {
        0.0
    };

    (passed, failed, pass_rate, mae)
}

fn categorize_failures(
    results: &[BlindValidationResult],
) -> std::collections::HashMap<String, Vec<&BlindValidationResult>> {
    let mut categories: std::collections::HashMap<String, Vec<&BlindValidationResult>> =
        std::collections::HashMap::new();

    for r in results.iter().filter(|r| !r.within_tolerance) {
        let category = if r.case_id.ends_with("FF") {
            "free-float".to_string()
        } else if r.case_id.starts_with("9") {
            "high-mass".to_string()
        } else if r.case_id.starts_with("6") {
            "low-mass".to_string()
        } else {
            "special".to_string()
        };

        categories.entry(category).or_default().push(r);
    }

    categories
}

// ─────────────────────────────────────────────────────────────────────────────
// Monthly energy validation (issue #1165)
// ─────────────────────────────────────────────────────────────────────────────
// Reporting-only: never panics. The monthly reference is an INTERIM
// degree-day-derived value pending direct EnergyPlus monthly output — see
// tests/reference_data/ashrae140/monthly/README.md. The pass rate is tracked
// separately in BLIND_VALIDATION_RESULTS.md.

/// Parsed monthly reference band for one case (heating + cooling, Jan..Dec).
struct MonthlyReference {
    heating_mid_mwh: [f64; 12],
    heating_accept_min_mwh: [f64; 12],
    heating_accept_max_mwh: [f64; 12],
    cooling_mid_mwh: [f64; 12],
    cooling_accept_min_mwh: [f64; 12],
    cooling_accept_max_mwh: [f64; 12],
}

/// One month × (heating|cooling) validation comparison.
struct MonthlyValidationResult {
    case_id: String,
    month_idx: usize,
    metric: &'static str, // "Heating" | "Cooling"
    simulated_mwh: f64,
    reference_mid_mwh: f64,
    accept_min_mwh: f64,
    accept_max_mwh: f64,
    percent_error: f64,
    within_tolerance: bool,
}

/// Parse `tests/reference_data/ashrae140/monthly/case_{id}_monthly_reference.csv`.
///
/// Returns `None` (and prints a notice) if the file is missing, so the monthly
/// test degrades gracefully instead of failing the build when the reference
/// artifact is unavailable. Matches the CSV-loading convention used in
/// `tests/zone_balance_eplus_isolation.rs`.
fn load_monthly_reference(case_id: &str) -> Option<MonthlyReference> {
    let filename =
        format!("tests/reference_data/ashrae140/monthly/case_{case_id}_monthly_reference.csv");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/");
    let full = format!("{path}{filename}");
    let data = match std::fs::read_to_string(&full) {
        Ok(d) => d,
        Err(e) => {
            println!(
                "[monthly] reference CSV not found for Case {case_id} ({filename}): {e}. \
                 Monthly metric will be skipped for this case."
            );
            return None;
        }
    };

    let mut ref_data = MonthlyReference {
        heating_mid_mwh: [0.0; 12],
        heating_accept_min_mwh: [0.0; 12],
        heating_accept_max_mwh: [0.0; 12],
        cooling_mid_mwh: [0.0; 12],
        cooling_accept_min_mwh: [0.0; 12],
        cooling_accept_max_mwh: [0.0; 12],
    };

    let mut rows_parsed = 0usize;
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // First non-comment line is the header (`month,heating_mid_mwh,...`).
        if line.starts_with("month,") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 7 {
            continue;
        }
        // Locate the month row by label so column-order changes can't silently
        // misalign the data.
        let label = cols[0].trim();
        let m_idx = match MONTH_LABELS.iter().position(|&m| m == label) {
            Some(i) => i,
            None => continue,
        };
        let parse = |s: &str| -> f64 { s.trim().parse::<f64>().unwrap_or(0.0) };
        ref_data.heating_mid_mwh[m_idx] = parse(cols[1]);
        ref_data.heating_accept_min_mwh[m_idx] = parse(cols[2]);
        ref_data.heating_accept_max_mwh[m_idx] = parse(cols[3]);
        ref_data.cooling_mid_mwh[m_idx] = parse(cols[4]);
        ref_data.cooling_accept_min_mwh[m_idx] = parse(cols[5]);
        ref_data.cooling_accept_max_mwh[m_idx] = parse(cols[6]);
        rows_parsed += 1;
    }

    if rows_parsed != 12 {
        println!(
            "[monthly] Case {case_id}: expected 12 month rows, parsed {rows_parsed}; \
             skipping monthly metric."
        );
        return None;
    }
    Some(ref_data)
}

/// Run the simulation for a single case and return its monthly energy results.
/// Reuses [`simulate_case_blind`] so monthly numbers are consistent with the
/// annual baseline.
fn simulate_monthly_for_case(case: ASHRAE140Case) -> (String, CaseResultsBlinded) {
    let spec = case.spec();
    (case.number(), simulate_case_blind(&spec))
}

/// Build per-month validation results for one case against its reference band.
///
/// Iterates metric-outer, month-inner so the printed breakdown groups all
/// heating months together then all cooling months (matches the reporting
/// format requested in issue #1165).
fn build_monthly_results(
    case_id: &str,
    sim: &CaseResultsBlinded,
    refr: &MonthlyReference,
    out: &mut Vec<MonthlyValidationResult>,
) {
    for (metric, sim_vals, mid, lo, hi) in [
        (
            "Heating",
            sim.monthly_heating_mwh,
            refr.heating_mid_mwh,
            refr.heating_accept_min_mwh,
            refr.heating_accept_max_mwh,
        ),
        (
            "Cooling",
            sim.monthly_cooling_mwh,
            refr.cooling_mid_mwh,
            refr.cooling_accept_min_mwh,
            refr.cooling_accept_max_mwh,
        ),
    ] {
        for m in 0..12 {
            let ref_mid = mid[m];
            let accept_min = lo[m];
            let accept_max = hi[m];
            let value = sim_vals[m];

            // Skip months where the reference is structurally zero (e.g.
            // Denver cooling in Jan/Feb) — a percent error is meaningless and
            // a 0/0 comparison carries no signal. These are reported as N/A.
            if ref_mid <= 1e-6 {
                continue;
            }

            let percent_error = ((value - ref_mid) / ref_mid * 100.0).abs();
            // Clamp negative simulated energy (numerical noise) to 0 for the
            // in-window test; negative energy is physically meaningless.
            let value_clamped = value.max(0.0);
            let within_tolerance = value_clamped >= accept_min && value_clamped <= accept_max;

            out.push(MonthlyValidationResult {
                case_id: case_id.to_string(),
                month_idx: m,
                metric,
                simulated_mwh: value,
                reference_mid_mwh: ref_mid,
                accept_min_mwh: accept_min,
                accept_max_mwh: accept_max,
                percent_error,
                within_tolerance,
            });
        }
    }
}

fn print_monthly_breakdown(results: &[MonthlyValidationResult]) {
    println!("\n====================================================================================================");
    println!("MONTHLY ENERGY BREAKDOWN (Phase D ±10% criterion, issue #1165)");
    println!("Reference = INTERIM degree-day-derived values — see tests/reference_data/ashrae140/monthly/README.md");
    println!("====================================================================================================");

    // Group by (case_id, metric) preserving input order.
    let mut current_case: Option<String> = None;
    let mut current_metric: Option<String> = None;
    for r in results {
        if current_case.as_deref() != Some(r.case_id.as_str())
            || current_metric.as_deref() != Some(r.metric)
        {
            current_case = Some(r.case_id.clone());
            current_metric = Some(r.metric.to_string());
            println!("\nCase {} Monthly {}:", r.case_id, r.metric);
        }
        let status = if r.within_tolerance { "PASS" } else { "FAIL" };
        println!(
            "  {}: sim={:7.4} ref_mid={:7.4} (±10%: {:7.4}..{:7.4})  error={:6.1}%  {}",
            MONTH_LABELS[r.month_idx],
            r.simulated_mwh,
            r.reference_mid_mwh,
            r.accept_min_mwh,
            r.accept_max_mwh,
            r.percent_error,
            status
        );
    }
    println!("----------------------------------------------------------------------------------------------------");
}

fn compute_monthly_summary(results: &[MonthlyValidationResult]) -> (usize, usize, f64, f64) {
    let total = results.len();
    let passed = results.iter().filter(|r| r.within_tolerance).count();
    let failed = total.saturating_sub(passed);
    let pass_rate = if total > 0 {
        passed as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let mae = if total > 0 {
        results.iter().map(|r| r.percent_error).sum::<f64>() / total as f64
    } else {
        0.0
    };
    (passed, failed, pass_rate, mae)
}

/// Cases that have a monthly reference CSV checked in (issue #1165 scope).
fn monthly_reference_cases() -> Vec<ASHRAE140Case> {
    vec![ASHRAE140Case::Case600, ASHRAE140Case::Case900]
}

/// Run the full monthly validation pass and return per-result rows plus a
/// (passed, total, pass_rate, mae) summary. Reporting-only: never panics.
fn run_monthly_validation() -> (Vec<MonthlyValidationResult>, usize, usize, f64, f64) {
    let mut results: Vec<MonthlyValidationResult> = Vec::new();

    for case in monthly_reference_cases() {
        let (case_id, sim) = simulate_monthly_for_case(case);
        let Some(refr) = load_monthly_reference(&case_id) else {
            // Reference missing — already logged; skip this case.
            continue;
        };
        build_monthly_results(&case_id, &sim, &refr, &mut results);

        // Internal consistency check: Σ(monthly) must equal the annual value
        // the same simulation reports (within float noise). This guards the
        // monthly aggregation logic, not the physics.
        let sum_h: f64 = sim.monthly_heating_mwh.iter().sum();
        let sum_c: f64 = sim.monthly_cooling_mwh.iter().sum();
        let dh = (sum_h - sim.annual_heating_mwh).abs();
        let dc = (sum_c - sim.annual_cooling_mwh).abs();
        if dh > 1e-6 || dc > 1e-6 {
            println!(
                "[monthly] WARNING Case {}: Σ(monthly) ≠ annual (Δheat={:.3e} MWh, \
                 Δcool={:.3e} MWh) — aggregation inconsistency.",
                case_id, dh, dc
            );
        }
    }

    let (passed, failed, pass_rate, mae) = compute_monthly_summary(&results);
    (results, passed, failed, pass_rate, mae)
}

#[test]
fn test_blind_validation_baseline() {
    println!("\nStarting ASHRAE 140 Blind Validation Baseline Measurement");
    println!("Expected: ~0% pass rate (corrections make current numbers look acceptable)\n");

    let results = run_blind_validation();

    print_results_table(&results);

    let (passed, failed, pass_rate, mae) = compute_summary_statistics(&results);

    println!("\n====================================================================================================");
    println!("SUMMARY STATISTICS");
    println!("====================================================================================================");
    println!("Total metrics: {}", results.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    println!("Pass rate: {:.2}%", pass_rate);
    println!("Mean Absolute Error: {:.2}%", mae);
    println!();

    let categories = categorize_failures(&results);
    println!("Failure categories:");
    for (category, failures) in &categories {
        println!("  {}: {} failures", category, failures.len());
        for f in failures.iter().take(3) {
            println!(
                "    - Case {} {}: {:.2}% error",
                f.case_id, f.metric, f.percent_error
            );
        }
        if failures.len() > 3 {
            println!("    ... and {} more", failures.len() - 3);
        }
    }
    println!("====================================================================================================");

    if pass_rate > 50.0 {
        println!("\nWARNING: Pass rate is higher than expected for blind validation.");
        println!("This suggests corrections may not be fully disabled.");
    }
}

/// Monthly energy validation (issue #1165).
///
/// This is **measurement infrastructure**: it computes per-month heating/cooling
/// energy for Cases 600 and 900 by bucketing the simulation's per-step energy
/// deltas into calendar months, compares each month against the Phase D ±10%
/// reference window, and reports the pass rate. It deliberately does **not**
/// panic on physics failure — the underlying fixes are tracked in #1163 and
/// #1168, and the monthly reference itself is an interim degree-day-derived
/// value (see `tests/reference_data/ashrae140/monthly/README.md`). Pass/fail is
/// reported to stdout and tracked in `BLIND_VALIDATION_RESULTS.md`.
#[test]
fn test_monthly_energy_validation_baseline() {
    println!("\nStarting ASHRAE 140 Monthly Energy Validation (issue #1165)");
    println!("Phase D criterion: each month within ±10% of reference midpoint.");
    println!("Reporting-only — physics not expected to pass yet (#1163, #1168).\n");

    let (results, passed, total, pass_rate, mae) = run_monthly_validation();

    if results.is_empty() {
        println!(
            "No monthly results produced (reference CSVs missing for Cases 600/900?). \
             See tests/reference_data/ashrae140/monthly/README.md."
        );
        // Still a successful run: the infrastructure executed without panicking.
        return;
    }

    print_monthly_breakdown(&results);

    let failed = total - passed;
    println!("\n====================================================================================================");
    println!("MONTHLY SUMMARY (separate from annual pass rate)");
    println!("====================================================================================================");
    println!("Total monthly metrics: {}", total);
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    println!("Monthly pass rate: {:.2}%", pass_rate);
    println!("Monthly Mean Absolute Error: {:.2}%", mae);
    println!();
    println!("Reference = INTERIM degree-day-derived values (not direct E+ output).");
    println!("Phase D acceptance requires: (1) replace interim reference with direct");
    println!("EnergyPlus monthly totals, (2) monthly pass rate ≥ target once physics");
    println!("fixes #1163/#1168 land. See BLIND_VALIDATION_RESULTS.md.");
    println!("====================================================================================================");

    // Intentionally no assert: this is reporting infrastructure. The build must
    // not turn red because the (known-broken) physics misses the monthly band.
    if pass_rate > 0.0 {
        println!(
            "\nNOTE: monthly pass rate is {:.2}% (>0) — unexpected for the blind baseline;",
            pass_rate
        );
        println!("either the interim reference is generous, or a physics fix has landed.");
    }
}

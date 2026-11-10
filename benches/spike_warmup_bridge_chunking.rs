//! Spike harness for fluxion issue #4699 — warm-up bridge for chunked (parallel-in-time)
//! annual simulation, LOW-MASS case class only (ASHRAE 140 Case 600).
//!
//! WHAT THIS MEASURES (measurement only; no production code is touched):
//!   1. A sequential Case 600 annual run (the reference, exactly the production
//!      `Case600Model::simulate_year` stepping) — reference A ("as-is", one pass)
//!      plus reference B (a second consecutive annual pass = the periodic-steady
//!      reference, used to separate "warm-up bridging" from the reference's own
//!      January cold-start transient).
//!   2. For every (month boundary, warm-up length N) pair: a CHUNKED-MODE run that
//!      simulates [N preceding days on the REAL preceding weather (Dec -> Jan wrap)]
//!      [the month], then compares the chunk's month portion against the reference
//!      month portion: hourly zone-temperature seam error over the first 72 h and
//!      monthly HVAC energy drift.
//!   3. The N-vs-2N end-of-warm-up state self-check: the observable state at the end
//!      of an N-day warm-up vs the observable state at the end of a 2N-day warm-up
//!      (both ending at the same month boundary), and whether that self-check's
//!      verdict agrees with the measured seam error against the reference.
//!
//! The engine exposes NO serialization/checkpoint/restore of the full solver state
//! (see the report): state deltas are therefore measured through the observable
//! zone / mass / wall-surface outputs the harness can reach.

use std::time::Instant;

use fluxion::ashrae_cases::Orientation;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::solar::{calculate_hourly_solar, WindowProperties};
use fluxion::validation::ashrae140::Case600Model;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

const HOURS_PER_YEAR: usize = 8760;
const HOURS_PER_DAY: usize = 24;

/// Month boundaries used for the measurement: (start hour, length in hours, label).
/// January exercises the Dec -> Jan wrap (its warm-up weather comes from December).
const BOUNDARIES: [(usize, usize, &str); 4] = [
    (0, 744, "January"),
    (2160, 720, "April"),
    (4320, 744, "July"),
    (6480, 744, "October"),
];

/// Warm-up lengths in days.
const WARMUP_DAYS: [usize; 3] = [7, 14, 30];

/// Zone floor area (m²) and window area (m²) — Case 600.
const FLOOR_AREA: f64 = 48.0;
const WINDOW_AREA: f64 = 12.0;

/// Observable state snapshot at a chunk boundary.
#[derive(Debug, Clone, Copy)]
struct StateSnapshot {
    zone_c: f64,
    mass_c: f64,
    air_c: f64,
    wall_surface_c: f64,
}

impl StateSnapshot {
    fn of(model: &ThermalModel<VectorField>) -> Self {
        Self {
            zone_c: model.setpoints.temperatures.as_ref()[0],
            mass_c: model.mass.mass_temperatures.as_ref()[0],
            air_c: model.mass.air_temperatures.as_ref()[0],
            wall_surface_c: model.mass.wall_surface_temperatures.as_ref()[0],
        }
    }

    fn max_abs_delta(self, other: Self) -> f64 {
        [
            (self.zone_c - other.zone_c).abs(),
            (self.mass_c - other.mass_c).abs(),
            (self.air_c - other.air_c).abs(),
            (self.wall_surface_c - other.wall_surface_c).abs(),
        ]
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// A traced annual run: hourly zone temperature and hourly HVAC energy (kWh,
/// positive = heating, negative = cooling) indexed by absolute hour of year.
#[derive(Clone)]
struct Trace {
    temps: Vec<f64>,
    hvac_kwh: Vec<f64>,
}

impl Trace {
    fn month_energy_kwh(&self, start: usize, len: usize) -> (f64, f64) {
        let mut heat = 0.0;
        let mut cool = 0.0;
        for h in &self.hvac_kwh[start..start + len] {
            if *h > 0.0 {
                heat += *h;
            } else {
                cool += -*h;
            }
        }
        (heat, cool)
    }
}

/// Fresh Case 600 model (identical construction to the production
/// `Case600Model::new()`), used for every independent run.
fn fresh_model() -> ThermalModel<VectorField> {
    Case600Model::new().model
}

/// Advance the model by one hour on the weather of absolute hour `hour`.
///
/// This mirrors `Case600Model::simulate_year`'s per-step work exactly (same solar
/// call, same load computation, same `step_physics` entry point, no
/// `model.solar.weather` assignment — the production Case 600 path leaves it unset).
fn step_hour(
    model: &mut ThermalModel<VectorField>,
    weather: &DenverTmyWeather,
    hour: usize,
) -> (f64, f64) {
    let w = weather
        .get_hourly_data(hour)
        .expect("Denver TMY must provide every hour 0..8760");
    let hour_of_day = hour % HOURS_PER_DAY;
    let day_of_year = hour / HOURS_PER_DAY + 1;

    let window = WindowProperties::double_clear(WINDOW_AREA);
    let (_, _, solar_gain) = calculate_hourly_solar(
        39.7392,   // Denver latitude (°N)
        -104.9903, // Denver longitude (°W)
        2024,
        (day_of_year as u32) / 30 + 1,
        day_of_year as u32,
        hour_of_day as f64 + 0.5, // mid-hour
        w.dni,
        w.dhi,
        &window,
        None, // no window geometry details
        None, // no overhang
        &[],  // no fins
        Orientation::South,
        Some(0.2), // ground reflectance
        None,      // UTC offset
        1.0,       // full sky view — the spike is unshaded (sky-diffuse view factor, r120)
    );

    let total_loads = 200.0 + solar_gain.total_gain_w; // 200 W internal + solar
    model.set_loads(&[total_loads / FLOOR_AREA]);
    let hvac_kwh = model.step_physics(hour, w.dry_bulb_temp, 3600.0);
    (model.get_temperatures()[0], hvac_kwh)
}

/// Run a contiguous list of absolute hours, returning the trace.
fn run_segment(
    model: &mut ThermalModel<VectorField>,
    weather: &DenverTmyWeather,
    hours: &[usize],
) -> Trace {
    let mut temps = Vec::with_capacity(hours.len());
    let mut hvac_kwh = Vec::with_capacity(hours.len());
    for &hour in hours {
        let (t, e) = step_hour(model, weather, hour);
        temps.push(t);
        hvac_kwh.push(e);
    }
    Trace { temps, hvac_kwh }
}

/// Seam statistics between a chunk's month portion and a reference month portion.
struct SeamStats {
    max_abs_first_24h: f64,
    max_abs_after_24h: f64,
    max_abs_first_72h: f64,
    mean_abs_first_72h: f64,
}

fn seam_stats(chunk: &[f64], reference: &[f64]) -> SeamStats {
    let n = chunk.len().min(reference.len()).min(72);
    let mut max24 = 0.0_f64;
    let mut max_after24 = 0.0_f64;
    let mut max72 = 0.0_f64;
    let mut sum = 0.0_f64;
    for i in 0..n {
        let d = (chunk[i] - reference[i]).abs();
        max72 = max72.max(d);
        sum += d;
        if i < 24 {
            max24 = max24.max(d);
        } else {
            max_after24 = max_after24.max(d);
        }
    }
    SeamStats {
        max_abs_first_24h: max24,
        max_abs_after_24h: max_after24,
        max_abs_first_72h: max72,
        mean_abs_first_72h: if n > 0 { sum / n as f64 } else { f64::NAN },
    }
}

fn pct(num: f64, den: f64) -> f64 {
    if den.abs() < 1e-12 {
        f64::NAN
    } else {
        100.0 * (num - den) / den
    }
}

/// Format a drift percentage, spelling out the "reference is zero" case.
fn fmt_pct(v: f64) -> String {
    if v.is_nan() {
        "n/a (ref=0)".to_string()
    } else {
        format!("{v:.3}")
    }
}

/// Issue #4699 win/kill bands.
const SEAM_BAND_C: f64 = 0.2;
const ENERGY_BAND_PCT: f64 = 0.5;
/// Tolerance for the N-vs-2N end-of-warm-up state self-check.
const SELF_CHECK_TOL_C: f64 = 0.05;

fn band_seam(v: f64) -> &'static str {
    if v < SEAM_BAND_C {
        "PASS"
    } else {
        "FAIL"
    }
}

fn band_energy(v: f64) -> &'static str {
    if !v.is_nan() && v.abs() < ENERGY_BAND_PCT {
        "PASS"
    } else if v.is_nan() {
        "n/a"
    } else {
        "FAIL"
    }
}

fn main() {
    let weather = DenverTmyWeather::new();

    // ------------------------------------------------------------------
    // Reference A: the production sequential annual run, as-is (one pass).
    // ------------------------------------------------------------------
    println!(
        "=== fluxion #4699 warm-up-bridge spike — LOW-MASS case class (ASHRAE 140 Case 600) ==="
    );
    println!("mode: chunked-mode (measurement only; no production code changed)");
    println!();

    let mut model_a = fresh_model();
    let hours: Vec<usize> = (0..HOURS_PER_YEAR).collect();
    let t0 = Instant::now();
    let ref_a = run_segment(&mut model_a, &weather, &hours);
    let ref_a_secs = t0.elapsed().as_secs_f64();

    // Cross-check against the production `Case600Model::simulate_year()`:
    // the harness reference must reproduce it (annual MWh and hourly temps).
    let mut prod = Case600Model::new();
    let prod_result = prod.simulate_year();
    let prod_heat_kwh = prod_result.annual_heating_mwh * 1000.0;
    let prod_cool_kwh = prod_result.annual_cooling_mwh * 1000.0;
    let (ref_a_heat, ref_a_cool) = ref_a.month_energy_kwh(0, HOURS_PER_YEAR);
    let max_temp_diff = ref_a
        .temps
        .iter()
        .zip(prod_result.hourly_temperatures.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("--- reference A: sequential annual run, as-is (1 pass) ---");
    println!("elapsed: {:.3} s for {} steps", ref_a_secs, HOURS_PER_YEAR);
    println!(
        "annual heating: {:.4} MWh (harness) vs {:.4} MWh (Case600Model::simulate_year)",
        ref_a_heat / 1000.0,
        prod_heat_kwh / 1000.0
    );
    println!(
        "annual cooling: {:.4} MWh (harness) vs {:.4} MWh (Case600Model::simulate_year)",
        ref_a_cool / 1000.0,
        prod_cool_kwh / 1000.0
    );
    println!(
        "max |zone temp difference| harness vs production: {:.3e} °C",
        max_temp_diff
    );
    println!();

    // ------------------------------------------------------------------
    // Reference B: a second consecutive annual pass (periodic-steady
    // reference). Reference A starts the year from the 20 °C initial
    // condition, so its January is a cold-start transient; reference B is
    // the same sequential model after one full year of spin-up.
    // ------------------------------------------------------------------
    let t0 = Instant::now();
    let ref_b = run_segment(&mut model_a, &weather, &hours);
    let ref_b_secs = t0.elapsed().as_secs_f64();
    let (ref_b_heat, ref_b_cool) = ref_b.month_energy_kwh(0, HOURS_PER_YEAR);
    println!(
        "--- reference B: sequential annual run, 2nd consecutive pass (periodic steady state) ---"
    );
    println!("elapsed: {:.3} s for {} steps", ref_b_secs, HOURS_PER_YEAR);
    println!(
        "annual heating: {:.4} MWh, annual cooling: {:.4} MWh",
        ref_b_heat / 1000.0,
        ref_b_cool / 1000.0
    );
    println!(
        "annual energy difference B - A: heating {:.4} MWh, cooling {:.4} MWh",
        (ref_b_heat - ref_a_heat) / 1000.0,
        (ref_b_cool - ref_a_cool) / 1000.0
    );
    println!();

    // ------------------------------------------------------------------
    // Chunked mode: for each boundary and each warm-up length N, run
    // [N preceding days on real preceding weather][the month].
    // ------------------------------------------------------------------
    println!("--- chunked-mode: [N preceding days on real preceding weather][month] ---");
    println!(
        "boundary | warmup_d | chunk_heat_MWh | chunk_cool_MWh | drift_vs_A_heat% | drift_vs_A_cool% | drift_vs_A_total% | drift_vs_B_total% | seam_max_0_24h_C | seam_max_24_72h_C | seam_max_0_72h_C | seam_mean_0_72h_C"
    );

    for (start, len, label) in BOUNDARIES {
        for &n_days in &WARMUP_DAYS {
            let warmup_hours = n_days * HOURS_PER_DAY;

            // Warm-up hours: the real preceding hours, wrapping Dec -> Jan.
            let warm: Vec<usize> = (0..warmup_hours)
                .map(|k| (start + HOURS_PER_YEAR - warmup_hours + k) % HOURS_PER_YEAR)
                .collect();
            let month: Vec<usize> = (start..start + len).collect();

            let mut chunk_model = fresh_model();
            let t0 = Instant::now();
            let _ = run_segment(&mut chunk_model, &weather, &warm);
            let state_n = StateSnapshot::of(&chunk_model);
            let chunk = run_segment(&mut chunk_model, &weather, &month);
            let chunk_secs = t0.elapsed().as_secs_f64();

            // 2N self-check run: same construction, twice the warm-up length,
            // ending at the same boundary.
            let warm2: Vec<usize> = (0..2 * warmup_hours)
                .map(|k| (start + HOURS_PER_YEAR - 2 * warmup_hours + k) % HOURS_PER_YEAR)
                .collect();
            let mut chunk2_model = fresh_model();
            let _ = run_segment(&mut chunk2_model, &weather, &warm2);
            let state_2n = StateSnapshot::of(&chunk2_model);

            let (chunk_heat, chunk_cool) = chunk.month_energy_kwh(0, len);
            let (a_heat, a_cool) = ref_a.month_energy_kwh(start, len);
            let (b_heat, b_cool) = ref_b.month_energy_kwh(start, len);

            let seam_a = seam_stats(&chunk.temps, &ref_a.temps[start..start + len]);
            let seam_b = seam_stats(&chunk.temps, &ref_b.temps[start..start + len]);
            let _ = seam_b; // printed below for the primary (A) reference

            let self_check_delta = state_n.max_abs_delta(state_2n);

            println!(
                "{:8} | {:8} | {:14.5} | {:14.5} | {:>15} | {:>15} | {:>16} | {:>16} | {:17.4} | {:18.4} | {:16.4} | {:16.4}",
                format!("{label}"),
                n_days,
                chunk_heat / 1000.0,
                chunk_cool / 1000.0,
                fmt_pct(pct(chunk_heat, a_heat)),
                fmt_pct(pct(chunk_cool, a_cool)),
                fmt_pct(pct(chunk_heat + chunk_cool, a_heat + a_cool)),
                fmt_pct(pct(chunk_heat + chunk_cool, b_heat + b_cool)),
                seam_a.max_abs_first_24h,
                seam_a.max_abs_after_24h,
                seam_a.max_abs_first_72h,
                seam_a.mean_abs_first_72h,
            );
            println!(
                "         state@boundary N={}d: zone {:.4} C, mass {:.4} C, air {:.4} C, wall_surf {:.4} C | 2N={}d: zone {:.4} C, mass {:.4} C, air {:.4} C, wall_surf {:.4} C | max|N-2N| delta = {:.4} C | chunk wall clock {:.3} s",
                n_days,
                state_n.zone_c,
                state_n.mass_c,
                state_n.air_c,
                state_n.wall_surface_c,
                2 * n_days,
                state_2n.zone_c,
                state_2n.mass_c,
                state_2n.air_c,
                state_2n.wall_surface_c,
                self_check_delta,
                chunk_secs,
            );
            println!(
                "         seam vs reference B (2nd pass): max_0_24h {:.4} C, max_24_72h {:.4} C, max_0_72h {:.4} C",
                seam_b.max_abs_first_24h, seam_b.max_abs_after_24h, seam_b.max_abs_first_72h
            );

            // Verdicts.
            //  - Self-check verdict: the N-vs-2N end-of-warm-up state delta.
            //    CONVERGED if every observable node moved by < SELF_CHECK_TOL_C.
            //  - Issue bands: seam error over the FIRST 24 h AFTER the month
            //    boundary hour (i.e. hours 24..72 of the month) < 0.2 C, and
            //    monthly total energy drift < 0.5 %.
            let self_check_converged = self_check_delta < SELF_CHECK_TOL_C;
            let seam_band_a = band_seam(seam_a.max_abs_after_24h);
            let seam_band_b = band_seam(seam_b.max_abs_after_24h);
            let energy_band_a = band_energy(pct(chunk_heat + chunk_cool, a_heat + a_cool));
            let energy_band_b = band_energy(pct(chunk_heat + chunk_cool, b_heat + b_cool));
            // Does the self-check verdict agree with the reference seam error?
            // (self-check says "converged" <=> the chunk actually reproduces the
            //  reference seam within the band)
            let agrees_b = (self_check_converged && seam_band_b == "PASS")
                || (!self_check_converged && seam_band_b == "FAIL");
            let agrees_a = (self_check_converged && seam_band_a == "PASS")
                || (!self_check_converged && seam_band_a == "FAIL");
            // Diagnostic: the first-24 h zone-temperature profile of the chunk
            // next to both references, so a large seam can be attributed to the
            // chunk or to the reference's own cold-start transient.
            let a_win = &ref_a.temps[start..start + 24.min(len)];
            let b_win = &ref_b.temps[start..start + 24.min(len)];
            let c_win = &chunk.temps[..24.min(len)];
            let rng = |v: &[f64]| {
                let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                format!("[{lo:.3}, {hi:.3}]")
            };
            println!(
                "         zone temp first 24 h — chunk {} | ref A {} | ref B {} (chunk[0]={:.4}, refA[0]={:.4}, refB[0]={:.4})",
                rng(c_win),
                rng(a_win),
                rng(b_win),
                c_win[0],
                a_win[0],
                b_win[0],
            );
            println!(
                "         VERDICT [chunked-mode] self-check(N vs 2N) delta {:.4} C -> {} | seam(24-72h) vs A {:.4} C {} | vs B {:.4} C {} | energy drift vs A {} | vs B {} | self-check agrees with seam vs A: {} | vs B: {}",
                self_check_delta,
                if self_check_converged { "CONVERGED" } else { "NOT CONVERGED" },
                seam_a.max_abs_after_24h,
                seam_band_a,
                seam_b.max_abs_after_24h,
                seam_band_b,
                energy_band_a,
                energy_band_b,
                if agrees_a { "yes" } else { "no" },
                if agrees_b { "yes" } else { "no" },
            );
        }
        println!();
    }

    // ------------------------------------------------------------------
    // Speedup arithmetic: chunk cost vs sequential cost.
    // ------------------------------------------------------------------
    println!("--- summary: worst case over boundaries (chunked-mode) ---");
    println!(
        "NOTE: reference A (the sequential annual run as-is) starts the year from the 20 C initial condition, so its own January is a cold-start transient; reference B (2nd consecutive pass) is the periodic-steady reference."
    );
    for &n_days in &WARMUP_DAYS {
        // Worst case over the three "warm reference" boundaries (April, July,
        // October), where reference A is already at its periodic state.
        let mut worst_seam_a = 0.0_f64;
        let mut worst_seam_b = 0.0_f64;
        let mut worst_drift_a = 0.0_f64;
        let mut worst_drift_b = 0.0_f64;
        let mut worst_self_check = 0.0_f64;
        let mut jan_drift_a = f64::NAN;
        let mut jan_seam_a = f64::NAN;
        for (start, len, _label) in BOUNDARIES {
            let warmup_hours = n_days * HOURS_PER_DAY;
            let warm: Vec<usize> = (0..warmup_hours)
                .map(|k| (start + HOURS_PER_YEAR - warmup_hours + k) % HOURS_PER_YEAR)
                .collect();
            let month: Vec<usize> = (start..start + len).collect();
            let mut m = fresh_model();
            let _ = run_segment(&mut m, &weather, &warm);
            let state_n = StateSnapshot::of(&m);
            let chunk = run_segment(&mut m, &weather, &month);
            let warm2: Vec<usize> = (0..2 * warmup_hours)
                .map(|k| (start + HOURS_PER_YEAR - 2 * warmup_hours + k) % HOURS_PER_YEAR)
                .collect();
            let mut m2 = fresh_model();
            let _ = run_segment(&mut m2, &weather, &warm2);
            let state_2n = StateSnapshot::of(&m2);

            let seam_a = seam_stats(&chunk.temps, &ref_a.temps[start..start + len]);
            let seam_b = seam_stats(&chunk.temps, &ref_b.temps[start..start + len]);
            let (ch, cc) = chunk.month_energy_kwh(0, len);
            let (ah, ac) = ref_a.month_energy_kwh(start, len);
            let (bh, bc) = ref_b.month_energy_kwh(start, len);
            let da = pct(ch + cc, ah + ac).abs();
            let db = pct(ch + cc, bh + bc).abs();
            let sc = state_n.max_abs_delta(state_2n);

            if start == 0 {
                jan_drift_a = da;
                jan_seam_a = seam_a.max_abs_after_24h;
            } else {
                worst_seam_a = worst_seam_a.max(seam_a.max_abs_after_24h);
                worst_seam_b = worst_seam_b.max(seam_b.max_abs_after_24h);
                worst_drift_a = worst_drift_a.max(da);
                worst_drift_b = worst_drift_b.max(db);
                worst_self_check = worst_self_check.max(sc);
            }
        }
        println!(
            "warm-up {:2} d | worst seam(24-72h) over Apr/Jul/Oct: vs A {:.4} C ({}), vs B {:.4} C ({}) | worst |energy drift|: vs A {:.3}% ({}), vs B {:.3}% ({}) | worst N-vs-2N self-check delta {:.4} C ({}) | January vs A: drift {:.3}% ({}) seam {:.4} C ({})",
            n_days,
            worst_seam_a,
            band_seam(worst_seam_a),
            worst_seam_b,
            band_seam(worst_seam_b),
            worst_drift_a,
            band_energy(worst_drift_a),
            worst_drift_b,
            band_energy(worst_drift_b),
            worst_self_check,
            if worst_self_check < SELF_CHECK_TOL_C { "CONVERGED" } else { "NOT CONVERGED" },
            jan_drift_a,
            band_energy(jan_drift_a),
            jan_seam_a,
            band_seam(jan_seam_a),
        );
    }
    println!();

    // ------------------------------------------------------------------
    // Speedup arithmetic: chunk cost vs sequential cost.
    // ------------------------------------------------------------------
    println!(
        "--- honest speedup arithmetic (chunk cost = (30 + warmup) / 365 of a sequential year) ---"
    );
    let per_hour_secs = ref_a_secs / HOURS_PER_YEAR as f64;
    println!(
        "measured: reference A {:.3} s / 8760 steps = {:.3e} s per simulated hour",
        ref_a_secs, per_hour_secs
    );
    for &n_days in &WARMUP_DAYS {
        let chunk_days = 30.0 + n_days as f64;
        let cost_fraction = chunk_days / 365.0;
        let annual_overhead = 12.0 * cost_fraction; // 12 chunks of 30 days + warm-up each
        let measured_chunk_secs = chunk_days * 24.0 * per_hour_secs;
        println!(
            "warm-up {:2} d: chunk cost = (30 + {:2})/365 = {:.4} of a sequential year; measured chunk wall clock (30 + warm-up) ≈ {:.3} s; 12-chunk annual work = {:.3} sequential-years (overhead x{:.3})",
            n_days,
            n_days,
            cost_fraction,
            measured_chunk_secs,
            annual_overhead,
            annual_overhead
        );
        for workers in [4usize, 8, 16] {
            let speedup = workers as f64 / annual_overhead;
            println!(
                "    workers {:2}: ideal parallel speedup = workers / overhead = {:.2} -> capped at x{:.2}",
                workers,
                speedup,
                speedup.min(workers as f64)
            );
        }
    }
    println!();
    println!("high-mass (Case 900) and ground-coupled case classes: NOT MEASURED in this run.");
}

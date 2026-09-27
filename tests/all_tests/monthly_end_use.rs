//! Issue #4102 — monthly end-use summary tests.
//!
//! Covers the pure post-processing aggregation of the timestep-indexed
//! end-use series (#4101) into 12 monthly bins per year per end use:
//!
//! - bin edges (Jan 31 → Feb 1 boundary) and leap-year February handling
//! - sub-hourly timesteps bin by actual time, not by entry count
//! - multi-year runs bin by (year, month)
//! - monthly sums reconcile exactly with the hourly series totals
//! - monthly peaks equal the maximum timestep-average kW in the month
//! - `SimulationOutput::monthly_end_use_summary` wiring (None handling,
//!   `[end_use][year][month]` shape) and the REST exposure
//! - Case 600 monthly heating/cooling shape cross-check against
//!   `tests/reference_data/ashrae140/monthly/case_600_monthly_reference.csv`

use fluxion::api::schema::SimulationOutput;
use fluxion::validation::report::BenchmarkReport;

/// Bin a single-entry series and return the (year, month) of the bin that
/// received the energy.
fn bin_of_single_entry(hour_index: usize, dt_seconds: f64, leap_year: bool) -> (usize, usize) {
    let mut series = vec![0.0; hour_index + 1];
    series[hour_index] = 1.0;
    let bins = BenchmarkReport::calculate_monthly_end_use(&series, dt_seconds, leap_year);
    let flat = bins
        .iter()
        .position(|b| b.kwh > 0.0)
        .expect("the single nonzero entry must land in exactly one bin");
    (flat / 12, flat % 12)
}

#[test]
fn monthly_bin_edges_jan_feb_boundary() {
    // Hour 743 = Jan 31 23:00-00:00; hour 744 = Feb 1 00:00-01:00.
    assert_eq!(bin_of_single_entry(743, 3600.0, false), (0, 0));
    assert_eq!(bin_of_single_entry(744, 3600.0, false), (0, 1));
    // December's last hour stays in December.
    assert_eq!(bin_of_single_entry(8759, 3600.0, false), (0, 11));
}

#[test]
fn monthly_leap_year_february_has_29_days() {
    // Non-leap: Feb has 28 days, so hour 1416 = Mar 1 00:00.
    assert_eq!(bin_of_single_entry(1415, 3600.0, false), (0, 1));
    assert_eq!(bin_of_single_entry(1416, 3600.0, false), (0, 2));
    // Leap: Feb has 29 days; hour 1416 = Feb 29 00:00, hour 1440 = Mar 1.
    assert_eq!(bin_of_single_entry(1416, 3600.0, true), (0, 1));
    assert_eq!(bin_of_single_entry(1439, 3600.0, true), (0, 1));
    assert_eq!(bin_of_single_entry(1440, 3600.0, true), (0, 2));
}

#[test]
fn monthly_sums_reconcile_exactly_with_series_totals() {
    // Two-year hourly run with a deterministic non-uniform pattern.
    let series: Vec<f64> = (0..2 * 8760)
        .map(|t| ((t * 37) % 11) as f64 * 0.25 + 0.5)
        .collect();
    let bins = BenchmarkReport::calculate_monthly_end_use(&series, 3600.0, false);
    assert_eq!(bins.len(), 24, "two years -> 24 monthly bins");
    let series_total: f64 = series.iter().sum();
    let bins_total: f64 = bins.iter().map(|b| b.kwh).sum();
    assert!(
        (series_total - bins_total).abs() < 1e-9,
        "monthly sums must reconcile exactly: series {series_total} vs bins {bins_total}"
    );
    // Per-year reconciliation as well.
    for year in 0..2 {
        let year_series: f64 = series[year * 8760..(year + 1) * 8760].iter().sum();
        let year_bins: f64 = bins[year * 12..(year + 1) * 12].iter().map(|b| b.kwh).sum();
        assert!(
            (year_series - year_bins).abs() < 1e-9,
            "year {year}: series {year_series} vs bins {year_bins}"
        );
    }
}

#[test]
fn monthly_peak_is_max_timestep_kw_in_month() {
    // Hourly baseline of 1 kWh with a 5 kWh spike in March (hour 1500).
    let mut series = vec![1.0; 8760];
    series[1500] = 5.0;
    let bins = BenchmarkReport::calculate_monthly_end_use(&series, 3600.0, false);
    assert_eq!(bins.len(), 12);
    for (m, bin) in bins.iter().enumerate() {
        if m == 2 {
            assert_eq!(bin.peak_kw, 5.0, "March peak must equal the spike");
        } else {
            assert_eq!(bin.peak_kw, 1.0, "month {m} peak must equal the baseline");
        }
        // Peak is never below any hourly value in its month.
        assert!(bin.peak_kw >= 1.0);
    }
}

#[test]
fn monthly_subhourly_binning_respects_timestep_duration() {
    // 2-hour timesteps (dt = 7200 s), one full year = 4380 entries.
    // Timestep 371 spans [742h, 744h) with midpoint 743h -> January;
    // timestep 372 spans [744h, 746h) with midpoint 745h -> February.
    let mut series = vec![0.0; 4380];
    series[371] = 2.0;
    series[372] = 4.0;
    let bins = BenchmarkReport::calculate_monthly_end_use(&series, 7200.0, false);
    assert_eq!(bins.len(), 12);
    assert_eq!(bins[0].kwh, 2.0, "timestep 371 belongs to January");
    assert_eq!(bins[1].kwh, 4.0, "timestep 372 belongs to February");
    // Peak demand divides by the actual 2-hour timestep, not by one hour.
    assert_eq!(bins[0].peak_kw, 1.0, "2 kWh over 2 h = 1 kW");
    assert_eq!(bins[1].peak_kw, 2.0, "4 kWh over 2 h = 2 kW");
    let total: f64 = bins.iter().map(|b| b.kwh).sum();
    assert_eq!(total, 6.0);
}

#[test]
fn monthly_subhourly_ten_per_hour_case_900_style() {
    // Case-900-style 6-minute timesteps: 10 entries per clock hour.
    // 24 hours of 0.1 kWh per timestep -> ~24 kWh in January, peak 1 kW.
    // (0.1 is inexact in binary, so the energy sum uses an epsilon; the
    // peak is exact because x/x == 1.0 for the identical f64 operands.)
    let series = vec![0.1; 240];
    let bins = BenchmarkReport::calculate_monthly_end_use(&series, 360.0, false);
    assert!(!bins.is_empty());
    assert!(
        (bins[0].kwh - 24.0).abs() < 1e-9,
        "January sum ≈ 24 kWh, got {}",
        bins[0].kwh
    );
    assert_eq!(bins[0].peak_kw, 1.0, "0.1 kWh per 6 min = 1 kW");
}

#[test]
fn monthly_multi_year_bins_by_year_not_folded() {
    // Year 1 January = 1 kWh/h, year 2 January = 3 kWh/h, rest zero.
    let mut series = vec![0.0; 2 * 8760];
    for t in 0..744 {
        series[t] = 1.0;
    }
    for t in 8760..8760 + 744 {
        series[t] = 3.0;
    }
    let bins = BenchmarkReport::calculate_monthly_end_use(&series, 3600.0, false);
    assert_eq!(bins.len(), 24);
    assert_eq!(bins[0].kwh, 744.0, "year 1 January");
    assert_eq!(bins[12].kwh, 3.0 * 744.0, "year 2 January stays separate");
    assert_eq!(bins[1].kwh, 0.0, "year 1 February untouched");
}

#[test]
fn monthly_end_use_summary_none_when_series_missing_or_empty() {
    let zeros = vec![0.0; 8760];
    // Any missing series -> None (all-or-nothing, mirroring #4101).
    assert!(SimulationOutput::monthly_end_use_summary(
        None,
        Some(zeros.clone()),
        Some(zeros.clone()),
        Some(zeros.clone()),
        3600.0,
    )
    .is_none());
    // Empty series -> None (nothing was recorded).
    assert!(SimulationOutput::monthly_end_use_summary(
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        3600.0,
    )
    .is_none());
    // Invalid dt -> None rather than a panic in the reporting path.
    assert!(SimulationOutput::monthly_end_use_summary(
        Some(zeros.clone()),
        Some(zeros.clone()),
        Some(zeros.clone()),
        Some(zeros),
        0.0,
    )
    .is_none());
}

#[test]
fn monthly_end_use_summary_shape_is_end_use_by_year_by_month() {
    let heating: Vec<f64> = (0..8760).map(|t| (t % 24) as f64 * 0.1).collect();
    let cooling = vec![0.5; 8760];
    let lighting = vec![0.25; 8760];
    let equipment = vec![0.125; 8760];
    let summary = SimulationOutput::monthly_end_use_summary(
        Some(heating.clone()),
        Some(cooling.clone()),
        Some(lighting.clone()),
        Some(equipment.clone()),
        3600.0,
    )
    .expect("all series present");
    for (u, series) in [&heating, &cooling, &lighting, &equipment]
        .into_iter()
        .enumerate()
    {
        assert_eq!(summary.kwh[u].len(), 1, "one year of bins");
        assert_eq!(summary.kwh[u][0].len(), 12, "twelve months");
        assert_eq!(summary.peak_kw[u].len(), 1);
        assert_eq!(summary.peak_kw[u][0].len(), 12);
        let series_total: f64 = series.iter().sum();
        let binned: f64 = summary.kwh[u][0].iter().sum();
        let tol = 1e-9 * series_total.abs().max(1.0);
        assert!(
            (series_total - binned).abs() < tol,
            "end use {u}: binned {binned} vs series {series_total}"
        );
    }
    // January heating peak: max hourly value in Jan = 23 * 0.1 kW
    // (0.1 is inexact in binary — compare against the same computation).
    let jan_peak = summary.peak_kw[0][0][0];
    assert!(
        (jan_peak - 23.0 * 0.1).abs() < 1e-12,
        "January heating peak {jan_peak}"
    );
}

#[test]
fn rest_simulation_exposes_monthly_summaries() {
    use fluxion::api::schema::{
        ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
        SimulationOutput, SimulationSchemaV1, WeatherData,
    };
    use fluxion::api::server::run_simulation;
    use fluxion::sim::thermal_selector::ThermalSelector;

    let schema = SimulationSchemaV1 {
        version: SchemaVersion::V1,
        metadata: SchemaMetadata::default(),
        geometry: Geometry::default(),
        constructions: ConstructionSet::default(),
        schedules: ScheduleSet::default(),
        weather: WeatherData::default(),
        controls: ControlSet::default(),
        output: SimulationOutput::default(),
    };

    let out = run_simulation(
        &schema,
        1,
        false,
        ThermalSelector::default(),
        "monthly-test",
    )
    .expect("REST run_simulation succeeds");

    // Hourly series present (precondition from #4101).
    let hourly_heating = out.hourly_heating_kwh.expect("hourly heating recorded");
    // Monthly fields: [1 year][12 months].
    let monthly_heating = out.monthly_heating_kwh.expect("monthly heating present");
    let monthly_heating_peak = out
        .monthly_heating_peak_kw
        .expect("monthly heating peaks present");
    assert_eq!(monthly_heating.len(), 1);
    assert_eq!(monthly_heating[0].len(), 12);
    assert_eq!(monthly_heating_peak.len(), 1);
    assert_eq!(monthly_heating_peak[0].len(), 12);
    for end_use in [
        &out.monthly_cooling_kwh,
        &out.monthly_lighting_kwh,
        &out.monthly_equipment_kwh,
    ] {
        let grid = end_use.as_ref().expect("monthly field present");
        assert_eq!((grid.len(), grid[0].len()), (1, 12));
    }
    // Monthly sums reconcile exactly with the hourly series totals
    // (relative tolerance: the two sums visit the same values in
    // different orders).
    let hourly_total: f64 = hourly_heating.iter().sum();
    let monthly_total: f64 = monthly_heating[0].iter().sum();
    assert!(
        (hourly_total - monthly_total).abs() < 1e-9 * hourly_total.abs().max(1.0),
        "monthly {monthly_total} vs hourly {hourly_total}"
    );
    // Peaks are never below any hourly value in their month.
    let jan_peak = monthly_heating_peak[0][0];
    assert!(hourly_heating[..744].iter().all(|&v| jan_peak >= v));
}

/// Issue #4102 acceptance: cross-check monthly heating/cooling shapes
/// against `case_600_monthly_reference.csv`.
///
/// The reference is the v1.3 documented-shape reference (authoritative
/// annual midpoint redistributed by degree-day share — see
/// `tests/reference_data/ashrae140/monthly/README.md`), and the engine's
/// cooling physics is a known gap (Issue #2239), so the CSV comparison is
/// **reporting-only**: the hard assertions cover structure, exact
/// reconciliation, and the heating shape (winter-dominated), which the
/// physics gets right.
#[test]
fn monthly_case_600_shape_cross_check() {
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::sim::thermal_selector::ThermalSelector;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    use fluxion::weather::denver::DenverTmyWeather;
    use fluxion::weather::WeatherSource;

    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    model.reset_heating_cooling_energy();
    let weather = DenverTmyWeather::new();

    // Manual hourly loop with Denver TMY weather (mirrors
    // `simulate_case_blind`), plus #4101-style end-use metering via
    // accumulator deltas.
    model.diagnostics_state.init_end_use_metering(8760);
    for step in 0..8760usize {
        let w = weather
            .get_hourly_data(step)
            .expect("TMY weather must cover all 8760 hours");
        model.solar.weather = Some(w.clone());
        if let Some(hvac) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            model.setpoints.heating_setpoint = hvac
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac.heating_setpoint);
            model.setpoints.cooling_setpoint = model.setpoints.cooling_schedule.value(step % 24);
        }
        let heat_before = model.get_heating_energy_kwh();
        let cool_before = model.get_cooling_energy_kwh();
        model.step_physics(step, w.dry_bulb_temp, 3600.0);
        let heat_step = (model.get_heating_energy_kwh() - heat_before).max(0.0);
        let cool_step = (model.get_cooling_energy_kwh() - cool_before).max(0.0);
        model
            .diagnostics_state
            .record_timestep(heat_step, cool_step, 0.0, 0.0);
    }

    let heating = model
        .get_hourly_heating_kwh()
        .expect("heating metering recorded");
    let cooling = model
        .get_hourly_cooling_kwh()
        .expect("cooling metering recorded");
    let summary = SimulationOutput::monthly_end_use_summary(
        Some(heating.clone()),
        Some(cooling.clone()),
        Some(vec![0.0; 8760]),
        Some(vec![0.0; 8760]),
        3600.0,
    )
    .expect("monthly summary");
    let monthly_heating = &summary.kwh[0][0];
    let monthly_cooling = &summary.kwh[1][0];
    assert_eq!(monthly_heating.len(), 12);
    assert_eq!(monthly_cooling.len(), 12);

    // Exact reconciliation with the annual accumulators.
    let annual_heating = model.get_heating_energy_kwh();
    let annual_cooling = model.get_cooling_energy_kwh();
    let sum_h: f64 = monthly_heating.iter().sum();
    let sum_c: f64 = monthly_cooling.iter().sum();
    assert!(
        (sum_h - annual_heating).abs() < 1e-6,
        "Σ(monthly heating) {sum_h} vs annual {annual_heating}"
    );
    assert!(
        (sum_c - annual_cooling).abs() < 1e-6,
        "Σ(monthly cooling) {sum_c} vs annual {annual_cooling}"
    );

    // Heating shape: winter-dominated (Dec/Jan/Feb >> Jun/Jul/Aug).
    let winter_h: f64 = monthly_heating[0] + monthly_heating[1] + monthly_heating[11];
    let summer_h: f64 = monthly_heating[5] + monthly_heating[6] + monthly_heating[7];
    assert!(
        winter_h > 3.0 * summer_h,
        "heating must be winter-dominated: winter {winter_h:.1} vs summer {summer_h:.1} kWh"
    );

    // Reporting-only comparison against the v1.3 documented-shape reference.
    let csv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/reference_data/ashrae140/monthly/case_600_monthly_reference.csv"
    );
    let csv = std::fs::read_to_string(csv_path).expect("monthly reference CSV readable");
    let mut ref_heating = [0.0f64; 12];
    let mut ref_cooling = [0.0f64; 12];
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("month") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        let m = match cols[0] {
            "Jan" => 0,
            "Feb" => 1,
            "Mar" => 2,
            "Apr" => 3,
            "May" => 4,
            "Jun" => 5,
            "Jul" => 6,
            "Aug" => 7,
            "Sep" => 8,
            "Oct" => 9,
            "Nov" => 10,
            "Dec" => 11,
            _ => continue,
        };
        ref_heating[m] = cols[1].parse::<f64>().unwrap_or(0.0) * 1000.0;
        ref_cooling[m] = cols[4].parse::<f64>().unwrap_or(0.0) * 1000.0;
    }
    println!("\n[monthly-4102] Case 600 monthly cross-check vs v1.3 reference (kWh):");
    println!(
        "[monthly-4102] {:>4} {:>12} {:>12} {:>12} {:>12}",
        "mon", "sim_heat", "ref_heat", "sim_cool", "ref_cool"
    );
    for m in 0..12 {
        println!(
            "[monthly-4102] {:>4} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
            m + 1,
            monthly_heating[m],
            ref_heating[m],
            monthly_cooling[m],
            ref_cooling[m]
        );
    }
    // Shape sanity vs the reference: heating peaks in a winter month in
    // both sim and reference (reporting context, not a band assert — the
    // reference is a documented shape and cooling physics is Issue #2239).
    let sim_heat_peak = monthly_heating
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(m, _)| m)
        .unwrap();
    let ref_heat_peak = ref_heating
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(m, _)| m)
        .unwrap();
    assert!(
        [0, 1, 11].contains(&sim_heat_peak),
        "sim heating peak month {sim_heat_peak} should be a winter month"
    );
    assert!(
        [0, 1, 11].contains(&ref_heat_peak),
        "reference heating peak month {ref_heat_peak} should be a winter month"
    );
}

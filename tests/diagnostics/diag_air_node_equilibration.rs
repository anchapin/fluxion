//! Diagnostic: Compare 5R1C vs GaugeZoneSolver air-node equilibration dynamics
//!
//! Issue #3916 / LIMIT-21 Phase 8 follow-up: measures the effective air-node
//! equilibration rate of both solvers under identical weather and initial
//! conditions to quantify the ~84%/hr (gauge) vs ~78%/hr (5R1C) gap
//! identified in the Phase 8 investigation.
//!
//! The test runs Case 640 (LowMass + thermostat setback) with both solvers
//! using the same Denver TMY3 weather file and identical initial conditions,
//! then computes the effective per-step equilibration fraction from the
//! air-node temperature trajectory.
//!
//! The mechanism:
//! - Winter (setback regime): Gauge's faster equilibration means more heat
//!   loss at night → more heating energy consumed during setback recovery.
//! - Summer (cooling regime): Gauge's fast equilibration dissipates solar
//!   gains to the envelope before they can drive meaningful cooling demand.
//!
//! **This is a diagnostic test only** — no production code changes, no
//! assertions on the gap. It emits structured output for offline analysis.
//!
//! Run with: `cargo test --test all_tests diag_air_node_equilibration -- --nocapture --ignored`

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::{ThermalSelector, ZoneSolverKind};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Effective equilibration fraction per timestep, computed as:
/// `eff_eq = 1 - |T_air_new - T_ext| / |T_air_old - T_ext|`
/// This measures what fraction of the air-to-outdoor temperature difference
/// is eliminated each hour.
fn compute_effective_equilibration(t_air_old: f64, t_air_new: f64, t_ext: f64) -> f64 {
    let delta_old = (t_air_old - t_ext).abs();
    let delta_new = (t_air_new - t_ext).abs();
    if delta_old < 0.01 {
        0.0 // No meaningful gradient, return 0
    } else {
        1.0 - (delta_new / delta_old)
    }
}

fn run_gauge_simulation() -> Vec<(usize, f64, f64, f64)> {
    // Gauge: ThermalSelector::default() = ZoneSolverKind::Gauge (under --features gauge-solver)
    // In default build, Gauge falls through to 5R1C, so we need --features gauge-solver
    let spec = ASHRAE140Case::Case640.spec();
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    let mut hourly = Vec::with_capacity(8760);

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let t_air_before = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let t_air_after = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);
        hourly.push((step, t_air_before, t_air_after, energy_kwh));
    }

    hourly
}

fn run_5r1c_simulation() -> Vec<(usize, f64, f64, f64)> {
    // Force 5R1C even under --features gauge-solver
    let spec = ASHRAE140Case::Case640.spec();
    let selector = ThermalSelector {
        zone_solver: ZoneSolverKind::FiveROneC,
        conduction_solver: Default::default(),
    };
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &selector)
        .expect("5R1C selector must initialize");
    let weather = DenverTmyWeather::new();

    let mut hourly = Vec::with_capacity(8760);

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let t_air_before = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let t_air_after = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);
        hourly.push((step, t_air_before, t_air_after, energy_kwh));
    }

    hourly
}

/// Monthly energy totals for attribution (in MWh)
fn monthly_energy(hourly: &[(usize, f64, f64, f64)]) -> [(f64, f64); 12] {
    let mut months = [(0.0_f64, 0.0_f64); 12];
    for &(step, _, _, energy_kwh) in hourly {
        let month = (step / 730) as usize; // 2 * 365 = 730 hours/month
        let month = month.min(11);
        if energy_kwh > 0.0 {
            months[month].0 += energy_kwh / 1000.0; // kWh -> MWh
        } else {
            months[month].1 += -energy_kwh / 1000.0; // kWh -> MWh
        }
    }
    months
}

#[test]
#[ignore = "diagnostic-only; run manually with --ignored if needed"]
fn diag_air_node_equilibration() {
    println!("=== 5R1C vs Gauge Air-Node Equilibration Diagnostic ===\n");

    // Run both simulations
    println!("Running 5R1C simulation...");
    let hourly_5r1c = run_5r1c_simulation();
    println!("Running Gauge simulation...");
    let hourly_gauge = run_gauge_simulation();

    // Compute annual totals
    // energy_kwh from step_physics: positive = heating, negative = cooling
    // Convert kWh -> MWh for display
    let (h_5r1c_kwh, c_5r1c_kwh) = hourly_5r1c.iter()
        .fold((0.0_f64, 0.0_f64), |(h, c), &(_, _, _, e)| {
            if e > 0.0 { (h + e, c) } else { (h, c - e) }
        });
    let (h_gauge_kwh, c_gauge_kwh) = hourly_gauge.iter()
        .fold((0.0_f64, 0.0_f64), |(h, c), &(_, _, _, e)| {
            if e > 0.0 { (h + e, c) } else { (h, c - e) }
        });

    let h_5r1c = h_5r1c_kwh / 1000.0;
    let c_5r1c = c_5r1c_kwh / 1000.0;
    let h_gauge = h_gauge_kwh / 1000.0;
    let c_gauge = c_gauge_kwh / 1000.0;

    println!("\n--- Annual Energy Totals ---");
    println!("5R1C:  heating = {:.2} MWh, cooling = {:.2} MWh, H/C = {:.2}", h_5r1c, c_5r1c, h_5r1c / c_5r1c);
    println!("Gauge: heating = {:.2} MWh, cooling = {:.2} MWh, H/C = {:.2}", h_gauge, c_gauge, h_gauge / c_gauge);
    println!("ASHRAE reference: heating = 2.75–3.80 MWh, cooling = 5.95–8.10 MWh");

    // Compute effective equilibration per timestep (exclude setback/transient hours)
    // Focus on stable hours where |T_air - T_out| > 2 K
    let eq_5r1c: Vec<f64> = hourly_5r1c.iter()
        .filter_map(|&(step, t_old, t_new, _)| {
            let weather = DenverTmyWeather::new();
            let t_ext = weather.get_hourly_data(step).unwrap().dry_bulb_temp;
            let delta = (t_old - t_ext).abs();
            if delta > 2.0 && delta < 40.0 {
                Some(compute_effective_equilibration(t_old, t_new, t_ext))
            } else {
                None
            }
        })
        .collect();

    let eq_gauge: Vec<f64> = hourly_gauge.iter()
        .filter_map(|&(step, t_old, t_new, _)| {
            let weather = DenverTmyWeather::new();
            let t_ext = weather.get_hourly_data(step).unwrap().dry_bulb_temp;
            let delta = (t_old - t_ext).abs();
            if delta > 2.0 && delta < 40.0 {
                Some(compute_effective_equilibration(t_old, t_new, t_ext))
            } else {
                None
            }
        })
        .collect();

    let mean_eq_5r1c = eq_5r1c.iter().sum::<f64>() / eq_5r1c.len() as f64;
    let mean_eq_gauge = eq_gauge.iter().sum::<f64>() / eq_gauge.len() as f64;

    println!("\n--- Effective Equilibration Rate (stable hours only) ---");
    println!("5R1C mean equilibration: {:.4} ({:.1}%/hr)", mean_eq_5r1c, 100.0 * mean_eq_5r1c);
    println!("Gauge mean equilibration:  {:.4} ({:.1}%/hr)", mean_eq_gauge, 100.0 * mean_eq_gauge);
    println!("Delta: {:.4} ({:.1}%/hr faster in gauge)", mean_eq_gauge - mean_eq_5r1c, 100.0 * (mean_eq_gauge - mean_eq_5r1c));

    // Monthly breakdown
    let months_5r1c = monthly_energy(&hourly_5r1c);
    let months_gauge = monthly_energy(&hourly_gauge);

    println!("\n--- Monthly Energy (MWh) ---");
    println!("Month | 5R1c H  | 5R1c C  | Gauge H | Gauge C | H delta | C delta");
    for i in 0..12 {
        let (h5, c5) = months_5r1c[i];
        let (hg, cg) = months_gauge[i];
        let month_name = match i {
            0 => "Jan", 1 => "Feb", 2 => "Mar", 3 => "Apr", 4 => "May", 5 => "Jun",
            6 => "Jul", 7 => "Aug", 8 => "Sep", 9 => "Oct", 10 => "Nov", 11 => "Dec",
            _ => "??",
        };
        println!("{:>4} | {:>7.2} | {:>7.2} | {:>7.2} | {:>7.2} | {:>+7.2} | {:>+7.2}",
            month_name, h5, c5, hg, cg, hg - h5, cg - c5);
    }

    // Summer peak analysis (June-August)
    let summer_cooling_5r1c: f64 = months_5r1c[5..8].iter().map(|&(_, c)| c).sum();
    let summer_cooling_gauge: f64 = months_gauge[5..8].iter().map(|&(_, c)| c).sum();
    let winter_heating_5r1c: f64 = months_5r1c[11].0 + months_5r1c[0].0 + months_5r1c[1].0;
    let winter_heating_gauge: f64 = months_gauge[11].0 + months_gauge[0].0 + months_gauge[1].0;

    println!("\n--- Seasonal Summary ---");
    println!("Summer (Jun-Aug) cooling: 5R1C = {:.2} MWh, Gauge = {:.2} MWh", summer_cooling_5r1c, summer_cooling_gauge);
    println!("Winter (Dec-Feb) heating: 5R1C = {:.2} MWh, Gauge = {:.2} MWh", winter_heating_5r1c, winter_heating_gauge);
    println!("Summer cooling gap: {:.2} MWh (gauge under-predicts by {:.0}%)",
        summer_cooling_5r1c - summer_cooling_gauge,
        100.0 * (summer_cooling_5r1c - summer_cooling_gauge) / summer_cooling_5r1c.max(0.01));
    println!("Winter heating gap: {:.2} MWh (gauge over-predicts by {:.0}%)",
        winter_heating_gauge - winter_heating_5r1c,
        100.0 * (winter_heating_gauge - winter_heating_5r1c) / winter_heating_5r1c.max(0.01));

    println!("\n=== End Diagnostic ===");
}

// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Measurement harness for the Issue #4693 error-controlled adaptive
//! timestepping spike (NOT shipped).
//!
//! Runs one ASHRAE 140 case for a full year in one of three modes and prints
//! the metered results as machine-readable key/value lines:
//!
//! * `ref360`  — fixed 6-minute steps (10 sub-steps per hour) — the reference
//! * `ref3600` — fixed hourly steps
//! * `adaptive` — error-controlled stepping through
//!   `fluxion::sim::adaptive_timestep::ErrorControlledStepper`
//!
//! Usage (from the repository root):
//!
//! ```text
//! CARGO_TARGET_DIR=target cargo run --release --manifest-path examples/Cargo.toml \
//!     --bin adaptive_timestep_spike -- <case:900|600> <mode:ref360|ref3600|adaptive> [tolerance]
//! ```

use fluxion::physics::cta::VectorField;
use fluxion::sim::adaptive_timestep::{ErrorControlConfig, ErrorControlledStepper};
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

const EPW_PATH: &str = "assets/weather/USA_CO_Golden-NREL.724666_TMY3.epw";
const HOURS_PER_YEAR: usize = 8760;
const J_TO_MWH: f64 = 1.0 / 3.6e9;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: adaptive_timestep_spike <case:900|600> <mode:ref360|ref3600|adaptive> [tol]"
        );
        std::process::exit(2);
    }
    let case_id = args[1].clone();
    let mode = args[2].clone();
    let tolerance: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.05);

    let spec = match case_id.as_str() {
        "900" => ASHRAE140Case::Case900.spec(),
        "600" => ASHRAE140Case::Case600.spec(),
        other => {
            eprintln!("unknown case {other}");
            std::process::exit(2);
        }
    };

    let mut model = match ThermalModel::<VectorField>::from_spec_with_selector(
        &spec,
        &ThermalSelector::default(),
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model init failed: {e}");
            std::process::exit(2);
        }
    };
    let weather = match EpwWeatherSource::from_file(EPW_PATH) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("cannot read {EPW_PATH}: {e:?}");
            std::process::exit(2);
        }
    };

    let t0 = std::time::Instant::now();
    let mut heating_j = 0.0_f64;
    let mut cooling_j = 0.0_f64;
    let mut peak_heating_w = 0.0_f64;
    let mut peak_cooling_w = 0.0_f64;
    let mut hourly_temps: Vec<f64> = Vec::with_capacity(HOURS_PER_YEAR);

    // Adaptive controller state.
    let mut stepper = ErrorControlledStepper::new(ErrorControlConfig {
        min_dt: 60.0,
        max_dt: 3600.0,
        tolerance,
        order: 1,
        safety_factor: 0.9,
        hour_alignment: true,
    });
    let mut t_now = 0.0_f64; // seconds since the start of the year
    let mut pending_error = 0.0_f64; // error estimate of the last accepted step
    let mut prev_temp: Option<f64> = None;
    let mut last_delta: Option<f64> = None;
    let mut physical_steps = 0usize;

    for hour in 0..HOURS_PER_YEAR {
        let weather_data = match weather.get_hourly_data(hour) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("weather failed at hour {hour}: {e:?}");
                std::process::exit(2);
            }
        };
        model.solar.weather = Some(weather_data.clone());
        let outdoor = weather_data.dry_bulb_temp;
        let hour_end = (hour as f64 + 1.0) * 3600.0;

        match mode.as_str() {
            "ref360" => {
                for _ in 0..10 {
                    let kwh = model.step_physics(hour, outdoor, 360.0);
                    accumulate(
                        kwh,
                        360.0,
                        &mut heating_j,
                        &mut cooling_j,
                        &mut peak_heating_w,
                        &mut peak_cooling_w,
                    );
                    physical_steps += 1;
                }
            }
            "ref3600" => {
                let kwh = model.step_physics(hour, outdoor, 3600.0);
                accumulate(
                    kwh,
                    3600.0,
                    &mut heating_j,
                    &mut cooling_j,
                    &mut peak_heating_w,
                    &mut peak_cooling_w,
                );
                physical_steps += 1;
            }
            "adaptive" => loop {
                let sel = stepper.select_step(pending_error, t_now);
                let kwh = model.step_physics(hour, outdoor, sel.dt);
                accumulate(
                    kwh,
                    sel.dt,
                    &mut heating_j,
                    &mut cooling_j,
                    &mut peak_heating_w,
                    &mut peak_cooling_w,
                );
                physical_steps += 1;

                // Local error estimate for the *next* step: deviation of this
                // step's temperature increment from the previous increment
                // (a first-order extrapolation error, same units as the
                // tolerance).
                let temp = model.get_temperatures()[0];
                if let (Some(prev), Some(last)) = (prev_temp, last_delta) {
                    let delta = temp - prev;
                    pending_error = (delta - last).abs();
                    last_delta = Some(delta);
                } else if prev_temp.is_some() {
                    pending_error = 0.0;
                }
                prev_temp = Some(temp);
                t_now += sel.dt;
                if t_now >= hour_end - 1e-6 {
                    break;
                }
            },
            other => {
                eprintln!("unknown mode {other}");
                std::process::exit(2);
            }
        }

        hourly_temps.push(model.get_temperatures()[0]);
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let stats = stepper.stats();

    println!("case {case_id}");
    println!("mode {mode}");
    println!("tolerance {tolerance}");
    println!("heating_mwh {:.6}", heating_j * J_TO_MWH);
    println!("cooling_mwh {:.6}", cooling_j * J_TO_MWH);
    println!("peak_heating_kw {:.6}", peak_heating_w / 1000.0);
    println!("peak_cooling_kw {:.6}", peak_cooling_w / 1000.0);
    println!("steps_accepted {}", stats.accepted_steps);
    println!("steps_rejected {}", stats.rejected_steps);
    println!("physical_steps {physical_steps}");
    println!("wall_clock_s {elapsed:.3}");
    println!("min_step_fraction {:.6}", stats.min_step_fraction());
    println!("min_step_steps {}", stats.min_step_steps);
    println!("hour_landings {}", stats.hour_landings);
    println!("short_landings {}", stats.short_landings);
    for (ms, count) in &stats.step_histogram {
        println!("histogram_ms {ms} {count}");
    }
    // Hourly zone-air temperature series (one value per line) for RMSE work.
    print!("temps");
    for t in &hourly_temps {
        print!(" {t:.6}");
    }
    println!();
}

#[allow(clippy::too_many_arguments)]
fn accumulate(
    kwh: f64,
    dt: f64,
    heating_j: &mut f64,
    cooling_j: &mut f64,
    peak_heating_w: &mut f64,
    peak_cooling_w: &mut f64,
) {
    let energy_j = kwh * 3.6e6;
    if energy_j > 0.0 {
        *heating_j += energy_j;
        *peak_heating_w = peak_heating_w.max(energy_j / dt);
    } else if energy_j < 0.0 {
        *cooling_j += -energy_j;
        *peak_cooling_w = peak_cooling_w.max(-energy_j / dt);
    }
}

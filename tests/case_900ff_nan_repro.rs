//! Regression guard for Case 900FF NaN instability (Issue #1219).
//!
//! Issue #1219 was resolved by PR #1243 (commit 64e076f), which added a
//! near-zero denominator guard to the 9R4C backward-Euler mass-temperature
//! update (`physics_impl.rs`, `denom = cm/dt + h_tr_em + h_tr_3`). This test
//! pins that fix: it runs the full 8760-step free-floating year with an
//! **early-exit** on the first non-finite / out-of-bounds zone temperature and,
//! on failure, dumps the per-step trajectory around the divergence — far more
//! diagnostic than the annual determinism test's end-of-year min/max assertion.
//!
//! The simulation itself is fast (~0.03s release / ~0.14s debug for 8760
//! steps); the historical >120s timeout came from solver divergence, which the
//! early-exit avoids.
//!
//! Run with:
//!   cargo test --test case_900ff_nan_repro --release -- --nocapture --include-ignored

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Find the first step at which the zone temperature (or any node in the
/// multi-node state vector) becomes non-finite. Returns (step, value).
fn find_first_non_finite(max_steps: usize) -> Option<(usize, f64)> {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    for step in 0..max_steps {
        let weather_data = weather.get_hourly_data(step % 8760).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        let temps = model.get_temperatures();
        for (i, &t) in temps.iter().enumerate() {
            if !t.is_finite() {
                return Some((step, t));
            }
            // Sanity bounds: free-floating zone air must stay in a physically
            // plausible envelope. Excursions beyond [-100, 200] °C indicate the
            // solver is diverging (precursor to NaN).
            if t < -100.0 || t > 200.0 {
                return Some((step, t));
            }
            let _ = i;
        }
    }
    None
}

#[test]
fn test_case_900ff_short_window_no_nan() {
    // Run the FULL year but with early-exit on first non-finite / out-of-bounds
    // value. The simulation itself is fast (<1s for 200 steps); the historical
    // >120s timeout came from solver divergence, which this early-exit avoids.
    let window = 8760;

    match find_first_non_finite(window) {
        None => {
            println!("OK: Case 900FF zone temperature stayed finite for all {window} steps");
        }
        Some((step, bad_val)) => {
            // Dump the state trajectory around the failure for diagnosis.
            println!(
                "FAIL: first non-finite / out-of-bounds zone temperature at step {step}: {bad_val}"
            );
            dump_trajectory_around_failure(step);
            panic!(
                "Case 900FF produced non-finite/out-of-bounds temperature {bad_val} at step {step} \
                 within {window}-step window (Issue #1219)"
            );
        }
    }
}

/// Re-run up to `fail_step + 5` and print the zone temperature each step so the
/// divergence slope is visible in the test output.
fn dump_trajectory_around_failure(fail_step: usize) {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let end = (fail_step + 5).min(8760);
    println!("--- trajectory (step: zone_temp [all nodes]) ---");
    for step in 0..=end {
        let weather_data = weather.get_hourly_data(step % 8760).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let t_out = weather_data.dry_bulb_temp;
        model.step_physics(step, t_out, 3600.0);

        let temps = model.get_temperatures();
        let marker = if step == fail_step {
            "  <-- FIRST BAD"
        } else {
            ""
        };
        println!(
            "step {step:>4}  T_out={t_out:>7.3}  T_zone_nodes={:?}{marker}",
            temps
        );
        if step >= fail_step && temps.iter().all(|t| !t.is_finite()) {
            break;
        }
    }
}

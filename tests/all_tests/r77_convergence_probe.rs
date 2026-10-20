//! [R77 HARNESS — round 77, scratch probe. NOT a regression test.]
//!
//! N-convergence sweep for the coupled single-N scheme
//! (`FLX_R77_COUPLED_N=N`): air and mass sub-stepped together at dt/N,
//! solar-lag as a per-sub-step driver instead of an additive
//! post-correction. Runs three simulated days of free-float cases at
//! N = 1, 3, 6, 12, 24, 48, 96 and prints per-N daily Tmin / Tmax and the
//! delta vs the hourly baseline, so convergence (dT → 0 monotonically
//! vs N) can be read directly.
//!
//! Run: cargo test --release --test all_tests r77_convergence_probe -- --include-ignored --nocapture
//! Case selection: FLX_R77_PROBE_CASE=900FF | 600FF (default 650FF).

/// ENV_LOCK (Issue #3453 convention): the probe mutates process env
/// (flag off/on), so the mutation is serialized through a file-local mutex.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore]
fn r77_convergence_probe() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::sim::thermal_selector::ThermalSelector;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    use fluxion::weather::epw::EpwWeatherSource;
    use fluxion::weather::WeatherSource;

    let case = match std::env::var("FLX_R77_PROBE_CASE").as_deref() {
        Ok("900FF") => ASHRAE140Case::Case900FF.spec(),
        Ok("600FF") => ASHRAE140Case::Case600FF.spec(),
        _ => ASHRAE140Case::Case650FF.spec(),
    };
    let label = match std::env::var("FLX_R77_PROBE_CASE").as_deref() {
        Ok("900FF") => "900FF",
        Ok("600FF") => "600FF",
        _ => "650FF",
    };

    let run = |n: Option<usize>, label: &str| -> (f64, f64) {
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &case,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let weather = EpwWeatherSource::from_file("assets/weather/WD600.epw")
            .expect("Failed to load WD600.epw — run from project root");
        model.setpoints.heating_setpoint = -999.0;
        model.setpoints.cooling_setpoint = 999.0;
        model.hvac.hvac_heating_capacity = 0.0;
        model.hvac.hvac_cooling_capacity = 0.0;
        match n {
            Some(n) => std::env::set_var("FLX_R77_COUPLED_N", n.to_string()),
            None => std::env::remove_var("FLX_R77_COUPLED_N"),
        }
        let mut tmin = f64::INFINITY;
        let mut tmax = f64::NEG_INFINITY;
        for step in 0..72 {
            let weather_data = weather.get_hourly_data(step).unwrap();
            model.solar.weather = Some(weather_data.clone());
            model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
            let t = model.setpoints.temperatures.as_slice()[0];
            if step >= 48 {
                tmin = tmin.min(t);
                tmax = tmax.max(t);
            }
        }
        println!("{label}: Tmin = {tmin:.9}  Tmax = {tmax:.9}");
        (tmin, tmax)
    };

    println!("--- r77_convergence_probe: coupled single-N sweep, case {label} ---");
    let base = run(None, "baseline (hourly, flag off)");
    let mut rows = Vec::new();
    for n in [1usize, 3, 6, 12, 24, 48, 96, 192, 384] {
        let r = run(Some(n), &format!("N={n}"));
        rows.push((n, r));
    }
    println!("--- payoff vs baseline (Tmin delta, Tmax delta), {label} 3-day probe ---");
    for (n, (tmin, tmax)) in &rows {
        println!(
            "N={n}: dTmin = {:+.9}  dTmax = {:+.9}",
            tmin - base.0,
            tmax - base.1
        );
    }
}

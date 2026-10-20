//! [R76 HARNESS — round 76, scratch probe. NOT a regression test.]
//!
//! Measures the sub-hour mass-state payoff at full float precision, by
//! running three simulated days of Case 650FF (free float) at mass
//! sub-step counts N = 1, 2, 4, 8, 16 (`FLX_R76_MASS_SUBSTEPS`) and
//! printing per-N daily Tmin / Tmax and the delta vs the hourly baseline.
//!
//! Run: cargo test --release --test all_tests r76_mass_probe -- --include-ignored --nocapture

/// ENV_LOCK (Issue #3453 convention): the probe mutates process env
/// (flag off/on), so the mutation is serialized through a file-local mutex.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore]
fn r76_mass_probe() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::sim::thermal_selector::ThermalSelector;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    use fluxion::weather::epw::EpwWeatherSource;
    use fluxion::weather::WeatherSource;

    let run = |n: Option<usize>, label: &str| -> (f64, f64) {
        let spec = if std::env::var("FLX_R76_PROBE_CASE").as_deref() == Ok("900FF") {
            ASHRAE140Case::Case900FF.spec()
        } else {
            ASHRAE140Case::Case650FF.spec()
        };
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
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
            Some(n) => std::env::set_var("FLX_R76_MASS_SUBSTEPS", n.to_string()),
            None => std::env::remove_var("FLX_R76_MASS_SUBSTEPS"),
        }
        let mut tmin = f64::INFINITY;
        let mut tmax = f64::NEG_INFINITY;
        for step in 0..72 {
            let weather_data = weather.get_hourly_data(step).unwrap();
            model.solar.weather = Some(weather_data.clone());
            model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
            let t = model.setpoints.temperatures.as_slice()[0];
            let tm = model.mass.mass_temperatures.as_slice()[0];
            if step >= 48 {
                println!("  day {} end: T={t:.6} Tm={tm:.6} ({label})", step / 24);
            }
            tmin = tmin.min(t);
            tmax = tmax.max(t);
        }
        println!("{label}: Tmin = {tmin:.9}  Tmax = {tmax:.9}");
        (tmin, tmax)
    };

    let base = run(None, "baseline (hourly mass)");
    let mut rows = Vec::new();
    for n in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 48, 96] {
        let r = run(Some(n), &format!("N={n}"));
        rows.push((n, r));
    }
    println!("--- payoff vs baseline (Tmin delta, Tmax delta), 650FF 3-day probe ---");
    for (n, (tmin, tmax)) in &rows {
        println!(
            "N={n}: dTmin = {:+.9}  dTmax = {:+.9}",
            tmin - base.0,
            tmax - base.1
        );
    }
    // Convergence sanity: N=1 must differ from the hourly baseline only
    // through the intra-hour coupling refresh (num_tm + t_i boundary at the
    // same single advance) — tiny but nonzero; N monotonically explores
    // resolution. No assertion on physics values: measurement, not a gate.
}

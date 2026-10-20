//! [R75 HARNESS — round 75, scratch probe. NOT a regression test.]
//!
//! Decides whether `FLX_R75_VENT_RAMP` actually executes and how large the
//! vent-transition effect is at full float precision, by running three
//! simulated days of Case 650FF (free float) twice within one process —
//! flag off, then flag on — and printing the transition-hour (07:00 / 18:00)
//! and extreme-hour zone temperatures.
//!
//! Run: cargo test --release --test all_tests r75_vent_probe -- --include-ignored --nocapture

/// ENV_LOCK (Issue #3453 convention): the probe mutates process env
/// (flag off/on), so the mutation is serialized through a file-local mutex.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore]
fn r75_vent_probe() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::sim::thermal_selector::ThermalSelector;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    use fluxion::weather::epw::EpwWeatherSource;
    use fluxion::weather::WeatherSource;

    let run = |flag: bool, label: &str| -> Vec<(usize, f64)> {
        let spec = ASHRAE140Case::Case650FF.spec();
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
        if flag {
            std::env::set_var("FLX_R75_VENT_RAMP", "1");
        } else {
            std::env::remove_var("FLX_R75_VENT_RAMP");
        }
        let mut samples = Vec::new();
        for step in 0..72 {
            let weather_data = weather.get_hourly_data(step).unwrap();
            model.solar.weather = Some(weather_data.clone());
            model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
            let hour = step % 24;
            if hour == 5 || hour == 6 || hour == 7 || hour == 17 || hour == 18 || hour == 19 {
                if let Some(&t) = model.setpoints.temperatures.as_slice().first() {
                    samples.push((step, t));
                }
            }
        }
        println!("--- {label} ---");
        for (step, t) in &samples {
            println!("step {step} (hour {}) T = {:.12}", step % 24, t);
        }
        samples
    };

    let off = run(false, "flag OFF");
    let on = run(true, "flag ON");
    let mut diffs = Vec::new();
    for ((so, to), (sn, tn)) in off.iter().zip(on.iter()) {
        assert_eq!(so, sn);
        diffs.push((so, tn - to));
    }
    println!("--- per-sample delta (ON - OFF) ---");
    let mut max_abs = 0.0f64;
    for (step, d) in &diffs {
        println!("step {step}: delta = {d:.12}");
        max_abs = max_abs.max(d.abs());
    }
    println!("MAX |delta| over probe = {max_abs:.12}");
    assert!(max_abs > 0.0, "arm produced zero effect — arm is inert");
}
